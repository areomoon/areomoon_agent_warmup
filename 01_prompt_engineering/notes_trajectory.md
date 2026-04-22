# Trajectory 實作細節筆記

> 前置：`notes_react.md`
> 對應檔案：`react_pattern.py` 裡的 `trajectory = []`（行 140）
> 串接 career path：Patsnap 科學論文 / 專利抽取 agent 的 debug / eval / fine-tune pipeline

---

## 0. Trajectory 的資料結構（先把 schema 定下來）

現行 `react_pattern.py` 的版本太陽春：
```python
trajectory.append({"step": 0, "agent": "..."})
trajectory.append({"step": 0, "observation": "..."})
```

**生產級 schema 建議**（你到 Patsnap 第一週就會想重構成這樣）：

```python
from dataclasses import dataclass, field
from typing import Literal, Any
import time, uuid

@dataclass
class Step:
    step_id: int
    role: Literal["thought", "action", "observation", "final"]
    content: str                    # thought 文字 or final answer
    tool_name: str | None = None    # 只有 action 有
    tool_args: dict | None = None   # 改用 dict 而非 list[str]
    tool_result: Any | None = None  # 只有 observation 有
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None        # 工具失敗 / 解析失敗

@dataclass
class Trajectory:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    final_answer: str | None = None
    steps: list[Step] = field(default_factory=list)
    status: Literal["success", "max_steps", "error"] = "success"
    total_tokens: int = 0
    total_latency_ms: int = 0
    model: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)  # paper_id, user_id, etc.
```

**為什麼這樣設計**：
- `run_id`：之後所有 log / eval 都用這個串
- `tokens_in/out` per step：算成本 + 找出哪個 step 最貴
- `latency_ms`：找 bottleneck（通常是某個工具 API）
- `error`：Reflector 直接用 error 分類失敗樣本
- `metadata`：存「這條 trajectory 是在處理哪篇 paper / 哪個 user query」

---

## 1. Debug — Agent 答錯時怎麼追

### 1.1 最小可行做法：結構化 log

```python
import json, logging

def log_trajectory(traj: Trajectory, path: str = "trajectories.jsonl"):
    with open(path, "a") as f:
        f.write(json.dumps(asdict(traj), ensure_ascii=False) + "\n")
```

每次 agent run 完 append 一行 JSONL。**不要** 存到 DB（成本高、schema 改動痛）。
JSONL → `jq` / DuckDB / pandas 直接查。

### 1.2 實務 case：Patsnap PM 說「這篇專利 A 的沉積溫度抽錯了」

你的 debug SOP：
```bash
# 1. 用 paper_id 找到那次 run
jq 'select(.metadata.paper_id == "US2024123456")' trajectories.jsonl

# 2. 把 trajectory 展開成可讀格式
python -m agent.tools.replay --run-id abc-123
```

然後你會看到類似：
```
Step 1 THOUGHT:  我要找 substrate temperature
Step 2 ACTION:   search_papers(query="substrate temperature LSMO")   ← ⚠️ query 太窄
Step 3 OBS:      [只返回 3 篇，都不是這個 paper]
Step 4 THOUGHT:  我找不到，用 200°C 作預設                           ← ⚠️ 幻覺
Step 5 FINAL:    {"temperature_C": 200}                               ← ❌
```

**Trajectory 讓你直接指出 Step 2 query 構造有問題**，不用再花 token 跑第二次。這就是 agent 系統的 stack trace。

### 1.3 進階：diff 兩條 trajectory

生產環境改 prompt / 換模型後，**跑同一個 question 存兩條 trajectory，diff 它們**：
```python
def diff_trajectories(old: Trajectory, new: Trajectory):
    for s_old, s_new in zip(old.steps, new.steps):
        if s_old.tool_name != s_new.tool_name:
            print(f"Step {s_old.step_id}: tool changed {s_old.tool_name} → {s_new.tool_name}")
```
這是 regression test 的基礎 —— prompt 微調最常見的副作用就是「某個 query 從 3 步變 7 步」。

---

## 2. Eval — 不只算答案對錯，還要算過程品質

### 2.1 Trajectory-level metrics

光看 final answer 對不對叫 **outcome eval**，太粗。
Agent 系統要看 **process eval**：

| Metric | 怎麼算 | 為什麼重要 |
|--------|-------|----------|
| `n_steps` | `len(traj.steps)` | 同樣答對，3 步比 8 步好 |
| `tool_precision` | 呼叫的工具中「該用這個工具」的比例 | 抓亂選工具的模型 |
| `tool_recall` | 該呼叫的工具中「實際有呼叫」的比例 | 抓偷懶跳過檢索直接猜的模型 |
| `redundant_calls` | 同一個工具 + 同樣 args 呼叫 >1 次 | 模型鬼打牆指標 |
| `cost_per_correct` | `total_tokens / 正確答案數` | 算 ROI |
| `latency_p50/p95` | 每個 step latency 分布 | 找出卡脖子的工具 |

### 2.2 實務 case：Patsnap 內部 benchmark

```python
# benchmark/eval_extractor.py
GOLD = [
    {"paper_id": "P001", "expected": {"temp_C": 700, "pressure_Pa": 26.66}},
    {"paper_id": "P002", "expected": {"temp_C": 750, "pressure_Pa": 13.33}},
    # ... 200 篇標注好的 paper
]

def eval_run(traj: Trajectory, gold: dict) -> dict:
    return {
        "correct": json.loads(traj.final_answer) == gold["expected"],
        "n_steps": len(traj.steps),
        "had_redundant": has_duplicate_action(traj),
        "used_search": any(s.tool_name == "search_papers" for s in traj.steps),
        "cost_tokens": traj.total_tokens,
    }

# 聚合
df = pd.DataFrame([eval_run(t, g) for t, g in zip(trajs, GOLD)])
print(df.groupby("correct")[["n_steps", "cost_tokens"]].mean())
```

你會看到像：
```
          n_steps  cost_tokens
correct
False        7.2       18400       ← 錯的案例更貴，多繞了好幾步
True         4.1        9200
```
**這種分析沒有 trajectory 就做不出來**。面試被問「agent 怎麼 eval」時，能講出「process eval vs outcome eval 差別」就加分。

---

## 3. Fine-tune 訓練料 — 把好的 trajectory 變成 SFT data

### 3.1 資料格式轉換

Anthropic / OpenAI SFT 要求 message 序列。把 trajectory 攤平：

```python
def trajectory_to_sft_example(traj: Trajectory) -> dict:
    """只拿 status=success 且 n_steps <= 6 的 trajectory"""
    messages = [{"role": "user", "content": traj.question}]
    for step in traj.steps:
        if step.role in ("thought", "action"):
            # 把 thought + action 合併成 assistant turn
            content = f"Thought: {step.content}"
            if step.tool_name:
                content += f"\nAction: {step.tool_name}({json.dumps(step.tool_args)})"
            messages.append({"role": "assistant", "content": content})
        elif step.role == "observation":
            messages.append({"role": "user", "content": f"Observation: {step.tool_result}"})
        elif step.role == "final":
            messages.append({"role": "assistant", "content": f"Final Answer: {step.content}"})
    return {"messages": messages}
```

### 3.2 實務 case：Patsnap 垂域蒸餾

**情境**：你用 Claude Opus 跑 5000 篇 paper 抽取，良率 92%。但 Opus 太貴，要訓一個 Qwen-7B 接力。

Pipeline：
1. 用 Opus 跑 5000 條 trajectory
2. **濾掉** 失敗的、步數 > 8 的、有 redundant_calls 的 → 留下約 3800 條「乾淨」trajectory
3. 進一步用 Reflector（見 §4）把這 3800 條標成 good/bad，留下 ~3000 條 gold
4. 轉成 SFT format → QLoRA 訓練（你 Week 5 會碰）
5. 小模型學到的不只是「答案」，是 **「Opus 的思考路徑」**

這就是 **agent distillation** —— 產業現在很紅的做法。Trajectory 是這個 pipeline 的原料。

### 3.3 關鍵：你要濾什麼樣的 trajectory

```python
def is_training_quality(traj: Trajectory) -> bool:
    return all([
        traj.status == "success",
        len(traj.steps) <= 8,
        not has_duplicate_action(traj),
        not any(s.error for s in traj.steps),
        traj.total_tokens < 10000,       # 太長的通常是在鬼打牆
        is_valid_json(traj.final_answer), # final answer 能被機器解析
    ])
```
**別把垃圾 trajectory 丟去訓練** —— 模型會學會垃圾的思考方式。

---

## 4. Reflector — 自動化 trajectory 品質評估

### 4.1 概念

訓練料太多人工標不完。用另一個 LLM（通常是更強的 Opus）當「老師」看 trajectory，自動標好壞。

```python
REFLECTOR_PROMPT = """你是一個 agent trajectory 品質審查員。
閱讀以下 agent 的解題過程，回答：

1. final answer 是否正確？（如有 gold 就對照）
2. 有沒有多餘步驟？
3. 有沒有選錯工具的 step？（指出 step_id）
4. Thought 的推理是否合理？
5. 評分 1–5：這條 trajectory 值得當訓練料嗎？

Trajectory:
{trajectory_str}

Gold answer (if available): {gold}

以 JSON 回覆：
{{
  "score": 1-5,
  "bad_steps": [step_ids],
  "reason": "...",
  "use_for_training": true/false
}}
"""
```

### 4.2 實務 case：Week 5 你可能會做的

```python
def reflect(traj: Trajectory, gold=None) -> dict:
    traj_str = format_trajectory_for_llm(traj)
    resp = client.messages.create(
        model="claude-opus-4-7",   # 用最強模型當老師
        messages=[{"role": "user", "content": REFLECTOR_PROMPT.format(
            trajectory_str=traj_str, gold=gold
        )}],
    )
    return json.loads(resp.content[0].text)

# 用法
for traj in load_trajectories():
    review = reflect(traj, gold=get_gold(traj.metadata["paper_id"]))
    if review["use_for_training"]:
        save_to_sft_corpus(traj)
    else:
        save_to_error_analysis(traj, review["reason"])
```

**延伸**：Reflector 還可以產生「修正版 trajectory」—— 看到 bad step 就重寫那步，之後整條也跟著重算。這叫 **self-correction** / **trajectory rewriting**，是 2024–2025 很多 agent 論文在做的事（STaR, Reflexion, Language Agent Tree Search 等）。

---

## 5. Replay — 不花 token 重現 bug

### 5.1 Deterministic replay

存了 trajectory 就可以**不呼叫 LLM** 重播整個 run：

```python
def replay(traj: Trajectory, verbose=True):
    """把一條 trajectory 以人類可讀格式播一遍，不耗 token。"""
    print(f"Q: {traj.question}\n")
    for step in traj.steps:
        if step.role == "thought":
            print(f"💭 Step {step.step_id} THOUGHT: {step.content}")
        elif step.role == "action":
            print(f"🔧 Step {step.step_id} ACTION:  {step.tool_name}({step.tool_args})")
        elif step.role == "observation":
            preview = str(step.tool_result)[:200]
            print(f"👁️  Step {step.step_id} OBS:     {preview}")
        elif step.role == "final":
            print(f"\n✅ FINAL: {step.content}")
    print(f"\nTotal: {len(traj.steps)} steps, {traj.total_tokens} tokens, {traj.total_latency_ms}ms")
```

### 5.2 實務 case：Patsnap 線上 bug 回報

**情境**：客戶說「你們 agent 昨天幫我抽的這篇專利 thickness 欄位錯了」。

SOP：
1. 從客戶提供的 `request_id` → 找到 `run_id` → 撈 trajectory
2. `replay(traj)` 一鍵看整條思考過程（0 cost）
3. 找出問題 step，**在該 step 後面 fork 一條新 trajectory** 測試修復後的 prompt/tool：
```python
def fork_and_retry(traj: Trajectory, fork_at_step: int, new_tool_version):
    """保留前 N 步，從第 N+1 步開始用新工具重跑。"""
    prefix = traj.steps[:fork_at_step]
    return continue_agent(
        question=traj.question,
        prefix_steps=prefix,
        tools=new_tool_version,
    )
```

這個 pattern 叫 **trajectory branching** —— A/B 測試 prompt/tool 改動的最低成本做法。

---

## 6. 小結：Trajectory 在 agent 系統的角色

```
           ┌────────────────────────────────┐
           │  Agent Run (ReAct loop)        │
           └───────────────┬────────────────┘
                           │
                      Trajectory  ←────── 單一 source of truth
                           │
    ┌──────────┬───────────┼───────────┬──────────┐
    ↓          ↓           ↓           ↓          ↓
  Debug       Eval      SFT Data    Reflector   Replay
  (jq)     (metrics)   (filter)    (LLM judge) (0-cost)
```

**記住一件事**：
> 沒有 trajectory 就沒有 agent ops。你只有一個黑盒子的輸入輸出，無法 debug、無法 eval、無法改進。
> 從 Day 1 就把 trajectory 用結構化 schema 存下來，是 agent 工程師的基本功。

---

## 7. 面試 / 日常對答可能會被問的

- **Q: 你們 agent 怎麼 debug？**
  A: 我們把每次 run 的 trajectory 以 JSONL 存下來，包含每步的 thought / action / observation / tokens / latency。出問題時用 `run_id` 撈出來 replay，通常能定位到某個 step 的 tool args 或 query 有問題。

- **Q: 怎麼判斷 agent 改版有沒有 regression？**
  A: 留一個 ~200 題的 benchmark set，每次改 prompt 或工具跑一遍。不只比 final answer accuracy，還比 trajectory 層級的 metrics：步數、tool precision、redundant calls。

- **Q: 怎麼做 agent 的 distillation？**
  A: 用強模型跑大量 trajectory → Reflector 篩選高品質 → 轉成 SFT 格式訓小模型。關鍵在 trajectory 篩選的 quality filter 要嚴。

---

## 8. 自我檢核

- [ ] 我能寫出一個合理的 `Trajectory` dataclass schema
- [ ] 我能解釋 outcome eval vs process eval 的差別
- [ ] 我能說出至少 4 種 trajectory metric 及其意義
- [ ] 我知道 trajectory → SFT training data 的轉換邏輯
- [ ] 我能描述 Reflector 的輸入輸出與用途
- [ ] 我能在 PM 說「這個 agent 答錯了」時給出完整的 debug SOP
