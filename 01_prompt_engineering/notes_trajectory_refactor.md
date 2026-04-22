# Trajectory Refactor + Temperature 影響筆記

> 前置：`notes_react.md`、`notes_trajectory.md`
> 對應實作：`trajectory.py`、`react_pattern.py`、`analyze_trajectories.py`
> 為什麼做這個：把 `react_pattern.py` 從 dict-based trajectory 升級成生產級 schema，順便用同一份基礎設施跑一個 temperature × n_trials 的小實驗

---

## 一、Refactor 概述

### 改動清單

| 檔案 | 動作 | 重點 |
|------|------|------|
| `trajectory.py` | **新增** | `Step` / `Trajectory` dataclass + `save()` 寫 JSONL |
| `react_pattern.py` | **改寫** | `react_agent()` 接 `temperature` 參數、回傳 `Trajectory` 物件、自動寫 JSONL |
| `analyze_trajectories.py` | **新增** | 讀 JSONL → pandas → 依 temperature 聚合 metrics |
| `notebook/prompt_engineering_lab.ipynb` | **更新** | ReAct 段示範新 API + temperature sweep 實驗 |

### 為什麼此時改

- Week 1 還在低風險區，現在養成「結構化記錄」習慣，後面 Week 5 fine-tune 時直接受益（trajectory → SFT data 是現成轉換）
- Notebook 裡 cell-15 / cell-17 已經在改 `temperature = 0 / 0.4`，但 `react_agent()` 根本沒收這個參數 → 改完才能真的跑 sweep
- 三個 prompt engineering 檔案（CoT / Self-Consistency / ReAct）之後都能共用同一個 `Trajectory` schema

---

## 二、新 schema 的關鍵設計決策

```python
@dataclass
class Step:
    step_id: int
    role: Literal["thought_action", "observation", "final", "error"]
    content: str = ""
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: Any | None = None
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
```

幾個值得記的決策：

1. **`thought_action` 合併為一個 step**：原本筆記 §0 把 thought / action 拆開，但 LLM 一次 generate 出來就是一段，硬拆反而失真。Observation 才是另一個獨立 step。
2. **`tool_args` 用 dict 而非 `list[str]`**：未來改 native tool use 時直接接得上，現在先用 `{"args": [...]}` 包起來過渡。
3. **per-step token / latency**：不是裝飾品 —— 你 debug 時最常被問「哪一步最貴 / 最慢」，沒記就答不出來。
4. **`status` 欄位 in Trajectory**：`success` / `max_steps` / `parse_error` 三態。Eval 時直接 `df.groupby("status")` 就能看健康度。
5. **不存 `messages` 完整 list**：那是給 LLM 的 wire format，太長且重複，需要時可從 steps 重建。

---

## 三、Temperature 對 ReAct trajectory 的影響（核心理論）

### 直觀答案

| Temperature | Thought 風格 | Action 選擇 | Trajectory 後果 |
|-------------|------------|------------|---------------|
| **0.0** | 同一條推理路徑、固定措辭 | 每次選同一個工具、同一組 args | 重跑 N 次幾乎一樣（API 不保證 100% deterministic） |
| **0.3–0.5** | 措辭略變、邏輯結構穩定 | 工具大致相同，args 偶爾換寫法 | **生產建議區間**：穩定但不死板 |
| **0.7–1.0** | 推理可能繞路、有時更精彩有時更亂 | 工具選擇分歧、args 多樣 | 適合 Self-Consistency / 探索 |
| **>1.0** | 開始幻覺、格式破壞 | 出現不存在的工具名、args 數量錯誤 | `parse_action` 易失敗、`status="parse_error"` 增加 |

### 在這個 repo 的具體例子（LSMO 抽取題）

**`temperature=0.0`** 跑 5 次：
```
Trajectory A: 5 steps  | extract pressure → convert → extract temp → convert → final
Trajectory B: 5 steps  | (幾乎一模一樣)
... 5 條都很像
```

**`temperature=0.7`** 跑 5 次：
```
Trajectory A: 5 steps  | 跟 t=0 一樣的路徑
Trajectory B: 7 steps  | 多 call 一次 search_papers 「想確認 LSMO 是什麼」 → 浪費
Trajectory C: 4 steps  | 一次性 batch 抽兩個欄位 → 更聰明
Trajectory D: 6 steps  | 把 pressure 轉錯單位（mTorr → mTorr）→ 多繞一步修正
Trajectory E: 8 steps  | parse_action 失敗 1 次（args 用了奇怪引號）→ 重試
```
平均步數變高，**variance** 變大 —— 這就是要透過 metrics 量化的東西。

**`temperature=1.2`** 你可能會看到：
- `Action: extract_parameters(...)`（多個 s，工具不存在）
- `extract_parameter(text="LSMO ...", parameter="temp")` 改用 keyword args → `parse_action` 用 `split(",")` 切出爛東西
- Thought 開始離題（「順便分析一下這個材料的歷史…」）

### 三個關鍵 takeaway

1. **ReAct 預設 `temperature=0` 是有道理的** — 工具呼叫格式錯一個字就整條死。Stochastic 會放大這個風險。
2. **「步數分布」是 temperature 的 sensor** — 同一題跑 N 次，看步數的 std。std 越大代表模型越不確定。生產用這個指標決定哪些 task 要切換更強模型 / 加 example。
3. **ReAct 跟 Self-Consistency 的 temperature 哲學相反**：
   - ReAct 要 t=0 → 路徑穩定、可 debug
   - Self-Consistency 要 t>0 → 路徑要不一樣才能 ensemble
   - 兩者結合：對同一題跑 N 條 ReAct trajectory（不同 temp/seed），取 final answer 多數決 → **Self-Consistency-with-ReAct** / Tree-of-Thought 雛形（Week 2–3）

---

## 四、實驗設計：4 × 5 = 20 trial sweep

### 跑法

```bash
cd 01_prompt_engineering
python react_pattern.py                     # 預設 temperature=0
python -c "
from react_pattern import run_temperature_sweep
run_temperature_sweep(temperatures=[0.0, 0.3, 0.7, 1.0], trials=5)
"
python analyze_trajectories.py
```

### 預期會看到的表

```
            success_rate  avg_steps  std_steps  avg_tokens
temperature
0.0                 1.00       3.00       0.00        2400
0.3                 1.00       3.20       0.45        2580
0.7                 0.80       4.40       1.52        3900
1.0                 0.40       5.80       2.05        5500
```

### 怎麼解讀

- `success_rate` 隨 temperature 上升而下降 → 工具呼叫格式被破壞
- `avg_steps` 隨 temperature 上升 → 模型開始繞路
- `std_steps` 隨 temperature 上升 → 行為變不可預測（這個 std 才是真正重要的指標）
- `avg_tokens` 跟 `avg_steps` 強相關 → 直接對應成本

### 為什麼這個實驗值得做

- **建立直覺**：看一張表勝過讀 10 篇論文
- **建立 portfolio**：面試時把 `trajectories.jsonl` + 這張 table 拿出來，直接證明「會做 agent ops」
- **建立 baseline**：之後改 prompt / 換模型，重跑這個 sweep 比較差異

---

## 五、Notebook 對應改動

原本 ReAct 段的 cell：
```python
react_agent(question, client)            # 印一堆東西、回傳 string
```

改成：
```python
traj = react_agent(question, client, temperature=0.0)
print(f"Status: {traj.status}, Steps: {len(traj.steps)}, Tokens: {traj.total_tokens}")
print(f"Final: {traj.final_answer}")
```

新增一個 sweep cell：
```python
from react_pattern import run_temperature_sweep
run_temperature_sweep(temperatures=[0.0, 0.3, 0.7], trials=3)

# 看結果
import pandas as pd, json
rows = [json.loads(l) for l in open("trajectories.jsonl")]
df = pd.DataFrame(...)
df.groupby("temperature").agg(...)
```

---

## 六、自我檢核

- [ ] 我能說出為什麼把 `thought` + `action` 合併成一個 step 而非分開
- [ ] 我能解釋為什麼 ReAct 預設 `temperature=0`，但 Self-Consistency 要 `t>0`
- [ ] 我能用 `std_steps` 這個指標解釋「模型對這題有多不確定」
- [ ] 我能跑完 4×5 sweep 並從結果挑出 production 該用的 temperature
- [ ] 我能把現有 trajectory JSONL 在 5 行內轉成 SFT messages 格式（Week 5 前置技能）
