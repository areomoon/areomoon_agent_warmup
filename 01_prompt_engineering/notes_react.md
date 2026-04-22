# ReAct Pattern — 快速建立知識筆記

> 對應檔案：`react_pattern.py`
> 串接 career path：Patsnap 科學論文 / 專利 Agentic Service 的核心 agent loop
> 後續延伸：Week 3 LangGraph + MCP / Week 5 Reflector for fine-tuning

---

## 1. 一句話定義

**ReAct = Reasoning + Acting**：讓 LLM 把「思考」與「呼叫工具」交錯產生，每步都觀察工具結果再決定下一步。

對比：
- **CoT**：只有 thought，純內部推理 → 容易幻想數字
- **Tool use only**：只有 action，沒有顯式 thought → 難 debug、難在多跳查詢間規劃
- **ReAct**：thought ↔ action ↔ observation 循環 → 可解釋、可中斷、可累積成訓練資料

論文：Yao et al., 2022 — *ReAct: Synergizing Reasoning and Acting in Language Models* (https://arxiv.org/abs/2210.03629)

---

## 2. 標準 Trace 結構（記住這個格式）

```
Question: 從這段文字抽取實驗參數，並把氧分壓轉成 Pa。
Thought:  我需要先用 extract_parameter 拿到 pressure 欄位
Action:   extract_parameter(text, "pressure")
Observation: 200 mTorr oxygen partial pressure
Thought:  好，現在把 200 mTorr 轉成 Pa
Action:   convert_units("200", "mTorr", "Pa")
Observation: 26.6644 Pa
Thought:  我有了答案，可以收束
Final Answer: {"pressure_Pa": 26.66, ...}
```

**三個迴圈元素**：
1. `Thought` — 模型的內部規劃（給人看 + 給未來 fine-tune 用）
2. `Action` — 結構化的工具呼叫
3. `Observation` — 工具回傳，下一輪 prompt 的新輸入

**關鍵停止訊號**：`Final Answer:` 出現 → 結束迴圈。

---

## 3. 代碼導讀（`react_pattern.py`）

### 3.1 Tools registry（行 87–100）
```python
TOOLS = {
  "search_papers":      (search_paper_database,    "..."),
  "extract_parameter":  (extract_parameter,        "..."),
  "convert_units":      (calculate_unit_conversion,"..."),
}
```
工具 = `(callable, description)`。description 直接灌進 system prompt 給模型挑選。

### 3.2 System prompt 模板（行 105–115）
強迫模型用固定格式回覆。**這就是「prompt-as-protocol」** —— 沒用 Anthropic native tool use，而是靠文字格式 + stop sequence 來模擬工具呼叫。

### 3.3 主迴圈骨架（行 142–192）
```python
for step in range(max_steps):
    response = client.messages.create(
        ...,
        stop_sequences=["Observation:"],   # ← 關鍵 1
    )
    agent_text = response.content[0].text

    if "Final Answer:" in agent_text:      # ← 關鍵 2：結束條件
        return final

    # 解析 Action: tool_name(args)
    tool_name, args = parse_action(...)
    observation = TOOLS[tool_name][0](*args)

    messages.append({"role": "user",
                     "content": f"Observation: {observation}"})  # ← 關鍵 3：把觀察灌回 context
```

三個你之後會反覆寫的核心機制：
| 機制 | 為什麼重要 |
|------|----------|
| `stop_sequences=["Observation:"]` | 不讓模型「自編」工具回傳值（hallucinated observation 是 ReAct 最常見 bug） |
| `Final Answer:` 偵測 | 沒有它你會無限迴圈 |
| Observation 用 `role: user` 接回去 | 讓模型把它當「外部資訊」而非「自己說的話」 |

### 3.4 為什麼用 `temperature=0`
ReAct 走 deterministic path 比較好 debug，且工具呼叫格式錯一個字就解析失敗。Self-Consistency 那邊才需要 temperature > 0。

---

## 4. Patsnap 場景對應（你之後可能要做的）

把這個 mock loop 想成你未來的真實工作：

| Mock 工具 | 真實對應 | 對應 API/系統 |
|----------|---------|-------------|
| `search_paper_database` | 內部論文/專利 vector search | Patsnap 自家 corpus + Qdrant/Milvus |
| `extract_parameter` | 段落級欄位抽取 | RAG retriever + 抽取 LLM call |
| `convert_units` | 單位/化學式正規化 | pint, RDKit, 自家 normalizer |

**典型 query 流程（材料科學專利）**：
```
Q: 「過去三年所有 LSMO PLD 沉積專利的最佳基板溫度是多少？」

Thought 1: 先檢索相關專利
Action 1:  search_papers("LSMO PLD deposition", date_range="2023-2026")
Obs 1:     找到 47 篇

Thought 2: 對每篇抽 substrate temperature
Action 2:  batch_extract(papers=[...], field="substrate_temperature")
Obs 2:     [700°C, 750°C, 680°C, ...]

Thought 3: 統計分布
Action 3:  aggregate(values, op="mean+std")
Obs 3:     mean=715°C, std=35°C

Final Answer: {"optimal_temperature_C": 715, "std": 35, "n_papers": 47}
```

這正是你 plan 裡寫的「multi-reasoning to assist decisions」。ReAct 是這個流程的最小可運作骨架。

---

## 5. 從這個版本 → 生產級 agent 的演進路線

| 階段 | 改動 | 你的 plan 對應 |
|------|------|--------------|
| 現在：Prompt-as-protocol | 文字格式 + stop sequence | Week 1 |
| Step 1：Native tool use | `client.messages.create(tools=[{...}])` 用 Anthropic SDK 內建 schema | Week 2 後段 |
| Step 2：LangGraph | 把 thought/action/observation 變成 graph node，可加 checkpoint、條件分支、人為審核 | Week 3 |
| Step 3：MCP server | 工具集從 in-process function → 外部 MCP server，讓多個 agent 共用 | Week 3 |
| Step 4：Reflector | 跑完之後另一個 LLM 看 trajectory，標出失敗 step → fine-tune 訓練料 | Week 5 |

**Native tool use 範例（之後你要改的方向）**：
```python
tools = [{
    "name": "extract_parameter",
    "description": "...",
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "parameter": {"type": "string"},
        },
        "required": ["text", "parameter"],
    },
}]
response = client.messages.create(model=MODEL, tools=tools, messages=[...])
# response.content 會包含 type="tool_use" 的 block，args 是 dict 不用 parse 字串
```
**為什麼 production 要遷移**：
- Args 解析交給 SDK，不會因為 `,` 在字串裡爆掉（`parse_action` 那段就有這問題）
- Schema 驗證、錯誤訊息、平行工具呼叫都內建
- LangGraph / Anthropic Agent SDK 都建在這之上

---

## 6. 容易踩的雷（面試 + 實務都會碰到）

1. **沒設 stop_sequence** → 模型自己編 `Observation: xxx`，整條 trajectory 被污染。
2. **Args 字串裡含逗號** → 此檔的 `parse_action` 用 `split(",")`，遇到 `extract_parameter("hello, world", "x")` 會錯切。生產一律走 native tool use 或用 JSON args。
3. **沒 max_steps** → 模型卡在 loop 燒 token。一般設 6–12，超過視為失敗。
4. **Observation 太長** → 把整篇 paper 塞回去 context 會爆。要先 summarize 或只回傳 chunk。
5. **Tool description 寫太模糊** → 模型亂選工具。Description 要包含「何時用 / 輸入格式 / 輸出範例」。
6. **Final Answer 不是合法 JSON** → 在 prompt 裡明示 schema，或上 structured output（Anthropic 的 tool-use 強制 schema 是最穩做法）。
7. **Thought 太長** → 模型把整個世界觀寫進去，慢且貴。實務上會限制 thought 字數或要求 bullet 形式。

---

## 7. 一句話帶回家

> **ReAct 不是一個演算法，是一個「protocol」**：你給 LLM 一個固定的「想 → 做 → 看 → 再想」格式，剩下都是工程化（解析、停止、錯誤處理、observation 縮減）。LangGraph、Anthropic Agent SDK、AutoGen、CrewAI 全部都是這個 protocol 的不同包裝。

把這個檔案的 50 行迴圈讀懂，你就能讀懂任何 agent framework 的核心。

---

## 8. 自我檢核清單

- [ ] 我能不看代碼，畫出 ReAct 的 thought/action/observation 流程圖
- [ ] 我能解釋為什麼要用 `stop_sequences=["Observation:"]`
- [ ] 我能說出 prompt-as-protocol 跟 native tool use 的差別與遷移時機
- [ ] 我能指出 `parse_action` 的兩個脆弱點（逗號、引號）
- [ ] 我能用 Patsnap 場景重畫一條 trajectory（例如「找出最常見的退火溫度」）
- [ ] 我知道 Reflector 階段會用 trajectory 做什麼（→ Week 5 fine-tune 料）
