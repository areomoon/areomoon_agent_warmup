# Anthropic Managed Agents vs LangChain Agents 架構對比

> 適用版本：LangChain ≥ 0.2 / LangGraph ≥ 0.1 | Anthropic SDK ≥ 0.34

---

## 一、核心哲學差異

| 面向 | Anthropic SDK | LangChain / LangGraph |
|------|--------------|----------------------|
| 設計哲學 | **極簡原語**，開發者掌控迴圈邏輯 | **框架抽象**，提供 Agent executor 與 Chain |
| 狀態管理 | 開發者自行維護 `messages` 列表 | `AgentState` TypedDict + reducer 自動合併 |
| 工具系統 | JSON Schema 宣告 + 手動 dispatch | `@tool` 裝飾器 + 自動解析/呼叫 |
| 記憶 | 開發者實作（全量/壓縮/檔案） | `ConversationBufferMemory`、`VectorStoreRetrieverMemory` |
| 串流 | 原生 SSE，細粒度事件控制 | `astream_events()` / `astream_log()` |
| 可觀測性 | 需自行加入 logging | LangSmith 原生整合 |

---

## 二、工具定義對比

### Anthropic SDK（JSON Schema 宣告）

```python
tools = [
    {
        "name": "lookup_material",
        "description": "查詢材料物理性質",
        "input_schema": {
            "type": "object",
            "properties": {
                "material": {"type": "string"},
                "property": {"type": "string"},
            },
            "required": ["material", "property"],
        },
    }
]

# 手動 dispatch
def dispatch(name, inputs):
    if name == "lookup_material":
        return lookup_material(**inputs)
```

### LangChain（裝飾器自動解析）

```python
from langchain.tools import tool

@tool
def lookup_material(material: str, property: str) -> dict:
    """查詢材料物理性質。material 為化學式，property 為性質名稱。"""
    return {"value": 68, "unit": "°C"}

# 自動掛載至 Agent
agent = create_tool_calling_agent(llm, [lookup_material], prompt)
```

**差異分析：**
- LangChain 工具的 description 從 docstring 自動提取，更易維護
- Anthropic 方式更靈活，可動態生成工具 schema
- LangChain 不需要手動 dispatch，框架自動執行工具

---

## 三、記憶管理對比

### Anthropic SDK（手動管理）

```python
# 全量記憶（開發者負責 token 限制）
messages = []

def add_turn(role, content):
    messages.append({"role": role, "content": content})
    if estimate_tokens(messages) > 100_000:
        # 手動觸發壓縮
        messages = compress_history(messages)
```

### LangChain（內建記憶類型）

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=4000,       # 超過自動壓縮
    return_messages=True,
)

# 自動注入對話歷史
chain = ConversationChain(llm=llm, memory=memory)
```

**差異分析：**
- LangChain 的 `SummaryBufferMemory` 對應本模組的 `SummarisationStore`，但更開箱即用
- Anthropic 方式給予更精細的壓縮策略控制（如保留哪些訊息）
- LangChain 記憶類型較多（Buffer / Summary / VectorStore），但需要學習各類 API

---

## 四、多 Agent 協作對比

### Anthropic SDK（手動協調）

```python
# 開發者手動實作 Agent 間通訊
class AgentOrchestrator:
    def __init__(self):
        self.extractor = ExtractionAgent()
        self.validator = ValidationAgent()
        self.writer = WriterAgent()

    def run(self, paper_text):
        data = self.extractor.extract(paper_text)
        report = self.validator.validate(data)
        return self.writer.summarize(data, report)
```

### LangGraph（圖結構自動調度）

```python
from langgraph.graph import StateGraph

workflow = StateGraph(ExtractionState)
workflow.add_node("extractor", extractor_node)
workflow.add_node("validator", validator_node)
workflow.add_node("writer", writer_node)
workflow.add_edge("extractor", "validator")
workflow.add_conditional_edges(
    "validator",
    lambda s: "writer" if s["is_valid"] else "extractor",
)

app = workflow.compile()
result = app.invoke({"text": paper_text})
```

**差異分析：**
- LangGraph 的圖結構更適合複雜的條件分支與迴圈（如重試邏輯）
- Anthropic 方式的控制流更直觀，適合線性或簡單條件的工作流
- 本課程的 `04_multi_store_session.py` 實現了類似 LangGraph 的手動協調

---

## 五、串流處理對比

### Anthropic SDK

```python
with client.messages.stream(...) as stream:
    for event in stream:
        # 細粒度：區分 text_delta vs input_json_delta
        if event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                yield event.delta.text
```

### LangChain

```python
async for event in chain.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        yield event["data"]["chunk"].content
```

**差異分析：**
- Anthropic 串流的事件類型更豐富，工具呼叫也可串流監聽
- LangChain 的 `astream_events` 覆蓋整個 Chain 的事件，適合複雜流程的監控

---

## 六、選擇建議

| 場景 | 建議選擇 | 原因 |
|------|---------|------|
| 學習 Agent 原理 | Anthropic SDK | 透明的控制流，無框架魔法 |
| 快速原型 | LangChain | 工具裝飾器、記憶類型開箱即用 |
| 複雜多 Agent 圖 | LangGraph | 條件分支、並行節點、狀態管理 |
| 生產部署 | Anthropic SDK + 自定義 | 更少依賴、更可控的行為 |
| 可觀測性需求高 | LangChain + LangSmith | 原生追蹤整合 |
| 材料科學抽取任務 | Anthropic SDK | 工具 schema 更精確，易整合領域邏輯 |

---

## 七、遷移對照表

| Anthropic SDK 概念 | LangChain 對應 |
|-------------------|---------------|
| `messages` 列表 | `ConversationBufferMemory` |
| `tool_use` + 手動 dispatch | `@tool` + AgentExecutor |
| `SummarisationStore` | `ConversationSummaryBufferMemory` |
| `client.messages.stream()` | `chain.astream()` |
| 手動 `SessionManager` | `RunnableWithMessageHistory` |
| `stop_reason == "tool_use"` 迴圈 | `AgentExecutor.invoke()` 內部迴圈 |

---

## 相關連結

- [LangChain 文件](https://python.langchain.com/docs/)
- [LangGraph 文件](https://langchain-ai.github.io/langgraph/)
- [Anthropic vs LangChain 官方討論](https://docs.anthropic.com/en/docs/tool-use)
- [本專案 04_ace_framework](../04_ace_framework/) — ACE 如何與 LangGraph 整合
