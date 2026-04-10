# Managed Agents vs LangChain / LangGraph

> 幫助有 LangChain 背景的開發者快速理解 Managed Agents 的定位差異

---

## 整體對比

| 維度 | LangChain / LangGraph | Claude Managed Agents |
|------|----------------------|----------------------|
| **Runtime** | 自建（你的伺服器/Colab）| Managed cloud runtime |
| **State 管理** | 手動（checkpointer, Redis...）| 內建 Session + Memory Store |
| **工具定義** | 自組（`@tool` decorator）| JSON Schema + Custom Tools |
| **內建工具** | 無（需自行實作）| bash, web_search, files, computer |
| **Observability** | 外接 LangSmith（需設定）| 內建 SSE events（原生）|
| **Multi-agent** | LangGraph orchestration | 多 session + shared memory stores |
| **Memory** | 自管（ConversationBuffer, VectorStore）| Memory Store API（version history）|
| **Streaming** | callback / `astream_events` | SSE event stream（標準格式）|
| **Interrupt / Steer** | 手動實作 HumanInTheLoop | 原生 `interrupt()` / `steer()` |
| **環境隔離** | 自負責（Docker, venv）| 雲端 Environment（Python runtime）|
| **費用模型** | API token + 自建基礎設施 | API token + Managed runtime 費用 |
| **學習曲線** | 較高（多概念、版本複雜）| 較低（官方 API，概念一致）|

---

## 概念對應

| LangChain / LangGraph 概念 | Managed Agents 對應 |
|---------------------------|---------------------|
| `AgentExecutor` | Session |
| `ChatPromptTemplate` + system | Agent.system_prompt |
| `@tool` / `StructuredTool` | Custom Tool（JSON Schema）|
| `ConversationBufferMemory` | Session 自動管理對話歷史 |
| `VectorStoreRetriever` | Memory Store（key-value，非向量）|
| `checkpointer` (LangGraph) | Memory Store version history |
| `CompiledGraph.stream()` | `session.send_message()` → SSE |
| `interrupt()` (LangGraph) | `session.interrupt()` |
| `Command(goto=..., update=...)` | `session.steer(content=...)` |
| `LangSmith` tracing | 內建 SSE events（memory_read, tool_use...）|

---

## 什麼時候選 LangChain / LangGraph？

- 需要複雜的有向圖（DAG）工作流，多個節點互相依賴
- 已有大量 LangChain 工具和 integration 生態
- 需要非 Anthropic 模型（GPT-4, Gemini...）
- 自建基礎設施，需要完全掌控 runtime
- 現有 LangSmith 監控設施

## 什麼時候選 Managed Agents？

- 使用 Claude 模型，想要最少設定快速上手
- 需要內建的 memory store（有 version history）
- 需要原生 interrupt / steer（Human-in-the-Loop）
- 不想自管 runtime 環境和 session state
- 需要內建工具（bash, web_search）而不想自行包裝

---

## Code 對比：定義 Tool

### LangChain
```python
from langchain_core.tools import tool

@tool
def lookup_material(material: str, property: str) -> dict:
    """查詢材料性質。"""
    return {"value": 1.57, "unit": "eV"}

agent = create_react_agent(llm, tools=[lookup_material])
```

### Managed Agents
```python
custom_tool = {
    "name": "lookup_material",
    "description": "查詢材料性質。回傳 JSON：{value, unit}",
    "input_schema": {
        "type": "object",
        "properties": {
            "material": {"type": "string"},
            "property": {"type": "string", "enum": ["band_gap", "density"]},
        },
        "required": ["material", "property"],
    },
}
agent = client.agents.create(custom_tools=[custom_tool], ...)
```

---

## Code 對比：Memory

### LangChain（ConversationBufferMemory）
```python
memory = ConversationBufferMemory()
chain = LLMChain(llm=llm, memory=memory, prompt=prompt)
# 只有對話歷史，沒有 version history，沒有 key-value
```

### Managed Agents（Memory Store）
```python
store = client.memory_stores.create(name="playbook")
store.entries.create(key="rules/step1", value="先找摘要...")

# 讀取
entry = store.entries.retrieve(key="rules/step1")
print(entry.version)  # 版本號

# Safe update（precondition）
store.entries.create(key="rules/step1", value="新規則",
                     precondition={"version": entry.version})
```

---

## Code 對比：Streaming

### LangChain
```python
async for event in agent.astream_events({"input": "..."}, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
    elif event["event"] == "on_tool_start":
        print(f"Tool: {event['name']}")
```

### Managed Agents（SSE）
```python
for event in session.send_message(session_id, content="..."):
    if event.type == "content_block_delta":
        print(event.delta.text, end="")
    elif event.type == "tool_use":
        print(f"Tool: {event.name}")
    elif event.type == "memory_write":
        print(f"Memory write: {event.key} v{event.new_version}")
```

Managed Agents 多了 `memory_*` 事件，讓你清楚看到 agent 的記憶體操作。
