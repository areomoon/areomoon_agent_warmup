# Managed Agents API Reference

> 整理自官方文件：https://platform.claude.com/docs/en/managed-agents/overview  
> 最後更新：2026-04

---

## 核心資源（Resources）

| 資源 | 說明 |
|------|------|
| Agent | 定義模型、系統提示、工具 |
| Environment | Python runtime（3.10/3.11）+ 安裝的套件 |
| Session | Agent + Environment 的執行實例，包含對話歷史 |
| Memory Store | Key-value 持久化儲存，可 attach 到 session |

---

## Agent API

### 建立 Agent
```python
agent = client.agents.create(
    name="my-agent",
    model="claude-sonnet-4-6",        # 必填
    system_prompt="You are...",        # 可選
    tools=["bash", "web_search"],      # 內建工具
    custom_tools=[...],                # 自定義工具定義（JSON Schema）
)
```

### Agent 可用的內建工具
| 工具 | 說明 |
|------|------|
| `bash` | 執行 shell 指令 |
| `web_search` | 搜尋網路 |
| `files` | 讀寫 session 檔案系統 |
| `computer` | 控制桌面（需額外權限）|

---

## Environment API

```python
env = client.environments.create(
    name="my-env",
    runtime="python3.11",
    packages=["numpy", "pandas", "pydantic"],
)
# Environment 建立後需等待 status="ready"
```

---

## Session API

### 建立 Session
```python
session = client.sessions.create(
    agent_id=agent.id,
    environment_id=env.id,
    memory_stores=[
        {"store_id": store.id, "access": "read"},        # read-only
        {"store_id": cases.id, "access": "read-write"},  # 可讀寫
    ],
)
```

### 發送 Message（SSE Stream）
```python
stream = client.sessions.send_message(
    session_id=session.id,
    content="分析這篇論文...",
)
for event in stream:
    # 處理 SSE 事件（見 Event Types）
    pass
```

### Interrupt Session
```python
client.sessions.interrupt(session_id=session.id)
```

### Steer Session（中途插入指令）
```python
client.sessions.steer(
    session_id=session.id,
    content="請改用 JSON 格式輸出",
)
```

---

## Memory Store API

### 建立與管理
```python
store = client.memory_stores.create(
    name="playbook",
    description="提取規則",
)
```

### CRUD 操作
```python
# Write（無條件覆蓋）
client.memory_stores.entries.create(store.id, key="k", value="v")

# Write（safe update，precondition）
client.memory_stores.entries.create(
    store.id, key="k", value="v_new",
    precondition={"version": 3},   # 只有 version=3 時才寫入
)

# Read
entry = client.memory_stores.entries.retrieve(store.id, key="k")
# entry.value, entry.version, entry.updated_at

# List all keys
entries = client.memory_stores.entries.list(store.id)

# Delete
client.memory_stores.entries.delete(store.id, key="k")

# Version history
history = client.memory_stores.entries.versions.list(store.id, key="k")
```

---

## SSE Event Types

| Event | 觸發時機 |
|-------|---------|
| `message_start` | session 開始新輪對話 |
| `content_block_start` | 新 block 開始（text 或 tool_use）|
| `content_block_delta` | 增量文字（streaming text）|
| `content_block_stop` | block 結束 |
| `tool_use` | agent 呼叫工具 |
| `tool_result` | 工具回傳結果 |
| `memory_list` | agent 列出 store keys |
| `memory_search` | agent 搜尋 store |
| `memory_read` | agent 讀取 entry |
| `memory_write` | agent 寫入 entry |
| `message_stop` | 本輪結束（含 stop_reason）|
| `error` | 錯誤（rate_limit, server_error, etc.）|

### stop_reason 說明
| 值 | 意義 |
|----|------|
| `end_turn` | 正常完成 |
| `max_tokens` | 達到 token 上限 |
| `interrupted` | 被 interrupt 中斷 |
| `tool_use` | 等待 tool result（manual tool） |

---

## Custom Tools 格式

```python
custom_tool = {
    "name": "lookup_material",
    "description": "查詢材料性質。回傳 JSON：{value, unit, confidence}",
    "input_schema": {
        "type": "object",
        "properties": {
            "material": {"type": "string", "description": "化學式，例如 TiO2"},
            "property": {
                "type": "string",
                "enum": ["band_gap", "density"],
            },
        },
        "required": ["material", "property"],
    },
}
```

---

## Rate Limits & Quotas（Beta）

| 項目 | 限制 |
|------|------|
| 同時進行的 sessions | 視 tier 而定 |
| Memory store entries | 1,000 per store（beta）|
| Entry value 大小 | 100 KB |
| Session duration | 24 hours max |

---

## 常見錯誤

| 錯誤碼 | 原因 | 處理方式 |
|--------|------|---------|
| `precondition_failed` | version 不符（concurrent write）| retry with latest version |
| `rate_limit_error` | 超過 API rate limit | 等待 `retry_after` 秒 |
| `store_not_found` | store_id 無效 | 確認 store 存在 |
| `access_denied` | 寫入 read-only store | 確認 access 設定 |
