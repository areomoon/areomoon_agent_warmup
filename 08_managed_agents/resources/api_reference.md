# Anthropic Managed Agents API 速查表

> 版本：Anthropic Python SDK ≥ 0.34.0 | 更新日期：2025-04

---

## 核心 API 呼叫

### 基本訊息建立

```python
from anthropic import Anthropic
client = Anthropic(api_key="sk-ant-...")

response = client.messages.create(
    model="claude-opus-4-5",           # 或 claude-sonnet-4-5
    max_tokens=1024,                   # 必填，最大輸出 token 數
    system="你是材料科學助理",           # 可選系統提示
    messages=[
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},  # 多輪對話
    ],
)
print(response.content[0].text)
print(response.usage.input_tokens, response.usage.output_tokens)
```

---

## Tool Use（工具呼叫）

### 工具定義格式

```python
tools = [
    {
        "name": "tool_name",                    # snake_case，唯一
        "description": "工具功能的精確描述",      # 影響 Agent 決策
        "input_schema": {                        # JSON Schema (draft-07)
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",            # string/number/boolean/array/object
                    "description": "參數說明",
                },
                "param2": {
                    "type": "number",
                    "description": "數值參數",
                },
            },
            "required": ["param1"],             # 必填欄位列表
        },
    }
]
```

### tool_use 回呼流程

```python
messages = [{"role": "user", "content": user_input}]

while True:
    response = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS,
        tools=tools, messages=messages,
    )

    if response.stop_reason == "end_turn":
        # 取得最終文字回應
        final_text = response.content[0].text
        break

    if response.stop_reason == "tool_use":
        # 加入 assistant 回應（含 tool_use blocks）
        messages.append({"role": "assistant", "content": response.content})

        # 執行工具並收集結果
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                output = dispatch_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,    # 必須與 block.id 對應
                    "content": json.dumps(output),
                })

        messages.append({"role": "user", "content": tool_results})
```

### 停止工具呼叫（tool_choice）

```python
# 強制使用特定工具
response = client.messages.create(
    ...,
    tool_choice={"type": "tool", "name": "specific_tool"},
)

# 不允許使用工具
response = client.messages.create(
    ...,
    tool_choice={"type": "none"},
)

# 預設（自動決定是否使用工具）
response = client.messages.create(
    ...,
    tool_choice={"type": "auto"},  # 預設值
)
```

---

## 串流（Streaming）

### 基本串流

```python
with client.messages.stream(
    model=MODEL,
    max_tokens=MAX_TOKENS,
    messages=messages,
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# 取得完整回應物件（串流結束後）
final_message = stream.get_final_message()
```

### 完整事件監聽

```python
with client.messages.stream(...) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
            elif event.delta.type == "input_json_delta":
                # 工具輸入的增量 JSON
                buffer += event.delta.partial_json
        elif event.type == "message_stop":
            break
```

### 串流事件類型一覽

| 事件類型 | 說明 | 關鍵欄位 |
|---------|------|---------|
| `message_start` | 串流開始 | `message.model`, `message.usage` |
| `content_block_start` | 新 block 開始 | `content_block.type` (text/tool_use) |
| `content_block_delta` | 增量更新 | `delta.type`, `delta.text` / `delta.partial_json` |
| `content_block_stop` | Block 完成 | `index` |
| `message_delta` | 訊息 delta | `delta.stop_reason`, `usage` |
| `message_stop` | 串流結束 | — |

---

## 重要參數與限制

### 模型選擇建議

| 模型 | 適用場景 | Context Window |
|------|---------|----------------|
| `claude-opus-4-5` | 複雜推理、長文本分析 | 200K tokens |
| `claude-sonnet-4-5` | 平衡速度與品質 | 200K tokens |
| `claude-haiku-4-5-20251001` | 快速分類、簡單任務 | 200K tokens |

### 常見錯誤與處理

```python
from anthropic import APIStatusError, RateLimitError, APIConnectionError

try:
    response = client.messages.create(...)
except RateLimitError:
    time.sleep(60)  # 等待後重試
except APIStatusError as e:
    print(f"API 錯誤 {e.status_code}: {e.message}")
except APIConnectionError:
    print("網路連線問題，請檢查代理設定")
```

### token 估算公式

```
估算 tokens ≈ 總字元數 / 4   （英文）
估算 tokens ≈ 總字元數 / 2   （中文）
```

---

## 環境設定

```bash
# .env 檔案
ANTHROPIC_API_KEY=sk-ant-api03-...

# 安裝
pip install anthropic python-dotenv

# 驗證
python -c "from anthropic import Anthropic; print(Anthropic().models.list())"
```

---

## 相關連結

- [官方文件首頁](https://docs.anthropic.com/)
- [Messages API 參考](https://docs.anthropic.com/en/api/messages)
- [Tool Use 指南](https://docs.anthropic.com/en/docs/tool-use)
- [Streaming 指南](https://docs.anthropic.com/en/api/messages-streaming)
- [Python SDK GitHub](https://github.com/anthropics/anthropic-sdk-python)
- [模型列表](https://docs.anthropic.com/en/docs/about-claude/models)
