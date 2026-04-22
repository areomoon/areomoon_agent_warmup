# Constrained Decoding / Structured Output 筆記

> KB 正本：[constrained-decoding](~/PycharmProjects/llm_knowledge_base/wiki/concepts/constrained-decoding.md)
> 參考：[Outlines](https://github.com/dottxt-ai/outlines)、[Willard & Louf 2023](https://arxiv.org/abs/2307.09702)

## 一句話

在 decode 時強制 next token 只能是讓 output 仍符合 schema 的那些。解決 **format 正確性**，不解決 **fact 正確性**。跟 self-consistency 互補，不是替代。

## 四種限制形式

- **JSON Schema** — 輸出必須是符合 schema 的 JSON
- **Regex** — 符合正則
- **CFG (context-free grammar)** — 符合 BNF 文法
- **Tool Use** — 合法的 function call + typed arguments

## 原理

Decoder 維護一個狀態機（schema 編譯後的 FSM）。每 step 計算 logit mask：可讓 state 合法推進的 token 保留原 logit，其他設為 -inf，再 sample。

## 實作選擇（2026 現狀）

| Provider | 機制 | 推薦度 |
|---|---|---|
| Anthropic | `tools=[...]` tool use（implicit） | ⭐ 強推，預設開啟 |
| OpenAI | `response_format={"type": "json_schema", "strict": True}` | ⭐ GPT-4o+ 可用 |
| Google Gemini | `responseSchema` | 支援 |
| Self-hosted (vLLM / TGI) | [Outlines](https://github.com/dottxt-ai/outlines) | 開源首選 |

## Anthropic tool use 寫法

```python
response = client.messages.create(
    model="claude-haiku-4-5",
    tools=[{
        "name": "extract_patent_params",
        "description": "Extract patent parameters from text",
        "input_schema": {
            "type": "object",
            "properties": {
                "substrate_temp_C": {"type": "number"},
                "oxygen_pressure_mTorr": {"type": "number"},
                "material": {"type": "string"},
            },
            "required": ["material"]
        }
    }],
    messages=[{"role": "user", "content": text}],
)
# response.content[0].input 保證符合 schema
```

## 核心屬性

- **格式正確性保證；事實正確性不保證**。model 會輸出 `{"substrate_temp_C": 700}` 但 700 是不是對的是另一回事
- **消滅一整類 bug**：再也不會半夜被 「model 返回 95% 合法 JSON 但這筆壞掉」叫醒。下游 code 不用再寫 defensive try/except
- **某些情況下會有品質下降**：schema 與 model 自然表達衝突時，heavily constrained 會比 parse-on-output 差。**要在目標 task 實測**
- **會為了滿足 schema 而幻覺**：若 required field 在來源沒提，model 會硬編一個。**解法：把 field 設 optional，允許 null**

## 業界應用

- Anthropic Claude Agent SDK 所有 tool call 都走 constrained decoding
- OpenAI GPT-4o Structured Outputs 企業級 JSON endpoint 大量使用
- Outlines + vLLM 是 self-hosted 抽取服務事實標準
- LangChain / LlamaIndex output parser 優先選底層支援的 API

## 什麼時候用 / 不用

**一定用**：
- 輸出進下游 structured system（DB insert, API call, analytics）
- format 錯誤佔 debug 時間大宗
- 用 Anthropic / OpenAI — 免費開關

**考慮不用**（unconstrained + parse）：
- 輸出是含結構的 free-form prose（報告、帶引用的摘要）
- 想要 CoT reasoning 在 structured output 之前
- Self-host model 對效率 constrained decoding 還沒支援

## Patsnap 落地

- **所有 Claude-based extractor 預設用 tool use**，不用 prompt-parsed JSON。永遠消除這類 parse error
- **Nested schema 表達 claim 結構**（tool use 原生支援）
- **Optional / nullable field**：不存在就不存在，別讓模型幻覺補值

## 與其他策略的關係

- 跟 [self-consistency](notes_self_consistency_impl.md) 互補：format vs fact
- 跟 [retrieval-augmented verification](notes_retrieval_augmented_verification.md) 互補：schema 要求 `{value, supporting_span}` 配對，decoder 保證雙方都出
