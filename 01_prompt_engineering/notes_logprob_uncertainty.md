# Logprob Uncertainty 筆記

> KB 正本：[logprob-uncertainty](~/PycharmProjects/llm_knowledge_base/wiki/concepts/logprob-uncertainty.md)
> 參考：[Jiang 2021 calibration](https://arxiv.org/abs/2012.00955)、[Kadavath 2022 "models know what they know"](https://arxiv.org/abs/2207.05221)

## 一句話

讀每個 token 的 log-probability 當 confidence 訊號。一次 forward 同時拿到 output 和 uncertainty。**但 Anthropic Claude API 不給 logprobs**，這條路對 Claude-only stack 關閉。OpenAI、self-hosted 可用。

## 誰給 / 誰不給（2026 狀況）

| Provider | logprobs 可用？ |
|---|---|
| OpenAI (`logprobs=True, top_logprobs=5`) | ✅ |
| Self-hosted（vLLM, TGI, Ollama） | ✅（全分布） |
| Anthropic Claude | ❌ **產品決策，不給** |
| Google Gemini | 部分（某些 API 有 `avgLogprobs`） |

## 寫法（OpenAI）

```python
resp = openai.chat.completions.create(
    model="gpt-5",
    messages=[...],
    logprobs=True,
    top_logprobs=5,
)
tokens = resp.choices[0].logprobs.content
# 最粗：整段平均
confidence = sum(t.logprob for t in tokens) / len(tokens)

# 實用：per-field（把 token 對齊到 JSON 欄位）
field_spans = locate_field_spans(extracted_json, tokens)
per_field_conf = {
    field: exp(mean(t.logprob for t in span))
    for field, span in field_spans.items()
}
```

## vs Self-Consistency

| | Logprob | Self-Consistency N=5 |
|---|---|---|
| Calls | 1 | 5 |
| Confidence 來源 | Token probabilities | 樣本 agreement |
| 粒度 | Per-token, per-field | Per-field |
| Claude 可用？ | ❌ | ✅ |
| Calibration | Model-dependent，常極端 miscalibrated | Vote-share 非機率 |

## 核心屬性

- **「免費」**—— 單次 forward 同時拿 output + uncertainty，比任何 sampling-based 方法便宜一個數量級
- **Miscalibration 是主要失敗模式**：RLHF 後的 model 常 overconfident-on-wrong。正式使用前必做 calibration（temperature scaling / isotonic regression on 一組 held-out）
- **抓不到 prior-driven 幻覺**：model 自信編出來的值，logprobs 很高。必須搭配 [retrieval verification](notes_retrieval_augmented_verification.md)
- **Per-field aggregation 是關鍵工藝**：raw token logprob 很 noisy，聚合到 schema field span（用 JSON parsing + token 對齊）才好用

## 業界應用

- **OpenAI 內部 eval**：logprobs 是 calibration 研究的 backbone
- **Self-hosted 抽取 stack**：vLLM + 自製 aggregation = 開源最便宜的 confidence pipeline
- **HuggingFace `evaluate` / `lm-eval-harness`**：ARC, MMLU 等 benchmark 用 logprob-based metric
- **Arize / Langfuse**（OpenAI-compatible API）顯示 per-token logprob 作 debug 輔助
- **Patsnap**（封鎖）：Claude-only stack 就不能用。若 OpenAI 或 self-hosted 進場，day 1 打開 logprobs —— 校準後 10× 便宜於 self-consistency 且同品質

## 什麼時候用 / 不用

**用**：API 暴露 logprobs（OpenAI / self-hosted）；高流量（單次 call 贏 N 倍 self-consistency）；能投入 calibration 驗證

**不用 / 用不了**：Claude-only（→ self-consistency 當工作替代）；失敗模式是 prior hallucination（→ retrieval verification）；沒 eval set 做 calibration（未校準的 logprob 當 router threshold 會誤導）

## Patsnap 特注

如果 production 鎖定 Claude，這條路完全不可用 —— self-consistency + retrieval verification + LLM-as-Judge 要扛起 confidence 訊號的角色。若有 OpenAI / self-hosted extractor 加入，重新評估 —— logprobs 校準後能 10× 降成本。
