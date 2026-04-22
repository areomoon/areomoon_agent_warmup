# LLM-as-Judge 筆記

> KB 正本：[llm-as-judge](~/PycharmProjects/llm_knowledge_base/wiki/concepts/llm-as-judge.md)
> 基礎論文：[Zheng et al. 2023 — MT-Bench / Chatbot Arena](https://arxiv.org/abs/2306.05685)

## 一句話

用另一個（通常更強的）LLM 給別人的答案打分/排序/裁決。比 self-consistency 便宜（1 call vs N），但會引入 **judge bias**，必須處理。

## 三種形態

1. **Pointwise scoring** — 打 1-10 分
2. **Pairwise preference** — A 還是 B 好？
3. **Rubric-based ruling** — 依照明確 criteria 判決（抽取情境最實用：「這個欄位的值，source text 有支持嗎？」）

## 程式模式

```python
# Cheap extract + strong judge
extraction = haiku.extract(patent_text)
judge_prompt = f"""
Source: {patent_text}
Extracted: {extraction}
每一欄位判斷: 值是否由 source 支持？（yes / no / partial），一句話說明。
Return JSON.
"""
rulings = sonnet.judge(judge_prompt)
```

## 三大 judge bias（面試必考）

1. **Position bias** — 判斷偏向第一個（或第二個，看 model）。處理：評 A-vs-B 再評 B-vs-A，都同意才算贏
2. **Verbosity bias** — 長答案被打高分即使內容一樣。處理：rubric 內明確要求 length-normalize，或把候選 truncate 到同長度
3. **Self-preference** — 模型偏好同家族的輸出（GPT-4 偏 GPT-4, Claude 偏 Claude）。處理：用不同家族 judge，或 ensemble

其他風險：
- **Judge 版本更新 → calibration drift**。把 judge model version pin 起來
- **Judge 在不熟悉的 domain 會默默失誤**（例如不知道 200 mTorr 對 PLD 合理）。用 domain-specific rubric + few-shot demo

## 業界應用

- Anthropic / OpenAI 內部 eval pipeline
- Ragas / RAG faithfulness scoring
- RLAIF（Anthropic Constitutional AI 用 LLM-as-Judge 標 preference）
- PR 自動審查 bot
- Patsnap：專利欄位抽取後 verify supporting span

## vs Self-Consistency

| | Self-Consistency | LLM-as-Judge |
|---|---|---|
| Calls/input | N (通常 5) | 1 judge + 1 extract |
| Confidence 來源 | 樣本 agreement | Judge 的顯式打分 |
| Bias 風險 | Model 自身系統性錯誤被放大 N 倍 | Position/verbosity/self-preference |
| 擅長 | Ambiguous reasoning | Source-consistency、pairwise |

## 什麼時候用 / 不用

**用**：需要 pairwise preference；成本不允許 N× 但允許 1× 強 model；task 是「判對錯」而非「想出答案」

**不用**：沒有明顯更強的 judge（self-preference 崩壞）；rubric 需要 domain expert knowledge（用 verifier model）；regulatory 要人類決策留底

## 在 Self-Consistency pipeline 的位置

Self-consistency 的三段式 router 中間段（0.5 ≤ conf < 0.8）就是 LLM-as-Judge 最自然的位置。
