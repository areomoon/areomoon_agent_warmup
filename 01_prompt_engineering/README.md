# 01 Prompt Engineering

**對應 Warmup Week 1 | 預估時數：8–10hr**

## 學習目標

在不修改模型權重的前提下，最大化 LLM 對科學文本的抽取能力。

完成後你能：
- [ ] 用 Zero-shot / Few-shot / CoT 對同一段科學文本做比較測試
- [ ] 實作 ReAct pattern 的基礎 agent loop
- [ ] 用 Self-consistency 提高數值抽取的可靠性
- [ ] 輸出符合 JSON schema 的結構化實驗參數

## 執行順序

1. `python chain_of_thought.py` — 理解 CoT 對科學推理的幫助
2. `python self_consistency.py` — 用多數決提高數值抽取準確率
3. `python react_pattern.py` — 看 Reasoning + Acting 的基礎循環
4. 打開 `notebook/prompt_engineering_lab.ipynb` — 互動實驗

## 關鍵概念

| 技術 | 一句話說明 | 適合場景 |
|------|-----------|---------|
| Zero-shot CoT | 加 "Let's think step by step" | 快速推理 |
| Few-shot | 給 2–3 個示例再要求抽取 | 格式固定的欄位 |
| Chain-of-Thought | 明確要求逐步推理 | 複雜推理、多步計算 |
| Self-Consistency | 多次生成 + 多數決 | 需要高可靠性的數值 |
| ReAct | Reason → Act → Observe 循環 | Agent 設計核心 |
| Structured Output | JSON mode / function calling | 程式可解析的輸出 |
