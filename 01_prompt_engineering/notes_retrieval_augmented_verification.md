# Retrieval-Augmented Verification 筆記

> KB 正本：[retrieval-augmented-verification](~/PycharmProjects/llm_knowledge_base/wiki/concepts/retrieval-augmented-verification.md)
> 參考：[Gao et al. 2023 (citation)](https://arxiv.org/abs/2305.14627)、[RAGAS (Es et al. 2023)](https://arxiv.org/abs/2309.15217)

## 一句話

抽取完每個值，回到原文驗證它真的存在（verbatim 或 semantic）。**這是 self-consistency 抓不到的失敗模式**：N 個樣本全都同意一個幻覺值，因為 model 的 prior 憑空編的。

## 三階段實作（由便宜到貴）

| Tier | 機制 | 成本 | 覆蓋範圍 |
|---|---|---|---|
| 1 | Exact match（normalize 後） | ~免費 | 漏 paraphrase |
| 2 | Embedding similarity on spans | ~$0.0001/field | 抓 paraphrase |
| 3 | 第二 pass LLM 驗證 | ~$0.001-0.01/field | 最高，支援推理 |

**起手：Tier 1。** Tier 1 漏掉的才升 Tier 2；Tier 2 漏掉高風險再升 Tier 3。

## Tier 3 citation-span pattern

```python
extraction = llm.extract(
    patent_text,
    schema_requires="value + supporting_span",
)
for field, v in extraction.items():
    span = v["supporting_span"]
    if span not in patent_text:
        flag(field, "hallucinated span")
    elif not contains_value(span, v["value"]):
        flag(field, "span does not support value")
    else:
        accept(field, v)
```

## 為什麼重要：捕 shared-prior 幻覺

這是最危險的失敗模式 —— **高 confidence + 錯誤答案**。

例：model 的 prior 覺得 PLD 氧壓常是 200 mTorr。即使專利寫 350 mTorr（或沒寫），N 個 self-consistency 樣本可能全給 200 mTorr。
- Self-consistency 說 agreement 1.0 → 高 confidence
- **Retrieval verification 抓得到**：宣稱的 span 要嘛不存在要嘛不含 200

## 業界應用

- **Anthropic Citations API (2025)**：原生 passage-level citation
- **Perplexity / You.com**：每句答案連結 source URL，是 UX 契約
- **Harvey / Casetext**（legal tech）：案例引用必須先驗證
- **Elicit / Consensus**（醫學證據抽取）：verification against source paper 是核心產品
- **Patsnap 是這個模式最關鍵的應用場景** —— 客戶拿抽取資料做 IP 訴訟/投資決策，span 可追溯性是 legal defensibility 的必要條件

## 什麼時候用 / 不用

**強制用**：
- 監管 / 法律要求可追溯（專利、醫學、金融、法律）
- 下游用戶會挑戰 output 真實性
- Source 很長且特定（專利平均 20-50 頁，記憶錯是真實失敗模式）

**可選用**：
- Output 是摘要 / paraphrase（沒有值對應）
- Source 很短且已在 context，幻覺風險低

## Patsnap 落地

- **Claim 1 抽取強制加 span**
- 和 [constrained-decoding](notes_constrained_decoding.md) 配合：schema 強制 `{value, supporting_span}` 配對
- **Span 存進 output record**，變 audit trail 給下游搜尋/分析工具 link 回去
- **Faithfulness 做 production KPI**（不是只看 accuracy）—— 用 RAGAS 或同類工具每月跑

## 跟 self-consistency 的關係

**完全互補**：
- Self-consistency 抓「隨機不一致」
- Retrieval verification 抓「一致但幻覺」
- **Patsnap 生產 pipeline 兩個都要有**
