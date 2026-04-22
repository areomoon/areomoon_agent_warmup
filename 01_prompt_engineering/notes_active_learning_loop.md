# Active Learning Loop 筆記

> KB 正本：[active-learning-loop](~/PycharmProjects/llm_knowledge_base/wiki/concepts/active-learning-loop.md)
> 參考：[Settles 2010 survey](https://burrsettles.com/pub/settles.activelearning.pdf)、[Snorkel (Ratner 2017)](https://arxiv.org/abs/1711.10160)

## 一句話

把低信心樣本路由給人工標註 → 用校正結果回訓 model / 更新 few-shot / 調整 prompt。**把 confidence-aware pipeline 變成會長進的系統**，而不是 static。

## 這是 self-consistency 的 **互補**，不是替代

- Self-consistency 產生訊號
- Active learning metabolize 低信心樣本變改進

## 循環五步驟

1. **Run** 抽取，附上 confidence 分數（來自 self-consistency / verifier / logprob）
2. **Route** 低信心樣本給人工 reviewer（最資訊量大的樣本）
3. **Capture** 校正進 labelled dataset
4. **Update** 系統（retrain verifier / refresh few-shot / adjust prompt / fine-tune base model）
5. **Redeploy** 再循環

## 選擇策略（哪些樣本送人審）

- **Uncertainty sampling** —— 信心最低優先。最簡單，預設起點
- **Query-by-committee** —— ensemble 成員分歧大的。利用 self-consistency 或 model-ensembling 輸出
- **Expected model change** —— 預期標註後對 model 改變最大的
- **Diversity sampling** —— 跨 domain / 文件類型分層，避免過度偏向單一技術領域

## 系統架構

```
Inputs → Extractor → Confidence Router
                          │
               ┌──────────┼──────────┐
               ▼          ▼          ▼
          auto-accept  LLM-Judge  human queue
                                      │
                                      ▼
                               Annotation UI (Argilla)
                                      │
                                      ▼
                               Labelled DB
                                      │
               ┌──────────────────────┘
               ▼
   retrain verifier / refresh few-shot / fine-tune base
```

## 核心屬性

- **標註預算是 binding constraint，不是 compute**。Expert annotator（專利律師、材料科學家）$50-200/hr；好的標註 UI + 好的 selection strategy 比 model 精巧重要
- **Drift detection 免費附贈**：低信心量 spike = 新術語 / 新文件類型到了
- **Fine-tune 在這裡才真的 pay off**：沒 active learning loop 就 fine-tune 通常 premature，沒 data flywheel
- **標註 UI 的 UX 乘數 3-5×**：context-rich 介面（source span 標亮、sibling extraction 可見、one-click accept-with-edit）

## 業界工具

- **Snorkel Flow, Humanloop, Argilla, Label Studio, Scale AI** —— 把 loop 產品化
- **Anthropic, OpenAI, DeepMind** 內部都跑持續標註；frontier model 訓練本身就是一個巨型 active learning loop
- **GitHub Copilot**：低信心 suggestion 與 accept/reject 訊號餵下一輪訓練
- **醫學 / 法律抽取平台**：active learning 通常是唯一經濟可行路 —— 全 supervised 太貴

## 什麼時候用 / 不用

**用**：
- 已有 confidence 訊號的 production extractor
- Expert annotation 預算稀缺要精準投放
- Horizon 長（6+ 月持續運營，改進會 compound）
- Domain 會演化（新專利領域、新術語）—— drift detection 是 built-in

**不用**：
- 一次性 static batch（不需持續改進）
- 還沒 confidence 訊號（先建那個）
- Team 人手不夠撐標註 UI + pipeline ops（半蓋的 loop 比不蓋更糟：data 爛掉、更新到不了 production）

## Patsnap 六個月落地

- **Month 1-2**：ship self-consistency + retrieval verification；所有低信心 item dump 進 table
- **Month 2-4**：Argilla（或同類）+ per-domain 標註 rubric；從 Claim 1 參數開始
- **Month 4-6**：每週刷新 few-shot example pool；per-domain confidence timeseries 上 dashboard
- **Month 6+**：annotated corpus ~5k-10k per domain 且 prompt 救不了的殘留錯誤明顯時，才重新考慮 fine-tune

## 和其他策略的關係

- 和 [self-consistency](notes_self_consistency_impl.md) 配對：一個產訊號一個消化
- 和 [verifier model](notes_verifier_model.md) 配對：loop 餵 verifier 的訓練資料
- 和 [cost-aware cascade](~/PycharmProjects/llm_knowledge_base/wiki/concepts/cost-aware-cascade-design.md) 是同一條路的不同切面
