# Verifier Model 筆記

> KB 正本：[verifier-model](~/PycharmProjects/llm_knowledge_base/wiki/concepts/verifier-model.md)
> 論文：[Cobbe et al. 2021](https://arxiv.org/abs/2110.14168)、[Lightman et al. 2023 (PRM)](https://arxiv.org/abs/2305.20050)

## 一句話

訓一個小 classifier 預測「LLM 這個輸出對不對」。有了標註資料後，取代 LLM-as-Judge，推理成本低 1-2 個數量級。OpenAI o-series 的核心訓練組件。

## 兩種子類

- **ORM (Outcome Reward Model)**：只看最終答案（Cobbe 2021, GSM8K 始祖）
- **PRM (Process Reward Model)**：逐步打分，支援 step-level correction（Lightman 2023）。抽取場景對應「逐欄位」打分

## 訓練流程

1. 收集 `(input, LLM output, is_correct)` triples
   - 來源：人工標註、self-consistency majority vote、ground-truth execution（code 執行）
2. Fine-tune 小 encoder（DeBERTa-v3-base / ModernBERT）為 binary classifier
3. 在 held-out 上做 calibration（verifier confidence vs 真實正確率）

## 推理模式：Best-of-N + Verifier

```python
candidates = [llm.generate(input, seed=s) for s in range(K)]
scored = [(c, verifier.score(input, c)) for c in candidates]
best = max(scored, key=lambda x: x[1])
```

2023 以後多數 reasoning benchmark 的 SOTA 都靠這個組合。

## 與替代方案成本對比

| | Verifier | LLM-as-Judge | Self-Consistency |
|---|---|---|---|
| 要標註資料？ | **要** | 不要 | 不要 |
| 單次推理成本 | ~$0.0001 (小 encoder) | ~$0.01 (強 LLM) | ~$0.005 × N |
| Bias profile | 訓練分布偏移 | position/verbosity/self-preference | 模型自身錯誤 N 倍 |
| 適用流量 | 高流量 production | 中流量 eval | 低流量高風險 |

## 業界應用

- **OpenAI o-series**（PRM 訓練 + 推理兩階段都用）
- **DeepMind AlphaCode 2**（verifier 過濾百萬候選程式）
- **Anthropic Constitutional AI / RLAIF**
- **LangSmith / Arize evaluator**（productized）
- **Patsnap**：累積 10k+ 人工修正的抽取記錄後，訓練 DeBERTa-base verifier 做 online quality gate

## 什麼時候用 / 不用

**用**：
- ≥5k 標註樣本累積完成（self-consistency gold set 算數）
- 流量 >10k call/day（訓練成本能 amortize）
- 要版控的品質訊號（verifier weights = git artifact；LLM judge 會隨 API 漂移）

**不用**：
- 還沒標註資料（先用 self-consistency 取得）
- Domain 快速演化（retrain cadence 跟不上價值）
- 低流量

## 主要失敗模式

- **Distribution shift 是 #1**：2024 專利訓的 verifier 在 2026 專利會默默 miscalibrate。Mitigation：periodic retraining + active learning loop
- **Calibration ≠ accuracy**：90% accuracy 的 verifier 可能 confidence 0.9 卻只對 70%。Reliability diagram 必做
- **Process > Outcome** for 多欄位抽取：逐欄位打分贏過「整份 JSON 對不對」的單一訊號

## Patsnap 路線圖

Month 0-3：self-consistency 累積 gold set
Month 3-6：self-consistency gold + 人工 QA → 第一版 verifier
Month 6+：verifier 取代 online self-consistency，成本 10× 下降
