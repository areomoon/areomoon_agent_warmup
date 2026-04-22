# 抽取 Pipeline 複合策略總覽

> Week 1 筆記。對應 notebook `prompt_engineering_lab.ipynb` 的五種做法比較 + 進階產業做法。
> KB 英文正本：[Extraction Pipeline Composite Strategies](~/PycharmProjects/llm_knowledge_base/wiki/derived/2026-04-23-extraction-pipeline-composite-strategies.md)

## 閱讀順序（Week 1 筆記導覽）

### Phase 1 — 先跑 code 建體感
1. `notebook/prompt_engineering_lab.ipynb` — 跑一次看 5 種方法輸出差異
2. `README.md` — 週 1 目的

### Phase 2 — 全景
3. **本檔** ⭐ — 三個複合策略 + 導航表

### Phase 3 — 核心三本（Patsnap 入職直接會用）
4. [notes_self_consistency_impl.md](notes_self_consistency_impl.md) ⭐ — 最重要，六個坑 + 系統圖
5. [notes_constrained_decoding.md](notes_constrained_decoding.md) — Anthropic tool use，抽取 pipeline 預設
6. [notes_retrieval_augmented_verification.md](notes_retrieval_augmented_verification.md) — 專利必要可追溯性

### Phase 4 — 互補工具（了解取捨）
7. [notes_llm_as_judge.md](notes_llm_as_judge.md) — self-consistency 路由中段的夥伴
8. [notes_verifier_model.md](notes_verifier_model.md) — 累積標註資料後的升級路徑
9. [notes_active_learning_loop.md](notes_active_learning_loop.md) — 把前面串成「會長進的系統」

### Phase 5 — 邊界 / 情境限定（知道就好）
10. [notes_self_refine_critic_loop.md](notes_self_refine_critic_loop.md) — 推理題用，抽取用處邊際
11. [notes_model_ensembling.md](notes_model_ensembling.md) — 極高價值低流量才用
12. [notes_logprob_uncertainty.md](notes_logprob_uncertainty.md) — Anthropic 不給，Claude-only 當知識

### Phase 6 — ReAct 相關（獨立主線）
13. [notes_react.md](notes_react.md) — ReAct pattern
14. [notes_trajectory.md](notes_trajectory.md) / [notes_trajectory_refactor.md](notes_trajectory_refactor.md)

**入職前兩週衝刺版**：只讀 3 → 4 → 5 → 6 → 9（五本）就能應付前三個月 Patsnap 工作，其他碰到再查。

---

## 五種基礎方法對比

| 方法 | 輸出 | 決定性 | 單次成本 | 中間產物 |
|---|---|---|---|---|
| Zero-shot | JSON | 高 (T=0) | 1× | 無 |
| Zero-shot CoT | 推理 + JSON | 高 | 1× | 文字 reasoning |
| Few-shot CoT | 推理 + JSON | 高 | 1× (prompt 最長) | domain-shaped reasoning |
| Self-Consistency | N 份 JSON → 投票 + 每欄位 confidence | 低 (T>0) | N× | agreement distribution |
| ReAct | Thought/Action/Observation trace | 中 | 多 round | tool-call trace |

記憶重點：
- **Zero-shot CoT = CP 值最高的升級**（一句話換顯著進步）
- **只有 Self-Consistency 產生 per-field confidence**（production router 真正需要的東西）
- **ReAct prompt-parsed 版本脆弱** — 今天踩的 `StopIteration` 就是例證；正式環境用 Anthropic tool use，不要 regex-on-text

## 三個複合策略（Patsnap 情境）

### 策略 1：Zero-shot CoT 預設 + 低信心欄位 fallback 到 Self-Consistency

**情境**：專利說明書參數抽取（組成、製程、效能），一份 10-50 欄位。

**做法**：90% 欄位一次 zero-shot CoT 抽完；觸發 N=5 self-consistency 的條件：
1. 模型自述 confidence low（便宜但不可盡信）
2. 輸出啟發式：空欄位、單位缺失、數值超出合理範圍
3. 規則閘：業務定義的高風險欄位（如 Claim 1 核心參數）

**成本**：~1.3-1.5×，而非全 self-consistency 的 5×。

### 策略 2：Few-shot CoT 讀懂 + tool use 做計算

**情境**：跨公司電池能量密度比較（Wh/kg vs mAh/g × V 單位換算）。

**做法**：
- 抽取層：LLM 只讀出原值與原單位，不做任何換算
- 計算層：deterministic Python function 做單位轉換、聚合

**為什麼**：LLM 算數字會錯。錯誤可歸因（讀錯 vs 算錯），計算層可 unit test。

### 策略 3：離線 Self-Consistency 建 gold set + 線上用輕量方法

**情境**：上新 extractor 無人工標註預算。

**做法**：
- **Phase 1 一次性**：500-2000 份代表性樣本 × N=10 × 最強模型（Opus/Sonnet）→ high agreement 當 silver label，low agreement 送 domain expert 最小審核
- **Phase 2 線上**：Haiku + zero-shot CoT，每週 vs gold set 計 F1/accuracy
- **Phase 3 持續**：accuracy 衰退 → 校準

**業界名稱**：weak supervision / LLM-as-annotator（Snorkel, Argilla）

## 九個 topic（每個有獨立筆記檔）

| Topic | Chinese note | KB concept |
|---|---|---|
| Self-Consistency 實作 | [notes_self_consistency_impl.md](notes_self_consistency_impl.md) | [self-consistency-implementation](~/PycharmProjects/llm_knowledge_base/wiki/concepts/self-consistency-implementation.md) |
| LLM-as-Judge | [notes_llm_as_judge.md](notes_llm_as_judge.md) | [llm-as-judge](~/PycharmProjects/llm_knowledge_base/wiki/concepts/llm-as-judge.md) |
| Verifier Model | [notes_verifier_model.md](notes_verifier_model.md) | [verifier-model](~/PycharmProjects/llm_knowledge_base/wiki/concepts/verifier-model.md) |
| Constrained Decoding | [notes_constrained_decoding.md](notes_constrained_decoding.md) | [constrained-decoding](~/PycharmProjects/llm_knowledge_base/wiki/concepts/constrained-decoding.md) |
| Retrieval-Augmented Verification | [notes_retrieval_augmented_verification.md](notes_retrieval_augmented_verification.md) | [retrieval-augmented-verification](~/PycharmProjects/llm_knowledge_base/wiki/concepts/retrieval-augmented-verification.md) |
| Self-Refine / Critic Loop | [notes_self_refine_critic_loop.md](notes_self_refine_critic_loop.md) | [self-refine-critic-loop](~/PycharmProjects/llm_knowledge_base/wiki/concepts/self-refine-critic-loop.md) |
| Model Ensembling | [notes_model_ensembling.md](notes_model_ensembling.md) | [model-ensembling](~/PycharmProjects/llm_knowledge_base/wiki/concepts/model-ensembling.md) |
| Logprob Uncertainty | [notes_logprob_uncertainty.md](notes_logprob_uncertainty.md) | [logprob-uncertainty](~/PycharmProjects/llm_knowledge_base/wiki/concepts/logprob-uncertainty.md) |
| Active Learning Loop | [notes_active_learning_loop.md](notes_active_learning_loop.md) | [active-learning-loop](~/PycharmProjects/llm_knowledge_base/wiki/concepts/active-learning-loop.md) |

## Patsnap 入職前三個月推薦組合

1. **Tool use + JSON schema**（格式穩定 + 計算/抽取分層）
2. **Retrieval-augmented verification**（抽取值回到原文 span，專利必要可追溯性）
3. **Self-Consistency** 只用在關鍵欄位 + 離線評測；線上全上成本會爆
4. **Fine-tune 先緩**：self-consistency gold set + retrieval verification 就能讓 zero-shot CoT 到可出貨品質；等看清系統性錯誤 prompt 救不了再 fine-tune

## 一句話總結

**Self-Consistency 的核心價值不是「投票選答案」，而是「產生 per-field confidence 讓你能做路由」** — 這是它比其他方法更適合建 production pipeline 的關鍵。
