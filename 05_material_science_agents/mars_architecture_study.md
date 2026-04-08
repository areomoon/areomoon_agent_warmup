# MARS 系統架構研究筆記

**論文：** Knowledge-driven autonomous materials research via collaborative multi-agent and robotic system
**新聞報導：** https://phys.org/news/2026-01-multi-agent-ai-robots-automate.html
**關鍵詞：** closed-loop, 19 agents, 16 domain tools, materials discovery

---

## 系統概覽

MARS (Multi-Agent Research System) 是目前最完整的閉環材料研究自動化系統。
核心設計：**多個 LLM 專家 agents** 透過結構化通訊協作，而非單一 LLM 包辦一切。

```
Hypothesis → Simulation/Experiment → Analysis → Refined Hypothesis
     ↑                                                    |
     └────────────────── closed loop ───────────────────┘
```

---

## 19 個 LLM Agents 分工

### Tier 1: Orchestration
| Agent | 職責 |
|-------|------|
| **Orchestrator** | 任務分解、agent 調度、結果整合 |
| **Planner** | 制定實驗計劃、決定下一步行動 |

### Tier 2: Scientific Reasoning
| Agent | 職責 |
|-------|------|
| **Scientist** | 高層次科學假設生成，基於領域知識 |
| **Hypothesis Generator** | 提出具體材料候選（組成、結構、摻雜） |
| **Literature Reviewer** | 搜尋和摘要相關文獻 |
| **Safety Officer** | 檢查化學安全性（毒性、爆炸風險） |

### Tier 3: Engineering & Execution
| Agent | 職責 |
|-------|------|
| **Engineer** | 設計具體合成/實驗流程 |
| **Synthesis Planner** | 詳細合成步驟規劃 |
| **Executor** | 呼叫工具執行計算或實驗 |
| **Error Handler** | 捕捉和處理工具執行錯誤 |

### Tier 4: Analysis
| Agent | 職責 |
|-------|------|
| **Analyst** | 解析實驗/計算結果 |
| **Property Predictor** | 預測材料性質（ML 代理模型） |
| **Characterization Analyst** | 解讀 XRD/SEM/Raman 等表徵數據 |
| **Data Validator** | 驗證數據一致性和物理合理性 |

### Tier 5: Knowledge Management
| Agent | 職責 |
|-------|------|
| **Knowledge Curator** | 更新材料知識庫（← ACE Curator 類比） |
| **Strategy Updater** | 根據結果更新搜索策略 |
| **Report Writer** | 撰寫結構化研究報告 |
| **Review Agent** | 最終品質審查 |

---

## 16 個 Domain Tools

### 計算工具
- `DFT Calculator` (VASP wrapper) — 第一性原理計算
- `ML Force Field` (MACE/CHGNet) — 快速分子動力學
- `CALPHAD Engine` — 相圖計算
- `Crystal Structure Predictor` — 結構預測（DiffCSP）

### 資料庫 API
- `Materials Project API` — 50,000+ 已知材料性質
- `ICSD` (Inorganic Crystal Structure Database)
- `COD` (Crystallography Open Database)
- `Springer Materials`

### 分析工具
- `XRD Analyzer` — Rietveld 精修
- `Spectra Interpreter` — Raman/FTIR 峰值識別
- `SEM/TEM Image Analyzer` (Vision LLM)
- `Property Calculator` — 基於成分的快速估算

### 工作流工具
- `Synthesis Optimizer` — 貝葉斯優化合成條件
- `Experiment Scheduler` (連接物理機器人)
- `Data Logger` — 結構化數據存儲
- `Literature Search` (Semantic Scholar API)

---

## 關鍵設計決策

### 1. 結構化 handoff（不傳 raw text）
Agents 之間傳遞 **結構化 JSON**，不是自然語言。
這消除了 inter-agent 的幻覺放大問題。

```json
// Good: structured handoff
{"material": "La₀.₈Sr₀.₂MnO₃", "T_substrate_C": 700, "confidence": 0.95}

// Bad: natural language handoff  
"I found that the LSMO material was deposited at around 700 degrees..."
```

### 2. Shared State Store
所有 agents 讀寫一個共同的 state store，不是點對點通訊。
類比：Git repo 的 shared state，而非 email chain。

### 3. Safety Gate
Safety Officer agent 是 hard gate — 任何計劃必須通過安全審查才能執行。
這對實驗室自動化尤其重要（有毒化學品、爆炸風險）。

---

## 與 ACE GRC 的對應

| MARS 組件 | ACE 角色 | 說明 |
|-----------|---------|------|
| Hypothesis Generator + Engineer | **Generator** | 提出並規劃實驗 |
| Analyst + Data Validator | **Reflector** | 分析結果，識別錯誤 |
| Knowledge Curator + Strategy Updater | **Curator** | 更新知識庫和策略 |

**關鍵差異：** MARS 沒有持久化 playbook；每次 session 的學習不會自動積累。
這是 ACE 在 MARS 架構上的主要改進點。

---

## 入職應用筆記

你的工作是 **extraction-focused**，不是 discovery。所以：

- 用 MARS 的 Analyst + Data Validator 概念 → 你的 Reflector
- 用 MARS 的 Knowledge Curator 概念 → 你的 ACE Curator（要加 playbook）
- 暫時不需要 DFT/CALPHAD tools → 你的 tools 是 PDF parser + DB API

**第一個月的 MVP：** Orchestrator + Extractor + Validator，3 個 agents 就夠了。
