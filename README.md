# areomoon_agent_warmup

**目標角色：** 材料科學 AI Agent 算法工程師（模型微調 + 算法設計）
**入職時間：** 2026 年 5 月
**核心任務：** 從 long-context multi-modal 科學文獻抽取實驗資料，輔助科學家決策

> 硬體限制：Mac (Apple Silicon) — 所有大型模型訓練使用雲端 API 或 Colab/Kaggle；本地僅跑小規模推理。

---

## 學習路線圖

```
Phase 0（入職前）: Warmup — 6 週掌握 LLM → RAG → Agent → Fine-tuning
Phase 1（第1-2月）: 實戰 Generator-Reflector extraction pattern
Phase 2（第3-4月）: QLoRA fine-tuning + evaluation benchmark
```

### 模組總覽

| 模組 | 主題 | 對應 Warmup Week | 預估時數 |
|------|------|-----------------|---------|
| [01 Prompt Engineering](#01-prompt-engineering) | CoT、ReAct、Self-consistency、Structured Output | Week 1 | 8–10hr |
| [02 RAG Fundamentals](#02-rag-fundamentals) | 基礎 RAG、Embedding、科學論文問答 | Week 2 | 10–12hr |
| [03 Agent Patterns](#03-agent-patterns) | Generator-Reflector、LangGraph、Multi-agent | Week 3 | 10–12hr |
| [04 ACE Framework](#04-ace-framework) | Playbook evolution、Curator delta updates | Week 3–4 | 6–8hr |
| [05 Material Science Agents](#05-material-science-agents) | MARS、LLMatDesign、Extraction Agent | Phase 1 | 持續 |
| [06 Fine-tuning](#06-fine-tuning) | LoRA、QLoRA、SFT、Evaluation | Week 5 | 12–15hr |
| [07 Multimodal](#07-multimodal) | 圖表理解、PDF multi-modal、Vision LLM | Week 4 | 8–10hr |

---

## 01 Prompt Engineering

**學習目標：** 在不動模型權重的情況下最大化 LLM 的科學文本抽取能力

```
01_prompt_engineering/
├── chain_of_thought.py       # Zero-shot CoT vs. Manual CoT 在科學文本的差異
├── react_pattern.py          # Reasoning + Acting 的基礎實作
├── self_consistency.py       # 多路徑推理 + 多數決取最終答案
└── notebook/
    └── prompt_engineering_lab.ipynb   # 互動實驗：同一段文本用不同 prompting 策略
```

**關鍵資源：**
- [Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [ReAct: Synergizing Reasoning and Acting (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Self-Consistency Improves CoT (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)

---

## 02 RAG Fundamentals

**學習目標：** 能對科學論文 PDF 做問答並取回結構化實驗參數

```
02_rag_fundamentals/
├── basic_rag.py              # LlamaIndex 基礎 RAG pipeline（PDF → chunk → embed → retrieve → generate）
├── embedding_search.py       # Embedding 模型比較（BGE-M3 vs. text-embedding-3-small）
└── notebook/
    └── rag_lab.ipynb         # 用真實論文做問答實驗
```

**關鍵資源：**
- [RAG for Knowledge-Intensive NLP (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [LlamaIndex 官方教學](https://docs.llamaindex.ai/)

---

## 03 Agent Patterns

**學習目標：** 掌握 LangGraph + Generator-Reflector pattern；建出最小 extraction agent

```
03_agent_patterns/
├── generator_reflector.py    # ACE Generator-Reflector pattern 核心實作
├── reflection_agent.py       # LangGraph reflection agent（自我修正循環）
├── multi_agent_basic.py      # Orchestrator + Specialist agents 基礎
└── notebook/
    └── agent_patterns_lab.ipynb
```

**關鍵資源：**
- [LLM Powered Autonomous Agents — Lilian Weng](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [LangGraph 官方教學](https://langchain-ai.github.io/langgraph/)
- [Reflexion (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366)
- [Andrew Ng — Reflection Pattern](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/)
- [ACE Playbook 開源實作](https://github.com/jmanhype/ace-playbook)

---

## 04 ACE Framework

**學習目標：** 理解 Grow-and-Refine 機制；實作 Curator delta update；建立 materials extraction playbook

```
04_ace_framework/
├── playbook_evolution.py     # Grow-and-Refine：append → merge → deduplicate
├── curator_pattern.py        # Curator 角色：delta update playbook
└── notebook/
    └── ace_lab.ipynb         # 完整 GRC loop 模擬
```

**關鍵資源：**
- [ACE: Agentic Context Engineering (arXiv 2510.04618)](https://arxiv.org/abs/2510.04618)
- [ACE GitHub](https://github.com/ace-agent/ace)

---

## 05 Material Science Agents

**學習目標：** 理解 MARS 和 LLMatDesign 架構；建出材料科學 extraction agent 原型

```
05_material_science_agents/
├── extraction_agent.py       # 材料論文 extraction agent（Generator-Reflector-Curator）
├── mars_architecture_study.md      # MARS 系統 19-agent 架構筆記
├── llmatdesign_study.md            # LLMatDesign strategy library 筆記
└── notebook/
    └── material_agent_lab.ipynb
```

**關鍵資源：**
- [Towards Agentic Intelligence for Materials Science (arXiv 2602.00169)](https://arxiv.org/abs/2602.00169)
- [LLMatDesign (arXiv 2406.13163)](https://arxiv.org/abs/2406.13163) | [GitHub](https://github.com/Fung-Lab/LLMatDesign)
- [MatAgent GitHub](https://github.com/adibgpt/MatAgent)
- [MADE Benchmark (arXiv 2601.20996)](https://arxiv.org/abs/2601.20996)
- [Agentic Material Science (OAE)](https://www.oaepublish.com/articles/jmi.2025.87)

---

## 06 Fine-tuning

**學習目標：** 完成一次 QLoRA 微調 + eval benchmark；理解 fine-tuning vs. prompt engineering 的取捨

```
06_finetuning/
├── lora_basics.py            # LoRA 原理 + 簡單設定範例
├── qlora_training.py         # QLoRA 訓練腳本骨架（Colab/GPU 執行）
└── notebook/
    └── finetuning_lab.ipynb
```

> ⚠️ **Apple Silicon 注意：** `bitsandbytes` 不支援 MPS。訓練請使用 Google Colab (T4) 或 Kaggle (P100)。本地可用 `mlx-lm` 做輕量實驗。

**關鍵資源：**
- [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) | [GitHub](https://github.com/artidoro/qlora)
- [HuggingFace PEFT](https://huggingface.co/docs/peft)
- [MLX Fine-tuning Guide](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)

---

## 07 Multimodal

**學習目標：** 能處理含圖表的科學論文，輸出結構化 JSON 數據

```
07_multimodal/
├── image_analysis.py         # 圖表理解（GPT-4.1-mini vision / Claude API）
└── notebook/
    └── multimodal_lab.ipynb
```

**關鍵資源：**
- [ColPali (Faysse et al., 2024)](https://arxiv.org/abs/2407.01449) — 視覺模型做文件檢索
- [DocLLM (Wang et al., 2024)](https://arxiv.org/abs/2401.00908)

---

## 參考資料

- [`resources/papers.md`](resources/papers.md) — 完整論文清單
- [`resources/tools.md`](resources/tools.md) — 工具與框架清單
- [`resources/references.md`](resources/references.md) — 學習資源彙整

---

## 快速開始

```bash
# 1. Clone repo
git clone https://github.com/areomoon/areomoon_agent_warmup.git
cd areomoon_agent_warmup

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 設定 API keys
cp .env.example .env
# 編輯 .env，填入 ANTHROPIC_API_KEY 或 OPENAI_API_KEY

# 4. 從 Week 1 開始
cd 01_prompt_engineering
python chain_of_thought.py
```

---

*由 areomoon 建立 | 2026-04 | 入職準備：材料科學 AI Agent 算法工程師*
