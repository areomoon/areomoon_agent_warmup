# 05 Material Science Agents

**對應 Phase 1（入職後第1-2月）| 持續進行**

## 學習目標

理解 MARS 和 LLMatDesign 架構，建出材料科學 extraction agent 原型，為入職直接對接真實系統。

完成後你能：
- [ ] 說明 MARS 19-agent 分工（Orchestrator/Scientist/Engineer/Executor/Analyst）
- [ ] 解釋 LLMatDesign 的 strategy library 如何運作
- [ ] 建出 materials paper extraction agent prototype
- [ ] 設計 extraction case 收集機制（為 fine-tuning 累積資料）

## 執行順序

1. 讀 `mars_architecture_study.md` — 理解 MARS 系統
2. 讀 `llmatdesign_study.md` — 理解 LLMatDesign strategy library
3. `python extraction_agent.py` — 建出 extraction agent 原型
4. 打開 `notebook/material_agent_lab.ipynb`

## 必讀論文（入職前完成）

| 論文 | 重點 | 連結 |
|------|------|------|
| MARS 系統 | 19 agents 分工 + 16 domain tools | [phys.org 報導](https://phys.org/news/2026-01-multi-agent-ai-robots-automate.html) |
| LLMatDesign | iterative propose/evaluate/refine | [arXiv 2406.13163](https://arxiv.org/abs/2406.13163) \| [GitHub](https://github.com/Fung-Lab/LLMatDesign) |
| MatAgent | physics-aware multi-agent framework | [GitHub](https://github.com/adibgpt/MatAgent) |
| Towards Agentic Intelligence | 材料科學 AI Agent 全景 | [arXiv 2602.00169](https://arxiv.org/abs/2602.00169) |
| MADE Benchmark | closed-loop discovery 評估 | [arXiv 2601.20996](https://arxiv.org/abs/2601.20996) |
