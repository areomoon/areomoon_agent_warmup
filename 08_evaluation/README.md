# 08 Agent Evaluation

**學習目標：** 建立 agent-level evaluation 能力，不只是 model eval，而是 multi-step task completion、efficiency、safety 的完整評估

> 這是 warmup repo 原本缺少的模組。缺口分析見 [llm_knowledge_base: agent-evaluation](https://github.com/areomoon/llm_knowledge_base/blob/main/wiki/concepts/agent-evaluation.md)

## 為什麼 Agent Eval ≠ Model Eval

Model eval 衡量單次生成品質（BLEU, F1, EM）。Agent eval 必須衡量：
- **多步驟任務完成率** — agent 能否端到端完成任務？
- **效率** — 花了多少步驟/token/時間？
- **安全性** — 有沒有做危險操作？
- **成本** — 每次 run 花多少錢？值不值？

## 權威框架

| 框架 | 提出者 | 核心指標 |
|------|--------|---------|
| SWE-bench | Princeton NLP (Shunyu Yao, Karthik Narasimhan) — ICLR 2024 Oral | GitHub issue resolve rate |
| METR | Beth Barnes; advisors: Yoshua Bengio, Alec Radford | Task-completion time horizon |
| HELM | Percy Liang (Stanford CRFM) | 7 metrics × 16 scenarios |
| MADE | Materials science community | 結構化數據抽取 EM/F1 |

**Andrew Ng 的關鍵數據：** GPT-3.5 + agent loop 在 HumanEval 達 95.1%，超過 GPT-4 zero-shot 的 67%。這證明 harness/workflow 的投資回報可能超過模型升級。（[來源](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/)）

## 模組內容

```
08_evaluation/
├── README.md                     # 本文件
├── task_completion_eval.py       # 多步驟任務完成率評估（SWE-bench 風格）
├── cost_efficiency_eval.py       # Token cost / step count / time tracking
├── made_style_benchmark.py       # 材料科學抽取 benchmark（MADE 風格）
└── notebook/
    └── eval_lab.ipynb            # 互動實驗：baseline vs agent vs fine-tuned
```

## 練習任務

### Task 1: 建立 Extraction Benchmark（MADE 風格）
1. 從 05_material_science_agents 收集 50 筆 ground truth（論文段落 → 結構化參數）
2. 定義成功標準（Exact Match for numeric fields, F1 for text fields）
3. 跑 3 個 baseline：
   - Pure LLM (zero-shot)
   - LLM + prompt engineering (CoT)
   - Agent (Generator-Reflector from 03)
4. 記錄 completion rate, step count, token cost
5. 輸出比較報告

### Task 2: Trajectory Eval（METR 風格）
1. 錄製 agent 的完整執行軌跡（每步的 tool call + observation）
2. 評估：是否有冗餘步驟？是否走了錯誤路徑再修正？
3. 計算 task-completion time horizon

### Task 3: Regression Test
1. 建立 CI-friendly 的 eval script
2. 每次 agent 更新後自動跑 benchmark
3. 輸出 pass/fail + metrics diff

## 閱讀清單

- [SWE-bench (Jimenez et al., ICLR 2024)](https://arxiv.org/abs/2310.06770)
- [METR Evaluations](https://metr.org/)
- [HELM (Liang et al., Stanford CRFM)](https://crfm.stanford.edu/helm/)
- [MADE Benchmark](https://arxiv.org/abs/2601.20996)
- [Building Effective Agents — Anthropic](https://www.anthropic.com/research/building-effective-agents) — "complexity must be justified by performance gains"
