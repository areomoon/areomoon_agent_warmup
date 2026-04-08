# 09 Production Patterns

**學習目標：** 掌握讓 agent 從 demo 變成 production-grade 的工程 patterns

> 這是 warmup repo 原本缺少的模組。缺口分析見 [llm_knowledge_base: agentic-harness](https://github.com/areomoon/llm_knowledge_base/blob/main/wiki/concepts/agentic-harness.md)

## 為什麼需要 Production Patterns

Phil Schmid（Hugging Face Technical Lead）的類比（[來源](https://www.philschmid.de/agent-harness-2026)）：

```
Model = CPU（原始算力）
Context window = RAM（有限、易失的工作記憶）
Agent harness = Operating System（管理 context、boot、lifecycle hooks）
Agent = Application（在 OS 上跑的用戶邏輯）
```

你的 warmup 01-07 教你寫 Application。這個模組教你寫 Operating System。

## 權威框架

| Pattern | 提出者 | 來源 |
|---------|--------|------|
| Workflow → Agent 光譜 | Anthropic (Schluntz & Zhang) | [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) |
| 12 Harness Patterns | Claude Code source analysis | [Generative Programmer](https://generativeprogrammer.com/p/12-agentic-harness-patterns-from) |
| Model=CPU, Harness=OS | Phil Schmid (Hugging Face) | [Agent Harness 2026](https://www.philschmid.de/agent-harness-2026) |
| IMPACT framework | Swyx (Latent Space) | [Agent essay](https://www.latent.space/p/agent) |
| Crawl-Walk-Run trust | Chip Huyen (Stanford/O'Reilly) | [Building AI Agents](https://huyenchip.com/2025/01/07/agents.html) |
| Programmable HITL | Microsoft Research (AutoGen) | [COLM 2024](https://arxiv.org/abs/2308.08155) |

## 模組內容

```
09_production_patterns/
├── README.md                     # 本文件
├── context_compression.py        # 3 層 context compression（MicroCompact → AutoCompact → Reset）
├── permission_gating.py          # 3 層 permission model（allow / prompt / deny）
├── observability.py              # LangSmith / Langfuse integration for agent tracing
└── notebook/
    └── production_lab.ipynb      # 把 03 的 agent 加上 production patterns
```

## 練習任務

### Task 1: Context Compression（Claude Code Pattern 3）
1. 在 03_agent_patterns 的 reflection_agent 上加入 context window tracking
2. 實作 3 層壓縮：
   - MicroCompact：每 5 步自動 trim 舊 tool output
   - AutoCompact：接近 context limit 時生成結構化 summary
   - Full Reset：極端情況下只保留 memory index + current task
3. 驗證：長對話（>50 步）不會 context overflow

### Task 2: Permission Gating（Claude Code Pattern 8）
1. 定義 3 個權限層級：
   - Allow：讀取操作（搜索論文、查 vector DB）
   - Prompt：寫入操作（修改 database、發送 email）
   - Deny：刪除操作（刪除數據、drop table）
2. 在 agent loop 中加入 pre-tool-use hook 檢查權限
3. 驗證：agent 在 Prompt 層級時會暫停等待人類確認

### Task 3: Observability
1. 用 LangSmith 或 Langfuse 記錄 agent 的完整 trace
2. 能看到：每步的 tool call、input/output、latency、token count
3. 能從 trace 回溯 debug agent 的錯誤決策

### Task 4: 整合 — Production-Grade Agent
1. 把 05_material_science_agents 的 extraction_agent 升級：
   - 加入 context compression
   - 加入 permission gating
   - 加入 observability
   - 加入 08_evaluation 的 benchmark
2. 這就是你能帶去 Patsnap 展示的 demo

## 閱讀清單

- [Building Effective Agents — Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [12 Agentic Harness Patterns from Claude Code](https://generativeprogrammer.com/p/12-agentic-harness-patterns-from)
- [Agent Harness as OS — Phil Schmid](https://www.philschmid.de/agent-harness-2026)
- [IMPACT Framework — Swyx/Latent Space](https://www.latent.space/p/agent)
- [Building AI Agents — Chip Huyen](https://huyenchip.com/2025/01/07/agents.html)
- [LangSmith Docs](https://docs.smith.langchain.com/)
- [Langfuse Docs](https://langfuse.com/docs)
