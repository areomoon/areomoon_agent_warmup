# 08 Managed Agents

**對應 Warmup Week 4–5 | 預估時數：8–10hr**

## 模組目標

學會 Claude Managed Agents API，為材料科學 agent 工作做準備。  
Managed Agents 提供雲端管理的 agent runtime、內建 session 管理、memory store，讓你專注在 agent 邏輯而非基礎設施。

完成後你能：
- [ ] 用 Managed Agents API 建立並執行 agent session
- [ ] 定義、註冊、呼叫 custom tools
- [ ] 操作 memory store（create/seed/attach/read/write）
- [ ] 處理 SSE event stream（tool_use、memory events、error）
- [ ] 設計材料科學 extraction agent，並利用 memory store 管理 playbook + cases

---

## 前置知識

| 模組 | 需要的概念 |
|------|-----------|
| [03_agent_patterns](../03_agent_patterns/README.md) | Generator-Reflector pattern、tool use |
| [04_ace_framework](../04_ace_framework/README.md) | Playbook 結構、Curator delta update |

---

## ⚠️ Beta Access 說明

Managed Agents 目前處於 **Private Beta**。  
執行需要 beta access — 本模組所有 `.py` 檔都提供 **mock 模式**，在沒有 beta access 的情況下可完整跑完學習流程。

申請 beta access：
- 官方文件：https://platform.claude.com/docs/en/managed-agents/overview
- 申請頁面：https://www.anthropic.com/contact/managed-agents-beta

---

## 學習順序

```
概念理解 → API 操作 → Memory → 完整 Demo
```

| 步驟 | 檔案 | 學習目標 | 預估時間 |
|------|------|---------|---------|
| 1 | `01_quickstart.py` | 建立 agent + environment + session，發送第一條訊息 | 45min |
| 2 | `02_custom_tools.py` | 定義 custom tools，掌握 schema 設計最佳實務 | 45min |
| 3 | `03_memory_stores.py` | Memory store CRUD 全操作，version history | 60min |
| 4 | `04_multi_store_session.py` | 多 store attach，access control，shared vs scoped | 45min |
| 5 | `05_event_streaming.py` | SSE 事件流處理，interrupt/steer | 60min |
| 6 | `06_material_extraction_demo.py` | 材料科學完整 demo，整合所有概念 | 90min |
| 7 | `notebook/managed_agents_lab.ipynb` | 互動式練習 + 小專案 | 90min |

---

## 核心概念速覽

```
┌─────────────────────────────────────────────────────┐
│                  Managed Agents                      │
│                                                     │
│  Agent Definition                                   │
│  ├── Model (claude-sonnet-4-6)                      │
│  ├── System Prompt                                  │
│  ├── Built-in Tools (bash, web_search, files)       │
│  └── Custom Tools (your API calls)                  │
│                                                     │
│  Environment                                        │
│  └── Runtime (Python 3.11, packages installed)      │
│                                                     │
│  Session                                            │
│  ├── Messages (user ↔ agent)                        │
│  ├── Memory Stores (read-only / read-write)         │
│  └── Event Stream (SSE)                             │
│                                                     │
│  Memory Store                                       │
│  ├── Key-value entries                              │
│  ├── Version history                                │
│  └── Preconditioned writes (safe update)            │
└─────────────────────────────────────────────────────┘
```

---

## 資源

- [resources/api_reference.md](resources/api_reference.md) — API 重點整理
- [resources/comparison_with_langchain.md](resources/comparison_with_langchain.md) — vs LangChain 對比
- [resources/ace_mapping.md](resources/ace_mapping.md) — ACE Framework 對應關係
- 官方文件：https://platform.claude.com/docs/en/managed-agents/overview
