# 08 · Managed Agents — Anthropic SDK 代理管理模組

**對應 Warmup Week 6 | 預估時數：4–6 hr**

---

## 學習目標

- [ ] 理解 Anthropic SDK 的 Managed Agent 架構與生命週期
- [ ] 掌握自定義工具（Custom Tools）的定義與 `tool_use` 回呼流程
- [ ] 實作多種 Memory Store 策略（In-memory / File-based / Summarisation）
- [ ] 建構跨工具多 Store 的 Session 管理機制
- [ ] 使用 Server-Sent Events (SSE) 實現串流輸出
- [ ] 整合材料科學論文抽取任務作為端對端展示

---

## 執行順序

```bash
cd 08_managed_agents

# 1. 快速入門：建立第一個 Managed Agent
python 01_quickstart.py

# 2. 自定義工具：tool_use 完整回呼迴圈
python 02_custom_tools.py

# 3. Memory Store 三種策略比較
python 03_memory_stores.py

# 4. 多 Store 跨工具 Session 管理
python 04_multi_store_session.py

# 5. SSE 串流事件處理
python 05_event_streaming.py

# 6. 材料科學抽取 Demo（端對端）
python 06_material_extraction_demo.py

# 7. 互動式 Notebook
jupyter lab notebook/managed_agents_lab.ipynb
```

> **注意：** 所有腳本支援 Mock 模式，未設定 `ANTHROPIC_API_KEY` 時自動切換。

---

## 目錄說明

| 檔案 | 說明 |
|------|------|
| `01_quickstart.py` | 建立最小可執行的 Managed Agent，理解 SDK 基本結構 |
| `02_custom_tools.py` | 定義工具 schema、處理 `tool_use` 事件、回傳 `tool_result` |
| `03_memory_stores.py` | In-memory / File-based / Summarisation 三種記憶策略 |
| `04_multi_store_session.py` | 多工具協作 Session，跨輪次狀態持久化 |
| `05_event_streaming.py` | 使用 `stream()` context manager 處理 SSE 事件 |
| `06_material_extraction_demo.py` | 以材料科學論文段落為輸入，展示完整 Agent 工作流 |
| `notebook/managed_agents_lab.ipynb` | 互動式實驗場，含練習題 |
| `resources/api_reference.md` | Anthropic Managed Agents API 速查表 |
| `resources/comparison_with_langchain.md` | 與 LangChain Agents 的架構對比 |
| `resources/ace_mapping.md` | 本模組概念如何對應 ACE Framework |

---

## 關鍵資源

- [Anthropic Managed Agents 官方文件](https://platform.claude.com/docs/en/managed-agents/overview)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Tool Use 指南](https://docs.anthropic.com/en/docs/tool-use)
- [Streaming 指南](https://docs.anthropic.com/en/api/messages-streaming)
- [ACE Framework (arXiv 2510.04618)](https://arxiv.org/abs/2510.04618)
- [ReAct Pattern (arXiv 2210.03629)](https://arxiv.org/abs/2210.03629)

---

## 環境設定

```bash
# .env 中需要設定：
ANTHROPIC_API_KEY=sk-ant-...

# 安裝依賴：
pip install anthropic python-dotenv
```

---

## 與其他模組的關聯

- **← 07_multimodal**：多模態輸入在此模組中可作為工具的輸入來源
- **← 04_ace_framework**：Managed Agents 是 ACE Playbook 的執行引擎
- **→ 09_production_patterns**：本模組的 Session 管理是生產部署的基礎
