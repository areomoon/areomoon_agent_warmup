# Managed Agents 概念與 ACE Framework 的對應關係

> ACE 論文：arXiv 2510.04618 | 本模組路徑：`08_managed_agents/`

---

## 一、ACE Framework 核心架構回顧

ACE（Agentic Context Engineering）的核心是一個**自我改進的三角循環**：

```
┌─────────────────────────────────────────────────┐
│                                                  │
│   Generator ──→ Reflector ──→ Curator            │
│       ↑                          │               │
│       └──────── Playbook ←───────┘               │
│                                                  │
│   Playbook = 跨任務累積的成功/失敗規則庫           │
└─────────────────────────────────────────────────┘
```

- **Generator**：產生輸出（論文抽取、假設生成）
- **Reflector**：分析輸出品質，提取可遷移的教訓
- **Curator**：將教訓合併至 Playbook（永久記憶）
- **Playbook**：在 System Prompt 中注入，影響下次 Generator 行為

---

## 二、本模組各檔案與 ACE 概念的對應

### `01_quickstart.py` → ACE 基礎迴圈

| Managed Agent 概念 | ACE 對應 |
|-------------------|---------|
| `AgentSession` | 單次 ACE 任務執行上下文 |
| `messages` 歷史 | ACE 的 Working Memory |
| 多輪 `chat_turn()` | Generator 的迭代執行 |
| `session.to_summary()` | Reflector 的輸出摘要 |

**連結點：** `AgentSession.messages` 是 ACE Working Memory 的直接實現。

---

### `02_custom_tools.py` → ACE Tool Layer

| Managed Agent 概念 | ACE 對應 |
|-------------------|---------|
| `TOOLS` 定義（JSON Schema） | ACE Tool Manifest |
| `TOOL_DISPATCH` 字典 | ACE Tool Executor |
| `run_agent_with_tools()` 迴圈 | ACE ReAct 執行模式 |
| `tool_use` → `tool_result` 流程 | ACE Observation Step |

**ACE 原文對應（§3.2 Tool Integration）：**
> "The agent must clearly specify the tools it can call and their expected input/output contracts."

本模組的 `input_schema` 即為此「合約」的 JSON Schema 實現。

---

### `03_memory_stores.py` → ACE Memory Architecture

```
ACE Memory 分層               本模組實現
─────────────────────────────────────────
Working Memory (短期)    →    FullHistoryStore.messages
Episodic Memory (中期)   →    FileBasedStore (JSONL)
Semantic Memory (長期)   →    SummarisationStore.summary
Playbook (跨任務)        →    (見 03_agent_patterns/curator_pattern.py)
```

**關鍵洞察：**
- ACE 認為記憶分層是減少 token 消耗的核心機制
- `SummarisationStore.maybe_compress()` 實現了 ACE 的 Memory Distillation
- 真正的 Playbook 更新需要 Curator Agent（見 `04_ace_framework/`）

---

### `04_multi_store_session.py` → ACE Session Coordination

| Managed Agent 概念 | ACE 對應 |
|-------------------|---------|
| `SessionManager` | ACE Agent Coordinator |
| `tool_cache` | ACE Execution Cache（避免重複工具呼叫） |
| `scratchpad` | ACE Chain-of-Thought Buffer |
| `extraction_log` | ACE Episodic Memory（持久化） |
| `_cache_key()` | ACE Deduplication Layer |

**ACE 設計原則體現：**
> "Avoid redundant computation. Cache tool outputs keyed by semantic fingerprint."

`_cache_key()` 以 MD5 hash 實現了此語義指紋（簡化版）。

---

### `05_event_streaming.py` → ACE Real-time Feedback

| Managed Agent 概念 | ACE 對應 |
|-------------------|---------|
| `StreamEventHandler` | ACE Event Bus 訂閱者 |
| `on_text_delta()` | ACE 即時輸出監控 |
| `on_tool_use` 事件 | ACE Tool Execution Trace |
| `event_counts` 統計 | ACE Performance Telemetry |

**ACE 可觀測性原則：**
串流事件是 ACE 系統中實現「透明度」的關鍵機制。
每個 `content_block_delta` 都是一個可被 Reflector 監控的觀察點。

---

### `06_material_extraction_demo.py` → ACE 完整執行流

```
本模組流程                    ACE 概念對應
──────────────────────────────────────────────────
step1_initial_extraction()  →  Generator（初稿生成）
step2_validate_with_tools() →  Reflector（工具輔助驗證）
step3_reflect_and_correct() →  Generator + Reflector 聯合修正
save_to_database()          →  Episodic Memory 更新
```

**完整 ACE 循環尚缺的部分：**
- Curator 更新 Playbook（需整合 `04_ace_framework/curator_pattern.py`）
- System Prompt 中注入 Playbook 規則（下次 Generator 呼叫時生效）

---

## 三、本模組未覆蓋的 ACE 概念

| ACE 概念 | 說明 | 在哪裡實作 |
|---------|------|----------|
| Curator Agent | 將教訓合併至 Playbook | `04_ace_framework/curator_pattern.py` |
| Playbook 注入 | System Prompt 動態更新 | `04_ace_framework/` |
| Multi-Agent Debate | 多個 Reflector 互相挑戰 | `03_agent_patterns/` |
| Long-horizon Planning | 跨多天的任務規劃 | `09_production_patterns/` |

---

## 四、整合路徑建議

若要將本模組與 ACE Framework 完整整合，建議路徑：

```python
# 整合示意（偽代碼）
from ace_framework import CuratorAgent, Playbook
from managed_agents import ExtractionSession, run_full_pipeline

playbook = Playbook.load("materials_extraction_v3.json")

session = ExtractionSession(
    system_prompt=playbook.to_system_prompt()  # 注入 ACE 規則
)

result = run_full_pipeline(paper_text, session)

lessons = reflector.extract_lessons(result)
playbook = curator.update(playbook, lessons)   # ACE 閉環
playbook.save()
```

---

## 五、推薦學習順序

```
03_agent_patterns/ (Generator-Reflector 基礎)
         ↓
04_ace_framework/ (Playbook 與 Curator)
         ↓
08_managed_agents/ (本模組：工具、記憶、串流)
         ↓
整合：06_material_extraction_demo.py + ACE Curator
         ↓
09_production_patterns/ (部署、監控、A/B 測試)
```

---

## 相關連結

- [ACE 論文 (arXiv 2510.04618)](https://arxiv.org/abs/2510.04618)
- [本專案 04_ace_framework/](../04_ace_framework/)
- [本專案 03_agent_patterns/](../03_agent_patterns/)
- [Generator-Reflector 實現](../03_agent_patterns/generator_reflector.py)
