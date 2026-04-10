# ACE Framework → Managed Agents 對應關係

> 說明 04_ace_framework 的 Generator / Reflector / Curator 如何對應 Managed Agents 概念  
> 前置閱讀：[04_ace_framework/README.md](../../04_ace_framework/README.md)

---

## ACE Framework 快速回顧

```
Generator  → 從文獻提取 raw data
Reflector  → 驗證提取結果的品質
Curator    → 將 delta updates 整合進 Playbook
```

---

## 對應關係總覽

| ACE 角色 | Managed Agents 實作 |
|---------|---------------------|
| **Generator** | Session（agent 執行提取任務）|
| **Reflector** | 同一 Session 的第二輪 message，或獨立 Session |
| **Curator** | Session 的 `memory_write`（safe update with precondition）|
| **Playbook** | Memory Store（read-write，跨 session 共用）|
| **Extraction Cases** | Memory Store（session scoped 或 shared read-only）|
| **GRC Loop** | 多個 Session 共用同一 Playbook Store |

---

## 詳細說明

### Generator → Session

Generator 的工作（讀論文、提取 raw data）直接對應到一個 Managed Agent session：

```
Session（Generator 角色）
├── 讀取 Playbook（memory_read）
├── 分析論文（tool: bash 或 files）
├── 呼叫性質查詢工具（tool_use: lookup_material_property）
└── 回傳 structured extraction result
```

### Reflector → 第二輪 Message 或獨立 Session

**選項 A：同一 session 的第二輪 message**
```
Session
├── Turn 1：Generator → 提取 raw data
└── Turn 2：system steer "請以 Reflector 角色驗證上方結果"
                → Reflector 輸出 critique + corrections
```

**選項 B：獨立 Reflector Session**
```
Generator Session → 寫入 cases store（draft）
Reflector Session → 讀取 draft → 驗證 → 更新 confidence 和 notes
```

選項 B 適合需要不同 system prompt 的 Reflector，或需要並行驗證多個 draft 的場景。

### Curator → Memory Write with Precondition

Curator 的核心是「安全地更新 Playbook」，對應 memory store 的 preconditioned write：

```python
# ACE Curator 的 delta update 邏輯
current = store.entries.retrieve(key="rules/heuristics")
new_value = current.value + "\n- 新啟發式：..." + delta

# 安全寫入：只有當 version 沒變時才更新
store.entries.create(
    key="rules/heuristics",
    value=new_value,
    precondition={"version": current.version},  # ← Curator 安全機制
)
```

這樣即使多個 session 同時嘗試更新 Playbook，也不會發生覆蓋衝突。

---

## GRC Loop → 多 Session 共用 Memory

```
┌─────────────────────────────────────────────────────────┐
│  Shared Memory Stores（跨所有 session 共用）              │
│  ├── playbook_store (read-write)                        │
│  └── reference_store (read-only)                        │
│                                                         │
│  Session A（Generator）                                 │
│  ├── reads playbook                                     │
│  ├── extracts paper_1                                   │
│  └── writes cases/paper_1                               │
│                                                         │
│  Session B（Generator）                                 │
│  ├── reads same playbook                                │
│  ├── extracts paper_2                                   │
│  └── writes cases/paper_2                               │
│                                                         │
│  Session C（Curator）                                   │
│  ├── reads cases/paper_1, cases/paper_2                │
│  ├── generates delta updates                            │
│  └── writes updated playbook (with precondition)        │
└─────────────────────────────────────────────────────────┘
```

這就是 ACE GRC Loop 在 Managed Agents 中的實作。

---

## ACE 四層 Playbook → Memory Store 結構

```
playbook_store/
├── layer1/structural_design_rules     ← PbTe + Bi doping → ZT 規則
├── layer2/synthesis_protocol_patterns ← Perovskite 固態合成 900°C
├── layer3/characterization_signatures ← Raman D band 識別
└── layer4/extraction_heuristics       ← 'typical' 表示單一樣品
```

每個 layer 是獨立的 key，支援獨立更新和 version 追蹤。  
Curator 只更新有新 evidence 的 layer，不需要重寫整個 playbook。

---

## Brevity Bias & Context Collapse 的防護

ACE Framework 提到兩個主要失敗模式：

| 失敗模式 | 在 Managed Agents 中的防護 |
|---------|--------------------------|
| **Brevity Bias**（Playbook 越改越短）| Version history 讓你比較前後差異；若 len(v_new) < len(v_old) * 0.8 可觸發警告 |
| **Context Collapse**（重要規則被覆蓋）| Preconditioned write 確保不會意外覆蓋；定期 `history(key)` 審查 |

```python
# 防止 Brevity Bias 的 Curator 寫入邏輯
def safe_curator_update(store, key, delta_text):
    current = store.read(key)
    new_value = current["value"] + "\n" + delta_text  # 只 append，不覆蓋

    # 長度保護
    if len(new_value) < len(current["value"]) * 0.9:
        raise ValueError("更新後內容過短，疑似 Brevity Bias，請人工審查")

    return store.write(key, new_value,
                       precondition_version=current["version"])
```

---

## 小結

| 你在 04 學的 | 在 08 的對應 |
|------------|------------|
| Generator 提取 | `session.send_message()` + tool_use |
| Reflector 驗證 | 同 session 第二輪 or 獨立 session |
| Curator delta update | `memory_store.write(precondition_version=...)` |
| Playbook 四層結構 | Memory store keys 按 layer 組織 |
| GRC Loop | 多 session 共用 playbook_store |
| Brevity Bias 防護 | 長度保護 + version history 審查 |
