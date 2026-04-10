"""
04_multi_store_session.py — 多個 Memory Store 的 Session 設計

學習目標：
    1. 同時 attach 多個 store 到同一個 session
    2. 理解 read-only vs read-write access control
    3. Shared store（多 session 共用）vs Scoped store（session 專屬）
    4. 設計材料科學 agent 的 memory 架構

設計說明：
    本模組展示三層 memory 架構：
    ┌─────────────────────────────────────────────────────┐
    │  Store 1: Reference Knowledge (read-only)           │
    │    → 材料性質常數、文獻標準值（不應被 agent 修改）      │
    │                                                     │
    │  Store 2: Extraction Playbook (read-write)          │
    │    → 提取規則、啟發式方法（agent 可學習並更新）         │
    │                                                     │
    │  Store 3: Extraction Cases (read-write, scoped)     │
    │    → 當前 session 的提取結果（每個 session 獨立）       │
    └─────────────────────────────────────────────────────┘

官方文件：https://platform.claude.com/docs/en/managed-agents/memory-stores

使用方式：
    python 04_multi_store_session.py
"""

import json
from typing import Optional
from dataclasses import dataclass, field

# 複用 03_memory_stores.py 的 Mock 實作
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from memory_stores_mock import MockMemoryStore, MockMemoryStoreClient  # type: ignore
except ImportError:
    # 內聯 minimal mock（獨立執行時使用）
    import copy
    from datetime import datetime, timezone

    class MockMemoryStore:
        def __init__(self, store_id, name, description=""):
            self.id = store_id
            self.name = name
            self.description = description
            self._entries = {}

        def write(self, key, value, precondition_version=None):
            versions = self._entries.get(key, [])
            current_version = len(versions)
            if precondition_version is not None and current_version != precondition_version:
                raise ValueError(f"Precondition failed for '{key}'")
            new_version = current_version + 1
            entry = {"key": key, "value": value, "version": new_version,
                     "updated_at": datetime.now(timezone.utc).isoformat()}
            self._entries.setdefault(key, []).append(copy.deepcopy(entry))
            return entry

        def read(self, key):
            versions = self._entries.get(key)
            return copy.deepcopy(versions[-1]) if versions else None

        def list_keys(self):
            return list(self._entries.keys())

        def seed(self, entries):
            for k, v in entries.items():
                self.write(k, v)

    class MockMemoryStoreClient:
        def __init__(self):
            self._stores = {}
        def create(self, name, description=""):
            sid = f"store_{len(self._stores)+1:04d}"
            s = MockMemoryStore(sid, name, description)
            self._stores[sid] = s
            return s


# ─────────────────────────────────────────────────────
#  Session with Multiple Stores
# ─────────────────────────────────────────────────────

@dataclass
class StoreAttachment:
    """Store 附加到 session 的設定。"""
    store: MockMemoryStore
    access: str  # "read" | "read-write"
    alias: str   # session 內的別名


@dataclass
class MockSession:
    """模擬帶有多個 memory store 的 session。"""
    session_id: str
    agent_id: str
    attachments: list[StoreAttachment] = field(default_factory=list)

    def attach_store(self, store: MockMemoryStore, access: str, alias: str):
        """Attach 一個 memory store 到此 session。"""
        self.attachments.append(StoreAttachment(store=store, access=access, alias=alias))
        print(f"  [attach] alias='{alias}' store={store.id} access={access}")

    def read_from(self, alias: str, key: str) -> Optional[dict]:
        """從特定 store 讀取。"""
        att = self._get_attachment(alias)
        if att is None:
            raise KeyError(f"Store alias '{alias}' 未附加到此 session")
        result = att.store.read(key)
        print(f"  [session.read] alias={alias}, key={key}, found={result is not None}")
        return result

    def write_to(self, alias: str, key: str, value: str,
                 precondition_version: Optional[int] = None) -> dict:
        """寫入特定 store（需要 read-write 權限）。"""
        att = self._get_attachment(alias)
        if att is None:
            raise KeyError(f"Store alias '{alias}' 未附加到此 session")
        if att.access == "read":
            raise PermissionError(
                f"Store '{alias}' 是 read-only，無法寫入。"
                f"（store_id={att.store.id}）"
            )
        return att.store.write(key, value, precondition_version)

    def _get_attachment(self, alias: str) -> Optional[StoreAttachment]:
        for att in self.attachments:
            if att.alias == alias:
                return att
        return None

    def list_stores(self):
        """列出附加的所有 store 和權限。"""
        print(f"\n  Session {self.session_id} 的 Store 列表：")
        for att in self.attachments:
            print(f"    [{att.access:10s}] alias='{att.alias}' → {att.store.name} ({att.store.id})")


# ─────────────────────────────────────────────────────
#  初始資料
# ─────────────────────────────────────────────────────

REFERENCE_KNOWLEDGE_SEED = {
    "constants/avogadro": "6.02214076e23 mol⁻¹",
    "constants/boltzmann": "1.380649e-23 J/K",
    "standards/perovskite_efficiency_record": "29.1% (Si/perovskite tandem, 2024)",
    "standards/thermoelectric_zt_target": "ZT > 3.0 at 500K（研究目標）",
    "known_materials/MAPbI3/band_gap": "1.57 eV (Snaith 2013)",
    "known_materials/GaN/band_gap": "3.4 eV (bulk)",
}

PLAYBOOK_SEED = {
    "rules/step1_locate": "先找摘要和結論中的主要材料描述",
    "rules/step2_tables": "掃描所有表格，記錄 caption 關鍵詞",
    "rules/step3_figures": "圖表中的數據需用 multimodal 確認",
    "heuristics/confidence": (
        "confidence 評分：\n"
        "  high: 摘要/結論明確陳述\n"
        "  medium: 表格數據有標題\n"
        "  low: 圖表讀取或文中推斷"
    ),
}


# ─────────────────────────────────────────────────────
#  Demo 主流程
# ─────────────────────────────────────────────────────

def demo_multi_store_session():
    print("=" * 60)
    print("  Multi-Store Session Demo")
    print("=" * 60)

    client = MockMemoryStoreClient()

    # ── 1. 建立三個 store ──────────────────────────────
    print("\n[1] 建立三個 Memory Stores")

    # Store A: 參考知識（共用、read-only）
    reference_store = client.create(
        name="materials-reference-knowledge",
        description="材料常數和文獻標準值，所有 session 共用，不可修改",
    )
    reference_store.seed(REFERENCE_KNOWLEDGE_SEED)

    # Store B: Playbook（共用、read-write）
    playbook_store = client.create(
        name="materials-extraction-playbook",
        description="提取規則，agent 可學習並更新",
    )
    playbook_store.seed(PLAYBOOK_SEED)

    # Store C: Cases（session 專屬、read-write）
    cases_store = client.create(
        name="session-extraction-cases",
        description="當前 session 的提取結果，session 結束後歸檔",
    )

    # ── 2. 建立 session 並 attach stores ──────────────
    print("\n[2] 建立 Session，Attach 三個 Stores")
    session = MockSession(session_id="sess_0001", agent_id="agent_0001")

    # Reference: read-only（不允許 agent 修改常數）
    session.attach_store(reference_store, access="read", alias="reference")

    # Playbook: read-write（agent 可以學習並更新規則）
    session.attach_store(playbook_store, access="read-write", alias="playbook")

    # Cases: read-write（agent 儲存提取結果）
    session.attach_store(cases_store, access="read-write", alias="cases")

    session.list_stores()

    # ── 3. 讀取 reference（OK）─────────────────────────
    print("\n[3] 從 reference store 讀取")
    val = session.read_from("reference", "standards/perovskite_efficiency_record")
    print(f"  value: {val['value']}")

    # ── 4. 嘗試寫入 reference（應失敗）─────────────────
    print("\n[4] 嘗試寫入 read-only reference（應失敗）")
    try:
        session.write_to("reference", "constants/avogadro", "6.02e23")
    except PermissionError as e:
        print(f"  ✓ 預期錯誤：{e}")

    # ── 5. 讀取並更新 playbook ─────────────────────────
    print("\n[5] 讀取並更新 playbook")
    rule = session.read_from("playbook", "heuristics/confidence")
    print(f"  current v{rule['version']}: {rule['value'][:60]}...")

    updated_value = rule["value"] + "\n  very_low: 只在方法部分提到，無數據"
    session.write_to("playbook", "heuristics/confidence",
                     updated_value, precondition_version=rule["version"])
    print("  ✓ Playbook 更新成功（新啟發式加入）")

    # ── 6. 寫入 extraction cases ──────────────────────
    print("\n[6] 儲存提取結果到 cases store")
    import json as _json
    case_1 = _json.dumps({
        "paper_doi": "10.1039/D3EE01234A",
        "material": "Cs0.05(MA0.17FA0.83)0.95Pb(I0.83Br0.17)3",
        "property": "PCE",
        "value": 24.3,
        "unit": "%",
        "confidence": "high",
        "location": "Abstract, Table 1",
    }, ensure_ascii=False, indent=2)

    session.write_to("cases", "paper/D3EE01234A/extraction_v1", case_1)
    print(f"  ✓ Case 儲存完成：{case_1[:80]}...")

    # ── 7. Shared vs Scoped 概念說明 ──────────────────
    print("\n[7] Shared vs Scoped Memory 模式")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  Shared Store（跨 session 共用）                         │
  │  ├── reference_store → 所有 session 共用，read-only      │
  │  └── playbook_store  → 所有 session 可讀，某些可寫        │
  │                                                         │
  │  Scoped Store（session 專屬）                           │
  │  └── cases_store     → 每個 session 有自己的 cases        │
  │      建立方式：每個 session 建立新的 cases_store           │
  │      Session 結束後歸檔（copy to long-term store）        │
  └─────────────────────────────────────────────────────────┘
    """)

    print("✅ Multi-Store Session Demo 完成！")
    print("   下一步：05_event_streaming.py — 處理 SSE 事件流")


if __name__ == "__main__":
    demo_multi_store_session()
