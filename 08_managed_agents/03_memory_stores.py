"""
03_memory_stores.py — Memory Store 完整操作

學習目標：
    1. 建立 memory store
    2. Seed 初始內容
    3. Attach 到 session
    4. CRUD：read / write / update / delete
    5. Version history 查詢
    6. Safe write with precondition（避免 race condition）

官方文件：https://platform.claude.com/docs/en/managed-agents/memory-stores

Memory Store 概念：
    - Key-value store，每個 entry 有 key、value、version
    - Agent 可在 session 中讀取或寫入（依 access control）
    - Version history 讓你追蹤 entry 的演變
    - Preconditioned write：只有當前 version 符合預期才寫入（safe update）

使用方式：
    python 03_memory_stores.py
"""

import json
import copy
from datetime import datetime, timezone
from typing import Optional

# ─────────────────────────────────────────────────────
#  Mock Memory Store 實作
# ─────────────────────────────────────────────────────

class MockMemoryStore:
    """
    模擬 Managed Agents Memory Store。
    真實 API 透過 client.memory_stores.* 操作。
    """

    def __init__(self, store_id: str, name: str, description: str = ""):
        self.id = store_id
        self.name = name
        self.description = description
        # entries: key → list of versions [{value, version, updated_at}]
        self._entries: dict[str, list[dict]] = {}
        print(f"[MemoryStore] 建立: id={self.id}, name={self.name}")

    # ── CRUD ──────────────────────────────────────────

    def write(self, key: str, value: str,
              precondition_version: Optional[int] = None) -> dict:
        """
        寫入 entry。

        precondition_version:
            None  → 無條件寫入（新建或覆蓋）
            int   → 只有當前 version 等於此值才寫入，否則 raise ConflictError
                    用於 safe update，避免覆蓋他人的修改。
        """
        current_versions = self._entries.get(key, [])
        current_version = len(current_versions)  # 0 = 不存在

        # Precondition 檢查
        if precondition_version is not None:
            if current_version != precondition_version:
                raise ValueError(
                    f"Precondition failed: key='{key}' "
                    f"current_version={current_version}, "
                    f"expected={precondition_version}"
                )

        new_version = current_version + 1
        entry = {
            "key": key,
            "value": value,
            "version": new_version,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if key not in self._entries:
            self._entries[key] = []
        self._entries[key].append(copy.deepcopy(entry))

        print(f"  [write] key='{key}' version={new_version}")
        return entry

    def read(self, key: str) -> Optional[dict]:
        """讀取 entry（最新版本）。None 表示不存在。"""
        versions = self._entries.get(key)
        if not versions:
            return None
        result = copy.deepcopy(versions[-1])
        print(f"  [read]  key='{key}' version={result['version']}")
        return result

    def delete(self, key: str) -> bool:
        """刪除 entry。回傳 True 表示成功刪除。"""
        if key in self._entries:
            del self._entries[key]
            print(f"  [delete] key='{key}'")
            return True
        print(f"  [delete] key='{key}' 不存在")
        return False

    def list_keys(self) -> list[str]:
        """列出所有 key。"""
        keys = list(self._entries.keys())
        print(f"  [list] {len(keys)} keys: {keys}")
        return keys

    def history(self, key: str) -> list[dict]:
        """
        查詢 key 的完整 version history。
        真實 API：client.memory_stores.entries.versions.list(store_id, key)
        """
        versions = self._entries.get(key, [])
        print(f"  [history] key='{key}' {len(versions)} versions")
        return copy.deepcopy(versions)

    # ── Seed ──────────────────────────────────────────

    def seed(self, entries: dict[str, str]) -> None:
        """
        批量初始化 store 內容。
        通常在建立 store 後立即執行，設定 playbook 或 reference data。
        """
        print(f"  [seed] 寫入 {len(entries)} 個 entries")
        for key, value in entries.items():
            self.write(key, value)


class MockMemoryStoreClient:
    """模擬 Memory Store 管理 client。"""

    def __init__(self):
        self._stores: dict[str, MockMemoryStore] = {}

    def create(self, name: str, description: str = "") -> MockMemoryStore:
        store_id = f"store_{len(self._stores) + 1:04d}"
        store = MockMemoryStore(store_id, name, description)
        self._stores[store_id] = store
        return store

    def get(self, store_id: str) -> Optional[MockMemoryStore]:
        return self._stores.get(store_id)


# ─────────────────────────────────────────────────────
#  材料科學 Playbook 初始內容
# ─────────────────────────────────────────────────────

MATERIALS_PLAYBOOK_SEED = {
    "extraction/composition_rules": (
        "## 組成提取規則\n"
        "1. 優先從摘要和結論提取化學式\n"
        "2. 注意 doping 濃度（通常用 mol% 或 at%）\n"
        "3. 區分 host material 和 dopant\n"
        "4. 鈣鈦礦格式：ABX3（A=有機陽離子, B=金屬, X=鹵素）\n"
        "5. 'x=0.1' 等變數定義在首次出現處"
    ),
    "extraction/synthesis_patterns": (
        "## 合成方法識別\n"
        "- solid-state: 高溫燒結（>800°C），粉末混合\n"
        "- sol-gel: 前驅物溶液 → 凝膠 → 煅燒\n"
        "- hydrothermal: 高壓釜，水溶液，100-300°C\n"
        "- CVD: 氣相沉積，適合薄膜\n"
        "- spin-coat: 溶液塗佈，鈣鈦礦太陽能電池常用"
    ),
    "extraction/property_units": (
        "## 性質標準單位\n"
        "- band_gap: eV\n"
        "- efficiency (PCE): %\n"
        "- thermal_conductivity: W/(m·K)\n"
        "- electrical_conductivity: S/m\n"
        "- ZT (thermoelectric figure of merit): dimensionless\n"
        "- lattice_parameter: Å\n"
        "- density: g/cm³"
    ),
    "heuristics/table_ambiguity": (
        "## 表格歧義處理\n"
        "- 'typical' 在 caption → 代表性單一樣品，非平均\n"
        "- 無標準差 → 可能是計算值而非實驗值\n"
        "- 多個值但無統計 → 列出所有，標 note\n"
        "- 表格標題含 'best' → 最佳樣品，非典型"
    ),
}


# ─────────────────────────────────────────────────────
#  Demo 主流程
# ─────────────────────────────────────────────────────

def demo_memory_stores():
    print("=" * 60)
    print("  Memory Store 完整操作 Demo")
    print("=" * 60)

    client = MockMemoryStoreClient()

    # ── 1. 建立 store ──────────────────────────────────
    print("\n[1] 建立 Memory Store")
    playbook_store = client.create(
        name="materials-extraction-playbook",
        description="材料科學文獻提取規則與啟發式方法",
    )

    # ── 2. Seed 初始內容 ───────────────────────────────
    print("\n[2] Seed 初始內容")
    playbook_store.seed(MATERIALS_PLAYBOOK_SEED)

    # ── 3. Read ────────────────────────────────────────
    print("\n[3] 讀取 entry")
    entry = playbook_store.read("extraction/composition_rules")
    print(f"  version={entry['version']}")
    print(f"  value preview: {entry['value'][:80]}...")

    # ── 4. 讀取不存在的 key ────────────────────────────
    print("\n[4] 讀取不存在的 key")
    missing = playbook_store.read("nonexistent/key")
    print(f"  result: {missing}")

    # ── 5. Update（append 新規則）─────────────────────
    print("\n[5] Update entry（append 新規則）")
    current = playbook_store.read("heuristics/table_ambiguity")
    new_value = current["value"] + "\n- 'nominally' → 理論值，需注意實際偏差"
    updated = playbook_store.write(
        key="heuristics/table_ambiguity",
        value=new_value,
        precondition_version=current["version"],  # ← safe write
    )
    print(f"  update 後 version={updated['version']}")

    # ── 6. Precondition 失敗示範 ───────────────────────
    print("\n[6] Precondition 失敗（模擬並發寫入衝突）")
    try:
        playbook_store.write(
            key="heuristics/table_ambiguity",
            value="試圖用舊 version 寫入",
            precondition_version=1,  # ← 當前是 version 2，應失敗
        )
    except ValueError as e:
        print(f"  ✓ 預期錯誤：{e}")

    # ── 7. Version History ─────────────────────────────
    print("\n[7] 查詢 Version History")
    history = playbook_store.history("heuristics/table_ambiguity")
    for h in history:
        print(f"  v{h['version']} ({h['updated_at'][:19]}): {h['value'][:60]}...")

    # ── 8. List all keys ───────────────────────────────
    print("\n[8] 列出所有 Keys")
    playbook_store.list_keys()

    # ── 9. Delete ──────────────────────────────────────
    print("\n[9] 刪除 entry")
    playbook_store.delete("extraction/composition_rules")
    playbook_store.list_keys()

    # ── 10. Attach 到 session（概念示範）─────────────
    print("\n[10] 將 store attach 到 session（概念示範）")
    session_config = {
        "agent_id": "agent_0001",
        "memory_stores": [
            {
                "store_id": playbook_store.id,
                "access": "read-write",
                # Agent 可以讀取和修改 playbook
            }
        ],
    }
    print(f"  Session config: {json.dumps(session_config, indent=2)}")

    print("\n✅ Memory Store Demo 完成！")
    print("   下一步：04_multi_store_session.py — 多個 store 的 access control")

    return playbook_store


if __name__ == "__main__":
    demo_memory_stores()
