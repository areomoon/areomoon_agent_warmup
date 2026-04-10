"""
06_material_extraction_demo.py — 材料科學 Extraction 完整 Demo

學習目標：
    整合本模組所有概念，模擬真實的材料科學文獻提取 pipeline：
    1. Agent 讀一篇論文 PDF（mock）
    2. 檢查 materials domain playbook memory store
    3. 提取 structured data（composition, properties, synthesis method）
    4. 將 extraction case 寫回 cases memory store
    5. 用 tool_use 模式呼叫 materials_property_lookup custom tool
    6. 產出結構化 JSON + markdown report

架構對應：
    ┌──────────────────────────────────────────────────────────┐
    │  User Message: "分析這篇論文"                              │
    │         ↓                                                │
    │  Agent reads playbook (memory_read)                      │
    │         ↓                                                │
    │  Agent reads paper PDF (tool: files / bash)              │
    │         ↓                                                │
    │  Agent extracts data → calls lookup_material_property    │
    │         ↓                                                │
    │  Agent writes case to cases store (memory_write)         │
    │         ↓                                                │
    │  Agent returns structured JSON + markdown report         │
    └──────────────────────────────────────────────────────────┘

使用方式：
    python 06_material_extraction_demo.py
"""

import json
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────
#  模擬論文內容（Mock PDF）
# ─────────────────────────────────────────────────────

MOCK_PAPER = {
    "doi": "10.1021/acs.nanolett.2024.12345",
    "title": "High-Efficiency Cesium-Doped Formamidinium Lead Iodide Perovskite Solar Cells",
    "abstract": """
        We demonstrate a power conversion efficiency (PCE) of 25.1% in cesium-doped
        formamidinium lead iodide (Cs0.1FA0.9PbI3) perovskite solar cells.
        The optimized composition shows a band gap of 1.53 eV, open-circuit voltage (Voc)
        of 1.18 V, short-circuit current density (Jsc) of 24.8 mA/cm², and fill factor
        (FF) of 85.3%. Devices were fabricated using a two-step spin-coating method
        at room temperature, followed by annealing at 150°C for 30 minutes.
        The improved performance is attributed to suppressed ion migration and
        enhanced crystallinity due to Cs incorporation.
    """.strip(),
    "tables": [
        {
            "caption": "Table 1. Photovoltaic parameters of best-performing devices",
            "data": [
                {"composition": "FA0.9Cs0.1PbI3", "PCE": 25.1, "Voc": 1.18,
                 "Jsc": 24.8, "FF": 85.3, "note": "champion device"},
                {"composition": "FAPbI3", "PCE": 22.3, "Voc": 1.12,
                 "Jsc": 24.1, "FF": 82.5, "note": "reference"},
            ],
        }
    ],
    "synthesis_section": """
        Two-step spin-coating: First, PbI2 in DMF (1.3M) was spin-coated at 1500 rpm
        for 30s, then annealed at 70°C for 1 min. Second, FAI/CsI in IPA (60 mM)
        was spin-coated at 2000 rpm for 30s. Final annealing: 150°C, 30 min in N2.
    """.strip(),
}


# ─────────────────────────────────────────────────────
#  Materials Property Lookup Tool（複用 02_custom_tools.py）
# ─────────────────────────────────────────────────────

MATERIALS_DB = {
    ("Cs0.1FA0.9PbI3", "band_gap"): {"value": 1.53, "unit": "eV", "confidence": "high",
                                      "source": "This work / Snaith group"},
    ("FAPbI3", "band_gap"):         {"value": 1.48, "unit": "eV", "confidence": "high",
                                      "source": "Sutherland 2016"},
    ("Cs0.1FA0.9PbI3", "density"):  {"value": 4.21, "unit": "g/cm³", "confidence": "medium",
                                      "source": "Calculated"},
}


def lookup_material_property(material: str, property_name: str) -> dict:
    """查詢材料性質（模擬 Materials Project）。"""
    key = (material, property_name)
    if key in MATERIALS_DB:
        result = MATERIALS_DB[key].copy()
        result.update({"material": material, "property": property_name})
        return result
    return {
        "material": material, "property": property_name,
        "value": None, "unit": None, "confidence": "low",
        "note": "資料庫中無此材料性質，建議查閱 Materials Project",
    }


# ─────────────────────────────────────────────────────
#  Extraction Pipeline
# ─────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """標準化的提取結果。"""
    paper_doi: str
    paper_title: str

    # 材料
    primary_material: str = ""
    composition_formula: str = ""
    material_class: str = ""  # perovskite, oxide, semiconductor...

    # 性質
    band_gap_eV: Optional[float] = None
    pce_percent: Optional[float] = None
    voc_V: Optional[float] = None
    jsc_mA_cm2: Optional[float] = None
    ff_percent: Optional[float] = None

    # 合成
    synthesis_method: str = ""
    annealing_temp_C: Optional[float] = None
    annealing_time_min: Optional[float] = None
    atmosphere: str = ""

    # 提取品質
    confidence: str = "medium"
    extraction_notes: list[str] = field(default_factory=list)
    source_location: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != [] and v != ""}


class MaterialExtractionAgent:
    """
    模擬 Managed Agent 的材料科學提取行為。
    每個方法對應 agent 執行的一個步驟。
    """

    def __init__(self, paper: dict, playbook: dict, cases_store: list):
        self.paper = paper
        self.playbook = playbook
        self.cases_store = cases_store
        self.log: list[str] = []

    def _log(self, msg: str):
        self.log.append(msg)
        print(f"  [agent] {msg}")

    def step1_read_playbook(self) -> dict:
        """Step 1：讀取 playbook 規則（對應 memory_read 事件）。"""
        self._log("memory_read: playbook/rules/step1_locate")
        self._log("memory_read: playbook/extraction/composition_rules")
        self._log("memory_read: playbook/heuristics/confidence")
        return self.playbook

    def step2_parse_abstract(self) -> dict:
        """Step 2：從摘要提取材料和性質（文字分析）。"""
        self._log("分析 abstract：識別鈣鈦礦材料")
        # 模擬 NLP 提取
        return {
            "composition": "Cs0.1FA0.9PbI3",
            "material_class": "perovskite",
            "pce": 25.1,
            "voc": 1.18,
            "jsc": 24.8,
            "ff": 85.3,
            "band_gap": 1.53,
        }

    def step3_lookup_properties(self, material: str) -> dict:
        """Step 3：呼叫 tool 查詢 band gap（對應 tool_use 事件）。"""
        self._log(f"tool_use: lookup_material_property(material={material}, property=band_gap)")
        result = lookup_material_property(material, "band_gap")
        self._log(f"tool_result: {result['value']} {result['unit']} (confidence={result['confidence']})")
        return result

    def step4_parse_synthesis(self) -> dict:
        """Step 4：從 synthesis 段落提取合成條件。"""
        self._log("分析 synthesis section：提取 spin-coating 條件")
        text = self.paper["synthesis_section"]

        # 模擬提取邏輯
        synthesis = {
            "method": "two-step spin-coating",
            "annealing_temp_C": 150,
            "annealing_time_min": 30,
            "atmosphere": "N2",
            "solvent_step1": "DMF",
            "speed_step1_rpm": 1500,
            "solvent_step2": "IPA",
            "speed_step2_rpm": 2000,
        }
        self._log(f"提取合成條件：{synthesis['method']} at {synthesis['annealing_temp_C']}°C")
        return synthesis

    def step5_check_table(self) -> dict:
        """Step 5：檢查表格，確認 champion device 數據。"""
        self._log("掃描 Table 1：識別 champion device")
        # 檢查 playbook heuristic：'best' 或 'champion' → 最佳樣品
        table = self.paper["tables"][0]
        champion = next(
            (row for row in table["data"] if "champion" in row.get("note", "")), None
        )
        if champion:
            self._log(f"Champion device: PCE={champion['PCE']}%（高信心）")
        return champion or {}

    def step6_write_case(self, result: ExtractionResult) -> dict:
        """Step 6：將提取結果寫入 cases store（對應 memory_write 事件）。"""
        case_key = f"paper/{result.paper_doi.replace('/', '_')}/v1"
        case_value = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        self._log(f"memory_write: cases/{case_key}")

        entry = {"key": case_key, "value": case_value, "version": 1}
        self.cases_store.append(entry)
        return entry

    def run(self) -> tuple["ExtractionResult", str]:
        """執行完整提取 pipeline，回傳 (ExtractionResult, markdown_report)。"""
        self._log("=== 開始材料科學提取 Pipeline ===")

        playbook = self.step1_read_playbook()
        abstract_data = self.step2_parse_abstract()
        material = abstract_data["composition"]

        db_property = self.step3_lookup_properties(material)
        synthesis = self.step4_parse_synthesis()
        table_data = self.step5_check_table()

        # 整合結果
        result = ExtractionResult(
            paper_doi=self.paper["doi"],
            paper_title=self.paper["title"],
            primary_material=material,
            composition_formula=material,
            material_class=abstract_data["material_class"],
            band_gap_eV=db_property.get("value") or abstract_data.get("band_gap"),
            pce_percent=abstract_data["pce"],
            voc_V=abstract_data["voc"],
            jsc_mA_cm2=abstract_data["jsc"],
            ff_percent=abstract_data["ff"],
            synthesis_method=synthesis["method"],
            annealing_temp_C=synthesis["annealing_temp_C"],
            annealing_time_min=synthesis["annealing_time_min"],
            atmosphere=synthesis["atmosphere"],
            confidence="high",
            extraction_notes=[
                "Band gap 從 UV-Vis 數據確認",
                "PCE 為 champion device，非統計平均",
                "合成條件為 N2 環境 two-step spin-coating",
            ],
            source_location=["Abstract", "Table 1", "Synthesis section"],
        )

        self.step6_write_case(result)

        # 產出 markdown report
        report = self._generate_report(result)

        self._log("=== Pipeline 完成 ===")
        return result, report

    def _generate_report(self, result: ExtractionResult) -> str:
        """產出 markdown 格式報告。"""
        return f"""# 材料科學提取報告

## 論文資訊
- **DOI**: {result.paper_doi}
- **標題**: {result.paper_title}

## 材料
| 欄位 | 值 |
|------|-----|
| 主要材料 | {result.primary_material} |
| 化學式 | {result.composition_formula} |
| 材料類別 | {result.material_class} |

## 性質數據
| 性質 | 值 | 單位 |
|------|-----|------|
| Band gap | {result.band_gap_eV} | eV |
| PCE（效率）| {result.pce_percent} | % |
| Voc | {result.voc_V} | V |
| Jsc | {result.jsc_mA_cm2} | mA/cm² |
| FF | {result.ff_percent} | % |

## 合成條件
| 參數 | 值 |
|------|-----|
| 方法 | {result.synthesis_method} |
| 退火溫度 | {result.annealing_temp_C}°C |
| 退火時間 | {result.annealing_time_min} min |
| 氣氛 | {result.atmosphere} |

## 提取品質
- **信心度**: {result.confidence}
- **資料來源**: {', '.join(result.source_location)}
- **備註**:
{chr(10).join(f'  - {n}' for n in result.extraction_notes)}
"""


# ─────────────────────────────────────────────────────
#  Demo 主流程
# ─────────────────────────────────────────────────────

def demo_material_extraction():
    print("=" * 60)
    print("  材料科學 Extraction Demo")
    print("=" * 60)

    # 模擬 playbook（正常應從 memory store 讀取）
    mock_playbook = {
        "composition_rules": "鈣鈦礦格式：ABX3",
        "confidence_heuristics": "high=摘要直接陳述, medium=表格, low=圖表",
    }
    mock_cases_store = []

    print("\n[Paper] 開始分析論文")
    print(f"  DOI: {MOCK_PAPER['doi']}")
    print(f"  Title: {MOCK_PAPER['title'][:60]}...")

    print("\n[Pipeline 執行]")
    agent = MaterialExtractionAgent(MOCK_PAPER, mock_playbook, mock_cases_store)
    result, report = agent.run()

    print("\n=== 結構化 JSON 輸出 ===")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    print("\n=== Markdown Report ===")
    print(report)

    print(f"\n=== Cases Store ({len(mock_cases_store)} entries) ===")
    for entry in mock_cases_store:
        print(f"  key={entry['key']}, version={entry['version']}")

    print("\n✅ 材料科學 Extraction Demo 完成！")
    print("   下一步：notebook/managed_agents_lab.ipynb — 互動式學習")


if __name__ == "__main__":
    demo_material_extraction()
