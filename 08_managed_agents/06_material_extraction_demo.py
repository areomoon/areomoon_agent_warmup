"""
材料科學論文抽取端對端 Demo
============================

本模組整合本章所有概念（Session 管理、自定義工具、記憶壓縮、串流輸出），
展示一個完整的材料科學論文結構化抽取工作流。

工作流架構（Generator-Reflector 模式）：
  ┌─────────────────────────────────────────────────────────┐
  │  論文段落 (raw text)                                     │
  │       ↓                                                  │
  │  [Step 1] 初步抽取 → structured_data (JSON)             │
  │       ↓                                                  │
  │  [Step 2] 工具驗證 → 查詢已知數據庫，標記異常值          │
  │       ↓                                                  │
  │  [Step 3] 反思修正 → 修正錯誤欄位，補充缺漏值           │
  │       ↓                                                  │
  │  [Step 4] 持久化   → 儲存至 JSONL + 更新 Session 記憶   │
  └─────────────────────────────────────────────────────────┘

目標抽取欄位（VO₂ 薄膜案例）：
  - material, chemical_formula
  - deposition_method, target_purity_percent
  - ar_o2_flow_ratio, base_pressure_torr, working_pressure_mTorr
  - substrate_temperature_C, deposition_rate_nm_per_min
  - film_thickness_nm, SMT_temperature_C, resistance_change_orders

Mock 模式說明：
  未設定 ANTHROPIC_API_KEY 時，使用預定義的抽取結果展示完整流程。

參考資料：
  - ACE Generator-Reflector: https://arxiv.org/abs/2510.04618
  - VO₂ 薄膜製備案例: https://doi.org/10.1016/j.apsusc.2015.01.050
"""

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── 常數設定 ──────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-5"
MAX_TOKENS = 2048
OUTPUT_LOG = Path("/tmp/material_extractions.jsonl")

# ── 論文測試段落 ──────────────────────────────────────────────────────────────

SAMPLE_PAPER_TEXT = """
The synthesis of VO₂ thin films was carried out using reactive DC magnetron sputtering.
A vanadium metal target (99.95% purity) was sputtered in a mixed Ar/O₂ atmosphere with
a flow ratio of 15:1 (Ar:O₂). The base pressure before deposition was 5 × 10⁻⁷ Torr,
and the working pressure during sputtering was 5 mTorr. The substrate temperature was
maintained at 500°C throughout the deposition. Film deposition rate was 2 nm/min,
yielding films of 100 nm thickness after 50 minutes. The VO₂ films underwent a sharp
semiconductor-to-metal transition at 68°C (341 K) with a resistance change of four
orders of magnitude (ΔR/R ~ 10⁴). X-ray diffraction confirmed the monoclinic VO₂ (M1)
phase with dominant (011) orientation. The activation energy for the MIT was determined
to be 0.48 eV from Arrhenius analysis of the resistivity data.
"""

# ── 目標抽取 Schema ───────────────────────────────────────────────────────────

EXTRACTION_SCHEMA = {
    "material": None,
    "chemical_formula": None,
    "deposition_method": None,
    "target_purity_percent": None,
    "ar_o2_flow_ratio": None,
    "base_pressure_torr": None,
    "working_pressure_mTorr": None,
    "substrate_temperature_C": None,
    "deposition_rate_nm_per_min": None,
    "film_thickness_nm": None,
    "deposition_duration_min": None,
    "SMT_temperature_C": None,
    "SMT_temperature_K": None,
    "resistance_change_orders": None,
    "crystal_phase": None,
    "preferred_orientation": None,
    "activation_energy_eV": None,
}

# Mock 模式預定義結果
MOCK_INITIAL_EXTRACTION = {
    "material": "VO₂",
    "chemical_formula": "VO2",
    "deposition_method": "reactive DC magnetron sputtering",
    "target_purity_percent": 99.95,
    "ar_o2_flow_ratio": "15:1",
    "base_pressure_torr": 5e-7,
    "working_pressure_mTorr": 5.0,
    "substrate_temperature_C": 500,
    "deposition_rate_nm_per_min": 2.0,
    "film_thickness_nm": 100,
    "deposition_duration_min": 50,
    "SMT_temperature_C": 68,
    "SMT_temperature_K": 341,
    "resistance_change_orders": 4,
    "crystal_phase": "monoclinic VO₂ (M1)",
    "preferred_orientation": "(011)",
    "activation_energy_eV": 0.48,
}

MOCK_REFLECTION = {
    "errors_found": [],
    "corrections": {},
    "missing_fields": [],
    "confidence": "high",
    "note": "所有欄位均成功抽取，數值與已知文獻一致（SMT 68°C、ΔR ~ 10⁴）。",
}

# ── 工具定義 ──────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "validate_against_literature",
        "description": (
            "對照已知文獻數據驗證抽取結果是否合理，"
            "回傳可疑欄位列表與建議的合理值範圍。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "material": {"type": "string"},
                "extracted_data": {
                    "type": "object",
                    "description": "待驗證的抽取結果 dict",
                },
            },
            "required": ["material", "extracted_data"],
        },
    },
    {
        "name": "save_to_database",
        "description": "將最終驗證後的抽取結果儲存至材料數據庫。",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string"},
                "extraction": {"type": "object"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["paper_id", "extraction", "confidence"],
        },
    },
]

# ── 工具實作 ──────────────────────────────────────────────────────────────────

KNOWN_RANGES = {
    "VO2": {
        "SMT_temperature_C": (55, 80),
        "resistance_change_orders": (3, 5),
        "substrate_temperature_C": (400, 600),
        "working_pressure_mTorr": (1, 20),
    }
}

_database = []


def validate_against_literature(material: str, extracted_data: dict) -> dict:
    """驗證抽取數據是否在已知合理範圍內。"""
    ranges = KNOWN_RANGES.get(material.replace("₂", "2").replace("₄", "4"), {})
    suspicious = []
    for field_name, (lo, hi) in ranges.items():
        val = extracted_data.get(field_name)
        if val is not None and not (lo <= val <= hi):
            suspicious.append({
                "field": field_name,
                "extracted": val,
                "expected_range": f"{lo}–{hi}",
            })

    return {
        "material": material,
        "suspicious_fields": suspicious,
        "validation_passed": len(suspicious) == 0,
        "fields_checked": len(ranges),
    }


def save_to_database(paper_id: str, extraction: dict, confidence: str) -> dict:
    """儲存最終抽取結果。"""
    record = {
        "paper_id": paper_id,
        "extraction": extraction,
        "confidence": confidence,
        "saved_at": time.time(),
    }
    _database.append(record)
    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "saved", "total_records": len(_database), "paper_id": paper_id}


TOOL_DISPATCH = {
    "validate_against_literature": validate_against_literature,
    "save_to_database": save_to_database,
}


# ── 抽取流程 ──────────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """完整抽取任務的結果容器。"""
    paper_id: str
    raw_text: str
    initial_extraction: dict = field(default_factory=dict)
    reflection: dict = field(default_factory=dict)
    final_extraction: dict = field(default_factory=dict)
    validation_report: dict = field(default_factory=dict)
    saved: bool = False
    processing_time_s: float = 0.0


def step1_initial_extraction(text: str, client, mock_mode: bool) -> dict:
    """步驟一：呼叫 Agent 從論文段落初步抽取結構化數據。"""
    if mock_mode:
        return MOCK_INITIAL_EXTRACTION.copy()

    prompt = (
        f"請從以下材料科學論文段落中抽取所有數值性參數，"
        f"以 JSON 格式回傳（使用以下欄位名稱，若未提及填 null）：\n"
        f"{json.dumps(list(EXTRACTION_SCHEMA.keys()), ensure_ascii=False)}\n\n"
        f"論文段落：\n{text}\n\n"
        f"僅回傳 JSON，不要其他說明文字。"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    text_response = response.content[0].text.strip()
    # 去除 markdown code block
    if text_response.startswith("```"):
        text_response = "\n".join(text_response.split("\n")[1:-1])
    return json.loads(text_response)


def step2_validate_with_tools(extraction: dict, client, mock_mode: bool) -> dict:
    """步驟二：使用工具驗證抽取結果的合理性。"""
    if mock_mode:
        result = validate_against_literature("VO₂", extraction)
        return result

    messages = [
        {
            "role": "user",
            "content": (
                f"請驗證以下 VO₂ 薄膜抽取結果是否與文獻一致，並說明任何可疑之處：\n"
                f"{json.dumps(extraction, ensure_ascii=False, indent=2)}"
            ),
        }
    ]

    for _ in range(3):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
        )
        if response.stop_reason == "end_turn":
            break
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                fn = TOOL_DISPATCH.get(block.name)
                result = fn(**block.input) if fn else {"error": "unknown tool"}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            messages.append({"role": "user", "content": tool_results})

    return {"validation": "completed", "messages": len(messages)}


def step3_reflect_and_correct(
    initial: dict, validation: dict, client, mock_mode: bool
) -> dict:
    """步驟三：根據驗證結果進行反思與修正。"""
    if mock_mode:
        return {**initial, **MOCK_REFLECTION.get("corrections", {})}

    prompt = (
        f"初步抽取結果：\n{json.dumps(initial, ensure_ascii=False, indent=2)}\n\n"
        f"驗證報告：\n{json.dumps(validation, ensure_ascii=False, indent=2)}\n\n"
        f"請修正任何錯誤欄位並補充可推算的缺漏值，回傳修正後的完整 JSON。"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return initial  # 解析失敗時保留初步抽取結果


def run_full_pipeline(
    paper_id: str,
    paper_text: str,
    client,
    mock_mode: bool = False,
) -> ExtractionResult:
    """執行完整的四步驟論文抽取工作流。"""
    result = ExtractionResult(paper_id=paper_id, raw_text=paper_text)
    start_time = time.time()

    print(f"\n[Step 1] 初步抽取中...")
    result.initial_extraction = step1_initial_extraction(paper_text, client, mock_mode)
    print(f"         抽取欄位：{sum(1 for v in result.initial_extraction.values() if v is not None)}"
          f"/{len(EXTRACTION_SCHEMA)}")

    print("[Step 2] 工具驗證中...")
    result.validation_report = step2_validate_with_tools(result.initial_extraction, client, mock_mode)
    suspicious_count = len(result.validation_report.get("suspicious_fields", []))
    print(f"         驗證通過：{result.validation_report.get('validation_passed', True)}"
          f"，可疑欄位：{suspicious_count} 個")

    print("[Step 3] 反思修正中...")
    result.final_extraction = step3_reflect_and_correct(
        result.initial_extraction, result.validation_report, client, mock_mode
    )
    print(f"         最終欄位：{sum(1 for v in result.final_extraction.values() if v is not None)}"
          f"/{len(EXTRACTION_SCHEMA)}")

    print("[Step 4] 持久化儲存...")
    OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    save_result = save_to_database(paper_id, result.final_extraction, "high")
    result.saved = save_result["status"] == "saved"
    print(f"         儲存成功：{result.saved}，資料庫總記錄：{save_result['total_records']}")

    result.processing_time_s = round(time.time() - start_time, 2)
    return result


# ── 主入口 ────────────────────────────────────────────────────────────────────

def run_example() -> None:
    """執行端對端材料科學論文抽取示範。"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    mock_mode = not bool(api_key)

    if mock_mode:
        print("[Mock 模式] 使用預定義抽取結果，展示完整四步驟流程\n")
        client = None
    else:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        print(f"[真實模式] 模型：{MODEL}\n")

    print("=" * 60)
    print("材料科學論文抽取 End-to-End Demo")
    print("論文段落（前 100 字）：")
    print(f"  {SAMPLE_PAPER_TEXT.strip()[:100]}...")
    print("=" * 60)

    result = run_full_pipeline(
        paper_id="arXiv:2401.demo001",
        paper_text=SAMPLE_PAPER_TEXT,
        client=client,
        mock_mode=mock_mode,
    )

    print("\n" + "=" * 60)
    print("最終抽取結果：")
    print(json.dumps(result.final_extraction, ensure_ascii=False, indent=2))
    print(f"\n處理時間：{result.processing_time_s}s")
    print(f"結果已儲存：{result.saved}（路徑：{OUTPUT_LOG}）")

    # TODO: 批次處理多篇論文，統計整體 precision/recall
    # TODO: 與 03_agent_patterns 的 Generator-Reflector 整合，加入 ACE Playbook 更新
    # TODO: 實作 PDF 解析前處理（配合 07_multimodal 的多模態輸入）


if __name__ == "__main__":
    run_example()
