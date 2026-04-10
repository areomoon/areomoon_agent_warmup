"""
02_custom_tools.py — Custom Tools 定義與呼叫

學習目標：
    1. 理解 custom tool schema（JSON Schema 格式）
    2. 定義 3 個材料科學相關工具
    3. 掌握 tool description 最佳實務
    4. 模擬 agent → tool call → result 完整流程

官方文件：https://platform.claude.com/docs/en/managed-agents/tools

⚠️  無 beta access 可用 mock 模式完整學習。

使用方式：
    python 02_custom_tools.py
"""

import json
from typing import Any

# ─────────────────────────────────────────────────────
#  Tool Schema 定義
#  最佳實務：
#    - description 要詳細、說明何時用、輸出什麼格式
#    - 參數用 enum 限制合法值
#    - 必填參數放 required
# ─────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        # ✅ 最佳實務：說明回傳格式和單位
        "description": (
            "取得指定城市的當前天氣資訊。"
            "回傳 JSON 格式：{city, temperature_celsius, humidity_percent, condition}。"
            "適用於需要知道實驗室或採樣地點天氣條件的情境。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名稱，例如 'Taipei', 'Tokyo', 'London'",
                },
                "unit": {
                    "type": "string",
                    # ✅ 最佳實務：用 enum 限制合法值
                    "enum": ["celsius", "fahrenheit"],
                    "description": "溫度單位，預設 celsius",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "lookup_material_property",
        # ✅ 最佳實務：說明資料來源和限制
        "description": (
            "查詢材料的物理或化學性質。"
            "資料來源：Materials Project 資料庫（模擬）。"
            "回傳 JSON：{material, property, value, unit, source, confidence}。"
            "若材料或性質不在資料庫中，回傳 confidence=low 並說明限制。"
            "適用性質：band_gap, density, melting_point, thermal_conductivity, "
            "electrical_conductivity, hardness, young_modulus。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "material": {
                    "type": "string",
                    "description": "材料名稱或化學式，例如 'TiO2', 'GaN', 'MAPbI3'",
                },
                "property": {
                    "type": "string",
                    "enum": [
                        "band_gap", "density", "melting_point",
                        "thermal_conductivity", "electrical_conductivity",
                        "hardness", "young_modulus",
                    ],
                    "description": "要查詢的性質",
                },
                "phase": {
                    "type": "string",
                    "enum": ["bulk", "thin_film", "nanoparticle"],
                    "description": "材料相態，影響性質數值，預設 bulk",
                },
            },
            "required": ["material", "property"],
        },
    },
    {
        "name": "convert_units",
        # ✅ 最佳實務：說明精度和邊界條件
        "description": (
            "科學單位換算工具。"
            "支援：能量（eV, J, kJ/mol, cm⁻¹）、溫度（K, °C, °F）、"
            "壓力（Pa, GPa, atm, bar）、長度（m, nm, Å, pm）。"
            "精度：float64，不處理極端值（< 1e-300 或 > 1e300）。"
            "回傳 JSON：{value, from_unit, to_unit, result, formula}。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "要換算的數值",
                },
                "from_unit": {
                    "type": "string",
                    "description": "原始單位，例如 'eV', 'K', 'GPa', 'nm'",
                },
                "to_unit": {
                    "type": "string",
                    "description": "目標單位",
                },
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
]


# ─────────────────────────────────────────────────────
#  Tool 實作（真實邏輯，可直接使用）
# ─────────────────────────────────────────────────────

# 材料性質資料庫（模擬 Materials Project）
MATERIALS_DB = {
    ("TiO2", "band_gap", "bulk"):          {"value": 3.0,    "unit": "eV",      "source": "Materials Project"},
    ("GaN", "band_gap", "bulk"):           {"value": 3.4,    "unit": "eV",      "source": "Materials Project"},
    ("MAPbI3", "band_gap", "bulk"):        {"value": 1.57,   "unit": "eV",      "source": "Snaith 2013"},
    ("TiO2", "density", "bulk"):           {"value": 4.23,   "unit": "g/cm³",   "source": "Materials Project"},
    ("GaN", "melting_point", "bulk"):      {"value": 2500.0, "unit": "°C",      "source": "Materials Project"},
    ("TiO2", "thermal_conductivity", "bulk"): {"value": 11.8, "unit": "W/(m·K)", "source": "Materials Project"},
}

# 單位換算係數
UNIT_CONVERSIONS = {
    ("eV", "J"):       lambda x: x * 1.602176634e-19,
    ("J", "eV"):       lambda x: x / 1.602176634e-19,
    ("eV", "kJ/mol"):  lambda x: x * 96.4853,
    ("K", "°C"):       lambda x: x - 273.15,
    ("°C", "K"):       lambda x: x + 273.15,
    ("°C", "°F"):      lambda x: x * 9/5 + 32,
    ("GPa", "Pa"):     lambda x: x * 1e9,
    ("Pa", "GPa"):     lambda x: x / 1e9,
    ("nm", "Å"):       lambda x: x * 10,
    ("Å", "nm"):       lambda x: x / 10,
    ("Å", "pm"):       lambda x: x * 100,
    ("nm", "m"):       lambda x: x * 1e-9,
}


def get_weather(city: str, unit: str = "celsius") -> dict:
    """取得天氣資訊（模擬）。"""
    mock_data = {
        "Taipei":  {"temp_c": 28, "humidity": 75, "condition": "partly_cloudy"},
        "Tokyo":   {"temp_c": 22, "humidity": 60, "condition": "clear"},
        "London":  {"temp_c": 15, "humidity": 80, "condition": "overcast"},
    }
    data = mock_data.get(city, {"temp_c": 20, "humidity": 65, "condition": "unknown"})
    temp = data["temp_c"]
    if unit == "fahrenheit":
        temp = temp * 9/5 + 32

    return {
        "city": city,
        "temperature": round(temp, 1),
        "unit": unit,
        "humidity_percent": data["humidity"],
        "condition": data["condition"],
    }


def lookup_material_property(material: str, property: str,
                              phase: str = "bulk") -> dict:
    """查詢材料性質（模擬 Materials Project）。"""
    key = (material, property, phase)
    if key in MATERIALS_DB:
        result = MATERIALS_DB[key].copy()
        result.update({"material": material, "property": property,
                       "phase": phase, "confidence": "high"})
        return result

    # 查不到時回傳低信心結果
    return {
        "material": material,
        "property": property,
        "phase": phase,
        "value": None,
        "unit": None,
        "source": None,
        "confidence": "low",
        "note": f"'{material}' 的 {property} ({phase}) 不在資料庫中，請參考文獻。",
    }


def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    """科學單位換算。"""
    if from_unit == to_unit:
        return {"value": value, "from_unit": from_unit,
                "to_unit": to_unit, "result": value, "formula": "same unit"}

    converter = UNIT_CONVERSIONS.get((from_unit, to_unit))
    if converter is None:
        return {"error": f"不支援 {from_unit} → {to_unit} 的換算",
                "supported": list(UNIT_CONVERSIONS.keys())}

    result = converter(value)
    return {
        "value": value,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "result": round(result, 10),
        "formula": f"{value} {from_unit} = {result:.6g} {to_unit}",
    }


# ─────────────────────────────────────────────────────
#  Tool 路由（agent 呼叫 tool 時使用）
# ─────────────────────────────────────────────────────

TOOL_HANDLERS = {
    "get_weather":             get_weather,
    "lookup_material_property": lookup_material_property,
    "convert_units":           convert_units,
}


def handle_tool_call(tool_name: str, tool_input: dict) -> Any:
    """
    統一 tool 呼叫入口。
    在 Managed Agents 中，tool_use events 會攜帶 tool_name 和 input，
    你的程式碼負責路由到對應 handler 並回傳 result。
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return {"error": f"未知的 tool: {tool_name}"}
    return handler(**tool_input)


# ─────────────────────────────────────────────────────
#  Demo：模擬 agent tool call 流程
# ─────────────────────────────────────────────────────

def demo_tool_calls():
    """模擬 agent 在處理材料科學問題時的 tool call 流程。"""
    print("=" * 60)
    print("  Custom Tools Demo")
    print("=" * 60)

    # 模擬 agent 發出的 tool calls
    mock_tool_calls = [
        {
            "id": "toolu_001",
            "type": "tool_use",
            "name": "lookup_material_property",
            "input": {"material": "MAPbI3", "property": "band_gap"},
        },
        {
            "id": "toolu_002",
            "type": "tool_use",
            "name": "convert_units",
            "input": {"value": 1.57, "from_unit": "eV", "to_unit": "kJ/mol"},
        },
        {
            "id": "toolu_003",
            "type": "tool_use",
            "name": "lookup_material_property",
            "input": {"material": "Cu2O", "property": "band_gap"},  # 不在 DB
        },
        {
            "id": "toolu_004",
            "type": "tool_use",
            "name": "get_weather",
            "input": {"city": "Taipei", "unit": "celsius"},
        },
    ]

    print("\n=== Tool Schema（前兩個）===")
    for tool in TOOL_DEFINITIONS[:2]:
        print(f"\n  Tool: {tool['name']}")
        print(f"  Description: {tool['description'][:80]}...")
        print(f"  Required params: {tool['input_schema'].get('required', [])}")

    print("\n=== 模擬 Tool Call 流程 ===")
    for call in mock_tool_calls:
        print(f"\n  [tool_use] id={call['id']} name={call['name']}")
        print(f"  [input]    {json.dumps(call['input'], ensure_ascii=False)}")

        result = handle_tool_call(call["name"], call["input"])
        print(f"  [result]   {json.dumps(result, ensure_ascii=False, indent=None)}")

    print("\n✅ Custom Tools Demo 完成！")
    print("   下一步：03_memory_stores.py — 學習 Memory Store 操作")


if __name__ == "__main__":
    demo_tool_calls()
