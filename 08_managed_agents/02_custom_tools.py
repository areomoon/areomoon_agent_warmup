"""
自定義工具（Custom Tools）與 tool_use 回呼迴圈
================================================

本模組示範如何為 Managed Agent 定義自定義工具，
並實作完整的 tool_use → tool_result 回呼流程。

核心流程（ReAct 模式）：
  1. 使用者發送請求
  2. Agent 回傳含 `tool_use` 的回應（stop_reason = "tool_use"）
  3. 本地端執行工具函式
  4. 將 `tool_result` 附加至對話歷史
  5. 再次呼叫 API，取得最終文字回應
  6. 重複直到 stop_reason = "end_turn"

工具設計原則：
  - 每個工具有明確的 input_schema（JSON Schema 格式）
  - 工具名稱使用 snake_case，描述需精確（影響 Agent 決策）
  - tool_result 應包含結構化資料，而非自然語言描述

參考資料：
  - Tool Use 完整指南: https://docs.anthropic.com/en/docs/tool-use
  - ReAct Pattern (arXiv 2210.03629): https://arxiv.org/abs/2210.03629
"""

import os
import json
import math
from typing import Any
from dotenv import load_dotenv

load_dotenv()

# ── 常數設定 ──────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-5"
MAX_TOKENS = 2048
MAX_TOOL_ROUNDS = 5  # 防止無限迴圈

# ── 工具定義（JSON Schema） ───────────────────────────────────────────────────

TOOLS = [
    {
        "name": "lookup_material_property",
        "description": (
            "從材料科學資料庫查詢特定材料的物理或化學性質。"
            "適用於查詢相變溫度、電阻率、能隙等基本參數。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "material": {
                    "type": "string",
                    "description": "材料名稱或化學式，例如 'VO2'、'TiO2'、'LiFePO4'",
                },
                "property": {
                    "type": "string",
                    "description": "欲查詢的性質，例如 'transition_temperature'、'bandgap'、'resistivity'",
                },
            },
            "required": ["material", "property"],
        },
    },
    {
        "name": "calculate_film_thickness",
        "description": (
            "根據沉積速率與時間計算薄膜厚度，或進行相關的單位換算。"
            "支援 nm/min、Å/s、nm/cycle 等常見速率單位。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rate": {
                    "type": "number",
                    "description": "沉積速率（數值部分）",
                },
                "rate_unit": {
                    "type": "string",
                    "description": "速率單位：'nm_per_min'、'angstrom_per_sec'、'nm_per_cycle'",
                },
                "duration": {
                    "type": "number",
                    "description": "沉積時間（分鐘，或 ALD 週期數）",
                },
            },
            "required": ["rate", "rate_unit", "duration"],
        },
    },
    {
        "name": "parse_synthesis_conditions",
        "description": (
            "從自然語言文本中解析並結構化合成條件，"
            "回傳包含溫度、壓力、氣體比例、時間等欄位的 JSON 物件。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "包含合成條件描述的論文段落或實驗記錄",
                },
            },
            "required": ["text"],
        },
    },
]


# ── 工具實作（本地端執行）────────────────────────────────────────────────────

# 簡化的材料性質資料庫（實際應用中應連接真實資料庫）
MATERIAL_DB = {
    ("VO2", "transition_temperature"): {"value": 68, "unit": "°C", "note": "bulk; thin film may shift ±10°C"},
    ("VO2", "resistivity"): {"value": "10^-3 to 10^-2", "unit": "Ω·cm", "phase": "metallic"},
    ("TiO2", "bandgap"): {"value": 3.2, "unit": "eV", "phase": "anatase"},
    ("LiFePO4", "discharge_plateau"): {"value": 3.4, "unit": "V vs Li/Li+"},
}


def lookup_material_property(material: str, property: str) -> dict:
    """查詢材料性質資料庫，回傳結構化結果。"""
    key = (material.replace("₂", "2").replace("₄", "4"), property)
    result = MATERIAL_DB.get(key)
    if result:
        return {"found": True, "material": material, "property": property, **result}
    return {
        "found": False,
        "material": material,
        "property": property,
        "message": f"資料庫中無 {material} 的 {property} 數據，建議查閱 Materials Project 或 ICSD。",
    }


def calculate_film_thickness(rate: float, rate_unit: str, duration: float) -> dict:
    """計算薄膜厚度，支援多種速率單位。"""
    if rate_unit == "nm_per_min":
        thickness_nm = rate * duration
    elif rate_unit == "angstrom_per_sec":
        thickness_nm = rate * duration * 60 / 10  # Å/s → nm/min → nm
    elif rate_unit == "nm_per_cycle":
        thickness_nm = rate * duration  # duration 為週期數
    else:
        return {"error": f"不支援的速率單位：{rate_unit}"}

    return {
        "thickness_nm": round(thickness_nm, 2),
        "thickness_angstrom": round(thickness_nm * 10, 1),
        "input": {"rate": rate, "rate_unit": rate_unit, "duration": duration},
    }


def parse_synthesis_conditions(text: str) -> dict:
    """從文本中以關鍵字比對方式抽取合成條件（簡化示範版）。"""
    import re

    conditions = {}

    # 溫度：支援 °C 和 K
    temp_match = re.search(r"(\d+)\s*°C", text)
    if temp_match:
        conditions["temperature_C"] = int(temp_match.group(1))

    # 壓力：支援 mTorr、Torr、Pa
    pressure_match = re.search(r"([\d.]+)\s*mTorr", text, re.IGNORECASE)
    if pressure_match:
        conditions["pressure_mTorr"] = float(pressure_match.group(1))

    # 時間：分鐘
    time_match = re.search(r"(\d+)\s*min", text, re.IGNORECASE)
    if time_match:
        conditions["duration_min"] = int(time_match.group(1))

    # 流量比：Ar:O₂
    ratio_match = re.search(r"(\d+):(\d+)\s*\(Ar:O", text)
    if ratio_match:
        conditions["Ar_O2_ratio"] = f"{ratio_match.group(1)}:{ratio_match.group(2)}"

    conditions["raw_text_length"] = len(text)
    conditions["fields_extracted"] = len(conditions) - 1  # 不計 raw_text_length
    return conditions


TOOL_DISPATCH = {
    "lookup_material_property": lookup_material_property,
    "calculate_film_thickness": calculate_film_thickness,
    "parse_synthesis_conditions": parse_synthesis_conditions,
}


# ── tool_use 回呼迴圈 ─────────────────────────────────────────────────────────

def run_agent_with_tools(
    user_message: str,
    client,
    mock_mode: bool = False,
    verbose: bool = True,
) -> str:
    """執行含工具呼叫的 Agent 迴圈，直到取得最終回應。

    參數：
      user_message — 使用者問題
      client       — Anthropic client（mock 模式下為 None）
      mock_mode    — 是否使用預設回應
      verbose      — 是否列印工具呼叫細節

    回傳：
      最終的助理回應文字
    """
    messages = [{"role": "user", "content": user_message}]

    if mock_mode:
        # Mock 模式：模擬一次工具呼叫然後回傳最終答案
        print("  [Mock] 模擬呼叫工具：lookup_material_property(VO2, transition_temperature)")
        result = lookup_material_property("VO2", "transition_temperature")
        print(f"  [Mock] 工具回傳：{json.dumps(result, ensure_ascii=False)}")
        return f"根據資料庫，VO₂ 的相變溫度為 {result['value']} {result['unit']}。{result['note']}"

    for round_num in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=messages,
        )

        if verbose:
            print(f"  [Round {round_num + 1}] stop_reason={response.stop_reason}, "
                  f"blocks={len(response.content)}")

        # 若已取得最終文字回應
        if response.stop_reason == "end_turn":
            text_blocks = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_blocks)

        # 處理 tool_use 事件
        if response.stop_reason == "tool_use":
            # 將 assistant 回應加入歷史（含 tool_use blocks）
            messages.append({"role": "assistant", "content": response.content})

            # 執行所有工具並收集結果
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input
                if verbose:
                    print(f"  [Tool] 呼叫 {tool_name}({json.dumps(tool_input, ensure_ascii=False)})")

                fn = TOOL_DISPATCH.get(tool_name)
                if fn:
                    output = fn(**tool_input)
                else:
                    output = {"error": f"未知工具：{tool_name}"}

                if verbose:
                    print(f"  [Tool] 結果：{json.dumps(output, ensure_ascii=False)}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output, ensure_ascii=False),
                })

            # 將 tool_result 加入歷史
            messages.append({"role": "user", "content": tool_results})

    return "[錯誤] 超過最大工具呼叫輪次限制"


# ── 示範場景 ──────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "VO₂ 薄膜的相變溫度是多少？並說明其在智慧窗應用中的意義。",
    "如果以 2 nm/min 的速率沉積 50 分鐘，薄膜厚度是多少奈米？",
    (
        "請從以下段落中抽取合成條件，並計算薄膜厚度：\n"
        "「The VO₂ films were deposited at 500°C with working pressure 5 mTorr "
        "for 50 min at a rate of 2 nm/min.」"
    ),
]


def run_example() -> None:
    """主入口：依序展示三個工具呼叫場景。"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    mock_mode = not bool(api_key)

    if mock_mode:
        print("[Mock 模式] 使用本地工具執行（無需 API key）\n")
        client = None
    else:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        print(f"[真實模式] 模型：{MODEL}\n")

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n{'=' * 60}")
        print(f"問題 {i}：{question[:80]}{'...' if len(question) > 80 else ''}")
        print("-" * 60)
        answer = run_agent_with_tools(question, client, mock_mode)
        print(f"回答：{answer}")

    # TODO: 新增工具執行錯誤重試機制（exponential backoff）
    # TODO: 支援並行工具呼叫（parallel tool use）以提升效率
    # TODO: 記錄工具呼叫日誌至 JSONL 格式，用於後續效能分析


if __name__ == "__main__":
    run_example()
