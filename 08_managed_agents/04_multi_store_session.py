"""
多 Store 跨工具 Session 管理
=============================

本模組示範如何在 Managed Agent 中同時協調多個工具與多種 Memory Store，
實現跨輪次的狀態持久化與工具狀態隔離。

架構設計：
  SessionManager
    ├── dialogue_store  (FullHistoryStore)   — 對話歷史
    ├── tool_cache      (dict)               — 工具輸出快取（避免重複呼叫）
    ├── scratchpad      (list[str])          — Agent 的思考暫存區
    └── extraction_db   (FileBasedStore)     — 已抽取結果的持久化 log

跨 Session 持久化策略：
  - 對話歷史：每輪結束後序列化至 JSON
  - 工具快取：以（工具名稱 + 輸入 hash）為 key，避免重複 API 呼叫
  - 抽取結果：追加至 JSONL，支援離線分析

實際應用場景：
  處理材料科學論文時，同一批次可能需要重複查詢相同材料的性質，
  工具快取可顯著減少 API 呼叫次數。

參考資料：
  - ACE Playbook 多 Session 架構: https://arxiv.org/abs/2510.04618
  - Anthropic Tool Use 文件: https://docs.anthropic.com/en/docs/tool-use
"""

import os
import json
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from dotenv import load_dotenv

load_dotenv()

# ── 常數設定 ──────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-5"
MAX_TOKENS = 2048
EXTRACTION_LOG = Path("/tmp/extraction_results.jsonl")
MAX_TOOL_ROUNDS = 6


# ── 工具定義（簡化版，重用 02 的工具集） ──────────────────────────────────────

TOOLS = [
    {
        "name": "lookup_material_property",
        "description": "從資料庫查詢材料性質（相變溫度、能隙、電阻率等）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "material": {"type": "string", "description": "材料化學式或名稱"},
                "property": {"type": "string", "description": "欲查詢的性質名稱"},
            },
            "required": ["material", "property"],
        },
    },
    {
        "name": "save_extraction_result",
        "description": "將從論文中抽取的結構化數據儲存至結果資料庫，供後續分析使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "論文識別碼（如 arXiv ID）"},
                "material": {"type": "string", "description": "主要材料"},
                "properties": {
                    "type": "object",
                    "description": "抽取的性質字典，key 為欄位名，value 含數值與單位",
                },
            },
            "required": ["paper_id", "material", "properties"],
        },
    },
]


# ── Session Manager ────────────────────────────────────────────────────────────

@dataclass
class SessionManager:
    """管理多個 Store 的 Session 協調器。

    屬性：
      session_id      — 唯一識別碼
      dialogue        — 對話歷史（list[dict]）
      tool_cache      — 工具呼叫快取 {cache_key: result}
      scratchpad      — Agent 思考暫存（純字串列表）
      extraction_log  — 抽取結果持久化路徑
      _stats          — 統計資訊（快取命中次數、呼叫次數等）
    """
    session_id: str
    dialogue: list = field(default_factory=list)
    tool_cache: dict = field(default_factory=dict)
    scratchpad: list = field(default_factory=list)
    extraction_log: Path = field(default_factory=lambda: EXTRACTION_LOG)
    _stats: dict = field(default_factory=lambda: {
        "tool_calls": 0,
        "cache_hits": 0,
        "extractions_saved": 0,
        "api_calls": 0,
    })

    def _cache_key(self, tool_name: str, tool_input: dict) -> str:
        """生成工具呼叫的唯一快取鍵（以 MD5 hash 為基礎）。"""
        payload = json.dumps({"tool": tool_name, "input": tool_input}, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:12]

    def dispatch_tool(self, tool_name: str, tool_input: dict) -> dict:
        """執行工具呼叫，優先從快取取值。

        若快取命中，直接回傳快取結果；否則執行工具並更新快取。
        """
        cache_key = self._cache_key(tool_name, tool_input)
        self._stats["tool_calls"] += 1

        if cache_key in self.tool_cache:
            self._stats["cache_hits"] += 1
            result = self.tool_cache[cache_key]
            result["_cached"] = True
            return result

        # 執行工具
        result = _execute_tool(tool_name, tool_input)
        result["_cached"] = False
        self.tool_cache[cache_key] = {k: v for k, v in result.items() if k != "_cached"}

        return result

    def save_extraction(self, data: dict) -> None:
        """將抽取結果持久化至 JSONL 日誌檔。"""
        record = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            **data,
        }
        with open(self.extraction_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stats["extractions_saved"] += 1

    def add_to_scratchpad(self, note: str) -> None:
        """記錄 Agent 的中間思考步驟。"""
        self.scratchpad.append(note)

    def get_stats(self) -> dict:
        """回傳 Session 統計摘要。"""
        return {
            "session_id": self.session_id,
            "dialogue_turns": len([m for m in self.dialogue if m["role"] == "user"]),
            "cache_hit_rate": (
                f"{self._stats['cache_hits'] / max(1, self._stats['tool_calls']):.1%}"
            ),
            **self._stats,
        }


# ── 工具實作 ──────────────────────────────────────────────────────────────────

MATERIAL_DB = {
    ("VO2", "transition_temperature"): {"value": 68, "unit": "°C"},
    ("TiO2", "bandgap"): {"value": 3.2, "unit": "eV"},
}

_saved_extractions = []  # 記憶體內的抽取記錄（Mock 模式使用）


def _execute_tool(tool_name: str, tool_input: dict) -> dict:
    """工具執行分派器。"""
    if tool_name == "lookup_material_property":
        key = (tool_input["material"].replace("₂", "2"), tool_input["property"])
        result = MATERIAL_DB.get(key, {"error": f"找不到 {tool_input['material']} 的 {tool_input['property']}"})
        return dict(result)

    if tool_name == "save_extraction_result":
        _saved_extractions.append(tool_input)
        return {"status": "saved", "record_count": len(_saved_extractions)}

    return {"error": f"未知工具：{tool_name}"}


# ── 多 Store Agent 迴圈 ────────────────────────────────────────────────────────

def run_multi_store_agent(
    user_message: str,
    session: SessionManager,
    client,
    mock_mode: bool = False,
) -> str:
    """執行含多 Store 管理的 Agent 迴圈。

    相比 02 的基本版本，本函式額外維護：
    - 工具呼叫快取（跨輪次共享）
    - Scratchpad 思考記錄
    - 抽取結果自動持久化
    """
    session.dialogue.append({"role": "user", "content": user_message})

    if mock_mode:
        # Mock：模擬查詢兩次相同工具（展示快取效果）
        r1 = session.dispatch_tool("lookup_material_property", {"material": "VO2", "property": "transition_temperature"})
        r2 = session.dispatch_tool("lookup_material_property", {"material": "VO2", "property": "transition_temperature"})
        session.add_to_scratchpad(f"[Mock] 第一次查詢結果：{r1}")
        session.add_to_scratchpad(f"[Mock] 第二次查詢（快取）：{r2}, cached={r2.get('_cached')}")

        answer = f"VO₂ 相變溫度為 {r1['value']} {r1['unit']}（第二次查詢命中快取：{r2.get('_cached')}）"
        session.dialogue.append({"role": "assistant", "content": answer})
        return answer

    session._stats["api_calls"] += 1
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOLS,
            messages=session.dialogue,
        )

        if response.stop_reason == "end_turn":
            text = "\n".join(b.text for b in response.content if b.type == "text")
            session.dialogue.append({"role": "assistant", "content": response.content})
            return text

        if response.stop_reason == "tool_use":
            session.dialogue.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = session.dispatch_tool(block.name, block.input)
                session.add_to_scratchpad(
                    f"[Tool] {block.name}({block.input}) → cached={result.pop('_cached', False)}"
                )
                if block.name == "save_extraction_result":
                    session.save_extraction(block.input)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            session.dialogue.append({"role": "user", "content": tool_results})
            session._stats["api_calls"] += 1

    return "[錯誤] 超過最大工具呼叫輪次"


# ── 多輪 Session 示範 ─────────────────────────────────────────────────────────

def run_example() -> None:
    """主入口：執行三輪對話，展示快取效果與多 Store 協同。"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    mock_mode = not bool(api_key)

    if mock_mode:
        print("[Mock 模式] 展示工具快取與 scratchpad 記錄\n")
        client = None
    else:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

    session = SessionManager(session_id="multi-store-001")
    EXTRACTION_LOG.touch(exist_ok=True)

    questions = [
        "查詢 VO₂ 的相變溫度，並說明是否適合用於節能窗。",
        "再次確認 VO₂ 的相變溫度（測試快取命中）。",
        (
            "請將以下抽取結果儲存至資料庫："
            "paper_id=arXiv:2401.00001, material=VO2, "
            "properties={transition_temperature_C: 68, resistance_change_orders: 4}"
        ),
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"[輪次 {i}] {q[:70]}{'...' if len(q) > 70 else ''}")
        answer = run_multi_store_agent(q, session, client, mock_mode)
        print(f"回答：{answer}")

    print("\n" + "=" * 60)
    print("Session 統計：")
    print(json.dumps(session.get_stats(), ensure_ascii=False, indent=2))

    print(f"\nScratchpad（{len(session.scratchpad)} 條記錄）：")
    for note in session.scratchpad:
        print(f"  {note}")

    # TODO: 實作 Session 快照序列化（pickle 或 JSON），支援斷點續跑
    # TODO: 加入 tool_cache 的 TTL 機制，定期清除過期快取
    # TODO: 多 Session 共享 extraction_db，實現跨論文的知識聚合


if __name__ == "__main__":
    run_example()
