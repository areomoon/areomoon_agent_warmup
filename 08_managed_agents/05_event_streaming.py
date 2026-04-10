"""
Server-Sent Events (SSE) 串流處理
===================================

本模組示範如何使用 Anthropic SDK 的串流 API 實現逐 token 輸出，
並正確處理各種 SSE 事件類型（text delta、tool_use、message_stop 等）。

核心概念：
  - 使用 `client.messages.stream()` context manager 取得 MessageStream
  - 監聽三種主要事件：text、tool_use、message_stop
  - 串流模式下 tool_use 的 input JSON 是分批傳遞的（需要累積後解析）
  - 結合 `with` 語句確保串流連線正確關閉

事件類型說明：
  message_start      — 串流開始，含模型資訊與初始 usage
  content_block_start — 新的 content block 開始（text 或 tool_use）
  content_block_delta — 增量更新（text_delta 或 input_json_delta）
  content_block_stop  — 一個 block 完成
  message_delta      — 整體訊息的 delta（stop_reason、usage 更新）
  message_stop       — 串流結束

串流 vs 非串流的使用時機：
  串流 → 即時 UI 顯示、互動式 CLI、長輸出監控
  非串流 → 批次處理、日誌分析、工具整合

參考資料：
  - Anthropic Streaming 指南: https://docs.anthropic.com/en/api/messages-streaming
  - SSE 規範: https://html.spec.whatwg.org/multipage/server-sent-events.html
"""

import os
import json
import sys
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── 常數設定 ──────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-5"
MAX_TOKENS = 1024

MOCK_TOKENS = [
    "VO", "₂", " 是", "一種", "相", "變", "材料", "，",
    "其", "半", "導", "體", "—", "金", "屬", "相", "變",
    "溫", "度", "約", "為", " **", "68", "°C", "**",
    "（", "341", " K", "）。",
]

TOOLS_FOR_STREAMING = [
    {
        "name": "get_material_data",
        "description": "查詢材料的基本物理性質。",
        "input_schema": {
            "type": "object",
            "properties": {
                "material": {"type": "string", "description": "材料化學式"},
            },
            "required": ["material"],
        },
    },
]


# ── 事件處理器 ────────────────────────────────────────────────────────────────

class StreamEventHandler:
    """收集並處理 Anthropic 串流事件的輔助類別。

    記錄完整的 text 輸出與工具呼叫資訊，
    同時支援即時列印效果（逐字輸出）。
    """

    def __init__(self, print_live: bool = True) -> None:
        """初始化事件處理器。

        參數：
          print_live — 是否即時列印 text delta（模擬打字效果）
        """
        self.print_live = print_live
        self.full_text: str = ""
        self.tool_calls: list = []
        self._current_tool: Optional[dict] = None
        self._current_json_buffer: str = ""
        self.event_counts: dict = {
            "text_delta": 0,
            "input_json_delta": 0,
            "tool_use_blocks": 0,
            "message_stop": 0,
        }

    def on_text_delta(self, text: str) -> None:
        """處理文字增量事件。"""
        self.full_text += text
        self.event_counts["text_delta"] += 1
        if self.print_live:
            print(text, end="", flush=True)

    def on_input_json_delta(self, partial_json: str) -> None:
        """處理工具輸入 JSON 的增量事件（需累積至完整 JSON）。"""
        self._current_json_buffer += partial_json
        self.event_counts["input_json_delta"] += 1

    def on_content_block_start(self, block_type: str, block_id: str, block_name: str = "") -> None:
        """處理新 content block 開始事件。"""
        if block_type == "tool_use":
            self._current_tool = {"id": block_id, "name": block_name, "input": None}
            self._current_json_buffer = ""

    def on_content_block_stop(self, block_type: str) -> None:
        """處理 content block 結束事件。"""
        if block_type == "tool_use" and self._current_tool:
            try:
                self._current_tool["input"] = json.loads(self._current_json_buffer)
            except json.JSONDecodeError:
                self._current_tool["input"] = {"raw": self._current_json_buffer}
            self.tool_calls.append(self._current_tool)
            self.event_counts["tool_use_blocks"] += 1
            self._current_tool = None
            self._current_json_buffer = ""

    def on_message_stop(self, stop_reason: str, usage: Optional[dict] = None) -> None:
        """處理串流結束事件。"""
        self.event_counts["message_stop"] += 1
        if self.print_live and self.full_text:
            print()  # 換行


# ── 串流工具呼叫迴圈 ──────────────────────────────────────────────────────────

def stream_with_tools(
    user_message: str,
    client,
    mock_mode: bool = False,
) -> str:
    """使用串流模式執行含工具呼叫的 Agent 輪次。

    參數：
      user_message — 使用者輸入
      client       — Anthropic client
      mock_mode    — 是否模擬串流輸出

    回傳：
      最終完整文字回應
    """
    messages = [{"role": "user", "content": user_message}]
    handler = StreamEventHandler(print_live=True)

    if mock_mode:
        print("[串流模擬] 逐 token 輸出：", end="")
        for token in MOCK_TOKENS:
            handler.on_text_delta(token)
            time.sleep(0.05)  # 模擬串流延遲
        handler.on_message_stop("end_turn")
        print(f"\n[Mock] 共輸出 {handler.event_counts['text_delta']} 個 delta 事件")
        return handler.full_text

    # 真實串流模式
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=TOOLS_FOR_STREAMING,
        messages=messages,
    ) as stream:
        for event in stream:
            event_type = event.type

            if event_type == "content_block_start":
                handler.on_content_block_start(
                    event.content_block.type,
                    getattr(event.content_block, "id", ""),
                    getattr(event.content_block, "name", ""),
                )

            elif event_type == "content_block_delta":
                delta_type = event.delta.type
                if delta_type == "text_delta":
                    handler.on_text_delta(event.delta.text)
                elif delta_type == "input_json_delta":
                    handler.on_input_json_delta(event.delta.partial_json)

            elif event_type == "content_block_stop":
                # 判斷 block 類型（從 stream 的 current_message 取得）
                pass  # handler 內部狀態已追蹤

            elif event_type == "message_stop":
                handler.on_message_stop(
                    stop_reason=stream.get_final_message().stop_reason,
                )

    # 處理工具呼叫（若有）
    if handler.tool_calls:
        print(f"\n[工具呼叫] 偵測到 {len(handler.tool_calls)} 個工具呼叫：")
        for tc in handler.tool_calls:
            print(f"  - {tc['name']}({json.dumps(tc['input'], ensure_ascii=False)})")

    return handler.full_text


# ── 純文字串流示範 ────────────────────────────────────────────────────────────

def stream_pure_text(user_message: str, client, mock_mode: bool = False) -> str:
    """示範最簡單的純文字串流（不含工具）。

    使用 stream.text_stream 迭代器，是最簡潔的串流方式。
    """
    if mock_mode:
        print("[純文字串流] ", end="")
        for token in MOCK_TOKENS[:15]:
            print(token, end="", flush=True)
            time.sleep(0.04)
        print()
        return "".join(MOCK_TOKENS[:15])

    full_text = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text_chunk in stream.text_stream:
            print(text_chunk, end="", flush=True)
            full_text += text_chunk
    print()
    return full_text


# ── 主示範 ────────────────────────────────────────────────────────────────────

def run_example() -> None:
    """主入口：展示兩種串流模式。"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    mock_mode = not bool(api_key)

    if mock_mode:
        print("[Mock 模式] 模擬 SSE 串流輸出，無需 API key\n")
        client = None
    else:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        print(f"[真實模式] 模型：{MODEL}\n")

    print("=" * 60)
    print("示範一：純文字串流")
    print("=" * 60)
    stream_pure_text("請簡述 VO₂ 的相變特性。", client, mock_mode)

    print("\n" + "=" * 60)
    print("示範二：含工具呼叫的串流")
    print("=" * 60)
    result = stream_with_tools(
        "請查詢 VO₂ 的材料性質並解釋其應用。",
        client,
        mock_mode,
    )
    print(f"\n完整回應（{len(result)} 字）：{result[:100]}...")

    # TODO: 實作串流中斷機制（用戶按 Ctrl+C 時優雅退出）
    # TODO: 將串流輸出即時寫入 WebSocket，實現前端即時顯示
    # TODO: 統計每個 token 的延遲時間（Time-To-First-Token、throughput）


if __name__ == "__main__":
    run_example()
