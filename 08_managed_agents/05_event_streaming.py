"""
05_event_streaming.py — SSE 事件流處理

學習目標：
    1. 理解 Managed Agents 的 Server-Sent Events (SSE) 格式
    2. 處理各類事件：text、tool_use、memory 操作、error
    3. Interrupt（中斷）和 Steer（引導）session
    4. 實作 event handler 架構

SSE 事件類型（Managed Agents）：
    message_start         → session 開始處理
    content_block_start   → 新 content block 開始
    content_block_delta   → 增量文字（streaming）
    content_block_stop    → block 結束
    tool_use              → agent 呼叫工具
    tool_result           → 工具執行結果
    memory_list           → agent 列出 memory store entries
    memory_search         → agent 搜尋 memory store
    memory_read           → agent 讀取 entry
    memory_write          → agent 寫入 entry
    message_stop          → 本輪對話結束
    error                 → 錯誤事件

官方文件：https://platform.claude.com/docs/en/managed-agents/events

使用方式：
    python 05_event_streaming.py
"""

import json
import time
from typing import Generator, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


# ─────────────────────────────────────────────────────
#  Event 型別定義
# ─────────────────────────────────────────────────────

class EventType(str, Enum):
    MESSAGE_START       = "message_start"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP  = "content_block_stop"
    TOOL_USE            = "tool_use"
    TOOL_RESULT         = "tool_result"
    MEMORY_LIST         = "memory_list"
    MEMORY_SEARCH       = "memory_search"
    MEMORY_READ         = "memory_read"
    MEMORY_WRITE        = "memory_write"
    MESSAGE_STOP        = "message_stop"
    ERROR               = "error"


@dataclass
class StreamEvent:
    """SSE 事件的通用結構。"""
    type: str
    data: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        return self.data.get(key, default)


# ─────────────────────────────────────────────────────
#  Mock Event Stream（模擬材料科學提取場景）
# ─────────────────────────────────────────────────────

def mock_extraction_event_stream(paper_abstract: str) -> Generator[StreamEvent, None, None]:
    """
    模擬材料科學論文提取任務的完整 event stream。
    包含：文字回應、tool_use、memory 操作。
    """
    events = [
        # ── 開始 ──
        StreamEvent("message_start", {"session_id": "sess_0001", "role": "assistant"}),

        # ── 文字分析 ──
        StreamEvent("content_block_start", {"index": 0, "content_block": {"type": "text"}}),
        StreamEvent("content_block_delta", {"delta": {"type": "text_delta",
            "text": "收到論文摘要，開始分析材料資訊...\n"}}),
        StreamEvent("content_block_delta", {"delta": {"type": "text_delta",
            "text": "識別到鈣鈦礦材料，檢查 playbook 規則。\n"}}),
        StreamEvent("content_block_stop", {}),

        # ── Memory: 讀取 playbook ──
        StreamEvent("memory_read", {
            "store_alias": "playbook",
            "key": "rules/step1_locate",
            "value": "先找摘要和結論中的主要材料描述",
            "version": 1,
        }),
        StreamEvent("memory_read", {
            "store_alias": "playbook",
            "key": "extraction/composition_rules",
            "value": "鈣鈦礦格式：ABX3（A=有機陽離子, B=金屬, X=鹵素）",
            "version": 2,
        }),

        # ── Tool Use: 查詢材料性質 ──
        StreamEvent("tool_use", {
            "id": "toolu_001",
            "name": "lookup_material_property",
            "input": {"material": "MAPbI3", "property": "band_gap"},
        }),
        StreamEvent("tool_result", {
            "tool_use_id": "toolu_001",
            "output": json.dumps({
                "material": "MAPbI3", "property": "band_gap",
                "value": 1.57, "unit": "eV", "confidence": "high"
            }),
        }),

        # ── Tool Use: 單位換算 ──
        StreamEvent("tool_use", {
            "id": "toolu_002",
            "name": "convert_units",
            "input": {"value": 1.57, "from_unit": "eV", "to_unit": "kJ/mol"},
        }),
        StreamEvent("tool_result", {
            "tool_use_id": "toolu_002",
            "output": json.dumps({"result": 151.4, "to_unit": "kJ/mol"}),
        }),

        # ── Memory: 搜尋現有 cases ──
        StreamEvent("memory_search", {
            "store_alias": "cases",
            "query": "MAPbI3 extraction",
            "results": [],
            "note": "無相關 case，將建立新記錄",
        }),

        # ── 文字：結果摘要 ──
        StreamEvent("content_block_start", {"index": 1, "content_block": {"type": "text"}}),
        StreamEvent("content_block_delta", {"delta": {"type": "text_delta",
            "text": "\n提取完成。主要發現：\n"}}),
        StreamEvent("content_block_delta", {"delta": {"type": "text_delta",
            "text": "- 材料：MAPbI3（甲銨鉛碘鈣鈦礦）\n"}}),
        StreamEvent("content_block_delta", {"delta": {"type": "text_delta",
            "text": "- Band gap：1.57 eV（151.4 kJ/mol）\n"}}),
        StreamEvent("content_block_delta", {"delta": {"type": "text_delta",
            "text": "- 信心度：高（文獻直接引用）\n"}}),
        StreamEvent("content_block_stop", {}),

        # ── Memory: 寫入 case ──
        StreamEvent("memory_write", {
            "store_alias": "cases",
            "key": "paper/test001/extraction_v1",
            "value": json.dumps({
                "material": "MAPbI3",
                "band_gap_eV": 1.57,
                "confidence": "high",
            }),
            "new_version": 1,
        }),

        # ── 結束 ──
        StreamEvent("message_stop", {"stop_reason": "end_turn"}),
    ]

    for event in events:
        time.sleep(0.05)  # 模擬網路延遲
        yield event


def mock_error_event_stream() -> Generator[StreamEvent, None, None]:
    """模擬發生錯誤的 event stream。"""
    yield StreamEvent("message_start", {"session_id": "sess_error"})
    yield StreamEvent("content_block_start", {"index": 0})
    yield StreamEvent("content_block_delta", {"delta": {"type": "text_delta", "text": "處理中..."}})
    yield StreamEvent("error", {
        "type": "rate_limit_error",
        "message": "Too many requests, please retry after 10 seconds",
        "retry_after": 10,
    })


# ─────────────────────────────────────────────────────
#  Event Handler
# ─────────────────────────────────────────────────────

class StreamEventHandler:
    """
    SSE 事件流處理器。
    可以繼承並覆寫 on_* 方法來自定義行為。
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.collected_text: list[str] = []
        self.tool_calls: list[dict] = []
        self.memory_ops: list[dict] = []
        self.errors: list[dict] = []
        self._interrupted = False

    def process_stream(self, stream: Generator[StreamEvent, None, None]) -> dict:
        """
        處理完整 event stream，回傳摘要。
        可在任意點呼叫 self.interrupt() 中斷。
        """
        for event in stream:
            if self._interrupted:
                print("  [stream] 已中斷，停止處理")
                break

            handler = getattr(self, f"on_{event.type}", self.on_unknown)
            handler(event)

        return {
            "text": "".join(self.collected_text),
            "tool_calls": self.tool_calls,
            "memory_ops": self.memory_ops,
            "errors": self.errors,
        }

    def interrupt(self):
        """
        中斷 stream 處理。
        真實 API 中，interrupt 會發送控制訊號給 session：
        client.sessions.interrupt(session_id)
        """
        self._interrupted = True
        print("  [interrupt] 中斷訊號已發送")

    # ── Event 處理方法 ──────────────────────────────

    def on_message_start(self, event: StreamEvent):
        if self.verbose:
            print(f"\n  [message_start] session={event.get('session_id')}")

    def on_content_block_start(self, event: StreamEvent):
        pass  # 靜默處理

    def on_content_block_delta(self, event: StreamEvent):
        text = event.get("delta", {}).get("text", "")
        self.collected_text.append(text)
        if self.verbose:
            print(text, end="", flush=True)

    def on_content_block_stop(self, event: StreamEvent):
        if self.verbose and self.collected_text:
            print()  # 換行

    def on_tool_use(self, event: StreamEvent):
        call = {
            "id": event.get("id"),
            "name": event.get("name"),
            "input": event.get("input", {}),
        }
        self.tool_calls.append(call)
        if self.verbose:
            print(f"  [tool_use] {call['name']}({json.dumps(call['input'], ensure_ascii=False)})")

    def on_tool_result(self, event: StreamEvent):
        if self.verbose:
            output = event.get("output", "")
            print(f"  [tool_result] id={event.get('tool_use_id')} → {str(output)[:80]}")

    def on_memory_read(self, event: StreamEvent):
        op = {"op": "read", "alias": event.get("store_alias"), "key": event.get("key")}
        self.memory_ops.append(op)
        if self.verbose:
            print(f"  [memory_read] {op['alias']}/{op['key']}")

    def on_memory_write(self, event: StreamEvent):
        op = {"op": "write", "alias": event.get("store_alias"),
              "key": event.get("key"), "new_version": event.get("new_version")}
        self.memory_ops.append(op)
        if self.verbose:
            print(f"  [memory_write] {op['alias']}/{op['key']} → v{op['new_version']}")

    def on_memory_search(self, event: StreamEvent):
        op = {"op": "search", "alias": event.get("store_alias"), "query": event.get("query")}
        self.memory_ops.append(op)
        if self.verbose:
            n = len(event.get("results", []))
            print(f"  [memory_search] {op['alias']} query='{op['query']}' → {n} results")

    def on_memory_list(self, event: StreamEvent):
        op = {"op": "list", "alias": event.get("store_alias")}
        self.memory_ops.append(op)
        if self.verbose:
            print(f"  [memory_list] {op['alias']}")

    def on_message_stop(self, event: StreamEvent):
        if self.verbose:
            print(f"  [message_stop] reason={event.get('stop_reason')}")

    def on_error(self, event: StreamEvent):
        error = {
            "type": event.get("type"),
            "message": event.get("message"),
        }
        self.errors.append(error)
        print(f"\n  [ERROR] {error['type']}: {error['message']}")
        if "retry_after" in event.data:
            print(f"  [ERROR] 請等待 {event.get('retry_after')}s 後重試")

    def on_unknown(self, event: StreamEvent):
        if self.verbose:
            print(f"  [unknown event] type={event.type}")


# ─────────────────────────────────────────────────────
#  Steer（引導）示範
# ─────────────────────────────────────────────────────

def demo_steer():
    """
    Steer：在 session 執行中途插入新指令。
    真實 API：client.sessions.steer(session_id, content="新指令")
    用途：修正 agent 方向、提供額外資訊、要求特定格式。
    """
    print("\n[Steer 概念說明]")
    steer_example = {
        "session_id": "sess_0001",
        "steer_message": (
            "請將結果改為繁體中文，"
            "並在 JSON 輸出中加入 synthesis_method 欄位。"
        ),
    }
    print(f"  Steer 訊息：{steer_example['steer_message']}")
    print("  效果：agent 會在當前 reasoning 中途改變方向")
    print("  注意：Steer 不是新的 user turn，不會重置 context")


# ─────────────────────────────────────────────────────
#  Demo 主流程
# ─────────────────────────────────────────────────────

def demo_event_streaming():
    print("=" * 60)
    print("  SSE Event Streaming Demo")
    print("=" * 60)

    paper_abstract = """
    We report a high-efficiency methylammonium lead iodide (MAPbI3)
    perovskite solar cell with a power conversion efficiency of 22.7%.
    The band gap of 1.57 eV was confirmed by UV-Vis spectroscopy.
    """

    # ── 1. 正常流程 ────────────────────────────────────
    print("\n[1] 正常 Event Stream 處理")
    handler = StreamEventHandler(verbose=True)
    stream = mock_extraction_event_stream(paper_abstract)
    result = handler.process_stream(stream)

    print("\n=== 處理摘要 ===")
    print(f"  Tool calls: {len(result['tool_calls'])}")
    for tc in result['tool_calls']:
        print(f"    - {tc['name']}")
    print(f"  Memory ops: {len(result['memory_ops'])}")
    for op in result['memory_ops']:
        print(f"    - {op['op']}: {op.get('alias')}/{op.get('key', op.get('query', ''))}")
    print(f"  Errors: {len(result['errors'])}")

    # ── 2. Error 處理 ──────────────────────────────────
    print("\n[2] Error Event 處理")
    error_handler = StreamEventHandler(verbose=True)
    error_result = error_handler.process_stream(mock_error_event_stream())
    print(f"  捕獲到 {len(error_result['errors'])} 個錯誤")

    # ── 3. Steer 說明 ──────────────────────────────────
    demo_steer()

    # ── 4. Interrupt 示範 ─────────────────────────────
    print("\n[4] Interrupt 示範（處理到一半中斷）")

    class InterruptAfterFirstTool(StreamEventHandler):
        """在第一個 tool_use 後中斷。"""
        def on_tool_use(self, event):
            super().on_tool_use(event)
            print("  [demo] 偵測到 tool_use，觸發中斷")
            self.interrupt()

    interrupt_handler = InterruptAfterFirstTool(verbose=True)
    interrupt_handler.process_stream(mock_extraction_event_stream(paper_abstract))
    print(f"  中斷前完成的 tool calls: {len(interrupt_handler.tool_calls)}")

    print("\n✅ Event Streaming Demo 完成！")
    print("   下一步：06_material_extraction_demo.py — 完整材料科學 Demo")


if __name__ == "__main__":
    demo_event_streaming()
