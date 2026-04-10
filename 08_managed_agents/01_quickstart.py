"""
Managed Agent 快速入門
======================

本模組示範如何使用 Anthropic SDK 建立最小可執行的 Managed Agent。

核心概念：
  - Managed Agent 本質上是一個持有對話記憶的有狀態迴圈
  - 每輪呼叫 `client.messages.create()` 並將歷史訊息傳入
  - SDK 負責管理 token 計數、角色輪替、工具回呼等底層細節

執行方式：
  python 01_quickstart.py

環境變數：
  ANTHROPIC_API_KEY — 未設定時自動切換至 Mock 模式

參考資料：
  - Anthropic Managed Agents 文件: https://platform.claude.com/docs/en/managed-agents/overview
  - Messages API: https://docs.anthropic.com/en/api/messages
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── 常數設定 ──────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-5"
MAX_TOKENS = 1024
SYSTEM_PROMPT = """你是一位專業的材料科學文獻助理，熟悉薄膜沉積、相變材料與電化學系統。
請用精確的科學語言回答問題，並在適當時引用具體數值（溫度、壓力、效率等）。"""

MOCK_RESPONSES = [
    "您好！我是材料科學文獻助理。請問您想了解哪方面的材料科學知識？",
    "VO₂ 的半導體—金屬相變溫度約為 68°C（341 K），電阻變化幅度可達 4 個數量級（ΔR/R ≈ 10⁴）。",
    "感謝您的提問。如需進一步分析特定論文段落，請提供文本內容。",
]


# ── 資料結構 ──────────────────────────────────────────────────────────────────

@dataclass
class AgentSession:
    """代表一個 Managed Agent 的對話 Session。

    屬性：
      session_id  — 唯一識別碼
      messages    — 完整對話歷史（符合 Anthropic 訊息格式）
      turn_count  — 已完成的輪次數
      token_total — 累計消耗的 token 數（近似值）
    """
    session_id: str
    messages: list = field(default_factory=list)
    turn_count: int = 0
    token_total: int = 0

    def add_user_message(self, content: str) -> None:
        """新增使用者訊息至對話歷史。"""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """新增助理回應至對話歷史，並更新輪次計數。"""
        self.messages.append({"role": "assistant", "content": content})
        self.turn_count += 1

    def to_summary(self) -> dict:
        """回傳 Session 的摘要資訊。"""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "message_count": len(self.messages),
            "token_total": self.token_total,
        }


# ── 核心 Agent 函式 ───────────────────────────────────────────────────────────

def chat_turn(
    session: AgentSession,
    user_input: str,
    client,
    mock_mode: bool = False,
) -> str:
    """執行單一對話輪次，回傳助理回應文字。

    參數：
      session    — 當前 AgentSession 物件
      user_input — 使用者輸入文字
      client     — Anthropic client 實例（mock 模式下可為 None）
      mock_mode  — 是否使用預設回應（無需 API key）

    回傳：
      assistant_text — 助理回應的純文字內容
    """
    session.add_user_message(user_input)

    if mock_mode:
        # Mock 模式：輪流回傳預設回應
        idx = session.turn_count % len(MOCK_RESPONSES)
        assistant_text = MOCK_RESPONSES[idx]
        session.add_assistant_message(assistant_text)
        return assistant_text

    # 真實 API 呼叫
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=session.messages,
    )

    assistant_text = response.content[0].text
    session.token_total += response.usage.input_tokens + response.usage.output_tokens
    session.add_assistant_message(assistant_text)
    return assistant_text


def run_multi_turn_demo(client, mock_mode: bool) -> None:
    """示範三輪對話，展示 Session 狀態持久化。

    此函式建立一個 AgentSession 並連續發送三個問題，
    驗證每輪回應都能參考先前對話上下文。
    """
    print("\n" + "=" * 60)
    print("Managed Agent 快速入門 Demo")
    print("=" * 60)

    session = AgentSession(session_id="quickstart-001")

    questions = [
        "請簡單介紹自己，並說明你擅長處理哪類材料科學問題。",
        "VO₂ 薄膜的相變溫度是多少？電阻變化幅度有多大？",
        "根據你剛才的說明，這個特性對智慧窗（smart window）有什麼應用潛力？",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n[Turn {i}] 使用者：{question}")
        response = chat_turn(session, question, client, mock_mode)
        print(f"[Turn {i}] 助理：{response[:200]}{'...' if len(response) > 200 else ''}")

    print("\n" + "-" * 60)
    print("Session 摘要：", json.dumps(session.to_summary(), ensure_ascii=False, indent=2))

    # TODO: 將 session 序列化至磁碟，實現跨程式執行的記憶持久化
    # TODO: 新增 token budget 警示，當 token_total 超過閾值時自動壓縮歷史


def run_example() -> None:
    """主入口：自動偵測 API key，切換真實或 Mock 模式。"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    mock_mode = not bool(api_key)

    if mock_mode:
        print("[Mock 模式] 未偵測到 ANTHROPIC_API_KEY，使用預設回應。")
        print("           設定 .env 中的 ANTHROPIC_API_KEY 可切換至真實 API。\n")
        client = None
    else:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        print(f"[真實模式] 使用模型：{MODEL}\n")

    run_multi_turn_demo(client, mock_mode)

    # TODO: 新增互動式 REPL，讓使用者手動輸入問題


if __name__ == "__main__":
    run_example()
