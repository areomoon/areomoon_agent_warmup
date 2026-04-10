"""
01_quickstart.py — Managed Agents API 快速入門

學習目標：
    1. 建立 Agent Definition（模型、系統提示、工具）
    2. 建立 Environment（Python runtime）
    3. 建立 Session 並 attach memory store
    4. 發送 user message 並接收 event stream
    5. 理解 session lifecycle

官方文件：https://platform.claude.com/docs/en/managed-agents/overview

⚠️  需要 Managed Agents beta access。
    無 access 時：MOCK_MODE = True 跑完整個學習流程。

使用方式：
    python 01_quickstart.py
    python 01_quickstart.py --live   # 有 beta access 時使用
"""

import os
import sys
import json
import argparse

# ── Beta SDK import（可能尚未 release，使用 try/except）──
try:
    import anthropic
    from anthropic import Anthropic
    # TODO: 確認 managed_agents 子模組的正確 import 路徑
    # from anthropic.managed_agents import ManagedAgentsClient
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("[WARN] anthropic SDK 未安裝，使用 mock 模式")
    print("       pip install anthropic")

# ── 設定 ──────────────────────────────────────────────
MOCK_MODE = True  # 改成 False 需要有 beta access + 正確 SDK

AGENT_CONFIG = {
    "name": "materials-extraction-agent",
    "model": "claude-sonnet-4-6",
    "system_prompt": (
        "你是一位材料科學文獻分析專家。"
        "你能從論文中精確提取材料組成、合成方法、性質數據。"
        "使用 structured JSON 格式回報結果。"
    ),
    # 內建工具：bash（執行程式碼）、web_search、files
    "tools": ["bash", "web_search", "files"],
}

ENVIRONMENT_CONFIG = {
    "name": "materials-env",
    "runtime": "python3.11",
    "packages": ["numpy", "pandas", "pydantic"],
}


# ─────────────────────────────────────────────────────
#  Mock 實作（無 beta access 時使用）
# ─────────────────────────────────────────────────────

class MockManagedAgentsClient:
    """模擬 Managed Agents Client，用於學習 API 結構。"""

    def __init__(self):
        self.agents = {}
        self.environments = {}
        self.sessions = {}
        print("[MOCK] ManagedAgentsClient 初始化")

    def create_agent(self, **kwargs) -> dict:
        """建立 agent definition。"""
        agent_id = f"agent_{len(self.agents) + 1:04d}"
        agent = {"id": agent_id, "status": "active", **kwargs}
        self.agents[agent_id] = agent
        return agent

    def create_environment(self, **kwargs) -> dict:
        """建立執行環境（Python runtime + 套件）。"""
        env_id = f"env_{len(self.environments) + 1:04d}"
        env = {"id": env_id, "status": "ready", **kwargs}
        self.environments[env_id] = env
        return env

    def create_session(self, agent_id: str, environment_id: str,
                       memory_stores: list = None) -> dict:
        """
        建立 session，連接 agent + environment + memory stores。
        memory_stores 格式：[{"store_id": "...", "access": "read" | "read-write"}]
        """
        session_id = f"sess_{len(self.sessions) + 1:04d}"
        session = {
            "id": session_id,
            "agent_id": agent_id,
            "environment_id": environment_id,
            "memory_stores": memory_stores or [],
            "status": "active",
        }
        self.sessions[session_id] = session
        return session

    def send_message(self, session_id: str, content: str):
        """
        發送 user message，回傳 event stream（generator）。
        真實 API 使用 Server-Sent Events (SSE)。
        """
        # 模擬事件流
        mock_events = [
            {"type": "message_start", "session_id": session_id},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta",
                "text": "收到您的請求，開始分析材料科學文獻..."}},
            {"type": "content_block_delta", "delta": {"type": "text_delta",
                "text": "\n\n我將提取以下資訊：\n- 材料組成（化學式）\n- 合成方法\n- 關鍵性質數據"}},
            {"type": "content_block_stop"},
            {"type": "tool_use", "tool": "bash",
             "input": {"command": "python -c \"print('提取完成')\""}},
            {"type": "tool_result", "output": "提取完成"},
            {"type": "message_stop", "stop_reason": "end_turn"},
        ]
        for event in mock_events:
            yield event


# ─────────────────────────────────────────────────────
#  Live 實作（有 beta access 時使用）
# ─────────────────────────────────────────────────────

class LiveManagedAgentsClient:
    """
    真實 Managed Agents Client。
    TODO: 等 SDK release 後補全實作。
    """

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("請設定 ANTHROPIC_API_KEY 環境變數")
        # TODO: 確認正確的 client 初始化方式
        # self.client = anthropic.ManagedAgents(api_key=api_key)
        self.client = Anthropic(api_key=api_key)
        print(f"[LIVE] 連接 Anthropic API（key: ...{api_key[-4:]}）")

    def create_agent(self, **kwargs) -> dict:
        # TODO: self.client.agents.create(**kwargs)
        raise NotImplementedError("等待 beta SDK release")

    def create_environment(self, **kwargs) -> dict:
        # TODO: self.client.environments.create(**kwargs)
        raise NotImplementedError("等待 beta SDK release")

    def create_session(self, agent_id: str, environment_id: str,
                       memory_stores: list = None) -> dict:
        # TODO: self.client.sessions.create(...)
        raise NotImplementedError("等待 beta SDK release")

    def send_message(self, session_id: str, content: str):
        # TODO: self.client.sessions.send_message(session_id, content=content)
        raise NotImplementedError("等待 beta SDK release")


# ─────────────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────────────

def run_quickstart(mock: bool = True):
    """
    Managed Agents 完整 quickstart 流程。

    步驟：
        1. 初始化 client
        2. 建立 agent definition
        3. 建立 environment
        4. 建立 session
        5. 發送 message 並處理 event stream
        6. 印出結果摘要
    """
    print("=" * 60)
    print("  Managed Agents Quickstart")
    print(f"  模式: {'MOCK' if mock else 'LIVE'}")
    print("=" * 60)

    # ── Step 1: 初始化 client ──────────────────────────
    print("\n[Step 1] 初始化 Managed Agents Client")
    client = MockManagedAgentsClient() if mock else LiveManagedAgentsClient()

    # ── Step 2: 建立 agent definition ─────────────────
    print("\n[Step 2] 建立 Agent Definition")
    print(f"  模型: {AGENT_CONFIG['model']}")
    print(f"  工具: {AGENT_CONFIG['tools']}")

    agent = client.create_agent(
        name=AGENT_CONFIG["name"],
        model=AGENT_CONFIG["model"],
        system_prompt=AGENT_CONFIG["system_prompt"],
        tools=AGENT_CONFIG["tools"],
    )
    print(f"  ✓ Agent 建立完成: {agent['id']}")

    # ── Step 3: 建立 environment ───────────────────────
    print("\n[Step 3] 建立 Environment（Python runtime）")
    print(f"  Runtime: {ENVIRONMENT_CONFIG['runtime']}")
    print(f"  套件: {ENVIRONMENT_CONFIG['packages']}")

    env = client.create_environment(
        name=ENVIRONMENT_CONFIG["name"],
        runtime=ENVIRONMENT_CONFIG["runtime"],
        packages=ENVIRONMENT_CONFIG["packages"],
    )
    print(f"  ✓ Environment 建立完成: {env['id']}")

    # ── Step 4: 建立 session ───────────────────────────
    print("\n[Step 4] 建立 Session")
    print(f"  Agent: {agent['id']}")
    print(f"  Environment: {env['id']}")

    session = client.create_session(
        agent_id=agent["id"],
        environment_id=env["id"],
        memory_stores=[],  # 之後在 03_memory_stores.py 會詳細說明
    )
    print(f"  ✓ Session 建立完成: {session['id']}")

    # ── Step 5: 發送 message，處理 event stream ────────
    print("\n[Step 5] 發送 User Message")
    user_message = "請分析這篇論文中的鈣鈦礦太陽能電池材料，提取關鍵的效率數據和製備條件。"
    print(f"  User: {user_message[:60]}...")

    print("\n[Step 5] 接收 Event Stream:")
    collected_text = []

    for event in client.send_message(session["id"], user_message):
        event_type = event.get("type", "unknown")

        if event_type == "content_block_delta":
            text = event.get("delta", {}).get("text", "")
            collected_text.append(text)
            print(f"  [text] {text}", end="", flush=True)

        elif event_type == "tool_use":
            tool = event.get("tool")
            inp = json.dumps(event.get("input", {}), ensure_ascii=False)
            print(f"\n  [tool_use] {tool}: {inp}")

        elif event_type == "tool_result":
            print(f"  [tool_result] {event.get('output', '')}")

        elif event_type == "message_stop":
            reason = event.get("stop_reason", "")
            print(f"\n  [stop] reason={reason}")

    # ── Step 6: 結果摘要 ───────────────────────────────
    print("\n" + "=" * 60)
    print("  完整回應：")
    print("  " + "".join(collected_text))
    print("=" * 60)

    print("\n✅ Quickstart 完成！")
    print("   下一步：02_custom_tools.py — 學習定義 custom tools")

    return {"agent": agent, "environment": env, "session": session}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Managed Agents Quickstart")
    parser.add_argument("--live", action="store_true",
                        help="使用 Live API（需要 beta access）")
    args = parser.parse_args()

    use_mock = not args.live
    if not use_mock and not SDK_AVAILABLE:
        print("[ERROR] Live 模式需要安裝 anthropic SDK")
        sys.exit(1)

    run_quickstart(mock=use_mock)
