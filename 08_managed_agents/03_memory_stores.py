"""
Memory Store 三種策略
=====================

本模組比較三種在 Managed Agent 中管理對話記憶的策略，
並分析各自的適用場景、優缺點與 token 消耗。

策略一：全量記憶（Full History）
  - 做法：保留所有 messages，每次呼叫完整傳入
  - 優點：零資訊損失，Agent 可參考全部歷史
  - 缺點：token 線性成長，長對話超出 context window
  - 適用：短期任務、少於 20 輪的精確提取

策略二：檔案持久化（File-based Persistence）
  - 做法：將 messages 序列化至 JSONL，按需載入
  - 優點：跨程式執行持久化，支援審計追蹤
  - 缺點：每次仍需載入全部歷史（需配合壓縮）
  - 適用：多日期跨 Session 的長期任務

策略三：摘要壓縮（Summarisation）
  - 做法：當 token 超過閾值，呼叫 API 生成歷史摘要
  - 優點：token 成本可控，保留關鍵資訊
  - 缺點：摘要可能遺漏細節，需設計摘要提示
  - 適用：長期對話助理、生產環境部署

參考資料：
  - MemGPT (arXiv 2310.08560): https://arxiv.org/abs/2310.08560
  - Anthropic Long Context Best Practices: https://docs.anthropic.com/en/docs/long-context-tips
"""

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── 常數設定 ──────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-5"
MAX_TOKENS = 1024
TOKEN_BUDGET = 4000        # 超過此數觸發摘要壓縮
SESSION_FILE = Path("/tmp/managed_agent_session.jsonl")  # 檔案持久化路徑

MOCK_SUMMARY = (
    "【對話摘要】使用者詢問了 VO₂ 薄膜的相變特性（68°C，4 個數量級電阻變化）"
    "以及磁控濺射的典型沉積條件（工作壓力 5 mTorr，溫度 500°C）。"
)


# ── 策略一：全量記憶 ──────────────────────────────────────────────────────────

@dataclass
class FullHistoryStore:
    """全量保留對話歷史的記憶策略。

    所有訊息皆保留於記憶體中，適合短期精確任務。
    """
    messages: list = field(default_factory=list)
    _estimated_tokens: int = 0

    def append(self, role: str, content: str) -> None:
        """新增訊息並更新 token 估算（以字元數 ÷ 4 近似）。"""
        self.messages.append({"role": role, "content": content})
        self._estimated_tokens += len(content) // 4

    def get_messages(self) -> list:
        """回傳完整訊息列表。"""
        return self.messages.copy()

    def stats(self) -> dict:
        """回傳記憶統計資訊。"""
        return {
            "strategy": "full_history",
            "message_count": len(self.messages),
            "estimated_tokens": self._estimated_tokens,
        }


# ── 策略二：檔案持久化 ────────────────────────────────────────────────────────

class FileBasedStore:
    """將對話歷史持久化至 JSONL 檔案的記憶策略。

    每條訊息為獨立的 JSON 行，支援增量追加與完整讀取。
    """

    def __init__(self, file_path: Path = SESSION_FILE) -> None:
        """初始化，若檔案不存在則建立。"""
        self.file_path = file_path
        self.file_path.touch(exist_ok=True)

    def append(self, role: str, content: str) -> None:
        """將訊息追加至 JSONL 檔案。"""
        record = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_messages(self) -> list:
        """從 JSONL 檔案讀取所有訊息（僅返回 role 與 content）。"""
        messages = []
        if not self.file_path.exists():
            return messages
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                messages.append({"role": record["role"], "content": record["content"]})
        return messages

    def clear(self) -> None:
        """清空持久化檔案（新 Session 開始時呼叫）。"""
        self.file_path.write_text("", encoding="utf-8")

    def stats(self) -> dict:
        """回傳檔案大小與訊息數統計。"""
        messages = self.get_messages()
        size_bytes = self.file_path.stat().st_size if self.file_path.exists() else 0
        return {
            "strategy": "file_based",
            "message_count": len(messages),
            "file_path": str(self.file_path),
            "file_size_bytes": size_bytes,
        }


# ── 策略三：摘要壓縮 ──────────────────────────────────────────────────────────

class SummarisationStore:
    """超過 token 閾值時自動壓縮歷史的記憶策略。

    維護一份「壓縮摘要」加上近期完整訊息的混合記憶。
    """

    def __init__(self, token_budget: int = TOKEN_BUDGET) -> None:
        """初始化，設定壓縮觸發閾值。"""
        self.token_budget = token_budget
        self.summary: str = ""          # 已壓縮的歷史摘要
        self.recent_messages: list = [] # 近期未壓縮的訊息
        self._total_tokens: int = 0
        self._compression_count: int = 0

    def append(self, role: str, content: str) -> None:
        """新增訊息，若超出 token 預算則觸發壓縮。"""
        self.recent_messages.append({"role": role, "content": content})
        self._total_tokens += len(content) // 4

    def maybe_compress(self, client, mock_mode: bool = False) -> bool:
        """若 token 超出預算，呼叫 API 生成摘要並壓縮歷史。

        回傳 True 表示執行了壓縮，False 表示未達閾值。
        """
        if self._total_tokens < self.token_budget:
            return False

        if mock_mode:
            self.summary = MOCK_SUMMARY
        else:
            # 呼叫 API 生成摘要
            history_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in self.recent_messages
            )
            prompt = (
                f"請用 200 字以內的繁體中文摘要以下對話，"
                f"保留所有數值、材料名稱與關鍵結論：\n\n{history_text}"
            )
            if self.summary:
                prompt = f"現有摘要：{self.summary}\n\n請將以下新對話整合進去：\n\n{history_text}"

            response = client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            self.summary = response.content[0].text

        # 保留最近 4 條訊息，清空舊歷史
        self.recent_messages = self.recent_messages[-4:]
        self._total_tokens = sum(len(m["content"]) // 4 for m in self.recent_messages)
        self._compression_count += 1
        return True

    def get_messages(self) -> list:
        """組合摘要系統訊息與近期訊息，供 API 呼叫使用。"""
        messages = []
        if self.summary:
            # 將摘要以 system-like user 訊息注入（Anthropic 無 system in messages）
            messages.append({
                "role": "user",
                "content": f"[對話背景摘要]\n{self.summary}\n\n請繼續對話：",
            })
            messages.append({"role": "assistant", "content": "了解，我已掌握先前的討論內容，請繼續。"})
        messages.extend(self.recent_messages)
        return messages

    def stats(self) -> dict:
        """回傳壓縮策略的統計資訊。"""
        return {
            "strategy": "summarisation",
            "recent_message_count": len(self.recent_messages),
            "has_summary": bool(self.summary),
            "compression_count": self._compression_count,
            "estimated_tokens": self._total_tokens,
            "token_budget": self.token_budget,
        }


# ── 比較示範 ──────────────────────────────────────────────────────────────────

SAMPLE_CONVERSATION = [
    ("user", "VO₂ 的相變溫度是多少？"),
    ("assistant", "VO₂ 的半導體—金屬相變溫度約為 68°C（341 K）。"),
    ("user", "電阻變化幅度呢？"),
    ("assistant", "相變時電阻可降低約 4 個數量級（ΔR/R ≈ 10⁴），在 10⁻³ 到 10⁻² Ω·cm 之間。"),
    ("user", "磁控濺射的典型條件？"),
    ("assistant", "常見條件：工作壓力 5 mTorr、Ar/O₂ 流量比 15:1、基板溫度 500°C、沉積速率 2 nm/min。"),
]


def compare_stores(mock_mode: bool, client) -> None:
    """比較三種記憶策略的 token 消耗與訊息結構。"""
    print("\n" + "=" * 60)
    print("Memory Store 策略比較")
    print("=" * 60)

    # 填入相同的對話歷史
    full = FullHistoryStore()
    file_store = FileBasedStore()
    file_store.clear()
    summ = SummarisationStore(token_budget=50)  # 低閾值以便示範壓縮

    for role, content in SAMPLE_CONVERSATION:
        full.append(role, content)
        file_store.append(role, content)
        summ.append(role, content)

    # 觸發摘要壓縮
    compressed = summ.maybe_compress(client, mock_mode)

    print("\n策略一（全量記憶）統計：")
    print(json.dumps(full.stats(), ensure_ascii=False, indent=2))

    print("\n策略二（檔案持久化）統計：")
    print(json.dumps(file_store.stats(), ensure_ascii=False, indent=2))

    print("\n策略三（摘要壓縮）統計：")
    stats = summ.stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if compressed:
        print(f"\n生成的摘要：{summ.summary}")

    print("\n策略三的 get_messages() 結構（前 2 條）：")
    msgs = summ.get_messages()
    for m in msgs[:2]:
        print(f"  [{m['role']}] {m['content'][:60]}...")

    # TODO: 實作 LRU 快取策略（保留最常被參考的歷史片段）
    # TODO: 加入向量相似度搜尋，從歷史中召回相關段落（RAG 式記憶）


def run_example() -> None:
    """主入口：執行三種記憶策略比較示範。"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    mock_mode = not bool(api_key)

    if mock_mode:
        print("[Mock 模式] 摘要壓縮將使用預設摘要文字。\n")
        client = None
    else:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

    compare_stores(mock_mode, client)


if __name__ == "__main__":
    run_example()
