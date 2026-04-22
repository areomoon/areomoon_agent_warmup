# Self-Consistency 實作筆記

> KB 正本：[self-consistency-implementation](~/PycharmProjects/llm_knowledge_base/wiki/concepts/self-consistency-implementation.md)
> 論文：[Wang et al. 2022](https://arxiv.org/abs/2203.11171)

## 一句話

**Production 價值不是「投票選答案」，是「產生 per-field confidence 讓 pipeline 可以 route」**。

## 核心骨架

```python
# 1. Fan-out N 樣本（temperature > 0）
samples = []
for _ in range(N):
    resp = client.messages.create(
        model="claude-haiku-4-5",   # Opus 4.7 不吃 temperature
        temperature=0.7,
        messages=[...],
    )
    samples.append(parse_json(resp))

# 2. Field-level 聚合（不是整份 JSON 比對）
aggregated = {}
for field in SCHEMA:
    values = [s.get(field) for s in samples if field in s]
    normalized = [normalize(field, v) for v in values]
    counter = Counter(normalized)
    top_value, top_count = counter.most_common(1)[0]
    aggregated[field] = {
        "value": top_value,
        "confidence": top_count / len(samples),
        "all_votes": dict(counter),
    }
```

## 六大實作坑（面試/code review 會問）

1. **必須 field-level 聚合** — whole-JSON diff 任何一欄不同就判「不一致」，長 schema 下永遠零 agreement
2. **Normalize 是真功夫** — `"700°C"` / `"700 C"` / `700` / `"seven hundred celsius"` 要先 canonical form。Pydantic validator / 正則管線
3. **數值要 tolerance-band 投票**（±1% 同一票）—— 不然浮點誤差讓所有樣本都不同
4. **Temperature 0.5-0.9 band**，0.7 是起點；<0.3 樣本太像，>1.0 開始亂編；**每個 task ablation 決定**
5. **N：3 → 5 → 10 遞減**。N=5 是 sweet spot。Batch API 可把 5× 成本壓到 2.5×；asyncio.gather 可把 latency 保持 1×
6. **Anthropic 家限制**：Opus 4.7 無 temperature → 無 diversity → self-consistency 崩潰。用 Haiku 4.5 或 Sonnet 4.6

## 系統設計圖

```
Input
  ↓
Dispatcher ── fan-out N async ──→ LLM
  ↓                                ↓
Sample Store ← N 份 raw JSON ──────┘
  ↓
Normalizer  (unit conversion, canonicalization, numeric binning)
  ↓
Field-level Aggregator  (per-field majority vote + confidence)
  ↓
Router      ├── conf ≥ 0.8  → auto-accept
            ├── 0.5-0.8     → LLM-as-Judge 二次裁決
            └── conf < 0.5  → human queue
  ↓
Output + Telemetry  (per-field confidence timeseries → drift 偵測)
```

## 關鍵設計決定

- **Sample Store 不是 optional** — N 份 raw JSON 是 ablation、audit、fine-tuning 語料來源。省儲存 = 毀掉下季的 initiative
- **三段式 router** 才是實務，不是「≥0.8 ship / else reject」兩段
- **Per-field confidence timeseries 是 drift canary** — 新專利領域或新術語出現時會先在這裡亮燈

## 什麼時候用 / 什麼時候不用

**用**：structured schema + 成本容忍 N× + 想要 per-field confidence + 失敗模式是 ambiguity 不是 format 錯

**不用**：latency < 300ms UI / Opus 4.7 鎖死 / 有標註資料 + 高流量（換 verifier model）/ 失敗模式是 format 錯（換 constrained decoding）

## Patsnap 情境

- **Claim 1 參數**：永遠 N=5，錯誤 propagate 到客戶投資決策
- **Example 段 table**：N=3 + 數值 tolerance-band，table 是此訊號的強項
- **離線 gold set**：N=10 × Sonnet 4.6 × 500-2000 樣本，high-agreement 當 silver label

## 關鍵洞察

- **Agreement ≠ 機率校準**。0.8 agreement 不代表 80% 正確。要客戶端承諾「80% 確信」之前必須先跑 reliability diagram
- **成本 N 線性、但 prompt 長度 super-linear**。長專利 + N=5 可能讓 $/doc 高到 model ensembling 變相對便宜
- **原論文沒講這些**。paper 是在 arithmetic QA benchmark 上證明，production 這些細節全得自己處理
