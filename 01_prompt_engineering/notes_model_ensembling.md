# Model Ensembling 筆記

> KB 正本：[model-ensembling](~/PycharmProjects/llm_knowledge_base/wiki/concepts/model-ensembling.md)
> 參考：[LLM-Blender (Jiang et al. 2023)](https://arxiv.org/abs/2306.02561)

## 一句話

**跨 model 家族投票**（Claude + GPT + Gemini），而不是同一 model 抽 N 次。逃出 self-consistency 的「共同 prior 陷阱」。成本與工程複雜度都遠高於 self-consistency，**只適合高價值低流量**。

## vs Self-Consistency

| | Self-Consistency N=5 | 3-model Ensemble |
|---|---|---|
| Call count | 5 | 3 |
| $/input | 5× 單 model | ~3×（pricing 不同，均價較高） |
| Diversity 來源 | Temperature noise | 訓練分布不同 |
| 抓共同 prior 幻覺？ | ❌（5 樣本都共享 prior） | ✅ |
| 工程複雜度 | 低（一個 SDK） | 高（3 SDK, auth, retry, schema） |

## 程式樣板

```python
results = await asyncio.gather(
    anthropic.extract(text),    # Claude Sonnet 4.6
    openai.extract(text),       # GPT-5
    google.extract(text),       # Gemini 2.5 Pro
)
# Field-level voting，和 self-consistency 一樣，但票來自不同家族
aggregated = field_level_vote(results)
```

## 核心屬性

- **逃出 shared-prior 幻覺**：如果 Claude、GPT、Gemini **各自獨立** 得到「200 mTorr」，才是真強證據；5 個 Claude 樣本都說 200 不是
- **工程複雜度才是主要成本**，不是 API 帳單。Schema 差異、rate limit、billing 對帳、每家 provider 的失敗模式都不同 —— 每項都是 team-month
- **跨 model calibration 非 trivial**：GPT 的 "confident" 跟 Claude 的 "confident" 對應不上。Vote share 可以；per-model confidence 對比不要
- **Vendor risk hedge**：一家掛了其他家扛，regulated industry 有時是採購硬條件
- **法律風險**：專利/法律資料可能有 per-provider ToS 限制，路由三家之前先確認

## 業界應用

- **NotDiamond / Martian** LLM router SaaS
- **Financial analysis**（Bloomberg GPT copilot）：finance-tuned + frontier 合用
- **Legal analysis** 事務所：GPT-4 + Claude-3-Opus + domain-tuned 審核 key clauses（採購條件）
- **Anthropic Constitutional AI 訓練**：用多 model preference corpus（不是 production ensemble 但同理）

## 什麼時候用 / 不用

**用**：
- 錯誤後果極度不對稱（客戶退款 > $1k、法律責任、safety-critical）
- 能吃下 3× 工程複雜度（multi-provider infrastructure）
- Task 會撞 shared-prior 幻覺（self-consistency 只會強化的那種）
- Regulatory / 採購要求 cross-vendor redundancy

**不用**：
- 高流量低單位價值（工程 amortize 不掉）
- 還沒試過 [self-consistency](notes_self_consistency_impl.md) + [retrieval verification](notes_retrieval_augmented_verification.md)（這組合解決大多數 case）
- Team 沒 multi-provider secret management / retry / observability infra

## Patsnap 落地策略

- **保留給 flagship 客戶的 Claim 1 抽取**，或新 extractor deployment 的 red-teaming
- **不用在 bulk 專利處理**（成本 / 複雜度攤不了）
