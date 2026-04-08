# LLMatDesign 架構研究筆記

**論文：** LLMatDesign: Autonomous Materials Discovery with Large Language Models
**arXiv：** https://arxiv.org/abs/2406.13163
**GitHub：** https://github.com/Fung-Lab/LLMatDesign
**關鍵詞：** propose/evaluate/refine, strategy library, self-reflection

---

## 核心貢獻

LLMatDesign 最重要的貢獻不是新演算法，而是：
> **Strategy Library** — 一個持久化的設計啟發式知識庫，跨實驗積累。

這就是 ACE Evolving Playbook 在材料發現領域的具體實現。

---

## 三步循環架構

```
┌──────────────────────────────────────────────┐
│              Strategy Library                │
│  (persistent, grows with each experiment)    │
└───────────┬──────────────────────────────────┘
            │ inject relevant strategies
            ▼
     ┌──────────────┐
     │   PROPOSE    │  LLM generates material candidate
     │              │  with explicit rationale + cites strategy
     └──────┬───────┘
            │ candidate composition
            ▼
     ┌──────────────┐
     │   EVALUATE   │  DFT / ML surrogate model
     │              │  → predicted property value
     └──────┬───────┘
            │ property result
            ▼
     ┌──────────────┐
     │   REFLECT    │  LLM analyzes gap (predicted vs target)
     │              │  → extracts transferable lesson
     └──────┬───────┘
            │ new strategy
            ▼
     Strategy Library updated (Curator step)
```

---

## Strategy Library 詳解

這是 LLMatDesign 最值得學習的組件。

### 結構
```python
{
  "strategies": [
    {
      "id": "S001",
      "domain": "perovskite_thermoelectric",
      "rule": "B-site doping with Bi in ABO₃ increases phonon scattering, reducing thermal conductivity",
      "evidence": ["paper_id_001", "experiment_run_042"],
      "success_rate": 0.73,
      "n_applications": 11
    }
  ]
}
```

### 注入方式
在每次 Propose step 之前，從 strategy library 檢索 top-k 相關策略注入 prompt：

```python
# 偽代碼
relevant_strategies = retrieve_strategies(target_property, material_class, top_k=5)
prompt = f"""Design a material to maximize {target_property}.

Relevant strategies from past experiments:
{format_strategies(relevant_strategies)}

Propose a candidate with rationale..."""
```

### 更新規則（Grow-and-Refine）
- 成功的實驗 → 策略 success_rate 提升
- 失敗的實驗 → 策略 success_rate 降低（或標記 deprecate）
- 新發現的啟發式 → append 新 strategy
- 重複/衝突的策略 → merge 或 deprecate

---

## Self-Reflection 機制

每次評估後，LLM 執行 self-reflection：

```
Given:
- Target property: ZT > 2.0 at 500°C
- Proposed material: PbTe + 5% Bi₂Te₃
- Predicted ZT: 1.6

Reflection prompt:
"Why did the prediction fall short?
What systematic bias should we correct?
What new strategy can be generalized?"

→ Lesson: "Higher Bi₂Te₃ fraction (8-12%) consistently increases ZT
   in PbTe matrix, contrary to our initial hypothesis of 5%"
```

這個 lesson 被加入 strategy library，下一次 propose 就不會重蹈覆轍。

---

## Cross-Material Transfer

LLMatDesign 的另一個重要特性：策略可以跨材料類別遷移。

例如：
- 在 PbTe thermoelectrics 學到「異質摻雜降低熱導率」
- 這個策略被遷移到 SnSe thermoelectrics → 同樣有效

實現方式：strategy 的標籤不只包含具體材料，也包含物理機制：
```
tag: phonon_scattering | thermal_conductivity | doping
```

這讓 strategy library 成為**可遷移的物理直覺庫**，而非特定材料的記憶。

---

## 與你的工作的關聯

你做的是 **extraction**（不是 discovery），但 LLMatDesign 的 strategy library 概念完全適用：

| LLMatDesign（discovery） | 你的工作（extraction） |
|------------------------|---------------------|
| 材料設計規則 | 論文解析規則 |
| 「摻雜 Bi 降低熱導率」 | 「'typical' 在表格標題 → 單樣本數據，降低信心」 |
| DFT 驗證失敗 → 策略更新 | 交叉驗證失敗 → extraction rule 更新 |
| 跨材料類別遷移 | 跨論文類型遷移（review vs. primary） |

**入職初期的行動：**
1. 建立一個 JSON 格式的 extraction strategy library（小型 playbook）
2. 每次 reflection 找到的 extraction rule 都加進去
3. 每次開始新論文時，注入 top-5 相關 rules 進 prompt
4. 這就是 ACE + LLMatDesign 的融合實踐

---

## 論文中的實驗結果

- 在 thermoelectric material discovery 任務上，比 random search 快 3-5×
- Strategy library 在 10 次實驗後開始顯著提升 proposal 品質
- Cross-material transfer 讓冷啟動問題降低（不需要從頭學習每種材料類別）

**對你的啟示：** Extraction playbook 在處理 ~10 篇同類論文後會開始提供顯著改善。
入職第一個月就值得開始累積。
