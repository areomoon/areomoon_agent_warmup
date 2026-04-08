# 04 ACE Framework

**對應 Warmup Week 3–4 | 預估時數：6–8hr**

## 學習目標

理解 ACE Grow-and-Refine 機制，實作 Curator delta update，建立 materials extraction 專用 playbook。

完成後你能：
- [ ] 解釋 Brevity Bias 和 Context Collapse 兩個失敗模式
- [ ] 實作 playbook append → merge → deduplicate 循環
- [ ] 為材料科學 extraction 設計四層 playbook 結構
- [ ] 理解 ACE 與 fine-tuning 的互補關係

## 執行順序

1. `python playbook_evolution.py` — Grow-and-Refine 核心機制
2. `python curator_pattern.py` — Curator delta update 實作
3. 打開 `notebook/ace_lab.ipynb` — 完整 GRC loop 模擬

## ACE 四層 Playbook（材料科學版）

```
Layer 1: Structural Design Rules    ("PbTe + Bi doping → ZT > 1.5 above 500K")
Layer 2: Synthesis Protocol Patterns ("Perovskite solid-state: 900°C/12h calcination")
Layer 3: Characterization Signatures ("Raman D band ~1350 cm⁻¹ = graphene indicator")
Layer 4: Extraction Heuristics       ("'typical' in table caption → single sample")
```

## 關鍵資源

- [ACE Paper (arXiv 2510.04618)](https://arxiv.org/abs/2510.04618)
- [ACE GitHub](https://github.com/ace-agent/ace)
- [ACE Playbook 社群實作](https://github.com/jmanhype/ace-playbook)
