# 07 Multimodal

**對應 Warmup Week 4 | 預估時數：8–10hr**

## 學習目標

能處理含圖表的科學論文（XRD patterns、SEM/TEM、spectroscopy plots），輸出結構化 JSON。

完成後你能：
- [ ] 用 GPT-4.1-mini / Claude vision API 分析科學圖表
- [ ] 區分並處理：表格 / 圖片 / 圖表 / 公式 四種模態
- [ ] 輸出含信心分數的結構化 JSON
- [ ] 設計 multi-modal extraction pipeline

## 科學論文四種模態

| 模態 | 內容 | 主要工具 |
|------|------|---------|
| 文字 | 摘要、方法、結論 | 直接 LLM 讀取 |
| 表格 | 實驗數據、結果比較 | Vision LLM + table parser |
| 圖片 | SEM/TEM、XRD pattern | Vision LLM |
| 圖表 | 折線圖、散點圖 | Vision LLM + chart parser |

## 執行順序

1. `python image_analysis.py` — 圖表理解基礎
2. 打開 `notebook/multimodal_lab.ipynb` — 完整 multi-modal 實驗

## 關鍵資源

- [ColPali (arXiv 2407.01449)](https://arxiv.org/abs/2407.01449) — 用視覺模型做文件檢索
- [DocLLM (arXiv 2401.00908)](https://arxiv.org/abs/2401.00908)
- [Claude API Vision](https://docs.anthropic.com/en/docs/build-with-claude/vision)
