# 02 RAG Fundamentals

**對應 Warmup Week 2 | 預估時數：10–12hr**

## 學習目標

建立能對科學論文 PDF 做問答、取回結構化實驗參數的 RAG prototype。

完成後你能：
- [ ] 用 LlamaIndex 建一個端到端 RAG pipeline
- [ ] 對科學論文 PDF 問答並取回實驗參數
- [ ] 比較不同 embedding 模型的效果
- [ ] 理解 naive RAG vs. advanced RAG 的差異

## 執行順序

1. `python basic_rag.py` — 基礎 RAG pipeline（需要一份 PDF）
2. `python embedding_search.py` — 比較 embedding 模型
3. 打開 `notebook/rag_lab.ipynb` — 用真實論文做問答

## 關鍵概念

**RAG Pipeline：**
```
PDF → Load → Chunk → Embed → Index → [Query] → Retrieve → Generate
```

**Chunking 策略比較：**
| 策略 | 說明 | 適合場景 |
|------|------|---------|
| Fixed-size | 固定 token 數切割 | 快速原型 |
| Recursive | 按段落/句子遞迴切割 | 一般文章 |
| Semantic | 按語意邊界切割 | 科學論文 |
| Sequential | 保留前後文關係 | 需要上下文的查詢 |
