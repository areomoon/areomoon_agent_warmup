# 06 Fine-tuning

**對應 Warmup Week 5 + Phase 2（第3-4月）| 預估時數：12–15hr**

## 學習目標

完成一次 QLoRA 微調實驗，建立 eval benchmark，理解 fine-tuning 與 prompt engineering 的取捨。

完成後你能：
- [ ] 解釋 LoRA 的低秩分解原理（r, alpha, target_modules）
- [ ] 在 Colab/Kaggle 上完成一次 QLoRA 微調
- [ ] 建立 (input, expected_output) 格式的 eval benchmark
- [ ] 用 Exact Match / F1 / LLM-as-Judge 評估微調效果
- [ ] 比較：base model vs. fine-tuned vs. prompt engineering

## ⚠️ 硬體注意

```
Apple Silicon Mac：bitsandbytes 4-bit 量化不支援 MPS
推薦方案 A：Google Colab (T4 免費) 或 Kaggle (P100 免費)
推薦方案 B：mlx-lm（Apple MLX 框架，原生支援 Apple Silicon）
推薦方案 C：雲端 GPU（Lambda Labs / RunPod / Vast.ai）
```

## 執行順序

1. `python lora_basics.py` — 理解 LoRA 原理（本地可執行，不需 GPU）
2. `python qlora_training.py` — 訓練腳本骨架（在 Colab 執行）
3. 打開 `notebook/finetuning_lab.ipynb` — 完整實驗記錄

## QLoRA 推薦設定（材料科學 extraction）

```python
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,
    target_modules="all-linear",
    use_dora=True,           # DoRA 通常比 LoRA 效果好
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# lr=2e-4, cosine warmup, 3 epochs
```

## 關鍵資源

| 資源 | 連結 |
|------|------|
| QLoRA 論文 | [arXiv 2305.14314](https://arxiv.org/abs/2305.14314) |
| QLoRA GitHub | [artidoro/qlora](https://github.com/artidoro/qlora) |
| LoRA 論文 | [arXiv 2106.09685](https://arxiv.org/abs/2106.09685) |
| HuggingFace PEFT | [docs.huggingface.co/peft](https://huggingface.co/docs/peft) |
| 2025 實用指南 | [reintech.io](https://reintech.io/blog/how-to-fine-tune-llms-with-lora-and-qlora-practical-guide) |
| 2026 完整教程 | [oneuptime.com](https://oneuptime.com/blog/post/2026-01-30-qlora-fine-tuning/view) |
