# 工具清單

## LLM API（主要用這些，不跑本地大模型）

| 工具 | 用途 | 備註 |
|------|------|------|
| OpenAI API (`gpt-4.1-mini`) | 主力 API，cost-effective | `pip install openai` |
| Anthropic API (`claude-sonnet-4-6`) | 長文本/PDF 理解強 | `pip install anthropic` |
| Google Gemini API (`gemini-2.5-flash`) | 免費額度高，1M context | `pip install google-genai` |

## Agent 框架

| 工具 | 用途 | 文件 |
|------|------|------|
| **LangGraph** | 狀態機 agent，生產首選 | [langgraph.com](https://langchain-ai.github.io/langgraph/) |
| LangChain | RAG、工具調用 | [langchain.com](https://python.langchain.com/) |
| LlamaIndex | 科學文件 RAG | [llamaindex.ai](https://docs.llamaindex.ai/) |
| Claude Agent SDK | Anthropic 原生 | [anthropic.com](https://docs.anthropic.com/) |

## RAG & 向量資料庫

| 工具 | 用途 | 備註 |
|------|------|------|
| ChromaDB | 輕量向量 DB | 本地，適合原型 |
| FAISS | 高效相似度搜索 | Meta 出品 |
| `BAAI/bge-small-en-v1.5` | 本地 embedding 模型 | `pip install sentence-transformers` |

## 文件處理

| 工具 | 用途 | 備註 |
|------|------|------|
| PyMuPDF (`fitz`) | PDF 文字/頁面提取 | `pip install pymupdf` |
| pypdf | PDF 解析 | 較輕量 |
| markdownify | HTML → Markdown | `pip install markdownify` |

## Fine-tuning（在 Colab/Kaggle 執行）

| 工具 | 用途 | 備註 |
|------|------|------|
| HuggingFace Transformers | 模型載入 | `pip install transformers` |
| PEFT | LoRA/QLoRA 實作 | `pip install peft` |
| TRL SFTTrainer | SFT 訓練 | `pip install trl` |
| bitsandbytes | 4-bit 量化 | **Linux/GPU only** |
| mlx-lm | Apple Silicon 微調 | Mac 替代方案 |

## 材料科學工具（入職後接觸）

| 工具 | 用途 | 備註 |
|------|------|------|
| Materials Project API | 材料性質資料庫 | 需申請 API key |
| pymatgen | 材料結構分析 | `pip install pymatgen` |
| ASE (Atomic Simulation Environment) | 原子模擬 | `pip install ase` |

## 訓練環境

| 環境 | GPU | 費用 | 用途 |
|------|-----|------|------|
| Google Colab | T4 (16GB) | 免費 | 7B 模型 QLoRA |
| Kaggle Notebooks | P100 (16GB) | 免費 | 7B 模型 QLoRA |
| Lambda Labs | A100 (40GB) | ~$1.5/hr | 大模型實驗 |
| Mac (Apple Silicon) | MPS | — | 推理、小規模實驗（mlx-lm） |
