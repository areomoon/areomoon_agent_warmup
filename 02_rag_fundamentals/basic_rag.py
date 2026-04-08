"""
Basic RAG Pipeline for Scientific Papers
=========================================

RAG (Retrieval-Augmented Generation) pipeline for scientific documents:
  1. Load PDF → parse text
  2. Chunk into segments (semantic or recursive)
  3. Embed chunks → vector index
  4. Query: retrieve top-k relevant chunks → feed to LLM → generate answer

For materials science: the "documents" are research papers;
the "queries" are requests for specific experimental parameters.

References:
  - RAG for Knowledge-Intensive NLP (Lewis et al., 2020): https://arxiv.org/abs/2005.11401
  - LlamaIndex documentation: https://docs.llamaindex.ai/
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LlamaIndex RAG Pipeline ───────────────────────────────────────────────────

def build_rag_pipeline(pdf_path: str, llm_model: str = "gpt-4.1-mini"):
    """
    Build a RAG query engine from a PDF file.

    Args:
        pdf_path: Path to a scientific paper PDF
        llm_model: OpenAI model to use for generation

    Returns:
        LlamaIndex query engine
    """
    try:
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError:
        print("Install: pip install llama-index llama-index-llms-openai llama-index-embeddings-huggingface")
        return None

    # Use a lightweight local embedding model to avoid embedding API costs
    # BGE-small is fast and good for scientific text
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = OpenAI(model=llm_model, temperature=0.1, api_key=os.getenv("OPENAI_API_KEY"))

    # Load document
    pdf_dir = str(Path(pdf_path).parent)
    documents = SimpleDirectoryReader(
        input_files=[pdf_path],
        filename_as_id=True,
    ).load_data()

    print(f"Loaded {len(documents)} document chunks from {Path(pdf_path).name}")

    # Build index
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    query_engine = index.as_query_engine(similarity_top_k=5)

    return query_engine


def extract_parameters_with_rag(query_engine, queries: list[str]) -> dict:
    """
    Run a set of extraction queries against the RAG pipeline.

    Args:
        query_engine: LlamaIndex query engine
        queries: List of natural language extraction queries

    Returns:
        Dict mapping query → response
    """
    results = {}
    for query in queries:
        print(f"\nQuery: {query}")
        response = query_engine.query(query)
        results[query] = str(response)
        print(f"Answer: {response}")

    return results


EXTRACTION_QUERIES = [
    "What is the synthesis temperature and duration for the main material in this paper?",
    "What characterization methods were used and what were the key measured properties?",
    "Extract all experimental parameters as a structured list with values and units.",
    "What material was synthesized and what is its chemical formula?",
    "What were the performance metrics or key results reported?",
]


def run_example():
    """
    Run the RAG pipeline on a PDF.
    Place a scientific paper PDF at data/sample_paper.pdf to test.
    """
    sample_pdf = "data/sample_paper.pdf"

    if not Path(sample_pdf).exists():
        print(f"Place a scientific paper PDF at: {sample_pdf}")
        print("Creating data/ directory...")
        Path("data").mkdir(exist_ok=True)
        print("\nDemo mode: showing pipeline structure without real PDF")
        print("\nRAG Pipeline steps:")
        print("1. PDF → load with SimpleDirectoryReader")
        print("2. Chunk with SentenceSplitter (chunk_size=512, overlap=50)")
        print("3. Embed with BGE-small-en-v1.5 (local, free)")
        print("4. Index in VectorStoreIndex (in-memory)")
        print("5. Query: top-5 chunks → GPT-4.1-mini → structured answer")
        return

    query_engine = build_rag_pipeline(sample_pdf)
    if query_engine:
        extract_parameters_with_rag(query_engine, EXTRACTION_QUERIES)


if __name__ == "__main__":
    run_example()

# TODO: Add chunking strategy comparison (fixed vs. semantic vs. hierarchical)
# TODO: Add ChromaDB for persistent vector storage
# TODO: Add re-ranking with Cohere reranker
# TODO: Add hybrid search (dense + BM25 sparse)
# TODO: Add structured output enforcement (JSON schema)
