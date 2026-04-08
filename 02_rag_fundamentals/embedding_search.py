"""
Embedding Models for Scientific Text Search
============================================

Compares embedding models for materials science retrieval tasks.
Key consideration: scientific text has domain-specific vocabulary
(chemical formulas, method names, units) that general-purpose embeddings
may not represent well.

Models tested:
- BAAI/bge-small-en-v1.5 (fast, local, good general quality)
- text-embedding-3-small (OpenAI, API cost, strong general)
- BAAI/bge-m3 (multilingual, handles Chinese+English mixed text)

References:
  - BGE: https://huggingface.co/BAAI/bge-small-en-v1.5
  - Text Embedding 3: https://platform.openai.com/docs/guides/embeddings
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Sample materials science knowledge base
KNOWLEDGE_BASE = [
    "La₀.₈Sr₀.₂MnO₃ thin films deposited by PLD at 700°C show metal-insulator transition at 370 K",
    "ZnO nanorods synthesized hydrothermally at 95°C for 6 hours, diameter 200 nm",
    "BaTiO₃ ceramics sintered at 1300°C exhibit dielectric constant of 2800 at room temperature",
    "Perovskite solar cell with MAPbI₃ achieves 23% power conversion efficiency",
    "LSMO films on SrTiO₃ substrate show colossal magnetoresistance below T_MI",
    "Carbon nanotube networks show electrical conductivity of 10⁴ S/cm",
    "LiFePO₄ cathode material cycling stability: 95% capacity retention after 500 cycles",
    "Graphene oxide reduction at 200°C in H₂/Ar gives conductivity ~6000 S/m",
]

QUERIES = [
    "What is the synthesis temperature for ZnO nanorods?",
    "Which material shows metal-insulator transition?",
    "Perovskite solar cell efficiency",
    "battery cathode cycling performance",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def embed_with_huggingface(texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    """Embed texts using a local HuggingFace model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install: pip install sentence-transformers")
        return np.array([])

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings


def embed_with_openai(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed texts using OpenAI embedding API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except ImportError:
        print("Install: pip install openai")
        return np.array([])

    response = client.embeddings.create(input=texts, model=model)
    embeddings = np.array([r.embedding for r in response.data])
    return embeddings


def retrieve_top_k(query: str, knowledge_base: list[str], embed_fn, k: int = 3) -> list[tuple[str, float]]:
    """Retrieve top-k most relevant items for a query."""
    all_texts = [query] + knowledge_base
    embeddings = embed_fn(all_texts)
    if len(embeddings) == 0:
        return []

    query_emb = embeddings[0]
    kb_embs = embeddings[1:]

    scores = [(knowledge_base[i], cosine_similarity(query_emb, kb_embs[i]))
              for i in range(len(knowledge_base))]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def run_comparison():
    """Compare embedding models on materials science queries."""
    print("=== Embedding Model Comparison ===\n")

    # Test with local HuggingFace model (no API cost)
    print("Using BAAI/bge-small-en-v1.5 (local)...\n")
    embed_fn = lambda texts: embed_with_huggingface(texts, "BAAI/bge-small-en-v1.5")

    for query in QUERIES:
        print(f"Query: {query}")
        results = retrieve_top_k(query, KNOWLEDGE_BASE, embed_fn, k=2)
        for text, score in results:
            print(f"  [{score:.3f}] {text[:80]}...")
        print()


if __name__ == "__main__":
    run_comparison()

# TODO: Add BAAI/bge-m3 for Chinese+English mixed scientific text
# TODO: Benchmark: precision@k for a labeled materials retrieval dataset
# TODO: Compare dense-only vs. hybrid (dense + BM25) retrieval
# TODO: Test domain-specific embedding models (MatBERT, ChemBERTa)
