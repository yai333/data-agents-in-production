"""Few-shot retrieval for Text-to-SQL agents.

This module provides retrieval capabilities for finding relevant
few-shot examples to include in SQL generation prompts.

Key components:
- FewShotExample: Data model for examples
- HybridRetriever: In-memory BM25 + semantic search with RRF fusion
- PgVectorRetriever: Database-backed retrieval via pgvector (production)
- reranker: LLM-based reranking for precision

Usage (in-memory, for development):
    from src.retrieval import HybridRetriever
    retriever = HybridRetriever(examples)
    results = retriever.retrieve("How many customers?")

Usage (pgvector, for production):
    from src.retrieval import PgVectorRetriever
    retriever = await PgVectorRetriever.create(database_url)
    results = await retriever.retrieve("How many customers?")
"""

from src.retrieval.models import FewShotExample, render_example, render_examples
from src.retrieval.retriever import HybridRetriever, SimpleRetriever
from src.retrieval.rrf import reciprocal_rank_fusion, rrf_with_weights
from src.retrieval.pgvector_store import PgVectorRetriever, create_embeddings

__all__ = [
    "FewShotExample",
    "render_example",
    "render_examples",
    "HybridRetriever",
    "SimpleRetriever",
    "PgVectorRetriever",
    "create_embeddings",
    "reciprocal_rank_fusion",
    "rrf_with_weights",
]
