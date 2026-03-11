"""Hybrid retrieval combining BM25 and semantic search.

This module implements a two-stage retrieval system:
1. BM25 (lexical) - matches exact terms, table names, SQL keywords
2. Semantic (embedding) - matches conceptual similarity

Results are combined using Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from src.retrieval.models import FewShotExample
from src.retrieval.rrf import rrf_with_weights


class HybridRetriever:
    """Retrieves few-shot examples using hybrid BM25 + semantic search.

    Usage:
        retriever = HybridRetriever(examples)
        results = retriever.retrieve("How many customers?", top_k=5)
    """

    def __init__(
        self,
        examples: list[FewShotExample],
        embedding_model: str = "all-MiniLM-L6-v2",
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ):
        """Initialize the retriever.

        Args:
            examples: Library of few-shot examples
            embedding_model: Sentence transformer model name
            bm25_weight: Weight for BM25 in fusion (0-1)
            semantic_weight: Weight for semantic in fusion (0-1)
        """
        self.examples = examples
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self._embedding_model_name = embedding_model

        # Lazy loading for optional dependencies
        self._bm25: Optional[object] = None
        self._embedder: Optional[object] = None
        self._embeddings: Optional[np.ndarray] = None

    def _ensure_bm25(self) -> None:
        """Lazy-load BM25 index."""
        if self._bm25 is not None:
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 required: pip install rank-bm25")

        tokenized = [self._tokenize(ex.question) for ex in self.examples]
        self._bm25 = BM25Okapi(tokenized)

    def _ensure_embeddings(self) -> None:
        """Lazy-load embedding model and index."""
        if self._embeddings is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

        self._embedder = SentenceTransformer(self._embedding_model_name)
        self._embeddings = self._embedder.encode(
            [ex.question for ex in self.examples],
            convert_to_numpy=True,
        )

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25.

        Lowercases and splits on whitespace.
        Could be enhanced with stemming, stopword removal, etc.
        """
        return text.lower().split()

    def retrieve_bm25(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """BM25-only retrieval.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (example_index, score) tuples
        """
        self._ensure_bm25()

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    def retrieve_semantic(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Semantic-only retrieval.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (example_index, score) tuples
        """
        self._ensure_embeddings()

        query_embedding = self._embedder.encode([query], convert_to_numpy=True)
        similarities = np.dot(self._embeddings, query_embedding.T).flatten()

        # Get top-k indices
        top_indices = np.argsort(-similarities)[:top_k]
        return [(int(i), float(similarities[i])) for i in top_indices]

    def retrieve(self, query: str, top_k: int = 5) -> list[FewShotExample]:
        """Hybrid retrieval combining BM25 and semantic search.

        Uses RRF to combine rankings from both methods.

        Args:
            query: Natural language question
            top_k: Number of examples to return

        Returns:
            List of most relevant examples
        """
        # Get rankings from both methods
        # Retrieve more candidates than final top_k for better fusion
        candidate_count = min(top_k * 4, len(self.examples))

        bm25_results = self.retrieve_bm25(query, candidate_count)
        semantic_results = self.retrieve_semantic(query, candidate_count)

        # Convert to rankings (lists of doc IDs as strings)
        bm25_ranking = [str(idx) for idx, _ in bm25_results]
        semantic_ranking = [str(idx) for idx, _ in semantic_results]

        # RRF fusion
        rankings = [bm25_ranking, semantic_ranking]
        weights = [self.bm25_weight, self.semantic_weight]

        fused = rrf_with_weights(rankings, weights)

        # Get top-k examples
        top_indices = [int(doc_id) for doc_id, _ in fused[:top_k]]
        return [self.examples[i] for i in top_indices]

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[FewShotExample, float]]:
        """Retrieve examples with their fusion scores.

        Useful for debugging or confidence thresholding.

        Args:
            query: Natural language question
            top_k: Number of examples to return

        Returns:
            List of (example, score) tuples
        """
        candidate_count = min(top_k * 4, len(self.examples))

        bm25_results = self.retrieve_bm25(query, candidate_count)
        semantic_results = self.retrieve_semantic(query, candidate_count)

        bm25_ranking = [str(idx) for idx, _ in bm25_results]
        semantic_ranking = [str(idx) for idx, _ in semantic_results]

        rankings = [bm25_ranking, semantic_ranking]
        weights = [self.bm25_weight, self.semantic_weight]

        fused = rrf_with_weights(rankings, weights)

        results = []
        for doc_id, score in fused[:top_k]:
            idx = int(doc_id)
            results.append((self.examples[idx], score))

        return results


class SimpleRetriever:
    """A simple keyword-based retriever for when ML models aren't available.

    This is a fallback that works without sentence-transformers or rank-bm25.
    Uses basic keyword matching with TF-IDF-like scoring.
    """

    def __init__(self, examples: list[FewShotExample]):
        """Initialize with examples.

        Args:
            examples: Library of few-shot examples
        """
        self.examples = examples

        # Build simple inverted index
        self._index: dict[str, set[int]] = {}
        for i, ex in enumerate(examples):
            for token in self._tokenize(ex.question):
                if token not in self._index:
                    self._index[token] = set()
                self._index[token].add(i)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Remove punctuation and lowercase
        import re
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()

    def retrieve(self, query: str, top_k: int = 5) -> list[FewShotExample]:
        """Retrieve examples by keyword overlap.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of matching examples
        """
        query_tokens = set(self._tokenize(query))
        scores: dict[int, float] = {}

        for token in query_tokens:
            if token in self._index:
                # IDF-like weighting: rare tokens are more valuable
                idf = 1.0 / len(self._index[token])
                for idx in self._index[token]:
                    scores[idx] = scores.get(idx, 0) + idf

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [self.examples[idx] for idx, _ in ranked[:top_k]]
