"""Reciprocal Rank Fusion (RRF) for combining retrieval results.

RRF is a simple, effective method to combine rankings from multiple
retrieval systems. It works by summing reciprocal ranks:

    RRF_score(d) = Σ 1 / (k + rank_i(d))

Where:
- d is a document
- rank_i(d) is the rank of d in retrieval system i
- k is a constant (typically 60)

The parameter k controls how much low ranks contribute:
- Small k: Top ranks dominate
- Large k: All ranks contribute similarly
- k=60: Empirically good balance
"""
from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple rankings using RRF.

    Args:
        rankings: List of rankings, each is a list of document IDs in rank order
                  (rank 1 first, then rank 2, etc.)
        k: RRF constant (default 60)

    Returns:
        List of (doc_id, score) tuples sorted by score descending

    Example:
        >>> rankings = [
        ...     ["doc1", "doc2", "doc3"],  # BM25 ranking
        ...     ["doc2", "doc1", "doc4"],  # Semantic ranking
        ... ]
        >>> rrf = reciprocal_rank_fusion(rankings)
        >>> print(rrf[0])  # Top document
        ('doc2', 0.032...)  # doc2 ranked well in both
    """
    scores: dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_scores


def rrf_with_weights(
    rankings: list[list[str]],
    weights: list[float],
    k: int = 60,
) -> list[tuple[str, float]]:
    """RRF with per-source weights.

    Allows giving more importance to certain retrieval systems.
    For example, if semantic search is more reliable for your domain,
    give it a higher weight.

    Args:
        rankings: List of rankings
        weights: Weight for each ranking source (should sum to ~1.0)
        k: RRF constant

    Returns:
        Weighted RRF scores

    Example:
        >>> rankings = [
        ...     ["doc1", "doc2", "doc3"],  # BM25
        ...     ["doc2", "doc1", "doc4"],  # Semantic
        ... ]
        >>> weights = [0.3, 0.7]  # Prefer semantic
        >>> rrf = rrf_with_weights(rankings, weights)
    """
    if len(rankings) != len(weights):
        raise ValueError("Rankings and weights must have same length")

    scores: dict[str, float] = defaultdict(float)

    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += weight / (k + rank)

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_scores


def rrf_with_scores(
    scored_rankings: list[list[tuple[str, float]]],
    k: int = 60,
    score_transform: str = "rank",
) -> list[tuple[str, float]]:
    """RRF variant that can use original scores instead of just ranks.

    By default, RRF ignores the original scores and only uses ranks.
    This variant can optionally incorporate original scores.

    Args:
        scored_rankings: List of rankings where each is [(doc_id, score), ...]
                        sorted by score descending
        k: RRF constant
        score_transform: How to use scores:
            - "rank": Ignore scores, use only ranks (default RRF)
            - "normalize": Normalize scores to 0-1, multiply with RRF
            - "direct": Use scores directly with RRF

    Returns:
        Combined scores
    """
    if score_transform == "rank":
        # Standard RRF - convert to rankings and ignore scores
        rankings = [[doc_id for doc_id, _ in ranking] for ranking in scored_rankings]
        return reciprocal_rank_fusion(rankings, k)

    scores: dict[str, float] = defaultdict(float)

    for ranking in scored_rankings:
        if not ranking:
            continue

        # Normalize scores if requested
        if score_transform == "normalize":
            max_score = max(s for _, s in ranking) if ranking else 1.0
            min_score = min(s for _, s in ranking) if ranking else 0.0
            score_range = max_score - min_score if max_score != min_score else 1.0

        for rank, (doc_id, orig_score) in enumerate(ranking, start=1):
            rrf_score = 1.0 / (k + rank)

            if score_transform == "normalize":
                norm_score = (orig_score - min_score) / score_range
                scores[doc_id] += rrf_score * (0.5 + 0.5 * norm_score)
            elif score_transform == "direct":
                scores[doc_id] += rrf_score * orig_score
            else:
                scores[doc_id] += rrf_score

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_scores
