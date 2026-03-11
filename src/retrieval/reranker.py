"""LLM-based reranking for retrieval results.

Reranking uses an LLM to score or rank candidate examples based on
their relevance to the query. This is slower but more accurate than
initial retrieval methods.

Two approaches are implemented:
1. Individual scoring: Score each candidate separately (N LLM calls)
2. Batch ranking: Rank all candidates in one call (1 LLM call)

Batch ranking is recommended for production due to lower latency and cost.
"""
from __future__ import annotations

from src.adapters.base import LLMAdapter
from src.retrieval.models import FewShotExample


async def rerank_examples(
    query: str,
    candidates: list[FewShotExample],
    adapter: LLMAdapter,
    top_k: int = 3,
) -> list[FewShotExample]:
    """Rerank candidates using individual LLM scoring.

    Each candidate is scored separately for relevance to the query.
    This is more accurate but requires N LLM calls.

    Args:
        query: The user's question
        candidates: Initial retrieval results
        adapter: LLM adapter for scoring
        top_k: Number of examples to return

    Returns:
        Reranked top-k examples
    """
    if len(candidates) <= top_k:
        return candidates

    # Score each candidate
    scored = []
    for ex in candidates:
        score = await score_example(query, ex, adapter)
        scored.append((score, ex))

    # Sort by score descending
    scored.sort(key=lambda x: -x[0])
    return [ex for _, ex in scored[:top_k]]


async def score_example(
    query: str,
    example: FewShotExample,
    adapter: LLMAdapter,
) -> float:
    """Score how relevant an example is for a query.

    Args:
        query: The user's question
        example: The candidate example
        adapter: LLM adapter

    Returns:
        Score from 0-10 indicating relevance
    """
    prompt = f"""Rate how helpful this SQL example would be for answering a similar question.
Score from 0 (not helpful) to 10 (very helpful).

Consider:
- Similar query patterns (aggregations, joins, filters)
- Same or related tables
- Similar complexity level

Target question: {query}

Example question: {example.question}
Example SQL: {example.sql}

Return only a number 0-10:"""

    response = await adapter.generate(prompt)

    try:
        score = float(response.content.strip())
        return min(max(score, 0), 10)  # Clamp to 0-10
    except ValueError:
        return 5.0  # Default middle score on parse failure


async def batch_rerank(
    query: str,
    candidates: list[FewShotExample],
    adapter: LLMAdapter,
    top_k: int = 3,
) -> list[FewShotExample]:
    """Rerank using a single LLM call (more efficient).

    Instead of scoring each example separately, ask the LLM
    to rank all candidates in one call. This is faster and cheaper.

    Args:
        query: The user's question
        candidates: Initial retrieval results
        adapter: LLM adapter
        top_k: Number of examples to return

    Returns:
        Reranked top-k examples
    """
    if len(candidates) <= top_k:
        return candidates

    # Format candidates for the prompt
    examples_text = "\n".join([
        f"{i}. Q: {ex.question}\n   SQL: {ex.sql}"
        for i, ex in enumerate(candidates)
    ])

    prompt = f"""Given this target question, rank the following SQL examples from most to least helpful.
Consider which examples have similar patterns, use related tables, or demonstrate relevant techniques.

Return only the numbers in order, comma-separated (e.g., "2,0,3,1,4").
Most helpful example first.

Target question: {query}

Examples:
{examples_text}

Ranking (most helpful first):"""

    response = await adapter.generate(prompt)

    try:
        # Parse ranking
        ranking_str = response.content.strip()
        # Handle various formats: "2,0,3" or "2, 0, 3" or "2 0 3"
        ranking_str = ranking_str.replace(" ", ",").replace(",,", ",")
        ranking = [int(x.strip()) for x in ranking_str.split(",") if x.strip()]

        # Filter valid indices
        ranking = [i for i in ranking if 0 <= i < len(candidates)]

        # Ensure we have enough results
        if len(ranking) < top_k:
            # Add any missing indices
            all_indices = set(range(len(candidates)))
            missing = all_indices - set(ranking)
            ranking.extend(sorted(missing))

        return [candidates[i] for i in ranking[:top_k]]

    except (ValueError, IndexError):
        # Fallback: return original order
        return candidates[:top_k]


async def rerank_with_explanation(
    query: str,
    candidates: list[FewShotExample],
    adapter: LLMAdapter,
    top_k: int = 3,
) -> list[tuple[FewShotExample, str]]:
    """Rerank with explanations for why each example is relevant.

    Useful for debugging and understanding retrieval decisions.

    Args:
        query: The user's question
        candidates: Initial retrieval results
        adapter: LLM adapter
        top_k: Number of examples to return

    Returns:
        List of (example, explanation) tuples
    """
    if not candidates:
        return []

    examples_text = "\n".join([
        f"{i}. Q: {ex.question}\n   SQL: {ex.sql}"
        for i, ex in enumerate(candidates)
    ])

    prompt = f"""Given this target question, select the {top_k} most helpful SQL examples.
For each selected example, briefly explain why it's relevant.

Target question: {query}

Examples:
{examples_text}

For each selected example (most helpful first), respond in this format:
[index]: [brief explanation]

Selected examples:"""

    response = await adapter.generate(prompt)

    # Parse response
    results = []
    lines = response.content.strip().split("\n")

    for line in lines:
        if ":" not in line:
            continue

        try:
            # Parse "[index]: explanation" format
            idx_str, explanation = line.split(":", 1)
            idx = int(idx_str.strip().strip("[]"))

            if 0 <= idx < len(candidates):
                results.append((candidates[idx], explanation.strip()))

            if len(results) >= top_k:
                break

        except (ValueError, IndexError):
            continue

    # Fill in if we don't have enough
    if len(results) < top_k:
        used_indices = {candidates.index(ex) for ex, _ in results}
        for i, ex in enumerate(candidates):
            if i not in used_indices:
                results.append((ex, "No explanation available"))
            if len(results) >= top_k:
                break

    return results[:top_k]
