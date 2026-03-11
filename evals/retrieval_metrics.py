"""Retrieval-specific metrics for few-shot example selection.

This module implements comprehensive retrieval evaluation inspired by Spider
and test-suite-sql-eval (https://github.com/taoyds/test-suite-sql-eval):

1. TABLE OVERLAP: Do retrieved examples use tables relevant to the query?
2. CATEGORY MATCH: Is the query type similar (count, join, group, etc.)?
3. SQL COMPONENT SIMILARITY: Spider-style component matching (SELECT, WHERE, GROUP BY, etc.)
4. COMBINED RELEVANCE: Weighted score across all dimensions

These metrics matter because retrieval quality directly impacts SQL generation.
A "relevant" example should teach the right pattern, not just mention the right tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Dict

from src.retrieval.models import FewShotExample
from evals.golden_set import GoldenQuery
from evals.sql_components import (
    parse_sql_components,
    compute_component_f1,
    compute_similarity_from_components,
    SQLComponents,
)


def compute_sql_similarity(sql1: str, sql2: str) -> float:
    """Compute SQL similarity using Spider-style component matching.

    This replaces simple Jaccard similarity with proper component-based
    comparison following the test-suite-sql-eval approach:
    - Parse both queries into components (SELECT, FROM, WHERE, GROUP BY, etc.)
    - Compute F1 for each component
    - Return weighted overall similarity

    Args:
        sql1: First SQL query
        sql2: Second SQL query

    Returns:
        Similarity score (0 to 1)
    """
    return compute_similarity_from_components(sql1, sql2)


@dataclass
class RelevanceScore:
    """Multi-dimensional relevance score for a single example."""
    example_id: str
    table_overlap: float  # 0-1: fraction of query tables covered
    category_match: float  # 0 or 1: same category?
    pattern_similarity: float  # 0-1: SQL component similarity (Spider-style)
    combined_score: float  # Weighted combination


@dataclass
class RetrievalResult:
    """Result of evaluating retrieval for a single query."""

    query_id: str
    question: str
    retrieved_examples: List[FewShotExample]
    query_tables: Set[str]
    query_category: str
    query_sql: str  # For Spider-style component matching

    # Per-example scores
    example_scores: List[RelevanceScore] = field(default_factory=list)

    # Aggregate metrics
    mrr_table: float = 0.0  # MRR based on table overlap
    mrr_category: float = 0.0  # MRR based on category match
    mrr_pattern: float = 0.0  # MRR based on SQL component similarity
    mrr_combined: float = 0.0  # MRR based on combined score

    precision_at_k: float = 0.0  # Fraction with combined_score > threshold
    mean_relevance: float = 0.0  # Average combined score of retrieved


def compute_example_relevance(
    example: FewShotExample,
    query_tables: Set[str],
    query_category: str,
    query_sql: str,
    weights: Dict[str, float] = None,
) -> RelevanceScore:
    """Compute multi-dimensional relevance score for one example.

    Uses Spider-style component matching for SQL similarity instead of
    simple pattern detection. This follows the approach from
    test-suite-sql-eval (https://github.com/taoyds/test-suite-sql-eval).

    Args:
        example: The retrieved example
        query_tables: Tables the query needs
        query_category: Category of the query (count, join, etc.)
        query_sql: The gold SQL query for component comparison
        weights: Optional weights for combining scores

    Returns:
        RelevanceScore with all dimensions
    """
    weights = weights or {"table": 0.3, "category": 0.3, "pattern": 0.4}

    # Table overlap: what fraction of query tables does the example cover?
    ex_tables = set(example.tables_used)
    if query_tables:
        overlap_count = len(ex_tables & query_tables)
        table_overlap = overlap_count / len(query_tables)
    else:
        table_overlap = 1.0 if not ex_tables else 0.0

    # Category match: exact match
    category_match = 1.0 if example.category == query_category else 0.0

    # SQL component similarity: Spider-style component matching
    # Compares SELECT, FROM, WHERE, GROUP BY, ORDER BY as sets
    pattern_similarity = compute_sql_similarity(example.sql, query_sql)

    # Combined score
    combined = (
        weights["table"] * table_overlap +
        weights["category"] * category_match +
        weights["pattern"] * pattern_similarity
    )

    return RelevanceScore(
        example_id=example.id,
        table_overlap=table_overlap,
        category_match=category_match,
        pattern_similarity=pattern_similarity,
        combined_score=combined,
    )


def compute_mrr(scores: List[float], threshold: float = 0.5) -> float:
    """Compute MRR: reciprocal rank of first relevant result.

    Args:
        scores: List of relevance scores in retrieval order
        threshold: Minimum score to be considered relevant

    Returns:
        Reciprocal rank (0 if no relevant result)
    """
    for rank, score in enumerate(scores, start=1):
        if score >= threshold:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval_single(
    query: GoldenQuery,
    retrieved: List[FewShotExample],
    k: int = 5,
    relevance_threshold: float = 0.5,
) -> RetrievalResult:
    """Evaluate retrieval quality for a single query.

    Uses Spider-style component matching for SQL similarity evaluation.
    See: https://github.com/taoyds/test-suite-sql-eval

    Args:
        query: The golden query with expected tables, category, SQL
        retrieved: Retrieved examples (top-k)
        k: Number of examples to consider
        relevance_threshold: Score threshold for "relevant"

    Returns:
        RetrievalResult with comprehensive metrics
    """
    query_tables = set(query.tables_used) if hasattr(query, 'tables_used') and query.tables_used else set()
    query_category = query.category if hasattr(query, 'category') else "general"
    query_sql = query.sql

    top_k = retrieved[:k]

    # Compute per-example relevance scores using Spider-style component matching
    example_scores = []
    for ex in top_k:
        score = compute_example_relevance(ex, query_tables, query_category, query_sql)
        example_scores.append(score)

    # Compute MRR for each dimension
    table_scores = [s.table_overlap for s in example_scores]
    category_scores = [s.category_match for s in example_scores]
    pattern_scores = [s.pattern_similarity for s in example_scores]
    combined_scores = [s.combined_score for s in example_scores]

    mrr_table = compute_mrr(table_scores, threshold=0.5)
    mrr_category = compute_mrr(category_scores, threshold=0.5)
    mrr_pattern = compute_mrr(pattern_scores, threshold=0.3)
    mrr_combined = compute_mrr(combined_scores, threshold=relevance_threshold)

    # Precision@k: fraction above threshold
    relevant_count = sum(1 for s in combined_scores if s >= relevance_threshold)
    precision = relevant_count / k if k > 0 else 0.0

    # Mean relevance
    mean_rel = sum(combined_scores) / len(combined_scores) if combined_scores else 0.0

    return RetrievalResult(
        query_id=query.id,
        question=query.question,
        retrieved_examples=top_k,
        query_tables=query_tables,
        query_category=query_category,
        query_sql=query_sql,
        example_scores=example_scores,
        mrr_table=mrr_table,
        mrr_category=mrr_category,
        mrr_pattern=mrr_pattern,
        mrr_combined=mrr_combined,
        precision_at_k=precision,
        mean_relevance=mean_rel,
    )


def evaluate_retrieval(
    golden_set: List[GoldenQuery],
    retriever,
    k: int = 5,
    verbose: bool = True,
) -> dict:
    """Evaluate retrieval quality across a golden set.

    Args:
        golden_set: Test queries with expected tables, category, SQL
        retriever: Retriever with .retrieve(query, top_k) method
        k: Number of examples to retrieve
        verbose: Print progress

    Returns:
        Dictionary with aggregate metrics and per-query results
    """
    results = []

    for i, query in enumerate(golden_set, 1):
        retrieved = retriever.retrieve(query.question, top_k=k)
        result = evaluate_retrieval_single(query, retrieved, k)
        results.append(result)

        if verbose:
            status = "✓" if result.mrr_combined > 0 else "✗"
            print(f"  [{i}/{len(golden_set)}] {query.id}: "
                  f"MRR={result.mrr_combined:.2f} "
                  f"P@{k}={result.precision_at_k:.2f} "
                  f"(tbl={result.mrr_table:.2f}, cat={result.mrr_category:.2f}, pat={result.mrr_pattern:.2f}) {status}")

    # Aggregate metrics
    n = len(results) if results else 1

    return {
        # Combined metrics (primary)
        "mrr_combined": sum(r.mrr_combined for r in results) / n,
        f"precision_at_{k}": sum(r.precision_at_k for r in results) / n,
        "mean_relevance": sum(r.mean_relevance for r in results) / n,

        # Per-dimension MRR
        "mrr_table": sum(r.mrr_table for r in results) / n,
        "mrr_category": sum(r.mrr_category for r in results) / n,
        "mrr_pattern": sum(r.mrr_pattern for r in results) / n,

        # Coverage
        "queries_with_relevant": sum(1 for r in results if r.mrr_combined > 0),
        "total_queries": len(results),
        "coverage": sum(1 for r in results if r.mrr_combined > 0) / n,

        # Raw results for analysis
        "results": results,
    }


def format_retrieval_report(metrics: dict, k: int = 5) -> str:
    """Format retrieval metrics as a readable report.

    Args:
        metrics: Dictionary from evaluate_retrieval()
        k: The k value used

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 70,
        "RETRIEVAL QUALITY METRICS (Spider-style)",
        "=" * 70,
        "",
        "Combined Metrics (Primary)",
        "-" * 40,
        f"  MRR (Combined):       {metrics['mrr_combined']:.3f}",
        f"  Precision@{k}:         {metrics[f'precision_at_{k}']:.3f}",
        f"  Mean Relevance:       {metrics['mean_relevance']:.3f}",
        f"  Coverage:             {metrics['coverage']:.1%} ({metrics['queries_with_relevant']}/{metrics['total_queries']})",
        "",
        "Per-Dimension MRR",
        "-" * 40,
        f"  MRR (Table Overlap):  {metrics['mrr_table']:.3f}",
        f"  MRR (Category Match): {metrics['mrr_category']:.3f}",
        f"  MRR (SQL Pattern):    {metrics['mrr_pattern']:.3f}",
        "",
        "Interpretation",
        "-" * 40,
    ]

    mrr = metrics['mrr_combined']
    if mrr >= 0.7:
        lines.append(f"  ✓ MRR {mrr:.2f} >= 0.7: Good retrieval quality")
    elif mrr >= 0.5:
        lines.append(f"  ⚠ MRR {mrr:.2f} in [0.5, 0.7): Moderate quality")
    else:
        lines.append(f"  ✗ MRR {mrr:.2f} < 0.5: Poor retrieval")

    # Diagnose weak dimensions
    if metrics['mrr_category'] < metrics['mrr_table']:
        lines.append("  → Category match is weak: add more diverse example categories")
    if metrics['mrr_pattern'] < metrics['mrr_table']:
        lines.append("  → SQL pattern match is weak: add examples with similar SQL structures")

    precision = metrics[f'precision_at_{k}']
    if precision >= 0.6:
        lines.append(f"  ✓ Precision@{k} {precision:.2f} >= 0.6: Good precision")
    elif precision >= 0.4:
        lines.append(f"  ⚠ Precision@{k} {precision:.2f} in [0.4, 0.6): Moderate precision")
    else:
        lines.append(f"  ✗ Precision@{k} {precision:.2f} < 0.4: Low precision")

    lines.append("=" * 70)
    return "\n".join(lines)


def compare_retrieval_methods(
    golden_set: List[GoldenQuery],
    retrievers: dict,  # name -> retriever
    k: int = 5,
) -> dict:
    """Compare multiple retrieval methods.

    Args:
        golden_set: Test queries
        retrievers: Dictionary of name -> retriever
        k: Number of examples

    Returns:
        Dictionary with metrics for each method
    """
    comparison = {}

    for name, retriever in retrievers.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_retrieval(golden_set, retriever, k, verbose=False)
        comparison[name] = {
            "mrr_combined": metrics["mrr_combined"],
            "mrr_table": metrics["mrr_table"],
            "mrr_category": metrics["mrr_category"],
            "mrr_pattern": metrics["mrr_pattern"],
            f"precision_at_{k}": metrics[f"precision_at_{k}"],
            "coverage": metrics["coverage"],
        }

    return comparison


def analyze_retrieval_failures(
    results: List[RetrievalResult],
    threshold: float = 0.5,
) -> dict:
    """Analyze why retrieval failed for certain queries.

    Args:
        results: List of RetrievalResult from evaluate_retrieval
        threshold: Relevance threshold for failure

    Returns:
        Analysis of failure patterns
    """
    failures = [r for r in results if r.mrr_combined == 0]
    successes = [r for r in results if r.mrr_combined > 0]

    # Categorize failures
    failure_analysis = {
        "total_failures": len(failures),
        "no_table_overlap": 0,
        "no_category_match": 0,
        "no_pattern_match": 0,
        "failure_by_category": {},
    }

    for r in failures:
        # Check why it failed
        if r.mrr_table == 0:
            failure_analysis["no_table_overlap"] += 1
        if r.mrr_category == 0:
            failure_analysis["no_category_match"] += 1
        if r.mrr_pattern == 0:
            failure_analysis["no_pattern_match"] += 1

        # Count by category
        cat = r.query_category
        failure_analysis["failure_by_category"][cat] = \
            failure_analysis["failure_by_category"].get(cat, 0) + 1

    return failure_analysis
