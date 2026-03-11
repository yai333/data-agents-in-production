#!/usr/bin/env python3
"""2.3 Retrieval - Comprehensive Evaluation.

This script evaluates retrieval quality and its impact on SQL generation:

1. RETRIEVAL METRICS (Spider-style evaluation):
   - MRR (Mean Reciprocal Rank): Where does the first relevant result appear?
   - Precision@k: What fraction of top-k results are relevant?
   - Table Overlap: Do retrieved examples use tables relevant to the query?

2. RETRIEVAL METHOD COMPARISON:
   - BM25 only (lexical matching)
   - Semantic only (embedding-based)
   - Hybrid (RRF fusion of both)

3. END-TO-END SQL GENERATION:
   - Schema only baseline (from 2.2)
   - With retrieved examples
   - Improvement measurement

Expected results:
- Hybrid retrieval: MRR > 0.7, Precision@3 > 60%
- SQL accuracy improvement: +10-20% with good retrieval

Usage:
    make db-up
    python scripts/generate_chinook_schema.py
    python scripts/run_chapter_2_3.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import asyncpg

from src.adapters import create_adapter
from src.schema import SchemaStore, render_schema
from src.retrieval import HybridRetriever, SimpleRetriever, render_examples
from src.retrieval.chinook_examples import CHINOOK_EXAMPLES, get_example_summary
from src.utils.config import load_config
from evals.runner import run_evaluation
from evals.metrics import format_metrics_report, compare_metrics
from evals.retrieval_metrics import (
    evaluate_retrieval,
    format_retrieval_report,
    compare_retrieval_methods,
)
from evals.chinook_golden_set import GOLDEN_SET


async def generate_with_schema_only(
    question: str,
    schema_store: SchemaStore,
) -> str:
    """Generate SQL with schema but no examples (from 2.2)."""
    adapter = create_adapter()

    relevant_tables = schema_store.search_tables(question)
    if not relevant_tables:
        relevant_tables = schema_store.get_all_tables()[:10]

    schema_text = render_schema(relevant_tables)

    prompt = f"""You are a SQL expert. Generate a SQL query for the given question.
Use only the tables and columns described in the schema below.
Return only the SQL query, no explanation or markdown.

DATABASE SCHEMA:
{schema_text}

QUESTION: {question}

SQL:"""

    response = await adapter.generate(prompt)

    sql = response.content.strip()
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]

    return sql.strip()


async def generate_with_retrieval(
    question: str,
    schema_store: SchemaStore,
    retriever,
) -> str:
    """Generate SQL with schema AND retrieved examples."""
    adapter = create_adapter()

    # Get relevant schema
    relevant_tables = schema_store.search_tables(question)
    if not relevant_tables:
        relevant_tables = schema_store.get_all_tables()[:10]

    # Get relevant examples
    examples = retriever.retrieve(question, top_k=3)

    schema_text = render_schema(relevant_tables)
    examples_text = render_examples(examples)

    prompt = f"""You are a SQL expert. Generate a SQL query for the given question.
Use only the tables and columns described in the schema below.
Study the examples to understand query patterns.
Return only the SQL query, no explanation or markdown.

DATABASE SCHEMA:
{schema_text}

{examples_text}

QUESTION: {question}

SQL:"""

    response = await adapter.generate(prompt)

    sql = response.content.strip()
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]

    return sql.strip()


async def execute_sql(sql: str) -> Any:
    """Execute SQL against the Chinook database."""
    settings = load_config()

    conn = await asyncpg.connect(settings.database.url)
    try:
        await conn.execute(f"SET statement_timeout = '{settings.safety.query_timeout_seconds}s'")
        result = await conn.fetch(sql)
        return [dict(row) for row in result]
    finally:
        await conn.close()


async def main():
    """Run comprehensive retrieval evaluation."""
    print("=" * 70)
    print("2.3 Retrieval - Comprehensive Evaluation")
    print("=" * 70)
    print()

    settings = load_config()
    print(f"Provider: {settings.llm.provider}")
    print(f"Model: {settings.llm.model}")
    print()

    # Load schema
    schema_path = project_root / "config" / "chinook_schema.json"
    if not schema_path.exists():
        print(f"Schema not found: {schema_path}")
        print("Run: python scripts/generate_chinook_schema.py")
        return

    schema_store = SchemaStore(schema_path)
    print(f"Loaded schema: {len(schema_store)} tables")

    # Example library summary
    summary = get_example_summary()
    print(f"Example library: {summary['total']} examples")
    print(f"  By category: {summary['by_category']}")
    print()

    # =========================================================================
    # PART 1: RETRIEVAL QUALITY EVALUATION
    # =========================================================================
    print("=" * 70)
    print("PART 1: RETRIEVAL QUALITY METRICS")
    print("=" * 70)
    print()
    print("Evaluating how well retrieval finds relevant examples.")
    print("Relevant = example uses at least one table the query needs.")
    print()

    # Create retrievers for comparison
    try:
        hybrid_retriever = HybridRetriever(CHINOOK_EXAMPLES)
        simple_retriever = SimpleRetriever(CHINOOK_EXAMPLES)
        has_hybrid = True
        print("Created HybridRetriever (BM25 + Semantic) and SimpleRetriever (BM25 only)")
    except ImportError as e:
        print(f"HybridRetriever not available: {e}")
        print("Using SimpleRetriever only")
        simple_retriever = SimpleRetriever(CHINOOK_EXAMPLES)
        hybrid_retriever = simple_retriever
        has_hybrid = False
    print()

    # Evaluate retrieval quality
    print("Evaluating Hybrid Retrieval Quality...")
    print("-" * 50)
    retrieval_metrics = evaluate_retrieval(
        GOLDEN_SET,
        hybrid_retriever,
        k=5,
        verbose=True,
    )

    print()
    print(format_retrieval_report(retrieval_metrics, k=5))

    # Compare retrieval methods if hybrid is available
    if has_hybrid:
        print()
        print("Comparing Retrieval Methods...")
        print("-" * 50)

        # Create BM25-only and Semantic-only retrievers for comparison
        from src.retrieval.retriever import HybridRetriever as HR

        bm25_only = HR(CHINOOK_EXAMPLES, bm25_weight=1.0, semantic_weight=0.0)
        semantic_only = HR(CHINOOK_EXAMPLES, bm25_weight=0.0, semantic_weight=1.0)

        retrievers = {
            "BM25 Only": bm25_only,
            "Semantic Only": semantic_only,
            "Hybrid (RRF)": hybrid_retriever,
        }

        comparison = compare_retrieval_methods(GOLDEN_SET, retrievers, k=5)

        print()
        print("Method Comparison")
        print("-" * 50)
        print(f"{'Method':<20} {'MRR':>8} {'P@5':>8} {'Coverage':>10}")
        print("-" * 50)
        for name, metrics in comparison.items():
            print(f"{name:<20} {metrics['mrr_combined']:>8.3f} {metrics['precision_at_5']:>8.3f} {metrics['coverage']:>10.1%}")
        print("-" * 50)

    # =========================================================================
    # PART 2: END-TO-END SQL GENERATION
    # =========================================================================
    print()
    print("=" * 70)
    print("PART 2: SQL GENERATION WITH RETRIEVAL")
    print("=" * 70)
    print()
    print("Comparing SQL accuracy: schema-only vs. with retrieved examples.")
    print()

    # Run schema-only evaluation
    print("Running Schema Only (no examples)...")
    print("-" * 50)

    async def schema_only_fn(question: str) -> str:
        return await generate_with_schema_only(question, schema_store)

    schema_metrics = await run_evaluation(
        GOLDEN_SET,
        schema_only_fn,
        execute_sql,
        verbose=True,
    )

    print()
    print(format_metrics_report(schema_metrics))

    # Run evaluation with retrieval
    print()
    print("Running With Retrieved Examples...")
    print("-" * 50)

    async def retrieval_fn(question: str) -> str:
        return await generate_with_retrieval(question, schema_store, hybrid_retriever)

    retrieval_sql_metrics = await run_evaluation(
        GOLDEN_SET,
        retrieval_fn,
        execute_sql,
        verbose=True,
    )

    print()
    print(format_metrics_report(retrieval_sql_metrics))

    # Show comparison
    print()
    print(compare_metrics(schema_metrics, retrieval_sql_metrics))

    # =========================================================================
    # PART 3: ANALYSIS AND INSIGHTS
    # =========================================================================
    print()
    print("=" * 70)
    print("PART 3: ANALYSIS")
    print("=" * 70)
    print()

    schema_acc = schema_metrics["result_accuracy"]
    retrieval_acc = retrieval_sql_metrics["result_accuracy"]
    improvement = retrieval_acc - schema_acc
    mrr = retrieval_metrics["mrr_combined"]

    print("Correlation Analysis")
    print("-" * 50)
    print(f"  Retrieval MRR:          {mrr:.3f}")
    print(f"  SQL Accuracy (schema):  {schema_acc:.1%}")
    print(f"  SQL Accuracy (w/retr):  {retrieval_acc:.1%}")
    print(f"  Improvement:            {improvement:+.1%}")
    print()

    if mrr >= 0.7 and improvement > 0.10:
        print("  ✓ Strong correlation: Good retrieval → Good SQL improvement")
    elif mrr >= 0.7 and improvement <= 0.05:
        print("  ⚠ Retrieval is good but SQL improvement is modest")
        print("    Consider: More diverse examples, better example quality")
    elif mrr < 0.5 and improvement <= 0:
        print("  ✗ Poor retrieval → No SQL improvement")
        print("    Fix retrieval first: expand example library, tune weights")
    else:
        print("  Mixed results. Analyze per-category performance.")

    # Sample retrievals with detailed analysis
    print()
    print("Sample Retrievals (Detailed)")
    print("-" * 50)

    sample_questions = [
        ("How many customers are there?", ["customer"]),
        ("What is the total revenue by country?", ["invoice", "customer"]),
        ("Which artists have the most albums?", ["artist", "album"]),
    ]

    for q, expected_tables in sample_questions:
        examples = hybrid_retriever.retrieve(q, top_k=3)
        print(f"\n  Q: {q}")
        print(f"  Expected tables: {expected_tables}")
        for i, ex in enumerate(examples, 1):
            overlap = set(ex.tables_used) & set(expected_tables)
            status = "✓" if overlap else "✗"
            print(f"    {i}. [{status}] {ex.question[:45]}...")
            print(f"       Tables: {ex.tables_used}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        "retrieval_quality": {
            "mrr": retrieval_metrics["mrr_combined"],
            "precision_at_5": retrieval_metrics["precision_at_5"],
            "coverage": retrieval_metrics["coverage"],
        },
        "sql_generation": {
            "schema_only": {k: v for k, v in schema_metrics.items() if k != "results"},
            "with_retrieval": {k: v for k, v in retrieval_sql_metrics.items() if k != "results"},
            "improvement": improvement,
        },
    }

    if has_hybrid:
        # Transform mrr_combined to mrr for consistent JSON output
        results["method_comparison"] = {
            name: {
                "mrr": metrics["mrr_combined"],
                "precision_at_5": metrics["precision_at_5"],
                "coverage": metrics["coverage"],
            }
            for name, metrics in comparison.items()
        }

    results_path = project_root / "evals" / "chapter_2_3_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {results_path}")

    # Final summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Retrieval MRR:     {mrr:.3f} {'✓' if mrr >= 0.7 else '⚠' if mrr >= 0.5 else '✗'}")
    print(f"  SQL Improvement:   {improvement:+.1%} {'✓' if improvement > 0.05 else '⚠' if improvement > 0 else '✗'}")
    print()
    print("Next steps:")
    if mrr < 0.7:
        print("  - Expand example library (target 50+ examples)")
        print("  - Tune BM25/Semantic weights based on query patterns")
    if improvement < 0.10:
        print("  - Add examples for failing categories")
        print("  - Consider LLM reranking for top candidates")
    print("  - Proceed to 2.4 Structured Generation")


if __name__ == "__main__":
    asyncio.run(main())
