#!/usr/bin/env python3
"""2.1 Measuring Before Building - Baseline Evaluation.

This script establishes a baseline measurement for Text-to-SQL accuracy
using the simplest possible approach: just ask the LLM to generate SQL
without any schema context, examples, or validation.

The purpose is to measure the "floor" - what accuracy do we get with
minimal investment? Every improvement in later chapters should measurably
beat this baseline.

Expected results:
- Execution Accuracy: 60-80% (many queries fail on non-existent columns)
- Result Accuracy: 40-60% (even successful queries often return wrong data)
- Easy queries: Higher accuracy (simple patterns in training data)
- Hard queries: Lower accuracy (complex reasoning without context)

Usage:
    # Make sure database is running
    make db-up

    # Run baseline evaluation
    python scripts/run_chapter_2_1.py

    # Run with specific provider
    LLM_PROVIDER=gemini python scripts/run_chapter_2_1.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.adapters import create_adapter
from src.utils.config import load_config
from evals.runner import run_evaluation, print_failures
from evals.metrics import format_metrics_report
from evals.chinook_golden_set import GOLDEN_SET, get_summary


# Minimal schema: just table names, no columns or relationships
BASELINE_SCHEMA = """
Tables: album, artist, customer, employee, genre, invoice, invoice_line,
        media_type, playlist, playlist_track, track
"""


async def baseline_generate(question: str) -> str:
    """Baseline SQL generation with minimal schema context.

    Provides only table names - enough for executable SQL, but no column
    info, relationships, or examples. This establishes a meaningful floor
    that Schema Cards (2.2) and retrieval (2.3) can improve upon.
    """
    adapter = create_adapter()

    prompt = f"""Generate a SQL query for this question.
Return only the SQL query, no explanation or markdown.

{BASELINE_SCHEMA}

Question: {question}

SQL:"""

    response = await adapter.generate(prompt)
    return response.content.strip()


async def execute_sql(sql: str) -> Any:
    """Execute SQL against the Chinook database.

    Uses asyncpg for PostgreSQL. The Chinook database should be
    running (use `make db-up` to start it).
    """
    import asyncpg

    settings = load_config()

    conn = await asyncpg.connect(settings.database.url)
    try:
        # Set a statement timeout to prevent long-running queries
        await conn.execute(f"SET statement_timeout = '{settings.safety.query_timeout_seconds}s'")

        # Execute and fetch results
        result = await conn.fetch(sql)

        # Convert to list of dicts for easier comparison
        return [dict(row) for row in result]
    finally:
        await conn.close()


async def main():
    """Run the baseline evaluation and print results."""
    print("=" * 60)
    print("2.1 Baseline Evaluation")
    print("=" * 60)
    print()

    # Show golden set summary
    summary = get_summary()
    print("Golden Set Summary")
    print("-" * 30)
    print(f"  Total queries: {summary['total']}")
    print(f"  By category: {summary['by_category']}")
    print(f"  By difficulty: {summary['by_difficulty']}")
    print()

    # Check adapter configuration
    settings = load_config()
    print("Configuration")
    print("-" * 30)
    print(f"  Provider: {settings.llm.provider}")
    print(f"  Model: {settings.llm.model}")
    print(f"  Database: {settings.database.url.split('@')[-1]}")  # Hide credentials
    print()

    # Verify database connection
    print("Verifying database connection...")
    try:
        import asyncpg
        conn = await asyncpg.connect(settings.database.url)
        await conn.close()
        print("  ✓ Database connected")
    except Exception as e:
        print(f"  ✗ Database connection failed: {e}")
        print("\n  Run 'make db-up' to start the database.")
        return

    # Run evaluation
    print()
    print("Running Evaluation")
    print("-" * 30)

    metrics = await run_evaluation(
        GOLDEN_SET,
        baseline_generate,
        execute_sql,
        verbose=True,
    )

    # Print report
    print()
    print(format_metrics_report(metrics))

    # Show some failures for debugging
    results = metrics.get("results", [])
    failures = [r for r in results if not r.result_matches]

    if failures:
        print_failures(results, limit=5)

    # Summary interpretation
    print()
    print("Interpretation")
    print("-" * 30)

    result_acc = metrics["result_accuracy"]
    if result_acc < 0.5:
        print("  ⚠ Low accuracy (<50%) - expected without schema context")
        print("  → Schema representation (2.2) should improve this significantly")
    elif result_acc < 0.7:
        print("  ⚠ Moderate accuracy (50-70%) - room for improvement")
        print("  → Few-shot examples (2.3) should help with complex queries")
    else:
        print("  ✓ Surprisingly high baseline - simple queries dominate")
        print("  → Focus on hard queries for further improvement")

    exec_acc = metrics["execution_accuracy"]
    if exec_acc < result_acc * 1.2:
        print("  ℹ Execution and result accuracy are close")
        print("  → Validation (3.3) can catch more errors before execution")

    # Save results for comparison
    results_path = project_root / "evals" / "baseline_results.json"
    try:
        import json

        # Convert results to serializable format
        serializable_metrics = {k: v for k, v in metrics.items() if k != "results"}
        serializable_metrics["failures"] = [
            {
                "query_id": r.query_id,
                "question": r.question,
                "generated_sql": r.generated_sql,
                "execution_error": r.execution_error,
            }
            for r in failures[:20]  # Save first 20 failures
        ]

        with open(results_path, "w") as f:
            json.dump(serializable_metrics, f, indent=2)

        print()
        print(f"Results saved to: {results_path}")
    except Exception as e:
        print(f"  Could not save results: {e}")


if __name__ == "__main__":
    asyncio.run(main())
