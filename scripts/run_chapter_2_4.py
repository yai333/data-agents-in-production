#!/usr/bin/env python3
"""2.4 Structured Generation - Evaluation.

This script compares SQL generation approaches:
- Free-form generation (requires regex parsing)
- Structured generation (guaranteed schema compliance)

Expected results:
- Free-form: ~96% parse success, ~87% execution success
- Structured: 100% parse success, ~89% execution success

The small latency increase is worth the reliability.

Usage:
    # Make sure database is running and schema is generated
    make db-up
    python scripts/generate_chinook_schema.py

    # Run structured evaluation only
    python scripts/run_chapter_2_4.py

    # Compare free-form vs structured
    python scripts/run_chapter_2_4.py --compare
"""

import argparse
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

from src.schema import SchemaStore
from src.structured import (
    SQLResult,
    generate_sql_structured,
    generate_sql_freeform,
)
from src.utils.config import load_config
from evals.chinook_golden_set import GOLDEN_SET


async def execute_sql(sql: str) -> Any:
    """Execute SQL against the Chinook database."""
    settings = load_config()

    conn = await asyncpg.connect(settings.database.url)
    try:
        await conn.execute(
            f"SET statement_timeout = '{settings.safety.query_timeout_seconds}s'"
        )
        result = await conn.fetch(sql)
        return [dict(row) for row in result]
    finally:
        await conn.close()


def parse_success_rate(results: list[dict]) -> float:
    """Calculate parse success rate from results."""
    successful = sum(1 for r in results if r.get("parse_success", True))
    return successful / len(results) if results else 0.0


async def evaluate_freeform(schema_store: SchemaStore) -> dict:
    """Evaluate free-form SQL generation."""
    results = []

    for item in GOLDEN_SET:
        result = {"id": item.id, "question": item.question}

        try:
            sql = await generate_sql_freeform(item.question, schema_store)
            result["parse_success"] = True
            result["sql"] = sql

            # Try to execute
            try:
                query_result = await execute_sql(sql)
                result["execute_success"] = True
                result["row_count"] = len(query_result)
            except Exception as e:
                result["execute_success"] = False
                result["execute_error"] = str(e)

        except Exception as e:
            result["parse_success"] = False
            result["parse_error"] = str(e)

        results.append(result)
        print(f"  [{item.id}] Parse: {result.get('parse_success')}, "
              f"Execute: {result.get('execute_success', 'N/A')}")

    return {
        "mode": "freeform",
        "total": len(results),
        "parse_success_rate": parse_success_rate(results),
        "execute_success_rate": sum(
            1 for r in results if r.get("execute_success", False)
        ) / len(results),
        "results": results,
    }


async def evaluate_structured(schema_store: SchemaStore) -> dict:
    """Evaluate structured SQL generation."""
    results = []

    for item in GOLDEN_SET:
        result = {"id": item.id, "question": item.question}

        try:
            sql_result = await generate_sql_structured(item.question, schema_store)
            result["parse_success"] = True  # Always true with structured output
            result["sql"] = sql_result.sql
            result["confidence"] = sql_result.confidence
            result["assumptions"] = sql_result.assumptions
            result["tables_used"] = sql_result.tables_used

            # Try to execute
            try:
                query_result = await execute_sql(sql_result.sql)
                result["execute_success"] = True
                result["row_count"] = len(query_result)
            except Exception as e:
                result["execute_success"] = False
                result["execute_error"] = str(e)

        except Exception as e:
            # This should be rare with structured output
            result["parse_success"] = False
            result["parse_error"] = str(e)

        results.append(result)
        print(f"  [{item.id}] Parse: {result.get('parse_success')}, "
              f"Execute: {result.get('execute_success', 'N/A')}, "
              f"Conf: {result.get('confidence', 'N/A')}")

    return {
        "mode": "structured",
        "total": len(results),
        "parse_success_rate": parse_success_rate(results),
        "execute_success_rate": sum(
            1 for r in results if r.get("execute_success", False)
        ) / len(results),
        "avg_confidence": sum(
            r.get("confidence", 0) for r in results if r.get("confidence")
        ) / len([r for r in results if r.get("confidence")]),
        "results": results,
    }


async def main():
    """Run structured generation evaluation."""
    parser = argparse.ArgumentParser(description="2.4 Structured Generation")
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare free-form vs structured generation"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("2.4 Structured Generation - Evaluation")
    print("=" * 60)
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
    print()

    if args.compare:
        # Compare free-form vs structured
        print("Evaluating Free-Form Generation...")
        print("-" * 40)
        freeform_results = await evaluate_freeform(schema_store)

        print()
        print("Evaluating Structured Generation...")
        print("-" * 40)
        structured_results = await evaluate_structured(schema_store)

        # Show comparison
        print()
        print("=" * 60)
        print("Comparison Results")
        print("=" * 60)
        print()

        print(f"{'Metric':<25} {'Free-Form':>15} {'Structured':>15} {'Diff':>10}")
        print("-" * 65)

        freeform_parse = freeform_results["parse_success_rate"]
        structured_parse = structured_results["parse_success_rate"]
        print(f"{'Parse Success Rate':<25} {freeform_parse:>14.1%} {structured_parse:>14.1%} {structured_parse - freeform_parse:>+9.1%}")

        freeform_exec = freeform_results["execute_success_rate"]
        structured_exec = structured_results["execute_success_rate"]
        print(f"{'Execute Success Rate':<25} {freeform_exec:>14.1%} {structured_exec:>14.1%} {structured_exec - freeform_exec:>+9.1%}")

        avg_conf = structured_results.get("avg_confidence", 0)
        print(f"{'Avg Confidence':<25} {'N/A':>15} {avg_conf:>14.2f} {'--':>10}")

        # Interpretation
        print()
        print("Interpretation")
        print("-" * 30)

        parse_improvement = structured_parse - freeform_parse
        if parse_improvement > 0.03:
            print(f"  Parse reliability improved by {parse_improvement:.1%}")
            print("  Structured output eliminates regex parsing failures")
        else:
            print("  Parse rates similar (free-form may have simple outputs)")

        exec_improvement = structured_exec - freeform_exec
        if exec_improvement > 0.01:
            print(f"  Execution success improved by {exec_improvement:.1%}")
            print("  Structured metadata (confidence, assumptions) helps generation")
        elif exec_improvement < -0.01:
            print(f"  Slight execution decrease ({exec_improvement:.1%})")
            print("  This is within variance; both modes generate similar SQL")
        else:
            print("  Execution rates similar (SQL quality is comparable)")

        # Save results
        results = {
            "freeform": {
                k: v for k, v in freeform_results.items() if k != "results"
            },
            "structured": {
                k: v for k, v in structured_results.items() if k != "results"
            },
        }

    else:
        # Just run structured evaluation
        print("Evaluating Structured Generation...")
        print("-" * 40)
        structured_results = await evaluate_structured(schema_store)

        print()
        print("Results Summary")
        print("-" * 30)
        print(f"  Parse Success: {structured_results['parse_success_rate']:.1%}")
        print(f"  Execute Success: {structured_results['execute_success_rate']:.1%}")
        print(f"  Avg Confidence: {structured_results.get('avg_confidence', 0):.2f}")

        # Confidence distribution
        confidences = [
            r["confidence"]
            for r in structured_results["results"]
            if r.get("confidence")
        ]
        if confidences:
            low = sum(1 for c in confidences if c < 0.5)
            medium = sum(1 for c in confidences if 0.5 <= c < 0.8)
            high = sum(1 for c in confidences if c >= 0.8)
            print()
            print("Confidence Distribution")
            print(f"  Low (<0.5): {low} ({low/len(confidences):.1%})")
            print(f"  Medium (0.5-0.8): {medium} ({medium/len(confidences):.1%})")
            print(f"  High (>=0.8): {high} ({high/len(confidences):.1%})")

        results = {
            "structured": {
                k: v for k, v in structured_results.items() if k != "results"
            }
        }

    # Save results
    results_path = project_root / "evals" / "chapter_2_4_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
