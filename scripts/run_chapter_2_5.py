#!/usr/bin/env python3
"""2.5 Reasoning Patterns - Evaluation.

This script compares different reasoning methods for SQL generation:
- Direct: No explicit reasoning
- CoT: Chain-of-thought with four phases
- Agentic CoT: CoT with error recovery
- Reasoning Model: o1/o3 (if available)

Expected results:
- Direct: Fastest, ~85% accuracy on simple queries
- CoT: +10-15% accuracy, 1.5x latency
- Agentic CoT: +15-20% accuracy, 2.5x latency
- Reasoning Model: +20-25% accuracy, 7x latency

Usage:
    # Make sure database is running
    make db-up

    # Run comparison
    python scripts/run_chapter_2_5.py

    # Test specific method
    python scripts/run_chapter_2_5.py --method cot

    # Quick test with fewer questions
    python scripts/run_chapter_2_5.py --quick
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import asyncpg

from src.reasoning import (
    ReasoningMethod,
    ReasoningResult,
    generate_sql_with_reasoning,
    select_reasoning_method,
    get_method_characteristics,
)
from src.schema import SchemaStore
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


async def evaluate_method(
    method: ReasoningMethod,
    schema_store: SchemaStore,
    questions: list[dict],
    verbose: bool = False,
) -> dict:
    """Evaluate a single reasoning method on the question set."""
    results = []

    for item in questions:
        question = item.question if hasattr(item, "question") else item["question"]
        item_id = item.id if hasattr(item, "id") else item.get("id", "unknown")

        result = {
            "id": item_id,
            "question": question,
            "method": method.value,
        }

        start_time = time.time()

        try:
            reasoning_result = await generate_sql_with_reasoning(
                question=question,
                schema_store=schema_store,
                method=method,
            )

            result["latency_ms"] = (time.time() - start_time) * 1000
            result["sql"] = reasoning_result.sql
            result["confidence"] = reasoning_result.confidence
            result["has_reasoning"] = bool(reasoning_result.reasoning)
            result["has_verification"] = bool(reasoning_result.verification)

            # Try to execute
            if reasoning_result.sql:
                try:
                    query_result = await execute_sql(reasoning_result.sql)
                    result["execute_success"] = True
                    result["row_count"] = len(query_result)
                except Exception as e:
                    result["execute_success"] = False
                    result["execute_error"] = str(e)[:200]
            else:
                result["execute_success"] = False
                result["execute_error"] = "No SQL generated"

        except Exception as e:
            result["latency_ms"] = (time.time() - start_time) * 1000
            result["error"] = str(e)[:200]
            result["execute_success"] = False

        results.append(result)

        if verbose:
            status = "OK" if result.get("execute_success") else "FAIL"
            latency = result.get("latency_ms", 0)
            conf = result.get("confidence", 0)
            print(f"  [{item_id}] {status} | {latency:.0f}ms | conf={conf:.2f}")

    # Calculate aggregate metrics
    total = len(results)
    execute_success = sum(1 for r in results if r.get("execute_success", False))
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    confidences = [r["confidence"] for r in results if "confidence" in r]

    return {
        "method": method.value,
        "total": total,
        "execute_success_rate": execute_success / total if total else 0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else 0,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "results": results,
    }


async def compare_methods(
    schema_store: SchemaStore,
    questions: list,
    methods: list[ReasoningMethod] | None = None,
    verbose: bool = False,
) -> dict:
    """Compare multiple reasoning methods."""
    if methods is None:
        # Skip reasoning model by default (too slow for comparison)
        methods = [
            ReasoningMethod.DIRECT,
            ReasoningMethod.COT,
            ReasoningMethod.AGENTIC_COT,
        ]

    all_results = {}

    for method in methods:
        print(f"\nEvaluating {method.value}...")
        print("-" * 40)
        method_results = await evaluate_method(
            method=method,
            schema_store=schema_store,
            questions=questions,
            verbose=verbose,
        )
        all_results[method.value] = method_results

    return all_results


def print_comparison(results: dict):
    """Print a comparison table of results."""
    print("\n" + "=" * 70)
    print("Reasoning Methods Comparison")
    print("=" * 70)

    # Header
    print(f"\n{'Method':<15} {'Exec Success':>12} {'Avg Latency':>12} {'P95 Latency':>12} {'Avg Conf':>10}")
    print("-" * 70)

    # Data rows
    for method, data in results.items():
        exec_rate = data["execute_success_rate"]
        avg_lat = data["avg_latency_ms"]
        p95_lat = data["p95_latency_ms"]
        avg_conf = data["avg_confidence"]

        print(f"{method:<15} {exec_rate:>11.1%} {avg_lat:>11.0f}ms {p95_lat:>11.0f}ms {avg_conf:>9.2f}")

    # Calculate improvements relative to direct
    print("\n" + "-" * 70)
    print("Improvement vs Direct:")
    print("-" * 70)

    if "direct" in results:
        direct_exec = results["direct"]["execute_success_rate"]
        direct_lat = results["direct"]["avg_latency_ms"]

        for method, data in results.items():
            if method == "direct":
                continue

            exec_diff = data["execute_success_rate"] - direct_exec
            lat_mult = data["avg_latency_ms"] / direct_lat if direct_lat > 0 else 0

            print(f"{method:<15} Exec: {exec_diff:>+6.1%} | Latency: {lat_mult:.1f}x")


def demonstrate_method_selection(schema_store: SchemaStore):
    """Demonstrate automatic method selection."""
    print("\n" + "=" * 70)
    print("Method Selection Demonstration")
    print("=" * 70)

    test_questions = [
        ("How many customers?", 1),
        ("List all genres", 1),
        ("Show orders for customer Smith", 2),
        ("Revenue by country by month", 3),
        ("Top 5 customers by spending", 3),
        ("Artists with most tracks in metal genre", 4),
        ("Year-over-year revenue growth by genre", 5),
    ]

    print(f"\n{'Question':<45} {'Tables':>7} {'Method':<15}")
    print("-" * 70)

    for question, expected_tables in test_questions:
        method = select_reasoning_method(
            question=question,
            schema_complexity=expected_tables,
        )
        chars = get_method_characteristics(method)

        print(f"{question[:44]:<45} {expected_tables:>7} {method.value:<15}")


async def main():
    """Run reasoning patterns evaluation."""
    parser = argparse.ArgumentParser(description="2.5 Reasoning Patterns")
    parser.add_argument(
        "--method",
        choices=["direct", "cot", "agentic_cot", "reasoning_model"],
        help="Test specific method only",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer questions",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demonstrate method selection only",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("2.5 Reasoning Patterns - Evaluation")
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

    if args.demo:
        demonstrate_method_selection(schema_store)
        return

    # Prepare questions
    questions = GOLDEN_SET[:5] if args.quick else GOLDEN_SET
    print(f"Evaluating on {len(questions)} questions")

    if args.method:
        # Single method evaluation
        method = ReasoningMethod(args.method)
        print(f"\nEvaluating {method.value} only...")
        print("-" * 40)

        results = await evaluate_method(
            method=method,
            schema_store=schema_store,
            questions=questions,
            verbose=args.verbose,
        )

        print("\nResults Summary:")
        print(f"  Execute Success: {results['execute_success_rate']:.1%}")
        print(f"  Avg Latency: {results['avg_latency_ms']:.0f}ms")
        print(f"  Avg Confidence: {results['avg_confidence']:.2f}")

    else:
        # Compare all methods
        results = await compare_methods(
            schema_store=schema_store,
            questions=questions,
            verbose=args.verbose,
        )

        print_comparison(results)

        # Also show method selection demo
        demonstrate_method_selection(schema_store)

    # Save results
    results_path = project_root / "evals" / "chapter_2_5_results.json"
    results_to_save = {
        k: {kk: vv for kk, vv in v.items() if kk != "results"}
        for k, v in (results if isinstance(results, dict) and "method" not in results else {"single": results}).items()
    }

    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
