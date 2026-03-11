"""Evaluation runner for Text-to-SQL agents.

This module orchestrates the evaluation process:
1. Take a golden set of test queries
2. Run each through a SQL generation function
3. Execute the generated SQL
4. Compare results to expected values
5. Collect metrics for analysis

The runner is designed to work with any SQL generation approach,
from simple LLM calls to complex multi-agent systems.
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from evals.golden_set import GoldenQuery
from evals.metrics import calculate_metrics


@dataclass
class EvalResult:
    """Result of evaluating a single query.

    Captures everything needed for analysis: the generated SQL,
    whether it executed, whether results matched, timing, and errors.
    """

    query_id: str
    question: str
    generated_sql: str
    reference_sql: str
    executed: bool
    execution_error: str | None
    result_matches: bool
    latency_ms: float
    actual_result: Any = None
    expected_result: Any = None


def normalize_value(v: Any) -> str:
    """Convert any value to comparable string.

    Handles mixed types (strings, numbers, None, JSON) consistently.
    """
    if v is None:
        return "NULL"
    if isinstance(v, float):
        return f"{v:.6f}"  # Consistent float precision
    return str(v)


def compare_results(actual: Any, expected: Any) -> bool:
    """Compare query results, ignoring column names and order.

    Handles:
    - Single scalar results: [{'count': 275}] -> 275
    - Multiple rows: normalize values, sort as multiset
    - Numeric comparisons with float precision tolerance
    - Count comparisons (expected int vs. result set length)

    Args:
        actual: The result from executing the generated SQL
        expected: The expected result from the golden set

    Returns:
        True if results match within tolerance
    """
    # Handle None
    if actual is None:
        return expected is None

    # Extract values from list of dicts (database result format)
    if isinstance(actual, list) and actual and isinstance(actual[0], dict):
        # Single scalar result: [{'count': 275}] -> 275
        if len(actual) == 1 and len(actual[0]) == 1:
            actual = list(actual[0].values())[0]
        else:
            # Multiple rows: normalize values, sort as multiset
            actual = sorted(
                tuple(sorted(normalize_value(v) for v in row.values()))
                for row in actual
            )

    # Exact match
    if actual == expected:
        return True

    # Numeric comparison (handles float precision issues)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        # Allow 1% tolerance for floats
        if expected == 0:
            return abs(actual) < 0.01
        return abs(actual - expected) / abs(expected) < 0.01

    # Row count comparison
    if isinstance(expected, int) and isinstance(actual, list):
        return len(actual) == expected

    return False


async def evaluate_query(
    query: GoldenQuery,
    generate_fn: Callable[[str], Awaitable[str]],
    execute_fn: Callable[[str], Awaitable[Any]],
) -> EvalResult:
    """Evaluate a single query against the golden set.

    Args:
        query: The golden query test case
        generate_fn: Async function that generates SQL from a question
        execute_fn: Async function that executes SQL and returns results

    Returns:
        EvalResult with all evaluation details
    """
    start = time.perf_counter()

    # Step 1: Generate SQL
    try:
        generated_sql = await generate_fn(query.question)
        generated_sql = generated_sql.strip()
    except Exception as e:
        return EvalResult(
            query_id=query.id,
            question=query.question,
            generated_sql="",
            reference_sql=query.sql,
            executed=False,
            execution_error=f"Generation failed: {e}",
            result_matches=False,
            latency_ms=(time.perf_counter() - start) * 1000,
            expected_result=query.expected_result,
        )

    # Step 2: Execute SQL
    try:
        result = await execute_fn(generated_sql)
        executed = True
        execution_error = None
    except Exception as e:
        result = None
        executed = False
        execution_error = str(e)

    # Step 3: Compare results
    result_matches = False
    if executed and result is not None:
        result_matches = compare_results(result, query.expected_result)

    return EvalResult(
        query_id=query.id,
        question=query.question,
        generated_sql=generated_sql,
        reference_sql=query.sql,
        executed=executed,
        execution_error=execution_error,
        result_matches=result_matches,
        latency_ms=(time.perf_counter() - start) * 1000,
        actual_result=result,
        expected_result=query.expected_result,
    )


async def run_evaluation(
    golden_set: list[GoldenQuery],
    generate_fn: Callable[[str], Awaitable[str]],
    execute_fn: Callable[[str], Awaitable[Any]],
    verbose: bool = True,
) -> dict:
    """Run full evaluation and return metrics.

    Evaluates each query in the golden set sequentially,
    printing progress if verbose mode is enabled.

    Args:
        golden_set: List of test queries
        generate_fn: Async function to generate SQL
        execute_fn: Async function to execute SQL
        verbose: Whether to print progress

    Returns:
        Dictionary of metrics including accuracy, latency, and slice breakdowns
    """
    results = []

    for i, query in enumerate(golden_set, 1):
        result = await evaluate_query(query, generate_fn, execute_fn)
        results.append(result)

        if verbose:
            status = "✓" if result.result_matches else "✗"
            exec_status = "" if result.executed else " (failed to execute)"
            print(f"  [{i}/{len(golden_set)}] {query.id}: {status}{exec_status}")

    return calculate_metrics(results, golden_set)


async def run_evaluation_parallel(
    golden_set: list[GoldenQuery],
    generate_fn: Callable[[str], Awaitable[str]],
    execute_fn: Callable[[str], Awaitable[Any]],
    concurrency: int = 5,
) -> dict:
    """Run evaluation with parallel execution.

    Useful for faster evaluation when the database can handle
    concurrent connections. Be mindful of rate limits.

    Args:
        golden_set: List of test queries
        generate_fn: Async function to generate SQL
        execute_fn: Async function to execute SQL
        concurrency: Maximum parallel evaluations

    Returns:
        Dictionary of metrics
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_evaluate(query: GoldenQuery) -> EvalResult:
        async with semaphore:
            return await evaluate_query(query, generate_fn, execute_fn)

    tasks = [bounded_evaluate(q) for q in golden_set]
    results = await asyncio.gather(*tasks)

    return calculate_metrics(list(results), golden_set)


def print_failures(results: list[EvalResult], limit: int = 10) -> None:
    """Print details of failed queries for debugging.

    Args:
        results: List of evaluation results
        limit: Maximum number of failures to print
    """
    failures = [r for r in results if not r.result_matches]

    print(f"\n{'='*60}")
    print(f"Failed Queries ({len(failures)} total, showing first {limit})")
    print("=" * 60)

    for result in failures[:limit]:
        print(f"\n[{result.query_id}] {result.question}")
        print(f"  Expected: {result.expected_result}")
        print(f"  Actual:   {result.actual_result}")
        print(f"  Generated SQL:")
        for line in result.generated_sql.split('\n'):
            print(f"    {line}")
        if result.execution_error:
            print(f"  Error: {result.execution_error}")
