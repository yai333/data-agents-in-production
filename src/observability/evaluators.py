"""Evaluation bridge: connects offline eval harness with Langfuse scoring.

This module bridges two subsystems:
- Offline evaluation (evals/sql_components.py, evals/runner.py)
- Online scoring (src/observability/annotation.py, Langfuse SDK)

Usage:
    from src.observability.evaluators import (
        run_component_eval,
        run_execution_eval,
        score_eval_results,
        upload_golden_set_to_dataset,
    )

    # Offline: compare SQL components
    component_result = run_component_eval(generated_sql, reference_sql)

    # Online: push scores to Langfuse
    score_eval_results(trace_id, component_result)
"""

from typing import Any, Callable, Awaitable

from langfuse.api import CreateDatasetRunItemRequest
from evals.sql_components import compare_sql_components, ComponentMatchResult
from evals.runner import evaluate_query, EvalResult, compare_results
from evals.golden_set import GoldenQuery
from src.observability.annotation import score_trace
from src.observability.tracing import get_langfuse_client


def run_component_eval(
    generated_sql: str,
    reference_sql: str,
) -> ComponentMatchResult:
    """Run Spider-style component matching between two SQL queries.

    Extracts SELECT, FROM, WHERE, GROUP BY, ORDER BY components
    and computes per-component F1 scores plus an overall score.

    Args:
        generated_sql: The SQL produced by the agent
        reference_sql: The golden reference SQL

    Returns:
        ComponentMatchResult with per-component and overall F1 scores
    """
    return compare_sql_components(generated_sql, reference_sql)


async def run_execution_eval(
    query: GoldenQuery,
    generated_sql: str,
    execute_fn: Callable[[str], Awaitable[Any]],
) -> dict[str, Any]:
    """Run execution-based evaluation: execute SQL and compare results.

    Args:
        query: The golden query with expected results
        generated_sql: The SQL produced by the agent
        execute_fn: Async function that executes SQL against the database

    Returns:
        Dict with executed (bool), result_matches (bool), error (str|None)
    """
    try:
        actual_result = await execute_fn(generated_sql)
        executed = True
        error = None
    except Exception as e:
        actual_result = None
        executed = False
        error = str(e)

    result_matches = False
    if executed and actual_result is not None:
        result_matches = compare_results(actual_result, query.expected_result)

    return {
        "executed": executed,
        "result_matches": result_matches,
        "error": error,
        "actual_result": actual_result,
    }


def score_eval_results(
    trace_id: str,
    component_result: ComponentMatchResult | None = None,
    execution_result: dict[str, Any] | None = None,
) -> list[str]:
    """Push evaluation scores to Langfuse as trace scores.

    Attaches both component-level and execution-level scores to a trace,
    making them visible in the Langfuse dashboard for filtering and analysis.

    Args:
        trace_id: The Langfuse trace ID
        component_result: Result from run_component_eval()
        execution_result: Result from run_execution_eval()

    Returns:
        List of score IDs created
    """
    score_ids = []

    if component_result is not None:
        # Overall component F1
        sid = score_trace(
            trace_id, "component_f1", component_result.overall_f1,
            comment=f"Exact match: {component_result.exact_match}",
        )
        score_ids.append(sid)

        # Per-component scores for drill-down
        sid = score_trace(trace_id, "select_f1", component_result.select_f1)
        score_ids.append(sid)
        sid = score_trace(trace_id, "from_f1", component_result.from_f1)
        score_ids.append(sid)
        sid = score_trace(trace_id, "where_f1", component_result.where_f1)
        score_ids.append(sid)

    if execution_result is not None:
        sid = score_trace(
            trace_id, "execution_success",
            1.0 if execution_result["executed"] else 0.0,
        )
        score_ids.append(sid)
        sid = score_trace(
            trace_id, "result_accuracy",
            1.0 if execution_result["result_matches"] else 0.0,
        )
        score_ids.append(sid)

    return score_ids


def upload_golden_set_to_dataset(
    dataset_name: str,
    golden_set: list[GoldenQuery],
    description: str | None = None,
) -> dict[str, Any]:
    """Upload golden set queries to a Langfuse dataset.

    Creates a dataset that can be used for Langfuse experiments:
    each item has the question as input and reference SQL as expected output.

    Args:
        dataset_name: Name for the Langfuse dataset
        golden_set: List of GoldenQuery items to upload
        description: Optional dataset description

    Returns:
        Dict with dataset_name, item_count
    """
    client = get_langfuse_client()

    dataset = client.create_dataset(
        name=dataset_name,
        description=description or f"Golden set with {len(golden_set)} queries",
    )

    count = 0
    for query in golden_set:
        # Skip negative test cases (no reference SQL)
        if not query.sql:
            continue

        client.create_dataset_item(
            dataset_name=dataset_name,
            input={"question": query.question},
            expected_output={"sql": query.sql},
            metadata={
                "query_id": query.id,
                "category": query.category,
                "difficulty": query.difficulty,
                "tables_used": query.tables_used,
                "expected_result": str(query.expected_result),
            },
        )
        count += 1

    client.flush()

    return {
        "dataset_name": dataset_name,
        "dataset_id": dataset.id,
        "item_count": count,
    }


def link_trace_to_dataset_item(
    dataset_item_id: str,
    trace_id: str,
    run_name: str,
) -> None:
    """Link an existing trace to a dataset item, creating a dataset run item.

    This is used when traces are created through a separate callback handler
    (e.g., LangGraph's CallbackHandler) and need to be linked to dataset
    items after the fact. The linked run items trigger any dataset-mode
    evaluators configured in Langfuse.

    Args:
        dataset_item_id: The Langfuse dataset item ID
        trace_id: The trace ID to link
        run_name: Name for the experiment run (e.g., "baseline-v1")
    """
    client = get_langfuse_client()
    client.api.dataset_run_items.create(
        request=CreateDatasetRunItemRequest(
            runName=run_name,
            datasetItemId=dataset_item_id,
            traceId=trace_id,
        )
    )
