"""Dataset export from annotated traces.

Two export paths:
1. Good traces (CORRECT) → fine-tuning dataset for Chapter 4
2. Bad traces (INCORRECT) → MDL revision candidates for Chapter 3.3

Langfuse v3 removed the Python SDK's fetch_traces/fetch_scores methods.
We use the Langfuse REST API directly for trace retrieval.

Usage:
    from src.observability.datasets import (
        export_good_traces_to_dataset,
        export_bad_traces_for_mdl_review,
    )

    # Export correct traces to a Langfuse dataset
    dataset = export_good_traces_to_dataset("finetune-v1")

    # Export incorrect traces for MDL review
    issues = export_bad_traces_for_mdl_review()
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from src.observability.tracing import get_langfuse_client


@dataclass
class FineTuningExample:
    """A single fine-tuning example extracted from a trace."""
    question: str
    sql: str
    tables_used: list[str]
    trace_id: str
    annotator: str | None = None


@dataclass
class MDLIssue:
    """An MDL revision candidate from an incorrect trace."""
    trace_id: str
    question: str
    generated_sql: str
    corrected_sql: str | None
    error_description: str | None
    tables_involved: list[str] = field(default_factory=list)
    suggested_fix: str | None = None


def _langfuse_api_get(path: str, params: dict | None = None) -> dict:
    """Call the Langfuse REST API.

    Langfuse v3 removed fetch_traces/fetch_scores from the Python SDK.
    We use the REST API directly for trace retrieval and filtering.
    """
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")

    resp = httpx.get(
        f"{host}/api/public{path}",
        params=params or {},
        auth=(public_key, secret_key),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def export_good_traces_to_dataset(
    dataset_name: str,
    description: str | None = None,
    min_score: float = 1.0,
    limit: int = 500,
) -> dict[str, Any]:
    """Export correctly-annotated traces to a Langfuse dataset.

    This creates a dataset that can be used for:
    - Fine-tuning a smaller model (Chapter 4)
    - Evaluation regression testing (Chapter 3.5)
    - Few-shot example curation (Chapter 2.3 feedback loop)

    Args:
        dataset_name: Name for the dataset (e.g., "finetune-v1")
        description: Optional description
        min_score: Minimum annotation score (1.0 = only CORRECT)
        limit: Max traces to export

    Returns:
        Dict with dataset_id, item_count, and examples
    """
    client = get_langfuse_client()

    # Create or get the dataset
    dataset = client.create_dataset(
        name=dataset_name,
        description=description or (
            f"Fine-tuning dataset from annotated traces. "
            f"Exported {datetime.now().isoformat()}"
        ),
    )

    # Fetch traces via REST API (removed from SDK in v3)
    traces_resp = _langfuse_api_get("/traces", {"limit": limit, "orderBy": "timestamp"})
    traces = traces_resp.get("data", [])

    examples = []
    for trace in traces:
        trace_id = trace.get("id", "")

        # Fetch scores for this trace
        scores_resp = _langfuse_api_get("/scores", {"traceId": trace_id})
        scores = scores_resp.get("data", [])

        annotation_score = None
        for score in scores:
            if score.get("name") == "human_annotation":
                annotation_score = score.get("value")
                break

        if annotation_score is None or annotation_score < min_score:
            continue

        # Extract question and SQL from trace
        question = trace.get("input")
        sql = trace.get("output")
        metadata = trace.get("metadata") or {}

        if not question or not sql:
            question = metadata.get("question", "")
            sql = metadata.get("sql", "")

        if not question or not sql:
            continue

        # Add to dataset
        client.create_dataset_item(
            dataset_name=dataset_name,
            input={"question": question},
            expected_output={"sql": sql},
            source_trace_id=trace_id,
            metadata={
                "trace_id": trace_id,
                "annotation_score": annotation_score,
                "tables_used": metadata.get("tables_used", []),
            },
        )

        examples.append(FineTuningExample(
            question=str(question),
            sql=str(sql),
            tables_used=metadata.get("tables_used", []),
            trace_id=trace_id,
        ))

    client.flush()

    return {
        "dataset_id": dataset.id,
        "dataset_name": dataset_name,
        "item_count": len(examples),
        "examples": examples,
    }


def export_bad_traces_for_mdl_review(
    max_score: float = 0.0,
    limit: int = 100,
) -> list[MDLIssue]:
    """Export incorrectly-annotated traces for MDL knowledge revision.

    These traces reveal gaps in the MDL that cause SQL generation errors:
    - Missing glossary terms → add to global_knowledge.json
    - Wrong column descriptions → update per-schema MDL files
    - Missing business rules → add to institutional knowledge
    - Missing relationships → update table relationships

    Args:
        max_score: Maximum annotation score (0.0 = only INCORRECT)
        limit: Max traces to export

    Returns:
        List of MDLIssue objects for review
    """
    # Fetch traces via REST API
    traces_resp = _langfuse_api_get("/traces", {"limit": limit, "orderBy": "timestamp"})
    traces = traces_resp.get("data", [])

    issues = []
    for trace in traces:
        trace_id = trace.get("id", "")

        # Fetch scores for this trace
        scores_resp = _langfuse_api_get("/scores", {"traceId": trace_id})
        scores = scores_resp.get("data", [])

        annotation_score = None
        annotation_comment = None
        for score in scores:
            if score.get("name") == "human_annotation":
                annotation_score = score.get("value")
                annotation_comment = score.get("comment")
                break

        if annotation_score is None or annotation_score > max_score:
            continue

        metadata = trace.get("metadata") or {}

        # Parse correction info from comment
        corrected_sql = None
        if annotation_comment and "Corrected SQL:" in annotation_comment:
            parts = annotation_comment.split("Corrected SQL:")
            corrected_sql = parts[-1].strip()

        question = trace.get("input") or metadata.get("question", "")
        generated_sql = trace.get("output") or metadata.get("sql", "")

        issues.append(MDLIssue(
            trace_id=trace_id,
            question=str(question),
            generated_sql=str(generated_sql),
            corrected_sql=corrected_sql,
            error_description=annotation_comment,
            tables_involved=metadata.get("tables_used", []),
        ))

    return issues


def export_dataset_to_jsonl(
    dataset_name: str,
    output_path: str,
    format: str = "openai",
) -> int:
    """Export a Langfuse dataset to JSONL for fine-tuning.

    Args:
        dataset_name: Name of the Langfuse dataset
        output_path: Path to write the JSONL file
        format: Output format — "openai" for OpenAI fine-tuning format

    Returns:
        Number of examples written
    """
    client = get_langfuse_client()
    dataset = client.get_dataset(dataset_name)

    count = 0
    with open(output_path, "w") as f:
        for item in dataset.items:
            question = item.input.get("question", "")
            sql = item.expected_output.get("sql", "")

            if not question or not sql:
                continue

            if format == "openai":
                # OpenAI fine-tuning format
                entry = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a SQL expert. Generate PostgreSQL "
                                "queries for the given question."
                            ),
                        },
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": sql},
                    ]
                }
            else:
                # Generic format
                entry = {
                    "input": question,
                    "output": sql,
                    "metadata": item.metadata or {},
                }

            f.write(json.dumps(entry) + "\n")
            count += 1

    return count
