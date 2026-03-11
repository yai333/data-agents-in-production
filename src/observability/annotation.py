"""Trace annotation and scoring for the Text-to-SQL agent.

Engineers annotate traces as correct/incorrect/partial.
Good annotations feed fine-tuning datasets (Chapter 4).
Bad annotations trigger MDL knowledge revision (Chapter 3.3).

Usage:
    from src.observability.annotation import score_trace, annotate_trace

    # Programmatic scoring (from eval pipeline)
    score_trace(trace_id, "accuracy", 1.0, comment="SQL correct")

    # Human annotation (from review UI or script)
    annotate_trace(trace_id, AnnotationLabel.CORRECT, reviewer="alice")
"""

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.observability.tracing import get_langfuse_client


class AnnotationLabel(str, Enum):
    """Labels for human annotation of traces.

    CORRECT: SQL is correct, answer is accurate → candidate for fine-tuning
    INCORRECT: SQL is wrong or answer is misleading → triggers MDL review
    PARTIAL: SQL runs but answer is incomplete → needs investigation
    """
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"


@dataclass
class AnnotationResult:
    """Result of an annotation operation."""
    trace_id: str
    score_id: str
    label: str
    value: float


def score_trace(
    trace_id: str,
    name: str,
    value: float,
    comment: str | None = None,
    data_type: str = "NUMERIC",
) -> str:
    """Add a numeric score to a trace.

    Use for automated evaluation scores (accuracy, latency, cost).

    Args:
        trace_id: The Langfuse trace ID
        name: Score name (e.g., "accuracy", "latency_ms", "cost_usd")
        value: Numeric value (0.0-1.0 for accuracy, raw for others)
        comment: Optional explanation of the score
        data_type: Score type — NUMERIC, BOOLEAN, or CATEGORICAL

    Returns:
        The score ID
    """
    client = get_langfuse_client()

    score_id = str(uuid.uuid4())
    client.create_score(
        score_id=score_id,
        trace_id=trace_id,
        name=name,
        value=value,
        comment=comment,
        data_type=data_type,
    )

    return score_id


def annotate_trace(
    trace_id: str,
    label: AnnotationLabel,
    reviewer: str | None = None,
    comment: str | None = None,
    corrected_sql: str | None = None,
) -> AnnotationResult:
    """Annotate a trace with a human judgment label.

    This is the core annotation workflow:
    - Engineers review traces in the Langfuse UI or via scripts
    - They label each trace as CORRECT, INCORRECT, or PARTIAL
    - CORRECT traces become fine-tuning candidates (Chapter 4)
    - INCORRECT traces trigger MDL revision (Chapter 3.3 feedback loop)

    Args:
        trace_id: The Langfuse trace ID to annotate
        label: The annotation label (CORRECT, INCORRECT, PARTIAL)
        reviewer: Who made the annotation (for audit trail)
        comment: Why this label was chosen
        corrected_sql: For INCORRECT traces, the corrected SQL

    Returns:
        AnnotationResult with the score ID
    """
    client = get_langfuse_client()

    # Map labels to numeric values for filtering
    label_values = {
        AnnotationLabel.CORRECT: 1.0,
        AnnotationLabel.INCORRECT: 0.0,
        AnnotationLabel.PARTIAL: 0.5,
    }

    # Build comment with reviewer and correction info
    parts = []
    if reviewer:
        parts.append(f"Reviewer: {reviewer}")
    if comment:
        parts.append(comment)
    if corrected_sql:
        parts.append(f"Corrected SQL: {corrected_sql}")

    full_comment = " | ".join(parts) if parts else None

    score_id = str(uuid.uuid4())
    client.create_score(
        score_id=score_id,
        trace_id=trace_id,
        name="human_annotation",
        value=label_values[label],
        comment=full_comment,
        data_type="NUMERIC",
    )

    # Also add the label as a categorical score for easy filtering
    client.create_score(
        trace_id=trace_id,
        name="annotation_label",
        value=label.value,
        data_type="CATEGORICAL",
    )

    return AnnotationResult(
        trace_id=trace_id,
        score_id=score_id,
        label=label.value,
        value=label_values[label],
    )


def batch_annotate(
    annotations: list[dict[str, Any]],
) -> list[AnnotationResult]:
    """Annotate multiple traces in batch.

    Args:
        annotations: List of dicts with keys:
            trace_id, label (AnnotationLabel), reviewer, comment,
            corrected_sql (optional)

    Returns:
        List of AnnotationResult
    """
    results = []
    for ann in annotations:
        result = annotate_trace(
            trace_id=ann["trace_id"],
            label=ann["label"],
            reviewer=ann.get("reviewer"),
            comment=ann.get("comment"),
            corrected_sql=ann.get("corrected_sql"),
        )
        results.append(result)

    # Flush after batch to ensure all scores are sent
    client = get_langfuse_client()
    client.flush()

    return results
