"""Observability module for the Text-to-SQL agent.

Provides Langfuse-based tracing, annotation, dataset export,
and evaluation bridge functions.
See Chapters 3.4 and 3.5 for usage patterns.
"""

from src.observability.tracing import (
    init_langfuse,
    get_langfuse_callback,
    get_langfuse_client,
    flush_langfuse,
)
from src.observability.annotation import (
    score_trace,
    annotate_trace,
    AnnotationLabel,
)
from src.observability.datasets import (
    export_good_traces_to_dataset,
    export_bad_traces_for_mdl_review,
)
from src.observability.evaluators import (
    run_component_eval,
    run_execution_eval,
    score_eval_results,
    upload_golden_set_to_dataset,
    link_trace_to_dataset_item,
)

__all__ = [
    "init_langfuse",
    "get_langfuse_callback",
    "get_langfuse_client",
    "flush_langfuse",
    "score_trace",
    "annotate_trace",
    "AnnotationLabel",
    "export_good_traces_to_dataset",
    "export_bad_traces_for_mdl_review",
    "run_component_eval",
    "run_execution_eval",
    "score_eval_results",
    "upload_golden_set_to_dataset",
    "link_trace_to_dataset_item",
]
