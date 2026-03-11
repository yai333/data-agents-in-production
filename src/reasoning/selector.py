"""Select appropriate reasoning method for a query.

This module provides heuristics to match reasoning complexity
to query complexity. The goal is to use the minimum reasoning
needed for accurate results.

See 1.5 for the decision framework rationale.
"""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schema import SchemaStore


class ReasoningMethod(str, Enum):
    """Available reasoning methods, ordered by complexity."""

    DIRECT = "direct"          # No explicit reasoning
    COT = "cot"                # Chain-of-thought
    AGENTIC_COT = "agentic_cot"  # CoT with tool use
    REASONING_MODEL = "reasoning_model"  # o1/o3/R1


# Patterns that indicate simple queries
SIMPLE_PATTERNS = [
    "count",
    "how many",
    "list all",
    "show all",
    "get all",
    "what is the",
    "what are the",
]

# Patterns that indicate complex queries
COMPLEX_PATTERNS = [
    "compared to",
    "versus",
    "growth",
    "trend",
    "more than average",
    "less than average",
    "top",
    "bottom",
    "rank",
    "percentile",
    "year over year",
    "month over month",
    "rolling",
    "cumulative",
    "running total",
    "window",
    "partition",
    "recursive",
    "hierarchy",
]

# Patterns that indicate very complex queries (need reasoning model)
ENTERPRISE_PATTERNS = [
    "seasonal adjustment",
    "forecas",
    "predict",
    "anomal",
    "outlier",
    "correlation",
    "regression",
    "statistical",
]


def select_reasoning_method(
    question: str,
    schema_complexity: int,
    latency_budget_ms: int = 2000,
) -> ReasoningMethod:
    """Select the appropriate reasoning method.

    The selection balances accuracy against latency:
    - Direct: Fast but may miss complex logic
    - CoT: Good for multi-table queries
    - Agentic CoT: Best for uncertain requirements
    - Reasoning Model: For enterprise-level complexity

    Args:
        question: Natural language question
        schema_complexity: Number of tables likely involved
        latency_budget_ms: Maximum acceptable latency

    Returns:
        ReasoningMethod to use

    Example:
        >>> select_reasoning_method("How many customers?", 1)
        ReasoningMethod.DIRECT
        >>> select_reasoning_method("Revenue by country by month", 3)
        ReasoningMethod.COT
    """
    question_lower = question.lower()

    # Check for enterprise-level patterns first
    if any(p in question_lower for p in ENTERPRISE_PATTERNS):
        if latency_budget_ms >= 5000:
            return ReasoningMethod.REASONING_MODEL
        else:
            return ReasoningMethod.AGENTIC_COT

    # Check for simple patterns with low complexity
    is_simple = any(p in question_lower for p in SIMPLE_PATTERNS)
    if is_simple and schema_complexity <= 1:
        return ReasoningMethod.DIRECT

    # Check latency constraints
    if latency_budget_ms < 800:
        # Very tight latency: direct only
        return ReasoningMethod.DIRECT
    elif latency_budget_ms < 1500:
        # Tight latency: direct or basic CoT
        if schema_complexity <= 2:
            return ReasoningMethod.DIRECT
        else:
            return ReasoningMethod.COT

    # Check for complex patterns
    has_complex_pattern = any(p in question_lower for p in COMPLEX_PATTERNS)

    # Select based on complexity
    if schema_complexity <= 1 and not has_complex_pattern:
        return ReasoningMethod.DIRECT
    elif schema_complexity <= 2 and not has_complex_pattern:
        return ReasoningMethod.COT
    elif schema_complexity <= 4:
        if has_complex_pattern:
            return ReasoningMethod.AGENTIC_COT
        else:
            return ReasoningMethod.COT
    else:
        # High complexity (5+ tables)
        if latency_budget_ms >= 5000:
            return ReasoningMethod.REASONING_MODEL
        else:
            return ReasoningMethod.AGENTIC_COT


def estimate_schema_complexity(
    question: str,
    schema_store: "SchemaStore",
) -> int:
    """Estimate how many tables a question likely needs.

    Uses the schema store's search to find relevant tables.

    Args:
        question: Natural language question
        schema_store: Schema information

    Returns:
        Number of tables likely needed
    """
    tables = schema_store.search_tables(question)
    return len(tables) if tables else 1


def get_method_characteristics(method: ReasoningMethod) -> dict:
    """Get latency and accuracy characteristics for a method.

    Args:
        method: Reasoning method

    Returns:
        Dict with latency_multiplier, accuracy_gain, token_overhead
    """
    characteristics = {
        ReasoningMethod.DIRECT: {
            "latency_multiplier": 1.0,
            "accuracy_gain": 0,
            "token_overhead": 0,
            "description": "Direct generation, no explicit reasoning",
        },
        ReasoningMethod.COT: {
            "latency_multiplier": 1.5,
            "accuracy_gain": 0.12,  # ~12% improvement
            "token_overhead": 200,
            "description": "Chain-of-thought with structured phases",
        },
        ReasoningMethod.AGENTIC_COT: {
            "latency_multiplier": 2.5,
            "accuracy_gain": 0.18,  # ~18% improvement
            "token_overhead": 500,
            "description": "CoT with tool use and error recovery",
        },
        ReasoningMethod.REASONING_MODEL: {
            "latency_multiplier": 7.0,
            "accuracy_gain": 0.23,  # ~23% improvement
            "token_overhead": 1000,
            "description": "o1/o3 reasoning models with built-in CoT",
        },
    }
    return characteristics[method]


def recommend_method_for_latency(
    target_latency_ms: int,
    base_latency_ms: int = 500,
) -> list[ReasoningMethod]:
    """Get methods that fit within a latency budget.

    Args:
        target_latency_ms: Maximum acceptable latency
        base_latency_ms: Baseline latency for direct generation

    Returns:
        List of methods that fit, ordered by preference
    """
    viable = []

    for method in ReasoningMethod:
        chars = get_method_characteristics(method)
        expected_latency = base_latency_ms * chars["latency_multiplier"]
        if expected_latency <= target_latency_ms:
            viable.append(method)

    # Order by accuracy gain (best first)
    return sorted(
        viable,
        key=lambda m: get_method_characteristics(m)["accuracy_gain"],
        reverse=True,
    )
