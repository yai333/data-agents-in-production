"""Structured output for Text-to-SQL agents.

This module provides Pydantic models and generation functions
for structured SQL output. Using constrained decoding eliminates
parse failures entirely.

Key components:
- SQLResult: Main SQL generation output model
- QueryPlan: Intermediate reasoning for complex queries
- NaturalLanguageAnswer: Human-readable answer with citations
- generate_sql_structured: Main entry point for SQL generation
"""

from src.structured.models import (
    SQLAgentResponse,
    SQLResult,
    QueryPlan,
    NaturalLanguageAnswer,
    ValidationResult,
    AmbiguityDetection,
    ConfidenceLevel,
)
from src.structured.generator import (
    generate_sql_structured,
    generate_sql_direct,
    generate_plan,
    generate_sql_from_plan,
    generate_answer,
    generate_sql_freeform,
)

__all__ = [
    # Models
    "SQLAgentResponse",
    "SQLResult",
    "QueryPlan",
    "NaturalLanguageAnswer",
    "ValidationResult",
    "AmbiguityDetection",
    "ConfidenceLevel",
    # Generation functions
    "generate_sql_structured",
    "generate_sql_direct",
    "generate_plan",
    "generate_sql_from_plan",
    "generate_answer",
    "generate_sql_freeform",
]
