"""Structured output models for Text-to-SQL agents.

These Pydantic models define the schema for constrained decoding.
Using structured output eliminates parse failures entirely.

See 1.4 for deep dive on constrained decoding mechanism.
"""

from enum import Enum
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Discrete confidence levels for query generation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class QueryPlan(BaseModel):
    """Intermediate reasoning before SQL generation.

    Use this for complex queries that benefit from explicit planning.
    The plan helps catch errors before SQL generation.

    Example:
        plan = QueryPlan(
            tables_needed=["customer", "invoice"],
            join_path=["customer -> invoice ON customer_id"],
            filters=["country = 'USA'"],
            aggregations=["SUM(total)"],
            reasoning="Need to sum invoice totals for US customers"
        )
    """

    tables_needed: list[str] = Field(
        ...,
        description="Tables required for this query"
    )
    join_path: list[str] = Field(
        default_factory=list,
        description="Join sequence, e.g., ['customer -> invoice ON customer_id']"
    )
    filters: list[str] = Field(
        default_factory=list,
        description="WHERE conditions in natural language"
    )
    aggregations: list[str] = Field(
        default_factory=list,
        description="GROUP BY and aggregate functions needed"
    )
    ordering: str | None = Field(
        default=None,
        description="ORDER BY specification if needed"
    )
    limit: int | None = Field(
        default=None,
        description="LIMIT value if needed"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of approach"
    )


class SQLAgentResponse(BaseModel):
    """Unified response from a Text-to-SQL agent.

    The agent handles reasoning, SQL generation, and execution internally.
    This model structures what the user sees.

    Example:
        response = SQLAgentResponse(
            answer="You have 968 customers total.",
            sql=["SELECT COUNT(*) FROM customer"],
            confidence=0.95,
        )
    """

    answer: str = Field(
        ...,
        description="Natural language answer synthesized from query results"
    )
    sql: list[str] = Field(
        ...,
        description="SQL queries executed (supports multi-step queries)"
    )
    reasoning: str | None = Field(
        default=None,
        description="Optional explanation of approach taken"
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made about ambiguous terms"
    )
    followup_questions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in the answer (0-1)"
    )


# Legacy model for backwards compatibility with scripts
class SQLResult(BaseModel):
    """Structured SQL generation result (legacy).

    Use SQLAgentResponse for new code. This model is kept for
    backwards compatibility with existing evaluation scripts.
    """

    sql: str = Field(
        ...,
        description="The SQL query (no markdown, no explanation)"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Model's confidence in the query (0-1)"
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made about ambiguous terms"
    )
    tables_used: list[str] = Field(
        default_factory=list,
        description="Tables referenced in the query"
    )
    explanation: str | None = Field(
        default=None,
        description="Optional brief explanation of the query approach"
    )


class NaturalLanguageAnswer(BaseModel):
    """Human-readable answer with citations.

    Use this for the final response to the user after query execution.
    The structured format ensures consistent presentation.

    Example:
        answer = NaturalLanguageAnswer(
            answer="The total revenue from US customers is $15,234.50",
            data_points=["USA: $15,234.50", "Total customers: 13"],
            caveats=["Based on invoice totals, not including refunds"]
        )
    """

    answer: str = Field(
        ...,
        description="Natural language answer to the question"
    )
    data_points: list[str] = Field(
        default_factory=list,
        description="Key data points supporting the answer"
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Limitations or assumptions in the answer"
    )


class ValidationResult(BaseModel):
    """Result of SQL validation.

    Used in 1.6 for the validate-first pattern.
    """

    is_valid: bool = Field(
        ...,
        description="Whether the SQL is valid"
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of validation errors"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking issues"
    )
    suggested_fix: str | None = Field(
        default=None,
        description="Suggested corrected SQL if errors found"
    )


class AmbiguityDetection(BaseModel):
    """Result of ambiguity detection.

    Used in 1.7 for handling unclear queries.
    """

    is_ambiguous: bool = Field(
        ...,
        description="Whether the query has ambiguity"
    )
    ambiguity_type: str | None = Field(
        default=None,
        description="Type: lexical, temporal, scope, or semantic"
    )
    clarifying_question: str | None = Field(
        default=None,
        description="Question to ask user for clarification"
    )
    possible_interpretations: list[str] = Field(
        default_factory=list,
        description="Different ways to interpret the query"
    )
    default_interpretation: str | None = Field(
        default=None,
        description="Most likely interpretation if user doesn't clarify"
    )
