"""SQL repair with error context.

When SQL execution fails, this module generates repaired SQL by including
the error message and hints in the prompt.

See 1.6 for the repair loop.
"""

from src.adapters import create_adapter
from src.adapters.base import LLMAdapter
from src.structured import SQLResult
from src.agent.error_classifier import ClassifiedError


REPAIR_PROMPT = """The previous SQL query failed. Fix it based on the error.

ORIGINAL QUESTION: {question}

FAILED SQL:
{failed_sql}

ERROR:
{error_message}

REPAIR HINT:
{repair_hint}

DATABASE SCHEMA:
{schema}

Generate a corrected SQL query that:
1. Fixes the specific error mentioned
2. Still answers the original question
3. Uses only valid tables and columns from the schema

Return the corrected SQL."""


REPAIR_SYSTEM_PROMPT = """You are a SQL repair expert. Your job is to fix SQL queries that failed.

Rules:
1. Focus on fixing the specific error mentioned
2. Don't change parts of the query that weren't causing the error
3. Use only tables and columns that exist in the schema
4. If the error is about a missing column, find an alternative that answers the question
5. Return valid, executable SQL"""


async def repair_sql(
    question: str,
    failed_sql: str,
    error: ClassifiedError,
    schema_text: str,
    adapter: LLMAdapter | None = None,
) -> SQLResult:
    """Attempt to repair failed SQL.

    Args:
        question: Original user question
        failed_sql: SQL that failed
        error: Classified error information
        schema_text: Database schema
        adapter: LLM adapter

    Returns:
        SQLResult with repaired SQL

    Example:
        >>> error = ClassifiedError(
        ...     error_type=ErrorType.SCHEMA,
        ...     message="column 'foo' does not exist",
        ...     repair_hint="Check column names",
        ...     retryable=True,
        ... )
        >>> result = await repair_sql(question, failed_sql, error, schema)
        >>> result.sql
        'SELECT bar FROM table...'  # Fixed column name
    """
    if adapter is None:
        adapter = create_adapter()

    prompt = REPAIR_PROMPT.format(
        question=question,
        failed_sql=failed_sql,
        error_message=error.message,
        repair_hint=error.repair_hint,
        schema=schema_text,
    )

    result = await adapter.generate_structured(
        prompt=prompt,
        response_model=SQLResult,
        system_prompt=REPAIR_SYSTEM_PROMPT,
    )

    # Lower confidence for repaired queries (we're uncertain)
    repaired = result.data
    repaired.confidence = min(repaired.confidence, 0.7)

    return repaired
