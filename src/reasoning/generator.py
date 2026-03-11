"""Reasoning-enhanced SQL generation.

This module provides the main entry point for SQL generation
with appropriate reasoning levels. It selects the reasoning
method based on query complexity and latency requirements.

See 1.5 for the principles behind reasoning-enhanced generation.
"""

from typing import TYPE_CHECKING, Optional

from src.adapters import create_adapter
from src.adapters.base import LLMAdapter
from src.reasoning.prompts import (
    REASONING_PROMPT,
    DIRECT_PROMPT,
    COT_PROMPT,
    RECOVERY_PROMPT,
    SYSTEM_PROMPTS,
)
from src.reasoning.parser import (
    ReasoningResult,
    parse_reasoning_response,
    extract_sql_from_text,
)
from src.reasoning.selector import (
    ReasoningMethod,
    select_reasoning_method,
    estimate_schema_complexity,
)
from src.schema import SchemaStore, render_schema

if TYPE_CHECKING:
    pass


async def generate_sql_with_reasoning(
    question: str,
    schema_store: SchemaStore,
    examples_text: str = "",
    method: Optional[ReasoningMethod] = None,
    latency_budget_ms: int = 2000,
    adapter: Optional[LLMAdapter] = None,
) -> ReasoningResult:
    """Generate SQL with appropriate reasoning level.

    This is the main entry point for reasoning-enhanced SQL generation.
    It automatically selects the reasoning method based on query
    complexity, or uses the specified method.

    Args:
        question: Natural language question
        schema_store: Schema information
        examples_text: Few-shot examples (from 1.3)
        method: Force a specific method (auto-select if None)
        latency_budget_ms: Maximum acceptable latency for method selection
        adapter: LLM adapter (creates default if None)

    Returns:
        ReasoningResult with SQL, reasoning trace, and confidence

    Example:
        >>> result = await generate_sql_with_reasoning(
        ...     "What are the top 5 customers by spending?",
        ...     schema_store,
        ... )
        >>> print(result.sql)
        SELECT c.first_name, c.last_name, SUM(i.total) as total_spent
        FROM customer c JOIN invoice i ON c.customer_id = i.customer_id
        GROUP BY c.customer_id ORDER BY total_spent DESC LIMIT 5
        >>> print(result.confidence)
        0.85
    """
    if adapter is None:
        adapter = create_adapter()

    # Get relevant schema
    tables = schema_store.search_tables(question)
    if not tables:
        tables = schema_store.get_all_tables()[:10]

    schema_text = render_schema(tables)

    # Auto-select method if not specified
    if method is None:
        schema_complexity = len(tables)
        method = select_reasoning_method(
            question=question,
            schema_complexity=schema_complexity,
            latency_budget_ms=latency_budget_ms,
        )

    # Dispatch to appropriate generator
    if method == ReasoningMethod.DIRECT:
        result = await _generate_direct(question, schema_text, examples_text, adapter)
    elif method == ReasoningMethod.COT:
        result = await _generate_cot(question, schema_text, examples_text, adapter)
    elif method == ReasoningMethod.AGENTIC_COT:
        result = await _generate_agentic(question, schema_text, examples_text, adapter)
    else:  # REASONING_MODEL
        result = await _generate_with_reasoning_model(
            question, schema_text, examples_text
        )

    result.method = method.value
    return result


async def _generate_direct(
    question: str,
    schema_text: str,
    examples_text: str,
    adapter: LLMAdapter,
) -> ReasoningResult:
    """Direct generation without explicit reasoning.

    Fastest but least accurate for complex queries.
    Best for: Single-table lookups, simple aggregations.
    """
    prompt = DIRECT_PROMPT.format(
        schema=schema_text,
        examples=examples_text,
        question=question,
    )

    response = await adapter.generate(
        prompt,
        system_prompt=SYSTEM_PROMPTS["direct"],
    )
    sql = response.content.strip()

    # Remove markdown formatting if present
    sql = _clean_sql(sql)

    return ReasoningResult(
        reasoning="[Direct generation - no explicit reasoning]",
        analysis="",
        query_explanation="",
        verification="",
        sql=sql,
        confidence=0.7,  # Lower confidence without reasoning
        method="direct",
        raw_response=response.content,
    )


async def _generate_cot(
    question: str,
    schema_text: str,
    examples_text: str,
    adapter: LLMAdapter,
) -> ReasoningResult:
    """Chain-of-thought generation with four phases.

    Good balance of accuracy and latency.
    Best for: Multi-table queries, basic joins, standard aggregations.
    """
    prompt = REASONING_PROMPT.format(
        schema=schema_text,
        examples=examples_text,
        question=question,
    )

    response = await adapter.generate(
        prompt,
        system_prompt=SYSTEM_PROMPTS["reasoning"],
    )

    result = parse_reasoning_response(response.content)
    result.raw_response = response.content
    return result


async def _generate_agentic(
    question: str,
    schema_text: str,
    examples_text: str,
    adapter: LLMAdapter,
) -> ReasoningResult:
    """Agentic CoT with error recovery.

    Includes a recovery step if initial confidence is low.
    Best for: Complex queries, uncertain requirements.

    Note: Full agentic implementation with tool use is in 1.6.
    This is a simplified version that includes error recovery.
    """
    # First attempt with standard CoT
    result = await _generate_cot(question, schema_text, examples_text, adapter)

    # If confidence is low, try recovery
    if result.confidence < 0.6:
        recovery_prompt = RECOVERY_PROMPT.format(
            previous_analysis=result.analysis,
            previous_sql=result.sql,
            schema=schema_text,
            examples=examples_text,
            question=question,
        )

        response = await adapter.generate(
            recovery_prompt,
            system_prompt=SYSTEM_PROMPTS["reasoning"],
        )

        recovered = parse_reasoning_response(response.content)

        # Use recovered result if it has better confidence
        if recovered.confidence > result.confidence:
            result = recovered
            # Boost confidence slightly for successful recovery
            result.confidence = min(result.confidence + 0.1, 0.95)

    result.method = "agentic_cot"
    return result


async def _generate_with_reasoning_model(
    question: str,
    schema_text: str,
    examples_text: str,
) -> ReasoningResult:
    """Use a reasoning model (o1/o3) for complex queries.

    Highest accuracy but slowest.
    Best for: Enterprise queries, multi-step logic, domain rules.

    Note: Requires OpenAI API with o1/o3 access.
    Falls back to GPT-4o if reasoning models unavailable.
    """
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()

        prompt = f"""Generate a SQL query for this question.

DATABASE SCHEMA:
{schema_text}

{examples_text}

QUESTION: {question}

Provide:
1. Your reasoning process
2. The SQL query
3. Verification that the query answers the question"""

        # Try o1 first
        try:
            response = await client.chat.completions.create(
                model="o1",
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            # Fall back to GPT-4o if o1 not available
            response = await client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a SQL expert. Think carefully step by step.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

        content = response.choices[0].message.content

        # Parse response (reasoning models structure their own output)
        result = parse_reasoning_response(content)
        result.method = "reasoning_model"

        # Reasoning models get higher base confidence
        result.confidence = min(result.confidence + 0.1, 0.95)

        return result

    except ImportError:
        # OpenAI not installed, use adapter
        adapter = create_adapter()
        result = await _generate_cot(question, schema_text, examples_text, adapter)
        result.method = "reasoning_model_fallback"
        return result


def _clean_sql(sql: str) -> str:
    """Clean up SQL from various formats.

    Handles:
    - Markdown code blocks
    - Leading/trailing whitespace
    - Multiple SQL statements (returns first)
    """
    sql = sql.strip()

    # Remove markdown code blocks
    if sql.startswith("```"):
        lines = sql.split("\n")
        # Skip first line (```sql or ```)
        lines = lines[1:]
        # Find closing ```
        for i, line in enumerate(lines):
            if line.strip() == "```":
                lines = lines[:i]
                break
        sql = "\n".join(lines)

    # Remove trailing semicolons and extra whitespace
    sql = sql.strip().rstrip(";").strip()

    return sql


async def generate_with_fallback(
    question: str,
    schema_store: SchemaStore,
    examples_text: str = "",
    adapter: Optional[LLMAdapter] = None,
) -> ReasoningResult:
    """Generate SQL with automatic fallback on failure.

    Tries methods in order of latency until one succeeds.
    Useful for production where reliability matters more than latency.

    Args:
        question: Natural language question
        schema_store: Schema information
        examples_text: Few-shot examples
        adapter: LLM adapter

    Returns:
        ReasoningResult from first successful method
    """
    if adapter is None:
        adapter = create_adapter()

    methods = [
        ReasoningMethod.DIRECT,
        ReasoningMethod.COT,
        ReasoningMethod.AGENTIC_COT,
    ]

    last_error = None
    for method in methods:
        try:
            result = await generate_sql_with_reasoning(
                question=question,
                schema_store=schema_store,
                examples_text=examples_text,
                method=method,
                adapter=adapter,
            )

            # Check if result seems valid
            if result.sql and "SELECT" in result.sql.upper():
                return result

        except Exception as e:
            last_error = e
            continue

    # All methods failed
    raise RuntimeError(f"All reasoning methods failed. Last error: {last_error}")
