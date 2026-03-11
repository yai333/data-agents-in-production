"""Structured SQL generation with schema enforcement.

This module provides SQL generation with guaranteed schema compliance
using constrained decoding. The structured output eliminates parse
failures entirely.

See 1.4 for the principles behind constrained decoding.
"""

from typing import Any

from src.adapters import create_adapter
from src.adapters.base import LLMAdapter
from src.schema import SchemaStore, render_schema
from src.structured.models import SQLResult, QueryPlan, NaturalLanguageAnswer


async def generate_sql_structured(
    question: str,
    schema_store: SchemaStore,
    examples_text: str = "",
    use_planning: bool = False,
    adapter: LLMAdapter | None = None,
) -> SQLResult:
    """Generate SQL with structured output guarantee.

    This is the primary entry point for SQL generation. It uses
    constrained decoding to ensure the output always matches
    the SQLResult schema.

    Args:
        question: Natural language question
        schema_store: Schema information
        examples_text: Few-shot examples (from 1.3)
        use_planning: Whether to generate a plan first (for complex queries)
        adapter: Optional LLM adapter (creates default if not provided)

    Returns:
        SQLResult with sql, confidence, assumptions, and metadata

    Example:
        result = await generate_sql_structured(
            "What is the total revenue by country?",
            schema_store,
        )
        print(result.sql)  # SELECT c.country, SUM(i.total)...
        print(result.confidence)  # 0.85
    """
    if adapter is None:
        adapter = create_adapter()

    # Get relevant schema
    tables = schema_store.search_tables(question)
    if not tables:
        tables = schema_store.get_all_tables()[:10]

    schema_text = render_schema(tables)

    if use_planning:
        # Step 1: Generate plan (for complex queries)
        plan = await generate_plan(question, schema_text, adapter)

        # Step 2: Generate SQL from plan
        return await generate_sql_from_plan(
            question, plan, schema_text, examples_text, adapter
        )
    else:
        # Direct generation (simpler, faster)
        return await generate_sql_direct(
            question, schema_text, examples_text, adapter
        )


async def generate_sql_direct(
    question: str,
    schema_text: str,
    examples_text: str,
    adapter: LLMAdapter,
) -> SQLResult:
    """Generate SQL directly without planning step.

    Use this for simple to medium complexity queries where
    planning overhead isn't justified.
    """
    prompt = f"""Generate a SQL query for the question below.
Return a JSON object with:
- sql: The SQL query (no markdown code fences)
- confidence: Your confidence level (0.0 to 1.0)
- assumptions: Any assumptions you made about ambiguous terms
- tables_used: List of tables referenced

DATABASE SCHEMA:
{schema_text}

{examples_text}

QUESTION: {question}

Generate the SQL query:"""

    system_prompt = """You are a SQL expert. Generate precise, efficient queries.
Rules:
1. Use only tables and columns from the provided schema
2. Return the SQL without markdown formatting
3. Set confidence lower if the question is ambiguous
4. List any assumptions you made"""

    result = await adapter.generate_structured(
        prompt=prompt,
        response_model=SQLResult,
        system_prompt=system_prompt,
    )

    return result.data


async def generate_plan(
    question: str,
    schema_text: str,
    adapter: LLMAdapter,
) -> QueryPlan:
    """Generate a query plan before SQL generation.

    Planning helps for complex queries (3+ tables, aggregations,
    subqueries). The explicit plan catches errors early.

    Args:
        question: Natural language question
        schema_text: Rendered schema
        adapter: LLM adapter

    Returns:
        QueryPlan with tables, joins, filters, and reasoning
    """
    prompt = f"""Analyze this question and create a query plan.
Identify the tables needed, join paths, filters, and any aggregations.

DATABASE SCHEMA:
{schema_text}

QUESTION: {question}

Think through:
1. What tables contain the data we need?
2. How do these tables connect (join paths)?
3. What filters (WHERE conditions) are needed?
4. Any aggregations (GROUP BY, SUM, COUNT)?
5. Any ordering or limits?

Create the query plan:"""

    system_prompt = """You are a SQL query planner. Think step by step.
Break down the question into components before generating SQL."""

    result = await adapter.generate_structured(
        prompt=prompt,
        response_model=QueryPlan,
        system_prompt=system_prompt,
    )

    return result.data


async def generate_sql_from_plan(
    question: str,
    plan: QueryPlan,
    schema_text: str,
    examples_text: str,
    adapter: LLMAdapter,
) -> SQLResult:
    """Generate SQL based on a query plan.

    The plan provides explicit guidance, reducing errors on
    complex queries.
    """
    plan_text = f"""QUERY PLAN:
- Tables needed: {', '.join(plan.tables_needed)}
- Join path: {', '.join(plan.join_path) if plan.join_path else 'No joins needed'}
- Filters: {', '.join(plan.filters) if plan.filters else 'No filters'}
- Aggregations: {', '.join(plan.aggregations) if plan.aggregations else 'No aggregations'}
- Ordering: {plan.ordering or 'None'}
- Limit: {plan.limit or 'None'}
- Reasoning: {plan.reasoning}"""

    prompt = f"""Generate a SQL query based on this plan.

{plan_text}

DATABASE SCHEMA:
{schema_text}

{examples_text}

QUESTION: {question}

Follow the plan and generate the SQL query:"""

    system_prompt = """You are a SQL expert. Follow the provided plan precisely.
The plan has already identified the tables and joins needed."""

    result = await adapter.generate_structured(
        prompt=prompt,
        response_model=SQLResult,
        system_prompt=system_prompt,
    )

    return result.data


async def generate_answer(
    question: str,
    sql: str,
    results: list[dict[str, Any]],
    adapter: LLMAdapter | None = None,
) -> NaturalLanguageAnswer:
    """Generate a natural language answer from query results.

    Use this after executing SQL to provide a human-readable response.

    Args:
        question: Original user question
        sql: The SQL that was executed
        results: Query results as list of dicts
        adapter: Optional LLM adapter

    Returns:
        NaturalLanguageAnswer with answer, data points, and caveats
    """
    if adapter is None:
        adapter = create_adapter()

    # Format results for prompt (limit to avoid token overflow)
    if len(results) > 20:
        results_text = (
            f"First 20 of {len(results)} results:\n"
            + "\n".join(str(r) for r in results[:20])
        )
    else:
        results_text = "\n".join(str(r) for r in results)

    prompt = f"""Based on the query results, provide a natural language answer.

QUESTION: {question}

SQL EXECUTED:
{sql}

RESULTS:
{results_text}

Provide:
1. A clear, concise answer to the question
2. Key data points that support the answer
3. Any caveats or limitations"""

    system_prompt = """You are a data analyst. Provide clear, accurate answers.
Cite specific numbers from the results. Note any limitations."""

    result = await adapter.generate_structured(
        prompt=prompt,
        response_model=NaturalLanguageAnswer,
        system_prompt=system_prompt,
    )

    return result.data


# Convenience function for free-form comparison
async def generate_sql_freeform(
    question: str,
    schema_store: SchemaStore,
    examples_text: str = "",
    adapter: LLMAdapter | None = None,
) -> str:
    """Generate SQL without structured output (for comparison).

    This is the traditional approach that requires regex parsing.
    Use generate_sql_structured instead for production.
    """
    if adapter is None:
        adapter = create_adapter()

    tables = schema_store.search_tables(question)
    if not tables:
        tables = schema_store.get_all_tables()[:10]

    schema_text = render_schema(tables)

    prompt = f"""Generate a SQL query for the question below.
Return only the SQL query, no explanation or markdown.

DATABASE SCHEMA:
{schema_text}

{examples_text}

QUESTION: {question}

SQL:"""

    response = await adapter.generate(prompt)

    # Parse SQL from response (the error-prone approach)
    sql = response.content.strip()
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.startswith("```"):
        sql = sql[3:]
    if sql.endswith("```"):
        sql = sql[:-3]

    return sql.strip()
