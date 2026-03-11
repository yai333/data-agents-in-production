"""Schema enrichment via LLM and database sampling.

This module provides functions to enrich raw schema metadata with
semantic descriptions and sample values. Used during schema card
generation (one-time cost), not at query time.
"""

from typing import Any

from src.schema.models import ColumnCard
from src.adapters.base import LLMAdapter


async def enrich_column(
    table_name: str,
    column_name: str,
    data_type: str,
    sample_values: list[Any],
    adapter: LLMAdapter,
) -> ColumnCard:
    """Use LLM to generate column description from name and samples.

    This is a one-time cost per column. Run during schema setup,
    not at query time. Results are cached in the schema JSON.
    """

    prompt = f"""Describe this database column based on its name and sample values.
Be concise (1 sentence). Focus on what the column represents.

Table: {table_name}
Column: {column_name}
Type: {data_type}
Sample values: {sample_values[:5]}

Description:"""

    response = await adapter.generate(prompt)

    return ColumnCard(
        name=column_name,
        data_type=data_type,
        description=response.content.strip(),
        examples=[str(v) for v in sample_values[:3]],
    )


async def extract_sample_values(
    conn,
    table_name: str,
    column_name: str,
    limit: int = 10,
) -> list[Any]:
    """Get distinct sample values from a column.

    Uses DISTINCT to get variety, LIMIT for efficiency.
    Excludes NULL values to show meaningful examples.
    """
    query = f"""
        SELECT DISTINCT {column_name}
        FROM {table_name}
        WHERE {column_name} IS NOT NULL
        LIMIT {limit}
    """
    rows = await conn.fetch(query)
    return [row[0] for row in rows]
