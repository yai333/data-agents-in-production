#!/usr/bin/env python3
"""2.2 Schema Representation - Evaluation with Schema Cards.

This script compares SQL generation accuracy:
- Baseline (from 2.1): Table names only, no column info
- With Schema Cards: Full table/column descriptions and relationships

Usage:
    make db-up
    python scripts/generate_chinook_schema.py
    python scripts/run_chapter_2_2.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import asyncpg

from src.adapters import create_adapter
from src.schema.store import SchemaStore
from src.schema.render import render_schema
from src.utils.config import load_config
from evals.runner import run_evaluation
from evals.chinook_golden_set import GOLDEN_SET


BASELINE_TABLES = """Tables: album, artist, customer, employee, genre, invoice,
        invoice_line, media_type, playlist, playlist_track, track"""


async def execute_sql(sql: str) -> Any:
    """Execute SQL against Chinook database."""
    settings = load_config()
    conn = await asyncpg.connect(settings.database.url)
    try:
        await conn.execute(f"SET statement_timeout = '{settings.safety.query_timeout_seconds}s'")
        result = await conn.fetch(sql)
        return [dict(row) for row in result]
    finally:
        await conn.close()


async def main():
    """Compare baseline vs Schema Cards."""
    print("2.2 Schema Cards Evaluation")
    print("=" * 50)
    print()

    settings = load_config()
    print(f"Provider: {settings.llm.provider}")
    print(f"Model: {settings.llm.model}")
    print()

    # Load schema
    schema_path = project_root / "config" / "chinook_schema.json"
    if not schema_path.exists():
        print(f"Schema not found: {schema_path}")
        print("Run: python scripts/generate_chinook_schema.py")
        return

    store = SchemaStore(schema_path)
    print(f"Loaded schema: {len(store)} tables")
    print()

    adapter = create_adapter()

    # Baseline: table names only
    async def baseline_generate(question: str) -> str:
        prompt = f"""Generate a SQL query for this question.
Return only the SQL query, no explanation or markdown.

{BASELINE_TABLES}

Question: {question}

SQL:"""
        response = await adapter.generate(prompt)
        return response.content.strip()

    baseline = await run_evaluation(GOLDEN_SET, baseline_generate, execute_sql, verbose=False)
    print("Baseline (table names only):")
    print(f"  Execution Accuracy: {baseline['execution_accuracy']:.0%}")
    print(f"  Result Accuracy:    {baseline['result_accuracy']:.0%}")
    print()

    # With Schema Cards
    async def schema_generate(question: str) -> str:
        cards = list(store.tables.values())  # All TableCard objects
        schema_text = render_schema(cards)

        prompt = f"""You are a SQL expert. Generate a SQL query for the given question.
Use only the tables and columns described in the schema below.
Return only the SQL query, no explanation or markdown.

DATABASE SCHEMA:
{schema_text}

QUESTION: {question}

SQL:"""
        response = await adapter.generate(prompt)
        sql = response.content.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        return sql.strip()

    with_schema = await run_evaluation(GOLDEN_SET, schema_generate, execute_sql, verbose=False)
    exec_diff = with_schema['execution_accuracy'] - baseline['execution_accuracy']
    result_diff = with_schema['result_accuracy'] - baseline['result_accuracy']

    print("With Schema Cards:")
    print(f"  Execution Accuracy: {with_schema['execution_accuracy']:.0%}  ({exec_diff*100:+.0f}pp)")
    print(f"  Result Accuracy:    {with_schema['result_accuracy']:.0%}  ({result_diff*100:+.0f}pp)")


if __name__ == "__main__":
    asyncio.run(main())
