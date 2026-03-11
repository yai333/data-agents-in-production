#!/usr/bin/env python3
"""Chapter 3.2: PII Protection - Four-Phase Pipeline Demo.

Requirements:
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download en_core_web_lg
    make db-up  # Start Chinook database

Usage:
    python scripts/run_chapter_3_2.py
"""

import asyncio
import re
import sys
from pathlib import Path
from typing import Literal, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from pydantic import BaseModel, field_validator
from langchain_community.utilities.sql_database import SQLDatabase

from chapter_3_2 import (
    PIIMappingStore,
    PresidioPIIDetector,
    DatabaseResultsAnonymizer,
    detect_and_pseudonymize,
    build_allow_list,
    resolve_parameters,
    deanonymize,
)
from chapter_3_2.core import SQLParameter as CoreSQLParameter
from src.adapters import create_adapter
from src.schema import SchemaStore, render_schema
from src.utils.config import load_config


class SQLParameter(BaseModel):
    param_name: str
    placeholder_ref: str
    param_type: Literal["string", "integer", "float"] = "string"


class SQLGenerationResult(BaseModel):
    reasoning: str
    sql_query: str
    parameters: list[SQLParameter]

    @field_validator("sql_query")
    @classmethod
    def validate_no_raw_pii(cls, v: str) -> str:
        if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", v):
            raise ValueError("SQL contains raw email address")
        return v


def create_constrained_model(allow_list: list[str]) -> type[SQLGenerationResult]:
    allowed_values = tuple(allow_list) if allow_list else ("__NONE__",)

    class ConstrainedSQLParameter(BaseModel):
        param_name: str
        placeholder_ref: Literal[allowed_values]  # type: ignore

    class ConstrainedSQLGenerationResult(BaseModel):
        reasoning: str
        sql_query: str
        parameters: list[ConstrainedSQLParameter]

    return ConstrainedSQLGenerationResult


def execute_query(sql_query: str, parameters: dict[str, Any], db: SQLDatabase) -> str:
    """Execute parameterized SQL using LangChain's built-in binding."""
    return db.run_no_throw(
        sql_query,
        include_columns=True,
        parameters=parameters,
    )


def parse_db_results(result_str: str) -> list[dict[str, Any]]:
    """Parse db.run_no_throw output into list of dicts."""
    import ast

    if not result_str or result_str.startswith("Error"):
        return []

    preprocessed = re.sub(r"Decimal\('([^']+)'\)", r"'\1'", result_str)
    preprocessed = re.sub(
        r"datetime\.datetime\(([^)]+)\)",
        lambda m: f"'{m.group(0)}'",
        preprocessed
    )

    try:
        rows = ast.literal_eval(preprocessed)
        if isinstance(rows, list):
            return rows
    except (ValueError, SyntaxError):
        pass

    return []


async def generate_sql_with_pii_protection(
    user_question: str,
    schema_context: str,
    mapping: PIIMappingStore,
    detector: PresidioPIIDetector,
) -> tuple[str, dict[str, Any], str]:
    adapter = create_adapter()

    print("\n" + "─" * 60)
    print("PHASE 1: PROTECT - Pseudonymize PII")
    print("─" * 60)

    pseudonymized_question = detect_and_pseudonymize(user_question, mapping, detector)
    print(f"Original:      {user_question}")
    print(f"Pseudonymized: {pseudonymized_question}")

    print("\n" + "─" * 60)
    print("PHASE 2: CONSTRAIN - Build allow-list")
    print("─" * 60)

    allow_list = build_allow_list(pseudonymized_question)
    print(f"Allow-list: {allow_list}")

    ConstrainedResult = create_constrained_model(allow_list)

    print("\n" + "─" * 60)
    print("PHASE 3: GENERATE - LLM generates SQL")
    print("─" * 60)

    system_prompt = f"""You are a SQL expert for PostgreSQL. Generate SQL queries for the Chinook music database.
Use parameterized queries with :param_name syntax (e.g., :email, :customer_name).
Return parameters in the order they appear in the query.

DATABASE SCHEMA:
{schema_context}"""

    prompt = f"Generate SQL for: {pseudonymized_question}"

    print(f"\nCalling LLM ({adapter.model})...")

    try:
        response = await adapter.generate_structured(
            prompt=prompt,
            response_model=ConstrainedResult,
            system_prompt=system_prompt,
        )
        result = response.data
    except Exception as e:
        print(f"Constrained generation failed: {e}")
        response = await adapter.generate_structured(
            prompt=prompt,
            response_model=SQLGenerationResult,
            system_prompt=system_prompt,
        )
        result = response.data

    print(f"\nLLM Reasoning: {result.reasoning}")
    print(f"\nGenerated SQL:\n{result.sql_query}")
    print(f"\nParameters (with placeholders):")
    for param in result.parameters:
        print(f"  :{param.param_name} = {param.placeholder_ref}")

    print("\n" + "─" * 60)
    print("PARAMETER BINDING - Resolve placeholders")
    print("─" * 60)

    core_params = [
        CoreSQLParameter(param_name=p.param_name, placeholder_ref=p.placeholder_ref)
        for p in result.parameters
    ]

    bound_params = resolve_parameters(core_params, mapping)
    print(f"Bound parameters (for database execution):")
    for name, value in bound_params.items():
        print(f"  :{name} = '{value}'")

    return result.sql_query, bound_params, result.reasoning


async def demo_full_pipeline():
    print("=" * 70)
    print("Chapter 3.2: Four-Phase PII Protection Pipeline")
    print("=" * 70)

    settings = load_config()
    db_uri = settings.database.url
    print(f"\nDatabase: {db_uri}")

    try:
        db = SQLDatabase.from_uri(db_uri)
        print(f"Connected to database: {db_uri.split('@')[-1] if '@' in db_uri else db_uri}")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Run: make db-up")
        return

    schema_path = project_root / "config" / "chinook_schema.json"
    if not schema_path.exists():
        print(f"Schema not found: {schema_path}")
        print("Run: python scripts/generate_chinook_schema.py")
        return

    schema_store = SchemaStore(schema_path)
    schema_context = render_schema(schema_store.get_all_tables())
    print(f"Loaded schema: {len(schema_store.tables)} tables")

    customers_result = db.run_no_throw(
        "SELECT first_name, last_name, email FROM customer LIMIT 3",
        include_columns=True
    )
    print(f"\nUsing real customers from database:\n{customers_result}")

    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", customers_result)

    mapping = PIIMappingStore()
    detector = PresidioPIIDetector()

    print(f"\nSession ID: {mapping.session_id}")
    print(f"PII Storage: SQLite ({mapping.db_path})")

    test_questions = [
        f"Find all invoices for customer with email {emails[0]}",
        f"What is the total spending for customer {emails[1]}?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {question}")
        print("=" * 70)

        try:
            sql, params, _ = await generate_sql_with_pii_protection(
                user_question=question,
                schema_context=schema_context,
                mapping=mapping,
                detector=detector,
            )

            print("\n" + "─" * 60)
            print("EXECUTE SQL (via db.run_no_throw)")
            print("─" * 60)

            results = execute_query(sql, params, db)
            print(f"Results:\n{results}")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("SESSION MAPPING STATE")
    print("=" * 70)
    for entity_type, entities in mapping.mappings.items():
        print(f"\n{entity_type}:")
        for placeholder, value in entities.items():
            print(f"  {placeholder} -> '{value}'")


async def demo_results_anonymization():
    print("\n" + "=" * 70)
    print("Database Results Anonymization Demo")
    print("=" * 70)
    print("\nThis demo tests that PII mappings are SHARED between:")
    print("  1. Question pseudonymization (user input)")
    print("  2. Database results anonymization (query output)")
    print("The same email should get the SAME placeholder in both places.")

    settings = load_config()
    db = SQLDatabase.from_uri(settings.database.url)

    mapping = PIIMappingStore()
    detector = PresidioPIIDetector()

    sample_result = db.run_no_throw("SELECT email FROM customer LIMIT 1", include_columns=True)
    customer_email = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", sample_result)[0]

    question = f"Show me invoices for {customer_email}"
    pseudonymized = detect_and_pseudonymize(question, mapping, detector)
    print(f"\n--- Step 1: Pseudonymize Question ---")
    print(f"Question: {question}")
    print(f"Pseudonymized: {pseudonymized}")
    print(f"Mapping created: {customer_email} -> {mapping.find_placeholder('EMAIL_ADDRESS', customer_email)}")

    raw_results_str = db.run_no_throw("""
        SELECT c.first_name || ' ' || c.last_name AS name,
               c.email, i.total
        FROM customer c
        JOIN invoice i ON c.customer_id = i.customer_id
        WHERE c.email = :customer_email
        LIMIT 3
    """, include_columns=True, parameters={"customer_email": customer_email})

    print(f"\n--- Step 2: Query Database ---")
    print(f"Raw results:\n{raw_results_str}")

    rows = parse_db_results(raw_results_str)

    if rows:
        print(f"\n--- Step 3: Anonymize Results ---")
        print(f"Parsed {len(rows)} rows")

        results_anonymizer = DatabaseResultsAnonymizer(mapping)
        anonymized_results = results_anonymizer.anonymize_results(rows)

        print(f"\nAnonymized results (what LLM sees):")
        for row in anonymized_results:
            print(f"  {row}")

        email_in_results = rows[0]['email']
        placeholder_from_question = mapping.find_placeholder('EMAIL_ADDRESS', customer_email)
        placeholder_in_results = anonymized_results[0]['email']

        print(f"\n--- Step 4: Verify Mapping Consistency ---")
        print(f"Email in question: {customer_email}")
        print(f"Email in results:  {email_in_results}")
        print(f"Placeholder from question: {placeholder_from_question}")
        print(f"Placeholder in results:    {placeholder_in_results}")

        if placeholder_from_question == placeholder_in_results:
            print("✓ PASSED: Same email gets same placeholder (mapping shared correctly)")
        else:
            print("✗ FAILED: Mapping not shared correctly!")

        new_mappings = results_anonymizer.get_new_mappings()
        if new_mappings:
            print(f"\nNew PII discovered in results (not in original question):")
            for entity_type, entities in new_mappings.items():
                for placeholder, value in entities.items():
                    print(f"  {placeholder} -> '{value}'")

        llm_response = "Found customers:\n"
        for row in anonymized_results:
            llm_response += f"- {row['name']} ({row['email']}): ${row['total']}\n"

        print(f"\nLLM Response (with placeholders):\n{llm_response}")

        final_response = deanonymize(llm_response, mapping)
        print(f"Final Response (user sees):\n{final_response}")


async def main():
    await demo_full_pipeline()
    await demo_results_anonymization()

    print("\n" + "=" * 70)
    print("Chapter 3.2 Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
