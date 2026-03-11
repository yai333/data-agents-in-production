#!/usr/bin/env python3
"""1.5 Environment Setup - Hello World SQL Generation.

This script verifies your environment is correctly set up:
1. Database connection works
2. LLM adapter works (OpenAI or Gemini)
3. Basic SQL generation succeeds

Usage:
    python scripts/run_chapter_1_5.py
    python scripts/run_chapter_1_5.py --provider gemini
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def check_environment() -> list[str]:
    """Check environment setup and return list of issues."""
    issues = []

    # Check Python version
    if sys.version_info < (3, 11):
        issues.append(f"Python 3.11+ required, found {sys.version_info.major}.{sys.version_info.minor}")

    # Check for at least one API key
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GOOGLE_API_KEY"))

    if not has_openai and not has_gemini:
        issues.append(
            "No API key found. Set OPENAI_API_KEY or GOOGLE_API_KEY in .env file.\n"
            "  Copy .env.example to .env and add your key."
        )

    return issues


def check_database() -> tuple[bool, str]:
    """Check if database is accessible."""
    try:
        import psycopg2

        db_url = os.getenv("CHINOOK_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/chinook")

        # Parse URL components
        # Format: postgresql://user:pass@host:port/dbname
        if "@" in db_url:
            auth_part = db_url.split("@")[0].split("//")[1]
            host_part = db_url.split("@")[1]
            user, password = auth_part.split(":")
            host_db = host_part.split("/")
            host_port = host_db[0].split(":")
            host = host_port[0]
            port = int(host_port[1]) if len(host_port) > 1 else 5432
            dbname = host_db[1] if len(host_db) > 1 else "chinook"
        else:
            return False, "Invalid CHINOOK_DATABASE_URL format"

        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            connect_timeout=5,
        )

        # Check if Chinook tables exist
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'artist'
        """)
        count = cursor.fetchone()[0]
        conn.close()

        if count == 0:
            return False, (
                "Database connected but Chinook tables not found.\n"
                "  Run: python scripts/setup_chinook.py\n"
                "  Then: make db-reset"
            )

        return True, "Database connected, Chinook tables found"

    except ImportError:
        return False, "psycopg2 not installed. Run: pip install -e ."
    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg:
            return False, (
                "Database connection refused. Is Postgres running?\n"
                "  Run: make db-up\n"
                "  Or: docker compose up -d postgres"
            )
        elif "password authentication failed" in error_msg:
            return False, "Database authentication failed. Check CHINOOK_DATABASE_URL in .env"
        else:
            return False, f"Database error: {error_msg}"


async def test_adapter(provider: str) -> tuple[bool, str]:
    """Test LLM adapter with a simple query."""
    try:
        from src.adapters import create_adapter

        print(f"\n[2/3] Testing {provider} adapter...")

        adapter = create_adapter(provider=provider)
        print(f"      Model: {adapter.model}")

        # Simple test prompt
        response = await adapter.generate(
            prompt="What is 2 + 2? Reply with just the number.",
            system_prompt="You are a helpful assistant. Be concise.",
        )

        if "4" in response.content:
            return True, f"Adapter working. Response: {response.content.strip()}"
        else:
            return True, f"Adapter working (unexpected response: {response.content.strip()})"

    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
            return False, f"API key issue: {error_msg}"
        else:
            return False, f"Adapter error: {error_msg}"


async def test_sql_generation(provider: str) -> tuple[bool, str]:
    """Test basic SQL generation."""
    try:
        from pydantic import BaseModel
        from src.adapters import create_adapter

        print(f"\n[3/3] Testing SQL generation with {provider}...")

        class SQLQuery(BaseModel):
            """Generated SQL query."""
            reasoning: str
            sql: str

        adapter = create_adapter(provider=provider)

        # Test structured output
        result = await adapter.generate_structured(
            prompt=(
                "Generate a SQL query to count the number of artists in the database. "
                "The table is named 'artist' with columns: artist_id, name."
            ),
            response_model=SQLQuery,
            system_prompt="You are a SQL expert. Generate valid PostgreSQL queries.",
        )

        sql = result.data.sql.strip().upper()
        if "SELECT" in sql and "COUNT" in sql and "ARTIST" in sql:
            return True, f"SQL generation working!\n      Generated: {result.data.sql}"
        else:
            return False, f"Unexpected SQL: {result.data.sql}"

    except Exception as e:
        return False, f"SQL generation error: {str(e)}"


async def main(provider: str) -> int:
    """Run all environment checks."""
    print("=" * 60)
    print("1.5 Environment Setup - Verification")
    print("=" * 60)

    # Check basic environment
    print("\n[1/3] Checking environment...")
    issues = check_environment()
    if issues:
        for issue in issues:
            print(f"      ERROR: {issue}")
        return 1
    print("      Python version OK")

    # Determine provider
    if provider is None:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            print("      ERROR: No API key found")
            return 1

    print(f"      Using provider: {provider}")

    # Check database
    db_ok, db_msg = check_database()
    if db_ok:
        print(f"      {db_msg}")
    else:
        print(f"      WARNING: {db_msg}")
        print("      (Continuing without database for now)")

    # Test adapter
    adapter_ok, adapter_msg = await test_adapter(provider)
    if adapter_ok:
        print(f"      {adapter_msg}")
    else:
        print(f"      ERROR: {adapter_msg}")
        return 1

    # Test SQL generation
    sql_ok, sql_msg = await test_sql_generation(provider)
    if sql_ok:
        print(f"      {sql_msg}")
    else:
        print(f"      ERROR: {sql_msg}")
        return 1

    # Success summary
    print("\n" + "=" * 60)
    print("SUCCESS! Environment is ready.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Read 1.1-1.4 for foundational concepts")
    print("  2. Continue to 2.1 to start building the agent")
    print(f"\nYour setup:")
    print(f"  Provider: {provider}")
    print(f"  Database: {'Connected' if db_ok else 'Not connected (optional for 0.5)'}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify environment setup")
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default=None,
        help="LLM provider to test (default: auto-detect from API keys)",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(main(args.provider))
    sys.exit(exit_code)
