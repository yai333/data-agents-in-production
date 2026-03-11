"""CLI entry point for the Text-to-SQL agent.

Usage:
    python -m src.agent.cli "How many artists are there?"
    python -m src.agent.cli --schema data/chinook_schema.json "List all genres"
    sql-agent "Show top 5 customers by total spending"
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


def _find_schema_path() -> Path:
    """Locate the default schema file relative to the project root."""
    root = Path(__file__).resolve().parent.parent.parent
    return root / "data" / "chinook_schema.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sql-agent",
        description="Ask natural-language questions against the Chinook database.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="The question to ask. If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="Path to schema JSON file (default: data/chinook_schema.json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output raw JSON instead of formatted text",
    )
    return parser


async def _run(question: str, schema_path: Path, json_output: bool) -> int:
    from src.agent.graph import run_agent
    from src.schema.store import SchemaStore

    store = SchemaStore(schema_path)
    if not store.tables:
        print(f"Error: no tables loaded from {schema_path}", file=sys.stderr)
        return 1

    result = await run_agent(question, store)

    if json_output:
        out = {k: v for k, v in result.items() if k != "rows"}
        out["row_count"] = result.get("row_count", 0)
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"\nQuestion: {question}")
        if result.get("sql"):
            print(f"SQL:      {result['sql']}")
        print(f"Answer:   {result.get('answer', 'No answer')}")
        if result.get("row_count"):
            print(f"Rows:     {result['row_count']}")

    return 0


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    question = args.question
    if not question:
        try:
            question = input("Enter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

    if not question:
        parser.print_help()
        sys.exit(1)

    schema_path = args.schema or _find_schema_path()
    if not schema_path.exists():
        print(f"Error: schema file not found at {schema_path}", file=sys.stderr)
        print("Run with --schema <path> or ensure data/chinook_schema.json exists.", file=sys.stderr)
        sys.exit(1)

    exit_code = asyncio.run(_run(question, schema_path, args.json_output))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
