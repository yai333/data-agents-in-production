"""Chapter 3.7 — Post-SQL UX with A2UI + Google ADK.

Starts the A2UI SQL Explorer agent. Two modes:

  Server mode (default):
    python scripts/run_chapter_3_7.py
    → Starts A2A server on http://localhost:10003
    → Connect with an A2UI client for interactive DataTable

  CLI mode:
    python scripts/run_chapter_3_7.py --cli "Show all artists"
    → Text-only agent, no UI, prints markdown table

Requirements:
    pip install google-adk a2a-sdk a2ui litellm jsonschema
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "chapter_3_7")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CHAPTER_DIR)

REQUIRED_PACKAGES = {
    "google.adk": "google-adk",
    "a2a": "a2a-sdk",
    "a2ui": "a2ui",
    "litellm": "litellm",
    "jsonschema": "jsonschema",
}


def check_deps() -> list[str]:
    """Check for required packages, return list of missing ones."""
    missing = []
    for module, package in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    return missing


def get_db_path() -> str:
    """Get the path to the Chinook SQLite database."""
    db_path = os.getenv("CHINOOK_DB_PATH")
    if db_path and os.path.exists(db_path):
        return db_path
    default = os.path.join(PROJECT_ROOT, "data", "chinook.db")
    if os.path.exists(default):
        return default
    print(f"ERROR: Chinook database not found at {default}")
    print("Download from: https://github.com/lerocha/chinook-database")
    print("Place the SQLite version at data/chinook.db")
    sys.exit(1)


async def run_cli(query: str) -> None:
    """Run a single query in text-only mode (no UI, no server)."""
    from agent import SQLExplorerAgent

    db_path = get_db_path()
    os.environ.setdefault("CHINOOK_DB_PATH", db_path)

    from sql_session_manager import get_session_manager
    get_session_manager(db_path)

    agent = SQLExplorerAgent(base_url="http://localhost:10003", use_ui=False)

    print(f"Query: {query}")
    print("---")

    async for item in agent.stream(query, session_id="cli-session"):
        if item["is_task_complete"]:
            print(item["content"])
        else:
            print(f"  ... {item.get('updates', 'processing')}")


def run_server(host: str = "localhost", port: int = 10003) -> None:
    """Start the A2A server with A2UI support."""
    db_path = get_db_path()
    os.environ.setdefault("CHINOOK_DB_PATH", db_path)

    from __main__ import main as server_main

    print(f"Starting SQL Explorer Agent...")
    print(f"  Database: {db_path}")
    print(f"  Server:   http://{host}:{port}")
    print(f"  Model:    {os.getenv('LITELLM_MODEL', 'openai/gpt-5.1-mini')}")
    print()

    server_main(standalone_mode=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Chapter 3.7 — A2UI SQL Explorer Agent"
    )
    parser.add_argument(
        "--cli",
        type=str,
        metavar="QUERY",
        help='Run a single query in CLI mode (e.g., --cli "Show all artists")',
    )
    parser.add_argument(
        "--host", default="localhost", help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=10003, help="Server port (default: 10003)"
    )

    args = parser.parse_args()

    missing = check_deps()
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        sys.exit(1)

    if args.cli:
        asyncio.run(run_cli(args.cli))
    else:
        run_server(args.host, args.port)


if __name__ == "__main__":
    main()
