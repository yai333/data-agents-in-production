#!/usr/bin/env python3
"""Chapter 4B.1 — SFT notebook launcher/executor."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = PROJECT_ROOT / "scripts" / "chapter_4B" / "finetune_text2sql.ipynb"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run chapter 4B.1 notebook")
    parser.add_argument("--execute", action="store_true", help="Execute the notebook via jupyter nbconvert")
    parser.add_argument("--output", default="finetune_text2sql.executed.ipynb", help="Output notebook name when --execute is used")
    args = parser.parse_args()

    if not NOTEBOOK.exists():
        print(f"Notebook not found: {NOTEBOOK}", file=sys.stderr)
        return 1

    if not args.execute:
        print(f"Notebook: {NOTEBOOK}")
        print("Run with:")
        print(f"  python -m jupyter lab {NOTEBOOK}")
        print("Or execute non-interactively:")
        print(f"  python scripts/run_chapter_4B_1.py --execute --output {args.output}")
        return 0

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(NOTEBOOK),
        "--output",
        args.output,
    ]
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
