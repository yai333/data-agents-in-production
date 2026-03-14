#!/usr/bin/env python3
"""Chapter 4C.3 - full SQL-agent GRPO wrapper."""

import importlib.util
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET = PROJECT_ROOT / "scripts" / "chapter_4C" / "train_grpo_sql.py"
REQUIRED_MODULES = {
    "agentlightning": "uv pip install -e '.[grpo]'",
}


def check_optional_deps() -> int:
    if platform.system() == "Darwin":
        print(
            "Chapter 4C.3 depends on Agent Lightning and VERL, which are not supported on macOS yet.",
            file=sys.stderr,
        )
        print("Run this chapter on Linux instead.", file=sys.stderr)
        return 1

    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if not missing:
        return 0

    print("Missing optional dependencies for chapter 4C.3:", file=sys.stderr)
    for name in missing:
        print(f"  - {name}", file=sys.stderr)
    print(f"Install with: {REQUIRED_MODULES[missing[0]]}", file=sys.stderr)
    return 1


def main() -> int:
    if check_optional_deps():
        return 1
    forwarded = sys.argv[1:] or ["--dry-run"]
    cmd = [sys.executable, str(TARGET), *forwarded]
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
