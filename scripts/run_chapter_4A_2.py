#!/usr/bin/env python3
"""Chapter 4A.2 - APO runner wrapper."""

import importlib.util
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET = PROJECT_ROOT / "scripts" / "chapter_4A" / "run_apo.py"
REQUIRED_MODULES = {
    "agentlightning": "uv pip install 'agentlightning[apo]'",
}


def check_optional_deps() -> int:
    if platform.system() == "Darwin":
        print(
            "Chapter 4A depends on Agent Lightning, which is not supported on macOS yet.",
            file=sys.stderr,
        )
        print("Run this chapter on Linux instead.", file=sys.stderr)
        return 1

    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if not missing:
        return 0

    print("Missing optional dependencies for chapter 4A:", file=sys.stderr)
    for name in missing:
        print(f"  - {name}", file=sys.stderr)
    print(f"Install with: {REQUIRED_MODULES[missing[0]]}", file=sys.stderr)
    return 1


def main() -> int:
    if check_optional_deps():
        return 1
    forwarded = sys.argv[1:] or ["--beam-width", "2", "--rounds", "1"]
    cmd = [sys.executable, str(TARGET), *forwarded]
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
