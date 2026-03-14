#!/usr/bin/env python3
"""Chapter 4D — workflow optimization helper.

This chapter combines earlier building blocks rather than introducing one
new standalone subsystem. The wrapper exposes the most relevant runnable
paths so readers can exercise the workflow-oriented pieces directly.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTEXT_PIPELINE = PROJECT_ROOT / "scripts" / "run_context_pipeline.py"
APO_RUNNER = PROJECT_ROOT / "scripts" / "run_chapter_4A_2.py"
RL_RUNNER = PROJECT_ROOT / "scripts" / "run_chapter_4C_3.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run chapter 4D helper flows")
    parser.add_argument("--context-dry-run", action="store_true", help="Dry-run the offline context pipeline")
    parser.add_argument("--apo-quick", action="store_true", help="Run the quick APO wrapper")
    parser.add_argument("--rl-dry-run", action="store_true", help="Run the SQL GRPO dry-run wrapper")
    args = parser.parse_args()

    if args.context_dry_run:
        return subprocess.run([sys.executable, str(CONTEXT_PIPELINE), "--dry-run"], cwd=PROJECT_ROOT).returncode
    if args.apo_quick:
        return subprocess.run([sys.executable, str(APO_RUNNER)], cwd=PROJECT_ROOT).returncode
    if args.rl_dry_run:
        return subprocess.run([sys.executable, str(RL_RUNNER)], cwd=PROJECT_ROOT).returncode

    print("Chapter 4D combines earlier workflow pieces. Common entrypoints:")
    print(f"  python {CONTEXT_PIPELINE.relative_to(PROJECT_ROOT)} --dry-run")
    print("  python scripts/run_chapter_4A_2.py")
    print("  python scripts/run_chapter_4C_3.py --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
