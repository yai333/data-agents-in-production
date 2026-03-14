#!/usr/bin/env python3
"""Chapter 4C.2 — reward design demo with multiple candidate SQL outputs."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.chapter_4C.reward_sql import compute_sql_reward

GOLD_SQL = "SELECT name FROM genre ORDER BY name"
CANDIDATES = {
    "exact": """```sql
SELECT name FROM genre ORDER BY name
```""",
    "missing_sort": """```sql
SELECT name FROM genre
```""",
    "wrong_table": """```sql
SELECT title FROM album ORDER BY title
```""",
}


def main() -> int:
    db_url = os.getenv("CHINOOK_DATABASE_URL")
    print("Reward design demo")
    print(f"Gold SQL: {GOLD_SQL}")
    print()
    for label, candidate in CANDIDATES.items():
        reward = compute_sql_reward(candidate, GOLD_SQL, db_url)
        print(f"[{label}]")
        print(f"  total={reward['total']:.3f}")
        print(f"  component_f1={reward['component_f1']:.3f}")
        print(f"  execution_match={reward['execution_match']}")
        print(f"  executed={reward['executed']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
