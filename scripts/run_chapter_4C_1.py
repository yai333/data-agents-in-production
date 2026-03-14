#!/usr/bin/env python3
"""Chapter 4C.1 — simple RL reward-signal demo."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.chapter_4C.reward_sql import compute_sql_reward

GOLD_SQL = "SELECT COUNT(*) FROM artist"
SAMPLES = {
    "correct": """```sql
SELECT COUNT(*) FROM artist
```""",
    "close_but_wrong": """```sql
SELECT COUNT(artist_id) FROM album
```""",
    "format_fail": "The answer is probably a count over the artist table.",
}


def main() -> int:
    db_url = os.getenv("CHINOOK_DATABASE_URL")
    print("Gold SQL:")
    print(f"  {GOLD_SQL}")
    print()
    for label, candidate in SAMPLES.items():
        result = compute_sql_reward(candidate, GOLD_SQL, db_url)
        print(f"[{label}]")
        print(f"  total={result['total']:.3f}")
        print(f"  format_ok={result['format_ok']}")
        print(f"  executed={result['executed']}")
        print(f"  component_f1={result['component_f1']:.3f}")
        print(f"  execution_match={result['execution_match']}")
        detail = result.get('detail')
        if detail:
            print(f"  detail={detail}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
