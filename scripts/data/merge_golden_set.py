"""Merge the 65-query golden set into the training dataset.

Extracts non-negative golden queries, validates against Postgres,
and merges with existing sql_train.jsonl + sql_val.jsonl.
Re-splits into train/val with deduplication.

Usage:
  source .venv/bin/activate && python scripts/data/merge_golden_set.py
"""

import asyncio
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
SEED = 42


async def main():
    import asyncpg
    from evals.chinook_golden_set import GOLDEN_SET
    from src.utils.config import load_config

    config = load_config()
    db_url = config.database.url
    pool = await asyncpg.create_pool(db_url)

    # ── Step 1: Extract non-negative golden queries ──
    golden_with_sql = [q for q in GOLDEN_SET if q.sql.strip()]
    print(f"Golden set: {len(GOLDEN_SET)} total, {len(golden_with_sql)} with SQL, "
          f"{len(GOLDEN_SET) - len(golden_with_sql)} negative (skipped)")

    # ── Step 2: Validate golden SQLs against Postgres ──
    print("\nValidating golden SQLs against PostgreSQL...")
    golden_records = []
    failures = []

    for q in golden_with_sql:
        sql = q.sql.strip()
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql)
                golden_records.append({
                    "question": q.question,
                    "gold_sql": sql,
                })
        except Exception as e:
            failures.append({"id": q.id, "question": q.question, "error": str(e)[:200]})

    print(f"  Passed: {len(golden_records)}/{len(golden_with_sql)}")
    if failures:
        print(f"  Failed: {len(failures)}")
        for f in failures:
            print(f"    {f['id']}: {f['error']}")
            print(f"      Q: {f['question']}")

    # ── Step 3: Load existing train + val ──
    existing = []
    for fname in ["sql_train.jsonl", "sql_val.jsonl"]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing.append(json.loads(line))
    print(f"\nExisting dataset: {len(existing)} examples")

    # ── Step 4: Merge with deduplication ──
    seen: set[str] = set()
    merged: list[dict] = []

    for record in existing:
        key = record["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            merged.append(record)

    added = 0
    for record in golden_records:
        key = record["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            merged.append(record)
            added += 1

    print(f"Golden queries added: {added} new "
          f"({len(golden_records) - added} already in dataset)")
    print(f"Total merged: {len(merged)}")

    # ── Step 5: Split into train/val ──
    rng = random.Random(SEED)
    rng.shuffle(merged)

    val_n = max(1, round(len(merged) * 0.13))  # ~13% for val
    val = merged[:val_n]
    train = merged[val_n:]

    print(f"\nNew split: train={len(train)}, val={len(val)}")

    # ── Step 6: Save ──
    for data, fname in [(train, "sql_train.jsonl"), (val, "sql_val.jsonl")]:
        fpath = DATA_DIR / fname
        with open(fpath, "w") as f:
            for item in data:
                record = {
                    "question": item["question"],
                    "gold_sql": item["gold_sql"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Saved {len(data)} → {fpath}")

    # ── Step 7: Re-validate all ──
    print(f"\nPost-merge validation...")
    for fname in ["sql_train.jsonl", "sql_val.jsonl"]:
        fpath = DATA_DIR / fname
        total = 0
        passed = 0
        fail_list = []

        with open(fpath) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                total += 1
                try:
                    async with pool.acquire() as conn:
                        rows = await conn.fetch(record["gold_sql"])
                        passed += 1
                except Exception as e:
                    fail_list.append({
                        "line": line_no,
                        "question": record["question"][:80],
                        "error": str(e)[:200],
                    })

        status = "ALL PASS" if not fail_list else f"{len(fail_list)} FAILED"
        print(f"  {fname}: {passed}/{total} ({status})")
        for f in fail_list[:5]:
            print(f"    Line {f['line']}: {f['error']}")
            print(f"      Q: {f['question']}")

    await pool.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
