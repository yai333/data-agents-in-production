"""Generate 500+ additional SQL training questions and merge with existing dataset.

Loads all existing questions (train + val) for deduplication,
generates new questions only, runs them through the chapter 3.3 agent,
validates against Postgres, then merges and re-splits.

Usage:
  source .venv/bin/activate && python scripts/data/augment_dataset.py
"""

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
SEED = 43  # Different seed from original run
BATCH_SIZE = 15

# Target ~625 new questions to get ~500 valid after 80% agent success
CATEGORIES = ["sales", "catalog", "playlist", "employee"]
DIFFICULTIES = ["single_table", "multi_join", "aggregation", "subquery"]

# Per category×difficulty targets (total ~625)
TARGETS = {
    ("sales", "single_table"): 30,
    ("sales", "multi_join"): 50,
    ("sales", "aggregation"): 50,
    ("sales", "subquery"): 55,
    ("catalog", "single_table"): 30,
    ("catalog", "multi_join"): 50,
    ("catalog", "aggregation"): 50,
    ("catalog", "subquery"): 55,
    ("playlist", "single_table"): 15,
    ("playlist", "multi_join"): 25,
    ("playlist", "aggregation"): 25,
    ("playlist", "subquery"): 30,
    ("employee", "single_table"): 15,
    ("employee", "multi_join"): 25,
    ("employee", "aggregation"): 25,
    ("employee", "subquery"): 30,
}


def load_existing_questions() -> set[str]:
    """Load all existing questions from train/val/test + golden set."""
    existing: set[str] = set()

    # From JSONL files
    for fname in ["sql_train.jsonl", "sql_val.jsonl", "sql_test_sample.jsonl"]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    existing.add(record["question"].strip().lower())

    # From golden set
    from evals.chinook_golden_set import GOLDEN_SET
    for gq in GOLDEN_SET:
        existing.add(gq.question.strip().lower())

    return existing


def load_existing_records() -> list[dict]:
    """Load existing train + val records."""
    records = []
    for fname in ["sql_train.jsonl", "sql_val.jsonl"]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ── Reuse from generate_sql_data.py ──────────────────

from scripts.data.generate_sql_data import (
    build_question_prompt,
    generate_questions,
    load_schema_summary,
    run_agent_for_sql,
    setup_agent,
    strip_schema_prefix,
    validate_sql_postgres,
)


async def generate_sql_for_questions(
    agent, questions: list[dict], pool, concurrency: int = 3,
) -> list[dict]:
    """Run the production agent on each question to produce gold SQL."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def process_one(item: dict) -> dict | None:
        async with semaphore:
            agent_result = await run_agent_for_sql(agent, item["question"])
            if agent_result is None:
                return None

            check = await validate_sql_postgres(agent_result["sql"], pool)
            if not check["valid"] or check["row_count"] == 0:
                return None

            return {
                "question": item["question"],
                "gold_sql": check["clean_sql"],
            }

    tasks = [process_one(q) for q in questions]

    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        if result is not None:
            results.append(result)
        if completed % 25 == 0:
            print(f"    Progress: {completed}/{len(questions)} "
                  f"({len(results)} valid)")

    return results


async def validate_all(pool, fpath: Path) -> tuple[int, int, list[dict]]:
    """Validate every SQL in a JSONL file."""
    total = 0
    failures = []
    with open(fpath) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            try:
                async with pool.acquire() as conn:
                    await conn.fetch(record["gold_sql"])
            except Exception as e:
                failures.append({
                    "line": line_no,
                    "question": record["question"][:80],
                    "error": str(e)[:200],
                })
    return total, total - len(failures), failures


def save_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            record = {
                "question": item["question"],
                "gold_sql": item["gold_sql"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} → {path}")


async def main():
    import asyncpg

    print("=" * 60)
    print("Augment SQL Dataset (+500 questions)")
    print("=" * 60)

    # Step 1: Load existing questions for dedup
    print("\n[1/6] Loading existing questions for deduplication...")
    existing_questions = load_existing_questions()
    existing_records = load_existing_records()
    print(f"  Existing questions: {len(existing_questions)}")
    print(f"  Existing records (train+val): {len(existing_records)}")

    # Step 2: Load schema
    print("\n[2/6] Loading schema summary...")
    schema_summary = load_schema_summary()

    # Step 3: Generate new questions (Phase 1)
    print("\n[3/6] Phase 1: Generating new questions via LLM...")
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.laozhang.ai/v1"),
    )

    all_questions: set[str] = set(existing_questions)
    question_pool: list[dict] = []

    for category in CATEGORIES:
        for difficulty in DIFFICULTIES:
            target = TARGETS[(category, difficulty)]
            cat_generated = 0
            batch_num = 0

            while cat_generated < target:
                batch_num += 1
                remaining = target - cat_generated
                batch_n = min(BATCH_SIZE, remaining + 3)

                print(
                    f"  [{category}/{difficulty}] Batch {batch_num}: "
                    f"generating {batch_n} (have {cat_generated}/{target})..."
                )

                batch = generate_questions(
                    client, category, difficulty,
                    schema_summary, list(all_questions)[:80],
                    batch_n,
                )

                new_in_batch = 0
                for item in batch:
                    key = item["question"].strip().lower()
                    if key not in all_questions:
                        all_questions.add(key)
                        question_pool.append(item)
                        cat_generated += 1
                        new_in_batch += 1
                        if cat_generated >= target:
                            break

                print(f"    Got {len(batch)}, {new_in_batch} new "
                      f"(total: {cat_generated}/{target})")

                time.sleep(0.5)
                if batch_num >= 15:
                    print(f"    Max batches reached for {category}/{difficulty}")
                    break

    print(f"\n  New questions generated: {len(question_pool)}")

    # Step 4: Set up production agent
    print("\n[4/6] Setting up chapter 3.3 production agent...")
    agent, pool, store_cm, checkpointer_cm = await setup_agent()

    try:
        # Step 5: Run agent on new questions (Phase 2)
        print(f"\n[5/6] Phase 2: Running agent on "
              f"{len(question_pool)} new questions...")
        new_records = await generate_sql_for_questions(
            agent, question_pool, pool, concurrency=3,
        )

        success_rate = len(new_records) / max(1, len(question_pool)) * 100
        print(f"\n  Agent success rate: {len(new_records)}/{len(question_pool)} "
              f"({success_rate:.0f}%)")

        # Step 6: Merge and re-split
        print("\n[6/6] Merging with existing dataset...")

        # Dedup merge
        seen: set[str] = set()
        merged: list[dict] = []

        for record in existing_records:
            key = record["question"].strip().lower()
            if key not in seen:
                seen.add(key)
                merged.append(record)

        added = 0
        for record in new_records:
            key = record["question"].strip().lower()
            if key not in seen:
                seen.add(key)
                merged.append(record)
                added += 1

        duped = len(new_records) - added
        print(f"  Existing: {len(existing_records)}")
        print(f"  New valid: {len(new_records)}, added: {added}"
              f"{f', deduped: {duped}' if duped else ''}")
        print(f"  Total merged: {len(merged)}")

        # Split: ~13% val
        rng = random.Random(SEED)
        rng.shuffle(merged)
        val_n = max(1, round(len(merged) * 0.13))
        val = merged[:val_n]
        train = merged[val_n:]

        print(f"\n  New split: train={len(train)}, val={len(val)}")

        save_jsonl(train, DATA_DIR / "sql_train.jsonl")
        save_jsonl(val, DATA_DIR / "sql_val.jsonl")

        # Verify: no duplicates, no golden leakage
        train_q = {r["question"].strip().lower() for r in train}
        val_q = {r["question"].strip().lower() for r in val}
        cross_dups = train_q & val_q
        print(f"\n  Cross-split duplicates: {len(cross_dups)}")

        # Post-merge validation
        from src.utils.config import load_config
        config = load_config()
        db_url = config.database.url
        vpool = await asyncpg.create_pool(db_url)

        print(f"\nPost-merge SQL validation...")
        try:
            for fname in ["sql_train.jsonl", "sql_val.jsonl"]:
                fpath = DATA_DIR / fname
                total, passed, failures = await validate_all(vpool, fpath)
                status = "ALL PASS" if not failures else f"{len(failures)} FAILED"
                print(f"  {fname}: {passed}/{total} ({status})")
                for f in failures[:5]:
                    print(f"    Line {f['line']}: {f['error']}")
                    print(f"      Q: {f['question']}")
        finally:
            await vpool.close()

        print("\nDone!")

    finally:
        await checkpointer_cm.__aexit__(None, None, None)
        await store_cm.__aexit__(None, None, None)
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
