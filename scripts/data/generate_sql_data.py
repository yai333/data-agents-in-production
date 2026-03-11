"""Generate SQL training dataset for Chapter 4C.

Two-phase pipeline:
  Phase 1: Generate natural language questions via LLM (questions only)
  Phase 2: Run each question through the chapter 3 production agent
           (disambiguate → retrieve schema → generate SQL → evaluate)
           to produce gold SQL

The gold SQL comes from the same agent architecture readers built
in chapters 2-3. Only questions where the agent produces executed,
validated SQL become training examples.

Deduplicates against the golden set (held out as test).
Splits into train/val with category stratification.

Prerequisites:
  make db-up                                   # Chinook DB with pgvector
  python scripts/run_chapter_3_3.py --offline  # Build the MDL index

Output format (one per line):
  {"question": "...", "gold_sql": "..."}

Usage:
  source .venv/bin/activate && python scripts/data/generate_sql_data.py
"""

import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ── Configuration ────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

CATEGORIES = ["sales", "catalog", "playlist", "employee"]
DIFFICULTIES = ["single_table", "multi_join", "aggregation", "subquery"]

TARGET_TOTAL = 575  # 500 train + 75 val
TRAIN_N = 500
VAL_N = 75
BATCH_SIZE = 15
SEED = 42



# ── Load golden set questions (to exclude from training) ──

def load_golden_questions() -> set[str]:
    """Load golden set questions for deduplication."""
    from evals.chinook_golden_set import GOLDEN_SET
    return {gq.question.strip().lower() for gq in GOLDEN_SET}


def load_existing_questions(data_dir: Path) -> set[str]:
    """Load questions from existing train/val JSONL files for dedup."""
    existing: set[str] = set()
    for fname in ["sql_train.jsonl", "sql_val.jsonl"]:
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    existing.add(record["question"].strip().lower())
    return existing


# ── Schema summary for question generation prompt ────

def load_schema_summary() -> str:
    """Load Chinook schema as a compact summary for the question generator."""
    schema_path = ROOT / "config" / "chinook_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)

    lines = []
    for table in schema["tables"]:
        cols = ", ".join(
            f"{c['name']} ({c['data_type'].split('(')[0]})"
            for c in table["columns"]
        )
        rels = []
        for r in table.get("relationships", []):
            rels.append(r["condition"])
        rel_str = "; ".join(rels) if rels else "none"
        lines.append(
            f"- {table['name']}: {table['description']}\n"
            f"  Columns: {cols}\n"
            f"  Relationships: {rel_str}"
        )
    return "\n".join(lines)


# ── SQL validation (SQLite, for final check) ─────────

def strip_schema_prefix(sql: str) -> str:
    """Remove 'chinook.' schema prefix from SQL.

    The production agent generates SQL with the MDL schema prefix
    (chinook.table), but tables live in the public schema.
    """
    return re.sub(r'\bchinook\.', '', sql)


async def validate_sql_postgres(sql: str, pool) -> dict:
    """Execute SQL against Chinook PostgreSQL and return result info.

    Strips chinook. schema prefix (tables live in public schema).
    """
    clean_sql = strip_schema_prefix(sql)
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(clean_sql)
            return {"valid": True, "row_count": len(rows), "error": None,
                    "clean_sql": clean_sql}
    except Exception as e:
        return {"valid": False, "row_count": 0, "error": str(e),
                "clean_sql": clean_sql}


# ── Phase 1: Generate questions only ──────────────────

def get_client() -> OpenAI:
    """Create OpenAI client."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE", "https://api.laozhang.ai/v1"),
    )


def build_question_prompt(
    category: str,
    difficulty: str,
    schema_summary: str,
    existing_questions: list[str],
    n: int = 15,
) -> str:
    """Build prompt that generates questions only (no SQL)."""
    existing_sample = existing_questions[:30]
    existing_str = "\n".join(f"- {q}" for q in existing_sample)

    difficulty_guidance = {
        "single_table": "Questions answerable from a single table (SELECT, WHERE, COUNT, SUM, AVG). No JOINs needed.",
        "multi_join": "Questions requiring 2-4 table JOINs to answer.",
        "aggregation": "Questions needing GROUP BY, HAVING, multiple aggregations, ORDER BY.",
        "subquery": "Questions needing subqueries, CTEs, window functions, or CASE WHEN.",
    }

    category_guidance = {
        "sales": "Questions about customers, invoices, purchases, revenue, spending patterns, billing.",
        "catalog": "Questions about artists, albums, tracks, genres, media types, song durations.",
        "playlist": "Questions about playlists, track membership, playlist composition.",
        "employee": "Questions about employees, org hierarchy, support representatives, reporting lines.",
    }

    return f"""You generate natural language questions for a music store database.

## Schema
{schema_summary}

## Task
Generate exactly {n} questions that a business user would ask.
Category: {category} — {category_guidance[category]}
Difficulty: {difficulty} — {difficulty_guidance[difficulty]}

## Requirements
1. Questions must be natural, varied, and realistic
2. Each question should be answerable using the Chinook schema above
3. Vary phrasing — don't start every question with "How many" or "List all"
4. Do NOT duplicate these existing questions:
{existing_str}

## Output format (JSON array of strings):
["What is the total revenue by country?", "Which artist has the most albums?"]

Output ONLY the JSON array, no other text."""


def generate_questions(
    client: OpenAI,
    category: str,
    difficulty: str,
    schema_summary: str,
    existing_questions: list[str],
    n: int = 15,
    max_retries: int = 3,
) -> list[dict]:
    """Generate a batch of questions (no SQL)."""
    prompt = build_question_prompt(
        category, difficulty, schema_summary,
        existing_questions, n,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
                messages=[
                    {"role": "system", "content": "You generate questions. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=2000,
            )

            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            questions = json.loads(text)
            if not isinstance(questions, list):
                print(f"  Warning: Expected list, got {type(questions)}")
                continue

            return [
                {"question": q.strip(), "category": category, "difficulty": difficulty}
                for q in questions
                if isinstance(q, str) and q.strip()
            ]

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt + 1}): {e}")
            time.sleep(2)
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}): {e}")
            time.sleep(5)

    return []


# ── Phase 2: Run production agent to generate SQL ────

async def setup_agent():
    """Set up the chapter 3.3 production agent.

    Reuses the same infrastructure as run_chapter_3_3.py:
    disambiguate → cache check → retrieve schema → generate SQL → evaluate.
    No chart node — we only need the SQL.
    """
    import asyncpg
    from langchain_community.utilities.sql_database import SQLDatabase
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore

    from src.adapters import get_model_name, get_provider_name
    from src.retrieval.pgvector_store import create_embeddings
    from src.schema import HybridSchemaStore
    from src.utils.config import load_config

    # Import the agent builder from chapter 3.3
    from scripts.run_chapter_3_3 import build_agent_with_hybrid_tools

    config = load_config()
    db_url = config.database.url
    pool = await asyncpg.create_pool(db_url)

    provider = get_provider_name()
    embeddings = create_embeddings(provider)
    hybrid_store = HybridSchemaStore(pool, embeddings)

    model_name = get_model_name()
    if provider == "openai":
        llm = ChatOpenAI(model=model_name, temperature=0)
    else:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    db = SQLDatabase.from_uri(db_url)

    store_cm = AsyncPostgresStore.from_conn_string(db_url)
    checkpointer_cm = AsyncPostgresSaver.from_conn_string(db_url)
    store = await store_cm.__aenter__()
    checkpointer = await checkpointer_cm.__aenter__()
    await store.setup()
    await checkpointer.setup()

    agent = build_agent_with_hybrid_tools(
        db, llm, hybrid_store,
        store=store, checkpointer=checkpointer,
    )

    print(f"  Agent ready: {provider}:{model_name}")
    return agent, pool, store_cm, checkpointer_cm


async def run_agent_for_sql(agent, question: str) -> dict | None:
    """Run the chapter 3.3 agent on a question and extract SQL.

    Returns {"sql": ..., "score": ...} or None if agent fails.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    initial_state = {
        "original_question": question,
        "disambiguated_question": "",
        "cache_hit": False,
        "cached_sql": "",
        "tables_used": [],
        "schema_overview": "",
        "sql": "",
        "results": "",
        "evaluation_score": 0.0,
        "response": "",
        "messages": [
            SystemMessage(content="You are a SQL agent."),
            HumanMessage(content=question),
        ],
    }

    try:
        result = await agent.ainvoke(
            initial_state,
            {"configurable": {"thread_id": f"gen-{hash(question)}"},
             "recursion_limit": 50},
        )

        sql = result.get("sql", "")
        score = result.get("evaluation_score", 0.0)

        if not sql or score < 0.5:
            return None

        return {"sql": sql, "score": score}

    except Exception as e:
        print(f"    Agent error: {str(e)[:80]}")
        return None


async def generate_sql_for_questions(
    agent,
    questions: list[dict],
    pool,
    concurrency: int = 5,
) -> list[dict]:
    """Run the production agent on each question to produce gold SQL."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def process_one(item: dict) -> dict | None:
        async with semaphore:
            agent_result = await run_agent_for_sql(
                agent, item["question"],
            )
            if agent_result is None:
                return None

            # Validate by executing against PostgreSQL
            check = await validate_sql_postgres(agent_result["sql"], pool)
            if not check["valid"] or check["row_count"] == 0:
                return None

            return {
                "question": item["question"],
                "gold_sql": check["clean_sql"],
                "category": item.get("category", "catalog"),
                "difficulty": item.get("difficulty", "multi_join"),
                "gold_row_count": check["row_count"],
                "agent_score": agent_result["score"],
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


# ── Split dataset ────────────────────────────────────

def stratified_split(
    examples: list[dict],
    train_n: int,
    val_n: int,
    seed: int = 42,
) -> tuple[list, list]:
    """Split into train/val with stratification by category."""
    rng = random.Random(seed)

    by_cat: dict[str, list[dict]] = {}
    for ex in examples:
        cat = ex.get("category", "catalog")
        by_cat.setdefault(cat, []).append(ex)

    for cat in by_cat:
        rng.shuffle(by_cat[cat])

    total = sum(len(v) for v in by_cat.values())
    train, val = [], []

    for cat, cat_examples in by_cat.items():
        n = len(cat_examples)
        cat_val_n = max(1, round(n * val_n / total))
        val.extend(cat_examples[:cat_val_n])
        train.extend(cat_examples[cat_val_n:])

    rng.shuffle(train)
    rng.shuffle(val)

    if len(train) > train_n:
        train = train[:train_n]
    if len(val) > val_n:
        val = val[:val_n]

    return train, val


# ── Save JSONL ───────────────────────────────────────

def save_jsonl(data: list[dict], path: Path) -> None:
    """Write examples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            record = {
                "question": item["question"],
                "gold_sql": item["gold_sql"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} examples to {path}")


# ── Post-generation validation ──────────────────────

async def validate_dataset(jsonl_path: Path, db_url: str) -> tuple[int, int, list[dict]]:
    """Execute every golden SQL in a JSONL file against Postgres.

    Returns (total, passed, failures) where failures is a list of
    {"line": int, "question": str, "sql": str, "error": str}.
    """
    import asyncpg

    pool = await asyncpg.create_pool(db_url)
    failures: list[dict] = []
    total = 0

    try:
        with open(jsonl_path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                total += 1
                sql = record["gold_sql"]

                try:
                    async with pool.acquire() as conn:
                        rows = await conn.fetch(sql)
                        if len(rows) == 0:
                            failures.append({
                                "line": line_no,
                                "question": record["question"][:80],
                                "sql": sql[:120],
                                "error": "0 rows returned",
                            })
                except Exception as e:
                    failures.append({
                        "line": line_no,
                        "question": record["question"][:80],
                        "sql": sql[:120],
                        "error": str(e)[:200],
                    })
    finally:
        await pool.close()

    passed = total - len(failures)
    return total, passed, failures


def deduplicate_questions(questions: list[dict]) -> list[dict]:
    """Remove duplicate questions (case-insensitive)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for q in questions:
        key = q["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


# ── Main ─────────────────────────────────────────────

async def async_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate SQL training data for Chinook database",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent agent runs",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing datasets (skip generation)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # ── Validate-only mode ──────────────────────────
    if args.validate_only:
        from src.utils.config import load_config
        config = load_config()
        db_url = config.database.url

        print("=" * 60)
        print("Validating existing datasets against PostgreSQL")
        print("=" * 60)

        for fname in ["sql_train.jsonl", "sql_val.jsonl", "sql_test_sample.jsonl"]:
            fpath = output_dir / fname
            if not fpath.exists():
                print(f"\n  {fname}: SKIPPED (not found)")
                continue

            total, passed, failures = await validate_dataset(fpath, db_url)
            status = "ALL PASS" if not failures else f"{len(failures)} FAILED"
            print(f"\n  {fname}: {passed}/{total} ({status})")

            for f in failures:
                print(f"    Line {f['line']}: {f['error']}")
                print(f"      Q: {f['question']}")
                print(f"      SQL: {f['sql']}")

        return

    # ── Full generation mode ────────────────────────
    print("=" * 60)
    print("SQL Training Data Generator (two-phase)")
    print("  Phase 1: LLM generates questions")
    print("  Phase 2: Chapter 3 agent generates SQL")
    print("=" * 60)

    # Step 1: Load golden set for dedup
    print("\n[1/6] Loading golden set for deduplication...")
    golden_questions = load_golden_questions()
    print(f"  Golden set: {len(golden_questions)} questions (held out)")

    # Step 2: Load schema
    print("\n[2/6] Loading schema summary...")
    schema_summary = load_schema_summary()

    # Step 3: Generate questions (Phase 1)
    print("\n[3/6] Phase 1: Generating questions via LLM...")
    client = get_client()
    all_questions: set[str] = set(golden_questions)
    question_pool: list[dict] = []

    for category in CATEGORIES:
        for difficulty in DIFFICULTIES:
            # Over-generate to account for agent failures
            target = {
                "single_table": 30,
                "multi_join": 55,
                "aggregation": 55,
                "subquery": 60,
            }[difficulty]

            if category in ("playlist", "employee"):
                target = max(8, target // 2)

            cat_generated = 0
            batch_num = 0

            while cat_generated < target:
                batch_num += 1
                remaining = target - cat_generated
                batch_n = min(BATCH_SIZE, remaining + 3)

                print(
                    f"  [{category}/{difficulty}] Batch {batch_num}: "
                    f"generating {batch_n} questions "
                    f"(have {cat_generated}/{target})..."
                )

                batch = generate_questions(
                    client, category, difficulty,
                    schema_summary, list(all_questions)[:50],
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

                print(
                    f"    Got {len(batch)}, {new_in_batch} new "
                    f"(total: {cat_generated}/{target})"
                )

                time.sleep(0.5)
                if batch_num >= 10:
                    print(f"    Max batches reached for {category}/{difficulty}")
                    break

    # Final dedup pass (belt and suspenders)
    question_pool = deduplicate_questions(question_pool)
    print(f"\n  Total unique questions: {len(question_pool)}")

    # Step 4: Set up production agent
    print("\n[4/6] Setting up chapter 3.3 production agent...")
    agent, pool, store_cm, checkpointer_cm = await setup_agent()

    # Step 5: Run agent on each question (Phase 2)
    try:
        print(f"\n[5/6] Phase 2: Running agent on "
              f"{len(question_pool)} questions...")
        generated = await generate_sql_for_questions(
            agent, question_pool, pool,
            concurrency=args.concurrency,
        )

        success_rate = len(generated) / len(question_pool) * 100
        print(f"\n  Agent success rate: {len(generated)}/{len(question_pool)} "
              f"({success_rate:.0f}%)")

        cat_counts: dict[str, int] = {}
        for ex in generated:
            cat = ex.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        print(f"  Category distribution: {cat_counts}")

        diff_counts: dict[str, int] = {}
        for ex in generated:
            diff = ex.get("difficulty", "unknown")
            diff_counts[diff] = diff_counts.get(diff, 0) + 1
        print(f"  Difficulty distribution: {diff_counts}")

        # Step 6: Split and save
        print("\n[6/6] Splitting and saving...")
        train, val = stratified_split(generated, TRAIN_N, VAL_N, SEED)

        save_jsonl(train, output_dir / "sql_train.jsonl")
        save_jsonl(val, output_dir / "sql_val.jsonl")

        # Verification: duplicates and leakage
        print(f"\n{'=' * 60}")
        print("Verification")
        print("=" * 60)
        print(f"  train: {len(train)}, val: {len(val)}")

        train_q = {ex["question"].strip().lower() for ex in train}
        val_q = {ex["question"].strip().lower() for ex in val}
        golden_leak = (train_q | val_q) & golden_questions
        print(f"  Golden set leakage: {len(golden_leak)} "
              f"{'(CLEAN)' if not golden_leak else '(LEAK!)'}")

        cross_dups = train_q & val_q
        print(f"  Cross-split duplicates: {len(cross_dups)}")

        # Post-generation validation: re-execute every golden SQL
        from src.utils.config import load_config
        config = load_config()
        db_url = config.database.url

        print(f"\n{'=' * 60}")
        print("Post-generation SQL validation")
        print("=" * 60)

        for fname in ["sql_train.jsonl", "sql_val.jsonl"]:
            fpath = output_dir / fname
            if not fpath.exists():
                continue
            total, passed, failures = await validate_dataset(fpath, db_url)
            status = "ALL PASS" if not failures else f"{len(failures)} FAILED"
            print(f"  {fname}: {passed}/{total} ({status})")

            if failures:
                for f in failures[:10]:
                    print(f"    Line {f['line']}: {f['error']}")
                    print(f"      Q: {f['question']}")
                if len(failures) > 10:
                    print(f"    ... and {len(failures) - 10} more")

        print("\nDone!")

    finally:
        await checkpointer_cm.__aexit__(None, None, None)
        await store_cm.__aexit__(None, None, None)
        await pool.close()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
