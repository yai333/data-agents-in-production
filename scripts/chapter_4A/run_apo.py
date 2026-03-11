#!/usr/bin/env python3
"""4A Automatic Prompt Optimization -- Run APO with Agent Lightning.

Uses the Chinook golden-set dataset as training data,
pre-computes schema retrieval via HybridSchemaStore (same as 3.5),
splits into train/val/test, and runs beam search over the prompt space.

The rollout executes the full chapter 3.5 LangGraph pipeline
(generate_sql → sql_tools → evaluate → eval_tools) for each task.
APO optimizes the SQL generation system prompt while everything else
stays fixed.

Tracing uses Agent Lightning's built-in AgentOps + dashboard (port 4747).
No Langfuse needed — avoids TracerProvider conflicts entirely.

Prerequisites:
    make db-up
    python scripts/run_chapter_3_3.py --offline   # builds MDL + embeddings
    pip install agentlightning[apo] sqlglot

Usage:
    source .venv/bin/activate
    python scripts/chapter_4A/run_apo.py
    python scripts/chapter_4A/run_apo.py --beam-width 2 --rounds 1  # quick test
    python scripts/chapter_4A/run_apo.py --dev                      # dry-run 3 tasks
"""

import argparse
import asyncio
import multiprocessing
import os
import sys
import time
from pathlib import Path

# macOS defaults to "spawn" which can't pickle nested functions used by
# ClientServerExecutionStrategy. "fork" avoids this pickling limitation.
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass  # already set

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import asyncpg
from openai import AsyncOpenAI

import agentlightning as agl
from agentlightning.algorithm.apo import APO
from agentlightning.adapter import TraceToMessages
from agentlightning.tracer import OtelTracer

from src.adapters import get_model_name, get_provider_name
from src.retrieval.pgvector_store import create_embeddings
from src.schema import HybridSchemaStore

from agent_rollout import sql_agent_rollout, baseline_prompt_template


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def load_dataset() -> list[dict]:
    """Load the golden-set dataset from the Python file."""
    from evals.chinook_golden_set import GOLDEN_SET

    items = [
        {
            "question": q.question,
            "expected_sql": q.sql,
            "difficulty": q.difficulty,
            "category": q.category,
        }
        for q in GOLDEN_SET
        if q.sql.strip()
    ]
    print(f"   Loaded {len(items)} items from golden set")
    return items


async def precompute_schema_contexts(
    items: list[dict],
    hybrid_store: HybridSchemaStore,
    db_url: str,
) -> list[dict]:
    """Pre-compute schema retrieval for each question.

    Mirrors chapter 3.5's retrieve_schema_node: search for relevant
    tables and format as context text. This avoids running async
    HybridSchemaStore calls inside the sync @rollout function.
    """
    print(f"   Pre-computing schema contexts for {len(items)} questions...")

    for i, item in enumerate(items):
        search_results = await hybrid_store.search_tables(
            item["question"], top_k=5
        )
        schema_lines = ["## Retrieved Schema Context\n"]
        for r in search_results:
            details = await hybrid_store.get_table_details(
                r["schema_name"], r["name"]
            )
            if not details:
                continue
            cols = ", ".join(c["name"] for c in details.get("columns", []))
            schema_lines.append(
                f"### {r['schema_name']}.{r['name']} "
                f"(score: {r['score']:.2f})\n"
                f"Description: {details['description']}\n"
                f"Columns: {cols}\n"
            )
        item["schema_context"] = "\n".join(schema_lines)
        item["database_url"] = db_url

        if (i + 1) % 10 == 0:
            print(f"   ... {i + 1}/{len(items)} done")

    print(f"   Schema contexts ready for {len(items)} items")
    return items


def split_dataset(
    items: list[dict],
    train_ratio: float = 0.50,
    val_ratio: float = 0.25,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split by difficulty."""
    groups: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    for item in items:
        d = item.get("difficulty", "medium")
        groups.get(d, groups["medium"]).append(item)

    train, val, test = [], [], []
    for group in groups.values():
        n = len(group)
        t_end = int(n * train_ratio)
        v_end = int(n * (train_ratio + val_ratio))
        train.extend(group[:t_end])
        val.extend(group[t_end:v_end])
        test.extend(group[v_end:])
    return train, val, test


# ---------------------------------------------------------------------------
# Test set evaluation
# ---------------------------------------------------------------------------

def evaluate_test_set(
    test_data: list[dict],
    best_template,
) -> list[float]:
    """Evaluate the optimized prompt on the held-out test set."""
    from reward import compute_sql_reward_detailed
    from agent_rollout import _get_agent
    from langchain_core.messages import SystemMessage, HumanMessage

    agent = _get_agent()
    test_rewards = []

    for i, task in enumerate(test_data, 1):
        template_text = best_template.template if hasattr(best_template, "template") else str(best_template)

        initial_state = {
            "original_question": task["question"],
            "sql_gen_prompt": template_text,
            "tables_used": [],
            "schema_overview": task.get("schema_context", ""),
            "sql": "",
            "results": "",
            "evaluation_score": 0.0,
            "response": "",
            "messages": [
                SystemMessage(content="You are a SQL agent."),
                HumanMessage(content=task["question"]),
            ],
        }

        config = {"recursion_limit": 25}

        generated_sql = ""
        try:
            result = agent.invoke(initial_state, config)
            generated_sql = result.get("sql", "")
        except Exception:
            generated_sql = ""

        breakdown = compute_sql_reward_detailed(generated_sql, task["expected_sql"])
        test_rewards.append(breakdown.total)

        marker = "." if breakdown.total > 0.5 else "F"
        print(marker, end="", flush=True)

    print()  # newline after progress dots
    return test_rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def prepare_data(db_url: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Async phase: load dataset, pre-compute schemas, split."""
    # ── 1. Load dataset ──
    print("\n1. Loading dataset...")
    raw_items = load_dataset()

    # ── 2. Pre-compute schema contexts (same as chapter 3.5's retrieve_schema) ──
    print("\n2. Pre-computing schema contexts...")
    provider = get_provider_name()
    embeddings = create_embeddings(provider)
    pool = await asyncpg.create_pool(db_url)
    hybrid_store = HybridSchemaStore(pool, embeddings)

    items = await precompute_schema_contexts(raw_items, hybrid_store, db_url)
    await pool.close()

    # ── 3. Split dataset ──
    train_data, val_data, test_data = split_dataset(items)
    print(f"\n   Split: {len(train_data)} train / {len(val_data)} val "
          f"/ {len(test_data)} test")
    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(description="Run APO on the SQL agent")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--branch-factor", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--gradient-batch", type=int, default=6)
    parser.add_argument("--val-batch", type=int, default=15)
    parser.add_argument("--runners", type=int, default=4)
    parser.add_argument("--dev", action="store_true",
                        help="Dry-run: test 3 tasks then exit (trainer.dev)")
    parser.add_argument("--db-url", default=os.getenv(
        "CHINOOK_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/chinook",
    ))
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Chapter 4A: Automatic Prompt Optimization")
    print("=" * 60)
    print("   Dashboard: http://localhost:4747")

    # ── Async phase: data loading + schema pre-computation ──
    train_data, val_data, test_data = asyncio.run(prepare_data(args.db_url))

    # ── 4. Configure APO (sync from here — avoids nested event-loop) ──
    print("\n3. Configuring APO...")
    print(f"   beam_width={args.beam_width}, branch_factor={args.branch_factor}, "
          f"rounds={args.rounds}")
    print(f"   gradient_batch={args.gradient_batch}, val_batch={args.val_batch}, "
          f"runners={args.runners}")

    openai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )

    algo = APO(
        openai_client,
        beam_width=args.beam_width,
        branch_factor=args.branch_factor,
        beam_rounds=args.rounds,
        gradient_batch_size=args.gradient_batch,
        val_batch_size=args.val_batch,
    )

    trainer = agl.Trainer(
        algorithm=algo,
        n_runners=args.runners,
        initial_resources={
            "prompt_template": baseline_prompt_template(),
        },
        adapter=TraceToMessages(),
        tracer=OtelTracer(),  # Local-only OTEL, no AgentOps cloud upload
        # Default strategy = ClientServerExecutionStrategy → dashboard on :4747
    )

    # ── 5. Run optimization (or dev dry-run) ──
    if args.dev:
        print("\n4. Running dev dry-run (3 train tasks, Baseline algorithm)...")
        from agentlightning.algorithm import Baseline
        dev_trainer = agl.Trainer(
            algorithm=Baseline(),
            n_runners=1,
            initial_resources={"prompt_template": baseline_prompt_template()},
            adapter=TraceToMessages(),
            tracer=OtelTracer(),
        )
        dev_trainer.dev(agent=sql_agent_rollout, train_dataset=train_data[:3])
        print("\n   Dev run complete. Check dashboard at http://localhost:4747")
        return

    print("\n4. Running APO optimization...")
    start = time.perf_counter()

    trainer.fit(
        agent=sql_agent_rollout,
        train_dataset=train_data,
        val_dataset=val_data,
    )

    elapsed = time.perf_counter() - start
    print(f"\n   Optimization completed in {elapsed:.1f}s")

    # ── 6. Retrieve best prompt from store ──
    print("\n5. Retrieving best prompt...")
    latest = asyncio.run(trainer.store.get_latest_resources())
    if latest and latest.resources:
        best_template = latest.resources.get("prompt_template")
    else:
        print("   WARNING: No optimized prompt found, using baseline")
        best_template = baseline_prompt_template()

    template_text = best_template.template if hasattr(best_template, "template") else str(best_template)
    print(f"\n   Best prompt template:\n   {'-' * 50}")
    for line in template_text.split("\n")[:10]:
        print(f"   {line}")
    if template_text.count("\n") > 10:
        print(f"   ... ({template_text.count(chr(10)) - 10} more lines)")
    print(f"   {'-' * 50}")

    # ── 7. Evaluate on held-out test set ──
    print("\n6. Evaluating on held-out test set...")
    test_rewards = evaluate_test_set(test_data, best_template)

    if test_rewards:
        mean_reward = sum(test_rewards) / len(test_rewards)
        print(f"\n   Test set mean reward: {mean_reward:.3f} "
              f"({len(test_rewards)} queries)")

    # ── 8. Save results ──
    output_dir = project_root / "experiments" / "apo_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = output_dir / "best_prompt.txt"
    prompt_path.write_text(template_text)
    print(f"\n7. Saved best prompt to {prompt_path}")

    print("\n" + "=" * 60)
    print("Done. Compare with baseline:")
    print(f"  python scripts/run_chapter_3_5.py --run-name prompt-apo-optimized")
    print(f"  View traces: http://localhost:4747")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
