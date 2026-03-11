"""Evaluate the same SQL agent with different LLM backends.

Runs the chapter 3.3 production agent (disambiguate → retrieve schema →
generate SQL → evaluate) on the golden set with two LLM backends:
  1. GPT-4.1-mini (cloud API baseline)
  2. Fine-tuned local model (SFT or GRPO checkpoint)

Same agent architecture, same tools, same schema retrieval — only the
LLM changes. This isolates the effect of fine-tuning from everything
else in the pipeline.

Outputs:
  - Side-by-side metrics (execution accuracy, result accuracy, latency)
  - Per-category and per-difficulty breakdowns
  - Head-to-head analysis: which queries each model wins/loses
  - Component F1 comparison (structural similarity to gold SQL)
  - JSON results file for further analysis

Prerequisites:
  make db-up
  python scripts/run_chapter_3_3.py --offline   # Build MDL index

Usage:
  source .venv/bin/activate

  # Compare GPT-4.1-mini vs GRPO-trained model
  python scripts/chapter_4C/evaluate_sql.py \\
      --local-model ./output_grpo_sql

  # Compare GPT-4.1-mini vs SFT checkpoint
  python scripts/chapter_4C/evaluate_sql.py \\
      --local-model ./output_4b_sft

  # Use a specific base model (default: Qwen/Qwen2.5-1.5B-Instruct)
  python scripts/chapter_4C/evaluate_sql.py \\
      --local-model ./output_grpo_sql \\
      --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct

  # Save results to JSON
  python scripts/chapter_4C/evaluate_sql.py \\
      --local-model ./output_grpo_sql \\
      --output data/eval_comparison.json

  # Only run the cloud model (skip local)
  python scripts/chapter_4C/evaluate_sql.py --cloud-only

  # Only run the local model (skip cloud)
  python scripts/chapter_4C/evaluate_sql.py \\
      --local-model ./output_grpo_sql --local-only
"""

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

import asyncpg
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from evals.chinook_golden_set import GOLDEN_SET
from evals.golden_set import GoldenQuery
from evals.metrics import (
    calculate_metrics,
    compare_metrics,
    format_metrics_report,
)
from evals.runner import EvalResult, evaluate_query, print_failures
from evals.sql_components import compare_sql_components
from scripts.chapter_4C.reward_sql import compare_results as reward_compare_results
from scripts.chapter_4C.reward_sql import execute_sql
from src.adapters import get_model_name, get_provider_name
from src.retrieval.pgvector_store import create_embeddings
from src.schema import HybridSchemaStore
from src.utils.config import load_config

# Reuse the production agent builder
from scripts.run_chapter_3_3 import build_agent_with_hybrid_tools


# ── Model result for comparison ──────────────────────

@dataclass
class ModelResult:
    """Per-query result from one model."""
    query_id: str
    question: str
    generated_sql: str
    gold_sql: str
    executed: bool
    execution_match: bool
    component_f1: float
    latency_ms: float
    error: str | None


# ── Build agent with a specific LLM ─────────────────

async def build_agent(
    llm,
    db_url: str,
    pool: asyncpg.Pool,
    embeddings,
):
    """Build the chapter 3.3 production agent with a given LLM.

    Returns (agent, store_cm, checkpointer_cm) — caller must manage
    the context managers.
    """
    hybrid_store = HybridSchemaStore(pool, embeddings)
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

    return agent, store_cm, checkpointer_cm


async def run_agent_query(
    agent,
    question: str,
    thread_id: str,
) -> dict:
    """Run a single question through the agent and return the state."""
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

    result = await agent.ainvoke(
        initial_state,
        {"configurable": {"thread_id": thread_id}, "recursion_limit": 50},
    )
    return result


# ── Evaluate one model on the golden set ─────────────

async def evaluate_model(
    agent,
    golden_set: list[GoldenQuery],
    db_url: str,
    model_name: str,
) -> list[ModelResult]:
    """Run golden set through the agent and collect per-query results."""
    results: list[ModelResult] = []

    for i, gq in enumerate(golden_set, 1):
        # Skip negative test cases (no gold SQL)
        if not gq.sql.strip():
            continue

        start = time.perf_counter()

        try:
            thread_id = f"eval-{model_name}-{gq.id}-{int(time.time())}"
            state = await run_agent_query(agent, gq.question, thread_id)
            generated_sql = state.get("sql", "")

            # Strip MDL schema prefix for execution
            generated_sql = re.sub(r"\bchinook\.", "", generated_sql)

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            results.append(ModelResult(
                query_id=gq.id,
                question=gq.question,
                generated_sql="",
                gold_sql=gq.sql,
                executed=False,
                execution_match=False,
                component_f1=0.0,
                latency_ms=elapsed,
                error=str(e)[:200],
            ))
            print(f"  [{i}/{len(golden_set)}] {gq.id}: ERROR {str(e)[:60]}")
            continue

        elapsed = (time.perf_counter() - start) * 1000

        # Execute and compare
        executed = False
        execution_match = False
        component_f1 = 0.0
        error = None

        if generated_sql:
            gen_result = execute_sql(generated_sql, db_url)
            executed = gen_result["success"]
            error = gen_result.get("error")

            if executed:
                gold_result = execute_sql(gq.sql, db_url)
                if gold_result["success"]:
                    execution_match = reward_compare_results(
                        gen_result["rows"], gold_result["rows"],
                    )

            try:
                comp = compare_sql_components(generated_sql, gq.sql)
                component_f1 = comp.overall_f1
            except Exception:
                component_f1 = 0.0

        status = "MATCH" if execution_match else ("EXEC" if executed else "FAIL")
        print(
            f"  [{i}/{len(golden_set)}] {gq.id}: {status} "
            f"(f1={component_f1:.2f}, {elapsed:.0f}ms)"
        )

        results.append(ModelResult(
            query_id=gq.id,
            question=gq.question,
            generated_sql=generated_sql,
            gold_sql=gq.sql,
            executed=executed,
            execution_match=execution_match,
            component_f1=component_f1,
            latency_ms=elapsed,
            error=error,
        ))

    return results


# ── Aggregate metrics from ModelResult ───────────────

def compute_aggregate(
    results: list[ModelResult],
    golden_set: list[GoldenQuery],
) -> dict:
    """Compute aggregate metrics from model results."""
    if not results:
        return {}

    query_map = {gq.id: gq for gq in golden_set}

    total = len(results)
    executed = sum(1 for r in results if r.executed)
    matched = sum(1 for r in results if r.execution_match)
    latencies = sorted(r.latency_ms for r in results)
    f1_scores = [r.component_f1 for r in results]

    p50_idx = len(latencies) // 2
    p95_idx = int(len(latencies) * 0.95)

    # By category
    by_cat: dict[str, dict[str, int]] = {}
    by_diff: dict[str, dict[str, int]] = {}

    for r in results:
        gq = query_map.get(r.query_id)
        if not gq:
            continue

        cat = gq.category
        diff = gq.difficulty

        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "matched": 0, "executed": 0}
        by_cat[cat]["total"] += 1
        if r.executed:
            by_cat[cat]["executed"] += 1
        if r.execution_match:
            by_cat[cat]["matched"] += 1

        if diff not in by_diff:
            by_diff[diff] = {"total": 0, "matched": 0, "executed": 0}
        by_diff[diff]["total"] += 1
        if r.executed:
            by_diff[diff]["executed"] += 1
        if r.execution_match:
            by_diff[diff]["matched"] += 1

    return {
        "total_queries": total,
        "execution_accuracy": executed / total if total else 0.0,
        "result_accuracy": matched / total if total else 0.0,
        "component_f1_mean": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "latency_p50_ms": latencies[p50_idx] if latencies else 0.0,
        "latency_p95_ms": latencies[p95_idx] if latencies else 0.0,
        "latency_mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "by_category": {
            cat: s["matched"] / s["total"] if s["total"] else 0.0
            for cat, s in by_cat.items()
        },
        "by_category_detailed": by_cat,
        "by_difficulty": {
            d: s["matched"] / s["total"] if s["total"] else 0.0
            for d, s in by_diff.items()
        },
        "by_difficulty_detailed": by_diff,
    }


# ── Head-to-head comparison ─────────────────────────

def head_to_head(
    cloud_results: list[ModelResult],
    local_results: list[ModelResult],
    cloud_name: str,
    local_name: str,
) -> str:
    """Compare two models query-by-query."""
    cloud_map = {r.query_id: r for r in cloud_results}
    local_map = {r.query_id: r for r in local_results}

    common_ids = sorted(set(cloud_map.keys()) & set(local_map.keys()))

    cloud_wins = []
    local_wins = []
    both_correct = 0
    both_wrong = 0

    for qid in common_ids:
        c = cloud_map[qid]
        l = local_map[qid]

        if c.execution_match and l.execution_match:
            both_correct += 1
        elif c.execution_match and not l.execution_match:
            cloud_wins.append(qid)
        elif not c.execution_match and l.execution_match:
            local_wins.append(qid)
        else:
            both_wrong += 1

    lines = [
        "",
        "=" * 60,
        f"HEAD-TO-HEAD: {cloud_name} vs {local_name}",
        "=" * 60,
        "",
        f"  Total compared: {len(common_ids)}",
        f"  Both correct:   {both_correct}",
        f"  Both wrong:     {both_wrong}",
        f"  {cloud_name} wins: {len(cloud_wins)}",
        f"  {local_name} wins: {len(local_wins)}",
    ]

    if cloud_wins:
        lines.append(f"\n  {cloud_name} wins on:")
        for qid in cloud_wins[:10]:
            c = cloud_map[qid]
            l = local_map[qid]
            lines.append(
                f"    {qid}: {c.question[:60]}"
            )
            lines.append(
                f"      {cloud_name} f1={c.component_f1:.2f}  "
                f"{local_name} f1={l.component_f1:.2f}"
            )
        if len(cloud_wins) > 10:
            lines.append(f"    ... and {len(cloud_wins) - 10} more")

    if local_wins:
        lines.append(f"\n  {local_name} wins on:")
        for qid in local_wins[:10]:
            c = cloud_map[qid]
            l = local_map[qid]
            lines.append(
                f"    {qid}: {l.question[:60]}"
            )
            lines.append(
                f"      {cloud_name} f1={c.component_f1:.2f}  "
                f"{local_name} f1={l.component_f1:.2f}"
            )
        if len(local_wins) > 10:
            lines.append(f"    ... and {len(local_wins) - 10} more")

    lines.append("=" * 60)
    return "\n".join(lines)


# ── Print comparison table ───────────────────────────

def print_comparison(
    cloud_metrics: dict | None,
    local_metrics: dict | None,
    cloud_name: str,
    local_name: str,
) -> None:
    """Print side-by-side metrics table."""
    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)

    def fval(metrics: dict | None, key: str, fmt: str = ".1%") -> str:
        if metrics is None:
            return "—"
        val = metrics.get(key, 0.0)
        return f"{val:{fmt}}"

    header = f"{'Metric':<28} {cloud_name:>16} {local_name:>16}"
    print(f"\n{header}")
    print("─" * 62)

    rows = [
        ("Queries", "total_queries", "d"),
        ("Execution Accuracy", "execution_accuracy", ".1%"),
        ("Result Accuracy", "result_accuracy", ".1%"),
        ("Component F1 (mean)", "component_f1_mean", ".3f"),
        ("Latency p50 (ms)", "latency_p50_ms", ".0f"),
        ("Latency p95 (ms)", "latency_p95_ms", ".0f"),
        ("Latency mean (ms)", "latency_mean_ms", ".0f"),
    ]

    for label, key, fmt in rows:
        c_val = fval(cloud_metrics, key, fmt)
        l_val = fval(local_metrics, key, fmt)
        print(f"  {label:<26} {c_val:>16} {l_val:>16}")

    # Category breakdown
    all_cats = set()
    if cloud_metrics:
        all_cats |= set(cloud_metrics.get("by_category", {}).keys())
    if local_metrics:
        all_cats |= set(local_metrics.get("by_category", {}).keys())

    if all_cats:
        print(f"\n{'By Category':<28} {cloud_name:>16} {local_name:>16}")
        print("─" * 62)
        for cat in sorted(all_cats):
            c_acc = cloud_metrics.get("by_category", {}).get(cat) if cloud_metrics else None
            l_acc = local_metrics.get("by_category", {}).get(cat) if local_metrics else None
            c_str = f"{c_acc:.1%}" if c_acc is not None else "—"
            l_str = f"{l_acc:.1%}" if l_acc is not None else "—"

            c_det = cloud_metrics.get("by_category_detailed", {}).get(cat, {}) if cloud_metrics else {}
            l_det = local_metrics.get("by_category_detailed", {}).get(cat, {}) if local_metrics else {}
            c_n = f"({c_det.get('matched', 0)}/{c_det.get('total', 0)})" if c_det else ""
            l_n = f"({l_det.get('matched', 0)}/{l_det.get('total', 0)})" if l_det else ""

            print(f"  {cat:<26} {c_str + ' ' + c_n:>16} {l_str + ' ' + l_n:>16}")

    # Difficulty breakdown
    print(f"\n{'By Difficulty':<28} {cloud_name:>16} {local_name:>16}")
    print("─" * 62)
    for diff in ["easy", "medium", "hard"]:
        c_acc = cloud_metrics.get("by_difficulty", {}).get(diff) if cloud_metrics else None
        l_acc = local_metrics.get("by_difficulty", {}).get(diff) if local_metrics else None
        c_str = f"{c_acc:.1%}" if c_acc is not None else "—"
        l_str = f"{l_acc:.1%}" if l_acc is not None else "—"

        c_det = cloud_metrics.get("by_difficulty_detailed", {}).get(diff, {}) if cloud_metrics else {}
        l_det = local_metrics.get("by_difficulty_detailed", {}).get(diff, {}) if local_metrics else {}
        c_n = f"({c_det.get('matched', 0)}/{c_det.get('total', 0)})" if c_det else ""
        l_n = f"({l_det.get('matched', 0)}/{l_det.get('total', 0)})" if l_det else ""

        print(f"  {diff:<26} {c_str + ' ' + c_n:>16} {l_str + ' ' + l_n:>16}")

    print("=" * 60)


# ── Create local LLM (vLLM or HuggingFace) ──────────

def create_local_llm(
    model_path: str,
    base_model: str | None = None,
    base_url: str | None = None,
):
    """Create a LangChain chat model for the local fine-tuned checkpoint.

    Two modes:
      1. vLLM server (recommended): pass --base-url to point at a running
         vLLM server that already loaded the model.
      2. OpenAI-compatible endpoint: same as vLLM but any endpoint.

    The fine-tuned model is served through an OpenAI-compatible API,
    so we use ChatOpenAI with a custom base_url.
    """
    if not base_url:
        raise ValueError(
            "Local model requires a running inference server. "
            "Start one with:\n"
            "  vllm serve {model} --enable-lora "
            "--lora-modules sql-agent={lora_path}\n\n"
            "Then pass --base-url http://localhost:8000/v1"
        )

    return ChatOpenAI(
        model=model_path if not base_model else base_model,
        base_url=base_url,
        api_key="not-needed",
        temperature=0,
        max_retries=1,
    )


# ── Main ─────────────────────────────────────────────

async def async_main():
    parser = argparse.ArgumentParser(
        description="Compare SQL agent with GPT-4.1-mini vs fine-tuned model",
    )
    parser.add_argument(
        "--local-model",
        help="Path to fine-tuned checkpoint (LoRA adapter or merged model)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible endpoint for local model "
             "(e.g., http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--cloud-model",
        default="gpt-4.1-mini",
        help="Cloud model name (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--cloud-only",
        action="store_true",
        help="Only run cloud model evaluation",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only run local model evaluation",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=0,
        help="Limit number of queries (0 = all, useful for quick tests)",
    )
    args = parser.parse_args()

    if args.local_only and not args.local_model:
        parser.error("--local-only requires --local-model")
    if args.local_only and args.cloud_only:
        parser.error("Cannot use both --local-only and --cloud-only")

    run_cloud = not args.local_only
    run_local = not args.cloud_only and args.local_model is not None

    if not run_cloud and not run_local:
        parser.error("Nothing to run. Provide --local-model or remove --cloud-only.")

    # ── Filter golden set ────────────────────────────
    golden_set = [gq for gq in GOLDEN_SET if gq.sql.strip()]
    if args.limit > 0:
        golden_set = golden_set[:args.limit]

    print("=" * 60)
    print("SQL Agent Evaluation: Same Agent, Different LLMs")
    print("=" * 60)
    print(f"\n  Golden set: {len(golden_set)} queries (non-negative)")
    if run_cloud:
        print(f"  Cloud model: {args.cloud_model}")
    if run_local:
        print(f"  Local model: {args.local_model}")
        print(f"  Base model:  {args.base_model}")

    # ── Connect ──────────────────────────────────────
    config = load_config()
    db_url = config.database.url

    pool = await asyncpg.create_pool(db_url)
    print(f"\n  Connected to database")

    provider = get_provider_name()
    embeddings = create_embeddings(provider)
    print(f"  Embeddings ready ({provider})")

    # Initialise reward DB for compare_results
    from scripts.chapter_4C.reward_sql import init_db
    init_db(db_url)

    cloud_results: list[ModelResult] | None = None
    local_results: list[ModelResult] | None = None
    cloud_metrics: dict | None = None
    local_metrics: dict | None = None

    cloud_name = args.cloud_model
    local_name = Path(args.local_model).name if args.local_model else "local"

    # ── Evaluate cloud model ─────────────────────────
    if run_cloud:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {cloud_name}")
        print("=" * 60)

        cloud_llm = ChatOpenAI(model=args.cloud_model, temperature=0)
        agent, store_cm, cp_cm = await build_agent(
            cloud_llm, db_url, pool, embeddings,
        )

        try:
            cloud_results = await evaluate_model(
                agent, golden_set, db_url, cloud_name,
            )
            cloud_metrics = compute_aggregate(cloud_results, GOLDEN_SET)
        finally:
            await cp_cm.__aexit__(None, None, None)
            await store_cm.__aexit__(None, None, None)

        print(f"\n  {cloud_name} done: "
              f"{cloud_metrics['result_accuracy']:.1%} result accuracy")

    # ── Evaluate local model ─────────────────────────
    if run_local:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {local_name}")
        print("=" * 60)

        local_llm = create_local_llm(
            args.local_model,
            base_model=args.base_model,
            base_url=args.base_url,
        )
        agent, store_cm, cp_cm = await build_agent(
            local_llm, db_url, pool, embeddings,
        )

        try:
            local_results = await evaluate_model(
                agent, golden_set, db_url, local_name,
            )
            local_metrics = compute_aggregate(local_results, GOLDEN_SET)
        finally:
            await cp_cm.__aexit__(None, None, None)
            await store_cm.__aexit__(None, None, None)

        print(f"\n  {local_name} done: "
              f"{local_metrics['result_accuracy']:.1%} result accuracy")

    # ── Print results ────────────────────────────────
    print_comparison(cloud_metrics, local_metrics, cloud_name, local_name)

    if cloud_results and local_results:
        print(head_to_head(
            cloud_results, local_results,
            cloud_name, local_name,
        ))

    # ── Individual model reports ─────────────────────
    if cloud_metrics and not run_local:
        print("\n" + format_metrics_report(cloud_metrics))
    if local_metrics and not run_cloud:
        print("\n" + format_metrics_report(local_metrics))

    # ── Print failures ───────────────────────────────
    if cloud_results:
        cloud_failures = [
            r for r in cloud_results if not r.execution_match
        ]
        if cloud_failures:
            print(f"\n{cloud_name} failures ({len(cloud_failures)}):")
            for r in cloud_failures[:5]:
                print(f"  {r.query_id}: {r.question[:60]}")
                if r.error:
                    print(f"    Error: {r.error[:80]}")

    if local_results:
        local_failures = [
            r for r in local_results if not r.execution_match
        ]
        if local_failures:
            print(f"\n{local_name} failures ({len(local_failures)}):")
            for r in local_failures[:5]:
                print(f"  {r.query_id}: {r.question[:60]}")
                if r.error:
                    print(f"    Error: {r.error[:80]}")

    # ── Save JSON ────────────────────────────────────
    if args.output:
        output_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "golden_set_size": len(golden_set),
        }
        if cloud_metrics and cloud_results:
            output_data["cloud"] = {
                "model": cloud_name,
                "metrics": {
                    k: v for k, v in cloud_metrics.items()
                    if k != "results"
                },
                "results": [
                    {
                        "query_id": r.query_id,
                        "question": r.question,
                        "generated_sql": r.generated_sql,
                        "gold_sql": r.gold_sql,
                        "executed": r.executed,
                        "execution_match": r.execution_match,
                        "component_f1": r.component_f1,
                        "latency_ms": round(r.latency_ms, 1),
                        "error": r.error,
                    }
                    for r in cloud_results
                ],
            }
        if local_metrics and local_results:
            output_data["local"] = {
                "model": local_name,
                "metrics": {
                    k: v for k, v in local_metrics.items()
                    if k != "results"
                },
                "results": [
                    {
                        "query_id": r.query_id,
                        "question": r.question,
                        "generated_sql": r.generated_sql,
                        "gold_sql": r.gold_sql,
                        "executed": r.executed,
                        "execution_match": r.execution_match,
                        "component_f1": r.component_f1,
                        "latency_ms": round(r.latency_ms, 1),
                        "error": r.error,
                    }
                    for r in local_results
                ],
            }

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    await pool.close()

    print(f"\n{'=' * 60}")
    print("Evaluation complete.")
    print("=" * 60)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
