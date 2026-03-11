#!/usr/bin/env python3
"""3.5 Evaluation as a System -- Two-mode evaluation pipeline.

This script demonstrates the complete evaluation workflow:

1. SETUP: Initialize Langfuse + load golden set
2. UPLOAD: Push golden set to a Langfuse dataset
3. EVALUATE: Run each query through the agent with two eval modes:
   - Offline: component matching (evals/sql_components.py)
   - Online: score traces in Langfuse (src/observability/evaluators.py)
4. REPORT: Print metrics with per-category and per-difficulty breakdown

Prerequisites:
    # Start Langfuse + Chinook DB
    docker compose -f docker-compose.langfuse.yml up -d
    make db-up

    # Build the 3.3 offline index (if not done already)
    python scripts/run_chapter_3_3.py --offline

    # Run this script
    python scripts/run_chapter_3_5.py

    # Open Langfuse UI to see traces and scores
    open http://localhost:3000
"""

from evals.runner import EvalResult
from evals.metrics import calculate_metrics, format_metrics_report
from evals.chinook_golden_set import GOLDEN_SET
from src.adapters import get_model_name, get_provider_name
from src.retrieval.pgvector_store import create_embeddings
from src.schema import HybridSchemaStore
from src.utils.config import load_config
from src.observability.evaluators import (
    run_component_eval,
    run_execution_eval,
    score_eval_results,
    upload_golden_set_to_dataset,
    link_trace_to_dataset_item,
)
from src.observability.tracing import (
    init_langfuse,
    get_langfuse_callback,
    get_langfuse_client,
    flush_langfuse,
)
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools.sql_database.tool import (
    QuerySQLDatabaseTool,
    QuerySQLCheckerTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import asyncpg
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
from typing import Sequence, Any
from dotenv import load_dotenv
import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")


# =============================================================================
# Agent State (same as 3.4)
# =============================================================================

class SQLAgentState(TypedDict):
    """State for the SQL agent."""
    original_question: str
    disambiguated_question: str
    cache_hit: bool
    cached_sql: str
    tables_used: list[str]
    schema_overview: str
    sql: str
    results: str
    evaluation_score: float
    response: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


class TextToSQLResult(BaseModel):
    """Output from the Text-to-SQL agent."""
    sql: str = Field(description="The generated SQL query")
    explanation: str = Field(
        description="Brief explanation of the query logic")
    tables_used: list[str] = Field(
        default_factory=list, description="Tables referenced in the query"
    )


class EvaluationResult(BaseModel):
    """Output from the evaluation agent."""
    score: float = Field(ge=0.0, le=1.0, description="Quality score")
    passed: bool = Field(description="Whether the result is acceptable")
    issues: list[str] = Field(default_factory=list,
                              description="Any issues found")
    response: str = Field(description="Final response to show the user")


# =============================================================================
# Build Agent (reused from 3.4)
# =============================================================================

def build_agent(db, llm, hybrid_store, store, checkpointer):
    """Build the LangGraph SQL agent (same as 3.4)."""

    @tool
    async def get_metrics(schema_name: str, table_name: str) -> str:
        """Get metrics (aggregations, KPIs) for a table."""
        results = await hybrid_store.get_metrics(schema_name, table_name)
        if not results:
            return f"No metrics found for '{schema_name}.{table_name}'."
        output = [f"Metrics for {schema_name}.{table_name}:"]
        for m in results:
            output.append(
                f"  - {m['name']}: {m.get('description', m.get('expression', ''))}")
        return "\n".join(output)

    @tool
    async def get_relationships(schema_name: str, table_name: str) -> str:
        """Get relationships (foreign keys, joins) for a table."""
        results = await hybrid_store.get_relationships(schema_name, table_name)
        if not results:
            return f"No relationships found for '{schema_name}.{table_name}'."
        output = [f"Relationships for {schema_name}.{table_name}:"]
        for r in results:
            output.append(
                f"  - {r['name']}: {r.get('condition', r.get('type', ''))}")
        return "\n".join(output)

    @tool
    async def get_glossary_entries(terms: str) -> str:
        """Look up business term definitions from the glossary."""
        term_list = [t.strip().lower() for t in terms.split(",")]
        results = await hybrid_store.get_glossary_entries(term_list)
        if not results:
            return f"No definitions found for: {terms}"
        output = ["Glossary definitions:"]
        for r in results:
            output.append(f"  - {r['term']}: {r['definition']}")
        return "\n".join(output)

    @tool
    async def search_business_context(question: str) -> str:
        """Find business context and rules relevant to the question."""
        results = await hybrid_store.search_business_context(question, top_k=3)
        if not results:
            return "No relevant business context found."
        output = ["Relevant business context:"]
        for r in results:
            output.append(f"  - ({r['type']}, score: {r['score']:.2f})")
            output.append(f"    {r['content'][:100]}...")
        return "\n".join(output)

    optional_mdl_tools = [
        get_metrics, get_relationships,
        get_glossary_entries, search_business_context,
    ]

    sql_checker_tool = QuerySQLCheckerTool(db=db, llm=llm)
    sql_query_tool = QuerySQLDatabaseTool(db=db)

    sql_gen_tools = optional_mdl_tools + [sql_checker_tool, sql_query_tool]
    eval_tools = [sql_query_tool]

    sql_model_with_tools = llm.bind_tools(sql_gen_tools + [TextToSQLResult])
    eval_model_with_tools = llm.bind_tools(eval_tools + [EvaluationResult])

    SQL_GENERATION_PROMPT = """You are an expert SQL generation agent.

The relevant tables and columns have ALREADY been retrieved for you.

<WORKFLOW>
1. READ the schema context already provided
2. OPTIONAL: Use tools for metrics, relationships, glossary, business context
3. GENERATE: Draft the SQL query using only columns from the schema context
4. CHECK: Use sql_db_query_checker to double check the query is correct
5. EXECUTE: Use sql_db_query to run the query and verify sample data
6. If sql_db_query returns an error, rewrite the query, check again, and retry once
7. OUTPUT: Call TextToSQLResult with the final sql, explanation, and tables_used
</WORKFLOW>

<RULES>
- Use exact column names from the schema context
- Do NOT invent schema objects not in the context
- SELECT statements only
- Always use sql_db_query_checker before sql_db_query
- If sql_db_query returns an error, you may retry once after fixing the query
- After a successful sql_db_query, call TextToSQLResult immediately
</RULES>
"""

    EVALUATION_PROMPT = """You are evaluating SQL and generating the final answer.

<WORKFLOW>
1. EXECUTE: Use sql_db_query tool to run the SQL query (ONCE only)
2. ANALYZE: Review the query results
3. SYNTHESIZE: Create a natural language response
4. SCORE: Quality score (1.0=correct, 0.9=mostly correct, 0.5=partial, 0.0=error)
5. OUTPUT: Call EvaluationResult
</WORKFLOW>
"""

    def sql_generation_node(state):
        return {
            "messages": [
                sql_model_with_tools.invoke(
                    [SystemMessage(content=SQL_GENERATION_PROMPT)]
                    + list(state.get("messages", []))
                )
            ]
        }

    def should_continue_sql(state):
        messages = state.get("messages", [])
        if not messages:
            return "respond_sql"
        last = messages[-1]
        if not hasattr(last, "tool_calls") or not last.tool_calls:
            return "respond_sql"
        if len(last.tool_calls) == 1 and last.tool_calls[0].get("name") == "TextToSQLResult":
            return "respond_sql"
        return "sql_tools"

    def respond_sql_node(state):
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        sql = ""
        tables_used = []
        tool_message = None

        if last and hasattr(last, "tool_calls") and last.tool_calls:
            tc = last.tool_calls[0]
            if tc.get("name") == "TextToSQLResult":
                args = tc.get("args", {})
                sql = args.get("sql", "")
                tables_used = args.get("tables_used", [])
                tool_message = ToolMessage(
                    content="SQL generation completed",
                    tool_call_id=tc.get("id", ""),
                )

        eval_instruction = HumanMessage(
            content=f"Execute this SQL and provide the final answer:\n\nSQL: {sql}"
        )
        new_messages = []
        if tool_message:
            new_messages.append(tool_message)
        new_messages.append(eval_instruction)
        return {"sql": sql, "tables_used": tables_used, "messages": new_messages}

    def evaluation_node(state):
        return {
            "messages": [
                eval_model_with_tools.invoke(
                    [SystemMessage(content=EVALUATION_PROMPT)]
                    + list(state.get("messages", []))
                )
            ]
        }

    def should_continue_eval(state):
        messages = state.get("messages", [])
        if not messages:
            return "respond_eval"
        last = messages[-1]
        if not hasattr(last, "tool_calls") or not last.tool_calls:
            return "respond_eval"
        if len(last.tool_calls) == 1 and last.tool_calls[0].get("name") == "EvaluationResult":
            return "respond_eval"
        return "eval_tools"

    def respond_eval_node(state):
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        if last and hasattr(last, "tool_calls") and last.tool_calls:
            tc = last.tool_calls[0]
            if tc.get("name") == "EvaluationResult":
                args = tc.get("args", {})
                tool_msg = ToolMessage(
                    content="Evaluation completed",
                    tool_call_id=tc.get("id", ""),
                )
                return {
                    "response": args.get("response", ""),
                    "evaluation_score": args.get("score", 0.0),
                    "messages": [tool_msg],
                }
        return {"response": "Unable to generate response", "evaluation_score": 0.0}

    async def retrieve_schema_node(state):
        question = state.get("disambiguated_question",
                             state["original_question"])
        search_results = await hybrid_store.search_tables(question, top_k=5)
        schema_lines = ["## Retrieved Schema Context\n"]
        tables_used = []
        for r in search_results:
            details = await hybrid_store.get_table_details(r["schema_name"], r["name"])
            if not details:
                continue
            tables_used.append(r["name"])
            cols = ", ".join(c["name"] for c in details.get("columns", []))
            schema_lines.append(
                f"### {r['schema_name']}.{r['name']} (score: {r['score']:.2f})\n"
                f"Description: {details['description']}\n"
                f"Columns: {cols}\n"
            )
        schema_context = "\n".join(schema_lines)
        return {
            "schema_overview": schema_context,
            "tables_used": tables_used,
            "messages": [SystemMessage(content=schema_context)],
        }

    graph = StateGraph(SQLAgentState)
    graph.add_node("retrieve_schema", retrieve_schema_node)
    graph.add_node("generate_sql", sql_generation_node)
    graph.add_node("sql_tools", ToolNode(
        sql_gen_tools, handle_tool_errors=True))
    graph.add_node("respond_sql", respond_sql_node)
    graph.add_node("evaluate", evaluation_node)
    graph.add_node("eval_tools", ToolNode(eval_tools, handle_tool_errors=True))
    graph.add_node("respond_eval", respond_eval_node)

    graph.set_entry_point("retrieve_schema")
    graph.add_edge("retrieve_schema", "generate_sql")
    graph.add_conditional_edges(
        "generate_sql", should_continue_sql,
        {"sql_tools": "sql_tools", "respond_sql": "respond_sql"},
    )
    graph.add_edge("sql_tools", "generate_sql")
    graph.add_edge("respond_sql", "evaluate")
    graph.add_conditional_edges(
        "evaluate", should_continue_eval,
        {"eval_tools": "eval_tools", "respond_eval": "respond_eval"},
    )
    graph.add_edge("eval_tools", "evaluate")
    graph.add_edge("respond_eval", END)

    return graph.compile(checkpointer=checkpointer, store=store)


# =============================================================================
# Execute SQL helper (for execution-based eval)
# =============================================================================

async def make_execute_fn(db_url: str):
    """Create an async SQL execution function."""
    pool = await asyncpg.create_pool(db_url)

    async def execute_sql(sql: str) -> Any:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql)
            return [dict(row) for row in rows]

    return execute_sql, pool


# =============================================================================
# Main evaluation pipeline
# =============================================================================

async def run_evaluation_pipeline(
    verbose: bool = False,
    limit: int | None = None,
    run_name: str | None = None,
):
    """Run the two-mode evaluation pipeline."""
    print("\n" + "=" * 60)
    print("Chapter 3.5: Evaluation as a System")
    print("=" * 60)

    # ── 1. Initialize ──
    print("\n1. Initializing Langfuse and loading golden set...")
    langfuse = init_langfuse()
    langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    # Filter to queries that have reference SQL (skip negative cases)
    golden_subset = [q for q in GOLDEN_SET if q.sql.strip()]
    if limit:
        golden_subset = golden_subset[:limit]
    print(
        f"   Golden set: {len(golden_subset)} queries (of {len(GOLDEN_SET)} total)")
    print(f"   Langfuse: {langfuse_host}")

    # ── 2. Upload golden set to Langfuse dataset ──
    print("\n2. Uploading golden set to Langfuse dataset...")
    dataset_result = upload_golden_set_to_dataset(
        dataset_name="chinook-golden-set",
        golden_set=golden_subset,
        description="Chinook golden set for evaluation experiments",
    )
    print(
        f"   Dataset '{dataset_result['dataset_name']}': {dataset_result['item_count']} items")

    # ── 2b. Fetch dataset items for experiment linking ──
    langfuse_client = get_langfuse_client()
    dataset = langfuse_client.get_dataset("chinook-golden-set")
    item_map: dict[str, str] = {}  # query_id -> dataset_item_id
    for item in dataset.items:
        qid = (item.metadata or {}).get("query_id", "")
        if qid:
            item_map[qid] = item.id
    print(f"   Mapped {len(item_map)} dataset items for experiment linking")

    # ── 3. Set up agent ──
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
    execute_fn, exec_pool = await make_execute_fn(db_url)

    experiment_run_name = run_name or f"eval-{provider}-{model_name}"
    print(f"\n3. Agent: {provider}:{model_name}")
    print(f"   Run name: {experiment_run_name}")

    # ── 4. Evaluate each query ──
    eval_results: list[EvalResult] = []
    component_results: list = []  # Store component match results for summary

    try:
        async with (
            AsyncPostgresStore.from_conn_string(db_url) as store,
            AsyncPostgresSaver.from_conn_string(db_url) as checkpointer,
        ):
            await store.setup()
            await checkpointer.setup()

            agent = build_agent(db, llm, hybrid_store,
                                store=store, checkpointer=checkpointer)

            print(
                f"\n4. Running evaluation ({len(golden_subset)} queries)...\n")

            for i, query in enumerate(golden_subset, 1):
                start_time = time.perf_counter()

                # -- Run agent with tracing --
                handler, langfuse_metadata, trace_id = get_langfuse_callback(
                    user_id="eval-pipeline",
                    session_id="eval-3-5",
                    tags=["chapter-3-5", "eval",
                          query.category, query.difficulty],
                )

                thread_id = f"eval-{int(time.time())}-{i}"
                config_dict = {
                    "configurable": {"thread_id": thread_id},
                    "callbacks": [handler],
                    "run_name": "eval-pipeline",
                    "metadata": langfuse_metadata,
                    "recursion_limit": 50,
                }

                initial_state = {
                    "original_question": query.question,
                    "disambiguated_question": query.question,
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
                        HumanMessage(content=query.question),
                    ],
                }

                generated_sql = ""
                executed = False
                result_matches = False
                exec_error = None

                try:
                    with langfuse_client.start_as_current_observation(
                        as_type="span",
                        name=f"eval-{query.id}",
                        trace_context={"trace_id": trace_id},
                    ) as span:
                        result = await agent.ainvoke(initial_state, config_dict)

                        generated_sql = result.get("sql", "")

                        # Set trace input/output so evaluators can extract variables.
                        # Use the span object directly because ainvoke() may displace
                        # the OTEL "current span" context in async code (see 3.4).
                        span.update_trace(
                            input={"question": query.question,
                                   "reference_sql": query.sql},
                            output={"sql": generated_sql},
                        )

                    # -- SQL matching --
                    component_result = run_component_eval(
                        generated_sql, query.sql)
                    component_results.append(component_result)

                    # -- SQL execution accuracy --
                    exec_result = await run_execution_eval(query, generated_sql, execute_fn)
                    executed = exec_result["executed"]
                    result_matches = exec_result["result_matches"]
                    exec_error = exec_result["error"]

                    # -- Push scores to Langfuse --
                    score_eval_results(
                        trace_id,
                        component_result=component_result,
                        execution_result=exec_result,
                    )

                    # -- Link trace to dataset item (creates dataset run item) --
                    dataset_item_id = item_map.get(query.id)
                    if dataset_item_id:
                        link_trace_to_dataset_item(
                            dataset_item_id=dataset_item_id,
                            trace_id=trace_id,
                            run_name=experiment_run_name,
                        )

                except Exception as e:
                    exec_error = str(e)
                    component_result = None

                latency_ms = (time.perf_counter() - start_time) * 1000

                # Collect result for metrics
                eval_results.append(EvalResult(
                    query_id=query.id,
                    question=query.question,
                    generated_sql=generated_sql,
                    reference_sql=query.sql,
                    executed=executed,
                    execution_error=exec_error,
                    result_matches=result_matches,
                    latency_ms=latency_ms,
                ))

                # Progress output
                status = "pass" if result_matches else "FAIL"
                comp_score = f"F1={component_result.overall_f1:.2f}" if component_result else "N/A"
                if verbose:
                    print(
                        f"  [{i:2d}/{len(golden_subset)}] {query.id:15s} {status:4s}  {comp_score}  {latency_ms:.0f}ms")
                    if exec_error and not executed:
                        print(f"           Error: {exec_error[:80]}")
                else:
                    marker = "." if result_matches else "F"
                    print(marker, end="", flush=True)

            if not verbose:
                print()  # newline after progress dots

        # ── 5. Calculate and print metrics ──
        print(f"\n5. Results\n")
        metrics = calculate_metrics(eval_results, golden_subset)
        report = format_metrics_report(metrics)
        print(report)

        # ── 6. Component matching summary (reuse results from step 4) ──
        print("\nComponent Matching Summary")
        print("-" * 30)
        if component_results:
            avg_f1 = sum(c.overall_f1 for c in component_results) / \
                len(component_results)
            exact_matches = sum(1 for c in component_results if c.exact_match)
            print(f"  Avg Component F1: {avg_f1:.3f}")
            print(
                f"  Exact Matches:    {exact_matches}/{len(component_results)}")
            print(
                f"  Avg SELECT F1:    {sum(c.select_f1 for c in component_results) / len(component_results):.3f}")
            print(
                f"  Avg FROM F1:      {sum(c.from_f1 for c in component_results) / len(component_results):.3f}")
            print(
                f"  Avg WHERE F1:     {sum(c.where_f1 for c in component_results) / len(component_results):.3f}")

        # ── 7. Flush and summary ──
        print(f"\n6. Flushing traces to Langfuse...")
        flush_langfuse()
        print("   Done.")

        print(f"\n{'=' * 60}")
        print("Next Steps")
        print(f"{'=' * 60}")
        print(f"  1. Open {langfuse_host} to see traces with scores")
        print(
            f"  2. Go to Datasets > 'chinook-golden-set' > Runs > '{experiment_run_name}'")
        print(f"  3. The dataset-mode evaluator fires automatically on new run items")
        print(f"  4. Compare runs by changing --run-name (e.g., 'new-prompt-v2')")

    finally:
        await pool.close()
        await exec_pool.close()


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Run 3.5 Evaluation Pipeline")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-query details")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Limit number of queries to evaluate")
    parser.add_argument("--run-name", "-r", type=str, default=None,
                        help="Experiment run name (default: eval-{provider}-{model})")
    args = parser.parse_args()

    try:
        await run_evaluation_pipeline(
            verbose=args.verbose, limit=args.limit, run_name=args.run_name,
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
