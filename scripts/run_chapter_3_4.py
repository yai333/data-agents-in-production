#!/usr/bin/env python3
"""3.4 Observability — Langfuse Tracing Demo.

This script demonstrates the observability pipeline:

1. SETUP: Initialize Langfuse v3 tracing (local Docker)
2. TRACE: Run the 3.3 LangGraph agent with Langfuse callbacks
3. SCORE: Attach automated eval scores to each trace

Prerequisites:
    # Start Langfuse (separate from Chinook DB)
    docker compose -f docker-compose.langfuse.yml up -d

    # Build the 3.3 offline index (if not done already)
    python scripts/run_chapter_3_3.py --offline

    # Run this script
    python scripts/run_chapter_3_4.py

    # Open Langfuse UI to see traces
    open http://localhost:3000
"""

from src.observability.annotation import score_trace
from src.observability.tracing import (
    init_langfuse,
    get_langfuse_callback,
    get_langfuse_client,
    flush_langfuse,
)
from src.utils.config import load_config
from src.schema import HybridSchemaStore
from src.retrieval.pgvector_store import create_embeddings
from src.adapters import get_model_name, get_provider_name
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from typing import Sequence
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    QuerySQLDatabaseTool,
    QuerySQLCheckerTool,
)
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
import asyncpg
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


# Observability imports (Chapter 3.4)


# =============================================================================
# Agent State (same as 3.3)
# =============================================================================

class SQLAgentState(TypedDict):
    """State for the SQL agent with observability."""
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
# Build Agent (from 3.3, with Langfuse callback support)
# =============================================================================

def build_observable_agent(
    db: SQLDatabase,
    llm,
    hybrid_store: HybridSchemaStore,
    store: AsyncPostgresStore,
    checkpointer: AsyncPostgresSaver,
    verbose: bool = False,
):
    """Build LangGraph agent — same as 3.3 but designed for Langfuse callbacks."""

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

    sql_gen_tools = optional_mdl_tools + [sql_checker_tool]
    eval_tools = [sql_query_tool]

    sql_model_with_tools = llm.bind_tools(sql_gen_tools + [TextToSQLResult])
    eval_model_with_tools = llm.bind_tools(eval_tools + [EvaluationResult])

    SQL_GENERATION_PROMPT = """You are an expert SQL generation agent.

The relevant tables and columns have ALREADY been retrieved for you.
You have optional tools for additional detail when needed.

<WORKFLOW>
1. READ the schema context already provided
2. OPTIONAL: Use tools for metrics, relationships, glossary, business context
3. GENERATE: Draft the SQL query using only columns from the schema context
4. VALIDATE: Use sql_db_query_checker to validate syntax
5. OUTPUT: Call TextToSQLResult with sql, explanation, and tables_used
</WORKFLOW>

<RULES>
- Use exact column names from the schema context
- Do NOT invent schema objects not in the context
- SELECT statements only
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

    def sql_generation_node(state: SQLAgentState) -> dict:
        return {
            "messages": [
                sql_model_with_tools.invoke(
                    [SystemMessage(content=SQL_GENERATION_PROMPT)]
                    + list(state.get("messages", []))
                )
            ]
        }

    def should_continue_sql(state: SQLAgentState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "respond_sql"
        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "respond_sql"
        if (
            len(last_message.tool_calls) == 1
            and last_message.tool_calls[0].get("name") == "TextToSQLResult"
        ):
            return "respond_sql"
        return "sql_tools"

    def respond_sql_node(state: SQLAgentState) -> dict:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        sql = ""
        tables_used: list[str] = []
        tool_message = None

        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            if tool_call.get("name") == "TextToSQLResult":
                args = tool_call.get("args", {})
                sql = args.get("sql", "")
                tables_used = args.get("tables_used", [])
                tool_message = ToolMessage(
                    content="SQL generation completed",
                    tool_call_id=tool_call.get("id", ""),
                )

        eval_instruction = HumanMessage(
            content=f"Execute this SQL and provide the final answer:\n\nSQL: {sql}"
        )
        new_messages = []
        if tool_message:
            new_messages.append(tool_message)
        new_messages.append(eval_instruction)

        return {"sql": sql, "tables_used": tables_used, "messages": new_messages}

    def evaluation_node(state: SQLAgentState) -> dict:
        return {
            "messages": [
                eval_model_with_tools.invoke(
                    [SystemMessage(content=EVALUATION_PROMPT)]
                    + list(state.get("messages", []))
                )
            ]
        }

    def should_continue_eval(state: SQLAgentState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "respond_eval"
        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "respond_eval"
        if (
            len(last_message.tool_calls) == 1
            and last_message.tool_calls[0].get("name") == "EvaluationResult"
        ):
            return "respond_eval"
        return "eval_tools"

    def respond_eval_node(state: SQLAgentState) -> dict:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            if tool_call.get("name") == "EvaluationResult":
                args = tool_call.get("args", {})
                tool_message = ToolMessage(
                    content="Evaluation completed",
                    tool_call_id=tool_call.get("id", ""),
                )
                return {
                    "response": args.get("response", ""),
                    "evaluation_score": args.get("score", 0.0),
                    "messages": [tool_message],
                }
        return {"response": "Unable to generate response", "evaluation_score": 0.0}

    async def retrieve_schema_node(state: SQLAgentState) -> dict:
        """Deterministic schema retrieval (no LLM)."""
        question = state.get("disambiguated_question",
                             state["original_question"])
        search_results = await hybrid_store.search_tables(question, top_k=5)
        if verbose:
            names = [f"{r['schema_name']}.{r['name']}" for r in search_results]
            print(f"  [Retrieve] Found {len(search_results)} tables: {names}")

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

    # Build graph
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
        "generate_sql",
        should_continue_sql,
        {"sql_tools": "sql_tools", "respond_sql": "respond_sql"},
    )
    graph.add_edge("sql_tools", "generate_sql")
    graph.add_edge("respond_sql", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        should_continue_eval,
        {"eval_tools": "eval_tools", "respond_eval": "respond_eval"},
    )
    graph.add_edge("eval_tools", "evaluate")
    graph.add_edge("respond_eval", END)

    compiled = graph.compile(
        checkpointer=checkpointer,
        store=store,
    )
    return compiled


# =============================================================================
# Demo: Run Agent with Langfuse Tracing
# =============================================================================

async def run_traced_demo(verbose: bool = False):
    """Run the agent with Langfuse tracing enabled."""
    print("\n" + "=" * 60)
    print("Chapter 3.4: Observability with Langfuse")
    print("=" * 60)

    # 1. Initialize Langfuse
    print("\n1. Initializing Langfuse...")
    langfuse = init_langfuse()
    langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    print(f"   Host: {langfuse_host}")
    print("   Connected to Langfuse")

    # 2. Set up database and agent (from 3.3)
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

    print(f"\n2. Agent setup: {provider}:{model_name}")

    # Demo questions that exercise different agent paths
    demo_questions = [
        ("How many customers are there?", "user-alice"),
        ("What are the top 5 genres by total sales revenue?", "user-bob"),
        ("Show monthly revenue for 2013", "user-alice"),
        ("Which country has the most customers?", "user-charlie"),
    ]

    trace_ids = []

    async with (
        AsyncPostgresStore.from_conn_string(db_url) as store,
        AsyncPostgresSaver.from_conn_string(db_url) as checkpointer,
    ):
        await store.setup()
        await checkpointer.setup()

        agent = build_observable_agent(
            db, llm, hybrid_store,
            store=store, checkpointer=checkpointer,
            verbose=verbose,
        )

        langfuse_client = get_langfuse_client()
        print(
            f"\n3. Running {len(demo_questions)} queries with Langfuse tracing...\n")

        for i, (question, user_id) in enumerate(demo_questions, 1):
            print(f"{'─' * 60}")
            print(f"  Query {i}: \"{question}\" (user: {user_id})")
            print(f"{'─' * 60}")

            # Create Langfuse callback for this request
            handler, langfuse_metadata, trace_id = get_langfuse_callback(
                user_id=user_id,
                session_id=f"demo-session-{user_id}",
                tags=["chapter-3-4", "demo"],
            )

            thread_id = f"trace-demo-{int(time.time())}-{i}"
            config_dict = {
                "configurable": {"thread_id": thread_id},
                "callbacks": [handler],
                "run_name": "text-to-sql-agent",
                "metadata": langfuse_metadata,
                "recursion_limit": 50,
            }

            initial_state = {
                "original_question": question,
                "disambiguated_question": question,
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
                # Wrap in start_as_current_observation to force a
                # separate trace per request (v3 OTEL shares context
                # within the same process otherwise).
                with langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"query-{i}",
                    trace_context={"trace_id": trace_id},
                ):
                    result = await agent.ainvoke(initial_state, config_dict)

                sql = result.get("sql", "N/A")
                eval_score = result.get("evaluation_score", 0.0)
                response = result.get("response", "N/A")

                print(f"  SQL: {sql[:120]}...")
                print(f"  Score: {eval_score}")
                print(f"  Response: {response[:150]}...")

                # trace_id was pre-generated by get_langfuse_callback()
                trace_ids.append((trace_id, question, sql, eval_score))
                print(f"  Trace ID: {trace_id}")

                # Attach automated eval score to the trace
                score_trace(trace_id, "auto_eval_score", eval_score,
                            comment="Automated eval from agent pipeline")

            except Exception as e:
                print(f"  ERROR: {e}")

            print()

    # 4. Flush traces to Langfuse
    print("\n4. Flushing traces to Langfuse...")
    flush_langfuse()
    print("   All traces sent.")

    # 5. Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  Traces sent: {len(trace_ids)}")
    print(f"  Langfuse UI: {langfuse_host}")
    print(f"  Login: admin@example.com / password123")
    if trace_ids:
        print(f"\n  Traces with auto_eval_score attached:")
        for trace_id, question, sql, eval_score in trace_ids:
            print(
                f"    {trace_id[:12]}... score={eval_score:.1f}  \"{question[:40]}\"")
    print(f"\n  Next steps:")
    print(f"    1. Open {langfuse_host} to view traces")
    print(f"    2. Click a trace to see the full span breakdown")
    print(f"    3. Use Langfuse's built-in annotation to label CORRECT/INCORRECT")
    print(f"    4. Export annotated traces to datasets for fine-tuning (Ch.4)")

    await pool.close()


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Run 3.4 Observability demo")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    try:
        await run_traced_demo(verbose=args.verbose)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
