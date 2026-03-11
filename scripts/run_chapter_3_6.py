#!/usr/bin/env python3
"""3.6 Chart Generation -- Extends the 3.5 agent with chart + frontend.

Builds on the 3.5 agent graph (retrieve → generate → evaluate) and adds:
- A chart_node after respond_eval that generates Vega-Lite specs
- A web frontend with Answer + Chart tabs

Prerequisites:
    # Start Chinook DB (if not running)
    make db-up

    # Build the 3.3 offline index (if not done already)
    python scripts/run_chapter_3_3.py --offline

Usage:
    # Start web server (open http://localhost:8080)
    python scripts/run_chapter_3_6.py

    # CLI mode: run a single question
    python scripts/run_chapter_3_6.py --cli "Show me the top 5 genres by track count"
"""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools.sql_database.tool import (
    QuerySQLDatabaseTool,
    QuerySQLCheckerTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
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
import json
import logging
import os
import sys
import time

from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

from src.adapters import create_adapter, get_model_name, get_provider_name
from src.retrieval.pgvector_store import create_embeddings
from src.schema import HybridSchemaStore
from src.utils.config import load_config
from src.chart.generator import generate_chart_from_answer
from src.observability.tracing import (
    init_langfuse,
    get_langfuse_callback,
    flush_langfuse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Agent State (extends 3.5 with chart fields)
# =============================================================================

class SQLAgentState(TypedDict):
    """State for the SQL agent with chart generation."""
    # From 3.5
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
    # New in 3.6: chart generation
    chart_spec: dict[str, Any] | None
    chart_reasoning: str | None


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
# Build Agent (3.5 graph + chart node)
# =============================================================================

def build_agent(db, llm, hybrid_store):
    """Build the LangGraph SQL agent with chart generation.

    Inherits the 3.5 graph (retrieve → generate → evaluate) and adds
    a chart_node after respond_eval.
    """

    # ── MDL tools (same as 3.5) ──

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
3. SYNTHESIZE: Create a natural language response that directly answers the user's question
4. SCORE: Quality score (1.0=correct, 0.9=mostly correct, 0.5=partial, 0.0=error)
5. OUTPUT: Call EvaluationResult with your response and score
</WORKFLOW>

<RULES>
- The response should directly answer the user's question in plain English
- Include specific numbers and data from the results
- If the data has multiple rows, summarize key patterns or list top items
</RULES>
"""

    # ── Nodes (same as 3.5) ──

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

    # ── Chart node (NEW in 3.6) ──

    async def chart_node(state):
        """Generate Vega-Lite chart from the synthesized answer text.

        Reads the final answer (already produced by the evaluate node)
        and asks the LLM to extract data sections and produce charts.
        No SQL re-execution needed — data comes from the answer itself.
        """
        question = state.get("original_question", "")
        response = state.get("response", "")

        if not response or len(response) < 50:
            return {"chart_spec": None, "chart_reasoning": None}

        # Precheck: skip answers without numeric data patterns (avoids wasted LLM call)
        has_data = any(c.isdigit() for c in response) and ':' in response
        if not has_data:
            return {"chart_spec": None, "chart_reasoning": "No numeric data in answer"}

        try:
            chart_data = await generate_chart_from_answer(
                question=question,
                final_answer=response,
                llm_model=llm,
            )

            if chart_data and chart_data.get("charts"):
                first = chart_data["charts"][0]
                return {
                    "chart_spec": first["chart_schema"],
                    "chart_reasoning": first.get("reasoning", ""),
                }

            return {"chart_spec": None, "chart_reasoning": "No chartable data in answer"}
        except Exception as e:
            logger.warning(f"Chart generation failed (non-blocking): {e}")
            return {"chart_spec": None, "chart_reasoning": None}

    # ── Build graph (3.5 + chart) ──

    graph = StateGraph(SQLAgentState)
    graph.add_node("retrieve_schema", retrieve_schema_node)
    graph.add_node("generate_sql", sql_generation_node)
    graph.add_node("sql_tools", ToolNode(
        sql_gen_tools, handle_tool_errors=True))
    graph.add_node("respond_sql", respond_sql_node)
    graph.add_node("evaluate", evaluation_node)
    graph.add_node("eval_tools", ToolNode(eval_tools, handle_tool_errors=True))
    graph.add_node("respond_eval", respond_eval_node)
    graph.add_node("chart", chart_node)  # NEW in 3.6

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
    # 3.6 change: respond_eval → chart → END (was respond_eval → END)
    graph.add_edge("respond_eval", "chart")
    graph.add_edge("chart", END)

    return graph.compile()


# =============================================================================
# Run a single question through the agent
# =============================================================================

async def run_question(agent, question: str) -> dict[str, Any]:
    """Run a question through the agent and return structured results."""
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
        "chart_spec": None,
        "chart_reasoning": None,
    }

    # Langfuse tracing (carries forward from 3.4/3.5)
    try:
        handler, langfuse_metadata, trace_id = get_langfuse_callback(
            session_id="chapter-3-6",
            tags=["chapter-3-6", "chart"],
        )
        config = {
            "configurable": {"thread_id": f"web-{int(time.time())}"},
            "callbacks": [handler],
            "metadata": langfuse_metadata,
            "recursion_limit": 50,
        }
    except Exception:
        config = {
            "configurable": {"thread_id": f"web-{int(time.time())}"},
            "recursion_limit": 50,
        }

    result = await agent.ainvoke(initial_state, config)

    return {
        "response": result.get("response", ""),
        "sql": result.get("sql", ""),
        "chart_spec": result.get("chart_spec"),
        "chart_reasoning": result.get("chart_reasoning"),
    }


# =============================================================================
# Infrastructure setup
# =============================================================================

async def setup_infrastructure():
    """Initialize database, LLM, schema store, and observability."""
    config = load_config()
    db_url = config.database.url

    # Observability (carries forward from 3.4/3.5)
    try:
        init_langfuse()
        logger.info("Langfuse tracing initialized")
    except Exception as e:
        logger.warning(f"Langfuse not available (tracing disabled): {e}")

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

    agent = build_agent(db, llm, hybrid_store)
    logger.info(f"Agent built: {provider}:{model_name}")

    return agent, pool


# =============================================================================
# Web server
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the web server with the agent frontend."""
    from aiohttp import web
    from src.api.server import create_app

    async def startup():
        agent, pool = await setup_infrastructure()
        app = create_app(agent)
        app["pool"] = pool
        return app

    async def run():
        app = await startup()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        print(f"\n  SQL Agent running at http://localhost:{port}\n")
        await site.start()
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                flush_langfuse()
            except Exception:
                pass
            pool = app.get("pool")
            if pool:
                await pool.close()
            await runner.cleanup()

    asyncio.run(run())


# =============================================================================
# CLI mode
# =============================================================================

def run_cli(question: str) -> None:
    """Run a single question through the agent and print results."""

    async def _run() -> dict:
        agent, pool = await setup_infrastructure()
        try:
            return await run_question(agent, question)
        finally:
            try:
                flush_langfuse()
            except Exception:
                pass
            await pool.close()

    result = asyncio.run(_run())

    print(json.dumps(result, indent=2, default=str))

    if result.get("response"):
        print(f"\nAnswer: {result['response']}")
    if result.get("chart_spec"):
        print(f"\nChart type: {result['chart_spec'].get('mark', 'unknown')}")
        print(f"Chart reasoning: {result.get('chart_reasoning', '')[:200]}")
    else:
        print(f"\nNo chart. Reasoning: {result.get('chart_reasoning', 'N/A')}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 3.6: Chart generation + frontend")
    parser.add_argument(
        "--cli",
        type=str,
        metavar="QUESTION",
        help="Run a single question in CLI mode (no server)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli(args.cli)
    else:
        run_server(port=args.port)


if __name__ == "__main__":
    main()
