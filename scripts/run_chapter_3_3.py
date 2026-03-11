#!/usr/bin/env python3
"""3.3 Context Engineering: Production Pipeline with Real Agent.

This script demonstrates the complete context engineering pipeline:
- OFFLINE: Build MDL index, glossary, and institutional knowledge embeddings
- ONLINE: LangGraph agent with hybrid schema retrieval and SQL caching
- COMPARE: Side-by-side comparison of 2.7 exact vs 3.3 hybrid retrieval

Usage:
    # Make sure database is running
    make db-up

    # Run offline pipeline (builds embedding index)
    python scripts/run_chapter_3_3.py --offline

    # Run online agent demo (default)
    python scripts/run_chapter_3_3.py --demo

    # Run comparison mode
    python scripts/run_chapter_3_3.py --compare

    # Test specific question
    python scripts/run_chapter_3_3.py --question "What was last month's revenue?"

    # Verbose output
    python scripts/run_chapter_3_3.py --verbose
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Sequence

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

import asyncpg
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
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated

from src.adapters import get_model_name, get_provider_name
from src.retrieval.pgvector_store import create_embeddings
from src.schema import (
    SchemaStore,
    HybridSchemaStore,
    build_mdl_index,
    build_glossary_table,
    ingest_additional_descriptions,
    ingest_institutional_knowledge,
    SQLCache,
    normalize_question,
    ensure_cache_table,
)
from src.utils.config import load_config


# =============================================================================
# Agent State
# =============================================================================

class SQLAgentState(TypedDict):
    """State for the SQL agent with cache pipeline and session memory."""
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


# =============================================================================
# Structured Output Models
# =============================================================================

class TextToSQLResult(BaseModel):
    """Output from the Text-to-SQL agent."""
    sql: str = Field(description="The generated SQL query")
    explanation: str = Field(description="Brief explanation of the query logic")
    tables_used: list[str] = Field(
        default_factory=list, description="Tables referenced in the query"
    )


class EvaluationResult(BaseModel):
    """Output from the evaluation agent."""
    score: float = Field(ge=0.0, le=1.0, description="Quality score")
    passed: bool = Field(description="Whether the result is acceptable")
    issues: list[str] = Field(default_factory=list, description="Any issues found")
    response: str = Field(description="Final response to show the user")


# =============================================================================
# Offline Pipeline: Index Building
# =============================================================================

async def run_offline_pipeline(verbose: bool = False):
    """Build the complete offline index: MDL, glossary, and institutional knowledge."""
    print("\n" + "=" * 60)
    print("OFFLINE PIPELINE: Building Indexes")
    print("=" * 60)

    config = load_config()
    db_url = config.database.url

    # Create connection pool
    pool = await asyncpg.create_pool(db_url)
    if not pool:
        print("Error: Could not connect to database")
        return

    print("\n✓ Connected to database")

    # Create LangChain embeddings
    provider = get_provider_name()
    embeddings = create_embeddings(provider)
    print(f"✓ Created embeddings ({provider})")

    # 1. Load per-schema MDL files
    mdl_dir = project_root / "data" / "mdl"
    store = SchemaStore.from_directory(mdl_dir)
    tables = store.list_tables()
    print(f"\n1. Loaded {len(tables)} tables from {mdl_dir}/")
    if verbose:
        for t in tables[:3]:
            print(f"   - {t['name']}: {t['description'][:60]}...")

    # 2. Build MDL index (embeds table descriptions, populates mdl_tables)
    print("\n2. Building MDL index...")
    count = await build_mdl_index(store, pool, embeddings)
    print(f"   ✓ Embedded {count} table descriptions")

    # 3. Load global knowledge
    global_path = project_root / "data" / "global_knowledge.json"
    with open(global_path) as f:
        global_knowledge = json.load(f)

    glossary = global_knowledge["glossary"]
    additional_descs = global_knowledge["additional_descriptions"]

    # 4. Build glossary table
    print("\n3. Building glossary table...")
    count = await build_glossary_table(glossary, pool)
    print(f"   ✓ Loaded {count} glossary terms")
    if verbose:
        for term, defn in list(glossary.items())[:3]:
            print(f"      - {term}: {defn}")

    # 5. Ingest additional descriptions
    print("\n4. Ingesting additional descriptions...")
    count = await ingest_additional_descriptions(additional_descs, pool, embeddings)
    print(f"   ✓ Embedded {count} additional descriptions")
    if verbose:
        for desc in additional_descs[:2]:
            print(f"      - {desc[:70]}...")

    # 6. Ingest institutional knowledge
    print("\n5. Ingesting institutional knowledge...")
    inst_path = project_root / "data" / "chinook_dbt" / "institutional_knowledge.txt"
    if inst_path.exists():
        text = inst_path.read_text()
        count = await ingest_institutional_knowledge(text, pool, embeddings)
        print(f"   ✓ Chunked and embedded {count} institutional knowledge chunks")
        if verbose:
            print(f"      Document: {len(text)} chars")
    else:
        # Fallback sample if file doesn't exist
        sample_text = (
            "Chinook is a digital media store with sales data from 2009-2013. "
            "Revenue is recognized at invoice creation. Customers are active if "
            "they have an invoice in the last 90 days, churned otherwise."
        )
        count = await ingest_institutional_knowledge(sample_text, pool, embeddings)
        print(f"   ⚠ Institutional knowledge file not found, using sample text")
        print(f"   ✓ Embedded {count} sample chunks")

    # 7. Ensure cache table exists
    print("\n6. Ensuring SQL cache table exists...")
    await ensure_cache_table(pool)
    print("   ✓ SQL cache table ready")

    await pool.close()

    print("\n" + "=" * 60)
    print("Offline pipeline complete!")
    print("=" * 60)
    print(f"\nIndexed:")
    print(f"  - {len(tables)} table descriptions")
    print(f"  - {len(glossary)} glossary terms")
    print(f"  - {len(additional_descs)} additional descriptions")
    print(f"  - Institutional knowledge chunks")
    print("\nReady for online agent queries.")


# =============================================================================
# Online Agent: Build LangGraph Agent with Hybrid Tools
# =============================================================================

def build_agent_with_hybrid_tools(
    db: SQLDatabase,
    llm,
    hybrid_store: HybridSchemaStore,
    store: AsyncPostgresStore,
    checkpointer: AsyncPostgresSaver,
    verbose: bool = False,
):
    """Build LangGraph agent with cache pipeline, hybrid tools, and store."""

    # ── Optional LLM tools (metrics, relationships, glossary, context) ──
    @tool
    async def get_metrics(schema_name: str, table_name: str) -> str:
        """Get metrics (aggregations, KPIs) for a table.

        Requires schema_name and table_name. Use when the question involves
        aggregations, calculated fields, or KPIs.
        """
        results = await hybrid_store.get_metrics(schema_name, table_name)
        if not results:
            return f"No metrics found for '{schema_name}.{table_name}'."

        output = [f"Metrics for {schema_name}.{table_name}:"]
        for m in results:
            output.append(f"  - {m['name']}: {m.get('description', m.get('expression', ''))}")
        return "\n".join(output)

    # get_relationships (OPTIONAL - use for joins)
    @tool
    async def get_relationships(schema_name: str, table_name: str) -> str:
        """Get relationships (foreign keys, joins) for a table.

        Requires schema_name and table_name. Use when the question requires
        joining multiple tables.
        """
        results = await hybrid_store.get_relationships(schema_name, table_name)
        if not results:
            return f"No relationships found for '{schema_name}.{table_name}'."

        output = [f"Relationships for {schema_name}.{table_name}:"]
        for r in results:
            output.append(f"  - {r['name']}: {r.get('condition', r.get('type', ''))}")
        return "\n".join(output)

    # get_glossary_entries (OPTIONAL - use when terms are ambiguous)
    @tool
    async def get_glossary_entries(terms: str) -> str:
        """Look up business term definitions from the glossary.

        Use this when the question contains business terms like 'active', 'churn', 'revenue'.
        """
        term_list = [t.strip().lower() for t in terms.split(",")]
        results = await hybrid_store.get_glossary_entries(term_list)

        if not results:
            return f"No definitions found for: {terms}"

        output = ["Glossary definitions:"]
        for r in results:
            output.append(f"  - {r['term']}: {r['definition']}")
        return "\n".join(output)

    # search_business_context (OPTIONAL - use for business rules)
    @tool
    async def search_business_context(question: str) -> str:
        """Find business context and rules relevant to the question.

        Use this to find institutional knowledge, fiscal calendar rules, etc.
        """
        results = await hybrid_store.search_business_context(question, top_k=3)
        if not results:
            return "No relevant business context found."

        output = ["Relevant business context:"]
        for r in results:
            output.append(f"  - ({r['type']}, score: {r['score']:.2f})")
            output.append(f"    {r['content'][:100]}...")
        return "\n".join(output)

    # Tools the LLM can still call (optional detail lookups)
    optional_mdl_tools = [
        get_metrics,
        get_relationships,
        get_glossary_entries,
        search_business_context,
    ]

    sql_checker_tool = QuerySQLCheckerTool(db=db, llm=llm)
    sql_query_tool = QuerySQLDatabaseTool(db=db)

    sql_gen_tools = optional_mdl_tools + [sql_checker_tool]
    eval_tools = [sql_query_tool]

    sql_model_with_tools = llm.bind_tools(sql_gen_tools + [TextToSQLResult])
    eval_model_with_tools = llm.bind_tools(eval_tools + [EvaluationResult])

    SQL_GENERATION_PROMPT = """You are an expert SQL generation agent.

The relevant tables and columns have ALREADY been retrieved for you (see schema context in messages).
You have optional tools for additional detail when needed:

<WORKFLOW>
1. READ the schema context already provided (tables + columns)
2. OPTIONAL: Use get_metrics(schema_name, table_name) if the question involves aggregations or KPIs
3. OPTIONAL: Use get_relationships(schema_name, table_name) if the question requires joining tables
4. OPTIONAL: Use get_glossary_entries("term1,term2,...") for ambiguous business terms
5. OPTIONAL: Use search_business_context() for business rules
6. GENERATE: Draft the SQL query using only columns from the schema context
7. VALIDATE: Use sql_db_query_checker to validate syntax
8. OUTPUT: Call TextToSQLResult with sql, explanation, and tables_used
</WORKFLOW>

<RULES>
- Schema context is already provided — do NOT search for tables
- Use exact column names from the schema context
- Every table/column in final SQL must come from the schema context or tool outputs
- Do NOT invent schema objects not in the context
- Validate SQL with sql_db_query_checker before final output
- SELECT statements only, no INSERT/UPDATE/DELETE
</RULES>
"""

    EVALUATION_PROMPT = """You are evaluating SQL and generating the final answer.

<WORKFLOW>
1. EXECUTE: Use sql_db_query tool to run the SQL query (ONCE only)
2. ANALYZE: Review the query results
3. SYNTHESIZE: Create a natural language response
4. SCORE: Evaluate quality (1.0 = correct SQL and useful answer, 0.9 = correct SQL with limited insight, 0.5 = partial/uncertain, 0.0 = execution error)
5. OUTPUT: Call EvaluationResult
</WORKFLOW>

<CRITICAL>
- Execute sql_db_query ONCE only
- After receiving results, immediately call EvaluationResult
- EvaluationResult requirements:
  - score: float in [0, 1]
  - passed: true when score >= 0.9, else false
  - issues: list concrete problems when score < 0.9; empty when passed=true
  - response: user-facing answer grounded in SQL results
</CRITICAL>
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

        tool_names = [tc.get("name", "") for tc in last_message.tool_calls]
        if verbose:
            print(f"    [SQL Gen] Tool calls: {tool_names}")

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
                    content="SQL generation completed", tool_call_id=tool_call.get("id", "")
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

        tool_names = [tc.get("name", "") for tc in last_message.tool_calls]
        if verbose:
            print(f"    [Eval] Tool calls: {tool_names}")

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
                    content="Evaluation completed", tool_call_id=tool_call.get("id", "")
                )
                return {
                    "response": args.get("response", ""),
                    "evaluation_score": args.get("score", 0.0),
                    "messages": [tool_message],
                }

        return {"response": "Unable to generate response", "evaluation_score": 0.0}

    # ── Programmatic retrieval node (no LLM needed) ────────────────

    async def retrieve_schema_node(state: SQLAgentState) -> dict:
        """Deterministic schema retrieval: search → fetch details.

        This is a PROGRAM node, not an LLM node.  It replaces the first
        two tool-calling steps (search_tables + get_table_details) with
        a fixed search-then-fetch pipeline — the same pattern WrenAI uses.
        Saves 2-4 LLM round-trips per query.
        """
        question = state.get("disambiguated_question", state["original_question"])

        # Step 1: Hybrid search for relevant tables
        search_results = await hybrid_store.search_tables(question, top_k=5)
        if verbose:
            names = [f"{r['schema_name']}.{r['name']}" for r in search_results]
            print(f"  [Retrieve] Found {len(search_results)} tables: {names}")

        # Step 2: Fetch full details for each table
        schema_lines = ["## Retrieved Schema Context\n"]
        tables_used = []
        for r in search_results:
            details = await hybrid_store.get_table_details(
                r["schema_name"], r["name"]
            )
            if not details:
                continue

            tables_used.append(r["name"])
            cols = ", ".join(c["name"] for c in details.get("columns", []))
            schema_lines.append(
                f"### {r['schema_name']}.{r['name']} "
                f"(score: {r['score']:.2f})\n"
                f"Description: {details['description']}\n"
                f"Columns: {cols}\n"
            )

        schema_context = "\n".join(schema_lines)
        return {
            "schema_overview": schema_context,
            "tables_used": tables_used,
            "messages": [SystemMessage(content=schema_context)],
        }

    # ── Cache pipeline nodes (Figure 3.12) ──────────────────────────

    async def disambiguate_node(state: SQLAgentState) -> dict:
        """Resolve ambiguous phrasing before cache lookup.

        In production this calls an LLM to resolve pronouns, time
        references, etc.  For the demo it passes through unchanged.
        """
        question = state["original_question"]
        if verbose:
            print(f"  [Disambiguate] \"{question}\"")
        return {"disambiguated_question": question}

    async def check_cache_node(state: SQLAgentState) -> dict:
        """Check SQL cache in the LangGraph store (cross-user)."""
        question = state.get("disambiguated_question", state["original_question"])
        cache_key = normalize_question(question)

        item = await store.aget(("cache", "sql"), cache_key)
        if item:
            cached = item.value
            if verbose:
                print(f"  [Cache] HIT (key: {cache_key[:40]}...)")
            return {
                "cache_hit": True,
                "cached_sql": cached["sql"],
                "sql": cached["sql"],
                "tables_used": cached.get("tables_used", []),
            }

        if verbose:
            print("  [Cache] MISS")
        return {"cache_hit": False, "cached_sql": "", "tables_used": []}

    def route_after_cache(state: SQLAgentState) -> str:
        """Route based on cache hit/miss (Figure 3.12 conditional edge)."""
        if state.get("cache_hit"):
            return "respond_cache_hit"
        return "load_preferences"

    async def respond_cache_hit_node(state: SQLAgentState) -> dict:
        """Execute cached SQL and synthesize answer (cache hit fast path).

        Cached SQL was validated (score >= 0.9) before storage,
        so we skip re-evaluation and go straight to synthesis.
        """
        sql = state.get("cached_sql", "")
        question = state.get("original_question", "")

        # Use the same tool the evaluation path uses (handles errors gracefully)
        result = sql_query_tool.run(sql)

        synthesis_msg = HumanMessage(
            content=(
                f"The user asked: \"{question}\"\n"
                f"Cached SQL: {sql}\n"
                f"Results: {result}\n\n"
                f"Provide a concise natural language answer."
            )
        )
        response = llm.invoke([
            SystemMessage(content="Synthesize a natural language answer from SQL results."),
            synthesis_msg,
        ])

        return {
            "results": str(result),
            "response": response.content,
            "evaluation_score": 1.0,
            "messages": [HumanMessage(content=question), response],
        }

    async def store_cache_node(state: SQLAgentState) -> dict:
        """Store high-confidence SQL in the LangGraph store (score >= 0.9).

        Uses ("cache", "sql") namespace — cross-user, backed by PostgreSQL.
        """
        score = state.get("evaluation_score", 0.0)
        sql = state.get("sql", "")
        question = state.get("disambiguated_question", state.get("original_question", ""))
        tables = state.get("tables_used", [])

        if score >= 0.9 and sql:
            # Strip MDL schema prefix — it's a logical grouping,
            # not a Postgres schema (tables live in public).
            clean_sql = re.sub(r"\bchinook\.", "", sql)
            if not tables:
                known = {
                    "invoice", "customer", "track", "artist", "album",
                    "genre", "invoice_line", "employee", "playlist",
                    "playlist_track", "media_type",
                }
                seen: set[str] = set()
                tables = [
                    w for w in clean_sql.lower().split()
                    if w in known and not (w in seen or seen.add(w))
                ]
            cache_key = normalize_question(question)
            await store.aput(
                ("cache", "sql"),
                cache_key,
                {"sql": clean_sql, "tables_used": tables or ["unknown"], "score": score},
            )
            if verbose:
                print(f"  [Cache] Stored (key: {cache_key[:40]}...)")
        elif verbose:
            print(f"  [Cache] Not stored (score: {score:.1f} < 0.9)")

        return {}

    # ── User preferences from LangGraph store (Chapter 2.8) ─────

    async def load_preferences_node(state: SQLAgentState) -> dict:
        """Load user preferences from LangGraph store into system prompt."""
        items = await store.asearch(("user", "demo-user"), limit=10)
        if not items:
            return {}

        prefs = "\n".join(f"- {item.value['content']}" for item in items)
        pref_msg = SystemMessage(content=f"## User preferences\n{prefs}")
        if verbose:
            print(f"  [Store] Loaded {len(items)} user preference(s)")
        return {"messages": [pref_msg]}

    # ── Build graph (Figure 3.12 cache pipeline) ─────────────────

    graph = StateGraph(SQLAgentState)

    # Cache pipeline nodes
    graph.add_node("disambiguate", disambiguate_node)
    graph.add_node("check_cache", check_cache_node)
    graph.add_node("respond_cache_hit", respond_cache_hit_node)
    graph.add_node("load_preferences", load_preferences_node)

    # Programmatic retrieval (no LLM) → then SQL generation (LLM)
    graph.add_node("retrieve_schema", retrieve_schema_node)
    graph.add_node("generate_sql", sql_generation_node)
    graph.add_node("sql_tools", ToolNode(sql_gen_tools, handle_tool_errors=True))
    graph.add_node("respond_sql", respond_sql_node)
    graph.add_node("evaluate", evaluation_node)
    graph.add_node("eval_tools", ToolNode(eval_tools, handle_tool_errors=True))
    graph.add_node("respond_eval", respond_eval_node)
    graph.add_node("store_cache", store_cache_node)

    # Entry: disambiguate → check_cache → conditional routing
    graph.set_entry_point("disambiguate")
    graph.add_edge("disambiguate", "check_cache")
    graph.add_conditional_edges(
        "check_cache",
        route_after_cache,
        {"respond_cache_hit": "respond_cache_hit", "load_preferences": "load_preferences"},
    )

    # Cache hit: synthesize directly → END
    graph.add_edge("respond_cache_hit", END)

    # Cache miss: load prefs → retrieve schema (program) → generate SQL (LLM)
    graph.add_edge("load_preferences", "retrieve_schema")
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
    graph.add_edge("respond_eval", "store_cache")
    graph.add_edge("store_cache", END)

    compiled = graph.compile(
        checkpointer=checkpointer,
        store=store,
    )
    return compiled


# =============================================================================
# Online Agent Demo
# =============================================================================

async def run_online_demo(question: str | None = None, verbose: bool = False):
    """Run the online agent with hybrid retrieval and SQL caching."""
    print("\n" + "=" * 60)
    print("ONLINE AGENT: Hybrid Retrieval + PostgreSQL Store")
    print("=" * 60)

    config = load_config()
    db_url = config.database.url

    # Create connection pool (for HybridSchemaStore — uses asyncpg)
    pool = await asyncpg.create_pool(db_url)
    if not pool:
        print("Error: Could not connect to database")
        return

    print("\n✓ Connected to database")

    # Create LangChain embeddings
    provider = get_provider_name()
    embeddings = create_embeddings(provider)

    # Create HybridSchemaStore
    hybrid_store = HybridSchemaStore(pool, embeddings)
    print("✓ Created HybridSchemaStore")

    # Create LLM
    model_name = get_model_name()
    if provider == "openai":
        llm = ChatOpenAI(model=model_name, temperature=0)
    elif provider == "gemini":
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    print(f"✓ Created LLM ({provider}:{model_name})")

    # Create SQLDatabase (for query execution tool)
    db = SQLDatabase.from_uri(db_url)

    # PostgreSQL-backed store and checkpointer (production-grade)
    async with (
        AsyncPostgresStore.from_conn_string(db_url) as store,
        AsyncPostgresSaver.from_conn_string(db_url) as checkpointer,
    ):
        await store.setup()
        await checkpointer.setup()
        print("✓ PostgresStore ready (cross-user cache + user preferences)")
        print("✓ PostgresSaver ready (session checkpoints)")

        # Seed user preferences into store (Chapter 2.8 pattern)
        await store.aput(
            ("user", "demo-user"), "pref-1",
            {"content": "Always show percentages as decimals"},
        )
        await store.aput(
            ("user", "demo-user"), "pref-2",
            {"content": "I work with the North America region"},
        )

        # Build agent with cache pipeline + store (Figure 3.12)
        agent = build_agent_with_hybrid_tools(
            db, llm, hybrid_store,
            store=store, checkpointer=checkpointer,
            verbose=verbose,
        )
        print("✓ Built LangGraph agent (cache pipeline + store)")

        # Demo questions — cache pipeline (Figure 3.12)
        demo_questions = [
            "How many customers are there?",          # cache miss → generate → store
            "How many customers are there?",          # cache hit → fast path
            "What are the top 5 selling genres?",     # cache miss → generate → store
            "What are the top 5 selling genres?",     # cache hit → fast path
        ]

        if question:
            demo_questions = [question]

        thread_id = "demo-session"
        config_dict = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,
        }

        for i, q in enumerate(demo_questions, 1):
            print(f"\n{'─' * 60}")
            print(f"Question {i}: \"{q}\"")
            print('─' * 60)

            # build_initial_state pattern from Chapter 2.8:
            # Only add SystemMessage on the FIRST message in a thread
            messages: list[BaseMessage] = []
            try:
                existing_state = await agent.aget_state(config_dict)
                has_history = bool(existing_state and existing_state.values.get("messages"))
            except Exception:
                has_history = False

            if not has_history:
                messages.append(SystemMessage(
                    content="You are a SQL agent. Use the hybrid schema tools to answer questions."
                ))
            messages.append(HumanMessage(content=q))

            initial_state = {
                "original_question": q,
                "disambiguated_question": "",
                "cache_hit": False,
                "cached_sql": "",
                "tables_used": [],
                "schema_overview": "",
                "sql": "",
                "results": "",
                "evaluation_score": 0.0,
                "response": "",
                "messages": messages,
            }

            # Cache check + store both happen INSIDE the graph (Figure 3.12)
            result = await agent.ainvoke(initial_state, config_dict)

            if result.get("cache_hit"):
                print(f"  ✓ CACHE HIT — used cached SQL")
                print(f"  SQL: {result.get('sql', '')[:100]}...")
            elif result.get("sql"):
                print(f"  Cache miss → generated new SQL")
                print(f"  SQL: {result['sql'][:150]}...")
                if result.get("evaluation_score", 0.0) >= 0.9:
                    print(f"  ✓ Stored in cache (score: {result['evaluation_score']:.1f})")

            if result.get("response"):
                print(f"\n  Response: {result['response'][:200]}...")

        # Show cache stats from store
        print(f"\n{'─' * 60}")
        print("Cache Statistics (from PostgresStore)")
        print('─' * 60)
        cached_items = await store.asearch(("cache", "sql"), limit=100)
        print(f"  Cached SQL entries: {len(cached_items)}")
        for item in cached_items[:5]:
            print(f"    - \"{item.key[:50]}\" → {item.value['sql'][:60]}...")

    await pool.close()

    print("\n" + "=" * 60)
    print("Online demo complete!")
    print("=" * 60)


# =============================================================================
# Comparison Mode: 2.7 Exact vs 3.3 Hybrid
# =============================================================================

async def run_comparison(question: str | None = None):
    """Compare 2.7 exact lookup vs 3.3 hybrid retrieval."""
    print("\n" + "=" * 60)
    print("COMPARISON: 2.7 Exact vs 3.3 Hybrid")
    print("=" * 60)

    test_question = question or "How are sales trending by country?"
    print(f"\nQuestion: \"{test_question}\"")

    config = load_config()
    db_url = config.database.url

    pool = await asyncpg.create_pool(db_url)
    if not pool:
        print("Error: Could not connect to database")
        return

    provider = get_provider_name()
    embeddings = create_embeddings(provider)
    hybrid_store = HybridSchemaStore(pool, embeddings)

    # Load 2.7 exact store
    mdl_dir = project_root / "data" / "mdl"
    exact_store = SchemaStore.from_directory(mdl_dir)

    print(f"\n{'─' * 60}")
    print("2.7 EXACT APPROACH")
    print('─' * 60)
    all_tables = exact_store.list_tables()
    print(f"  list_tables() returns ALL {len(all_tables)} tables:")
    for t in all_tables[:5]:
        print(f"    - {t['name']}: {t['description'][:50]}...")
    if len(all_tables) > 5:
        print(f"    ... and {len(all_tables) - 5} more")

    print(f"\n{'─' * 60}")
    print("3.3 HYBRID APPROACH")
    print('─' * 60)
    results = await hybrid_store.search_tables(test_question, top_k=5)
    print(f"  search_tables() returns TOP {len(results)} relevant tables:")
    for r in results:
        print(f"    - {r['schema_name']}.{r['name']} (score: {r['score']:.2f}): {r['description'][:50]}...")

    print(f"\n{'─' * 60}")
    print("ANALYSIS")
    print('─' * 60)
    print("  2.7: Agent must scan all table descriptions → high token cost")
    print("  3.3: Hybrid search pre-filters to top-k relevant → lower token cost")
    print(f"  Token savings: ~{(1 - len(results)/len(all_tables)) * 100:.0f}% reduction in table context")

    await pool.close()


# =============================================================================
# Main
# =============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run 3.3 Context Engineering pipeline")
    parser.add_argument("--offline", action="store_true", help="Run offline index build")
    parser.add_argument("--demo", action="store_true", help="Run online agent demo (default)")
    parser.add_argument("--compare", action="store_true", help="Run 2.7 vs 3.3 comparison")
    parser.add_argument("--question", "-q", help="Test specific question")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("3.3 Context Engineering: Production Pipeline")
    print("=" * 60)

    try:
        if args.offline:
            await run_offline_pipeline(args.verbose)
        elif args.demo:
            await run_online_demo(args.question, args.verbose)
        elif args.compare:
            await run_comparison(args.question)
        else:
            # Default (no flags): run both offline + demo
            await run_offline_pipeline(args.verbose)
            await run_online_demo(args.question, args.verbose)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
