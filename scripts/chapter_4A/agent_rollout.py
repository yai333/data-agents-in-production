"""APO rollout wrapper for the SQL agent.

Wraps the chapter 3.5 LangGraph pipeline as an Agent Lightning rollout.
Same graph structure as 3.5:
  retrieve_schema → generate_sql ↔ sql_tools → respond_sql
  → evaluate ↔ eval_tools → respond_eval → END

Differences from 3.5:
  - sql_generation_node reads prompt from state["sql_gen_prompt"] (APO-optimized)
  - No cache lookup / disambiguation nodes
  - No checkpointer / store (stateless rollouts)
  - retrieve_schema uses pre-computed schema when available (APO efficiency)
"""

from src.adapters import get_model_name, get_provider_name
from reward import compute_sql_reward_detailed
from agentlightning.types import PromptTemplate
from agentlightning import rollout
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    QuerySQLDatabaseTool,
    QuerySQLCheckerTool,
)
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
import asyncio
import os
import sys
import threading
from pathlib import Path
from typing import Sequence
from typing_extensions import TypedDict, Annotated

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# Background event loop for async database operations
# ---------------------------------------------------------------------------
# asyncpg connections are bound to the event loop they were created on.
# Since the @rollout runs inside SharedMemoryExecutionStrategy's thread
# context, we use a dedicated background loop for ALL async operations:
# pool creation, schema retrieval, MDL tool calls. This avoids nested
# asyncio.run() failures and event-loop mismatch errors.

_bg_loop = None
_bg_thread = None


def _ensure_bg_loop():
    """Start a persistent background event loop in a daemon thread."""
    global _bg_loop, _bg_thread
    if _bg_loop is not None and _bg_loop.is_running():
        return _bg_loop
    _bg_loop = asyncio.new_event_loop()
    _bg_thread = threading.Thread(target=_bg_loop.run_forever, daemon=True)
    _bg_thread.start()
    return _bg_loop


def _run_async(coro):
    """Run an async coroutine on the background event loop (thread-safe)."""
    loop = _ensure_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=60)


# ---------------------------------------------------------------------------
# Module-level shared resources (lazy-initialized on first use)
# ---------------------------------------------------------------------------

_db: SQLDatabase | None = None
_llm = None
_hybrid_store = None
_pool = None
_agent = None
def _get_db() -> SQLDatabase:
    global _db
    if _db is None:
        db_url = os.getenv(
            "CHINOOK_DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/chinook",
        )
        _db = SQLDatabase.from_uri(db_url)
    return _db


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    provider = get_provider_name()
    model = get_model_name()
    if provider == "gemini":
        _llm = ChatGoogleGenerativeAI(model=model, temperature=0, cache=False)
    else:
        _llm = ChatOpenAI(model=model, temperature=0, cache=False)
    return _llm


def _get_hybrid_store():
    """Lazy-init HybridSchemaStore using the background event loop."""
    global _hybrid_store, _pool
    if _hybrid_store is not None:
        return _hybrid_store

    import asyncpg
    from src.retrieval.pgvector_store import create_embeddings
    from src.schema import HybridSchemaStore

    db_url = os.getenv(
        "CHINOOK_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/chinook",
    )
    provider = get_provider_name()
    embeddings = create_embeddings(provider)

    async def _init_pool():
        pool = asyncpg.create_pool(db_url)
        await pool  # Initialize connections
        return pool

    _pool = _run_async(_init_pool())
    _hybrid_store = HybridSchemaStore(_pool, embeddings)
    return _hybrid_store


# ---------------------------------------------------------------------------
# Agent State (matches chapter 3.5 SQLAgentState, minus cache fields)
# ---------------------------------------------------------------------------

class APOAgentState(TypedDict):
    """State for the APO SQL agent."""
    original_question: str
    # APO-optimized prompt (replaces SQL_GENERATION_PROMPT)
    sql_gen_prompt: str
    tables_used: list[str]
    schema_overview: str          # Pre-computed or retrieved schema context
    sql: str
    results: str
    evaluation_score: float
    response: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


class TextToSQLResult(BaseModel):
    """Structured output from the SQL generation node."""
    sql: str = Field(description="The generated SQL query")
    explanation: str = Field(
        description="Brief explanation of the query logic")
    tables_used: list[str] = Field(
        default_factory=list, description="Tables referenced in the query")


class EvaluationResult(BaseModel):
    """Structured output from the evaluation node."""
    score: float = Field(ge=0.0, le=1.0, description="Quality score")
    passed: bool = Field(description="Whether the result is acceptable")
    issues: list[str] = Field(
        default_factory=list, description="Any issues found")
    response: str = Field(description="Final response to show the user")


# ---------------------------------------------------------------------------
# Default prompts (chapter 3.5 originals)
# APO replaces SQL_GENERATION_PROMPT; EVALUATION_PROMPT stays fixed.
# ---------------------------------------------------------------------------

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
</WORKFLOW>"""


# ---------------------------------------------------------------------------
# Build the agent graph (same structure as chapter 3.5, minus cache)
# ---------------------------------------------------------------------------

def _build_apo_agent():
    """Build the LangGraph SQL agent for APO.

    Same pipeline as chapter 3.5:
      retrieve_schema → generate_sql ↔ sql_tools → respond_sql
      → evaluate ↔ eval_tools → respond_eval → END

    Only APO-specific change: sql_generation_node reads the prompt from
    state["sql_gen_prompt"] instead of the hardcoded SQL_GENERATION_PROMPT.
    No cache, no chart, no checkpointer/store.
    """
    db = _get_db()
    llm = _get_llm()
    hybrid_store = _get_hybrid_store()

    # -- MDL tools (same as chapter 3.5, sync wrappers via _run_async) --

    @tool
    def get_metrics(schema_name: str, table_name: str) -> str:
        """Get metrics (aggregations, KPIs) for a table."""
        results = _run_async(hybrid_store.get_metrics(schema_name, table_name))
        if not results:
            return f"No metrics found for '{schema_name}.{table_name}'."
        output = [f"Metrics for {schema_name}.{table_name}:"]
        for m in results:
            output.append(
                f"  - {m['name']}: {m.get('description', m.get('expression', ''))}")
        return "\n".join(output)

    @tool
    def get_relationships(schema_name: str, table_name: str) -> str:
        """Get relationships (foreign keys, joins) for a table."""
        results = _run_async(
            hybrid_store.get_relationships(schema_name, table_name))
        if not results:
            return f"No relationships found for '{schema_name}.{table_name}'."
        output = [f"Relationships for {schema_name}.{table_name}:"]
        for r in results:
            output.append(
                f"  - {r['name']}: {r.get('condition', r.get('type', ''))}")
        return "\n".join(output)

    @tool
    def get_glossary_entries(terms: str) -> str:
        """Look up business term definitions from the glossary."""
        term_list = [t.strip().lower() for t in terms.split(",")]
        results = _run_async(hybrid_store.get_glossary_entries(term_list))
        if not results:
            return f"No definitions found for: {terms}"
        output = ["Glossary definitions:"]
        for r in results:
            output.append(f"  - {r['term']}: {r['definition']}")
        return "\n".join(output)

    @tool
    def search_business_context(question: str) -> str:
        """Find business context and rules relevant to the question."""
        results = _run_async(
            hybrid_store.search_business_context(question, top_k=3))
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

    # -- Nodes (matching chapter 3.5 structure) --

    def retrieve_schema_node(state):
        """Retrieve schema context for the question.

        Uses pre-computed schema_overview when available (APO optimization:
        same question always produces the same schema, no need to
        re-retrieve for every prompt iteration).
        Falls back to dynamic HybridSchemaStore retrieval.
        """
        if state.get("schema_overview"):
            schema_context = state["schema_overview"]
        else:
            question = state["original_question"]
            search_results = _run_async(
                hybrid_store.search_tables(question, top_k=5))
            schema_lines = ["## Retrieved Schema Context\n"]
            for r in search_results:
                details = _run_async(
                    hybrid_store.get_table_details(r["schema_name"], r["name"]))
                if not details:
                    continue
                cols = ", ".join(
                    c["name"] for c in details.get("columns", []))
                schema_lines.append(
                    f"### {r['schema_name']}.{r['name']} "
                    f"(score: {r['score']:.2f})\n"
                    f"Description: {details['description']}\n"
                    f"Columns: {cols}\n"
                )
            schema_context = "\n".join(schema_lines)

        return {
            "schema_overview": schema_context,
            "messages": [SystemMessage(content=schema_context)],
        }

    def sql_generation_node(state):
        """Generate SQL using the APO-optimized prompt from state."""
        prompt = state.get("sql_gen_prompt", SQL_GENERATION_PROMPT)
        return {
            "messages": [
                sql_model_with_tools.invoke(
                    [SystemMessage(content=prompt)]
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

    # -- Graph (same structure as chapter 3.5) --

    graph = StateGraph(APOAgentState)
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

    return graph.compile()


def _get_agent():
    """Lazy-init the compiled agent graph."""
    global _agent
    if _agent is None:
        _agent = _build_apo_agent()
    return _agent


# ---------------------------------------------------------------------------
# Rollout function (called by Agent Lightning for each task)
# ---------------------------------------------------------------------------

@rollout
def sql_agent_rollout(task: dict, prompt_template: PromptTemplate) -> float:
    """Execute the full SQL agent pipeline and return a reward.

    Agent Lightning calls this for each task during optimization.
    The prompt_template changes between APO iterations (beam search);
    everything else stays fixed.

    Schema and question flow through LangGraph messages (same as 3.5),
    NOT through prompt placeholders. APO optimizes the instructions only.

    Args:
        task: Dict with keys: question, expected_sql, schema_context
        prompt_template: APO-optimized prompt (no data placeholders)

    Returns:
        Reward float in [0.0, 1.0]
    """
    agent = _get_agent()

    # APO-optimized prompt — no placeholders to fill.
    # Schema arrives via retrieve_schema node; question via HumanMessage.
    system_prompt = prompt_template.template

    initial_state = {
        "original_question": task["question"],
        "sql_gen_prompt": system_prompt,
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

    # No Langfuse — AgentOps captures all spans for the built-in dashboard.
    config = {"recursion_limit": 50}

    generated_sql = ""
    try:
        result = agent.invoke(initial_state, config)
        generated_sql = result.get("sql") or ""
    except Exception:
        generated_sql = ""

    breakdown = compute_sql_reward_detailed(
        generated_sql=generated_sql,
        expected_sql=task["expected_sql"],
    )

    return breakdown.total


# ---------------------------------------------------------------------------
# Baseline prompt template (chapter 3.5's SQL_GENERATION_PROMPT as APO seed)
# ---------------------------------------------------------------------------

def baseline_prompt_template() -> PromptTemplate:
    """The chapter 3.5 SQL generation prompt as an APO PromptTemplate.

    No data placeholders ({schema_context}, {question}) — schema and question
    flow through LangGraph messages, same as chapter 3.5. APO optimizes
    the instructions while the data pipeline stays fixed.
    """
    return PromptTemplate(
        template=SQL_GENERATION_PROMPT,
        engine="f-string",
    )
