#!/usr/bin/env python3
"""2.8 Agent Memory: Session Memory and Checkpoints.

This script demonstrates the memory patterns from Chapter 2.8:
- Working memory: LangGraph state within a single request
- Session memory: Multi-turn conversations via thread_id
- Long-term memory: InMemorySaver checkpointer (SQLite/Postgres in production)
- Context window management: Sliding window for history

The chapter builds on 2.7's disambiguation pipeline by adding memory for continuity.

Usage:
    # Make sure database is running
    make db-up

    # Run memory demonstrations
    python scripts/run_chapter_2_8.py

    # Show multi-turn conversation
    python scripts/run_chapter_2_8.py --multi-turn

    # Show checkpoint inspection
    python scripts/run_chapter_2_8.py --checkpoints

    # Interactive mode
    python scripts/run_chapter_2_8.py --interactive
"""

from src.adapters import get_model_name, get_provider_name
from src.utils.config import load_config
from langgraph.prebuilt import ToolNode
# Long-term memory (Chapter 2.8)
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_community.tools.sql_database.tool import (
    QuerySQLDatabaseTool,
    QuerySQLCheckerTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Annotated, Sequence
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")


# =============================================================================
# build_initial_state Pattern (Chapter 2.8 - Key Pattern)
# =============================================================================

async def build_initial_state(
    graph: CompiledStateGraph,
    question: str,
    config: dict,
    user_name: str | None = None
) -> dict:
    """Construct initial state, adding system context only for new threads.

    Why check existing state? If we add SystemMessage on every invocation,
    the conversation accumulates multiple SystemMessages. This breaks
    message trimming (which expects Human/AI alternation) and wastes tokens.

    This is the canonical pattern from Chapter 2.8.
    """
    # Check if this thread already has messages
    existing_state = None
    try:
        existing_state = await graph.aget_state(config)
    except Exception:
        pass  # New thread, no checkpoint exists

    messages = []

    # Only add SystemMessage for the FIRST message in a thread
    if not existing_state or not existing_state.values.get("messages"):
        if user_name:
            messages.append(SystemMessage(
                content=f"Current user: {user_name}"))

    messages.append(HumanMessage(content=question))
    return {"messages": messages}


# =============================================================================
# Agent State with Memory Fields (Chapter 2.8)
# =============================================================================

class SQLAgentState(TypedDict):
    """State for the SQL agent with memory support."""
    original_question: str
    schema_overview: str
    few_shot_examples: str
    sql: str
    results: str
    evaluation_score: float
    response: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


# =============================================================================
# Structured Output Models (Chapter 2.7 pattern)
# =============================================================================

class TextToSQLResult(BaseModel):
    """Output from the Text-to-SQL agent."""
    sql: str = Field(description="The generated SQL query")
    explanation: str = Field(
        description="Brief explanation of the query logic")
    tables_used: list[str] = Field(
        default_factory=list, description="Tables referenced in the query")


class EvaluationResult(BaseModel):
    """Output from the evaluation agent."""
    score: float = Field(ge=0.0, le=1.0, description="Quality score")
    passed: bool = Field(description="Whether the result is acceptable")
    issues: list[str] = Field(default_factory=list,
                              description="Any issues found")
    response: str = Field(description="Final response to show the user")


# =============================================================================
# Schema and Semantic Layer (same as Chapter 2.7)
# =============================================================================

CHINOOK_SCHEMA = {
    "artist": {
        "columns": ["artist_id", "name"],
        "description": "Music artists",
        "sample_values": {"name": ["AC/DC", "Accept", "Aerosmith"]}
    },
    "album": {
        "columns": ["album_id", "title", "artist_id"],
        "description": "Albums by artists",
        "relationships": ["artist_id -> artist.artist_id"]
    },
    "track": {
        "columns": ["track_id", "name", "album_id", "media_type_id", "genre_id",
                    "composer", "milliseconds", "bytes", "unit_price"],
        "description": "Individual tracks/songs",
        "relationships": ["album_id -> album.album_id", "genre_id -> genre.genre_id"]
    },
    "genre": {
        "columns": ["genre_id", "name"],
        "description": "Music genres",
        "sample_values": {"name": ["Rock", "Jazz", "Metal", "Alternative"]}
    },
    "customer": {
        "columns": ["customer_id", "first_name", "last_name", "company", "address",
                    "city", "state", "country", "postal_code", "phone", "fax",
                    "email", "support_rep_id"],
        "description": "Customers who purchased music"
    },
    "invoice": {
        "columns": ["invoice_id", "customer_id", "invoice_date", "billing_address",
                    "billing_city", "billing_state", "billing_country",
                    "billing_postal_code", "total"],
        "description": "Sales invoices",
        "relationships": ["customer_id -> customer.customer_id"]
    },
    "invoice_line": {
        "columns": ["invoice_line_id", "invoice_id", "track_id", "unit_price", "quantity"],
        "description": "Line items on invoices",
        "relationships": ["invoice_id -> invoice.invoice_id", "track_id -> track.track_id"]
    },
}

GLOSSARY = {
    "top customers": "Customers ranked by total invoice amount (SUM of invoice.total)",
    "active customer": "Customer with at least one invoice in the last 12 months",
    "churned customer": "Customer with no invoice in the last 12 months",
    "revenue": "SUM(invoice.total) - represents total sales",
    "bestselling": "Ranked by quantity sold (SUM of invoice_line.quantity)",
    "popular": "Ranked by number of purchases or plays",
}

METRICS = {
    "invoice": {
        "revenue": {"definition": "Total sales amount", "sql_pattern": "SUM(invoice.total)"},
        "average_order_value": {"definition": "Average invoice total", "sql_pattern": "AVG(invoice.total)"},
    },
    "customer": {
        "customer_count": {"definition": "Number of unique customers", "sql_pattern": "COUNT(DISTINCT customer_id)"},
    },
}

BUSINESS_INFO = {
    "fiscal_year": "Calendar year (Jan 1 - Dec 31)",
    "default_limit": "10 for 'top N' queries unless specified",
    "default_time_range": "Last 12 months for trend queries",
    "currency": "USD for all monetary values",
}


# =============================================================================
# Disambiguation Tools (same as Chapter 2.7)
# =============================================================================

@tool
def get_table_details(table_name: str) -> str:
    """Get detailed schema for a specific table including columns and relationships."""
    table_name_lower = table_name.lower()
    if table_name_lower not in CHINOOK_SCHEMA:
        available = ", ".join(CHINOOK_SCHEMA.keys())
        return f"Error: Table '{table_name}' not found. Available tables: {available}"

    info = CHINOOK_SCHEMA[table_name_lower]
    result = [
        f"Table: {table_name_lower}",
        f"Description: {info['description']}",
        f"Columns: {', '.join(info['columns'])}"
    ]
    if "relationships" in info:
        result.append(f"Relationships: {', '.join(info['relationships'])}")
    if "sample_values" in info:
        samples = [f"{col}: {vals[:3]}" for col,
                   vals in info["sample_values"].items()]
        result.append(f"Sample values: {'; '.join(samples)}")
    return "\n".join(result)


@tool
def get_relationships(table_name: str) -> str:
    """Get foreign key relationships for a table."""
    table_name_lower = table_name.lower()
    if table_name_lower not in CHINOOK_SCHEMA:
        return f"Error: Table '{table_name}' not found."

    info = CHINOOK_SCHEMA[table_name_lower]
    if "relationships" not in info:
        return f"Table '{table_name_lower}' has no foreign key relationships defined."

    result = [f"Relationships for {table_name_lower}:"]
    for rel in info["relationships"]:
        result.append(f"  - {rel}")
    return "\n".join(result)


@tool
def get_metrics(table_name: str) -> str:
    """Get metric definitions for a table."""
    table_name_lower = table_name.lower()
    if table_name_lower not in METRICS:
        available = ", ".join(METRICS.keys())
        return f"No metrics for '{table_name}'. Tables with metrics: {available}"

    table_metrics = METRICS[table_name_lower]
    results = [f"Metrics for {table_name_lower}:"]
    for metric_name, m in table_metrics.items():
        results.append(f"  - {metric_name}: {m['definition']}")
        results.append(f"    SQL: {m['sql_pattern']}")
    return "\n".join(results)


@tool
def get_glossaries(terms: str) -> str:
    """Look up business term definitions."""
    term_list = [t.strip().lower() for t in terms.split(",")]
    results = []
    for term in term_list:
        if term in GLOSSARY:
            results.append(f"'{term}': {GLOSSARY[term]}")
        else:
            matches = [(k, v)
                       for k, v in GLOSSARY.items() if term in k or k in term]
            if matches:
                for k, v in matches:
                    results.append(f"'{k}': {v}")
            else:
                results.append(f"'{term}': No definition found")
    return "\n".join(results)


@tool
def get_additional_business_info(topic: str) -> str:
    """Get business conventions and rules."""
    topic_lower = topic.lower().replace(" ", "_")
    if topic_lower in BUSINESS_INFO:
        return f"{topic}: {BUSINESS_INFO[topic_lower]}"

    matches = [(k, v) for k, v in BUSINESS_INFO.items()
               if topic_lower in k or any(word in k for word in topic_lower.split("_"))]
    if matches:
        return "\n".join([f"{k.replace('_', ' ')}: {v}" for k, v in matches])

    available = ", ".join(k.replace("_", " ") for k in BUSINESS_INFO.keys())
    return f"No information for '{topic}'. Available: {available}"


# =============================================================================
# Agent Prompts (XML-tagged, Chapter 2.7 pattern)
# =============================================================================

def get_schema_overview() -> str:
    """Generate schema overview."""
    tables = [f"- {name}: {info['description']}" for name,
              info in CHINOOK_SCHEMA.items()]
    return "Available tables:\n" + "\n".join(tables)


SQL_GENERATION_PROMPT = """You are an expert SQL generation agent with session memory.

<SCHEMA>
{schema_overview}
</SCHEMA>

<WORKFLOW>
1. DISCOVERY: Use tools to understand the schema
   - get_table_details: Get column names and types
   - get_relationships: Understand join paths
   - get_metrics: Get metric definitions
   - get_glossaries: Understand business terms
   - get_additional_business_info: Get conventions

2. SQL GENERATION: Draft the query
   - Use exact table and column names
   - Build clear SQL with correct joins
   - Add LIMIT 100 for detail queries

3. VALIDATION: Validate before outputting
   - Use sql_db_query_checker to validate syntax

4. FINAL OUTPUT: Call TextToSQLResult with your SQL
</WORKFLOW>

<MULTI_TURN_CONTEXT>
- If the question references previous context ("break it down", "top 5 of those"), use conversation history
- "Break it down by X" = Add GROUP BY X to previous query concept
- "Just the top N" = Add ORDER BY ... LIMIT N
- Pronouns like "those", "them", "it" refer to previous query's subject
- Remember user clarifications (e.g., "by 'active' I mean...")
</MULTI_TURN_CONTEXT>

<RULES>
- SELECT statements only
- Use explicit column names (not SELECT *)
- Always validate SQL with sql_db_query_checker before final output
- Emit TextToSQLResult only after successful validation
</RULES>
"""

EVALUATION_PROMPT = """You are evaluating SQL and generating the final answer.

<WORKFLOW>
1. EXECUTE: Use sql_db_query tool to run the SQL query
   - Execute ONCE only

2. ANALYZE: Review the query results
   - What data was returned?
   - Does it answer the user's question?

3. SYNTHESIZE: Create a natural language response
   - Summarize key findings
   - Present data clearly

4. SCORE: Evaluate quality
   - 1.0: Completely answers the question
   - 0.5: Empty results but query was correct
   - 0.0: Error occurred

5. FINAL OUTPUT: Call EvaluationResult
</WORKFLOW>

<CRITICAL>
- Execute sql_db_query ONCE only
- After receiving results, immediately call EvaluationResult
</CRITICAL>
"""


# =============================================================================
# Build Agent with Memory (Chapter 2.7 pattern + Chapter 2.8 checkpointer)
# =============================================================================

def build_agent_with_memory(db: SQLDatabase, llm, store: InMemoryStore | None = None):
    """Build SQL agent with checkpointer and store for memory.

    Uses the canonical LangGraph pattern from Chapter 2.7:
    - bind_tools([...tools, OutputSchema]) for ReAct loops
    - Routing checks for output tool name to exit loop
    - ToolMessage responses for OpenAI compatibility

    Chapter 2.8 memory additions:
    - InMemorySaver checkpointer for session memory (tied to thread_id)
    - InMemoryStore for long-term memory (tied to user_id, cross-session)
    """
    mdl_tools = [
        get_table_details,
        get_relationships,
        get_metrics,
        get_glossaries,
        get_additional_business_info,
    ]

    sql_checker_tool = QuerySQLCheckerTool(db=db, llm=llm)
    sql_query_tool = QuerySQLDatabaseTool(db=db)

    sql_gen_tools = mdl_tools + [sql_checker_tool]
    eval_tools = [sql_query_tool]

    sql_model_with_tools = llm.bind_tools(sql_gen_tools + [TextToSQLResult])
    eval_model_with_tools = llm.bind_tools(eval_tools + [EvaluationResult])

    def sql_generation_node(state: SQLAgentState) -> dict:
        prompt = SQL_GENERATION_PROMPT.format(
            schema_overview=state.get("schema_overview", get_schema_overview())
        )
        return {
            "messages": [
                sql_model_with_tools.invoke(
                    [SystemMessage(content=prompt)] +
                    list(state.get("messages", []))
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
        tool_message = None

        if last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            if tool_call.get("name") == "TextToSQLResult":
                args = tool_call.get("args", {})
                sql = args.get("sql", "")
                tool_message = ToolMessage(
                    content="SQL generation completed",
                    tool_call_id=tool_call.get("id", "")
                )

        eval_instruction = HumanMessage(
            content=f"Execute this SQL and provide the final answer:\n\nSQL: {sql}"
        )

        new_messages = []
        if tool_message:
            new_messages.append(tool_message)
        new_messages.append(eval_instruction)

        return {"sql": sql, "messages": new_messages}

    def evaluation_node(state: SQLAgentState) -> dict:
        return {
            "messages": [
                eval_model_with_tools.invoke(
                    [SystemMessage(content=EVALUATION_PROMPT)] +
                    list(state.get("messages", []))
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
                    content="Evaluation completed",
                    tool_call_id=tool_call.get("id", "")
                )
                return {
                    "response": args.get("response", ""),
                    "evaluation_score": args.get("score", 0.0),
                    "messages": [tool_message]
                }

        return {"response": "Unable to generate response", "evaluation_score": 0.0}

    graph = StateGraph(SQLAgentState)

    graph.add_node("generate_sql", sql_generation_node)
    graph.add_node("sql_tools", ToolNode(sql_gen_tools))
    graph.add_node("respond_sql", respond_sql_node)
    graph.add_node("evaluate", evaluation_node)
    graph.add_node("eval_tools", ToolNode(eval_tools))
    graph.add_node("respond_eval", respond_eval_node)

    graph.set_entry_point("generate_sql")

    graph.add_conditional_edges(
        "generate_sql",
        should_continue_sql,
        {"sql_tools": "sql_tools", "respond_sql": "respond_sql"}
    )
    graph.add_edge("sql_tools", "generate_sql")
    graph.add_edge("respond_sql", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        should_continue_eval,
        {"eval_tools": "eval_tools", "respond_eval": "respond_eval"}
    )
    graph.add_edge("eval_tools", "evaluate")
    graph.add_edge("respond_eval", END)

    # Chapter 2.8: Compile with both checkpointer and store
    checkpointer = InMemorySaver()
    compiled = graph.compile(checkpointer=checkpointer, store=store)
    return compiled, checkpointer, store


# =============================================================================
# Demo Functions
# =============================================================================

async def demo_session_memory(agent: CompiledStateGraph, config: dict):
    """Demonstrate session memory with multi-turn conversation.

    Uses the build_initial_state pattern from Chapter 2.8:
    - First message: SystemMessage + HumanMessage
    - Follow-up messages: Only HumanMessage (SystemMessage already in history)
    """
    print("\n" + "=" * 60)
    print("Session Memory Demo: Multi-turn Conversation")
    print("=" * 60)
    print(f"\nUsing thread_id: {config['configurable']['thread_id']}")

    conversation = [
        "How many customers do we have?",
        "Break it down by country",
        "Just the top 5",
    ]

    user_name = "Alice"  # Simulated user context

    for i, question in enumerate(conversation, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {question}")

        # Use build_initial_state pattern from Chapter 2.8
        # This only adds SystemMessage on the FIRST message
        initial_state = await build_initial_state(
            agent, question, config, user_name=user_name
        )

        # Merge with other state fields
        full_state = {
            "original_question": question,
            "schema_overview": get_schema_overview(),
            "few_shot_examples": "",
            "sql": "",
            "results": "",
            "evaluation_score": 0.0,
            "response": "",
            **initial_state,  # messages from build_initial_state
        }

        result = await agent.ainvoke(full_state, config)

        if result.get("sql"):
            print(f"SQL: {result['sql'][:150]}...")
        if result.get("response"):
            print(f"Response: {result['response'][:300]}...")

    # Verify SystemMessage count
    final_state = await agent.aget_state(config)
    messages = final_state.values.get("messages", [])
    system_count = sum(1 for m in messages if isinstance(m, SystemMessage))
    print(
        f"\n[Verification] Total messages: {len(messages)}, SystemMessages: {system_count}")
    if system_count == 1:
        print("✓ build_initial_state pattern working correctly (only 1 SystemMessage)")


async def demo_checkpoint_inspection(agent: CompiledStateGraph, checkpointer, config: dict):
    """Demonstrate checkpoint history inspection."""
    print("\n" + "=" * 60)
    print("Checkpoint Inspection Demo")
    print("=" * 60)

    questions = [
        "How many artists?",
        "How many albums?",
    ]

    user_name = "Bob"

    for question in questions:
        # Use build_initial_state pattern
        initial_state = await build_initial_state(
            agent, question, config, user_name=user_name
        )

        full_state = {
            "original_question": question,
            "schema_overview": get_schema_overview(),
            "few_shot_examples": "",
            "sql": "",
            "results": "",
            "evaluation_score": 0.0,
            "response": "",
            **initial_state,
        }

        await agent.ainvoke(full_state, config)

    print("\nCheckpoint History:")
    print("-" * 40)

    try:
        history = [h async for h in agent.aget_state_history(config)]
        for i, checkpoint in enumerate(history[:5]):
            state = checkpoint.values
            print(f"\nCheckpoint {i}:")
            print(
                f"  Question: {state.get('original_question', 'N/A')[:50]}...")
            print(f"  SQL: {state.get('sql', 'N/A')[:50]}...")
            if state.get("messages"):
                # Count message types
                msgs = state["messages"]
                sys_count = sum(
                    1 for m in msgs if isinstance(m, SystemMessage))
                human_count = sum(
                    1 for m in msgs if isinstance(m, HumanMessage))
                print(
                    f"  Messages: {len(msgs)} total (System: {sys_count}, Human: {human_count})")
    except Exception as e:
        print(f"  Error inspecting history: {e}")


def demo_sliding_window():
    """Demonstrate sliding window for context management."""
    print("\n" + "=" * 60)
    print("Sliding Window Demo: Context Management")
    print("=" * 60)

    from langchain_core.messages.utils import trim_messages, count_tokens_approximately

    messages = [
        HumanMessage(content="What tables do we have?"),
        AIMessage(
            content="You have artist, album, track, genre, customer, invoice..."),
        HumanMessage(content="Show me customer count"),
        AIMessage(content="SELECT COUNT(*) FROM customer -- Result: 59 customers"),
        HumanMessage(content="Break it down by country"),
        AIMessage(
            content="SELECT country, COUNT(*) FROM customer GROUP BY country"),
        HumanMessage(
            content="When I say 'active', I mean ordered in last 90 days"),
        AIMessage(content="Got it! I'll use that definition going forward."),
        HumanMessage(content="Show active customers by country"),
        AIMessage(content="SELECT country, COUNT(*) FROM customer WHERE..."),
    ]

    print(f"\nOriginal conversation: {len(messages)} messages")

    trimmed = trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=200,
        start_on="human",
        end_on=("human", "tool"),
    )
    print(f"After trim_messages: {len(trimmed)} messages")
    for i, msg in enumerate(trimmed):
        prefix = msg.__class__.__name__[:3]
        print(f"  {i}. [{prefix}] {msg.content[:60]}...")


async def demo_store_long_term_memory(
    agent: CompiledStateGraph,
    store: InMemoryStore,
    config: dict
):
    """Demonstrate Store for long-term memory (Chapter 2.8).

    Store is separate from checkpointer:
    - Checkpointer (InMemorySaver): Session memory, tied to thread_id
    - Store (InMemoryStore): Long-term memory, tied to user_id, cross-session

    Store API:
    - store.put(namespace, key, value) → save/update item
    - store.get(namespace, key) → exact key lookup, returns Item | None
    - store.search(namespace, query, limit) → semantic search (requires index)
    """
    print("\n" + "=" * 60)
    print("Store Demo: Long-term Memory (Chapter 2.8)")
    print("=" * 60)

    user_id = "user-123"

    # 1. Save user preferences
    print("\n1. Saving user preferences...")
    preferences = {
        "date_format": "YYYY-MM-DD",
        "timezone": "UTC",
        "default_limit": 10,
    }
    namespace = ("preferences", user_id)
    store.put(namespace, "settings", preferences)
    print(f"   Saved: {preferences}")

    # 2. Load preferences
    print("\n2. Loading user preferences...")
    item = store.get(namespace, "settings")
    if item:
        print(f"   Loaded: {item.value}")
    else:
        print("   Not found")

    # 3. Handle query with preferences pattern (Chapter 2.8)
    print("\n3. Demonstrating handle_query with preferences...")

    def handle_query_with_preferences(
        agent: CompiledStateGraph,
        store: InMemoryStore,
        user_id: str,
        session_id: str,
        question: str
    ):
        """Load preferences from store and inject into initial state.

        This is the canonical pattern from Chapter 2.8 for combining
        session memory (checkpointer) with long-term memory (store).
        """
        # Load user preferences from store
        namespace = ("preferences", user_id)
        item = store.get(namespace, "settings")
        preferences = item.value if item else {}

        config = {
            "configurable": {
                "thread_id": f"{user_id}-{session_id}",
                "user_id": user_id,
            }
        }

        # Inject preferences into initial state
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "user_preferences": preferences,  # Available to all nodes
        }

        print(f"   Config: thread_id={config['configurable']['thread_id']}")
        print(f"   Preferences injected: {preferences}")
        return initial_state, config

    initial_state, query_config = handle_query_with_preferences(
        agent, store, user_id, "session-abc", "Show top customers"
    )

    # 4. Cross-session persistence
    print("\n4. Cross-session persistence...")
    print("   Store data persists across sessions (different thread_ids)")
    print("   - Session 1 thread_id: user-123-session-abc")
    print("   - Session 2 thread_id: user-123-session-xyz")
    print("   - Both sessions access same store namespace: ('preferences', 'user-123')")

    # Verify data persists
    item = store.get(("preferences", user_id), "settings")
    print(f"   Preferences still available: {item is not None}")

    print("\n✓ Store demo complete")
    print("  - Checkpointer: Session memory (conversation history)")
    print("  - Store: Long-term memory (user preferences)")


async def interactive_mode(agent: CompiledStateGraph, store: InMemoryStore, config: dict):
    """Interactive conversation mode using build_initial_state pattern."""
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    print("Type your questions. Type 'quit' to exit, 'history' to see conversation.")
    print(f"Session: {config['configurable']['thread_id']}")

    user_name = "Interactive User"

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue

        if question.lower() == "quit":
            break

        if question.lower() == "history":
            try:
                state = await agent.aget_state(config)
                messages = state.values.get("messages", [])
                sys_count = sum(
                    1 for m in messages if isinstance(m, SystemMessage))
                print(
                    f"\nConversation history ({len(messages)} messages, {sys_count} SystemMessage):")
                for msg in messages[-10:]:
                    if isinstance(msg, SystemMessage):
                        prefix = "System"
                    elif isinstance(msg, HumanMessage):
                        prefix = "You"
                    else:
                        prefix = "Agent"
                    print(f"  {prefix}: {msg.content[:80]}...")
            except Exception as e:
                print(f"  Error: {e}")
            continue

        # Use build_initial_state pattern from Chapter 2.8
        initial_state = await build_initial_state(
            agent, question, config, user_name=user_name
        )

        full_state = {
            "original_question": question,
            "schema_overview": get_schema_overview(),
            "few_shot_examples": "",
            "sql": "",
            "results": "",
            "evaluation_score": 0.0,
            "response": "",
            **initial_state,
        }

        result = await agent.ainvoke(full_state, config)

        if result.get("response"):
            print(f"Agent: {result['response']}")
        if result.get("sql"):
            print(f"  SQL: {result['sql'][:100]}...")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run 2.8 Agent Memory demos")
    parser.add_argument("--multi-turn", action="store_true",
                        help="Demo multi-turn conversation")
    parser.add_argument("--checkpoints", action="store_true",
                        help="Demo checkpoint inspection")
    parser.add_argument("--sliding-window",
                        action="store_true", help="Demo sliding window")
    parser.add_argument("--store", action="store_true",
                        help="Demo store long-term memory")
    parser.add_argument("--interactive", "-i",
                        action="store_true", help="Interactive mode")
    parser.add_argument("--thread-id", default=None, help="Custom thread ID")
    args = parser.parse_args()

    print("=" * 60)
    print("2.8 Agent Memory: Session Memory, Checkpoints, and Store")
    print("=" * 60)

    if args.sliding_window:
        demo_sliding_window()
        return

    config = load_config()
    provider = get_provider_name()
    model_name = get_model_name()

    print(f"\nUsing model: {provider}:{model_name}")

    if provider == "openai":
        llm = ChatOpenAI(model=model_name, temperature=0)
    elif provider == "gemini":
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    db_uri = config.database.url
    try:
        db = SQLDatabase.from_uri(db_uri)
        print("Connected to database")
    except Exception as e:
        print(f"\nError: Could not connect to database: {e}")
        print("Run: make db-up")
        return

    # Chapter 2.8: Create store for long-term memory
    store = InMemoryStore()

    # Build agent with both checkpointer and store
    agent, checkpointer, store = build_agent_with_memory(db, llm, store=store)
    print("Agent ready with InMemorySaver checkpointer and InMemoryStore")

    thread_id = args.thread_id or f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    session_config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50,
    }

    if args.multi_turn:
        await demo_session_memory(agent, session_config)
    elif args.checkpoints:
        await demo_checkpoint_inspection(agent, checkpointer, session_config)
    elif args.store:
        await demo_store_long_term_memory(agent, store, session_config)
    elif args.interactive:
        await interactive_mode(agent, store, session_config)
    else:
        # Run all demos
        await demo_session_memory(agent, session_config)
        await demo_checkpoint_inspection(agent, checkpointer, session_config)
        await demo_store_long_term_memory(agent, store, session_config)
        demo_sliding_window()

    print("\n" + "=" * 60)
    print("Memory demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
