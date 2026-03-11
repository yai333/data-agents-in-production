#!/usr/bin/env python3
"""2.6 The Agent Loop: Tool Calling in LangGraph.

This script demonstrates the ReAct pattern from Chapter 2.6:
- Messages-based state with add_messages reducer
- Tool-calling agent with LangGraph
- LangChain's native SQL tools (QuerySQLDatabaseTool)
- Implicit repair loop (agent sees errors and retries)

The agent uses a simple loop: agent → tools → agent → ... → end

Usage:
    # Make sure database is running
    make db-up

    # Run agent on sample questions
    python scripts/run_chapter_2_6.py

    # Run with specific question
    python scripts/run_chapter_2_6.py --question "How many artists?"

    # Show detailed trace
    python scripts/run_chapter_2_6.py --verbose
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Annotated, Sequence

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.utils.config import load_config


# =============================================================================
# State (Chapter 2.6: Messages-based state)
# =============================================================================

class AgentState(TypedDict):
    """State for the Text-to-SQL agent.

    Uses messages-based state with the add_messages reducer.
    Each node appends messages; the reducer handles accumulation.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


# =============================================================================
# System Prompt (Chapter 2.6)
# =============================================================================

SYSTEM_PROMPT = """You are a Text-to-SQL agent that helps users query a database.

TOOLS
Schema discovery:
- list_tables: Get all tables with short descriptions
- get_table_details: Get columns, types, and relationships for a table

SQL execution:
- sql_db_query: Execute SQL and return results (or errors)

RULES (must follow)
- Generate READ-ONLY SQL only: SELECT / WITH. Never write or modify data.
- Use a single statement only. No multiple statements.
- Prefer explicit column lists; avoid SELECT *.
- Control result size: add LIMIT 100 unless the user explicitly requests full output.

WORKFLOW (not fixed)
- Use list_tables when you need to see what tables are available.
- Use get_table_details to confirm columns/keys for chosen tables.
- Draft SQL, then run sql_db_query.

ERROR HANDLING
- Read tool errors carefully and fix SQL accordingly.

RESPONSE STYLE
- Provide: (1) the SQL you executed, (2) a plain-language explanation, (3) a concise summary of results.
"""


# =============================================================================
# Tools (Chapter 2.6)
# =============================================================================

# Simple in-memory schema for demo (in production, use SchemaStore from Chapter 2.2)
# Note: Column names match actual Chinook DB (snake_case)
CHINOOK_SCHEMA = {
    "artist": {
        "columns": ["artist_id", "name"],
        "description": "Music artists"
    },
    "album": {
        "columns": ["album_id", "title", "artist_id"],
        "description": "Albums by artists"
    },
    "track": {
        "columns": ["track_id", "name", "album_id", "media_type_id", "genre_id", "composer", "milliseconds", "bytes", "unit_price"],
        "description": "Individual tracks/songs"
    },
    "genre": {
        "columns": ["genre_id", "name"],
        "description": "Music genres"
    },
    "customer": {
        "columns": ["customer_id", "first_name", "last_name", "company", "address", "city", "state", "country", "postal_code", "phone", "fax", "email", "support_rep_id"],
        "description": "Customers who purchased music"
    },
    "invoice": {
        "columns": ["invoice_id", "customer_id", "invoice_date", "billing_address", "billing_city", "billing_state", "billing_country", "billing_postal_code", "total"],
        "description": "Sales invoices"
    },
    "invoice_line": {
        "columns": ["invoice_line_id", "invoice_id", "track_id", "unit_price", "quantity"],
        "description": "Line items on invoices"
    },
}


@tool
def list_tables() -> str:
    """Get all tables with short descriptions.

    Use when: Starting a new question to see what tables are available.

    Returns:
        List of all tables with descriptions
    """
    tables = [f"- {name}: {info['description']}" for name, info in CHINOOK_SCHEMA.items()]
    return "Available tables:\n" + "\n".join(tables)


@tool
def get_table_details(table_name: str) -> str:
    """Get detailed schema for a specific table.

    Use when: You need column names, types, and relationships for a table
    you found via search_tables.

    Args:
        table_name: Exact name of the table (case-insensitive)

    Returns:
        Table schema with columns
    """
    table_name_lower = table_name.lower()

    if table_name_lower not in CHINOOK_SCHEMA:
        return f"Error: Table '{table_name}' not found. Use search_tables to find available tables."

    info = CHINOOK_SCHEMA[table_name_lower]
    columns = ", ".join(info["columns"])
    return f"Table: {table_name_lower}\nDescription: {info['description']}\nColumns: {columns}"


def create_tools(db: SQLDatabase):
    """Create tools for the agent."""
    # Native SQL query tool from LangChain
    sql_query_tool = QuerySQLDatabaseTool(db=db)  # tool.name = "sql_db_query"

    return [sql_query_tool, list_tables, get_table_details]


# =============================================================================
# Agent Graph (Chapter 2.6: ReAct Pattern)
# =============================================================================

def build_agent(tools, model):
    """Build the Text-to-SQL agent graph.

    Graph structure:
        START → agent → (tool calls?) → tools → agent → ... → END
    """
    # Bind tools to model
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """Call the LLM with the current messages."""
        messages = state["messages"]

        # Add system prompt if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """Determine whether to continue to tools or end."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # Build graph
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )

    graph.add_edge("tools", "agent")

    return graph.compile()


# =============================================================================
# Runner
# =============================================================================

async def run_agent(question: str, agent, verbose: bool = False) -> str:
    """Run the agent on a question."""
    initial_state = {
        "messages": [HumanMessage(content=question)]
    }

    config = {"recursion_limit": 15}  # Max iterations

    if verbose:
        print("\n[Trace]")
        async for event in agent.astream(initial_state, config=config):
            for node_name, node_state in event.items():
                print(f"  → {node_name}")
                if "messages" in node_state:
                    last_msg = node_state["messages"][-1] if node_state["messages"] else None
                    if last_msg:
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                print(f"    Tool call: {tc['name']}({tc['args']})")
                        elif hasattr(last_msg, "content") and last_msg.content:
                            preview = last_msg.content[:100].replace("\n", " ")
                            print(f"    Content: {preview}...")

        # Get final state
        result = await agent.ainvoke(initial_state, config=config)
    else:
        result = await agent.ainvoke(initial_state, config=config)

    # Get the last assistant message
    last_message = result["messages"][-1]
    return last_message.content


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run 2.6 Agent Loop (ReAct pattern)")
    parser.add_argument("--question", "-q", help="Specific question to ask")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed trace")
    args = parser.parse_args()

    print("=" * 60)
    print("2.6 The Agent Loop: Tool Calling in LangGraph")
    print("=" * 60)

    # Load config and create LangChain model
    config = load_config()
    provider = config.llm.provider

    if provider == "openai":
        model = ChatOpenAI(
            model=config.llm.model,
            temperature=0,
        )
    elif provider == "gemini":
        model = ChatGoogleGenerativeAI(
            model=config.llm.model,
            temperature=0,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    print(f"\nUsing provider: {provider}, model: {config.llm.model}")

    # Connect to database
    db_uri = config.database.url
    try:
        db = SQLDatabase.from_uri(db_uri)
        print(f"Connected to database: {db_uri.split('@')[-1] if '@' in db_uri else db_uri}")
    except Exception as e:
        print(f"\nError: Could not connect to database: {e}")
        print("Run: make db-up")
        return

    # Create tools and agent
    tools = create_tools(db)
    agent = build_agent(tools, model)

    print(f"Agent ready with {len(tools)} tools: {[t.name for t in tools]}")

    # Run questions
    if args.question:
        questions = [args.question]
    else:
        questions = [
            "How many artists are in the database?",
            "What are the top 5 genres by number of tracks?",
            "Show total revenue by country, top 5",
        ]

    for question in questions:
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        print("=" * 60)

        try:
            answer = await run_agent(question, agent, args.verbose)
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    asyncio.run(main())
