"""ADK tools for the SQL Explorer agent.

Three LLM-callable tools:
1. get_database_schema — returns all tables with columns, types, row counts
2. execute_sql_query — runs SELECT queries with session-based pagination
3. get_sample_data — returns sample rows from a specified table
"""

import json
import logging
import os
import sqlite3

from google.adk.tools.tool_context import ToolContext

from sql_session_manager import DEFAULT_PAGE_SIZE, get_session_manager

logger = logging.getLogger(__name__)


def _get_db_path() -> str:
    """Get the path to the Chinook database."""
    # Check for DB_PATH env var first, then fall back to data/chinook.db
    db_path = os.getenv("CHINOOK_DB_PATH")
    if db_path and os.path.exists(db_path):
        return db_path
    # Default: project root data/chinook.db (scripts/chapter_3_7/ → project root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "data", "chinook.db")


def get_database_schema(tool_context: ToolContext) -> str:
    """Get the database schema including all tables and their columns.

    Call this tool first to understand what data is available.

    Returns:
        JSON string with table names and their columns.
    """
    logger.info("--- TOOL: get_database_schema ---")

    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = []
            for col in cursor.fetchall():
                columns.append({
                    "name": col[1],
                    "type": col[2],
                    "nullable": not col[3],
                    "primary_key": bool(col[5]),
                })

            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]

            schema[table] = {"columns": columns, "row_count": row_count}

        logger.info(f"  Found {len(tables)} tables")
        return json.dumps(schema, indent=2)

    finally:
        conn.close()


def execute_sql_query(
    sql_query: str,
    tool_context: ToolContext,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> str:
    """Execute a SQL query and return the first page of results.

    The query is stored for pagination — subsequent pages can be fetched
    without re-involving the LLM.

    Args:
        sql_query: The SQL SELECT query to execute (without LIMIT/OFFSET).
        page_size: Number of rows per page (default: 20).

    Returns:
        JSON string with query_id, columns, first page of rows, and pagination info.
    """
    logger.info(f"--- TOOL: execute_sql_query ---")
    logger.info(f"  Query: {sql_query}")

    # Only allow SELECT queries
    sql_lower = sql_query.strip().lower()
    if not sql_lower.startswith("select"):
        return json.dumps({"error": "Only SELECT queries are allowed."})

    dangerous_keywords = ["drop", "delete", "update", "insert", "alter", "create", "truncate"]
    for keyword in dangerous_keywords:
        if keyword in sql_lower:
            return json.dumps({"error": f"Query contains disallowed keyword: {keyword}"})

    try:
        db_path = _get_db_path()
        session_manager = get_session_manager(db_path)

        context_id = getattr(tool_context, "invocation_id", None) or \
                     getattr(tool_context, "session_id", None) or \
                     str(id(tool_context.state)) if hasattr(tool_context, "state") else "default"
        context_id = str(context_id)

        session = session_manager.create_session(sql_query, context_id, page_size)
        result = session_manager.fetch_page(context_id, session.query_id, page=1)

        logger.info(
            f"  {result['total_count']} total rows, "
            f"page 1 of {result['total_pages']}"
        )

        if "active_queries" not in tool_context.state:
            tool_context.state["active_queries"] = {}
        tool_context.state["active_queries"][session.query_id] = {
            "sql": sql_query,
            "created_at": session.created_at,
        }

        return json.dumps(result)

    except sqlite3.Error as e:
        logger.error(f"  SQL Error: {e}")
        return json.dumps({"error": f"SQL Error: {str(e)}"})
    except Exception as e:
        logger.error(f"  Error: {e}")
        return json.dumps({"error": str(e)})


def get_sample_data(table_name: str, tool_context: ToolContext, limit: int = 5) -> str:
    """Get sample rows from a specific table to understand its data.

    Args:
        table_name: The name of the table to sample.
        limit: Number of sample rows (default: 5).

    Returns:
        JSON string with sample rows.
    """
    logger.info(f"--- TOOL: get_sample_data({table_name}) ---")

    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        if not cursor.fetchone():
            return json.dumps({"error": f"Table not found: {table_name}"})

        cursor.execute(f"SELECT * FROM {table_name} LIMIT ?", (limit,))
        rows = [dict(row) for row in cursor.fetchall()]

        return json.dumps({"table": table_name, "sample_rows": rows, "count": len(rows)}, indent=2)

    finally:
        conn.close()
