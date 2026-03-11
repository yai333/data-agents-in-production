"""Safe SQL execution with guardrails.

This module executes SQL queries with timeout, row limits, and read-only
transaction support. These guardrails protect against runaway queries
and accidental data modification.

See 1.6 for execution guardrails.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import asyncpg

from src.utils.config import load_config


@dataclass
class ExecutionResult:
    """Result of SQL execution."""

    success: bool
    rows: list[dict[str, Any]] | None
    row_count: int
    error: str | None
    truncated: bool = False
    execution_time_ms: float = 0.0


async def execute_sql(
    sql: str,
    timeout_seconds: int = 30,
    max_rows: int = 1000,
) -> ExecutionResult:
    """Execute SQL with guardrails.

    Guardrails:
    - Statement timeout (default 30s)
    - Row limit (default 1000)
    - Read-only transaction

    Args:
        sql: SQL query to execute
        timeout_seconds: Maximum execution time
        max_rows: Maximum rows to return

    Returns:
        ExecutionResult with rows or error

    Example:
        >>> result = await execute_sql("SELECT * FROM customer LIMIT 10")
        >>> result.success
        True
        >>> len(result.rows)
        10
    """
    import time

    settings = load_config()
    start_time = time.time()

    # Try to connect
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(settings.database.url),
            timeout=5.0,  # Connection timeout
        )
    except asyncio.TimeoutError:
        return ExecutionResult(
            success=False,
            rows=None,
            row_count=0,
            error="Database connection timeout",
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            rows=None,
            row_count=0,
            error=f"Connection error: {e}",
        )

    try:
        # Set statement timeout
        await conn.execute(f"SET statement_timeout = '{timeout_seconds}s'")

        # Start read-only transaction
        await conn.execute("BEGIN READ ONLY")

        try:
            # Add LIMIT if not present (safety net)
            sql_with_limit = _ensure_limit(sql, max_rows + 1)

            # Execute query
            rows = await asyncio.wait_for(
                conn.fetch(sql_with_limit),
                timeout=timeout_seconds,
            )

            execution_time = (time.time() - start_time) * 1000

            # Convert to dicts
            result_rows = [dict(row) for row in rows]

            # Check if truncated
            truncated = len(result_rows) > max_rows
            if truncated:
                result_rows = result_rows[:max_rows]

            await conn.execute("COMMIT")

            return ExecutionResult(
                success=True,
                rows=result_rows,
                row_count=len(result_rows),
                error=None,
                truncated=truncated,
                execution_time_ms=execution_time,
            )

        except asyncio.TimeoutError:
            await conn.execute("ROLLBACK")
            return ExecutionResult(
                success=False,
                rows=None,
                row_count=0,
                error=f"Query timeout ({timeout_seconds}s exceeded)",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    except asyncpg.PostgresError as e:
        return ExecutionResult(
            success=False,
            rows=None,
            row_count=0,
            error=f"Database error: {e}",
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    finally:
        await conn.close()


def _ensure_limit(sql: str, limit: int) -> str:
    """Add LIMIT clause if not present.

    Args:
        sql: SQL query
        limit: Maximum rows to return

    Returns:
        SQL with LIMIT clause
    """
    sql_upper = sql.upper().strip()

    # Don't modify if already has LIMIT
    if "LIMIT" in sql_upper:
        return sql

    # Don't modify aggregation-only queries (they return single row)
    aggregations = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]
    has_aggregation = any(agg in sql_upper for agg in aggregations)
    has_group_by = "GROUP BY" in sql_upper

    if has_aggregation and not has_group_by:
        return sql

    # Add LIMIT
    return f"{sql.rstrip(';')} LIMIT {limit}"
