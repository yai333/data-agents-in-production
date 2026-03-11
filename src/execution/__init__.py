"""Safe SQL execution with guardrails.

This module provides execution with timeouts, row limits, and read-only
transaction support.

See 1.6 for execution guardrails.
"""

from src.execution.runner import execute_sql, ExecutionResult

__all__ = ["execute_sql", "ExecutionResult"]
