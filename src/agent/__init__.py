"""SQL agent with validate-execute-repair loop.

This module provides the main agent implementation using LangGraph.
The agent generates SQL, validates it, executes with guardrails,
and repairs on failure.

See 1.6 for the agent loop architecture.
"""

from src.agent.error_classifier import ErrorType, ClassifiedError, classify_error
from src.agent.repair import repair_sql
from src.agent.retry import RetryConfig, should_retry, get_retry_delay_ms
from src.agent.graph import (
    AgentState,
    build_agent_graph,
    run_agent,
    generate_node,
    validate_node,
    execute_node,
    repair_node,
    format_node,
    chart_node,
)

__all__ = [
    # Main entry points
    "run_agent",
    "build_agent_graph",
    "AgentState",
    # Node functions (for testing/customization)
    "generate_node",
    "validate_node",
    "execute_node",
    "repair_node",
    "format_node",
    "chart_node",
    # Error classification
    "ErrorType",
    "ClassifiedError",
    "classify_error",
    # Repair
    "repair_sql",
    # Retry
    "RetryConfig",
    "should_retry",
    "get_retry_delay_ms",
]
