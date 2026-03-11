"""Langfuse tracing setup for the Text-to-SQL agent.

Provides initialization, callback handler for LangGraph, and
helper functions for trace metadata configuration.

Langfuse v3 auto-instruments LLM calls, tool invocations, and graph
runs via a CallbackHandler. Each request gets its own trace by wrapping
the agent call inside ``start_as_current_observation()`` with a unique
trace_id.

Usage:
    from src.observability.tracing import (
        init_langfuse, get_langfuse_callback, get_langfuse_client,
        flush_langfuse,
    )

    init_langfuse()  # reads from env vars
    handler, metadata, trace_id = get_langfuse_callback(
        user_id="user-123", session_id="sess-1",
    )

    client = get_langfuse_client()
    with client.start_as_current_observation(
        as_type="span", name="agent-request",
        trace_context={"trace_id": trace_id},
    ):
        result = await agent.ainvoke(state, config={
            "callbacks": [handler],
            "metadata": metadata,
        })
    # trace_id is known upfront — use it for scoring
"""

import os
import uuid
from typing import Any

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler


# Module-level client, initialized once
_langfuse_client: Langfuse | None = None


def init_langfuse(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
) -> Langfuse:
    """Initialize the Langfuse client.

    Reads from environment variables if arguments not provided:
      LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

    Returns:
        Configured Langfuse client
    """
    global _langfuse_client

    _langfuse_client = Langfuse(
        public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
        host=host or os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
    )

    return _langfuse_client


def get_langfuse_client() -> Langfuse:
    """Get the initialized Langfuse client.

    Initializes from env vars if not already done.
    """
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = init_langfuse()
    return _langfuse_client


def get_langfuse_callback(
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    trace_name: str = "text-to-sql-agent",
) -> tuple[CallbackHandler, dict[str, Any], str]:
    """Create a Langfuse callback handler, metadata dict, and trace ID.

    In Langfuse v3 (OTEL-based), multiple ``ainvoke()`` calls in the
    same process share a single trace context by default.  To produce
    separate traces, wrap each call in
    ``client.start_as_current_observation(trace_context=...)`` using the
    ``trace_id`` returned here.

    Args:
        user_id: Associate trace with a user (for per-user analytics)
        session_id: Group traces into a session (for multi-turn)
        tags: Labels for filtering in the Langfuse UI
        trace_name: Name shown in the Langfuse trace list

    Returns:
        Tuple of (CallbackHandler, metadata dict, trace_id string)
    """
    handler = CallbackHandler()
    trace_id = uuid.uuid4().hex

    metadata: dict[str, Any] = {}
    if user_id:
        metadata["langfuse_user_id"] = user_id
    if session_id:
        metadata["langfuse_session_id"] = session_id
    if tags:
        metadata["langfuse_tags"] = tags

    return handler, metadata, trace_id


def flush_langfuse() -> None:
    """Flush pending traces to Langfuse.

    Call this before program exit to ensure all traces are sent.
    The SDK batches traces for efficiency; flush forces immediate send.
    """
    client = get_langfuse_client()
    client.flush()
