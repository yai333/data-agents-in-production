"""LangGraph agent with validate-execute-repair loop.

This module provides the main SQL agent using LangGraph's StateGraph.
The agent follows the pattern: Generate → Validate → Execute → Format → Chart,
with retry loops for recovery.

See 1.6 for the architecture documentation.
"""

import logging
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import StateGraph, END

from src.adapters import create_adapter
from src.chart import preprocess_for_chart, generate_chart_spec, finalize_chart_spec
from src.schema import SchemaStore, render_schema
from src.structured import generate_sql_structured, SQLResult
from src.validators import validate_sql, FullValidationResult
from src.execution.runner import execute_sql, ExecutionResult
from src.agent.error_classifier import classify_error, ClassifiedError
from src.agent.repair import repair_sql
from src.agent.retry import should_retry, RetryConfig

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """State for the SQL agent.

    Attributes:
        question: The natural language question to answer
        schema_store: Schema information for the database
        sql_result: Generated SQL with reasoning
        validation: Validation result (schema, safety, etc.)
        execution: Query execution result
        error: Classified error if any step failed
        attempt: Current attempt number (for retry logic)
        final_answer: Formatted response to user
    """
    # Input
    question: str
    schema_store: SchemaStore

    # Generation
    sql_result: SQLResult | None

    # Validation
    validation: FullValidationResult | None

    # Execution
    execution: ExecutionResult | None

    # Repair
    error: ClassifiedError | None
    attempt: int

    # Output
    final_answer: str | None

    # Chart
    chart_spec: dict[str, Any] | None
    chart_reasoning: str | None


async def generate_node(state: AgentState) -> AgentState:
    """Generate SQL from the question.

    Uses structured generation to produce SQL with reasoning.
    """
    result = await generate_sql_structured(
        question=state["question"],
        schema_store=state["schema_store"],
    )

    return {
        **state,
        "sql_result": result,
        "attempt": state.get("attempt", 0) + 1,
    }


async def validate_node(state: AgentState) -> AgentState:
    """Validate the generated SQL.

    Checks schema validity, safety constraints, and join paths.
    """
    validation = validate_sql(
        sql=state["sql_result"].sql,
        schema_store=state["schema_store"],
    )

    return {**state, "validation": validation}


async def execute_node(state: AgentState) -> AgentState:
    """Execute the validated SQL.

    Runs with guardrails: timeout, row limit, read-only connection.
    """
    execution = await execute_sql(state["sql_result"].sql)

    if not execution.success:
        error = classify_error(execution.error)
        return {**state, "execution": execution, "error": error}

    return {**state, "execution": execution, "error": None}


async def repair_node(state: AgentState) -> AgentState:
    """Repair failed SQL with error context.

    Uses the error message and schema to generate a corrected query.
    """
    schema_text = render_schema(
        state["schema_store"].search_tables(state["question"])
    )

    repaired = await repair_sql(
        question=state["question"],
        failed_sql=state["sql_result"].sql,
        error=state["error"],
        schema_text=schema_text,
    )

    return {
        **state,
        "sql_result": repaired,
        "attempt": state["attempt"] + 1,
    }


async def format_node(state: AgentState) -> AgentState:
    """Format the final response.

    Creates a user-friendly summary of the query results.
    """
    execution = state["execution"]

    if execution.truncated:
        note = f" (showing first {len(execution.rows)} of more results)"
    else:
        note = ""

    answer = f"Query returned {execution.row_count} rows{note}."

    return {**state, "final_answer": answer}


async def chart_node(state: AgentState) -> AgentState:
    """Generate a Vega-Lite chart spec from execution results.

    Uses the LLM to select chart type and produce a spec
    based on column statistics and sample rows.
    Skips chart generation for single-row results or errors.
    """
    execution = state.get("execution")

    # Skip if no data or not enough rows to chart
    if not execution or not execution.success or execution.row_count <= 1:
        return {**state, "chart_spec": None, "chart_reasoning": None}

    # Skip wide tables (too many columns for meaningful chart)
    if len(execution.rows[0]) > 10:
        return {
            **state,
            "chart_spec": None,
            "chart_reasoning": "Wide table with many columns — table display is more appropriate",
        }

    try:
        df = pd.DataFrame(execution.rows)
        context = preprocess_for_chart(
            df=df,
            query=state["question"],
            sql=state["sql_result"].sql,
        )
        adapter = create_adapter()
        spec_result = await generate_chart_spec(context, adapter)
        final = finalize_chart_spec(spec_result, execution.rows)

        return {
            **state,
            "chart_spec": final,
            "chart_reasoning": spec_result.reasoning,
        }
    except Exception as e:
        logger.warning(f"Chart generation failed (non-blocking): {e}")
        return {**state, "chart_spec": None, "chart_reasoning": None}


async def classify_validation_error_node(state: AgentState) -> AgentState:
    """Classify a validation failure as an error for the repair loop."""
    error_msg = "; ".join(state["validation"].all_errors)
    return {**state, "error": classify_error(error_msg)}


async def give_up_node(state: AgentState) -> AgentState:
    """Handle case where we can't fix the query.

    Returns an error message after max retries exhausted.
    """
    return {
        **state,
        "final_answer": f"Unable to generate a valid query after {state['attempt']} attempts. "
                        f"Last error: {state['error'].message}",
    }


# Routing functions

def route_after_validation(state: AgentState) -> str:
    """Route based on validation result."""
    if state["validation"].can_execute:
        return "execute"
    else:
        return "classify_validation_error"


def route_after_execution(state: AgentState) -> str:
    """Route based on execution result."""
    if state["execution"].success:
        return "format"
    else:
        return "check_retry"


def route_retry(state: AgentState) -> str:
    """Decide whether to retry or give up."""
    config = RetryConfig()

    if should_retry(state["error"], state["attempt"], config):
        return "repair"
    else:
        return "give_up"


def build_agent_graph() -> StateGraph:
    """Build the SQL agent graph.

    Returns:
        Compiled LangGraph StateGraph for the SQL agent

    Graph structure:
        generate → validate → execute → format → chart → END
                     ↓           ↓
                  repair ← check_retry
                     ↓
                  give_up → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("repair", repair_node)
    graph.add_node("format", format_node)
    graph.add_node("chart", chart_node)
    graph.add_node("classify_validation_error", classify_validation_error_node)
    graph.add_node("give_up", give_up_node)

    # Add edges
    graph.set_entry_point("generate")
    graph.add_edge("generate", "validate")

    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "execute": "execute",
            "classify_validation_error": "classify_validation_error",
        }
    )

    graph.add_edge("classify_validation_error", "repair")

    graph.add_conditional_edges(
        "execute",
        route_after_execution,
        {
            "format": "format",
            "check_retry": "repair",
        }
    )

    graph.add_conditional_edges(
        "repair",
        lambda s: route_retry(s),
        {
            "repair": "validate",  # Try again after repair
            "give_up": "give_up",
        }
    )

    graph.add_edge("format", "chart")
    graph.add_edge("chart", END)
    graph.add_edge("give_up", END)

    return graph.compile()


async def run_agent(question: str, schema_store: SchemaStore) -> dict[str, Any]:
    """Run the SQL agent on a question.

    This is the main entry point for the agent.

    Args:
        question: Natural language question about the data
        schema_store: Schema information for the database

    Returns:
        Dict with answer, sql, chart_spec, chart_reasoning, row_count

    Example:
        >>> store = SchemaStore("data/chinook_schema.json")
        >>> result = await run_agent("How many customers?", store)
        >>> print(result["answer"])
        Query returned 59 rows.
    """
    graph = build_agent_graph()

    initial_state: AgentState = {
        "question": question,
        "schema_store": schema_store,
        "sql_result": None,
        "validation": None,
        "execution": None,
        "error": None,
        "attempt": 0,
        "final_answer": None,
        "chart_spec": None,
        "chart_reasoning": None,
    }

    final_state = await graph.ainvoke(initial_state)

    sql_result = final_state.get("sql_result")
    execution = final_state.get("execution")

    return {
        "answer": final_state.get("final_answer"),
        "sql": sql_result.sql if sql_result else None,
        "rows": execution.rows if execution and execution.success else None,
        "row_count": execution.row_count if execution else 0,
        "chart_spec": final_state.get("chart_spec"),
        "chart_reasoning": final_state.get("chart_reasoning"),
    }
