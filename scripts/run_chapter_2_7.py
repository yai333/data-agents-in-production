#!/usr/bin/env python3
"""2.7 Handling Ambiguity: Disambiguation Tools and Pipeline.

This script demonstrates the disambiguation patterns from Chapter 2.7:
- Five disambiguation tools backed by Schema Cards
- Parallel pre-retrieval (schema overview + few-shot examples)
- Three-agent pipeline: Disambiguation → Text-to-SQL → Evaluation
- Clarification flow when confidence is low

The chapter builds on 2.6's ReAct agent by adding disambiguation before SQL generation.

Usage:
    # Make sure database is running
    make db-up

    # Run disambiguation pipeline on sample questions
    python scripts/run_chapter_2_7.py

    # Test specific question
    python scripts/run_chapter_2_7.py --question "Show top customers"

    # Show detailed trace
    python scripts/run_chapter_2_7.py --verbose
"""

from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool, QuerySQLCheckerTool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from src.utils.config import load_config
from src.adapters import get_model_name, get_provider_name
from dotenv import load_dotenv
import argparse
import asyncio
import sys
from pathlib import Path
from typing import Annotated, Sequence, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")


# =============================================================================
# Pydantic Models for Structured Output (Chapter 2.7)
# =============================================================================

class DisambiguationResult(BaseModel):
    """Output from the disambiguation agent."""
    disambiguated_question: str | None = Field(
        default=None,
        description="The rewritten, specific question with details"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made, e.g., 'interpreted top as top 10 by revenue'"
    )
    unclear_terms: list[str] = Field(
        default_factory=list,
        description="Terms that couldn't be resolved from context"
    )
    clarification_needed: str | None = Field(
        default=None,
        description="Question to ask user if confidence is low"
    )


class TextToSQLResult(BaseModel):
    """Output from the Text-to-SQL agent."""
    sql: str = Field(description="The generated SQL query")
    explanation: str = Field(
        description="Brief explanation of the query logic")
    tables_used: list[str] = Field(
        description="Tables referenced in the query")


class EvaluationResult(BaseModel):
    """Output from the evaluation agent."""
    score: float = Field(ge=0.0, le=1.0, description="Quality score")
    passed: bool = Field(description="Whether the result is acceptable")
    issues: list[str] = Field(
        default_factory=list,
        description="Any issues found during evaluation"
    )
    response: str = Field(description="Final response to show the user")


# =============================================================================
# Schema and Semantic Layer (from Chapter 2.2)
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

# Business glossary (from semantic layer)
GLOSSARY = {
    "top customers": "Customers ranked by total invoice amount (SUM of invoice.total)",
    "active customer": "Customer with at least one invoice in the last 12 months",
    "churned customer": "Customer with no invoice in the last 12 months",
    "revenue": "SUM(invoice.total) - represents total sales",
    "bestselling": "Ranked by quantity sold (SUM of invoice_line.quantity)",
    "popular": "Ranked by number of purchases or plays",
}

# Metrics definitions (organized by table)
METRICS = {
    "invoice": {
        "revenue": {
            "definition": "Total sales amount",
            "sql_pattern": "SUM(invoice.total)",
        },
        "average_order_value": {
            "definition": "Average invoice total",
            "sql_pattern": "AVG(invoice.total)",
        },
    },
    "customer": {
        "customer_count": {
            "definition": "Number of unique customers",
            "sql_pattern": "COUNT(DISTINCT customer_id)",
        },
    },
    "invoice_line": {
        "quantity_sold": {
            "definition": "Total quantity sold",
            "sql_pattern": "SUM(invoice_line.quantity)",
        },
    },
    "track": {
        "track_count": {
            "definition": "Number of tracks",
            "sql_pattern": "COUNT(track_id)",
        },
    },
}

# Business info (conventions, defaults)
BUSINESS_INFO = {
    "fiscal_year": "Calendar year (Jan 1 - Dec 31)",
    "default_limit": "10 for 'top N' queries unless specified",
    "default_time_range": "Last 12 months for trend queries",
    "currency": "USD for all monetary values",
}

# Few-shot examples for SQL patterns
FEW_SHOT_EXAMPLES = [
    {
        "question": "How many artists are in the database?",
        "sql": "SELECT COUNT(*) as artist_count FROM artist"
    },
    {
        "question": "Show top 5 customers by total spending",
        "sql": """SELECT c.first_name, c.last_name, SUM(i.total) as total_spent
FROM customer c
JOIN invoice i ON c.customer_id = i.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY total_spent DESC
LIMIT 5"""
    },
    {
        "question": "What genres have the most tracks?",
        "sql": """SELECT g.name as genre, COUNT(t.track_id) as track_count
FROM genre g
JOIN track t ON g.genre_id = t.genre_id
GROUP BY g.genre_id, g.name
ORDER BY track_count DESC
LIMIT 10"""
    },
]


# =============================================================================
# Five Disambiguation Tools (Chapter 2.7)
# =============================================================================

@tool
def get_table_details(table_name: str) -> str:
    """Get detailed schema for a specific table including columns, relationships, and sample values.

    Use when: You need column names, types, relationships, or sample values for a table.

    Args:
        table_name: Name of the table (case-insensitive)

    Returns:
        Detailed table information
    """
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
    """Get foreign key relationships for a table to understand join paths.

    Use when: Question involves multiple entities and you need to determine how tables connect.

    Args:
        table_name: Name of the table (case-insensitive)

    Returns:
        Relationship information including join paths
    """
    table_name_lower = table_name.lower()

    if table_name_lower not in CHINOOK_SCHEMA:
        return f"Error: Table '{table_name}' not found."

    info = CHINOOK_SCHEMA[table_name_lower]

    if "relationships" not in info:
        return f"Table '{table_name_lower}' has no foreign key relationships defined."

    result = [f"Relationships for {table_name_lower}:"]
    for rel in info["relationships"]:
        result.append(f"  - {rel}")

    referencing = []
    for other_table, other_info in CHINOOK_SCHEMA.items():
        if "relationships" in other_info:
            for rel in other_info["relationships"]:
                if f"-> {table_name_lower}." in rel:
                    referencing.append(f"{other_table}: {rel}")

    if referencing:
        result.append(f"Referenced by:")
        for ref in referencing:
            result.append(f"  - {ref}")

    return "\n".join(result)


@tool
def get_metrics(table_name: str) -> str:
    """Get metric definitions (SQL patterns) for a table.

    Use when: Question mentions calculated values like revenue, count, average.
    Metrics are table-level - pass the table name to get available metrics.

    Args:
        table_name: Name of the table to get metrics for

    Returns:
        Available metrics with definitions and SQL patterns
    """
    table_name_lower = table_name.lower()

    if table_name_lower not in METRICS:
        available = ", ".join(METRICS.keys())
        return f"No metrics defined for table '{table_name}'. Tables with metrics: {available}"

    table_metrics = METRICS[table_name_lower]
    results = [f"Metrics for {table_name_lower}:"]
    for metric_name, m in table_metrics.items():
        results.append(f"  - {metric_name}: {m['definition']}")
        results.append(f"    SQL: {m['sql_pattern']}")

    return "\n".join(results)


@tool
def get_glossaries(terms: str) -> str:
    """Look up business term definitions from the glossary.

    Use when: Question contains domain-specific jargon like "churned", "active", "top customers".

    Args:
        terms: Comma-separated business terms to look up

    Returns:
        Definitions for the requested terms
    """
    term_list = [t.strip().lower() for t in terms.split(",")]
    results = []

    for term in term_list:
        if term in GLOSSARY:
            results.append(f"'{term}': {GLOSSARY[term]}")
            continue

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
    """Get business conventions and rules not captured in schema or glossary.

    Use when: Ambiguity relates to time periods, defaults, or organizational conventions.

    Args:
        topic: Topic to look up (e.g., "fiscal year", "default limit", "time range")

    Returns:
        Business convention or rule
    """
    topic_lower = topic.lower().replace(" ", "_")

    if topic_lower in BUSINESS_INFO:
        return f"{topic}: {BUSINESS_INFO[topic_lower]}"

    matches = [(k, v) for k, v in BUSINESS_INFO.items()
               if topic_lower in k or any(word in k for word in topic_lower.split("_"))]

    if matches:
        return "\n".join([f"{k.replace('_', ' ')}: {v}" for k, v in matches])

    available = ", ".join(k.replace("_", " ") for k in BUSINESS_INFO.keys())
    return f"No information found for '{topic}'. Available topics: {available}"


# =============================================================================
# Parallel Pre-Retrieval (Chapter 2.7)
# =============================================================================

async def get_schema_overview() -> str:
    """Retrieve schema overview for all tables (parallel task 1)."""
    tables = []
    for name, info in CHINOOK_SCHEMA.items():
        tables.append(f"- {name}: {info['description']}")
    return "Available tables:\n" + "\n".join(tables)


async def get_few_shot_examples(question: str) -> str:
    """Retrieve relevant few-shot examples (parallel task 2).

    In production, this would use semantic search. For demo, return all examples.
    """
    examples = []
    for ex in FEW_SHOT_EXAMPLES:
        examples.append(f"Q: {ex['question']}\nSQL: {ex['sql']}")
    return "Example queries:\n\n" + "\n\n".join(examples)


async def parallel_pre_retrieval(question: str) -> tuple[str, str]:
    """Run schema and few-shot retrieval in parallel."""
    schema_task = get_schema_overview()
    examples_task = get_few_shot_examples(question)

    schema_overview, few_shot_examples = await asyncio.gather(schema_task, examples_task)
    return schema_overview, few_shot_examples


# =============================================================================
# Pipeline State (Chapter 2.7)
# =============================================================================

class PipelineState(TypedDict):
    """State for the three-agent pipeline."""
    original_question: str
    schema_overview: str
    few_shot_examples: str
    disambiguated_question: str
    confidence: float
    unclear_terms: list[str]  # Terms Agent 1 couldn't resolve
    sql: str
    results: str  # Query results from execution
    evaluation_score: float
    response: str  # Final synthesized response
    messages: Annotated[Sequence[BaseMessage],
                        add_messages]  # SQL generation only


# =============================================================================
# Three-Agent Pipeline (Chapter 2.7)
# =============================================================================

def create_disambiguation_tools():
    """Create the five disambiguation tools."""
    return [get_table_details, get_relationships, get_metrics, get_glossaries, get_additional_business_info]


def create_sql_checker_tool(db: SQLDatabase, llm):
    """Create SQL validation tool for SQL generation agent."""
    return QuerySQLCheckerTool(db=db, llm=llm)


def create_sql_query_tool(db: SQLDatabase):
    """Create SQL execution tool for evaluation agent."""
    return QuerySQLDatabaseTool(db=db)


DISAMBIGUATION_PROMPT = """You clarify ambiguous questions about a database.

<SCHEMA>
{schema_overview}
</SCHEMA>

Use this context to:
1. Identify ambiguous terms (jargon, pronouns, unclear scope)
2. Rewrite the question to be self-contained and specific
3. Add details that will help generate accurate SQL
4. Flag terms you cannot resolve from the context provided
5. Assess your confidence (0.0-1.0)

IMPORTANT:
- Do NOT guess definitions you're unsure about
- If a term has no clear meaning from the schema, add it to unclear_terms
- If confidence < 0.7, provide a clarification_needed question to ask the user
  Example: "Did you mean top 5, 10, or 20 customers?" or "What time period defines 'active'?"
"""


TEXT_TO_SQL_PROMPT = """You are an expert SQL generation agent.

<SCHEMA>
{schema_overview}
</SCHEMA>

<EXAMPLES>
{few_shot_examples}
</EXAMPLES>

<WORKFLOW>
1. DISCOVERY: Use tools to understand the schema
   - get_table_details: Get column names and types for specific tables
   - get_relationships: Understand join paths between tables
   - get_metrics: Get metric definitions and SQL patterns
   - get_glossaries: Understand business terms
   - get_additional_business_info: Get conventions and defaults

2. SQL GENERATION: Draft the query
   - Use exact table and column names from discovery
   - Build clear SQL with correct joins and filters
   - Add LIMIT 100 for detail queries

3. VALIDATION: Validate before outputting
   - Use sql_db_query_checker to validate syntax

4. FINAL OUTPUT: Call TextToSQLResult with your SQL
   - sql: The final SQL query
   - explanation: Brief explanation of the query logic
   - tables_used: List of tables in the query
</WORKFLOW>

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
   - Execute ONCE only - do not loop back for more queries

2. ANALYZE: Review the query results
   - What data was returned?
   - Does it answer the user's question?

3. SYNTHESIZE: Create a natural language response
   - Summarize the key findings
   - Present data in a clear format

4. SCORE: Evaluate how well the SQL answered the question
   - 1.0: SQL completely answers the question
   - 0.5: Empty results but query was correct
   - 0.0: Error occurred or query failed

5. FINAL OUTPUT: Call EvaluationResult with your response
   - response: Natural language answer for the user
   - score: Quality score (0.0-1.0)
   - passed: Whether the result is acceptable
   - issues: Any problems found
</WORKFLOW>

<CRITICAL>
- Execute sql_db_query ONCE only
- After receiving results, immediately call EvaluationResult
- Do NOT loop back for additional queries
</CRITICAL>
"""


def build_pipeline(disambiguation_tools, sql_checker_tool, sql_query_tool, llm):
    """Build the three-agent disambiguation pipeline using with_structured_output pattern.

    Uses the canonical LangGraph pattern:
    - model.bind_tools() for tool calling (ReAct loop)
    - ToolNode for tool execution
    - model.with_structured_output() for final structured extraction

    Architecture:
    - SQL Generation: 5 MDL tools + sql_query_checker (ReAct loop)
    - Structured extraction: with_structured_output for final SQL/response

    Pipeline:
        disambiguate → generate_sql ⟺ sql_tools → extract_sql → evaluate ⟺ eval_tools → finalize
    """
    # SQL Generation: disambiguation tools + validation
    sql_gen_tools = disambiguation_tools + [sql_checker_tool]

    # Evaluation: execution only
    eval_tools = [sql_query_tool]

    # Disambiguation uses structured output (single-turn, no ReAct loop)
    disambiguation_model = llm.with_structured_output(DisambiguationResult)

    sql_model_with_tools = llm.bind_tools(sql_gen_tools + [TextToSQLResult])
    eval_model_with_tools = llm.bind_tools(eval_tools + [EvaluationResult])

    def disambiguation_node(state: PipelineState) -> dict:
        prompt = DISAMBIGUATION_PROMPT.format(
            schema_overview=state["schema_overview"]
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state["original_question"])
        ]

        response = disambiguation_model.invoke(messages)

        if response is None:
            response = DisambiguationResult(
                disambiguated_question=state["original_question"],
                confidence=0.5
            )

        if response.confidence < 0.7 and response.clarification_needed:
            return {
                "disambiguated_question": response.disambiguated_question or state["original_question"],
                "confidence": response.confidence,
                "unclear_terms": response.unclear_terms or [],
                "response": response.clarification_needed,
            }

        question = response.disambiguated_question or state["original_question"]
        return {
            "disambiguated_question": question,
            "confidence": response.confidence,
            "unclear_terms": response.unclear_terms or [],
            "response": "",
            "messages": [HumanMessage(content=question)]
        }

    def sql_generation_node(state: PipelineState) -> dict:
        prompt = TEXT_TO_SQL_PROMPT.format(
            schema_overview=state["schema_overview"],
            few_shot_examples=state["few_shot_examples"]
        )
        return {
            "messages": [
                sql_model_with_tools.invoke(
                    [SystemMessage(content=prompt)] + state["messages"]
                )
            ]
        }

    def should_continue_sql(state: PipelineState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "respond_sql"

        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "respond_sql"

        tool_names = [tc.get("name", "unknown")
                      for tc in last_message.tool_calls]
        print(f"    [SQL Gen] Tool calls: {tool_names}")

        if (
            len(last_message.tool_calls) == 1
            and last_message.tool_calls[0].get("name") == "TextToSQLResult"
        ):
            return "respond_sql"

        return "sql_tools"

    def respond_sql_node(state: PipelineState) -> dict:
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
            content=f"Execute this SQL and provide the final answer:\n\nSQL: {sql}")

        new_messages = []
        if tool_message:
            new_messages.append(tool_message)
        new_messages.append(eval_instruction)

        return {"sql": sql, "messages": new_messages}

    def evaluation_node(state: PipelineState) -> dict:
        return {
            "messages": [
                eval_model_with_tools.invoke(
                    [SystemMessage(content=EVALUATION_PROMPT)] +
                    state["messages"]
                )
            ]
        }

    def should_continue_eval(state: PipelineState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "respond_eval"

        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "respond_eval"

        tool_names = [tc.get("name", "unknown")
                      for tc in last_message.tool_calls]
        print(f"    [Eval] Tool calls: {tool_names}")

        if (
            len(last_message.tool_calls) == 1
            and last_message.tool_calls[0].get("name") == "EvaluationResult"
        ):
            return "respond_eval"

        return "eval_tools"

    def respond_eval_node(state: PipelineState) -> dict:
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

    def route_after_disambiguation(state: PipelineState) -> str:
        confidence = state.get("confidence", 0.5)
        unclear_terms = state.get("unclear_terms", [])
        if confidence < 0.7 and unclear_terms:
            return "clarify"
        return "generate_sql"

    graph = StateGraph(PipelineState)

    graph.add_node("disambiguate", disambiguation_node)
    graph.add_node("generate_sql", sql_generation_node)
    graph.add_node("sql_tools", ToolNode(sql_gen_tools))
    graph.add_node("respond_sql", respond_sql_node)
    graph.add_node("evaluate", evaluation_node)
    graph.add_node("eval_tools", ToolNode(eval_tools))
    graph.add_node("respond_eval", respond_eval_node)

    graph.set_entry_point("disambiguate")

    graph.add_conditional_edges(
        "disambiguate",
        route_after_disambiguation,
        {"generate_sql": "generate_sql", "clarify": END}
    )

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

    return graph.compile()


# =============================================================================
# Demo Functions
# =============================================================================

def demo_disambiguation_tools():
    """Demonstrate the five disambiguation tools."""
    print("\n" + "=" * 60)
    print("Disambiguation Tools Demo")
    print("=" * 60)

    print("\n1. get_table_details('customer'):")
    print(get_table_details.invoke({"table_name": "customer"}))

    print("\n2. get_relationships('invoice'):")
    print(get_relationships.invoke({"table_name": "invoice"}))

    print("\n3. get_metrics('revenue'):")
    print(get_metrics.invoke({"metric_name": "revenue"}))

    print("\n4. get_glossaries('top customers, churned'):")
    print(get_glossaries.invoke({"terms": "top customers, churned"}))

    print("\n5. get_additional_business_info('default limit'):")
    print(get_additional_business_info.invoke({"topic": "default limit"}))


async def demo_parallel_retrieval():
    """Demonstrate parallel pre-retrieval."""
    print("\n" + "=" * 60)
    print("Parallel Pre-Retrieval Demo")
    print("=" * 60)

    question = "Show top customers last quarter"
    print(f"\nQuestion: {question}")

    import time
    start = time.time()
    schema, examples = await parallel_pre_retrieval(question)
    elapsed = time.time() - start

    print(f"\nRetrieved in {elapsed:.3f}s (parallel)")
    print(f"\nSchema overview ({len(schema)} chars):")
    print(schema[:200] + "...")
    print(f"\nFew-shot examples ({len(examples)} chars):")
    print(examples[:300] + "...")


async def run_pipeline(question: str, pipeline, verbose: bool = False) -> dict:
    """Run the three-agent pipeline on a question."""
    schema_overview, few_shot_examples = await parallel_pre_retrieval(question)

    initial_state = {
        "original_question": question,
        "schema_overview": schema_overview,
        "few_shot_examples": few_shot_examples,
        "disambiguated_question": "",
        "confidence": 0.0,
        "unclear_terms": [],
        "sql": "",
        "results": "",
        "evaluation_score": 0.0,
        "response": "",
        "messages": []
    }

    config = {"recursion_limit": 50}

    if verbose:
        print("\n[Pipeline Trace]")
        async for event in pipeline.astream(initial_state, config=config):
            for node_name, node_state in event.items():
                print(f"  → {node_name}")
                if "confidence" in node_state:
                    print(f"    Confidence: {node_state['confidence']}")

    result = await pipeline.ainvoke(initial_state, config=config)
    return result


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run 2.7 Handling Ambiguity")
    parser.add_argument("--question", "-q", help="Specific question to test")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed trace")
    parser.add_argument("--tools-only", action="store_true",
                        help="Only demo the tools")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Only demo pre-retrieval")
    args = parser.parse_args()

    print("=" * 60)
    print("2.7 Handling Ambiguity: Disambiguation Pipeline")
    print("=" * 60)

    if args.tools_only:
        demo_disambiguation_tools()
        return

    if args.retrieval_only:
        await demo_parallel_retrieval()
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
        print(f"Connected to database")
    except Exception as e:
        print(f"\nError: Could not connect to database: {e}")
        print("Run: make db-up")
        return

    disambiguation_tools = create_disambiguation_tools()
    sql_checker_tool = create_sql_checker_tool(db, llm)
    sql_query_tool = create_sql_query_tool(db)
    pipeline = build_pipeline(disambiguation_tools,
                              sql_checker_tool, sql_query_tool, llm)

    print(
        f"Pipeline ready with {len(disambiguation_tools)} disambiguation tools")

    if args.question:
        questions = [args.question]
    else:
        questions = [
            "Show top customers",
            "Revenue last quarter",
            "Active customers by country",
            "How many artists are in the database?",
            "List all genres",
        ]

    for question in questions:
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        print("=" * 60)

        try:
            result = await run_pipeline(question, pipeline, args.verbose)

            print(f"\nConfidence: {result.get('confidence', 'N/A')}")
            print(f"Evaluation: {result.get('evaluation_score', 'N/A')}")

            if result.get("response"):
                print(f"\nResponse:\n{result['response'][:500]}")
            elif result.get("sql"):
                print(f"\nSQL:\n{result['sql']}")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            if args.verbose:
                traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
