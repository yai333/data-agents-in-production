"""Hybrid schema retrieval tools for the Text-to-SQL agent.

Replaces create_schema_tools() from 2.7 for production use.
Two tools use semantic search; four use exact lookup.
"""

from src.schema.hybrid_store import HybridSchemaStore


def create_hybrid_schema_tools(hybrid_store: HybridSchemaStore) -> list[dict]:
    """Create tool definitions using hybrid (semantic + exact) retrieval.

    Replaces create_schema_tools() from Chapter 2.7.
    Two semantic search tools (search_tables, search_business_context),
    four exact lookup tools (get_table_details, get_metrics,
    get_relationships, get_glossary_entries).
    """

    return [
        # -- Semantic search --
        {
            "name": "search_tables",
            "description": (
                "Find tables relevant to the user's question using semantic search. "
                "Returns the most relevant tables with schema_name, name, "
                "description, and score. "
                "Use FIRST to discover which tables might answer the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question or topic to search for",
                    }
                },
                "required": ["question"],
            },
            "handler": lambda question: hybrid_store.search_tables(question),
        },
        {
            "name": "search_business_context",
            "description": (
                "Find business context relevant to the question using hybrid search. "
                "Searches institutional knowledge and business descriptions. "
                "Use for fiscal year, time conventions, data quality notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The business context to search for",
                    }
                },
                "required": ["question"],
            },
            "handler": lambda question: hybrid_store.search_business_context(
                question
            ),
        },
        # -- Exact lookup (queries mdl_tables DB table) --
        {
            "name": "get_table_details",
            "description": (
                "Get table description and columns. "
                "Requires both schema_name and table_name "
                "(from search_tables results). "
                "Use AFTER search_tables identifies relevant tables."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "schema_name": {
                        "type": "string",
                        "description": "Schema name from search_tables results",
                    },
                    "table_name": {
                        "type": "string",
                        "description": "Table name from search_tables results",
                    },
                },
                "required": ["schema_name", "table_name"],
            },
            "handler": lambda schema_name, table_name: (
                hybrid_store.get_table_details(schema_name, table_name)
            ),
        },
        {
            "name": "get_metrics",
            "description": (
                "Get metrics (aggregations, KPIs) for a table. "
                "Requires schema_name and table_name. "
                "Use when the question involves aggregations or calculated fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "schema_name": {
                        "type": "string",
                        "description": "Schema name from search_tables results",
                    },
                    "table_name": {
                        "type": "string",
                        "description": "Table name from search_tables results",
                    },
                },
                "required": ["schema_name", "table_name"],
            },
            "handler": lambda schema_name, table_name: (
                hybrid_store.get_metrics(schema_name, table_name)
            ),
        },
        {
            "name": "get_relationships",
            "description": (
                "Get relationships (foreign keys, joins) for a table. "
                "Requires schema_name and table_name. "
                "Use when the question requires joining multiple tables."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "schema_name": {
                        "type": "string",
                        "description": "Schema name from search_tables results",
                    },
                    "table_name": {
                        "type": "string",
                        "description": "Table name from search_tables results",
                    },
                },
                "required": ["schema_name", "table_name"],
            },
            "handler": lambda schema_name, table_name: (
                hybrid_store.get_relationships(schema_name, table_name)
            ),
        },
        {
            "name": "get_glossary_entries",
            "description": (
                "Search business glossary for term definitions. "
                "Use for domain jargon like 'churn', 'active', 'revenue'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "terms": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["terms"],
            },
            "handler": lambda terms: hybrid_store.get_glossary_entries(terms),
        },
    ]
