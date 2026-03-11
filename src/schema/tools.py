# src/schema/tools.py
"""Schema retrieval tools for the Text-to-SQL agent."""

from src.schema.store import SchemaStore


def create_schema_tools(store: SchemaStore) -> list[dict]:
    """Create tool definitions for schema retrieval and disambiguation."""

    return [
        {
            "name": "list_tables",
            "description": "Get all table names with short descriptions. Use first to understand what tables exist.",
            "parameters": {"type": "object", "properties": {}},
            "handler": lambda: store.list_tables(),
        },
        {
            "name": "get_table_details",
            "description": "Get full schema for a table: columns, types, relationships, and context.",
            "parameters": {
                "type": "object",
                "properties": {"table_name": {"type": "string"}},
                "required": ["table_name"],
            },
            "handler": lambda table_name: store.get_table(table_name).model_dump() if store.get_table(table_name) else None,
        },
        {
            "name": "get_relationships",
            "description": "Get join paths from a table to related tables.",
            "parameters": {
                "type": "object",
                "properties": {"table_name": {"type": "string"}},
                "required": ["table_name"],
            },
            "handler": lambda table_name: [r.model_dump() for r in store.get_relationships(table_name)],
        },
        {
            "name": "get_glossaries",
            "description": "Search business glossary for term definitions. Use for domain jargon.",
            "parameters": {
                "type": "object",
                "properties": {"terms": {"type": "array", "items": {"type": "string"}}},
                "required": ["terms"],
            },
            "handler": lambda terms: store.get_glossary_entries(terms),
        },
        {
            "name": "get_metrics",
            "description": "Get metric definitions (SQL patterns) for a table. Metrics are table-level. Use when question mentions calculated values like revenue, count, average.",
            "parameters": {
                "type": "object",
                "properties": {"table_name": {"type": "string", "description": "Table to get metrics for"}},
                "required": ["table_name"],
            },
            "handler": lambda table_name: store.get_metrics(table_name),
        },
        {
            "name": "get_additional_descriptions",
            "description": "Get all business context: fiscal year, time conventions, default values. Use for 'Q3', 'recent', 'top'.",
            "parameters": {"type": "object", "properties": {}},
            "handler": lambda: store.get_additional_descriptions(),
        },
    ]
