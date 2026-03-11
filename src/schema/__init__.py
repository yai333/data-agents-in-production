"""Schema representation for Text-to-SQL agents.

Schema Cards provide a structured, semantic-rich representation of database
schema that is designed for LLM consumption. Unlike raw DDL, Schema Cards
include descriptions, examples, business rules, and relationships.
"""

from src.schema.models import TableCard, ColumnCard
from src.schema.render import render_table_card, render_column_card, render_schema
from src.schema.store import SchemaStore
from src.schema.tools import create_schema_tools
from src.schema.hybrid_store import (
    HybridSchemaStore,
    build_mdl_index,
    build_glossary_table,
    ingest_additional_descriptions,
    ingest_institutional_knowledge,
)
from src.schema.hybrid_tools import create_hybrid_schema_tools
from src.schema.cache import SQLCache, normalize_question, ensure_cache_table

__all__ = [
    "TableCard",
    "ColumnCard",
    "render_table_card",
    "render_column_card",
    "render_schema",
    "SchemaStore",
    "create_schema_tools",
    # 3.3: hybrid schema store and tools
    "HybridSchemaStore",
    "build_mdl_index",
    "build_glossary_table",
    "ingest_additional_descriptions",
    "ingest_institutional_knowledge",
    "create_hybrid_schema_tools",
    # 3.3: SQL caching
    "SQLCache",
    "normalize_question",
    "ensure_cache_table",
]
