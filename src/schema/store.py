"""Backend for schema retrieval tools."""

from pathlib import Path
import json
from typing import Optional

from src.schema.models import TableCard, Relationship


class SchemaStore:
    """In-memory store for schema cards and global context."""

    def __init__(self, schema_path: Path | str):
        """Load schema from JSON file at startup."""
        self.schema_path = Path(schema_path)
        self.tables: dict[str, TableCard] = {}
        # Global: term definitions
        self.glossary: dict[str, str] = {}
        # Global: business context
        self.additional_descriptions: list[str] = []
        self._load()

    @classmethod
    def from_directory(cls, mdl_dir: Path | str) -> "SchemaStore":
        """Load per-schema MDL files from a directory.

        Expected layout:
            mdl_dir/
                chinook.json       # one file per schema / dbt project
                analytics.json     # another schema
                ...

        Each file has the same shape as chinook_schema.json:
        {"tables": [...], "glossary": {...}}

        Business context (additional_descriptions) is not stored in
        per-schema MDL files — it lives in the embedding database,
        ingested from institutional knowledge documents.
        """
        mdl_dir = Path(mdl_dir)
        instance = cls.__new__(cls)
        instance.schema_path = mdl_dir
        instance.tables = {}
        instance.glossary = {}
        instance.additional_descriptions = []  # populated from embedding DB, not JSON

        # Load each schema file and merge
        for schema_file in sorted(mdl_dir.glob("*.json")):
            schema_name = schema_file.stem  # chinook.json → "chinook"
            with open(schema_file) as f:
                data = json.load(f)

            for table_data in data.get("tables", []):
                table_data.setdefault("schema_name", schema_name)
                table = TableCard(**table_data)
                instance.tables[table.name] = table

            for term, definition in data.get("glossary", {}).items():
                instance.glossary[term.lower()] = definition

        return instance

    def _load(self) -> None:
        """Parse JSON and build lookup dictionaries."""
        if not self.schema_path.exists():
            return
        with open(self.schema_path) as f:
            data = json.load(f)

        # Tables (relationships and metrics are inside each TableCard, not global)
        for table_data in data.get("tables", []):
            # Pydantic deserializes nested Relationship/Metric objects
            table = TableCard(**table_data)
            self.tables[table.name] = table

        # Global context
        for term, definition in data.get("glossary", {}).items():
            self.glossary[term.lower()] = definition

        self.additional_descriptions = data.get("additional_descriptions", [])

    def get_table(self, name: str) -> Optional[TableCard]:
        """Get full schema for one table. Used by get_table_details tool."""
        return self.tables.get(name)

    def list_tables(self) -> list[dict]:
        """Get all table names with short descriptions.

        For this book, we return all tables since Chinook has only 11.
        For large schemas (100+ tables), use semantic search with embeddings
        to find relevant tables based on the user's question.
        """
        return [{"name": t.name, "description": t.description}
                for t in self.tables.values()]

    def get_tables(self, names: list[str]) -> list[TableCard]:
        """Get multiple tables by name."""
        return [self.tables[n] for n in names if n in self.tables]

    def get_all_tables(self) -> list[TableCard]:
        """Get all tables as TableCard objects."""
        return list(self.tables.values())

    def search_tables(self, question: str) -> list[TableCard]:
        """Find tables relevant to a question.

        For Chinook (11 tables), returns all tables since the schema is small.
        For large schemas (100+ tables), this should use semantic search
        with embeddings to find relevant tables based on the question.

        Args:
            question: Natural language question

        Returns:
            List of relevant TableCard objects
        """
        # For small schemas like Chinook, return all tables
        # The LLM can filter to what's needed
        if len(self.tables) <= 20:
            return list(self.tables.values())

        # For larger schemas, do simple keyword matching
        # (In production, use embedding-based semantic search)
        return None

    def get_relationships(self, table_name: str) -> list[Relationship]:
        """Get join paths from a table. Used by get_relationships tool."""
        table = self.tables.get(table_name)
        return table.relationships if table else []

    def get_glossary_entries(self, terms: list[str]) -> list[dict]:
        """Get glossary entries by keywords. Used by get_glossaries tool.

        Deprecated: In 3.3+, glossary terms live in the DB glossary table
        and are accessed via HybridSchemaStore.get_glossary_entries().
        Kept for backward compatibility with 2.x scripts.
        """
        if not self.glossary or not terms:
            return []

        results = []
        for term in terms:
            term_lower = term.lower()
            for k, v in self.glossary.items():
                if term_lower in k or k in term_lower:
                    entry = {"term": k, "definition": v}
                    if entry not in results:  # Avoid duplicates
                        results.append(entry)
        return results

    def get_metrics(self, table_name: str | None = None) -> list[dict]:
        """Get metric definitions from tables.

        Metrics are table-level, so we return metrics for a specific table
        or all metrics across tables if no table specified.
        """
        if table_name:
            # Get metrics for specific table
            table = self.tables.get(table_name.lower())
            if table and table.metrics:
                return [{"table": table_name, **m.model_dump()} for m in table.metrics]
            return [{"error": f"No metrics for table '{table_name}'"}]

        # Return all metrics across all tables
        all_metrics = []
        for table in self.tables.values():
            for metric in table.metrics:
                all_metrics.append(
                    {"table": table.name, **metric.model_dump()})
        return all_metrics if all_metrics else [{"error": "No metrics defined"}]

    def get_additional_descriptions(self) -> list[str]:
        """Get all business context. Used by get_additional_descriptions tool.

        Deprecated: In 3.3+, additional descriptions are embedded into
        pgvector and searched via HybridSchemaStore.search_business_context().
        Kept for backward compatibility with 2.x scripts.
        """
        return self.additional_descriptions

    def __len__(self) -> int:
        """Return number of tables in the store."""
        return len(self.tables)
