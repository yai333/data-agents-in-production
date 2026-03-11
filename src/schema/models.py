"""Schema Card data models.

Schema Cards are structured representations of database schema optimized
for LLM consumption. They include semantic information (descriptions,
examples, business rules) that raw DDL lacks.

The core models:
- ColumnCard: Metadata for a single database column
- MetricDefinition: Business metric with SQL calculation pattern
- Relationship: Join path between tables (WrenAI/Looker MDL pattern)
- TableCard: Metadata for a database table including its columns
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ColumnCard(BaseModel):
    """Metadata for a single database column."""

    name: str                           # Exact column name from database
    data_type: str                      # SQL type: INTEGER, VARCHAR, etc.
    description: str = ""               # Human-readable, for LLM context
    nullable: bool = True               # Whether NULL values allowed
    is_primary_key: bool = False        # Part of primary key?
    is_foreign_key: bool = False        # References another table?
    references: str | None = None       # Foreign key target: "table.column"
    examples: list[str] = []            # Sample values to illustrate content
    business_rules: str | None = None   # Business logic not in schema


class MetricDefinition(BaseModel):
    """Business metric with SQL calculation pattern. Defined per table."""

    name: str                           # Metric name: "revenue", "churn_rate"
    description: str                    # What this metric measures
    sql_pattern: str                    # SQL calculation: "SUM(total)"
    default_aggregation: str | None = None  # Default grouping if unspecified
    default_limit: int | None = None    # Default LIMIT if "top" is unspecified


class Relationship(BaseModel):
    """Join path between tables. Follows WrenAI/Looker MDL pattern."""

    name: str                           # Identifier: "invoice_customer"
    models: list[str]                   # Tables involved: ["invoice", "customer"]
    join_type: str                      # Cardinality: "MANY_TO_ONE", "ONE_TO_MANY"
    condition: str                      # SQL join: "invoice.customer_id = customer.customer_id"

class TableCard(BaseModel):
    """Metadata for a database table."""

    name: str                           # Exact table name from database
    schema_name: str = ""               # Schema/project: "chinook", "analytics"
    description: str                    # What this table represents
    columns: list[ColumnCard] = []      # All columns with their metadata
    primary_key: list[str] = []         # Primary key column(s)
    relationships: list[Relationship] = []  # Join paths to other tables
    metrics: list[MetricDefinition] = []  # Table-level metrics (revenue, count, etc.)

    def get_column(self, name: str) -> ColumnCard | None:
        """Get a column by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def get_foreign_keys(self) -> list[ColumnCard]:
        """Get all foreign key columns."""
        return [col for col in self.columns if col.is_foreign_key]

    def get_referenced_tables(self) -> set[str]:
        """Get names of all tables this table references."""
        tables = set()
        for col in self.columns:
            if col.references:
                table_name = col.references.split(".")[0]
                tables.add(table_name)
        return tables
