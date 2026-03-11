"""Schema validation for generated SQL.

This module validates that SQL queries reference valid tables and columns
from the database schema. Catching schema errors before execution is cheaper
than letting them fail at runtime.

See 1.6 for the validate-first pattern.
"""

import sqlglot
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schema import SchemaStore


@dataclass
class ValidationResult:
    """Result of SQL validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


def validate_schema(sql: str, schema_store: "SchemaStore") -> ValidationResult:
    """Validate SQL against the database schema.

    Checks:
    - All referenced tables exist
    - All referenced columns exist in their tables
    - Ambiguous column references are flagged

    Args:
        sql: SQL query to validate
        schema_store: Schema information

    Returns:
        ValidationResult with errors and warnings

    Example:
        >>> result = validate_schema("SELECT foo FROM bar", schema_store)
        >>> result.is_valid
        False
        >>> result.errors
        ['Unknown table: bar']
    """
    errors = []
    warnings = []

    # Parse SQL
    try:
        parsed = sqlglot.parse_one(sql)
    except sqlglot.errors.ParseError as e:
        return ValidationResult(
            is_valid=False,
            errors=[f"SQL parse error: {e}"],
            warnings=[],
        )

    # Extract referenced tables
    referenced_tables = set()
    table_aliases = {}  # alias -> table name

    for table in parsed.find_all(sqlglot.exp.Table):
        table_name = table.name.lower()
        referenced_tables.add(table_name)

        # Track aliases
        if table.alias:
            table_aliases[table.alias.lower()] = table_name

    # Check tables exist
    known_tables = {t.name.lower() for t in schema_store.get_all_tables()}

    for table in referenced_tables:
        if table not in known_tables:
            errors.append(f"Unknown table: {table}")

    # Skip column validation if table errors exist
    if errors:
        return ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
        )

    # Extract and validate columns
    for column in parsed.find_all(sqlglot.exp.Column):
        col_name = column.name.lower()
        table_ref = column.table.lower() if column.table else None

        # Resolve alias to table name
        if table_ref and table_ref in table_aliases:
            table_ref = table_aliases[table_ref]

        if table_ref:
            # Qualified column - check specific table
            table_info = schema_store.get_table(table_ref)
            if table_info:
                col_names = {c.name.lower() for c in table_info.columns}
                if col_name not in col_names and col_name != "*":
                    errors.append(f"Unknown column: {table_ref}.{col_name}")
        else:
            # Unqualified column - check if exists and if ambiguous
            found_in = []
            for table in referenced_tables:
                table_info = schema_store.get_table(table)
                if table_info:
                    col_names = {c.name.lower() for c in table_info.columns}
                    if col_name in col_names:
                        found_in.append(table)

            if not found_in and col_name != "*":
                errors.append(f"Unknown column: {col_name}")
            elif len(found_in) > 1:
                warnings.append(
                    f"Ambiguous column '{col_name}' found in: {', '.join(found_in)}"
                )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
