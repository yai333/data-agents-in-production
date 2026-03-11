"""Render Schema Cards as text for LLM prompts.

Schema Cards are structured data. This module converts them into
human-readable text that fits naturally into prompts.

The rendering format is optimized for:
- LLM comprehension: Clear structure, semantic labels
- Token efficiency: Minimal redundancy
- Accuracy: All critical information included
"""

from src.schema.models import TableCard, ColumnCard, Relationship


def render_column_card(col: ColumnCard, indent: str = "  ") -> list[str]:
    """Render a single column as text lines.

    Format:
      - column_name (TYPE) [PK] [-> ref]: Description
        Examples: val1, val2
        Note: business rule

    Args:
        col: The column to render
        indent: Prefix for each line

    Returns:
        List of text lines
    """
    lines = []

    # Main column line
    col_line = f"{indent}- {col.name} ({col.data_type})"

    if col.is_primary_key:
        col_line += " [PK]"

    if col.is_foreign_key and col.references:
        col_line += f" -> {col.references}"

    if col.description:
        col_line += f": {col.description}"

    lines.append(col_line)

    # Examples (if present)
    if col.examples:
        examples_str = ", ".join(str(e) for e in col.examples[:3])
        lines.append(f"{indent}  Examples: {examples_str}")

    # Business rules (if present)
    if col.business_rules:
        lines.append(f"{indent}  Note: {col.business_rules}")

    return lines


def render_table_card(card: TableCard) -> str:
    """Render a TableCard as text for LLM prompts.

    Format:
    ## table_name
    Description of the table.

    Columns:
      - col1 (TYPE): description
      - col2 (TYPE) -> ref: description

    Relationships:
      - table -> other (JOIN_TYPE): condition

    Args:
        card: The table to render

    Returns:
        Multi-line text representation
    """
    lines = [
        f"## {card.name}",
        card.description,
        "",
        "Columns:",
    ]

    # Render each column
    for col in card.columns:
        lines.extend(render_column_card(col))

    # Relationships (structured format: WrenAI/Looker MDL pattern)
    if card.relationships:
        lines.append("")
        lines.append("Relationships:")
        for rel in card.relationships:
            # Format: table -> other_table (JOIN_TYPE): condition
            other_table = rel.models[1] if len(rel.models) > 1 else "?"
            lines.append(f"  - {card.name} -> {other_table} ({rel.join_type}): {rel.condition}")

    return "\n".join(lines)


def render_schema(tables: list[TableCard], include_row_counts: bool = True) -> str:
    """Render multiple tables for a prompt.

    Args:
        tables: List of tables to render
        include_row_counts: Whether to include row count estimates

    Returns:
        Combined text for all tables, separated by blank lines
    """
    rendered = []
    for table in tables:
        rendered.append(render_table_card(table))

    return "\n\n".join(rendered)


def render_schema_summary(tables: list[TableCard]) -> str:
    """Render a brief summary of available tables.

    Useful when you need to show the LLM what tables exist
    without providing full column details.

    Args:
        tables: List of tables

    Returns:
        Compact summary of table names and purposes
    """
    lines = ["Available tables:"]

    for table in tables:
        lines.append(f"  - {table.name}: {table.description}")

    return "\n".join(lines)


def render_table_names(tables: list[TableCard]) -> str:
    """Render just the table names as a comma-separated list.

    Args:
        tables: List of tables

    Returns:
        Comma-separated table names
    """
    return ", ".join(t.name for t in tables)
