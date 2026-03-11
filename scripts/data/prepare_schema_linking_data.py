"""Prepare schema linking training data from sql_train.jsonl.

For each example:
1. Parse gold_sql with sqlglot to extract tables + qualified columns
2. Build ddl_context (full DDL for all 11 Chinook tables)
3. Build glossary_context from question keywords
4. Save (without gold_sql — model must never see the answer)

Usage:
    source .venv/bin/activate && python -m scripts.data.prepare_schema_linking_data
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import sqlglot
from sqlglot import exp

from src.schema.render import render_schema
from src.schema.store import SchemaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
PROMPT_TEMPLATE_PATH = ROOT / "scripts" / "chapter_4C" / "schema_linking_prompt.txt"


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL records from disk."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: Path) -> None:
    """Save records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d examples to %s", len(data), path)


def extract_gold_labels(
    gold_sql: str,
    valid_table_names: set[str],
    table_columns: dict[str, set[str]],
) -> tuple[list[str], list[str]]:
    """Extract gold tables and table-qualified columns via sqlglot AST.

    Uses sqlglot to parse the SQL into an AST, then:
    1. Tables: find all Table nodes, filter against valid schema names
       (removes CTE names, subquery aliases).
    2. Columns: find all Column nodes, resolve alias prefixes to real
       table names via the alias map built from Table nodes. For bare
       columns (no prefix), qualify against gold tables.

    Returns:
        (gold_tables, gold_columns) — columns are "table.column" format.
    """
    cleaned_sql = gold_sql.rstrip().rstrip(";")

    try:
        parsed = sqlglot.parse_one(cleaned_sql, dialect="postgres")
    except sqlglot.errors.ParseError:
        logger.warning("sqlglot parse failed, returning empty: %s", cleaned_sql[:80])
        return [], []

    # --- Tables ---
    # Build alias → real table name map and collect gold tables
    alias_map: dict[str, str] = {}
    gold_tables: set[str] = set()

    for table_node in parsed.find_all(exp.Table):
        table_name = table_node.name.lower()
        if table_name not in valid_table_names:
            continue
        gold_tables.add(table_name)
        alias = table_node.alias
        if alias:
            alias_map[alias.lower()] = table_name
        alias_map[table_name] = table_name

    # --- Columns ---
    valid_column_names = set()
    for cols in table_columns.values():
        valid_column_names.update(cols)

    qualified_cols: set[str] = set()

    for col_node in parsed.find_all(exp.Column):
        col_name = col_node.name.lower()
        if col_name not in valid_column_names:
            continue

        prefix = col_node.table.lower() if col_node.table else ""

        if prefix and prefix in alias_map:
            # Alias-qualified: e.first_name → employee.first_name
            qualified_cols.add(f"{alias_map[prefix]}.{col_name}")
        elif prefix and prefix in valid_table_names:
            # Direct table reference: employee.first_name
            qualified_cols.add(f"{prefix}.{col_name}")
        else:
            # Bare column: qualify against gold tables
            matching = [
                t for t in gold_tables
                if col_name in table_columns.get(t, set())
            ]
            if len(matching) == 1:
                qualified_cols.add(f"{matching[0]}.{col_name}")
            elif len(matching) > 1:
                for t in matching:
                    qualified_cols.add(f"{t}.{col_name}")
            # No match → skip (likely an alias like total_sales)

    return sorted(gold_tables), sorted(qualified_cols)


def build_glossary_context(
    question: str,
    store: SchemaStore,
) -> str:
    """Build glossary context by extracting keywords from the question.

    Splits question into words, queries the glossary for matches.
    Returns formatted glossary text or "None" if no matches.
    """
    # Extract keywords: words longer than 2 chars, lowercased
    words = re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
    # Deduplicate while preserving order
    seen: set[str] = set()
    keywords: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            keywords.append(w)

    entries = store.get_glossary_entries(keywords)

    if not entries:
        return "None"

    lines = []
    for entry in entries:
        lines.append(f"- {entry['term']}: {entry['definition']}")
    return "\n".join(lines)


def prepare_examples(
    examples: list[dict],
    ddl_context: str,
    prompt_template: str,
    store: SchemaStore,
) -> list[dict]:
    """Prepare schema linking examples from raw SQL training data.

    Args:
        examples: Raw examples with 'question' and 'gold_sql'.
        ddl_context: Pre-rendered DDL for all tables.
        prompt_template: Prompt template with {ddl_context}, {glossary_context},
            {question} placeholders.
        store: SchemaStore for glossary lookups.

    Returns:
        List of prepared examples with prompt, gold_tables, gold_columns.
    """
    valid_table_names = {t.name for t in store.get_all_tables()}
    # Map table_name → set of column names (for qualifying bare columns)
    table_columns: dict[str, set[str]] = {}
    for t in store.get_all_tables():
        table_columns[t.name] = {c.name.lower() for c in t.columns}

    prepared = []
    for i, ex in enumerate(examples):
        question = ex["question"]
        gold_sql = ex["gold_sql"]

        gold_tables, gold_columns = extract_gold_labels(
            gold_sql, valid_table_names, table_columns,
        )

        if not gold_tables:
            logger.warning(
                "Example %d has no tables extracted from SQL: %s",
                i, gold_sql[:100],
            )

        glossary_context = build_glossary_context(question, store)

        # No gold_sql in output — the model must never see the answer.
        # Only gold_tables and gold_columns are kept (for reward computation).
        prepared.append({
            "question": question,
            "ddl_context": ddl_context,
            "glossary_context": glossary_context,
            "gold_tables": gold_tables,
            "gold_columns": gold_columns,
        })

    return prepared


def main() -> None:
    """Prepare schema linking data from existing train/val splits."""
    train_path = ROOT / "data" / "sql_train.jsonl"
    val_path = ROOT / "data" / "sql_val.jsonl"
    out_train = ROOT / "data" / "schema_linking_train.jsonl"
    out_val = ROOT / "data" / "schema_linking_val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    # Load schema
    schema_path = ROOT / "config" / "chinook_schema.json"
    store = SchemaStore(schema_path)
    all_tables = store.get_all_tables()
    logger.info("Loaded %d tables from schema", len(all_tables))

    # Build full DDL context (all tables, rendered once)
    ddl_context = render_schema(all_tables)
    logger.info("DDL context: %d chars", len(ddl_context))

    # Load prompt template
    prompt_template = PROMPT_TEMPLATE_PATH.read_text()

    # Process train split
    train_data = load_jsonl(train_path)
    logger.info("Loaded %d train examples", len(train_data))
    train_prepared = prepare_examples(
        train_data, ddl_context, prompt_template, store,
    )
    save_jsonl(train_prepared, out_train)

    # Process val split (if exists)
    if val_path.exists():
        val_data = load_jsonl(val_path)
        logger.info("Loaded %d val examples", len(val_data))
        val_prepared = prepare_examples(
            val_data, ddl_context, prompt_template, store,
        )
        save_jsonl(val_prepared, out_val)

    # Summary statistics
    all_prepared = train_prepared
    tables_per_example = [len(ex["gold_tables"]) for ex in all_prepared]
    cols_per_example = [len(ex["gold_columns"]) for ex in all_prepared]
    logger.info(
        "Stats (train): avg_tables=%.1f avg_columns=%.1f",
        sum(tables_per_example) / len(tables_per_example),
        sum(cols_per_example) / len(cols_per_example),
    )

    # Spot-check first 5 examples
    print("\n" + "=" * 60)
    print("SPOT CHECK — first 5 examples")
    print("=" * 60)
    for ex in train_prepared[:5]:
        print(f"\nQ: {ex['question']}")
        print(f"  Tables:  {ex['gold_tables']}")
        print(f"  Columns: {ex['gold_columns']}")


if __name__ == "__main__":
    main()
