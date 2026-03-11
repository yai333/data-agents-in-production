"""Reward function for schema linking GRPO training.

Scores model predictions against gold tables/columns using F2 + format bonus:
  content_score = 0.4 * table_f2 + 0.6 * column_f2
  total = 0.9 * content_score + 0.1 * format_score

Uses F2 (beta=2) instead of F1 because in schema linking:
- Extra tables/columns are low cost (downstream SQL generator can ignore)
- Missing tables/columns are high cost (SQL generator can't produce correct joins)
F2 weights recall 2x more than precision, matching this asymmetry.

Format score (0.1 weight) provides gradient signal in early GRPO training
when the model doesn't yet know the expected output format. Without it,
correct content in wrong format gets 0 reward — no learning signal.

Gold columns are table-qualified ("invoice.total", "customer.first_name").
Column comparison is also qualified — the model must predict BOTH the
correct column AND the correct table it belongs to. This matters because
21 of 39 Chinook columns are ambiguous (e.g., "name" appears in 5 tables).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# F2 weights recall 2x over precision: missing > extra
BETA = 2.0

# Reward component weights
CONTENT_WEIGHT = 0.9
FORMAT_WEIGHT = 0.1


def parse_prediction(text: str) -> tuple[set[str], set[str]]:
    """Parse model output into predicted tables and columns.

    Handles multiple formats a 1.5B model might produce:
      - Comma-separated: Tables: table1, table2
      - Bullet points:   Tables:\n- table1\n- table2
      - Numbered:        Tables:\n1. table1\n2. table2

    Returns:
        (pred_tables, pred_columns) — raw strings, not yet normalized.
    """
    tables: set[str] = set()
    columns: set[str] = set()

    lines = text.strip().splitlines()
    current_section: str | None = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for section headers: "Tables:" or "Columns:"
        tables_match = re.match(r"^tables?\s*:\s*(.*)$", line, re.IGNORECASE)
        cols_match = re.match(r"^columns?\s*:\s*(.*)$", line, re.IGNORECASE)

        if tables_match:
            current_section = "tables"
            rest = tables_match.group(1).strip()
            if rest:
                _add_items(rest, tables)
            continue

        if cols_match:
            current_section = "columns"
            rest = cols_match.group(1).strip()
            if rest:
                _add_items(rest, columns)
            continue

        # Continuation lines: only bullet/numbered items stay in section
        if current_section and re.match(r"^[-*]\s+|^\d+[.)]\s+", line):
            item = re.sub(r"^[-*]\s+|^\d+[.)]\s+", "", line).strip()
            if item:
                target = tables if current_section == "tables" else columns
                _add_items(item, target)
        else:
            # Non-bullet line breaks the section
            current_section = None

    return tables, columns


def _add_items(text: str, target: set[str]) -> None:
    """Split comma-separated items and add non-empty ones to the target set."""
    for item in text.split(","):
        item = item.strip()
        if item:
            target.add(item)


def _compute_f_beta(pred_set: set, gold_set: set, beta: float = BETA) -> float:
    """Compute F-beta score between predicted and gold sets.

    F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)

    beta > 1 weights recall more than precision.
    beta = 2 means recall is 2x as important as precision.
    """
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def _strip_function_wrapper(col: str) -> str:
    """Strip SQL function wrappers: sum(invoice.total) → invoice.total.

    The model sometimes outputs columns wrapped in functions.
    Extract the innermost table.column reference.
    """
    # Match patterns like sum(table.col), avg(table.col), count(table.col)
    m = re.match(r"^[a-z_]+\((.+)\)$", col.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return col


def _normalize_qualified_columns(cols: set[str]) -> set[str]:
    """Normalize qualified columns: lowercase, strip wrappers, strip whitespace.

    Handles function-wrapped columns: sum(invoice.total) → invoice.total.
    Keeps table.column format intact for qualified comparison.
    """
    result = set()
    for c in cols:
        c = _strip_function_wrapper(c.strip())
        c = c.lower().strip()
        if c:
            result.add(c)
    return result


def _normalize_tables(tables: set[str]) -> set[str]:
    """Normalize tables: lowercase, strip whitespace."""
    return {t.lower().strip() for t in tables if t.strip()}


def _compute_format_score(text: str) -> float:
    """Score how well the output matches expected format.

    Returns 0.0–1.0 based on presence of Tables/Columns sections.
    Provides gradient signal in early training when content is wrong
    but format is partially right.
    """
    has_tables = bool(re.search(r"^tables?\s*:", text, re.IGNORECASE | re.MULTILINE))
    has_columns = bool(re.search(r"^columns?\s*:", text, re.IGNORECASE | re.MULTILINE))

    if has_tables and has_columns:
        return 1.0
    if has_tables or has_columns:
        return 0.5
    return 0.0


def compute_schema_linking_reward(
    predicted_text: str,
    gold_tables: list[str],
    gold_columns: list[str],
) -> dict[str, float]:
    """Compute schema linking reward from model output.

    Reward = 0.9 * content_score + 0.1 * format_score

    Content: 0.4 * table_F2 + 0.6 * column_F2
    Format: 1.0 if both Tables:/Columns: present, 0.5 if one, 0.0 if neither

    Args:
        predicted_text: Raw model output text.
        gold_tables: Gold-standard table names.
        gold_columns: Gold-standard columns as "table.column".

    Returns:
        Dict with total reward and component scores.
    """
    pred_tables, pred_columns = parse_prediction(predicted_text)

    pred_tables_norm = _normalize_tables(pred_tables)
    gold_tables_norm = _normalize_tables(set(gold_tables))

    pred_cols_norm = _normalize_qualified_columns(pred_columns)
    gold_cols_norm = _normalize_qualified_columns(set(gold_columns))

    table_f2 = _compute_f_beta(pred_tables_norm, gold_tables_norm)
    column_f2 = _compute_f_beta(pred_cols_norm, gold_cols_norm)

    content_score = 0.4 * table_f2 + 0.6 * column_f2
    format_score = _compute_format_score(predicted_text)
    total = CONTENT_WEIGHT * content_score + FORMAT_WEIGHT * format_score

    logger.info(
        "schema_linking reward=%.3f content=%.3f format=%.3f "
        "table_f2=%.3f column_f2=%.3f "
        "pred_tables=%s gold_tables=%s pred_cols=%s gold_cols=%s",
        total, content_score, format_score, table_f2, column_f2,
        pred_tables_norm, gold_tables_norm,
        pred_cols_norm, gold_cols_norm,
    )

    return {
        "total": total,
        "content_score": content_score,
        "format_score": format_score,
        "table_f2": table_f2,
        "column_f2": column_f2,
        "pred_tables": sorted(pred_tables_norm),
        "pred_columns": sorted(pred_cols_norm),
    }
