"""Reward function for SQL agent RL training.

Scores generated SQL by executing both queries and comparing results.
Composite reward: execution match + component F1 + format compliance.

Reward weights:
  execution_match: 0.50  (binary — correct results?)
  component_f1:    0.35  (partial credit — Spider-style structural similarity)
  execution_ok:    0.15  (ran without error, even if wrong)

Format gate: if no extractable SQL → total = 0.0

Why not pure binary? The SFT model starts at ~28% exact match.
Binary reward is too sparse. Component F1 (Spider-style clause
matching) provides gradient for "close but wrong" queries —
correct tables/joins but wrong WHERE still gets partial credit.

Why 0.50/0.35 and not 0.60/0.25? Early in training the model
produces many structurally close but wrong queries. Boosting
component F1 weight gives stronger shaping signal. Execution
match still dominates: a correct query scores at least 0.50,
while a structurally perfect but wrong query caps at 0.50.
"""

import re
from typing import Any

import psycopg2

from evals.sql_components import compare_sql_components


# ── Module-level connection pool ─────────────────────
# Training rollouts are synchronous (VERL calls rollout() in threads).
# Each thread gets its own connection from a shared pool.
_db_url: str | None = None


def init_db(db_url: str) -> None:
    """Store the database URL for execute_sql calls."""
    global _db_url
    _db_url = db_url


def extract_sql(text: str) -> str | None:
    """Extract SQL from model output.

    Supports:
      - ```sql ... ``` code blocks
      - <sql> ... </sql> tags
      - Raw SELECT/WITH statements

    Returns None if no extractable SQL found (triggers format gate).
    """
    # Try ```sql ... ``` blocks first
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try <sql> tags
    match = re.search(r"<sql>(.*?)</sql>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try raw SQL (SELECT or WITH at start of a line)
    match = re.search(
        r"(?:^|\n)\s*((?:SELECT|WITH)\b.*?)(?:\n\n|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        sql = match.group(1).strip().rstrip(";")
        if sql:
            return sql

    return None


def execute_sql(
    sql: str,
    db_url: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Execute SQL against PostgreSQL.

    Uses psycopg2 (synchronous) for compatibility with VERL's
    thread-based rollout execution.

    Returns:
        Dict with success, rows, error keys.
    """
    url = db_url or _db_url
    if not url:
        return {"success": False, "rows": None, "error": "No database URL configured"}

    try:
        conn = psycopg2.connect(url, options=f"-c statement_timeout={int(timeout * 1000)}")
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return {"success": True, "rows": rows, "error": None}
    except Exception as e:
        return {"success": False, "rows": None, "error": str(e)}


def compare_results(
    actual: list[dict] | None,
    expected: list[dict] | None,
) -> bool:
    """Compare query results ignoring column names and row order.

    Normalizes both result sets to multisets of stringified values.
    Handles:
      - Column name differences (aliases)
      - Row ordering differences
      - Type differences (int vs Decimal via str())
      - Superset columns: if the model returns all gold columns plus
        extra columns (e.g., adding album_id), this still matches.
        This is important because the model often includes IDs that
        the gold query omits — the data is correct, just more verbose.
    """
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False
    if len(actual) != len(expected):
        return False
    if not actual and not expected:
        return True

    def row_values(row: dict) -> tuple:
        return tuple(sorted(str(v) for v in row.values()))

    # Fast path: same column count → exact multiset comparison
    actual_ncols = len(actual[0]) if actual else 0
    expected_ncols = len(expected[0]) if expected else 0

    if actual_ncols == expected_ncols:
        actual_norm = sorted(row_values(r) for r in actual)
        expected_norm = sorted(row_values(r) for r in expected)
        return actual_norm == expected_norm

    # Superset path: model (actual) may have MORE columns than gold
    # (expected). That's fine — extra IDs are harmless. But if model
    # has FEWER columns than gold, that's missing data → fail.
    if actual_ncols < expected_ncols:
        return False

    wider, narrower = actual, expected

    def narrow_in_wide(narrow_vals: tuple, wide_vals: tuple) -> bool:
        """Check if all narrow values exist in wide values (multiset)."""
        remaining = list(wide_vals)
        for v in narrow_vals:
            if v in remaining:
                remaining.remove(v)
            else:
                return False
        return True

    # Greedy 1:1 matching: for each narrow row, find a matching wide row
    wide_pool = [row_values(r) for r in wider]
    for narrow_row in narrower:
        nv = row_values(narrow_row)
        matched = False
        for i, wv in enumerate(wide_pool):
            if narrow_in_wide(nv, wv):
                wide_pool.pop(i)
                matched = True
                break
        if not matched:
            return False

    return True


def compute_sql_reward(
    generated_sql: str,
    gold_sql: str,
    db_url: str | None = None,
) -> dict:
    """Score SQL by executing both and comparing results.

    Args:
        generated_sql: Raw model output (may contain markdown, tags, etc.)
        gold_sql: Reference SQL query (clean SQL)
        db_url: PostgreSQL connection URL (uses module default if None)

    Returns:
        Dict with:
          total: float [0, 1]
          execution_match: bool (1.0 if results identical)
          component_f1: float (partial credit for structure)
          format_ok: bool (produced valid SQL)
          executed: bool (SQL ran without error)
    """
    # ── Gate: format adherence ──────────────────────
    sql = extract_sql(generated_sql)
    if sql is None:
        return {
            "total": 0.0,
            "execution_match": False,
            "component_f1": 0.0,
            "format_ok": False,
            "executed": False,
            "detail": "Format gate failed: no extractable SQL",
        }

    # ── Execute generated SQL ───────────────────────
    gen_result = execute_sql(sql, db_url)
    executed = gen_result["success"]

    # ── Execute gold SQL ────────────────────────────
    gold_result = execute_sql(gold_sql, db_url)

    # ── Execution match (weight 0.60) ───────────────
    execution_match = False
    if executed and gold_result["success"]:
        execution_match = compare_results(
            gen_result["rows"],
            gold_result["rows"],
        )

    # ── Component F1 (weight 0.25) ──────────────────
    try:
        comp = compare_sql_components(sql, gold_sql)
        component_f1 = comp.overall_f1
    except Exception:
        component_f1 = 0.0

    # ── Weighted total ──────────────────────────────
    total = (
        0.50 * (1.0 if execution_match else 0.0)
        + 0.35 * component_f1
        + 0.15 * (1.0 if executed else 0.0)
    )
    total = max(0.0, min(1.0, total))

    return {
        "total": round(total, 4),
        "execution_match": execution_match,
        "component_f1": round(component_f1, 4),
        "format_ok": True,
        "executed": executed,
        "detail": gen_result.get("error"),
    }
