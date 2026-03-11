"""Detect potential PII leakage through aggregates.

Aggregation queries can reveal individual PII values when
groups are too small.

See 2.3 for aggregate protection patterns.
"""

import sqlglot
from sqlglot import exp


def detect_aggregate_leakage(sql: str) -> list[str]:
    """Detect queries that might leak PII through aggregates.

    Flags:
    - Aggregates with highly specific filters
    - Small group sizes (filters on name, email, etc.)
    - Unique identifier in GROUP BY

    Args:
        sql: SQL query to analyze

    Returns:
        List of warnings

    Example:
        >>> warnings = detect_aggregate_leakage(
        ...     "SELECT AVG(salary) FROM employees WHERE name = 'John'"
        ... )
        >>> len(warnings) > 0
        True
    """
    warnings = []

    try:
        parsed = sqlglot.parse_one(sql)
    except Exception:
        return warnings

    # Check for aggregation functions
    has_aggregation = any(
        parsed.find(agg_type)
        for agg_type in [exp.Count, exp.Sum, exp.Avg, exp.Max, exp.Min]
    )

    if not has_aggregation:
        return warnings

    # Columns that suggest identity
    identity_columns = [
        "name", "first_name", "last_name", "full_name",
        "email", "email_address",
        "ssn", "social_security",
        "phone", "phone_number",
        "employee_id", "user_id", "customer_id",
        "id",
    ]

    # Check WHERE clause for filters on identity columns
    where = parsed.find(exp.Where)
    if where:
        for eq in where.find_all(exp.EQ):
            col = eq.find(exp.Column)
            if col:
                col_name = col.name.lower()
                if any(ident in col_name for ident in identity_columns):
                    warnings.append(
                        f"Aggregate query with filter on '{col.name}' "
                        "may reveal individual records"
                    )

    # Check GROUP BY for potentially small groups
    group = parsed.find(exp.Group)
    if group:
        for col in group.find_all(exp.Column):
            col_name = col.name.lower()
            if any(ident in col_name for ident in identity_columns):
                warnings.append(
                    f"GROUP BY on '{col.name}' may create small groups "
                    "revealing individual values"
                )

    # Check for HAVING with small count thresholds
    having = parsed.find(exp.Having)
    if having:
        for count_cond in having.find_all(exp.Count):
            # If there's a comparison with a small number
            parent = count_cond.parent
            if isinstance(parent, (exp.LT, exp.LTE)):
                warnings.append(
                    "HAVING clause with low count threshold may allow "
                    "identification of individuals"
                )

    return warnings


def is_aggregate_safe(
    sql: str,
    min_group_size: int = 5,
) -> tuple[bool, list[str]]:
    """Check if an aggregate query is safe to execute.

    Args:
        sql: SQL query to check
        min_group_size: Minimum acceptable group size

    Returns:
        Tuple of (is_safe, warnings)
    """
    warnings = detect_aggregate_leakage(sql)

    if not warnings:
        return True, []

    # If there are warnings, query may not be safe
    return False, warnings


def suggest_privacy_fix(sql: str) -> str | None:
    """Suggest modifications to make query more privacy-safe.

    Args:
        sql: Original SQL query

    Returns:
        Suggested modified query, or None if no suggestions

    Example:
        >>> sql = "SELECT name, AVG(salary) FROM employees GROUP BY name"
        >>> fix = suggest_privacy_fix(sql)
        >>> "HAVING COUNT(*)" in (fix or "")
        True
    """
    try:
        parsed = sqlglot.parse_one(sql)
    except Exception:
        return None

    # Check if already has HAVING clause
    if parsed.find(exp.Having):
        return None

    # Check if this is an aggregate query
    has_aggregation = any(
        parsed.find(agg_type)
        for agg_type in [exp.Count, exp.Sum, exp.Avg, exp.Max, exp.Min]
    )

    if not has_aggregation:
        return None

    # Check for GROUP BY
    group = parsed.find(exp.Group)
    if not group:
        return None

    # Suggest adding HAVING COUNT(*) >= 5
    return f"{sql.rstrip(';')} HAVING COUNT(*) >= 5"


class AggregateProtection:
    """Protection layer for aggregate queries.

    Enforces minimum group sizes to prevent PII leakage.
    """

    def __init__(self, min_group_size: int = 5):
        """Initialize with minimum group size.

        Args:
            min_group_size: Minimum rows per group
        """
        self.min_group_size = min_group_size

    def check(self, sql: str) -> tuple[bool, str]:
        """Check if query is safe to execute.

        Args:
            sql: SQL query

        Returns:
            Tuple of (allowed, reason)
        """
        warnings = detect_aggregate_leakage(sql)

        if not warnings:
            return True, ""

        return False, "; ".join(warnings)

    def protect(self, sql: str) -> str:
        """Add protection to an aggregate query.

        Adds HAVING COUNT(*) >= min_group_size if needed.

        Args:
            sql: Original SQL

        Returns:
            Protected SQL
        """
        try:
            parsed = sqlglot.parse_one(sql)
        except Exception:
            return sql

        # Only modify if it's a GROUP BY query
        group = parsed.find(exp.Group)
        if not group:
            return sql

        # Skip if already has HAVING
        if parsed.find(exp.Having):
            return sql

        # Add HAVING clause
        return f"{sql.rstrip(';')} HAVING COUNT(*) >= {self.min_group_size}"
