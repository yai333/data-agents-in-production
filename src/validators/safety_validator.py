"""Safety validation for SQL queries.

This module checks SQL queries for dangerous operations that should never
be allowed in a Text-to-SQL agent (DDL, DML, privilege commands).

See 1.6 for the safety validation pattern.
"""

from dataclasses import dataclass


@dataclass
class SafetyResult:
    """Result of safety validation."""

    is_safe: bool
    violations: list[str]
    risk_level: str  # "low", "medium", "high"


# Operations that should never be allowed
BLOCKED_OPERATIONS = {
    "DROP",
    "TRUNCATE",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "CREATE",
    "GRANT",
    "REVOKE",
    "EXEC",
    "EXECUTE",
}

# SQL comments that might hide injection
COMMENT_PATTERNS = ["--", "/*", "*/"]


def validate_safety(sql: str) -> SafetyResult:
    """Validate SQL for safety concerns.

    Checks:
    - No DDL statements (DROP, ALTER, CREATE)
    - No DML statements (INSERT, UPDATE, DELETE)
    - No privilege commands (GRANT, REVOKE)
    - Warns about risky patterns (SELECT *, no LIMIT)
    - Detects potential injection patterns

    Args:
        sql: SQL query to validate

    Returns:
        SafetyResult with violations and risk level

    Example:
        >>> result = validate_safety("DROP TABLE users")
        >>> result.is_safe
        False
        >>> result.violations
        ['Blocked operation: DROP']
    """
    violations = []
    risk_level = "low"

    sql_upper = sql.upper()

    # Check for blocked operations
    # Use word boundaries to avoid false positives (e.g., "UPDATED_AT")
    for op in BLOCKED_OPERATIONS:
        # Check if operation appears as a word (not part of identifier)
        import re

        if re.search(rf"\b{op}\b", sql_upper):
            violations.append(f"Blocked operation: {op}")
            risk_level = "high"

    # Check for SELECT * (may expose sensitive columns)
    if "SELECT *" in sql_upper or "SELECT  *" in sql_upper:
        violations.append("SELECT * may expose sensitive columns")
        if risk_level == "low":
            risk_level = "medium"

    # Check for missing LIMIT on potentially large results
    if "LIMIT" not in sql_upper:
        # Aggregation-only queries are OK without LIMIT
        is_aggregation_only = (
            "COUNT(" in sql_upper
            or "SUM(" in sql_upper
            or "AVG(" in sql_upper
            or "MAX(" in sql_upper
            or "MIN(" in sql_upper
        ) and "GROUP BY" not in sql_upper

        if not is_aggregation_only:
            violations.append("No LIMIT clause - may return excessive rows")
            if risk_level == "low":
                risk_level = "medium"

    # Check for UNION-based injection patterns
    if "UNION" in sql_upper:
        # Multiple SELECT statements after UNION is suspicious
        parts = sql_upper.split("UNION")
        if len(parts) > 2:  # More than one UNION
            violations.append("Multiple UNION clauses detected - potential injection")
            risk_level = "high"

    # Check for comment-based injection
    for pattern in COMMENT_PATTERNS:
        if pattern in sql:
            violations.append(f"SQL comment detected: {pattern}")
            if risk_level == "low":
                risk_level = "medium"

    # Check for multiple statements (semicolon injection)
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    if len(statements) > 1:
        violations.append("Multiple SQL statements detected")
        risk_level = "high"

    # Check for suspicious string patterns
    suspicious_patterns = [
        ("'1'='1'", "Always-true condition (injection pattern)"),
        ("OR 1=1", "Always-true condition (injection pattern)"),
        ("' OR '", "Quote-based injection pattern"),
        ("SLEEP(", "Time-based injection attempt"),
        ("BENCHMARK(", "Time-based injection attempt"),
        ("LOAD_FILE(", "File access attempt"),
        ("INTO OUTFILE", "File write attempt"),
        ("INTO DUMPFILE", "File write attempt"),
    ]

    for pattern, description in suspicious_patterns:
        if pattern.upper() in sql_upper:
            violations.append(description)
            risk_level = "high"

    return SafetyResult(
        is_safe=risk_level != "high",
        violations=violations,
        risk_level=risk_level,
    )
