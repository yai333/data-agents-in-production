"""Output filtering for sensitive data protection.

Prevents sensitive columns from being exposed in query results.

See 2.1 for the complete threat model.
"""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class FilteredOutput:
    """Result of output filtering."""

    data: list[dict[str, Any]]
    redactions: list[str]
    row_count_original: int
    row_count_filtered: int


# Columns that should never be exposed
BLOCKED_COLUMNS = {
    # Credentials
    "password",
    "password_hash",
    "passwd",
    "pwd",
    # Secrets
    "secret",
    "api_key",
    "apikey",
    "token",
    "access_token",
    "refresh_token",
    # PII
    "ssn",
    "social_security",
    "social_security_number",
    # Financial
    "credit_card",
    "cc_number",
    "cvv",
    "card_number",
    # Crypto
    "private_key",
    "encryption_key",
    "secret_key",
    "salt",
}

# Patterns in column names that indicate sensitive data
SENSITIVE_PATTERNS = [
    r".*_secret$",
    r".*_key$",
    r".*_token$",
    r".*_hash$",
    r"^hash$",
    r"^salt$",
    r".*password.*",
    r".*credential.*",
    r".*_ssn$",
    r".*credit_card.*",
]


def filter_output(
    rows: list[dict[str, Any]],
    allowed_columns: set[str] | None = None,
    blocked_columns: set[str] | None = None,
) -> FilteredOutput:
    """Filter sensitive columns from query results.

    Args:
        rows: Query result rows
        allowed_columns: If provided, only allow these columns (allowlist)
        blocked_columns: Additional columns to block (extends default blocklist)

    Returns:
        FilteredOutput with sensitive data removed

    Example:
        >>> rows = [{"name": "Alice", "password_hash": "abc123"}]
        >>> result = filter_output(rows)
        >>> result.data
        [{'name': 'Alice'}]
        >>> "password_hash" in result.redactions[0]
        True
    """
    if not rows:
        return FilteredOutput(
            data=[],
            redactions=[],
            row_count_original=0,
            row_count_filtered=0,
        )

    # Combine default and additional blocked columns
    effective_blocklist = BLOCKED_COLUMNS.copy()
    if blocked_columns:
        effective_blocklist.update(blocked_columns)

    redactions = []
    filtered_rows = []

    for row in rows:
        filtered_row = {}

        for col, value in row.items():
            col_lower = col.lower()

            # Check against blocklist
            if col_lower in effective_blocklist:
                redactions.append(f"Blocked column: {col}")
                continue

            # Check against patterns
            blocked_by_pattern = False
            for pattern in SENSITIVE_PATTERNS:
                if re.match(pattern, col_lower):
                    redactions.append(f"Blocked by pattern: {col}")
                    blocked_by_pattern = True
                    break

            if blocked_by_pattern:
                continue

            # Check against allowlist if provided
            if allowed_columns and col not in allowed_columns:
                redactions.append(f"Not in allowlist: {col}")
                continue

            filtered_row[col] = value

        if filtered_row:  # Only add non-empty rows
            filtered_rows.append(filtered_row)

    return FilteredOutput(
        data=filtered_rows,
        redactions=list(set(redactions)),  # Deduplicate
        row_count_original=len(rows),
        row_count_filtered=len(filtered_rows),
    )


def is_column_sensitive(column_name: str) -> bool:
    """Check if a column name indicates sensitive data.

    Args:
        column_name: Column name to check

    Returns:
        True if column appears sensitive
    """
    col_lower = column_name.lower()

    if col_lower in BLOCKED_COLUMNS:
        return True

    for pattern in SENSITIVE_PATTERNS:
        if re.match(pattern, col_lower):
            return True

    return False
