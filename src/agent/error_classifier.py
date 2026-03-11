"""Classify SQL errors for targeted repair.

Different error types require different repair strategies. This module
categorizes errors so the repair logic can be targeted.

See 1.6 for error classification and repair.
"""

from enum import Enum
from dataclasses import dataclass


class ErrorType(str, Enum):
    """Types of SQL errors."""

    SYNTAX = "syntax"  # SQL syntax error
    SCHEMA = "schema"  # Missing table/column
    TYPE = "type"  # Type mismatch
    TIMEOUT = "timeout"  # Query too slow
    PERMISSION = "permission"  # Access denied
    CONSTRAINT = "constraint"  # Constraint violation
    UNKNOWN = "unknown"


@dataclass
class ClassifiedError:
    """Classified error with repair hints."""

    error_type: ErrorType
    message: str
    repair_hint: str
    retryable: bool


def classify_error(error_message: str) -> ClassifiedError:
    """Classify a database error for repair.

    Args:
        error_message: Error message from database or validation

    Returns:
        ClassifiedError with type and repair hints

    Example:
        >>> error = classify_error("column 'foo' does not exist")
        >>> error.error_type
        ErrorType.SCHEMA
        >>> error.retryable
        True
    """
    error_lower = error_message.lower()

    # Syntax errors
    if "syntax error" in error_lower or "parse error" in error_lower:
        return ClassifiedError(
            error_type=ErrorType.SYNTAX,
            message=error_message,
            repair_hint="Fix SQL syntax. Check for missing keywords, commas, or parentheses.",
            retryable=True,
        )

    # Schema errors (missing tables/columns)
    if any(
        phrase in error_lower
        for phrase in [
            "does not exist",
            "unknown column",
            "unknown table",
            "no such column",
            "no such table",
            "undefined column",
        ]
    ):
        return ClassifiedError(
            error_type=ErrorType.SCHEMA,
            message=error_message,
            repair_hint="Check table and column names against the schema. The referenced object doesn't exist.",
            retryable=True,
        )

    # Type errors
    if any(
        phrase in error_lower
        for phrase in [
            "type mismatch",
            "cannot cast",
            "invalid input syntax for type",
            "operator does not exist",
        ]
    ):
        return ClassifiedError(
            error_type=ErrorType.TYPE,
            message=error_message,
            repair_hint="Check column types. You may need to cast values or use different comparisons.",
            retryable=True,
        )

    # Timeout errors
    if any(
        phrase in error_lower
        for phrase in ["timeout", "canceling statement", "statement timeout"]
    ):
        return ClassifiedError(
            error_type=ErrorType.TIMEOUT,
            message=error_message,
            repair_hint="Query is too slow. Add more filters, simplify joins, or reduce data scope.",
            retryable=True,
        )

    # Permission errors
    if any(
        phrase in error_lower
        for phrase in ["permission denied", "access denied", "insufficient privilege"]
    ):
        return ClassifiedError(
            error_type=ErrorType.PERMISSION,
            message=error_message,
            repair_hint="Insufficient permissions. This operation may not be allowed.",
            retryable=False,
        )

    # Constraint errors
    if any(
        phrase in error_lower
        for phrase in [
            "constraint",
            "violates",
            "unique violation",
            "foreign key",
            "not null",
        ]
    ):
        return ClassifiedError(
            error_type=ErrorType.CONSTRAINT,
            message=error_message,
            repair_hint="Query violates a database constraint.",
            retryable=False,
        )

    # Unknown errors
    return ClassifiedError(
        error_type=ErrorType.UNKNOWN,
        message=error_message,
        repair_hint="Unknown error. Review the query and error message carefully.",
        retryable=True,
    )
