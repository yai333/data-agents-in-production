"""SQL validation combining all validators.

This module provides a unified interface for validating SQL queries
before execution. It combines schema, safety, and join validation.

See 1.6 for the validate-first pattern.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.validators.schema_validator import validate_schema, ValidationResult
from src.validators.safety_validator import validate_safety, SafetyResult

if TYPE_CHECKING:
    from src.schema import SchemaStore


@dataclass
class FullValidationResult:
    """Combined validation result from all validators."""

    is_valid: bool
    is_safe: bool
    schema_errors: list[str]
    safety_violations: list[str]
    warnings: list[str]
    risk_level: str

    @property
    def can_execute(self) -> bool:
        """Whether the query is safe to execute."""
        return self.is_valid and self.is_safe

    @property
    def all_errors(self) -> list[str]:
        """All errors combined."""
        return self.schema_errors + self.safety_violations

    def summary(self) -> str:
        """Human-readable summary of validation results."""
        if self.can_execute:
            if self.warnings:
                return f"Valid with warnings: {'; '.join(self.warnings)}"
            return "Valid"
        else:
            return f"Invalid: {'; '.join(self.all_errors)}"


def validate_sql(sql: str, schema_store: "SchemaStore") -> FullValidationResult:
    """Run all validators on SQL query.

    Performs:
    1. Schema validation (tables/columns exist)
    2. Safety validation (no dangerous operations)

    Args:
        sql: SQL query to validate
        schema_store: Schema information

    Returns:
        FullValidationResult with all checks

    Example:
        >>> result = validate_sql("SELECT * FROM customer", schema_store)
        >>> result.can_execute
        True
        >>> result.warnings
        ['SELECT * may expose sensitive columns', 'No LIMIT clause...']
    """
    schema_result = validate_schema(sql, schema_store)
    safety_result = validate_safety(sql)

    return FullValidationResult(
        is_valid=schema_result.is_valid,
        is_safe=safety_result.is_safe,
        schema_errors=schema_result.errors,
        safety_violations=safety_result.violations if not safety_result.is_safe else [],
        warnings=schema_result.warnings + (
            safety_result.violations if safety_result.is_safe else []
        ),
        risk_level=safety_result.risk_level,
    )


__all__ = [
    "validate_sql",
    "validate_schema",
    "validate_safety",
    "FullValidationResult",
    "ValidationResult",
    "SafetyResult",
]
