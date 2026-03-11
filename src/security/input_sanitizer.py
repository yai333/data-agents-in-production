"""Input sanitization for prompt injection defense.

This module provides first-line defense against prompt injection attacks.
It should NOT be relied upon alone—use as part of defense-in-depth.

See 2.1 for the complete threat model.
"""

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class SanitizationResult:
    """Result of input sanitization."""

    is_clean: bool
    cleaned_input: str
    violations: list[str]
    risk_score: float


# Patterns that indicate prompt injection attempts
INJECTION_PATTERNS = [
    (r"ignore\s+(previous|all|above)\s+instructions?", "instruction_override"),
    (r"system\s*:\s*", "system_prompt_injection"),
    (r"\[INST\]|\[\/INST\]", "instruction_tags"),
    (r"<\|im_start\|>|<\|im_end\|>", "chat_ml_injection"),
    (r"<\|system\|>|<\|user\|>|<\|assistant\|>", "role_injection"),
    (r"```\s*(sql|python|bash|sh)\s*\n.*?(DROP|DELETE|TRUNCATE)", "code_block_injection"),
    (r"DROP\s+TABLE|DELETE\s+FROM|TRUNCATE\s+TABLE", "ddl_injection"),
    (r"UNION\s+(ALL\s+)?SELECT", "union_injection"),
    (r"--\s*$|;\s*--", "sql_comment_injection"),
    (r"'(\s*OR\s*'?1'?\s*=\s*'?1|.*--)", "sql_always_true"),
    (r"EXEC(\s+|UTE\s+)", "exec_injection"),
    (r"xp_cmdshell|sp_oacreate", "mssql_injection"),
]

# Maximum input length to prevent context stuffing
MAX_INPUT_LENGTH = 2000


def sanitize_input(user_input: str) -> SanitizationResult:
    """Sanitize user input for potential injection attacks.

    This is a FIRST LINE defense. It catches obvious attacks
    but should not be relied upon alone.

    Args:
        user_input: Raw user input

    Returns:
        SanitizationResult with cleaned input and violations

    Example:
        >>> result = sanitize_input("Show me sales for Q1")
        >>> result.is_clean
        True

        >>> result = sanitize_input("Ignore previous instructions...")
        >>> result.is_clean
        False
        >>> result.risk_score > 0.3
        True
    """
    violations = []
    risk_score = 0.0
    cleaned = user_input

    # Length check
    if len(user_input) > MAX_INPUT_LENGTH:
        violations.append(f"Input too long: {len(user_input)} > {MAX_INPUT_LENGTH}")
        risk_score += 0.3
        cleaned = cleaned[:MAX_INPUT_LENGTH]

    # Unicode normalization (prevent homograph attacks)
    normalized = unicodedata.normalize("NFKC", cleaned)
    if normalized != cleaned:
        violations.append("Unicode normalization applied")
        risk_score += 0.1
        cleaned = normalized

    # Pattern matching for injection attempts
    for pattern, violation_type in INJECTION_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE | re.DOTALL):
            violations.append(f"Potential injection: {violation_type}")
            risk_score += 0.4

    # Suspicious character sequences
    if cleaned.count("'") > 5:
        violations.append("Excessive single quotes")
        risk_score += 0.2

    if cleaned.count('"') > 5:
        violations.append("Excessive double quotes")
        risk_score += 0.2

    if cleaned.count(";") > 2:
        violations.append("Multiple semicolons")
        risk_score += 0.2

    # Check for encoded characters that might bypass filters
    if "&#" in cleaned or "%27" in cleaned or "%3B" in cleaned:
        violations.append("Encoded characters detected")
        risk_score += 0.3

    # Check for nested quotes/escapes
    if re.search(r"\\['\"\\]", cleaned):
        violations.append("Escape sequences detected")
        risk_score += 0.2

    return SanitizationResult(
        is_clean=len(violations) == 0,
        cleaned_input=cleaned,
        violations=violations,
        risk_score=min(risk_score, 1.0),
    )


def is_suspicious(user_input: str, threshold: float = 0.5) -> bool:
    """Quick check if input is suspicious.

    Args:
        user_input: Input to check
        threshold: Risk score threshold

    Returns:
        True if risk score exceeds threshold
    """
    result = sanitize_input(user_input)
    return result.risk_score >= threshold
