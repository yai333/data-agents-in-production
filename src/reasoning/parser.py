"""Parse four-phase reasoning responses.

This module extracts structured information from LLM responses
that use the four-phase reasoning pattern with XML tags.

See 1.5 for the reasoning pattern specification.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReasoningResult:
    """Parsed result from four-phase reasoning.

    Attributes:
        reasoning: User intent understanding
        analysis: Schema mapping (tables, joins, filters)
        query_explanation: SQL with explanation
        verification: Self-check results
        sql: Extracted SQL query
        confidence: Calculated confidence (0.0-1.0)
        method: Reasoning method used
        raw_response: Original LLM response
    """

    reasoning: str
    analysis: str
    query_explanation: str
    verification: str
    sql: str
    confidence: float = 0.8
    method: str = "cot"
    raw_response: str = ""


def parse_reasoning_response(response: str) -> ReasoningResult:
    """Parse a four-phase reasoning response.

    Args:
        response: Raw LLM response with XML tags

    Returns:
        ReasoningResult with extracted components

    Example:
        >>> response = '''
        ... <reasoning>User wants customer count</reasoning>
        ... <analysis>Need customer table, COUNT aggregation</analysis>
        ... <query>SELECT COUNT(*) FROM customer</query>
        ... <verification>Returns single count value</verification>
        ... '''
        >>> result = parse_reasoning_response(response)
        >>> result.sql
        'SELECT COUNT(*) FROM customer'
    """
    def extract_tag(tag: str) -> str:
        """Extract content between XML tags."""
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    reasoning = extract_tag("reasoning")
    analysis = extract_tag("analysis")
    query_section = extract_tag("query")
    verification = extract_tag("verification")

    # Extract SQL from query section
    sql = extract_sql_from_text(query_section or response)

    # Calculate confidence based on verification quality
    confidence = calculate_confidence(verification, sql, reasoning)

    return ReasoningResult(
        reasoning=reasoning,
        analysis=analysis,
        query_explanation=query_section,
        verification=verification,
        sql=sql,
        confidence=confidence,
        method="cot",
        raw_response=response,
    )


def extract_sql_from_text(text: str) -> str:
    """Extract SQL query from text that may include explanation.

    Handles multiple formats:
    - Code blocks (```sql ... ```)
    - Inline SQL (SELECT ... )
    - SQL at end of text

    Args:
        text: Text potentially containing SQL

    Returns:
        Extracted SQL query
    """
    if not text:
        return ""

    # Try to extract from code blocks first (most reliable)
    code_block = re.search(
        r"```sql?\n?(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE
    )
    if code_block:
        return code_block.group(1).strip()

    # Look for SQL statements (SELECT, WITH, INSERT, UPDATE, DELETE)
    sql_match = re.search(
        r"((?:WITH|SELECT|INSERT|UPDATE|DELETE)\s+.*?)(?:\n\n|\Z|<)",
        text,
        re.DOTALL | re.IGNORECASE
    )
    if sql_match:
        sql = sql_match.group(1).strip()
        # Clean up any trailing explanation
        sql = re.sub(r"\n\n.*$", "", sql, flags=re.DOTALL)
        return sql

    # Fallback: return cleaned text
    return text.strip()


def calculate_confidence(
    verification: str,
    sql: str,
    reasoning: str,
) -> float:
    """Calculate confidence score based on response quality.

    Factors:
    - Presence of verification phase
    - Specific checks mentioned in verification
    - SQL complexity vs reasoning depth
    - Warning signs in reasoning

    Args:
        verification: Verification phase content
        sql: Generated SQL
        reasoning: Reasoning phase content

    Returns:
        Confidence score (0.0-1.0)
    """
    if not sql:
        return 0.0

    # Base confidence
    confidence = 0.6

    # Boost for having verification
    if verification:
        confidence += 0.15

    # Boost for having reasoning
    if reasoning:
        confidence += 0.05

    # Boost if verification mentions specific checks
    verification_checks = [
        "columns", "joins", "filters", "aggregat",
        "group by", "correct", "matches", "returns"
    ]
    for check in verification_checks:
        if check.lower() in verification.lower():
            confidence += 0.02

    # Penalty for warning signs
    warning_signs = [
        "unsure", "might", "maybe", "not certain",
        "could be", "possibly", "unclear"
    ]
    all_text = f"{reasoning} {verification}".lower()
    for warning in warning_signs:
        if warning in all_text:
            confidence -= 0.05

    # Penalty for very simple SQL with complex reasoning
    # (might indicate misunderstanding)
    if len(sql) < 30 and len(reasoning) > 200:
        confidence -= 0.1

    # Cap confidence
    return max(0.1, min(confidence, 0.95))


def has_complete_reasoning(response: str) -> bool:
    """Check if response has all four reasoning phases.

    Args:
        response: LLM response to check

    Returns:
        True if all phases present
    """
    required_tags = ["reasoning", "analysis", "query", "verification"]
    return all(
        f"<{tag}>" in response and f"</{tag}>" in response
        for tag in required_tags
    )


def extract_partial_reasoning(response: str) -> dict[str, str]:
    """Extract whatever reasoning phases are present.

    Useful for handling incomplete responses.

    Args:
        response: Potentially incomplete LLM response

    Returns:
        Dict of present phases and their content
    """
    phases = ["reasoning", "analysis", "query", "verification"]
    result = {}

    for phase in phases:
        pattern = f"<{phase}>(.*?)</{phase}>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            result[phase] = match.group(1).strip()

    return result
