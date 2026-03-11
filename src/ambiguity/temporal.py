"""Detect temporal ambiguity in questions.

Temporal ambiguity occurs when time references don't have clear boundaries.

See 1.7 for ambiguity types.
"""

import re
from dataclasses import dataclass


@dataclass
class TemporalReference:
    """A temporal reference found in text."""

    text: str
    ambiguity_type: str
    clarification_needed: str


TEMPORAL_PATTERNS = [
    # "last month/week/year" - calendar vs rolling
    (
        r"\blast\s+(month|week|year|quarter)\b",
        "rolling_vs_calendar",
        "Do you mean the calendar {0} or the last 30/7/365 days?",
    ),
    # "recently" - undefined
    (
        r"\brecently\b",
        "undefined_period",
        "What time period do you consider 'recent'? (Last week, month, etc.)",
    ),
    # "this year/quarter" - calendar vs fiscal
    (
        r"\bthis\s+(year|quarter)\b",
        "calendar_vs_fiscal",
        "Calendar {0} or fiscal {0}?",
    ),
    # "Q1-Q4" - which year
    (
        r"\bQ([1-4])\b",
        "fiscal_year",
        "Which year's Q{0}?",
    ),
    # "last N days/weeks" - boundary
    (
        r"\b(?:last|past)\s+(\d+)\s+(days?|weeks?|months?)\b",
        "boundary",
        "Including today or before today?",
    ),
    # "yesterday/today/tomorrow"
    (
        r"\b(yesterday|today|tomorrow)\b",
        "timezone",
        "Which timezone for '{0}'?",
    ),
]


def detect_temporal_ambiguity(question: str) -> list[TemporalReference]:
    """Detect temporal references that may need clarification.

    Args:
        question: User's question

    Returns:
        List of ambiguous temporal references

    Example:
        >>> refs = detect_temporal_ambiguity("Show sales last month")
        >>> refs[0].ambiguity_type
        'rolling_vs_calendar'
    """
    references = []

    for pattern, amb_type, clarification_template in TEMPORAL_PATTERNS:
        matches = re.finditer(pattern, question, re.IGNORECASE)
        for match in matches:
            text = match.group(0)
            # Format clarification with captured groups
            groups = match.groups()
            if groups:
                clarification = clarification_template.format(*groups)
            else:
                clarification = clarification_template

            references.append(
                TemporalReference(
                    text=text,
                    ambiguity_type=amb_type,
                    clarification_needed=clarification,
                )
            )

    return references
