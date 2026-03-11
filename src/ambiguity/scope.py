"""Detect scope ambiguity in questions.

Scope ambiguity occurs when aggregation level or grouping is unclear.

See 1.7 for ambiguity types.
"""

from dataclasses import dataclass


@dataclass
class ScopeAmbiguity:
    """Detected scope ambiguity."""

    phrase: str
    ambiguity_type: str
    options: list[str]
    default: str | None = None


# Common scope-related phrases and their ambiguities
SCOPE_PATTERNS = {
    "top": {
        "type": "limit",
        "options": ["Top 5", "Top 10", "Top 20", "Top 100"],
        "default": "Top 10",
    },
    "bottom": {
        "type": "limit",
        "options": ["Bottom 5", "Bottom 10", "Bottom 20"],
        "default": "Bottom 10",
    },
    "by region": {
        "type": "grouping",
        "options": ["Geographic region", "Sales region", "Shipping region"],
        "default": None,
    },
    "by category": {
        "type": "grouping",
        "options": ["Product category", "Customer category"],
        "default": None,
    },
    "total": {
        "type": "aggregation_scope",
        "options": ["All time total", "Period total", "Per-group total"],
        "default": "All time total",
    },
    "average": {
        "type": "aggregation_scope",
        "options": ["Overall average", "Per-customer average", "Per-period average"],
        "default": "Overall average",
    },
    "by country": {
        "type": "grouping",
        "options": ["Billing country", "Shipping country", "Customer country"],
        "default": "Customer country",
    },
}


def detect_scope_ambiguity(question: str) -> list[ScopeAmbiguity]:
    """Detect scope-related ambiguity in a question.

    Args:
        question: User's question

    Returns:
        List of scope ambiguities found

    Example:
        >>> ambs = detect_scope_ambiguity("Show top customers by region")
        >>> [a.phrase for a in ambs]
        ['top', 'by region']
    """
    question_lower = question.lower()
    ambiguities = []

    for phrase, config in SCOPE_PATTERNS.items():
        if phrase in question_lower:
            # Check if context clarifies (e.g., "top 10" is clear)
            if phrase == "top":
                import re

                if re.search(r"top\s+\d+", question_lower):
                    continue  # "top N" is clear

            ambiguities.append(
                ScopeAmbiguity(
                    phrase=phrase,
                    ambiguity_type=config["type"],
                    options=config["options"],
                    default=config.get("default"),
                )
            )

    return ambiguities
