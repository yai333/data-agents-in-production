"""Main ambiguity detection and handling.

This module combines all ambiguity detectors and decides whether to
proceed, clarify, or reject based on confidence.

See 1.7 for the ambiguity handling flow.
"""

from dataclasses import dataclass
from enum import Enum

from src.ambiguity.lexical import detect_lexical_ambiguity
from src.ambiguity.temporal import detect_temporal_ambiguity, TemporalReference
from src.ambiguity.scope import detect_scope_ambiguity, ScopeAmbiguity, SCOPE_PATTERNS


class AmbiguityAction(str, Enum):
    """What to do when ambiguity is detected."""

    PROCEED = "proceed"  # Confidence high enough, use defaults
    CLARIFY = "clarify"  # Ask user for clarification
    REJECT = "reject"  # Too ambiguous to proceed


@dataclass
class AmbiguityAnalysis:
    """Result of ambiguity analysis."""

    action: AmbiguityAction
    confidence: float
    lexical: list[str]
    temporal: list[TemporalReference]
    scope: list[ScopeAmbiguity]
    clarifying_questions: list[str]
    assumptions: list[str]


def analyze_ambiguity(
    question: str,
    sql_confidence: float,
    schema_tables: set[str],
    confidence_threshold: float = 0.7,
) -> AmbiguityAnalysis:
    """Analyze a question for ambiguity and decide action.

    Args:
        question: User's question
        sql_confidence: Confidence from SQL generation
        schema_tables: Available table names
        confidence_threshold: Below this, ask for clarification

    Returns:
        AmbiguityAnalysis with action and details

    Example:
        >>> result = analyze_ambiguity(
        ...     "Show top customers last month",
        ...     sql_confidence=0.6,
        ...     schema_tables={"customer", "order"},
        ... )
        >>> result.action
        AmbiguityAction.CLARIFY
    """
    # Run all detectors
    lexical = detect_lexical_ambiguity(question, schema_tables)
    temporal = detect_temporal_ambiguity(question)
    scope = detect_scope_ambiguity(question)

    # Count ambiguity factors
    ambiguity_count = len(lexical) + len(temporal) + len(scope)

    # Adjust confidence based on ambiguity
    # Each ambiguity reduces confidence by 10%
    adjusted_confidence = sql_confidence - (ambiguity_count * 0.1)
    adjusted_confidence = max(0.0, adjusted_confidence)

    # Generate clarifying questions and assumptions
    questions = []
    assumptions = []

    # Handle lexical ambiguity
    for term in lexical:
        questions.append(
            f"When you say '{term}', do you mean the database table or the general concept?"
        )

    # Handle temporal ambiguity
    for temp in temporal:
        questions.append(temp.clarification_needed)

    # Handle scope ambiguity
    for sc in scope:
        if sc.default:
            assumptions.append(f"Assuming '{sc.phrase}' means {sc.default}")
        else:
            options_str = " or ".join(sc.options[:3])
            questions.append(f"For '{sc.phrase}', which do you mean: {options_str}?")

    # Decide action
    if adjusted_confidence >= confidence_threshold and len(questions) == 0:
        action = AmbiguityAction.PROCEED
    elif adjusted_confidence >= 0.4 or len(questions) <= 2:
        action = AmbiguityAction.CLARIFY
    else:
        action = AmbiguityAction.REJECT

    # If we have assumptions and confidence is reasonable, proceed
    if assumptions and not questions and adjusted_confidence >= 0.5:
        action = AmbiguityAction.PROCEED

    return AmbiguityAnalysis(
        action=action,
        confidence=adjusted_confidence,
        lexical=lexical,
        temporal=temporal,
        scope=scope,
        clarifying_questions=questions[:3],  # Limit to 3 questions
        assumptions=assumptions,
    )
