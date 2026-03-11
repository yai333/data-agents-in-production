"""Ambiguity detection and handling for Text-to-SQL.

This module detects ambiguous queries and generates clarifying questions.

See 1.7 for ambiguity handling patterns.
"""

from src.ambiguity.detector import (
    analyze_ambiguity,
    AmbiguityAction,
    AmbiguityAnalysis,
)
from src.ambiguity.lexical import detect_lexical_ambiguity
from src.ambiguity.temporal import detect_temporal_ambiguity, TemporalReference
from src.ambiguity.scope import detect_scope_ambiguity, ScopeAmbiguity

__all__ = [
    "analyze_ambiguity",
    "AmbiguityAction",
    "AmbiguityAnalysis",
    "detect_lexical_ambiguity",
    "detect_temporal_ambiguity",
    "TemporalReference",
    "detect_scope_ambiguity",
    "ScopeAmbiguity",
]
