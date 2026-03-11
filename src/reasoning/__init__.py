"""Reasoning patterns for Text-to-SQL agents.

This module provides reasoning-enhanced SQL generation with
multiple levels of thinking: direct, CoT, agentic CoT, and
reasoning models (o1/o3).

Key components:
- ReasoningMethod: Enum of available reasoning levels
- ReasoningResult: Parsed result with SQL and reasoning trace
- generate_sql_with_reasoning: Main entry point
- select_reasoning_method: Auto-select based on complexity

See 1.5 for the principles behind reasoning patterns.
"""

from src.reasoning.selector import (
    ReasoningMethod,
    select_reasoning_method,
    estimate_schema_complexity,
    get_method_characteristics,
    recommend_method_for_latency,
)
from src.reasoning.parser import (
    ReasoningResult,
    parse_reasoning_response,
    extract_sql_from_text,
    calculate_confidence,
    has_complete_reasoning,
)
from src.reasoning.generator import (
    generate_sql_with_reasoning,
    generate_with_fallback,
)
from src.reasoning.prompts import (
    REASONING_PROMPT,
    DIRECT_PROMPT,
    COT_PROMPT,
    SYSTEM_PROMPTS,
)

__all__ = [
    # Enums
    "ReasoningMethod",
    # Data classes
    "ReasoningResult",
    # Main functions
    "generate_sql_with_reasoning",
    "generate_with_fallback",
    # Selection
    "select_reasoning_method",
    "estimate_schema_complexity",
    "get_method_characteristics",
    "recommend_method_for_latency",
    # Parsing
    "parse_reasoning_response",
    "extract_sql_from_text",
    "calculate_confidence",
    "has_complete_reasoning",
    # Prompts
    "REASONING_PROMPT",
    "DIRECT_PROMPT",
    "COT_PROMPT",
    "SYSTEM_PROMPTS",
]
