"""Data models for few-shot retrieval.

This module defines the FewShotExample structure used to store
and retrieve example queries for in-context learning.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class FewShotExample(BaseModel):
    """A single few-shot example for retrieval.

    Each example pairs a natural language question with its correct SQL,
    along with metadata for filtering and evaluation.

    The core fields (question, sql, explanation) are what the LLM sees.
    The metadata fields (id, tables_used, category, difficulty) support
    retrieval and evaluation but are NOT included in prompts.
    """

    # Core: what the LLM sees in the prompt
    question: str = Field(..., description="Natural language question")
    sql: str = Field(..., description="Correct SQL query")
    explanation: Optional[str] = Field(
        default=None,
        description="Why this SQL answers the question (optional)"
    )

    # Metadata: for retrieval and evaluation (NOT shown to LLM)
    id: str = Field(..., description="Unique identifier")
    tables_used: List[str] = Field(
        default_factory=list,
        description="Tables involved in the query"
    )
    category: str = Field(default="general", description="Query type")
    difficulty: str = Field(default="medium", description="Complexity level")

    class Config:
        """Pydantic configuration."""
        frozen = True


def render_example(example: FewShotExample, include_explanation: bool = True) -> str:
    """Render a single example for prompt insertion.

    Args:
        example: The example to render
        include_explanation: Whether to include the explanation note

    Returns:
        Formatted text for the example
    """
    lines = [
        f"Q: {example.question}",
        f"SQL: {example.sql}",
    ]

    if include_explanation and example.explanation:
        lines.append(f"Note: {example.explanation}")

    return "\n".join(lines)


def render_examples(
    examples: List[FewShotExample],
    include_explanations: bool = True,
) -> str:
    """Render multiple examples for prompt insertion.

    Args:
        examples: List of examples to render
        include_explanations: Whether to include explanation notes

    Returns:
        Formatted text with all examples
    """
    if not examples:
        return ""

    lines = ["Here are some example queries:\n"]

    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(render_example(ex, include_explanations))
        lines.append("")

    return "\n".join(lines)
