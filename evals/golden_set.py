"""Golden set data models for Text-to-SQL evaluation.

A golden set is your ground truth: queries with known-correct SQL and results.
This module defines the structure for test cases.
"""

from typing import Any
from pydantic import BaseModel, Field


class GoldenQuery(BaseModel):
    """A single test case in the golden set.

    Each golden query represents a natural language question paired with
    its reference SQL answer and expected result. Categories and difficulty
    levels enable slice analysis to identify specific weaknesses.

    Attributes:
        id: Unique identifier for tracking and reporting
        question: Natural language question from the user
        sql: Reference SQL (one correct answer, though others may be equivalent)
        expected_result: Expected query result for validation
        category: Query type for slice analysis (count, join, group, etc.)
        difficulty: Complexity level (easy, medium, hard)
        tables_used: Tables involved, for retrieval evaluation
    """

    id: str = Field(..., description="Unique identifier for the query")
    question: str = Field(..., description="Natural language question")
    sql: str = Field(..., description="Reference SQL query")
    expected_result: Any = Field(..., description="Expected result from execution")
    category: str = Field(..., description="Query category for slice analysis")
    difficulty: str = Field(..., description="Difficulty: easy, medium, hard")
    tables_used: list[str] = Field(
        default_factory=list,
        description="Tables used in the query, for retrieval eval"
    )

    class Config:
        """Pydantic configuration."""
        frozen = True  # Immutable after creation
