"""Multi-turn conversation context.

This module maintains state across conversation turns, tracking
entities, tables, and filters for context-aware SQL generation.

See 1.7 for multi-turn context patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Turn:
    """A single turn in the conversation."""

    timestamp: datetime
    question: str
    sql: str | None
    result_summary: str | None
    entities_mentioned: list[str] = field(default_factory=list)


@dataclass
class ConversationContext:
    """Context maintained across conversation turns."""

    turns: list[Turn] = field(default_factory=list)
    current_tables: list[str] = field(default_factory=list)
    current_filters: dict[str, Any] = field(default_factory=dict)
    resolved_entities: dict[str, str] = field(default_factory=dict)

    def add_turn(self, turn: Turn):
        """Add a turn and update context.

        Args:
            turn: The turn to add
        """
        self.turns.append(turn)

        # Update current tables from SQL
        if turn.sql:
            self._extract_tables(turn.sql)

        # Track mentioned entities for pronoun resolution
        for entity in turn.entities_mentioned:
            self.resolved_entities[entity] = entity

    def _extract_tables(self, sql: str):
        """Extract tables from SQL for context."""
        try:
            import sqlglot

            parsed = sqlglot.parse_one(sql)
            self.current_tables = [
                table.name.lower()
                for table in parsed.find_all(sqlglot.exp.Table)
            ]
        except Exception:
            pass

    def get_recent_context(self, max_turns: int = 3) -> str:
        """Get recent context for the LLM.

        Args:
            max_turns: Maximum recent turns to include

        Returns:
            Formatted context string
        """
        if not self.turns:
            return ""

        recent = self.turns[-max_turns:]
        context_parts = []

        for i, turn in enumerate(recent, 1):
            context_parts.append(f"Turn {i}: {turn.question}")
            if turn.result_summary:
                context_parts.append(f"  Result: {turn.result_summary}")

        return "\n".join(context_parts)

    def should_reset(self, new_question: str) -> bool:
        """Determine if context should be reset.

        Reset when:
        - User explicitly asks to start over
        - Too many turns have passed
        - Topic completely changes

        Args:
            new_question: The new question

        Returns:
            True if context should be reset
        """
        reset_phrases = [
            "start over",
            "new question",
            "forget that",
            "never mind",
            "different topic",
            "something else",
        ]

        question_lower = new_question.lower()
        if any(phrase in question_lower for phrase in reset_phrases):
            return True

        # Context too old
        if len(self.turns) > 10:
            return True

        return False

    def get_last_entity(self) -> str | None:
        """Get the most recently mentioned entity.

        Returns:
            Entity name or None
        """
        for turn in reversed(self.turns):
            if turn.entities_mentioned:
                return turn.entities_mentioned[-1]
        return None

    def clear(self):
        """Clear all context."""
        self.turns = []
        self.current_tables = []
        self.current_filters = {}
        self.resolved_entities = {}
