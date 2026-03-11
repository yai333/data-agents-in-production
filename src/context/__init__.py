"""Multi-turn conversation context management.

This module manages context across conversation turns, including
entity tracking and pronoun resolution.

See 1.7 for multi-turn context patterns.
"""

from src.context.session import ConversationContext, Turn
from src.context.pronouns import resolve_pronouns, PronounResolution

__all__ = [
    "ConversationContext",
    "Turn",
    "resolve_pronouns",
    "PronounResolution",
]
