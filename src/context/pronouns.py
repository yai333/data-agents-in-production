"""Resolve pronouns in multi-turn conversations.

This module handles pronoun resolution ("their", "it", "those") using
conversation context.

See 1.7 for pronoun resolution patterns.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.context.session import ConversationContext


@dataclass
class PronounResolution:
    """Result of pronoun resolution."""

    original: str
    resolved: str
    pronoun: str
    referent: str


# Pronouns and their types
PRONOUNS = {
    "their": "possessive",
    "them": "object",
    "they": "subject",
    "it": "thing",
    "its": "possessive_thing",
    "those": "plural_thing",
    "these": "plural_thing",
    "that": "singular_thing",
    "this": "singular_thing",
}


def resolve_pronouns(
    question: str,
    context: "ConversationContext",
) -> tuple[str, list[PronounResolution]]:
    """Resolve pronouns using conversation context.

    Args:
        question: Current question with pronouns
        context: Conversation context

    Returns:
        Tuple of (resolved question, list of resolutions)

    Example:
        >>> context.resolved_entities = {"customers": "customer"}
        >>> resolved, _ = resolve_pronouns("show me their orders", context)
        >>> resolved
        'show me customer orders'
    """
    resolutions = []
    resolved_question = question

    for pronoun, pronoun_type in PRONOUNS.items():
        # Check if pronoun exists in question
        pattern = rf"\b{pronoun}\b"
        if not re.search(pattern, question, re.IGNORECASE):
            continue

        # Find the referent
        referent = _find_referent(pronoun_type, context)

        if referent:
            # Replace pronoun with referent
            resolved_question = re.sub(
                pattern,
                referent,
                resolved_question,
                count=1,  # Only replace first occurrence
                flags=re.IGNORECASE,
            )

            resolutions.append(
                PronounResolution(
                    original=question,
                    resolved=resolved_question,
                    pronoun=pronoun,
                    referent=referent,
                )
            )

    return resolved_question, resolutions


def _find_referent(
    pronoun_type: str,
    context: "ConversationContext",
) -> str | None:
    """Find the referent for a pronoun type.

    Args:
        pronoun_type: Type of pronoun (possessive, object, etc.)
        context: Conversation context

    Returns:
        Referent string or None
    """
    if not context.turns:
        return None

    # For people pronouns (their, them, they), look for entity mentions
    if pronoun_type in ("possessive", "object", "subject"):
        # Check recent turns for entities
        for turn in reversed(context.turns[-3:]):
            if turn.entities_mentioned:
                return turn.entities_mentioned[-1]

        # Fall back to current tables that look like entities
        entity_tables = ["customer", "employee", "user", "artist", "author"]
        for table in context.current_tables:
            if table in entity_tables:
                return table

    # For thing pronouns (it, those, this), look for tables or results
    elif pronoun_type in ("thing", "possessive_thing", "singular_thing", "plural_thing"):
        # Return most recent table
        if context.current_tables:
            return context.current_tables[-1]

    return None


def needs_resolution(question: str) -> bool:
    """Check if a question contains pronouns that need resolution.

    Args:
        question: The question to check

    Returns:
        True if pronouns are present
    """
    question_lower = question.lower()
    return any(
        re.search(rf"\b{pronoun}\b", question_lower)
        for pronoun in PRONOUNS
    )
