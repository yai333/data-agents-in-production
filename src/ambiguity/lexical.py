"""Detect lexical ambiguity in questions.

Lexical ambiguity occurs when the same word can mean different things
in the context of a database query.

See 1.7 for ambiguity types.
"""

# Terms that commonly have multiple meanings in SQL contexts
AMBIGUOUS_TERMS = {
    "orders": ["order table", "purchase transactions"],
    "sales": ["revenue amount", "transaction count"],
    "active": ["recent activity", "subscription status", "not deleted"],
    "users": ["user table", "customers", "employees"],
    "top": ["highest value", "most frequent", "most recent"],
    "best": ["highest revenue", "highest rating", "most sold"],
    "recent": ["last few days", "last week", "last month"],
    "new": ["recently created", "not existing before"],
}


def detect_lexical_ambiguity(
    question: str,
    schema_tables: set[str],
) -> list[str]:
    """Detect potentially ambiguous terms in a question.

    Args:
        question: User's natural language question
        schema_tables: Set of table names in the schema

    Returns:
        List of ambiguous terms found

    Example:
        >>> detect_lexical_ambiguity("show top orders", {"order", "customer"})
        ['top', 'orders']
    """
    question_lower = question.lower()
    ambiguous = []

    for term in AMBIGUOUS_TERMS:
        if term in question_lower:
            # Check if it could be a table name (more ambiguous)
            if term in schema_tables or f"{term}s" in schema_tables:
                ambiguous.append(term)
            # Check if context clarifies the meaning
            elif _is_context_clear(term, question_lower):
                continue
            else:
                ambiguous.append(term)

    return ambiguous


def _is_context_clear(term: str, question: str) -> bool:
    """Check if context clarifies an ambiguous term."""
    # "top 10" is clear (it's a limit)
    if term == "top" and any(f"top {i}" in question for i in range(1, 101)):
        return True

    # "sales revenue" is clear (it's the amount)
    if term == "sales" and "revenue" in question:
        return True

    # "active users" with "last" is clearer
    if term == "active" and "last" in question:
        return True

    return False
