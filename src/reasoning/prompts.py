"""Four-phase reasoning prompts for SQL generation.

This module provides prompt templates that enforce structured thinking
through explicit reasoning phases. The four phases are:
1. Reasoning: Understand what the user wants
2. Analysis: Map to schema (tables, joins, filters)
3. Query: Generate the SQL with explanation
4. Verification: Confirm the query answers the question

See 1.5 for the principles behind structured reasoning.
"""

# Four-phase reasoning prompt template
REASONING_PROMPT = """You are a SQL expert. Use structured thinking to generate accurate queries.

DATABASE SCHEMA:
{schema}

{examples}

QUESTION: {question}

Think through this step by step using the following structure:

<reasoning>
Understand what the user is asking. Identify:
- What data they want to see
- Any implicit requirements (ordering, limits, time periods)
- Ambiguous terms that need interpretation
</reasoning>

<analysis>
Map the question to the database:
- Which tables contain the needed data?
- What joins connect these tables?
- What filters (WHERE conditions) are needed?
- What aggregations (GROUP BY, SUM, COUNT) apply?
</analysis>

<query>
Write the SQL query. Explain any non-obvious choices.
</query>

<verification>
Verify the query answers the original question:
- Does it return the right columns?
- Are the joins correct?
- Are filters applied correctly?
- Will the results be in the expected format?
</verification>

After verification, provide your final SQL query."""


# Direct generation prompt (no reasoning)
DIRECT_PROMPT = """Generate a SQL query for this question.
Return only the SQL, no explanation.

DATABASE SCHEMA:
{schema}

{examples}

QUESTION: {question}

SQL:"""


# Simple CoT prompt (think step by step)
COT_PROMPT = """Generate a SQL query for this question.

DATABASE SCHEMA:
{schema}

{examples}

QUESTION: {question}

Think step by step:
1. What tables are needed?
2. What columns to select?
3. Any joins required?
4. Any filters or aggregations?

Then write the SQL query."""


# Recovery prompt for low-confidence results
RECOVERY_PROMPT = """The previous attempt had low confidence. Let me reconsider.

Previous analysis:
{previous_analysis}

Previous SQL (potentially incorrect):
{previous_sql}

Please reconsider the question and generate a more accurate query.
Focus on verifying:
1. All required tables are included
2. Join conditions are correct
3. Aggregations match the question

DATABASE SCHEMA:
{schema}

{examples}

QUESTION: {question}

Use the four-phase structure for your response."""


# System prompts for different reasoning modes
SYSTEM_PROMPTS = {
    "direct": "You are a SQL expert. Generate precise, efficient queries.",
    "cot": "You are a SQL expert. Think through queries step by step.",
    "reasoning": """You are a SQL expert using structured reasoning.
Follow the four-phase pattern: reasoning, analysis, query, verification.
Use XML tags to structure your response.""",
    "agentic": """You are a SQL expert with access to tools.
You can use tools during reasoning to verify schema details.
Always verify your assumptions before generating the final query.""",
}
