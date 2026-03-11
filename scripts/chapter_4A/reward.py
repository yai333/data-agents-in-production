"""Reward function for APO optimization.

Combines three evaluation signals (like the Agent Lightning room_selector
example, but adapted for SQL):

  1. Spider-style component matching (evals/sql_components.py): per-clause F1
  2. Execution accuracy: run both queries, compare results
  3. LLM-as-judge: structured grading of semantic equivalence

Reward = 0.5 * component_f1 + 0.5 * judge_score  (exec_match disabled)

Safety check is a hard gate: dangerous SQL gets reward 0.0.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pydantic import BaseModel, Field

from evals.sql_components import compare_sql_components
from evals.runner import compare_results


@dataclass
class RewardBreakdown:
    """Detailed breakdown of the reward signal."""
    total: float
    component_f1: float
    execution_match: float
    judge_score: float
    generated_sql: str
    expected_sql: str


# ---------------------------------------------------------------------------
# Module-level shared resources (lazy-initialized on first use)
# ---------------------------------------------------------------------------

_judge_client = None  # openai.OpenAI instance for LLM-as-judge
_engine = None        # sqlalchemy.Engine for execution eval


def _get_judge_client():
    """Lazy-init the OpenAI client for LLM-as-judge."""
    global _judge_client
    if _judge_client is None:
        from openai import OpenAI
        _judge_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )
    return _judge_client


def _get_engine():
    """Lazy-init the SQLAlchemy engine for execution eval."""
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine
        db_url = os.getenv(
            "CHINOOK_DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/chinook",
        )
        _engine = create_engine(db_url)
    return _engine


# ---------------------------------------------------------------------------
# LLM-as-judge (follows the Agent Lightning room_selector pattern)
# ---------------------------------------------------------------------------

class JudgeResponse(BaseModel):
    """Structured output for the LLM-as-judge grader."""
    reason: str = Field(description="Brief reasoning for the score")
    score: float = Field(ge=0.0, le=1.0, description="Quality score 0-1")


JUDGE_PROMPT = """You are a strict SQL grader. Compare the generated SQL against the expected SQL.

Assess whether the generated query would produce the same results as the expected query.

Expected SQL:
{expected_sql}

Generated SQL:
{generated_sql}

Score on a 0-1 scale:
- 1.0: Semantically equivalent (may differ in style but produces same results)
- 0.7-0.9: Mostly correct with minor differences (e.g., extra columns, slight filter variation)
- 0.3-0.6: Partially correct (some right elements, some wrong)
- 0.0-0.2: Incorrect or unrelated"""


DANGEROUS_KEYWORDS = {"DROP", "DELETE", "TRUNCATE", "ALTER", "INSERT", "UPDATE"}


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_sql_reward(generated_sql: str, expected_sql: str) -> float:
    """Compute combined reward. Returns a float in [0.0, 1.0]."""
    return compute_sql_reward_detailed(generated_sql, expected_sql).total


def compute_sql_reward_detailed(
    generated_sql: str, expected_sql: str,
) -> RewardBreakdown:
    """Compute combined reward with per-signal breakdown."""
    gen_sql = _extract_sql(generated_sql)
    exp_sql = _extract_sql(expected_sql)

    if not gen_sql or not _passes_safety(gen_sql):
        return RewardBreakdown(
            total=0.0, component_f1=0.0, execution_match=0.0,
            judge_score=0.0, generated_sql=gen_sql, expected_sql=exp_sql,
        )

    # 1. Spider-style component matching (0.3 weight)
    try:
        component_result = compare_sql_components(gen_sql, exp_sql)
        component_score = component_result.overall_f1
    except Exception:
        component_score = 0.0

    # 2. Execution accuracy (commented out — requires DB connection in runner process)
    # exec_score = _execution_match(gen_sql, exp_sql)
    exec_score = 0.0

    # 3. LLM-as-judge (0.5 weight with exec disabled, was 0.3)
    judge_score = _llm_judge(gen_sql, exp_sql)

    # With exec disabled: 0.5 * component + 0.5 * judge
    reward = min(0.5 * component_score + 0.5 * judge_score, 1.0)
    return RewardBreakdown(
        total=reward,
        component_f1=component_score,
        execution_match=exec_score,
        judge_score=judge_score,
        generated_sql=gen_sql,
        expected_sql=exp_sql,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_sql(text: str | None) -> str:
    """Extract SQL from markdown code blocks or raw text."""
    if not text:
        return ""
    if "```sql" in text:
        start = text.find("```sql") + 6
        end = text.find("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        return text[start:end].strip()
    return text.strip()


def _llm_judge(gen_sql: str, exp_sql: str) -> float:
    """Use LLM-as-judge to assess semantic equivalence."""
    try:
        client = _get_judge_client()
        prompt = JUDGE_PROMPT.format(expected_sql=exp_sql, generated_sql=gen_sql)
        response = client.chat.completions.parse(
            model=os.getenv("APO_JUDGE_MODEL", "gpt-4.1-mini"),
            messages=[{"role": "user", "content": prompt}],
            response_format=JudgeResponse,
            temperature=0.0,
        )
        result = response.choices[0].message.parsed
        return result.score if result else 0.0
    except Exception:
        return 0.0


def _execution_match(gen_sql: str, exp_sql: str) -> float:
    """Execute both queries via SQLAlchemy and compare results."""
    try:
        engine = _get_engine()
        from sqlalchemy import text

        with engine.connect() as conn:
            gen_result = [
                dict(row._mapping) for row in conn.execute(text(gen_sql))
            ]
            exp_result = [
                dict(row._mapping) for row in conn.execute(text(exp_sql))
            ]

        if compare_results(gen_result, exp_result):
            return 1.0

        # Partial credit: set overlap
        gen_set = {
            tuple(sorted(str(v) for v in r.values())) for r in gen_result
        }
        exp_set = {
            tuple(sorted(str(v) for v in r.values())) for r in exp_result
        }
        if exp_set:
            return len(gen_set & exp_set) / len(exp_set) * 0.8
        return 0.0
    except Exception:
        return 0.0


def _passes_safety(sql: str) -> bool:
    """Check for dangerous SQL keywords."""
    tokens = sql.upper().split()
    return not any(t in DANGEROUS_KEYWORDS for t in tokens)
