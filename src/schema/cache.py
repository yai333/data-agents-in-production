"""SQL caching module for Text-to-SQL systems.

Implements question normalization and SQL result caching with schema versioning.
Cache keys are based on normalized questions + context, allowing multi-tenant
deployments while invalidating stale queries after schema changes.
"""

from __future__ import annotations

import re
from datetime import datetime

import asyncpg


def normalize_question(question: str) -> str:
    """Normalize a natural language question for cache lookup.

    Converts to lowercase, strips punctuation (keeping alphanumeric + spaces),
    collapses whitespace, and strips leading/trailing space.

    Args:
        question: Raw user question

    Returns:
        Normalized question string
    """
    # Lowercase
    normalized = question.lower()
    # Keep only alphanumeric and spaces
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    # Collapse multiple spaces to single space
    normalized = re.sub(r"\s+", " ", normalized)
    # Strip leading/trailing space
    return normalized.strip()


class SQLCache:
    """SQL query cache with schema versioning and context isolation.

    Stores normalized question -> SQL mappings with hit tracking and
    automatic invalidation on schema changes.
    """

    def __init__(self, pool: asyncpg.Pool, schema_version: str = "v1"):
        """Initialize SQL cache.

        Args:
            pool: asyncpg connection pool
            schema_version: Current schema version (e.g., "v1", "2024-01")
        """
        self.pool = pool
        self.schema_version = schema_version

    async def lookup(
        self, question: str, context_key: str = "default"
    ) -> dict | None:
        """Look up cached SQL for a question.

        Args:
            question: User's natural language question
            context_key: Context identifier (tenant, department, etc.)

        Returns:
            Dict with sql, tables_used, hit_count if found, else None
        """
        normalized = normalize_question(question)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT sql, tables_used, hit_count
                FROM sql_cache
                WHERE context_key = $1
                  AND normalized_question = $2
                  AND schema_version = $3
                """,
                context_key,
                normalized,
                self.schema_version,
            )

            if row:
                # Increment hit count
                await conn.execute(
                    """
                    UPDATE sql_cache
                    SET hit_count = hit_count + 1
                    WHERE context_key = $1
                      AND normalized_question = $2
                    """,
                    context_key,
                    normalized,
                )

                return {
                    "sql": row["sql"],
                    "tables_used": list(row["tables_used"]),
                    "hit_count": row["hit_count"] + 1,
                }

            return None

    async def store(
        self,
        question: str,
        sql: str,
        tables_used: list[str],
        context_key: str = "default",
    ) -> None:
        """Store SQL in cache for a question.

        Args:
            question: User's natural language question
            sql: Generated SQL query
            tables_used: List of table names referenced
            context_key: Context identifier (tenant, department, etc.)
        """
        normalized = normalize_question(question)

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sql_cache (
                    context_key,
                    normalized_question,
                    sql,
                    tables_used,
                    schema_version
                )
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (context_key, normalized_question)
                DO UPDATE SET
                    sql = EXCLUDED.sql,
                    tables_used = EXCLUDED.tables_used,
                    schema_version = EXCLUDED.schema_version,
                    created_at = NOW(),
                    hit_count = 0
                """,
                context_key,
                normalized,
                sql,
                tables_used,
                self.schema_version,
            )

    async def invalidate(self, schema_version: str | None = None) -> int:
        """Invalidate cache entries not matching schema version.

        Args:
            schema_version: Version to keep (default: current version).
                           If None, clears entire cache.

        Returns:
            Number of entries deleted
        """
        async with self.pool.acquire() as conn:
            if schema_version is None:
                # Clear entire cache
                result = await conn.execute("DELETE FROM sql_cache")
            else:
                # Delete entries not matching version
                result = await conn.execute(
                    """
                    DELETE FROM sql_cache
                    WHERE schema_version != $1
                    """,
                    schema_version,
                )

            # Extract count from "DELETE N"
            return int(result.split()[-1])

    async def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with total_entries, total_hits, and top_questions list
        """
        async with self.pool.acquire() as conn:
            # Total entries and hits
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) as total_entries,
                       COALESCE(SUM(hit_count), 0) as total_hits
                FROM sql_cache
                WHERE schema_version = $1
                """,
                self.schema_version,
            )

            total_entries = row["total_entries"]
            total_hits = row["total_hits"]

            # Top questions
            rows = await conn.fetch(
                """
                SELECT normalized_question, hit_count
                FROM sql_cache
                WHERE schema_version = $1
                ORDER BY hit_count DESC
                LIMIT 10
                """,
                self.schema_version,
            )

            top_questions = [
                {"question": row["normalized_question"], "hits": row["hit_count"]}
                for row in rows
            ]

            return {
                "total_entries": total_entries,
                "total_hits": total_hits,
                "top_questions": top_questions,
            }


async def ensure_cache_table(pool: asyncpg.Pool) -> None:
    """Ensure sql_cache table exists in the database.

    Args:
        pool: asyncpg connection pool
    """
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sql_cache (
                context_key          TEXT NOT NULL,
                normalized_question  TEXT NOT NULL,
                sql                  TEXT NOT NULL,
                tables_used          TEXT[] NOT NULL,
                created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                hit_count            INTEGER NOT NULL DEFAULT 0,
                schema_version       TEXT NOT NULL,
                PRIMARY KEY (context_key, normalized_question)
            )
            """
        )
