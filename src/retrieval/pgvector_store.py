"""pgvector-backed retrieval for production use.

This module provides database-backed retrieval instead of in-memory.
Benefits:
- Embeddings persist across restarts (no recomputation)
- Scales to large example libraries
- Enables hybrid search via PostgreSQL full-text + pgvector
- Supports filtering by category, difficulty, tables

Usage:
    # First, index examples:
    # python scripts/index_fewshot_examples.py

    # Then use the retriever:
    retriever = await PgVectorRetriever.create(database_url, provider="openai")
    examples = await retriever.retrieve("How many customers?", top_k=5)
"""

from __future__ import annotations

from typing import Optional

import asyncpg
from langchain_core.embeddings import Embeddings

from src.retrieval.models import FewShotExample
from src.retrieval.rrf import rrf_with_weights


def create_embeddings(provider: str) -> Embeddings:
    """Create a LangChain Embeddings instance for the specified provider.

    Args:
        provider: One of 'openai', 'gemini', or 'local'

    Returns:
        LangChain Embeddings instance with embed_query() and embed_documents()
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    else:  # local / sentence-transformers
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _format_vector(embedding: list[float]) -> str:
    """Format embedding as pgvector string literal."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


class PgVectorRetriever:
    """Retrieves few-shot examples from pgvector using hybrid search.

    Combines:
    - Semantic search (pgvector cosine similarity)
    - BM25 full-text search (pg_textsearch extension)
    - RRF fusion of both rankings
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        embeddings: Embeddings,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ):
        """Initialize with a connection pool.

        Use PgVectorRetriever.create() instead of calling this directly.

        Args:
            pool: asyncpg connection pool
            embeddings: LangChain Embeddings instance
            bm25_weight: Weight for full-text search in RRF
            semantic_weight: Weight for semantic search in RRF
        """
        self._pool = pool
        self._embeddings = embeddings
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

    @classmethod
    async def create(
        cls,
        database_url: str,
        provider: str = "openai",
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        min_pool_size: int = 1,
        max_pool_size: int = 5,
    ) -> "PgVectorRetriever":
        """Create a retriever with a connection pool.

        Args:
            database_url: PostgreSQL connection URL
            provider: Embedding provider ('openai', 'gemini', or 'local')
            bm25_weight: Weight for full-text search
            semantic_weight: Weight for semantic search
            min_pool_size: Minimum connections in pool
            max_pool_size: Maximum connections in pool

        Returns:
            Configured PgVectorRetriever instance
        """
        pool = await asyncpg.create_pool(
            database_url,
            min_size=min_pool_size,
            max_size=max_pool_size,
        )
        embeddings = create_embeddings(provider)
        return cls(pool, embeddings, bm25_weight, semantic_weight)

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()

    def _embed_query(self, query: str) -> str:
        """Embed a query string and return as pgvector literal."""
        return _format_vector(self._embeddings.embed_query(query))

    async def retrieve_semantic(
        self,
        query: str,
        top_k: int = 20,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        required_tables: Optional[list[str]] = None,
    ) -> list[tuple[str, float]]:
        """Semantic-only retrieval via pgvector.

        Args:
            query: Search query
            top_k: Number of results
            category: Filter by category (optional)
            difficulty: Filter by difficulty (optional)
            required_tables: Filter to examples using these tables (optional)

        Returns:
            List of (example_id, similarity_score) tuples
        """
        # Embed query and format as pgvector literal
        query_embedding_str = self._embed_query(query)

        # Build query with optional filters
        where_clauses = []
        params = [top_k]  # Only top_k as parameter
        param_idx = 2

        if category:
            where_clauses.append(f"category = ${param_idx}")
            params.append(category)
            param_idx += 1

        if difficulty:
            where_clauses.append(f"difficulty = ${param_idx}")
            params.append(difficulty)
            param_idx += 1

        if required_tables:
            where_clauses.append(f"tables_used && ${param_idx}")
            params.append(required_tables)
            param_idx += 1

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # Use string interpolation for vector literal (asyncpg doesn't support pgvector natively)
        sql = f"""
            SELECT id,
                   1 - (question_embedding <=> '{query_embedding_str}'::vector) as similarity
            FROM fewshot_examples
            {where_sql}
            ORDER BY question_embedding <=> '{query_embedding_str}'::vector
            LIMIT $1
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [(row["id"], row["similarity"]) for row in rows]

    async def retrieve_bm25(
        self,
        query: str,
        top_k: int = 20,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        required_tables: Optional[list[str]] = None,
    ) -> list[tuple[str, float]]:
        """BM25 full-text search retrieval via pg_textsearch.

        Args:
            query: Search query
            top_k: Number of results
            category: Filter by category (optional)
            difficulty: Filter by difficulty (optional)
            required_tables: Filter to examples using these tables (optional)

        Returns:
            List of (example_id, bm25_score) tuples. Scores are negative; lower = better.
        """
        # Build query with optional filters
        # BM25 scores are negative; < 0 means there's a match
        where_clauses = ["question <@> $1 < 0"]
        params = [query, top_k]
        param_idx = 3

        if category:
            where_clauses.append(f"category = ${param_idx}")
            params.append(category)
            param_idx += 1

        if difficulty:
            where_clauses.append(f"difficulty = ${param_idx}")
            params.append(difficulty)
            param_idx += 1

        if required_tables:
            where_clauses.append(f"tables_used && ${param_idx}")
            params.append(required_tables)
            param_idx += 1

        where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
            SELECT id,
                   question <@> $1 as bm25_score
            FROM fewshot_examples
            {where_sql}
            ORDER BY bm25_score  -- Lower (more negative) = better match
            LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [(row["id"], row["bm25_score"]) for row in rows]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        required_tables: Optional[list[str]] = None,
    ) -> list[FewShotExample]:
        """Hybrid retrieval combining semantic and BM25 search.

        Uses RRF fusion in a single SQL query with CTEs for efficiency.
        This approach:
        - Single database round-trip (vs 3 in naive implementation)
        - RRF computed in SQL using ROW_NUMBER() and FULL OUTER JOIN
        - Filters applied once, not duplicated

        Args:
            query: Natural language question
            top_k: Number of examples to return
            category: Filter by category (optional)
            difficulty: Filter by difficulty (optional)
            required_tables: Filter to examples using these tables (optional)

        Returns:
            List of most relevant examples
        """
        # Embed query
        query_embedding_str = self._embed_query(query)

        # Build filter clauses
        filters = []
        params = [top_k]
        param_idx = 2

        if category:
            filters.append(f"category = ${param_idx}")
            params.append(category)
            param_idx += 1

        if difficulty:
            filters.append(f"difficulty = ${param_idx}")
            params.append(difficulty)
            param_idx += 1

        if required_tables:
            filters.append(f"tables_used && ${param_idx}")
            params.append(required_tables)
            param_idx += 1

        filter_sql = ""
        if filters:
            filter_sql = "WHERE " + " AND ".join(filters)

        # BM25 filter: scores are negative, < 0 means match
        bm25_filter = f"question <@> ${param_idx} < 0"
        params.append(query)

        # Single query with RRF fusion in SQL
        # k=60 is the standard RRF constant
        sql = f"""
            WITH semantic AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           ORDER BY question_embedding <=> '{query_embedding_str}'::vector
                       ) as rank
                FROM fewshot_examples
                {filter_sql}
                LIMIT 50
            ),
            bm25 AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           ORDER BY question <@> ${param_idx}
                       ) as rank
                FROM fewshot_examples
                {"WHERE " + " AND ".join(filters + [bm25_filter]) if filters else "WHERE " + bm25_filter}
                LIMIT 50
            ),
            rrf AS (
                SELECT
                    COALESCE(s.id, b.id) as id,
                    COALESCE({self.semantic_weight}::float / (60 + s.rank), 0) +
                    COALESCE({self.bm25_weight}::float / (60 + b.rank), 0) as score
                FROM semantic s
                FULL OUTER JOIN bm25 b ON s.id = b.id
            )
            SELECT e.id, e.question, e.sql, e.explanation,
                   e.tables_used, e.category, e.difficulty,
                   rrf.score
            FROM rrf
            JOIN fewshot_examples e ON e.id = rrf.id
            ORDER BY rrf.score DESC
            LIMIT $1
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [
            FewShotExample(
                id=row["id"],
                question=row["question"],
                sql=row["sql"],
                explanation=row["explanation"],
                tables_used=list(row["tables_used"]),
                category=row["category"],
                difficulty=row["difficulty"],
            )
            for row in rows
        ]

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> list[tuple[FewShotExample, float]]:
        """Retrieve examples with their RRF fusion scores.

        Useful for debugging or confidence thresholding.

        Args:
            query: Natural language question
            top_k: Number of examples to return
            category: Filter by category (optional)
            difficulty: Filter by difficulty (optional)

        Returns:
            List of (example, rrf_score) tuples
        """
        candidate_count = min(top_k * 4, 50)

        semantic_results = await self.retrieve_semantic(
            query, candidate_count, category, difficulty
        )
        bm25_results = await self.retrieve_bm25(
            query, candidate_count, category, difficulty
        )

        semantic_ranking = [id for id, _ in semantic_results]
        bm25_ranking = [id for id, _ in bm25_results]

        rankings = [bm25_ranking, semantic_ranking]
        weights = [self.bm25_weight, self.semantic_weight]

        fused = rrf_with_weights(rankings, weights)

        top_results = fused[:top_k]
        top_ids = [doc_id for doc_id, _ in top_results]
        score_by_id = {doc_id: score for doc_id, score in top_results}

        if not top_ids:
            return []

        # Fetch examples
        placeholders = ", ".join(f"${i+1}" for i in range(len(top_ids)))
        sql = f"""
            SELECT id, question, sql, explanation, tables_used, category, difficulty
            FROM fewshot_examples
            WHERE id IN ({placeholders})
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *top_ids)

        examples_by_id = {}
        for row in rows:
            examples_by_id[row["id"]] = FewShotExample(
                id=row["id"],
                question=row["question"],
                sql=row["sql"],
                explanation=row["explanation"],
                tables_used=list(row["tables_used"]),
                category=row["category"],
                difficulty=row["difficulty"],
            )

        return [
            (examples_by_id[id], score_by_id[id])
            for id in top_ids
            if id in examples_by_id
        ]

    async def get_example_count(self) -> int:
        """Get the total number of indexed examples."""
        async with self._pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM fewshot_examples")

    async def get_categories(self) -> list[str]:
        """Get all unique categories."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT category FROM fewshot_examples ORDER BY category"
            )
        return [row["category"] for row in rows]
