"""Hybrid schema store with DB-backed lookups and semantic search.

Uses pgvector for embedding-backed search over descriptive MDL
fields, and the mdl_tables DB table for on-demand structural
lookups (columns, relationships, primary keys). No in-memory
SchemaStore: everything comes from the database.
"""

from __future__ import annotations

import json

import asyncpg
from langchain_core.embeddings import Embeddings

from src.schema.models import TableCard
from src.schema.store import SchemaStore


def _format_vector(embedding: list[float]) -> str:
    """Format embedding as pgvector string literal."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


class HybridSchemaStore:
    """Schema store with hybrid (exact + semantic) retrieval.

    Uses pgvector for embedding-backed search over descriptive MDL
    fields, and the mdl_tables DB table for on-demand structural
    lookups (columns, relationships, primary keys).
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        embeddings: Embeddings,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ):
        self._pool = pool
        self._embeddings = embeddings
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

    # ── Semantic search methods ──

    async def search_tables(self, question: str, top_k: int = 5) -> list[dict]:
        """Find tables relevant to a question using hybrid search.

        Searches over table descriptions (one embedding per table).
        Replaces the exact list_tables() for large schemas.
        """
        query_vec = _format_vector(self._embeddings.embed_query(question))

        sql = f"""
            WITH semantic AS (
                SELECT schema_name, table_name,
                       ROW_NUMBER() OVER (
                           ORDER BY embedding <=> '{query_vec}'::vector
                       ) AS rank
                FROM mdl_embeddings
                WHERE field_type = 'table_description'
                LIMIT {top_k * 4}
            ),
            bm25 AS (
                SELECT schema_name, table_name,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank(
                               to_tsvector('english', content),
                               plainto_tsquery('english', $1)
                           ) DESC
                       ) AS rank
                FROM mdl_embeddings
                WHERE field_type = 'table_description'
                  AND to_tsvector('english', content)
                      @@ plainto_tsquery('english', $1)
                LIMIT {top_k * 4}
            ),
            rrf AS (
                SELECT
                    COALESCE(s.schema_name, b.schema_name) AS schema_name,
                    COALESCE(s.table_name, b.table_name) AS table_name,
                    COALESCE({self.semantic_weight} / (60 + s.rank), 0) +
                    COALESCE({self.bm25_weight} / (60 + b.rank), 0) AS score
                FROM semantic s
                FULL OUTER JOIN bm25 b
                    ON s.schema_name = b.schema_name
                   AND s.table_name = b.table_name
            )
            SELECT schema_name, table_name, score
            FROM rrf ORDER BY score DESC LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, question, top_k)

        # Enrich results with table descriptions from mdl_tables
        if not rows:
            return []
        async with self._pool.acquire() as conn:
            detail_rows = await conn.fetch(
                "SELECT schema_name, name, "
                "table_data->>'description' AS description "
                "FROM mdl_tables WHERE (schema_name, name) IN "
                "(" + ", ".join(
                    f"(${i*2+1}, ${i*2+2})"
                    for i in range(len(rows))
                ) + ")",
                *[v for row in rows
                  for v in (row["schema_name"], row["table_name"])],
            )
        detail_map = {
            (r["schema_name"], r["name"]): r["description"]
            for r in detail_rows
        }

        results = []
        for row in rows:
            key = (row["schema_name"], row["table_name"])
            if key in detail_map:
                results.append({
                    "schema_name": row["schema_name"],
                    "name": row["table_name"],
                    "description": detail_map[key],
                    "score": float(row["score"]),
                })
        return results

    async def search_business_context(
        self, question: str, top_k: int = 5
    ) -> list[dict]:
        """Find business context relevant to a question using hybrid search.

        Searches over additional descriptions, institutional knowledge,
        and dbt-enriched business rules.
        """
        query_vec = _format_vector(self._embeddings.embed_query(question))

        sql = f"""
            WITH semantic AS (
                SELECT table_name, content, field_type,
                       ROW_NUMBER() OVER (
                           ORDER BY embedding <=> '{query_vec}'::vector
                       ) AS rank
                FROM mdl_embeddings
                WHERE field_type IN (
                    'additional_description', 'institutional', 'business_rule'
                )
                LIMIT {top_k * 4}
            ),
            bm25 AS (
                SELECT table_name, content, field_type,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank(
                               to_tsvector('english', content),
                               plainto_tsquery('english', $1)
                           ) DESC
                       ) AS rank
                FROM mdl_embeddings
                WHERE field_type IN (
                    'additional_description', 'institutional', 'business_rule'
                )
                  AND to_tsvector('english', content)
                      @@ plainto_tsquery('english', $1)
                LIMIT {top_k * 4}
            ),
            rrf AS (
                SELECT
                    COALESCE(s.table_name, b.table_name) AS table_name,
                    COALESCE(s.content, b.content) AS content,
                    COALESCE(s.field_type, b.field_type) AS field_type,
                    COALESCE({self.semantic_weight} / (60 + s.rank), 0) +
                    COALESCE({self.bm25_weight} / (60 + b.rank), 0) AS score
                FROM semantic s
                FULL OUTER JOIN bm25 b
                    ON s.content = b.content
            )
            SELECT table_name, content, field_type, score
            FROM rrf ORDER BY score DESC LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, question, top_k)

        return [
            {
                "table": row["table_name"],
                "content": row["content"],
                "type": row["field_type"],
                "score": float(row["score"]),
            }
            for row in rows
        ]

    # ── Exact lookup methods ──

    async def get_table_details(
        self, schema_name: str, name: str,
    ) -> dict | None:
        """Exact lookup for table description and columns.

        Returns description and columns only. Use get_metrics() and
        get_relationships() for additional detail when needed.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT table_data FROM mdl_tables "
                "WHERE schema_name = $1 AND name = $2",
                schema_name, name,
            )
        if not row:
            return None
        data = json.loads(row["table_data"])
        return {
            "name": data.get("name"),
            "schema_name": data.get("schema_name"),
            "description": data.get("description"),
            "columns": data.get("columns", []),
        }

    async def get_metrics(
        self, schema_name: str, name: str,
    ) -> list[dict]:
        """Exact lookup for table metrics (aggregations, KPIs).

        Optional — call when the question involves aggregations,
        KPIs, or calculated fields.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT table_data FROM mdl_tables "
                "WHERE schema_name = $1 AND name = $2",
                schema_name, name,
            )
        if not row:
            return []
        data = json.loads(row["table_data"])
        return data.get("metrics", [])

    async def get_relationships(
        self, schema_name: str, name: str,
    ) -> list[dict]:
        """Exact lookup for table relationships (foreign keys, joins).

        Optional — call when the question requires joining tables.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT table_data FROM mdl_tables "
                "WHERE schema_name = $1 AND name = $2",
                schema_name, name,
            )
        if not row:
            return []
        data = json.loads(row["table_data"])
        return data.get("relationships", [])

    async def get_glossary_entries(self, terms: list[str]) -> list[dict]:
        """Exact-match lookup for glossary terms from the database.

        Unlike 2.7 which read from the JSON file, this queries the
        glossary table for centralized, cross-schema term definitions.
        """
        if not terms:
            return []
        placeholders = ", ".join(f"${i+1}" for i in range(len(terms)))
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT term, definition FROM glossary "
                f"WHERE term IN ({placeholders})",
                *[t.lower() for t in terms],
            )
        return [{"term": r["term"], "definition": r["definition"]} for r in rows]


# ── Offline index-building functions ──


async def build_mdl_index(
    store: SchemaStore,
    pool: asyncpg.Pool,
    embeddings: Embeddings,
) -> int:
    """Build the embedding index from per-schema MDL files.

    Extracts table-level descriptive fields, embeds them, stores in pgvector.
    Run offline, not at query time. Glossary and additional descriptions
    are loaded by separate functions (build_glossary_table,
    ingest_additional_descriptions).
    """
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS mdl_embeddings (
                id SERIAL PRIMARY KEY,
                schema_name TEXT,
                table_name TEXT,
                field_type TEXT,
                content TEXT,
                embedding vector(1536)
            )
        """)
        # Preserve institutional chunks and additional descriptions;
        # those are managed by separate ingestion functions.
        await conn.execute("""
            DELETE FROM mdl_embeddings
            WHERE field_type = 'table_description'
        """)

        # Also populate the mdl_tables lookup table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS mdl_tables (
                schema_name TEXT NOT NULL,
                name TEXT NOT NULL,
                table_data JSONB NOT NULL,
                PRIMARY KEY (schema_name, name)
            )
        """)
        await conn.execute("DELETE FROM mdl_tables")

        tables = list(store.get_all_tables())

        # Store full table data in mdl_tables for on-demand lookup
        for table in tables:
            await conn.execute(
                "INSERT INTO mdl_tables (schema_name, name, table_data) "
                "VALUES ($1, $2, $3::jsonb)",
                table.schema_name, table.name, table.model_dump_json(),
            )

        # Batch-embed all table descriptions in one call
        desc_texts = [f"{t.name}: {t.description}" for t in tables]
        vectors = embeddings.embed_documents(desc_texts)

        for table, desc_text, vec in zip(tables, desc_texts, vectors):
            await conn.execute(
                """INSERT INTO mdl_embeddings
                   (schema_name, table_name, field_type, content, embedding)
                   VALUES ($1, $2, $3, $4, $5::vector)""",
                table.schema_name, table.name, "table_description",
                desc_text, _format_vector(vec),
            )

    return len(tables)


async def build_glossary_table(
    glossary: dict[str, str],
    pool: asyncpg.Pool,
) -> int:
    """Load glossary terms into the database for exact-match lookup.

    Run offline alongside the embedding index build.
    """
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS glossary (
                term TEXT PRIMARY KEY,
                definition TEXT NOT NULL
            )
        """)
        await conn.execute("DELETE FROM glossary")

        for term, definition in glossary.items():
            await conn.execute(
                "INSERT INTO glossary (term, definition) VALUES ($1, $2)",
                term.lower(), definition,
            )
    return len(glossary)


async def ingest_additional_descriptions(
    descriptions: list[str],
    pool: asyncpg.Pool,
    embeddings: Embeddings,
) -> int:
    """Embed additional descriptions into the MDL index.

    These are global business context entries (fiscal calendar,
    time conventions, etc.) stored alongside institutional knowledge.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM mdl_embeddings WHERE field_type = 'additional_description'"
        )
        vectors = embeddings.embed_documents(descriptions)
        for desc, vec in zip(descriptions, vectors):
            await conn.execute(
                """INSERT INTO mdl_embeddings
                   (table_name, field_type, content, embedding)
                   VALUES ($1, $2, $3, $4::vector)""",
                None, "additional_description", desc,
                _format_vector(vec),
            )
    return len(descriptions)


async def ingest_institutional_knowledge(
    text: str,
    pool: asyncpg.Pool,
    embeddings: Embeddings,
) -> int:
    """Chunk and embed institutional knowledge into the MDL index.

    Uses RecursiveCharacterTextSplitter (paragraph-aware) to split
    long documents into chunks, then embeds each chunk into pgvector.
    Chunks land in mdl_embeddings with field_type='institutional'.

    Args:
        text: The full institutional knowledge document text.
        pool: asyncpg connection pool.
        embeddings: LangChain Embeddings instance.

    Returns:
        Number of chunks embedded.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_text(text)

    async with pool.acquire() as conn:
        # Clear previous institutional chunks
        await conn.execute(
            "DELETE FROM mdl_embeddings WHERE field_type = 'institutional'"
        )
        vectors = embeddings.embed_documents(chunks)
        for chunk, vec in zip(chunks, vectors):
            await conn.execute(
                """INSERT INTO mdl_embeddings
                   (table_name, field_type, content, embedding)
                   VALUES ($1, $2, $3, $4::vector)""",
                None, "institutional", chunk,
                _format_vector(vec),
            )
    return len(chunks)
