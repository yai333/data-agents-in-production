#!/usr/bin/env python3
"""Index few-shot examples into pgvector for production retrieval.

This script:
1. Loads all few-shot examples from chinook_examples.py
2. Computes embeddings using the configured LLM provider's embedding model
3. Stores examples + embeddings in the fewshot_examples table

The pgvector table enables:
- Fast semantic similarity search via vector index
- Full-text search for BM25-style keyword matching
- Hybrid retrieval combining both methods

Usage:
    make db-up  # Start PostgreSQL with pgvector
    python scripts/index_fewshot_examples.py

    # With specific provider
    python scripts/index_fewshot_examples.py --provider openai
    python scripts/index_fewshot_examples.py --provider gemini

    # Verify
    make db-shell
    SELECT id, question, array_length(question_embedding::float[], 1) FROM fewshot_examples LIMIT 5;
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import asyncpg

from src.retrieval.chinook_examples import CHINOOK_EXAMPLES
from src.utils.config import load_config


# Embedding dimensions by provider/model
EMBEDDING_DIMS = {
    "openai": 1536,      # text-embedding-3-small
    "gemini": 768,       # text-embedding-004
    "local": 384,        # all-MiniLM-L6-v2
}


def get_embedder(provider: str):
    """Create an embedder for the specified provider.

    Args:
        provider: One of 'openai', 'gemini', or 'local'

    Returns:
        Tuple of (embed_function, dimension)
    """
    if provider == "openai":
        import openai
        client = openai.OpenAI()

        def embed_openai(texts: list[str]) -> list[list[float]]:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [item.embedding for item in response.data]

        return embed_openai, EMBEDDING_DIMS["openai"]

    elif provider == "gemini":
        import google.generativeai as genai

        def embed_gemini(texts: list[str]) -> list[list[float]]:
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                )
                embeddings.append(result["embedding"])
            return embeddings

        return embed_gemini, EMBEDDING_DIMS["gemini"]

    else:  # local / sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required for local embedding: "
                "pip install sentence-transformers"
            )

        model = SentenceTransformer("all-MiniLM-L6-v2")

        def embed_local(texts: list[str]) -> list[list[float]]:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            return [e.tolist() for e in embeddings]

        return embed_local, EMBEDDING_DIMS["local"]


async def ensure_schema(conn: asyncpg.Connection, dim: int) -> None:
    """Ensure the fewshot_examples table exists with correct dimensions."""
    # Check if table exists
    exists = await conn.fetchval("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'fewshot_examples'
        )
    """)

    if exists:
        # Check if dimension matches
        current_dim = await conn.fetchval("""
            SELECT atttypmod
            FROM pg_attribute
            WHERE attrelid = 'fewshot_examples'::regclass
            AND attname = 'question_embedding'
        """)

        # pgvector stores dimension + 4 in atttypmod
        if current_dim and current_dim != dim + 4:
            print(f"Warning: Table has different dimension ({current_dim - 4} vs {dim})")
            print("Dropping and recreating table...")
            await conn.execute("DROP TABLE IF EXISTS fewshot_examples CASCADE")
            exists = False

    if not exists:
        # Create table with correct dimension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        await conn.execute(f"""
            CREATE TABLE fewshot_examples (
                id VARCHAR(50) PRIMARY KEY,
                question TEXT NOT NULL,
                sql TEXT NOT NULL,
                explanation TEXT,
                tables_used TEXT[] NOT NULL DEFAULT '{{}}',
                category VARCHAR(50) NOT NULL DEFAULT 'general',
                difficulty VARCHAR(20) NOT NULL DEFAULT 'medium',
                question_embedding vector({dim}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        await conn.execute(f"""
            CREATE INDEX idx_fewshot_embedding
            ON fewshot_examples
            USING ivfflat (question_embedding vector_cosine_ops)
            WITH (lists = 10)
        """)

        await conn.execute("""
            CREATE INDEX idx_fewshot_category ON fewshot_examples(category)
        """)

        await conn.execute("""
            CREATE INDEX idx_fewshot_difficulty ON fewshot_examples(difficulty)
        """)

        await conn.execute("""
            CREATE INDEX idx_fewshot_tables ON fewshot_examples USING GIN(tables_used)
        """)

        await conn.execute("""
            CREATE INDEX idx_fewshot_question_fts
            ON fewshot_examples
            USING GIN(to_tsvector('english', question))
        """)

        print(f"Created fewshot_examples table (dim={dim})")
    else:
        print(f"fewshot_examples table exists (dim={dim})")


async def clear_examples(conn: asyncpg.Connection) -> int:
    """Clear existing examples and return count deleted."""
    result = await conn.execute("DELETE FROM fewshot_examples")
    # Extract count from "DELETE N"
    count = int(result.split()[-1]) if result else 0
    return count


def format_vector(embedding: list[float]) -> str:
    """Format embedding as pgvector string literal."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def index_examples(
    conn: asyncpg.Connection,
    embed_fn,
    batch_size: int = 20,
) -> int:
    """Embed and store all few-shot examples.

    Args:
        conn: Database connection
        embed_fn: Function to compute embeddings
        batch_size: Number of examples to embed per API call

    Returns:
        Number of examples indexed
    """
    questions = [ex.question for ex in CHINOOK_EXAMPLES]
    print(f"Embedding {len(questions)} questions...")

    # Embed in batches to avoid API limits
    all_embeddings = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        batch_embeddings = embed_fn(batch)
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded {len(all_embeddings)}/{len(questions)}")

    print(f"Generated {len(all_embeddings)} embeddings")

    # Insert using raw SQL with vector literal
    indexed = 0
    for ex, embedding in zip(CHINOOK_EXAMPLES, all_embeddings):
        # Format embedding as pgvector literal
        embedding_str = format_vector(embedding)

        # Use raw SQL with string interpolation for the vector
        # (asyncpg doesn't have native pgvector type support)
        await conn.execute(f"""
            INSERT INTO fewshot_examples (
                id, question, sql, explanation, tables_used,
                category, difficulty, question_embedding
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, '{embedding_str}'::vector)
            ON CONFLICT (id) DO UPDATE SET
                question = EXCLUDED.question,
                sql = EXCLUDED.sql,
                explanation = EXCLUDED.explanation,
                tables_used = EXCLUDED.tables_used,
                category = EXCLUDED.category,
                difficulty = EXCLUDED.difficulty,
                question_embedding = EXCLUDED.question_embedding,
                updated_at = CURRENT_TIMESTAMP
        """,
            ex.id,
            ex.question,
            ex.sql,
            ex.explanation,
            ex.tables_used,
            ex.category,
            ex.difficulty,
        )
        indexed += 1

    print(f"Indexed {indexed} examples")
    return indexed


async def verify_index(conn: asyncpg.Connection, embed_fn) -> None:
    """Verify the index was created correctly."""
    # Count examples
    count = await conn.fetchval("SELECT COUNT(*) FROM fewshot_examples")
    print(f"\nVerification:")
    print(f"  Total examples: {count}")

    # Check a sample
    sample = await conn.fetchrow("""
        SELECT id, question
        FROM fewshot_examples
        LIMIT 1
    """)
    if sample:
        print(f"  Sample: {sample['id']}")
        print(f"  Question: {sample['question'][:50]}...")

    # Test similarity search
    print("\nTesting similarity search...")
    test_question = "How many customers are there?"

    # Embed the test question
    test_embedding = embed_fn([test_question])[0]
    embedding_str = format_vector(test_embedding)

    results = await conn.fetch(f"""
        SELECT id, question,
               1 - (question_embedding <=> '{embedding_str}'::vector) as similarity
        FROM fewshot_examples
        ORDER BY question_embedding <=> '{embedding_str}'::vector
        LIMIT 3
    """)

    print(f"  Query: {test_question}")
    for i, row in enumerate(results, 1):
        print(f"  {i}. [{row['similarity']:.3f}] {row['question'][:50]}...")


async def main():
    """Index all few-shot examples into pgvector."""
    parser = argparse.ArgumentParser(description="Index few-shot examples into pgvector")
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini", "local"],
        default=None,
        help="Embedding provider (default: from config)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Indexing Few-Shot Examples into pgvector")
    print("=" * 60)
    print()

    settings = load_config()

    # Use specified provider or fall back to config
    provider = args.provider or settings.llm.provider
    print(f"Embedding provider: {provider}")
    print(f"Database: {settings.database.url.split('@')[-1]}")  # Hide password
    print()

    # Create embedder
    embed_fn, dim = get_embedder(provider)
    print(f"Embedding dimension: {dim}")

    # Connect to database
    conn = await asyncpg.connect(settings.database.url)
    try:
        # Ensure schema exists with correct dimension
        await ensure_schema(conn, dim)

        # Clear existing examples
        deleted = await clear_examples(conn)
        if deleted > 0:
            print(f"Cleared {deleted} existing examples")

        # Index new examples
        indexed = await index_examples(conn, embed_fn)
        print(f"\nIndexed {indexed} examples successfully")

        # Verify
        await verify_index(conn, embed_fn)

        print()
        print("=" * 60)
        print("SUCCESS! Few-shot examples indexed in pgvector")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Run chapter 2.3 script: python scripts/run_chapter_2_3.py")
        print("  2. The PgVectorRetriever will use these indexed embeddings")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
