#!/usr/bin/env python3
"""3.4 Context Engineering - Daily Context Refresh Pipeline.

This script implements the offline preprocessing phase of context engineering.
It aggregates, normalizes, and embeds context from multiple layers.

Based on OpenAI's architecture:
> "We run a daily offline pipeline that aggregates table usage, human annotations,
>  and Codex-derived enrichment into a single, normalized representation."

The Six Context Layers:
1. Table Usage - Schema metadata, lineage, query patterns
2. Human Annotations - Domain expert descriptions (Schema Cards from 2.2)
3. Codex Enrichment - Code-derived table definitions
4. Institutional Knowledge - Docs, wikis, Slack messages
5. Memory - User corrections and learned definitions
6. Runtime Context - (Not pre-processed, queried live)

Usage:
    make db-up
    python scripts/run_context_pipeline.py

    # Run specific layers only
    python scripts/run_context_pipeline.py --layers 1,2,5

    # Dry run (show what would be processed)
    python scripts/run_context_pipeline.py --dry-run
"""

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import asyncpg

from src.adapters import create_adapter
from src.schema import SchemaStore
from src.utils.config import load_config


@dataclass
class ContextChunk:
    """A single piece of retrievable context."""

    content: str
    layer: int  # 1-6
    layer_name: str
    source: str  # Where this came from (table name, doc path, etc.)
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""

    layer_counts: dict[int, int] = field(default_factory=dict)
    total_chunks: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Layer 1: Table Usage
# =============================================================================

async def extract_layer1_table_usage(conn: asyncpg.Connection) -> list[ContextChunk]:
    """Extract schema metadata and query patterns.

    OpenAI: "Metadata grounding: The agent relies on schema metadata
    (column names and data types) to inform SQL writing."
    """
    chunks = []

    # Get all tables with their columns
    tables = await conn.fetch("""
        SELECT
            t.table_name,
            t.table_type,
            array_agg(
                c.column_name || ' ' || c.data_type ||
                CASE WHEN c.is_nullable = 'NO' THEN ' NOT NULL' ELSE '' END
            ) as columns
        FROM information_schema.tables t
        JOIN information_schema.columns c
            ON t.table_name = c.table_name
            AND t.table_schema = c.table_schema
        WHERE t.table_schema = 'public'
        GROUP BY t.table_name, t.table_type
        ORDER BY t.table_name
    """)

    for table in tables:
        content = f"Table: {table['table_name']}\n"
        content += f"Type: {table['table_type']}\n"
        content += "Columns:\n"
        for col in table['columns']:
            content += f"  - {col}\n"

        chunks.append(ContextChunk(
            content=content,
            layer=1,
            layer_name="Table Usage",
            source=table['table_name'],
            metadata={"table_name": table['table_name'], "column_count": len(table['columns'])}
        ))

    # Extract foreign key relationships (lineage)
    fks = await conn.fetch("""
        SELECT
            tc.table_name as from_table,
            kcu.column_name as from_column,
            ccu.table_name as to_table,
            ccu.column_name as to_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu
            ON tc.constraint_name = ccu.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
    """)

    if fks:
        lineage_content = "Table Relationships (Foreign Keys):\n"
        for fk in fks:
            lineage_content += f"  {fk['from_table']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}\n"

        chunks.append(ContextChunk(
            content=lineage_content,
            layer=1,
            layer_name="Table Usage",
            source="foreign_keys",
            metadata={"relationship_count": len(fks)}
        ))

    return chunks


def extract_query_patterns(query_logs: list[dict]) -> list[ContextChunk]:
    """Mine common join patterns from historical queries.

    OpenAI: "Ingesting historical queries helps the agent understand
    how to write its own queries and which tables are typically joined together."
    """
    chunks = []
    co_occurrence = defaultdict(Counter)

    for log in query_logs:
        # Simple extraction: find table names in FROM/JOIN clauses
        sql = log.get('sql', '').upper()
        tables = extract_tables_from_sql(sql)

        for t1, t2 in combinations(sorted(tables), 2):
            co_occurrence[t1][t2] += 1
            co_occurrence[t2][t1] += 1

    if co_occurrence:
        patterns_content = "Common Table Join Patterns (from query logs):\n"
        for table, partners in sorted(co_occurrence.items()):
            top_partners = partners.most_common(3)
            if top_partners:
                partner_str = ", ".join(f"{p[0]} ({p[1]}x)" for p in top_partners)
                patterns_content += f"  {table} often joined with: {partner_str}\n"

        chunks.append(ContextChunk(
            content=patterns_content,
            layer=1,
            layer_name="Table Usage",
            source="query_patterns",
            metadata={"table_count": len(co_occurrence)}
        ))

    return chunks


def extract_tables_from_sql(sql: str) -> set[str]:
    """Extract table names from SQL (simple heuristic)."""
    import re
    tables = set()

    # Match FROM table and JOIN table patterns
    patterns = [
        r'\bFROM\s+(\w+)',
        r'\bJOIN\s+(\w+)',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, sql, re.IGNORECASE):
            tables.add(match.group(1).lower())

    return tables


# =============================================================================
# Layer 2: Human Annotations (from Schema Cards)
# =============================================================================

def extract_layer2_human_annotations(schema_store: SchemaStore) -> list[ContextChunk]:
    """Extract human-authored descriptions from Schema Cards.

    OpenAI: "Curated descriptions of tables and columns provided by domain experts,
    capturing intent, semantics, business meaning, and known caveats."

    This layer comes from Chapter 2.2 Schema Representation.
    """
    chunks = []

    for table in schema_store.get_all_tables():
        content = f"Table: {table.name}\n"
        content += f"Description: {table.description}\n"

        if table.columns:
            content += "Column Details:\n"
            for col in table.columns:
                col_desc = f"  - {col.name} ({col.data_type})"
                if col.description:
                    col_desc += f": {col.description}"
                if col.business_rules:
                    col_desc += f" [Rule: {col.business_rules}]"
                content += col_desc + "\n"

        chunks.append(ContextChunk(
            content=content,
            layer=2,
            layer_name="Human Annotations",
            source=table.name,
            metadata={
                "table_name": table.name,
                "column_count": len(table.columns) if table.columns else 0,
                "has_description": bool(table.description),
            }
        ))

    return chunks


# =============================================================================
# Layer 3: Codex Enrichment
# =============================================================================

async def extract_layer3_codex_enrichment(
    codebase_path: Path | None = None
) -> list[ContextChunk]:
    """Extract context from code that uses the database.

    OpenAI: "We crawl the codebase for files that reference tables or columns
    to generate detailed descriptions through our Codex model."

    This layer scans Python/SQL files to understand:
    - How tables are used in transformations
    - Update frequency and data scope
    - Business logic encoded in code
    """
    chunks = []

    if codebase_path is None:
        # Use project's src directory as example
        codebase_path = project_root / "src"

    if not codebase_path.exists():
        return chunks

    # Find Python files that might reference database tables
    table_references = defaultdict(list)

    for py_file in codebase_path.rglob("*.py"):
        try:
            content = py_file.read_text()

            # Look for SQL or table references
            import re
            table_patterns = [
                r"['\"](\w+)['\"].*(?:table|TABLE)",
                r"FROM\s+['\"]?(\w+)['\"]?",
                r"INSERT\s+INTO\s+['\"]?(\w+)['\"]?",
                r"UPDATE\s+['\"]?(\w+)['\"]?",
            ]

            for pattern in table_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    table_name = match.group(1).lower()
                    if len(table_name) > 2:  # Filter out short matches
                        table_references[table_name].append({
                            "file": str(py_file.relative_to(project_root)),
                            "context": content[max(0, match.start()-50):match.end()+50],
                        })
        except Exception:
            continue

    # Create chunks for tables with code references
    for table, refs in table_references.items():
        if len(refs) >= 1:  # At least one reference
            content = f"Code Usage for table '{table}':\n"
            content += f"Referenced in {len(refs)} location(s):\n"
            for ref in refs[:5]:  # Limit to 5 references
                content += f"  - {ref['file']}\n"

            chunks.append(ContextChunk(
                content=content,
                layer=3,
                layer_name="Codex Enrichment",
                source=table,
                metadata={"reference_count": len(refs), "files": [r["file"] for r in refs[:5]]}
            ))

    return chunks


# =============================================================================
# Layer 4: Institutional Knowledge
# =============================================================================

def extract_layer4_institutional_knowledge(
    docs_path: Path | None = None
) -> list[ContextChunk]:
    """Extract context from documentation and institutional sources.

    OpenAI: "Crawling Slack messages, Google Docs, Notion pages, etc. to surface
    information like launch announcements, incident reports, and canonical metric
    definitions."

    In production, this would integrate with:
    - Slack API for data-related discussions
    - Notion/Confluence for documentation
    - Google Docs for specifications
    """
    chunks = []

    if docs_path is None:
        # Look for docs in the project
        docs_path = project_root / "docs"

    if not docs_path.exists():
        return chunks

    # Process markdown files
    for md_file in docs_path.rglob("*.md"):
        try:
            content = md_file.read_text()

            # Extract title (first heading)
            import re
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else md_file.stem

            # Chunk the document if it's long
            if len(content) > 2000:
                # Split into sections
                sections = re.split(r'^##\s+', content, flags=re.MULTILINE)
                for i, section in enumerate(sections[:5]):  # Max 5 sections
                    if section.strip():
                        chunks.append(ContextChunk(
                            content=f"Document: {title}\nSection {i+1}:\n{section[:1500]}",
                            layer=4,
                            layer_name="Institutional Knowledge",
                            source=str(md_file.relative_to(project_root)),
                            metadata={"document": title, "section": i+1}
                        ))
            else:
                chunks.append(ContextChunk(
                    content=f"Document: {title}\n\n{content}",
                    layer=4,
                    layer_name="Institutional Knowledge",
                    source=str(md_file.relative_to(project_root)),
                    metadata={"document": title}
                ))
        except Exception:
            continue

    return chunks


# =============================================================================
# Layer 5: Memory
# =============================================================================

def extract_layer5_memory(memory_store: dict | None = None) -> list[ContextChunk]:
    """Extract learned context from user corrections and feedback.

    OpenAI: "What makes memory distinct is that it's learned: it captures
    information from previous queries that helps answer future queries more
    accurately."

    Memory is scoped:
    - Global: Shared definitions (e.g., "churn = no purchase in 90 days")
    - Personal: User-specific preferences

    This connects to Chapter 2.8 Agent Memory.
    """
    chunks = []

    if memory_store is None:
        # Example memories (would come from database in production)
        memory_store = {
            "global": [
                {
                    "content": "'Active customer' means customer with purchase in last 90 days",
                    "source": "user_correction_2024_01",
                },
                {
                    "content": "Revenue should always exclude refunds unless specified",
                    "source": "user_correction_2024_02",
                },
                {
                    "content": "Test accounts (email ending in @test.com) should be excluded from metrics",
                    "source": "data_quality_rule",
                },
            ],
            "personal": {}  # Would be keyed by user_id
        }

    # Global memories
    for mem in memory_store.get("global", []):
        chunks.append(ContextChunk(
            content=f"Global Memory: {mem['content']}",
            layer=5,
            layer_name="Memory",
            source=mem.get("source", "unknown"),
            metadata={"scope": "global"}
        ))

    # Personal memories would be filtered by user at retrieval time
    # but pre-embedded here
    for user_id, user_memories in memory_store.get("personal", {}).items():
        for mem in user_memories:
            chunks.append(ContextChunk(
                content=f"Personal Memory ({user_id}): {mem['content']}",
                layer=5,
                layer_name="Memory",
                source=mem.get("source", "unknown"),
                metadata={"scope": "personal", "user_id": user_id}
            ))

    return chunks


# =============================================================================
# Embedding and Storage
# =============================================================================

async def embed_chunks(chunks: list[ContextChunk]) -> list[ContextChunk]:
    """Embed all chunks using OpenAI embeddings.

    OpenAI: "This enriched context is then converted into embeddings
    using the OpenAI embeddings API and stored for retrieval."
    """
    adapter = create_adapter()

    # Embed in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.content for c in batch]

        try:
            embeddings = await adapter.embed_batch(texts)
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding
        except Exception as e:
            print(f"  Warning: Embedding batch {i//batch_size + 1} failed: {e}")
            # Continue without embeddings for failed batch

    return chunks


async def store_chunks(chunks: list[ContextChunk], conn: asyncpg.Connection) -> int:
    """Store context chunks in PostgreSQL with pgvector.

    Creates the table if it doesn't exist.
    """
    # Create table if needed
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS context_chunks (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            layer INTEGER NOT NULL,
            layer_name VARCHAR(50) NOT NULL,
            source VARCHAR(255),
            metadata JSONB,
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(layer, source, content)
        )
    """)

    # Create index on embeddings if not exists
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS context_chunks_embedding_idx
        ON context_chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)

    # Insert chunks (upsert to handle duplicates)
    stored = 0
    for chunk in chunks:
        try:
            embedding_str = (
                f"[{','.join(str(x) for x in chunk.embedding)}]"
                if chunk.embedding else None
            )

            await conn.execute("""
                INSERT INTO context_chunks (content, layer, layer_name, source, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5, $6::vector)
                ON CONFLICT (layer, source, content) DO UPDATE SET
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    created_at = NOW()
            """, chunk.content, chunk.layer, chunk.layer_name, chunk.source,
                json.dumps(chunk.metadata), embedding_str)
            stored += 1
        except Exception as e:
            print(f"  Warning: Failed to store chunk from {chunk.source}: {e}")

    return stored


# =============================================================================
# Main Pipeline
# =============================================================================

async def run_pipeline(
    layers: list[int] | None = None,
    dry_run: bool = False,
    query_logs: list[dict] | None = None,
) -> PipelineStats:
    """Run the full context preprocessing pipeline.

    Args:
        layers: Which layers to process (1-5). None means all.
        dry_run: If True, don't store results.
        query_logs: Historical query logs for Layer 1 pattern extraction.
    """
    stats = PipelineStats()
    start_time = datetime.now()

    if layers is None:
        layers = [1, 2, 3, 4, 5]

    settings = load_config()
    all_chunks: list[ContextChunk] = []

    print("=" * 60)
    print("Context Engineering - Daily Pipeline")
    print("=" * 60)
    print(f"Processing layers: {layers}")
    print(f"Dry run: {dry_run}")
    print()

    # Connect to database
    conn = await asyncpg.connect(settings.database.url)

    try:
        # Load schema store for Layer 2
        schema_store = SchemaStore.from_yaml(project_root / "config" / "chinook_schema.yaml")

        # Layer 1: Table Usage
        if 1 in layers:
            print("Layer 1: Table Usage...")
            l1_chunks = await extract_layer1_table_usage(conn)
            if query_logs:
                l1_chunks.extend(extract_query_patterns(query_logs))
            all_chunks.extend(l1_chunks)
            stats.layer_counts[1] = len(l1_chunks)
            print(f"  Extracted {len(l1_chunks)} chunks")

        # Layer 2: Human Annotations
        if 2 in layers:
            print("Layer 2: Human Annotations...")
            l2_chunks = extract_layer2_human_annotations(schema_store)
            all_chunks.extend(l2_chunks)
            stats.layer_counts[2] = len(l2_chunks)
            print(f"  Extracted {len(l2_chunks)} chunks")

        # Layer 3: Codex Enrichment
        if 3 in layers:
            print("Layer 3: Codex Enrichment...")
            l3_chunks = await extract_layer3_codex_enrichment()
            all_chunks.extend(l3_chunks)
            stats.layer_counts[3] = len(l3_chunks)
            print(f"  Extracted {len(l3_chunks)} chunks")

        # Layer 4: Institutional Knowledge
        if 4 in layers:
            print("Layer 4: Institutional Knowledge...")
            l4_chunks = extract_layer4_institutional_knowledge()
            all_chunks.extend(l4_chunks)
            stats.layer_counts[4] = len(l4_chunks)
            print(f"  Extracted {len(l4_chunks)} chunks")

        # Layer 5: Memory
        if 5 in layers:
            print("Layer 5: Memory...")
            l5_chunks = extract_layer5_memory()
            all_chunks.extend(l5_chunks)
            stats.layer_counts[5] = len(l5_chunks)
            print(f"  Extracted {len(l5_chunks)} chunks")

        stats.total_chunks = len(all_chunks)
        print()
        print(f"Total chunks extracted: {stats.total_chunks}")

        if dry_run:
            print("\n[DRY RUN] Would process these chunks:")
            for layer in sorted(stats.layer_counts.keys()):
                print(f"  Layer {layer}: {stats.layer_counts[layer]} chunks")
        else:
            # Embed chunks
            print("\nEmbedding chunks...")
            all_chunks = await embed_chunks(all_chunks)
            embedded_count = sum(1 for c in all_chunks if c.embedding)
            print(f"  Embedded {embedded_count}/{len(all_chunks)} chunks")

            # Store chunks
            print("\nStoring chunks in PostgreSQL...")
            stored = await store_chunks(all_chunks, conn)
            print(f"  Stored {stored} chunks")

    finally:
        await conn.close()

    stats.duration_seconds = (datetime.now() - start_time).total_seconds()

    print()
    print("=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Duration: {stats.duration_seconds:.1f}s")
    print(f"Total chunks: {stats.total_chunks}")
    for layer in sorted(stats.layer_counts.keys()):
        print(f"  Layer {layer}: {stats.layer_counts[layer]}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Run context preprocessing pipeline")
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list of layers to process (1-5). Default: all"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without storing"
    )

    args = parser.parse_args()

    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    asyncio.run(run_pipeline(layers=layers, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
