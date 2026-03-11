-- Chapter 3.3: Context Engineering
-- Tables for the hybrid MDL architecture:
--   mdl_tables:     on-demand table loading (JSONB per table)
--   mdl_embeddings: semantic search over descriptive fields (pgvector)
--   glossary:       exact-match term definitions
--   sql_cache:      cached high-confidence (question, SQL) pairs

CREATE EXTENSION IF NOT EXISTS vector;

-- Per-table metadata for on-demand loading.
-- Populated from per-schema JSON files (data/mdl/*.json) by the offline pipeline.
CREATE TABLE IF NOT EXISTS mdl_tables (
    name          TEXT PRIMARY KEY,
    schema_name   TEXT NOT NULL,          -- e.g. 'chinook', 'analytics'
    table_data    JSONB NOT NULL           -- full TableCard JSON
);

-- Embedding index for semantic search over descriptive MDL fields.
-- Each row is one embedded text chunk (table description, business context, etc.).
CREATE TABLE IF NOT EXISTS mdl_embeddings (
    id            SERIAL PRIMARY KEY,
    table_name    TEXT,                    -- NULL for global entries (additional descriptions)
    field_type    TEXT NOT NULL,           -- 'table_description',
                                           -- 'additional_description', 'institutional'
    content       TEXT NOT NULL,           -- original text
    embedding     vector(1536)             -- text-embedding-3-small dimension
);

-- Business glossary for exact-match term lookup.
-- Populated from data/global_knowledge.json by the offline pipeline.
CREATE TABLE IF NOT EXISTS glossary (
    term          TEXT PRIMARY KEY,         -- lowercased, e.g. 'churn'
    definition    TEXT NOT NULL             -- e.g. 'Customer with no invoice in last 90 days'
);

-- SQL cache for repeated high-confidence questions.
-- Scoped by context_key (tenant/role) to prevent cross-tenant leakage.
CREATE TABLE IF NOT EXISTS sql_cache (
    context_key          TEXT NOT NULL,     -- e.g. 'tenant_id:role_id'
    normalized_question  TEXT NOT NULL,
    sql                  TEXT NOT NULL,
    tables_used          TEXT[] NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hit_count            INTEGER NOT NULL DEFAULT 0,
    schema_version       TEXT NOT NULL,     -- invalidate when schema changes
    PRIMARY KEY (context_key, normalized_question)
);
