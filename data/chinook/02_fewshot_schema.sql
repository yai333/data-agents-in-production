-- Enable extensions for hybrid search
-- pgvector: semantic search via embeddings
-- pg_textsearch: true BM25 full-text search ranking
--
-- The fewshot_examples table is created by scripts/index_fewshot_examples.py
-- since the vector dimension depends on the embedding provider:
--   - OpenAI (text-embedding-3-small): 1536 dimensions
--   - Gemini (text-embedding-004): 768 dimensions
--   - Local (all-MiniLM-L6-v2): 384 dimensions

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;
