-- Agentic Cloud - Phase 4, Step 4.4
-- Persistent embedding cache of the model router (level 2 behind its in-memory LRU).
--
--   key        sha256(model \0 prefix+text) - the same key the in-memory cache uses
--   embedding  unconstrained pgvector column (any dimension), read back as text
--
-- Why: on this CPU one embedding costs ~1.2 s. Without persistence every control-plane
-- restart threw the vectors away and the next rank re-embedded everything (18-20 s).
-- With it, each paper is embedded once, ever.
--
-- Lookups are by primary key only, so no vector index is needed.
-- Applied automatically by the control plane on startup.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embedding_cache (
    key           TEXT PRIMARY KEY,
    model         TEXT        NOT NULL,
    dim           INTEGER     NOT NULL,
    embedding     vector      NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    hits          INTEGER     NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_model ON embedding_cache (model, last_used_at);