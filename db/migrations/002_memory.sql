-- Agentic Cloud - Phase 2, Step 2.1
-- L3 long-term memory: one row per remembered fact, with its embedding vector.
--
-- Runs automatically on a fresh database (mounted as 02_memory.sql in docker-compose.yml).
-- Idempotent: safe to run again by hand on an existing database:
--   Get-Content db/migrations/002_memory.sql | docker exec -i agentic-postgres psql -U agentic -d agentic_cloud

-- ---------------------------------------------------------------------------
-- pgvector extension: adds the `vector(n)` type, distance operators and HNSW indexes
--   <=>  cosine distance   = 1 - cosine similarity
--   <->  Euclidean (L2) distance
--   <#>  negative inner product
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- memories
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id               BIGSERIAL PRIMARY KEY,
    user_id          TEXT        NOT NULL,              -- every query MUST filter on this
    kind             TEXT        NOT NULL
                     CHECK (kind IN ('paper_read', 'topic_researched', 'preference', 'note')),
    content          TEXT        NOT NULL,              -- the text that was embedded
    embedding        vector(768) NOT NULL,              -- 768 dims = nomic-embed-text (mock uses 768 too)
    -- Vectors from different models live in different "spaces" and must never be
    -- compared with each other, so we record which model produced each one and
    -- every search filters on it.
    embedding_model  TEXT        NOT NULL,
    metadata         JSONB       NOT NULL DEFAULT '{}'::jsonb,   -- e.g. {"paper_id", "title", "venue", "year"}
    source_session   TEXT        REFERENCES sessions(session_id) ON DELETE SET NULL,
    use_count        INTEGER     NOT NULL DEFAULT 0,    -- how often it was recalled or re-confirmed (dedup)
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Plain filter index: "all memories of this user (of this kind)"
CREATE INDEX IF NOT EXISTS idx_memories_user_kind
    ON memories (user_id, kind);

CREATE INDEX IF NOT EXISTS idx_memories_user_model
    ON memories (user_id, embedding_model);

-- ---------------------------------------------------------------------------
-- HNSW index for fast nearest-neighbour search on cosine distance.
--   m               = max links per node in the graph (higher = better recall, more memory)
--   ef_construction = candidate list size while building (higher = better graph, slower build)
-- Search-time knob (set per query later): SET hnsw.ef_search = 40 (default 40).
--
-- Note on filtering: the index finds the nearest vectors over ALL users first, then
-- the WHERE user_id = ... filter is applied. With many users this can return fewer
-- than k rows; pgvector >= 0.8 fixes it with  SET hnsw.iterative_scan = relaxed_order.
-- At our scale (one user, thousands of rows) Postgres will often just scan exactly.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw
    ON memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- Keep updated_at fresh (touch_updated_at() is defined in init.sql)
-- ---------------------------------------------------------------------------
DROP TRIGGER IF EXISTS trg_memories_updated_at ON memories;
CREATE TRIGGER trg_memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();