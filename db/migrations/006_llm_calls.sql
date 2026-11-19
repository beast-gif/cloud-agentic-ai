-- Agentic Cloud - Phase 4, Step 4.2
-- Audit log of the MODEL ROUTER: one row per model call it makes (local or cloud).
--
--   task      score | summarize | generic
--   mode      the routing mode of the request: local | cloud | cascade
--   route     where THIS call went: local | cloud
--   kind      embed (batch of papers) | chat (one LLM call)
--   outcome   ok        the answer was used
--             escalated a local result was not trusted, so the cloud was asked (row of the CLOUD call)
--             fallback  the preferred route failed, this call is the replacement
--             failed    this call failed (error / timeout / unparsable / validator) -> see reason
--             skipped   a cloud call was wanted but not made (budget exhausted, cap reached)
--   reason    why it escalated / fell back / failed, e.g. "uncertain band cos=0.634"
--   items     papers in an embed batch (1 for chat calls); cache_hits = vectors reused
--   cost_inr  NULL when prices are not configured, 0 for local calls
--
-- Applied automatically by the control plane on startup.

CREATE TABLE IF NOT EXISTS llm_calls (
    id           BIGSERIAL PRIMARY KEY,
    session_id   TEXT,                               -- NULL for admin test calls
    user_id      TEXT        NOT NULL,
    task         TEXT        NOT NULL,
    mode         TEXT        NOT NULL CHECK (mode IN ('local', 'cloud', 'cascade')),
    route        TEXT        NOT NULL CHECK (route IN ('local', 'cloud')),
    kind         TEXT        NOT NULL CHECK (kind IN ('embed', 'chat')),
    model        TEXT        NOT NULL,
    outcome      TEXT        NOT NULL
                 CHECK (outcome IN ('ok', 'escalated', 'fallback', 'failed', 'skipped')),
    reason       TEXT,
    items        INTEGER     NOT NULL DEFAULT 1,
    cache_hits   INTEGER     NOT NULL DEFAULT 0,
    tokens_in    INTEGER     NOT NULL DEFAULT 0,
    tokens_out   INTEGER     NOT NULL DEFAULT 0,
    cost_inr     DOUBLE PRECISION,
    latency_ms   DOUBLE PRECISION,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_session ON llm_calls (session_id, id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_user    ON llm_calls (user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_calls_route   ON llm_calls (route, task, created_at);