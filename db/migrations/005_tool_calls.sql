-- Agentic Cloud - Phase 3, Step 3.3
-- Audit log of the MCP gateway: one row per tools/call attempt (allowed or not).
--
--   decision      allowed | denied | rate_limited | error
--   reason        why it was denied / limited / failed (NULL when allowed)
--   args_hash     sha256 of the canonical JSON arguments (first 16 hex chars):
--                 lets you spot identical calls without storing everything
--   args_preview  first 300 characters of the arguments AS SENT BY THE AGENT
--                 (agents never hold secrets, so this never contains a credential;
--                 it is what lets you see e.g. to='attacker@evil.com' afterwards)
--   credential_used  vault provider whose secret the gateway attached (never the secret)
--
-- Applied automatically by the control plane on startup.

CREATE TABLE IF NOT EXISTS tool_calls (
    id               BIGSERIAL PRIMARY KEY,
    session_id       TEXT        NOT NULL,
    user_id          TEXT        NOT NULL,
    agent_type       TEXT        NOT NULL,
    tool             TEXT        NOT NULL,           -- e.g. papers.search
    decision         TEXT        NOT NULL
                     CHECK (decision IN ('allowed', 'denied', 'rate_limited', 'error')),
    reason           TEXT,
    args_hash        TEXT        NOT NULL,
    args_preview     TEXT        NOT NULL DEFAULT '',
    latency_ms       DOUBLE PRECISION,               -- whole gateway handling time
    backend_ms       DOUBLE PRECISION,               -- time spent waiting for the MCP server
    tool_is_error    BOOLEAN,                        -- the server ran it but returned isError
    credential_used  TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls (session_id, id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_user    ON tool_calls (user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_decision ON tool_calls (decision, created_at);