-- Agentic Cloud - Phase 2, Step 2.2
-- Session tokens: how an agent proves "I am session X" to the Memory API
-- without ever holding database credentials for the memories table.
--
-- Only the SHA-256 HASH of the token is stored. If the database leaks, the hashes
-- cannot be used as tokens (same idea as storing password hashes).
--
-- A new token is issued on every container start (cold start AND every wake), so the
-- token of a frozen/deleted container stops working automatically.
--
-- Applied automatically by the control plane on startup (control_plane/migrations.py).

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS token_hash      TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS token_issued_at TIMESTAMPTZ;

-- lookup by hash on every memory API call; partial index skips sessions without a token
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_token_hash
    ON sessions (token_hash)
    WHERE token_hash IS NOT NULL;