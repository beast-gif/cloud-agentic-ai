-- Agentic Cloud - Phase 1, Step 1.1
-- Control-plane tables. Runs automatically on the first `docker compose up`.
--
-- NOTE: LangGraph's checkpoint tables (checkpoints, checkpoint_writes, checkpoint_blobs,
-- checkpoint_migrations) are NOT created here. The agent creates them itself by calling
-- PostgresSaver.setup() in Step 1.2, so they always match the installed LangGraph version.

-- ---------------------------------------------------------------------------
-- 1. sessions: one row per agent session (the "session table" of the control plane)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,                 -- e.g. "s-7f3a9c"
    user_id         TEXT        NOT NULL,
    agent_type      TEXT        NOT NULL,             -- which agent image / graph
    thread_id       TEXT        NOT NULL,             -- LangGraph thread_id (checkpoint key)
    state           TEXT        NOT NULL DEFAULT 'STARTING'
                    CHECK (state IN ('STARTING', 'RUNNING', 'WAITING',
                                     'FREEZING', 'FROZEN', 'WAKING',
                                     'COMPLETED', 'FAILED')),
    container_id    TEXT,                             -- NULL when FROZEN / COMPLETED
    agent_url       TEXT,                             -- e.g. http://127.0.0.1:49213
    waiting_for     TEXT,                             -- what the agent waits on (e.g. "human_approval")
    wake_count      INTEGER     NOT NULL DEFAULT 0,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at  TIMESTAMPTZ NOT NULL DEFAULT now() -- used by the idle reaper
);

-- The idle reaper scans: state = 'WAITING' AND last_active_at < now() - idle_timeout
CREATE INDEX IF NOT EXISTS idx_sessions_state_active
    ON sessions (state, last_active_at);

CREATE INDEX IF NOT EXISTS idx_sessions_user
    ON sessions (user_id);

-- ---------------------------------------------------------------------------
-- 2. session_events: audit log of everything that happens to a session
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_events (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT        NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    event_type  TEXT        NOT NULL,   -- created | started | waiting | freeze | frozen | wake | resumed | completed | failed | user_event
    payload     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_events_session
    ON session_events (session_id, created_at);

-- ---------------------------------------------------------------------------
-- 3. wake_metrics: timing breakdown of every start / wake (our Phase 1 baseline)
--    Phase 5 (golden.snap + Firecracker) will be compared against these numbers.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS wake_metrics (
    id                  BIGSERIAL PRIMARY KEY,
    session_id          TEXT        NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    wake_no             INTEGER     NOT NULL,          -- 0 = first start, 1+ = wakes after freeze
    kind                TEXT        NOT NULL CHECK (kind IN ('cold_start', 'wake')),
    container_start_ms  DOUBLE PRECISION,              -- docker run -> process started
    import_ms           DOUBLE PRECISION,              -- python imports (langgraph etc.)
    graph_build_ms      DOUBLE PRECISION,              -- build + compile the graph
    checkpoint_load_ms  DOUBLE PRECISION,              -- read state from Postgres
    ready_ms            DOUBLE PRECISION,              -- control plane: request -> /health OK
    total_ms            DOUBLE PRECISION,              -- control plane: request -> agent resumed
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_wake_metrics_session
    ON wake_metrics (session_id, wake_no);

-- ---------------------------------------------------------------------------
-- Keep sessions.updated_at fresh on every UPDATE
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sessions_updated_at ON sessions;
CREATE TRIGGER trg_sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();