"""
Agentic Cloud - Phase 1, Step 1.3
Session Store: the control plane's only gateway to Postgres.

Tables (created by db/init.sql):
  sessions        one row per agent session  (the source of truth)
  session_events  audit log of everything that happens to a session
  wake_metrics    timing breakdown of every cold start / wake

Correctness rule: every state change is a COMPARE-AND-SET update:

    UPDATE sessions SET state = <to>
    WHERE session_id = <id> AND state = ANY(<allowed from-states>)
    RETURNING *

Postgres runs each UPDATE atomically, so if two actors (e.g. the idle reaper and an
incoming user event) try to move the same session at the same time, exactly one gets
the row back; the other gets None and must react (retry / wait / skip).
"""

from __future__ import annotations

import logging
import secrets
from typing import Any, Iterable, Optional

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

log = logging.getLogger("control_plane.store")

# ---- states ------------------------------------------------------------------

STARTING = "STARTING"
RUNNING = "RUNNING"
WAITING = "WAITING"
FREEZING = "FREEZING"
FROZEN = "FROZEN"
WAKING = "WAKING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"

ALL_STATES = (STARTING, RUNNING, WAITING, FREEZING, FROZEN, WAKING, COMPLETED, FAILED)
TERMINAL_STATES = (COMPLETED, FAILED)
# states in which a container is (or should be) alive
LIVE_STATES = (STARTING, RUNNING, WAITING, FREEZING, WAKING)

# columns that transition() / update_fields() are allowed to set
_SETTABLE = {"container_id", "agent_url", "waiting_for", "error"}


class SessionStore:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 8) -> None:
        self.pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=True,
        )
        self.pool.wait(timeout=30)
        log.info("session store connected")

    def close(self) -> None:
        self.pool.close()

    # ================================================================ sessions

    def create_session(
        self,
        user_id: str,
        agent_type: str,
        session_id: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> dict:
        """Insert a new session in STARTING state. thread_id = session_id (1 session = 1 LangGraph thread)."""
        session_id = session_id or f"s-{secrets.token_hex(4)}"
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                """
                INSERT INTO sessions (session_id, user_id, agent_type, thread_id, state)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING *
                """,
                (session_id, user_id, agent_type, session_id, STARTING),
            ).fetchone()
            self._log_event(conn, session_id, "created", payload or {})
        return row

    def get(self, session_id: str) -> Optional[dict]:
        with self.pool.connection() as conn:
            return conn.execute(
                "SELECT * FROM sessions WHERE session_id = %s", (session_id,)
            ).fetchone()

    def list_sessions(
        self,
        state: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        clauses, params = [], []
        if state:
            clauses.append("state = %s")
            params.append(state)
        if user_id:
            clauses.append("user_id = %s")
            params.append(user_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self.pool.connection() as conn:
            return conn.execute(
                f"SELECT * FROM sessions {where} ORDER BY created_at DESC LIMIT %s", params
            ).fetchall()

    # ================================================================ state changes

    def transition(
        self,
        session_id: str,
        from_states: str | Iterable[str],
        to_state: str,
        *,
        event: Optional[str] = None,
        payload: Optional[dict] = None,
        touch_active: bool = False,
        increment_wake: bool = False,
        **fields: Any,
    ) -> Optional[dict]:
        """Compare-and-set state change.

        Returns the updated row, or None if the session was NOT in one of `from_states`
        (someone else changed it first, or it does not exist).

        fields         : extra columns to set (container_id, agent_url, waiting_for, error);
                         pass None to clear a column.
        touch_active   : also set last_active_at = now()
        increment_wake : also do wake_count = wake_count + 1
        event          : if given, an audit event is written in the SAME transaction
        """
        if to_state not in ALL_STATES:
            raise ValueError(f"unknown state {to_state!r}")
        allowed = [from_states] if isinstance(from_states, str) else list(from_states)
        bad = set(fields) - _SETTABLE
        if bad:
            raise ValueError(f"cannot set columns: {sorted(bad)}")

        sets = [sql.SQL("state = %s")]
        params: list[Any] = [to_state]
        for col, val in fields.items():
            sets.append(sql.SQL("{} = %s").format(sql.Identifier(col)))
            params.append(val)
        if touch_active:
            sets.append(sql.SQL("last_active_at = now()"))
        if increment_wake:
            sets.append(sql.SQL("wake_count = wake_count + 1"))

        query = sql.SQL(
            "UPDATE sessions SET {sets} WHERE session_id = %s AND state = ANY(%s) RETURNING *"
        ).format(sets=sql.SQL(", ").join(sets))
        params.extend([session_id, allowed])

        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(query, params).fetchone()
            if row is None:
                log.debug("CAS failed: %s not in %s -> %s", session_id, allowed, to_state)
                return None
            if event:
                self._log_event(
                    conn, session_id, event,
                    {"from": allowed, "to": to_state, **(payload or {})},
                )
        log.info("session %s -> %s", session_id, to_state)
        return row

    def touch(self, session_id: str) -> None:
        """Mark activity (resets the idle timer)."""
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE sessions SET last_active_at = now() WHERE session_id = %s", (session_id,)
            )

    # ================================================================ queries for background loops

    def find_idle(self, idle_timeout_s: float) -> list[dict]:
        """WAITING sessions whose last activity is older than idle_timeout_s (uses idx_sessions_state_active)."""
        with self.pool.connection() as conn:
            return conn.execute(
                """
                SELECT * FROM sessions
                WHERE state = %s
                  AND last_active_at < now() - make_interval(secs => %s)
                ORDER BY last_active_at
                """,
                (WAITING, float(idle_timeout_s)),
            ).fetchall()

    def find_live(self) -> list[dict]:
        """Sessions that should currently have a container (used by reconciliation)."""
        with self.pool.connection() as conn:
            return conn.execute(
                "SELECT * FROM sessions WHERE state = ANY(%s)", (list(LIVE_STATES),)
            ).fetchall()

    # ================================================================ events

    def log_event(self, session_id: str, event_type: str, payload: Optional[dict] = None) -> None:
        with self.pool.connection() as conn:
            self._log_event(conn, session_id, event_type, payload or {})

    def events(self, session_id: str, limit: int = 200) -> list[dict]:
        with self.pool.connection() as conn:
            return conn.execute(
                """
                SELECT id, event_type, payload, created_at FROM session_events
                WHERE session_id = %s ORDER BY id LIMIT %s
                """,
                (session_id, limit),
            ).fetchall()

    @staticmethod
    def _log_event(conn, session_id: str, event_type: str, payload: dict) -> None:
        conn.execute(
            "INSERT INTO session_events (session_id, event_type, payload) VALUES (%s, %s, %s)",
            (session_id, event_type, Jsonb(payload)),
        )

    # ================================================================ wake metrics

    def record_wake_metrics(
        self,
        session_id: str,
        wake_no: int,
        kind: str,
        *,
        container_start_ms: Optional[float] = None,
        import_ms: Optional[float] = None,
        graph_build_ms: Optional[float] = None,
        checkpoint_load_ms: Optional[float] = None,
        ready_ms: Optional[float] = None,
        total_ms: Optional[float] = None,
    ) -> None:
        if kind not in ("cold_start", "wake"):
            raise ValueError("kind must be 'cold_start' or 'wake'")
        with self.pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO wake_metrics
                    (session_id, wake_no, kind, container_start_ms, import_ms,
                     graph_build_ms, checkpoint_load_ms, ready_ms, total_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (session_id, wake_no, kind, container_start_ms, import_ms,
                 graph_build_ms, checkpoint_load_ms, ready_ms, total_ms),
            )

    def wake_metrics(self, session_id: Optional[str] = None, limit: int = 500) -> list[dict]:
        with self.pool.connection() as conn:
            if session_id:
                return conn.execute(
                    "SELECT * FROM wake_metrics WHERE session_id = %s ORDER BY id LIMIT %s",
                    (session_id, limit),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM wake_metrics ORDER BY id DESC LIMIT %s", (limit,)
            ).fetchall()