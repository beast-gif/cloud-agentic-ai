"""
Agentic Cloud - Phase 2, Step 2.2
Session tokens for the Memory API.

  issue(session_id)  -> new random token (256 bits); only its SHA-256 hash is stored
  resolve(token)     -> the session row, if the token matches AND the session is live
  revoke(session_id) -> token no longer valid

Why the agent can only ever reach its OWN user's memories:
  the agent sends only the token; the control plane looks up which session (and so
  which user_id) the token belongs to, and the Memory API filters on THAT user_id.
  Nothing the agent sends can change the user_id.

Token lifetime:
  issued on every container start (cold start and each wake) -> rotated on every wake
  accepted only while the session is RUNNING or WAITING      -> useless once FROZEN
  cleared when the session finishes                          -> useless once COMPLETED
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import Optional

from psycopg_pool import ConnectionPool

from control_plane.session_store import RUNNING, WAITING

log = logging.getLogger("control_plane.auth")

ACCEPTED_STATES = [RUNNING, WAITING]


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class SessionAuth:
    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool

    def issue(self, session_id: str) -> str:
        token = secrets.token_urlsafe(32)  # 32 random bytes = 256 bits
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE sessions SET token_hash = %s, token_issued_at = now() WHERE session_id = %s",
                (_hash(token), session_id),
            )
        return token

    def resolve(self, token: Optional[str]) -> Optional[dict]:
        if not token:
            return None
        with self.pool.connection() as conn:
            return conn.execute(
                "SELECT * FROM sessions WHERE token_hash = %s AND state = ANY(%s)",
                (_hash(token), ACCEPTED_STATES),
            ).fetchone()

    def revoke(self, session_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE sessions SET token_hash = NULL WHERE session_id = %s", (session_id,)
            )