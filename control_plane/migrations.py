"""
Agentic Cloud - Phase 2, Step 2.2
Tiny migration runner: applies db/migrations/*.sql in name order, once each.

Applied versions are recorded in the schema_migrations table, so every file runs
exactly once per database. Files are written to be idempotent anyway (IF NOT EXISTS),
so re-running one (e.g. 002_memory, which docker-compose already ran on a fresh
database) is harmless.

Called by the control plane on startup, before it accepts any request.
"""

from __future__ import annotations

import logging
from pathlib import Path

from psycopg_pool import ConnectionPool

from control_plane.settings import REPO_ROOT

log = logging.getLogger("control_plane.migrations")

MIGRATIONS_DIR = REPO_ROOT / "db" / "migrations"


def apply_migrations(pool: ConnectionPool, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations. Returns the versions applied in this call."""
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        done = {r["version"] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}

    applied: list[str] = []
    for path in sorted(directory.glob("*.sql")):
        version = path.stem  # e.g. "002_memory"
        if version in done:
            continue
        sql_text = path.read_text(encoding="utf-8")
        # one transaction per file: either the whole file applies, or nothing does
        with pool.connection() as conn, conn.transaction():
            conn.execute(sql_text)  # no parameters -> multiple statements allowed
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
        applied.append(version)
        log.info("applied migration %s", version)

    if not applied:
        log.info("database schema up to date (%d migrations)", len(done))
    return applied