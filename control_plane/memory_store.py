"""
Agentic Cloud - Phase 2, Step 2.2
L3 long-term memory storage (pgvector).

Every method takes user_id and every SQL statement filters on it. Callers never pass
SQL; the Memory API decides the user_id from the session token, not from the agent.

Similarity:
  pgvector's  a <=> b  is COSINE DISTANCE = 1 - cosine similarity
  so  score = 1 - (embedding <=> query)   (1.0 = same direction, 0 = unrelated)

Dedup on write:
  find the nearest existing memory (same user, model, kind);
  if score >= dedup_threshold -> it is the same fact: bump use_count / last_used_at
  and merge metadata, instead of inserting a duplicate row.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

log = logging.getLogger("control_plane.memory")

KINDS = ("paper_read", "topic_researched", "preference", "note")


def _vec(v: list[float]) -> str:
    """Python list -> pgvector text literal '[0.1,0.2,...]' (cast with ::vector in SQL)."""
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


class MemoryStore:
    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool

    # ------------------------------------------------------------------ search

    def search(
        self,
        user_id: str,
        query_vec: list[float],
        model: str,
        k: int = 5,
        min_score: float = 0.0,
        kinds: Optional[list[str]] = None,
        touch: bool = True,
    ) -> list[dict]:
        """Top-k memories of this user by cosine similarity, keeping only score >= min_score."""
        params: dict[str, Any] = {"v": _vec(query_vec), "u": user_id, "m": model, "k": int(k)}
        kind_clause = ""
        if kinds:
            bad = set(kinds) - set(KINDS)
            if bad:
                raise ValueError(f"unknown kinds: {sorted(bad)}")
            kind_clause = "AND kind = ANY(%(kinds)s)"
            params["kinds"] = list(kinds)

        query = f"""
            SELECT id, kind, content, metadata, use_count, created_at, last_used_at,
                   source_session,
                   1 - (embedding <=> %(v)s::vector) AS score
            FROM memories
            WHERE user_id = %(u)s AND embedding_model = %(m)s {kind_clause}
            ORDER BY embedding <=> %(v)s::vector
            LIMIT %(k)s
        """
        with self.pool.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            hits = [dict(r, score=round(float(r["score"]), 4)) for r in rows
                    if float(r["score"]) >= min_score]
            if touch and hits:
                conn.execute(
                    "UPDATE memories SET use_count = use_count + 1, last_used_at = now() "
                    "WHERE id = ANY(%s) AND user_id = %s",
                    ([h["id"] for h in hits], user_id),
                )
        return hits

    # ------------------------------------------------------------------ write

    def write(
        self,
        user_id: str,
        kind: str,
        content: str,
        vec: list[float],
        model: str,
        metadata: Optional[dict] = None,
        source_session: Optional[str] = None,
        dedup_threshold: float = 0.95,
    ) -> dict:
        """Insert a memory, or merge into a near-identical existing one. Returns {id, action, similarity}."""
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; allowed: {KINDS}")
        content = content.strip()
        if not content:
            raise ValueError("content is empty")
        v = _vec(vec)
        meta = Jsonb(metadata or {})

        with self.pool.connection() as conn, conn.transaction():
            nearest = conn.execute(
                """
                SELECT id, 1 - (embedding <=> %(v)s::vector) AS score
                FROM memories
                WHERE user_id = %(u)s AND embedding_model = %(m)s AND kind = %(kind)s
                ORDER BY embedding <=> %(v)s::vector
                LIMIT 1
                """,
                {"v": v, "u": user_id, "m": model, "kind": kind},
            ).fetchone()

            if nearest and float(nearest["score"]) >= dedup_threshold:
                conn.execute(
                    """
                    UPDATE memories
                    SET use_count = use_count + 1, last_used_at = now(),
                        metadata = metadata || %s
                    WHERE id = %s AND user_id = %s
                    """,
                    (meta, nearest["id"], user_id),
                )
                return {"id": nearest["id"], "action": "deduplicated",
                        "similarity": round(float(nearest["score"]), 4)}

            row = conn.execute(
                """
                INSERT INTO memories
                    (user_id, kind, content, embedding, embedding_model, metadata, source_session)
                VALUES (%s, %s, %s, %s::vector, %s, %s, %s)
                RETURNING id
                """,
                (user_id, kind, content, v, model, meta, source_session),
            ).fetchone()
            return {"id": row["id"], "action": "inserted",
                    "similarity": round(float(nearest["score"]), 4) if nearest else None}

    # ------------------------------------------------------------------ admin helpers

    def list(self, user_id: str, limit: int = 100, kind: Optional[str] = None) -> list[dict]:
        clause, params = "", [user_id]
        if kind:
            clause = "AND kind = %s"
            params.append(kind)
        params.append(limit)
        with self.pool.connection() as conn:
            return conn.execute(
                f"""
                SELECT id, kind, content, metadata, embedding_model, use_count,
                       source_session, created_at, last_used_at
                FROM memories WHERE user_id = %s {clause}
                ORDER BY created_at DESC LIMIT %s
                """,
                params,
            ).fetchall()

    def delete_user(self, user_id: str) -> int:
        """'Forget me': delete every memory of one user. Returns rows deleted."""
        with self.pool.connection() as conn:
            cur = conn.execute("DELETE FROM memories WHERE user_id = %s", (user_id,))
            return cur.rowcount

    def stats(self) -> list[dict]:
        with self.pool.connection() as conn:
            return conn.execute(
                """
                SELECT user_id, embedding_model, kind, count(*) AS n
                FROM memories GROUP BY user_id, embedding_model, kind
                ORDER BY user_id, embedding_model, kind
                """
            ).fetchall()