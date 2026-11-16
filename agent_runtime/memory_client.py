"""
Agentic Cloud - Phase 2, Step 2.4
Agent-side client for the control plane's Memory API (L3 long-term memory).

The agent knows only two things (environment variables set by the control plane):
  MEMORY_API_URL  e.g. http://host.docker.internal:8000
  MEMORY_TOKEN    this container's session token (rotated on every wake)

It never knows its user_id and never touches the memories table directly:
the control plane resolves token -> session -> user_id on every call.

Memory is BEST-EFFORT: if the API is unreachable or the token is rejected, calls
return empty results and the agent continues normally (it just "forgets").
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

log = logging.getLogger("agent.memory")


class MemoryClient:
    def __init__(self, base_url: str, token: str, timeout_s: float = 30.0) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.token = token or ""
        self.enabled = bool(self.base_url and self.token)
        self._http = httpx.Client(
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {self.token}"},
        ) if self.enabled else None

        self.calls = 0
        self.errors = 0
        self.total_ms = 0.0
        if not self.enabled:
            log.warning("memory disabled (MEMORY_API_URL or MEMORY_TOKEN not set)")

    # ------------------------------------------------------------------ API

    def search(self, query: str, k: int = 10, kinds: Optional[list[str]] = None,
               min_score: Optional[float] = None) -> list[dict]:
        body: dict[str, Any] = {"query": query, "k": k}
        if kinds:
            body["kinds"] = kinds
        if min_score is not None:
            body["min_score"] = min_score
        data = self._post("/memory/search", body)
        return (data or {}).get("hits", [])

    def write(self, kind: str, content: str, metadata: Optional[dict] = None) -> Optional[dict]:
        return self._post("/memory/write", {"kind": kind, "content": content, "metadata": metadata or {}})

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "calls": self.calls,
            "errors": self.errors,
            "total_ms": round(self.total_ms, 1),
            "avg_ms": round(self.total_ms / self.calls, 1) if self.calls else 0.0,
        }

    def close(self) -> None:
        if self._http is not None:
            self._http.close()

    # ------------------------------------------------------------------ internals

    def _post(self, path: str, body: dict) -> Optional[dict]:
        if not self.enabled:
            return None
        t0 = time.perf_counter()
        try:
            resp = self._http.post(f"{self.base_url}{path}", json=body)
            if resp.status_code != 200:
                self.errors += 1
                log.warning("memory %s -> HTTP %s: %s", path, resp.status_code, resp.text[:200])
                return None
            return resp.json()
        except httpx.HTTPError as exc:
            self.errors += 1
            log.warning("memory %s failed: %r", path, exc)
            return None
        finally:
            self.calls += 1
            self.total_ms += (time.perf_counter() - t0) * 1000.0