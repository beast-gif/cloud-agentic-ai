"""
Agentic Cloud - Phase 1, Step 1.4
Session lifecycle: the state machine of the control plane.

    POST /sessions            -> create_session()  STARTING -> RUNNING -> WAITING | COMPLETED
    POST /sessions/{id}/events-> send_event()      WAITING  -> RUNNING            (fast path, container alive)
                                                   FROZEN   -> WAKING -> RUNNING  (wake: new container)
    idle reaper               -> freeze()          WAITING  -> FREEZING -> FROZEN
    startup                   -> reconcile()       fix sessions/containers after a crash

Two layers of protection against races (e.g. reaper freezing while a user event arrives):
  1. a per-session threading.Lock  -> operations on ONE session run one at a time
  2. compare-and-set SQL updates    -> even without the lock, only one actor can win a transition
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import httpx

from control_plane.auth import SessionAuth
from control_plane.docker_driver import DockerDriver
from control_plane.session_store import (
    COMPLETED, FAILED, FREEZING, FROZEN, RUNNING, STARTING, TERMINAL_STATES, WAITING, WAKING,
    SessionStore,
)

log = logging.getLogger("control_plane.lifecycle")

SUPPORTED_AGENTS = {"research"}


# ---- errors (mapped to HTTP status codes in main.py) -------------------------

class SessionError(Exception):
    status_code = 400


class SessionNotFound(SessionError):
    status_code = 404


class InvalidState(SessionError):
    status_code = 409


class AgentStartError(SessionError):
    status_code = 502


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


class SessionManager:
    def __init__(self, cfg: dict, store: SessionStore, driver: DockerDriver,
                 auth: Optional[SessionAuth] = None) -> None:
        self.store = store
        self.driver = driver
        # Phase 2: issues a fresh Memory-API token on every container start
        self.auth = auth
        self.idle_timeout_s = float(cfg["lifecycle"]["idle_timeout_s"])
        # Phase 2 (L2): how long a FINISHED session's workspace volume is kept for inspection
        self.workspace_keep_s = float((cfg.get("workspace") or {}).get("keep_after_complete_s", 3600))

        # agent calls can include several LLM calls -> generous read timeout
        llm_timeout = float((cfg["llm"].get("ollama") or {}).get("timeout_s", 120))
        self.http = httpx.Client(timeout=httpx.Timeout(llm_timeout * 3, connect=5.0))

        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._stats_lock = threading.Lock()
        self.stats = {"cold_starts": 0, "wakes": 0, "fast_path_events": 0,
                      "freezes": 0, "lost": 0, "failures": 0, "cleanups": 0,
                      "workspaces_deleted": 0}

        # Background workers that stop containers of finished sessions, so the user's
        # response does not wait for `docker stop` (~1 s).
        self._cleanup = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cleanup")

    def close(self) -> None:
        # wait=True: finish pending cleanups so no container is left running on shutdown
        self._cleanup.shutdown(wait=True)
        self.http.close()

    # ================================================================ helpers

    def _lock(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(session_id, threading.Lock())

    def _forget_lock(self, session_id: str) -> None:
        with self._locks_guard:
            self._locks.pop(session_id, None)

    def _inc(self, key: str) -> None:
        with self._stats_lock:
            self.stats[key] += 1

    # ================================================================ create (cold start)

    def create_session(self, user_id: str, agent_type: str, input_: dict) -> dict:
        if agent_type not in SUPPORTED_AGENTS:
            raise SessionError(f"unknown agent_type {agent_type!r}; supported: {sorted(SUPPORTED_AGENTS)}")

        t0 = time.perf_counter()
        row = self.store.create_session(user_id, agent_type, payload={"input": input_})
        sid = row["session_id"]

        with self._lock(sid):
            url, timings = self._start_container(sid, STARTING, kind="cold_start")
            res, code = self._call_agent(url, "/task", {"thread_id": row["thread_id"], "input": input_})
            total_ms = _ms(t0)
            self.store.record_wake_metrics(sid, 0, "cold_start", **timings, total_ms=total_ms)
            session = self._apply_result(sid, res, code)

        self._inc("cold_starts")
        return self._response(session, res, timings, total_ms, path="cold_start")

    # ================================================================ events (fast path / wake)

    def send_event(self, session_id: str, value: Any) -> dict:
        t0 = time.perf_counter()
        with self._lock(session_id):
            row = self.store.get(session_id)
            if row is None:
                raise SessionNotFound(f"session {session_id} not found")
            state = row["state"]

            # ---- fast path: container still alive, just resume it
            if state == WAITING:
                if self.store.transition(session_id, WAITING, RUNNING, event="user_event",
                                         payload={"path": "fast"}, touch_active=True) is None:
                    raise InvalidState(f"session {session_id} changed state; retry")
                res, code = self._call_agent(row["agent_url"], "/resume",
                                             {"thread_id": row["thread_id"], "value": value})
                # measured right after the agent answers - same point as the wake path
                total_ms = _ms(t0)
                session = self._apply_result(session_id, res, code)
                self._inc("fast_path_events")
                return self._response(session, res, None, total_ms, path="fast")

            # ---- wake: no container -> start a brand-new one, it loads the checkpoint
            if state == FROZEN:
                if self.store.transition(session_id, FROZEN, WAKING, event="user_event",
                                         payload={"path": "wake"}, increment_wake=True) is None:
                    raise InvalidState(f"session {session_id} changed state; retry")
                wake_no = int(row["wake_count"]) + 1
                url, timings = self._start_container(session_id, WAKING, kind="wake")
                res, code = self._call_agent(url, "/resume", {"thread_id": row["thread_id"], "value": value})
                total_ms = _ms(t0)
                self.store.record_wake_metrics(
                    session_id, wake_no, "wake", **timings,
                    checkpoint_load_ms=res.get("checkpoint_load_ms"), total_ms=total_ms,
                )
                session = self._apply_result(session_id, res, code)
                self._inc("wakes")
                return self._response(session, res, timings, total_ms, path="wake")

            if state in TERMINAL_STATES:
                raise InvalidState(f"session {session_id} is {state}")
            raise InvalidState(f"session {session_id} is busy ({state}); retry shortly")

    # ================================================================ freeze

    def freeze(self, session_id: str, reason: str = "idle", blocking: bool = False) -> bool:
        """WAITING -> FREEZING -> FROZEN. Returns True if this call froze the session.

        The reaper uses blocking=False: if someone holds the session lock (an event is
        being processed) it simply skips and tries again on the next tick.
        """
        lock = self._lock(session_id)
        if not lock.acquire(blocking=blocking):
            return False
        try:
            row = self.store.transition(session_id, WAITING, FREEZING, event="freeze",
                                        payload={"reason": reason})
            if row is None:
                return False
            stop_ms = self.driver.stop(row["container_id"]) if row["container_id"] else 0.0
            self.store.transition(session_id, FREEZING, FROZEN, event="frozen",
                                  payload={"stop_ms": stop_ms}, container_id=None, agent_url=None)
            self._inc("freezes")
            return True
        finally:
            lock.release()

    # ================================================================ liveness (used by the reaper)

    def check_liveness(self) -> int:
        """WAITING sessions whose container died (OOM, crash, docker rm) -> FROZEN.

        Safe because the agent's memory is in the Postgres checkpoint; the next event wakes it.
        """
        lost = 0
        for s in self.store.list_sessions(state=WAITING, limit=10_000):
            if s["agent_url"] and self.driver.check_alive(s["agent_url"]):
                continue
            lock = self._lock(s["session_id"])
            if not lock.acquire(blocking=False):
                continue
            try:
                if s["container_id"]:
                    self.driver.stop(s["container_id"])
                if self.store.transition(s["session_id"], WAITING, FROZEN, event="lost",
                                         payload={"reason": "container not responding"},
                                         container_id=None, agent_url=None):
                    lost += 1
                    self._inc("lost")
            finally:
                lock.release()
        return lost

    # ================================================================ reconciliation

    def reconcile(self) -> dict:
        """Make the sessions table and the real Docker containers agree (run on startup).

        WAITING + container alive          -> keep
        WAITING / FREEZING / WAKING        -> FROZEN  (agent was paused at an interrupt: checkpoint is safe)
        STARTING / RUNNING                 -> FAILED  (a run was in progress when we crashed: lost)
        container without a live session   -> remove (orphan, wasting resources)
        """
        report: dict[str, list] = {"kept": [], "frozen": [], "failed": [], "orphans_removed": []}
        containers = {c["session_id"]: c for c in self.driver.list_managed()}
        live_ids: set[str] = set()

        for s in self.store.find_live():
            sid, state = s["session_id"], s["state"]
            live_ids.add(sid)
            c = containers.get(sid)
            running = (c is not None and c["status"] == "running"
                       and c["container_id"] == s["container_id"])

            if state == WAITING and running and s["agent_url"] and self.driver.check_alive(s["agent_url"]):
                report["kept"].append(sid)
                continue

            if c is not None:
                self.driver.stop(c["container_id"])

            if state in (WAITING, FREEZING, WAKING):
                self.store.transition(sid, state, FROZEN, event="reconciled", payload={"was": state},
                                      container_id=None, agent_url=None)
                report["frozen"].append(sid)
            else:
                self.store.transition(sid, state, FAILED, event="reconciled", payload={"was": state},
                                      error="control plane restarted while the agent was running",
                                      container_id=None, agent_url=None)
                report["failed"].append(sid)

        for sid, c in containers.items():
            if sid not in live_ids:
                self.driver.stop(c["container_id"])
                report["orphans_removed"].append(c["name"])

        # L2: remove workspace volumes of finished/unknown sessions (respecting retention)
        report["workspaces_deleted"] = self.gc_workspaces()

        log.info("reconcile: %s", {k: len(v) for k, v in report.items()})
        return report

    # ================================================================ L2 workspace garbage collection

    def gc_workspaces(self, older_than_s: Optional[float] = None) -> list[str]:
        """Delete workspace volumes that are no longer needed.

        orphan (no session row)                                  -> delete now
        session COMPLETED/FAILED, finished > keep_after_complete_s ago -> delete
        session FROZEN / WAITING / RUNNING / ...                 -> KEEP (it will wake again)

        older_than_s overrides the retention (e.g. 0 = delete all finished now).
        """
        keep_s = self.workspace_keep_s if older_than_s is None else float(older_than_s)
        deleted: list[str] = []
        for ws in self.driver.list_workspaces():
            sid = ws["session_id"]
            if not sid:
                continue  # not one of ours (no session label)
            row = self.store.get(sid)

            if row is None:
                reason = "orphan"
            elif row["state"] in TERMINAL_STATES:
                age_s = (time.time() - row["updated_at"].timestamp())
                if age_s < keep_s:
                    continue
                reason = f"{row['state'].lower()} {age_s:.0f}s ago"
            else:
                continue  # session can still wake -> its files must stay

            if self.driver.delete_workspace(sid):
                deleted.append(ws["name"])
                self._inc("workspaces_deleted")
                if row is not None:
                    self.store.log_event(sid, "workspace_deleted", {"volume": ws["name"], "reason": reason})
        if deleted:
            log.info("workspace gc: deleted %d volume(s)", len(deleted))
        return deleted

    # ================================================================ internals

    def _start_container(self, session_id: str, from_state: str, kind: str) -> tuple[str, dict]:
        """Start a container, wait for /health, move from_state -> RUNNING. Returns (agent_url, timings)."""
        t0 = time.perf_counter()
        # New token for every container (cold start AND wake): the previous container's
        # token stops working the moment this one is issued.
        extra_env = {"MEMORY_TOKEN": self.auth.issue(session_id)} if self.auth else None
        try:
            started = self.driver.start(session_id, extra_env=extra_env)
        except Exception as exc:  # noqa: BLE001
            self.store.transition(session_id, from_state, FAILED, event="failed",
                                  payload={"stage": kind}, error=f"docker start failed: {exc!r}")
            self._inc("failures")
            raise AgentStartError(f"docker start failed: {exc!r}") from exc

        health = self.driver.wait_healthy(started.agent_url, started.container_id)
        if not health.ok:
            self.driver.stop(started.container_id)
            self.store.transition(session_id, from_state, FAILED, event="failed",
                                  payload={"stage": kind}, error=health.error)
            self._inc("failures")
            raise AgentStartError(health.error or "agent failed to become healthy")

        row = self.store.transition(
            session_id, from_state, RUNNING,
            event="started" if kind == "cold_start" else "resumed",
            payload={"agent_url": started.agent_url, "polls": health.polls,
                     "workspace": started.workspace, "workspace_created": started.workspace_created},
            touch_active=True, container_id=started.container_id, agent_url=started.agent_url,
        )
        if row is None:
            self.driver.stop(started.container_id)
            raise InvalidState(f"session {session_id} changed state during start")

        t = health.body.get("timings_ms", {})
        timings = {
            "container_start_ms": t.get("container_start_ms"),
            "import_ms": t.get("import_ms"),
            "graph_build_ms": t.get("graph_build_ms"),
            "ready_ms": _ms(t0),
        }
        if t.get("firecracker"):   # Phase 5: restored from golden.snap -> restore/bind breakdown
            timings.update({"db_connect_ms": t.get("db_connect_ms"), "bind_ms": t.get("bind_ms"),
                            "firecracker": t["firecracker"]})
        return started.agent_url, timings

    def _call_agent(self, agent_url: str, path: str, body: dict) -> tuple[dict, int]:
        try:
            resp = self.http.post(f"{agent_url}{path}", json=body)
        except httpx.HTTPError as exc:
            return {"status": "failed", "error": f"agent unreachable: {exc!r}"}, 0
        try:
            return resp.json(), resp.status_code
        except ValueError:
            return {"status": "failed", "error": resp.text[:500]}, resp.status_code

    def _apply_result(self, session_id: str, res: dict, code: int) -> dict:
        """Translate the agent's answer into a state transition."""
        status = res.get("status")

        if code == 200 and status == "waiting":
            self.store.transition(session_id, RUNNING, WAITING, event="waiting",
                                        payload={"waiting_for": res.get("waiting_for")},
                                        touch_active=True, waiting_for=res.get("waiting_for"))
        elif code == 200 and status == "completed":
            self.store.transition(session_id, RUNNING, COMPLETED, event="completed",
                                        touch_active=True, waiting_for=None)
            self._release_container_async(session_id)
            self._forget_lock(session_id)
        else:
            error = res.get("error") or f"agent returned HTTP {code}"
            self.store.transition(session_id, RUNNING, FAILED, event="failed",
                                  payload={"http_status": code}, error=error)
            self._release_container_async(session_id)
            self._inc("failures")

        return self.store.get(session_id)

    def _release_container_async(self, session_id: str) -> None:
        """Schedule the container stop in the background; the caller returns immediately.

        Safe because the session is already COMPLETED/FAILED (terminal): no other
        operation will touch it, so nothing races with the cleanup.
        """
        self._cleanup.submit(self._release_container, session_id)

    def _release_container(self, session_id: str) -> None:
        """Stop the container of a finished session and clear its container fields."""
        try:
            if self.auth:
                self.auth.revoke(session_id)  # finished session: token no longer valid
            row = self.store.get(session_id)
            if row and row["container_id"]:
                stop_ms = self.driver.stop(row["container_id"])
                self.store.transition(session_id, row["state"], row["state"],
                                      event="cleaned_up", payload={"stop_ms": stop_ms},
                                      container_id=None, agent_url=None)
                self._inc("cleanups")
            # L2: retention 0 -> delete the workspace right away; otherwise the reaper's
            # periodic gc_workspaces() removes it after keep_after_complete_s.
            if self.workspace_keep_s <= 0 and self.driver.delete_workspace(session_id):
                self._inc("workspaces_deleted")
                self.store.log_event(session_id, "workspace_deleted",
                                     {"volume": self.driver.workspace_name(session_id),
                                      "reason": "retention 0"})
        except Exception:  # noqa: BLE001  (background thread: log, never crash)
            log.exception("cleanup of %s failed", session_id)

    @staticmethod
    def _response(session: dict, res: dict, timings: Optional[dict], total_ms: float, path: str) -> dict:
        return {
            "path": path,  # cold_start | fast | wake
            # never expose the token hash through the API
            "session": {k: v for k, v in (session or {}).items() if k != "token_hash"},
            "agent": {k: res.get(k) for k in
                      ("status", "waiting_for", "data", "error", "run_ms", "checkpoint_load_ms",
                       "llm", "memory", "mcp")},
            "timings_ms": {**(timings or {}), "total_ms": total_ms},
        }