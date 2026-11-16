"""
Agentic Cloud - control plane API (Phase 1 + Phase 2 Step 2.2 Memory API).

Run from the agentic-cloud/ folder:
  python -m control_plane.main

Sessions (Phase 1):
  POST /sessions                     create a session + cold start  {"user_id", "agent_type", "input": {"topic"}}
  POST /sessions/{id}/events         send the answer the agent waits for  {"value": {"selected": [1, 3]}}
  POST /sessions/{id}/freeze         freeze now (testing; normally the reaper does it)
  GET  /sessions                     list (?state=WAITING&user_id=amrith)
  GET  /sessions/{id}                one session
  GET  /sessions/{id}/events         audit log
  GET  /sessions/{id}/metrics        wake metrics of one session
  GET  /metrics/wake                 all wake metrics + averages per kind

Memory API for AGENTS (Phase 2) - needs header  Authorization: Bearer <MEMORY_TOKEN>
  POST /memory/search                {"query", "k"?, "min_score"?, "kinds"?}
  POST /memory/write                 {"kind", "content", "metadata"?}

Memory admin (local testing only, no token):
  POST   /admin/memory/search        {"user_id", "query", ...}
  POST   /admin/memory/write         {"user_id", "kind", "content", "metadata"?}
  GET    /users/{user_id}/memories   list
  DELETE /users/{user_id}/memories   forget everything about a user
  GET    /admin/memory/stats         counts per user / model / kind

L2 workspaces (Phase 2, Step 2.3):
  GET  /sessions/{id}/workspace      list files in the session's volume (?content=true to read them)
  GET  /admin/workspaces             all workspace volumes
  POST /admin/workspaces/gc          delete finished/orphan volumes now (?older_than_s=0 ignores retention)

Other:
  POST /admin/reconcile              run reconciliation again
  GET  /healthz                      control plane status
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from statistics import mean
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from control_plane.auth import SessionAuth
from control_plane.docker_driver import DockerDriver
from control_plane.embeddings import Embedder, make_embedder, provider_settings
from control_plane.idle_reaper import IdleReaper
from control_plane.lifecycle import SessionError, SessionManager, SessionNotFound
from control_plane.memory_store import KINDS, MemoryStore
from control_plane.migrations import apply_migrations
from control_plane.session_store import SessionStore
from control_plane.settings import agent_env, load_config

CFG = load_config()

logging.basicConfig(
    level=CFG.get("logging", {}).get("level", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide the per-poll /health lines
log = logging.getLogger("control_plane.main")


class Services:
    store: SessionStore
    driver: DockerDriver
    manager: SessionManager
    reaper: IdleReaper
    auth: SessionAuth
    embedder: Embedder
    memory: MemoryStore
    mem_settings: dict
    startup_reconcile: dict
    migrations_applied: list


S = Services()


@asynccontextmanager
async def lifespan(_: FastAPI):
    S.store = SessionStore(CFG["postgres"]["host_dsn"])

    # Phase 2: bring the schema up to date before anything else touches the DB
    S.migrations_applied = apply_migrations(S.store.pool)

    S.auth = SessionAuth(S.store.pool)
    S.memory = MemoryStore(S.store.pool)
    S.embedder = make_embedder(CFG)
    S.mem_settings = provider_settings(CFG)
    log.info("embeddings: %s (min_score=%s, dedup=%s)", S.embedder.name,
             S.mem_settings["min_score"], S.mem_settings["dedup_threshold"])

    S.driver = DockerDriver.from_config(CFG, base_env=agent_env(CFG))
    S.driver.ensure_ready()
    S.manager = SessionManager(CFG, S.store, S.driver, auth=S.auth)

    # fix anything left over from a previous run BEFORE accepting traffic
    S.startup_reconcile = S.manager.reconcile()

    lc = CFG["lifecycle"]
    S.reaper = IdleReaper(
        S.manager, S.store,
        idle_timeout_s=float(lc["idle_timeout_s"]),
        interval_s=float(lc["reaper_interval_s"]),
        workspace_gc_every=int((CFG.get("workspace") or {}).get("gc_every_ticks", 12)),
    )
    S.reaper.start()
    log.info("control plane ready")
    yield
    S.reaper.stop()
    S.manager.close()
    S.driver.close()
    S.store.close()
    log.info("control plane stopped")


app = FastAPI(title="Agentic Cloud - control plane", lifespan=lifespan)


@app.exception_handler(SessionError)
async def _session_error(_: Request, exc: SessionError):
    return JSONResponse({"error": str(exc), "type": type(exc).__name__}, status_code=exc.status_code)


# ============================================================ request models

class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., examples=["amrith"])
    agent_type: str = Field("research", examples=["research"])
    input: dict[str, Any] = Field(..., examples=[{"topic": "KV cache compression"}])


class EventRequest(BaseModel):
    value: Any = Field(..., examples=[{"selected": [1, 3]}])


class MemorySearchRequest(BaseModel):
    query: str = Field(..., examples=["KV cache compression"])
    k: int = Field(5, ge=1, le=50)
    min_score: Optional[float] = Field(None, ge=-1.0, le=1.0,
                                       description="default: the active provider's min_score")
    kinds: Optional[list[str]] = Field(None, examples=[["paper_read"]])


class MemoryWriteRequest(BaseModel):
    kind: str = Field(..., examples=["paper_read"])
    content: str = Field(..., examples=["Read 'Adaptive KV cache compression' (EMNLP 2021): ..."])
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdminMemorySearchRequest(MemorySearchRequest):
    user_id: str = Field(..., examples=["amrith"])


class AdminMemoryWriteRequest(MemoryWriteRequest):
    user_id: str = Field(..., examples=["amrith"])


# ============================================================ sessions (Phase 1)
# Plain `def` endpoints: FastAPI runs them in a thread pool, so slow agent calls
# do not block other requests.

@app.post("/sessions", status_code=201)
def create_session(req: CreateSessionRequest):
    return S.manager.create_session(req.user_id, req.agent_type, req.input)


@app.post("/sessions/{session_id}/events")
def send_event(session_id: str, req: EventRequest):
    return S.manager.send_event(session_id, req.value)


@app.post("/sessions/{session_id}/freeze")
def freeze_session(session_id: str):
    if S.store.get(session_id) is None:
        raise SessionNotFound(f"session {session_id} not found")
    frozen = S.manager.freeze(session_id, reason="manual", blocking=True)
    return {"frozen": frozen, "session": _public_session(S.store.get(session_id))}


@app.get("/sessions")
def list_sessions(state: Optional[str] = None, user_id: Optional[str] = None, limit: int = 100):
    return [_public_session(r) for r in S.store.list_sessions(state=state, user_id=user_id, limit=limit)]


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    row = S.store.get(session_id)
    if row is None:
        raise SessionNotFound(f"session {session_id} not found")
    return _public_session(row)


@app.get("/sessions/{session_id}/events")
def session_events(session_id: str):
    return S.store.events(session_id)


@app.get("/sessions/{session_id}/metrics")
def session_metrics(session_id: str):
    return S.store.wake_metrics(session_id)


@app.get("/sessions/{session_id}/workspace")
def session_workspace(session_id: str, content: bool = False):
    """Files in the session's L2 workspace volume - works in ANY state, including FROZEN."""
    row = S.store.get(session_id)
    if row is None:
        raise SessionNotFound(f"session {session_id} not found")
    try:
        files = S.driver.list_workspace_files(session_id, with_content=content)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {
        "session_id": session_id,
        "state": row["state"],
        "volume": S.driver.workspace_name(session_id),
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files),
    }


def _public_session(row: Optional[dict]) -> Optional[dict]:
    """Never return token hashes through the API."""
    if row is None:
        return None
    return {k: v for k, v in row.items() if k != "token_hash"}


# ============================================================ memory core (shared by agent + admin endpoints)

def _embed(text: str, mode: str) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    try:
        vec = S.embedder.embed([text], mode=mode)[0]
    except httpx.HTTPError as exc:
        raise HTTPException(503, f"embedding provider {S.embedder.name} unavailable: {exc!r}") from exc
    return vec, round((time.perf_counter() - t0) * 1000.0, 1)


def _search(user_id: str, req: MemorySearchRequest) -> dict:
    vec, embed_ms = _embed(req.query, mode="query")
    min_score = req.min_score if req.min_score is not None else S.mem_settings["min_score"]
    t0 = time.perf_counter()
    try:
        hits = S.memory.search(user_id, vec, S.embedder.name, k=req.k,
                               min_score=min_score, kinds=req.kinds)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    search_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return {
        "user_id": user_id,
        "model": S.embedder.name,
        "min_score": min_score,
        "hits": hits,
        "timings_ms": {"embed_ms": embed_ms, "search_ms": search_ms},
    }


def _write(user_id: str, req: MemoryWriteRequest, source_session: Optional[str] = None) -> dict:
    if req.kind not in KINDS:
        raise HTTPException(400, f"unknown kind {req.kind!r}; allowed: {list(KINDS)}")
    vec, embed_ms = _embed(req.content, mode="document")
    t0 = time.perf_counter()
    try:
        result = S.memory.write(user_id, req.kind, req.content, vec, S.embedder.name,
                                metadata=req.metadata, source_session=source_session,
                                dedup_threshold=S.mem_settings["dedup_threshold"])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    write_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return {"user_id": user_id, "model": S.embedder.name, **result,
            "timings_ms": {"embed_ms": embed_ms, "write_ms": write_ms}}


def _session_from_token(authorization: Optional[str]) -> dict:
    """'Bearer <token>' -> session row. The user_id ALWAYS comes from here, never from the request body."""
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    session = S.auth.resolve(token)
    if session is None:
        raise HTTPException(401, "invalid, expired or missing session token")
    return session


# ============================================================ memory API for agents (token required)

@app.post("/memory/search")
def memory_search(req: MemorySearchRequest, authorization: Optional[str] = Header(None)):
    session = _session_from_token(authorization)
    out = _search(session["user_id"], req)
    S.store.log_event(session["session_id"], "memory_search",
                      {"query": req.query[:200], "hits": len(out["hits"]), **out["timings_ms"]})
    return {"session_id": session["session_id"], **out}


@app.post("/memory/write")
def memory_write(req: MemoryWriteRequest, authorization: Optional[str] = Header(None)):
    session = _session_from_token(authorization)
    out = _write(session["user_id"], req, source_session=session["session_id"])
    S.store.log_event(session["session_id"], "memory_write",
                      {"kind": req.kind, "action": out["action"], "memory_id": out["id"]})
    return {"session_id": session["session_id"], **out}


# ============================================================ memory admin (local testing, no token)

@app.post("/admin/memory/search")
def admin_memory_search(req: AdminMemorySearchRequest):
    return _search(req.user_id, req)


@app.post("/admin/memory/write")
def admin_memory_write(req: AdminMemoryWriteRequest):
    return _write(req.user_id, req)


@app.get("/users/{user_id}/memories")
def list_memories(user_id: str, kind: Optional[str] = None, limit: int = 100):
    return S.memory.list(user_id, limit=limit, kind=kind)


@app.delete("/users/{user_id}/memories")
def forget_user(user_id: str):
    return {"user_id": user_id, "deleted": S.memory.delete_user(user_id)}


@app.get("/admin/memory/stats")
def memory_stats():
    return {"embedder": S.embedder.stats(), "settings": S.mem_settings, "counts": S.memory.stats()}


# ============================================================ metrics / admin

def _avg(rows: list[dict], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(mean(vals), 1) if vals else None


@app.get("/metrics/wake")
def wake_metrics(limit: int = 500):
    rows = S.store.wake_metrics(limit=limit)
    summary = {}
    for kind in ("cold_start", "wake"):
        k_rows = [r for r in rows if r["kind"] == kind]
        summary[kind] = {
            "count": len(k_rows),
            **{f"avg_{f}": _avg(k_rows, f) for f in
               ("container_start_ms", "import_ms", "graph_build_ms",
                "checkpoint_load_ms", "ready_ms", "total_ms")},
        }
    return {"summary": summary, "rows": rows}


@app.post("/admin/reconcile")
def reconcile():
    return S.manager.reconcile()


@app.get("/admin/workspaces")
def list_workspaces():
    out = []
    for ws in S.driver.list_workspaces():
        row = S.store.get(ws["session_id"]) if ws["session_id"] else None
        out.append({**ws, "session_state": row["state"] if row else None})
    return out


@app.post("/admin/workspaces/gc")
def gc_workspaces(older_than_s: Optional[float] = None):
    return {"deleted": S.manager.gc_workspaces(older_than_s=older_than_s)}


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "reaper": S.reaper.status(),
        "manager_stats": S.manager.stats,
        "startup_reconcile": S.startup_reconcile,
        "migrations_applied_at_startup": S.migrations_applied,
        "embedder": S.embedder.stats(),
    }


if __name__ == "__main__":
    cp = CFG["control_plane"]
    uvicorn.run(app, host=cp["host"], port=int(cp["port"]), log_level="info")