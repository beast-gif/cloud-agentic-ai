"""
Agentic Cloud - Phase 1, Step 1.4
Control plane API.

Run from the agentic-cloud/ folder:
  python -m control_plane.main

Endpoints:
  POST /sessions                     create a session + cold start  {"user_id", "agent_type", "input": {"topic"}}
  POST /sessions/{id}/events         send the answer the agent waits for  {"value": {"selected": [1, 3]}}
  POST /sessions/{id}/freeze         freeze now (testing; normally the reaper does it)
  GET  /sessions                     list (?state=WAITING&user_id=amrith)
  GET  /sessions/{id}                one session
  GET  /sessions/{id}/events         audit log
  GET  /sessions/{id}/metrics        wake metrics of one session
  GET  /metrics/wake                 all wake metrics + averages per kind
  POST /admin/reconcile              run reconciliation again
  GET  /healthz                      control plane status
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from statistics import mean
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from control_plane.docker_driver import DockerDriver
from control_plane.idle_reaper import IdleReaper
from control_plane.lifecycle import SessionError, SessionManager, SessionNotFound
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
    startup_reconcile: dict


S = Services()


@asynccontextmanager
async def lifespan(_: FastAPI):
    S.store = SessionStore(CFG["postgres"]["host_dsn"])
    S.driver = DockerDriver.from_config(CFG, base_env=agent_env(CFG))
    S.driver.ensure_ready()
    S.manager = SessionManager(CFG, S.store, S.driver)

    # fix anything left over from a previous run BEFORE accepting traffic
    S.startup_reconcile = S.manager.reconcile()

    lc = CFG["lifecycle"]
    S.reaper = IdleReaper(
        S.manager, S.store,
        idle_timeout_s=float(lc["idle_timeout_s"]),
        interval_s=float(lc["reaper_interval_s"]),
    )
    S.reaper.start()
    log.info("control plane ready")
    yield
    S.reaper.stop()
    S.manager.close()
    S.driver.close()
    S.store.close()
    log.info("control plane stopped")


app = FastAPI(title="Agentic Cloud - control plane (Phase 1)", lifespan=lifespan)


@app.exception_handler(SessionError)
async def _session_error(_: Request, exc: SessionError):
    return JSONResponse({"error": str(exc), "type": type(exc).__name__}, status_code=exc.status_code)


# ---- request models ----------------------------------------------------------

class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., examples=["amrith"])
    agent_type: str = Field("research", examples=["research"])
    input: dict[str, Any] = Field(..., examples=[{"topic": "KV cache compression"}])


class EventRequest(BaseModel):
    value: Any = Field(..., examples=[{"selected": [1, 3]}])


# ---- sessions ----------------------------------------------------------------
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
    return {"frozen": frozen, "session": S.store.get(session_id)}


@app.get("/sessions")
def list_sessions(state: Optional[str] = None, user_id: Optional[str] = None, limit: int = 100):
    return S.store.list_sessions(state=state, user_id=user_id, limit=limit)


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    row = S.store.get(session_id)
    if row is None:
        raise SessionNotFound(f"session {session_id} not found")
    return row


@app.get("/sessions/{session_id}/events")
def session_events(session_id: str):
    return S.store.events(session_id)


@app.get("/sessions/{session_id}/metrics")
def session_metrics(session_id: str):
    return S.store.wake_metrics(session_id)


# ---- metrics / admin -----------------------------------------------------------

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


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "reaper": S.reaper.status(),
        "manager_stats": S.manager.stats,
        "startup_reconcile": S.startup_reconcile,
    }


if __name__ == "__main__":
    cp = CFG["control_plane"]
    uvicorn.run(app, host=cp["host"], port=int(cp["port"]), log_level="info")