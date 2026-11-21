"""
Agentic Cloud - Phase 1, Step 1.2
Agent server: runs INSIDE every agent container.

Startup is split in two on purpose:
  1. Light part (FastAPI + uvicorn) -> the server starts listening quickly.
  2. Heavy part in a background thread (LangGraph imports, Postgres checkpointer,
     graph build). While it runs, GET /health returns 503 "starting".
     When it finishes, /health returns 200 "ok" with a timing breakdown.

Endpoints:
  GET  /health              readiness/liveness + startup timings
  POST /task                start a new run:   {"thread_id": "...", "input": {"topic": "..."}}
  POST /resume              continue a paused run: {"thread_id": "...", "value": {"selected": [1, 4]}}
  GET  /state/{thread_id}   debug: current checkpointed state

Environment variables (set by the control plane in Step 1.3):
  AGENT_DSN           Postgres DSN reachable from the container (required)
  SESSION_ID          control-plane session id (for logs)
  REQUESTED_AT_MS     epoch ms when the control plane asked Docker to start this container
  PORT                default 8080
  LLM_PROVIDER        mock | ollama | router   (default mock; Phase 4: router)
  LLM_ROUTER_URL      Phase 4: control plane base URL for /llm/rank + /llm/complete
  LLM_ROUTER_TIMEOUT_S Phase 4: default 180 (local embeddings / local summaries are slow on CPU)
  OLLAMA_BASE_URL     default http://host.docker.internal:11434
  OLLAMA_MODEL        default llama3.2:3b
  OLLAMA_TEMPERATURE  default 0.2
  OLLAMA_TIMEOUT_S    default 120
  MOCK_LATENCY_MS     default 150 (fake LLM delay per call)
  LOG_LEVEL           default INFO
  MEMORY_API_URL      Phase 2: control plane Memory API (L3)
  MEMORY_TOKEN        Phase 2: this container's session token (rotated on every wake)
                      Phase 3: the same token authenticates the agent to the MCP gateway
  MCP_GATEWAY_URL     Phase 3: the control plane's MCP gateway, e.g. http://host.docker.internal:8000/mcp
  WORKSPACE_DIR       Phase 2: L2 workspace mount, default /workspace
  FC_PREBIND          Phase 5: "1" = import everything, then wait for POST /bind (golden snapshot)
  FC_GC_FREEZE        Phase 5: "1" (default) = gc.freeze() before the snapshot (copy-on-write friendly)

Phase 5 endpoint:
  POST /bind                {"env": {"AGENT_DSN": ..., "SESSION_ID": ..., "MEMORY_TOKEN": ...}}
"""

# ---- timing starts before ANY other import ---------------------------------
import time

PROCESS_START_EPOCH_MS = time.time() * 1000.0
_T_PROCESS = time.perf_counter()

import logging  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from typing import Any, Optional  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

_T_WEB_IMPORTED = time.perf_counter()

# ---- configuration -----------------------------------------------------------

AGENT_DSN = os.environ.get("AGENT_DSN", "")
SESSION_ID = os.environ.get("SESSION_ID", "local")
PORT = int(os.environ.get("PORT", "8080"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
# Phase 5: Firecracker golden snapshot mode (see _init_agent)
PREBIND = os.environ.get("FC_PREBIND", "0") == "1"

_requested_at = os.environ.get("REQUESTED_AT_MS")
REQUESTED_AT_MS: Optional[float] = float(_requested_at) if _requested_at else None

logging.basicConfig(
    level=LOG_LEVEL,
    format=f"%(asctime)s %(levelname)s [{SESSION_ID}] %(name)s: %(message)s",
)
log = logging.getLogger("agent.server")


def _ms(a: float, b: float) -> float:
    return round((b - a) * 1000.0, 1)


# ---- runtime holder (filled by the background init thread) -------------------


class Runtime:
    def __init__(self) -> None:
        self.ready = False
        self.error: Optional[str] = None
        self.graph = None
        self.pool = None
        self.llm = None
        self.memory = None
        self.mcp = None
        self.Command = None
        self.timings: dict[str, Any] = {}
        self.prebind = False                 # Phase 5: imports done, waiting for /bind
        self.bind_lock = threading.Lock()


RT = Runtime()


class _Mods:
    """Modules imported by the heavy stage (kept so /bind can use them later)."""
    dict_row = ConnectionPool = PostgresSaver = Command = None
    build_graph = LLM = MCPClient = MemoryClient = None


def _import_stage() -> float:
    """Stage 1: heavy imports. Independent of any session -> safe to snapshot (Phase 5)."""
    t0 = time.perf_counter()
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.types import Command

    from graph import build_graph
    from llm import LLM
    from mcp_client import MCPClient
    from memory_client import MemoryClient
    _Mods.dict_row, _Mods.ConnectionPool, _Mods.PostgresSaver, _Mods.Command = (
        dict_row, ConnectionPool, PostgresSaver, Command)
    _Mods.build_graph, _Mods.LLM, _Mods.MCPClient, _Mods.MemoryClient = (
        build_graph, LLM, MCPClient, MemoryClient)
    return time.perf_counter() - t0


def _bind_stage() -> tuple[float, float]:
    """Stage 2: everything that depends on THIS session's environment:
    Postgres connection + checkpointer, LLM/Memory/MCP clients (token), graph build.
    Reads os.environ at call time, so /bind can set it first. Returns (db_s, graph_s)."""
    dsn = os.environ.get("AGENT_DSN", "")
    if not dsn:
        raise RuntimeError("AGENT_DSN environment variable is not set")
    t1 = time.perf_counter()
    pool = _Mods.ConnectionPool(
        conninfo=dsn,
        min_size=1,
        max_size=4,
        # settings required by PostgresSaver
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": _Mods.dict_row},
        open=True,
    )
    pool.wait(timeout=30)
    checkpointer = _Mods.PostgresSaver(pool)
    checkpointer.setup()  # creates LangGraph tables on first run, no-op afterwards
    t2 = time.perf_counter()

    llm = _Mods.LLM(
        provider=os.environ.get("LLM_PROVIDER", "mock"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
        temperature=float(os.environ.get("OLLAMA_TEMPERATURE", "0.2")),
        timeout_s=float(os.environ.get("OLLAMA_TIMEOUT_S", "120")),
        mock_latency_ms=float(os.environ.get("MOCK_LATENCY_MS", "150")),
        # Phase 4: the model router authenticates the agent with the same session token
        router_url=os.environ.get("LLM_ROUTER_URL", ""),
        router_token=os.environ.get("MEMORY_TOKEN", ""),
        router_timeout_s=float(os.environ.get("LLM_ROUTER_TIMEOUT_S", "180")),
    )
    memory = _Mods.MemoryClient(
        base_url=os.environ.get("MEMORY_API_URL", ""),
        token=os.environ.get("MEMORY_TOKEN", ""),
    )
    mcp = _Mods.MCPClient(
        url=os.environ.get("MCP_GATEWAY_URL", ""),
        token=os.environ.get("MEMORY_TOKEN", ""),
    )
    graph = _Mods.build_graph(
        llm, checkpointer, memory=memory, mcp=mcp,
        workspace_dir=os.environ.get("WORKSPACE_DIR", "/workspace"),
    )
    t3 = time.perf_counter()

    RT.pool, RT.llm, RT.graph, RT.Command = pool, llm, graph, _Mods.Command
    RT.memory = memory
    RT.mcp = mcp
    return t2 - t1, t3 - t2


def _init_agent() -> None:
    """Heavy initialization. Runs in a background thread so /health can answer 503 meanwhile.

    Normal (Docker, Phase 1-4): imports -> bind -> ready.
    Pre-bind (Firecracker golden snapshot, Phase 5, env FC_PREBIND=1): imports -> gc.freeze()
    -> STOP and wait for POST /bind. The VM is snapshotted in this state, so every restored
    clone starts with all imports done and no session data / no open connections inside.
    """
    try:
        import_s = _import_stage()
        RT.timings = {"import_ms": round(_ms(_T_PROCESS, _T_WEB_IMPORTED) + import_s * 1000.0, 1)}

        if PREBIND:
            import gc
            if os.environ.get("FC_GC_FREEZE", "1") == "1":
                # Move every object that exists now into a permanent generation the GC never
                # scans again. Otherwise the first GC pass after restore writes to (and thereby
                # copies) nearly every page holding imported objects -> no copy-on-write sharing.
                gc.collect()
                gc.freeze()
                RT.timings["gc_frozen_objects"] = gc.get_freeze_count()
            RT.prebind = True
            log.info("pre-bind: imports done, waiting for POST /bind (%s)", RT.timings)
            return

        db_s, graph_s = _bind_stage()
        container_start_ms = (
            round(PROCESS_START_EPOCH_MS - REQUESTED_AT_MS, 1) if REQUESTED_AT_MS else None
        )
        RT.timings.update({
            # docker run requested -> python process started (host and VM clocks may differ slightly)
            "container_start_ms": container_start_ms,
            "db_connect_ms": round(db_s * 1000.0, 1),
            "graph_build_ms": round(graph_s * 1000.0, 1),
            # python process start -> agent ready
            "process_to_ready_ms": _ms(_T_PROCESS, time.perf_counter()),
        })
        RT.ready = True
        log.info("agent ready: %s", RT.timings)

    except Exception as exc:  # noqa: BLE001
        RT.error = repr(exc)
        log.exception("agent initialization failed")


# ---- FastAPI app -------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_init_agent, name="agent-init", daemon=True).start()
    yield
    # graceful shutdown (docker stop sends SIGTERM -> uvicorn -> here)
    if RT.llm is not None:
        RT.llm.close()
    if RT.memory is not None:
        RT.memory.close()
    if RT.mcp is not None:
        RT.mcp.close()
    if RT.pool is not None:
        RT.pool.close()
    log.info("agent shut down")


app = FastAPI(title="Agentic Cloud - research agent", lifespan=lifespan)


class TaskRequest(BaseModel):
    thread_id: str
    input: dict[str, Any]


class ResumeRequest(BaseModel):
    thread_id: str
    value: Any


class BindRequest(BaseModel):
    env: dict[str, str]


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _not_ready() -> JSONResponse:
    if RT.error:
        return JSONResponse({"status": "error", "error": RT.error}, status_code=500)
    return JSONResponse({"status": "starting"}, status_code=503)


def _result(thread_id: str, run_ms: float, checkpoint_load_ms: Optional[float] = None) -> dict:
    """Read the checkpoint after a run and describe it for the control plane."""
    snap = RT.graph.get_state(_config(thread_id))
    common = {
        "thread_id": thread_id,
        "run_ms": run_ms,
        "checkpoint_load_ms": checkpoint_load_ms,
        "llm": RT.llm.stats(),
        "memory": RT.memory.stats(),
        "mcp": dict(RT.mcp.stats),
    }

    if snap.next:  # graph paused -> waiting for a human
        interrupts = [i.value for task in snap.tasks for i in (task.interrupts or ())]
        payload = interrupts[0] if interrupts else {}
        return {
            **common,
            "status": "waiting",
            "waiting_for": payload.get("waiting_for", "input") if isinstance(payload, dict) else "input",
            "data": payload,
        }

    values = snap.values or {}
    return {
        **common,
        "status": "completed",
        "waiting_for": None,
        "data": {
            "report": values.get("report"),
            "summaries": values.get("summaries", []),
            "skipped_already_read": values.get("skipped_already_read", []),
            "workspace_files": values.get("workspace_files", []),
            "memories_written": values.get("memories_written", []),
            "search_source": values.get("search_source"),
            "search_note": values.get("search_note"),
            "ranking": values.get("ranking"),
            "notification": values.get("notification"),
        },
    }


@app.get("/health")
def health():
    if RT.prebind:
        # Phase 5: imports done, no session yet -> the moment to take the golden snapshot
        return {"status": "prebind", "timings_ms": RT.timings}
    if not RT.ready:
        return _not_ready()
    return {
        "status": "ok",
        "session_id": SESSION_ID,
        "uptime_ms": _ms(_T_PROCESS, time.perf_counter()),
        "timings_ms": RT.timings,
    }


@app.post("/bind")
def bind(req: BindRequest):
    """Phase 5: give a restored golden-snapshot clone its session (DSN, SESSION_ID, MEMORY_TOKEN,
    URLs). Allowed exactly once, only in pre-bind mode."""
    global SESSION_ID
    with RT.bind_lock:
        if not RT.prebind:
            return JSONResponse({"status": "failed", "error": "not in pre-bind mode (or already bound)"},
                                status_code=409)
        start = time.perf_counter()
        os.environ.update({str(k): str(v) for k, v in req.env.items()})
        SESSION_ID = os.environ.get("SESSION_ID", SESSION_ID)
        try:
            db_s, graph_s = _bind_stage()
        except Exception as exc:  # noqa: BLE001
            log.exception("bind failed")
            return JSONResponse({"status": "failed", "error": repr(exc)}, status_code=500)
        RT.timings.update({"db_connect_ms": round(db_s * 1000.0, 1),
                           "graph_build_ms": round(graph_s * 1000.0, 1),
                           "bind_ms": _ms(start, time.perf_counter())})
        RT.prebind = False
        RT.ready = True
        log.info("bound to session %s: %s", SESSION_ID, RT.timings)
        return {"status": "ok", "session_id": SESSION_ID, "timings_ms": RT.timings}


@app.post("/task")
def task(req: TaskRequest):
    if not RT.ready:
        return _not_ready()

    topic = str(req.input.get("topic", "")).strip()
    if not topic:
        return JSONResponse({"status": "failed", "error": "input.topic is required"}, status_code=400)

    if RT.graph.get_state(_config(req.thread_id)).values:
        return JSONResponse(
            {"status": "failed", "error": f"thread {req.thread_id} already exists; use /resume"},
            status_code=409,
        )

    # Phase 4: optional routing mode for this session's model calls
    llm_mode = req.input.get("llm_mode")
    if llm_mode is not None and llm_mode not in ("local", "cloud", "cascade"):
        return JSONResponse({"status": "failed", "error": "input.llm_mode must be local | cloud | cascade"},
                            status_code=400)

    start = time.perf_counter()
    try:
        RT.graph.invoke({"topic": topic, "llm_mode": llm_mode}, _config(req.thread_id))
    except Exception as exc:  # noqa: BLE001
        log.exception("task failed")
        return JSONResponse({"status": "failed", "error": repr(exc)}, status_code=500)
    return _result(req.thread_id, run_ms=_ms(start, time.perf_counter()))


@app.post("/resume")
def resume(req: ResumeRequest):
    if not RT.ready:
        return _not_ready()

    cfg = _config(req.thread_id)

    # Load the checkpoint from Postgres. On a fresh (woken) container this is the
    # moment the agent "remembers" everything. (invoke() reads it again internally;
    # this separate read is what we time as checkpoint_load_ms.)
    t0 = time.perf_counter()
    snap = RT.graph.get_state(cfg)
    checkpoint_load_ms = _ms(t0, time.perf_counter())

    if not snap.values:
        return JSONResponse(
            {"status": "failed", "error": f"no checkpoint for thread {req.thread_id}"}, status_code=404
        )
    if not snap.next:
        return JSONResponse(
            {"status": "failed", "error": f"thread {req.thread_id} is already completed"}, status_code=409
        )

    start = time.perf_counter()
    try:
        RT.graph.invoke(RT.Command(resume=req.value), cfg)
    except Exception as exc:  # noqa: BLE001
        log.exception("resume failed")
        return JSONResponse({"status": "failed", "error": repr(exc)}, status_code=500)
    return _result(req.thread_id, run_ms=_ms(start, time.perf_counter()), checkpoint_load_ms=checkpoint_load_ms)


@app.get("/state/{thread_id}")
def state(thread_id: str):
    if not RT.ready:
        return _not_ready()
    snap = RT.graph.get_state(_config(thread_id))
    if not snap.values:
        return JSONResponse({"error": f"no checkpoint for thread {thread_id}"}, status_code=404)
    return {"thread_id": thread_id, "next": list(snap.next), "values": snap.values}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL.lower())