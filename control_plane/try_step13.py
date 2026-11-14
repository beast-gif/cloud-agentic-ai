"""
Agentic Cloud - Phase 1, Step 1.3
Smoke test for session_store.py + docker_driver.py (no API yet - that is Step 1.4).

Runs the full cycle the control plane will automate:
  create session -> start container -> /task (pauses) -> freeze (stop container)
  -> wake (new container) -> /resume -> completed
and records everything in sessions / session_events / wake_metrics.

Run from the agentic-cloud/ folder:
  python -m control_plane.try_step13
"""

from __future__ import annotations

import json
import logging
import time

import httpx

from control_plane.docker_driver import DockerDriver
from control_plane.session_store import (
    COMPLETED, FAILED, FREEZING, FROZEN, RUNNING, STARTING, WAITING, WAKING, SessionStore,
)
from control_plane.settings import agent_env, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("try_step13")


def start_and_wait(driver: DockerDriver, store: SessionStore, session_id: str,
                   from_state: str, wake_no: int, kind: str) -> tuple[str, dict]:
    """Start a container, wait for /health, move session to RUNNING. Returns (agent_url, health)."""
    t0 = time.perf_counter()
    started = driver.start(session_id)
    health = driver.wait_healthy(started.agent_url, started.container_id)
    if not health.ok:
        store.transition(session_id, from_state, FAILED, event="failed", error=health.error)
        driver.stop(started.container_id)
        raise RuntimeError(health.error)

    row = store.transition(
        session_id, from_state, RUNNING,
        event="started" if kind == "cold_start" else "resumed",
        payload={"agent_url": started.agent_url},
        touch_active=True,
        container_id=started.container_id,
        agent_url=started.agent_url,
    )
    assert row is not None, "CAS failed unexpectedly"

    t = health.body.get("timings_ms", {})
    ready_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    store.record_wake_metrics(
        session_id, wake_no, kind,
        container_start_ms=t.get("container_start_ms"),
        import_ms=t.get("import_ms"),
        graph_build_ms=t.get("graph_build_ms"),
        ready_ms=ready_ms,
    )
    log.info("%s ready in %.0f ms (polls=%d, agent timings=%s)", kind, ready_ms, health.polls, t)
    return started.agent_url, health.body


def main() -> None:
    cfg = load_config()
    store = SessionStore(cfg["postgres"]["host_dsn"])
    driver = DockerDriver.from_config(cfg, base_env=agent_env(cfg))
    driver.ensure_ready()
    http = httpx.Client(timeout=120.0)

    try:
        # 1) create session
        s = store.create_session(user_id="amrith", agent_type="research",
                                 payload={"topic": "speculative decoding"})
        sid = s["session_id"]
        log.info("created session %s", sid)

        # 2) cold start
        url, _ = start_and_wait(driver, store, sid, STARTING, wake_no=0, kind="cold_start")

        # 3) run the task until it pauses
        r = http.post(f"{url}/task", json={"thread_id": sid, "input": {"topic": "speculative decoding"}}).json()
        log.info("task -> %s (waiting_for=%s)", r["status"], r.get("waiting_for"))
        store.transition(sid, RUNNING, WAITING, event="waiting", touch_active=True,
                         waiting_for=r.get("waiting_for"))
        for p in r["data"]["shortlist"]:
            log.info("   %d. %s (score %.2f)", p["rank"], p["title"], p["score"])

        # 4) freeze (what the idle reaper will do automatically in Step 1.4)
        row = store.transition(sid, WAITING, FREEZING, event="freeze")
        stop_ms = driver.stop(row["container_id"])
        store.transition(sid, FREEZING, FROZEN, event="frozen", payload={"stop_ms": stop_ms},
                         container_id=None, agent_url=None)
        log.info("frozen (stop took %.0f ms) - no container is running for %s now", stop_ms, sid)
        time.sleep(2)

        # 5) wake: brand-new container
        store.transition(sid, FROZEN, WAKING, event="wake", increment_wake=True)
        t_wake = time.perf_counter()
        url, _ = start_and_wait(driver, store, sid, WAKING, wake_no=1, kind="wake")

        # 6) resume with the user's choice
        r = http.post(f"{url}/resume", json={"thread_id": sid, "value": {"selected": [1, 3]}}).json()
        total_ms = round((time.perf_counter() - t_wake) * 1000.0, 1)
        log.info("resume -> %s, checkpoint_load_ms=%s, wake+resume total=%.0f ms",
                 r["status"], r.get("checkpoint_load_ms"), total_ms)

        store.transition(sid, RUNNING, COMPLETED, event="completed", touch_active=True, waiting_for=None)
        row = store.get(sid)
        driver.stop(row["container_id"])
        store.transition(sid, COMPLETED, COMPLETED, container_id=None, agent_url=None)

        # 7) show what was recorded
        print("\n=== session ===")
        print(json.dumps(store.get(sid), indent=2, default=str))
        print("\n=== events ===")
        for e in store.events(sid):
            print(f"{e['created_at']:%H:%M:%S.%f}  {e['event_type']:<10} {json.dumps(e['payload'])}")
        print("\n=== wake_metrics ===")
        for m in store.wake_metrics(sid):
            print({k: m[k] for k in ("wake_no", "kind", "container_start_ms", "import_ms",
                                     "graph_build_ms", "ready_ms")})
        print("\n=== report ===")
        print(r["data"]["report"])

        # 8) CAS demo: a second freeze attempt on a COMPLETED session must fail
        assert store.transition(sid, WAITING, FREEZING) is None
        log.info("CAS check OK: cannot freeze a session that is not WAITING")

    finally:
        http.close()
        driver.close()
        store.close()


if __name__ == "__main__":
    main()