"""
Agentic Cloud - Phase 5, Step 5.4
Firecracker Driver: same methods as DockerDriver, so the lifecycle, the idle reaper, the
gateway and the benchmarks stay unchanged. Selected with  driver: firecracker  in config.yaml.

  start()        POST fc-agent /vms -> new microVM RESTORED from golden.snap (no boot, no imports)
                 + clock fix + POST /bind (session env: DSN, SESSION_ID, MEMORY_TOKEN, URLs)
  wait_healthy() poll the agent's GET /health (normally already 200: bind finished in start())
  stop()         DELETE fc-agent /vms/<id>  (= FREEZE: the LangGraph checkpoint is in Postgres,
                 nothing inside the VM needs saving)
  list_managed() microVMs known to fc-agent (for reconciliation)

The control plane runs on Windows; fc-agent runs inside the fc-host Hyper-V VM and is reached
over the Hyper-V Default Switch. Every request carries the shared secret FC_AGENT_SECRET (.env).

L2 workspace: inside a microVM /workspace is RAM (tmpfs) - files exist while the VM runs and
are returned in the session result, but there are no persistent workspace volumes (yet).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx

from control_plane.docker_driver import HealthResult, StartResult

log = logging.getLogger("control_plane.firecracker")

WORKSPACE_LABEL = "microvm-tmpfs"


class FirecrackerDriver:
    kind = "firecracker"

    def __init__(self, agent_url: str, secret: str, golden: str = "golden",
                 base_env: Optional[dict[str, str]] = None, env_overrides: Optional[dict[str, str]] = None,
                 start_timeout_s: float = 90.0, health_timeout_s: float = 30.0,
                 health_poll_ms: float = 50.0) -> None:
        if not secret:
            raise RuntimeError("FC_AGENT_SECRET is not set in .env (driver: firecracker)")
        self.fc_url = agent_url.rstrip("/")
        self.golden = golden
        self.base_env = dict(base_env or {})
        self.env_overrides = dict(env_overrides or {})
        self.health_timeout_s = health_timeout_s
        self.health_poll_s = health_poll_ms / 1000.0
        self.api = httpx.Client(base_url=self.fc_url, headers={"X-FC-Secret": secret},
                                timeout=httpx.Timeout(start_timeout_s, connect=5.0))
        self.http = httpx.Client(timeout=2.0)
        self.last_start: dict[str, dict] = {}   # vm_id -> fc-agent timings of its start

    @classmethod
    def from_config(cls, cfg: dict, base_env: dict[str, str]) -> "FirecrackerDriver":
        fc, lc = cfg["firecracker"], cfg["lifecycle"]
        cp_url = str(fc.get("control_plane_url", "http://{WIN_HOST}:8000")).rstrip("/")
        overrides = {
            # The agent inside the microVM reaches Windows through fc-host; "{WIN_HOST}" is
            # replaced by fc-agent with the Windows address on the Hyper-V Default Switch.
            "AGENT_DSN": fc.get("agent_dsn", "postgresql://agentic:agentic_dev_pw@{WIN_HOST}:5432/agentic_cloud"),
            "MEMORY_API_URL": cp_url,
            "MCP_GATEWAY_URL": cp_url + "/mcp",
            "LLM_ROUTER_URL": cp_url,
            "OLLAMA_BASE_URL": fc.get("ollama_url", "http://{WIN_HOST}:11434"),
            **{str(k): str(v) for k, v in (fc.get("env") or {}).items()},
        }
        return cls(
            agent_url=fc["agent_url"],
            secret=os.environ.get(fc.get("secret_env", "FC_AGENT_SECRET"), ""),
            golden=fc.get("golden", "golden"),
            base_env=base_env,
            env_overrides=overrides,
            start_timeout_s=float(fc.get("start_timeout_s", 90)),
            health_timeout_s=float(lc.get("health_timeout_s", 60)),
            health_poll_ms=float(fc.get("health_poll_ms", 50)),
        )

    def close(self) -> None:
        self.api.close()
        self.http.close()

    # ================================================================ checks

    def ensure_ready(self) -> None:
        try:
            r = self.api.get("/health")
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"fc-agent not reachable at {self.fc_url}. In fc-host run:\n"
                f"  sudo FC_AGENT_SECRET=... python3 ~/fc-host/fc_agent.py\n"
                f"(and check firecracker.agent_url in config.yaml = the fc-host IP)") from exc
        if r.status_code == 401:
            raise RuntimeError("fc-agent rejected FC_AGENT_SECRET (different value in .env and fc-host?)")
        h = r.json()
        if self.golden not in (h.get("goldens") or {}):
            raise RuntimeError(f"golden snapshot '{self.golden}' missing in fc-host "
                               f"(have {list(h.get('goldens') or {})}); run make_golden.sh")
        log.info("fc-agent ok: %s free slots of %s, Windows host %s, golden %s",
                 h.get("free_slots"), h.get("slots"), h.get("windows_host"), self.golden)

    # ================================================================ start (= cold start AND wake)

    @staticmethod
    def container_name(session_id: str) -> str:
        return f"microvm-{session_id}"

    def start(self, session_id: str, extra_env: Optional[dict[str, str]] = None) -> StartResult:
        requested_at_ms = time.time() * 1000.0
        env = {**self.base_env, **self.env_overrides, **(extra_env or {}),
               "SESSION_ID": session_id, "WORKSPACE_DIR": "/workspace"}
        env.pop("REQUESTED_AT_MS", None)   # meaningless for a restored process
        t0 = time.perf_counter()
        r = self.api.post("/vms", json={"session_id": session_id, "env": env, "golden": self.golden})
        elapsed = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 201:
            raise RuntimeError(f"fc-agent could not start a microVM: HTTP {r.status_code} {r.text[:400]}")
        vm = r.json()
        self.last_start[vm["vm_id"]] = vm.get("timings_ms", {})
        t = vm.get("timings_ms", {})
        log.info("restored %s for %s at %s in %.0f ms (load %s, bind %s)", vm["vm_id"], session_id,
                 vm["agent_url"], elapsed, t.get("load_resume_ms"), t.get("bind_ms"))
        return StartResult(
            container_id=vm["vm_id"],
            name=vm["vm_id"],
            agent_url=vm["agent_url"],
            host_port=int(vm["agent_url"].rsplit(":", 1)[1]),
            requested_at_ms=requested_at_ms,
            run_returned_ms=round(elapsed, 1),
            workspace=WORKSPACE_LABEL,
            workspace_created=True,
        )

    # ================================================================ readiness / liveness

    def wait_healthy(self, agent_url: str, container_id: str) -> HealthResult:
        t0 = time.perf_counter()
        deadline = t0 + self.health_timeout_s
        polls = 0
        while time.perf_counter() < deadline:
            polls += 1
            try:
                resp = self.http.get(f"{agent_url}/health")
                if resp.status_code == 200 and resp.json().get("status") == "ok":
                    body = resp.json()
                    body.setdefault("timings_ms", {})["firecracker"] = self.last_start.get(container_id, {})
                    return HealthResult(True, self._since(t0), polls, body)
                if resp.status_code == 500:
                    return HealthResult(False, self._since(t0), polls, error=f"agent error: {resp.text[:300]}")
            except (httpx.HTTPError, ValueError):
                pass
            if polls % 20 == 0 and not self.is_running(container_id):
                return HealthResult(False, self._since(t0), polls,
                                    error=f"microVM exited:\n{self.logs(container_id, 30)}")
            time.sleep(self.health_poll_s)
        return HealthResult(False, self._since(t0), polls, error=f"/health not OK after {self.health_timeout_s:.0f}s")

    def check_alive(self, agent_url: str) -> bool:
        try:
            return self.http.get(f"{agent_url}/health").status_code == 200
        except httpx.HTTPError:
            return False

    # ================================================================ stop (= freeze)

    def stop(self, container_id: str, timeout_s: int = 5) -> float:
        t0 = time.perf_counter()
        try:
            r = self.api.delete(f"/vms/{container_id}")
            if r.status_code not in (200, 404):
                log.warning("stop %s: HTTP %s %s", container_id, r.status_code, r.text[:200])
        except httpx.HTTPError as exc:
            log.warning("stop %s: fc-agent unreachable: %r", container_id, exc)
        self.last_start.pop(container_id, None)
        elapsed = self._since(t0)
        log.info("stopped %s in %.0f ms", container_id, elapsed)
        return elapsed

    # ================================================================ inspection

    def is_running(self, container_id: str) -> bool:
        try:
            r = self.api.get(f"/vms/{container_id}")
            return r.status_code == 200 and r.json().get("status") == "running"
        except httpx.HTTPError:
            return False

    def logs(self, container_id: str, tail: int = 50) -> str:
        try:
            r = self.api.get(f"/vms/{container_id}/console", params={"tail": tail})
            return r.json().get("console", "") if r.status_code == 200 else f"<{r.status_code}>"
        except httpx.HTTPError as exc:
            return f"<fc-agent unreachable: {exc!r}>"

    def list_managed(self) -> list[dict]:
        try:
            vms = self.api.get("/vms").json().get("vms", [])
        except httpx.HTTPError:
            return []
        return [{"container_id": v["vm_id"], "name": v["vm_id"], "session_id": v["session_id"],
                 "status": v["status"]} for v in vms]

    # ================================================================ L2 workspace (RAM inside the VM)

    def workspace_name(self, session_id: str) -> str:
        return WORKSPACE_LABEL

    def ensure_workspace(self, session_id: str) -> tuple[str, bool]:
        return WORKSPACE_LABEL, True

    def delete_workspace(self, session_id: str) -> bool:
        return False

    def list_workspaces(self) -> list[dict]:
        return []

    def list_workspace_files(self, session_id: str, with_content: bool = False, **_) -> list[dict]:
        raise FileNotFoundError("Firecracker driver: the workspace lives in the microVM's RAM; "
                                "report and summaries are returned in the session result")

    @staticmethod
    def _since(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000.0, 1)