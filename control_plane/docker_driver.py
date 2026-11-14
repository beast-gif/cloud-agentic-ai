"""
Agentic Cloud - Phase 1, Step 1.3
Docker Driver: the ONLY part of the control plane that knows agents are Docker containers.

  start()        docker run  (one container per session, random localhost port)
  wait_healthy() poll GET /health until 200 (readiness), fail fast on crash/error
  stop()         docker stop + rm  (= FREEZE in Phase 1)
  list_managed() containers carrying our label (for reconciliation / cleanup)

In Phase 5 this file is replaced by a Firecracker driver with the same methods
(start = clone golden.snap, stop = snapshot), and the rest of the control plane stays.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import docker
import httpx
from docker.errors import APIError, ImageNotFound, NotFound

log = logging.getLogger("control_plane.docker")


@dataclass
class StartResult:
    container_id: str
    name: str
    agent_url: str
    host_port: int
    requested_at_ms: float      # epoch ms just before `docker run` (also sent to the agent)
    run_returned_ms: float      # how long the docker run API call itself took


@dataclass
class HealthResult:
    ok: bool
    ready_ms: float                         # start of polling -> /health 200 (or failure)
    polls: int
    body: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class DockerDriver:
    def __init__(
        self,
        image: str,
        network: str,
        internal_port: int = 8080,
        bind_host: str = "127.0.0.1",
        label: str = "agentic-cloud.session",
        memory_limit: str = "512m",
        cpu_limit: float = 1.0,
        health_timeout_s: float = 60.0,
        health_poll_ms: float = 100.0,
        base_env: Optional[dict[str, str]] = None,
    ) -> None:
        self.image = image
        self.network = network
        self.internal_port = internal_port
        self.bind_host = bind_host
        self.label = label
        self.memory_limit = memory_limit
        self.nano_cpus = int(cpu_limit * 1_000_000_000)
        self.health_timeout_s = health_timeout_s
        self.health_poll_s = health_poll_ms / 1000.0
        self.base_env = dict(base_env or {})

        self.client = docker.from_env()
        self.http = httpx.Client(timeout=2.0)

    @classmethod
    def from_config(cls, cfg: dict, base_env: dict[str, str]) -> "DockerDriver":
        d, lc = cfg["docker"], cfg["lifecycle"]
        return cls(
            image=d["agent_image"],
            network=d["network"],
            internal_port=int(d["agent_internal_port"]),
            bind_host=d.get("bind_host", "127.0.0.1"),
            label=d.get("label", "agentic-cloud.session"),
            memory_limit=d.get("memory_limit", "512m"),
            cpu_limit=float(d.get("cpu_limit", 1.0)),
            health_timeout_s=float(lc.get("health_timeout_s", 60)),
            health_poll_ms=float(lc.get("health_poll_ms", 100)),
            base_env=base_env,
        )

    def close(self) -> None:
        self.http.close()
        self.client.close()

    # ================================================================ checks

    def ensure_ready(self) -> None:
        """Fail early with a clear message if Docker / image / network are missing."""
        try:
            self.client.ping()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Docker is not reachable. Is Docker Desktop running?") from exc
        try:
            self.client.images.get(self.image)
        except ImageNotFound as exc:
            raise RuntimeError(
                f"Agent image '{self.image}' not found. Build it first:\n"
                f"  docker build -t {self.image} ./agent_runtime"
            ) from exc
        try:
            self.client.networks.get(self.network)
        except NotFound as exc:
            raise RuntimeError(
                f"Docker network '{self.network}' not found. Start Postgres first:\n"
                f"  docker compose up -d"
            ) from exc

    # ================================================================ start

    @staticmethod
    def container_name(session_id: str) -> str:
        return f"agent-{session_id}"

    def start(self, session_id: str, extra_env: Optional[dict[str, str]] = None) -> StartResult:
        name = self.container_name(session_id)
        self._remove_if_exists(name)  # leftover from a crash -> remove before reusing the name

        requested_at_ms = time.time() * 1000.0
        env = {
            **self.base_env,
            **(extra_env or {}),
            "SESSION_ID": session_id,
            "REQUESTED_AT_MS": f"{requested_at_ms:.1f}",
        }

        t0 = time.perf_counter()
        container = self.client.containers.run(
            self.image,
            name=name,
            detach=True,
            environment=env,
            network=self.network,
            # host port None -> Docker picks a free random port on bind_host
            ports={f"{self.internal_port}/tcp": (self.bind_host, None)},
            labels={self.label: session_id},
            mem_limit=self.memory_limit,
            nano_cpus=self.nano_cpus,
        )
        run_returned_ms = (time.perf_counter() - t0) * 1000.0

        host_port = self._host_port(container)
        agent_url = f"http://{self.bind_host}:{host_port}"
        log.info("started %s (%s) at %s in %.0f ms", name, container.short_id, agent_url, run_returned_ms)

        return StartResult(
            container_id=container.id,
            name=name,
            agent_url=agent_url,
            host_port=host_port,
            requested_at_ms=requested_at_ms,
            run_returned_ms=round(run_returned_ms, 1),
        )

    def _host_port(self, container, attempts: int = 20) -> int:
        """Read the random host port Docker assigned (may take a moment to appear)."""
        key = f"{self.internal_port}/tcp"
        for _ in range(attempts):
            container.reload()
            bindings = (container.attrs.get("NetworkSettings", {}).get("Ports") or {}).get(key)
            if bindings:
                return int(bindings[0]["HostPort"])
            time.sleep(0.05)
        raise RuntimeError(f"no host port published for {container.name}")

    # ================================================================ readiness

    def wait_healthy(self, agent_url: str, container_id: str) -> HealthResult:
        """Poll /health every health_poll_ms until 200.

        503 / connection refused -> still starting, keep polling
        500 {"status": "error"}  -> agent init failed, stop immediately
        container exited         -> crashed, stop immediately (checked every 10 polls)
        timeout                  -> fail
        """
        t0 = time.perf_counter()
        deadline = t0 + self.health_timeout_s
        polls = 0

        while time.perf_counter() < deadline:
            polls += 1
            try:
                resp = self.http.get(f"{agent_url}/health")
                if resp.status_code == 200:
                    return HealthResult(True, self._since(t0), polls, resp.json())
                if resp.status_code == 500:
                    return HealthResult(False, self._since(t0), polls, self._json(resp),
                                        error=f"agent init failed: {resp.text[:300]}")
            except httpx.HTTPError:
                pass  # not listening yet

            if polls % 10 == 0 and not self.is_running(container_id):
                return HealthResult(False, self._since(t0), polls,
                                    error=f"container exited during startup:\n{self.logs(container_id, 30)}")
            time.sleep(self.health_poll_s)

        return HealthResult(False, self._since(t0), polls,
                            error=f"/health not OK after {self.health_timeout_s:.0f}s")

    def check_alive(self, agent_url: str) -> bool:
        """Liveness: one quick /health call."""
        try:
            return self.http.get(f"{agent_url}/health").status_code == 200
        except httpx.HTTPError:
            return False

    # ================================================================ stop (= freeze in Phase 1)

    def stop(self, container_id: str, timeout_s: int = 5) -> float:
        """docker stop (SIGTERM, then SIGKILL after timeout_s) + rm. Returns elapsed ms.

        Safe to call on a container that is already gone.
        """
        t0 = time.perf_counter()
        try:
            c = self.client.containers.get(container_id)
            try:
                c.stop(timeout=timeout_s)
            except APIError as exc:
                log.warning("stop %s: %s", container_id[:12], exc)
            c.remove(force=True)
        except NotFound:
            pass
        elapsed = self._since(t0)
        log.info("stopped %s in %.0f ms", container_id[:12], elapsed)
        return elapsed

    def _remove_if_exists(self, name: str) -> None:
        try:
            self.client.containers.get(name).remove(force=True)
            log.warning("removed leftover container %s", name)
        except NotFound:
            pass

    # ================================================================ inspection

    def is_running(self, container_id: str) -> bool:
        try:
            c = self.client.containers.get(container_id)
            c.reload()
            return c.status == "running"
        except NotFound:
            return False

    def logs(self, container_id: str, tail: int = 50) -> str:
        try:
            raw = self.client.containers.get(container_id).logs(tail=tail)
            return raw.decode("utf-8", errors="replace")
        except NotFound:
            return "<container not found>"

    def list_managed(self) -> list[dict]:
        """All containers (running or not) that carry our session label."""
        out = []
        for c in self.client.containers.list(all=True, filters={"label": self.label}):
            out.append({
                "container_id": c.id,
                "name": c.name,
                "session_id": c.labels.get(self.label),
                "status": c.status,
            })
        return out

    # ================================================================ helpers

    @staticmethod
    def _since(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000.0, 1)

    @staticmethod
    def _json(resp: httpx.Response) -> dict:
        try:
            return resp.json()
        except ValueError:
            return {}