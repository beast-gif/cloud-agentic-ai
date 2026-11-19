"""
Agentic Cloud - Phase 3, Step 3.5
Shared helpers for the red-team script and the gateway benchmark.

They act like a real agent: create a session through the control plane, then read
the session token from INSIDE the agent container (`docker exec ... printenv`),
exactly the value a compromised agent would have.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import time
from typing import Any, Optional

import httpx

from control_plane.settings import load_config, load_dotenv

load_dotenv()                      # .env -> MCP_GATEWAY_SECRET, NOTIFY_ACCOUNTS for direct-server tests
CFG = load_config()
API = f"http://{CFG['control_plane']['host']}:{CFG['control_plane']['port']}"
PAPERS_URL = CFG["mcp"]["servers"]["papers"]["url"]
NOTIFY_URL = CFG["mcp"]["servers"]["notify"]["url"]
NOTIFY_ADMIN = NOTIFY_URL.rsplit("/mcp", 1)[0] + "/admin/messages"


class ControlPlane:
    def __init__(self, base: str = API, timeout_s: float = 300.0) -> None:
        self.base = base.rstrip("/")
        self.http = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self.http.close()

    def get(self, path: str, **kw) -> Any:
        r = self.http.get(f"{self.base}{path}", **kw)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: Optional[dict] = None, **kw) -> httpx.Response:
        return self.http.post(f"{self.base}{path}", json=body or {}, **kw)

    def create_session(self, user_id: str, topic: str) -> dict:
        r = self.post("/sessions", {"user_id": user_id, "agent_type": "research", "input": {"topic": topic}})
        r.raise_for_status()
        return r.json()

    def event(self, sid: str, value: Any) -> dict:
        r = self.post(f"/sessions/{sid}/events", {"value": value})
        r.raise_for_status()
        return r.json()

    def freeze(self, sid: str) -> dict:
        r = self.post(f"/sessions/{sid}/freeze")
        r.raise_for_status()
        return r.json()


def container_env(session_id: str, name: str) -> str:
    """Read an environment variable from inside the agent container (what the agent itself has)."""
    out = subprocess.run(["docker", "exec", f"agent-{session_id}", "printenv", name],
                         capture_output=True, text=True, timeout=30)
    return out.stdout.strip()


def container_python(session_id: str, code: str) -> str:
    out = subprocess.run(["docker", "exec", f"agent-{session_id}", "python", "-c", code],
                         capture_output=True, text=True, timeout=60)
    return (out.stdout + out.stderr).strip()


class GatewayClient:
    """MCP client for the gateway, authenticated with a session token (like the agent)."""

    def __init__(self, token: str, base: str = API) -> None:
        self.url = f"{base}/mcp"
        self.http = httpx.Client(timeout=120.0)
        self.token = token
        self._id = 0

    def rpc(self, method: str, params: Optional[dict] = None, token: Optional[str] = None) -> tuple[int, dict, float]:
        """Returns (http_status, json_body, client_ms)."""
        self._id += 1
        t0 = time.perf_counter()
        r = self.http.post(self.url, json={"jsonrpc": "2.0", "id": self._id, "method": method,
                                           "params": params or {}},
                           headers={"Authorization": f"Bearer {token if token is not None else self.token}"})
        ms = (time.perf_counter() - t0) * 1000.0
        try:
            body = r.json()
        except ValueError:
            body = {}
        return r.status_code, body or {}, ms

    def call(self, tool: str, args: Optional[dict] = None, token: Optional[str] = None) -> tuple[int, dict, float]:
        return self.rpc("tools/call", {"name": tool, "arguments": args or {}}, token=token)

    def close(self) -> None:
        self.http.close()


# ONE reusable client (keep-alive), like the gateway's own client. Creating a new
# httpx client per call (httpx.post) builds a new SSL context and a new TCP connection
# every time, which cost ~700 ms per call on Windows and made "direct" look slower
# than going through the gateway in the first benchmark run.
_direct_http = httpx.Client(timeout=60.0)


def direct_mcp(url: str, tool: str, args: dict, headers: Optional[dict] = None,
               with_secret: bool = True) -> tuple[int, dict, float]:
    """Call an MCP server directly (bypassing the gateway)."""
    h = dict(headers or {})
    if with_secret:
        h["X-Gateway-Secret"] = os.environ.get("MCP_GATEWAY_SECRET", "")
    t0 = time.perf_counter()
    r = _direct_http.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                     "params": {"name": tool, "arguments": args}}, headers=h)
    ms = (time.perf_counter() - t0) * 1000.0
    try:
        return r.status_code, r.json() or {}, ms
    except ValueError:
        return r.status_code, {}, ms


def notify_messages() -> list[dict]:
    return httpx.get(NOTIFY_ADMIN, timeout=10.0).json().get("messages", [])


def provider_key(account: str) -> Optional[str]:
    """The mock notify provider's key for an account (from NOTIFY_ACCOUNTS in .env)."""
    for pair in os.environ.get("NOTIFY_ACCOUNTS", "").split(","):
        if ":" in pair:
            key, name = pair.split(":", 1)
            if name.strip() == account:
                return key.strip()
    return None


def describe(values: list[float]) -> dict:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {}
    p95 = v[max(1, -(-95 * len(v) // 100)) - 1]   # nearest-rank: ceil(0.95 n)
    return {"n": len(v), "mean": round(statistics.fmean(v), 2), "p50": round(statistics.median(v), 2),
            "p95": round(p95, 2), "min": round(v[0], 2), "max": round(v[-1], 2)}