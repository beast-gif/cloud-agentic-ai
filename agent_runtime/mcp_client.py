"""
Agentic Cloud - Phase 3, Step 3.4
Agent-side MCP client: talks ONLY to the control plane's MCP gateway.

The agent knows two things (environment variables set by the control plane):
  MCP_GATEWAY_URL  e.g. http://host.docker.internal:8000/mcp
  MEMORY_TOKEN     this container's session token (the same one the Memory API uses)

It does NOT know where the tool servers are, and holds no API keys: the gateway
decides which tools exist for this agent, fills in owner-only arguments, attaches
credentials from the vault and rate-limits calls.

Errors:
  MCPError        protocol / policy / backend problem (JSON-RPC "error")
  RateLimited     the gateway's token bucket said "not yet" (has retry_after_s)
  ToolFailed      the tool ran but reported isError (e.g. arXiv down)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import httpx

log = logging.getLogger("agent.mcp")

ERR_RATE_LIMITED = -32002


class MCPError(Exception):
    def __init__(self, code: int, message: str, data: Optional[dict] = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code, self.message, self.data = code, message, data or {}


class RateLimited(MCPError):
    @property
    def retry_after_s(self) -> float:
        return float(self.data.get("retry_after_s", 3.0))


class ToolFailed(Exception):
    """The tool executed but returned isError: true."""


class MCPClient:
    def __init__(self, url: str, token: str, timeout_s: float = 120.0) -> None:
        self.url = url or ""
        self.enabled = bool(url and token)
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            headers={"Authorization": f"Bearer {token}",
                     "Accept": "application/json, text/event-stream"},
        ) if self.enabled else None
        self._id = 0
        self.stats = {"calls": 0, "errors": 0, "rate_limited": 0, "retries": 0, "total_ms": 0.0}
        if not self.enabled:
            log.warning("MCP disabled (MCP_GATEWAY_URL or MEMORY_TOKEN not set)")

    # ------------------------------------------------------------------ low level

    def _rpc(self, method: str, params: Optional[dict] = None) -> dict:
        if not self.enabled:
            raise MCPError(-32003, "MCP gateway not configured")
        self._id += 1
        body = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        t0 = time.perf_counter()
        try:
            resp = self._http.post(self.url, json=body)
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            self.stats["errors"] += 1
            raise MCPError(-32003, f"gateway unreachable: {exc!r}") from exc
        finally:
            self.stats["calls"] += 1
            self.stats["total_ms"] = round(self.stats["total_ms"] + (time.perf_counter() - t0) * 1000.0, 1)

        if "error" in data:
            err = data["error"]
            cls = RateLimited if err.get("code") == ERR_RATE_LIMITED else MCPError
            if cls is RateLimited:
                self.stats["rate_limited"] += 1
            else:
                self.stats["errors"] += 1
            raise cls(err.get("code", -32000), err.get("message", "error"), err.get("data"))
        return data.get("result") or {}

    # ------------------------------------------------------------------ API

    def list_tools(self) -> list[str]:
        return [t["name"] for t in self._rpc("tools/list").get("tools", [])]

    def call(self, name: str, arguments: Optional[dict] = None) -> Any:
        """Call a tool. Returns structuredContent, or the JSON-decoded text content."""
        result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        text = "".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        if result.get("isError"):
            self.stats["errors"] += 1
            raise ToolFailed(text or f"{name} failed")
        if "structuredContent" in result:
            return result["structuredContent"]
        try:
            return json.loads(text)
        except ValueError:
            return text

    def call_with_retry(self, name: str, arguments: Optional[dict] = None,
                        max_wait_s: float = 10.0) -> Any:
        """Like call(), but if rate-limited waits retry_after_s once (if <= max_wait_s) and retries."""
        try:
            return self.call(name, arguments)
        except RateLimited as exc:
            wait = exc.retry_after_s
            if wait > max_wait_s:
                raise
            log.info("%s rate-limited, retrying in %.1f s", name, wait)
            self.stats["retries"] += 1
            time.sleep(wait + 0.05)
            return self.call(name, arguments)

    def close(self) -> None:
        if self._http is not None:
            self._http.close()