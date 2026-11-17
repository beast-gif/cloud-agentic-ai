"""
Agentic Cloud - Phase 3, Step 3.2
Minimal MCP server over HTTP (JSON-RPC 2.0), shared by the papers and notify servers.

Implements the subset of MCP that tool servers need, in the spec's "Streamable HTTP"
style, stateless, with plain JSON responses:

  POST /mcp   one JSON-RPC message per request
      initialize                -> server info + capabilities
      notifications/initialized -> (notification, no response body: HTTP 202)
      ping                      -> {}
      tools/list                -> {"tools": [{name, description, inputSchema}]}
      tools/call                -> {"content": [{"type": "text", "text": ...}], "isError": bool}
  GET  /health  plain liveness check for Docker

Two kinds of failure, as in MCP:
  - protocol errors (unknown method, bad params)   -> JSON-RPC "error" object
  - tool errors (arXiv down, bad input to the tool) -> normal result with isError: true,
    so the CALLER (agent) sees the message and can react

Gateway-only access (zero trust between components):
  If MCP_GATEWAY_SECRET is set, every /mcp request must carry the header
      X-Gateway-Secret: <that value>
  Only the control plane's gateway knows it; agent containers do not. So even an agent
  that discovers this server's address cannot call it directly and skip the gateway's
  auth, policy, rate limits and audit log.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

log = logging.getLogger("mcp")

LATEST_PROTOCOL_VERSION = "2025-06-18"


class ToolError(Exception):
    """The tool ran but failed (returned to the caller as isError: true)."""


class Unauthorized(Exception):
    """The caller's credentials for THIS server are missing or wrong (HTTP 401)."""


@dataclass
class CallContext:
    headers: dict[str, str]    # lower-cased request headers (e.g. "authorization")


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict, CallContext], Any]


class MCPServer:
    def __init__(self, name: str, version: str = "0.1.0", instructions: str = "") -> None:
        self.name = name
        self.version = version
        self.instructions = instructions
        self.tools: dict[str, Tool] = {}
        self.gateway_secret = os.environ.get("MCP_GATEWAY_SECRET", "")
        self.stats = {"requests": 0, "tool_calls": 0, "tool_errors": 0,
                      "unauthorized": 0, "forbidden": 0}

    # ------------------------------------------------------------------ registration

    def tool(self, name: str, description: str, input_schema: dict):
        """Decorator: @server.tool("search", "...", {...json schema...})"""
        def register(fn: Callable[[dict, CallContext], Any]):
            self.tools[name] = Tool(name, description, input_schema, fn)
            return fn
        return register

    # ------------------------------------------------------------------ JSON-RPC

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _result(msg_id: Any, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _call_tool(self, msg_id: Any, params: dict, ctx: CallContext) -> dict:
        name = params.get("name")
        args = params.get("arguments") or {}
        tool = self.tools.get(name)
        if tool is None:
            return self._error(msg_id, -32602, f"unknown tool: {name!r}")
        if not isinstance(args, dict):
            return self._error(msg_id, -32602, "arguments must be an object")
        missing = [k for k in tool.input_schema.get("required", []) if k not in args]
        if missing:
            return self._error(msg_id, -32602, f"missing required argument(s): {missing}")

        self.stats["tool_calls"] += 1
        t0 = time.perf_counter()
        try:
            out = tool.handler(args, ctx)
        except ToolError as exc:
            self.stats["tool_errors"] += 1
            return self._result(msg_id, {"content": [{"type": "text", "text": str(exc)}],
                                         "isError": True})
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        log.info("tool %s ok in %.1f ms", name, ms)

        result: dict[str, Any] = {
            "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}],
            "isError": False,
        }
        if isinstance(out, dict):
            result["structuredContent"] = out   # machine-readable copy of the same data
        return self._result(msg_id, result)     # wrap in the JSON-RPC envelope

    def handle(self, msg: dict, ctx: CallContext) -> Optional[dict]:
        """Process one JSON-RPC message. Returns None for notifications."""
        self.stats["requests"] += 1
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
            return self._error(msg.get("id") if isinstance(msg, dict) else None,
                               -32600, "invalid JSON-RPC request")
        method, msg_id = msg["method"], msg.get("id")
        params = msg.get("params") or {}

        if "id" not in msg:  # notification (e.g. notifications/initialized): no reply
            return None
        if method == "initialize":
            return self._result(msg_id, {
                "protocolVersion": params.get("protocolVersion") or LATEST_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self.name, "version": self.version},
                "instructions": self.instructions,
            })
        if method == "ping":
            return self._result(msg_id, {})
        if method == "tools/list":
            return self._result(msg_id, {"tools": [
                {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
                for t in self.tools.values()
            ]})
        if method == "tools/call":
            return self._call_tool(msg_id, params, ctx)
        return self._error(msg_id, -32601, f"method not found: {method}")

    # ------------------------------------------------------------------ HTTP

    def app(self) -> FastAPI:
        app = FastAPI(title=f"MCP server: {self.name}")

        @app.get("/health")
        def health():
            return {"status": "ok", "server": self.name, "tools": list(self.tools),
                    "gateway_secret_required": bool(self.gateway_secret), "stats": self.stats}

        @app.post("/mcp")
        async def mcp(request: Request):
            # 1) only the gateway may talk to this server
            if self.gateway_secret:
                got = request.headers.get("x-gateway-secret", "")
                if not hmac.compare_digest(got, self.gateway_secret):  # constant-time compare
                    self.stats["forbidden"] += 1
                    return JSONResponse(self._error(None, -32003, "forbidden: calls must come "
                                                    "through the MCP gateway"), status_code=403)
            # 2) parse JSON-RPC
            try:
                msg = await request.json()
            except (json.JSONDecodeError, ValueError):
                return JSONResponse(self._error(None, -32700, "parse error"), status_code=400)

            ctx = CallContext(headers={k.lower(): v for k, v in request.headers.items()})
            try:
                # tool handlers are blocking (HTTP calls, locks) -> run in a worker thread
                from starlette.concurrency import run_in_threadpool
                reply = await run_in_threadpool(self.handle, msg, ctx)
            except Unauthorized as exc:
                self.stats["unauthorized"] += 1
                return JSONResponse(self._error(msg.get("id"), -32001, f"unauthorized: {exc}"),
                                    status_code=401)
            if reply is None:
                return Response(status_code=202)
            return JSONResponse(reply)

        @app.get("/mcp")
        def mcp_get():
            # this server never opens server->client streams
            return Response(status_code=405)

        return app


def bearer_token(ctx: CallContext) -> str:
    auth = ctx.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else ""