"""
Agentic Cloud - Phase 3, Step 3.3
MCP Gateway: the ONE MCP endpoint agents talk to.

  agent --(session token)--> GATEWAY --(X-Gateway-Secret [+ vault credential])--> MCP servers

For every JSON-RPC message from an agent (already authenticated by main.py:
token -> session -> user_id + agent_type):

  initialize / ping        answered by the gateway itself
  tools/list               tools of ALL backend servers, renamed "<server>.<tool>",
                           FILTERED by the agent type's allowlist (hidden = invisible)
  tools/call               1. tool exists and is in the allowlist?        else denied
                           2. argument rules (e.g. notify.send to=owner)  else denied
                           3. token bucket for the tool's server          else rate_limited
                           4. credential from the vault, if the server needs one
                           5. forward to the server with X-Gateway-Secret
                           6. audit row in tool_calls (always, whatever the decision)

Token bucket (rate limit), per bucket:
    tokens(t) = min(C, tokens(t0) + r * (t - t0))      C = capacity, r = refill per second
    a call needs 1 token; if tokens < 1 -> rejected, retry after (1 - tokens) / r seconds
  e.g. C = 3, r = 0.33/s: 3 calls at once are fine, the 4th must wait ~3 s.
  scope "global"  = one bucket for the whole platform (protects arXiv: one IP for everyone)
  scope "session" = one bucket per session (stops one agent from spamming)

Secrets: the decrypted credential lives only in a local variable for the duration of one
forwarded request. It is never returned to the agent, logged, or written to the audit table.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from psycopg_pool import ConnectionPool

from control_plane.vault import CredentialNotFound, Vault, VaultError

log = logging.getLogger("control_plane.gateway")

PROTOCOL_VERSION = "2025-06-18"

# JSON-RPC error codes used by the gateway (application range: -32000 .. -32099)
ERR_DENIED = -32001
ERR_RATE_LIMITED = -32002
ERR_BACKEND = -32003


# ============================================================ token bucket

class TokenBucket:
    def __init__(self, capacity: float, refill_per_s: float) -> None:
        self.capacity = float(capacity)
        self.rate = float(refill_per_s)
        self.tokens = float(capacity)           # starts full: a burst of C is allowed
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def take(self) -> tuple[bool, float, float]:
        """Try to use 1 token. Returns (allowed, tokens_left, retry_after_s)."""
        with self._lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + self.rate * (now - self.last))
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, round(self.tokens, 2), 0.0
            retry = (1.0 - self.tokens) / self.rate if self.rate > 0 else float("inf")
            return False, round(self.tokens, 2), round(retry, 2)

    def peek(self) -> float:
        with self._lock:
            now = time.monotonic()
            return round(min(self.capacity, self.tokens + self.rate * (now - self.last)), 2)


# ============================================================ config objects

@dataclass
class Backend:
    name: str
    url: str
    credential: Optional[str] = None     # vault provider name, or None


@dataclass
class CatalogTool:
    full_name: str                       # papers.search
    server: str                          # papers
    tool: str                            # search
    description: str
    input_schema: dict


@dataclass
class Policy:
    allow: set[str]
    rules: dict[str, dict] = field(default_factory=dict)        # tool -> {arg: rule}
    rate_limits: dict[str, dict] = field(default_factory=dict)  # server -> {capacity, refill_per_s, scope}


class GatewayDenied(Exception):
    """Policy said no (allowlist or argument rule)."""


# ============================================================ gateway

class Gateway:
    def __init__(self, cfg: dict, pool: ConnectionPool, vault: Vault) -> None:
        mcp = cfg.get("mcp") or {}
        self.pool = pool
        self.vault = vault
        self.secret = os.environ.get(mcp.get("gateway_secret_env", "MCP_GATEWAY_SECRET"), "")
        self.enabled = bool(self.secret)
        self.timeout_s = float(mcp.get("request_timeout_s", 90))
        self.catalog_ttl_s = float(mcp.get("tools_cache_s", 300))

        self.backends = {
            name: Backend(name, s["url"], s.get("credential"))
            for name, s in (mcp.get("servers") or {}).items()
        }
        self.policies = {
            agent_type: Policy(
                allow=set(p.get("allow") or []),
                rules=p.get("rules") or {},
                rate_limits=p.get("rate_limits") or {},
            )
            for agent_type, p in (mcp.get("policies") or {}).items()
        }

        self.http = httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=5.0))
        self._catalog: list[CatalogTool] = []
        self._catalog_at = 0.0
        self._catalog_lock = threading.Lock()
        self._buckets: dict[str, TokenBucket] = {}
        self._buckets_lock = threading.Lock()
        self._next_id = 0
        self._id_lock = threading.Lock()
        self.stats = {"tools_list": 0, "calls": 0, "allowed": 0, "denied": 0,
                      "rate_limited": 0, "errors": 0}

    def close(self) -> None:
        self.http.close()

    # ================================================================ backend client

    def _rpc_id(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def _backend_rpc(self, backend: Backend, method: str, params: Optional[dict] = None,
                     extra_headers: Optional[dict] = None) -> dict:
        """One JSON-RPC request to an MCP server. Raises RuntimeError on transport problems."""
        headers = {
            "X-Gateway-Secret": self.secret,
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
            **(extra_headers or {}),
        }
        body = {"jsonrpc": "2.0", "id": self._rpc_id(), "method": method, "params": params or {}}
        try:
            resp = self.http.post(backend.url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"{backend.name} unreachable: {exc!r}") from exc
        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError(f"{backend.name} returned non-JSON (HTTP {resp.status_code})") from exc
        if resp.status_code == 403:
            raise RuntimeError(f"{backend.name} rejected the gateway secret (check MCP_GATEWAY_SECRET)")
        # 401 from a provider = bad/missing credential: pass its JSON-RPC error through
        return data

    # ================================================================ catalog (tools/list of all servers)

    def catalog(self, force: bool = False) -> list[CatalogTool]:
        with self._catalog_lock:
            if not force and self._catalog and time.time() - self._catalog_at < self.catalog_ttl_s:
                return self._catalog
            tools: list[CatalogTool] = []
            for b in self.backends.values():
                try:
                    data = self._backend_rpc(b, "tools/list")
                except RuntimeError as exc:
                    log.warning("catalog: %s", exc)
                    continue
                for t in (data.get("result") or {}).get("tools", []):
                    tools.append(CatalogTool(
                        full_name=f"{b.name}.{t['name']}", server=b.name, tool=t["name"],
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema") or {"type": "object"},
                    ))
            self._catalog, self._catalog_at = tools, time.time()
            log.info("catalog: %d tools from %d servers", len(tools), len(self.backends))
            return tools

    def _visible_tools(self, agent_type: str) -> list[CatalogTool]:
        policy = self.policies.get(agent_type)
        if policy is None:
            return []
        return [t for t in self.catalog() if t.full_name in policy.allow]

    # ================================================================ policy

    def _apply_rules(self, policy: Policy, tool: CatalogTool, args: dict, user_id: str) -> dict:
        """Check/complete arguments. Returns the (possibly completed) arguments or raises GatewayDenied."""
        rules = policy.rules.get(tool.full_name) or {}
        args = copy.deepcopy(args)
        for arg, rule in rules.items():
            if rule == "owner_only":
                backend = self.backends[tool.server]
                if not backend.credential:
                    raise GatewayDenied(f"rule owner_only needs a credential provider for {tool.server}")
                try:
                    owner = self.vault.metadata(user_id, backend.credential).get("owner_chat")
                except CredentialNotFound as exc:
                    raise GatewayDenied(f"no {backend.credential} account configured for this user") from exc
                if not owner:
                    raise GatewayDenied(f"no owner_chat set in the {backend.credential} credential metadata")
                given = args.get(arg)
                if given in (None, ""):
                    args[arg] = owner                       # fill in the only allowed value
                elif str(given) != owner:
                    raise GatewayDenied(f"{tool.full_name} may only send to the session owner "
                                        f"(got {arg}={given!r})")
            else:
                raise GatewayDenied(f"unknown rule {rule!r} for {tool.full_name}.{arg}")
        return args

    def _bucket(self, policy: Policy, server: str, session_id: str) -> Optional[TokenBucket]:
        rl = policy.rate_limits.get(server)
        if not rl:
            return None
        scope = rl.get("scope", "global")
        key = f"{server}:global" if scope == "global" else f"{server}:session:{session_id}"
        with self._buckets_lock:
            b = self._buckets.get(key)
            if b is None:
                b = TokenBucket(rl.get("capacity", 3), rl.get("refill_per_s", 0.33))
                self._buckets[key] = b
            return b

    # ================================================================ JSON-RPC entry point

    def handle(self, msg: Any, session: dict) -> Optional[dict]:
        """Handle one JSON-RPC message from an authenticated agent session."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
            return _error(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid JSON-RPC request")
        if "id" not in msg:
            return None                                           # notification: no reply
        method, msg_id, params = msg["method"], msg["id"], msg.get("params") or {}

        if method == "initialize":
            return _result(msg_id, {
                "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "agentic-cloud-gateway", "version": "0.3.0"},
                "instructions": "All tools of this platform. Calls are checked, rate-limited and logged.",
            })
        if method == "ping":
            return _result(msg_id, {})
        if method == "tools/list":
            self.stats["tools_list"] += 1
            return _result(msg_id, {"tools": [
                {"name": t.full_name, "description": t.description, "inputSchema": t.input_schema}
                for t in self._visible_tools(session["agent_type"])
            ]})
        if method == "tools/call":
            return self._call(msg_id, params, session)
        return _error(msg_id, -32601, f"method not found: {method}")

    # ================================================================ tools/call

    def _call(self, msg_id: Any, params: dict, session: dict) -> dict:
        t0 = time.perf_counter()
        self.stats["calls"] += 1
        name = str(params.get("name", ""))
        args = params.get("arguments") or {}
        sid, user, agent_type = session["session_id"], session["user_id"], session["agent_type"]
        audit = {"session_id": sid, "user_id": user, "agent_type": agent_type, "tool": name,
                 "args": args, "credential_used": None, "backend_ms": None, "tool_is_error": None}

        def finish(decision: str, reply: dict, reason: Optional[str] = None) -> dict:
            self.stats[{"allowed": "allowed", "denied": "denied",
                        "rate_limited": "rate_limited", "error": "errors"}[decision]] += 1
            self._audit(**audit, decision=decision, reason=reason,
                        latency_ms=round((time.perf_counter() - t0) * 1000.0, 1))
            return reply

        policy = self.policies.get(agent_type)
        tool = next((t for t in self.catalog() if t.full_name == name), None)

        # 1) exists + allowlist (a hidden tool looks exactly like an unknown tool)
        if policy is None or tool is None or name not in policy.allow:
            reason = f"tool {name!r} is not available to agent type {agent_type!r}"
            return finish("denied", _error(msg_id, ERR_DENIED, f"policy: {reason}"), reason)
        if not isinstance(args, dict):
            return finish("denied", _error(msg_id, -32602, "arguments must be an object"),
                          "arguments not an object")

        # 2) argument rules
        try:
            args = self._apply_rules(policy, tool, args, user)
        except GatewayDenied as exc:
            return finish("denied", _error(msg_id, ERR_DENIED, f"policy: {exc}"), str(exc))

        # 3) rate limit
        bucket = self._bucket(policy, tool.server, sid)
        if bucket is not None:
            ok, left, retry = bucket.take()
            if not ok:
                reason = f"rate limited on {tool.server}: retry in {retry} s"
                return finish("rate_limited",
                              _error(msg_id, ERR_RATE_LIMITED, reason, {"retry_after_s": retry}), reason)

        # 4) credential from the vault (decrypted only here, only for this request)
        backend = self.backends[tool.server]
        headers: dict[str, str] = {}
        if backend.credential:
            try:
                secret = self.vault.get(user, backend.credential, session_id=sid, actor="gateway")
            except VaultError as exc:
                reason = f"credential {backend.credential}: {exc}"
                return finish("denied", _error(msg_id, ERR_DENIED, f"policy: {reason}"), reason)
            headers["Authorization"] = f"Bearer {secret}"
            audit["credential_used"] = backend.credential
            del secret

        # 5) forward
        tb = time.perf_counter()
        try:
            data = self._backend_rpc(backend, "tools/call",
                                     {"name": tool.tool, "arguments": args}, extra_headers=headers)
        except RuntimeError as exc:
            audit["backend_ms"] = round((time.perf_counter() - tb) * 1000.0, 1)
            return finish("error", _error(msg_id, ERR_BACKEND, str(exc)), str(exc))
        finally:
            headers.clear()                                        # drop the secret
        audit["backend_ms"] = round((time.perf_counter() - tb) * 1000.0, 1)

        if "error" in data:                                        # protocol error from the server
            err = data["error"]
            return finish("error", _error(msg_id, err.get("code", ERR_BACKEND),
                                          f"{tool.server}: {err.get('message')}"),
                          f"{tool.server}: {err.get('message')}")
        result = data.get("result") or {}
        audit["tool_is_error"] = bool(result.get("isError"))
        return finish("allowed", _result(msg_id, result))

    # ================================================================ audit

    def _audit(self, *, session_id: str, user_id: str, agent_type: str, tool: str, args: Any,
               decision: str, reason: Optional[str], latency_ms: float, backend_ms: Optional[float],
               tool_is_error: Optional[bool], credential_used: Optional[str]) -> None:
        canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
        try:
            with self.pool.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO tool_calls (session_id, user_id, agent_type, tool, decision, reason,
                        args_hash, args_preview, latency_ms, backend_ms, tool_is_error, credential_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (session_id, user_id, agent_type, tool, decision, reason,
                     hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16], canonical[:300],
                     latency_ms, backend_ms, tool_is_error, credential_used),
                )
        except Exception:  # noqa: BLE001  (auditing must never break the call path)
            log.exception("could not write tool_calls audit row")

    def tool_calls(self, session_id: Optional[str] = None, limit: int = 100) -> list[dict]:
        with self.pool.connection() as conn:
            if session_id:
                return conn.execute(
                    "SELECT * FROM tool_calls WHERE session_id = %s ORDER BY id LIMIT %s",
                    (session_id, limit),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM tool_calls ORDER BY id DESC LIMIT %s", (limit,)
            ).fetchall()

    def status(self) -> dict:
        with self._buckets_lock:
            buckets = {k: {"tokens": b.peek(), "capacity": b.capacity, "refill_per_s": b.rate}
                       for k, b in self._buckets.items()}
        return {
            "enabled": self.enabled,
            "servers": {n: {"url": b.url, "credential": b.credential} for n, b in self.backends.items()},
            "policies": {a: sorted(p.allow) for a, p in self.policies.items()},
            "catalog": [t.full_name for t in self._catalog],
            "buckets": buckets,
            "stats": self.stats,
        }


# ============================================================ JSON-RPC helpers

def _result(msg_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str, data: Optional[dict] = None) -> dict:
    err: dict[str, Any] = {"code": code, "message": message}
    if data:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}