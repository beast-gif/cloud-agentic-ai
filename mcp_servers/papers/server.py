"""
Agentic Cloud - Phase 3, Step 3.2
MCP server "notify": a MOCK notification provider that requires an API key.

It plays the role of an external service (like a Telegram bot or an email API):
  - every account has an API key; calls must send  Authorization: Bearer <key>
  - messages are stored in memory per account (nothing leaves your machine)

Tools (all need a valid API key):
  send(to, message)       -> deliver a message from this account
  list_messages(limit)    -> this account's sent messages
  delete_all_messages()   -> DANGEROUS: wipes the account's history
                             (exists on purpose: the gateway must hide it from agents)

Accounts come from the env var NOTIFY_ACCOUNTS:  "key1:account1,key2:account2"
(set in .env - these play the role of the keys the provider issued to its customers).

Debug (mock only, NOT part of MCP):  GET /admin/messages   all stored messages
"""

from __future__ import annotations

import logging
import os
import threading
import time

import uvicorn

from common.mcp_http import CallContext, MCPServer, ToolError, Unauthorized, bearer_token

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s notify: %(message)s")
log = logging.getLogger("notify")

MAX_MESSAGE_CHARS = 2000


def _parse_accounts(raw: str) -> dict[str, str]:
    accounts = {}
    for pair in raw.split(","):
        if ":" in pair:
            key, name = pair.split(":", 1)
            if key.strip() and name.strip():
                accounts[key.strip()] = name.strip()
    return accounts


ACCOUNTS = _parse_accounts(os.environ.get("NOTIFY_ACCOUNTS", ""))
_messages: list[dict] = []
_lock = threading.Lock()
_next_id = 1

server = MCPServer("notify", "0.1.0",
                   instructions="Send short notifications. Requires the account's API key.")


def _account(ctx: CallContext) -> str:
    """Resolve the API key in the Authorization header to an account, or 401."""
    key = bearer_token(ctx)
    if not key:
        raise Unauthorized("missing API key (Authorization: Bearer <key>)")
    account = ACCOUNTS.get(key)
    if account is None:
        raise Unauthorized("invalid API key")
    return account


@server.tool(
    "send",
    "Send a notification message to a recipient (chat id).",
    {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "recipient chat id"},
            "message": {"type": "string", "maxLength": MAX_MESSAGE_CHARS},
        },
        "required": ["to", "message"],
    },
)
def send(args: dict, ctx: CallContext) -> dict:
    global _next_id
    account = _account(ctx)
    to, message = str(args["to"]).strip(), str(args["message"])
    if not to:
        raise ToolError("'to' is empty")
    if len(message) > MAX_MESSAGE_CHARS:
        raise ToolError(f"message longer than {MAX_MESSAGE_CHARS} characters")
    with _lock:
        msg = {"id": _next_id, "account": account, "to": to, "message": message,
               "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        _messages.append(msg)
        _next_id += 1
    log.info("delivered #%d from %s to %s (%d chars)", msg["id"], account, to, len(message))
    return {"delivered": True, "id": msg["id"], "account": account, "to": to}


@server.tool(
    "list_messages",
    "List messages sent by this account (newest first).",
    {"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}},
)
def list_messages(args: dict, ctx: CallContext) -> dict:
    account = _account(ctx)
    limit = max(1, min(int(args.get("limit", 20)), 200))
    with _lock:
        mine = [m for m in reversed(_messages) if m["account"] == account][:limit]
    return {"account": account, "messages": mine}


@server.tool(
    "delete_all_messages",
    "Delete ALL messages of this account. Irreversible.",
    {"type": "object", "properties": {}},
)
def delete_all_messages(args: dict, ctx: CallContext) -> dict:
    account = _account(ctx)
    with _lock:
        before = len(_messages)
        _messages[:] = [m for m in _messages if m["account"] != account]
        deleted = before - len(_messages)
    log.warning("account %s deleted %d messages", account, deleted)
    return {"account": account, "deleted": deleted}


app = server.app()


@app.get("/admin/messages")
def admin_messages():
    """Mock-only debug view (a real provider would never expose this)."""
    with _lock:
        return {"accounts": sorted(set(ACCOUNTS.values())), "messages": list(reversed(_messages))}


if __name__ == "__main__":
    if not ACCOUNTS:
        log.warning("NOTIFY_ACCOUNTS is empty: every call will be unauthorized")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9000")), log_level="info")