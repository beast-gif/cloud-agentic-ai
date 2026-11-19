"""
Agentic Cloud - Phase 3, Step 3.5
Gateway benchmark: what does going through the MCP gateway cost?

For each tool it measures N calls two ways, interleaved (so both see the same conditions):
  direct   : this script -> MCP server            (with the gateway secret; no policy, no audit)
  gateway  : this script -> control plane /mcp -> MCP server   (session token, full checks)

and reports:
  client_ms         end-to-end time seen by the caller
  added_ms          gateway client_ms - direct client_ms  (p50 vs p50): the real price of the extra hop
  internal_ms       gateway latency_ms - backend_ms, from the tool_calls audit table:
                    time spent INSIDE the gateway (auth is before this, audit write after it)
  backend_ms        time the MCP server took, as seen by the gateway

Tools measured:
  papers.search  cached query (no arXiv traffic after the first call): pure overhead, no credential
  notify.send    owner_only rule + vault decrypt (DB read + HMAC/AES + audit row) + forward

IMPORTANT: the gateway's token buckets (papers 3 burst / 0.33 per s, notify 3 / 0.05 per s)
would rate-limit a benchmark. For the run, raise them in config.yaml, e.g.
    papers: {capacity: 1000, refill_per_s: 100, scope: global}
    notify: {capacity: 1000, refill_per_s: 100, scope: session}
restart the control plane, benchmark, then PUT THE REAL LIMITS BACK.
(Or keep the limits and pass --pace 3.1 for papers only: slow, and notify will still be limited.)

Usage (from the agentic-cloud/ folder):
  python -m benchmarks.gateway_benchmark --n 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from psycopg import connect
from psycopg.rows import dict_row

from benchmarks.common import (
    CFG, NOTIFY_URL, PAPERS_URL, ControlPlane, GatewayClient, container_env,
    describe, direct_mcp, provider_key,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
USER = "amrith"
QUERY = "KV cache compression"


def main() -> int:
    ap = argparse.ArgumentParser(description="MCP gateway overhead benchmark")
    ap.add_argument("--n", type=int, default=50, help="calls per tool and path")
    ap.add_argument("--pace", type=float, default=0.0, help="seconds to sleep between iterations")
    args = ap.parse_args()

    cp = ControlPlane()
    hz = cp.get("/healthz")
    if hz["reaper"]["idle_timeout_s"] < 120:
        print("set lifecycle.idle_timeout_s >= 120 for the benchmark (the session must stay alive).")
        return 1
    limits = CFG["mcp"]["policies"]["research"].get("rate_limits", {})
    for server in ("papers", "notify"):
        cap = (limits.get(server) or {}).get("capacity", 0)
        if cap < args.n * 2 and args.pace == 0:
            print(f"rate limit for '{server}' (capacity {cap}) would block this benchmark.\n"
                  f"Temporarily raise it in config.yaml (see the docstring), restart the control plane, re-run.")
            return 1

    creds = {c["provider"]: c for c in cp.get(f"/admin/vault/{USER}")}
    owner_chat = ((creds.get("notify") or {}).get("metadata") or {}).get("owner_chat")
    key = provider_key(USER)
    if not owner_chat or not key:
        print(f"need {USER}'s notify credential with owner_chat in the vault and NOTIFY_ACCOUNTS in .env")
        return 1

    # one live session = one real agent token
    r = cp.create_session(USER, QUERY)
    sid = r["session"]["session_id"]
    gw = GatewayClient(container_env(sid, "MEMORY_TOKEN"))
    print(f"session {sid}; n={args.n} per tool and path")

    # warm-up: fill the papers cache and open connections (not measured)
    direct_mcp(PAPERS_URL, "search", {"query": QUERY, "max_results": 5})
    gw.call("papers.search", {"query": QUERY, "max_results": 5})

    db = connect(CFG["postgres"]["host_dsn"], autocommit=True, row_factory=dict_row)
    start_id = db.execute("SELECT coalesce(max(id), 0) AS m FROM tool_calls").fetchone()["m"]

    times = {"papers.search": {"direct": [], "gateway": []}, "notify.send": {"direct": [], "gateway": []}}
    problems = 0
    for i in range(args.n):
        s, body, ms = direct_mcp(PAPERS_URL, "search", {"query": QUERY, "max_results": 5})
        times["papers.search"]["direct"].append(ms) if s == 200 and "result" in body else None
        s, body, ms = gw.call("papers.search", {"query": QUERY, "max_results": 5})
        if s == 200 and "result" in body:
            times["papers.search"]["gateway"].append(ms)
        else:
            problems += 1

        s, body, ms = direct_mcp(NOTIFY_URL, "send", {"to": owner_chat, "message": f"bench direct {i}"},
                                 headers={"Authorization": f"Bearer {key}"})
        times["notify.send"]["direct"].append(ms) if s == 200 and "result" in body else None
        s, body, ms = gw.call("notify.send", {"message": f"bench gateway {i}"})
        if s == 200 and "result" in body:
            times["notify.send"]["gateway"].append(ms)
        else:
            problems += 1
        if args.pace:
            time.sleep(args.pace)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.n}")

    rows = db.execute(
        "SELECT tool, decision, latency_ms, backend_ms FROM tool_calls "
        "WHERE session_id = %s AND id > %s", (sid, start_id)).fetchall()
    db.close()

    report: dict = {}
    print(f"\n{'tool':<15}{'path':<9}{'p50 ms':>10}{'p95 ms':>10}{'n':>5}")
    print("-" * 49)
    for tool, paths in times.items():
        d, g = describe(paths["direct"]), describe(paths["gateway"])
        allowed = [r for r in rows if r["tool"] == tool and r["decision"] == "allowed" and r["backend_ms"] is not None]
        internal = describe([r["latency_ms"] - r["backend_ms"] for r in allowed])
        backend = describe([r["backend_ms"] for r in allowed])
        added = round(g["p50"] - d["p50"], 2) if d and g else None
        report[tool] = {"direct_client_ms": d, "gateway_client_ms": g, "added_p50_ms": added,
                        "gateway_internal_ms": internal, "backend_ms": backend}
        for name, st in (("direct", d), ("gateway", g)):
            if st:
                print(f"{tool:<15}{name:<9}{st['p50']:>10}{st['p95']:>10}{st['n']:>5}")
        if internal:
            print(f"{'':<15}{'inside':<9}{internal['p50']:>10}{internal['p95']:>10}{internal['n']:>5}"
                  f"   <- gateway's own work (latency_ms - backend_ms)")
        print(f"{'':<15}added p50 by going through the gateway: {added} ms\n")

    if problems:
        print(f"WARNING: {problems} gateway calls failed or were rate-limited (see /sessions/{sid}/tool-calls)")

    cp.freeze(sid)
    gw.close()
    cp.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"gateway_bench_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"session_id": sid, "n": args.n, "problems": problems, "results": report}, f, indent=2)
    print(f"saved: {out}\nRemember to restore the real rate limits in config.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())