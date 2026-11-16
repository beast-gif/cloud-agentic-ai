"""
Agentic Cloud - Phase 1 Step 1.5, updated in Phase 2 Step 2.5
Benchmark: cold start vs wake vs fast path, over many runs (+ memory cost).

Talks ONLY to the control plane API (it must be running: python -m control_plane.main).

For each run:
  wake cycle : POST /sessions (cold start)  -> POST /freeze  -> POST /events (wake)
  fast cycle : POST /sessions (cold start)  -> POST /events immediately (fast path)

Phase 2 additions:
  - every cycle uses its OWN user_id, so long-term memory never makes the agent skip
    papers between runs (each run does the same amount of work)
  - records the agent's memory stats per measurement: memory_calls, memory_ms, memory_errors
      cold start: recall_memory (1 search)
      wake/fast : remember (one write per summarized paper + 1 topic write)
  - --baseline <phase1 json>: prints p50 differences against Phase 1
  - cleans up afterwards: deletes the benchmark users' memories and their workspace volumes

Statistics per path and per stage:
  mean, p50 (median), p95 (nearest-rank), min, max, stdev
  nearest-rank p95: sort the n values, take the value at position ceil(0.95 * n)

Outputs:
  table in the terminal
  benchmarks/results/<tag>_<timestamp>.csv   (one row per measurement)
  benchmarks/results/<tag>_<timestamp>.json  (summary)

Usage (from the agentic-cloud/ folder):
  python -m benchmarks.wake_benchmark
  python -m benchmarks.wake_benchmark --n 20 --baseline benchmarks/results/phase1_XXXX.json
  python -m benchmarks.wake_benchmark --mode wake --n 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import secrets
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# stages reported by the control plane (timings_ms / agent fields)
STAGES = [
    "container_start_ms",
    "import_ms",
    "graph_build_ms",
    "ready_ms",
    "checkpoint_load_ms",
    "run_ms",
    "total_ms",
    "client_ms",       # measured by THIS script: request sent -> response received
    "memory_calls",    # Phase 2: Memory API calls made by the agent in this request
    "memory_ms",       # Phase 2: total time the agent spent in Memory API calls
    "memory_errors",   # Phase 2: failed Memory API calls (should be 0)
]

CSV_FIELDS = ["run", "path", "session_id", "user_id", "ok", "error"] + STAGES


# ============================================================ statistics

def percentile_nearest_rank(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile: value at position ceil(p/100 * n) (1-based)."""
    n = len(sorted_vals)
    k = max(1, math.ceil(p / 100.0 * n))
    return sorted_vals[k - 1]


def describe(values: list[float]) -> Optional[dict[str, float]]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 1),
        "p50": round(statistics.median(vals), 1),
        "p95": round(percentile_nearest_rank(vals, 95), 1),
        "min": round(vals[0], 1),
        "max": round(vals[-1], 1),
        "stdev": round(statistics.stdev(vals), 1) if len(vals) > 1 else 0.0,
    }


# ============================================================ API helpers

class Api:
    def __init__(self, base: str, timeout_s: float = 600.0) -> None:
        self.base = base.rstrip("/")
        self.http = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self.http.close()

    def _post(self, path: str, body: Optional[dict] = None) -> tuple[dict, float]:
        t0 = time.perf_counter()
        resp = self.http.post(f"{self.base}{path}", json=body or {})
        client_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code >= 400:
            raise RuntimeError(f"POST {path} -> HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json(), client_ms

    def check(self) -> dict:
        resp = self.http.get(f"{self.base}/healthz")
        resp.raise_for_status()
        return resp.json()

    def create(self, topic: str, user_id: str) -> tuple[dict, float]:
        return self._post("/sessions", {"user_id": user_id, "agent_type": "research",
                                        "input": {"topic": topic}})

    def freeze(self, sid: str) -> dict:
        data, _ = self._post(f"/sessions/{sid}/freeze")
        return data

    def event(self, sid: str, value: Any) -> tuple[dict, float]:
        return self._post(f"/sessions/{sid}/events", {"value": value})

    def forget_user(self, user_id: str) -> int:
        resp = self.http.delete(f"{self.base}/users/{user_id}/memories")
        return resp.json().get("deleted", 0) if resp.status_code == 200 else 0

    def gc_workspaces(self) -> list:
        resp = self.http.post(f"{self.base}/admin/workspaces/gc", params={"older_than_s": 0})
        return resp.json().get("deleted", []) if resp.status_code == 200 else []


def to_row(run: int, path: str, sid: str, user_id: str, resp: Optional[dict],
           client_ms: Optional[float], error: Optional[str] = None) -> dict:
    row: dict[str, Any] = {"run": run, "path": path, "session_id": sid, "user_id": user_id,
                           "ok": error is None, "error": error or ""}
    timings = (resp or {}).get("timings_ms", {}) or {}
    agent = (resp or {}).get("agent", {}) or {}
    memory = agent.get("memory") or {}
    for s in STAGES:
        row[s] = None
    for s in ("container_start_ms", "import_ms", "graph_build_ms", "ready_ms", "total_ms"):
        row[s] = timings.get(s)
    row["checkpoint_load_ms"] = agent.get("checkpoint_load_ms")
    row["run_ms"] = agent.get("run_ms")
    row["client_ms"] = round(client_ms, 1) if client_ms is not None else None
    if memory:
        row["memory_calls"] = memory.get("calls")
        row["memory_ms"] = memory.get("total_ms")
        row["memory_errors"] = memory.get("errors")
    return row


# ============================================================ cycles
# NOTE: the agent's memory stats are cumulative per CONTAINER. A wake starts a new
# container (stats = this request only). On the fast path the container is the same
# one that did the cold start, so its stats include the recall from the cold start;
# we subtract the cold-start numbers to get the fast-path request alone.

def _subtract_memory(row: dict, before: dict) -> None:
    for k in ("memory_calls", "memory_ms", "memory_errors"):
        if row.get(k) is not None and before.get(k) is not None:
            row[k] = round(row[k] - before[k], 1)


def wake_cycle(api: Api, run: int, topic: str, user_id: str) -> list[dict]:
    rows = []
    sid = ""
    try:
        created, c_ms = api.create(topic, user_id)
        sid = created["session"]["session_id"]
        rows.append(to_row(run, "cold_start", sid, user_id, created, c_ms))

        fr = api.freeze(sid)
        if not fr.get("frozen"):
            raise RuntimeError(f"freeze failed, state={fr.get('session', {}).get('state')}")

        woke, w_ms = api.event(sid, {"selected": [1, 3]})
        if woke.get("path") != "wake":
            raise RuntimeError(f"expected path 'wake', got {woke.get('path')!r}")
        rows.append(to_row(run, "wake", sid, user_id, woke, w_ms))
    except Exception as exc:  # noqa: BLE001
        rows.append(to_row(run, "wake", sid, user_id, None, None, error=repr(exc)))
    return rows


def fast_cycle(api: Api, run: int, topic: str, user_id: str) -> list[dict]:
    rows = []
    sid = ""
    try:
        created, c_ms = api.create(topic, user_id)
        sid = created["session"]["session_id"]
        cold = to_row(run, "cold_start", sid, user_id, created, c_ms)
        rows.append(cold)

        fast, f_ms = api.event(sid, {"selected": [2, 4]})
        if fast.get("path") != "fast":
            raise RuntimeError(f"expected path 'fast', got {fast.get('path')!r}")
        row = to_row(run, "fast", sid, user_id, fast, f_ms)
        _subtract_memory(row, cold)
        rows.append(row)
    except Exception as exc:  # noqa: BLE001
        rows.append(to_row(run, "fast", sid, user_id, None, None, error=repr(exc)))
    return rows


# ============================================================ reporting

def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for path in ("cold_start", "wake", "fast"):
        p_rows = [r for r in rows if r["path"] == path and r["ok"]]
        if not p_rows:
            continue
        summary[path] = {"runs": len(p_rows)}
        for s in STAGES:
            d = describe([r[s] for r in p_rows])
            if d:
                summary[path][s] = d

    # derived comparisons (use p50 = typical case)
    derived = {}
    wake, fast = summary.get("wake", {}), summary.get("fast", {})
    if "client_ms" in wake and "client_ms" in fast and fast["client_ms"]["p50"]:
        derived["wake_vs_fast_p50_ratio"] = round(wake["client_ms"]["p50"] / fast["client_ms"]["p50"], 1)
    if "checkpoint_load_ms" in wake and "total_ms" in wake and wake["total_ms"]["p50"]:
        derived["checkpoint_share_of_wake_pct"] = round(
            100.0 * wake["checkpoint_load_ms"]["p50"] / wake["total_ms"]["p50"], 2)
    if all(k in wake for k in ("container_start_ms", "import_ms", "total_ms")) and wake["total_ms"]["p50"]:
        derived["docker_plus_imports_share_of_wake_pct"] = round(
            100.0 * (wake["container_start_ms"]["p50"] + wake["import_ms"]["p50"]) / wake["total_ms"]["p50"], 1)
    for path in ("cold_start", "wake", "fast"):
        p = summary.get(path, {})
        if "memory_ms" in p and "total_ms" in p and p["total_ms"]["p50"]:
            derived[f"memory_share_of_{path}_pct"] = round(
                100.0 * p["memory_ms"]["p50"] / p["total_ms"]["p50"], 2)
    summary["derived"] = derived
    return summary


def print_table(summary: dict) -> None:
    cols = ["n", "mean", "p50", "p95", "min", "max", "stdev"]
    for path in ("cold_start", "wake", "fast"):
        if path not in summary:
            continue
        print(f"\n=== {path}  ({summary[path]['runs']} runs) ===")
        print(f"{'stage':<20}" + "".join(f"{c:>10}" for c in cols))
        print("-" * (20 + 10 * len(cols)))
        for s in STAGES:
            d = summary[path].get(s)
            if not d:
                continue
            print(f"{s:<20}" + "".join(f"{d[c]:>10}" for c in cols))

    if summary.get("derived"):
        print("\n=== derived (p50) ===")
        for k, v in summary["derived"].items():
            print(f"  {k:<40} {v}")


def print_baseline_diff(summary: dict, baseline_path: str) -> Optional[dict]:
    """Compare p50s with an earlier run (e.g. the Phase 1 JSON)."""
    try:
        with open(baseline_path, encoding="utf-8") as f:
            base = json.load(f)["summary"]
    except Exception as exc:  # noqa: BLE001
        print(f"\n(could not read baseline {baseline_path}: {exc!r})")
        return None

    diff: dict[str, dict] = {}
    print(f"\n=== vs baseline {Path(baseline_path).name} (p50) ===")
    print(f"{'path':<12}{'stage':<14}{'baseline':>12}{'now':>12}{'delta':>12}{'delta %':>10}")
    print("-" * 72)
    for path in ("cold_start", "wake", "fast"):
        for stage in ("run_ms", "total_ms", "client_ms"):
            b = ((base.get(path) or {}).get(stage) or {}).get("p50")
            n = ((summary.get(path) or {}).get(stage) or {}).get("p50")
            if b is None or n is None:
                continue
            d = round(n - b, 1)
            pct = round(100.0 * d / b, 1) if b else None
            diff.setdefault(path, {})[stage] = {"baseline": b, "now": n, "delta": d, "delta_pct": pct}
            print(f"{path:<12}{stage:<14}{b:>12}{n:>12}{d:>12}{pct:>10}")
    print("(run_ms is the agent's own work: the memory calls show up there.\n"
          " total_ms/client_ms also include container start + imports, which vary run to run.)")
    return diff


def save(rows: list[dict], summary: dict, meta: dict, tag: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"{tag}_{stamp}.csv"
    json_path = RESULTS_DIR / f"{tag}_{stamp}.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary}, f, indent=2)

    return csv_path, json_path


# ============================================================ main

def main() -> int:
    ap = argparse.ArgumentParser(description="Wake / fast-path benchmark (with memory cost)")
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="control plane URL")
    ap.add_argument("--n", type=int, default=10, help="runs per mode")
    ap.add_argument("--mode", choices=["both", "wake", "fast"], default="both")
    ap.add_argument("--topic", default="KV cache compression")
    ap.add_argument("--warmup", type=int, default=1, help="untimed warm-up cycles (results discarded)")
    ap.add_argument("--pause", type=float, default=2.0,
                    help="seconds between runs, so background cleanup does not overlap the next run")
    ap.add_argument("--tag", default="phase2", help="prefix of the result file names")
    ap.add_argument("--baseline", default=None, help="earlier result JSON to compare p50s against")
    ap.add_argument("--keep", action="store_true", help="keep benchmark memories and workspaces")
    args = ap.parse_args()

    api = Api(args.api)
    try:
        hz = api.check()
    except Exception as exc:  # noqa: BLE001
        print(f"Control plane not reachable at {args.api}: {exc!r}\n"
              f"Start it first:  python -m control_plane.main")
        return 1

    run_id = f"bench-{datetime.now():%H%M%S}-{secrets.token_hex(2)}"
    users: list[str] = []

    def user_for(mode: str, run: int) -> str:
        u = f"{run_id}-{mode}-{run}"
        users.append(u)
        return u

    modes = ["wake", "fast"] if args.mode == "both" else [args.mode]
    total_cycles = args.warmup + args.n * len(modes)
    print(f"Benchmark {run_id}: modes={modes}, n={args.n}, warmup={args.warmup}, pause={args.pause}s "
          f"-> {total_cycles} cycles (~{total_cycles * 9 // 60 + 1} min)")
    print(f"reaper idle timeout: {hz['reaper']['idle_timeout_s']} s, "
          f"embedder: {(hz.get('embedder') or {}).get('name', '?')}")

    # warm-up: first container start after a while is slower (disk cache), so discard it
    for i in range(args.warmup):
        print(f"warm-up {i + 1}/{args.warmup} ...", end=" ", flush=True)
        wake_cycle(api, -1, args.topic, user_for("warmup", i))
        print("done")
        time.sleep(args.pause)

    rows: list[dict] = []
    started = time.perf_counter()
    try:
        for mode in modes:
            cycle = wake_cycle if mode == "wake" else fast_cycle
            for run in range(1, args.n + 1):
                r = cycle(api, run, args.topic, user_for(mode, run))
                rows.extend(r)
                last = r[-1]
                if last["ok"]:
                    status = (f"client {last['client_ms']:.0f} ms, "
                              f"memory {last['memory_calls']} calls / {last['memory_ms']} ms")
                else:
                    status = f"ERROR {last['error'][:80]}"
                print(f"[{mode} {run:>2}/{args.n}] {status}", flush=True)
                time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\ninterrupted - summarizing what we have")

    elapsed_s = round(time.perf_counter() - started, 1)
    summary = summarize(rows)
    print_table(summary)

    baseline_diff = print_baseline_diff(summary, args.baseline) if args.baseline else None

    errors = [r for r in rows if not r["ok"]]
    if errors:
        print(f"\n{len(errors)} failed run(s):")
        for e in errors:
            print(f"  run {e['run']} {e['path']}: {e['error'][:150]}")

    if not args.keep:
        time.sleep(3)  # let background container cleanup finish before deleting volumes
        deleted_mem = sum(api.forget_user(u) for u in users)
        deleted_ws = api.gc_workspaces()
        print(f"\ncleanup: {deleted_mem} benchmark memories, {len(deleted_ws)} workspace volume(s) deleted")
    api.close()

    meta = {"run_id": run_id, "modes": modes, "n": args.n, "warmup": args.warmup,
            "pause_s": args.pause, "topic": args.topic, "elapsed_s": elapsed_s,
            "embedder": (hz.get("embedder") or {}).get("name"),
            "baseline": args.baseline, "baseline_diff": baseline_diff,
            "timestamp": datetime.now().isoformat()}
    csv_path, json_path = save(rows, summary, meta, args.tag)
    print(f"\nsaved: {csv_path}\n       {json_path}\nelapsed: {elapsed_s} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())