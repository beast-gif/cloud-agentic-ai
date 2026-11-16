"""
Agentic Cloud - Phase 2, Step 2.5
Memory scaling benchmark: how does L3 search time grow with the number of memories,
and what does the HNSW index trade between speed and accuracy?

It talks to Postgres DIRECTLY (not through the API) so it can insert thousands of
vectors quickly and switch the query plan. Uses a separate user and embedding model
("scale-bench" / "bench-random-768"), so it never mixes with real memories, and
deletes its rows at the end (unless --keep).

For each size N (cumulative: 100 -> 1,000 -> 5,000 -> 10,000 by default):
  1. insert vectors until the user has N memories (COPY = fast bulk insert)
  2. run Q queries in three modes and time them:
       exact : sequential scan + sort over all N rows       (always correct, O(N))
       hnsw  : forced HNSW index scan                        (approximate, ~O(log N))
       auto  : the real API query (user filter), planner decides
  3. recall@k of hnsw = |hnsw top-k  intersect  exact top-k| / k   (1.0 = identical results)

Theory to compare against:
  exact search does N x 768 multiply-adds per query:
     N = 10,000 -> 7.68 million;   N = 1,000,000 -> 768 million
  HNSW visits roughly ef_search x (a few) nodes per layer, almost independent of N.

Data: random unit vectors in 768 dimensions; queries are noisy copies of stored vectors
(like "a topic close to something I read"). Random high-dimensional data is a HARD case
for HNSW, so recall here is a lower bound of what real embeddings get.

Usage (from the agentic-cloud/ folder; control plane does NOT need to run):
  python -m benchmarks.memory_scaling
  python -m benchmarks.memory_scaling --sizes 100 1000 5000 10000 20000 --queries 50
  python -m benchmarks.memory_scaling --ef-search 100      # more accurate, slower HNSW
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from psycopg import connect
from psycopg.rows import dict_row

from control_plane.settings import load_config

USER = "scale-bench"
MODEL = "bench-random-768"
DIM = 768
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ============================================================ vectors

def unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def random_unit(rng: random.Random) -> list[float]:
    return unit([rng.gauss(0.0, 1.0) for _ in range(DIM)])


def noisy_copy(v: list[float], rng: random.Random, noise: float) -> list[float]:
    return unit([x + rng.gauss(0.0, noise / math.sqrt(DIM)) for x in v])


def lit(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# ============================================================ db helpers

def count_rows(conn) -> int:
    return conn.execute(
        "SELECT count(*) AS n FROM memories WHERE user_id = %s AND embedding_model = %s",
        (USER, MODEL),
    ).fetchone()["n"]


def bulk_insert(conn, vectors: list[list[float]], start_idx: int) -> float:
    """COPY rows in one stream. Returns seconds taken (includes HNSW index maintenance)."""
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        with cur.copy(
            "COPY memories (user_id, kind, content, embedding, embedding_model) FROM STDIN"
        ) as copy:
            for i, v in enumerate(vectors):
                copy.write_row((USER, "note", f"synthetic memory #{start_idx + i}", lit(v), MODEL))
    return time.perf_counter() - t0


def query(conn, mode: str, qvec: str, k: int, ef_search: int) -> tuple[list[int], float, str]:
    """Run one top-k query in a given mode. Returns (ids, ms, plan_index_used)."""
    if mode == "exact":
        settings = ["SET LOCAL enable_indexscan = off", "SET LOCAL enable_bitmapscan = off"]
        sql = ("SELECT id FROM memories WHERE embedding_model = %(m)s "
               "ORDER BY embedding <=> %(v)s::vector LIMIT %(k)s")
    elif mode == "hnsw":
        settings = ["SET LOCAL enable_seqscan = off", "SET LOCAL enable_bitmapscan = off",
                    f"SET LOCAL hnsw.ef_search = {int(ef_search)}"]
        sql = ("SELECT id FROM memories WHERE embedding_model = %(m)s "
               "ORDER BY embedding <=> %(v)s::vector LIMIT %(k)s")
    else:  # auto: exactly what the Memory API runs
        settings = [f"SET LOCAL hnsw.ef_search = {int(ef_search)}"]
        sql = ("SELECT id FROM memories WHERE user_id = %(u)s AND embedding_model = %(m)s "
               "ORDER BY embedding <=> %(v)s::vector LIMIT %(k)s")

    params = {"m": MODEL, "u": USER, "v": qvec, "k": k}
    with conn.transaction():
        for s in settings:
            conn.execute(s)
        t0 = time.perf_counter()
        ids = [r["id"] for r in conn.execute(sql, params).fetchall()]
        ms = (time.perf_counter() - t0) * 1000.0
        plan = conn.execute("EXPLAIN " + sql, params).fetchall()
    plan_text = " ".join(next(iter(r.values())) for r in plan)
    if "idx_memories_embedding_hnsw" in plan_text:
        used = "hnsw"
    elif "Seq Scan" in plan_text:
        used = "seq_scan"
    elif "idx_memories_user" in plan_text:
        used = "btree(user)"
    else:
        used = "other"
    return ids, ms, used


def p95(vals: list[float]) -> float:
    s = sorted(vals)
    return s[max(1, math.ceil(0.95 * len(s))) - 1]


# ============================================================ main

def main() -> int:
    ap = argparse.ArgumentParser(description="L3 memory search scaling benchmark")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 5000, 10000])
    ap.add_argument("--queries", type=int, default=30, help="queries per size and mode")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef-search", type=int, default=40, help="HNSW candidate list size at query time")
    ap.add_argument("--noise", type=float, default=0.6, help="query = stored vector + this much noise")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep", action="store_true", help="keep the synthetic rows afterwards")
    args = ap.parse_args()

    cfg = load_config()
    rng = random.Random(args.seed)
    sizes = sorted(set(args.sizes))

    conn = connect(cfg["postgres"]["host_dsn"], autocommit=True, row_factory=dict_row)
    existing = count_rows(conn)
    if existing:
        print(f"removing {existing} leftover rows from a previous run")
        conn.execute("DELETE FROM memories WHERE user_id = %s AND embedding_model = %s", (USER, MODEL))

    stored: list[list[float]] = []
    results = []
    print(f"sizes={sizes}, queries={args.queries}, k={args.k}, ef_search={args.ef_search}, "
          f"noise={args.noise}\n")
    print(f"{'N':>8} {'insert_s':>9} | {'exact p50':>10} {'p95':>8} | {'hnsw p50':>9} {'p95':>8} "
          f"{'recall@k':>9} | {'auto p50':>9} {'plan':>12} | {'N x 768 ops':>12}")
    print("-" * 118)

    try:
        for n in sizes:
            # 1) grow the dataset to n rows
            new = [random_unit(rng) for _ in range(n - len(stored))]
            insert_s = bulk_insert(conn, new, len(stored)) if new else 0.0
            stored.extend(new)
            conn.execute("ANALYZE memories")  # fresh statistics so the planner sees the new size

            # 2) queries = noisy copies of random stored vectors
            qs = [lit(noisy_copy(stored[rng.randrange(len(stored))], rng, args.noise))
                  for _ in range(args.queries)]

            timing = {"exact": [], "hnsw": [], "auto": []}
            recalls = []
            auto_plans = set()
            for q in qs:
                exact_ids, ms, _ = query(conn, "exact", q, args.k, args.ef_search)
                timing["exact"].append(ms)
                hnsw_ids, ms, _ = query(conn, "hnsw", q, args.k, args.ef_search)
                timing["hnsw"].append(ms)
                _, ms, plan = query(conn, "auto", q, args.k, args.ef_search)
                timing["auto"].append(ms)
                auto_plans.add(plan)
                recalls.append(len(set(exact_ids) & set(hnsw_ids)) / args.k)

            row = {
                "n": n,
                "insert_s": round(insert_s, 2),
                "exact_p50_ms": round(statistics.median(timing["exact"]), 2),
                "exact_p95_ms": round(p95(timing["exact"]), 2),
                "hnsw_p50_ms": round(statistics.median(timing["hnsw"]), 2),
                "hnsw_p95_ms": round(p95(timing["hnsw"]), 2),
                "hnsw_recall_at_k": round(statistics.fmean(recalls), 3),
                "auto_p50_ms": round(statistics.median(timing["auto"]), 2),
                "auto_plan": "/".join(sorted(auto_plans)),
                "exact_ops": n * DIM,
            }
            results.append(row)
            print(f"{n:>8} {row['insert_s']:>9} | {row['exact_p50_ms']:>10} {row['exact_p95_ms']:>8} | "
                  f"{row['hnsw_p50_ms']:>9} {row['hnsw_p95_ms']:>8} {row['hnsw_recall_at_k']:>9} | "
                  f"{row['auto_p50_ms']:>9} {row['auto_plan']:>12} | {row['exact_ops']:>12,}")
    finally:
        if not args.keep:
            deleted = conn.execute(
                "DELETE FROM memories WHERE user_id = %s AND embedding_model = %s", (USER, MODEL)
            ).rowcount
            print(f"\ncleanup: deleted {deleted} synthetic rows")
        conn.close()

    if len(results) >= 2:
        a, b = results[0], results[-1]
        growth_n = b["n"] / a["n"]
        print(f"\nN grew {growth_n:.0f}x ({a['n']} -> {b['n']}):")
        print(f"  exact p50 grew {b['exact_p50_ms'] / max(a['exact_p50_ms'], 0.01):.1f}x "
              f"(theory for O(N): {growth_n:.0f}x)")
        print(f"  hnsw  p50 grew {b['hnsw_p50_ms'] / max(a['hnsw_p50_ms'], 0.01):.1f}x "
              f"(theory for O(log N): {math.log(b['n']) / math.log(a['n']):.1f}x)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"memory_scaling_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": vars(args), "results": results}, f, indent=2)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())