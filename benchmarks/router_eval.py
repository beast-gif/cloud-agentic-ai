"""
Agentic Cloud - Phase 4, Step 4.5
Final evaluation of the model router: the same candidate lists ranked in every mode.

Per topic, candidates = 10 papers of the topic + 2 papers from each of 2 other topics
(all real arXiv via the papers MCP server, cached 1 h; never fallback placeholders).

Phase A  COLD cache (optional, --cold): the embedding cache is cleared, then every topic is
         ranked once in cascade mode with the latency budget -> shows the latency switch
         (fresh papers -> cloud) and the background warm-up.
         The script then waits until the background warm-up has finished.
Phase B  WARM cache: every topic in 4 modes
           local            embeddings only                         Rs 0
           cascade_norecal  cascade without online recalibration    (ablation)
           cascade          cascade + recalibration (the design)
           cloud            the cloud scores every paper             (reference)

Quality is measured against the cloud run of the same topic (k = 5):
  tie-aware hits  = shortlisted papers whose cloud score >= the cloud's k-th best score
                    (the cloud scores in integers, so many papers tie)
  quality ratio   = mean cloud score of the shortlist / mean of the best possible shortlist
  intrusions      = papers from OTHER topics in the shortlist (should be 0)
Cost and latency come from the router's own response (cost_inr, latency_ms).

Usage (control plane + MCP servers + Ollama running; prices configured in config.yaml):
  python -m benchmarks.router_eval                 # warm-cache comparison only (~Rs 0.5)
  python -m benchmarks.router_eval --cold          # + cold-cache phase (clears the cache, ~Rs 1-2)
  python -m benchmarks.router_eval --topics "KV cache compression" "mixture of experts"
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from benchmarks.common import PAPERS_URL, ControlPlane, direct_mcp

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOPICS = [
    "KV cache compression",
    "speculative decoding",
    "protein structure prediction",
    "mixture of experts",                      # stop word in the query
    "retrieval augmented generation",
    "diffusion models for image generation",   # stop word + broad topic
]
WARM_MODES = ["local", "cascade_norecal", "cascade", "cloud"]


# ============================================================ data

def search_real(topic: str, n: int, attempts: int = 4, wait_s: float = 45.0) -> list[dict]:
    for a in range(1, attempts + 1):
        _, body, _ = direct_mcp(PAPERS_URL, "search", {"query": topic, "max_results": n})
        sc = (body.get("result") or {}).get("structuredContent") or {}
        if sc.get("source") == "arxiv":
            return [{"id": p["id"], "title": p["title"], "abstract": p.get("abstract", "")}
                    for p in sc.get("papers", []) if p.get("abstract")]
        print(f"  '{topic}': source={sc.get('source')} ({sc.get('note')}) - attempt {a}/{attempts}, "
              f"waiting {wait_s:.0f} s")
        if a < attempts:
            time.sleep(wait_s)
    raise SystemExit(f"arXiv unavailable for '{topic}' - try again later")


def build_candidates(topics: list[str], per_topic: int, negatives: int) -> dict[str, dict]:
    own = {}
    for t in topics:
        own[t] = search_real(t, per_topic)
        print(f"  {t}: {len(own[t])} papers")
    out = {}
    for i, t in enumerate(topics):
        if not own[t]:
            print(f"  skipping '{t}': 0 papers")
            continue
        papers = list(own[t])
        ids = {p["id"] for p in papers}
        others = [topics[(i + 1) % len(topics)], topics[(i + 2) % len(topics)]]
        for o in others:
            for p in own.get(o, [])[:negatives]:
                if p["id"] not in ids:
                    papers.append(p)
                    ids.add(p["id"])
        out[t] = {"papers": papers, "own_ids": {p["id"] for p in own[t]}}
    return out


# ============================================================ calls + metrics

def rank(cp: ControlPlane, topic: str, papers: list[dict], k: int, mode: str,
         latency_budget_ms: float | None = None) -> dict:
    body = {"user_id": "router-eval", "topic": topic, "papers": papers, "k": k,
            "mode": "cascade" if mode.startswith("cascade") else mode}
    if mode == "cascade_norecal":
        body["recalibrate"] = False
    if mode.startswith("cascade"):
        # warm runs: a huge budget so the latency switch never fires; cold runs pass a real one
        body["latency_budget_ms"] = latency_budget_ms if latency_budget_ms is not None else 1e9
    r = cp.post("/admin/llm/rank", body)
    if r.status_code != 200:
        raise RuntimeError(f"{mode} on '{topic}': HTTP {r.status_code} {r.text[:300]}")
    return r.json()


def score_vs_cloud(out: dict, cloud_scores: dict[str, float], own_ids: set[str], k: int) -> dict:
    ranked_cloud = sorted(cloud_scores.values(), reverse=True)
    kth = ranked_cloud[k - 1]
    ideal = sum(ranked_cloud[:k]) / k
    short = out["shortlist"]
    mean_sc = sum(cloud_scores.get(pid, 0.0) for pid in short) / k
    s = out["summary"]
    return {
        "effective_mode": out["effective_mode"],
        "latency_ms": s["latency_ms"],
        "cost_inr": s.get("cost_inr"),
        "cloud_calls": s["cloud_calls"],
        "escalated": s.get("escalated", 0),
        "cache_hits": s.get("cache_hits"),
        "embedded": s.get("embedded"),
        "tie_aware_hits": sum(cloud_scores.get(pid, 0.0) >= kth for pid in short),
        "quality_ratio": round(mean_sc / ideal, 4) if ideal else None,
        "intrusions": sum(pid not in own_ids for pid in short),
        "recal_shift": (s.get("recalibration") or {}).get("shift"),
        "latency_decision": (s.get("latency_estimate") or {}).get("decision"),
        "shortlist": short,
    }


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    idx = min(len(v) - 1, max(0, round(q * (len(v) - 1))))
    return round(v[idx], 1)


def wait_for_warmup(cp: ControlPlane, timeout_s: float = 600.0) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        st = cp.get("/admin/router")
        if not st.get("warming_pending"):
            return round(time.perf_counter() - t0, 1)
        print(f"  waiting for background warm-up: {st['warming_pending']} papers pending, "
              f"postgres rows {st['vector_cache'].get('postgres_rows')}")
        time.sleep(10)
    print("  warm-up still running after the timeout - continuing anyway")
    return round(time.perf_counter() - t0, 1)


# ============================================================ main

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 model-router evaluation")
    ap.add_argument("--topics", nargs="+", default=TOPICS)
    ap.add_argument("--per-topic", type=int, default=10)
    ap.add_argument("--negatives", type=int, default=2, help="papers from EACH of 2 other topics")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cold", action="store_true", help="also run the cold-cache phase (clears the cache!)")
    ap.add_argument("--latency-budget", type=float, default=10000.0, help="ms, cold phase")
    args = ap.parse_args()

    cp = ControlPlane(timeout_s=600.0)
    st = cp.get("/admin/router")
    if not st.get("prices_configured"):
        print("note: prices are not configured in config.yaml -> costs will be None")

    print("building candidate lists from arXiv...")
    data = build_candidates(args.topics, args.per_topic, args.negatives)
    topics = list(data)
    report: dict = {"started": datetime.now().isoformat(timespec="seconds"), "k": args.k,
                    "topics": {t: {"candidates": len(d["papers"]), "own": len(d["own_ids"])} for t, d in data.items()},
                    "router_settings": {"rank": st.get("rank"), "latency": st.get("latency"),
                                        "cloud_model": (st.get("cloud") or {}).get("model")},
                    "cold": {}, "warm": {t: {} for t in topics}}

    # ---------------- Phase A: cold cache
    if args.cold:
        print(f"\nPHASE A - cold cache (latency budget {args.latency_budget:.0f} ms)")
        print(f"  cleared: {cp.post('/admin/router/clear-cache').json()}")
        for t in topics:
            out = rank(cp, t, data[t]["papers"], args.k, "cascade", latency_budget_ms=args.latency_budget)
            s = out["summary"]
            report["cold"][t] = {"effective_mode": out["effective_mode"], "latency_ms": s["latency_ms"],
                                 "cost_inr": s.get("cost_inr"), "cloud_calls": s["cloud_calls"],
                                 "latency_estimate": s.get("latency_estimate"),
                                 "background_warmup": s.get("background_warmup"), "shortlist": out["shortlist"]}
            print(f"  {t:42} {out['effective_mode']:8} {s['latency_ms']:>8.0f} ms  "
                  f"cloud {s['cloud_calls']:>2}  Rs {s.get('cost_inr')}")
        report["warmup_wait_s"] = wait_for_warmup(cp)
        print(f"  warm-up finished after {report['warmup_wait_s']} s")

    # ---------------- Phase B: warm cache, all modes
    print("\nPHASE B - warm cache, 4 modes per topic")
    # make sure every candidate is embedded (no-op if Phase A warmed them)
    for t in topics:
        rank(cp, t, data[t]["papers"], args.k, "local")
    for t in topics:
        outs = {m: rank(cp, t, data[t]["papers"], args.k, m) for m in WARM_MODES}
        cloud_scores = {p["id"]: p["score"] for p in outs["cloud"]["papers"]}
        report["warm"][t]["cloud_scores"] = cloud_scores
        for m in WARM_MODES:
            report["warm"][t][m] = score_vs_cloud(outs[m], cloud_scores, data[t]["own_ids"], args.k)
        row = "  ".join(f"{m}={report['warm'][t][m]['tie_aware_hits']}/{args.k}" for m in WARM_MODES)
        print(f"  {t:42} {row}  (shift {report['warm'][t]['cascade']['recal_shift']})")

    # ---------------- aggregate
    agg = {}
    for m in WARM_MODES:
        rows = [report["warm"][t][m] for t in topics]
        costs = [r["cost_inr"] for r in rows if r["cost_inr"] is not None]
        agg[m] = {
            "tie_aware_mean": round(statistics.fmean(r["tie_aware_hits"] for r in rows), 2),
            "tie_aware_5of5": sum(r["tie_aware_hits"] == args.k for r in rows),
            "quality_ratio_mean": round(statistics.fmean(r["quality_ratio"] for r in rows), 4),
            "intrusions_total": sum(r["intrusions"] for r in rows),
            "cloud_calls_mean": round(statistics.fmean(r["cloud_calls"] for r in rows), 2),
            "cost_inr_total": round(sum(costs), 4) if costs else None,
            "cost_inr_per_rank": round(statistics.fmean(costs), 5) if costs else None,
            "latency_p50_ms": pct([r["latency_ms"] for r in rows], 0.5),
            "latency_p95_ms": pct([r["latency_ms"] for r in rows], 0.95),
        }
    cloud_cost = agg["cloud"]["cost_inr_per_rank"]
    for m in WARM_MODES:
        c = agg[m]["cost_inr_per_rank"]
        agg[m]["cost_vs_cloud"] = round(c / cloud_cost, 3) if c is not None and cloud_cost else None
    report["aggregate_warm"] = agg
    if report["cold"]:
        cold = list(report["cold"].values())
        report["aggregate_cold"] = {
            "switched_to_cloud": sum(r["effective_mode"] == "cloud" for r in cold),
            "topics": len(cold),
            "latency_p50_ms": pct([r["latency_ms"] for r in cold], 0.5),
            "cost_inr_total": round(sum(r["cost_inr"] or 0 for r in cold), 4),
        }

    # ---------------- print
    print(f"\nWARM CACHE - {len(topics)} topics, k={args.k}, quality measured against the cloud run")
    print(f"{'mode':16} {'tie-aware':>9} {'5/5':>5} {'quality':>8} {'intrude':>8} {'cloud/rank':>10} "
          f"{'Rs/rank':>9} {'vs cloud':>9} {'p50 ms':>8} {'p95 ms':>8}")
    for m in WARM_MODES:
        a = agg[m]
        print(f"{m:16} {a['tie_aware_mean']:>7}/5 {a['tie_aware_5of5']:>3}/{len(topics)} "
              f"{a['quality_ratio_mean']:>8} {a['intrusions_total']:>8} {a['cloud_calls_mean']:>10} "
              f"{str(a['cost_inr_per_rank']):>9} {str(a['cost_vs_cloud']):>9} "
              f"{str(a['latency_p50_ms']):>8} {str(a['latency_p95_ms']):>8}")
    if report.get("aggregate_cold"):
        c = report["aggregate_cold"]
        print(f"\nCOLD CACHE (cascade, budget {args.latency_budget:.0f} ms): switched to cloud on "
              f"{c['switched_to_cloud']}/{c['topics']} topics, p50 {c['latency_p50_ms']} ms, Rs {c['cost_inr_total']}; "
              f"background warm-up took {report.get('warmup_wait_s')} s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"router_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved: {out_path}")
    cp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())