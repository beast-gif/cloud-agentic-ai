"""
Agentic Cloud - Phase 4, Step 4.2
Try the model router through its admin endpoints (no agent, no session needed).

  1. builds a candidate list for ONE topic from real arXiv (papers MCP server, cached):
     the topic's own results + a few papers from other topics (should be ranked low)
  2. POST /admin/llm/rank in the three modes  local | cascade | cloud  and compares:
     shortlist, cloud calls, cost, latency, and how far the cascade shortlist is from the cloud one
  3. POST /admin/llm/complete (summarize) in cascade and local mode
  4. GET /admin/router totals

Usage (control plane + MCP servers + Ollama running):
  python -m benchmarks.try_router
  python -m benchmarks.try_router --topic "speculative decoding" --per-topic 10
  Step 4.4 (latency budget):
  python -m benchmarks.try_router --topic "mixture of experts" --modes cascade --latency-budget 10000 --no-summary
"""

from __future__ import annotations

import argparse
import json

from benchmarks.common import PAPERS_URL, ControlPlane, direct_mcp

OTHER_TOPICS = ["protein structure prediction", "graph neural networks", "speculative decoding",
                "KV cache compression"]


def search(topic: str, n: int) -> list[dict]:
    _, body, _ = direct_mcp(PAPERS_URL, "search", {"query": topic, "max_results": n})
    sc = (body.get("result") or {}).get("structuredContent") or {}
    if sc.get("source") != "arxiv":
        raise SystemExit(f"'{topic}' came from source={sc.get('source')} ({sc.get('note')}) - "
                         f"arXiv is throttling, wait a minute and retry")
    return [{"id": p["id"], "title": p["title"], "abstract": p.get("abstract", "")}
            for p in sc.get("papers", []) if p.get("abstract")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="KV cache compression")
    ap.add_argument("--per-topic", type=int, default=10, help="papers from the topic itself")
    ap.add_argument("--negatives", type=int, default=2, help="papers from EACH other topic")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["local", "cascade", "cloud"], choices=["local", "cascade", "cloud"])
    ap.add_argument("--latency-budget", type=float, default=None, help="ms; cascade only (default: config)")
    ap.add_argument("--no-summary", action="store_true", help="skip the summarize part")
    args = ap.parse_args()

    print(f"candidates for '{args.topic}' (real arXiv, cached)...")
    papers = search(args.topic, args.per_topic)
    if not papers:
        raise SystemExit(f"arXiv returned 0 papers for '{args.topic}' - try another topic")
    own = {p["id"] for p in papers}
    for t in OTHER_TOPICS:
        if t.lower() != args.topic.lower():
            seen = {p["id"] for p in papers}
            papers += [p for p in search(t, args.negatives) if p["id"] not in seen]
    print(f"  {len(own)} from the topic + {len(papers) - len(own)} from other topics = {len(papers)}\n")

    cp = ControlPlane()
    results = {}
    for mode in args.modes:
        body = {"user_id": "router-test", "topic": args.topic, "papers": papers, "k": args.k, "mode": mode}
        if args.latency_budget is not None:
            body["latency_budget_ms"] = args.latency_budget
        r = cp.post("/admin/llm/rank", body)
        if r.status_code != 200:
            print(f"{mode}: HTTP {r.status_code} {r.text[:300]}")
            continue
        out = r.json()
        results[mode] = out
        s = out["summary"]
        print(f"=== mode {mode}  (effective {out['effective_mode']})")
        print(f"  latency {s['latency_ms']} ms | embedded {s['embedded']} (cache hits {s['cache_hits']}) | "
              f"cloud calls {s['cloud_calls']} (escalated {s['escalated']}, failed {s['cloud_failed']}, "
              f"skipped {s['skipped']}) | cost INR {s['cost_inr']}")
        if s.get("latency_estimate"):
            print(f"  latency estimate: {s['latency_estimate']}")
        if s.get("recalibration"):
            print(f"  recalibration: {s['recalibration']}")
        if s.get("background_warmup"):
            print(f"  background warm-up queued: {s['background_warmup']} papers")
        for p in out["papers"][: args.k + 3]:
            tag = "own " if p["id"] in own else "OTHER"
            cos = f"{p['cosine']:.3f}" if p["cosine"] is not None else "  -  "
            print(f"  #{p['rank']:<2} {tag} score {p['score']:>5}  cos {cos}  {p['route']:<5}  "
                  f"{p['title'][:55]:<55}  {p['reason'] or ''}")
        print()

    if "cloud" in results:
        # The cloud gives INTEGER scores, so many papers tie (e.g. seven 10s). Which 5 of the tied
        # papers it lists first is arbitrary, so plain overlap under-counts. Tie-aware version:
        #   hit = a shortlisted paper whose cloud score >= the cloud's k-th best score
        cloud_score = {p["id"]: p["score"] for p in results["cloud"]["papers"]}
        kth = sorted(cloud_score.values(), reverse=True)[args.k - 1]
        ideal = sum(sorted(cloud_score.values(), reverse=True)[:args.k]) / args.k
        print(f"cloud k-th best score = {kth}  (papers at or above it: "
              f"{sum(v >= kth for v in cloud_score.values())})")
        ref = set(results["cloud"]["shortlist"])
        for mode in ("local", "cascade"):
            if mode in results:
                got = results[mode]["shortlist"]
                hits = sum(cloud_score.get(pid, 0) >= kth for pid in got)
                mean = sum(cloud_score.get(pid, 0) for pid in got) / args.k
                print(f"{mode:8} vs cloud: plain overlap {len(set(got) & ref)}/{args.k} | tie-aware {hits}/{args.k}"
                      f" | mean cloud score of shortlist {mean:.2f} (best possible {ideal:.2f})")
    for mode, out in results.items():
        others = sum(1 for pid in out["shortlist"] if pid not in own)
        print(f"{mode}: papers from OTHER topics in the shortlist: {others}")

    if args.no_summary or not results:
        st = cp.get("/admin/router")
        print(f"\nlatency model: {st['latency']}")
        print(f"vector cache: {st['vector_cache']}")
        cp.close()
        return 0
    print("\nsummarize the top paper:")
    first = results.get("cascade") or results.get("local") or results["cloud"]
    top = next(p for p in papers if p["id"] == first["shortlist"][0])
    for mode in ("cascade", "local"):
        r = cp.post("/admin/llm/complete", {"user_id": "router-test", "task": "summarize", "mode": mode,
                                            "input": {"title": top["title"], "abstract": top["abstract"]}})
        if r.status_code != 200:
            print(f"  {mode}: HTTP {r.status_code} {r.text[:300]}")
            continue
        out = r.json()
        print(f"  {mode}: route {out['route']} ({out['model']}), {out['latency_ms']} ms, "
              f"cost INR {out['cost_inr']}, fallback {out['fallback']}")
        print(f"    {out['text'][:240]}")

    st = cp.get("/admin/router")
    print("\nrouter totals (all calls so far):")
    print(json.dumps(st["totals"], indent=1, default=str))
    print(f"vector cache: {st['vector_cache']}")
    cp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())