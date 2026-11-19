"""
Agentic Cloud - Phase 4, Step 4.1
LLM probe: measure local Ollama models against the cloud model on OUR task,
before building the router.

Dataset (built from REAL arXiv papers via the papers MCP server, direct + cached):
  for each topic, its own search results    -> mostly relevant pairs
  plus papers from the OTHER topics         -> clearly irrelevant pairs (negatives)

For every (topic, paper) pair:
  cloud (reference)  : 1 call at temperature 0          -> ref score
  each local model   : 2 calls at temperature 0.7       -> s1, s2  (self-consistency)

Reported per local model:
  speed      prompt tokens/s (reading), output tokens/s (writing), wall time per call
             (from Ollama's own counters: prompt_eval_count/_duration, eval_count/_duration)
  quality    MAE = mean |mean(s1,s2) - ref|,  parse failures,
             top-5 overlap per topic with the cloud's top-5 (5/5 = same shortlist)
  escalation disagreement rate p_fail = share of pairs with |s1 - s2| > 2
             and whether disagreement points at the real mistakes:
             error when they agree vs when they disagree, and the share of
             big errors (|error| > 3) that disagreement would have caught
  summary    time and validator result for 2 summaries
Cloud: latency, tokens (from the API's usage field), cost if prices are configured.

The Velona key is read from the VAULT (user=platform, provider=velona), never from a file.

Usage (from the agentic-cloud/ folder; Ollama running, MCP servers up):
  python -m benchmarks.llm_probe
  python -m benchmarks.llm_probe --models qwen2.5:1.5b --per-topic 6
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from benchmarks.common import CFG, PAPERS_URL, direct_mcp
from control_plane.vault import Vault

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOPICS = ["KV cache compression", "speculative decoding", "protein structure prediction"]
DISAGREE_THRESHOLD = 2.0
BIG_ERROR = 3.0
SCORE_SYSTEM = ("You are a scoring function. Output exactly one integer from 0 to 10 and nothing else: "
                "no words, no punctuation, no explanation.")
_NUM = re.compile(r"\d+(\.\d+)?")


def score_prompt(topic: str, title: str, abstract: str) -> str:
    return ("You rate how relevant a research paper is to a topic.\n"
            f"Topic: {topic}\nTitle: {title}\nAbstract: {abstract[:800]}\n\n"
            "Reply with ONLY a number from 0 to 10, where 10 means perfectly relevant.")


def summary_prompt(title: str, abstract: str) -> str:
    return ("Summarize this research paper in 2 short sentences for a student.\n"
            f"Title: {title}\nAbstract: {abstract[:1600]}\n")


def parse_score(text: str) -> Optional[float]:
    m = _NUM.search(text or "")
    if not m:
        return None
    v = float(m.group())
    return v if 0.0 <= v <= 10.0 else None


def summary_ok(text: str, title: str) -> tuple[bool, str]:
    t = (text or "").strip()
    if len(t) < 40:
        return False, "too short"
    if re.search(r"\b(i cannot|i can't|as an ai)\b", t.lower()):
        return False, "refusal"
    sentences = [s for s in re.split(r"[.!?]+\s", t) if s.strip()]
    if len(sentences) > 4:
        return False, f"{len(sentences)} sentences"
    title_words = {w for w in re.findall(r"[a-z]{5,}", title.lower())}
    if title_words and not title_words & set(re.findall(r"[a-z]{5,}", t.lower())):
        return False, "mentions no title keyword"
    return True, "ok"


# ============================================================ providers

class Ollama:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(timeout=timeout_s)

    def available(self) -> list[str]:
        return [m["name"] for m in self.http.get(f"{self.base}/api/tags").json().get("models", [])]

    def generate(self, model: str, prompt: str, temperature: float, num_predict: int) -> dict:
        t0 = time.perf_counter()
        r = self.http.post(f"{self.base}/api/generate", json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict}})
        r.raise_for_status()
        d = r.json()
        d["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        return d


    def embed(self, model: str, texts: list[str]) -> tuple[list[list[float]], float]:
        t0 = time.perf_counter()
        r = self.http.post(f"{self.base}/api/embed", json={"model": model, "input": texts})
        r.raise_for_status()
        return r.json()["embeddings"], (time.perf_counter() - t0) * 1000.0


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class Cloud:
    def __init__(self, url: str, model: str, key: str, timeout_s: float) -> None:
        self.url, self.model = url, model
        self.http = httpx.Client(timeout=timeout_s, headers={"Authorization": f"Bearer {key}"})

    def chat(self, prompt: str, max_tokens: int, system: Optional[str] = None) -> dict:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        t0 = time.perf_counter()
        r = self.http.post(self.url, json={"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                                           "messages": msgs})
        wall = (time.perf_counter() - t0) * 1000.0
        if r.status_code >= 400:
            raise RuntimeError(f"cloud HTTP {r.status_code}: {r.text[:300]}")
        d = r.json()
        usage = d.get("usage") or {}
        choice = (d.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return {"text": msg.get("content") or "", "wall_ms": wall,
                "finish_reason": choice.get("finish_reason"),
                "message_keys": sorted(k for k, v in msg.items() if v),
                "in_tokens": usage.get("prompt_tokens", 0), "out_tokens": usage.get("completion_tokens", 0)}


def cloud_cost_inr(in_tok: int, out_tok: int) -> Optional[float]:
    c = CFG["router"]["cloud"]
    p = c.get("price_usd_per_mtok") or {}
    if p.get("input") is None or p.get("output") is None or not c.get("usd_inr"):
        return None
    usd = in_tok / 1e6 * p["input"] + out_tok / 1e6 * p["output"]
    return usd * c["usd_inr"] * c.get("forex_buffer", 1.005) * c.get("platform_fee", 1.07)


# ============================================================ dataset

def _search_real(topic: str, n: int, attempts: int = 4, wait_s: float = 30.0) -> list[dict]:
    """Search via the papers server, insisting on REAL arXiv data (fallback placeholders are useless
    for measuring quality). On fallback (429 / timeout / cool-down) wait and retry."""
    for a in range(1, attempts + 1):
        _, body, _ = direct_mcp(PAPERS_URL, "search", {"query": topic, "max_results": n})
        sc = (body.get("result") or {}).get("structuredContent") or {}
        if sc.get("source") == "arxiv":
            return [p for p in sc.get("papers", []) if p.get("abstract")][:n]
        print(f"  '{topic}': source={sc.get('source')} ({sc.get('note')}) - "
              f"attempt {a}/{attempts}, waiting {wait_s:.0f}s for arXiv...")
        if a < attempts:
            time.sleep(wait_s)
    raise RuntimeError(f"arXiv unavailable for '{topic}' after {attempts} attempts - "
                       f"try again in a few minutes (results are cached once they succeed)")


def build_dataset(per_topic: int, allow_fallback: bool) -> list[dict]:
    by_topic: dict[str, list[dict]] = {}
    for t in TOPICS:
        if allow_fallback:
            _, body, _ = direct_mcp(PAPERS_URL, "search", {"query": t, "max_results": per_topic})
            sc = (body.get("result") or {}).get("structuredContent") or {}
            by_topic[t] = [p for p in sc.get("papers", []) if p.get("abstract")][:per_topic]
        else:
            by_topic[t] = _search_real(t, per_topic)
        print(f"  {t}: {len(by_topic[t])} papers")
    pairs = []
    for t in TOPICS:
        for p in by_topic[t]:
            pairs.append({"topic": t, "paper": p, "kind": "same-topic"})
        others = [p for o in TOPICS if o != t for p in by_topic[o][:2]]   # 2 from each other topic
        for p in others:
            pairs.append({"topic": t, "paper": p, "kind": "other-topic"})
    return pairs


def top5_overlap(pairs: list[dict], scores: list[Optional[float]], ref: list[Optional[float]]) -> float:
    """Average over topics of |top5(model) & top5(cloud)| / 5."""
    vals = []
    for t in TOPICS:
        idx = [i for i, p in enumerate(pairs) if p["topic"] == t and scores[i] is not None and ref[i] is not None]
        if len(idx) < 5:
            continue
        top_m = set(sorted(idx, key=lambda i: -scores[i])[:5])
        top_r = set(sorted(idx, key=lambda i: -ref[i])[:5])
        vals.append(len(top_m & top_r) / 5.0)
    return round(statistics.fmean(vals), 3) if vals else 0.0


def ranking_auc(pairs: list[dict], scores: list[Optional[float]]) -> Optional[float]:
    """Per topic: probability that a same-topic paper scores ABOVE an other-topic paper (ties = 0.5).
    1.0 = perfect separation, 0.5 = coin flip. Needs no reference model and works for any score scale."""
    vals = []
    for t in TOPICS:
        pos = [s for p, s in zip(pairs, scores) if p["topic"] == t and p["kind"] == "same-topic" and s is not None]
        neg = [s for p, s in zip(pairs, scores) if p["topic"] == t and p["kind"] == "other-topic" and s is not None]
        if not pos or not neg:
            continue
        wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
        vals.append(wins / (len(pos) * len(neg)))
    return round(statistics.fmean(vals), 3) if vals else None


def top5_random_baseline(pairs: list[dict]) -> float:
    """Expected top-5 overlap of a RANDOM ranking: E|A&B|/5 = 5/N for N candidates per topic."""
    vals = [min(1.0, 5.0 / sum(p["topic"] == t for p in pairs)) for t in TOPICS]
    return round(statistics.fmean(vals), 3)


def by_kind(pairs: list[dict], scores: list[Optional[float]]) -> dict:
    """Mean score on same-topic vs other-topic papers: does the model SEPARATE relevant from irrelevant?"""
    out = {}
    for kind in ("same-topic", "other-topic"):
        v = [s for p, s in zip(pairs, scores) if p["kind"] == kind and s is not None]
        out[kind] = round(statistics.fmean(v), 2) if v else None
    if out["same-topic"] is not None and out["other-topic"] is not None:
        out["separation"] = round(out["same-topic"] - out["other-topic"], 2)
    return out


def tok_rate(count: int, dur_ns: int) -> Optional[float]:
    return round(count / (dur_ns / 1e9), 1) if count and dur_ns else None


# ============================================================ main

def main() -> int:
    ap = argparse.ArgumentParser(description="Measure local vs cloud LLMs on the scoring task")
    ap.add_argument("--models", nargs="*", default=CFG["router"]["local"]["models"],
                    help="local LLMs to test; pass --models with nothing after it to skip LLMs")
    ap.add_argument("--embed", default=None, metavar="MODEL",
                    help="also test an EMBEDDING ranker, e.g. --embed nomic-embed-text")
    ap.add_argument("--per-topic", type=int, default=6, help="papers per topic (default 6)")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="run even if arXiv is down (placeholder papers - speed only, quality is meaningless)")
    args = ap.parse_args()

    rc = CFG["router"]
    ollama = Ollama(rc["local"]["base_url"], rc["local"]["timeout_s"])
    try:
        have = ollama.available()
    except httpx.HTTPError as exc:
        print(f"Ollama not reachable at {rc['local']['base_url']}: {exc!r}")
        return 1
    def pulled(m: str) -> bool:
        return m in have or f"{m}:latest" in have
    missing = [m for m in args.models + ([args.embed] if args.embed else []) if not pulled(m)]
    if missing:
        print(f"not pulled in Ollama: {missing}  (have: {have})\n  ollama pull {' / '.join(missing)}")
        return 1

    pool = ConnectionPool(CFG["postgres"]["host_dsn"], kwargs={"autocommit": True, "row_factory": dict_row},
                          open=True)
    vault = Vault.from_env(pool, (CFG.get("vault") or {}).get("master_key_env", "VAULT_MASTER_KEY"))
    try:
        key = vault.get(rc["cloud"]["vault_user"], rc["cloud"]["vault_provider"], actor="llm-probe")
    except Exception as exc:  # noqa: BLE001
        print(f"cloud key not in the vault: {exc}\n  PUT /admin/vault/{rc['cloud']['vault_user']}/"
              f"{rc['cloud']['vault_provider']}  {{\"secret\": \"<your Velona key>\"}}")
        return 1
    finally:
        pool.close()          # only needed to read the key; avoids "couldn't stop thread" warnings
    cloud = Cloud(rc["cloud"]["url"], rc["cloud"]["model"], key, rc["cloud"]["timeout_s"])
    del key

    print("building dataset from arXiv (papers MCP server, cached)...")
    try:
        pairs = build_dataset(args.per_topic, args.allow_fallback)
    except RuntimeError as exc:
        print(f"stopped: {exc}")
        return 1
    n = len(pairs)
    print(f"{n} (topic, paper) pairs\n")

    # ---- cloud reference
    print(f"cloud reference: {rc['cloud']['model']}")
    ref: list[Optional[float]] = []
    c_wall, c_in, c_out = [], 0, 0
    cloud_fail_samples: list[dict] = []
    for p in pairs:
        out = cloud.chat(score_prompt(p["topic"], p["paper"]["title"], p["paper"]["abstract"]), 64,
                         system=SCORE_SYSTEM)
        s_ref = parse_score(out["text"])
        ref.append(s_ref)
        if s_ref is None and len(cloud_fail_samples) < 5:
            cloud_fail_samples.append({"text": out["text"][:120], "finish_reason": out["finish_reason"],
                                       "message_keys": out["message_keys"], "out_tokens": out["out_tokens"]})
        c_wall.append(out["wall_ms"]); c_in += out["in_tokens"]; c_out += out["out_tokens"]
    cloud_sum = []
    for p in [x for x in pairs if x["kind"] == "same-topic"][:2]:
        out = cloud.chat(summary_prompt(p["paper"]["title"], p["paper"]["abstract"]), 160)
        ok, why = summary_ok(out["text"], p["paper"]["title"])
        cloud_sum.append({"wall_ms": round(out["wall_ms"]), "ok": ok, "why": why, "text": out["text"][:300]})
        c_in += out["in_tokens"]; c_out += out["out_tokens"]
    cost = cloud_cost_inr(c_in, c_out)
    report = {"pairs": n, "topics": TOPICS, "top5_overlap_random_baseline": top5_random_baseline(pairs),
              "cloud": {
        "model": rc["cloud"]["model"],
        "score_wall_ms_p50": round(statistics.median(c_wall), 1),
        "parse_failures": sum(r is None for r in ref),
        "parse_failure_samples": cloud_fail_samples,
        "mean_score_by_kind": by_kind(pairs, ref),
        "ranking_auc": ranking_auc(pairs, ref),
        "tokens_in": c_in, "tokens_out": c_out,
        "cost_inr_total": round(cost, 4) if cost is not None else None,
        "summaries": cloud_sum}, "local": {}}
    print(f"  score call p50 {report['cloud']['score_wall_ms_p50']} ms, tokens in/out {c_in}/{c_out}, "
          f"cost INR {report['cloud']['cost_inr_total']}\n")

    # ---- local models
    for model in args.models:
        print(f"local: {model}  (warming up / loading into RAM...)")
        ollama.generate(model, "Reply with OK.", 0.0, 3)
        s1s, s2s, walls, p_rates, e_rates = [], [], [], [], []
        for i, p in enumerate(pairs):
            prompt = score_prompt(p["topic"], p["paper"]["title"], p["paper"]["abstract"])
            outs = [ollama.generate(model, prompt, 0.7, 6) for _ in range(2)]
            s1s.append(parse_score(outs[0].get("response", "")))
            s2s.append(parse_score(outs[1].get("response", "")))
            for o in outs:
                walls.append(o["wall_ms"])
                p_rates.append(tok_rate(o.get("prompt_eval_count", 0), o.get("prompt_eval_duration", 0)))
                e_rates.append(tok_rate(o.get("eval_count", 0), o.get("eval_duration", 0)))
            if (i + 1) % 6 == 0:
                print(f"  {i + 1}/{n} pairs")

        mean_s = [(a + b) / 2 if a is not None and b is not None else (a if a is not None else b)
                  for a, b in zip(s1s, s2s)]
        errs = [abs(m - r) for m, r in zip(mean_s, ref) if m is not None and r is not None]
        disagree = [a is None or b is None or abs(a - b) > DISAGREE_THRESHOLD for a, b in zip(s1s, s2s)]
        err_agree = [abs(m - r) for m, r, d in zip(mean_s, ref, disagree) if not d and m is not None and r is not None]
        err_dis = [abs(m - r) for m, r, d in zip(mean_s, ref, disagree) if d and m is not None and r is not None]
        big = [d for m, r, d in zip(mean_s, ref, disagree) if m is not None and r is not None and abs(m - r) > BIG_ERROR]

        sums = []
        for p in [x for x in pairs if x["kind"] == "same-topic"][:2]:
            o = ollama.generate(model, summary_prompt(p["paper"]["title"], p["paper"]["abstract"]), 0.2, 120)
            ok, why = summary_ok(o.get("response", ""), p["paper"]["title"])
            sums.append({"wall_ms": round(o["wall_ms"]), "ok": ok, "why": why,
                         "out_tok_per_s": tok_rate(o.get("eval_count", 0), o.get("eval_duration", 0)),
                         "text": o.get("response", "")[:300]})

        r = {
            "score_wall_ms_p50": round(statistics.median(walls), 1),
            "prompt_tok_per_s_p50": statistics.median([x for x in p_rates if x]) if any(p_rates) else None,
            "output_tok_per_s_p50": statistics.median([x for x in e_rates if x]) if any(e_rates) else None,
            "parse_failures": sum(x is None for x in s1s + s2s),
            "mae_vs_cloud": round(statistics.fmean(errs), 2) if errs else None,
            "top5_overlap_vs_cloud": top5_overlap(pairs, mean_s, ref),
            "mean_score_by_kind": by_kind(pairs, mean_s),
            "ranking_auc": ranking_auc(pairs, mean_s),
            "bias_vs_cloud": round(statistics.fmean([m - r for m, r in zip(mean_s, ref)
                                                     if m is not None and r is not None]), 2) if errs else None,
            "p_fail_disagree_rate": round(sum(disagree) / n, 3),
            "mae_when_agree": round(statistics.fmean(err_agree), 2) if err_agree else None,
            "mae_when_disagree": round(statistics.fmean(err_dis), 2) if err_dis else None,
            "big_errors": len(big),
            "big_errors_caught_by_disagreement": sum(big),
            "summaries": sums,
        }
        report["local"][model] = r
        print(f"  speed: {r['score_wall_ms_p50']} ms/call, read {r['prompt_tok_per_s_p50']} tok/s, "
              f"write {r['output_tok_per_s_p50']} tok/s")
        print(f"  quality: MAE {r['mae_vs_cloud']}, top-5 overlap {r['top5_overlap_vs_cloud']}, "
              f"parse failures {r['parse_failures']}")
        print(f"  escalation: p_fail {r['p_fail_disagree_rate']}, MAE agree {r['mae_when_agree']} vs "
              f"disagree {r['mae_when_disagree']}, big errors caught {r['big_errors_caught_by_disagreement']}"
              f"/{r['big_errors']}")
        print(f"  summaries: {[(s['wall_ms'], s['ok'], s['why']) for s in sums]}\n")

    # ---- embedding ranker: cosine(topic, paper) instead of asking an LLM
    if args.embed:
        print(f"embedding ranker: {args.embed}")
        q_pre, d_pre = ("search_query: ", "search_document: ") if "nomic" in args.embed else ("", "")
        ollama.embed(args.embed, ["warm up"])
        q_vec = {}
        for t in TOPICS:
            v, _ = ollama.embed(args.embed, [q_pre + t])
            q_vec[t] = v[0]
        uniq = {p["paper"]["id"]: p["paper"] for p in pairs}
        doc_vec, per_doc_ms = {}, []
        for pid, paper in uniq.items():                      # one call per paper -> per-paper latency
            v, ms = ollama.embed(args.embed, [d_pre + paper["title"] + ". " + paper["abstract"][:1600]])
            doc_vec[pid] = v[0]
            per_doc_ms.append(ms)
        sims = [cosine(q_vec[p["topic"]], doc_vec[p["paper"]["id"]]) for p in pairs]
        e = {
            "ms_per_paper_p50": round(statistics.median(per_doc_ms), 1),
            "ranking_auc": ranking_auc(pairs, sims),
            "top5_overlap_vs_cloud": top5_overlap(pairs, sims, ref),
            "mean_cosine_by_kind": by_kind(pairs, [round(x, 3) for x in sims]),
        }
        report["embedding"] = {args.embed: e}
        print(f"  {e['ms_per_paper_p50']} ms/paper, AUC {e['ranking_auc']}, "
              f"top-5 overlap {e['top5_overlap_vs_cloud']}, cosine {e['mean_cosine_by_kind']}\n")

    print(f"ranking AUC (1.0 perfect, 0.5 coin flip): cloud {report['cloud']['ranking_auc']}, "
          + ", ".join(f"{m} {r['ranking_auc']}" for m, r in report["local"].items())
          + (f", {args.embed} {report['embedding'][args.embed]['ranking_auc']}" if args.embed else ""))
    print(f"top-5 overlap of a RANDOM ranking: {report['top5_overlap_random_baseline']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"llm_probe_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())