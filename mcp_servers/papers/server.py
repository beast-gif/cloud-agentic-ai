"""
Agentic Cloud - Phase 3, Step 3.2
MCP server "papers": real paper search on arXiv.

Tools:
  search(query, max_results=8)  -> papers matching ALL words of the query
  get(id)                       -> one paper by arXiv id (e.g. "2401.01234")

Being a good API citizen (arXiv asks for at most ~1 request every 3 seconds):
  - upstream calls are serialized with a lock and spaced >= ARXIV_MIN_INTERVAL_S apart
  - results are cached for CACHE_TTL_S, so repeated searches never hit arXiv again
The gateway (Step 3.3) adds a platform-wide rate limit in front of this as well.

Handling arXiv saying "slow down" (HTTP 429 / 503) - added after real 429s in testing:
  1. RETRY with backoff: wait Retry-After (if arXiv sends it) or 5 s, 10 s, ... (capped),
     up to ARXIV_RETRIES times
  2. COOL-DOWN (a simple circuit breaker): if it still fails, stop calling arXiv for
     ARXIV_COOLDOWN_S seconds - every request during that window fails FAST instead of
     waiting ~25 s for another 429, and arXiv gets time to forgive us
  3. FALLBACK (if PAPERS_FALLBACK=mock): answer with deterministic placeholder papers,
     clearly marked "source": "fallback", so the agent keeps working; never cached,
     so the next search after the cool-down gets real data again

arXiv API: GET https://export.arxiv.org/api/query?search_query=...&max_results=N
Response: Atom XML, one <entry> per paper.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx
import uvicorn

from common.mcp_http import CallContext, MCPServer, ToolError

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s papers: %(message)s")
log = logging.getLogger("papers")

ARXIV_URL = os.environ.get("ARXIV_URL", "https://export.arxiv.org/api/query")
MIN_INTERVAL_S = float(os.environ.get("ARXIV_MIN_INTERVAL_S", "3.0"))
CACHE_TTL_S = float(os.environ.get("CACHE_TTL_S", "3600"))
TIMEOUT_S = float(os.environ.get("ARXIV_TIMEOUT_S", "20"))
RETRIES = int(os.environ.get("ARXIV_RETRIES", "2"))
BACKOFF_S = float(os.environ.get("ARXIV_BACKOFF_S", "5"))
MAX_WAIT_S = float(os.environ.get("ARXIV_MAX_WAIT_S", "30"))
COOLDOWN_S = float(os.environ.get("ARXIV_COOLDOWN_S", "60"))
FALLBACK = os.environ.get("PAPERS_FALLBACK", "none").lower()   # none | mock
MAX_RESULTS_LIMIT = 20

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_WORD = re.compile(r"[A-Za-z0-9\-]+")

server = MCPServer(
    "papers", "0.1.0",
    instructions="Search research papers on arXiv. Use 'search' with a short topic, "
                 "then 'get' for one paper's full abstract.",
)

_http = httpx.Client(timeout=TIMEOUT_S,
                     headers={"User-Agent": "agentic-cloud-papers/0.1 (student project)"})
_upstream_lock = threading.Lock()
_last_upstream = 0.0
_cooldown_until = 0.0
_cache: dict[tuple, tuple[float, list[dict]]] = {}
_by_id: dict[str, dict] = {}
_stats = {"upstream_calls": 0, "cache_hits": 0, "waited_ms": 0.0,
          "rate_limited": 0, "retries": 0, "cooldown_rejections": 0, "fallbacks": 0}


# ============================================================ arXiv

def _parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for e in root.findall("a:entry", NS):
        abs_url = (e.findtext("a:id", "", NS) or "").strip()        # http://arxiv.org/abs/2401.01234v2
        arxiv_id = re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1])
        if not arxiv_id:
            continue
        title = " ".join((e.findtext("a:title", "", NS) or "").split())
        abstract = " ".join((e.findtext("a:summary", "", NS) or "").split())
        published = e.findtext("a:published", "", NS) or ""
        authors = [" ".join((a.findtext("a:name", "", NS) or "").split())
                   for a in e.findall("a:author", NS)]
        cat = e.find("arxiv:primary_category", NS)
        papers.append({
            "id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors[:5],
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "venue": "arXiv",
            "category": cat.get("term") if cat is not None else None,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return papers


def _retry_after_s(resp: httpx.Response, attempt: int) -> float:
    """Seconds to wait before retry #attempt: arXiv's Retry-After if given, else exponential."""
    header = resp.headers.get("retry-after", "")
    if header.isdigit():
        return min(float(header), MAX_WAIT_S)
    return min(BACKOFF_S * (2 ** attempt), MAX_WAIT_S)   # 5 s, 10 s, 20 s, ...


def _fetch(params: dict) -> list[dict]:
    """Polite upstream request: serialized, spaced, retried on 429/503, with a cool-down."""
    global _last_upstream, _cooldown_until
    with _upstream_lock:
        remaining = _cooldown_until - time.monotonic()
        if remaining > 0:
            _stats["cooldown_rejections"] += 1
            raise ToolError(f"arXiv is rate-limiting us; cooling down for {remaining:.0f} more s")

        for attempt in range(RETRIES + 1):
            wait = _last_upstream + MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                _stats["waited_ms"] += wait * 1000.0
                time.sleep(wait)
            try:
                resp = _http.get(ARXIV_URL, params=params)
            except httpx.TimeoutException as exc:
                # arXiv often stalls ~25 s before a 429: treat a timeout as "slow down" too
                _last_upstream = time.monotonic()
                _stats["upstream_calls"] += 1
                _stats["rate_limited"] += 1
                _cooldown_until = time.monotonic() + COOLDOWN_S
                raise ToolError(f"arXiv did not answer within {TIMEOUT_S:.0f} s; "
                                f"cooling down for {COOLDOWN_S:.0f} s") from exc
            except httpx.HTTPError as exc:
                _last_upstream = time.monotonic()
                _stats["upstream_calls"] += 1
                raise ToolError(f"arXiv request failed: {exc!r}") from exc
            _last_upstream = time.monotonic()
            _stats["upstream_calls"] += 1

            if resp.status_code in (429, 503):
                _stats["rate_limited"] += 1
                if attempt < RETRIES:
                    delay = _retry_after_s(resp, attempt)
                    _stats["retries"] += 1
                    log.warning("arXiv %d, retry %d/%d in %.0f s",
                                resp.status_code, attempt + 1, RETRIES, delay)
                    time.sleep(delay)
                    continue
                _cooldown_until = time.monotonic() + COOLDOWN_S
                log.warning("arXiv still %d after %d retries -> cool-down %.0f s",
                            resp.status_code, RETRIES, COOLDOWN_S)
                raise ToolError(f"arXiv rate limit (HTTP {resp.status_code}); "
                                f"cooling down for {COOLDOWN_S:.0f} s")
            if resp.status_code >= 400:
                raise ToolError(f"arXiv returned HTTP {resp.status_code}")
            break

    try:
        return _parse_feed(resp.text)
    except ET.ParseError as exc:
        raise ToolError(f"could not parse arXiv response: {exc}") from exc


# ============================================================ fallback (PAPERS_FALLBACK=mock)

_METHODS = ["quantization", "token eviction", "sparse attention", "low-rank projection",
            "knowledge distillation", "speculative decoding", "retrieval augmentation"]
_ANGLES = ["efficient", "adaptive", "memory-aware", "training-free", "scalable"]


def _fallback_papers(query: str, n: int) -> list[dict]:
    """Deterministic placeholder papers (same query -> same papers), clearly labelled."""
    seed = int(hashlib.sha256(query.lower().encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    papers = []
    for i in range(n):
        method = rng.choice(_METHODS)
        p = {
            "id": f"fallback-{seed % 100000:05d}-{i}",
            "title": f"{rng.choice(_ANGLES).capitalize()} {query} with {method}",
            "abstract": f"Placeholder result (arXiv unavailable). A {method}-based approach to {query}.",
            "authors": [],
            "year": rng.randint(2021, 2026),
            "venue": "fallback",
            "category": None,
            "url": None,
        }
        papers.append(p)
        _by_id[p["id"]] = p
    return papers


_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it",
              "of", "on", "or", "the", "to", "via", "with", "without", "towards", "using"}


# ============================================================ tools

@server.tool(
    "search",
    "Search arXiv papers whose title/abstract contain ALL words of the query. "
    "Returns id, title, abstract, authors, year, url.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "short topic, e.g. 'KV cache compression'"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS_LIMIT, "default": 8},
        },
        "required": ["query"],
    },
)
def search(args: dict, ctx: CallContext) -> dict:
    words = _WORD.findall(str(args["query"]))
    if not words:
        raise ToolError("query must contain at least one word")
    # arXiv ignores stop words, so "all:of" matches NOTHING and the AND of all words returned
    # 0 papers for queries like "mixture of experts". Drop them (keep them only if nothing else is left).
    content = [w for w in words if w.lower() not in _STOPWORDS]
    words = content or words
    n = max(1, min(int(args.get("max_results", 8)), MAX_RESULTS_LIMIT))

    key = (" ".join(w.lower() for w in words), n)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        _stats["cache_hits"] += 1
        return {"query": args["query"], "cached": True, "source": "arxiv", "papers": hit[1]}

    try:
        papers = _fetch({
            "search_query": " AND ".join(f"all:{w}" for w in words),
            "start": 0,
            "max_results": n,
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
    except ToolError as exc:
        if FALLBACK != "mock":
            raise
        _stats["fallbacks"] += 1
        log.warning("search %r -> fallback (%s)", args["query"], exc)
        return {"query": args["query"], "cached": False, "source": "fallback",
                "note": str(exc), "papers": _fallback_papers(" ".join(words), n)}

    if papers:                      # never cache an empty answer for an hour
        _cache[key] = (time.time(), papers)
    for p in papers:
        _by_id[p["id"]] = p
    return {"query": args["query"], "cached": False, "source": "arxiv", "papers": papers}


@server.tool(
    "get",
    "Get one arXiv paper by id (e.g. '2401.01234').",
    {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
)
def get(args: dict, ctx: CallContext) -> dict:
    pid = re.sub(r"v\d+$", "", str(args["id"]).strip())
    if pid in _by_id:
        _stats["cache_hits"] += 1
        return _by_id[pid]
    papers = _fetch({"id_list": pid, "max_results": 1})
    if not papers:
        raise ToolError(f"paper {pid!r} not found")
    _by_id[pid] = papers[0]
    return papers[0]


app = server.app()


@app.get("/stats")
def stats():
    return {**_stats, "cache_entries": len(_cache), "papers_known": len(_by_id),
            "min_interval_s": MIN_INTERVAL_S, "fallback": FALLBACK,
            "cooldown_remaining_s": max(0.0, round(_cooldown_until - time.monotonic(), 1)),
            "mcp": server.stats}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9000")), log_level="info")