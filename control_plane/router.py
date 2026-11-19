"""
Agentic Cloud - Phase 4, Step 4.2
Model Router: decides, per call, whether a LOCAL model (Ollama on this machine, free)
or the CLOUD model (Velona, paid) does the work - and logs every decision.

  agent --(session token)--> /llm/rank, /llm/complete --> ROUTER --> Ollama (localhost)
                                                              +----> Velona (key from the vault)

Why this design (measured in Step 4.1, 24 real arXiv (topic, paper) pairs):
  ranker              AUC    separation         time/paper   cost
  cloud Haiku 5.5     1.00   8.92 vs 1.09       1.34 s       ~Rs 0.047
  nomic-embed-text    1.00   cos 0.76 vs 0.52   0.94 s       Rs 0
  qwen2.5:1.5b (LLM)  0.55   7.88 vs 7.75       2.0 s        Rs 0      -> NOT used for scoring
  summaries: cloud 1.6 s, qwen 9-13 s (acceptable text) -> qwen is the summary FALLBACK

------------------------------------------------------------------ task "score" (/llm/rank)
1. Embed the topic and every paper locally (one batched call; vectors cached per text):
       cos = (q . d) / (|q| |d|)
2. Map cosine to the 0-10 scale with a straight line through the two class means
   measured in Step 4.1  (cos 0.52 -> 1.09, cos 0.76 -> 8.92):
       score = s_lo + (cos - c_lo) * (s_hi - s_lo) / (c_hi - c_lo)      clamped to [0, 10]
       e.g. cos 0.78 -> 9.6   cos 0.64 -> 5.0   cos 0.50 -> 0.4
3. Escalate to the cloud ONLY where a mistake could change the shortlist (mode cascade):
     a) uncertain band   band_low <= cos <= band_high       (e.g. 0.60 .. 0.68)
     b) shortlist cut    score(rank k) - score(rank k+1) < cut_gap   -> both papers
   capped at max_cloud_calls, run in parallel. The cloud score replaces the local one.
   Typical run: 15 candidates -> 0-3 cloud calls instead of 15.

------------------------------------------------------------------ task "summarize" (/llm/complete)
   cloud first (fast); on error / timeout / empty / validator failure / budget -> qwen locally.
   Validators: >= 40 chars, no refusal, <= 4 sentences, mentions a title keyword.

------------------------------------------------------------------ modes (per request)
   local    : embeddings only for ranking, qwen for summaries        (Rs 0, never leaves the machine)
   cloud    : cloud scores every paper, cloud summaries              (the all-cloud baseline)
   cascade  : the design above (default)

------------------------------------------------------------------ cloud details
   - The Velona key is decrypted from the vault for EACH cloud call (audited) and never stored.
   - Haiku 5.5 sometimes spends hidden reasoning tokens first: an EMPTY reply with
     finish_reason "length" is retried once with max_tokens x retry_multiplier.
   - cost_inr = (tokens_in/1e6 x price_in + tokens_out/1e6 x price_out) x usd_inr x 1.005 x 1.07
   - Per-session budget: when SUM(cost_inr) of the session reaches budget_inr_per_session,
     every further call goes local (logged as outcome "skipped").
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import httpx
from psycopg_pool import ConnectionPool

from control_plane.vault import Vault, VaultError

log = logging.getLogger("control_plane.router")

MODES = ("local", "cloud", "cascade")
COMPLETE_TASKS = ("summarize", "generic")

SCORE_SYSTEM = ("You are a scoring function. Output exactly one integer from 0 to 10 and nothing else: "
                "no words, no punctuation, no explanation.")
_NUM = re.compile(r"\d+(\.\d+)?")
_PREAMBLE = re.compile(r"^\s*(here('s| is)[^\n:]*:)\s*", re.IGNORECASE)


# ============================================================ errors

class RouterError(Exception):
    status_code = 400


class RouterUnavailable(RouterError):
    status_code = 503


# ============================================================ prompts / parsing / validators
# Same prompts as benchmarks/llm_probe.py, so the Step 4.1 measurements apply.

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


def clean_summary(text: str) -> str:
    """Drop chatty preambles like "Here's a 2-sentence summary ...:" (llama/qwen habit)."""
    return _PREAMBLE.sub("", (text or "").strip()).strip()


def summary_ok(text: str, title: str) -> tuple[bool, str]:
    t = (text or "").strip()
    if len(t) < 40:
        return False, "too short"
    if re.search(r"\b(i cannot|i can't|as an ai)\b", t.lower()):
        return False, "refusal"
    sentences = [s for s in re.split(r"[.!?]+\s", t) if s.strip()]
    if len(sentences) > 4:
        return False, f"{len(sentences)} sentences"
    title_words = set(re.findall(r"[a-z]{5,}", title.lower()))
    if title_words and not title_words & set(re.findall(r"[a-z]{5,}", t.lower())):
        return False, "mentions no title keyword"
    return True, "ok"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ============================================================ vector cache

class VectorCache:
    """LRU cache: sha256(model | prefix | text) -> embedding. Each paper is embedded once."""

    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self._d: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()

    def get(self, k: str) -> Optional[list[float]]:
        with self._lock:
            v = self._d.get(k)
            if v is None:
                self.misses += 1
                return None
            self._d.move_to_end(k)
            self.hits += 1
            return v

    def put(self, k: str, v: list[float]) -> None:
        with self._lock:
            self._d[k] = v
            self._d.move_to_end(k)
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {"items": len(self._d), "max_items": self.max_items, "hits": self.hits, "misses": self.misses}


# ============================================================ router

class Router:
    def __init__(self, cfg: dict, pool: ConnectionPool, vault: Vault) -> None:
        rc = cfg.get("router") or {}
        self.pool = pool
        self.vault = vault

        self.default_mode = str(rc.get("mode", "cascade"))
        if self.default_mode not in MODES:
            raise ValueError(f"router.mode must be one of {MODES}")
        b = rc.get("budget_inr_per_session")
        self.budget_inr = float(b) if b is not None else None

        lo = rc.get("local") or {}
        self.local_url = str(lo.get("base_url", "http://localhost:11434")).rstrip("/")
        self.embed_model = lo.get("embed_model", "nomic-embed-text")
        self.chat_model = lo.get("chat_model", "qwen2.5:1.5b")
        self.keep_alive = lo.get("keep_alive", "30m")
        self.http_local = httpx.Client(timeout=float(lo.get("timeout_s", 180)))

        rk = rc.get("rank") or {}
        cal = rk.get("calibration") or {}
        self.c_lo, self.s_lo = float(cal.get("cos_low", 0.52)), float(cal.get("score_low", 1.09))
        self.c_hi, self.s_hi = float(cal.get("cos_high", 0.76)), float(cal.get("score_high", 8.92))
        band = rk.get("uncertain_band") or [0.60, 0.68]
        self.band_lo, self.band_hi = float(band[0]), float(band[1])
        self.cut_gap = float(rk.get("cut_gap", 1.5))
        self.max_cloud_calls = int(rk.get("max_cloud_calls", 6))
        self.embed_chars = int(rk.get("embed_chars", 1600))
        self.cache = VectorCache(int(rk.get("cache_items", 5000)))
        self.executor = ThreadPoolExecutor(max_workers=int(rk.get("cloud_parallel", 4)),
                                           thread_name_prefix="router-cloud")

        cl = rc.get("cloud") or {}
        self.cloud_url = cl.get("url")
        self.cloud_model = cl.get("model")
        self.vault_user = cl.get("vault_user", "platform")
        self.vault_provider = cl.get("vault_provider", "velona")
        self.max_tokens = {"score": int(cl.get("score_max_tokens", 64)),
                           "summarize": int(cl.get("summary_max_tokens", 300)),
                           "generic": int(cl.get("generic_max_tokens", 400))}
        self.retry_multiplier = int(cl.get("retry_multiplier", 4))
        self.price = cl.get("price_usd_per_mtok") or {}
        self.usd_inr = cl.get("usd_inr")
        self.forex_buffer = float(cl.get("forex_buffer", 1.005))
        self.platform_fee = float(cl.get("platform_fee", 1.07))
        self.http_cloud = httpx.Client(timeout=httpx.Timeout(float(cl.get("timeout_s", 60)), connect=10.0))

        self._stats_lock = threading.Lock()
        self.stats = {"rank_requests": 0, "complete_requests": 0, "cloud_calls": 0, "local_chat_calls": 0,
                      "embed_calls": 0, "escalations": 0, "fallbacks": 0, "failures": 0, "budget_skips": 0}

    # ------------------------------------------------------------------ small helpers

    def _count(self, key: str, n: int = 1) -> None:
        with self._stats_lock:
            self.stats[key] += n

    @property
    def prices_configured(self) -> bool:
        return (self.price.get("input") is not None and self.price.get("output") is not None
                and bool(self.usd_inr))

    def cost_inr(self, tokens_in: int, tokens_out: int) -> Optional[float]:
        if not self.prices_configured:
            return None
        usd = tokens_in / 1e6 * float(self.price["input"]) + tokens_out / 1e6 * float(self.price["output"])
        return round(usd * float(self.usd_inr) * self.forex_buffer * self.platform_fee, 6)

    def spent_inr(self, session_id: Optional[str]) -> float:
        if not session_id:
            return 0.0
        with self.pool.connection() as conn:
            row = conn.execute("SELECT COALESCE(SUM(cost_inr), 0) AS s FROM llm_calls WHERE session_id = %s",
                               (session_id,)).fetchone()
        return float(row["s"])

    def _cloud_allowed(self, session_id: Optional[str]) -> tuple[bool, str]:
        """Budget check before a cloud call. No budget / no prices / no session -> always allowed."""
        if not self.cloud_url or not self.cloud_model:
            return False, "cloud not configured"
        if self.budget_inr is None or not self.prices_configured or not session_id:
            return True, ""
        spent = self.spent_inr(session_id)
        if spent >= self.budget_inr:
            return False, f"budget exhausted (spent Rs {spent:.4f} of Rs {self.budget_inr})"
        return True, ""

    def resolve_mode(self, mode: Optional[str]) -> str:
        m = (mode or self.default_mode).lower()
        if m not in MODES:
            raise RouterError(f"mode must be one of {MODES}, got {mode!r}")
        return m

    def calibrate(self, cos: float) -> float:
        s = self.s_lo + (cos - self.c_lo) * (self.s_hi - self.s_lo) / (self.c_hi - self.c_lo)
        return round(max(0.0, min(10.0, s)), 2)

    # ------------------------------------------------------------------ audit

    def _log(self, *, user_id: str, session_id: Optional[str], task: str, mode: str, route: str,
             kind: str, model: str, outcome: str, reason: Optional[str] = None, items: int = 1,
             cache_hits: int = 0, tokens_in: int = 0, tokens_out: int = 0,
             cost_inr: Optional[float] = None, latency_ms: Optional[float] = None) -> None:
        try:
            with self.pool.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO llm_calls (session_id, user_id, task, mode, route, kind, model, outcome,
                                           reason, items, cache_hits, tokens_in, tokens_out, cost_inr, latency_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (session_id, user_id, task, mode, route, kind, model, outcome, (reason or None) and reason[:300],
                     items, cache_hits, tokens_in, tokens_out, cost_inr,
                     round(latency_ms, 1) if latency_ms is not None else None),
                )
        except Exception:  # noqa: BLE001 - the audit must never break a model call
            log.exception("could not write llm_calls row")

    # ------------------------------------------------------------------ providers

    def _embed(self, texts: list[str], prefix: str) -> tuple[list[list[float]], float, int]:
        """Embed texts with the local model, reusing cached vectors. Returns (vectors, ms, cache_hits)."""
        t0 = time.perf_counter()
        keys = [VectorCache.key(self.embed_model, prefix + t) for t in texts]
        out: list[Optional[list[float]]] = [self.cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            r = self.http_local.post(f"{self.local_url}/api/embed", json={
                "model": self.embed_model, "keep_alive": self.keep_alive,
                "input": [prefix + texts[i] for i in missing]})
            r.raise_for_status()
            vecs = r.json().get("embeddings") or []
            if len(vecs) != len(missing):
                raise RuntimeError(f"Ollama returned {len(vecs)} embeddings for {len(missing)} texts")
            for i, v in zip(missing, vecs):
                out[i] = v
                self.cache.put(keys[i], v)
            self._count("embed_calls")
        return out, (time.perf_counter() - t0) * 1000.0, len(texts) - len(missing)  # type: ignore[return-value]

    def _ollama_chat(self, prompt: str, num_predict: int, temperature: float = 0.2) -> dict:
        t0 = time.perf_counter()
        r = self.http_local.post(f"{self.local_url}/api/generate", json={
            "model": self.chat_model, "prompt": prompt, "stream": False, "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_predict": num_predict}})
        r.raise_for_status()
        d = r.json()
        self._count("local_chat_calls")
        return {"text": d.get("response", "") or "", "ms": (time.perf_counter() - t0) * 1000.0,
                "tokens_in": int(d.get("prompt_eval_count", 0) or 0), "tokens_out": int(d.get("eval_count", 0) or 0)}

    def _cloud_once(self, messages: list[dict], max_tokens: int, session_id: Optional[str]) -> dict:
        key = self.vault.get(self.vault_user, self.vault_provider, session_id=session_id, actor="router")
        t0 = time.perf_counter()
        try:
            r = self.http_cloud.post(self.cloud_url, headers={"Authorization": f"Bearer {key}"}, json={
                "model": self.cloud_model, "temperature": 0, "max_tokens": max_tokens, "messages": messages})
        finally:
            del key
        ms = (time.perf_counter() - t0) * 1000.0
        self._count("cloud_calls")
        if r.status_code >= 400:
            raise RouterUnavailable(f"cloud HTTP {r.status_code}: {r.text[:200]}")
        d = r.json()
        choice = (d.get("choices") or [{}])[0]
        usage = d.get("usage") or {}
        return {"text": (choice.get("message") or {}).get("content") or "", "ms": ms,
                "finish_reason": choice.get("finish_reason"),
                "tokens_in": int(usage.get("prompt_tokens", 0) or 0),
                "tokens_out": int(usage.get("completion_tokens", 0) or 0)}

    def _cloud_chat(self, messages: list[dict], max_tokens: int, session_id: Optional[str]) -> dict:
        """One cloud call; an EMPTY reply cut off by the token limit (hidden reasoning) is retried
        once with a bigger budget. Tokens/time of both attempts are summed (both are billed)."""
        first = self._cloud_once(messages, max_tokens, session_id)
        if first["text"].strip() or first["finish_reason"] != "length":
            return {**first, "retried": False}
        second = self._cloud_once(messages, max_tokens * self.retry_multiplier, session_id)
        return {**second, "retried": True, "ms": first["ms"] + second["ms"],
                "tokens_in": first["tokens_in"] + second["tokens_in"],
                "tokens_out": first["tokens_out"] + second["tokens_out"]}

    # ================================================================== task: score / rank

    def rank(self, user_id: str, session_id: Optional[str], topic: str, papers: list[dict],
             k: int = 5, mode: Optional[str] = None) -> dict:
        mode = self.resolve_mode(mode)
        if not topic.strip():
            raise RouterError("topic is empty")
        if not papers:
            raise RouterError("papers is empty")
        if len(papers) > 100:
            raise RouterError("at most 100 papers per request")
        self._count("rank_requests")
        t_start = time.perf_counter()
        n = len(papers)
        items = [{"id": str(p.get("id") or i), "title": str(p.get("title") or ""),
                  "abstract": str(p.get("abstract") or ""), "cosine": None, "local_score": None,
                  "cloud_score": None, "score": None, "route": None, "reason": None} for i, p in enumerate(papers)]
        summary = {"embedded": 0, "cache_hits": 0, "embed_ms": None, "cloud_calls": 0, "escalated": 0,
                   "skipped": 0, "cloud_failed": 0, "cost_inr": 0.0 if self.prices_configured else None}
        slock = threading.Lock()          # cloud_score() runs in parallel threads

        # ---- 1) local embeddings (modes local + cascade)
        def embed_all(idx: list[int]) -> None:
            q_vecs, q_ms, q_hits = self._embed([topic], "search_query: ")
            d_vecs, d_ms, d_hits = self._embed(
                [items[i]["title"] + ". " + items[i]["abstract"][:self.embed_chars] for i in idx], "search_document: ")
            for i, v in zip(idx, d_vecs):
                cos = _cosine(q_vecs[0], v)
                items[i]["cosine"] = round(cos, 4)
                items[i]["local_score"] = self.calibrate(cos)
            summary["embedded"] += len(idx)
            summary["cache_hits"] += d_hits + q_hits
            summary["embed_ms"] = round((summary["embed_ms"] or 0) + q_ms + d_ms, 1)
            self._log(user_id=user_id, session_id=session_id, task="score", mode=mode, route="local",
                      kind="embed", model=self.embed_model, outcome="ok" if mode != "cloud" else "fallback",
                      reason=None if mode != "cloud" else "cloud failed for these papers",
                      items=len(idx) + 1, cache_hits=d_hits + q_hits, latency_ms=q_ms + d_ms)

        if mode in ("local", "cascade"):
            try:
                embed_all(list(range(n)))
            except (httpx.HTTPError, RuntimeError) as exc:
                self._count("failures")
                self._log(user_id=user_id, session_id=session_id, task="score", mode=mode, route="local",
                          kind="embed", model=self.embed_model, outcome="failed", reason=repr(exc)[:300], items=n)
                if mode == "local":
                    raise RouterUnavailable(f"local embedding model unavailable: {exc!r}") from exc
                mode_effective = "cloud"          # cascade without embeddings = ask the cloud for everything
            else:
                mode_effective = mode
        else:
            mode_effective = "cloud"

        # ---- 2) which papers go to the cloud?
        wanted: list[tuple[int, str]] = []
        if mode_effective == "cloud":
            wanted = [(i, "mode=cloud" if mode == "cloud" else "embeddings unavailable") for i in range(n)]
        elif mode_effective == "cascade":
            mid = (self.band_lo + self.band_hi) / 2
            band = sorted((i for i in range(n) if self.band_lo <= items[i]["cosine"] <= self.band_hi),
                          key=lambda i: abs(items[i]["cosine"] - mid))
            wanted = [(i, f"uncertain band cos={items[i]['cosine']:.3f}") for i in band]
            order = sorted(range(n), key=lambda i: -items[i]["local_score"])
            if n > k:
                a, b = order[k - 1], order[k]
                gap = items[a]["local_score"] - items[b]["local_score"]
                if gap < self.cut_gap:
                    for i in (a, b):
                        if all(i != j for j, _ in wanted):
                            wanted.append((i, f"shortlist cut: rank {k} vs {k + 1} gap={gap:.2f}"))

        # caps: per-request limit (cascade only) and the session budget
        if mode_effective == "cascade" and len(wanted) > self.max_cloud_calls:
            for i, why in wanted[self.max_cloud_calls:]:
                items[i]["reason"] = f"{why}; not escalated (max_cloud_calls={self.max_cloud_calls})"
                summary["skipped"] += 1
                self._log(user_id=user_id, session_id=session_id, task="score", mode=mode, route="cloud",
                          kind="chat", model=self.cloud_model or "-", outcome="skipped", reason=items[i]["reason"])
            wanted = wanted[:self.max_cloud_calls]
        if wanted:
            ok, why_not = self._cloud_allowed(session_id)
            if not ok:
                self._count("budget_skips", len(wanted))
                for i, why in wanted:
                    items[i]["reason"] = f"{why}; not escalated ({why_not})"
                    summary["skipped"] += 1
                self._log(user_id=user_id, session_id=session_id, task="score", mode=mode, route="cloud",
                          kind="chat", model=self.cloud_model or "-", outcome="skipped", reason=why_not,
                          items=len(wanted))
                wanted = []

        # ---- 3) cloud scores, in parallel
        def cloud_score(i: int, why: str) -> None:
            it = items[i]
            outcome = "escalated" if mode_effective == "cascade" else "ok"
            try:
                res = self._cloud_chat([{"role": "system", "content": SCORE_SYSTEM},
                                        {"role": "user", "content": score_prompt(topic, it["title"], it["abstract"])}],
                                       self.max_tokens["score"], session_id)
            except (httpx.HTTPError, RouterUnavailable, VaultError) as exc:
                self._count("failures")
                with slock:
                    summary["cloud_failed"] += 1
                it["reason"] = f"{why}; cloud failed: {str(exc)[:120]}"
                self._log(user_id=user_id, session_id=session_id, task="score", mode=mode, route="cloud",
                          kind="chat", model=self.cloud_model, outcome="failed", reason=it["reason"])
                return
            cost = self.cost_inr(res["tokens_in"], res["tokens_out"])
            s = parse_score(res["text"])
            note = why + ("; retried (empty reply)" if res["retried"] else "")
            if s is None:
                self._count("failures")
                with slock:
                    summary["cloud_failed"] += 1
                note += f"; unparsable reply {res['text'][:40]!r} finish={res['finish_reason']}"
                outcome = "failed"
            else:
                it["cloud_score"] = s
                if mode_effective == "cascade":
                    self._count("escalations")
            it["reason"] = note
            with slock:
                summary["cloud_calls"] += 1
                if cost is not None:
                    summary["cost_inr"] = round((summary["cost_inr"] or 0.0) + cost, 6)
            self._log(user_id=user_id, session_id=session_id, task="score", mode=mode, route="cloud",
                      kind="chat", model=self.cloud_model, outcome=outcome, reason=note,
                      tokens_in=res["tokens_in"], tokens_out=res["tokens_out"], cost_inr=cost, latency_ms=res["ms"])

        list(self.executor.map(lambda w: cloud_score(*w), wanted))
        summary["escalated"] = sum(1 for i, _ in wanted if items[i]["cloud_score"] is not None) \
            if mode_effective == "cascade" else 0

        # ---- 4) papers the cloud could not score in cloud mode -> local embeddings as fallback
        if mode_effective == "cloud":
            missing = [i for i in range(n) if items[i]["cloud_score"] is None and items[i]["local_score"] is None]
            if missing:
                self._count("fallbacks", len(missing))
                try:
                    embed_all(missing)
                except (httpx.HTTPError, RuntimeError) as exc:
                    raise RouterUnavailable(f"cloud and local both failed: {exc!r}") from exc

        # ---- 5) final score = cloud if we have it, else local
        for it in items:
            if it["cloud_score"] is not None:
                it["score"], it["route"] = it["cloud_score"], "cloud"
            else:
                it["score"], it["route"] = it["local_score"], "local"
        ranked = sorted(items, key=lambda it: -(it["score"] if it["score"] is not None else -1))
        for r, it in enumerate(ranked, 1):
            it["rank"] = r
            it.pop("abstract", None)

        summary["latency_ms"] = round((time.perf_counter() - t_start) * 1000.0, 1)
        return {"mode": mode, "effective_mode": mode_effective, "k": k, "topic": topic,
                "shortlist": [it["id"] for it in ranked[:k]], "papers": ranked, "summary": summary}

    # ================================================================== task: summarize / generic

    def complete(self, user_id: str, session_id: Optional[str], task: str, inp: dict,
                 mode: Optional[str] = None) -> dict:
        mode = self.resolve_mode(mode)
        if task not in COMPLETE_TASKS:
            raise RouterError(f"task must be one of {COMPLETE_TASKS}")
        self._count("complete_requests")
        t_start = time.perf_counter()

        if task == "summarize":
            title, abstract = str(inp.get("title") or ""), str(inp.get("abstract") or "")
            if not title and not abstract:
                raise RouterError("summarize needs input.title and/or input.abstract")
            prompt = summary_prompt(title, abstract)

            def check(text: str) -> tuple[bool, str]:
                return summary_ok(text, title)
            local_tokens = 160
        else:
            prompt = str(inp.get("prompt") or "")
            if not prompt.strip():
                raise RouterError("generic needs input.prompt")

            def check(text: str) -> tuple[bool, str]:
                return (True, "ok") if text.strip() else (False, "empty")
            local_tokens = int(min(int(inp.get("max_tokens", 300)), 1000))

        plan = ["local"] if mode == "local" else ["cloud", "local"]
        attempts: list[dict] = []
        total_cost: Optional[float] = 0.0 if self.prices_configured else None

        for n_try, route in enumerate(plan):
            outcome_ok = "ok" if n_try == 0 else "fallback"
            if route == "cloud":
                allowed, why_not = self._cloud_allowed(session_id)
                if not allowed:
                    self._count("budget_skips")
                    attempts.append({"route": "cloud", "ok": False, "reason": why_not})
                    self._log(user_id=user_id, session_id=session_id, task=task, mode=mode, route="cloud",
                              kind="chat", model=self.cloud_model or "-", outcome="skipped", reason=why_not)
                    continue
                try:
                    res = self._cloud_chat([{"role": "user", "content": prompt}], self.max_tokens[task], session_id)
                except (httpx.HTTPError, RouterUnavailable, VaultError) as exc:
                    self._count("failures")
                    reason = f"cloud failed: {str(exc)[:150]}"
                    attempts.append({"route": "cloud", "ok": False, "reason": reason})
                    self._log(user_id=user_id, session_id=session_id, task=task, mode=mode, route="cloud",
                              kind="chat", model=self.cloud_model, outcome="failed", reason=reason)
                    continue
                cost = self.cost_inr(res["tokens_in"], res["tokens_out"])
                if cost is not None:
                    total_cost = round((total_cost or 0.0) + cost, 6)
                model = self.cloud_model
            else:
                try:
                    res = self._ollama_chat(prompt, local_tokens)
                except httpx.HTTPError as exc:
                    self._count("failures")
                    reason = f"local failed: {exc!r}"[:200]
                    attempts.append({"route": "local", "ok": False, "reason": reason})
                    self._log(user_id=user_id, session_id=session_id, task=task, mode=mode, route="local",
                              kind="chat", model=self.chat_model, outcome="failed", reason=reason)
                    continue
                res["retried"], res["finish_reason"], cost = False, None, (0.0 if self.prices_configured else None)
                model = self.chat_model

            text = clean_summary(res["text"]) if task == "summarize" else res["text"].strip()
            ok, why = check(text)
            reason = None if ok else f"validator: {why}"
            if res.get("retried"):
                reason = (reason + "; " if reason else "") + "retried (empty reply)"
            if n_try > 0 and ok:
                self._count("fallbacks")
                reason = "after: " + "; ".join(a["reason"] for a in attempts if not a["ok"]) + \
                         (f"; {reason}" if reason else "")
            if not ok:
                self._count("failures")
            self._log(user_id=user_id, session_id=session_id, task=task, mode=mode, route=route, kind="chat",
                      model=model, outcome=outcome_ok if ok else "failed", reason=reason,
                      tokens_in=res["tokens_in"], tokens_out=res["tokens_out"], cost_inr=cost, latency_ms=res["ms"])
            attempts.append({"route": route, "ok": ok, "reason": why if not ok else "ok",
                             "ms": round(res["ms"], 1), "tokens_in": res["tokens_in"], "tokens_out": res["tokens_out"],
                             "cost_inr": cost})
            if ok:
                return {"task": task, "mode": mode, "route": route, "model": model, "text": text,
                        "fallback": n_try > 0, "attempts": attempts, "cost_inr": total_cost,
                        "latency_ms": round((time.perf_counter() - t_start) * 1000.0, 1)}

        raise RouterUnavailable(f"no route produced a valid answer: {attempts}")

    # ================================================================== reporting

    def llm_calls(self, session_id: Optional[str] = None, user_id: Optional[str] = None,
                  limit: int = 200) -> list[dict]:
        where, args = [], []
        if session_id:
            where.append("session_id = %s"); args.append(session_id)
        if user_id:
            where.append("user_id = %s"); args.append(user_id)
        sql = "SELECT * FROM llm_calls" + (" WHERE " + " AND ".join(where) if where else "") + \
              " ORDER BY id DESC LIMIT %s"
        with self.pool.connection() as conn:
            return conn.execute(sql, (*args, limit)).fetchall()

    def totals(self, session_id: Optional[str] = None) -> list[dict]:
        sql = """
            SELECT task, route, kind, outcome, count(*) AS calls, SUM(items) AS items,
                   SUM(tokens_in) AS tokens_in, SUM(tokens_out) AS tokens_out,
                   ROUND(SUM(cost_inr)::numeric, 4) AS cost_inr,
                   ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)::numeric, 1) AS p50_ms
            FROM llm_calls {where}
            GROUP BY task, route, kind, outcome ORDER BY task, route, kind, outcome
        """.format(where="WHERE session_id = %s" if session_id else "")
        with self.pool.connection() as conn:
            return conn.execute(sql, (session_id,) if session_id else ()).fetchall()

    def status(self) -> dict:
        with self._stats_lock:
            stats = dict(self.stats)
        return {
            "default_mode": self.default_mode,
            "budget_inr_per_session": self.budget_inr,
            "prices_configured": self.prices_configured,
            "local": {"base_url": self.local_url, "embed_model": self.embed_model, "chat_model": self.chat_model,
                      "keep_alive": self.keep_alive},
            "rank": {"calibration": {"cos_low": self.c_lo, "score_low": self.s_lo,
                                     "cos_high": self.c_hi, "score_high": self.s_hi},
                     "uncertain_band": [self.band_lo, self.band_hi], "cut_gap": self.cut_gap,
                     "max_cloud_calls": self.max_cloud_calls, "embed_chars": self.embed_chars},
            "cloud": {"model": self.cloud_model, "max_tokens": self.max_tokens,
                      "retry_multiplier": self.retry_multiplier},
            "vector_cache": self.cache.stats(),
            "stats": stats,
            "totals": self.totals(),
        }

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.http_local.close()
        self.http_cloud.close()