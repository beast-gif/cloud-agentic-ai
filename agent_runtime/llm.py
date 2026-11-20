"""
Agentic Cloud - Phase 1, Step 1.2 (+ Phase 4, Step 4.3: the model router)
LLM client used by the research-assistant agent.

provider = "mock"   -> no model; deterministic answers + a small artificial delay
provider = "ollama" -> local model through Ollama's /api/generate endpoint (Phase 1-3)
provider = "router" -> the control plane's MODEL ROUTER (Phase 4):
                         POST /llm/rank      score + rank all candidate papers in ONE call
                         POST /llm/complete  task "summarize"
                       The agent only says WHAT it needs; the router decides WHERE it runs
                       (local embeddings / local qwen / cloud), using the session's mode
                       (local | cloud | cascade). The agent never sees a model API key:
                       it authenticates with its session token, like Memory and MCP.

The agent needs three operations:
  rank(topic, papers, k, mode)          -> (papers with "score" + "scored_by", info dict)
  summarize_ex(title, abstract, mode)   -> {"text", "by", "cost_inr", "fallback"}
  summarize(title, abstract)            -> str  (old interface, kept for mock/ollama)

Router problems never fail the run: if /llm/rank or /llm/complete is unreachable or
returns an error, the agent falls back to the deterministic mock (and says so in the
report via the "by" / info fields).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any, Optional

import httpx

log = logging.getLogger("agent.llm")

_WORD = re.compile(r"[a-z0-9]+")
PROVIDERS = ("mock", "ollama", "router")
MODES = ("local", "cloud", "cascade")


class LLM:
    def __init__(
        self,
        provider: str = "mock",
        ollama_base_url: str = "http://host.docker.internal:11434",
        ollama_model: str = "llama3.2:3b",
        temperature: float = 0.2,
        timeout_s: float = 120.0,
        mock_latency_ms: float = 150.0,
        router_url: str = "",
        router_token: str = "",
        router_timeout_s: float = 180.0,
    ) -> None:
        self.provider = provider.lower().strip()
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider: {provider!r} (use one of {PROVIDERS})")

        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.mock_latency_ms = mock_latency_ms

        self.calls = 0
        self.total_ms = 0.0
        # Phase 4: what the router did for this agent (summed over all calls of this container)
        self.router_stats = {"rank_calls": 0, "complete_calls": 0, "cloud_calls": 0, "cost_inr": 0.0,
                             "summaries_cloud": 0, "summaries_local": 0, "fallbacks": 0, "errors": 0}

        self._http = httpx.Client(timeout=timeout_s) if self.provider == "ollama" else None
        self.router_url = (router_url or "").rstrip("/")
        self._router: Optional[httpx.Client] = None
        if self.provider == "router":
            if not self.router_url or not router_token:
                log.warning("LLM_PROVIDER=router but LLM_ROUTER_URL / MEMORY_TOKEN missing -> mock answers")
            else:
                self._router = httpx.Client(timeout=httpx.Timeout(router_timeout_s, connect=5.0),
                                            headers={"Authorization": f"Bearer {router_token}"})

    # ================================================================== ranking

    def rank(self, topic: str, papers: list[dict], k: int = 5,
             mode: Optional[str] = None) -> tuple[list[dict], dict]:
        """Score every paper. Returns (papers + "score" + "scored_by", info).
        With the router this is ONE request for all papers (batched embeddings, parallel cloud calls)."""
        if self.provider == "router" and self._router is not None:
            start = time.perf_counter()
            try:
                body = {"topic": topic, "k": k,
                        "papers": [{"id": p["id"], "title": p["title"], "abstract": p.get("abstract", "")[:1600]}
                                   for p in papers]}
                if mode:
                    body["mode"] = mode
                data = self._router_post("/llm/rank", body)
                self.router_stats["rank_calls"] += 1
                by_id = {r["id"]: r for r in data["papers"]}
                out = []
                for p in papers:
                    r = by_id.get(p["id"]) or {}
                    out.append({**p, "score": r.get("score", 0.0),
                                "scored_by": r.get("route", "?"), "score_reason": r.get("reason")})
                s = data.get("summary") or {}
                self.router_stats["cloud_calls"] += int(s.get("cloud_calls") or 0)
                self.router_stats["cost_inr"] += float(s.get("cost_inr") or 0.0)
                info = {"provider": "router", "mode": data.get("mode"),
                        "effective_mode": data.get("effective_mode"),
                        "cloud_calls": s.get("cloud_calls"), "escalated": s.get("escalated"),
                        "cost_inr": s.get("cost_inr"), "latency_ms": s.get("latency_ms")}
                return out, info
            except Exception as exc:  # noqa: BLE001 - the router must never fail the run
                self.router_stats["errors"] += 1
                self.router_stats["fallbacks"] += 1
                log.warning("router /llm/rank failed (%s) -> mock scoring", exc)
                out = [{**p, "score": self._mock_score(topic, p["title"], p.get("abstract", ""), sleep=False),
                        "scored_by": "mock"} for p in papers]
                return out, {"provider": "router", "mode": mode, "error": str(exc)[:200], "fallback": "mock"}
            finally:
                self._record(start)

        # mock / ollama: one call per paper (Phase 1-3 behaviour)
        out = [{**p, "score": self.score_relevance(topic, p["title"], p.get("abstract", "")[:800]),
                "scored_by": self.provider} for p in papers]
        return out, {"provider": self.provider, "mode": None}

    def score_relevance(self, topic: str, title: str, abstract: str) -> float:
        start = time.perf_counter()
        try:
            if self.provider != "ollama":
                return self._mock_score(topic, title, abstract)
            prompt = (
                "You rate how relevant a research paper is to a topic.\n"
                f"Topic: {topic}\n"
                f"Title: {title}\n"
                f"Abstract: {abstract}\n\n"
                "Reply with ONLY a number from 0 to 10, where 10 means perfectly relevant."
            )
            text = self._ollama(prompt, num_predict=8)
            match = re.search(r"\d+(\.\d+)?", text)
            score = float(match.group()) if match else 0.0
            return max(0.0, min(10.0, score))
        finally:
            self._record(start)

    # ================================================================== summaries

    def summarize_ex(self, title: str, abstract: str, mode: Optional[str] = None) -> dict:
        """Summary + who wrote it. by = 'cloud:<model>' | 'local:<model>' | 'ollama:<model>' | 'mock'."""
        if self.provider == "router" and self._router is not None:
            start = time.perf_counter()
            try:
                body: dict[str, Any] = {"task": "summarize",
                                        "input": {"title": title, "abstract": abstract[:1600]}}
                if mode:
                    body["mode"] = mode
                data = self._router_post("/llm/complete", body)
                self.router_stats["complete_calls"] += 1
                route = data.get("route", "?")
                self.router_stats["summaries_cloud" if route == "cloud" else "summaries_local"] += 1
                self.router_stats["cost_inr"] += float(data.get("cost_inr") or 0.0)
                if route == "cloud":
                    self.router_stats["cloud_calls"] += 1
                if data.get("fallback"):
                    self.router_stats["fallbacks"] += 1
                return {"text": data["text"], "by": f"{route}:{data.get('model')}",
                        "cost_inr": data.get("cost_inr"), "fallback": bool(data.get("fallback"))}
            except Exception as exc:  # noqa: BLE001
                self.router_stats["errors"] += 1
                self.router_stats["fallbacks"] += 1
                log.warning("router /llm/complete failed (%s) -> mock summary", exc)
                return {"text": self._mock_summary(title, abstract, sleep=False), "by": "mock",
                        "cost_inr": 0.0, "fallback": True, "error": str(exc)[:200]}
            finally:
                self._record(start)

        text = self.summarize(title, abstract)
        return {"text": text, "by": f"ollama:{self.ollama_model}" if self.provider == "ollama" else "mock",
                "cost_inr": 0.0, "fallback": False}

    def summarize(self, title: str, abstract: str) -> str:
        start = time.perf_counter()
        try:
            if self.provider != "ollama":
                return self._mock_summary(title, abstract)
            prompt = (
                "Summarize this research paper in 2 short sentences for a student.\n"
                f"Title: {title}\n"
                f"Abstract: {abstract}\n"
            )
            return self._ollama(prompt, num_predict=120).strip()
        finally:
            self._record(start)

    # ================================================================== stats / lifecycle

    def stats(self) -> dict:
        out = {
            "provider": self.provider,
            "calls": self.calls,
            "total_ms": round(self.total_ms, 1),
            "avg_ms": round(self.total_ms / self.calls, 1) if self.calls else 0.0,
        }
        if self.provider == "router":
            out["router"] = {**self.router_stats, "cost_inr": round(self.router_stats["cost_inr"], 6)}
        return out

    def close(self) -> None:
        for client in (self._http, self._router):
            if client is not None:
                client.close()

    # ================================================================== internals

    def _record(self, start: float) -> None:
        self.calls += 1
        self.total_ms += (time.perf_counter() - start) * 1000

    def _router_post(self, path: str, body: dict) -> dict:
        assert self._router is not None
        resp = self._router.post(f"{self.router_url}{path}", json=body)
        if resp.status_code != 200:
            raise RuntimeError(f"{path} HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def _ollama(self, prompt: str, num_predict: int) -> str:
        assert self._http is not None
        resp = self._http.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature, "num_predict": num_predict},
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _mock_sleep(self) -> None:
        if self.mock_latency_ms > 0:
            time.sleep(self.mock_latency_ms / 1000.0)

    def _mock_score(self, topic: str, title: str, abstract: str, sleep: bool = True) -> float:
        """Score = 10 x (fraction of topic words found in title+abstract) + tiny stable jitter."""
        if sleep:
            self._mock_sleep()
        topic_words = set(_WORD.findall(topic.lower()))
        text_words = set(_WORD.findall(f"{title} {abstract}".lower()))
        overlap = len(topic_words & text_words) / len(topic_words) if topic_words else 0.0
        # stable jitter in [0, 0.5) so equal-overlap papers still get a fixed order
        jitter = (int(hashlib.md5(title.encode()).hexdigest(), 16) % 50) / 100.0
        return round(min(10.0, overlap * 9.5 + jitter), 2)

    def _mock_summary(self, title: str, abstract: str, sleep: bool = True) -> str:
        if sleep:
            self._mock_sleep()
        sentences = [s.strip() for s in abstract.split(".") if s.strip()]
        core = sentences[1] if len(sentences) > 1 else (sentences[0] if sentences else title)
        result = sentences[-1] if sentences else ""
        return f"'{title}': {core}. {result}."