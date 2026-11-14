"""
Agentic Cloud - Phase 1, Step 1.2
Minimal LLM client used by the research-assistant agent.

provider = "mock"   -> no model; deterministic answers + a small artificial delay
provider = "ollama" -> local model through Ollama's /api/generate endpoint

The agent only needs two operations:
  score_relevance(topic, title, abstract) -> float in [0, 10]
  summarize(title, abstract)              -> str

Every call is counted and timed (self.calls, self.total_ms) so later phases can
compare the cost/latency of local vs cloud models (Phase 4: Model Router).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time

import httpx

log = logging.getLogger("agent.llm")

_WORD = re.compile(r"[a-z0-9]+")


class LLM:
    def __init__(
        self,
        provider: str = "mock",
        ollama_base_url: str = "http://host.docker.internal:11434",
        ollama_model: str = "llama3.2:3b",
        temperature: float = 0.2,
        timeout_s: float = 120.0,
        mock_latency_ms: float = 150.0,
    ) -> None:
        self.provider = provider.lower().strip()
        if self.provider not in ("mock", "ollama"):
            raise ValueError(f"Unknown LLM provider: {provider!r} (use 'mock' or 'ollama')")

        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.mock_latency_ms = mock_latency_ms

        self.calls = 0
        self.total_ms = 0.0

        self._http = httpx.Client(timeout=timeout_s) if self.provider == "ollama" else None

    # ------------------------------------------------------------------ public API

    def score_relevance(self, topic: str, title: str, abstract: str) -> float:
        start = time.perf_counter()
        try:
            if self.provider == "mock":
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

    def summarize(self, title: str, abstract: str) -> str:
        start = time.perf_counter()
        try:
            if self.provider == "mock":
                return self._mock_summary(title, abstract)
            prompt = (
                "Summarize this research paper in 2 short sentences for a student.\n"
                f"Title: {title}\n"
                f"Abstract: {abstract}\n"
            )
            return self._ollama(prompt, num_predict=120).strip()
        finally:
            self._record(start)

    def stats(self) -> dict:
        return {
            "provider": self.provider,
            "calls": self.calls,
            "total_ms": round(self.total_ms, 1),
            "avg_ms": round(self.total_ms / self.calls, 1) if self.calls else 0.0,
        }

    def close(self) -> None:
        if self._http is not None:
            self._http.close()

    # ------------------------------------------------------------------ internals

    def _record(self, start: float) -> None:
        self.calls += 1
        self.total_ms += (time.perf_counter() - start) * 1000

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

    def _mock_score(self, topic: str, title: str, abstract: str) -> float:
        """Score = 10 x (fraction of topic words found in title+abstract) + tiny stable jitter."""
        self._mock_sleep()
        topic_words = set(_WORD.findall(topic.lower()))
        text_words = set(_WORD.findall(f"{title} {abstract}".lower()))
        overlap = len(topic_words & text_words) / len(topic_words) if topic_words else 0.0
        # stable jitter in [0, 0.5) so equal-overlap papers still get a fixed order
        jitter = (int(hashlib.md5(title.encode()).hexdigest(), 16) % 50) / 100.0
        return round(min(10.0, overlap * 9.5 + jitter), 2)

    def _mock_summary(self, title: str, abstract: str) -> str:
        self._mock_sleep()
        sentences = [s.strip() for s in abstract.split(".") if s.strip()]
        core = sentences[1] if len(sentences) > 1 else (sentences[0] if sentences else title)
        result = sentences[-1] if sentences else ""
        return f"'{title}': {core}. {result}."