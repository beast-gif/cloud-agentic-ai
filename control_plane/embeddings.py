"""
Agentic Cloud - Phase 2, Step 2.2
Embedding providers: text -> vector of 768 floats (unit length).

  mock   : feature hashing of words + character trigrams. Deterministic, ~0 ms, no model.
           Only "understands" shared words/word-parts, not meaning. For testing plumbing.
  ollama : nomic-embed-text through Ollama's /api/embed. Real semantic similarity.

All vectors are L2-normalized (length 1), so cosine similarity = dot product.

IMPORTANT: vectors from different providers/models are NOT comparable. Each embedder
has a `name` that is stored with every memory (memories.embedding_model) and every
search filters on it.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Literal, Protocol

import httpx

Mode = Literal["document", "query"]

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of",
    "on", "or", "over", "that", "the", "this", "to", "we", "with", "show", "shows", "our",
}


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str], mode: Mode = "document") -> list[list[float]]: ...

    def stats(self) -> dict: ...


class _Timed:
    def __init__(self) -> None:
        self.calls = 0
        self.texts = 0
        self.total_ms = 0.0

    def _record(self, t0: float, n: int) -> None:
        self.calls += 1
        self.texts += n
        self.total_ms += (time.perf_counter() - t0) * 1000.0

    def stats(self) -> dict:
        return {
            "name": getattr(self, "name", "?"),
            "calls": self.calls,
            "texts": self.texts,
            "avg_ms_per_text": round(self.total_ms / self.texts, 2) if self.texts else 0.0,
        }


# ============================================================ mock

class MockEmbedder(_Timed):
    """Feature hashing ("hashing trick").

    Each feature (a word, or a 3-letter piece of a word) is hashed to one of `dim`
    slots and adds +weight or -weight there. Texts sharing words/word-parts end up
    pointing in similar directions. Example: "compression" and "compressing" share
    the trigrams "com", "omp", "mpr", "pre", "res", "ess" -> partial match.
    """

    WORD_WEIGHT = 1.0
    TRIGRAM_WEIGHT = 0.3

    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dim = dim
        self.name = f"mock-hash-{dim}"

    def embed(self, texts: list[str], mode: Mode = "document") -> list[list[float]]:
        t0 = time.perf_counter()
        out = [self._one(t) for t in texts]
        self._record(t0, len(texts))
        return out

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]
        features: list[tuple[str, float]] = [(w, self.WORD_WEIGHT) for w in words]
        for w in words:
            if len(w) > 3:
                features.extend((f"#{w[i:i + 3]}", self.TRIGRAM_WEIGHT) for i in range(len(w) - 2))
        for feat, weight in features:
            h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[idx] += sign * weight
        return _normalize(vec)


# ============================================================ ollama

class OllamaEmbedder(_Timed):
    """nomic-embed-text via Ollama.

    nomic-embed-text is trained with task prefixes: documents are embedded as
    "search_document: <text>" and queries as "search_query: <text>". Using them
    noticeably improves retrieval quality.
    """

    PREFIXES = {"document": "search_document: ", "query": "search_query: "}

    def __init__(self, base_url: str, model: str = "nomic-embed-text", dim: int = 768,
                 timeout_s: float = 60.0, use_prefixes: bool = True) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.use_prefixes = use_prefixes
        self.name = f"ollama:{model}"
        self.http = httpx.Client(timeout=timeout_s)

    def embed(self, texts: list[str], mode: Mode = "document") -> list[list[float]]:
        t0 = time.perf_counter()
        prefix = self.PREFIXES[mode] if self.use_prefixes else ""
        resp = self.http.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": [prefix + t for t in texts]},
        )
        resp.raise_for_status()
        vecs = resp.json().get("embeddings") or []
        if len(vecs) != len(texts):
            raise RuntimeError(f"Ollama returned {len(vecs)} embeddings for {len(texts)} texts")
        for v in vecs:
            if len(v) != self.dim:
                raise RuntimeError(f"{self.model} returned {len(v)} dims, expected {self.dim} "
                                   f"(memories.embedding is vector({self.dim}))")
        self._record(t0, len(texts))
        return [_normalize(v) for v in vecs]


# ============================================================ factory

def make_embedder(cfg: dict) -> Embedder:
    e = (cfg.get("memory") or {}).get("embeddings") or {}
    provider = str(e.get("provider", "mock")).lower()
    dim = int(e.get("dim", 768))
    if provider == "mock":
        return MockEmbedder(dim)
    if provider == "ollama":
        o = e.get("ollama") or {}
        return OllamaEmbedder(
            base_url=o.get("base_url", "http://localhost:11434"),
            model=o.get("model", "nomic-embed-text"),
            dim=dim,
            timeout_s=float(o.get("timeout_s", 60)),
            use_prefixes=bool(o.get("use_prefixes", True)),
        )
    raise ValueError(f"unknown embeddings provider {provider!r} (use 'mock' or 'ollama')")


def provider_settings(cfg: dict) -> dict:
    """min_score / dedup threshold for the active provider (they differ per model)."""
    e = (cfg.get("memory") or {}).get("embeddings") or {}
    provider = str(e.get("provider", "mock")).lower()
    block = e.get(provider) or {}
    return {
        "min_score": float(block.get("min_score", 0.5)),
        "dedup_threshold": float(block.get("dedup_threshold", 0.95)),
    }