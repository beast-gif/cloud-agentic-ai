"""
Agentic Cloud - Phase 1, Step 1.2
Mock tools for the research-assistant agent.

Phase 1 tests the INFRASTRUCTURE (start / freeze / wake), not search quality, so the
paper search is fake but deterministic: the same topic always returns the same papers.
That makes runs repeatable and benchmarks comparable.

Real tools (arXiv, Semantic Scholar, ...) arrive in Phase 3 through the MCP Gateway.
"""

from __future__ import annotations

import hashlib
import random

METHODS = [
    "low-rank projection", "quantization", "token eviction", "sparse attention",
    "knowledge distillation", "mixture-of-experts routing", "retrieval augmentation",
    "speculative decoding", "contrastive pretraining", "reinforcement learning",
]

ANGLES = [
    "efficient", "scalable", "adaptive", "memory-aware", "robust",
    "training-free", "hardware-friendly", "lossless",
]

RESULTS = [
    "reduces memory footprint without retraining",
    "keeps accuracy within 1% of the full model",
    "cuts inference latency on long contexts",
    "generalizes across model sizes",
    "works as a drop-in module for existing pipelines",
]

DATASETS = ["LongBench", "MMLU", "GSM8K", "WikiText-103", "HELM", "RULER"]

VENUES = ["NeurIPS", "ICML", "ICLR", "ACL", "EMNLP", "MLSys", "arXiv"]

# Used to inject some off-topic results so the ranking step actually has work to do
OFF_TOPICS = [
    "image segmentation", "protein folding", "graph neural networks",
    "speech enhancement", "recommender systems", "robot navigation",
]


def search_papers(topic: str, k: int = 8) -> list[dict]:
    """Return k fake papers for a topic, deterministically.

    Uses a private random.Random seeded from the topic, NOT the global random module,
    so it never depends on (or changes) the process-wide RNG state. That matters later:
    in Phase 5 every agent is cloned from golden.snap and the global RNG is reseeded.
    """
    topic = topic.strip()
    seed = int(hashlib.sha256(topic.lower().encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)

    papers: list[dict] = []
    for i in range(k):
        method = rng.choice(METHODS)
        angle = rng.choice(ANGLES)
        # roughly 1 in 3 results is off-topic
        on_topic = (i % 3) != 2
        subject = topic if on_topic else rng.choice(OFF_TOPICS)

        title = f"{angle.capitalize()} {subject} with {method}"
        abstract = (
            f"We study {subject}. We propose a {method}-based approach that "
            f"{rng.choice(RESULTS)}. Experiments on {rng.choice(DATASETS)} show a "
            f"{rng.randint(5, 40)}% improvement over strong baselines."
        )
        papers.append(
            {
                "id": f"mock-{seed % 10000:04d}-{i}",
                "title": title,
                "abstract": abstract,
                "year": rng.randint(2021, 2026),
                "venue": rng.choice(VENUES),
            }
        )
    return papers