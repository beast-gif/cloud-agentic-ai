"""
Agentic Cloud - Phase 1, Step 1.3
  updated in Phase 2, Step 2.2: MEMORY_API_URL
  updated in Phase 3, Step 3.1: load_dotenv() for secrets such as VAULT_MASTER_KEY
Loads config.yaml and builds the environment variables passed to agent containers.

Config path: $AGENTIC_CONFIG if set, otherwise <repo root>/config.yaml
Secrets   : <repo root>/.env (git-ignored), loaded into the CONTROL PLANE's environment only.
            They are never copied into agent containers (agent_env() below does not read them).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | os.PathLike | None = None) -> list[str]:
    """Minimal .env loader: KEY=VALUE lines, '#' comments, optional quotes.

    Existing environment variables win (os.environ.setdefault), so a value set in the
    shell overrides the file. Returns the names that were loaded (never the values).
    """
    env_path = Path(path or REPO_ROOT / ".env")
    if not env_path.exists():
        return []
    loaded = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("AGENTIC_CONFIG") or REPO_ROOT / "config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for section in ("postgres", "docker", "control_plane", "lifecycle", "llm"):
        if section not in cfg:
            raise KeyError(f"config.yaml is missing the '{section}' section")
    return cfg


def agent_env(cfg: dict[str, Any]) -> dict[str, str]:
    """Environment variables every agent container receives (same for all sessions).

    Per-session variables (SESSION_ID, REQUESTED_AT_MS) are added by the Docker driver.
    """
    llm = cfg["llm"]
    ollama = llm.get("ollama", {}) or {}
    env = {
        "AGENT_DSN": cfg["postgres"]["container_dsn"],
        "PORT": str(cfg["docker"]["agent_internal_port"]),
        "LLM_PROVIDER": str(llm.get("provider", "mock")),
        "OLLAMA_BASE_URL": str(ollama.get("base_url", "http://host.docker.internal:11434")),
        "OLLAMA_MODEL": str(ollama.get("model", "llama3.2:3b")),
        "OLLAMA_TEMPERATURE": str(ollama.get("temperature", 0.2)),
        "OLLAMA_TIMEOUT_S": str(ollama.get("timeout_s", 120)),
        "LOG_LEVEL": str(cfg.get("logging", {}).get("level", "INFO")),
        # Phase 2: where the agent reaches the control plane's Memory API.
        # The per-session MEMORY_TOKEN is added by the lifecycle on every container start.
        "MEMORY_API_URL": str(cfg["control_plane"].get("agent_facing_url",
                                                       "http://host.docker.internal:8000")),
    }
    if "mock_latency_ms" in llm:
        env["MOCK_LATENCY_MS"] = str(llm["mock_latency_ms"])
    return env