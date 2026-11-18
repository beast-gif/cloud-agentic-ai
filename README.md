**# Agentic Cloud

**Infrastructure for hosting autonomous AI agents**: freeze idle agents to zero cost, wake them on demand, give them layered memory, and let them use tools only through a secure MCP gateway.

Most cloud platforms are built for stateless web requests that last milliseconds. AI agents are different: they are **stateful**, run for **minutes to days**, sit **idle 90%+ of the time** waiting for LLMs, tools or humans, and execute **untrusted, LLM-generated actions**. This project builds the platform layer for that kind of workload, step by step, and measures every design decision.

> Student project (B.Tech, Electronics and Computer Engineering) by **Sai Amrith Reddy** — [github.com/beast-gif](https://github.com/beast-gif)

---

## Contents

- [Architecture](#architecture)
- [Status](#status)
- [Key results](#key-results)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [API reference](#api-reference)
- [Security model](#security-model)
- [Configuration](#configuration)
- [Benchmarks](#benchmarks)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Known limitations](#known-limitations)

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
  user / app ──────────▶ │               CONTROL PLANE (FastAPI)        │
  POST /sessions         │                                              │
  POST /sessions/{id}/.. │  session store ─ lifecycle (state machine)   │
                         │  idle reaper ─ reconciliation ─ docker driver│
                         │  Memory API (L3) ─ session tokens            │
                         │  credential vault ─ MCP GATEWAY (/mcp)       │
                         └───────┬───────────────┬──────────────┬───────┘
                 start/stop      │               │ JSON-RPC     │ X-Gateway-Secret
                                 ▼               │ + token      │ (+ vault credential)
            ┌──────────────────────────────┐     │              ▼
            │ agent containers (1/session) │─────┘     ┌──────────────────────┐
            │ LangGraph research agent     │           │ tools-net            │
            │ /workspace volume (L2)       │           │  mcp-papers ──▶ arXiv│
            └──────────────┬───────────────┘           │  mcp-notify (mock)   │
                           │ agentic-net               └──────────────────────┘
                           ▼
            ┌──────────────────────────────┐
            │ Postgres 16 + pgvector       │
            │ sessions · checkpoints (L1)  │
            │ memories (L3) · credentials  │
            │ audit tables                 │
            └──────────────────────────────┘
```

**Session lifecycle**

```
POST /sessions ─▶ STARTING ─▶ RUNNING ─▶ WAITING ──(idle 30 s)──▶ FREEZING ─▶ FROZEN
                                 ▲          │                                   │
                                 │          └── event < 30 s: fast path ──┐     │ event
                                 └──────────────────────────────────── RUNNING ◀┘ (wake)
                              RUNNING ─▶ COMPLETED          any state ─▶ FAILED
```

Every state change is a **compare-and-set** SQL update (`UPDATE ... WHERE state = ANY(allowed)`), plus a per-session lock, so the reaper and an incoming event can never both win.

**Three memory layers**

| Layer | Holds | Lifetime | Stored in |
|---|---|---|---|
| **L1** execution state | graph position, pending input, history | one task | LangGraph checkpoints (Postgres) |
| **L2** workspace | files the agent writes (`report.md`, ...) | one session, survives freeze/wake | Docker volume `ws-<session_id>` |
| **L3** long-term memory | what the user already read/researched | across sessions, per user | `memories` table + pgvector (HNSW) |

---

## Status

| Phase | What | Status |
|---|---|---|
| **1** | Agent runtime + control plane: cold start, automatic freeze, wake, fast path, reconciliation, benchmark baseline | ✅ done |
| **2** | Memory layer: pgvector, embeddings (mock / Ollama), Memory API with session tokens, L2 workspace volumes, agent uses L1+L2+L3 | ✅ done |
| **3** | MCP gateway: credential vault, MCP servers (arXiv + mock notify), gateway with allowlists, argument rules, token buckets, audit | 🔄 3.1–3.3 built; 3.4 (agent uses the gateway) and 3.5 (attack + latency tests) next |
| **4** | Model router: local Ollama vs cloud LLM per step, with escalation | ⏳ planned |
| **5** | Firecracker microVMs + `golden.snap` copy-on-write snapshots | ⏳ planned |
| **6** | Final benchmarks vs the Phase 1 baseline | ⏳ planned |

The demo agent is a **research assistant**: it recalls what the user already read, searches papers, shortlists 5, **pauses for the user to pick**, summarizes the picks, writes a report to its workspace and remembers what was read.

---

## Key results

Measured on a laptop (Intel i5-6200U, 2 cores / 4 threads, 16 GB RAM, Docker Desktop on Windows).

**Phase 1 baseline: wake time with plain Docker containers** (n = 20, `client_ms` = what the user waits)

| Path | p50 | p95 | stdev |
|---|---|---|---|
| Fast path (container still alive) | **404 ms** | 438 ms | 21.5 ms |
| Wake (new container + checkpoint) | **5,560 ms** | 14,081 ms | 3,264 ms |

- A wake is **13.8× slower** than the fast path at p50, 32× at p95.
- **88.1%** of a wake is container start + Python imports; loading the agent's own state (checkpoint) is **0.38%** (21.3 ms).
- Wakes are also **unpredictable** (import time ranged 2.1–10.4 s on 2 shared cores). Removing that waste is the goal of Phase 5 (`golden.snap`).
- Moving container cleanup off the response path cut the fast path from ~1,418 ms to **419 ms**.

**Phase 2: L3 memory search scaling** (pgvector, 768-dim vectors, k = 10)

| Memories (N) | Exact search p50 | HNSW p50 | HNSW recall@10 |
|---|---|---|---|
| 100 | 1.75 ms | 1.71 ms | 1.00 |
| 1,000 | 7.27 ms | 2.52 ms | 0.983 |
| 10,000 | 45.84 ms | **3.33 ms** | 0.55* |

- Exact search grows linearly (~4.3 µs per row); HNSW grows ~logarithmically (1.95× for 100× more data vs 2.0× predicted).
- \*Random test vectors are the hardest case for HNSW: most "neighbours" are near-ties (cosine ≈ 0.1) that the similarity threshold discards anyway.

---

## Project structure

```
agentic-cloud/
├── docker-compose.yml        Postgres+pgvector, mcp-papers, mcp-notify; networks agentic-net / tools-net
├── config.yaml               all non-secret settings (lifecycle, memory, workspace, vault, MCP policies)
├── .env.example              template for secrets (.env is git-ignored)
├── db/
│   ├── init.sql              sessions, session_events, wake_metrics
│   └── migrations/           002 memory (pgvector) · 003 session tokens · 004 credentials · 005 tool_calls
├── agent_runtime/            the agent image (one container per session)
│   ├── server.py             FastAPI: /health, /task, /resume, /state
│   ├── graph.py              LangGraph research agent (recall → search → rank → pause → summarize → report → remember)
│   ├── llm.py                mock or Ollama LLM
│   ├── tools.py              mock paper search (Phase 1/2)
│   └── memory_client.py      client for the Memory API (L3)
├── control_plane/            runs on the host
│   ├── main.py               HTTP API
│   ├── lifecycle.py          state machine: create, fast path, wake, freeze, liveness, reconcile, workspace GC
│   ├── session_store.py      Postgres access with compare-and-set transitions
│   ├── docker_driver.py      start/stop containers, workspace volumes, /health polling
│   ├── idle_reaper.py        background freeze / liveness / workspace GC
│   ├── auth.py               per-container session tokens (SHA-256 hashed, rotated on every wake)
│   ├── embeddings.py         mock hashing embedder / Ollama nomic-embed-text
│   ├── memory_store.py       pgvector search + deduplicating writes
│   ├── vault.py              Fernet-encrypted credentials, user/provider binding, key rotation
│   ├── gateway.py            MCP gateway: allowlist, argument rules, token buckets, vault injection, audit
│   ├── migrations.py         applies db/migrations/*.sql once each
│   └── settings.py           config.yaml + .env loading
├── mcp_servers/
│   ├── common/mcp_http.py    minimal MCP-over-HTTP (JSON-RPC 2.0) server, gateway-secret check
│   ├── papers/server.py      arXiv search: caching, 3 s spacing, retry + cool-down on 429, labelled fallback
│   └── notify/server.py      mock notification provider that requires an API key
└── benchmarks/
    ├── wake_benchmark.py     cold start / wake / fast path, mean / p50 / p95, memory cost, baseline diff
    ├── memory_scaling.py     exact vs HNSW search time and recall as memories grow
    └── results/              JSON summaries are committed; raw CSVs are git-ignored
```

---

## Prerequisites

- **Docker Desktop** (WSL 2 backend on Windows) with **hardware virtualization enabled** in BIOS/UEFI (Intel VT-x / AMD-V)
- **Python 3.11+** on the host (tested with 3.13) and a virtual environment
- Internet access for real arXiv search (optional: the papers server falls back to labelled placeholder results)
- Optional: **Ollama** with `nomic-embed-text` for real semantic embeddings

---

## Setup

```powershell
git clone https://github.com/beast-gif/agentic-cloud.git
cd agentic-cloud
python -m venv venv
venv\Scripts\activate
pip install -r control_plane/requirements.txt
```

**1. Secrets.** Create `.env` from the template and fill in three values:

```powershell
Copy-Item .env.example .env
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"   # -> VAULT_MASTER_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"                              # -> MCP_GATEWAY_SECRET
notepad .env
```

Keep one `KEY=VALUE` per line, no spaces around `=`, and save as ASCII/UTF-8 **without BOM**. Back up `VAULT_MASTER_KEY` separately: losing it makes every stored credential unrecoverable.

**2. Infrastructure** (Postgres + MCP servers):

```powershell
docker compose up -d --build
docker compose ps          # wait until all services are "healthy"
```

**3. Agent image:**

```powershell
docker build -t agentic-cloud/agent:phase1 ./agent_runtime
```

**4. Control plane:**

```powershell
python -m control_plane.main
```

Startup applies pending migrations and logs `vault enabled`, `gateway enabled: 2 servers, 5 tools ...` and `control plane ready`. Interactive API docs: <http://127.0.0.1:8000/docs>.

---

## Usage

PowerShell helpers:

```powershell
$api = "http://127.0.0.1:8000"
function Post($path, $obj, $headers = @{}) {
  Invoke-RestMethod -Method Post -Uri "$api$path" -ContentType "application/json" -Body ($obj | ConvertTo-Json -Depth 6) -Headers $headers
}
```

**Start a research session** (cold start; the agent pauses with a shortlist):

```powershell
$r = Post "/sessions" @{ user_id = "amrith"; agent_type = "research"; input = @{ topic = "KV cache compression" } }
$sid = $r.session.session_id
$r.agent.data.shortlist | Format-Table rank, title
```

**Answer the agent** (fast path if within 30 s, otherwise it is woken from FROZEN):

```powershell
$d = Post "/sessions/$sid/events" @{ value = @{ selected = @(1, 3) } }
$d.path; $d.session.state; $d.timings_ms
```

**Inspect everything:**

```powershell
Invoke-RestMethod "$api/sessions/$sid/events"            # audit trail of the session
Invoke-RestMethod "$api/sessions/$sid/workspace?content=true"   # L2 files (works even when FROZEN)
Invoke-RestMethod "$api/users/amrith/memories"           # L3 memories
Invoke-RestMethod "$api/sessions/$sid/tool-calls"        # MCP gateway audit
Invoke-RestMethod "$api/healthz"
```

**Store a provider credential in the vault** (needed for `notify.send`; `owner_chat` is where the user's own notifications go):

```powershell
Invoke-RestMethod -Method Put -Uri "$api/admin/vault/amrith/notify" -ContentType "application/json" `
  -Body (@{ secret = "ntf_live_..."; metadata = @{ owner_chat = "amrith-phone" } } | ConvertTo-Json)
```

---

## API reference

| Area | Endpoint | Purpose |
|---|---|---|
| Sessions | `POST /sessions` | create + cold start `{user_id, agent_type, input}` |
| | `POST /sessions/{id}/events` | send the value the agent waits for (fast path or wake) |
| | `POST /sessions/{id}/freeze` | freeze now (testing) |
| | `GET /sessions`, `GET /sessions/{id}` | list / inspect |
| | `GET /sessions/{id}/events`, `/metrics`, `/workspace`, `/tool-calls` | audit, wake metrics, L2 files, gateway audit |
| Memory (agents) | `POST /memory/search`, `POST /memory/write` | L3, needs `Authorization: Bearer <MEMORY_TOKEN>` |
| Memory (admin) | `POST /admin/memory/search`, `/write`; `GET`/`DELETE /users/{id}/memories`; `GET /admin/memory/stats` | local testing |
| Vault (admin) | `PUT /admin/vault/{user}/{provider}`, `GET /admin/vault/{user}`, `POST .../verify`, `DELETE ...`, `GET /admin/vault/audit`, `POST /admin/vault/rotate` | secrets are never returned |
| MCP gateway | `POST /mcp` | JSON-RPC `initialize`, `ping`, `tools/list`, `tools/call` for agents (session token) |
| | `GET /admin/gateway`, `POST /admin/gateway/refresh` | servers, policies, catalog, token buckets |
| Ops | `GET /metrics/wake`, `GET /admin/workspaces`, `POST /admin/workspaces/gc`, `POST /admin/reconcile`, `GET /healthz` | |

`/admin/*` endpoints have **no authentication**: the control plane listens on `127.0.0.1` only.

---

## Security model

| Threat | Defence |
|---|---|
| An agent pretends to be another user | Per-container **session token** (256-bit, only its SHA-256 hash stored); the server derives `user_id` from the token, never from the request; tokens die on freeze and rotate on every wake |
| A prompt-injected agent calls a dangerous tool | Gateway **allowlist** per agent type: hidden tools do not appear in `tools/list` and are denied in `tools/call` |
| A prompt-injected agent exfiltrates data via a tool argument | **Argument rules**, e.g. `notify.send` may only send to the user's own `owner_chat` (filled in by the gateway) |
| An agent bypasses the gateway | MCP servers live on a separate network (`tools-net`) **and** require `X-Gateway-Secret` (constant-time comparison) |
| Credentials leak through the database (agents have DB access, backups, logs) | **Fernet** encryption (AES-128-CBC + HMAC-SHA256); master key only in the control plane's environment |
| Ciphertext tampering / swapping between users | HMAC rejects any modification; user + provider are encrypted **with** the secret and checked on decrypt |
| A single agent or burst overloads an external API | **Token buckets** per server (`global` or per-`session` scope) + server-side spacing, retry and cool-down |
| "What did this agent do?" | Audit tables: `session_events`, `vault_audit`, `tool_calls` (including denied and rate-limited attempts) |

---

## Configuration

Non-secret settings live in **`config.yaml`**; secrets live in **`.env`** (git-ignored, loaded only into the control plane and the MCP servers, never into agent containers).

| Section | Notable keys |
|---|---|
| `lifecycle` | `idle_timeout_s` (30), `reaper_interval_s` (5), `health_poll_ms` (100) |
| `llm` | `provider: mock \| ollama` |
| `memory.embeddings` | `provider: mock \| ollama`, per-provider `min_score` and `dedup_threshold` |
| `workspace` | `keep_after_complete_s` (3600), `gc_every_ticks` |
| `vault` | `master_key_env` |
| `mcp` | `servers` (URL + vault credential), `policies.<agent_type>.allow / rules / rate_limits` |

| `.env` variable | Used by |
|---|---|
| `VAULT_MASTER_KEY` | control plane (comma-separated, newest first, for rotation) |
| `MCP_GATEWAY_SECRET` | control plane + MCP servers |
| `NOTIFY_ACCOUNTS` | mock notify provider (`key:account,...`) |

Changes to `.env` or `config.yaml` take effect after restarting the control plane (and `docker compose up -d` for the MCP servers).

---

## Benchmarks

```powershell
# wake vs fast path (control plane must be running), compared with an earlier result
python -m benchmarks.wake_benchmark --n 20 --baseline benchmarks\results\phase1_<timestamp>.json

# vector search scaling (talks to Postgres directly)
python -m benchmarks.memory_scaling --sizes 100 1000 5000 10000 --ef-search 40
```

Results are written to `benchmarks/results/` (JSON summaries committed, raw CSVs ignored). Reported statistics: mean, p50, p95 (nearest-rank), min, max, stdev. Set `idle_timeout_s` back to 30 and close heavy applications before benchmarking.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `docker` not recognized | Restart the terminal / VS Code after installing Docker Desktop |
| Docker Desktop: "Virtualization is disabled in firmware" | Enable Intel VT-x / AMD-V in BIOS/UEFI |
| `PoolTimeout` on control-plane start | Postgres not running: `docker compose up -d` |
| Agent init fails with `No module named 'agent_runtime'` | Imports inside `agent_runtime/` must be plain (`from graph import ...`); rebuild the image |
| `vault DISABLED` | `VAULT_MASTER_KEY` missing, or `.env` saved with a BOM; recreate it with `-Encoding ascii` |
| `.env` change has no effect | Secrets are read at startup: restart the control plane |
| Session token suddenly gives 401 | The session was frozen (token dies on freeze); raise `idle_timeout_s` while testing |
| arXiv search returns `source: fallback` / 429 | arXiv rate-limits per IP; wait a few minutes. The server retries, cools down for 60 s and serves labelled placeholders meanwhile |
| Gateway lists no tools | MCP servers down or secret mismatch: `docker compose ps`, then `POST /admin/gateway/refresh` |

---

## Roadmap

- **3.4** Agent uses the gateway: real arXiv papers via `papers.search`, "brief ready" via `notify.send`
- **3.5** Prompt-injection test (planted abstract tries to message an outsider), rate-limit bursts, gateway latency per hop
- **4** Model router: send easy steps to a small local model (Ollama, CPU) and hard steps to a cloud LLM; escalate on low confidence; measure cost and latency
- **5** Firecracker microVMs: a pre-warmed `golden.snap` cloned with copy-on-write (`MAP_PRIVATE`), diff snapshots, working-set prefetch; target wake of tens of milliseconds instead of ~5.5 s
- **6** Final benchmarks against the Phase 1 baseline (p50 and especially p95)

---

## Known limitations

- Agents receive the full Postgres DSN (needed by the LangGraph checkpointer), so they can read every table; secrets are encrypted, but a least-privilege database role for agents is planned.
- `POST /sessions` trusts the `user_id` it is given; a real deployment needs user authentication in front of it.
- `/admin/*` endpoints are unauthenticated and rely on the control plane binding to `127.0.0.1`.
- Single host: workspace volumes and snapshots are local (multi-host would need object storage).
- `MCP_GATEWAY_SECRET` is one shared secret; production systems typically use mutual TLS per service.
- The notify provider is a mock; the papers server's fallback results are placeholders, not real papers.**
