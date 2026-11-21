# Agentic Cloud

**Infrastructure for hosting autonomous AI agents**: freeze idle agents to zero cost, wake them from a pre-warmed **Firecracker snapshot** in about a second, give them layered memory, let them use tools only through a **secure MCP gateway**, and route every model call to the **cheapest model that can handle it**.

Most cloud platforms are built for stateless web requests that last milliseconds. AI agents are different: they are **stateful**, run for **minutes to days**, sit **idle 90%+ of the time** waiting for LLMs, tools or humans, and execute **untrusted, LLM-generated actions**. This project builds the platform layer for that kind of workload, phase by phase, and measures every design decision.

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
- [Known limitations and future work](#known-limitations-and-future-work)

---

## Architecture

```
                 ┌──────────────────────────────────────────────────────────────┐
 user / app ───▶ │                  CONTROL PLANE (FastAPI, Windows host)       │
 POST /sessions  │  session store · lifecycle state machine · idle reaper       │
                 │  Memory API (L3) · session tokens · credential vault         │
                 │  MCP GATEWAY (/mcp) · MODEL ROUTER (/llm/rank, /llm/complete)│
                 │  driver: docker │ firecracker                                │
                 └──────┬──────────────────────┬──────────────┬─────────────────┘
       docker run/stop  │                      │ POST /vms    │ X-Gateway-Secret + vault key
                        ▼                      ▼              ▼
   ┌──────────────────────────┐  ┌─────────────────────────────┐  ┌─────────────────────┐
   │ Docker (Phases 1-4)      │  │ fc-host (Hyper-V VM, KVM)   │  │ tools-net           │
   │ 1 container per session  │  │ fc-agent :8090              │  │  mcp-papers ─▶ arXiv│
   │ /workspace volume (L2)   │  │  slot i = netns fc<i>       │  │  mcp-notify (mock)  │
   └────────────┬─────────────┘  │  microVM restored from      │  └─────────────────────┘
                │                │  golden.snap (MAP_PRIVATE)  │
                │                └──────────────┬──────────────┘   Ollama (local models)
                ▼                               ▼                  Velona (cloud model)
   ┌─────────────────────────────────────────────────────────┐
   │ Postgres 16 + pgvector: sessions · checkpoints (L1) ·   │
   │ memories (L3) · credentials · embedding cache · audit   │
   └─────────────────────────────────────────────────────────┘
```

**Session lifecycle** (identical for both drivers)

```
POST /sessions ─▶ STARTING ─▶ RUNNING ─▶ WAITING ──(idle 30 s)──▶ FREEZING ─▶ FROZEN
                                 ▲          │                                   │
                                 │          └── event < 30 s: fast path ──┐     │ event
                                 └──────────────────────────────────── RUNNING ◀┘ (wake)
                              RUNNING ─▶ COMPLETED          any state ─▶ FAILED
```

Every state change is a **compare-and-set** SQL update plus a per-session lock, so the reaper and an incoming event can never both win.

**Three memory layers**

| Layer | Holds | Lifetime | Stored in |
|---|---|---|---|
| **L1** execution state | graph position, pending input, history | one task | LangGraph checkpoints (Postgres) |
| **L2** workspace | files the agent writes (`report.md`, ...) | one session | Docker volume `ws-<session_id>` (Firecracker: RAM inside the microVM) |
| **L3** long-term memory | what the user already read/researched | across sessions, per user | `memories` table + pgvector (HNSW) |

**Wake with Firecracker (Phase 5)**: the expensive part of starting an agent (kernel boot, init, ~3 s of Python imports) is done **once**, then the paused VM is saved as `golden.snap`. A wake restores a copy of it and only has to **bind** the session:

```
cold boot :  kernel 730 ms → init 540 ms → imports ~2,900 ms → bind 70 ms        ≈ 4.6 s
restore   :  spawn ~20 ms → load golden.snap ~30 ms → clock fix → bind ~600-1,200 ms ≈ 0.8-1.5 s
```

**Bind** gives an anonymous clone its session: `SESSION_ID`, a fresh `MEMORY_TOKEN` (rotated on every start/wake), the Postgres connection and the compiled graph. None of that may be inside the shared snapshot.

---

## Status

| Phase | What | Status |
|---|---|---|
| **1** | Agent runtime + control plane: cold start, automatic freeze, wake, fast path, reconciliation, benchmark baseline | ✅ |
| **2** | Memory: pgvector, embeddings, Memory API with session tokens, L2 workspace volumes | ✅ |
| **3** | MCP gateway: credential vault, MCP servers (arXiv + mock notify), allowlists, argument rules, token buckets, audit, red team | ✅ |
| **4** | Model router: local embeddings / local LLM / cloud LLM per task, cascade with escalation, budget cap, latency budget, persistent embedding cache | ✅ |
| **5** | Firecracker microVMs on nested Hyper-V: `golden.snap`, copy-on-write clones, `FirecrackerDriver` + `fc-agent`, one network namespace per microVM | ✅ (working-set prefetch: future work) |

The demo agent is a **research assistant**: it recalls what the user already read, searches arXiv, ranks the papers, **pauses for the user to pick**, summarizes the picks, writes a report, remembers what was read and notifies the user.

---

## Key results

Measured on a laptop: **Intel i5-6200U (2 cores / 4 threads), 16 GB RAM, Windows 10 Pro**, Docker Desktop (WSL 2), Firecracker v1.18 inside a Hyper-V VM (nested virtualization, 2 vCPU, 6 GB). Timings on this machine drift by up to 2× within minutes, so comparisons were measured **back to back in the same session** and are reported as p50 / p95 and as ranges across sessions.

### Phase 1: baseline with Docker containers (n = 20)

| Path | p50 | p95 |
|---|---|---|
| Fast path (container still alive) | **404 ms** | 438 ms |
| Wake (new container + checkpoint) | **5,560 ms** | 14,081 ms |

**88.1%** of a wake is container start + Python imports; loading the agent's own state (checkpoint) is **0.38%** (21 ms).

### Phase 2: L3 memory search (pgvector, 768-dim, k = 10)

| Memories | Exact p50 | HNSW p50 |
|---|---|---|
| 1,000 | 7.27 ms | 2.52 ms |
| 10,000 | 45.84 ms | **3.33 ms** |

Exact search grows linearly (~4.3 µs per row), HNSW roughly logarithmically.

### Phase 3: MCP gateway and security

- **Red team: 11 / 11 tests passed** (`benchmarks/redteam.py`: attacks on tokens, the tool allowlist, argument rules, the gateway secret, rate limits and the vault). Audit log of the run: 6 allowed, 5 denied, 8 rate-limited.
- **Gateway overhead** per tool call: **+14.3 ms** (`papers.search`) and **+27.9 ms** (`notify.send`, including ~13.5 ms vault decryption).

### Phase 4: model router (6 topics, real arXiv papers, k = 5)

Small local LLMs on this CPU **cannot judge relevance** (qwen2.5:1.5b AUC **0.55**, a coin flip), but a local **embedding model can** (nomic-embed-text AUC **1.00**, same as the cloud model).

| Ranking mode | Quality vs cloud | Cost per ranking | Latency p50 |
|---|---|---|---|
| local (embeddings) | 0.971–0.974 | **₹0** | **9 ms** (cache warm) |
| cascade (local + cloud for uncertain papers) | 0.971–0.978 | ₹0.013 (**17–18% of cloud**) | 1.4 s |
| cloud (Claude Haiku 5.5 via Velona) | 1.000 (reference) | ₹0.071–0.075 | 5.5–5.7 s |
| *cloud vs a second cloud run* | *0.988* | | |

- The cloud model doesn't reproduce its own shortlist exactly (0.988, ±1-point integer scores), so local and cascade are close to the achievable ceiling.
- Summaries: cloud **1.6–1.9 s, ≈ ₹0.01**; local qwen ≈ 15 s, ₹0 (offline fallback).
- **Per-session budget**: once ₹0.01 was spent, further summaries switched to the local model automatically; the run completed.
- **Latency budget**: on fresh papers (cache cold), cascade switched to the cloud on 5 of 6 topics and warmed the cache in the background; repeat topics then ran locally (0 cache misses, 2–4 s).
- Online recalibration (aligning local scores with the cloud scale per request) fixed one topic (MoE: 3/5 → 5/5) but had **no significant effect** across 6 topics.

### Phase 5: Firecracker and `golden.snap`

**Boot vs restore** (same session, back to back)

| | Session 1 | Session 2 |
|---|---|---|
| Cold boot of the agent in a microVM (p50) | 4,568 ms | 9,098 ms |
| **Restore from golden.snap + bind (p50)** | **799 ms** | **1,034 ms** |
| Speed-up | **5.7×** | **8.8×** |
| Snapshot load + resume | 32 ms | 49 ms |

- Cold-boot breakdown: kernel 730 ms · init + config 540 ms · **Python imports ~2,900 ms (63%)** · Postgres 55 ms · graph 12 ms.
- The restore is also **more predictable**: cold boots ranged 7.5–12.3 s in one session, restores 0.93–1.45 s (no file reading from disk).
- **Bind after restore is ~10× slower than in a cold boot** (≈ 600–1,200 ms vs 70 ms): lazily loaded memory means ~26,000 first-touch page faults, each expensive under nested virtualization.
- A minimal init (`fc-init`) instead of systemd: the agent process starts ≈ 1.3 s after power-on, compared with ≈ 3.3 s until SSH on a full Ubuntu guest.

**Memory: 4 clones of golden.snap at the same time** (copy-on-write)

| Per clone | RSS 70 MiB | **shared 61 MiB** | **private 9 MiB** | PSS 24 MiB |
|---|---|---|---|---|

4 clones: **280 MiB without sharing vs 96 MiB with copy-on-write** (host free memory dropped by 53 MiB). A cold-booted agent VM uses 180–193 MiB RSS; a restored, bound one ≈ 99–130 MiB.

**Integrated into the control plane** (`driver: firecracker`, real sessions with arXiv, memory, router, notify)

| Lifecycle step | Docker (Phase 1) | Firecracker |
|---|---|---|
| cold start | `docker run` + imports | restore + bind ≈ **1.3–1.6 s** |
| freeze | `docker stop` ≈ 1 s | kill microVM ≈ **33–120 ms** |
| wake | **5,560 ms** p50 | ≈ **1.5–1.7 s** |
| isolation | shared kernel | **own kernel per agent (KVM)** |

Three sessions ran concurrently in three microVMs (identical guest IP, separated by network namespaces), each ≈ 130 MiB RSS.

> Docker and Firecracker wake figures above were measured in different sessions. `benchmarks/driver_benchmark.py` measures both interleaved in one session (see [Benchmarks](#benchmarks)).

---

## Project structure

```
agentic-cloud/
├── docker-compose.yml        Postgres+pgvector (localhost only), mcp-papers, mcp-notify
├── config.yaml               all non-secret settings (lifecycle, memory, MCP policies, router, driver)
├── .env.example              template for secrets (.env is git-ignored)
├── db/migrations/            002 memory · 003 tokens · 004 credentials · 005 tool_calls · 006 llm_calls · 007 embedding_cache
├── agent_runtime/            the agent (Docker image, also the microVM root filesystem)
│   ├── server.py             /health, /task, /resume, /state, /bind (pre-bind mode for snapshots)
│   ├── graph.py              LangGraph research agent
│   ├── llm.py                mock / Ollama / model-router client
│   ├── memory_client.py, mcp_client.py, tools.py
├── control_plane/
│   ├── main.py               HTTP API
│   ├── lifecycle.py          state machine (driver-independent)
│   ├── docker_driver.py      agents as containers
│   ├── firecracker_driver.py agents as microVMs (talks to fc-agent)
│   ├── router.py             model router: embeddings, cascade, budget, latency budget, caches
│   ├── gateway.py, vault.py, auth.py, memory_store.py, embeddings.py, session_store.py, ...
├── mcp_servers/              papers (arXiv: cache, spacing, retry, fallback) · notify (mock provider)
├── fc-host/                  runs inside the Hyper-V VM "fc-host"
│   ├── setup.sh              Firecracker binary + CI kernel + test rootfs
│   ├── hello_vm.sh           boot test of a plain Ubuntu microVM
│   ├── build_agent_rootfs.sh Docker image export → read-only ext4 root disk
│   ├── agent_vm.sh           cold boot of the agent, timings
│   ├── make_golden.sh        create golden.snap (pre-bind, gc.freeze, pause, snapshot)
│   ├── restore_vm.sh         restore + bind timings, CLONES=N memory test
│   ├── fc_agent.py           service: POST /vms (restore + bind), DELETE, network slots
│   ├── install_fc_agent_service.sh   fc-agent as a systemd service
│   ├── guest/fc-init, fc_boot.py     PID 1 inside the microVM, MMDS config, clock helper
│   └── windows/bridge.ps1, export_agent_rootfs.ps1
└── benchmarks/               wake, memory scaling, red team, gateway, llm_probe, router_eval,
                              try_router, driver_benchmark (Docker vs Firecracker) + results/*.json
```

---

## Prerequisites

- **Windows 10/11 Pro** (Hyper-V) or Linux with KVM; hardware virtualization (VT-x + EPT) enabled in BIOS
- **Docker Desktop** (WSL 2 backend), **Python 3.11+**
- **Ollama** with `nomic-embed-text` and `qwen2.5:1.5b` (model router)
- A **Velona** API key for the cloud model (optional: the router falls back to local models)
- For Phase 5: a Hyper-V VM with Ubuntu Server and nested virtualization (see below)

---

## Setup

```powershell
git clone https://github.com/beast-gif/cloud-agentic-ai.git agentic-cloud
cd agentic-cloud
python -m venv venv; venv\Scripts\activate
pip install -r control_plane/requirements.txt
Copy-Item .env.example .env      # fill in VAULT_MASTER_KEY, MCP_GATEWAY_SECRET (commands are in the file)
docker compose up -d --build
docker build -t agentic-cloud/agent:phase1 ./agent_runtime
ollama pull nomic-embed-text; ollama pull qwen2.5:1.5b
python -m control_plane.main     # with driver: docker in config.yaml
```

Store the cloud model key in the vault (never in `.env`):

```powershell
Invoke-RestMethod -Method Put -Uri "http://127.0.0.1:8000/admin/vault/platform/velona" `
  -ContentType "application/json" -Body (@{ secret = "<key>" } | ConvertTo-Json)
```

### Phase 5: Firecracker

1. **Hyper-V VM** `fc-host` (Generation 2, 6 GB fixed RAM, 2 vCPU, `Set-VMProcessor -ExposeVirtualizationExtensions $true`, Secure Boot template `MicrosoftUEFICertificateAuthority`), Ubuntu Server with OpenSSH. Check `ls -l /dev/kvm` inside.
2. **Windows bridge** (Administrator): `.\fc-host\windows\bridge.ps1` forwards :8000 and :5432 to the Hyper-V switch address only, with a firewall rule limited to that subnet.
3. **fc-host**: `scp -r fc-host <user>@<fc-host-ip>:~/`, then `~/fc-host/setup.sh`.
4. **Agent root disk**: `.\fc-host\windows\export_agent_rootfs.ps1 -FcHost <user>@<ip>`, then in fc-host `~/fc-host/build_agent_rootfs.sh`.
5. **golden.snap**: `sudo ~/fc-host/make_golden.sh`.
6. **fc-agent**: add `FC_AGENT_SECRET` to `.env`, then in fc-host `sudo ~/fc-host/install_fc_agent_service.sh`.
7. `config.yaml`: `driver: firecracker`, `firecracker.agent_url: http://<fc-host-ip>:8090`, then restart the control plane (`fc-agent ok`, `agent driver: firecracker`).

---

## Usage

```powershell
$api = "http://127.0.0.1:8000"
$r = Invoke-RestMethod -Method Post "$api/sessions" -ContentType "application/json" -Body (@{
       user_id = "amrith"; agent_type = "research"; input = @{ topic = "KV cache compression"; llm_mode = "cascade" } } | ConvertTo-Json -Depth 5)
$sid = $r.session.session_id
$r.agent.data.shortlist | Format-Table rank, score, scored_by, title

# answer the agent (fast path within 30 s, otherwise it is woken from FROZEN)
$d = Invoke-RestMethod -Method Post "$api/sessions/$sid/events" -ContentType "application/json" `
       -Body (@{ value = @{ selected = @(1, 2) } } | ConvertTo-Json -Depth 5)
$d.path; $d.timings_ms | ConvertTo-Json -Depth 4

Invoke-RestMethod "$api/sessions/$sid/events"        # audit trail
Invoke-RestMethod "$api/sessions/$sid/tool-calls"    # MCP gateway audit
Invoke-RestMethod "$api/sessions/$sid/llm-calls"     # model router audit (route, tokens, ₹)
```

`llm_mode`: `local` (nothing leaves the machine, ₹0) · `cascade` (default) · `cloud`.

---

## API reference

| Area | Endpoint | Purpose |
|---|---|---|
| Sessions | `POST /sessions`, `POST /sessions/{id}/events`, `POST /sessions/{id}/freeze`, `GET /sessions[/{id}]` | lifecycle |
| | `GET /sessions/{id}/events`, `/metrics`, `/workspace`, `/tool-calls`, `/llm-calls` | audit and inspection |
| Agents (session token) | `POST /memory/search`, `/memory/write` | L3 memory |
| | `POST /mcp` | MCP gateway (JSON-RPC) |
| | `POST /llm/rank`, `/llm/complete` | model router |
| Admin (localhost only) | `/admin/vault/...`, `/admin/gateway`, `/admin/router`, `/admin/router/clear-cache`, `/admin/llm/rank`, `/admin/llm/complete`, `/admin/llm-calls`, `/admin/memory/...`, `/admin/reconcile`, `/healthz` | operations and testing |
| fc-agent (X-FC-Secret) | `GET /health`, `POST /vms`, `GET /vms[/{id}]`, `DELETE /vms/{id}`, `GET /vms/{id}/console` | microVMs in fc-host |
| Agent (inside container / microVM) | `GET /health`, `POST /task`, `/resume`, `/bind`, `GET /state/{thread}` | |

---

## Security model

| Threat | Defence |
|---|---|
| An agent pretends to be another user | Per-start **session token** (256-bit, only its SHA-256 hash stored); `user_id` derived from the token; tokens die on freeze and rotate on every wake |
| Prompt-injected agent calls a dangerous tool | Gateway **allowlist** per agent type |
| Exfiltration through tool arguments | **Argument rules** (`notify.send` only to the user's own chat) |
| Agent bypasses the gateway | MCP servers on a separate network **and** `X-Gateway-Secret` |
| Credentials leak (DB, backups, logs) | **Fernet** encryption, user/provider binding, master key only in the control plane; the cloud LLM key is also in the vault and decrypted per call |
| Runaway cloud spending | **Per-session ₹ budget**; after it, calls go to local models |
| Agent escapes its sandbox (Firecracker) | **One kernel per agent** (KVM hardware isolation), minimal device model, agent runs as uid 10001, **read-only root disk**, writes only to RAM |
| Secrets inside the shared snapshot | `golden.snap` is taken **before** bind: no token, no session id, no DB connection inside |
| Unauthorized control of microVMs | fc-agent requires `X-FC-Secret` (from `/etc/fc-agent.env`, root only); reachable only on the Hyper-V network |
| Database exposed on the LAN | Postgres and the control plane bind to `127.0.0.1`; `bridge.ps1` forwards them only to the Hyper-V switch address with a subnet-restricted firewall rule |
| "What did this agent do?" | `session_events`, `vault_audit`, `tool_calls`, `llm_calls` |

---

## Configuration

| `config.yaml` section | Notable keys |
|---|---|
| `driver` | `docker` \| `firecracker` |
| `firecracker` | `agent_url` (fc-agent), `secret_env`, `golden`, `agent_dsn` / `control_plane_url` with `{WIN_HOST}` |
| `lifecycle` | `idle_timeout_s` (30), `reaper_interval_s` (5) |
| `llm` | `provider: mock \| ollama \| router` |
| `router` | `mode`, `budget_inr_per_session`, `rank` (calibration, uncertain band, latency budget, recalibration), `cloud` (model, prices, retries) |
| `memory`, `workspace`, `vault`, `mcp` | see comments in the file |

| `.env` | Used by |
|---|---|
| `VAULT_MASTER_KEY` | control plane (comma-separated for rotation) |
| `MCP_GATEWAY_SECRET` | control plane + MCP servers |
| `NOTIFY_ACCOUNTS` | mock notify provider |
| `FC_AGENT_SECRET` | control plane + fc-agent (Phase 5) |

---

## Benchmarks

```powershell
python -m benchmarks.wake_benchmark --n 20                 # Phase 1: wake / fast path
python -m benchmarks.memory_scaling --sizes 100 1000 10000  # Phase 2: exact vs HNSW
python -m benchmarks.redteam                                # Phase 3: 11 attacks
python -m benchmarks.gateway_benchmark                      # Phase 3: gateway overhead
python -m benchmarks.llm_probe --models qwen2.5:1.5b --embed nomic-embed-text   # Phase 4
python -m benchmarks.router_eval --cold                     # Phase 4: local / cascade / cloud
python -m benchmarks.driver_benchmark --rounds 10           # Phase 5: Docker vs Firecracker, interleaved
```

In fc-host: `sudo RUNS=10 ~/fc-host/restore_vm.sh`, `sudo RUNS=5 ~/fc-host/agent_vm.sh`, `sudo CLONES=4 ~/fc-host/restore_vm.sh`.

JSON summaries are written to `benchmarks/results/`. Close heavy applications, plug in the charger and keep `idle_timeout_s` at 30 when benchmarking.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Virtualization is disabled in firmware` | Enable VT-x in BIOS |
| `PoolTimeout` on start | Postgres not running: `docker compose up -d` |
| `vault DISABLED` | `VAULT_MASTER_KEY` missing or `.env` has a BOM |
| arXiv `source: fallback` / 429 | arXiv rate limit; wait a few minutes |
| `/dev/kvm` missing in fc-host | VM must be off when setting `ExposeVirtualizationExtensions`; Windows 10 WSL2 has no nested KVM, use a Hyper-V VM |
| `fc-agent not reachable` | `systemctl status fc-agent` in fc-host; after a Windows reboot the fc-host IP and the Windows switch address change: update `firecracker.agent_url` and run `bridge.ps1` again |
| microVM can't reach Postgres | `bridge.ps1` not run since the last reboot; Postgres must be published as `127.0.0.1:5432` |
| Rebuilt the agent image | Re-export the rootfs, `build_agent_rootfs.sh`, then `make_golden.sh` (old snapshots keep the old code) |
| Scripts fail with `\r: command not found` | Windows line endings: `sed -i 's/\r$//' ~/fc-host/*.sh` |

---

## Known limitations and future work

- **Working-set prefetch** (future work): record the pages a bind touches and load them in one batch at restore (userfaultfd, as in REAP). Expected to bring bind from ~1 s towards the 70 ms of a cold boot.
- **L2 workspace in Firecracker mode** lives in the microVM's RAM and is lost on freeze (Docker keeps it in a volume). A per-session data disk would fix this.
- **Nested virtualization**: all Firecracker numbers come from KVM inside Hyper-V on a 2-core laptop; bare-metal numbers would be lower.
- Agents receive the Postgres DSN (needed by the LangGraph checkpointer); a least-privilege role per agent is planned. The bind request travels as plain HTTP on the host-only network; vsock or TLS would be the production choice.
- Router: the latency estimate ignores the background embedding queue (one cold-cache topic was misjudged); recalibration showed no significant effect at n = 6 topics.
- `POST /sessions` trusts the given `user_id`; `/admin/*` endpoints rely on binding to localhost.
- Single host; the fc-host IP must be updated in `config.yaml` after a Windows reboot.
