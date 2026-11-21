#!/usr/bin/env bash
# =============================================================================
# Agentic Cloud - Phase 5, Step 5.3
# Restore agents from golden.snap and measure.
#
#   sudo ~/fc-host/restore_vm.sh                         # 1 restore + bind, timings
#   sudo RUNS=10 ~/fc-host/restore_vm.sh                 # p50 / p95 over 10 restores
#   sudo TASK="KV cache compression" ~/fc-host/restore_vm.sh   # + a research run on the clone
#   sudo CLONES=4 ~/fc-host/restore_vm.sh                # memory: 4 clones at once (copy-on-write)
#   sudo NAME=golden-nofreeze CLONES=4 ~/fc-host/restore_vm.sh  # same, snapshot without gc.freeze()
#
# One restore, step by step:
#   spawn    start a new firecracker process (API socket ready)
#   load     PUT /snapshot/load: map golden.mem MAP_PRIVATE (copy-on-write) + restore vCPU and
#            devices + resume. No kernel boot, no init, no imports - the guest continues exactly
#            where the golden VM was paused.
#   clock    POST :8081/clock - first round-trip into the restored guest; fixes its wall clock
#   bind     POST :8080/bind  - DSN, session id: Postgres connect + graph build (agent ready)
#   total    spawn start -> bind done  (compare: cold boot 4,568 ms, Docker wake 5,560 ms)
# =============================================================================
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

NAME="${NAME:-golden}"; RUNS="${RUNS:-1}"; TASK="${TASK:-}"; CLONES="${CLONES:-0}"; KEEP="${KEEP:-0}"
SNAP="$GOLDEN_DIR/$NAME"
[ -f "$SNAP/golden.mem" ] || { echo "missing $SNAP - run make_golden.sh first"; exit 1; }

load_snapshot() {  # <socket>
  fc_api "$1" PUT /snapshot/load "{\"snapshot_path\": \"$SNAP/golden.vmstate\",
       \"mem_backend\": {\"backend_type\": \"File\", \"backend_path\": \"$SNAP/golden.mem\"},
       \"resume_vm\": true}"
}
clock_fix() {  # [netns]
  local pre=(); [ -n "${1:-}" ] && pre=(ip netns exec "$1")
  "${pre[@]}" curl -sf --max-time 5 -X POST "http://$GUEST_IP:8081/clock" -H 'Content-Type: application/json' \
       -d "{\"epoch_ns\": $(date +%s%N)}"
}

FC_PID=""; declare -a PIDS=() NSS=()
cleanup() {
  stop_pid "$FC_PID"; FC_PID=""
  for p in "${PIDS[@]:-}"; do stop_pid "$p"; done; PIDS=()
  for ns in "${NSS[@]:-}"; do [ -n "$ns" ] && ip netns del "$ns" 2>/dev/null || true; done; NSS=()
  rm -f "$RUN"/restore*.sock
  ip link del "$TAP_DEV" 2>/dev/null || true
}
trap cleanup EXIT

echo "snapshot: $SNAP ($(python3 -c "import json;m=json.load(open('$SNAP/golden.json'));print('gc_freeze=%s, created %s' % (m['gc_freeze'], m['created']))"))"

# =============================================================== memory test: N clones at once
if [ "$CLONES" -gt 0 ]; then
  # Every clone has the SAME guest IP/MAC (they are copies of one VM), so each one gets its own
  # network namespace: an isolated copy of the host network stack with its own fc-tap0.
  # (No route to Windows from these namespaces: the clones stay in pre-bind, we only measure memory.)
  sync; echo 3 > /proc/sys/vm/drop_caches
  FREE0=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  for i in $(seq 1 "$CLONES"); do
    ns="fcclone$i"; ip netns del "$ns" 2>/dev/null || true; ip netns add "$ns"; NSS+=("$ns")
    make_tap "$ns"
    sock="$RUN/restore$i.sock"
    start_fc "$sock" "$RUN/restore$i.console.log" "$ns"; PIDS+=("$FC_PID"); FC_PID=""
    t0=$(now_ms); load_snapshot "$sock"
    clock_fix "$ns" >/dev/null
    ip netns exec "$ns" curl -sf --max-time 5 "http://$GUEST_IP:8080/health" >/dev/null
    echo "clone $i restored + answering in $(( $(now_ms) - t0 )) ms"
  done
  sleep 2
  echo "--- memory per firecracker process (MiB). RSS counts shared pages fully; PSS divides each"
  echo "    shared page by the number of processes sharing it; Private = pages only this clone has."
  printf "%-8s %6s %6s %8s %7s\n" clone RSS PSS Private Shared
  TR=0; TP=0; TPRIV=0
  for i in $(seq 1 "$CLONES"); do
    read -r r p pv sh < <(mem_stats "${PIDS[$((i-1))]}")
    printf "%-8s %6s %6s %8s %7s\n" "$i" "$r" "$p" "$pv" "$sh"
    TR=$((TR + r)); TP=$((TP + p)); TPRIV=$((TPRIV + pv))
  done
  FREE1=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  echo "sum: RSS $TR MiB | PSS $TP MiB | private $TPRIV MiB | host MemAvailable dropped by $((FREE0 - FREE1)) MiB"
  echo "     (without sharing, $CLONES clones would need about $TR MiB; with copy-on-write ~ $TP MiB)"
  cleanup
  exit 0
fi

# =============================================================== restore + bind (timed)
setup_nat
declare -a TOT=() LOADS=() BINDS=()
for run in $(seq 1 "$RUNS"); do
  cleanup
  make_tap
  sock="$RUN/restore.sock"; log="$RUN/restore.console.log"
  sid="fc-restore-$(date +%s)-$run"
  t0=$(now_ms)
  start_fc "$sock" "$log"
  t1=$(now_ms)
  load_snapshot "$sock"
  t2=$(now_ms)
  clk="$(clock_fix)"
  t3=$(now_ms)
  bind="$(python3 - "$DSN" "$sid" <<'PY'
import json, sys
print(json.dumps({"env": {"AGENT_DSN": sys.argv[1], "SESSION_ID": sys.argv[2], "LLM_PROVIDER": "mock",
                          "MOCK_LATENCY_MS": "150", "LOG_LEVEL": "INFO"}}))
PY
)"
  resp="$(curl -sf --max-time 30 -X POST "http://$GUEST_IP:8080/bind" -H 'Content-Type: application/json' -d "$bind")" \
    || { echo "bind failed:"; curl -s -X POST "http://$GUEST_IP:8080/bind" -H 'Content-Type: application/json' -d "$bind"; echo; exit 1; }
  t4=$(now_ms)
  read -r r p pv sh < <(mem_stats "$FC_PID")
  python3 - "$run" "$t0" "$t1" "$t2" "$t3" "$t4" "$resp" "$clk" "$r" "$pv" <<'PY'
import json, sys
run = sys.argv[1]; t0, t1, t2, t3, t4 = map(int, sys.argv[2:7])
b = json.loads(sys.argv[7]).get("timings_ms", {}); c = json.loads(sys.argv[8])
print(f"run {run}: total {t4 - t0:5d} ms = spawn {t1 - t0:4d} | load+resume {t2 - t1:5d} | clock {t3 - t2:4d}"
      f" | bind {t4 - t3:5d} (db {b.get('db_connect_ms', 0):.0f}, graph {b.get('graph_build_ms', 0):.0f})"
      f"   clock fixed by {c.get('corrected_by_ms', 0) / 1000:.0f} s, RSS {sys.argv[9]} MiB, private {sys.argv[10]} MiB")
PY
  TOT+=($((t4 - t0))); LOADS+=($((t2 - t1))); BINDS+=($((t4 - t3)))

  if [ -n "$TASK" ] && [ "$run" -eq 1 ]; then
    tid="fc-restored-$(date +%s)"
    echo "--- research run on the restored clone: '$TASK'"
    curl -sf -X POST "http://$GUEST_IP:8080/task" -H 'Content-Type: application/json' \
         -d "{\"thread_id\": \"$tid\", \"input\": {\"topic\": \"$TASK\"}}" | python3 -c '
import json, sys
r = json.load(sys.stdin); d = r.get("data") or {}
print("status:", r.get("status"), "| waiting_for:", r.get("waiting_for"), "| run_ms:", r.get("run_ms"))
for p in (d.get("shortlist") or [])[:5]:
    print("  #{} {}  {}".format(p.get("rank"), p.get("score"), str(p.get("title"))[:70]))'
    curl -sf -X POST "http://$GUEST_IP:8080/resume" -H 'Content-Type: application/json' \
         -d "{\"thread_id\": \"$tid\", \"value\": {\"selected\": [1, 2]}}" | python3 -c '
import json, sys
r = json.load(sys.stdin); d = r.get("data") or {}
print("status:", r.get("status"), "| run_ms:", r.get("run_ms"), "| summaries:", len(d.get("summaries") or []),
      "| files:", d.get("workspace_files"))'
  fi
  if [ "$KEEP" = "1" ]; then echo "--- clone left running (Ctrl+C to stop)"; wait "$FC_PID" || true; fi
  cleanup
done

echo "--- summary (n=$RUNS): total p50 $(median "${TOT[@]}") ms, p95 $(pct95 "${TOT[@]}") ms |" \
     "load+resume p50 $(median "${LOADS[@]}") ms | bind p50 $(median "${BINDS[@]}") ms"
echo "    compare: cold boot (5.2) p50 4,568 ms | Docker wake (Phase 1) p50 5,560 ms"