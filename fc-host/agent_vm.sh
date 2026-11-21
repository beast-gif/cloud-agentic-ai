#!/usr/bin/env bash
# =============================================================================
# Agentic Cloud - Phase 5, Step 5.2
# Cold-boot the AGENT in a Firecracker microVM and measure until /health = 200.
#
#   sudo ~/fc-host/agent_vm.sh                        # 1 boot, timings
#   sudo RUNS=5 ~/fc-host/agent_vm.sh                 # p50 over 5 boots
#   sudo TASK="KV cache compression" ~/fc-host/agent_vm.sh   # + a full research run
#   sudo KEEP=1 ~/fc-host/agent_vm.sh                 # leave the VM running (Ctrl+C stops it)
#   sudo LOG_CONSOLE=1 ~/fc-host/agent_vm.sh          # agent logs to the console log (debug)
#
# Path of a request from the agent to Postgres / the control plane on Windows:
#   microVM 172.16.0.2 -> fc-tap0 -> fc-host (NAT, MASQUERADE) -> eth0 -> Windows (Default
#   Switch address = fc-host's default gateway) -> portproxy -> 127.0.0.1:5432 / :8000
#
# Timeline reported:
#   kernel        InstanceStart -> /sbin/fc-init runs        (guest /proc/uptime at init)
#   init+mmds     fc-init -> config read from MMDS
#   python        agent server: imports, Postgres connect, graph build (its own timings)
#   total         InstanceStart -> GET /health returns 200   (measured from the host)
# =============================================================================
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

USER_HOME="$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)"
FC_DIR="${FC_DIR:-$USER_HOME/fc}"
IMG="$FC_DIR/images"; RUN="$FC_DIR/run"; mkdir -p "$RUN"
VCPUS="${VCPUS:-1}"; MEM_MIB="${MEM_MIB:-512}"; RUNS="${RUNS:-1}"
TASK="${TASK:-}"; KEEP="${KEEP:-0}"; LOG_CONSOLE="${LOG_CONSOLE:-0}"

TAP_DEV="fc-tap0"; TAP_IP="172.16.0.1"; GUEST_IP="172.16.0.2"; GUEST_MAC="06:00:AC:10:00:02"
API_SOCKET="$RUN/agent.sock"; CONSOLE_LOG="$RUN/agent.console.log"
KERNEL="$(ls "$IMG"/vmlinux-* 2>/dev/null | sort -V | tail -1 || true)"
ROOTFS="$IMG/agent-rootfs.ext4"
for f in "$KERNEL" "$ROOTFS"; do
  [ -n "$f" ] && [ -f "$f" ] || { echo "missing $f (setup.sh / build_agent_rootfs.sh)"; exit 1; }
done

# Windows = fc-host's default gateway on the Hyper-V Default Switch
WIN_HOST="${WIN_HOST:-$(ip -4 route show default | awk '{print $3; exit}')}"
UPLINK="$(ip -4 route show default | awk '{print $5; exit}')"
DSN="postgresql://agentic:agentic_dev_pw@$WIN_HOST:5432/agentic_cloud"

# Monotonic clock in ms. NOT "date": the fc-host VM's wall clock is stepped by Hyper-V time
# sync (hv_utils), which produced negative stage times. CLOCK_MONOTONIC never jumps.
now_ms() { perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC -e 'printf "%d\n", clock_gettime(CLOCK_MONOTONIC)*1000'; }
api() {
  local out code
  out="$(curl -sS --unix-socket "$API_SOCKET" -X "$1" "http://localhost$2" \
         -H 'Content-Type: application/json' -d "$3" -w '\n%{http_code}')"
  code="${out##*$'\n'}"
  [ "${code:0:1}" = "2" ] || { echo "API $1 $2 failed ($code): ${out%$'\n'*}"; return 1; }
}
port_open() { timeout 0.5 bash -c "</dev/tcp/$1/$2" 2>/dev/null; }

# ---------------------------------------------------------------- network + NAT (once)
echo "Windows host: $WIN_HOST (uplink $UPLINK)"
for p in 5432 8000; do
  if port_open "$WIN_HOST" "$p"; then echo "  reachable: $WIN_HOST:$p"
  else echo "  NOT reachable: $WIN_HOST:$p  -> run fc-host\\windows\\bridge.ps1 on Windows (Admin)"; [ "$p" = 5432 ] && exit 1; fi
done
sysctl -qw net.ipv4.ip_forward=1
iptables -t nat -C POSTROUTING -s 172.16.0.0/16 -o "$UPLINK" -j MASQUERADE 2>/dev/null \
  || iptables -t nat -A POSTROUTING -s 172.16.0.0/16 -o "$UPLINK" -j MASQUERADE
iptables -C FORWARD -i "fc-tap+" -o "$UPLINK" -j ACCEPT 2>/dev/null || iptables -A FORWARD -i "fc-tap+" -o "$UPLINK" -j ACCEPT
iptables -C FORWARD -i "$UPLINK" -o "fc-tap+" -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null \
  || iptables -A FORWARD -i "$UPLINK" -o "fc-tap+" -m state --state RELATED,ESTABLISHED -j ACCEPT

FC_PID=""
cleanup() {
  if [ -n "$FC_PID" ] && kill -0 "$FC_PID" 2>/dev/null; then kill "$FC_PID" 2>/dev/null || true; wait "$FC_PID" 2>/dev/null || true; fi
  FC_PID=""; rm -f "$API_SOCKET"; ip link del "$TAP_DEV" 2>/dev/null || true
}
trap cleanup EXIT

declare -a TOTALS=() KERNELS=() PYS=()

one_run() {
  local run="$1" sid="fc-test-$(date +%s)-$1"
  cleanup
  ip tuntap add dev "$TAP_DEV" mode tap
  ip addr add "$TAP_IP/30" dev "$TAP_DEV"
  ip link set dev "$TAP_DEV" up

  firecracker --api-sock "$API_SOCKET" >"$CONSOLE_LOG" 2>&1 &
  FC_PID=$!
  while [ ! -S "$API_SOCKET" ]; do kill -0 "$FC_PID" 2>/dev/null || { cat "$CONSOLE_LOG"; exit 1; }; sleep 0.005; done

  api PUT /machine-config "{\"vcpu_count\": $VCPUS, \"mem_size_mib\": $MEM_MIB}"
  api PUT /boot-source "{\"kernel_image_path\": \"$KERNEL\", \"boot_args\":
       \"console=ttyS0 reboot=k panic=1 pci=off quiet ip=$GUEST_IP::$TAP_IP:255.255.255.252::eth0:off init=/sbin/fc-init\"}"
  api PUT /drives/rootfs "{\"drive_id\": \"rootfs\", \"path_on_host\": \"$ROOTFS\", \"is_root_device\": true, \"is_read_only\": true}"
  api PUT /network-interfaces/eth0 "{\"iface_id\": \"eth0\", \"guest_mac\": \"$GUEST_MAC\", \"host_dev_name\": \"$TAP_DEV\"}"
  # MMDS: this VM's configuration, served to the guest at 169.254.169.254 (token-protected V2)
  api PUT /mmds/config '{"version": "V2", "network_interfaces": ["eth0"]}'
  api PUT /mmds "$(python3 - "$DSN" "$sid" "$LOG_CONSOLE" <<'PY'
import json, sys
dsn, sid, log_console = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
print(json.dumps({"agent": {
    "env": {"AGENT_DSN": dsn, "PORT": "8080", "SESSION_ID": sid, "LLM_PROVIDER": "mock",
            "MOCK_LATENCY_MS": "150", "LOG_LEVEL": "INFO"},
    "nameserver": "8.8.8.8", "log_to_console": log_console}}))
PY
)"

  local t0 t_port t_ready deadline
  t0="$(now_ms)"
  api PUT /actions '{"action_type": "InstanceStart"}'
  deadline=$(( t0 + 180000 ))
  until port_open "$GUEST_IP" 8080; do
    [ "$(now_ms)" -lt "$deadline" ] || { echo "agent port never opened; console:"; tail -40 "$CONSOLE_LOG"; exit 1; }
    sleep 0.02
  done
  t_port="$(now_ms)"
  local health=""
  until health="$(curl -sf "http://$GUEST_IP:8080/health")"; do
    [ "$(now_ms)" -lt "$deadline" ] || { echo "agent never became ready:"; curl -s "http://$GUEST_IP:8080/health"; echo; tail -20 "$CONSOLE_LOG"; exit 1; }
    sleep 0.05
  done
  t_ready="$(now_ms)"

  local rss_mb init_up fcboot_up mmds_ms
  rss_mb=$(( $(ps -o rss= -p "$FC_PID") / 1024 ))
  init_up="$(grep -oP 'init_start_uptime_s=\K[0-9.]+' "$CONSOLE_LOG" | tail -1 || true)"
  fcboot_up="$(grep -oP 'fc_boot_uptime_s=\K[0-9.]+' "$CONSOLE_LOG" | tail -1 || true)"
  mmds_ms="$(grep -oP 'mmds_ms=\K[0-9.]+' "$CONSOLE_LOG" | tail -1 || true)"

  python3 - "$run" "$t0" "$t_port" "$t_ready" "$health" "${init_up:-0}" "${fcboot_up:-0}" "${mmds_ms:-0}" "$rss_mb" <<'PY'
import json, sys
run, t0, tp, tr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
h = json.loads(sys.argv[5]); t = h.get("timings_ms", {})
init_up, fcb_up, mmds, rss = float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), sys.argv[9]
kernel = init_up * 1000
print(f"run {run}: total {tr - t0:6d} ms = kernel {kernel:5.0f} | init+mmds {(fcb_up - init_up) * 1000 + mmds:5.0f}"
      f" | python imports {t.get('import_ms', 0):6.0f} | postgres {t.get('db_connect_ms', 0):5.0f}"
      f" | graph {t.get('graph_build_ms', 0):4.0f}  (port open at {tp - t0} ms, RSS {rss} MiB)")
PY
  TOTALS+=($(( t_ready - t0 )))
  KERNELS+=("$(python3 -c "print(int(float('${init_up:-0}')*1000))")")
  PYS+=("$(echo "$health" | python3 -c 'import json,sys; print(int(json.load(sys.stdin)["timings_ms"].get("process_to_ready_ms") or 0))')")

  if [ -n "$TASK" ] && [ "$run" -eq 1 ]; then
    local tid="fc-thread-$(date +%s)"
    echo "--- research run inside the microVM: '$TASK'"
    curl -sf -X POST "http://$GUEST_IP:8080/task" -H 'Content-Type: application/json' \
         -d "{\"thread_id\": \"$tid\", \"input\": {\"topic\": \"$TASK\"}}" | python3 -c '
import json, sys
r = json.load(sys.stdin); d = r.get("data") or {}
print("status:", r.get("status"), "| waiting_for:", r.get("waiting_for"), "| run_ms:", r.get("run_ms"),
      "| search_source:", d.get("search_source"))
for p in (d.get("shortlist") or [])[:5]:
    print("  #{} {}  {}".format(p.get("rank"), p.get("score"), str(p.get("title"))[:70]))'
    curl -sf -X POST "http://$GUEST_IP:8080/resume" -H 'Content-Type: application/json' \
         -d "{\"thread_id\": \"$tid\", \"value\": {\"selected\": [1, 2]}}" | python3 -c '
import json, sys
r = json.load(sys.stdin); d = r.get("data") or {}
print("status:", r.get("status"), "| run_ms:", r.get("run_ms"), "| checkpoint_load_ms:", r.get("checkpoint_load_ms"),
      "| summaries:", len(d.get("summaries") or []), "| files:", d.get("workspace_files"))'
    echo "    (thread $tid is checkpointed in Postgres on Windows)"
  fi

  if [ "$KEEP" = "1" ]; then
    echo "--- VM left running: curl http://$GUEST_IP:8080/health  (Ctrl+C to stop)"
    wait "$FC_PID" || true
  fi
  cleanup
}

echo "Firecracker $(firecracker --version 2>&1 | sed -n 1p | awk '{print $2}') | $(basename "$KERNEL") | $VCPUS vCPU, $MEM_MIB MiB | runs: $RUNS"
for i in $(seq 1 "$RUNS"); do one_run "$i"; done

median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END {print (NR%2 ? a[(NR+1)/2] : int((a[NR/2]+a[NR/2+1])/2))}'; }
echo "--- summary (n=$RUNS): total p50 $(median "${TOTALS[@]}") ms | kernel p50 $(median "${KERNELS[@]}") ms | python process->ready p50 $(median "${PYS[@]}") ms"
echo "console log: $CONSOLE_LOG"