#!/usr/bin/env bash
# Agentic Cloud - Phase 5, Step 5.3: helpers shared by make_golden.sh and restore_vm.sh
# (source it; expects to run as root)

USER_HOME="$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)"
FC_DIR="${FC_DIR:-$USER_HOME/fc}"
IMG="$FC_DIR/images"
RUN="$FC_DIR/run"
GOLDEN_DIR="$FC_DIR/golden"
mkdir -p "$RUN" "$GOLDEN_DIR"

TAP_DEV="fc-tap0"; TAP_IP="172.16.0.1"; GUEST_IP="172.16.0.2"; GUEST_MAC="06:00:AC:10:00:02"

# Windows = fc-host's default gateway on the Hyper-V Default Switch
WIN_HOST="${WIN_HOST:-$(ip -4 route show default | awk '{print $3; exit}')}"
UPLINK="$(ip -4 route show default | awk '{print $5; exit}')"
DSN="postgresql://agentic:agentic_dev_pw@$WIN_HOST:5432/agentic_cloud"

# Monotonic clock in ms. NOT "date": the fc-host VM's wall clock is stepped by Hyper-V time
# sync (hv_utils), which produced negative stage times. CLOCK_MONOTONIC never jumps.
now_ms() { perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC -e 'printf "%d\n", clock_gettime(CLOCK_MONOTONIC)*1000'; }
port_open() { timeout 0.5 bash -c "</dev/tcp/$1/$2" 2>/dev/null; }

# fc_api <socket> <METHOD> <path> <json>
fc_api() {
  local out code
  out="$(curl -sS --unix-socket "$1" -X "$2" "http://localhost$3" \
         -H 'Content-Type: application/json' -d "$4" -w '\n%{http_code}')"
  code="${out##*$'\n'}"
  [ "${code:0:1}" = "2" ] || { echo "API $2 $3 failed ($code): ${out%$'\n'*}" >&2; return 1; }
}

# NAT so microVMs reach Windows (Postgres, control plane); idempotent
setup_nat() {
  sysctl -qw net.ipv4.ip_forward=1
  iptables -t nat -C POSTROUTING -s 172.16.0.0/16 -o "$UPLINK" -j MASQUERADE 2>/dev/null \
    || iptables -t nat -A POSTROUTING -s 172.16.0.0/16 -o "$UPLINK" -j MASQUERADE
  iptables -C FORWARD -i "fc-tap+" -o "$UPLINK" -j ACCEPT 2>/dev/null \
    || iptables -A FORWARD -i "fc-tap+" -o "$UPLINK" -j ACCEPT
  iptables -C FORWARD -i "$UPLINK" -o "fc-tap+" -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null \
    || iptables -A FORWARD -i "$UPLINK" -o "fc-tap+" -m state --state RELATED,ESTABLISHED -j ACCEPT
}

# make_tap [netns]   - the host end of the guest's virtual cable (172.16.0.1/30)
make_tap() {
  local ns="${1:-}" pre=()
  [ -n "$ns" ] && pre=(ip netns exec "$ns")
  "${pre[@]}" ip link del "$TAP_DEV" 2>/dev/null || true
  "${pre[@]}" ip tuntap add dev "$TAP_DEV" mode tap
  "${pre[@]}" ip addr add "$TAP_IP/30" dev "$TAP_DEV"
  "${pre[@]}" ip link set dev "$TAP_DEV" up
  [ -n "$ns" ] && "${pre[@]}" ip link set lo up
  return 0
}

# start_fc <socket> <console_log> [netns]  -> sets FC_PID
start_fc() {
  local sock="$1" log="$2" ns="${3:-}" pre=()
  [ -n "$ns" ] && pre=(ip netns exec "$ns")
  rm -f "$sock"
  "${pre[@]}" firecracker --api-sock "$sock" >"$log" 2>&1 &
  FC_PID=$!
  while [ ! -S "$sock" ]; do
    kill -0 "$FC_PID" 2>/dev/null || { echo "firecracker exited:"; cat "$log"; return 1; }
    sleep 0.002
  done
}

stop_pid() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null && { kill "$1" 2>/dev/null; wait "$1" 2>/dev/null || true; }; return 0; }

# memory of one firecracker process (MiB): "rss pss private shared"
mem_stats() {
  awk '/^Rss:/{r=$2} /^Pss:/{p=$2} /^Private_Clean:/{pc=$2} /^Private_Dirty:/{pd=$2}
       /^Shared_Clean:/{sc=$2} /^Shared_Dirty:/{sd=$2}
       END {printf "%.0f %.0f %.0f %.0f\n", r/1024, p/1024, (pc+pd)/1024, (sc+sd)/1024}' "/proc/$1/smaps_rollup"
}

median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END {print (NR%2 ? a[(NR+1)/2] : int((a[NR/2]+a[NR/2+1])/2))}'; }
pct95() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END {i=int(0.95*(NR-1)+0.5)+1; print a[i]}'; }