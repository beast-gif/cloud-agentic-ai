#!/usr/bin/env bash
# =============================================================================
# Agentic Cloud - Phase 5, Step 5.1
# Boot ONE Firecracker microVM from scratch (cold boot) and measure every stage.
#
#   sudo ~/fc-host/hello_vm.sh                 # 1 vCPU, 512 MiB
#   sudo VCPUS=2 MEM_MIB=1024 ~/fc-host/hello_vm.sh
#   sudo RUNS=5 ~/fc-host/hello_vm.sh          # repeat 5 times -> p50
#
# Stages timed (ms):
#   spawn      start the firecracker process until its API socket exists
#   configure  4 API calls: machine-config, boot-source, rootfs drive, network
#   boot       InstanceStart -> guest's SSH port (22) accepts a TCP connection
#   ssh        InstanceStart -> first command executed over SSH ("true")
# The guest is then shut down and everything (tap device, socket) cleaned up.
#
# Networking: a TAP device is a virtual Ethernet cable; one end is fc-tap0 on this
# host (172.16.0.1/30), the other end is eth0 inside the guest (172.16.0.2).
# The CI rootfs derives its IP from the MAC 06:00:AC:10:00:02 -> 172.16.0.2.
# =============================================================================
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then echo "run with sudo"; exit 1; fi

USER_HOME="$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)"
FC_DIR="${FC_DIR:-$USER_HOME/fc}"
IMG="$FC_DIR/images"
RUN="$FC_DIR/run"
mkdir -p "$RUN"

VCPUS="${VCPUS:-1}"
MEM_MIB="${MEM_MIB:-512}"
RUNS="${RUNS:-1}"
TAP_DEV="fc-tap0"
TAP_IP="172.16.0.1"
GUEST_IP="172.16.0.2"
GUEST_MAC="06:00:AC:10:00:02"
API_SOCKET="$RUN/hello.sock"
CONSOLE_LOG="$RUN/hello.console.log"

KERNEL="$(ls "$IMG"/vmlinux-* 2>/dev/null | sort -V | tail -1 || true)"
ROOTFS="$IMG/rootfs.ext4"
KEY="$IMG/rootfs.id_rsa"
for f in "$KERNEL" "$ROOTFS" "$KEY"; do
  [ -n "$f" ] && [ -f "$f" ] || { echo "missing $f - run ~/fc-host/setup.sh first"; exit 1; }
done

SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -o LogLevel=ERROR -o ConnectTimeout=1 "root@$GUEST_IP")

# Monotonic clock in ms. NOT "date": the fc-host VM's wall clock is stepped by Hyper-V time
# sync (hv_utils), which produced negative stage times. CLOCK_MONOTONIC never jumps.
now_ms() { perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC -e 'printf "%d\n", clock_gettime(CLOCK_MONOTONIC)*1000'; }
api() {  # api PUT /path '{json}'
  local out code
  out="$(curl -sS --unix-socket "$API_SOCKET" -X "$1" "http://localhost$2" \
         -H 'Content-Type: application/json' -d "$3" -w '\n%{http_code}')"
  code="${out##*$'\n'}"
  if [ "${code:0:1}" != "2" ]; then echo "API $1 $2 failed ($code): ${out%$'\n'*}"; return 1; fi
}

FC_PID=""
cleanup() {
  if [ -n "$FC_PID" ] && kill -0 "$FC_PID" 2>/dev/null; then kill "$FC_PID" 2>/dev/null || true; wait "$FC_PID" 2>/dev/null || true; fi
  FC_PID=""
  rm -f "$API_SOCKET"
  ip link del "$TAP_DEV" 2>/dev/null || true
}
trap cleanup EXIT

one_run() {
  cleanup
  # --- network: tap device on the host side
  ip tuntap add dev "$TAP_DEV" mode tap
  ip addr add "$TAP_IP/30" dev "$TAP_DEV"
  ip link set dev "$TAP_DEV" up

  # --- spawn the VMM process
  local t0 t1 t2 t3 t4 t5
  t0="$(now_ms)"
  firecracker --api-sock "$API_SOCKET" >"$CONSOLE_LOG" 2>&1 &
  FC_PID=$!
  while [ ! -S "$API_SOCKET" ]; do
    kill -0 "$FC_PID" 2>/dev/null || { echo "firecracker exited:"; cat "$CONSOLE_LOG"; exit 1; }
    sleep 0.005
  done
  t1="$(now_ms)"

  # --- configure the microVM (nothing runs yet)
  api PUT /machine-config "{\"vcpu_count\": $VCPUS, \"mem_size_mib\": $MEM_MIB}"
  api PUT /boot-source "{\"kernel_image_path\": \"$KERNEL\",
        \"boot_args\": \"console=ttyS0 reboot=k panic=1 pci=off\"}"
  api PUT /drives/rootfs "{\"drive_id\": \"rootfs\", \"path_on_host\": \"$ROOTFS\",
        \"is_root_device\": true, \"is_read_only\": false}"
  api PUT /network-interfaces/eth0 "{\"iface_id\": \"eth0\", \"guest_mac\": \"$GUEST_MAC\",
        \"host_dev_name\": \"$TAP_DEV\"}"
  t2="$(now_ms)"

  # --- boot
  api PUT /actions '{"action_type": "InstanceStart"}'
  t3="$(now_ms)"
  local deadline=$(( t3 + 120000 ))
  until timeout 0.3 bash -c "</dev/tcp/$GUEST_IP/22" 2>/dev/null; do
    [ "$(now_ms)" -lt "$deadline" ] || { echo "guest SSH port never opened; console:"; tail -30 "$CONSOLE_LOG"; exit 1; }
    sleep 0.02
  done
  t4="$(now_ms)"
  until "${SSH[@]}" true 2>/dev/null; do
    [ "$(now_ms)" -lt "$deadline" ] || { echo "SSH login failed"; exit 1; }
    sleep 0.05
  done
  t5="$(now_ms)"

  SPAWN=$((t1 - t0)); CONFIG=$((t2 - t1)); BOOT=$((t4 - t3)); SSHMS=$((t5 - t3))
  printf "run %d: spawn %5d ms | configure %4d ms | boot (port 22) %6d ms | ssh ready %6d ms\n" \
         "$1" "$SPAWN" "$CONFIG" "$BOOT" "$SSHMS"
  BOOTS+=("$BOOT"); SSHS+=("$SSHMS")

  if [ "$1" -eq 1 ]; then
    echo "--- inside the guest:"
    "${SSH[@]}" 'echo "kernel: $(uname -r)"; echo "vcpus: $(nproc)";
                 free -m | awk "/Mem:/ {print \"memory: \" \$2 \" MiB total, \" \$3 \" MiB used\"}";
                 . /etc/os-release; echo "os: $PRETTY_NAME"; echo "ip: $(hostname -I)"'
    echo "--- firecracker process on the host (resident memory):"
    ps -o pid,rss,vsz,cmd -p "$FC_PID" | awk 'NR==1 {print} NR>1 {printf "%s  RSS=%.0f MiB  VSZ=%.0f MiB  %s\n", $1, $2/1024, $3/1024, $4}'
  fi

  # --- shut down: reboot=k makes the guest exit, which ends the firecracker process
  "${SSH[@]}" 'reboot' 2>/dev/null || true
  for _ in $(seq 1 100); do kill -0 "$FC_PID" 2>/dev/null || break; sleep 0.05; done
  cleanup
}

BOOTS=(); SSHS=()
echo "Firecracker $(firecracker --version 2>&1 | sed -n 1p | awk '{print $2}') | kernel $(basename "$KERNEL") | $VCPUS vCPU, $MEM_MIB MiB | runs: $RUNS"
for i in $(seq 1 "$RUNS"); do one_run "$i"; done

median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END {print (NR%2 ? a[(NR+1)/2] : int((a[NR/2]+a[NR/2+1])/2))}'; }
echo "--- summary: boot p50 $(median "${BOOTS[@]}") ms, ssh-ready p50 $(median "${SSHS[@]}") ms (n=$RUNS)"
echo "console log of the last run: $CONSOLE_LOG"