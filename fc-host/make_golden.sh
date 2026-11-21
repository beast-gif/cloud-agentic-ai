#!/usr/bin/env bash
# =============================================================================
# Agentic Cloud - Phase 5, Step 5.3
# Create golden.snap: boot the agent in PRE-BIND mode (imports done, gc.freeze(), no session,
# no DB connection), pause the VM and save its complete state:
#   golden.mem      guest RAM, byte for byte (as large as the VM's memory, e.g. 512 MiB)
#   golden.vmstate  vCPU registers + device state (a few KB)
#   rootfs.ext4     frozen copy of the read-only root disk the snapshot was taken with
#
#   sudo ~/fc-host/make_golden.sh                    # -> ~/fc/golden/
#   sudo GC_FREEZE=0 NAME=golden-nofreeze ~/fc-host/make_golden.sh   # ablation: no gc.freeze()
#
# Re-run it whenever the agent image / rootfs changes (old snapshots keep the old code).
# =============================================================================
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
source "$(cd "$(dirname "$0")" && pwd)/lib.sh"

NAME="${NAME:-golden}"
GC_FREEZE="${GC_FREEZE:-1}"
VCPUS="${VCPUS:-1}"; MEM_MIB="${MEM_MIB:-512}"
OUT="$GOLDEN_DIR/$NAME"
SOCK="$RUN/golden.sock"; LOG="$RUN/golden.console.log"
KERNEL="$(ls "$IMG"/vmlinux-* 2>/dev/null | sort -V | tail -1 || true)"
[ -f "$IMG/agent-rootfs.ext4" ] && [ -n "$KERNEL" ] || { echo "run setup.sh + build_agent_rootfs.sh first"; exit 1; }

FC_PID=""
cleanup() { stop_pid "$FC_PID"; rm -f "$SOCK"; ip link del "$TAP_DEV" 2>/dev/null || true; }
trap cleanup EXIT

echo "==> preparing $OUT"
rm -rf "$OUT"; mkdir -p "$OUT"
# the snapshot records this exact disk path; a private read-only copy keeps it valid even if
# agent-rootfs.ext4 is rebuilt later
cp --sparse=always "$IMG/agent-rootfs.ext4" "$OUT/rootfs.ext4"
chmod 0444 "$OUT/rootfs.ext4"

setup_nat
make_tap
start_fc "$SOCK" "$LOG"

fc_api "$SOCK" PUT /machine-config "{\"vcpu_count\": $VCPUS, \"mem_size_mib\": $MEM_MIB}"
fc_api "$SOCK" PUT /boot-source "{\"kernel_image_path\": \"$KERNEL\", \"boot_args\":
     \"console=ttyS0 reboot=k panic=1 pci=off quiet ip=$GUEST_IP::$TAP_IP:255.255.255.252::eth0:off init=/sbin/fc-init\"}"
fc_api "$SOCK" PUT /drives/rootfs "{\"drive_id\": \"rootfs\", \"path_on_host\": \"$OUT/rootfs.ext4\", \"is_root_device\": true, \"is_read_only\": true}"
fc_api "$SOCK" PUT /network-interfaces/eth0 "{\"iface_id\": \"eth0\", \"guest_mac\": \"$GUEST_MAC\", \"host_dev_name\": \"$TAP_DEV\"}"
fc_api "$SOCK" PUT /mmds/config '{"version": "V2", "network_interfaces": ["eth0"]}'
# NO session data here: no DSN, no token, no session id -> safe to share with everyone
fc_api "$SOCK" PUT /mmds "{\"agent\": {\"env\": {\"FC_PREBIND\": \"1\", \"FC_GC_FREEZE\": \"$GC_FREEZE\",
     \"PORT\": \"8080\", \"LOG_LEVEL\": \"INFO\", \"SESSION_ID\": \"prebind\"}, \"nameserver\": \"8.8.8.8\"}}"

T0="$(now_ms)"
fc_api "$SOCK" PUT /actions '{"action_type": "InstanceStart"}'
echo "==> booting in pre-bind mode ..."
HEALTH=""
until HEALTH="$(curl -sf "http://$GUEST_IP:8080/health" 2>/dev/null)" && echo "$HEALTH" | grep -q '"prebind"'; do
  [ $(( $(now_ms) - T0 )) -lt 180000 ] || { echo "agent did not reach pre-bind; console:"; tail -30 "$LOG"; exit 1; }
  sleep 0.05
done
T1="$(now_ms)"
echo "    pre-bind reached after $((T1 - T0)) ms: $HEALTH"
sleep 1    # let the guest go idle (no half-finished work in the snapshot)

echo "==> pausing and writing the snapshot"
T2="$(now_ms)"
fc_api "$SOCK" PATCH /vm '{"state": "Paused"}'
T3="$(now_ms)"
fc_api "$SOCK" PUT /snapshot/create "{\"snapshot_type\": \"Full\",
     \"snapshot_path\": \"$OUT/golden.vmstate\", \"mem_file_path\": \"$OUT/golden.mem\"}"
T4="$(now_ms)"
cleanup
chmod 0444 "$OUT/golden.mem" "$OUT/golden.vmstate"

python3 - "$OUT" "$NAME" "$GC_FREEZE" "$KERNEL" "$VCPUS" "$MEM_MIB" "$((T1 - T0))" "$((T3 - T2))" "$((T4 - T3))" "$HEALTH" <<'PY'
import json, os, sys, time
out, name, gcf, kernel, vcpus, mem, boot, pause, snap, health = sys.argv[1:11]
meta = {"name": name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "gc_freeze": gcf == "1",
        "kernel": os.path.basename(kernel), "vcpus": int(vcpus), "mem_mib": int(mem),
        "boot_to_prebind_ms": int(boot), "pause_ms": int(pause), "snapshot_create_ms": int(snap),
        "agent": json.loads(health).get("timings_ms", {})}
for f in ("golden.mem", "golden.vmstate", "rootfs.ext4"):
    st = os.stat(os.path.join(out, f))
    meta[f] = {"size_mib": round(st.st_size / 2**20, 1), "on_disk_mib": round(st.st_blocks * 512 / 2**20, 1)}
json.dump(meta, open(os.path.join(out, "golden.json"), "w"), indent=2)
print(json.dumps(meta, indent=2))
PY
echo "next:  sudo ~/fc-host/restore_vm.sh"