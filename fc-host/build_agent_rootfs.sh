#!/usr/bin/env bash
# =============================================================================
# Agentic Cloud - Phase 5, Step 5.2 (+5.3: read-only root, /etc/resolv.conf + hosts -> /run)
# Build the agent microVM's root filesystem from the exported Docker image.
#
#   ~/fc/images/agent-rootfs.tar   (from fc-host/windows/export_agent_rootfs.ps1)
#     + /sbin/fc-init              (PID 1, replaces systemd)
#     + /opt/fc/fc_boot.py         (reads MMDS config, starts the agent as uid 10001)
#   -> ~/fc/images/agent-rootfs.ext4
#
# Usage (inside fc-host, as your user; uses sudo):  ~/fc-host/build_agent_rootfs.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FC_DIR="${FC_DIR:-$HOME/fc}"
IMG="$FC_DIR/images"
TAR="$IMG/agent-rootfs.tar"
OUT="$IMG/agent-rootfs.ext4"
WORK="$(mktemp -d)"
EXTRA_MB="${EXTRA_MB:-128}"     # free space inside the image (read-only at runtime; writes go to tmpfs)

[ -f "$TAR" ] || { echo "missing $TAR - run export_agent_rootfs.ps1 on Windows first"; exit 1; }
for f in "$HERE/guest/fc-init" "$HERE/guest/fc_boot.py"; do
  [ -f "$f" ] || { echo "missing $f (copy the whole fc-host folder)"; exit 1; }
done

echo "==> unpacking $(du -h "$TAR" | cut -f1) tarball"
sudo tar -xf "$TAR" -C "$WORK"

echo "==> adding the microVM boot files"
sudo install -m 0755 "$HERE/guest/fc-init" "$WORK/sbin/fc-init"
sudo sed -i 's/\r$//' "$WORK/sbin/fc-init"                 # in case Windows added CR line endings
sudo mkdir -p "$WORK/opt/fc" "$WORK/proc" "$WORK/sys" "$WORK/dev" "$WORK/run" "$WORK/tmp"
sudo install -m 0644 "$HERE/guest/fc_boot.py" "$WORK/opt/fc/fc_boot.py"
sudo chmod 1777 "$WORK/tmp"
# Phase 5.3: the root disk is mounted READ-ONLY (one file shared by every snapshot clone).
# Files written at boot therefore live in /run (tmpfs); /etc points there.
sudo rm -f "$WORK/.dockerenv" "$WORK/etc/resolv.conf" "$WORK/etc/hosts"
sudo ln -s /run/resolv.conf "$WORK/etc/resolv.conf"
sudo ln -s /run/hosts "$WORK/etc/hosts"
sudo mkdir -p "$WORK/home/agent" "$WORK/workspace"

# sanity: the things fc-init / fc_boot.py rely on
for p in /bin/sh /usr/local/bin/python /app/server.py /bin/mount; do
  sudo test -e "$WORK$p" || { echo "ERROR: $p not found in the image"; exit 1; }
done
sudo test -d "$WORK/workspace" || { echo "ERROR: /workspace missing"; exit 1; }

USED_MB=$(sudo du -sm "$WORK" | cut -f1)
SIZE_MB=$(( USED_MB + EXTRA_MB ))
echo "==> creating ext4 image: ${USED_MB} MB used + ${EXTRA_MB} MB free = ${SIZE_MB} MB"
rm -f "$OUT"
truncate -s "${SIZE_MB}M" "$OUT"
sudo mkfs.ext4 -q -F -d "$WORK" -L agentroot "$OUT"
sudo rm -rf "$WORK"

ls -lh "$OUT"
echo "next:  sudo ~/fc-host/agent_vm.sh"