#!/usr/bin/env bash
# =============================================================================
# Agentic Cloud - Phase 5, Step 5.1
# Run INSIDE the fc-host VM (Ubuntu, nested KVM). Installs Firecracker and
# downloads a guest kernel + Ubuntu rootfs from Firecracker's public CI bucket.
#
#   ~/fc-host/setup.sh            (as your normal user; it uses sudo where needed)
#
# Result (in $FC_DIR, default ~/fc):
#   /usr/local/bin/firecracker      the VMM binary (one process per microVM)
#   images/vmlinux-<ver>            uncompressed guest kernel
#   images/rootfs.ext4              1 GiB Ubuntu root filesystem (ext4)
#   images/rootfs.id_rsa            SSH key that root@guest accepts
# =============================================================================
set -euo pipefail

FC_DIR="${FC_DIR:-$HOME/fc}"
IMG="$FC_DIR/images"
ARCH="$(uname -m)"
S3="https://s3.amazonaws.com/spec.ccfc.min"
RELEASES="https://github.com/firecracker-microvm/firecracker/releases"

say() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------- 0. checks
say "checking KVM"
if [ ! -e /dev/kvm ]; then
  echo "ERROR: /dev/kvm missing - nested virtualization is not enabled for this VM"; exit 1
fi
if [ ! -r /dev/kvm ] || [ ! -w /dev/kvm ]; then
  echo "NOTE: your user cannot open /dev/kvm yet. Run: sudo usermod -aG kvm \$USER, then log out/in."
  echo "      (hello_vm.sh runs with sudo, so it still works)"
fi
ls -l /dev/kvm

# ---------------------------------------------------------------- 1. packages
say "installing tools (curl, squashfs-tools, e2fsprogs, iproute2, jq, python3)"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  curl wget squashfs-tools e2fsprogs iproute2 iptables jq python3 openssh-client >/dev/null

# ---------------------------------------------------------------- 2. firecracker binary
say "downloading the latest Firecracker release"
LATEST="$(basename "$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$RELEASES/latest")")"
echo "latest release: $LATEST"
TMP="$(mktemp -d)"
curl -fsSL "$RELEASES/download/$LATEST/firecracker-$LATEST-$ARCH.tgz" | tar -xz -C "$TMP"
sudo install -m 0755 "$TMP/release-$LATEST-$ARCH/firecracker-$LATEST-$ARCH" /usr/local/bin/firecracker
rm -rf "$TMP"
# NOTE: no "| head -1" here: head closes the pipe early, firecracker gets SIGPIPE and
# "set -o pipefail" would silently end the whole script. sed reads all input.
firecracker --version 2>&1 | sed -n 1p

# ---------------------------------------------------------------- 3. kernel + rootfs from CI
# CI artifacts live under firecracker-ci/v<major.minor>/<arch>/. The newest release may not
# have artifacts yet, so walk back through minor versions until we find a kernel.
mkdir -p "$IMG"
cd "$IMG"
list_keys() {  # $1 = prefix -> S3 object keys
  curl -fsSL "$S3?list-type=2&prefix=$1" | grep -oP '(?<=<Key>)[^<]+(?=</Key>)' || true
}
MAJOR="$(echo "$LATEST" | sed -E 's/^v([0-9]+)\.([0-9]+).*/\1/')"
MINOR="$(echo "$LATEST" | sed -E 's/^v([0-9]+)\.([0-9]+).*/\2/')"
KERNEL_KEY=""; ROOTFS_KEY=""
for m in $(seq "$MINOR" -1 0); do
  PREFIX="firecracker-ci/v$MAJOR.$m/$ARCH"
  # "|| true": a version without artifacts makes grep find nothing (exit 1); with pipefail
  # that would end the script instead of trying the previous version.
  KERNEL_KEY="$( (list_keys "$PREFIX/vmlinux-" | grep -E "/vmlinux-[0-9]+\.[0-9]+\.[0-9]{1,3}$" | sort -V | tail -1) || true)"
  ROOTFS_KEY="$( (list_keys "$PREFIX/ubuntu-" | grep -E "/ubuntu-[0-9]+\.[0-9]+\.squashfs$" | sort -V | tail -1) || true)"
  echo "  v$MAJOR.$m: kernel=${KERNEL_KEY:-none} rootfs=${ROOTFS_KEY:-none}"
  if [ -n "$KERNEL_KEY" ] && [ -n "$ROOTFS_KEY" ]; then
    echo "using CI artifacts of v$MAJOR.$m"; break
  fi
done
if [ -z "$KERNEL_KEY" ] || [ -z "$ROOTFS_KEY" ]; then
  echo "ERROR: could not find CI kernel/rootfs in $S3"; exit 1
fi

say "kernel: $KERNEL_KEY"
[ -f "$(basename "$KERNEL_KEY")" ] || wget -q --show-progress "$S3/$KERNEL_KEY"

say "rootfs: $ROOTFS_KEY"
[ -f rootfs.squashfs ] || wget -q --show-progress -O rootfs.squashfs "$S3/$ROOTFS_KEY"

# ---------------------------------------------------------------- 4. squashfs -> ext4 with our SSH key
# The CI rootfs is read-only squashfs without keys. We unpack it, add a fresh SSH key for root,
# and pack it into a writable 1 GiB ext4 image.
if [ ! -f rootfs.ext4 ]; then
  say "building rootfs.ext4 (1 GiB) with a new SSH key"
  sudo rm -rf squashfs-root
  sudo unsquashfs -q -d squashfs-root rootfs.squashfs
  rm -f rootfs.id_rsa rootfs.id_rsa.pub
  ssh-keygen -q -t ed25519 -f rootfs.id_rsa -N ""
  sudo mkdir -p squashfs-root/root/.ssh
  sudo cp rootfs.id_rsa.pub squashfs-root/root/.ssh/authorized_keys
  sudo chmod 700 squashfs-root/root/.ssh
  sudo chmod 600 squashfs-root/root/.ssh/authorized_keys
  sudo chown -R root:root squashfs-root
  truncate -s 1G rootfs.ext4
  sudo mkfs.ext4 -q -d squashfs-root -F rootfs.ext4
  sudo rm -rf squashfs-root
  chmod 400 rootfs.id_rsa
fi

say "done"
ls -lh "$IMG"
echo
echo "next:  sudo ~/fc-host/hello_vm.sh"