#!/usr/bin/env python3
"""
Agentic Cloud - Phase 5, Step 5.4
fc-agent: the small service inside the fc-host VM that the control plane (on Windows) calls to
run agents as Firecracker microVMs restored from golden.snap. Standard library only; runs as root.

    sudo FC_AGENT_SECRET=<secret> python3 ~/fc-host/fc_agent.py

API (every request needs header  X-FC-Secret: <secret>):
  GET    /health              golden snapshot info, Windows address, slots in use
  POST   /vms                 {"session_id": "...", "env": {...}}  -> restore + clock + bind
                              -> {"vm_id", "agent_url", "timings_ms": {...}, "agent": {...}}
  GET    /vms                 all microVMs (alive or exited)
  GET    /vms/<vm_id>         one microVM
  DELETE /vms/<vm_id>         stop it (the session state is already in Postgres)
  GET    /vms/<vm_id>/console?tail=40

Networking - one SLOT per running microVM, prepared once and reused:
  Every clone is a copy of the same VM, so every guest has IP 172.16.0.2 on a tap "fc-tap0".
  Each slot i is a network namespace fc<i> with its own fc-tap0, connected to fc-host by a
  veth pair 10.200.<i>.1 (fc-host side) <-> 10.200.<i>.2 (namespace side).

  control plane -> agent:  Windows -> fc-host eth0:(PORT_BASE+i) --DNAT--> 10.200.i.2:8080
                           --DNAT inside fc<i>--> 172.16.0.2:8080
  agent -> Windows:        172.16.0.2 --MASQUERADE in fc<i>--> 10.200.i.2
                           --MASQUERADE on fc-host--> eth0 -> Windows (Postgres / control plane)
  "{WIN_HOST}" inside env values is replaced with the Windows address (fc-host's default gateway).
"""
from __future__ import annotations

import hmac
import http.client
import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

SECRET = os.environ.get("FC_AGENT_SECRET", "")
LISTEN_PORT = int(os.environ.get("FC_AGENT_PORT", "8090"))
SLOTS = int(os.environ.get("FC_SLOTS", "8"))
PORT_BASE = int(os.environ.get("FC_PORT_BASE", "20000"))
USER_HOME = Path(os.path.expanduser(f"~{os.environ.get('SUDO_USER', 'root')}"))
FC_DIR = Path(os.environ.get("FC_DIR", str(USER_HOME / "fc")))
GOLDEN_DIR = FC_DIR / "golden"
RUN = FC_DIR / "run"
GUEST_IP = "172.16.0.2"


def sh(cmd: str, ok_fail: bool = False) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0 and not ok_fail:
        raise RuntimeError(f"`{cmd}` failed: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def mono_ms() -> float:
    return time.monotonic() * 1000.0


def default_route() -> tuple[str, str]:
    parts = sh("ip -4 route show default").split()
    return parts[parts.index("via") + 1], parts[parts.index("dev") + 1]


WIN_HOST, UPLINK = default_route()
FCHOST_IP = sh(f"ip -4 -o addr show dev {UPLINK}").split()[3].split("/")[0]


# =============================================================== Firecracker API over a unix socket

class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float = 30.0):
        super().__init__("localhost", timeout=timeout)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def fc_api(sock: str, method: str, path: str, body: dict, timeout: float = 60.0) -> None:
    conn = UnixHTTPConnection(sock, timeout)
    conn.request(method, path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read().decode()
    conn.close()
    if resp.status >= 300:
        raise RuntimeError(f"firecracker {method} {path} -> {resp.status}: {data[:300]}")


def http_json(url: str, body: Optional[dict] = None, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


# =============================================================== network slots

def setup_host_network() -> None:
    sh("sysctl -qw net.ipv4.ip_forward=1")
    rules = [
        f"-t nat -A POSTROUTING -s 10.200.0.0/16 -o {UPLINK} -j MASQUERADE",
        "-A FORWARD -i fcv+ -j ACCEPT",
        "-A FORWARD -o fcv+ -j ACCEPT",
    ]
    for r in rules:
        if subprocess.run(f"iptables {r.replace('-A ', '-C ', 1)}", shell=True,
                          capture_output=True).returncode != 0:
            sh(f"iptables {r}")


def setup_slot(i: int) -> None:
    """(Re)create namespace fc<i>: veth to fc-host, fc-tap0 for the guest, NAT rules."""
    ns, hv, nv = f"fc{i}", f"fcv{i}h", f"fcv{i}n"
    hip, nip = f"10.200.{i}.1", f"10.200.{i}.2"
    sh(f"ip netns del {ns}", ok_fail=True)
    sh(f"ip link del {hv}", ok_fail=True)
    sh(f"ip netns add {ns}")
    sh(f"ip link add {hv} type veth peer name {nv}")
    sh(f"ip link set {nv} netns {ns}")
    sh(f"ip addr add {hip}/30 dev {hv}")
    sh(f"ip link set {hv} up")
    x = f"ip netns exec {ns} "
    sh(x + "ip link set lo up")
    sh(x + f"ip addr add {nip}/30 dev {nv}")
    sh(x + f"ip link set {nv} up")
    sh(x + "ip tuntap add dev fc-tap0 mode tap")
    sh(x + "ip addr add 172.16.0.1/30 dev fc-tap0")
    sh(x + "ip link set fc-tap0 up")
    sh(x + f"ip route add default via {hip}")
    sh(x + "sysctl -qw net.ipv4.ip_forward=1")
    for port in (8080, 8081):   # agent API, clock helper
        sh(x + f"iptables -t nat -A PREROUTING -i {nv} -p tcp --dport {port} -j DNAT --to-destination {GUEST_IP}:{port}")
    sh(x + f"iptables -t nat -A POSTROUTING -s 172.16.0.0/30 -o {nv} -j MASQUERADE")
    # Everything sent INTO the guest appears to come from 172.16.0.1 (the guest's own gateway):
    # simple return routing, and the clock helper only accepts requests from that address.
    sh(x + "iptables -t nat -A POSTROUTING -o fc-tap0 -j MASQUERADE")
    # Windows -> fc-host:PORT_BASE+i -> this slot's agent
    rule = f"PREROUTING -i {UPLINK} -p tcp --dport {PORT_BASE + i} -j DNAT --to-destination {nip}:8080"
    if subprocess.run(f"iptables -t nat -C {rule}", shell=True, capture_output=True).returncode != 0:
        sh(f"iptables -t nat -A {rule}")


# =============================================================== microVM registry

class VM:
    def __init__(self, vm_id: str, slot: int, session_id: str, golden: str) -> None:
        self.vm_id, self.slot, self.session_id, self.golden = vm_id, slot, session_id, golden
        self.sock = str(RUN / f"{vm_id}.sock")
        self.console = str(RUN / f"{vm_id}.console.log")
        self.proc: Optional[subprocess.Popen] = None
        self.created = time.time()
        self.timings: dict = {}
        self.stopped = False

    @property
    def agent_url(self) -> str:
        return f"http://{FCHOST_IP}:{PORT_BASE + self.slot}"

    @property
    def internal_ip(self) -> str:
        return f"10.200.{self.slot}.2"

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def info(self) -> dict:
        rss = None
        if self.alive():
            try:
                for line in open(f"/proc/{self.proc.pid}/status"):
                    if line.startswith("VmRSS:"):
                        rss = round(int(line.split()[1]) / 1024)
            except OSError:
                pass
        return {"vm_id": self.vm_id, "session_id": self.session_id, "slot": self.slot,
                "golden": self.golden, "agent_url": self.agent_url,
                "status": "running" if self.alive() else ("stopped" if self.stopped else "exited"),
                "pid": self.proc.pid if self.proc else None, "rss_mib": rss,
                "created": self.created, "timings_ms": self.timings}


VMS: dict[str, VM] = {}
FREE_SLOTS = list(range(1, SLOTS + 1))
LOCK = threading.Lock()


def console_tail(vm: VM, n: int = 40) -> str:
    try:
        return "".join(open(vm.console, errors="replace").readlines()[-n:])
    except OSError:
        return ""


def substitute(env: dict) -> dict:
    return {str(k): str(v).replace("{WIN_HOST}", WIN_HOST).replace("{FCHOST}", FCHOST_IP)
            for k, v in env.items()}


def create_vm(session_id: str, env: dict, golden: str) -> VM:
    snap = GOLDEN_DIR / golden
    if not (snap / "golden.mem").exists():
        raise FileNotFoundError(f"golden snapshot {snap} not found (run make_golden.sh)")
    with LOCK:
        if not FREE_SLOTS:
            raise RuntimeError(f"all {SLOTS} slots in use")
        slot = FREE_SLOTS.pop(0)
    vm = VM(f"vm{slot}-{uuid.uuid4().hex[:8]}", slot, session_id, golden)
    try:
        t0 = mono_ms()
        Path(vm.sock).unlink(missing_ok=True)
        log = open(vm.console, "w")
        vm.proc = subprocess.Popen(["ip", "netns", "exec", f"fc{slot}", "firecracker", "--api-sock", vm.sock],
                                   stdout=log, stderr=subprocess.STDOUT)
        while not os.path.exists(vm.sock):
            if vm.proc.poll() is not None:
                raise RuntimeError(f"firecracker exited: {console_tail(vm)}")
            time.sleep(0.002)
        t1 = mono_ms()
        fc_api(vm.sock, "PUT", "/snapshot/load", {
            "snapshot_path": str(snap / "golden.vmstate"),
            "mem_backend": {"backend_type": "File", "backend_path": str(snap / "golden.mem")},
            "resume_vm": True})
        t2 = mono_ms()
        deadline = time.monotonic() + 15
        while True:   # first round-trip into the restored guest: fix its wall clock
            try:
                clock = http_json(f"http://{vm.internal_ip}:8081/clock", {"epoch_ns": time.time_ns()}, timeout=3)
                break
            except (urllib.error.URLError, OSError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.02)
        t3 = mono_ms()
        env = substitute({**env, "SESSION_ID": session_id})
        bound = http_json(f"http://{vm.internal_ip}:8080/bind", {"env": env}, timeout=60)
        t4 = mono_ms()
        vm.timings = {"spawn_ms": round(t1 - t0, 1), "load_resume_ms": round(t2 - t1, 1),
                      "clock_ms": round(t3 - t2, 1), "bind_ms": round(t4 - t3, 1),
                      "total_ms": round(t4 - t0, 1),
                      "clock_corrected_s": round((clock.get("corrected_by_ms") or 0) / 1000, 1),
                      "agent": bound.get("timings_ms", {})}
        with LOCK:
            VMS[vm.vm_id] = vm
        return vm
    except Exception:
        stop_vm(vm)
        raise


def stop_vm(vm: VM) -> float:
    t0 = mono_ms()
    if vm.proc is not None and vm.proc.poll() is None:
        vm.proc.terminate()
        try:
            vm.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            vm.proc.kill()
            vm.proc.wait(timeout=2)
    vm.stopped = True
    Path(vm.sock).unlink(missing_ok=True)
    with LOCK:
        if vm.slot not in FREE_SLOTS:
            FREE_SLOTS.append(vm.slot)
            FREE_SLOTS.sort()
    return round(mono_ms() - t0, 1)


def reaper() -> None:
    """Free the slot of a microVM whose firecracker process died on its own."""
    while True:
        time.sleep(5)
        with LOCK:
            dead = [vm for vm in VMS.values() if not vm.alive() and not vm.stopped]
        for vm in dead:
            stop_vm(vm)


# =============================================================== HTTP API

class Handler(BaseHTTPRequestHandler):
    server_version = "fc-agent/1.0"

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _auth(self) -> bool:
        if hmac.compare_digest(self.headers.get("X-FC-Secret", ""), SECRET):
            return True
        self._send(401, {"error": "missing or wrong X-FC-Secret"})
        return False

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _vm(self, vm_id: str) -> Optional[VM]:
        with LOCK:
            vm = VMS.get(vm_id)
        if vm is None:
            self._send(404, {"error": f"vm {vm_id} not found"})
        return vm

    def do_GET(self):  # noqa: N802
        if not self._auth():
            return
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        if parts == ["health"]:
            goldens = {}
            for d in sorted(GOLDEN_DIR.glob("*/golden.json")):
                m = json.loads(d.read_text())
                goldens[d.parent.name] = {k: m.get(k) for k in ("created", "gc_freeze", "mem_mib", "agent")}
            with LOCK:
                running = sum(vm.alive() for vm in VMS.values())
                free = len(FREE_SLOTS)
            return self._send(200, {"status": "ok", "windows_host": WIN_HOST, "fc_host": FCHOST_IP,
                                    "slots": SLOTS, "free_slots": free, "running": running,
                                    "goldens": goldens})
        if parts == ["vms"]:
            with LOCK:
                vms = list(VMS.values())
            return self._send(200, {"vms": [vm.info() for vm in vms]})
        if len(parts) == 2 and parts[0] == "vms":
            vm = self._vm(parts[1])
            return vm and self._send(200, vm.info())
        if len(parts) == 3 and parts[0] == "vms" and parts[2] == "console":
            vm = self._vm(parts[1])
            tail = int((self.path.split("tail=")[1:] or ["40"])[0].split("&")[0])
            return vm and self._send(200, {"vm_id": vm.vm_id, "console": console_tail(vm, tail)})
        self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if not self._auth():
            return
        if self.path.rstrip("/") != "/vms":
            return self._send(404, {"error": "not found"})
        body = self._body()
        sid = str(body.get("session_id") or "")
        if not sid:
            return self._send(400, {"error": "session_id is required"})
        try:
            vm = create_vm(sid, dict(body.get("env") or {}), str(body.get("golden") or "golden"))
        except Exception as exc:  # noqa: BLE001
            return self._send(500, {"error": repr(exc)[:500]})
        print(f"[fc-agent] started {vm.vm_id} for {sid} slot {vm.slot}: {vm.timings['total_ms']} ms", flush=True)
        self._send(201, {**vm.info(), "agent": vm.timings.get("agent", {})})

    def do_DELETE(self):  # noqa: N802
        if not self._auth():
            return
        parts = [p for p in self.path.split("/") if p]
        if len(parts) != 2 or parts[0] != "vms":
            return self._send(404, {"error": "not found"})
        vm = self._vm(parts[1])
        if vm is None:
            return
        ms = stop_vm(vm)
        with LOCK:
            VMS.pop(vm.vm_id, None)
        print(f"[fc-agent] stopped {vm.vm_id} ({vm.session_id}) in {ms} ms", flush=True)
        self._send(200, {"vm_id": vm.vm_id, "stop_ms": ms})

    def log_message(self, fmt, *args):  # quieter access log
        pass


def main() -> None:
    if os.geteuid() != 0:
        raise SystemExit("run with sudo")
    if len(SECRET) < 16:
        raise SystemExit("set FC_AGENT_SECRET (at least 16 characters; the same value as in Windows .env)")
    RUN.mkdir(parents=True, exist_ok=True)
    sh("pkill -f 'firecracker --api-sock .*/run/vm'", ok_fail=True)   # leftovers of a previous run
    setup_host_network()
    for i in range(1, SLOTS + 1):
        setup_slot(i)
    threading.Thread(target=reaper, daemon=True).start()
    print(f"[fc-agent] listening on {FCHOST_IP}:{LISTEN_PORT} | Windows host {WIN_HOST} | "
          f"{SLOTS} slots, agent ports {PORT_BASE + 1}-{PORT_BASE + SLOTS} | goldens: "
          f"{[p.name for p in GOLDEN_DIR.glob('*') if (p / 'golden.mem').exists()]}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()