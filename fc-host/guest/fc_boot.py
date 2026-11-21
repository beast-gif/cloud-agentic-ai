"""
Agentic Cloud - Phase 5, Steps 5.2 + 5.3
PID 1 supervisor inside the agent microVM (started by /sbin/fc-init, runs as root).

1. Reads THIS VM's configuration from MMDS, Firecracker's metadata service
   (http://169.254.169.254, V2 = session token first, like AWS IMDSv2).
   The rootfs holds no secrets and no per-session data.
2. Writes /run/resolv.conf and /run/hosts (/etc/* are symlinks there: the root disk is read-only).
3. Starts the agent server as uid 10001 ("agent", same user as in the Docker image) in a child process.
4. Stays alive as root for ONE privileged job the agent must not have: setting the clock.
   After a snapshot restore the guest's wall clock still shows the moment of the snapshot;
   the host fixes it with  POST http://172.16.0.2:8081/clock {"epoch_ns": ...}
   (accepted only from the host side of the tap, 172.16.0.1).

MMDS document (PUT by the host before InstanceStart):
  {"agent": {"env": {"AGENT_DSN": "...", "FC_PREBIND": "1", ...},
             "nameserver": "8.8.8.8", "log_to_console": false}}
"""
import json
import os
import signal
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

T0 = time.monotonic()
MMDS = "http://169.254.169.254"
HOST_IP = "172.16.0.1"
AGENT_UID = 10001


def uptime_s() -> float:
    with open("/proc/uptime") as f:
        return float(f.read().split()[0])


def console(msg: str) -> None:
    try:
        with open("/dev/console", "w") as c:
            c.write(msg + "\n")
    except OSError:
        pass


def fetch_mmds(deadline_s: float = 20.0) -> dict:
    last = None
    while time.monotonic() - T0 < deadline_s:
        try:
            req = urllib.request.Request(f"{MMDS}/latest/api/token", method="PUT",
                                         headers={"X-metadata-token-ttl-seconds": "300"})
            token = urllib.request.urlopen(req, timeout=1).read().decode()
            req = urllib.request.Request(f"{MMDS}/", headers={"X-metadata-token": token,
                                                              "Accept": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=1).read().decode())
        except Exception as exc:  # noqa: BLE001 - network may need a few ms after boot
            last = exc
            time.sleep(0.02)
    raise RuntimeError(f"MMDS not reachable: {last!r}")


class ClockHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        if self.path != "/clock" or self.client_address[0] != HOST_IP:
            self.send_response(403); self.end_headers(); return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            before = time.time_ns()
            time.clock_settime_ns(time.CLOCK_REALTIME, int(body["epoch_ns"]))
            out = {"ok": True, "corrected_by_ms": round((int(body["epoch_ns"]) - before) / 1e6, 1)}
            code = 200
        except Exception as exc:  # noqa: BLE001
            out, code = {"ok": False, "error": repr(exc)}, 400
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # keep the serial console quiet
        pass


def start_agent(env: dict, log_target: str) -> int:
    pid = os.fork()
    if pid:
        return pid
    try:  # child: logs -> file in RAM (or console), drop root, become the agent server
        fd = os.open(log_target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        if log_target != "/dev/console":
            os.fchown(fd, AGENT_UID, AGENT_UID)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.setgroups([])
        os.setgid(AGENT_UID)
        os.setuid(AGENT_UID)
        os.chdir("/app")
        os.execve("/usr/local/bin/python", ["/usr/local/bin/python", "/app/server.py"], env)
    except Exception as exc:  # noqa: BLE001
        console(f"FCBOOT agent exec failed {exc!r}")
    os._exit(1)


def main() -> None:
    start_uptime = uptime_s()
    doc = fetch_mmds().get("agent") or {}
    mmds_ms = (time.monotonic() - T0) * 1000.0
    env = {str(k): str(v) for k, v in (doc.get("env") or {}).items()}

    with open("/run/resolv.conf", "w") as f:
        f.write(f"nameserver {doc.get('nameserver', '8.8.8.8')}\n")
    with open("/run/hosts", "w") as f:
        f.write("127.0.0.1 localhost\n127.0.1.1 fc-agent\n")

    env.setdefault("PORT", "8080")
    env.setdefault("WORKSPACE_DIR", "/workspace")
    env.update({"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/home/agent",
                "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "FC_BOOT_UPTIME_S": f"{start_uptime:.3f}", "FC_MMDS_MS": f"{mmds_ms:.1f}"})
    console(f"FCBOOT fc_boot_uptime_s={start_uptime:.3f} mmds_ms={mmds_ms:.1f} "
            f"env_keys={','.join(sorted(k for k in env if k != 'AGENT_DSN' and 'TOKEN' not in k))}")

    pid = start_agent(env, "/dev/console" if doc.get("log_to_console") else "/run/agent.log")

    server = ThreadingHTTPServer(("0.0.0.0", 8081), ClockHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    # PID 1 duty: reap children; report if the agent dies
    while True:
        try:
            done, status = os.wait()
        except ChildProcessError:
            time.sleep(1)
            continue
        if done == pid:
            console(f"FCBOOT agent exited status={status}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - PID 1 must not just vanish silently
        console(f"FCBOOT error {exc!r}")
        time.sleep(3600)