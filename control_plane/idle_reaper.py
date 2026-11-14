"""
Agentic Cloud - Phase 1, Step 1.4
Idle Reaper: background thread that freezes idle agents.

Every `interval_s` seconds:
  1. find WAITING sessions idle longer than idle_timeout_s -> manager.freeze()
  2. every `liveness_every` ticks: check WAITING containers still answer /health;
     dead ones -> FROZEN (their memory is safe in the Postgres checkpoint)

Timing math (defaults: idle_timeout_s=30, interval_s=5):
  a session that went idle at t is frozen at the first tick after t + 30 s,
  i.e. between 30 s and 35 s after going idle (2.5 s late on average).
"""

from __future__ import annotations

import logging
import threading
import time

from control_plane.lifecycle import SessionManager
from control_plane.session_store import SessionStore

log = logging.getLogger("control_plane.reaper")


class IdleReaper:
    def __init__(
        self,
        manager: SessionManager,
        store: SessionStore,
        idle_timeout_s: float = 30.0,
        interval_s: float = 5.0,
        liveness_every: int = 6,
    ) -> None:
        self.manager = manager
        self.store = store
        self.idle_timeout_s = idle_timeout_s
        self.interval_s = interval_s
        self.liveness_every = max(1, liveness_every)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.ticks = 0
        self.frozen_total = 0
        self.lost_total = 0
        self.last_tick_ms = 0.0
        self.last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="idle-reaper", daemon=True)
        self._thread.start()
        log.info("idle reaper started (timeout=%ss, interval=%ss)", self.idle_timeout_s, self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 5)
        log.info("idle reaper stopped")

    def _run(self) -> None:
        # Event.wait returns True as soon as stop() is called -> fast shutdown
        while not self._stop.wait(self.interval_s):
            self.tick()

    def tick(self) -> None:
        t0 = time.perf_counter()
        self.ticks += 1
        try:
            for s in self.store.find_idle(self.idle_timeout_s):
                if self.manager.freeze(s["session_id"], reason="idle", blocking=False):
                    self.frozen_total += 1
                    log.info("froze idle session %s", s["session_id"])

            if self.ticks % self.liveness_every == 0:
                lost = self.manager.check_liveness()
                if lost:
                    self.lost_total += lost
                    log.warning("liveness: %d dead agent(s) moved to FROZEN", lost)

            self.last_error = None
        except Exception as exc:  # noqa: BLE001  (never let the loop die)
            self.last_error = repr(exc)
            log.exception("reaper tick failed")
        finally:
            self.last_tick_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    def status(self) -> dict:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "idle_timeout_s": self.idle_timeout_s,
            "interval_s": self.interval_s,
            "ticks": self.ticks,
            "frozen_total": self.frozen_total,
            "lost_total": self.lost_total,
            "last_tick_ms": self.last_tick_ms,
            "last_error": self.last_error,
        }