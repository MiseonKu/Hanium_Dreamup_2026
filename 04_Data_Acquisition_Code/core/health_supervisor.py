"""Health aggregation + escalation-policy signals (master spec section 15).

On a periodic timer, folds every registered stream's
`core.bounded_queue.BoundedQueue` depth state into one
`core.schemas.health_record`. This module only computes/exposes signals --
it does not itself stop sensors/targets/output. `session_controller.py`
(built separately) polls `should_suppress_live_output()` /
`should_block_new_targets()` / `should_request_safe_stop()` and acts on
them.

Escalation policy (master spec section 15), cumulative by severity rank:
    WARN     -> emit a health_state_changed event only
    DEGRADED -> should_suppress_live_output()  True (stop live preview/console)
    CRITICAL -> should_block_new_targets()     True (block new target dispatch)
    OVERFLOW -> should_request_safe_stop()     True (request safe stop)

CPU/memory: `psutil` is not a dependency anywhere else in this repo
(checked via `grep -rl psutil` from the capstone-code root -- zero hits),
so this task does not add it either. `os.getloadavg()` is POSIX-only and
unavailable on the Windows dev machine this package targets, so it isn't
used as a substitute. `cpu_percent`/`memory_percent` are therefore left as
`None` -- never a fabricated number. Add `psutil` as a real dependency (and
to `requirements.txt`) if real CPU/mem readings become necessary.

`WindowsCollectionGuard` reimplemented from
`07_WT901C/WT901C_imu0/robust_io.py` (commit
3a0c757e913454ade5007cbacbb52bae48f321a0, cited per master-spec section 4
and DECISIONS.md #4) -- best-effort sleep prevention + Above-Normal process
priority while a collection session is active, no-op on non-Windows.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import sys
import threading
from typing import Optional

# Needed for the `__main__` self-check at the bottom to run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import schemas  # noqa: E402
from core.bounded_queue import CRITICAL, DEGRADED, NORMAL, OVERFLOW, WARN, BoundedQueue  # noqa: E402
from core.event_bus import EventBus

_STATE_RANK = {NORMAL: 0, WARN: 1, DEGRADED: 2, CRITICAL: 3, OVERFLOW: 4}


class WindowsCollectionGuard:
    """Best-effort sleep prevention + Above-Normal priority while active.

    Reimplemented from `07_WT901C/WT901C_imu0/robust_io.py`
    (commit 3a0c757e913454ade5007cbacbb52bae48f321a0) -- trivial, low-risk,
    same behavior, with a self-check added (that file had none)."""

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
    NORMAL_PRIORITY_CLASS = 0x00000020

    def __init__(self) -> None:
        self.active = False

    def start(self) -> None:
        if os.name != "nt" or self.active:
            return
        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadExecutionState(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), self.ABOVE_NORMAL_PRIORITY_CLASS)
        self.active = True

    def stop(self) -> None:
        if os.name != "nt" or not self.active:
            return
        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), self.NORMAL_PRIORITY_CLASS)
        self.active = False


class HealthSupervisor:
    def __init__(
        self,
        session_id: str,
        segment_id: int,
        *,
        disk_path: str = ".",
        event_bus: Optional[EventBus] = None,
        interval_s: float = 1.0,
    ) -> None:
        self.session_id = session_id
        self.segment_id = segment_id
        self.disk_path = disk_path
        self.event_bus = event_bus
        self.interval_s = interval_s

        self._queues: dict[str, BoundedQueue] = {}
        self._lock = threading.Lock()
        self._host_seq = 0
        self._worst_state = NORMAL
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def register_queue(self, stream_name: str, bounded_queue: BoundedQueue) -> None:
        with self._lock:
            self._queues[stream_name] = bounded_queue

    def start(self) -> None:
        """Start the periodic sampling timer thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="health-supervisor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.sample()
            self._stop_event.wait(self.interval_s)

    def sample(self) -> dict:
        """Take one health snapshot now (also called by the timer thread).
        Returns the `core.schemas.health_record` dict; does not write it
        anywhere -- the caller (a JsonlWriter, typically) owns persistence.
        """
        with self._lock:
            queues = dict(self._queues)

        queue_depth: dict[str, int] = {}
        queue_state: dict[str, str] = {}
        worst = NORMAL
        for name, q in queues.items():
            queue_depth[name] = q.qsize()
            state = q.depth_state()
            queue_state[name] = state
            if _STATE_RANK[state] > _STATE_RANK[worst]:
                worst = state

        try:
            disk_free = shutil.disk_usage(self.disk_path).free
        except OSError:
            disk_free = None

        with self._lock:
            self._host_seq += 1
            host_seq = self._host_seq
            previous_worst = self._worst_state
            self._worst_state = worst

        base = schemas.base_record(
            schema=schemas.SCHEMA_HEALTH,
            session_id=self.session_id,
            segment_id=self.segment_id,
            stream=schemas.STREAM_HEALTH,
            host_seq=host_seq,
            t_pc_rx_ns=schemas.monotonic_ns(),
            valid=True,
            parse_status="ok",
        )
        record = schemas.health_record(
            base,
            queue_depth=queue_depth,
            cpu_percent=None,  # see module docstring: no psutil in this repo
            memory_percent=None,
            disk_free_bytes=disk_free,
            queue_state=queue_state,
        )

        if self.event_bus is not None and worst != previous_worst:
            self.event_bus.publish(
                "health_state_changed",
                {
                    "previous_state": previous_worst,
                    "new_state": worst,
                    "queue_state": queue_state,
                    "host_seq": host_seq,
                },
            )

        return record

    # --- escalation-policy signals (master spec section 15) ---------------
    def should_suppress_live_output(self) -> bool:
        """DEGRADED or worse -> stop live preview/console output."""
        return _STATE_RANK[self._worst_state] >= _STATE_RANK[DEGRADED]

    def should_block_new_targets(self) -> bool:
        """CRITICAL or worse -> block new target dispatch."""
        return _STATE_RANK[self._worst_state] >= _STATE_RANK[CRITICAL]

    def should_request_safe_stop(self) -> bool:
        """OVERFLOW -> request safe stop."""
        return self._worst_state == OVERFLOW


def _self_check() -> None:
    import json

    q_ok = BoundedQueue(expected_rate_hz=20, buffer_seconds=1.0)
    q_overflow = BoundedQueue(expected_rate_hz=1, buffer_seconds=1.0)
    q_overflow.put_nowait_tracked("x", seq=0)  # 1/1 = 100% = OVERFLOW

    bus = EventBus()
    events = []
    bus.subscribe("health_state_changed", lambda et, payload: events.append(payload))

    sup = HealthSupervisor("session_selfcheck", 0, event_bus=bus, disk_path=".")
    sup.register_queue("imu3_tip", q_ok)
    sup.register_queue("encoder", q_overflow)

    record = sup.sample()
    json.dumps(record)  # must be JSON-serializable
    assert record["data"]["queue_state"]["encoder"] == OVERFLOW
    assert record["data"]["cpu_percent"] is None
    assert sup.should_request_safe_stop() is True
    assert sup.should_block_new_targets() is True
    assert sup.should_suppress_live_output() is True
    assert len(events) == 1 and events[0]["new_state"] == OVERFLOW

    guard = WindowsCollectionGuard()
    guard.start()
    guard.stop()  # must not raise, on any platform

    print("health_supervisor.py self-check OK")


if __name__ == "__main__":
    _self_check()
