"""Bounded queue wrapper with depth tracking and a never-silent-drop policy.

Thin wrapper over stdlib `queue.Queue(maxsize=...)` -- no new dependency.
Depth is tracked as a percentage of capacity and classified into one of
five states (master spec section 15):

    0-60%   NORMAL
    60-80%  WARN
    80-95%  DEGRADED
    95-100% CRITICAL
    100%    OVERFLOW   (queue full -- a further blocking put would block)

On overflow, `put_nowait_tracked` never raises and never silently discards
the dropped item: it records `first_dropped_seq`/`last_dropped_seq`/
`drop_count` so a health supervisor or event logger can surface the loss.
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Optional

NORMAL = "NORMAL"
WARN = "WARN"
DEGRADED = "DEGRADED"
CRITICAL = "CRITICAL"
OVERFLOW = "OVERFLOW"


class BoundedQueue:
    def __init__(self, expected_rate_hz: float, buffer_seconds: float = 10.0) -> None:
        if expected_rate_hz <= 0:
            raise ValueError("expected_rate_hz must be > 0")
        if buffer_seconds <= 0:
            raise ValueError("buffer_seconds must be > 0")
        self.expected_rate_hz = expected_rate_hz
        self.buffer_seconds = buffer_seconds
        # ponytail: floor of 1 so a low rate/short buffer never produces a
        # zero-capacity (permanently-OVERFLOW) queue.
        self.maxsize = max(int(expected_rate_hz * buffer_seconds), 1)
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=self.maxsize)

        self._drop_lock = threading.Lock()
        self._first_dropped_seq: Optional[int] = None
        self._last_dropped_seq: Optional[int] = None
        self._drop_count = 0

    # --- pass-through queue operations (this is still a real queue) ------
    def qsize(self) -> int:
        return self._queue.qsize()

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        return self._queue.get(block=block, timeout=timeout)

    def get_nowait(self) -> Any:
        return self._queue.get_nowait()

    def task_done(self) -> None:
        self._queue.task_done()

    # --- depth classification ---------------------------------------------
    def depth_percent(self) -> float:
        return (self._queue.qsize() / self.maxsize) * 100.0

    def depth_state(self) -> str:
        pct = self.depth_percent()
        if pct >= 100.0:
            return OVERFLOW
        if pct >= 95.0:
            return CRITICAL
        if pct >= 80.0:
            return DEGRADED
        if pct >= 60.0:
            return WARN
        return NORMAL

    # --- never-silently-drop overflow path ---------------------------------
    def put_nowait_tracked(self, item: Any, seq: int) -> bool:
        """Non-blocking put. Returns True on success.

        On overflow (queue full -- the blocking `put()` equivalent would
        block), returns False instead of raising `queue.Full`, and records
        the drop in `first_dropped_seq`/`last_dropped_seq`/`drop_count` so
        the loss is tracked, never silently discarded.
        """
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            with self._drop_lock:
                if self._first_dropped_seq is None:
                    self._first_dropped_seq = seq
                self._last_dropped_seq = seq
                self._drop_count += 1
            return False

    def drop_stats(self) -> dict[str, Optional[int]]:
        """Read-only snapshot for the health supervisor / event logger."""
        with self._drop_lock:
            return {
                "first_dropped_seq": self._first_dropped_seq,
                "last_dropped_seq": self._last_dropped_seq,
                "drop_count": self._drop_count,
            }


def _self_check() -> None:
    # maxsize=20 so every classification boundary lands on a whole item count.
    q = BoundedQueue(expected_rate_hz=20, buffer_seconds=1.0)
    assert q.maxsize == 20
    assert q.depth_state() == NORMAL

    for i in range(12):  # 12/20 = 60%
        assert q.put_nowait_tracked(i, seq=i) is True
    assert q.depth_state() == WARN

    for i in range(12, 16):  # 16/20 = 80%
        assert q.put_nowait_tracked(i, seq=i) is True
    assert q.depth_state() == DEGRADED

    for i in range(16, 19):  # 19/20 = 95%
        assert q.put_nowait_tracked(i, seq=i) is True
    assert q.depth_state() == CRITICAL

    assert q.put_nowait_tracked(19, seq=19) is True  # 20/20 = 100%
    assert q.depth_state() == OVERFLOW

    ok = q.put_nowait_tracked("overflow-item", seq=20)
    assert ok is False, "overflow put must return False, never raise"
    stats = q.drop_stats()
    assert stats == {"first_dropped_seq": 20, "last_dropped_seq": 20, "drop_count": 1}

    # a second overflow extends last_dropped_seq/drop_count, keeps first
    q.put_nowait_tracked("overflow-item-2", seq=21)
    stats = q.drop_stats()
    assert stats == {"first_dropped_seq": 20, "last_dropped_seq": 21, "drop_count": 2}

    # the queue itself is untouched by the failed overflow puts
    assert q.qsize() == 20
    assert q.get_nowait() == 0

    print("bounded_queue.py self-check OK")


if __name__ == "__main__":
    _self_check()
