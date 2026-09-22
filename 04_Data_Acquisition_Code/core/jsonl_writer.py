"""Append-only JSONL writer, one file per stream/segment.

Design carried forward from `07_WT901C/WT901C_imu0/robust_io.py`'s
`AsyncJsonArrayWriter` (commit 3a0c757e913454ade5007cbacbb52bae48f321a0,
cited DECISIONS.md #4): a dedicated writer thread owns the file handle,
producers only ever touch a `queue.Queue` via `put()`, periodic flush,
drain-before-close, `fsync` before final close. That class wrote a single
growing JSON array; master spec section 8 requires JSONL (one JSON object
per line) instead, so this is a fresh implementation of the same pattern,
not an import across package boundaries.

Master spec section 16 ("Crash 후 같은 segment에 이어 쓰지 않는다"): only a
clean/normal session end renames `<stream>_segment{NNN}.jsonl.partial` to
the final `.jsonl` name. An interrupted session leaves the `.partial` file
in place for `core/recovery.py`, and the next segment is a brand new
`JsonlWriter` instance with a higher `segment_id` -- this class refuses to
open a `.partial` path that already exists, so nothing ever resumes a
crashed segment file.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Optional


class JsonlWriter:
    _STOP = object()

    def __init__(
        self,
        output_dir: str,
        stream_name: str,
        segment_id: int,
        *,
        flush_interval_s: float = 0.4,
        fsync_interval_s: float = 1.5,
    ) -> None:
        if flush_interval_s <= 0:
            raise ValueError("flush_interval_s must be > 0")
        if fsync_interval_s < flush_interval_s:
            # master spec: fsync cadence must be less frequent than flush.
            raise ValueError("fsync_interval_s must be >= flush_interval_s")

        self.output_dir = output_dir
        self.stream_name = stream_name
        self.segment_id = segment_id
        self.flush_interval_s = float(flush_interval_s)
        self.fsync_interval_s = float(fsync_interval_s)

        filename = f"{stream_name}_segment{segment_id:03d}.jsonl"
        self.final_path = os.path.join(output_dir, filename)
        self.partial_path = self.final_path + ".partial"

        self._queue: "queue.Queue" = queue.Queue()
        self._error: Optional[BaseException] = None
        self._ready = threading.Event()
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name=f"jsonl-writer-{stream_name}-{segment_id:03d}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(5.0):
            raise TimeoutError("jsonl writer thread did not start in time")
        self._raise_if_failed()

    def put(self, record: dict) -> None:
        """Enqueue one record (thread-safe, callable from any producer
        thread). Raises if this writer has already been closed."""
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot put() into a closed JsonlWriter")
            self._raise_if_failed()
            self._queue.put(record)

    def pending_count(self) -> int:
        return self._queue.qsize()

    def close(self, clean: bool = True, timeout: float = 30.0) -> None:
        """Drain the queue, flush+fsync, close the file handle.

        `clean=True` (normal end of session) atomically renames
        `<name>.jsonl.partial` -> `<name>.jsonl` via `os.replace`.
        `clean=False` leaves the `.partial` file in place -- use this (or
        `close_interrupted()`) when the session ended abnormally, so the
        recovery tool has something to work on.
        """
        with self._lock:
            if self._closed:
                self._raise_if_failed()
                return
            self._closed = True
            self._queue.put(self._STOP)

        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("jsonl writer thread did not drain within timeout")
        self._raise_if_failed()

        if clean:
            os.replace(self.partial_path, self.final_path)
        # else: leave <path>.partial for core/recovery.py.

    def close_interrupted(self, timeout: float = 30.0) -> None:
        """Equivalent to `close(clean=False)` -- drains/flushes/fsyncs but
        leaves the `.partial` file un-renamed."""
        self.close(clean=False, timeout=timeout)

    def start_new_segment(self, clean: bool = True) -> "JsonlWriter":
        """Close this writer and return a fresh writer for `segment_id + 1`.
        The preferred way to rotate segments -- never keep writing into the
        same file after a crash (master spec section 16)."""
        self.close(clean=clean)
        return JsonlWriter(
            self.output_dir,
            self.stream_name,
            self.segment_id + 1,
            flush_interval_s=self.flush_interval_s,
            fsync_interval_s=self.fsync_interval_s,
        )

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"jsonl writer thread failed: {self._error}") from self._error

    def _run(self) -> None:
        handle = None
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            # "x" (exclusive create) -- refuse to open/append into a
            # .partial file left behind by a crashed segment; the caller
            # must bump segment_id instead (see module docstring).
            handle = open(self.partial_path, "x", encoding="utf-8")
            self._ready.set()
            last_flush = time.monotonic()
            last_fsync = time.monotonic()

            while True:
                try:
                    item = self._queue.get(timeout=0.2)
                except queue.Empty:
                    last_flush, last_fsync = self._maybe_flush_fsync(handle, last_flush, last_fsync)
                    continue
                try:
                    if item is self._STOP:
                        break
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    last_flush, last_fsync = self._maybe_flush_fsync(handle, last_flush, last_fsync)
                finally:
                    self._queue.task_done()

            # final flush + fsync before close, on every close path
            # (clean or interrupted) -- whatever made it into the queue is
            # durable on disk even if the file stays a `.partial`.
            handle.flush()
            os.fsync(handle.fileno())
        except Exception as exc:  # pragma: no cover - defensive
            self._error = exc
            self._ready.set()
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

    def _maybe_flush_fsync(self, handle, last_flush: float, last_fsync: float):
        now = time.monotonic()
        if now - last_flush >= self.flush_interval_s:
            handle.flush()
            last_flush = now
        if now - last_fsync >= self.fsync_interval_s:
            os.fsync(handle.fileno())
            last_fsync = now
        return last_flush, last_fsync
