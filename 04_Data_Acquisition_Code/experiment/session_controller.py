"""Top-level session orchestrator (master spec section 7's process
architecture / section 8's session directory layout / section 16's
graceful-shutdown sequence / section 22's execution modes).

Threads, not processes (documented simplification): with 8 components that
only ever run in `mock`/`passive`(mock-backed)/test modes -- no real
hardware anywhere in this task -- `multiprocessing` would add IPC/pickling
complexity this task doesn't need. Everything here is a `threading.Thread`.
Switch to processes only if a real-hardware run shows actual GIL
contention; nothing about the collector interfaces below would need to
change (they already communicate via queues/files, not shared Python
objects).

Collector interfaces actually found on disk at the time this was written
(concurrent workstreams; re-check before relying on this note):
  - `collectors.collect_imu_rs232.IMURS232Collector` -- fully self-managing:
    owns its own reader+pump threads, an internal `core.bounded_queue.
    BoundedQueue` (exposed as `.queue`), and its own `core.jsonl_writer.
    JsonlWriter`. Lifecycle is `.start()` / `.stop(clean=bool, timeout=...)`.
    One instance per sensor_id in `VALID_SENSOR_IDS`.
  - `collectors.collect_esp32_encoder.Esp32EncoderCollector` and
    `collectors.collect_vive_tracker.ViveTrackerCollector` -- pull-style:
    construct with a source, optionally call `.start()` (Esp32 has one,
    Vive does not -- guarded with `hasattr`), then the CALLER loops
    `collector.read_record()` and owns the `JsonlWriter` + thread + when to
    stop. This module supplies that loop (`_pull_reader_loop`) and, since
    neither pull-style collector owns a `BoundedQueue`, a thin
    `_WriterQueueDepthAdapter` so `core.health_supervisor.HealthSupervisor`
    still gets a real backpressure signal (sourced from the `JsonlWriter`'s
    own `pending_count()`) instead of nothing.
  - `collectors.collect_events.EventCollector` (the "event_logger" box in
    the architecture diagram): `.log_event(event_type, *, message=None,
    details=None, ...)` + `.close(clean=True)`, no `.start()`. This module
    tries to import and use it, and on ANY failure (import, missing
    method, construction) falls back to a small built-in event logger
    (`_fallback_log_event`) built directly from `core.jsonl_writer.
    JsonlWriter` + `core.schemas.event_record` instead -- both of which
    already existed and are the same primitives `EventCollector` itself
    uses, so the fallback produces byte-for-byte-equivalent records.

`core/health_supervisor.py`'s own docstring says it "does not itself
persist" a health record -- "the caller (a JsonlWriter, typically) owns
persistence." This module is that caller: `_health_loop` samples on
`HealthSupervisor.interval_s` and writes each record to a `health` stream
JsonlWriter itself, rather than using `HealthSupervisor.start()`'s built-in
timer thread (which only updates internal state / publishes events, never
persists anything).
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))
_MOCKS_DIR = _PACKAGE_ROOT / "mocks"
if str(_MOCKS_DIR) not in sys.path:
    sys.path.insert(0, str(_MOCKS_DIR))  # mocks/*.py use flat imports, matching tests/test_mocks.py's own convention

from core import interrupt_handler, schemas
from core.bounded_queue import CRITICAL, DEGRADED, NORMAL, OVERFLOW, WARN, BoundedQueue
from core.event_bus import EventBus
from core.health_supervisor import HealthSupervisor
from core.jsonl_writer import JsonlWriter

from mock_esp32 import MockESP32Encoder  # noqa: E402
from mock_imu import MockWT901CImu  # noqa: E402
from mock_vive import MockViveTracker  # noqa: E402

# Each collector import is independently tolerant of the module not
# existing yet (concurrent workstream) -- see module docstring.
try:
    from collectors.collect_imu_rs232 import VALID_SENSOR_IDS as _IMU_SENSOR_IDS
    from collectors.collect_imu_rs232 import IMURS232Collector
except Exception:
    IMURS232Collector = None
    _IMU_SENSOR_IDS = ("imu0_base", "imu1_link_root", "imu2_link_mid", "imu3_tip")

try:
    from collectors.collect_esp32_encoder import Esp32EncoderCollector
except Exception:
    Esp32EncoderCollector = None

try:
    from collectors.collect_vive_tracker import ViveTrackerCollector
except Exception:
    ViveTrackerCollector = None

# Real-hardware-mode constants (Task 1/2, closing the gap the archived
# experiment_scripts/*.py ad hoc scripts papered over). Both are
# documented, overridable defaults -- never invisible magic numbers.
DEFAULT_REAL_IMU_BAUD = 9600  # confirmed real-hardware value: experiment_scripts/run_real_imu_capture2.py
DEFAULT_REAL_ENCODER_BAUD = 115200  # confirmed real-hardware value: experiment_scripts/collect_3d_sweep_50_v2.py (ser = serial.Serial("COM20", 115200, ...))
# OpenVR's getDeviceToAbsoluteTrackingPose() has no artificial rate limit
# (unlike the ad hoc script's ~2Hz, which was bounded by sharing the
# encoder's STATE-poll loop). 100Hz is a reasonable starting poll rate for
# a real bring-up -- NOT a measured achievable rate; measure the actual
# delivered rate from the vive_tracker stream's own host_seq/timestamps
# once real hardware is available (no existing "measured vs configured
# rate" convention elsewhere in this repo to reuse -- see report).
REAL_VIVE_POLL_HZ = 100.0

DATA_SCHEMA_VERSION = {
    schemas.STREAM_IMU: schemas.SCHEMA_IMU,
    schemas.STREAM_ENCODER: schemas.SCHEMA_ENCODER,
    schemas.STREAM_COMMAND: schemas.SCHEMA_COMMAND,
    schemas.STREAM_VIVE: schemas.SCHEMA_VIVE,
    schemas.STREAM_EVENT: schemas.SCHEMA_EVENT,
    schemas.STREAM_HEALTH: schemas.SCHEMA_HEALTH,
    schemas.STREAM_STABILITY: schemas.SCHEMA_STABILITY,
}

MODES = ("inventory", "mock", "passive", "soak", "experiment")


def _allocate_session_dir(sessions_root: Path) -> tuple[str, Path]:
    """`session_YYYYMMDD_NNN/` (master spec section 8), race-safe via
    atomic `mkdir(exist_ok=False)` instead of check-then-create."""
    sessions_root.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().strftime("%Y%m%d")
    n = 1
    while True:
        session_id = f"session_{date_str}_{n:03d}"
        path = sessions_root / session_id
        try:
            path.mkdir(parents=False, exist_ok=False)
            return session_id, path
        except FileExistsError:
            n += 1


def _paced(fn, period_s: float):
    """Wrap a zero-arg callable so repeated calls are rate-limited to
    roughly one call per `period_s`.

    `mocks/*.py` sources return instantly (by design -- see their
    docstrings) instead of blocking on I/O the way a real serial line
    would. Without this, `IMURS232Collector`'s reader thread (which this
    module feeds via `byte_source=`, see `_start_imu`) spins as fast as
    the CPU allows: observed in practice to queue a backlog that then
    takes tens of seconds to drain per stream on shutdown, several times
    longer than a short mock/smoke-test session's own duration. Pacing at
    the source is the minimal fix that doesn't require touching
    `collectors/collect_imu_rs232.py` (owned by a concurrent workstream).
    """
    lock = threading.Lock()
    state = {"next_due": time.monotonic()}

    def call():
        with lock:
            now = time.monotonic()
            wait = state["next_due"] - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            state["next_due"] = max(now, state["next_due"]) + period_s
        return fn()

    return call


class _WriterQueueDepthAdapter:
    """`core.bounded_queue`-compatible `.qsize()`/`.depth_state()` pair
    sourced from a real `JsonlWriter`'s own backlog (`pending_count()`),
    for the pull-style collectors (encoder, vive_tracker) that don't own a
    `BoundedQueue` themselves -- see module docstring. `nominal_capacity`
    is a soft classification scale only; `JsonlWriter`'s queue is
    unbounded, so this reports "falling behind," never a hard reject.
    """

    def __init__(self, writer: JsonlWriter, nominal_capacity: int) -> None:
        self._writer = writer
        self._nominal_capacity = max(int(nominal_capacity), 1)

    def qsize(self) -> int:
        return self._writer.pending_count()

    def depth_state(self) -> str:
        pct = (self.qsize() / self._nominal_capacity) * 100.0
        if pct >= 100.0:
            return OVERFLOW
        if pct >= 95.0:
            return CRITICAL
        if pct >= 80.0:
            return DEGRADED
        if pct >= 60.0:
            return WARN
        return NORMAL


class _Esp32StatePollReadRecordAdapter:
    """Bridges `collectors.collect_esp32_state_poll.Esp32StatePollCollector`
    -- whose telemetry method is `poll_once(**context) -> dict`, driven by
    an explicit caller loop in its own design (matching how the archived
    real sweep script drove it) -- to the pull-style `.read_record() ->
    dict` contract `_pull_reader_loop` already drives everything else
    through (encoder mock, vive_tracker). `controller._current_target_context`
    is forwarded straight into `poll_once`'s own `**context` on every call,
    so target-context threading (Task 4) reaches this stream natively,
    without needing `_pull_reader_loop`'s generic post-hoc overlay at all
    (that overlay still runs afterward too, harmlessly -- it never clobbers
    a field the collector already set)."""

    def __init__(self, poll_collector: Any, controller: "SessionController") -> None:
        self._poll_collector = poll_collector
        self._controller = controller

    def read_record(self) -> dict:
        return self._poll_collector.poll_once(**self._controller._current_target_context)


class SessionController:
    """Owns one `session_YYYYMMDD_NNN/` directory end to end: manifest,
    collectors, health, events, and the section-16 shutdown sequence.

    Usage (see `tools/run_mock_session.py`):
        controller = SessionController("mock")
        controller.start()
        controller.run_for(60.0)   # blocks; calls shutdown() internally
    """

    def __init__(
        self,
        mode: str,
        *,
        sessions_root: Optional[str] = None,
        experiment_cli_flag: bool = False,
        sensor_map_path: Optional[str] = None,
        imu_baud: int = DEFAULT_REAL_IMU_BAUD,
        encoder_collector_factory: Optional[Any] = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.experiment_cli_flag = experiment_cli_flag
        self.segment_id = 0  # single segment per session in this task's scope -- see report

        # Real-hardware-mode config (Task 1). `sensor_map_path` gates real
        # IMU port resolution -- no config, no guessed COM port, real IMUs
        # stay "unavailable" (see _start_imu). `encoder_collector_factory`
        # is the pluggable integration point for the real ESP32/03_SO-101_
        # control encoder collector a concurrent workstream owns -- see
        # _start_encoder's docstring-comment for the exact interface it
        # must satisfy.
        self.imu_baud = imu_baud
        self.sensor_map_path = sensor_map_path
        self._sensor_map: Optional[dict] = self._load_sensor_map(sensor_map_path)
        self._matched_ports: dict[str, Any] = {}
        self._encoder_collector_factory = encoder_collector_factory

        # Task 5: common start barrier. Each successfully-constructed
        # collector's actual "begin producing data" call is deferred into
        # this list instead of firing immediately; _release_start_barrier
        # fires them all together once every collector that's GOING to
        # start has been constructed (sized to however many that actually
        # is -- an unavailable collector never counts against it, so one
        # missing sensor can't block the rest forever).
        self._pending_launches: list = []

        # Task 4: per-target context (trial_id/target_id/planned_order_index/
        # workspace_mode/experiment_state), stamped into every subsequently
        # -read pull-style record (vive_tracker, future real encoder) so
        # every stream stays joinable by target_id. Set via
        # set_target_context() -- see that method's docstring for the one
        # documented gap (the 4 self-managed IMU collectors aren't reached).
        self._current_target_context: dict = {}
        self._pull_stream_stats: dict[str, dict] = {}  # Task 3 raw material -- see _collect_stream_stats
        self._start_monotonic_ns: Optional[int] = None

        self.package_root = _PACKAGE_ROOT
        self.capstone_root = self.package_root.parent
        self.sessions_root = Path(sessions_root) if sessions_root else self.package_root / "sessions"

        self.session_id, self.session_dir = _allocate_session_dir(self.sessions_root)
        self.raw_dir = self.session_dir / "raw"
        self.indexes_dir = self.session_dir / "indexes"
        self.reports_dir = self.session_dir / "reports"
        self.manifest_path = self.session_dir / "manifest.json"

        self.event_bus = EventBus()
        self.health = HealthSupervisor(
            self.session_id, self.segment_id, disk_path=str(self.session_dir), event_bus=self.event_bus
        )
        self.event_bus.subscribe("health_state_changed", lambda et, payload: self._log_event(et, details=payload))
        # Task 4: adaptive-settling event wiring. experiment/
        # experiment_state_machine.py publishes on this same EventBus
        # (constructor-injected, same pattern as HealthSupervisor above) --
        # this module never adds a parallel event mechanism, it translates
        # that machine's own transitions into the exact event_type strings
        # master spec section 10 asks for and persists them the normal way
        # (_log_event -> EventCollector/fallback writer). Harmless no-op
        # today (mock-mode sessions don't construct an
        # ExperimentStateMachine yet -- see report); wiring here now means
        # a future experiment loop only has to construct one with
        # event_bus=self.event_bus, nothing in this file needs to change.
        self.event_bus.subscribe("experiment_state_changed", self._on_experiment_state_changed)
        self.event_bus.subscribe("target_selected", lambda et, payload: self._log_event("TARGET_SELECTED", details=payload))
        self.event_bus.subscribe("command_sent", lambda et, payload: self._log_event("COMMAND_SENT", details=payload))
        self.event_bus.subscribe(
            "encoder_motion_onset", lambda et, payload: self._log_event("ENCODER_MOTION_ONSET", details=payload)
        )

        self._collectors: dict[str, Any] = {}
        self._collector_status: dict[str, str] = {}
        self._mock_sources: dict[str, Any] = {}
        self._pull_threads: dict[str, threading.Thread] = {}
        self._pull_writers: dict[str, JsonlWriter] = {}

        self._events_collector: Any = None
        self._fallback_event_writer: Optional[JsonlWriter] = None
        self._event_log_fn = None
        self._event_seq = 0

        self._health_writer: Optional[JsonlWriter] = None
        self._health_thread: Optional[threading.Thread] = None
        self._health_stop_event: Optional[threading.Event] = None

        self.start_utc: Optional[str] = None
        self.git_commit_hash: Optional[str] = None
        self.interrupted = False
        self._started = False
        self._ended = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # real-hardware-mode config + cross-stream context (Tasks 1/4)
    # ------------------------------------------------------------------
    @staticmethod
    def _load_sensor_map(path: Optional[str]) -> Optional[dict]:
        """Load `config/sensor_map.example.yaml`-shaped YAML (top-level
        `sensor_map:` key -> `{logical_id: {vid, pid, serial_number, ...}}`,
        see that file). Returns None if no path was given or the file
        doesn't exist -- callers must then fail closed (never guess a COM
        port), not fall back to any hardcoded default."""
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        import yaml  # PyYAML is already a declared dependency (requirements.txt); not hardware-tied, so no lazy-import-for-CI reason to defer it further

        with p.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        return doc.get("sensor_map") or None

    def set_target_context(
        self,
        *,
        trial_id: Optional[int] = None,
        target_id: Optional[str] = None,
        planned_order_index: Optional[int] = None,
        workspace_mode: Optional[str] = None,
        experiment_state: Optional[str] = None,
    ) -> None:
        """Master spec section 5: every stream joinable by `target_id`.
        Call once per target from a future experiment loop (e.g. right
        after `ExperimentStateMachine.start_new_target()`) -- every
        record subsequently read from a pull-style collector this module
        drives (vive_tracker today; the future real encoder collector once
        wired via `encoder_collector_factory`) is stamped with these
        fields in `_pull_reader_loop`, unless the collector already set a
        field itself (never clobbered).

        Documented gap: this does NOT reach the 4 self-managed IMU
        collectors. `collectors.collect_imu_rs232.IMURS232Collector` owns
        its own writer internally and builds `base_record` without
        accepting trial/target kwargs, and that file is owned by a
        concurrent workstream this task does not touch -- see report. IMU
        records can still be joined to a target post-hoc via their
        `t_pc_rx_ns` falling inside the interval between two
        `TARGET_SELECTED` event records for now.
        """
        with self._lock:
            self._current_target_context = {
                "trial_id": trial_id,
                "target_id": target_id,
                "planned_order_index": planned_order_index,
                "workspace_mode": workspace_mode,
                "experiment_state": experiment_state,
            }

    def _on_experiment_state_changed(self, event_type: str, payload: dict) -> None:
        from experiment.experiment_state_machine import translate_transition_to_event_types

        for mapped_type in translate_transition_to_event_types(payload):
            self._log_event(mapped_type, details=payload)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        interrupt_handler.shutdown_requested.clear()  # fresh state -- a prior session/test may have left it set

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.start_utc = schemas.utc_iso_now()
        self._start_monotonic_ns = schemas.monotonic_ns()  # Task 3: session_summary actual_rate_hz denominator
        self.git_commit_hash = self._get_git_commit_hash()
        self._write_manifest(session_complete=False, interrupted=False)

        self._start_event_logger()
        self._log_event("session_started", details={"mode": self.mode, "session_id": self.session_id})
        self._start_health()

        if self.mode == "inventory":
            self._run_inventory()
            self.shutdown(interrupted=False)
            return

        if self.mode == "experiment":
            if not self._check_experiment_gate():
                self._log_event(
                    "experiment_mode_refused",
                    message="safety gate not satisfied: requires BOTH config hardware_motion_enabled=true "
                    "AND --enable-hardware-motion (experiment/safety_gate.hardware_motion_allowed)",
                )
                self.shutdown(interrupted=True)
                raise RuntimeError(
                    "experiment mode refused by safety gate -- this task never actually "
                    "exercises real hardware motion, so this path is implemented but unreachable here"
                )
            # NOTE: never reached with real hardware in this task -- see report.

        is_mock = self.mode in ("mock", "soak")
        self._start_all_collectors(is_mock)

    def run_for(self, duration_s: float) -> None:
        """Block for `duration_s`, polling for an interrupt or a
        health-triggered safe-stop, then run the clean-shutdown sequence.
        No-op if the session already ended (e.g. `inventory` mode, which
        shuts itself down inside `start()`)."""
        if self._ended:
            return
        deadline = time.monotonic() + duration_s
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            if interrupt_handler.shutdown_requested.is_set():
                self.shutdown(interrupted=True)
                return
            if self.health.should_request_safe_stop():
                self._log_event("health_triggered_safe_stop")
                self.shutdown(interrupted=True)
                return
            time.sleep(min(0.05, deadline - now))
        self.shutdown(interrupted=False)

    def shutdown(self, interrupted: bool = False) -> None:
        """Master spec section 16: set flag -> stop dispatching new targets
        (no-op -- no target dispatcher exists in this package's scope yet)
        -> request safe-stop -> drain/flush/fsync -> update manifest ->
        close files. Idempotent -- a second call is a no-op."""
        with self._lock:
            if self._ended:
                return
            self._ended = True

        interrupt_handler.shutdown_requested.set()
        self._log_event("shutdown_initiated", details={"interrupted": interrupted, "mode": self.mode})

        # Task 3: snapshot writer backlog *now*, before the drain/close
        # sequence below empties every queue -- "writer lag at the moment
        # shutdown was requested" is the meaningful number for
        # session_summary.json; pending_count() after close() is always 0.
        writer_pending_snapshot: dict[str, Optional[int]] = {}
        for logical_id, writer in self._pull_writers.items():
            writer_pending_snapshot[logical_id] = writer.pending_count()
        for sensor_id in sorted(_IMU_SENSOR_IDS):
            collector = self._collectors.get(sensor_id)
            imu_writer = getattr(collector, "writer", None) if collector is not None else None
            if imu_writer is not None:
                writer_pending_snapshot[sensor_id] = imu_writer.pending_count()

        for collector in self._collectors.values():
            fn = getattr(collector, "request_safe_stop", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

        clean = not interrupted

        # pull-style collectors (encoder, vive_tracker): join the reader
        # thread this module owns, then close the writer this module owns.
        for logical_id, thread in list(self._pull_threads.items()):
            thread.join(timeout=5.0)
            writer = self._pull_writers.get(logical_id)
            if writer is not None:
                try:
                    writer.close(clean=clean)
                except Exception as exc:
                    self._log_event("writer_close_error", details={"logical_id": logical_id, "error": str(exc)})

        # self-managed collectors (the 4 IMUs): .stop() drains their own
        # internal queue + writer.
        for sensor_id in sorted(_IMU_SENSOR_IDS):
            collector = self._collectors.get(sensor_id)
            if collector is None or sensor_id in self._pull_threads:
                continue
            try:
                collector.stop(clean=clean, timeout=10.0)
            except Exception as exc:
                self._log_event("collector_stop_error", details={"logical_id": sensor_id, "error": str(exc)})

        if self._health_thread is not None:
            self._health_stop_event.set()
            self._health_thread.join(timeout=5.0)
            if self._health_writer is not None:
                self._health_writer.close(clean=clean)

        self._log_event("session_ending", details={"interrupted": interrupted})
        if self._fallback_event_writer is not None:
            self._fallback_event_writer.close(clean=clean)
        elif self._events_collector is not None:
            stop_fn = getattr(self._events_collector, "stop", None) or getattr(self._events_collector, "close", None)
            if stop_fn is not None:
                try:
                    stop_fn(clean=clean)
                except TypeError:
                    try:
                        stop_fn()
                    except Exception:
                        pass
                except Exception:
                    pass

        self._write_manifest(session_complete=clean, interrupted=interrupted)
        self._write_session_summary(writer_pending_snapshot)
        self.interrupted = interrupted

    # ------------------------------------------------------------------
    # session_summary.json (Task 3, master spec section 13)
    # ------------------------------------------------------------------
    def _write_session_summary(self, writer_pending_snapshot: dict[str, Optional[int]]) -> None:
        """Per-stream: record count, valid count, first/last timestamp,
        actual rate, writer lag at shutdown. Built entirely from what each
        collector already exposes (IMURS232Collector.stats(), the
        JsonlWriter each collector already owns, and the cheap per-record
        bookkeeping `_pull_reader_loop` already does) -- no new stats
        pipeline. Anything genuinely not tracked at this layer is `null`
        with a comment, never fabricated (core/schemas.py's own
        convention). Called after every collector has been stopped/joined,
        so counts are final; `writer_pending_snapshot` was taken earlier,
        before the drain -- see shutdown()."""
        duration_s = None
        if self._start_monotonic_ns is not None:
            duration_s = (schemas.monotonic_ns() - self._start_monotonic_ns) / 1e9

        def _rate(count: Optional[int]) -> Optional[float]:
            # Coarse session-wide average, not a filtered/instantaneous
            # rate -- ponytail: a real per-window rate estimate would need
            # its own rolling-window pipeline; this is the cheap, honest
            # version until that's actually needed.
            if count is None or not duration_s or duration_s <= 0:
                return None
            return count / duration_s

        streams: dict[str, dict] = {}

        for sensor_id in sorted(_IMU_SENSOR_IDS):
            collector = self._collectors.get(sensor_id)
            if collector is None or sensor_id in self._pull_threads:
                continue
            collector_stats = collector.stats() if hasattr(collector, "stats") else {}
            record_count = collector_stats.get("frames_emitted")
            streams[sensor_id] = {
                "record_count": record_count,
                # collect_imu_rs232.py always emits valid=True (no invalid
                # path in that collector today) but doesn't expose a
                # separate valid-count stat -- not fabricated here.
                "valid_count": None,
                # Self-managed collector owns its own writer internally;
                # session_controller never sees individual IMU records, so
                # first/last sample timestamp isn't available at this layer.
                "first_t_pc_rx_ns": None,
                "last_t_pc_rx_ns": None,
                "packet_gap_count": None,  # would need a per-stream expected-rate model beyond what's exposed today
                "actual_rate_hz": _rate(record_count),
                "writer_pending_at_shutdown": writer_pending_snapshot.get(sensor_id),
                "resync_count": collector_stats.get("resync_count"),
                "disconnect_count": collector_stats.get("disconnect_count"),
            }

        for logical_id in self._pull_writers:
            rec = self._pull_stream_stats.get(logical_id, {})
            record_count = rec.get("count")
            streams[logical_id] = {
                "record_count": record_count,
                "valid_count": rec.get("valid_count"),
                "first_t_pc_rx_ns": rec.get("first_t_pc_rx_ns"),
                "last_t_pc_rx_ns": rec.get("last_t_pc_rx_ns"),
                "packet_gap_count": None,  # same as above -- no expected-rate model plumbed through yet
                "actual_rate_hz": _rate(record_count),
                "writer_pending_at_shutdown": writer_pending_snapshot.get(logical_id),
            }

        summary = {
            "session_id": self.session_id,
            "start_utc": self.start_utc,
            "end_utc": schemas.utc_iso_now(),
            "session_duration_s": duration_s,
            "streams": streams,
        }
        summary_path = self.session_dir / "session_summary.json"
        tmp_path = summary_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, summary_path)

    # ------------------------------------------------------------------
    # mode-specific setup
    # ------------------------------------------------------------------
    def _run_inventory(self) -> None:
        from core.port_discovery import list_ports

        ports = list_ports()
        report_path = self.reports_dir / "port_inventory.json"
        report_path.write_text(json.dumps(ports, indent=2, ensure_ascii=False), encoding="utf-8")
        self._log_event("port_inventory_written", details={"port_count": len(ports)})

    def _check_experiment_gate(self) -> bool:
        """Requires BOTH a config flag and the CLI flag (master spec
        section 22). Delegates to `experiment/safety_gate.
        hardware_motion_allowed` when importable; a fail-closed inline
        fallback (same two-flag AND) covers the window before that module
        existed -- see report."""
        config_flag = os.environ.get("SO101_ALLOW_EXPERIMENT_MODE") == "1"
        try:
            from experiment.safety_gate import hardware_motion_allowed
        except Exception:
            return bool(config_flag and self.experiment_cli_flag)
        return bool(hardware_motion_allowed(config_flag, self.experiment_cli_flag))

    def _start_all_collectors(self, is_mock: bool) -> None:
        # Real-mode port resolution (Task 1): matched once, up front, and
        # reused by every _start_imu call below -- match_sensor_map's own
        # first-match-wins/no-port-reuse behavior needs one call over the
        # whole sensor_map, not one call per sensor_id.
        self._matched_ports = {}
        if not is_mock and self._sensor_map:
            from core.port_discovery import list_ports, match_sensor_map

            self._matched_ports = match_sensor_map(list_ports(), self._sensor_map)

        if IMURS232Collector is not None:
            for sensor_id in sorted(_IMU_SENSOR_IDS):
                self._start_imu(sensor_id, is_mock)
        else:
            for sensor_id in sorted(_IMU_SENSOR_IDS):
                self._mark_unavailable(sensor_id, "collectors.collect_imu_rs232", "module not importable")

        if Esp32EncoderCollector is not None:
            self._start_encoder(is_mock)
        else:
            self._mark_unavailable("encoder", "collectors.collect_esp32_encoder", "module not importable")

        if ViveTrackerCollector is not None:
            self._start_vive(is_mock)
        else:
            self._mark_unavailable("vive_tracker", "collectors.collect_vive_tracker", "module not importable")

        self._release_start_barrier()

    def _start_imu(self, sensor_id: str, is_mock: bool) -> None:
        if not is_mock:
            # passive/experiment: real port resolution via
            # core.port_discovery.match_sensor_map, gated on sensor_map_path
            # actually being configured -- never a guessed COM port.
            if not self._sensor_map:
                self._mark_unavailable(sensor_id, "collectors.collect_imu_rs232", "no sensor_map configured")
                return
            matched = self._matched_ports.get(sensor_id)
            if not matched or not matched.get("device"):
                self._mark_unavailable(
                    sensor_id, "collectors.collect_imu_rs232", f"sensor_map has no matching port for {sensor_id!r}"
                )
                return
            try:
                collector = IMURS232Collector(
                    sensor_id=sensor_id,
                    session_id=self.session_id,
                    segment_id=self.segment_id,
                    output_dir=str(self.raw_dir),
                    port=matched["device"],
                    baud=self.imu_baud,
                    event_bus=self.event_bus,
                )
            except Exception as exc:
                self._mark_unavailable(sensor_id, "collectors.collect_imu_rs232", exc)
                return
            self._register_started_imu(sensor_id, collector)
            return

        output_rate_hz = 100.0
        mock_source = MockWT901CImu(sensor_id=sensor_id, output_rate_hz=output_rate_hz, seed=abs(hash(sensor_id)) % 10_000)
        self._mock_sources[sensor_id] = mock_source  # keep a reference alive for the lifetime of the collector
        # 4 frame types (accel/gyro/angle/quaternion) per sample tick -- see
        # mocks/mock_imu.py's next_packet_bytes docstring -- so pace at
        # output_rate_hz * 4 frames/s, matching IMURS232Collector's own
        # queue_expected_rate_hz=400.0 default assumption. See _paced().
        paced_byte_source = _paced(mock_source.next_packet_bytes, period_s=1.0 / (output_rate_hz * 4))
        try:
            collector = IMURS232Collector(
                sensor_id=sensor_id,
                session_id=self.session_id,
                segment_id=self.segment_id,
                output_dir=str(self.raw_dir),
                byte_source=paced_byte_source,
                configured_output_rate_hz=output_rate_hz,
                event_bus=self.event_bus,
            )
        except Exception as exc:
            self._mark_unavailable(sensor_id, "collectors.collect_imu_rs232", exc)
            return

        self._register_started_imu(sensor_id, collector)

    def _register_started_imu(self, sensor_id: str, collector: Any) -> None:
        """Common tail for both the mock and real IMU paths: record it as
        started, register its queue with health, and defer the actual
        `.start()` call into the start barrier (Task 5) instead of firing
        it immediately -- see _release_start_barrier."""
        self._collectors[sensor_id] = collector
        self._collector_status[sensor_id] = "started"
        queue_obj = getattr(collector, "queue", None)
        if isinstance(queue_obj, BoundedQueue):
            self.health.register_queue(sensor_id, queue_obj)
        self._pending_launches.append(collector.start)

    def _start_encoder(self, is_mock: bool) -> None:
        logical_id = "encoder"
        if not is_mock:
            # --- real-mode integration point (Task 1) ------------------
            # The real ESP32 collector for 03_SO-101_control/main's
            # STATE-poll + D2/D3/D5 legacy command interface (3 joints) is
            # collectors.collect_esp32_state_poll.Esp32StatePollCollector
            # (built by a concurrent workstream, not this task -- do not
            # touch that file). It is NOT collectors.collect_esp32_encoder.
            # Esp32EncoderCollector, which targets 05_SO-101_rtos_motor_
            # driver's different (and not-actually-used) STREAM protocol.
            #
            # `encoder_collector_factory`, if given, always wins (full
            # override -- must return an object with pull-style
            # `.read_record() -> dict`, the exact contract `_wire_pull_reader`
            # already drives for vive_tracker below, and an optional
            # `.start()`). Without one, this defaults to constructing the
            # real Esp32StatePollCollector itself, gated on sensor_map
            # having a matched `esp32_encoder:` port (same match_sensor_map
            # call _start_all_collectors already ran for the IMUs) -- never
            # a guessed COM port. Esp32StatePollCollector's own method is
            # `poll_once(**context) -> dict`, not `read_record()`, so
            # `_Esp32StatePollReadRecordAdapter` below bridges the two
            # (and forwards this controller's current target context
            # straight into `poll_once`'s own `**context`, natively --
            # see set_target_context's docstring).
            if self._encoder_collector_factory is not None:
                try:
                    collector = self._encoder_collector_factory(self)
                except Exception as exc:
                    self._mark_unavailable(logical_id, "encoder_collector_factory", exc)
                    return
                self._wire_pull_reader(logical_id, collector, schemas.STREAM_ENCODER, period_s=0.1)
                return

            if not self._sensor_map:
                self._mark_unavailable(logical_id, "collectors.collect_esp32_state_poll", "no sensor_map configured")
                return
            matched = self._matched_ports.get(logical_id) or self._matched_ports.get("esp32_encoder")
            if not matched or not matched.get("device"):
                self._mark_unavailable(
                    logical_id, "collectors.collect_esp32_state_poll", "sensor_map has no matching port for 'esp32_encoder'"
                )
                return
            try:
                import serial  # local import -- keep this file importable without pyserial's hardware backend touched, matches _open_real_serial's pattern
                from collectors.collect_esp32_state_poll import Esp32StatePollCollector

                conn = serial.Serial(matched["device"], DEFAULT_REAL_ENCODER_BAUD, timeout=0.3)
                poll_collector = Esp32StatePollCollector(conn, self.session_id, self.segment_id)
                collector = _Esp32StatePollReadRecordAdapter(poll_collector, self)
            except Exception as exc:
                self._mark_unavailable(logical_id, "collectors.collect_esp32_state_poll", exc)
                return
            self._wire_pull_reader(logical_id, collector, schemas.STREAM_ENCODER, period_s=0.1)
            return

        mock_source = MockESP32Encoder(period_ms=100, seed=100)
        self._mock_sources[logical_id] = mock_source
        try:
            collector = Esp32EncoderCollector(mock_source, self.session_id, self.segment_id, period_ms=100)
        except Exception as exc:
            self._mark_unavailable(logical_id, "collectors.collect_esp32_encoder", exc)
            return

        self._wire_pull_reader(logical_id, collector, schemas.STREAM_ENCODER, period_s=0.1)

    def _start_vive(self, is_mock: bool) -> None:
        logical_id = "vive_tracker"
        if not is_mock:
            # Real mode: OpenVR doesn't go through a serial COM port at
            # all (SteamVR runtime does the device I/O), so there's no
            # "guessed port" risk to fail closed against the way IMU/
            # encoder have -- this always attempts to construct the real
            # source. `tracker_serial` (config-driven tracker selection,
            # section 7) comes from sensor_map's optional `vive_tracker:`
            # entry if given; omitted entirely is fine too --
            # OpenVRTrackerSource falls back to first-found + a logged
            # warning rather than hard-failing (see its docstring).
            tracker_serial = None
            if self._sensor_map:
                vive_cfg = self._sensor_map.get(logical_id) or {}
                tracker_serial = vive_cfg.get("tracker_serial") or vive_cfg.get("serial_number")
            try:
                from collectors.collect_vive_tracker import OpenVRTrackerSource

                source = OpenVRTrackerSource(
                    tracker_serial=tracker_serial,
                    on_warning=lambda msg: self._log_event("vive_tracker_selection_warning", message=msg),
                )
                collector = ViveTrackerCollector(
                    source,
                    self.session_id,
                    self.segment_id,
                    on_error=lambda exc: self._log_event("vive_read_error", details={"error": str(exc)}),
                )
            except Exception as exc:
                self._mark_unavailable(logical_id, "collectors.collect_vive_tracker", exc)
                return
            # period_s: see REAL_VIVE_POLL_HZ's module-level comment --
            # a starting point, not a measured achievable rate.
            self._wire_pull_reader(logical_id, collector, schemas.STREAM_VIVE, period_s=1.0 / REAL_VIVE_POLL_HZ)
            return

        mock_source = MockViveTracker(seed=200)
        self._mock_sources[logical_id] = mock_source
        try:
            collector = ViveTrackerCollector(
                mock_source,
                self.session_id,
                self.segment_id,
                on_error=lambda exc: self._log_event("vive_read_error", details={"error": str(exc)}),
            )
        except Exception as exc:
            self._mark_unavailable(logical_id, "collectors.collect_vive_tracker", exc)
            return

        self._wire_pull_reader(logical_id, collector, schemas.STREAM_VIVE, period_s=1.0 / 90.0)

    def _wire_pull_reader(self, logical_id: str, collector: Any, stream_name: str, period_s: Optional[float]) -> None:
        writer = JsonlWriter(str(self.raw_dir), stream_name, self.segment_id)
        self._pull_writers[logical_id] = writer
        nominal_capacity = max(int((1.0 / period_s if period_s else 100.0) * 10), 10)
        self.health.register_queue(logical_id, _WriterQueueDepthAdapter(writer, nominal_capacity))

        thread = threading.Thread(
            target=self._pull_reader_loop,
            args=(logical_id, collector, writer, period_s),
            name=f"collector-{logical_id}",
            daemon=True,
        )
        self._pull_threads[logical_id] = thread
        self._collectors[logical_id] = collector
        self._collector_status[logical_id] = "started"

        # Deferred into the start barrier (Task 5) instead of starting
        # immediately: `collector.start()` (optional -- Vive's collector
        # has none) then the reader thread itself, bundled as one launch
        # so both fire together with every other collector's launch.
        def _launch(c: Any = collector, t: threading.Thread = thread) -> None:
            if hasattr(c, "start"):
                c.start()
            t.start()

        self._pending_launches.append(_launch)

    def _release_start_barrier(self) -> None:
        """Task 5: common start barrier. Every collector that actually got
        far enough to be constructed (i.e. was NOT `_mark_unavailable`'d)
        has its real "begin producing data" call queued in
        `self._pending_launches` instead of firing immediately -- see
        `_register_started_imu` / `_wire_pull_reader`. This fires them all
        at once: one short-lived thread per launch, all rendezvousing on a
        single `threading.Barrier` sized to exactly how many are actually
        starting, so a sensor that never got this far can't block the
        rest. `timeout=10.0` is a safety net only (in practice every gate
        thread reaches the barrier within milliseconds of being spawned,
        with no other work in between) -- if it's ever hit, the broken
        barrier still lets every already-waiting launch proceed
        individually rather than deadlocking the session.

        This method itself blocks (joining every gate thread, bounded by
        `join_timeout_s`) until every launch has actually fired, so
        `start()` never returns to its caller while a collector's own
        `.start()`/pull-thread-`.start()` call is still in flight -- e.g.
        `shutdown()` calling `thread.join()` on a pull-reader thread that
        hadn't been `.start()`-ed yet would raise `RuntimeError: cannot
        join thread before it is started`. Blocking here does not change
        `start()`'s existing blocking behavior: every launch was already a
        direct, synchronous call before this task (Task 5 only added the
        barrier rendezvous in between)."""
        launches = self._pending_launches
        self._pending_launches = []
        n = len(launches)
        if n == 0:
            return
        barrier = threading.Barrier(n)

        def _gated_launch(start_fn) -> None:
            try:
                barrier.wait(timeout=10.0)
            except threading.BrokenBarrierError:
                pass  # a peer failed to arrive -- proceed anyway rather than deadlock the session
            try:
                start_fn()
            except Exception as exc:
                self._log_event("collector_start_error", details={"error": str(exc)})

        gate_threads = []
        for start_fn in launches:
            t = threading.Thread(target=_gated_launch, args=(start_fn,), name="collector-start-gate", daemon=True)
            gate_threads.append(t)
            t.start()
        for t in gate_threads:
            t.join(timeout=15.0)

    def _pull_reader_loop(self, logical_id: str, collector: Any, writer: JsonlWriter, period_s: Optional[float]) -> None:
        shutdown_event = interrupt_handler.shutdown_requested
        stats = self._pull_stream_stats.setdefault(
            logical_id, {"count": 0, "valid_count": 0, "first_t_pc_rx_ns": None, "last_t_pc_rx_ns": None}
        )
        while not shutdown_event.is_set():
            try:
                record = collector.read_record()
            except Exception as exc:
                self._log_event("collector_read_error", details={"logical_id": logical_id, "error": str(exc)})
                break

            # Task 4: stamp per-target context, never clobbering a field
            # the collector already set itself.
            ctx = self._current_target_context
            if ctx:
                for key, value in ctx.items():
                    if value is not None and record.get(key) is None:
                        record[key] = value

            # Task 3 raw material: cheap per-record bookkeeping, no new
            # stats pipeline -- record is already in hand here.
            stats["count"] += 1
            if record.get("valid"):
                stats["valid_count"] += 1
            t_pc_rx_ns = record.get("t_pc_rx_ns")
            if t_pc_rx_ns is not None:
                if stats["first_t_pc_rx_ns"] is None:
                    stats["first_t_pc_rx_ns"] = t_pc_rx_ns
                stats["last_t_pc_rx_ns"] = t_pc_rx_ns

            try:
                writer.put(record)
            except RuntimeError:
                break  # writer already closed -- shutdown raced ahead of this put
            if period_s:
                shutdown_event.wait(period_s)

    def _mark_unavailable(self, logical_id: str, module_name: str, error: Any) -> None:
        status = f"unavailable ({module_name}): {error}"
        self._collector_status[logical_id] = status
        self._log_event(
            "collector_unavailable",
            message=f"{logical_id} not started",
            details={"logical_id": logical_id, "module": module_name, "error": str(error)},
        )

    # ------------------------------------------------------------------
    # health (persists core.health_supervisor.HealthSupervisor.sample())
    # ------------------------------------------------------------------
    def _start_health(self) -> None:
        self._health_writer = JsonlWriter(str(self.raw_dir), schemas.STREAM_HEALTH, self.segment_id)
        self._health_stop_event = threading.Event()
        self._health_thread = threading.Thread(target=self._health_loop, name="health-writer", daemon=True)
        self._collector_status["health"] = "started"
        self._health_thread.start()

    def _health_loop(self) -> None:
        """A failed health sample/put is real data loss -- it must never be
        swallowed silently. The loop still keeps running (one bad sample
        must not kill health monitoring for the rest of the session), but
        the failure is counted and surfaced in `manifest.collector_status`
        and as an event record."""
        failures = 0
        while True:
            try:
                record = self.health.sample()
                self._health_writer.put(record)
            except Exception as exc:
                failures += 1
                self._collector_status["health"] = f"started ({failures} sample/write failure(s), last: {exc})"
                if failures == 1:  # first only -- a broken writer would otherwise flood the event stream
                    self._log_event(
                        "health_sample_error",
                        message="health record could not be written",
                        details={"error": str(exc)},
                    )
            if self._health_stop_event.wait(self.health.interval_s):
                break

    # ------------------------------------------------------------------
    # events (collectors/collect_events.py if present, else a built-in
    # fallback -- see module docstring)
    # ------------------------------------------------------------------
    def _start_event_logger(self) -> None:
        try:
            from collectors.collect_events import EventCollector

            collector = EventCollector(
                session_id=self.session_id,
                segment_id=self.segment_id,
                output_dir=str(self.raw_dir),
                event_bus=self.event_bus,
            )
            if hasattr(collector, "start"):
                collector.start()
            log_fn = getattr(collector, "log_event", None) or getattr(collector, "log", None)
            if log_fn is None:
                raise AttributeError("EventCollector exposes neither log_event() nor log()")
        except Exception as exc:
            self._collector_status["events"] = f"unavailable (collectors.collect_events): {exc} -> builtin fallback event logger"
            self._fallback_event_writer = JsonlWriter(str(self.raw_dir), schemas.STREAM_EVENT, self.segment_id)
            self._event_log_fn = self._fallback_log_event
            return

        self._events_collector = collector
        self._collector_status["events"] = "started"
        self._event_log_fn = log_fn

    def _fallback_log_event(self, event_type: str, message: Optional[str] = None, details: Optional[dict] = None) -> None:
        self._event_seq += 1
        base = schemas.base_record(
            schema=schemas.SCHEMA_EVENT,
            session_id=self.session_id,
            segment_id=self.segment_id,
            stream=schemas.STREAM_EVENT,
            host_seq=self._event_seq,
            t_pc_rx_ns=schemas.monotonic_ns(),
            valid=True,
            parse_status="ok",
        )
        rec = schemas.event_record(base, event_type=event_type, message=message, details=details)
        self._fallback_event_writer.put(rec)

    def _log_event(self, event_type: str, message: Optional[str] = None, details: Optional[dict] = None) -> None:
        fn = self._event_log_fn
        if fn is None:
            return
        try:
            fn(event_type, message=message, details=details)
        except TypeError:
            try:
                fn(event_type)
            except Exception:
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # manifest (master spec section 8)
    # ------------------------------------------------------------------
    def _get_git_commit_hash(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.capstone_root),
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    def _write_manifest(self, *, session_complete: bool, interrupted: bool) -> None:
        # "events"/"health" are always attempted (real collector or
        # fallback writer) so they belong in enabled_streams unconditionally;
        # everything else only if it actually started (self._collectors).
        enabled_streams = sorted(set(self._collectors.keys()) | {"events", "health"})
        manifest = {
            "session_id": self.session_id,
            "start_utc": self.start_utc,
            "end_utc": schemas.utc_iso_now() if (session_complete or interrupted) else None,
            # Not "workspace_mode": that name is reserved (core/schemas.py,
            # master spec sections 9/21.5) for the azimuth-sector target
            # mode (pilot/main/front_120/front_180_legacy). This is the
            # session *execution* mode (inventory/mock/passive/soak/
            # experiment, master spec section 22) -- a different axis.
            "execution_mode": self.mode,
            "git_commit_hash": self.git_commit_hash,
            "data_schema_version": DATA_SCHEMA_VERSION,
            "session_complete": session_complete,
            "interrupted": interrupted,
            "segment_id": self.segment_id,
            "enabled_streams": enabled_streams,
            "collector_status": dict(self._collector_status),
        }
        tmp_path = self.manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, self.manifest_path)
