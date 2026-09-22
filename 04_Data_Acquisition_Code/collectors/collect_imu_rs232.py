"""Per-IMU WT901C-RS232 raw collector (master spec section 7).

Protocol layer (`extract_frames`/`s16`/`decode_frame`/`FRAME_NAMES`) is
reimplemented -- not imported across package boundaries -- from
`07_WT901C/WT901C_imu0/collect_4imu.py` (commit
`3a0c757e913454ade5007cbacbb52bae48f321a0`), audited REUSABLE_AFTER_FIX in
`docs/07_WT901C_CODE_AUDIT.md` / `DECISIONS.md` #4. Same header (`0x55`),
frame length (11 bytes), 8-bit-sum-mod-256 checksum, resync-by-discarding-
one-byte-and-retrying behavior, and full-scale-range scaling formulas. Unit
conversion differs on purpose: `collect_4imu.py` emits g / deg-per-second /
degrees for a human-readable CSV; this module returns SI units (m/s^2,
rad/s, rad) to match `core/schemas.py`'s `imu_record` contract, and reorders
the quaternion from the wire's (w,x,y,z) to the schema's
`quaternion_xyzw=(x,y,z,w)`.

Reconnect design (reader thread owns the connection object, retries with a
1s backoff, stamps a `connection_epoch`) is carried from
`07_WT901C/WT901C_imu0/live_4imu_visualizer.py`'s `SensorReader` (same
commit, also REUSABLE_AFTER_FIX) -- but FIXED per that audit's one flagged
gap: there, `self.serial` was written by the reader thread and read/closed
by the Tk main thread with no lock (a narrow cross-thread race). Here the
open connection is a plain local variable inside `_reader_loop`; no other
thread ever touches it, so the race is eliminated structurally instead of
papered over with a lock.

Byte source (pick exactly one at construction):
  - `port="COM5"` (+ `baud`, `serial_timeout_s`) -- real hardware, opens
    `serial.Serial` lazily inside the reader thread.
  - `byte_source=some_zero_arg_callable` -- test hook. Called with no
    arguments, returns the next available bytes (need not be a whole
    frame), may raise to simulate a dropped connection. Matches
    `mocks/mock_imu.py`'s `next_packet_bytes` directly -- including
    anything `mocks/fault_injector.py` wraps around it -- no adapter needed:
    `IMURS232Collector(..., byte_source=mock.next_packet_bytes)`.
  - `connection_factory=some_zero_arg_callable` -- advanced hook for a fake
    object that already implements `.read(n)`/`.close()` itself.

Architecture: reader thread (serial/byte-source I/O + frame extraction +
decode) -> `core.bounded_queue.BoundedQueue` (never blocks; overflow is
tracked, never silently dropped) -> pump thread -> `core.jsonl_writer.
JsonlWriter` (owns its own writer thread + flush/fsync/drain-before-close).
No pandas/matplotlib/filtering/resampling anywhere in this loop, per master
spec section 7's explicit prohibition list.
"""
from __future__ import annotations

import math
import os
import queue
import struct
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Optional

# Same pattern as the sibling collectors -- without it the `__main__`
# self-check below can't run standalone (`python collectors/collect_imu_rs232.py`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import schemas  # noqa: E402
from core.bounded_queue import BoundedQueue
from core.event_bus import EventBus
from core.jsonl_writer import JsonlWriter

VALID_SENSOR_IDS = {"imu0_base", "imu1_link_root", "imu2_link_mid", "imu3_tip"}

# --- WT901C wire protocol (07_WT901C/WT901C_imu0/collect_4imu.py, commit
# 3a0c757e913454ade5007cbacbb52bae48f321a0) ---------------------------------
HEADER = 0x55
FRAME_LENGTH = 11  # header(1) + type(1) + payload(8) + checksum(1)

FRAME_TIME = 0x50
FRAME_ACCEL = 0x51
FRAME_GYRO = 0x52
FRAME_ANGLE = 0x53
FRAME_MAGNETIC = 0x54
FRAME_PORT_STATUS = 0x55
FRAME_PRESSURE = 0x56
FRAME_GPS = 0x57
FRAME_VELOCITY = 0x58
FRAME_QUATERNION = 0x59

FRAME_NAMES = {
    FRAME_TIME: "time",
    FRAME_ACCEL: "acceleration",
    FRAME_GYRO: "angular_velocity",
    FRAME_ANGLE: "angle",
    FRAME_MAGNETIC: "magnetic_field",
    FRAME_PORT_STATUS: "port_status",
    FRAME_PRESSURE: "pressure",
    FRAME_GPS: "gps",
    FRAME_VELOCITY: "velocity",
    FRAME_QUATERNION: "quaternion",
}

STANDARD_GRAVITY_MPS2 = 9.80665  # g -> m/s^2, ISO 80000-3 standard gravity (not sensor-specific)


@dataclass(frozen=True)
class IMUScaleConfig:
    """Full-scale-range constants used to convert raw int16 counts to
    physical units. Defaults carried over from `collect_4imu.py:decode_frame`
    (same commit as above) -- per `docs/07_WT901C_CODE_AUDIT.md` these were
    NEVER confirmed against a real WT901C's configured range register.

    UNVERIFIED_HARDWARE_SETTING: if a physical unit is configured for a
    different accel/gyro/angle full-scale range, these three constants must
    be updated to match, or every `accel_mps2`/`gyro_rads`/`euler_rad` value
    this module emits is silently wrong (still self-consistent, just off by
    whatever ratio the real range differs by).
    """

    accel_full_scale_g: float = 16.0
    gyro_full_scale_dps: float = 2000.0
    angle_full_scale_deg: float = 180.0


DEFAULT_SCALE = IMUScaleConfig()


def extract_frames(
    buffer: bytearray, discarded_carry: int = 0
) -> tuple[list[tuple[bytes, bool]], int]:
    """Reimplemented from `collect_4imu.py:extract_frames` (same commit as
    module docstring) -- identical control flow: find `HEADER`, skip
    leading garbage, validate the 8-bit-sum-mod-256 checksum over an
    11-byte candidate, discard exactly one byte and retry on a checksum
    mismatch, break (leaving partial trailing bytes buffered) when fewer
    than `FRAME_LENGTH` bytes remain. Mutates `buffer` in place.

    Adds resync bookkeeping on top of the original (needed for this
    package's `quality_flags` contract, not present in the source file):
    returns `(frames, discarded_carry)` where `frames` is a list of
    `(frame_bytes, resynced)` pairs -- `resynced=True` means at least one
    byte (header-hunt skip and/or checksum-failure single-byte discard) was
    thrown away since the previous frame. `discarded_carry` is the leftover
    discard count not yet attributed to any frame in this call (e.g. a
    checksum failure right at the end of the buffer, before enough bytes
    for the next candidate have arrived) -- pass it back in on the next
    call so a resync spanning a `read()`/chunk boundary still gets
    attributed to the frame that follows it.
    """
    frames: list[tuple[bytes, bool]] = []
    discarded = discarded_carry
    while len(buffer) >= FRAME_LENGTH:
        try:
            start = buffer.index(HEADER)
        except ValueError:
            discarded += len(buffer)
            buffer.clear()
            break
        if start:
            del buffer[:start]
            discarded += start
        if len(buffer) < FRAME_LENGTH:
            break
        candidate = bytes(buffer[:FRAME_LENGTH])
        if (sum(candidate[: FRAME_LENGTH - 1]) & 0xFF) == candidate[FRAME_LENGTH - 1]:
            frames.append((candidate, discarded > 0))
            discarded = 0
            del buffer[:FRAME_LENGTH]
        else:
            del buffer[0]
            discarded += 1
    return frames, discarded


def s16(low: int, high: int) -> int:
    """Little-endian signed int16 reconstruction. Reimplemented from
    `collect_4imu.py:s16` (same commit as module docstring)."""
    return struct.unpack("<h", bytes((low, high)))[0]


def decode_frame(frame: bytes, scale: IMUScaleConfig = DEFAULT_SCALE) -> dict:
    """Decode one validated 11-byte WT901C frame. See module docstring for
    the source this is reimplemented from and the unit differences."""
    kind = frame[1]
    a = s16(frame[2], frame[3])
    b = s16(frame[4], frame[5])
    c = s16(frame[6], frame[7])
    d = s16(frame[8], frame[9])

    out: dict = {
        "frame_type": kind,
        "frame_name": FRAME_NAMES.get(kind, "unknown"),
        "packet_type": FRAME_NAMES.get(kind, "unknown_0x{:02X}".format(kind)),
    }
    if kind == FRAME_ACCEL:
        out["accel_raw"] = [a, b, c]
        out["accel_mps2"] = [
            v / 32768.0 * scale.accel_full_scale_g * STANDARD_GRAVITY_MPS2 for v in (a, b, c)
        ]
        out["temperature_c"] = d / 100.0
    elif kind == FRAME_GYRO:
        out["gyro_raw"] = [a, b, c]
        out["gyro_rads"] = [math.radians(v / 32768.0 * scale.gyro_full_scale_dps) for v in (a, b, c)]
        out["temperature_c"] = d / 100.0
    elif kind == FRAME_ANGLE:
        out["euler_rad"] = [math.radians(v / 32768.0 * scale.angle_full_scale_deg) for v in (a, b, c)]
        # Word 4 (bytes 8-9) is NOT temperature for this packet -- confirmed
        # both by WitMotion's own published serial protocol (angle-packet
        # byte 8-9 = "VL/VH", Version=(VH<<8)|VL) and by real hardware: in
        # `sessions/real_motion_demo/imu*_segment{006,007,012}.jsonl` this
        # word is the exact same constant (19582) across all 95856 angle
        # records, all 4 independently-wired physical sensors, and all
        # segments -- while accel/gyro `temperature_c` from the same dies at
        # the same instants reads a normal, drifting 26-29C. A real
        # temperature can't be bit-identical across 4 separate sensors; a
        # firmware/protocol version number can. See
        # docs/IMU_DATA_COVERAGE.md for the full investigation. Kept raw and
        # honestly unlabeled instead of the previous (wrong) `temperature_c`.
        out["angle_packet_word4_raw"] = d
    elif kind == FRAME_QUATERNION:
        qw, qx, qy, qz = a / 32768.0, b / 32768.0, c / 32768.0, d / 32768.0
        out["quaternion_xyzw"] = [qx, qy, qz, qw]
    elif kind == FRAME_MAGNETIC:
        out["mag_raw"] = [a, b, c]
        # No protocol-documented raw-count -> uT scale factor exists (it's
        # calibration-dependent on real hardware) -- store raw counts only,
        # never invent a scale factor (mag_uT left unset/None).
        #
        # Word 4 (bytes 8-9): WitMotion's published serial protocol labels
        # this "TL/TH" = temperature, same convention as accel/gyro. But on
        # the real WT901C-RS232 units in this project it is EXACTLY 0 in
        # all 50074 magnetic-field records checked (12 real_motion_demo
        # files x 4 sensors x segments 006/007/012) -- never once tracking
        # the ~26-29C accel/gyro temperature read from the same die at the
        # same instant. That contradicts it actually being a live
        # temperature reading here, so it is NOT decoded as temperature_c;
        # kept as an honestly-separate raw field instead. See
        # docs/IMU_DATA_COVERAGE.md.
        out["mag_packet_word4_raw"] = d
    elif kind == FRAME_TIME:
        # WitMotion time packet is NOT 4x int16 like the other frame types --
        # it's raw calendar bytes: YY MM DD HH MN SS MSL MSH (confirmed from
        # WitMotion's own published serial protocol doc; see module
        # docstring for other protocol-layer sourcing). `frame[2:8]` are
        # taken as literal byte values (not s16 pairs -- `a`/`b`/`c` above do
        # not apply here), `ms` is an UNSIGNED little-endian 16-bit word
        # (0-999), unlike every other frame type's signed int16 words.
        # UNVERIFIED_HARDWARE_SETTING: no 0x50 frame appears anywhere in
        # `sessions/real_motion_demo/` (the device apparently never emits
        # this packet type in the captured sessions), so this decode has
        # never been cross-checked against a real sample -- only against
        # the protocol doc and a synthetic frame (tests/test_wt901_parser.py).
        yy, mm, dd, hh, mn, ss = frame[2], frame[3], frame[4], frame[5], frame[6], frame[7]
        ms = frame[8] | (frame[9] << 8)
        # "20{yy}" assumes the standard embedded-RTC 2-digit-year-since-2000
        # convention (never explicitly confirmed in the WitMotion doc text
        # fetched for this task) -- also UNVERIFIED_HARDWARE_SETTING.
        out["t_sensor"] = f"20{yy:02d}-{mm:02d}-{dd:02d}T{hh:02d}:{mn:02d}:{ss:02d}.{ms:03d}"
    return out


class _CallableByteSource:
    """Adapts a zero-arg `next_bytes_fn` (e.g. `MockWT901CImu.
    next_packet_bytes`, or the same wrapped by `mocks/fault_injector.py`)
    into a `.read(n)`/`.close()` object so `_reader_loop` can treat a real
    `serial.Serial` and a test byte source identically. `next_bytes_fn` is
    called once whenever the internal buffer is empty; its output is doled
    out up to `n` bytes at a time. Any exception `next_bytes_fn` raises
    (e.g. `with_disconnect` -> `ConnectionError`) propagates out of
    `.read()` unchanged, so the normal disconnect/reconnect path handles it.
    """

    def __init__(self, next_bytes_fn: Callable[[], bytes]) -> None:
        self._next_bytes_fn = next_bytes_fn
        self._buf = bytearray()

    def read(self, n: int = 1) -> bytes:
        if not self._buf:
            self._buf.extend(self._next_bytes_fn())
        take = bytes(self._buf[:n])
        del self._buf[:n]
        return take

    def close(self) -> None:
        pass


class IMURS232Collector:
    """One instance per physical WT901C IMU (`sensor_id` one of
    `VALID_SENSOR_IDS`). See module docstring for architecture and byte
    source options."""

    def __init__(
        self,
        sensor_id: str,
        session_id: str,
        segment_id: int,
        output_dir: str,
        *,
        port: Optional[str] = None,
        baud: int = 9600,
        serial_timeout_s: float = 0.05,
        byte_source: Optional[Callable[[], bytes]] = None,
        connection_factory: Optional[Callable[[], object]] = None,
        scale: IMUScaleConfig = DEFAULT_SCALE,
        configured_output_rate_hz: Optional[float] = None,
        queue_expected_rate_hz: float = 400.0,
        queue_buffer_seconds: float = 10.0,
        reconnect_wait_s: float = 1.0,
        stale_run_length: int = 20,
        event_bus: Optional[EventBus] = None,
        on_record: Optional[Callable[[dict], None]] = None,
        flush_interval_s: float = 0.4,
        fsync_interval_s: float = 1.5,
    ) -> None:
        assert sensor_id in VALID_SENSOR_IDS, (
            f"unexpected sensor_id {sensor_id!r}, expected one of {sorted(VALID_SENSOR_IDS)}"
        )
        if sum(x is not None for x in (port, byte_source, connection_factory)) != 1:
            raise ValueError("provide exactly one of: port, byte_source, connection_factory")

        self.sensor_id = sensor_id
        self.session_id = session_id
        self.segment_id = segment_id
        self.port = port
        self.baud = baud
        self.serial_timeout_s = serial_timeout_s
        self.scale = scale
        self.configured_output_rate_hz = configured_output_rate_hz
        self.reconnect_wait_s = reconnect_wait_s
        self.stale_run_length = stale_run_length
        self.event_bus = event_bus
        # Optional live tap: called on the reader thread with each record, in
        # addition to (never instead of) the normal queue -> writer path, so a
        # consumer that needs samples in real time -- e.g. the adaptive
        # settling detector watching imu3_tip -- doesn't have to tail the
        # JSONL file or race the writer. Exceptions raised here are swallowed
        # (see _handle_frame): a misbehaving observer must not be able to kill
        # data capture.
        self.on_record = on_record

        if connection_factory is not None:
            self._connection_factory = connection_factory
        elif byte_source is not None:
            self._connection_factory = lambda: _CallableByteSource(byte_source)
        else:
            self._connection_factory = self._open_real_serial

        self.queue = BoundedQueue(queue_expected_rate_hz, queue_buffer_seconds)
        self.writer = JsonlWriter(
            output_dir,
            sensor_id,
            segment_id,
            flush_interval_s=flush_interval_s,
            fsync_interval_s=fsync_interval_s,
        )

        # Mutated only by the reader thread (see module docstring) -- safe
        # to read from any thread under CPython's GIL, same single-writer
        # pattern `collect_4imu.py` itself uses for `stats[port_name]`.
        self.connection_epoch = 0
        self.host_seq = 0
        self.frames_emitted = 0
        self.resync_count = 0
        self.disconnect_count = 0
        self.on_record_errors = 0
        self._last_frame_bytes: Optional[bytes] = None
        self._last_bytes_by_type: dict[int, bytes] = {}
        self._stale_run_by_type: dict[int, int] = {}

        self._stop_event = threading.Event()
        self._reader_done_event = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._pump_thread: Optional[threading.Thread] = None

    def _open_real_serial(self):
        import serial  # local import: only touched on the real-hardware path

        return serial.Serial(self.port, self.baud, timeout=self.serial_timeout_s)

    # --- lifecycle ----------------------------------------------------
    def start(self) -> None:
        self._stop_event.clear()
        self._reader_done_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name=f"imu-reader-{self.sensor_id}", daemon=True
        )
        self._pump_thread = threading.Thread(
            target=self._pump_loop, name=f"imu-pump-{self.sensor_id}", daemon=True
        )
        self._reader_thread.start()
        self._pump_thread.start()

    def stop(self, clean: bool = True, timeout: float = 5.0) -> None:
        """Stop the reader, then drain the pump thread, then close the
        writer. Ordering matters: `_reader_done_event` is only set AFTER
        the reader thread has fully joined, so the pump thread can't decide
        "queue looks empty, I'm done" while the reader is still mid-flight
        producing one more record -- see `_pump_loop`'s docstring."""
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout)
        self._reader_done_event.set()
        if self._pump_thread is not None:
            self._pump_thread.join(timeout)
        self.writer.close(clean=clean)

    def _publish(self, event_type: str, message: str) -> None:
        if self.event_bus is not None:
            self.event_bus.publish(
                event_type,
                {
                    "sensor_id": self.sensor_id,
                    "connection_epoch": self.connection_epoch,
                    "message": message,
                },
            )

    # --- reader thread --------------------------------------------------
    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            conn = None
            buffer = bytearray()
            discarded_carry = 0
            try:
                conn = self._connection_factory()
                self.connection_epoch += 1
                epoch = self.connection_epoch
                self._publish("sensor_connected", f"{self.sensor_id} connected (epoch {epoch})")
                while not self._stop_event.is_set():
                    # Approximate first-byte/frame-end bracketing timestamps
                    # (task allows "approximate, document how"): first_byte
                    # = start of the read() call that (at least) completed
                    # this batch of frames; frame_end = right after
                    # extraction/validation finished for that batch. A frame
                    # spanning multiple read()s, or bytes already sitting in
                    # `buffer` from an earlier read, means the true first
                    # byte could be slightly earlier -- bounded by one
                    # read-loop iteration's latency, negligible at this
                    # sensor's data rate. `t_pc_rx_ns` (the authoritative
                    # per-record timestamp) is taken individually per frame
                    # in `_handle_frame`, not approximated.
                    t_pc_first_byte_ns = schemas.monotonic_ns()
                    # Read whatever's currently available in one call
                    # (`SensorReader` pattern from live_4imu_visualizer.py,
                    # same commit) instead of collect_4imu.py's slower
                    # byte-at-a-time loop; falls back to 1 when the
                    # connection object doesn't expose `in_waiting` (e.g.
                    # `_CallableByteSource`).
                    n = getattr(conn, "in_waiting", 0) or 1
                    data = conn.read(max(1, n))
                    if not data:
                        continue
                    buffer.extend(data)
                    frames, discarded_carry = extract_frames(buffer, discarded_carry)
                    t_pc_frame_end_ns = schemas.monotonic_ns()
                    for frame_bytes, resynced in frames:
                        self._handle_frame(
                            frame_bytes, resynced, t_pc_first_byte_ns, t_pc_frame_end_ns, epoch
                        )
            except Exception as exc:  # noqa: BLE001 - any I/O failure means "reconnect"
                self.disconnect_count += 1
                self._publish("sensor_disconnected", f"{self.sensor_id} disconnected: {exc}")
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
            if not self._stop_event.is_set():
                self._stop_event.wait(self.reconnect_wait_s)

    def _handle_frame(
        self,
        frame_bytes: bytes,
        resynced: bool,
        t_pc_first_byte_ns: int,
        t_pc_frame_end_ns: int,
        epoch: int,
    ) -> None:
        t_pc_rx_ns = schemas.monotonic_ns()
        decoded = decode_frame(frame_bytes, self.scale)
        frame_type = decoded["frame_type"]

        quality_flags: list[str] = []
        if resynced:
            quality_flags.append("resynced")
            self.resync_count += 1

        # Single-step exact-repeat check (any frame type) -- catches a
        # transient duplicate (e.g. mocks/fault_injector.py's
        # with_duplicate_packets literally re-emitting the previous bytes).
        duplicate = self._last_frame_bytes is not None and frame_bytes == self._last_frame_bytes
        self._last_frame_bytes = frame_bytes

        # Sustained-freeze check, accel/gyro only (task's explicit wording,
        # matches the spec 19.3 "frozen sensor" guard input signal) --
        # per-frame-type run length of exact byte repeats. ponytail: exact
        # byte-equality is a naive freeze heuristic (a real ADC could in
        # principle repeat a value by chance); upgrade to a
        # decoded-value/tolerance comparison if that ever produces a false
        # positive in practice.
        if frame_type in (FRAME_ACCEL, FRAME_GYRO):
            if self._last_bytes_by_type.get(frame_type) == frame_bytes:
                self._stale_run_by_type[frame_type] = self._stale_run_by_type.get(frame_type, 1) + 1
            else:
                self._stale_run_by_type[frame_type] = 1
                self._last_bytes_by_type[frame_type] = frame_bytes
            if self._stale_run_by_type[frame_type] >= self.stale_run_length:
                quality_flags.append("stale_suspect")

        self.host_seq += 1
        host_seq = self.host_seq

        base = schemas.base_record(
            schema=schemas.SCHEMA_IMU,
            session_id=self.session_id,
            segment_id=self.segment_id,
            stream=schemas.STREAM_IMU,
            sensor_id=self.sensor_id,
            host_seq=host_seq,
            t_pc_rx_ns=t_pc_rx_ns,
            t_pc_first_byte_ns=t_pc_first_byte_ns,
            t_pc_frame_end_ns=t_pc_frame_end_ns,
            valid=True,
            parse_status="ok",
            checksum_ok=True,
            duplicate=duplicate,
            # The WT901C wire format carries no in-frame sequence (audit
            # finding, DECISIONS.md #4) -- host_seq is assigned by us in
            # strict read order, so there is no independent signal this
            # collector could use to detect reordering for THIS sensor.
            # Always False here; a real out-of-order signal would need an
            # externally-verifiable ordering source.
            out_of_order=False,
            quality_flags=quality_flags,
            raw_hex=frame_bytes.hex(),
        )
        record = schemas.imu_record(
            base,
            accel_raw=decoded.get("accel_raw"),
            gyro_raw=decoded.get("gyro_raw"),
            accel_mps2=decoded.get("accel_mps2"),
            gyro_rads=decoded.get("gyro_rads"),
            mag_raw=decoded.get("mag_raw"),
            mag_packet_word4_raw=decoded.get("mag_packet_word4_raw"),
            euler_rad=decoded.get("euler_rad"),
            quaternion_xyzw=decoded.get("quaternion_xyzw"),
            temperature_c=decoded.get("temperature_c"),
            angle_packet_word4_raw=decoded.get("angle_packet_word4_raw"),
            packet_type=decoded.get("packet_type"),
            frame_length=len(frame_bytes),
            serial_port=self.port,
            configured_baud=self.baud,
            configured_output_rate_hz=self.configured_output_rate_hz,
            t_sensor=decoded.get("t_sensor"),
        )
        # `core/schemas.py` has no dedicated connection_epoch field (it's a
        # fixed shared contract -- not modified here, see module docstring).
        # Stamped as a plain extra top-level key instead, same placement
        # `live_4imu_visualizer.py`'s SensorReader used, so a gap across a
        # reconnect is recorded (not silently ignored) even though
        # host_seq itself keeps advancing monotonically through the gap.
        record["connection_epoch"] = epoch
        self.frames_emitted += 1

        if self.on_record is not None:
            # Never let an observer break capture: this runs on the reader
            # thread, so an exception here would kill the connection and
            # trigger a spurious reconnect.
            try:
                self.on_record(record)
            except Exception as exc:  # noqa: BLE001
                self.on_record_errors += 1
                if self.on_record_errors == 1:
                    self._publish("imu_on_record_error", f"{self.sensor_id} on_record raised: {exc}")

        if not self.queue.put_nowait_tracked(record, seq=host_seq):
            self._publish("imu_queue_overflow", f"{self.sensor_id} dropped host_seq={host_seq}")

    # --- pump thread: BoundedQueue -> JsonlWriter -----------------------
    def _pump_loop(self) -> None:
        """Drains `self.queue` and forwards to `self.writer`. Exits only
        once `_reader_done_event` is set (meaning the reader thread has
        fully stopped producing, see `stop()`) AND the queue is observed
        empty -- never on `_stop_event` alone, which would race against
        the reader thread still flushing its last few records."""
        while True:
            try:
                record = self.queue.get(timeout=0.2)
            except queue.Empty:
                if self._reader_done_event.is_set():
                    return
                continue
            self.writer.put(record)
            self.queue.task_done()

    def stats(self) -> dict:
        return {
            "connection_epoch": self.connection_epoch,
            "host_seq": self.host_seq,
            "frames_emitted": self.frames_emitted,
            "resync_count": self.resync_count,
            "disconnect_count": self.disconnect_count,
            "queue_depth_state": self.queue.depth_state(),
            "queue_drop_stats": self.queue.drop_stats(),
        }


def _self_check() -> None:
    # extract_frames: clean valid frame.
    body = bytes((HEADER, FRAME_ACCEL)) + bytes(range(8))
    frame = body + bytes((sum(body) & 0xFF,))
    buf = bytearray(frame)
    frames, carry = extract_frames(buf)
    assert frames == [(frame, False)]
    assert carry == 0
    assert len(buf) == 0

    # extract_frames: corrupted checksum requires exactly one resync,
    # next valid frame still found and flagged.
    bad = bytearray(frame)
    bad[-1] ^= 0xFF
    good2_body = bytes((HEADER, FRAME_GYRO)) + bytes(range(8, 16))
    good2 = good2_body + bytes((sum(good2_body) & 0xFF,))
    buf = bytearray(bytes(bad) + good2)
    frames, carry = extract_frames(buf)
    assert len(frames) == 1 and frames[0] == (good2, True)

    # extract_frames: split across two "read()" chunks.
    buf = bytearray(frame[:4])
    frames, carry = extract_frames(buf)
    assert frames == []
    buf.extend(frame[4:])
    frames, carry = extract_frames(buf, carry)
    assert frames == [(frame, False)]

    # decode_frame: accel round-trips g -> m/s^2, quaternion reorders to xyzw.
    accel_payload = struct.pack("<hhhh", 0, 0, 2048, 2500)  # az raw 2048/32768*16 = +1g, temp=25.00C
    accel_body = bytes((HEADER, FRAME_ACCEL)) + accel_payload
    accel_frame = accel_body + bytes((sum(accel_body) & 0xFF,))
    decoded = decode_frame(accel_frame)
    assert abs(decoded["accel_mps2"][2] - STANDARD_GRAVITY_MPS2 * 1.0) < 1e-6
    assert abs(decoded["temperature_c"] - 25.0) < 1e-6

    quat_payload = struct.pack("<hhhh", 16384, 0, 0, 0)  # w=0.5
    quat_body = bytes((HEADER, FRAME_QUATERNION)) + quat_payload
    quat_frame = quat_body + bytes((sum(quat_body) & 0xFF,))
    decoded = decode_frame(quat_frame)
    qx, qy, qz, qw = decoded["quaternion_xyzw"]
    assert abs(qw - 0.5) < 1e-6 and abs(qx) < 1e-6

    print("collect_imu_rs232.py self-check OK")


if __name__ == "__main__":
    _self_check()
