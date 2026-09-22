"""PC-side serial client for `02_SO-101_rtos_motor_driver/main`'s read-only
`STREAM` ASCII telemetry protocol (DECISIONS.md #3, #7).

NOT THE FIRMWARE ON THIS RIG. The board is flashed with
`03_SO-101_control/main`, which speaks a `STATE`-poll protocol over 3 axes
(shoulder_pan/shoulder_lift/wrist_flex) -- see
`collectors/collect_esp32_state_poll.py`, which is the real-hardware
collector. This module is kept only because the 05 STREAM protocol it was
written against is genuinely 5-joint (shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, gripper); its `NUM_JOINTS = 5` and joint order
below describe THAT firmware, not this rig. It is exercised in mock mode
only -- do not point it at the connected hardware.

This is a serial CLIENT, not new firmware -- it only ever sends `STREAM ON <period_ms>` on
connect. It NEVER sends `TORQUE`, `INIT`, `MOVE_RAW`, `CAL_AUTO`, or any
other motion/write/calibration command -- root `AGENTS.md`'s hardware-safety
rules forbid that without explicit user intent, which this task does not
have.

Wire format, matched from `02_SO-101_rtos_motor_driver/main/README.md`
("Compact Status Stream") and `mocks/mock_esp32.py` (built to the same
contract):

    STREAM t_ms=12345 seq=1 fresh=1 armed=1 cal=0 fault=0 fault_axis=-1
    mode=PID enabled=0x1F pos=2225,2100,1988,1900,700
    upd=12344,12344,12344,12344,12344 deg=80.00,35.00,77.00,155.00,50.00
    target=80.00,35.00,77.00,155.00,50.00

`t_ms` is milliseconds since ESP32 boot -> converted to microseconds for
`t_esp_us`/`t_source` (`core.schemas.encoder_record` expects microseconds).
Joint order is fixed and positional (05's axis IDs 1-5, same order the mock
emits `pos=`/`deg=`/`target=` in):
`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, gripper`.
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
from typing import Any, Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schemas import (  # noqa: E402
    SCHEMA_ENCODER,
    STREAM_ENCODER,
    base_record,
    encoder_record,
    monotonic_ns,
)

# Fixed joint order 05's firmware and mocks/mock_esp32.py both use for
# pos=/upd=/deg=/target= -- do not reorder.
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "gripper")
NUM_JOINTS = len(JOINT_NAMES)

_LIMITS_NOT_READY_DEG = -999.0

_LINE_RE = re.compile(
    r"^STREAM t_ms=(?P<t_ms>\d+) seq=(?P<seq>\d+) fresh=(?P<fresh>\d) "
    r"armed=(?P<armed>\d) cal=(?P<cal>\d) fault=(?P<fault>\d) "
    r"fault_axis=(?P<fault_axis>-?\d+) mode=(?P<mode>\S+) "
    r"enabled=0x(?P<enabled>[0-9A-Fa-f]+) pos=(?P<pos>[\-\d,]+) "
    r"upd=(?P<upd>[\d,]+) deg=(?P<deg>[\-\d.,]+) target=(?P<target>[\-\d.,]+)$"
)


def parse_stream_line(line: str) -> Optional[dict[str, Any]]:
    """Parse one `STREAM ...` line into typed fields, or None if it doesn't
    match the documented format (the caller still records the raw line --
    see `Esp32EncoderCollector.read_record`)."""
    m = _LINE_RE.match(line.strip())
    if not m:
        return None
    pos = [int(v) for v in m.group("pos").split(",")]
    upd = [int(v) for v in m.group("upd").split(",")]
    deg = [float(v) for v in m.group("deg").split(",")]
    target = [float(v) for v in m.group("target").split(",")]
    if len(pos) != NUM_JOINTS or len(upd) != NUM_JOINTS or len(deg) != NUM_JOINTS or len(target) != NUM_JOINTS:
        return None
    return {
        "t_ms": int(m.group("t_ms")),
        "seq": int(m.group("seq")),
        "fresh": m.group("fresh") == "1",
        "armed": m.group("armed") == "1",
        "cal": m.group("cal") == "1",
        "fault": m.group("fault") == "1",
        "fault_axis": int(m.group("fault_axis")),
        "mode": m.group("mode"),
        "enabled_mask": int(m.group("enabled"), 16),
        "pos": pos,
        "upd": upd,
        "deg": deg,
        "target": target,
    }


def _line_getter(source: Any) -> Callable[[], str]:
    """Adapt `source` to a zero-arg callable returning the next raw line
    (no trailing newline). Accepts, in priority order: an object with
    `.next_line()` (`mocks/mock_esp32.py`'s `MockESP32Encoder`, or a
    `mocks/fault_injector.py`-wrapped one), a `serial.Serial`-like object
    with `.readline() -> bytes`, or a bare zero-arg callable."""
    if hasattr(source, "next_line"):
        return source.next_line
    if hasattr(source, "readline"):
        def _get() -> str:
            raw = source.readline()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            return raw.rstrip("\r\n")

        return _get
    if callable(source):
        return source
    raise TypeError(f"{source!r} has no next_line/readline and isn't callable")


def _esp32_state(armed: bool, cal: bool, fault: bool) -> str:
    if fault:
        return "fault"
    if cal:
        return "calibrating"
    if armed:
        return "armed"
    return "idle"


class Esp32EncoderCollector:
    """PC-side client for 05's `STREAM ON <period_ms>` telemetry.

    `conn` is anything `_line_getter` accepts: a `serial.Serial` opened at
    115200 baud (05's README) for real hardware, or a
    `mocks.mock_esp32.MockESP32Encoder` (optionally
    `mocks.fault_injector`-wrapped) for tests.
    """

    def __init__(
        self,
        conn: Any,
        session_id: str,
        segment_id: int = 0,
        period_ms: int = 100,
        reconnect_fn: Optional[Callable[[], Any]] = None,
        retry_backoff_s: float = 1.0,
        max_backoff_s: float = 10.0,
    ):
        self.conn = conn
        self.session_id = session_id
        self.segment_id = segment_id
        self.period_ms = period_ms
        self.reconnect_fn = reconnect_fn
        self.retry_backoff_s = retry_backoff_s
        self.max_backoff_s = max_backoff_s
        self._line_get = _line_getter(conn)
        self.host_seq = 0
        self.reconnect_count = 0

    def start(self) -> None:
        """Read-only telemetry subscription. NEVER send TORQUE/INIT/
        MOVE_RAW/CAL_AUTO or any other write/motion command from this
        collector (DECISIONS.md #3/#7, root AGENTS.md hardware-safety
        rules). No-op if `conn` has no `.write()` (e.g. the mock)."""
        if hasattr(self.conn, "write"):
            self.conn.write(f"STREAM ON {self.period_ms}\n".encode("ascii"))

    def _reconnect_with_backoff(self) -> None:
        """1s-default-backoff retry for a dropped link, doubling (capped at
        `max_backoff_s`) between attempts when `reconnect_fn` is given
        (real hardware: actually reopen the serial port and re-send
        `STREAM ON`). Without a `reconnect_fn` -- e.g. a
        fault-injector-wrapped mock whose drop clears itself after N calls
        -- this just waits `retry_backoff_s` and lets `read_record`'s next
        attempt try the existing `conn` again."""
        if self.reconnect_fn is None:
            time.sleep(self.retry_backoff_s)
            self.reconnect_count += 1
            return

        backoff = self.retry_backoff_s
        while True:
            time.sleep(backoff)
            try:
                self.conn = self.reconnect_fn()
                self._line_get = _line_getter(self.conn)
                self.start()
                self.reconnect_count += 1
                return
            except Exception:
                backoff = min(backoff * 2, self.max_backoff_s)

    def read_record(self) -> dict[str, Any]:
        """Block for the next line and return exactly one `encoder_record`.
        A disconnect (`ConnectionError`/`OSError`) retries with backoff
        internally and never raises out of this method for a transient
        link drop. An unparsable line still produces a record
        (`valid=False`, `parse_status="parse_error"`, raw text preserved)
        rather than being silently skipped."""
        while True:
            try:
                line = self._line_get()
                break
            except (ConnectionError, OSError):
                self._reconnect_with_backoff()

        t_pc_rx_ns = monotonic_ns()
        self.host_seq += 1
        parsed = parse_stream_line(line)

        if parsed is None:
            base = base_record(
                schema=SCHEMA_ENCODER,
                session_id=self.session_id,
                segment_id=self.segment_id,
                stream=STREAM_ENCODER,
                sensor_id="esp32_encoder",
                host_seq=self.host_seq,
                t_pc_rx_ns=t_pc_rx_ns,
                valid=False,
                parse_status="parse_error",
                raw_line=line,
            )
            return encoder_record(base)

        t_esp_us = parsed["t_ms"] * 1000
        q_encoder_rad = [
            None if d == _LIMITS_NOT_READY_DEG else _deg2rad(d) for d in parsed["deg"]
        ]
        # Same -999.0 "limits not ready" sentinel applies to target= as to
        # deg= (05's README): it must stay None, never radians(-999).
        q_cmd_rad = [
            None if d == _LIMITS_NOT_READY_DEG else _deg2rad(d) for d in parsed["target"]
        ]
        fault_list = [
            "fault" if parsed["fault"] and i == parsed["fault_axis"] else "ok"
            for i in range(NUM_JOINTS)
        ]
        motor_status = [
            "enabled" if parsed["enabled_mask"] & (1 << i) else "disabled"
            for i in range(NUM_JOINTS)
        ]
        quality_flags = (
            ["joint_limits_not_ready"]
            if any(v is None for v in q_encoder_rad) or any(v is None for v in q_cmd_rad)
            else []
        )

        base = base_record(
            schema=SCHEMA_ENCODER,
            session_id=self.session_id,
            segment_id=self.segment_id,
            stream=STREAM_ENCODER,
            sensor_id="esp32_encoder",
            source_seq=parsed["seq"],
            host_seq=self.host_seq,
            t_source=t_esp_us,
            t_pc_rx_ns=t_pc_rx_ns,
            valid=True,
            parse_status="ok",
            quality_flags=quality_flags,
            raw_line=line,
        )
        return encoder_record(
            base,
            q_cmd_rad=q_cmd_rad,
            encoder_raw_tick=parsed["pos"],
            q_encoder_rad=q_encoder_rad,
            esp32_state=_esp32_state(parsed["armed"], parsed["cal"], parsed["fault"]),
            fault=fault_list,
            motor_status=motor_status,
            t_esp_us=t_esp_us,
        )


def _deg2rad(deg: float) -> float:
    return math.radians(deg)


def _run_cli() -> None:  # pragma: no cover - real hardware only, not exercised by tests
    import argparse

    import serial

    from core.interrupt_handler import installed
    from core.jsonl_writer import JsonlWriter

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True, help="Serial port, e.g. COM18")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--period-ms", type=int, default=100)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--segment-id", type=int, default=0)
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1.0)

    def _reconnect() -> "serial.Serial":
        return serial.Serial(args.port, args.baud, timeout=1.0)

    collector = Esp32EncoderCollector(
        ser,
        args.session_id,
        args.segment_id,
        args.period_ms,
        reconnect_fn=_reconnect,
    )
    collector.start()

    writer = JsonlWriter(args.output_dir, STREAM_ENCODER, args.segment_id)
    try:
        with installed() as shutdown_requested:
            while not shutdown_requested.is_set():
                writer.put(collector.read_record())
        writer.close(clean=True)
    except Exception:
        writer.close(clean=False)
        raise


if __name__ == "__main__":  # pragma: no cover
    _run_cli()
