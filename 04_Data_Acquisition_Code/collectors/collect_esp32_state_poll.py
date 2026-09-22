"""PC-side collector for the ESP32 firmware that is ACTUALLY on the board:
`03_SO-101_control/main`, polled via its `STATE` command.

Not to be confused with `collect_esp32_encoder.py`, which speaks
`02_SO-101_rtos_motor_driver`'s push-style `STREAM t_ms=... seq=...` protocol.
That firmware was never flashed on this rig -- see `DECISIONS.md` #3/#7 for
how the two got mixed up. Both collectors are kept: this one for the real
hardware, that one for the protocol it was written against.

Wire format, transcribed from `03_SO-101_control/main/src/main.cpp`:

- `printJointDetailLine()` (main.cpp:1400), one line per axis per `STATE`:

      axis=shoulder_pan alias=D2 id=1 ping=true torque=true raw_pos=2225
      raw_speed=0 raw_load=12 raw_voltage=118 raw_temp=34 raw_current=3
      logical_deg=80.0 mapping_ready=true raw_cal_min=1054 raw_cal_max=3189
      raw_active_min=1054 raw_active_max=3189 speed=120 acc=10
      range_percent=100 direction=normal last_update_ms=12344 fault=false
      read_failures=0 overload_samples=0

  `logical_deg` is the literal string `NA` when the axis has no valid
  raw->logical mapping yet (main.cpp:1366) -- parsed to `None`, never 0.0.

- `printFaultSummaryIfLatched()` (main.cpp:1378), at most one line per
  `STATE`, only when a fault is latched:

      fault_latched=true axis=shoulder_lift reason=overload latched_ms=98765

- `queueLegacyAngleCommand()` (main.cpp:3189) echoes each accepted legacy
  `D<axis><deg>` command:

      [LEGACY] queued axis=shoulder_pan logical=100

  and `flushLegacyBatch()` (main.cpp:3141) confirms actual dispatch:

      [LEGACY] batch dispatched count=1 reason=timer

  Those two echoes are what make a real command ACK timestamp possible;
  `send_axis_command()` below waits for the first one.

Raw units are preserved as-is under `data.raw_telemetry` (they are servo
register counts -- `raw_voltage`, `raw_temp`, `raw_current`, `raw_load` are
NOT volts/degrees/amps/newton-metres, and this module does not invent scale
factors for them; the ST3215 datasheet needed to convert them is not in this
repo, see `docs/ST3215_THERMAL_AND_SEQUENCING.md`).

`logical_deg` IS NOT A PHYSICAL ANGLE. The firmware linearly maps each
axis's calibrated raw span (`raw_cal_min`..`raw_cal_max`) onto an arbitrary
per-axis label interval (`logical_min_deg`..`logical_max_deg`), so one
"logical degree" is a different physical size on every axis. Measured
against the real rig, physical degrees per logical degree are:

    shoulder_pan  1.173   shoulder_lift 1.300   wrist_flex 1.101

The ST3215 encoder is 4096 counts / 360 deg, i.e. 11.378 raw ticks per
PHYSICAL degree. The authoritative physical quantity in these records is
therefore `data.encoder_raw_tick` (raw counts), NOT any radian field.

`data.q_encoder_rad` / `data.q_cmd_rad` are `radians(logical_deg)` and are
retained ONLY for backward compatibility with sessions already collected
under that definition (`sessions/real_motion_demo/`) -- they are radians of
a *label*, not of a joint angle, and must NOT be fed to FK/kinematics.
`data.logical_deg` / `data.q_cmd_logical_deg` (added here) carry the
firmware's own numbers untransformed so downstream code can apply the
correct per-axis conversion itself. `data.estimated_velocity_rad_s` is a
finite difference of the same label quantity and inherits the same caveat.

Joint order is this rig's intentional 3-axis scope

(`docs/THREE_AXIS_EXPERIMENT_SCOPE.md`): shoulder_pan, shoulder_lift,
wrist_flex.
"""
from __future__ import annotations

import math
import os
import re
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schemas import (  # noqa: E402
    SCHEMA_COMMAND,
    SCHEMA_ENCODER,
    STREAM_COMMAND,
    STREAM_ENCODER,
    base_record,
    command_record,
    encoder_record,
    monotonic_ns,
)

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "wrist_flex")
AXIS_ALIASES = ("D2", "D3", "D5")

# Tolerant of firmware field additions: each field is matched independently
# rather than as one rigid whole-line pattern, so a future extra `foo=bar`
# doesn't silently drop every axis on the floor (which a single anchored
# regex would). Missing field -> None, never a fabricated default.
_AXIS_LINE_RE = re.compile(r"^axis=(?P<axis>\S+)\s")
_FIELD_RE = re.compile(r"(\w+)=(\S+)")

_FAULT_LINE_RE = re.compile(
    r"fault_latched=(?P<latched>\w+)(?:\s+axis=(?P<axis>\S+))?"
    r"(?:\s+reason=(?P<reason>\S+))?(?:\s+latched_ms=(?P<latched_ms>\d+))?"
)
_LEGACY_QUEUED_RE = re.compile(r"\[LEGACY\] queued axis=(?P<axis>\S+) logical=(?P<logical>-?\d+)")
# main.cpp:1858 -- "[MOVE_RAW] axis=%s id=%u raw=%d speed=%u acc=%u".
# The failure path (main.cpp:1848) prints "[MOVE_RAW] write failed axis=..."
# instead, which this deliberately does NOT match, so a refused move is
# reported as no_ack rather than as a success.
_MOVE_RAW_ECHO_RE = re.compile(r"\[MOVE_RAW\] axis=(?P<axis>\S+) id=\d+ raw=(?P<raw>-?\d+)")

_INT_FIELDS = (
    "id", "raw_pos", "raw_speed", "raw_load", "raw_voltage", "raw_temp",
    "raw_current", "raw_cal_min", "raw_cal_max", "raw_active_min",
    "raw_active_max", "speed", "acc", "range_percent", "last_update_ms",
    "read_failures", "overload_samples",
)
_BOOL_FIELDS = ("ping", "torque", "mapping_ready", "fault")


def parse_state_text(text: str) -> dict[str, Any]:
    """Parse one whole `STATE` response into
    `{"axes": {axis_name: {...}}, "fault": {...} or None}`.

    Unknown/extra `key=value` pairs are kept as raw strings rather than
    dropped -- a firmware that starts reporting something new should show up
    in the data, not vanish."""
    axes: dict[str, dict[str, Any]] = {}
    fault: Optional[dict[str, Any]] = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("fault_latched="):
            m = _FAULT_LINE_RE.search(line)
            if m:
                fault = {
                    "latched": m.group("latched") == "true",
                    "axis": m.group("axis"),
                    "reason": m.group("reason"),
                    "latched_ms": int(m.group("latched_ms")) if m.group("latched_ms") else None,
                }
            continue

        if not _AXIS_LINE_RE.match(line):
            continue

        fields: dict[str, Any] = {}
        for key, value in _FIELD_RE.findall(line):
            if key in _INT_FIELDS:
                try:
                    fields[key] = int(value)
                except ValueError:
                    fields[key] = None
            elif key in _BOOL_FIELDS:
                fields[key] = value == "true"
            elif key == "logical_deg":
                # Firmware emits the literal "NA" when raw->logical mapping
                # isn't ready (main.cpp:1366). Must stay None, not 0.0.
                fields[key] = None if value == "NA" else float(value)
            else:
                fields[key] = value
        axes[fields.get("axis", "")] = fields

    return {"axes": axes, "fault": fault}


class Esp32StatePollCollector:
    """Poll-style collector: the caller drives it (`poll_once()` per sample),
    matching how the real sweep loop already works. `conn` is a
    `serial.Serial`-like object needing only `.write(bytes)` and
    `.read(n) -> bytes`.
    """

    def __init__(
        self,
        conn: Any,
        session_id: str,
        segment_id: int = 0,
        read_bytes: int = 4000,
        joint_names: tuple = JOINT_NAMES,
    ) -> None:
        self.conn = conn
        self.session_id = session_id
        self.segment_id = segment_id
        self.read_bytes = read_bytes
        self.joint_names = tuple(joint_names)
        self.host_seq = 0
        self.command_seq = 0
        self._prev_q_rad: Optional[list] = None
        self._prev_t_ns: Optional[int] = None
        # Last commanded logical degrees per axis, so every encoder record can
        # carry the command it is responding to (join key for later analysis).
        self._last_cmd_deg: list = [None] * len(self.joint_names)

    # -- telemetry -----------------------------------------------------
    def poll_once(self, **context: Any) -> dict[str, Any]:
        """Send `STATE`, parse the response, return one `encoder_record`.

        `context` passes through `trial_id`/`target_id`/
        `planned_order_index`/`workspace_mode`/`experiment_state` to
        `base_record` so every stream can be joined on `target_id`.

        An unparsable response still produces a record (`valid=False`,
        `parse_status="parse_error"`, `raw_line` preserved) rather than
        being dropped -- silent loss is worse than a flagged bad sample.
        """
        self.conn.write(b"STATE\r\n")
        raw = self.conn.read(self.read_bytes)
        t_pc_rx_ns = monotonic_ns()
        text = raw.decode("ascii", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        self.host_seq += 1

        parsed = parse_state_text(text)
        axes = parsed["axes"]

        if not axes:
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
                raw_line=text.strip(),
                **context,
            )
            return encoder_record(base)

        ticks, q_rad, motor_status, temps, loads = [], [], [], [], []
        faults, raw_telemetry, logical_degs = [], [], []
        for name in self.joint_names:
            a = axes.get(name)
            if a is None:
                ticks.append(None)
                q_rad.append(None)
                logical_degs.append(None)
                motor_status.append("missing")
                temps.append(None)
                loads.append(None)
                faults.append("missing")
                raw_telemetry.append(None)
                continue
            ticks.append(a.get("raw_pos"))
            deg = a.get("logical_deg")
            logical_degs.append(deg)
            # radians of a firmware LABEL, not of a joint angle -- see the
            # module docstring. Kept for backward compatibility only.
            q_rad.append(None if deg is None else math.radians(deg))
            motor_status.append("enabled" if a.get("torque") else "disabled")
            temps.append(a.get("raw_temp"))
            loads.append(a.get("raw_load"))
            faults.append("fault" if a.get("fault") else "ok")
            raw_telemetry.append(a)

        velocities = self._estimate_velocities(q_rad, t_pc_rx_ns)

        fault_info = parsed["fault"]
        latched = bool(fault_info and fault_info["latched"])
        quality_flags = []
        if any(v is None for v in q_rad):
            quality_flags.append("mapping_not_ready")
        if latched:
            quality_flags.append("fault_latched")

        base = base_record(
            schema=SCHEMA_ENCODER,
            session_id=self.session_id,
            segment_id=self.segment_id,
            stream=STREAM_ENCODER,
            sensor_id="esp32_encoder",
            host_seq=self.host_seq,
            t_pc_rx_ns=t_pc_rx_ns,
            valid=True,
            parse_status="ok",
            quality_flags=quality_flags,
            raw_line=text.strip(),
            **context,
        )
        rec = encoder_record(
            base,
            q_cmd_rad=[None if d is None else math.radians(d) for d in self._last_cmd_deg],
            encoder_raw_tick=ticks,
            q_encoder_rad=q_rad,
            estimated_velocity_rad_s=velocities,
            motor_status=motor_status,
            temperature_c=temps,      # raw servo register counts, NOT degrees C -- see module docstring
            current_or_load=loads,    # raw servo register counts, NOT amps/Nm
            fault=faults,
            esp32_state="fault" if latched else "ok",
        )
        # Everything the firmware reports that core.schemas.encoder_record has
        # no dedicated slot for. Kept whole rather than cherry-picked, so a
        # later analysis pass isn't blocked on a schema change.
        rec["data"]["raw_telemetry"] = raw_telemetry
        rec["data"]["fault_detail"] = fault_info
        # Untransformed firmware numbers, so downstream code never has to
        # un-do the radians(logical_deg) misnomer -- see module docstring.
        rec["data"]["logical_deg"] = logical_degs
        rec["data"]["q_cmd_logical_deg"] = list(self._last_cmd_deg)
        return rec

    def _estimate_velocities(self, q_rad: list, t_ns: int) -> list:
        """Backward finite difference against the previous poll. First
        sample (and any axis whose mapping isn't ready) is None, not 0.0 --
        "unknown" and "not moving" must stay distinguishable."""
        if self._prev_q_rad is None or self._prev_t_ns is None:
            out = [None] * len(q_rad)
        else:
            dt = (t_ns - self._prev_t_ns) / 1e9
            if dt <= 0:
                out = [None] * len(q_rad)
            else:
                out = [
                    None if (cur is None or prev is None) else (cur - prev) / dt
                    for cur, prev in zip(q_rad, self._prev_q_rad)
                ]
        self._prev_q_rad = list(q_rad)
        self._prev_t_ns = t_ns
        return out

    # -- commands ------------------------------------------------------
    def send_axis_command(
        self,
        axis_index: int,
        logical_deg: float,
        ack_read_bytes: int = 2000,
        **context: Any,
    ) -> dict[str, Any]:
        """Send one legacy `D<axis><deg>` command and return a
        `command_record` bracketing it with real timestamps.

        ONE axis per call, deliberately. The firmware batches every axis
        queued inside `kLegacyBatchWindowMs` into a single `RegWriteAction`
        (main.cpp:3099/3199), which starts all servos simultaneously and
        stacks their inrush on a shared power rail -- see
        `docs/ST3215_THERMAL_AND_SEQUENCING.md`. Callers must space these
        calls past that window.

        `command_ack` is True only if the firmware's `[LEGACY] queued`
        echo for THIS axis was actually seen -- never assumed from the
        write succeeding.

        HAZARD: `queueLegacyAngleCommand` (main.cpp:3159) refuses every
        `D<axis><deg>` command unless `g_is_initialized` is set, and only
        the `INIT` command sets it -- but INIT enables torque
        (main.cpp:3278) BEFORE writing goal positions (main.cpp:3309), so
        it makes the arm lunge to whatever stale goal the servos still
        hold. This method therefore either silently no-acks (no INIT) or
        requires a lunge to have happened first. `MOVE_RAW` (main.cpp:1847)
        writes the goal without touching torque and is the safe primitive;
        prefer it for real motion. See `docs/FIRMWARE_MOTION_HAZARDS.md`.
        """
        axis_name = self.joint_names[axis_index]
        alias = AXIS_ALIASES[axis_index]
        payload = f"{alias}{round(logical_deg):03d}"

        t_command_pc_ns = monotonic_ns()
        self.conn.write((payload + "\r\n").encode("ascii"))
        raw = self.conn.read(ack_read_bytes)
        t_ack_pc_ns = monotonic_ns()
        text = raw.decode("ascii", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)

        acked = any(
            m.group("axis") == axis_name for m in _LEGACY_QUEUED_RE.finditer(text)
        )
        self._last_cmd_deg[axis_index] = logical_deg
        self.command_seq += 1

        base = base_record(
            schema=SCHEMA_COMMAND,
            session_id=self.session_id,
            segment_id=self.segment_id,
            stream=STREAM_COMMAND,
            sensor_id="esp32_encoder",
            host_seq=self.command_seq,
            t_pc_rx_ns=t_ack_pc_ns,
            valid=True,
            parse_status="ok" if acked else "no_ack",
            quality_flags=[] if acked else ["command_not_acked"],
            raw_line=text.strip(),
            **context,
        )
        return command_record(
            base,
            command_type="legacy_axis_move",
            payload={
                "axis": axis_name,
                "alias": alias,
                "logical_deg": logical_deg,
                "wire": payload,
                "t_command_pc_ns": t_command_pc_ns,
                "t_ack_pc_ns": t_ack_pc_ns,
                "ack_latency_ms": (t_ack_pc_ns - t_command_pc_ns) / 1e6,
                "command_ack": acked,
            },
        )

    def send_axis_raw(
        self,
        axis_index: int,
        raw_tick: int,
        ack_read_bytes: int = 2000,
        **context: Any,
    ) -> dict[str, Any]:
        """Move one axis with `MOVE_RAW` -- the safe primitive, and the one
        real runs use.

        Preferred over `send_axis_command()` for three reasons:

        - `MOVE_RAW` (main.cpp:1847) writes the goal register without
          touching torque, so it needs no `INIT` and cannot trigger INIT's
          torque-before-goal lunge (`docs/FIRMWARE_MOTION_HAZARDS.md`).
        - Raw ticks are the lossless quantity. `logical_deg` is a per-axis
          label scale, not an angle (see module docstring), so commanding in
          degrees round-trips through a distortion.
        - One axis per command by construction, so the shared-rail thermal
          rule (`docs/ST3215_THERMAL_AND_SEQUENCING.md`) holds structurally
          instead of depending on caller discipline. Callers should still
          space calls to let each servo finish accelerating.

        The firmware clamps the target to that axis's active raw range
        (main.cpp:1843), so an out-of-range value is refused rather than
        driven into a hard stop.

        `command_ack` is True only when the firmware's own `[MOVE_RAW]` echo
        for THIS axis was seen -- never inferred from the write succeeding.
        A refusal (`write failed`, missing mapping) leaves it False.
        """
        axis_name = self.joint_names[axis_index]
        payload = f"MOVE_RAW {axis_name} {round(raw_tick)}"

        t_command_pc_ns = monotonic_ns()
        self.conn.write((payload + "\r\n").encode("ascii"))
        raw = self.conn.read(ack_read_bytes)
        t_ack_pc_ns = monotonic_ns()
        text = raw.decode("ascii", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)

        acked = any(
            m.group("axis") == axis_name for m in _MOVE_RAW_ECHO_RE.finditer(text)
        )
        self.command_seq += 1

        base = base_record(
            schema=SCHEMA_COMMAND,
            session_id=self.session_id,
            segment_id=self.segment_id,
            stream=STREAM_COMMAND,
            sensor_id="esp32_encoder",
            host_seq=self.command_seq,
            t_pc_rx_ns=t_ack_pc_ns,
            valid=True,
            parse_status="ok" if acked else "no_ack",
            quality_flags=[] if acked else ["command_not_acked"],
            raw_line=text.strip(),
            **context,
        )
        return command_record(
            base,
            command_type="move_raw",
            payload={
                "axis": axis_name,
                "raw_tick": round(raw_tick),
                "wire": payload,
                "t_command_pc_ns": t_command_pc_ns,
                "t_ack_pc_ns": t_ack_pc_ns,
                "ack_latency_ms": (t_ack_pc_ns - t_command_pc_ns) / 1e6,
                "command_ack": acked,
            },
        )


def _self_check() -> None:
    sample = (
        "axis=shoulder_pan alias=D2 id=1 ping=true torque=true raw_pos=2225 "
        "raw_speed=0 raw_load=12 raw_voltage=118 raw_temp=34 raw_current=3 "
        "logical_deg=80.0 mapping_ready=true raw_cal_min=1054 raw_cal_max=3189 "
        "raw_active_min=1054 raw_active_max=3189 speed=120 acc=10 "
        "range_percent=100 direction=normal last_update_ms=12344 fault=false "
        "read_failures=0 overload_samples=0\n"
        "axis=shoulder_lift alias=D3 id=2 ping=true torque=true raw_pos=2100 "
        "raw_speed=-5 raw_load=880 raw_voltage=117 raw_temp=41 raw_current=90 "
        "logical_deg=NA mapping_ready=false raw_cal_min=1092 raw_cal_max=3296 "
        "raw_active_min=1092 raw_active_max=3296 speed=100 acc=10 "
        "range_percent=100 direction=normal last_update_ms=12344 fault=true "
        "read_failures=0 overload_samples=3\n"
        "fault_latched=true axis=shoulder_lift reason=overload latched_ms=98765\n"
    )
    parsed = parse_state_text(sample)
    assert set(parsed["axes"]) == {"shoulder_pan", "shoulder_lift"}
    pan = parsed["axes"]["shoulder_pan"]
    assert pan["raw_pos"] == 2225 and pan["torque"] is True and pan["fault"] is False
    assert pan["logical_deg"] == 80.0
    lift = parsed["axes"]["shoulder_lift"]
    assert lift["logical_deg"] is None, "NA must parse to None, never 0.0"
    assert lift["overload_samples"] == 3 and lift["raw_speed"] == -5
    assert parsed["fault"]["latched"] is True
    assert parsed["fault"]["axis"] == "shoulder_lift"
    assert parsed["fault"]["reason"] == "overload"

    # tolerant to an added firmware field
    extended = sample.replace("read_failures=0 overload_samples=0", "read_failures=0 overload_samples=0 newfield=7")
    assert parse_state_text(extended)["axes"]["shoulder_pan"]["raw_pos"] == 2225

    class _FakeConn:
        def __init__(self, reply):
            self.reply = reply
            self.written = []

        def write(self, data):
            self.written.append(data)

        def read(self, n=1):
            return self.reply

    c = Esp32StatePollCollector(_FakeConn(sample.encode()), "s", 0)
    rec = c.poll_once(target_id="T001", planned_order_index=1)
    assert rec["valid"] is True
    assert rec["data"]["encoder_raw_tick"] == [2225, 2100, None]
    assert rec["data"]["q_encoder_rad"][1] is None, "NA axis must stay None"
    assert rec["data"]["q_encoder_rad"][2] is None, "absent axis must stay None"
    assert rec["data"]["logical_deg"] == [80.0, None, None], "raw firmware logical_deg must be preserved"
    assert rec["target_id"] == "T001"
    assert "fault_latched" in rec["quality_flags"]
    assert rec["data"]["raw_telemetry"][0]["overload_samples"] == 0
    # first sample has no previous point to difference against
    assert all(v is None for v in rec["data"]["estimated_velocity_rad_s"])

    # second poll now has a delta available for the axes that report degrees
    rec2 = c.poll_once()
    assert rec2["data"]["estimated_velocity_rad_s"][0] is not None

    ack_conn = _FakeConn(b"[LEGACY] queued axis=shoulder_pan logical=100\n")
    c2 = Esp32StatePollCollector(ack_conn, "s", 0)
    cmd = c2.send_axis_command(0, 100.4)
    assert ack_conn.written == [b"D2100\r\n"]
    assert cmd["data"]["payload"]["command_ack"] is True
    assert cmd["parse_status"] == "ok"

    noack = _FakeConn(b"[LEGACY] axis=shoulder_pan id=1 is offline; skipped\n")
    cmd2 = Esp32StatePollCollector(noack, "s", 0).send_axis_command(0, 100)
    assert cmd2["data"]["payload"]["command_ack"] is False
    assert "command_not_acked" in cmd2["quality_flags"]

    print("collect_esp32_state_poll.py self-check OK")


if __name__ == "__main__":
    _self_check()
