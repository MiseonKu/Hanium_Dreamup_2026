"""Common raw-record schema builders.

Single source of truth for the data contract described in the master spec,
sections 9-12, and `DECISIONS.md`. Collectors, mocks, and tools all build
records through these functions instead of hand-assembling dicts, so a
schema change happens in exactly one place.

Timestamp policy (DECISIONS.md #6): PC-master sample time is always
`time.perf_counter_ns()` (monotonic). `utc_iso` is a human-readable
wall-clock cross-check only and is never used as the authoritative sample
time or for ordering.
"""
from __future__ import annotations

import datetime
import time
from typing import Any, Optional

SCHEMA_IMU = "alphaf.imu.raw.v1"
SCHEMA_ENCODER = "alphaf.encoder.raw.v1"
SCHEMA_COMMAND = "alphaf.command.raw.v1"
SCHEMA_VIVE = "alphaf.vive.raw.v1"
SCHEMA_EVENT = "alphaf.event.raw.v1"
SCHEMA_HEALTH = "alphaf.health.raw.v1"
SCHEMA_STABILITY = "alphaf.stability_metrics.raw.v1"

STREAM_IMU = "imu"
STREAM_ENCODER = "encoder"
STREAM_COMMAND = "commands"
STREAM_VIVE = "vive_tracker"
STREAM_EVENT = "events"
STREAM_HEALTH = "health"
STREAM_STABILITY = "stability_metrics"


def monotonic_ns() -> int:
    """PC-master clock. Use for every `t_pc_*_ns` field. Never use
    `time.time()`/`datetime.now()` for a sample timestamp — see
    DECISIONS.md #6 (monotonic-clock-only policy, from 실험절차.md:481-490)."""
    return time.perf_counter_ns()


def utc_iso_now() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def base_record(
    *,
    schema: str,
    session_id: str,
    segment_id: int,
    stream: str,
    host_seq: int,
    t_pc_rx_ns: int,
    valid: bool,
    parse_status: str,
    sensor_id: Optional[str] = None,
    source_seq: Optional[int] = None,
    trial_id: Optional[int] = None,
    target_id: Optional[str] = None,
    planned_order_index: Optional[int] = None,
    workspace_mode: Optional[str] = None,
    experiment_state: Optional[str] = None,
    t_source: Optional[int] = None,
    t_pc_first_byte_ns: Optional[int] = None,
    t_pc_frame_end_ns: Optional[int] = None,
    t_sync_ns: Optional[int] = None,
    checksum_ok: Optional[bool] = None,
    duplicate: bool = False,
    out_of_order: bool = False,
    late_by_ms: Optional[float] = None,
    quality_flags: Optional[list[str]] = None,
    raw_hex: Optional[str] = None,
    raw_line: Optional[str] = None,
) -> dict[str, Any]:
    """Fields common to every stream (master spec section 9). Missing
    values are `None`, never `0` or `""` — a present-but-zero value and an
    absent value must stay distinguishable downstream."""
    return {
        "schema": schema,
        "session_id": session_id,
        "segment_id": segment_id,
        "stream": stream,
        "sensor_id": sensor_id,
        "source_seq": source_seq,
        "host_seq": host_seq,
        "trial_id": trial_id,
        "target_id": target_id,
        "planned_order_index": planned_order_index,
        "workspace_mode": workspace_mode,
        "experiment_state": experiment_state,
        "t_source": t_source,
        "t_pc_first_byte_ns": t_pc_first_byte_ns,
        "t_pc_frame_end_ns": t_pc_frame_end_ns,
        "t_pc_rx_ns": t_pc_rx_ns,
        "t_sync_ns": t_sync_ns,
        "utc_iso": utc_iso_now(),
        "valid": valid,
        "parse_status": parse_status,
        "checksum_ok": checksum_ok,
        "duplicate": duplicate,
        "out_of_order": out_of_order,
        "late_by_ms": late_by_ms,
        "quality_flags": list(quality_flags) if quality_flags else [],
        "raw_hex": raw_hex,
        "raw_line": raw_line,
    }


def imu_record(
    base: dict[str, Any],
    *,
    accel_raw: Optional[list[int]] = None,
    gyro_raw: Optional[list[int]] = None,
    accel_mps2: Optional[list[float]] = None,
    gyro_rads: Optional[list[float]] = None,
    mag_raw: Optional[list[int]] = None,
    mag_uT: Optional[list[float]] = None,
    mag_packet_word4_raw: Optional[int] = None,
    euler_rad: Optional[list[float]] = None,
    quaternion_xyzw: Optional[list[float]] = None,
    temperature_c: Optional[float] = None,
    angle_packet_word4_raw: Optional[int] = None,
    packet_type: Optional[str] = None,
    frame_length: Optional[int] = None,
    serial_port: Optional[str] = None,
    configured_baud: Optional[int] = None,
    configured_output_rate_hz: Optional[float] = None,
    measured_output_rate_hz: Optional[float] = None,
    t_sensor: Optional[str] = None,
) -> dict[str, Any]:
    """Master spec section 10. `base` must have been built with
    `schema=SCHEMA_IMU, stream=STREAM_IMU`.

    `angle_packet_word4_raw` / `mag_packet_word4_raw`: the WT901C angle
    (0x53) and magnetic-field (0x54) packets both carry an undecoded 4th
    int16 word (bytes 8-9) whose meaning is NOT the same as accel/gyro's
    `temperature_c` word -- see `decode_frame` in
    `collectors/collect_imu_rs232.py` and `docs/IMU_DATA_COVERAGE.md` for
    the investigation. Kept raw and separately named rather than mislabeled
    `temperature_c`.

    `t_sensor`: device-reported wall-clock timestamp decoded from a WT901C
    time (0x50) packet only (`None` for every other packet type). Purely
    supplementary -- `t_pc_rx_ns` (DECISIONS.md #6) remains the
    authoritative sample time, never overwritten or replaced by this.
    """
    rec = dict(base)
    rec["packet_type"] = packet_type
    rec["frame_length"] = frame_length
    rec["t_sensor"] = t_sensor
    rec["data"] = {
        "accel_raw": accel_raw,
        "gyro_raw": gyro_raw,
        "accel_mps2": accel_mps2,
        "gyro_rads": gyro_rads,
        "mag_raw": mag_raw,
        "mag_uT": mag_uT,
        "mag_packet_word4_raw": mag_packet_word4_raw,
        "euler_rad": euler_rad,
        "quaternion_xyzw": quaternion_xyzw,
        "temperature_c": temperature_c,
        "angle_packet_word4_raw": angle_packet_word4_raw,
    }
    rec["sensor_config"] = {
        "serial_port": serial_port,
        "configured_baud": configured_baud,
        "configured_output_rate_hz": configured_output_rate_hz,
        "measured_output_rate_hz": measured_output_rate_hz,
    }
    return rec


def encoder_record(
    base: dict[str, Any],
    *,
    q_cmd_raw: Optional[list[int]] = None,
    q_cmd_rad: Optional[list[float]] = None,
    encoder_raw_tick: Optional[list[int]] = None,
    q_encoder_rad: Optional[list[float]] = None,
    estimated_velocity_rad_s: Optional[list[float]] = None,
    motor_status: Optional[list[str]] = None,
    temperature_c: Optional[list[float]] = None,
    current_or_load: Optional[list[float]] = None,
    fault: Optional[list[str]] = None,
    esp32_state: Optional[str] = None,
    command_ack: Optional[bool] = None,
    watchdog_state: Optional[str] = None,
    t_esp_us: Optional[int] = None,
) -> dict[str, Any]:
    """Master spec section 11. `base` must have
    `schema=SCHEMA_ENCODER, stream=STREAM_ENCODER`, `t_source=t_esp_us`."""
    rec = dict(base)
    rec["t_esp_us"] = t_esp_us
    rec["data"] = {
        "q_cmd_raw": q_cmd_raw,
        "q_cmd_rad": q_cmd_rad,
        "encoder_raw_tick": encoder_raw_tick,
        "q_encoder_rad": q_encoder_rad,
        "estimated_velocity_rad_s": estimated_velocity_rad_s,
        "motor_status": motor_status,
        "temperature_c": temperature_c,
        "current_or_load": current_or_load,
        "fault": fault,
        "esp32_state": esp32_state,
        "command_ack": command_ack,
        "watchdog_state": watchdog_state,
    }
    return rec


def vive_record(
    base: dict[str, Any],
    *,
    tracker_serial: Optional[str] = None,
    device_index: Optional[int] = None,
    tracking_result: Optional[str] = None,
    tracking_valid: Optional[bool] = None,
    device_connected: Optional[bool] = None,
    tracking_origin: Optional[str] = None,
    pose_matrix_3x4: Optional[list[list[float]]] = None,
    position_m: Optional[list[float]] = None,
    quaternion_xyzw: Optional[list[float]] = None,
    source_timestamp: Optional[float] = None,
) -> dict[str, Any]:
    """Master spec section 12. `base` must have
    `schema=SCHEMA_VIVE, stream=STREAM_VIVE`. Never forward-fill a missing
    pose — an invalid sample is still written, with `valid=False` and
    `tracking_valid=False`, never carried forward from the last-good pose
    (DECISIONS.md #6 / master spec 1.3, 17.2)."""
    rec = dict(base)
    rec["frame_id"] = "vive_world"
    rec["data"] = {
        "tracker_serial": tracker_serial,
        "device_index": device_index,
        "tracking_result": tracking_result,
        "tracking_valid": tracking_valid,
        "device_connected": device_connected,
        "tracking_origin": tracking_origin,
        "pose_matrix_3x4": pose_matrix_3x4,
        "position_m": position_m,
        "quaternion_xyzw": quaternion_xyzw,
        "source_timestamp": source_timestamp,
    }
    return rec


def event_record(
    base: dict[str, Any],
    *,
    event_type: str,
    message: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """`base` must have `schema=SCHEMA_EVENT, stream=STREAM_EVENT`."""
    rec = dict(base)
    rec["event_type"] = event_type
    rec["message"] = message
    rec["details"] = details or {}
    return rec


def health_record(
    base: dict[str, Any],
    *,
    queue_depth: Optional[dict[str, int]] = None,
    writer_lag_ms: Optional[dict[str, float]] = None,
    cpu_percent: Optional[float] = None,
    memory_percent: Optional[float] = None,
    disk_free_bytes: Optional[int] = None,
    packet_rate_hz: Optional[dict[str, float]] = None,
    packet_loss_ratio: Optional[dict[str, float]] = None,
    checksum_failures: Optional[dict[str, int]] = None,
    reconnect_counts: Optional[dict[str, int]] = None,
    sampling_jitter_ms: Optional[dict[str, float]] = None,
    late_count: Optional[dict[str, int]] = None,
    out_of_order_count: Optional[dict[str, int]] = None,
    duplicate_count: Optional[dict[str, int]] = None,
    queue_state: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Master spec section 15. `base` must have
    `schema=SCHEMA_HEALTH, stream=STREAM_HEALTH`."""
    rec = dict(base)
    rec["data"] = {
        "queue_depth": queue_depth or {},
        "writer_lag_ms": writer_lag_ms or {},
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "disk_free_bytes": disk_free_bytes,
        "packet_rate_hz": packet_rate_hz or {},
        "packet_loss_ratio": packet_loss_ratio or {},
        "checksum_failures": checksum_failures or {},
        "reconnect_counts": reconnect_counts or {},
        "sampling_jitter_ms": sampling_jitter_ms or {},
        "late_count": late_count or {},
        "out_of_order_count": out_of_order_count or {},
        "duplicate_count": duplicate_count or {},
        "queue_state": queue_state or {},
    }
    return rec


def stability_record(
    base: dict[str, Any],
    *,
    tip_gyro_rms: Optional[float] = None,
    dynamic_accel_rms: Optional[float] = None,
    orientation_spread_rad: Optional[float] = None,
    jerk_rms: Optional[float] = None,
    drift_slope: Optional[float] = None,
    stable_candidate_duration_s: Optional[float] = None,
    stability_margin: Optional[float] = None,
    settling_state: Optional[str] = None,
    guard_flags: Optional[dict[str, bool]] = None,
) -> dict[str, Any]:
    """Master spec section 19.2. `base` must have
    `schema=SCHEMA_STABILITY, stream=STREAM_STABILITY`."""
    rec = dict(base)
    rec["data"] = {
        "tip_gyro_rms": tip_gyro_rms,
        "dynamic_accel_rms": dynamic_accel_rms,
        "orientation_spread_rad": orientation_spread_rad,
        "jerk_rms": jerk_rms,
        "drift_slope": drift_slope,
        "stable_candidate_duration_s": stable_candidate_duration_s,
        "stability_margin": stability_margin,
        "settling_state": settling_state,
        "guard_flags": guard_flags or {},
    }
    return rec


def command_record(
    base: dict[str, Any],
    *,
    command_type: str,
    q_cmd_rad: Optional[list[float]] = None,
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """`base` must have `schema=SCHEMA_COMMAND, stream=STREAM_COMMAND`."""
    rec = dict(base)
    rec["command_type"] = command_type
    rec["data"] = {
        "q_cmd_rad": q_cmd_rad,
        "payload": payload or {},
    }
    return rec


def _self_check() -> None:
    import json

    b = base_record(
        schema=SCHEMA_IMU,
        session_id="session_selfcheck",
        segment_id=0,
        stream=STREAM_IMU,
        sensor_id="imu3_tip",
        host_seq=1,
        t_pc_rx_ns=monotonic_ns(),
        valid=True,
        parse_status="ok",
    )
    rec = imu_record(b, accel_mps2=[0.0, 0.0, 9.8], gyro_rads=[0.0, 0.0, 0.0])
    json.dumps(rec)  # every field must be JSON-serializable
    assert rec["schema"] == SCHEMA_IMU
    assert rec["data"]["mag_uT"] is None, "missing field must be null, not 0"
    assert rec["quality_flags"] == [], "default quality_flags must be [], not None"

    vb = base_record(
        schema=SCHEMA_VIVE,
        session_id="session_selfcheck",
        segment_id=0,
        stream=STREAM_VIVE,
        host_seq=1,
        t_pc_rx_ns=monotonic_ns(),
        valid=False,
        parse_status="tracking_lost",
    )
    vrec = vive_record(vb, tracking_valid=False)
    json.dumps(vrec)
    assert vrec["data"]["position_m"] is None, "invalid Vive sample must not be forward-filled"

    print("schemas.py self-check OK")


if __name__ == "__main__":
    _self_check()
