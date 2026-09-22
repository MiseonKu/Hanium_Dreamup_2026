from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .math3d import vector_rmse
from .types import Vector3


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def read_vector(row: Mapping[str, object], prefix: str) -> Vector3:
    return (
        _float(row, f"{prefix}_x"),
        _float(row, f"{prefix}_y"),
        _float(row, f"{prefix}_z"),
    )


def compute_position_rmse(
    rows: Sequence[Mapping[str, object]],
    *,
    estimate_prefix: str,
    actual_prefix: str = "P_actual_base_mm",
) -> float:
    estimates = [read_vector(row, estimate_prefix) for row in rows]
    actuals = [read_vector(row, actual_prefix) for row in rows]
    return vector_rmse(estimates, actuals)


def compute_shape_rmse(
    estimated_points_base_mm: Sequence[Vector3],
    tracker_points_base_mm: Sequence[Vector3],
) -> float:
    return vector_rmse(list(estimated_points_base_mm), list(tracker_points_base_mm))


def error_norm_series(
    rows: Sequence[Mapping[str, object]],
    *,
    estimate_prefix: str,
    actual_prefix: str = "P_actual_base_mm",
) -> list[float]:
    errors: list[float] = []
    for row in rows:
        estimate = read_vector(row, estimate_prefix)
        actual = read_vector(row, actual_prefix)
        dx = estimate[0] - actual[0]
        dy = estimate[1] - actual[1]
        dz = estimate[2] - actual[2]
        errors.append(math.sqrt((dx * dx) + (dy * dy) + (dz * dz)))
    return errors


def compute_settling_time(
    times_s: Sequence[float],
    errors_mm: Sequence[float],
    *,
    threshold_mm: float,
    hold_time_s: float = 0.0,
) -> float | None:
    if len(times_s) != len(errors_mm):
        raise ValueError("times_s and errors_mm must have the same length")
    for index, time_s in enumerate(times_s):
        window_end = time_s + hold_time_s
        window_errors = [
            error
            for candidate_time, error in zip(times_s[index:], errors_mm[index:])
            if candidate_time <= window_end
        ]
        remaining_errors = list(errors_mm[index:])
        if window_errors and all(error <= threshold_mm for error in window_errors):
            if hold_time_s > 0.0 or all(error <= threshold_mm for error in remaining_errors):
                return time_s
    return None


def compute_transient_rms(errors_mm: Sequence[float], *, end_index: int | None = None) -> float:
    values = list(errors_mm if end_index is None else errors_mm[:end_index])
    if not values:
        return 0.0
    return math.sqrt(sum(error * error for error in values) / len(values))


def compute_residual_vibration_rms(
    values_mm: Sequence[float],
    *,
    start_index: int = 0,
) -> float:
    tail = list(values_mm[start_index:])
    if not tail:
        return 0.0
    mean = sum(tail) / len(tail)
    return math.sqrt(sum((value - mean) ** 2 for value in tail) / len(tail))


def dropped_sample_ratio(rows: Sequence[Mapping[str, object]]) -> float:
    if not rows:
        return 0.0
    dropped = 0
    for row in rows:
        flags = str(row.get("quality_flags", ""))
        if "dropped_sample_flag" in flags:
            dropped += 1
    return dropped / len(rows)


def build_analysis_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    trial_id: str,
    condition_id: str,
    robot_model: str,
    target_id: str,
    estimate_prefix: str = "P_hat_imu_base_mm",
    settling_threshold_mm: float = 5.0,
) -> dict[str, object]:
    errors = error_norm_series(rows, estimate_prefix=estimate_prefix)
    times = [_float(row, "t_common_s") for row in rows]
    settling = compute_settling_time(
        times,
        errors,
        threshold_mm=settling_threshold_mm,
        hold_time_s=0.0,
    )
    return {
        "trial_id": trial_id,
        "condition_id": condition_id,
        "robot_model": robot_model,
        "target_id": target_id,
        "position_rmse_mm": f"{compute_position_rmse(rows, estimate_prefix=estimate_prefix):.6f}",
        "shape_rmse_mm": "",
        "settling_time_s": "" if settling is None else f"{settling:.6f}",
        "transient_rms_mm": f"{compute_transient_rms(errors):.6f}",
        "residual_vibration_rms_mm": f"{compute_residual_vibration_rms(errors):.6f}",
        "dropped_sample_ratio": f"{dropped_sample_ratio(rows):.6f}",
    }
