"""Derived feature columns and named, leakage-safe feature views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .cleaning import IMU_PREFIXES


LABEL_COLUMNS = (
    "vive_pos_raw_x_mm",
    "vive_pos_raw_y_mm",
    "vive_pos_raw_z_mm",
)
ENCODER_COLUMNS = tuple(f"encoder_delta_rad_j{joint}" for joint in range(3))
PLANNED_COLUMNS = (
    "planned_p_x_mm",
    "planned_p_y_mm",
    "planned_p_z_mm",
)


def _euler_source_columns() -> tuple[str, ...]:
    return tuple(
        f"{prefix}_euler_{axis}_rad"
        for prefix in IMU_PREFIXES
        for axis in ("x", "y", "z")
    )


EULER_SOURCE_COLUMNS = _euler_source_columns()
EULER_TRIG_COLUMNS = tuple(
    name
    for source in EULER_SOURCE_COLUMNS
    for name in (f"{source}_sin", f"{source}_cos")
)
ACCEL_COLUMNS = tuple(
    f"{prefix}_accel_{axis}_mps2"
    for prefix in IMU_PREFIXES
    for axis in ("x", "y", "z")
)
GYRO_COLUMNS = tuple(f"{prefix}_gyro_rms_rad_s" for prefix in IMU_PREFIXES)
MAG_COLUMNS = tuple(
    f"{prefix}_mag_{axis}_raw"
    for prefix in IMU_PREFIXES
    for axis in ("x", "y", "z")
)
FOUR_IMU_ENDPOINT_COLUMNS = EULER_TRIG_COLUMNS + ACCEL_COLUMNS + GYRO_COLUMNS
FUSION_COLUMNS = ENCODER_COLUMNS + FOUR_IMU_ENDPOINT_COLUMNS


_FEATURE_VIEWS: dict[str, tuple[str, ...]] = {
    "encoder_only_v1": ENCODER_COLUMNS,
    "four_imu_endpoint_v1": FOUR_IMU_ENDPOINT_COLUMNS,
    "fusion_v1": FUSION_COLUMNS,
    "command_context_ablation_v1": FUSION_COLUMNS + PLANNED_COLUMNS,
    "mag_ablation_v1": FUSION_COLUMNS + MAG_COLUMNS,
}


def feature_views() -> dict[str, list[str]]:
    """Return independent copies of the versioned feature-view definitions."""

    return {name: list(columns) for name, columns in _FEATURE_VIEWS.items()}


def add_feature_columns(frame: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
    """Add sin/cos Euler representation without interpreting the Euler convention."""

    missing = sorted(set(EULER_SOURCE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"cannot derive Euler features; missing columns: {missing}")
    result = frame.copy() if copy else frame
    for source in EULER_SOURCE_COLUMNS:
        values = result[source].to_numpy(dtype=np.float64)
        result[f"{source}_sin"] = np.sin(values)
        result[f"{source}_cos"] = np.cos(values)
    return result


def validate_feature_views(
    frame: pd.DataFrame, views: Mapping[str, Sequence[str]] | None = None
) -> None:
    """Fail when a declared view is absent, non-finite, or leaks label context."""

    selected_views = feature_views() if views is None else views
    for name, columns in selected_views.items():
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f"feature view {name!r} is missing columns: {missing}")
        values = frame[list(columns)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"feature view {name!r} contains non-finite values")

    for primary in ("encoder_only_v1", "four_imu_endpoint_v1", "fusion_v1"):
        if primary not in selected_views:
            continue
        columns = selected_views[primary]
        leaked = [
            column
            for column in columns
            if column.startswith("vive_") or column.startswith("planned_")
        ]
        if leaked:
            raise ValueError(f"primary feature view {primary!r} leaks context: {leaked}")


def select_feature_view(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Select a named feature matrix after validating its columns and values."""

    views = feature_views()
    if name not in views:
        raise KeyError(f"unknown feature view {name!r}; choose from {sorted(views)}")
    columns = views[name]
    validate_feature_views(frame, {name: columns})
    return frame.loc[:, columns].copy()
