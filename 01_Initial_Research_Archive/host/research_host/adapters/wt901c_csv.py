from __future__ import annotations

import csv
import math
from pathlib import Path

from ..math3d import quaternion_from_euler
from ..types import ImuAttitudeSample


class WT901CCsvReader:
    def __init__(
        self,
        path: str | Path,
        *,
        imu_id: str,
        link_id: str,
        imu_position_id: str,
        filter_type: str = "wt901c_internal",
    ) -> None:
        self.path = Path(path)
        self.imu_id = imu_id
        self.link_id = link_id
        self.imu_position_id = imu_position_id
        self.filter_type = filter_type

    def samples(self) -> list[ImuAttitudeSample]:
        rows: list[ImuAttitudeSample] = []
        with self.path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if _has_quaternion(row):
                    quaternion = (
                        _read_float(row, "q_imu_world_w", "qw"),
                        _read_float(row, "q_imu_world_x", "qx"),
                        _read_float(row, "q_imu_world_y", "qy"),
                        _read_float(row, "q_imu_world_z", "qz"),
                    )
                    euler = (0.0, 0.0, 0.0)
                else:
                    euler = (
                        math.radians(_read_float(row, "roll_deg", "roll")),
                        math.radians(_read_float(row, "pitch_deg", "pitch")),
                        math.radians(_read_float(row, "yaw_deg", "yaw")),
                    )
                    quaternion = quaternion_from_euler(*euler)

                rows.append(
                    ImuAttitudeSample(
                        imu_id=self.imu_id,
                        link_id=self.link_id,
                        imu_position_id=self.imu_position_id,
                        t_imu_corrected_s=_read_float(
                            row, "t_imu_corrected_s", "host_time_s", "t_s"
                        ),
                        q_imu_world=quaternion,
                        roll_pitch_yaw_world_rad=euler,
                        filter_type=self.filter_type,
                        yaw_drift_estimate_rad=_read_float(row, "yaw_drift_rad"),
                        attitude_quality_flag=row.get("attitude_quality_flag", "ok"),
                    )
                )
        return rows


def _has_quaternion(row: dict[str, str]) -> bool:
    return any(row.get(name) not in (None, "") for name in ("q_imu_world_w", "qw"))


def _read_float(row: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return 0.0
