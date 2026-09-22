from __future__ import annotations

import math

from ..math3d import quaternion_from_axis_angle
from ..types import ImuAttitudeSample, Quaternion, TrialMetadata


def synthetic_trial_metadata() -> TrialMetadata:
    return TrialMetadata(
        trial_id="0001",
        condition_id="C2_imu_array_only",
        robot_model="long",
        target_id="T001",
        repeat_id=1,
        date="2026-05-01",
        calibration_versions={
            "imu_bias": "template",
            "imu_alignment": "template",
            "dh_params": "template",
            "latency": "template",
        },
        quality_thresholds={
            "max_jitter_s": 0.002,
            "max_dropped_sample_ratio": 0.01,
            "settling_threshold_mm": 5.0,
        },
        notes="Synthetic fixture for offline smoke tests.",
    )


def synthetic_link_attitudes(
    *,
    bend_deg: float = 12.0,
) -> list[tuple[Quaternion, Quaternion, Quaternion]]:
    mid_bend = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.radians(bend_deg))
    small_bend = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.radians(bend_deg * 0.4))
    return [
        ((1.0, 0.0, 0.0, 0.0), mid_bend, small_bend),
        (small_bend, mid_bend, small_bend),
        (small_bend, small_bend, small_bend),
    ]


class SyntheticAttitudeReader:
    def __init__(self, *, sample_count: int = 20, dt_s: float = 0.01) -> None:
        self.sample_count = sample_count
        self.dt_s = dt_s

    def samples(self) -> list[ImuAttitudeSample]:
        rows: list[ImuAttitudeSample] = []
        for index in range(self.sample_count):
            t_s = index * self.dt_s
            bend = 4.0 + (2.0 * math.sin(t_s * math.pi))
            q = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.radians(bend))
            rows.append(
                ImuAttitudeSample(
                    imu_id="synthetic_link1_mid",
                    link_id="link_1",
                    imu_position_id="mid",
                    t_imu_corrected_s=t_s,
                    q_imu_world=q,
                    roll_pitch_yaw_world_rad=(0.0, 0.0, math.radians(bend)),
                    filter_type="synthetic",
                )
            )
        return rows
