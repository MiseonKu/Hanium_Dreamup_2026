from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PoseSample:
    t_common_s: float
    p_base_mm: Vector3
    q_world: Quaternion
    source: str
    quality_flag: str = "ok"


@dataclass(frozen=True, slots=True)
class ImuRawSample:
    imu_id: str
    link_id: str
    imu_position_id: str
    t_request_s: float
    t_receive_s: float
    t_imu_corrected_s: float
    acc_raw_m_s2: Vector3
    gyro_raw_rad_s: Vector3
    dropped_sample_flag: bool = False
    delayed_sample_flag: bool = False


@dataclass(frozen=True, slots=True)
class ImuAttitudeSample:
    imu_id: str
    link_id: str
    imu_position_id: str
    t_imu_corrected_s: float
    q_imu_world: Quaternion
    roll_pitch_yaw_world_rad: Vector3
    filter_type: str
    yaw_drift_estimate_rad: float = 0.0
    attitude_quality_flag: str = "ok"


@dataclass(frozen=True, slots=True)
class MotorSample:
    motor_id: int
    joint_id: str
    t_encoder_corrected_s: float
    q_encoder_rad: float
    motor_velocity_rad_s: float
    motor_current: float
    motor_temperature_c: float
    communication_status: str = "ok"


@dataclass(frozen=True, slots=True)
class TrackerSample:
    marker_id: str
    t_tracker_corrected_s: float
    p_actual_base_mm: Vector3
    q_marker_world: Quaternion
    occluded: bool = False


@dataclass(frozen=True, slots=True)
class QualityFlags:
    dropped_sample_flag: bool = False
    delayed_sample_flag: bool = False
    timestamp_gap_flag: bool = False
    tracker_occlusion_flag: bool = False
    base_motion_flag: bool = False
    yaw_drift_flag: bool = False
    motor_fault_flag: bool = False

    def as_pipe_string(self) -> str:
        active = [name for name, value in self.__dict__.items() if value]
        return "ok" if not active else "|".join(active)

    def to_jsonable(self) -> dict[str, bool]:
        return dict(self.__dict__)


@dataclass(frozen=True, slots=True)
class TrialMetadata:
    trial_id: str
    condition_id: str
    robot_model: str
    target_id: str
    repeat_id: int
    date: str
    calibration_versions: dict[str, str] = field(default_factory=dict)
    filter_params_id: str = "filter_default"
    fusion_params_id: str = "fusion_default"
    quality_thresholds: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    @property
    def filename_stem(self) -> str:
        return (
            f"trial_{self.trial_id}_{self.robot_model}_{self.condition_id}_"
            f"{self.target_id}_rep{self.repeat_id:02d}_{self.date}"
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "condition_id": self.condition_id,
            "robot_model": self.robot_model,
            "target_id": self.target_id,
            "repeat_id": self.repeat_id,
            "date": self.date,
            "calibration_versions": self.calibration_versions,
            "filter_params_id": self.filter_params_id,
            "fusion_params_id": self.fusion_params_id,
            "quality_thresholds": self.quality_thresholds,
            "notes": self.notes,
            "filename_stem": self.filename_stem,
        }
