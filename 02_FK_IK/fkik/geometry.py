"""Shared geometry, config, and result types.

Units are millimeters for position/link lengths and radians for joint angles.
The base frame is x forward, y left, z up. The local link frame is x_l along
the link, y_l along the wide face, and z_l along the weak bending axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


@dataclass(frozen=True)
class RobotConfig:
    """Parsed robot and experiment configuration."""

    name: str
    dof: int
    link_length_preset: str
    L1: float
    L2: float
    use_c_link: bool
    tracker_offset_mm: np.ndarray
    material_preset: str
    wall_count: int
    top_bottom_wall_count: int
    infill_percent: int
    infill_type: str
    cross_section_mm: np.ndarray
    k1: float
    k2: float
    max_delta_rad: float
    bending_score_weights: np.ndarray
    joint_limits_rad: np.ndarray
    q_init_rad: np.ndarray
    n_candidates: int
    n_targets: int
    repeats_per_point: int
    random_seed: int
    latin_hypercube: bool
    azimuth_bins: int
    min_joint_margin_rad: float
    min_singularity_score: float
    max_d_flex_mm: float
    floor_min_z_mm: float
    round_trip_epsilon_mm: float
    validate_round_trip: bool
    trajectory_type: str
    duration_sec: float
    dt_sec: float
    speed_scale: float
    ik_tolerance_mm: float
    ik_max_iter: int
    ik_damping_lambda: float
    finite_difference_eps: float
    ik_max_step_rad: float
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def link_a_mm(self) -> float:
        """A-link length in millimeters."""

        return self.L1

    @property
    def link_b_mm(self) -> float:
        """B-link length in millimeters."""

        return self.L2

    @property
    def nominal_reach_mm(self) -> float:
        """Nominal rigid reach for q=[0,0,0], excluding tracker offset."""

        return self.L1 + self.L2


@dataclass(frozen=True)
class FlexFKResult:
    """Forward kinematics result for the pseudo-rigid flex model."""

    p_flex: np.ndarray
    p_rigid: np.ndarray
    delta1: float
    delta2: float
    M1_weak: float
    M2_weak: float
    d_flex: float
    bending_score: float


@dataclass(frozen=True)
class IKSolution:
    """One candidate IK solution."""

    q: np.ndarray
    elbow: str
    within_limits: bool
    joint_margin: float
    singularity_score: float
    round_trip_error: float
    score: float
    reason: str = ""


@dataclass(frozen=True)
class IKResult:
    """Inverse kinematics result with all candidates and selected command."""

    success: bool
    q: np.ndarray | None
    reason: str
    residual_error: float
    iterations: int
    solutions: list[IKSolution] = field(default_factory=list)
    selected_index: int | None = None


@dataclass(frozen=True)
class Trajectory:
    """Joint-space trajectory in radians, radians/s, and radians/s^2."""

    time: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    qddot: np.ndarray
    move_duration: float
    speed_scale: float


def load_config(path: str | Path) -> RobotConfig:
    """Load a YAML config file into a typed RobotConfig."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    robot = data["robot"]
    material = data["material"]
    flex_model = data["flex_model"]
    sampling = data["sampling"]
    trajectory = data["trajectory"]
    ik = data["ik"]
    limits = data["joint_limits_rad"]

    joint_limits = np.array([limits["q1"], limits["q2"], limits["q3"]], dtype=float)
    link_a_mm, link_b_mm = _parse_link_lengths(robot["link_lengths_mm"])

    return RobotConfig(
        name=str(robot["name"]),
        dof=int(robot["dof"]),
        link_length_preset=str(robot.get("link_length_preset", "custom_AB")),
        L1=link_a_mm,
        L2=link_b_mm,
        use_c_link=bool(robot.get("use_c_link", False)),
        tracker_offset_mm=np.asarray(robot.get("tracker_offset_mm", [0.0, 0.0]), dtype=float),
        q_init_rad=np.asarray(robot.get("q_init_rad", [0.0, 0.0, 0.0]), dtype=float),
        material_preset=str(material["preset"]),
        wall_count=int(material["wall_count"]),
        top_bottom_wall_count=int(material.get("top_bottom_wall_count", material["wall_count"])),
        infill_percent=int(material["infill_percent"]),
        infill_type=str(material["infill_type"]),
        cross_section_mm=np.asarray(material["cross_section_mm"], dtype=float),
        k1=float(flex_model["k1"]),
        k2=float(flex_model["k2"]),
        max_delta_rad=float(flex_model["max_delta_rad"]),
        bending_score_weights=np.asarray(flex_model["bending_score_weights"], dtype=float),
        joint_limits_rad=joint_limits,
        n_candidates=int(sampling["n_candidates"]),
        n_targets=int(sampling["n_targets"]),
        repeats_per_point=int(sampling["repeats_per_point"]),
        random_seed=int(sampling["random_seed"]),
        latin_hypercube=bool(sampling.get("latin_hypercube", True)),
        azimuth_bins=int(sampling.get("azimuth_bins", 4)),
        min_joint_margin_rad=float(sampling.get("min_joint_margin_rad", 0.0)),
        min_singularity_score=float(sampling.get("min_singularity_score", 0.0)),
        max_d_flex_mm=float(sampling.get("max_d_flex_mm", np.inf)),
        floor_min_z_mm=float(sampling.get("floor_min_z_mm", -np.inf)),
        round_trip_epsilon_mm=float(sampling.get("round_trip_epsilon_mm", ik["tolerance_mm"])),
        validate_round_trip=bool(sampling.get("validate_round_trip", True)),
        trajectory_type=str(trajectory.get("type", "minimum_jerk")),
        duration_sec=float(trajectory["duration_sec"]),
        dt_sec=float(trajectory["dt_sec"]),
        speed_scale=float(trajectory.get("speed_scale", 1.0)),
        ik_tolerance_mm=float(ik["tolerance_mm"]),
        ik_max_iter=int(ik["max_iter"]),
        ik_damping_lambda=float(ik["damping_lambda"]),
        finite_difference_eps=float(ik["finite_difference_eps"]),
        ik_max_step_rad=float(ik.get("max_step_rad", 0.25)),
        raw=data,
    )


def _parse_link_lengths(link_lengths: dict[str, Any]) -> tuple[float, float]:
    """Read configurable A/B link lengths with L1/L2 backward compatibility."""

    def read(primary: str, aliases: tuple[str, ...]) -> float:
        for key in (primary, *aliases):
            if key in link_lengths:
                value = float(link_lengths[key])
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"link_lengths_mm.{key} must be a positive finite millimeter value")
                return value
        names = ", ".join((primary, *aliases))
        raise KeyError(f"missing link length key; expected one of: {names}")

    return read("A", ("L1", "link_a_mm")), read("B", ("L2", "link_b_mm"))


def as_q(q: ArrayLike) -> np.ndarray:
    """Return q as a shape-(3,) float vector in radians."""

    arr = np.asarray(q, dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError("q contains non-finite values")
    return arr


def as_point(p: ArrayLike) -> np.ndarray:
    """Return p as a shape-(3,) float vector in millimeters."""

    arr = np.asarray(p, dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError("point contains non-finite values")
    return arr


def rotation2(theta: float) -> np.ndarray:
    """Return a 2D rotation matrix for theta radians."""

    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=float)


def tracker_effective_link(config: RobotConfig) -> tuple[float, float]:
    """Return effective distal length and angular offset for planar IK.

    The tracker offset is expressed in the B-link local x-z plane from the
    B-link endpoint. It is folded into an equivalent second link vector.
    """

    dx, dz = config.tracker_offset_mm
    x_eff = config.L2 + float(dx)
    z_eff = float(dz)
    return float(np.hypot(x_eff, z_eff)), float(np.arctan2(z_eff, x_eff))


def apply_base_yaw(rho_z: np.ndarray, q1: float) -> np.ndarray:
    """Map planar [rho, z] into base-frame [x, y, z]."""

    rho = float(rho_z[0])
    z = float(rho_z[1])
    return np.array([rho * np.cos(q1), rho * np.sin(q1), z], dtype=float)


def joint_margin(q: ArrayLike, config: RobotConfig) -> float:
    """Return the smallest distance to a joint limit in radians."""

    q_arr = as_q(q)
    lower = config.joint_limits_rad[:, 0]
    upper = config.joint_limits_rad[:, 1]
    return float(np.min(np.minimum(q_arr - lower, upper - q_arr)))


def within_joint_limits(q: ArrayLike, config: RobotConfig, margin: float = 0.0) -> bool:
    """Return True when q is inside configured joint limits plus margin."""

    q_arr = as_q(q)
    lower = config.joint_limits_rad[:, 0] + margin
    upper = config.joint_limits_rad[:, 1] - margin
    return bool(np.all((q_arr >= lower) & (q_arr <= upper)))


def singularity_score(q: ArrayLike, config: RobotConfig) -> float:
    """Return a 0..1 score where larger is farther from planar singularity."""

    q_arr = as_q(q)
    _, beta = tracker_effective_link(config)
    return float(abs(np.sin(q_arr[2] + beta)))


def normalize_features(values: np.ndarray) -> np.ndarray:
    """Normalize columns to zero mean and unit range for distance metrics."""

    arr = np.asarray(values, dtype=float)
    center = np.nanmean(arr, axis=0)
    span = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
    span = np.where(span > 1e-12, span, 1.0)
    return (arr - center) / span


def point_class_string(row: Any) -> str:
    """Build a stable point-class string from bin label fields."""

    return "|".join(
        [
            str(row["radial_shell"]),
            str(row["azimuth_bin"]),
            str(row["height_bin"]),
            str(row["bending_class"]),
            str(row["joint_posture_class"]),
        ]
    )
