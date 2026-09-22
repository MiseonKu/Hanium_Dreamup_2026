"""High-level trajectory helpers."""

from __future__ import annotations

import numpy as np

from fkik.geometry import RobotConfig, Trajectory
from trajectories.minimum_jerk import generate_minimum_jerk


def generate_joint_trajectory(q_goal: np.ndarray, config: RobotConfig, q_start: np.ndarray | None = None) -> Trajectory:
    """Generate the configured joint trajectory from q_start to q_goal."""

    start = config.q_init_rad if q_start is None else np.asarray(q_start, dtype=float).reshape(3)
    duration = config.duration_sec / max(config.speed_scale, 1e-9)
    traj = generate_minimum_jerk(start, q_goal, duration, config.dt_sec)
    return Trajectory(traj.time, traj.q, traj.qdot, traj.qddot, duration, config.speed_scale)
