"""Minimum-jerk joint trajectory generation."""

from __future__ import annotations

import numpy as np

from fkik.geometry import Trajectory


def generate_minimum_jerk(q_start: np.ndarray, q_goal: np.ndarray, duration: float, dt: float) -> Trajectory:
    """Generate a minimum-jerk trajectory between joint vectors.

    Args:
        q_start: Start q in radians.
        q_goal: Goal q in radians.
        duration: Move duration in seconds.
        dt: Sample period in seconds.

    Returns:
        Trajectory with time, q, qdot, and qddot.
    """

    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    q0 = np.asarray(q_start, dtype=float).reshape(3)
    q1 = np.asarray(q_goal, dtype=float).reshape(3)
    n = int(np.floor(duration / dt)) + 1
    time = np.linspace(0.0, duration, n, dtype=float)
    tau = time / duration
    dq = q1 - q0

    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    ds = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / duration
    dds = (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / (duration * duration)

    q = q0[None, :] + s[:, None] * dq[None, :]
    qdot = ds[:, None] * dq[None, :]
    qddot = dds[:, None] * dq[None, :]
    return Trajectory(time=time, q=q, qdot=qdot, qddot=qddot, move_duration=float(duration), speed_scale=1.0)
