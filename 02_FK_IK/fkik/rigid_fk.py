"""Rigid 3DoF FK for the SO-101 A/B-link setup."""

from __future__ import annotations

import numpy as np

from .geometry import RobotConfig, apply_base_yaw, as_q, rotation2


def fk_rigid(q: np.ndarray, config: RobotConfig) -> np.ndarray:
    """Compute rigid FK.

    Args:
        q: [q1, q2, q3] in radians.
        config: RobotConfig with L1/L2 in millimeters.

    Returns:
        End-effector or tracker point [x, y, z] in millimeters.
    """

    q1, q2, q3 = as_q(q)
    alpha2 = q2 + q3
    p_plane = np.array(
        [
            config.L1 * np.cos(q2) + config.L2 * np.cos(alpha2),
            config.L1 * np.sin(q2) + config.L2 * np.sin(alpha2),
        ],
        dtype=float,
    )
    if config.tracker_offset_mm is not None:
        p_plane = p_plane + rotation2(alpha2) @ config.tracker_offset_mm
    return apply_base_yaw(p_plane, q1)
