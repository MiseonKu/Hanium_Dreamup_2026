"""Lightweight pseudo-rigid FK for TPU flexible links."""

from __future__ import annotations

import numpy as np

from .geometry import FlexFKResult, RobotConfig, apply_base_yaw, as_q, rotation2
from .rigid_fk import fk_rigid


def compute_weak_moment_proxy(q: np.ndarray, config: RobotConfig) -> tuple[float, float]:
    """Return replaceable posture-based weak-axis bending proxies.

    The proxy is gravity/posture based, not a calibrated physical moment. A
    horizontal link receives a larger sag proxy than a vertical link. Positive
    proxy values bend downward in the planar FK convention.
    """

    _, q2, q3 = as_q(q)
    alpha1 = q2
    alpha2 = q2 + q3
    m1 = 0.5 * config.L1 * np.cos(alpha1) + config.L2 * np.cos(alpha2)
    m2 = 0.5 * config.L2 * np.cos(alpha2)
    return float(m1), float(m2)


def compute_deltas(q: np.ndarray, config: RobotConfig) -> tuple[float, float]:
    """Compute pseudo-rigid deflection angles delta1 and delta2 in radians."""

    m1, m2 = compute_weak_moment_proxy(q, config)
    return float(config.k1 * m1), float(config.k2 * m2)


def fk_flex(q: np.ndarray, config: RobotConfig) -> FlexFKResult:
    """Compute flex-aware FK using a pseudo-rigid two-link approximation.

    Args:
        q: [q1, q2, q3] in radians.
        config: RobotConfig.

    Returns:
        FlexFKResult containing flex and rigid positions in millimeters,
        deflection angles in radians, weak moment proxies, and scores.
    """

    q1, q2, q3 = as_q(q)
    m1, m2 = compute_weak_moment_proxy(q, config)
    delta1 = float(config.k1 * m1)
    delta2 = float(config.k2 * m2)

    alpha1 = q2
    alpha1_mid = alpha1 - delta1 / 2.0
    p1_plane = config.L1 * np.array([np.cos(alpha1_mid), np.sin(alpha1_mid)])

    alpha2 = q2 + q3 - delta1
    alpha2_mid = alpha2 - delta2 / 2.0
    p2_plane = p1_plane + config.L2 * np.array([np.cos(alpha2_mid), np.sin(alpha2_mid)])

    if config.tracker_offset_mm is not None:
        p2_plane = p2_plane + rotation2(alpha2 - delta2) @ config.tracker_offset_mm

    p_flex = apply_base_yaw(p2_plane, q1)
    p_rigid = fk_rigid(np.array([q1, q2, q3], dtype=float), config)
    d_flex = float(np.linalg.norm(p_flex - p_rigid))
    w1, w2, w3 = config.bending_score_weights
    bending_score = float(w1 * abs(m1) + w2 * abs(m2) + w3 * d_flex)

    return FlexFKResult(
        p_flex=p_flex,
        p_rigid=p_rigid,
        delta1=delta1,
        delta2=delta2,
        M1_weak=m1,
        M2_weak=m2,
        d_flex=d_flex,
        bending_score=bending_score,
    )
