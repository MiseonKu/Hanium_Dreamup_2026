"""Analytic rigid IK with elbow-up/down candidate scoring."""

from __future__ import annotations

import math

import numpy as np

from .geometry import (
    IKResult,
    IKSolution,
    RobotConfig,
    as_point,
    joint_margin,
    singularity_score,
    tracker_effective_link,
    within_joint_limits,
)
from .rigid_fk import fk_rigid


def _wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""

    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _candidate_score(
    q: np.ndarray,
    p_target: np.ndarray,
    config: RobotConfig,
    q_init: np.ndarray,
    within_limits: bool,
) -> tuple[float, float, float, float]:
    margin = joint_margin(q, config)
    singular = singularity_score(q, config)
    error = float(np.linalg.norm(fk_rigid(q, config) - p_target))
    q_span = config.joint_limits_rad[:, 1] - config.joint_limits_rad[:, 0]
    q_dist = float(np.linalg.norm((q - q_init) / q_span))
    score = q_dist - 0.10 * margin - 0.05 * singular + 0.01 * error
    if not within_limits:
        score += 1e6
    return score, margin, singular, error


def ik_rigid(p_target: np.ndarray, config: RobotConfig, q_init: np.ndarray | None = None) -> IKResult:
    """Solve rigid IK for target point [x, y, z] in millimeters.

    Both elbow-up and elbow-down branches are evaluated. The selected solution
    must be inside joint limits and is scored by q_init distance, joint-limit
    margin, singularity margin, and round-trip FK error.
    """

    p = as_point(p_target)
    q_ref = config.q_init_rad if q_init is None else np.asarray(q_init, dtype=float).reshape(3)
    x, y, z = p
    theta = math.atan2(y, x)
    radial = math.hypot(x, y)
    l1 = config.L1
    l2_eff, beta = tracker_effective_link(config)

    denom = 2.0 * l1 * l2_eff
    if denom <= 0.0:
        return IKResult(False, None, "invalid link length", np.inf, 0, [])

    D = (radial * radial + z * z - l1 * l1 - l2_eff * l2_eff) / denom
    if D < -1.0 - 1e-9 or D > 1.0 + 1e-9:
        return IKResult(False, None, "target unreachable", np.inf, 0, [])
    D = float(np.clip(D, -1.0, 1.0))

    solutions: list[IKSolution] = []
    root = math.sqrt(max(0.0, 1.0 - D * D))
    yaw_branches = [(theta, radial, "positive_radial"), (_wrap_pi(theta + math.pi), -radial, "negative_radial")]
    seen: set[tuple[int, int, int]] = set()
    for q1, rho, radial_name in yaw_branches:
        for sign, elbow_name in ((1.0, "elbow_down"), (-1.0, "elbow_up")):
            gamma = math.atan2(sign * root, D)
            q3 = gamma - beta
            q2 = math.atan2(z, rho) - math.atan2(l2_eff * math.sin(gamma), l1 + l2_eff * math.cos(gamma))
            q = np.array([q1, q2, q3], dtype=float)
            key = tuple(np.round(q, decimals=12))
            if key in seen:
                continue
            seen.add(key)
            ok = within_joint_limits(q, config)
            score, margin, singular, error = _candidate_score(q, p, config, q_ref, ok)
            reason = "" if ok else "joint limit violation"
            solutions.append(
                IKSolution(
                    q=q,
                    elbow=f"{elbow_name}_{radial_name}",
                    within_limits=ok,
                    joint_margin=margin,
                    singularity_score=singular,
                    round_trip_error=error,
                    score=score,
                    reason=reason,
                )
            )

    valid_indices = [i for i, sol in enumerate(solutions) if sol.within_limits]
    if not valid_indices:
        best = min(range(len(solutions)), key=lambda i: solutions[i].score)
        return IKResult(False, None, "no solution within joint limits", solutions[best].round_trip_error, 0, solutions)

    selected = min(valid_indices, key=lambda i: solutions[i].score)
    sol = solutions[selected]
    success = sol.round_trip_error <= max(config.ik_tolerance_mm, 1e-6)
    reason = "ok" if success else "round-trip error above tolerance"
    return IKResult(success, sol.q.copy(), reason, sol.round_trip_error, 0, solutions, selected)
