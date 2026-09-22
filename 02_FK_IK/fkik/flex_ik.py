"""Damped least-squares flex-aware IK."""

from __future__ import annotations

import numpy as np

from .flex_fk import fk_flex
from .geometry import IKResult, IKSolution, RobotConfig, as_point, joint_margin, singularity_score, within_joint_limits
from .jacobian import numerical_jacobian_flex
from .rigid_ik import ik_rigid


def _result_from_state(
    success: bool,
    q: np.ndarray | None,
    reason: str,
    residual: float,
    iterations: int,
    config: RobotConfig,
) -> IKResult:
    solutions: list[IKSolution] = []
    if q is not None:
        solutions.append(
            IKSolution(
                q=q.copy(),
                elbow="refined",
                within_limits=within_joint_limits(q, config),
                joint_margin=joint_margin(q, config),
                singularity_score=singularity_score(q, config),
                round_trip_error=residual,
                score=residual,
                reason=reason,
            )
        )
    return IKResult(success, None if q is None else q.copy(), reason, residual, iterations, solutions, 0 if q is not None else None)


def ik_flex(p_target: np.ndarray, config: RobotConfig, q_init: np.ndarray | None = None) -> IKResult:
    """Solve flex-aware IK with rigid IK initialization and DLS refinement.

    Args:
        p_target: Desired [x, y, z] in millimeters.
        config: RobotConfig.
        q_init: Optional reference posture in radians for branch selection.

    Returns:
        IKResult with command q, residual in millimeters, iteration count, and
        failure reason when convergence or safety checks fail.
    """

    p = as_point(p_target)
    rigid = ik_rigid(p, config, q_init=q_init)
    if not rigid.success or rigid.q is None:
        return IKResult(False, None, f"rigid init failed: {rigid.reason}", rigid.residual_error, 0, rigid.solutions)

    q = rigid.q.copy()
    lam = float(config.ik_damping_lambda)
    tol = float(config.ik_tolerance_mm)

    for iteration in range(config.ik_max_iter + 1):
        fk = fk_flex(q, config)
        e = p - fk.p_flex
        residual = float(np.linalg.norm(e))
        if residual <= tol:
            if abs(fk.delta1) > config.max_delta_rad or abs(fk.delta2) > config.max_delta_rad:
                return _result_from_state(False, q, "flex delta exceeds max_delta_rad", residual, iteration, config)
            return _result_from_state(True, q, "ok", residual, iteration, config)
        if iteration >= config.ik_max_iter:
            break

        jac = numerical_jacobian_flex(q, config)
        system = jac @ jac.T + (lam * lam) * np.eye(3)
        try:
            dq = jac.T @ np.linalg.solve(system, e)
        except np.linalg.LinAlgError:
            dq = jac.T @ np.linalg.pinv(system) @ e

        step_norm = float(np.linalg.norm(dq))
        if step_norm > config.ik_max_step_rad:
            dq *= config.ik_max_step_rad / step_norm

        accepted = False
        base_residual = residual
        for scale in (1.0, 0.5, 0.25, 0.1):
            q_next = q + scale * dq
            if not within_joint_limits(q_next, config):
                continue
            fk_next = fk_flex(q_next, config)
            if abs(fk_next.delta1) > config.max_delta_rad or abs(fk_next.delta2) > config.max_delta_rad:
                continue
            next_residual = float(np.linalg.norm(p - fk_next.p_flex))
            if next_residual <= base_residual or scale == 0.1:
                q = q_next
                accepted = True
                break
        if not accepted:
            return _result_from_state(False, q, "no valid DLS step within limits", residual, iteration, config)

    final_residual = float(np.linalg.norm(p - fk_flex(q, config).p_flex))
    return _result_from_state(False, q, "convergence failed", final_residual, config.ik_max_iter, config)
