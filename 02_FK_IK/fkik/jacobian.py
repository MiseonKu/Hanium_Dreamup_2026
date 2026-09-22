"""Numerical Jacobians for flex-aware IK."""

from __future__ import annotations

import numpy as np

from .flex_fk import fk_flex
from .geometry import RobotConfig, as_q


def numerical_jacobian_flex(q: np.ndarray, config: RobotConfig, eps: float | None = None) -> np.ndarray:
    """Return finite-difference d(position)/d(q) for fk_flex.

    Position is in millimeters and q is in radians, so each column has units
    mm/rad.
    """

    q_arr = as_q(q)
    step = float(config.finite_difference_eps if eps is None else eps)
    p0 = fk_flex(q_arr, config).p_flex
    jac = np.zeros((3, 3), dtype=float)
    for i in range(3):
        q_step = q_arr.copy()
        q_step[i] += step
        jac[:, i] = (fk_flex(q_step, config).p_flex - p0) / step
    return jac
