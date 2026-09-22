from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectories.minimum_jerk import generate_minimum_jerk


def test_minimum_jerk_starts_and_ends_at_requested_q():
    q0 = np.array([0.0, 1.0, -1.0])
    q1 = np.array([0.4, 0.2, -0.6])
    traj = generate_minimum_jerk(q0, q1, duration=2.0, dt=0.01)
    assert np.allclose(traj.q[0], q0, atol=1e-12)
    assert np.allclose(traj.q[-1], q1, atol=1e-12)


def test_minimum_jerk_endpoint_velocity_is_zero():
    q0 = np.array([0.0, 1.0, -1.0])
    q1 = np.array([0.4, 0.2, -0.6])
    traj = generate_minimum_jerk(q0, q1, duration=2.0, dt=0.01)
    assert np.allclose(traj.qdot[0], 0.0, atol=1e-12)
    assert np.allclose(traj.qdot[-1], 0.0, atol=1e-12)


def test_minimum_jerk_endpoint_acceleration_is_zero():
    q0 = np.array([0.0, 1.0, -1.0])
    q1 = np.array([0.4, 0.2, -0.6])
    traj = generate_minimum_jerk(q0, q1, duration=2.0, dt=0.01)
    assert np.allclose(traj.qddot[0], 0.0, atol=1e-12)
    assert np.allclose(traj.qddot[-1], 0.0, atol=1e-12)
