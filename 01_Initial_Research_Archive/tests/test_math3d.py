from __future__ import annotations

import math

from research_host.math3d import (
    quaternion_from_axis_angle,
    quaternion_normalize,
    quaternion_rotate_vector,
    quaternion_slerp,
    vector_rmse,
)


def test_quaternion_normalize() -> None:
    assert quaternion_normalize((2.0, 0.0, 0.0, 0.0)) == (1.0, 0.0, 0.0, 0.0)


def test_quaternion_rotate_local_x_about_z() -> None:
    q = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2.0)
    rotated = quaternion_rotate_vector(q, (1.0, 0.0, 0.0))
    assert abs(rotated[0]) < 1e-9
    assert abs(rotated[1] - 1.0) < 1e-9
    assert abs(rotated[2]) < 1e-9


def test_slerp_endpoints() -> None:
    start = (1.0, 0.0, 0.0, 0.0)
    end = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.pi)
    assert quaternion_slerp(start, end, 0.0) == start
    assert abs(quaternion_slerp(start, end, 1.0)[3] - 1.0) < 1e-9


def test_vector_rmse() -> None:
    actual = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)]
    expected = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    assert abs(vector_rmse(actual, expected) - math.sqrt(12.5)) < 1e-9
