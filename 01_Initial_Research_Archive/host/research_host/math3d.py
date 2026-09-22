from __future__ import annotations

import math

from .types import Quaternion, Vector3


EPSILON = 1e-12
IDENTITY_QUATERNION: Quaternion = (1.0, 0.0, 0.0, 0.0)
UNIT_X: Vector3 = (1.0, 0.0, 0.0)


def vector_add(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def vector_sub(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def vector_scale(vector: Vector3, scale: float) -> Vector3:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def vector_dot(left: Vector3, right: Vector3) -> float:
    return (left[0] * right[0]) + (left[1] * right[1]) + (left[2] * right[2])


def vector_norm(vector: Vector3) -> float:
    return math.sqrt(vector_dot(vector, vector))


def vector_normalize(vector: Vector3) -> Vector3:
    norm = vector_norm(vector)
    if norm <= EPSILON:
        return (0.0, 0.0, 0.0)
    return vector_scale(vector, 1.0 / norm)


def vector_rmse(actual: list[Vector3], expected: list[Vector3]) -> float:
    if len(actual) != len(expected):
        raise ValueError("actual and expected must have the same length")
    if not actual:
        return 0.0
    squared = 0.0
    for actual_vector, expected_vector in zip(actual, expected):
        delta = vector_sub(actual_vector, expected_vector)
        squared += vector_dot(delta, delta)
    return math.sqrt(squared / len(actual))


def quaternion_normalize(quaternion: Quaternion) -> Quaternion:
    w, x, y, z = quaternion
    norm = math.sqrt((w * w) + (x * x) + (y * y) + (z * z))
    if norm <= EPSILON:
        return IDENTITY_QUATERNION
    return (w / norm, x / norm, y / norm, z / norm)


def quaternion_conjugate(quaternion: Quaternion) -> Quaternion:
    w, x, y, z = quaternion_normalize(quaternion)
    return (w, -x, -y, -z)


def _quaternion_multiply_raw(left: Quaternion, right: Quaternion) -> Quaternion:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        (lw * rw) - (lx * rx) - (ly * ry) - (lz * rz),
        (lw * rx) + (lx * rw) + (ly * rz) - (lz * ry),
        (lw * ry) - (lx * rz) + (ly * rw) + (lz * rx),
        (lw * rz) + (lx * ry) - (ly * rx) + (lz * rw),
    )


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    return quaternion_normalize(_quaternion_multiply_raw(left, right))


def quaternion_from_axis_angle(axis: Vector3, angle_rad: float) -> Quaternion:
    unit_axis = vector_normalize(axis)
    if vector_norm(unit_axis) <= EPSILON:
        return IDENTITY_QUATERNION
    half = angle_rad * 0.5
    sin_half = math.sin(half)
    return quaternion_normalize(
        (
            math.cos(half),
            unit_axis[0] * sin_half,
            unit_axis[1] * sin_half,
            unit_axis[2] * sin_half,
        )
    )


def quaternion_from_euler(roll_rad: float, pitch_rad: float, yaw_rad: float) -> Quaternion:
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)
    return quaternion_normalize(
        (
            (cr * cp * cy) + (sr * sp * sy),
            (sr * cp * cy) - (cr * sp * sy),
            (cr * sp * cy) + (sr * cp * sy),
            (cr * cp * sy) - (sr * sp * cy),
        )
    )


def quaternion_rotate_vector(quaternion: Quaternion, vector: Vector3) -> Vector3:
    q = quaternion_normalize(quaternion)
    v = (0.0, vector[0], vector[1], vector[2])
    rotated = _quaternion_multiply_raw(
        _quaternion_multiply_raw(q, v), quaternion_conjugate(q)
    )
    return (rotated[1], rotated[2], rotated[3])


def quaternion_slerp(start: Quaternion, end: Quaternion, fraction: float) -> Quaternion:
    t = max(0.0, min(1.0, float(fraction)))
    q0 = quaternion_normalize(start)
    q1 = quaternion_normalize(end)
    dot = (q0[0] * q1[0]) + (q0[1] * q1[1]) + (q0[2] * q1[2]) + (q0[3] * q1[3])

    if dot < 0.0:
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
        dot = -dot

    if dot > 0.9995:
        blended = (
            q0[0] + (t * (q1[0] - q0[0])),
            q0[1] + (t * (q1[1] - q0[1])),
            q0[2] + (t * (q1[2] - q0[2])),
            q0[3] + (t * (q1[3] - q0[3])),
        )
        return quaternion_normalize(blended)

    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    scale_0 = math.cos(theta) - (dot * sin_theta / sin_theta_0)
    scale_1 = sin_theta / sin_theta_0
    return quaternion_normalize(
        (
            (scale_0 * q0[0]) + (scale_1 * q1[0]),
            (scale_0 * q0[1]) + (scale_1 * q1[1]),
            (scale_0 * q0[2]) + (scale_1 * q1[2]),
            (scale_0 * q0[3]) + (scale_1 * q1[3]),
        )
    )
