from __future__ import annotations

from dataclasses import dataclass

from .math3d import (
    UNIT_X,
    quaternion_rotate_vector,
    quaternion_slerp,
    vector_add,
    vector_norm,
    vector_scale,
    vector_sub,
)
from .types import Quaternion, Vector3


@dataclass(frozen=True, slots=True)
class LinkShapeResult:
    p_i_mid_base_mm: Vector3
    p_i_end_base_mm: Vector3
    shape_points_i_base_mm: tuple[Vector3, ...]
    delta_i_link_mm: Vector3
    arc_length_mm: float
    chord_length_mm: float
    shape_quality_flag: str


def _integrate_segment(
    q_start_world: Quaternion,
    q_end_world: Quaternion,
    segment_length_mm: float,
    *,
    steps: int,
) -> tuple[Vector3, list[Vector3]]:
    if steps < 1:
        raise ValueError("steps must be >= 1")

    ds = segment_length_mm / steps
    position = (0.0, 0.0, 0.0)
    local_points = [position]
    previous_tangent = quaternion_rotate_vector(q_start_world, UNIT_X)

    for step_index in range(1, steps + 1):
        fraction = step_index / steps
        q_current = quaternion_slerp(q_start_world, q_end_world, fraction)
        current_tangent = quaternion_rotate_vector(q_current, UNIT_X)
        average_tangent = vector_scale(
            vector_add(previous_tangent, current_tangent), 0.5
        )
        position = vector_add(position, vector_scale(average_tangent, ds))
        local_points.append(position)
        previous_tangent = current_tangent

    return position, local_points


def reconstruct_link_shape(
    imu_start_q_world: Quaternion,
    imu_mid_q_world: Quaternion,
    imu_end_q_world: Quaternion,
    link_length_mm: float,
    *,
    p_i_start_base_mm: Vector3 = (0.0, 0.0, 0.0),
    steps_per_half: int = 12,
    model: str = "arc_length_integration",
) -> LinkShapeResult:
    if model != "arc_length_integration":
        raise ValueError("Only arc_length_integration is implemented in this baseline")
    if link_length_mm <= 0.0:
        raise ValueError("link_length_mm must be positive")

    half_length = link_length_mm * 0.5
    first_delta, first_points = _integrate_segment(
        imu_start_q_world, imu_mid_q_world, half_length, steps=steps_per_half
    )
    second_delta, second_points = _integrate_segment(
        imu_mid_q_world, imu_end_q_world, half_length, steps=steps_per_half
    )

    p_mid = vector_add(p_i_start_base_mm, first_delta)
    p_end = vector_add(p_mid, second_delta)

    shape_points: list[Vector3] = []
    for point in first_points:
        shape_points.append(vector_add(p_i_start_base_mm, point))
    for point in second_points[1:]:
        shape_points.append(vector_add(p_mid, point))

    start_tangent = quaternion_rotate_vector(imu_start_q_world, UNIT_X)
    rigid_end = vector_add(
        p_i_start_base_mm, vector_scale(start_tangent, link_length_mm)
    )
    delta_i_link = vector_sub(p_end, rigid_end)
    chord_length = vector_norm(vector_sub(p_end, p_i_start_base_mm))
    quality = "ok"
    if chord_length > link_length_mm * 1.01:
        quality = "arc_length_violation"

    return LinkShapeResult(
        p_i_mid_base_mm=p_mid,
        p_i_end_base_mm=p_end,
        shape_points_i_base_mm=tuple(shape_points),
        delta_i_link_mm=delta_i_link,
        arc_length_mm=link_length_mm,
        chord_length_mm=chord_length,
        shape_quality_flag=quality,
    )
