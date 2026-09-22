from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .math3d import vector_add, vector_scale
from .shapefk import LinkShapeResult, reconstruct_link_shape
from .types import Quaternion, Vector3


@dataclass(frozen=True, slots=True)
class ConditionEstimate:
    condition_id: str
    p_hat_base_mm: Vector3
    quality_flag: str
    details: dict[str, float | str]


def no_correction(p_command_base_mm: Vector3) -> ConditionEstimate:
    return ConditionEstimate(
        condition_id="C0_no_correction",
        p_hat_base_mm=p_command_base_mm,
        quality_flag="open_loop",
        details={},
    )


def encoder_only(
    q_encoder_rad: Sequence[float],
    fk_rigid: Callable[[Sequence[float]], Vector3],
) -> ConditionEstimate:
    return ConditionEstimate(
        condition_id="C1_encoder_only",
        p_hat_base_mm=fk_rigid(q_encoder_rad),
        quality_flag="ok",
        details={"source": "encoder_fk"},
    )


def imu_array_only(
    link_attitudes: Sequence[tuple[Quaternion, Quaternion, Quaternion]],
    link_lengths_mm: Sequence[float],
) -> tuple[ConditionEstimate, list[LinkShapeResult]]:
    if len(link_attitudes) != len(link_lengths_mm):
        raise ValueError("link_attitudes and link_lengths_mm must have the same length")

    current_start = (0.0, 0.0, 0.0)
    shapes: list[LinkShapeResult] = []
    quality_flags: list[str] = []
    for attitudes, length_mm in zip(link_attitudes, link_lengths_mm):
        shape = reconstruct_link_shape(
            attitudes[0],
            attitudes[1],
            attitudes[2],
            length_mm,
            p_i_start_base_mm=current_start,
        )
        shapes.append(shape)
        quality_flags.append(shape.shape_quality_flag)
        current_start = shape.p_i_end_base_mm

    non_ok_flags = [flag for flag in quality_flags if flag != "ok"]
    quality = "ok" if not non_ok_flags else "|".join(non_ok_flags)
    return (
        ConditionEstimate(
            condition_id="C2_imu_array_only",
            p_hat_base_mm=current_start,
            quality_flag=quality,
            details={"source": "shape_fk"},
        ),
        shapes,
    )


def encoder_imu_fusion(
    p_hat_encoder_base_mm: Vector3,
    link_shapes: Sequence[LinkShapeResult],
    *,
    fusion_gain: float,
) -> ConditionEstimate:
    gain = max(0.0, min(1.0, float(fusion_gain)))
    accumulated_deflection = (0.0, 0.0, 0.0)
    for shape in link_shapes:
        accumulated_deflection = vector_add(
            accumulated_deflection, shape.delta_i_link_mm
        )

    p_fused = vector_add(
        p_hat_encoder_base_mm, vector_scale(accumulated_deflection, gain)
    )
    return ConditionEstimate(
        condition_id="C6_encoder_imu_fusion",
        p_hat_base_mm=p_fused,
        quality_flag="ok",
        details={
            "fusion_gain": gain,
            "deflection_norm_mm": sum(abs(v) for v in accumulated_deflection),
        },
    )
