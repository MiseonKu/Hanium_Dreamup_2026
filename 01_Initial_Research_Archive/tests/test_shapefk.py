from __future__ import annotations

import math

from research_host.math3d import quaternion_from_axis_angle
from research_host.shapefk import reconstruct_link_shape


def test_straight_link_keeps_arc_length_and_endpoint() -> None:
    result = reconstruct_link_shape(
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        100.0,
    )
    assert result.shape_quality_flag == "ok"
    assert abs(result.arc_length_mm - 100.0) < 1e-9
    assert abs(result.p_i_end_base_mm[0] - 100.0) < 1e-9
    assert abs(result.p_i_end_base_mm[1]) < 1e-9
    assert abs(result.chord_length_mm - 100.0) < 1e-9


def test_mid_bend_reduces_chord_length() -> None:
    mid = quaternion_from_axis_angle((0.0, 0.0, 1.0), math.radians(30.0))
    result = reconstruct_link_shape(
        (1.0, 0.0, 0.0, 0.0),
        mid,
        (1.0, 0.0, 0.0, 0.0),
        100.0,
        steps_per_half=24,
    )
    assert result.shape_quality_flag == "ok"
    assert result.chord_length_mm < result.arc_length_mm
    assert result.p_i_end_base_mm[1] > 0.0
    assert len(result.shape_points_i_base_mm) == 49
