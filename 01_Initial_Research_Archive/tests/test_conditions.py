from __future__ import annotations

import research_host.conditions as conditions
from research_host.shapefk import LinkShapeResult


def _shape(flag: str) -> LinkShapeResult:
    return LinkShapeResult(
        p_i_mid_base_mm=(0.0, 0.0, 0.0),
        p_i_end_base_mm=(1.0, 0.0, 0.0),
        shape_points_i_base_mm=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        delta_i_link_mm=(0.0, 0.0, 0.0),
        arc_length_mm=1.0,
        chord_length_mm=1.0,
        shape_quality_flag=flag,
    )


def test_imu_array_quality_flag_omits_ok_noise(monkeypatch) -> None:
    flags = iter(["ok", "arc_length_violation", "ok"])

    def fake_reconstruct_link_shape(*args, **kwargs):
        return _shape(next(flags))

    monkeypatch.setattr(
        conditions,
        "reconstruct_link_shape",
        fake_reconstruct_link_shape,
    )

    estimate, _shapes = conditions.imu_array_only(
        [
            ((1.0, 0.0, 0.0, 0.0),) * 3,
            ((1.0, 0.0, 0.0, 0.0),) * 3,
            ((1.0, 0.0, 0.0, 0.0),) * 3,
        ],
        [1.0, 1.0, 1.0],
    )

    assert estimate.quality_flag == "arc_length_violation"
