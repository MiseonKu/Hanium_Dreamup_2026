from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from imu_vive_learning.augmentation import (
    AugmentationError,
    circular_interpolate_rad,
    cmixup_offline_v1,
    local_interp_v2,
)


PLANNED = ("planned_x_mm", "planned_y_mm", "planned_z_mm")
LABELS = ("vive_pos_raw_x_mm", "vive_pos_raw_y_mm", "vive_pos_raw_z_mm")
JOINTS = ("encoder_delta_rad_j0", "encoder_delta_rad_j1", "encoder_delta_rad_j2")
LINEAR = (*JOINTS, "accel_x", "gyro_rms")
EULER = ("euler_x_rad",)
TRIG = {"euler_x_rad": ("euler_x_sin", "euler_x_cos")}
CONTEXT = {
    "split_scheme": "segment_loso",
    "fold": 36,
    "base_seed": 20260818,
    "source_hash": "source-sha256",
    "config_hash": "config-sha256",
}


def _row(
    uid: str,
    *,
    planned_x: float,
    label_x: float,
    joint_deg: float,
    euler_deg: float,
    segment: int = 36,
    set_id: str = "set-a",
    repeat: int = 0,
) -> dict[str, object]:
    joint = math.radians(joint_deg)
    angle = math.radians(euler_deg)
    return {
        "trial_uid": uid,
        "is_synthetic": False,
        "segment_id": segment,
        "set_id": set_id,
        "repeat_index": repeat,
        "planned_x_mm": planned_x,
        "planned_y_mm": 0.0,
        "planned_z_mm": 0.0,
        "vive_pos_raw_x_mm": label_x,
        "vive_pos_raw_y_mm": label_x / 2.0,
        "vive_pos_raw_z_mm": -label_x,
        "encoder_delta_rad_j0": joint,
        "encoder_delta_rad_j1": joint / 2.0,
        "encoder_delta_rad_j2": -joint / 3.0,
        "accel_x": planned_x / 10.0,
        "gyro_rms": 0.01 + planned_x / 100_000.0,
        "euler_x_rad": angle,
        "euler_x_sin": math.sin(angle),
        "euler_x_cos": math.cos(angle),
        # These real-only fields must never be copied into a synthetic row.
        "target_id": f"target-{uid}",
        "t_start_ns": 1_000_000 + int(planned_x),
        "quality_warning": uid.endswith("b"),
    }


def _local_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("a", planned_x=0.0, label_x=10.0, joint_deg=0.0, euler_deg=179.0),
            _row("b", planned_x=10.0, label_x=12.0, joint_deg=4.0, euler_deg=-179.0),
            _row("c", planned_x=50.0, label_x=20.0, joint_deg=8.0, euler_deg=30.0),
            _row(
                "d",
                planned_x=5.0,
                label_x=50.0,
                joint_deg=2.0,
                euler_deg=45.0,
                segment=37,
            ),
            _row(
                "e",
                planned_x=15.0,
                label_x=52.0,
                joint_deg=5.0,
                euler_deg=55.0,
                segment=37,
            ),
        ]
    )


def _cmixup_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("a", planned_x=0.0, label_x=10.0, joint_deg=0.0, euler_deg=179.0),
            _row("b", planned_x=10.0, label_x=11.0, joint_deg=5.0, euler_deg=-179.0),
            _row("c", planned_x=300.0, label_x=100.0, joint_deg=50.0, euler_deg=20.0),
            _row("d", planned_x=310.0, label_x=101.0, joint_deg=55.0, euler_deg=30.0),
        ]
    )


def _local(frame: pd.DataFrame, **overrides: object) -> pd.DataFrame:
    kwargs: dict[str, object] = {
        "planned_columns": PLANNED,
        "label_columns": LABELS,
        "linear_columns": LINEAR,
        "euler_columns": EULER,
        "euler_trig_columns": TRIG,
        **CONTEXT,
    }
    kwargs.update(overrides)
    return local_interp_v2(frame, **kwargs)


def _cmixup(frame: pd.DataFrame, **overrides: object) -> pd.DataFrame:
    kwargs: dict[str, object] = {
        "planned_columns": PLANNED,
        "label_columns": LABELS,
        "joint_rad_columns": JOINTS,
        "linear_columns": LINEAR,
        "euler_columns": EULER,
        "euler_trig_columns": TRIG,
        **CONTEXT,
    }
    kwargs.update(overrides)
    return cmixup_offline_v1(frame, **kwargs)


def test_circular_interpolation_crosses_wrap_boundary() -> None:
    angle = circular_interpolate_rad(math.radians(179.0), math.radians(-179.0), 0.5)
    assert abs(abs(math.degrees(angle)) - 180.0) < 1e-10
    assert abs(math.degrees(angle)) > 179.9


def test_local_is_one_to_one_group_constrained_and_has_complete_provenance() -> None:
    real = _local_frame()
    synthetic = _local(real, lambdas=(0.5,))

    assert len(synthetic) == len(real)
    assert synthetic["synthetic_id"].is_unique
    assert synthetic["is_synthetic"].eq(True).all()
    assert synthetic["augmentation_method"].eq("local_interp_v2").all()
    assert synthetic["split_scheme"].eq("segment_loso").all()
    assert synthetic["outer_fold"].eq(36).all()
    assert synthetic["inner_fold"].eq(-1).all()
    assert synthetic["split_context"].eq(
        "segment_loso:outer=36:inner=refit"
    ).all()
    assert synthetic["source_hash"].eq("source-sha256").all()
    assert synthetic["config_hash"].eq("config-sha256").all()
    assert synthetic["sample_weight"].sum() == pytest.approx(float(len(real)))

    by_id = real.set_index("trial_uid")
    for row in synthetic.itertuples(index=False):
        parent_a = by_id.loc[row.parent_a_trial_uid]
        parent_b = by_id.loc[row.parent_b_trial_uid]
        assert row.anchor_trial_uid == row.parent_a_trial_uid
        assert row.parent_a_trial_uid != row.parent_b_trial_uid
        assert parent_a["segment_id"] == parent_b["segment_id"]
        assert parent_a["set_id"] == parent_b["set_id"]
        assert parent_a["repeat_index"] == parent_b["repeat_index"]
        assert row.planned_parent_distance_mm <= 100.0
        expected = row.mix_lambda * parent_a["accel_x"] + (1 - row.mix_lambda) * parent_b[
            "accel_x"
        ]
        assert row.accel_x == pytest.approx(expected)
        assert row.euler_x_sin == pytest.approx(math.sin(row.euler_x_rad))
        assert row.euler_x_cos == pytest.approx(math.cos(row.euler_x_rad))

    assert "target_id" not in synthetic
    assert "t_start_ns" not in synthetic
    assert "quality_warning" not in synthetic


def test_local_uses_shortest_arc_before_recomputing_sin_cos() -> None:
    real = _local_frame().iloc[:2].copy()
    synthetic = _local(real, lambdas=(0.5,))

    assert np.allclose(np.abs(np.rad2deg(synthetic["euler_x_rad"])), 180.0)
    assert np.allclose(synthetic["euler_x_sin"], 0.0, atol=1e-12)
    assert np.allclose(synthetic["euler_x_cos"], -1.0, atol=1e-12)


def test_local_is_deterministic_and_row_order_invariant() -> None:
    real = _local_frame()
    first = _local(real)
    second = _local(real.sample(frac=1.0, random_state=91).reset_index(drop=True))

    pdt.assert_frame_equal(first, second, check_exact=True)


def test_inner_context_has_distinct_seed_and_explicit_provenance() -> None:
    real = _local_frame()
    outer = _local(real)
    inner = _local(real, inner_fold=37)

    assert inner["inner_fold"].eq(37).all()
    assert inner["split_context"].eq("segment_loso:outer=36:inner=37").all()
    assert not inner["derived_seed"].equals(outer["derived_seed"])
    assert set(inner["synthetic_id"]).isdisjoint(set(outer["synthetic_id"]))


def test_cmixup_is_one_to_one_train_only_and_honours_parent_constraints() -> None:
    train = _cmixup_frame()
    held_out_ids = {"held-out-1", "held-out-2"}
    synthetic = _cmixup(train)

    assert len(synthetic) == len(train)
    assert set(synthetic["parent_a_trial_uid"]) == set(train["trial_uid"])
    assert set(synthetic["parent_b_trial_uid"]) <= set(train["trial_uid"])
    assert not (set(synthetic["parent_a_trial_uid"]) | set(synthetic["parent_b_trial_uid"])) & held_out_ids
    assert synthetic["planned_parent_distance_mm"].le(100.0 + 1e-12).all()
    assert synthetic["joint_parent_distance_deg"].le(20.0 + 1e-12).all()
    assert synthetic["mix_lambda"].between(0.0, 1.0, inclusive="neither").all()
    assert synthetic["cmixup_sigma_mm"].nunique() == 1
    # Nearest raw-Vive label distance is identical for all four anchors.
    expected_label_distance = math.sqrt(1.0**2 + 0.5**2 + 1.0**2)
    assert synthetic.attrs["cmixup_sigma_mm"] == pytest.approx(expected_label_distance)
    assert synthetic["cmixup_sigma_mm"].iloc[0] == pytest.approx(expected_label_distance)


def test_cmixup_is_deterministic_and_row_order_invariant() -> None:
    train = _cmixup_frame()
    first = _cmixup(train)
    second = _cmixup(train.sample(frac=1.0, random_state=17).reset_index(drop=True))

    pdt.assert_frame_equal(first, second, check_exact=True)
    assert first.attrs == second.attrs


@pytest.mark.parametrize("method", ["local", "cmixup"])
def test_no_candidate_fails_without_relaxing_constraints(method: str) -> None:
    isolated = pd.DataFrame(
        [_row("only", planned_x=0.0, label_x=0.0, joint_deg=0.0, euler_deg=0.0)]
    )

    with pytest.raises(AugmentationError, match="constraints were not relaxed"):
        _local(isolated) if method == "local" else _cmixup(isolated)


@pytest.mark.parametrize("method", ["local", "cmixup"])
def test_rejects_synthetic_rows_in_training_input(method: str) -> None:
    frame = _local_frame() if method == "local" else _cmixup_frame()
    frame.loc[0, "is_synthetic"] = True

    with pytest.raises(AugmentationError, match="real rows only"):
        _local(frame) if method == "local" else _cmixup(frame)


def test_cmixup_does_not_relax_joint_limit_to_reach_nearby_planned_parent() -> None:
    frame = pd.DataFrame(
        [
            _row("a", planned_x=0.0, label_x=0.0, joint_deg=0.0, euler_deg=0.0),
            _row("b", planned_x=1.0, label_x=1.0, joint_deg=21.0, euler_deg=1.0),
        ]
    )

    with pytest.raises(AugmentationError, match="max joint delta <= 20 deg"):
        _cmixup(frame)


def test_local_interp_refills_missing_anchor_without_relaxing_constraints() -> None:
    frame = pd.DataFrame(
        [
            _row("a", planned_x=0.0, label_x=0.0, joint_deg=0.0, euler_deg=0.0),
            _row("b", planned_x=10.0, label_x=1.0, joint_deg=1.0, euler_deg=1.0),
            _row(
                "isolated",
                planned_x=500.0,
                label_x=50.0,
                joint_deg=10.0,
                euler_deg=10.0,
            ),
        ]
    )

    result = _local(frame, planned_max_distance_mm=100.0)

    assert len(result) == len(frame)
    assert result["replacement_for_missing_anchor"].sum() == 1
    assert "isolated" not in set(result["anchor_trial_uid"])
    assert result["planned_parent_distance_mm"].le(100.0).all()
    replacement = result.loc[result["replacement_for_missing_anchor"]].iloc[0]
    assert replacement["anchor_replica_index"] > 0
