from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from imu_vive_learning.cleaning import IMU_PREFIXES
from imu_vive_learning.features import (
    LABEL_COLUMNS,
    add_feature_columns,
    feature_views,
    select_feature_view,
    validate_feature_views,
)
from imu_vive_learning.splitting import (
    SplitContractError,
    build_split_assignments,
    iter_outer_folds,
)


SPLIT_CONFIG = {
    "segment_outer_values": [36, 37, 38, 39, 40],
    "target_group_count": 5,
    "expected_targets_per_group": 10,
    "expected_assignment_rows": 1500,
}


def _feature_frame() -> pd.DataFrame:
    values: dict[str, list[float]] = {
        "encoder_delta_rad_j0": [0.0, 0.1],
        "encoder_delta_rad_j1": [0.0, 0.2],
        "encoder_delta_rad_j2": [0.0, 0.3],
        "planned_p_x_mm": [1.0, 2.0],
        "planned_p_y_mm": [3.0, 4.0],
        "planned_p_z_mm": [5.0, 6.0],
        LABEL_COLUMNS[0]: [10.0, 20.0],
        LABEL_COLUMNS[1]: [30.0, 40.0],
        LABEL_COLUMNS[2]: [50.0, 60.0],
    }
    for prefix_index, prefix in enumerate(IMU_PREFIXES):
        for axis_index, axis in enumerate(("x", "y", "z")):
            values[f"{prefix}_euler_{axis}_rad"] = [
                0.0,
                np.pi / 2 + 0.01 * (prefix_index + axis_index),
            ]
            values[f"{prefix}_accel_{axis}_mps2"] = [1.0, 2.0]
            values[f"{prefix}_mag_{axis}_raw"] = [3.0, 4.0]
        values[f"{prefix}_gyro_rms_rad_s"] = [0.01, 0.02]
    return pd.DataFrame(values)


def _canonical_for_splits() -> pd.DataFrame:
    rows = []
    segments = [36, 37, 38, 39, 40]
    for target_number in range(50):
        rows.append(
            {
                "trial_uid": f"trial_{target_number:03d}",
                "segment_id": segments[target_number // 10],
                "target_id": f"T{target_number:03d}",
                "train_eligible": True,
            }
        )
    rows.append(
        {
            "trial_uid": "excluded",
            "segment_id": 40,
            "target_id": "T999",
            "train_eligible": False,
        }
    )
    return pd.DataFrame(rows)


def test_feature_views_have_frozen_sizes_and_no_primary_label_context() -> None:
    featured = add_feature_columns(_feature_frame())
    views = feature_views()
    assert {name: len(columns) for name, columns in views.items()} == {
        "encoder_only_v1": 3,
        "four_imu_endpoint_v1": 40,
        "fusion_v1": 43,
        "command_context_ablation_v1": 46,
        "mag_ablation_v1": 55,
    }
    for name in ("encoder_only_v1", "four_imu_endpoint_v1", "fusion_v1"):
        assert not any(
            column.startswith(("vive_", "planned_")) for column in views[name]
        )
    validate_feature_views(featured)
    assert select_feature_view(featured, "fusion_v1").shape == (2, 43)


def test_euler_trig_column_names_and_values_match_augmentation_contract() -> None:
    featured = add_feature_columns(_feature_frame())
    source = "imu0_base_euler_x_rad"
    assert f"{source}_sin" in featured
    assert f"{source}_cos" in featured
    np.testing.assert_allclose(featured[f"{source}_sin"], [0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(featured[f"{source}_cos"], [1.0, 0.0], atol=1e-15)


def test_segment_nested_and_target_modulo_splits_are_disjoint_and_deterministic() -> None:
    canonical = _canonical_for_splits()
    config = SPLIT_CONFIG
    first = build_split_assignments(canonical, config)
    second = build_split_assignments(canonical.sample(frac=1.0, random_state=3), config)
    pdt.assert_frame_equal(first, second)

    assert len(first) == 50 * 30
    assert first["split_context"].nunique() == 30
    assert "excluded" not in set(first["trial_uid"])

    outer = first.loc[
        first["split_context"].eq("segment_loso:outer=36:inner=refit")
    ]
    assert outer["role"].value_counts().to_dict() == {"train": 40, "test": 10}
    nested = first.loc[
        first["split_context"].eq("segment_loso:outer=36:inner=37")
    ]
    assert nested["role"].value_counts().to_dict() == {
        "train": 30,
        "test": 10,
        "validation": 10,
    }
    assert set(nested.loc[nested["role"].eq("test"), "segment_id"]) == {36}
    assert set(nested.loc[nested["role"].eq("validation"), "segment_id"]) == {37}

    target_outer = first.loc[
        first["split_context"].eq("target_group:outer=0:inner=refit")
    ]
    assert target_outer["role"].value_counts().to_dict() == {"train": 40, "test": 10}
    assert (
        target_outer.loc[target_outer["role"].eq("test"), "target_id"]
        .str.slice(1)
        .astype(int)
        .mod(5)
        .eq(0)
        .all()
    )


def test_iter_outer_folds_returns_refit_only() -> None:
    assignments = build_split_assignments(
        _canonical_for_splits(),
        SPLIT_CONFIG,
    )
    folds = list(iter_outer_folds(assignments, "segment_loso"))
    assert [fold for fold, _ in folds] == [36, 37, 38, 39, 40]
    assert all(rows["inner_fold"].eq(-1).all() for _, rows in folds)
    assert all("validation" not in set(rows["role"]) for _, rows in folds)


def test_invalid_target_id_fails_closed() -> None:
    canonical = _canonical_for_splits()
    canonical.loc[0, "target_id"] = "bad"
    with pytest.raises(SplitContractError, match="target_id must match"):
        build_split_assignments(
            canonical,
            SPLIT_CONFIG,
        )
