from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imu_vive_learning.experiment_augmentation import (
    AugmentationSettings,
    ExperimentAugmentationError,
    build_training_recipe,
    context_seed,
    materialize_inner_synthetic,
)
from imu_vive_learning.experiment_release import (
    PINNED_RELEASE_SHA256,
    ReleaseContractError,
    ReleaseData,
    verify_and_load_release,
)
from imu_vive_learning.experiment_splits import (
    ExperimentSplitError,
    assignment_sha256,
    build_experiment_assignments,
    context_assignments,
)
from imu_vive_learning.features import (
    ACCEL_COLUMNS,
    ENCODER_COLUMNS,
    EULER_SOURCE_COLUMNS,
    GYRO_COLUMNS,
    LABEL_COLUMNS,
    MAG_COLUMNS,
    PLANNED_COLUMNS,
)
from imu_vive_learning.splitting import ASSIGNMENT_COLUMNS


WORKSPACE = Path(__file__).parents[1]
REAL_RELEASE = (
    WORKSPACE.parent
    / "06_simulation_and_data_collect_preparation"
    / "imu_added_experiment"
    / "ml_preparation_v1.zip"
)
CONFIG_HASH = "a" * 64
SOURCE_HASH = "b" * 64


def _tiny_canonical() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 0
    for segment in range(36, 41):
        for target_number in (index % 5, (index + 5) % 10):
            row: dict[str, object] = {
                "trial_uid": f"trial-{index:02d}",
                "segment_id": segment,
                "target_id": f"T{target_number:02d}",
                "target_group": target_number % 5,
                "set_id": 0,
                "repeat_index": 0,
                "train_eligible": True,
                "strict_qc_eligible": True,
                "is_synthetic": False,
            }
            for offset, column in enumerate(PLANNED_COLUMNS):
                row[column] = float(index + offset)
            for offset, column in enumerate(LABEL_COLUMNS):
                row[column] = float(index * 2 + offset)
            for offset, column in enumerate(ENCODER_COLUMNS):
                row[column] = float(index) * 0.01 + offset * 0.001
            for offset, column in enumerate(EULER_SOURCE_COLUMNS):
                value = float(index) * 0.01 + offset * 0.001
                row[column] = value
                row[f"{column}_sin"] = np.sin(value)
                row[f"{column}_cos"] = np.cos(value)
            for columns, scale in (
                (ACCEL_COLUMNS, 0.1),
                (MAG_COLUMNS, 1.0),
                (GYRO_COLUMNS, 0.001),
            ):
                for offset, column in enumerate(columns):
                    row[column] = float(index) * scale + offset * 0.0001
            rows.append(row)
            index += 1
    return pd.DataFrame(rows)


def _tiny_preparation_assignments(canonical: pd.DataFrame) -> pd.DataFrame:
    base = canonical[
        ["trial_uid", "segment_id", "target_id", "target_group"]
    ].copy()
    contexts: list[pd.DataFrame] = []
    segments = base["segment_id"].to_numpy()
    groups = base["target_group"].to_numpy()
    for outer in range(36, 41):
        for inner in (-1, *(value for value in range(36, 41) if value != outer)):
            frame = base.copy()
            frame.insert(0, "scheme", "segment_loso")
            frame.insert(1, "outer_fold", outer)
            frame.insert(2, "inner_fold", inner)
            label = "refit" if inner == -1 else str(inner)
            frame.insert(3, "split_context", f"segment_loso:outer={outer}:inner={label}")
            frame["role"] = np.where(
                segments == outer,
                "test",
                np.where((inner != -1) & (segments == inner), "validation", "train"),
            )
            contexts.append(frame.loc[:, ASSIGNMENT_COLUMNS])
    for outer in range(5):
        frame = base.copy()
        frame.insert(0, "scheme", "target_group")
        frame.insert(1, "outer_fold", outer)
        frame.insert(2, "inner_fold", -1)
        frame.insert(3, "split_context", f"target_group:outer={outer}:inner=refit")
        frame["role"] = np.where(groups == outer, "test", "train")
        contexts.append(frame.loc[:, ASSIGNMENT_COLUMNS])
    return pd.concat(contexts, ignore_index=True)


def _fake_release(canonical: pd.DataFrame, assignments: pd.DataFrame) -> ReleaseData:
    return ReleaseData(
        path=Path("unused.zip"),
        archive_sha256="c" * 64,
        manifest={"config_sha256": "d" * 64},
        preparation_config={
            "contract": {"base_seed": 20260818},
            "source": {"member_sha256": SOURCE_HASH},
            "augmentation": {
                "planned_max_distance_mm": 100.0,
                "joint_max_distance_deg": 20.0,
                "local_neighbor_count": 5,
                "local_lambdas": [0.25, 0.5, 0.75],
                "cmixup_alpha": 2.0,
            },
        },
        canonical=canonical,
        assignments=assignments,
        feature_sets={},
        _member_hashes={},
        _outer_members={},
    )


def test_target_nested_assignments_are_complete_disjoint_and_stable() -> None:
    canonical = _tiny_canonical()
    parent = _tiny_preparation_assignments(canonical)
    assignments = build_experiment_assignments(
        canonical,
        parent,
        expected_rows=500,
        expected_trial_count=10,
    )
    assert len(assignments) == 500
    assert assignments["split_context"].nunique() == 50
    context = context_assignments(
        assignments,
        scheme="target_group",
        outer_fold=0,
        inner_fold=1,
    )
    assert set(context.loc[context["role"].eq("test"), "target_group"]) == {0}
    assert set(context.loc[context["role"].eq("validation"), "target_group"]) == {1}
    assert set(context.loc[context["role"].eq("train"), "target_group"]) == {2, 3, 4}
    shuffled = assignments.sample(frac=1.0, random_state=42)
    assert assignment_sha256(assignments) == assignment_sha256(shuffled)


def test_target_nested_assignment_rejects_parent_metadata_drift() -> None:
    canonical = _tiny_canonical()
    parent = _tiny_preparation_assignments(canonical)
    parent.loc[0, "target_id"] = "T99"
    with pytest.raises(ExperimentSplitError, match="target_id metadata drifted"):
        build_experiment_assignments(
            canonical,
            parent,
            expected_rows=500,
            expected_trial_count=10,
        )


def test_inner_augmentation_parents_and_sigma_are_train_context_only() -> None:
    canonical = _tiny_canonical()
    parent = _tiny_preparation_assignments(canonical)
    assignments = build_experiment_assignments(
        canonical,
        parent,
        expected_rows=500,
        expected_trial_count=10,
    )
    context = context_assignments(
        assignments,
        scheme="segment_loso",
        outer_fold=36,
        inner_fold=37,
    )
    train_ids = set(context.loc[context["role"].eq("train"), "trial_uid"])
    held_ids = set(context.loc[context["role"].ne("train"), "trial_uid"])
    train = canonical.loc[canonical["trial_uid"].isin(train_ids)].copy()
    generated = materialize_inner_synthetic(
        train,
        scheme="segment_loso",
        outer_fold=36,
        inner_fold=37,
        method="cmixup_offline_v1",
        source_hash=SOURCE_HASH,
        experiment_config_sha256=CONFIG_HASH,
        settings=AugmentationSettings(),
    )
    parents = set(generated["parent_a_trial_uid"]) | set(generated["parent_b_trial_uid"])
    assert len(generated) == len(train)
    assert parents <= train_ids
    assert parents.isdisjoint(held_ids)
    assert generated["cmixup_sigma_mm"].nunique() == 1
    assert generated["cmixup_sigma_mm"].iloc[0] > 0
    assert np.allclose(generated["sample_weight"], 0.5)


def test_replay_is_exact_duplicate_with_equal_objective_mass() -> None:
    canonical = _tiny_canonical()
    parent = _tiny_preparation_assignments(canonical)
    assignments = build_experiment_assignments(
        canonical,
        parent,
        expected_rows=500,
        expected_trial_count=10,
    )
    release = _fake_release(canonical, parent)
    recipe = build_training_recipe(
        release,
        assignments,
        scheme="segment_loso",
        outer_fold=36,
        inner_fold=37,
        recipe="replay",
        experiment_config_sha256=CONFIG_HASH,
    )
    ordered_real = recipe.real.sort_values("trial_uid")
    ordered_replay = recipe.additional.sort_values("trial_uid")
    feature_and_label = list(ENCODER_COLUMNS + LABEL_COLUMNS)
    np.testing.assert_allclose(
        ordered_real[feature_and_label].to_numpy(),
        ordered_replay[feature_and_label].to_numpy(),
    )
    assert len(recipe.real) == len(recipe.additional)
    assert recipe.combined["sample_weight"].sum() == len(recipe.real)


def test_context_seed_is_stable_and_context_specific() -> None:
    first = context_seed(
        scheme="segment_loso",
        outer_fold=36,
        inner_fold=37,
        recipe="local_interp_v2",
    )
    assert first == context_seed(
        scheme="segment_loso",
        outer_fold=36,
        inner_fold=37,
        recipe="local_interp_v2",
    )
    assert first != context_seed(
        scheme="segment_loso",
        outer_fold=36,
        inner_fold=38,
        recipe="local_interp_v2",
    )


def test_materialized_outer_member_is_rejected_for_inner_context() -> None:
    canonical = _tiny_canonical()
    parent = _tiny_preparation_assignments(canonical)
    assignments = build_experiment_assignments(
        canonical,
        parent,
        expected_rows=500,
        expected_trial_count=10,
    )
    release = _fake_release(canonical, parent)
    with pytest.raises(ExperimentAugmentationError, match="no materialized outer synthetic"):
        # The inner path regenerates, so force the outer path with a fake release
        # context whose member index is deliberately absent.
        build_training_recipe(
            release,
            assignments,
            scheme="segment_loso",
            outer_fold=36,
            inner_fold=-1,
            recipe="local_interp_v2",
            experiment_config_sha256=CONFIG_HASH,
        )


def test_release_reader_rejects_lfs_pointer_before_zip_parse(tmp_path: Path) -> None:
    pointer = tmp_path / "release.zip"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{PINNED_RELEASE_SHA256}\nsize 104446557\n",
        encoding="ascii",
    )
    with pytest.raises(ReleaseContractError, match="Git LFS pointer"):
        verify_and_load_release(pointer)


@pytest.mark.skipif(not REAL_RELEASE.is_file(), reason="tracked LFS release is unavailable")
def test_real_release_and_one_outer_fold_smoke() -> None:
    release = verify_and_load_release(REAL_RELEASE)
    assert (len(release.canonical), len(release.eligible_real), len(release.assignments)) == (
        7707,
        7702,
        231060,
    )
    assignments = build_experiment_assignments(release.canonical, release.assignments)
    assert len(assignments) == 385100
    synthetic = release.load_outer_synthetic(
        "segment_loso", 36, "cmixup_offline_v1"
    )
    train_ids = set(
        context_assignments(
            assignments,
            scheme="segment_loso",
            outer_fold=36,
            inner_fold=-1,
        )
        .loc[lambda frame: frame["role"].eq("train"), "trial_uid"]
        .astype(str)
    )
    parents = set(synthetic["parent_a_trial_uid"].astype(str)) | set(
        synthetic["parent_b_trial_uid"].astype(str)
    )
    assert len(synthetic) == len(train_ids) == 6030
    assert parents <= train_ids
