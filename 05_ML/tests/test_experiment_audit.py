from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from imu_vive_learning.experiment_config import load_experiment_config
from imu_vive_learning.experiment_augmentation import BASE_SEED, context_seed
from imu_vive_learning.experiment_io import (
    atomic_write_parquet,
    prepare_experiment_directory,
    sha256_file,
    stable_hash,
)
from imu_vive_learning.experiment_matrix import ExperimentCell
from imu_vive_learning.experiment_models import configuration_id, model_parameter_grid
from imu_vive_learning.experiment_runner import (
    AUGMENTATION_ROW_PROVENANCE_COLUMNS,
    ExperimentRunError,
    PREDICTION_COLUMNS,
    TRUE_COLUMNS,
    _aggregate_inner_selection,
    _cache_paths,
    _eligible_real_audit_table,
    _fold_and_aggregate_metrics,
    _git_revision,
    _public_augmentation_row_provenance,
)
from imu_vive_learning.experiment_validation import (
    EXPECTED_ENVIRONMENT_VERSIONS,
    ExperimentValidationError,
    _regenerate_seeded_augmentation_provenance,
    _recompute_cmixup_sigma,
    _validate_environment,
    _validate_final_synthetic,
    _validate_preprocessor_state,
    _validate_seeded_augmentation_provenance,
)
from imu_vive_learning.experiment_preprocessing import TrainOnlyPreprocessor
from imu_vive_learning.features import (
    ENCODER_COLUMNS,
    FUSION_COLUMNS,
    LABEL_COLUMNS,
    PLANNED_COLUMNS,
)


WORKSPACE = Path(__file__).parents[1]
CONFIG_PATH = WORKSPACE / "configs" / "endpoint_regression_v1.yaml"


def _eligible_fixture() -> pd.DataFrame:
    rows = []
    for index, uid in enumerate(("trial-a", "trial-b")):
        row: dict[str, object] = {
            "trial_uid": uid,
            "segment_id": 36,
            "target_id": f"T{index + 1:02d}",
            "set_id": 1.0,
            "repeat_index": 0,
            "strict_qc_eligible": True,
        }
        for offset, column in enumerate(PLANNED_COLUMNS):
            row[column] = float(index * 10 + offset)
        for offset, column in enumerate(ENCODER_COLUMNS):
            row[column] = float(index * 0.1 + offset * 0.01)
        for offset, column in enumerate(LABEL_COLUMNS):
            row[column] = float(index * 2 + offset)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def test_public_provenance_exposes_audited_joint_distance(tmp_path: Path) -> None:
    eligible = _eligible_fixture()
    audit = _eligible_real_audit_table(eligible)
    cache_root = tmp_path / "cache"
    path, _ = _cache_paths(
        cache_root,
        scheme="segment_loso",
        outer_fold=36,
        inner_fold=37,
        recipe="local_interp_v2",
    )
    records = []
    for anchor, partner in (("trial-a", "trial-b"), ("trial-b", "trial-a")):
        row = {column: None for column in AUGMENTATION_ROW_PROVENANCE_COLUMNS}
        row.update(
            {
                "synthetic_id": f"syn-{anchor}",
                "is_synthetic": True,
                "augmentation_method": "local_interp_v2",
                "split_scheme": "segment_loso",
                "outer_fold": 36,
                "inner_fold": 37,
                "split_context": "segment_loso:outer=36:inner=37",
                "anchor_trial_uid": anchor,
                "anchor_replica_index": 0,
                "replacement_for_missing_anchor": False,
                "parent_a_trial_uid": anchor,
                "parent_b_trial_uid": partner,
                "mix_lambda": 0.5,
                "planned_parent_distance_mm": np.sqrt(300.0),
                "label_parent_distance_mm": np.sqrt(12.0),
                "joint_parent_distance_deg": np.nan,
                "cmixup_sigma_mm": np.nan,
                "derived_seed": 123,
                "source_hash": "a" * 64,
                "config_hash": "b" * 64,
                "experiment_config_sha256": "c" * 64,
                "sample_weight": 0.5,
            }
        )
        records.append(row)
    atomic_write_parquet(path, pd.DataFrame.from_records(records))
    metadata = {
        "scheme": "segment_loso",
        "outer_fold": 36,
        "inner_fold": 37,
        "recipe": "local_interp_v2",
        "additional_rows": 2,
        "task_hash": "d" * 64,
        "parquet_sha256": sha256_file(path),
        "real_train_uid_sha256": "e" * 64,
    }
    public = _public_augmentation_row_provenance(
        cache_root, [metadata], audit, jobs=1
    )
    expected_joint = np.rad2deg(0.1)
    assert public["generator_joint_parent_distance_deg"].isna().all()
    assert public["joint_parent_distance_deg"].to_numpy() == pytest.approx(
        [expected_joint, expected_joint]
    )
    assert not public["joint_constraint_applied"].any()
    assert public["context_task_hash"].eq("d" * 64).all()


def test_cmixup_sigma_skips_isolated_anchors_like_generator() -> None:
    frame = pd.DataFrame(
        {
            PLANNED_COLUMNS[0]: [0.0, 10.0, 1000.0],
            PLANNED_COLUMNS[1]: [0.0, 0.0, 1000.0],
            PLANNED_COLUMNS[2]: [0.0, 0.0, 1000.0],
            ENCODER_COLUMNS[0]: [0.0, 0.0, 0.0],
            ENCODER_COLUMNS[1]: [0.0, 0.0, 0.0],
            ENCODER_COLUMNS[2]: [0.0, 0.0, 0.0],
            LABEL_COLUMNS[0]: [0.0, 3.0, 99.0],
            LABEL_COLUMNS[1]: [0.0, 4.0, 99.0],
            LABEL_COLUMNS[2]: [0.0, 0.0, 99.0],
        }
    )
    assert _recompute_cmixup_sigma(frame) == pytest.approx(5.0)


def _augmentation_replay_fixture(positions: tuple[float, ...]) -> pd.DataFrame:
    records = []
    for index, position in enumerate(positions):
        row: dict[str, object] = {
            "trial_uid": f"replay-{index}",
            "segment_id": 36,
            "target_id": f"T{index:02d}",
            "set_id": 1.0,
            "repeat_index": 0,
            "strict_qc_eligible": True,
        }
        for axis, column in enumerate(PLANNED_COLUMNS):
            row[column] = position if axis == 0 else 0.0
        for axis, column in enumerate(ENCODER_COLUMNS):
            row[column] = 0.01 * index + 0.001 * axis
        for axis, column in enumerate(LABEL_COLUMNS):
            row[column] = 0.2 * position + float(axis)
        records.append(row)
    return pd.DataFrame.from_records(records)


@pytest.mark.parametrize("recipe", ["local_interp_v2", "cmixup_offline_v1"])
def test_seeded_augmentation_replay_rejects_seed_and_replacement_tampering(
    recipe: str,
) -> None:
    train = _augmentation_replay_fixture((0.0, 10.0, 20.0, 500.0))
    seed = context_seed(
        scheme="segment_loso",
        outer_fold=36,
        inner_fold=37,
        recipe=recipe,
        base_seed=BASE_SEED,
    )
    expected = _regenerate_seeded_augmentation_provenance(
        train,
        scheme="segment_loso",
        outer_fold=36,
        inner_fold=37,
        recipe=recipe,
        augmentation_seed=seed,
        source_hash="a" * 64,
        generator_config_hash="b" * 64,
    )
    _validate_seeded_augmentation_provenance(
        expected,
        train,
        scheme="segment_loso",
        outer_fold=36,
        inner_fold=37,
        recipe=recipe,
        augmentation_seed=seed,
        source_hash="a" * 64,
        generator_config_hash="b" * 64,
        generator_joint_column="generator_joint_parent_distance_deg",
    )

    def assert_rejected(tampered: pd.DataFrame) -> None:
        with pytest.raises(ExperimentValidationError, match="seeded augmentation provenance"):
            _validate_seeded_augmentation_provenance(
                tampered,
                train,
                scheme="segment_loso",
                outer_fold=36,
                inner_fold=37,
                recipe=recipe,
                augmentation_seed=seed,
                source_hash="a" * 64,
                generator_config_hash="b" * 64,
                generator_joint_column="generator_joint_parent_distance_deg",
            )

    tampered_seed = expected.copy()
    tampered_seed.loc[0, "derived_seed"] = int(tampered_seed.loc[0, "derived_seed"]) + 1
    assert_rejected(tampered_seed)

    replacement_index = expected.index[expected["replacement_for_missing_anchor"]][0]
    tampered_replacement = expected.copy()
    tampered_replacement.loc[replacement_index, "replacement_for_missing_anchor"] = False
    assert_rejected(tampered_replacement)

    tampered_replica = expected.copy()
    tampered_replica.loc[replacement_index, "anchor_replica_index"] = (
        int(tampered_replica.loc[replacement_index, "anchor_replica_index"]) + 1
    )
    assert_rejected(tampered_replica)

    tampered_lambda = expected.copy()
    original_lambda = float(tampered_lambda.loc[0, "mix_lambda"])
    tampered_lambda.loc[0, "mix_lambda"] = (
        next(value for value in (0.25, 0.5, 0.75) if value != original_lambda)
        if recipe == "local_interp_v2"
        else original_lambda / 2.0
    )
    assert_rejected(tampered_lambda)


def test_final_local_replay_rejects_partner_outside_five_nearest() -> None:
    cohort = _augmentation_replay_fixture((0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0))
    config_hash = "b" * 64
    source_hash = "a" * 64
    seed = context_seed(
        scheme="final_full",
        outer_fold=0,
        inner_fold=0,
        recipe="local_interp_v2",
        base_seed=BASE_SEED,
    )
    normalized = _regenerate_seeded_augmentation_provenance(
        cohort,
        scheme="final_full",
        outer_fold=0,
        inner_fold=-1,
        recipe="local_interp_v2",
        augmentation_seed=seed,
        source_hash=source_hash,
        generator_config_hash=config_hash,
    )
    frame = normalized.rename(
        columns={"generator_joint_parent_distance_deg": "joint_parent_distance_deg"}
    )
    frame["experiment_config_sha256"] = config_hash
    _validate_final_synthetic(
        frame,
        cohort,
        recipe="local_interp_v2",
        config_hash=config_hash,
        source_hash=source_hash,
        scheme="final_full",
        outer_fold=0,
        inner_fold=-1,
        seed_inner_fold=0,
    )

    tampered = frame.copy()
    row_index = tampered.index[0]
    anchor_id = str(tampered.loc[row_index, "parent_a_trial_uid"])
    index = cohort.set_index("trial_uid")
    anchor = index.loc[anchor_id]
    candidates = index.drop(index=anchor_id).copy()
    candidates["distance"] = np.linalg.norm(
        candidates.loc[:, list(PLANNED_COLUMNS)].to_numpy(float)
        - anchor.loc[list(PLANNED_COLUMNS)].to_numpy(float),
        axis=1,
    )
    outside_top_five = str(
        candidates.sort_values(["distance"], kind="mergesort").index[-1]
    )
    partner = index.loc[outside_top_five]
    tampered.loc[row_index, "parent_b_trial_uid"] = outside_top_five
    tampered.loc[row_index, "planned_parent_distance_mm"] = float(
        np.linalg.norm(
            anchor.loc[list(PLANNED_COLUMNS)].to_numpy(float)
            - partner.loc[list(PLANNED_COLUMNS)].to_numpy(float)
        )
    )
    tampered.loc[row_index, "label_parent_distance_mm"] = float(
        np.linalg.norm(
            anchor.loc[list(LABEL_COLUMNS)].to_numpy(float)
            - partner.loc[list(LABEL_COLUMNS)].to_numpy(float)
        )
    )
    with pytest.raises(ExperimentValidationError, match="seeded augmentation provenance"):
        _validate_final_synthetic(
            tampered,
            cohort,
            recipe="local_interp_v2",
            config_hash=config_hash,
            source_hash=source_hash,
            scheme="final_full",
            outer_fold=0,
            inner_fold=-1,
            seed_inner_fold=0,
        )


def test_fusion_audit_recomputes_outer_and_final_scaler_state() -> None:
    records = []
    for index in range(8):
        row = _eligible_fixture().iloc[index % 2].to_dict()
        row["trial_uid"] = f"fusion-{index}"
        row["target_id"] = f"T{index + 1:02d}"
        row["strict_qc_eligible"] = index < 6
        for feature_index, column in enumerate(FUSION_COLUMNS):
            row[column] = float(index * 0.25 + feature_index * 0.01)
        for axis, column in enumerate(LABEL_COLUMNS):
            row[column] = float(index * 2.0 + axis)
        records.append(row)
    eligible = pd.DataFrame.from_records(records)
    audit = _eligible_real_audit_table(
        eligible, feature_columns=FUSION_COLUMNS
    )
    assert set(FUSION_COLUMNS) <= set(audit.columns)
    X = audit.loc[:, list(FUSION_COLUMNS)].to_numpy(float)
    y = audit.loc[:, list(LABEL_COLUMNS)].to_numpy(float)
    uids = audit["trial_uid"].astype(str).tolist()
    outer = TrainOnlyPreprocessor(
        feature_names=FUSION_COLUMNS,
        scale_features=True,
        split_context="segment_loso:outer=36:inner=refit",
    ).fit_real_train(
        X,
        y,
        trial_uids=uids,
        forbidden_uids=("held-out",),
    )
    _validate_preprocessor_state(
        outer.metadata(),
        audit,
        feature_columns=list(FUSION_COLUMNS),
        scale_features=True,
        expected_uid_hash=hashlib.sha256(
            "\n".join(sorted(uids)).encode("utf-8")
        ).hexdigest(),
        split_context="segment_loso:outer=36:inner=refit",
        forbidden_uid_count=1,
    )
    tampered = outer.metadata()
    tampered["feature_mean"][0] += 1.0
    with pytest.raises(ExperimentValidationError, match="statistics"):
        _validate_preprocessor_state(
            tampered,
            audit,
            feature_columns=list(FUSION_COLUMNS),
            scale_features=True,
            expected_uid_hash=outer.metadata()["leakage_audit"][
                "fit_trial_uid_sha256"
            ],
            split_context="segment_loso:outer=36:inner=refit",
            forbidden_uid_count=1,
        )

    strict = audit.loc[audit["strict_qc_eligible"].eq(True)].reset_index(drop=True)
    strict_uids = strict["trial_uid"].astype(str).tolist()
    final_strict = TrainOnlyPreprocessor(
        feature_names=FUSION_COLUMNS,
        scale_features=True,
        split_context="final_strict_qc:resubstitution_only",
    ).fit_real_train(
        strict.loc[:, list(FUSION_COLUMNS)].to_numpy(float),
        strict.loc[:, list(LABEL_COLUMNS)].to_numpy(float),
        trial_uids=strict_uids,
    )
    _validate_preprocessor_state(
        final_strict.metadata(),
        strict,
        feature_columns=list(FUSION_COLUMNS),
        scale_features=True,
        expected_uid_hash=final_strict.metadata()["leakage_audit"][
            "fit_trial_uid_sha256"
        ],
        split_context="final_strict_qc:resubstitution_only",
        forbidden_uid_count=0,
    )


def _selection_trace(
    cells: list[ExperimentCell], *, real_score: float, augmented_score: float
) -> pd.DataFrame:
    rows = []
    for cell in cells:
        base = real_score if cell.recipe == "real_only" else augmented_score
        for parameter_index, params in enumerate(model_parameter_grid(cell.model)):
            params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
            for context in range(20):
                rows.append(
                    {
                        "scheme": "segment_loso",
                        "outer_fold": context // 4,
                        "inner_fold": context % 4,
                        "prediction_kind": "ensemble",
                        "cell_id": cell.cell_id,
                        "config_id": configuration_id(cell.model, params),
                        "params_json": params_json,
                        "rmse_3d_mm": base + 10.0 * parameter_index,
                        "p95_3d_error_mm": base + 1.0,
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_final_selection_keeps_real_only_in_pool_when_augmentation_passes() -> None:
    config = load_experiment_config(CONFIG_PATH)
    cells = [
        ExperimentCell("ridge", "encoder", "real_only", ("sensor",), False, True),
        ExperimentCell(
            "ridge", "fusion", "local_interp_v2", ("augmentation",), False, True
        ),
    ]
    gates = pd.DataFrame(
        {"candidate_cell_id": [cells[1].cell_id], "passed": [True]}
    )
    _, selected = _aggregate_inner_selection(
        _selection_trace(cells, real_score=1.0, augmented_score=2.0),
        cells,
        gates,
        config,
    )
    assert selected["cell_id"] == cells[0].cell_id
    assert selected["final_pool_candidate_count"] == 2
    assert not selected["augmentation_selected"]

    _, augmented = _aggregate_inner_selection(
        _selection_trace(cells, real_score=2.0, augmented_score=1.0),
        cells,
        gates,
        config,
    )
    assert augmented["cell_id"] == cells[1].cell_id
    assert augmented["augmentation_selected"]


def test_fold_metrics_materialize_segment_target_details() -> None:
    cell = ExperimentCell(
        "ridge", "encoder", "real_only", ("sensor",), False, True
    )
    rows = []
    for outer in range(5):
        for target_index in range(2):
            truth = np.array([outer, target_index, outer + target_index], dtype=float)
            prediction = truth + np.array([1.0, 0.0, 0.0])
            rows.append(
                {
                    "scheme": "segment_loso",
                    "outer_fold": outer,
                    "cell_id": cell.cell_id,
                    "model": cell.model,
                    "feature_view": cell.feature_view,
                    "recipe": cell.recipe,
                    "config_id": "fixture",
                    "params_json": "{}",
                    "trial_uid": f"{outer}-{target_index}",
                    "segment_id": outer,
                    "target_id": f"T{target_index:02d}",
                    "strict_qc_eligible": True,
                    **dict(zip(TRUE_COLUMNS, truth, strict=True)),
                    **dict(zip(PREDICTION_COLUMNS, prediction, strict=True)),
                }
            )
    ensemble = pd.DataFrame.from_records(rows)
    seeds = ensemble.copy()
    seeds["seed_index"] = 0
    folds, aggregate, strict, details = _fold_and_aggregate_metrics(
        ensemble, seeds, [cell]
    )
    assert len(folds) == 5
    assert len(aggregate) == 1
    assert len(strict) == 1
    assert len(details) == 10
    assert not details.duplicated(
        ["scheme", "outer_fold", "cell_id", "segment_id", "target_id"]
    ).any()
    assert details["rmse_3d_mm"].to_numpy() == pytest.approx(np.ones(10))


def test_environment_is_frozen_and_resume_identity_includes_it(tmp_path: Path) -> None:
    environment = {
        "python": "3.12.13",
        "python_implementation": "CPython",
        "platform": "fixture",
        "machine": "AMD64",
        "byteorder": "little",
        **EXPECTED_ENVIRONMENT_VERSIONS,
    }
    manifest = {
        "runtime": {
            "environment": environment,
            "environment_sha256": stable_hash(environment),
            "context_jobs": 4,
            "estimator_threads": 1,
        }
    }
    _validate_environment(environment, manifest, current_environment=environment)
    tampered = dict(environment)
    tampered["xgboost"] = "3.2.0"
    tampered_manifest = {
        "runtime": {
            **manifest["runtime"],
            "environment": tampered,
            "environment_sha256": stable_hash(tampered),
        }
    }
    with pytest.raises(ExperimentValidationError, match="xgboost"):
        _validate_environment(
            tampered, tampered_manifest, current_environment=tampered
        )

    identity = {"environment_sha256": stable_hash(environment), "context_jobs": 4}
    prepare_experiment_directory(tmp_path / "run", identity=identity, resume=False)
    with pytest.raises(ValueError, match="identity changed"):
        prepare_experiment_directory(
            tmp_path / "run",
            identity={**identity, "context_jobs": 2},
            resume=True,
        )


def test_full_run_git_revision_rejects_untracked_code(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"], cwd=repository, check=True
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", "freeze"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    assert len(_git_revision(repository)) == 40
    (repository / "untracked.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ExperimentRunError, match="clean.*Git worktree"):
        _git_revision(repository)
