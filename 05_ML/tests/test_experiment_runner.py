from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import pytest

from imu_vive_learning.experiment_config import (
    EXPERIMENT_CONTRACT,
    load_experiment_config,
)
from imu_vive_learning.experiment_ensemble import SerializableEndpointEnsemble
from imu_vive_learning.experiment_inference import predict_endpoint
from imu_vive_learning.experiment_io import (
    atomic_joblib_dump,
    atomic_write_json,
    manifest_files,
    sha256_file,
)
from imu_vive_learning.experiment_matrix import ExperimentCell
from imu_vive_learning.experiment_models import (
    ConstantXYZRegressor,
    SerializableEndpointRegressor,
)
from imu_vive_learning.experiment_packaging import (
    EXPERIMENT_PACKAGE_ROOT,
    package_experiment_run,
    validate_experiment_package,
)
from imu_vive_learning.experiment_runner import (
    PREDICTION_COLUMNS,
    TRUE_COLUMNS,
    ExperimentRunError,
    _load_task_if_complete,
    _prediction_frame,
    _run_outer_cell,
    _task_directory,
)
from imu_vive_learning.features import LABEL_COLUMNS


WORKSPACE = Path(__file__).parents[1]
CONFIG_PATH = WORKSPACE / "configs" / "endpoint_regression_v1.yaml"


def _dummy_cell() -> ExperimentCell:
    return ExperimentCell(
        model="dummy_mean",
        feature_view="none",
        recipe="real_only",
        purposes=("baseline",),
        diagnostic_only=False,
        deployment_candidate=False,
    )


def _tiny_eligible() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for segment in range(36, 41):
        for repeat in range(2):
            value = float((segment - 36) * 10 + repeat)
            records.append(
                {
                    "trial_uid": f"s{segment}-r{repeat}",
                    "segment_id": segment,
                    "target_id": f"T{segment - 35:02d}",
                    "strict_qc_eligible": repeat == 0,
                    LABEL_COLUMNS[0]: value,
                    LABEL_COLUMNS[1]: value + 1.0,
                    LABEL_COLUMNS[2]: value + 2.0,
                }
            )
    return pd.DataFrame.from_records(records)


def _outer_contexts(
    eligible: pd.DataFrame, *, outer_fold: int
) -> dict[tuple[str, int, int], pd.DataFrame]:
    contexts: dict[tuple[str, int, int], pd.DataFrame] = {}
    segments = eligible["segment_id"].to_numpy(int)
    for inner_fold in (-1, *(value for value in range(36, 41) if value != outer_fold)):
        label = "refit" if inner_fold == -1 else str(inner_fold)
        frame = pd.DataFrame(
            {
                "split_context": f"segment_loso:outer={outer_fold}:inner={label}",
                "trial_uid": eligible["trial_uid"].astype(str),
                "role": np.where(
                    segments == outer_fold,
                    "test",
                    np.where(
                        (inner_fold != -1) & (segments == inner_fold),
                        "validation",
                        "train",
                    ),
                ),
            }
        )
        contexts[("segment_loso", outer_fold, inner_fold)] = frame
    return contexts


def _constant_member(mean_xyz: tuple[float, float, float], *, seed: int) -> SerializableEndpointRegressor:
    return SerializableEndpointRegressor(
        model_name="fixture_constant",
        recipe="real_only",
        feature_names=("f0",),
        estimator=ConstantXYZRegressor(mean_xyz),
        preprocessor=None,
        seed=seed,
        params={"fixture": True},
        fit_metadata={"fit_scope": "measured_real_fixture"},
    )


def _fixture_ensemble() -> SerializableEndpointEnsemble:
    return SerializableEndpointEnsemble(
        members=(
            _constant_member((1.0, 2.0, 3.0), seed=11),
            _constant_member((3.0, 4.0, 5.0), seed=22),
        ),
        model_name="fixture_constant",
        recipe="real_only",
        feature_names=("f0",),
        metadata={"fixture": "deterministic"},
    )


def test_one_cell_outer_pipeline_checkpoint_is_atomic_and_resume_safe(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(CONFIG_PATH)
    eligible = _tiny_eligible()
    eligible_indexed = eligible.set_index(eligible["trial_uid"].astype(str), drop=False)
    contexts = _outer_contexts(eligible, outer_fold=36)
    state_root = tmp_path / "state"
    cache_root = tmp_path / "cache"
    cell = _dummy_cell()

    first = _run_outer_cell(
        eligible_indexed,
        contexts,
        {},
        cache_root=cache_root,
        state_root=state_root,
        config=config,
        release_sha256="a" * 64,
        assignment_hash="b" * 64,
        scheme="segment_loso",
        outer_fold=36,
        cell=cell,
        feature_columns=(),
    )
    task_dir = _task_directory(
        state_root, scheme="segment_loso", outer_fold=36, cell_id=cell.cell_id
    )
    member_hashes = dict(first["member_sha256"])
    member_mtimes = {
        name: (task_dir / name).stat().st_mtime_ns for name in member_hashes
    }

    second = _run_outer_cell(
        eligible_indexed,
        contexts,
        {},
        cache_root=cache_root,
        state_root=state_root,
        config=config,
        release_sha256="a" * 64,
        assignment_hash="b" * 64,
        scheme="segment_loso",
        outer_fold=36,
        cell=cell,
        feature_columns=(),
    )
    assert second == first
    assert {
        name: sha256_file(task_dir / name) for name in member_hashes
    } == member_hashes
    assert {
        name: (task_dir / name).stat().st_mtime_ns for name in member_hashes
    } == member_mtimes

    ensemble_oof = pd.read_parquet(task_dir / "oof_ensemble.parquet")
    assert ensemble_oof["trial_uid"].tolist() == ["s36-r0", "s36-r1"]
    assert ensemble_oof["prediction_kind"].eq("ensemble").all()
    assert len(pd.read_parquet(task_dir / "inner_trace.parquet")) == 8
    assert not list(task_dir.glob(".*.tmp"))

    with pytest.raises(ExperimentRunError, match="task hash changed"):
        _load_task_if_complete(task_dir, "c" * 64)

    (task_dir / "preprocessing.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ExperimentRunError, match="checkpoint member changed"):
        _load_task_if_complete(task_dir, str(first["task_hash"]))


def test_prediction_frame_has_exact_oof_schema_and_preserves_trial_order() -> None:
    evaluation = pd.DataFrame(
        {
            "trial_uid": ["trial-b", "trial-a"],
            "segment_id": [40, 36],
            "target_id": ["T07", "T01"],
            "strict_qc_eligible": [False, True],
            LABEL_COLUMNS[0]: [1.0, 4.0],
            LABEL_COLUMNS[1]: [2.0, 5.0],
            LABEL_COLUMNS[2]: [3.0, 6.0],
        }
    )
    prediction = np.array([[2.0, 2.0, 3.0], [4.0, 3.0, 6.0]])
    frame = _prediction_frame(
        evaluation,
        prediction,
        scheme="segment_loso",
        outer_fold=36,
        cell=_dummy_cell(),
        config_id="fixture-config",
        params={},
        prediction_kind="ensemble",
        seed_index=-1,
        model_seed=-1,
    )

    assert frame.columns.tolist() == [
        "scheme",
        "outer_fold",
        "cell_id",
        "model",
        "feature_view",
        "recipe",
        "config_id",
        "params_json",
        "prediction_kind",
        "seed_index",
        "model_seed",
        "trial_uid",
        "segment_id",
        "target_id",
        "target_group",
        "strict_qc_eligible",
        *TRUE_COLUMNS,
        *PREDICTION_COLUMNS,
        "error_3d_mm",
    ]
    assert frame["trial_uid"].tolist() == ["trial-b", "trial-a"]
    assert frame["target_group"].tolist() == [2, 1]
    assert frame["error_3d_mm"].tolist() == pytest.approx([1.0, 2.0])


def test_ensemble_serialization_and_inference_allowlist_are_exact(tmp_path: Path) -> None:
    ensemble = _fixture_ensemble()
    with_ids = pd.DataFrame({"row_id": ["b", "a"], "f0": [2.0, 1.0]})
    prediction, row_ids = ensemble.predict_frame(with_ids)
    assert row_ids is not None
    assert row_ids.tolist() == ["b", "a"]
    assert prediction == pytest.approx(np.tile([2.0, 3.0, 4.0], (2, 1)))

    prediction_without_ids, absent_ids = ensemble.predict_frame(
        pd.DataFrame({"f0": [5.0]})
    )
    assert absent_ids is None
    assert prediction_without_ids.tolist() == [[2.0, 3.0, 4.0]]
    with pytest.raises(ValueError, match="schema/order mismatch"):
        ensemble.predict_frame(with_ids.loc[:, ["f0", "row_id"]])
    with pytest.raises(ValueError, match="row_id values must be unique"):
        ensemble.predict_frame(pd.DataFrame({"row_id": ["x", "x"], "f0": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="non-empty and finite"):
        ensemble.predict_frame(pd.DataFrame({"f0": [np.nan]}))

    model_path = tmp_path / "ensemble.joblib"
    atomic_joblib_dump(model_path, ensemble)
    loaded = joblib.load(model_path)
    assert isinstance(loaded, SerializableEndpointEnsemble)
    assert loaded.contract_fingerprint() == ensemble.contract_fingerprint()
    assert loaded.predict_frame(with_ids)[0] == pytest.approx(prediction)


def test_package_and_predict_are_deterministic_and_checksum_closed(
    tmp_path: Path,
) -> None:
    ensemble = _fixture_ensemble()
    run = tmp_path / "run"
    model_path = run / "final" / "model.joblib"
    atomic_joblib_dump(model_path, ensemble)
    model_hash = sha256_file(model_path)
    atomic_write_json(
        run / "final" / "model_metadata.json",
        {
            "contract": EXPERIMENT_CONTRACT,
            "model": {
                "sha256": model_hash,
                "contract_fingerprint": ensemble.contract_fingerprint(),
            },
        },
    )
    atomic_write_json(
        run / "manifest.json",
        {
            "contract": EXPERIMENT_CONTRACT,
            "final_model": {"independent_test_score_available": False},
            "files": manifest_files(run),
        },
    )

    first_package = package_experiment_run(run, tmp_path / "first.zip")
    second_package = package_experiment_run(run, tmp_path / "second.zip")
    assert first_package.read_bytes() == second_package.read_bytes()
    validation = validate_experiment_package(first_package)
    assert validation["status"] == "ok"
    assert validation["files"] == 3
    with ZipFile(first_package) as archive:
        assert archive.namelist() == [
            f"{EXPERIMENT_PACKAGE_ROOT}/manifest.json",
            f"{EXPERIMENT_PACKAGE_ROOT}/final/model.joblib",
            f"{EXPERIMENT_PACKAGE_ROOT}/final/model_metadata.json",
        ]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())

    input_path = tmp_path / "features.parquet"
    output_path = tmp_path / "predictions.parquet"
    pd.DataFrame({"row_id": ["row-2", "row-1"], "f0": [9.0, 8.0]}).to_parquet(
        input_path, index=False
    )
    result = predict_endpoint(first_package, input_path, output_path)
    assert result["status"] == "ok"
    assert result["rows"] == 2
    output = pd.read_parquet(output_path)
    assert output.columns.tolist() == [
        "row_id",
        *PREDICTION_COLUMNS,
        "model_sha256",
        "model_contract_fingerprint",
        "model_package_sha256",
        "output_frame",
        "output_unit",
    ]
    assert output["row_id"].tolist() == ["row-2", "row-1"]
    assert output.loc[:, list(PREDICTION_COLUMNS)].to_numpy() == pytest.approx(
        np.tile([2.0, 3.0, 4.0], (2, 1))
    )
    assert output["model_sha256"].eq(model_hash).all()
    assert output["output_frame"].eq("vive_world").all()
    assert output["output_unit"].eq("mm").all()

    model_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="manifest size mismatch before packaging"):
        package_experiment_run(run, tmp_path / "tampered.zip")
