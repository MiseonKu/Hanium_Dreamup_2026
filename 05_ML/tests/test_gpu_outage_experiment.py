from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from imu_vive_learning.cli import build_parser, main
from imu_vive_learning.experiment_metrics import regression_metrics
from imu_vive_learning.features import FUSION_COLUMNS
from imu_vive_learning.gpu_outage_data import (
    ChallengeIdentityData,
    DevelopmentData,
    ordered_trial_uid_sha256,
)
from imu_vive_learning.gpu_outage_experiment import (
    OUTAGE_SOURCE_FILES,
    _artifact_reference,
    _execute_development_task,
    _exclusive_write_json,
    _fit_preprocessor,
    _load_valid_task_result,
    _open_challenge_once,
    _prelabel_gate_then_open,
    _read_exact_parquet_projection,
    _read_resolved_outage_config,
    _refit_and_validate_development_preprocessor,
    _refit_and_validate_final_preprocessor,
    _require_canonical_reference,
    _run_identity_payload,
    _validate_challenge_identity_contract,
    _validate_development_best_rmse_binding,
    _validate_development_prediction_rows,
    _validate_development_seed_result,
    _validate_mask_schedule_history,
    _validate_source_provenance,
    _write_challenge_identity,
    outage_development_tasks,
    outage_final_tasks,
    outage_training_source_payload,
    run_outage_gpu_experiment,
    score_outage_candidates,
)
from imu_vive_learning.gpu_outage_experiment_config import (
    OUTAGE_CANDIDATE_IDS,
    load_outage_gpu_config,
)
from imu_vive_learning.experiment_io import atomic_write_json, atomic_write_parquet
from imu_vive_learning.gpu_outage_experiment_config import public_outage_gpu_config
from imu_vive_learning.gpu_outage_training import deterministic_mask_states
from imu_vive_learning.gpu_v3_backbones import IMU_SENSOR_INDEX_MAP


WORKSPACE = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORKSPACE / "configs" / "endpoint_regression_gpu_outage_v3.yaml"


def _config() -> dict:
    return load_outage_gpu_config(CONFIG_PATH)


def _uid(index: int) -> str:
    return "trial_" + hashlib.sha256(f"synthetic-{index}".encode()).hexdigest()


def _development_data() -> DevelopmentData:
    rng = np.random.default_rng(7)
    rows = 8
    return DevelopmentData(
        feature_names=FUSION_COLUMNS,
        X=rng.normal(size=(rows, 43)),
        y=rng.normal(size=(rows, 3)) * 10.0,
        availability=np.ones((rows, 4)),
        trial_uids=np.asarray([_uid(index) for index in range(rows)]),
        segment_ids=np.repeat(np.asarray((36, 37, 38, 39)), 2),
        target_ids=np.asarray([f"T{index % 2}" for index in range(rows)]),
    )


def _challenge_fixture(
    *, interleaved_segments: bool = False
) -> tuple[dict, ChallengeIdentityData]:
    config = _config()
    rows = 4
    rng = np.random.default_rng(11)
    X = rng.normal(size=(rows, 43))
    X[:, np.asarray(IMU_SENSOR_INDEX_MAP[1], dtype=np.int64)] = np.nan
    segments = np.asarray(
        (40, 41, 40, 41) if interleaved_segments else (40, 40, 41, 41),
        dtype=np.int64,
    )
    identity = ChallengeIdentityData(
        feature_names=FUSION_COLUMNS,
        X=X,
        availability=np.tile(np.asarray((1.0, 0.0, 1.0, 1.0)), (rows, 1)),
        trial_uids=np.asarray([_uid(100 + index) for index in range(rows)]),
        segment_ids=segments,
        target_ids=np.asarray(("T0", "T1", "T0", "T1")),
    )
    role = config["row_roles"]["challenge"]
    role["expected_rows"] = rows
    role["expected_segment_rows"] = {40: 2, 41: 2}
    role["ordered_trial_uid_sha256"] = ordered_trial_uid_sha256(
        identity.trial_uids.tolist()
    )
    return config, identity


def test_development_grid_is_exactly_five_by_sixteen_by_three() -> None:
    tasks = outage_development_tasks(_config())
    assert len(tasks) == 240
    assert len({task["task_id"] for task in tasks}) == 240
    assert {task["candidate_id"] for task in tasks} == set(OUTAGE_CANDIDATE_IDS[:-1])
    assert {task["held_out_segment"] for task in tasks} == {36, 37, 38, 39}
    assert {task["forced_outage_index"] for task in tasks} == {0, 1, 2, 3}
    assert {task["paired_seed_index"] for task in tasks} == {0, 1, 2}
    assert {
        (task["paired_seed_index"], task["model_seed"]) for task in tasks
    } == {(0, 20260824), (1, 20260825), (2, 20260826)}


def test_resolved_json_restores_integer_segment_count_keys(tmp_path: Path) -> None:
    config = _config()
    path = atomic_write_json(tmp_path / "config.resolved.json", public_outage_gpu_config(config))
    restored = _read_resolved_outage_config(path)
    assert restored["row_roles"]["development"]["expected_segment_rows"] == {
        36: 1672,
        37: 1597,
        38: 1707,
        39: 1659,
    }


def test_score_uses_mean_worst_segment_and_worst_sensor() -> None:
    config = _config()
    contexts = []
    for candidate_priority, candidate_id in enumerate(OUTAGE_CANDIDATE_IDS):
        base = 5.0 + candidate_priority
        for segment in (36, 37, 38, 39):
            for outage in range(4):
                rmse = base + 0.1 * (segment - 36) + 0.2 * outage
                contexts.append(
                    {
                        "candidate_id": candidate_id,
                        "held_out_segment": segment,
                        "forced_outage_index": outage,
                        "metrics": {
                            "rmse_3d_mm": rmse,
                            "p95_3d_error_mm": rmse + 1.0,
                        },
                        "seed_rmse_population_std_mm": 0.1,
                    }
                )
    selection = score_outage_candidates(contexts, config)
    assert selection["selected_candidate_id"] == "masked_mlp"
    winner = selection["trace"][0]
    expected_mean = np.mean(
        [5.0 + 0.1 * segment_offset + 0.2 * outage for segment_offset in range(4) for outage in range(4)]
    )
    expected_ws = np.mean([5.0 + 0.3 + 0.2 * outage for outage in range(4)])
    expected_wm = np.mean([5.0 + 0.1 * segment_offset + 0.6 for segment_offset in range(4)])
    assert np.isclose(winner["mean_rmse_over_16_contexts_mm"], expected_mean)
    assert np.isclose(winner["worst_segment_mean_rmse_mm"], expected_ws)
    assert np.isclose(winner["worst_sensor_mean_rmse_mm"], expected_wm)
    assert np.isclose(
        winner["outage_robust_j_mm"], (expected_mean + expected_ws + expected_wm) / 3.0
    )


def test_missing_context_is_rejected_instead_of_silently_reweighted() -> None:
    config = _config()
    contexts = []
    for candidate_priority, candidate_id in enumerate(OUTAGE_CANDIDATE_IDS):
        for segment in (36, 37, 38, 39):
            for outage in range(4):
                if candidate_id == "masked_mlp" and segment == 36 and outage == 0:
                    continue
                contexts.append(
                    {
                        "candidate_id": candidate_id,
                        "held_out_segment": segment,
                        "forced_outage_index": outage,
                        "metrics": {
                            "rmse_3d_mm": 1.0 + candidate_priority,
                            "p95_3d_error_mm": 2.0 + candidate_priority,
                        },
                        "seed_rmse_population_std_mm": 0.1,
                    }
                )
    with pytest.raises(Exception, match="exactly 96 unique"):
        score_outage_candidates(contexts, config)


def test_challenge_label_access_claim_is_exclusive(tmp_path: Path) -> None:
    claim = tmp_path / "challenge" / "label_access_1.started.json"
    _exclusive_write_json(claim, {"access_index": 1, "status": "started_fail_closed"})
    assert json.loads(claim.read_text(encoding="utf-8"))["access_index"] == 1
    with pytest.raises(Exception, match="already claimed"):
        _exclusive_write_json(claim, {"access_index": 2})
    assert json.loads(claim.read_text(encoding="utf-8"))["access_index"] == 1


def test_development_prediction_identity_and_truth_bind_to_replay() -> None:
    data = _development_data()
    eval_rows = data.segment_ids == 36
    frame = pd.DataFrame(
        {
            "trial_uid": data.trial_uids[eval_rows],
            "segment_id": data.segment_ids[eval_rows],
            "target_id": data.target_ids[eval_rows],
            **{
                name: data.y[eval_rows, index]
                for index, name in enumerate(("true_x_mm", "true_y_mm", "true_z_mm"))
            },
            "pred_x_mm": data.y[eval_rows, 0],
            "pred_y_mm": data.y[eval_rows, 1],
            "pred_z_mm": data.y[eval_rows, 2],
        }
    )
    assert np.array_equal(
        _validate_development_prediction_rows(frame, data, 36), eval_rows
    )
    tampered = frame.copy()
    tampered.loc[0, "target_id"] = "T999"
    with pytest.raises(Exception, match="identity/truth"):
        _validate_development_prediction_rows(tampered, data, 36)
    tampered = frame.copy()
    tampered.loc[0, "true_x_mm"] += 1.0
    with pytest.raises(Exception, match="identity/truth"):
        _validate_development_prediction_rows(tampered, data, 36)


def test_fold_preprocessor_is_refit_from_durable_rows() -> None:
    data = _development_data()
    task = outage_development_tasks(_config())[0]
    eval_rows = data.segment_ids == int(task["held_out_segment"])
    expected = _fit_preprocessor(
        data,
        ~eval_rows,
        forbidden_rows=eval_rows,
        split_context=(
            f"outage_v3_segment{task['held_out_segment']}_"
            f"drop{task['forced_outage_index']}_seed{task['paired_seed_index']}"
        ),
    ).metadata()
    _refit_and_validate_development_preprocessor(data, task, expected)
    tampered = json.loads(json.dumps(expected))
    tampered["feature_mean"][0] += 0.01
    with pytest.raises(Exception, match="does not reproduce durable fold"):
        _refit_and_validate_development_preprocessor(data, task, tampered)


def test_final_preprocessor_is_refit_from_all_durable_rows() -> None:
    data = _development_data()
    all_rows = np.ones(data.rows, dtype=bool)
    expected = _fit_preprocessor(
        data,
        all_rows,
        forbidden_rows=np.zeros(data.rows, dtype=bool),
        split_context="outage_v3_final_all_development",
    ).metadata()
    _refit_and_validate_final_preprocessor(data, expected)
    tampered = json.loads(json.dumps(expected))
    tampered["target_mean"][0] += 0.01
    with pytest.raises(Exception, match="does not reproduce all durable"):
        _refit_and_validate_final_preprocessor(data, tampered)


def test_mask_schedule_hash_and_counts_recompute_from_train_uids() -> None:
    config = _config()
    task = outage_development_tasks(config)[0]
    data = _development_data()
    train_uids = data.trial_uids[data.segment_ids != int(task["held_out_segment"])]
    rows = []
    for epoch in (1, 2):
        states = deterministic_mask_states(
            train_uids,
            base_seed=int(config["contract"]["base_seed"]),
            held_out_segment=task["held_out_segment"],
            forced_outage=int(task["forced_outage_index"]),
            paired_seed_index=int(task["paired_seed_index"]),
            epoch=epoch,
        )
        rows.append(
            {
                "epoch": epoch,
                "mask_schedule_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
                "mask_full_row_count": int(np.count_nonzero(states == -1)),
                **{
                    f"mask_drop_imu{sensor}_row_count": int(
                        np.count_nonzero(states == sensor)
                    )
                    for sensor in range(4)
                },
            }
        )
    history = pd.DataFrame(rows)
    _validate_mask_schedule_history(
        history,
        train_uids,
        base_seed=int(config["contract"]["base_seed"]),
        held_out_segment=task["held_out_segment"],
        forced_outage=int(task["forced_outage_index"]),
        paired_seed_index=int(task["paired_seed_index"]),
    )
    tampered = history.copy()
    tampered.loc[0, "mask_schedule_sha256"] = "0" * 64
    with pytest.raises(Exception, match="mask schedule evidence"):
        _validate_mask_schedule_history(
            tampered,
            train_uids,
            base_seed=int(config["contract"]["base_seed"]),
            held_out_segment=task["held_out_segment"],
            forced_outage=int(task["forced_outage_index"]),
            paired_seed_index=int(task["paired_seed_index"]),
        )


def test_challenge_identity_contract_binds_schema_and_segment_uid_order(
    tmp_path: Path,
) -> None:
    config, identity = _challenge_fixture()
    _write_challenge_identity(identity, tmp_path, config)
    _validate_challenge_identity_contract(tmp_path, identity, config)

    metadata_path = tmp_path / "challenge" / "identity.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config_sha256"] = "0" * 64
    atomic_write_json(metadata_path, metadata)
    with pytest.raises(Exception, match="metadata/config binding"):
        _validate_challenge_identity_contract(tmp_path, identity, config)

    bad_dir = tmp_path / "interleaved"
    bad_config, bad_identity = _challenge_fixture(interleaved_segments=True)
    _write_challenge_identity(bad_identity, bad_dir, bad_config)
    with pytest.raises(Exception, match="segment-to-UID"):
        _validate_challenge_identity_contract(bad_dir, bad_identity, bad_config)

    wrong_outage_dir = tmp_path / "wrong-outage"
    wrong_X = np.array(identity.X, copy=True)
    wrong_X[:, np.asarray(IMU_SENSOR_INDEX_MAP[0], dtype=np.int64)] = np.nan
    wrong_X[:, np.asarray(IMU_SENSOR_INDEX_MAP[1], dtype=np.int64)] = 0.0
    wrong_identity = ChallengeIdentityData(
        feature_names=FUSION_COLUMNS,
        X=wrong_X,
        availability=np.tile(np.asarray((0.0, 1.0, 1.0, 1.0)), (identity.rows, 1)),
        trial_uids=identity.trial_uids,
        segment_ids=identity.segment_ids,
        target_ids=identity.target_ids,
    )
    _write_challenge_identity(wrong_identity, wrong_outage_dir, config)
    with pytest.raises(Exception, match="forced IMU1 availability"):
        _validate_challenge_identity_contract(
            wrong_outage_dir, wrong_identity, config
        )


def test_parquet_schema_is_rejected_before_extra_label_projection(tmp_path: Path) -> None:
    path = atomic_write_parquet(
        tmp_path / "identity.parquet",
        pd.DataFrame(
            {
                "trial_uid": [_uid(1)],
                "segment_id": np.asarray([40], dtype=np.int64),
                "target_id": ["T0"],
                "feature": np.asarray([1.0], dtype=np.float64),
                "true_x_mm": np.asarray([123.0], dtype=np.float64),
            }
        ),
    )
    with pytest.raises(Exception, match="schema changed before projection"):
        _read_exact_parquet_projection(
            path, ["trial_uid", "segment_id", "target_id", "feature"]
        )


def test_canonical_reference_rejects_byte_identical_shadow_path(tmp_path: Path) -> None:
    canonical = atomic_write_json(tmp_path / "development" / "selection.json", {"x": 1})
    shadow = atomic_write_json(tmp_path / "shadow" / "selection.json", {"x": 1})
    reference = _artifact_reference(shadow, tmp_path)
    with pytest.raises(Exception, match="canonical path"):
        _require_canonical_reference(tmp_path, reference, canonical)


def test_development_best_rmse_binds_durable_truth() -> None:
    truth = np.asarray(((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)))
    prediction = truth + 1.0
    rmse = regression_metrics(truth, prediction)["rmse_3d_mm"]
    result = {"summary": {"best_validation_rmse_3d_mm": rmse}}
    _validate_development_best_rmse_binding(
        result=result,
        durable_truth=truth,
        stored_prediction=prediction,
        replay_prediction=prediction,
    )
    result["summary"]["best_validation_rmse_3d_mm"] = float(rmse) + 1.0
    with pytest.raises(Exception, match="selected history best"):
        _validate_development_best_rmse_binding(
            result=result,
            durable_truth=truth,
            stored_prediction=prediction,
            replay_prediction=prediction,
        )
def test_injected_seed_smoke_writes_atomic_hashed_artifacts(tmp_path: Path) -> None:
    config = _config()
    config["training"]["epoch_budget"] = 3
    # The real grid stays immutable.  This smoke calls one already-created task
    # with an injected trainer and never launches CUDA or reads challenge data.
    frozen_config = _config()
    task = outage_development_tasks(frozen_config)[0]
    data = _development_data()

    def fake_train(**kwargs):
        truth = np.asarray(kwargs["y_eval_raw"], dtype=np.float64)
        prediction = truth + np.asarray((0.1, -0.2, 0.3))
        metrics = regression_metrics(truth, prediction)
        train_truth = kwargs["preprocessor"].inverse_targets(kwargs["y_train"])
        history = [
            {
                "epoch": epoch,
                "scaled_objective_loss": float(4 - epoch),
                "scaled_endpoint_member_mse": float(4 - epoch),
                "scaled_masked_reconstruction_mse": 0.0,
                "validation_rmse_3d_mm": float(metrics["rmse_3d_mm"] + epoch - 1),
                "validation_p95_3d_error_mm": float(metrics["p95_3d_error_mm"]),
                "checkpoint_improved": epoch == 1,
                "best_epoch_so_far": 1,
                "best_validation_rmse_3d_mm": float(metrics["rmse_3d_mm"]),
                "mask_schedule_sha256": "0" * 64,
                "mask_full_row_count": 1,
                "mask_drop_imu0_row_count": 1,
                "mask_drop_imu1_row_count": 1,
                "mask_drop_imu2_row_count": 1,
                "mask_drop_imu3_row_count": 2,
            }
            for epoch in range(1, 4)
        ]
        return {
            "candidate_id": task["candidate_id"],
            "model_seed": task["model_seed"],
            "paired_seed_index": task["paired_seed_index"],
            "held_out_segment": str(task["held_out_segment"]),
            "forced_outage": task["forced_outage_index"],
            "trained_epochs": 3,
            "best_epoch": 1,
            "best_validation_rmse_3d_mm": metrics["rmse_3d_mm"],
            "history": history,
            "state_dict": {"weight": np.asarray([1.0], dtype=np.float32)},
            "eval_prediction_raw_mm": prediction,
            "eval_metrics": metrics,
            "train_prediction_raw_mm": train_truth,
            "train_metrics": regression_metrics(train_truth, train_truth),
        }

    result = _execute_development_task(
        data,
        task,
        config,
        tmp_path,
        gpu_id=0,
        train_fn=fake_train,
    )
    assert set(result["artifacts"]) == {
        "preprocessing",
        "history",
        "predictions",
        "checkpoint",
    }
    assert all(len(reference["sha256"]) == 64 for reference in result["artifacts"].values())
    resumed = _load_valid_task_result(tmp_path, task)
    assert resumed is not None
    assert resumed["task"] == task
    tampered_summary = dict(resumed)
    tampered_summary["summary"] = dict(resumed["summary"])
    tampered_summary["summary"]["best_epoch"] = 2
    with pytest.raises(Exception, match="earliest minimum"):
        _validate_development_seed_result(
            tmp_path, config, tampered_summary
        )
    for artifact_name in ("history", "checkpoint"):
        artifact = tmp_path / resumed["artifacts"][artifact_name]["path"]
        original = artifact.read_bytes()
        artifact.write_bytes(original + b"tamper")
        assert _load_valid_task_result(tmp_path, task) is None
        artifact.write_bytes(original)
        assert _load_valid_task_result(tmp_path, task) is not None


def test_prelabel_gate_never_calls_unseal_after_semantic_failure(tmp_path: Path) -> None:
    calls = {"open": 0}

    def reject_tampered(_run_dir):
        raise RuntimeError("tampered summary/history/checkpoint")

    def forbidden_open(**_kwargs):
        calls["open"] += 1
        return {}

    with pytest.raises(RuntimeError, match="tampered"):
        _prelabel_gate_then_open(
            run_dir=tmp_path,
            validate_fn=reject_tampered,
            open_fn=forbidden_open,
        )
    assert calls["open"] == 0


def test_transitive_training_dependency_changes_source_fingerprint(tmp_path: Path) -> None:
    for relative in OUTAGE_SOURCE_FILES:
        source = WORKSPACE / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    before = outage_training_source_payload(tmp_path)["fingerprint"]
    required_transitive = (
        "src/imu_vive_learning/gpu_robust_experiment_config.py",
        "src/imu_vive_learning/gpu_experiment_config.py",
        "src/imu_vive_learning/splitting.py",
        "src/imu_vive_learning/experiment_splits.py",
        "src/imu_vive_learning/gpu_experiment.py",
        "src/imu_vive_learning/gpu_inference.py",
    )
    assert set(required_transitive).issubset(OUTAGE_SOURCE_FILES)
    for relative in required_transitive:
        dependency = tmp_path / relative
        original = dependency.read_bytes()
        dependency.write_bytes(original + b"\n# fingerprint mutation\n")
        assert outage_training_source_payload(tmp_path)["fingerprint"] != before
        dependency.write_bytes(original)


def test_source_provenance_binds_current_snapshot_identity_and_config(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = tmp_path / "run"
    for relative in OUTAGE_SOURCE_FILES:
        source = WORKSPACE / relative
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    current = outage_training_source_payload(workspace)
    config = _config()
    config["_training_source_fingerprint"] = current["fingerprint"]
    atomic_write_json(run_dir / "provenance" / "source_snapshot.json", current)
    atomic_write_json(
        run_dir / "_state" / "run_identity.json",
        _run_identity_payload(config, current["fingerprint"]),
    )
    _validate_source_provenance(run_dir, config, workspace_root=workspace)
    dependency = workspace / "src" / "imu_vive_learning" / "splitting.py"
    dependency.write_bytes(dependency.read_bytes() + b"\n# drift\n")
    with pytest.raises(Exception, match="differs from the frozen snapshot"):
        _validate_source_provenance(run_dir, config, workspace_root=workspace)


def test_fresh_finalize_is_rejected_before_creating_run(tmp_path: Path) -> None:
    output = tmp_path / "fresh-finalize"
    with pytest.raises(Exception, match="prior sealed development run"):
        run_outage_gpu_experiment(
            _config(),
            output,
            resume=False,
            finalize_challenge=True,
        )
    assert not output.exists()


def test_staged_finalize_validates_manifest_before_any_final_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "staged"
    atomic_write_json(output / "development_manifest.json", {"tampered": True})
    calls = {"validation": 0}

    def reject(_run_dir, *, _allow_final_artifacts=False):
        calls["validation"] += 1
        assert _allow_final_artifacts is True
        raise RuntimeError("tampered development manifest")

    monkeypatch.setattr(
        "imu_vive_learning.gpu_outage_experiment.validate_outage_development_run",
        reject,
    )
    with pytest.raises(RuntimeError, match="tampered development"):
        run_outage_gpu_experiment(
            _config(),
            output,
            resume=True,
            finalize_challenge=True,
        )
    assert calls["validation"] == 1
    assert not (output / "final").exists()


def test_in_memory_freeze_mismatch_blocks_label_loader(tmp_path: Path) -> None:
    config, identity = _challenge_fixture()
    selection = {"selected_candidate_id": "masked_mlp"}
    durable_freeze = {"status": "frozen_before_label_access", "token": "durable"}
    atomic_write_json(
        tmp_path / "challenge" / "freeze.before_label_access.json", durable_freeze
    )
    atomic_write_json(
        tmp_path / "challenge" / "prelabel_semantic_validation.json",
        {
            "status": "ok",
            "challenge_labels_accessed": False,
            "selected_candidate_id": "masked_mlp",
        },
    )
    atomic_write_json(tmp_path / "development" / "selection.json", selection)
    calls = {"labels": 0}

    def forbidden_loader(*_args, **_kwargs):
        calls["labels"] += 1
        raise AssertionError("label loader must remain sealed")

    with pytest.raises(Exception, match="differs from durable frozen evidence"):
        _open_challenge_once(
            config=config,
            source_config={},
            identity=identity,
            selection=selection,
            freeze={"status": "frozen_before_label_access", "token": "memory"},
            run_dir=tmp_path,
            label_loader=forbidden_loader,
        )
    assert calls["labels"] == 0


def test_final_epoch_is_half_up_per_component_and_paired_seed() -> None:
    config = _config()
    tasks = outage_development_tasks(config)
    results = []
    for task in tasks:
        # Seed 0 has an even median halfway between 10 and 11 -> 11 half-up.
        context_position = len(
            [
                row
                for row in results
                if row["task"]["candidate_id"] == task["candidate_id"]
                and row["task"]["paired_seed_index"] == task["paired_seed_index"]
            ]
        )
        best_epoch = (10 if context_position < 8 else 11) + task["paired_seed_index"]
        results.append({"task": task, "summary": {"best_epoch": best_epoch}})
    selection = {"selected_candidate_id": "masked_mlp"}
    final = outage_final_tasks(config, selection, results)
    assert len(final) == 3
    assert [task["epochs"] for task in final] == [11, 12, 13]
    assert all(task["mask_held_out_segment"] == "final_refit_all_development" for task in final)
    assert all(task["mask_forced_outage_index"] == 1 for task in final)


def test_cli_exposes_run_and_source_free_validator() -> None:
    parser = build_parser()
    run = parser.parse_args(
        [
            "run-gpu-outage-experiment",
            "--development-release",
            "dev.zip",
            "--source",
            "source.zip",
            "--config",
            "outage.yaml",
            "--output",
            "run",
        ]
    )
    assert run.gpus == [0, 1, 2, 3]
    assert run.finalize_challenge is False
    validate = parser.parse_args(["validate-gpu-outage-experiment", "--run-dir", "run"])
    assert validate.command == "validate-gpu-outage-experiment"


def test_outage_cli_dispatches_explicit_finalize_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import imu_vive_learning.gpu_outage_experiment as experiment
    import imu_vive_learning.gpu_outage_experiment_config as experiment_config

    calls: dict[str, object] = {}
    loaded = {"loaded": True}

    def fake_load(
        path: Path,
        *,
        development_release_override: Path,
        source_override: Path,
    ) -> dict:
        calls["load"] = (
            path,
            development_release_override,
            source_override,
        )
        return loaded

    def fake_run(
        config: dict,
        output: Path,
        *,
        gpu_ids: list[int],
        resume: bool,
        finalize_challenge: bool,
    ) -> dict:
        calls["run"] = (
            config,
            output,
            gpu_ids,
            resume,
            finalize_challenge,
        )
        return {"status": "ok", "challenge_labels_sealed": False}

    monkeypatch.setattr(experiment_config, "load_outage_gpu_config", fake_load)
    monkeypatch.setattr(experiment, "run_outage_gpu_experiment", fake_run)
    status = main(
        [
            "run-gpu-outage-experiment",
            "--development-release",
            "dev.zip",
            "--source",
            "source.zip",
            "--config",
            "outage.yaml",
            "--output",
            "outage-run",
            "--gpus",
            "0",
            "1",
            "2",
            "3",
            "--resume",
            "--finalize-challenge",
        ]
    )
    assert status == 0
    assert calls == {
        "load": (Path("outage.yaml"), Path("dev.zip"), Path("source.zip")),
        "run": (loaded, Path("outage-run"), [0, 1, 2, 3], True, True),
    }
    assert json.loads(capsys.readouterr().out) == {
        "status": "ok",
        "challenge_labels_sealed": False,
    }
