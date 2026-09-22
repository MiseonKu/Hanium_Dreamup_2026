from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import math
import os
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import imu_vive_learning.gpu_outage_gbdt_v4_experiment as v4_experiment
import imu_vive_learning.gpu_outage_gbdt_v4_training as v4_training
from imu_vive_learning.experiment_io import (
    atomic_write_json,
    read_json_object,
    sha256_file,
    stable_hash,
)
from imu_vive_learning.features import FUSION_COLUMNS
from imu_vive_learning.gpu_outage_data import DevelopmentData
from imu_vive_learning.gpu_outage_experiment import (
    outage_development_tasks as v3_development_tasks,
    score_outage_candidates,
)
from imu_vive_learning.gpu_outage_experiment_config import load_outage_gpu_config
from imu_vive_learning.gpu_outage_experiment_config import (
    outage_gpu_config_sha256,
    public_outage_gpu_config,
)
from imu_vive_learning.gpu_outage_gbdt_v4_config import (
    GBDT_V4_CANDIDATE_IDS,
    GPUOutageGBDTV4ConfigError,
    gbdt_v4_config_sha256,
    load_gbdt_v4_config,
    validate_gbdt_v4_config,
)
from imu_vive_learning.gpu_outage_gbdt_v4_experiment import (
    GBDT_V4_SOURCE_FILES,
    GPUOutageGBDTV4Error,
    _execute_xgb_development_task,
    _exclusive_write_json,
    _quarantine_task_directory,
    _reconstruct_xgb_development_result,
    _require_sealed_v3_development_validation,
    _prepared_manifest_files,
    _run_xgb_development_pool,
    _v3_development_tasks,
    _validate_aggregate_worker_ledger,
    _validate_parent_cuda_device_grid,
    _validate_prepare_run_paths,
    _validate_quarantine_phase,
    _validate_sealed_v3_development,
    _validate_worker_cuda_mapping,
    gbdt_v4_environment_payload,
    gbdt_v4_source_payload,
    gbdt_v4_development_tasks,
    prepare_gbdt_v4_run,
    run_gbdt_v4_experiment,
    score_gbdt_v4_candidates,
)
from imu_vive_learning.gpu_outage_gbdt_v4_config import public_gbdt_v4_config
from imu_vive_learning.gpu_outage_gbdt_v4_training import fixed_xgb_parameter_sha256
from imu_vive_learning.gpu_outage_gbdt_v4_training import (
    GBDT_V4_AXIS_SEED_RECIPE,
    GBDT_V4_FIXED_XGB_PARAMS,
    GBDT_V4_NUM_BOOST_ROUND,
)


WORKSPACE = Path(__file__).resolve().parents[1]
CONFIG_PATH = WORKSPACE / "configs" / "endpoint_regression_gpu_outage_v4_gbdt.yaml"
V3_CONFIG_PATH = WORKSPACE / "configs" / "endpoint_regression_gpu_outage_v3.yaml"


def _config() -> dict:
    return load_gbdt_v4_config(CONFIG_PATH)


def _context_rows(
    *,
    rmse_by_candidate: dict[str, float] | None = None,
    p95_by_candidate: dict[str, float] | None = None,
) -> list[dict]:
    rmses = rmse_by_candidate or {
        "v3_incumbent": 10.0,
        "xgboost_mask_replicated": 20.0,
        "equal_blend_incumbent_xgboost": 15.0,
    }
    p95s = p95_by_candidate or {candidate: value + 5.0 for candidate, value in rmses.items()}
    return [
        {
            "candidate_id": candidate,
            "held_out_segment": segment,
            "forced_outage_index": outage,
            "metrics": {
                "rmse_3d_mm": float(rmses[candidate]),
                "p95_3d_error_mm": float(p95s[candidate]),
            },
            "seed_rmse_population_std_mm": 0.5,
        }
        for candidate in GBDT_V4_CANDIDATE_IDS
        for segment in (36, 37, 38, 39)
        for outage in range(4)
    ]


def _fake_cuda_devices() -> list[dict]:
    return [
        {
            "physical_gpu_id": index,
            "uuid": f"GPU-FAKE-{index:04d}",
            "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
            "name": "NVIDIA CMP 170HX",
            "total_memory_bytes": 8 * 1024**3,
            "compute_capability": [8, 0],
        }
        for index in range(4)
    ]


def _write_synthetic_provenance(run: Path) -> None:
    import pyarrow
    import torch
    import xgboost

    source = gbdt_v4_source_payload(WORKSPACE)
    library = Path(str(xgboost.core._LIB._name)).resolve()
    payload = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "torch": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda),
        "xgboost": str(xgboost.__version__),
        "xgboost_build_info": dict(xgboost.build_info()),
        "libxgboost": {
            "path": str(library),
            "bytes": int(library.stat().st_size),
            "sha256": sha256_file(library),
        },
        "fixed_xgb_parameter_sha256": fixed_xgb_parameter_sha256(),
        "cuda_devices": _fake_cuda_devices(),
    }
    atomic_write_json(run / "provenance" / "source_snapshot.json", source)
    atomic_write_json(
        run / "provenance" / "environment_snapshot.json",
        {"payload": payload, "fingerprint": stable_hash(payload)},
    )


def _tiny_development_data() -> DevelopmentData:
    rows = 12
    row_axis = np.arange(rows, dtype=np.float64)[:, None]
    feature_axis = np.arange(len(FUSION_COLUMNS), dtype=np.float64)[None, :]
    X = np.sin(0.17 * row_axis + 0.03 * feature_axis)
    y = np.column_stack(
        (
            10.0 + np.arange(rows),
            -5.0 + 0.5 * np.arange(rows),
            3.0 - 0.25 * np.arange(rows),
        )
    ).astype(np.float64)
    segments = np.repeat(np.asarray([36, 37, 38, 39], dtype=np.int64), 3)
    return DevelopmentData(
        feature_names=FUSION_COLUMNS,
        X=X,
        y=y,
        availability=np.ones((rows, 4), dtype=np.float64),
        trial_uids=np.asarray([f"trial_{index:064x}" for index in range(rows)]),
        segment_ids=segments,
        target_ids=np.asarray([f"T{index % 3:02d}" for index in range(rows)]),
    )


def _write_synthetic_prepared_run(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> dict:
    """Materialize a complete synthetic prepared run for the real validator."""

    config = _config()
    atomic_write_json(run / "config.resolved.json", public_gbdt_v4_config(config))
    _write_synthetic_provenance(run)
    environment_payload = read_json_object(
        run / "provenance" / "environment_snapshot.json"
    )["payload"]
    monkeypatch.setattr(
        v4_experiment,
        "gbdt_v4_environment_payload",
        lambda: deepcopy(environment_payload),
    )

    v3_config = load_outage_gpu_config(V3_CONFIG_PATH)
    v3_public = public_outage_gpu_config(v3_config)
    v3_config_sha = outage_gpu_config_sha256(v3_config)
    copied_v3_config = atomic_write_json(
        run / "incumbent" / "v3_config.resolved.json", v3_public
    )
    v3_manifest = {
        "stage": "development_complete_challenge_labels_sealed",
        "challenge_labels_sealed": True,
        "config_sha256": v3_config_sha,
        "counts": {"scored_contexts": 96},
    }
    copied_v3_manifest = atomic_write_json(
        run / "incumbent" / "v3_development_manifest.json", v3_manifest
    )
    selection = {
        "selected_candidate_id": "masked_mlp",
        "challenge_used_for_selection": False,
    }
    copied_selection = atomic_write_json(
        run / "incumbent" / "v3_selection.json", selection
    )
    validator_payload = {
        "status": "ok",
        "stage": "development_complete_challenge_labels_sealed",
        "challenge_labels_sealed": True,
        "development_seed_fits": 240,
        "scored_contexts": 96,
    }
    copied_validator = atomic_write_json(
        run / "incumbent" / "v3_validator_payload.json", validator_payload
    )
    task_tree = [{"task_id": f"development__synthetic__{index:03d}"} for index in range(240)]
    task_tree_sha = stable_hash(task_tree)
    copied_tree = atomic_write_json(
        run / "incumbent" / "v3_task_prediction_tree.json",
        {"task_count": 240, "tree_sha256": task_tree_sha, "tasks": task_tree},
    )

    segment_ids = np.concatenate(
        [
            np.full(rows, segment, dtype=np.int64)
            for segment, rows in v4_experiment._DEVELOPMENT_SEGMENT_ROWS.items()
        ]
    )
    development_rows = len(segment_ids)
    development_uids = np.asarray(
        [f"trial_{index:064x}" for index in range(development_rows)]
    )
    development_frame = pd.DataFrame(
        {
            "trial_uid": development_uids,
            "segment_id": segment_ids,
            "target_id": np.asarray(
                [f"T{index % 8:02d}" for index in range(development_rows)]
            ),
            **{
                name: np.zeros(development_rows, dtype=np.float64)
                for name in FUSION_COLUMNS
            },
            "true_x_mm": np.zeros(development_rows, dtype=np.float64),
            "true_y_mm": np.ones(development_rows, dtype=np.float64),
            "true_z_mm": np.full(development_rows, 2.0, dtype=np.float64),
        }
    )
    development_path = v4_experiment.atomic_write_parquet(
        run / "inputs" / "development_replay.parquet", development_frame
    )
    development_metadata_path = atomic_write_json(
        run / "inputs" / "development_replay.json",
        {
            "rows": development_rows,
            "feature_names": list(FUSION_COLUMNS),
            "label_names": ["true_x_mm", "true_y_mm", "true_z_mm"],
            "sorted_trial_uid_sha256": hashlib.sha256(
                "\n".join(sorted(development_uids.tolist())).encode("utf-8")
            ).hexdigest(),
            "input_file": v4_experiment._artifact_reference(
                development_path, run, rows=development_rows
            ),
        },
    )

    challenge_segment_ids = np.concatenate(
        (
            np.full(1672, 40, dtype=np.int64),
            np.full(1623, 41, dtype=np.int64),
        )
    )
    challenge_rows = len(challenge_segment_ids)
    challenge_uids = np.asarray(
        [f"trial_{10000 + index:064x}" for index in range(challenge_rows)]
    )
    challenge_features = {
        name: (
            np.full(challenge_rows, np.nan, dtype=np.float64)
            if name.startswith("imu1_link_root_")
            else np.zeros(challenge_rows, dtype=np.float64)
        )
        for name in FUSION_COLUMNS
    }
    availability = (1.0, 0.0, 1.0, 1.0)
    challenge_frame = pd.DataFrame(
        {
            "trial_uid": challenge_uids,
            "segment_id": challenge_segment_ids,
            "target_id": np.asarray(
                [f"C{index % 8:02d}" for index in range(challenge_rows)]
            ),
            **challenge_features,
            **{
                name: np.full(challenge_rows, availability[index], dtype=np.float64)
                for index, name in enumerate(v4_experiment.AVAILABILITY_NAMES)
            },
        }
    )
    challenge_path = v4_experiment.atomic_write_parquet(
        run / "inputs" / "challenge_identity_inputs.parquet", challenge_frame
    )
    challenge_metadata_path = atomic_write_json(
        run / "inputs" / "challenge_identity.json",
        {
            "rows": challenge_rows,
            "feature_names": list(FUSION_COLUMNS),
            "availability_names": list(v4_experiment.AVAILABILITY_NAMES),
            "ordered_trial_uid_sha256": stable_hash(challenge_uids.tolist()),
            "identity_file": v4_experiment._artifact_reference(
                challenge_path, run, rows=challenge_rows
            ),
        },
    )

    snapshots: list[dict] = []
    for segment, rows in v4_experiment._DEVELOPMENT_SEGMENT_ROWS.items():
        segment_frame = development_frame.loc[
            development_frame["segment_id"].eq(segment),
            [
                "trial_uid",
                "segment_id",
                "target_id",
                "true_x_mm",
                "true_y_mm",
                "true_z_mm",
            ],
        ].reset_index(drop=True)
        for name in ("pred_x_mm", "pred_y_mm", "pred_z_mm"):
            segment_frame[name] = 0.0
        for outage_index in range(4):
            for seed_index in range(3):
                snapshot_path = v4_experiment.atomic_write_parquet(
                    run
                    / "incumbent"
                    / "development_predictions"
                    / f"segment{segment}__drop{outage_index}__seed{seed_index}.parquet",
                    segment_frame,
                )
                snapshots.append(
                    {
                        "held_out_segment": segment,
                        "forced_outage_index": outage_index,
                        "paired_seed_index": seed_index,
                        "snapshot": v4_experiment._artifact_reference(
                            snapshot_path, run, rows=rows
                        ),
                    }
                )

    source_path = run / "provenance" / "source_snapshot.json"
    environment_path = run / "provenance" / "environment_snapshot.json"
    source = read_json_object(source_path)
    environment = read_json_object(environment_path)
    references = {
        "v3_config": v4_experiment._artifact_reference(copied_v3_config, run),
        "v3_development_manifest": v4_experiment._artifact_reference(
            copied_v3_manifest, run
        ),
        "v3_selection": v4_experiment._artifact_reference(copied_selection, run),
        "v3_validator_payload": v4_experiment._artifact_reference(
            copied_validator, run
        ),
        "v3_task_prediction_tree": v4_experiment._artifact_reference(
            copied_tree, run, tasks=240
        ),
        "development_replay_metadata": v4_experiment._artifact_reference(
            development_metadata_path, run
        ),
        "challenge_identity_metadata": v4_experiment._artifact_reference(
            challenge_metadata_path, run
        ),
        "source_snapshot": v4_experiment._artifact_reference(source_path, run),
        "environment_snapshot": v4_experiment._artifact_reference(
            environment_path, run
        ),
    }
    freeze = {
        "contract": v4_experiment.GBDT_V4_FREEZE_CONTRACT,
        "status": "prepared_from_validated_sealed_v3_development",
        "v4_config_sha256": gbdt_v4_config_sha256(config),
        "v3_origin_run": "synthetic-sealed-v3",
        "v3_resolved_config_sha256": v3_config_sha,
        "v3_development_manifest_sha256": sha256_file(copied_v3_manifest),
        "v3_validator_payload_sha256": stable_hash(validator_payload),
        "v3_selection_sha256": sha256_file(copied_selection),
        "v3_selected_candidate_id": selection["selected_candidate_id"],
        "v3_240_task_prediction_tree_sha256": task_tree_sha,
        "training_source_fingerprint": source["fingerprint"],
        "environment_fingerprint": environment["fingerprint"],
        "v3_task_count": 240,
        "v3_challenge_labels_sealed_at_prepare": True,
        "v3_run_modified_by_prepare": False,
        "references": references,
        "incumbent_seed_prediction_count": 48,
        "incumbent_seed_predictions": snapshots,
    }
    freeze_path = atomic_write_json(run / "incumbent.freeze.json", freeze)
    atomic_write_json(
        run / "_state" / "run_identity.json",
        {
            "contract": v4_experiment.GBDT_V4_CONTRACT,
            "config_sha256": gbdt_v4_config_sha256(config),
            "v3_config_sha256": v3_config_sha,
            "v3_development_manifest_sha256": freeze[
                "v3_development_manifest_sha256"
            ],
            "v3_selection_sha256": freeze["v3_selection_sha256"],
            "v3_task_prediction_tree_sha256": task_tree_sha,
            "training_source_fingerprint": source["fingerprint"],
            "environment_fingerprint": environment["fingerprint"],
        },
    )
    prepared = {
        "contract": v4_experiment.GBDT_V4_RUN_CONTRACT,
        "stage": "prepared_v3_incumbent_frozen_challenge_labels_sealed",
        "config_sha256": gbdt_v4_config_sha256(config),
        "incumbent_freeze": v4_experiment._artifact_reference(freeze_path, run),
        "challenge_labels_sealed": True,
        "files": v4_experiment._prepared_manifest_files(run),
    }
    atomic_write_json(run / "prepared_manifest.json", prepared)
    return environment_payload


def test_static_config_validates_without_guessing_v3_result_hashes() -> None:
    config = _config()
    assert len(gbdt_v4_config_sha256(config)) == 64
    assert config["contract"]["development_only"] is True
    assert config["contract"]["challenge_gate_owner"] == (
        "outage_v5_separate_sole_owner_contract"
    )
    assert config["execution"]["maximum_source_label_access_passes"] == 0
    assert config["final_refit"]["enabled"] is False
    assert config["challenge_report"]["enabled"] is False
    freeze = config["incumbent_freeze"]
    assert freeze["deferred_values_must_not_be_guessed_in_static_config"] is True
    assert freeze["bindings_materialized_after_v3_completion"] == [
        "v3_resolved_config_sha256",
        "v3_development_manifest_sha256",
        "v3_validator_payload_sha256",
        "v3_selection_sha256",
        "v3_selected_candidate_id",
        "v3_240_task_prediction_tree_sha256",
    ]
    assert not any(key.endswith("sha256") for key in freeze)


def test_config_matches_training_module_xgboost_recipe() -> None:
    xgb = _config()["xgboost"]
    assert xgb["num_boost_round"] == GBDT_V4_NUM_BOOST_ROUND
    assert xgb["axis_seed_recipe"] == GBDT_V4_AXIS_SEED_RECIPE
    for key, value in GBDT_V4_FIXED_XGB_PARAMS.items():
        assert xgb[key] == value
    assert xgb["early_stopping"] is False
    assert xgb["base_score"] == "exact_train_only_weighted_mean_per_axis"


def test_source_closure_and_environment_bind_xgboost_binary() -> None:
    source = gbdt_v4_source_payload(WORKSPACE)
    assert len(source["files"]) == len(GBDT_V4_SOURCE_FILES)
    assert [row["path"] for row in source["files"]] == list(GBDT_V4_SOURCE_FILES)
    assert "src/imu_vive_learning/gpu_outage_gbdt_v4_training.py" in GBDT_V4_SOURCE_FILES
    assert len(source["fingerprint"]) == 64
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        pytest.skip("live four-GPU identity is unavailable in this test sandbox")
    environment = gbdt_v4_environment_payload()
    assert environment["xgboost"] == "3.3.0"
    assert environment["xgboost_build_info"]["USE_CUDA"] is True
    assert environment["libxgboost"]["bytes"] > 0
    assert len(environment["libxgboost"]["sha256"]) == 64
    assert len(environment["fixed_xgb_parameter_sha256"]) == 64
    assert len(environment["cuda_devices"]) == 4


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("data_roles", "challenge_availability"), [1, 1, 1, 1]),
        (("selection", "score_name"), "plain_mean_rmse"),
        (("xgboost", "num_boost_round"), 999),
        (("execution", "maximum_source_label_access_passes"), 2),
    ],
)
def test_config_rejects_semantic_drift(path: tuple[str, str], value: object) -> None:
    config = deepcopy(_config())
    config[path[0]][path[1]] = value
    with pytest.raises(GPUOutageGBDTV4ConfigError):
        validate_gbdt_v4_config(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("contract", "confirmatory_claim_allowed", 0),
        ("xgboost", "min_child_weight", 5),
        ("execution", "gpu_count", 4.0),
    ],
)
def test_config_rejects_equal_but_wrong_scalar_types(
    section: str, key: str, value: object
) -> None:
    config = deepcopy(_config())
    config[section][key] = value
    with pytest.raises(GPUOutageGBDTV4ConfigError):
        validate_gbdt_v4_config(config)


def test_config_rejects_nested_extra_and_duplicate_yaml_key(tmp_path: Path) -> None:
    config = deepcopy(_config())
    config["xgboost"]["undeclared"] = 1
    with pytest.raises(GPUOutageGBDTV4ConfigError):
        validate_gbdt_v4_config(config)

    config = deepcopy(_config())
    config["_undeclared_runtime_override"] = "not-hash-covered"
    with pytest.raises(GPUOutageGBDTV4ConfigError, match="internal config keys"):
        validate_gbdt_v4_config(config)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        CONFIG_PATH.read_text(encoding="utf-8")
        + "\ncontract:\n  name: silently_overwrite_frozen_contract\n",
        encoding="utf-8",
    )
    with pytest.raises(GPUOutageGBDTV4ConfigError, match="duplicate"):
        load_gbdt_v4_config(duplicate)


def test_development_grid_has_twelve_fits_not_forty_eight() -> None:
    tasks = gbdt_v4_development_tasks(_config())
    assert len(tasks) == 12
    assert len({(task["held_out_segment"], task["paired_seed_index"]) for task in tasks}) == 12
    assert all(task["forced_outage_evaluations"] == [0, 1, 2, 3] for task in tasks)
    assert all("forced_outage_index" not in task for task in tasks)


def test_local_v3_grid_matches_authoritative_v3_development_grid() -> None:
    v3_config = load_outage_gpu_config(V3_CONFIG_PATH)
    assert _v3_development_tasks(v3_config) == v3_development_tasks(v3_config)


def test_parent_and_worker_gpu_uuid_pci_mapping_is_one_to_one() -> None:
    devices = _fake_cuda_devices()
    assert _validate_parent_cuda_device_grid(devices) == devices
    runtime = {
        "physical_gpu_id": 2,
        "visible_device": "cuda:0",
        "visible_cuda_device_uuid": devices[2]["uuid"],
        "visible_cuda_pci_bus_id": devices[2]["pci_bus_id"],
        "visible_cuda_name": devices[2]["name"],
    }
    _validate_worker_cuda_mapping(runtime, {"payload": {"cuda_devices": devices}})

    wrong = deepcopy(runtime)
    wrong["visible_cuda_pci_bus_id"] = devices[1]["pci_bus_id"]
    with pytest.raises(GPUOutageGBDTV4Error, match="does not map"):
        _validate_worker_cuda_mapping(
            wrong, {"payload": {"cuda_devices": devices}}
        )
    duplicate = deepcopy(devices)
    duplicate[3]["uuid"] = duplicate[0]["uuid"]
    with pytest.raises(GPUOutageGBDTV4Error, match="one-to-one"):
        _validate_parent_cuda_device_grid(duplicate)

    for invalid in (True, 0.0, "0"):
        invalid_devices = deepcopy(devices)
        invalid_devices[0]["physical_gpu_id"] = invalid
        with pytest.raises(GPUOutageGBDTV4Error, match="exact integer"):
            _validate_parent_cuda_device_grid(invalid_devices)


def test_requested_gpu_ids_reject_bool_float_and_string_coercion() -> None:
    for invalid in (True, 0.0, "0"):
        with pytest.raises(GPUOutageGBDTV4Error, match="exact integer"):
            v4_experiment._validate_gpu_ids([invalid, 1, 2, 3])


def test_closed_world_requires_exact_12_bundle_and_48_context_trees(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    config = _config()
    development = run / "development"
    (development / "xgb_bundles").mkdir(parents=True)
    for task in gbdt_v4_development_tasks(config):
        task_dir = development / "xgb_bundles" / task["task_id"]
        (task_dir / "bundle").mkdir(parents=True)
        for name in (
            "task_execution.json",
            "preprocessing.json",
            "result.json",
            *(f"predictions.drop{index}.parquet" for index in range(4)),
        ):
            (task_dir / name).touch()
    (development / "contexts").mkdir()
    for candidate in GBDT_V4_CANDIDATE_IDS:
        candidate_dir = development / "contexts" / candidate
        candidate_dir.mkdir()
        for segment in (36, 37, 38, 39):
            for outage in range(4):
                (candidate_dir / f"segment{segment}__drop{outage}.parquet").touch()
    (development / "context_metrics.json").touch()
    (development / "selection.json").touch()
    v4_experiment._assert_development_closed_world(run, config)

    extra = development / "xgb_bundles" / "xgb__segment36__seed0" / "extra.bin"
    extra.touch()
    with pytest.raises(GPUOutageGBDTV4Error, match="members changed"):
        v4_experiment._assert_development_closed_world(run, config)


def test_prepared_manifest_ignores_only_development_worker_ready_ledgers(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    atomic_write_json(run / "config.resolved.json", {"frozen": True})
    atomic_write_json(run / "provenance" / "source_snapshot.json", {"frozen": True})
    atomic_write_json(
        run / "provenance" / "worker_ready" / "development" / "gpu0.json",
        {"late_resume_ledger": True},
    )
    paths = {row["path"] for row in _prepared_manifest_files(run)}
    assert "config.resolved.json" in paths
    assert "provenance/source_snapshot.json" in paths
    assert not any(
        path.startswith("provenance/worker_ready/development/") for path in paths
    )


def test_real_prepared_validator_accepts_declared_late_development_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "prepared"
    _write_synthetic_prepared_run(run, monkeypatch)
    first = v4_experiment.validate_gbdt_v4_prepared_run(run)
    assert first["stage"] == "prepared_v3_incumbent_frozen_challenge_labels_sealed"

    atomic_write_json(
        run / "provenance" / "worker_ready" / "development" / "gpu0.json",
        {"late_worker_ledger": True},
    )
    atomic_write_json(
        run / "recovery" / "development" / "late_attempt.audit.json",
        {"late_recovery_evidence": True},
    )
    atomic_write_json(
        run / "development_manifest.json",
        {"stage": "synthetic_late_development_manifest"},
    )
    second = v4_experiment.validate_gbdt_v4_prepared_run(run)
    assert second == first
    frozen_paths = {
        row["path"]
        for row in read_json_object(run / "prepared_manifest.json")["files"]
    }
    assert "development_manifest.json" not in frozen_paths
    assert not any(path.startswith("recovery/") for path in frozen_paths)
    assert not any(
        path.startswith("provenance/worker_ready/development/")
        for path in frozen_paths
    )


def test_artifact_references_reject_extra_fields_and_scalar_coercion(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "prediction.parquet"
    artifact.write_bytes(b"exact-reference-bytes")
    reference = v4_experiment._artifact_reference(
        artifact, tmp_path, rows=7
    )
    assert (
        v4_experiment._verify_reference(
            tmp_path, reference, expected_rows=7
        )
        == artifact
    )
    with pytest.raises(GPUOutageGBDTV4Error, match="schema"):
        v4_experiment._verify_reference(tmp_path, reference)
    for field, invalid in (
        ("path", True),
        ("bytes", True),
        ("bytes", float(reference["bytes"])),
        ("sha256", "A" + str(reference["sha256"])[1:]),
        ("rows", "7"),
        ("rows", 7.0),
        ("rows", True),
    ):
        malformed = deepcopy(reference)
        malformed[field] = invalid
        with pytest.raises(GPUOutageGBDTV4Error):
            v4_experiment._verify_reference(
                tmp_path, malformed, expected_rows=7
            )
    malformed = {**reference, "undeclared": 1}
    with pytest.raises(GPUOutageGBDTV4Error, match="schema"):
        v4_experiment._verify_reference(
            tmp_path, malformed, expected_rows=7
        )
    task_reference = v4_experiment._artifact_reference(
        artifact, tmp_path, tasks=240
    )
    assert (
        v4_experiment._verify_reference(
            tmp_path, task_reference, expected_tasks=240
        )
        == artifact
    )
    for invalid in (True, 240.0, "240"):
        malformed_tasks = {**task_reference, "tasks": invalid}
        with pytest.raises(GPUOutageGBDTV4Error, match="exact integer"):
            v4_experiment._verify_reference(
                tmp_path, malformed_tasks, expected_tasks=240
            )


def test_pool_enqueues_sentinels_only_after_all_results_authenticate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = [
        {"task_id": "slow", "held_out_segment": 36},
        {"task_id": "fast", "held_out_segment": 37},
    ]
    authenticated: set[str] = set()
    queue_writes: list[object] = []

    class FakeTaskQueue:
        def put(self, value: object) -> None:
            if value is None:
                assert authenticated == {"slow", "fast"}
            queue_writes.append(value)

    class FakeResultQueue:
        def __init__(self) -> None:
            self.events = [
                {"event": "task_complete", "gpu_id": 1, "task_id": "fast"},
                {"event": "task_complete", "gpu_id": 0, "task_id": "slow"},
            ]

        def get(self, timeout: float) -> dict:
            del timeout
            event = self.events.pop(0)
            authenticated.add(str(event["task_id"]))
            assert None not in queue_writes
            return event

    class FakeProcess:
        exitcode = 0

        def __init__(self, *, name: str, **_: object) -> None:
            self.name = name
            self.alive = True

        def start(self) -> None:
            return None

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.alive = False

        def kill(self) -> None:
            self.alive = False

        def join(self, timeout: float) -> None:
            del timeout
            self.alive = False

    class FakeContext:
        def __init__(self) -> None:
            self.queue_index = 0

        def Queue(self) -> object:
            self.queue_index += 1
            return FakeTaskQueue() if self.queue_index == 1 else FakeResultQueue()

        def Process(self, **kwargs: object) -> FakeProcess:
            return FakeProcess(**kwargs)

    monkeypatch.setattr(v4_experiment, "gbdt_v4_development_tasks", lambda _: tasks)
    monkeypatch.setattr(v4_experiment, "_resume_xgb_development_task", lambda *_: None)
    monkeypatch.setattr(
        v4_experiment,
        "_load_valid_xgb_result",
        lambda _run, task: (
            {"task_id": task["task_id"]}
            if str(task["task_id"]) in authenticated
            else None
        ),
    )
    monkeypatch.setattr(v4_experiment.mp, "get_context", lambda _: FakeContext())
    monkeypatch.setattr(
        v4_experiment, "_materialize_worker_aggregate_ledger", lambda *args, **kwargs: {}
    )
    results = _run_xgb_development_pool(
        config={}, run_dir=tmp_path / "run", gpu_ids=[0, 1, 2, 3]
    )
    assert [row["task_id"] for row in results] == ["slow", "fast"]
    assert queue_writes[:2] == tasks
    assert queue_writes[2:] == [None, None, None, None]


def test_score_uses_exact_a_ws_wm_formula() -> None:
    rows = _context_rows()
    for row in rows:
        if row["candidate_id"] == "v3_incumbent":
            row["metrics"]["rmse_3d_mm"] = (
                10.0
                + (row["held_out_segment"] - 36) * 2.0
                + row["forced_outage_index"]
            )
    result = score_gbdt_v4_candidates(rows, _config())
    incumbent = next(row for row in result["trace"] if row["candidate_id"] == "v3_incumbent")
    values = [10.0 + segment_offset * 2.0 + outage for segment_offset in range(4) for outage in range(4)]
    expected_a = sum(values) / 16.0
    expected_ws = sum(16.0 + outage for outage in range(4)) / 4.0
    expected_wm = sum(10.0 + segment_offset * 2.0 + 3.0 for segment_offset in range(4)) / 4.0
    assert incumbent["mean_rmse_over_16_contexts_mm"] == pytest.approx(expected_a)
    assert incumbent["worst_segment_mean_rmse_mm"] == pytest.approx(expected_ws)
    assert incumbent["worst_sensor_mean_rmse_mm"] == pytest.approx(expected_wm)
    assert incumbent["outage_robust_j_mm"] == pytest.approx(
        (expected_a + expected_ws + expected_wm) / 3.0
    )


def test_score_ties_use_p95_then_listed_order() -> None:
    tied = {candidate: 10.0 for candidate in GBDT_V4_CANDIDATE_IDS}
    p95 = {
        "v3_incumbent": 12.0,
        "xgboost_mask_replicated": 11.0,
        "equal_blend_incumbent_xgboost": 11.0,
    }
    result = score_gbdt_v4_candidates(
        _context_rows(rmse_by_candidate=tied, p95_by_candidate=p95), _config()
    )
    assert result["selected_candidate_id"] == "xgboost_mask_replicated"


def test_score_missing_context_fails_closed_and_nonfinite_is_infinite() -> None:
    rows = _context_rows()
    with pytest.raises(GPUOutageGBDTV4Error, match="exactly 48"):
        score_gbdt_v4_candidates(rows[:-1], _config())
    rows[0]["metrics"]["rmse_3d_mm"] = math.nan
    result = score_gbdt_v4_candidates(rows, _config())
    incumbent = next(row for row in result["trace"] if row["candidate_id"] == "v3_incumbent")
    assert incumbent["score_status"] == "positive_infinity"
    assert incumbent["missing_contexts"] == []
    for invalid in (True, "1.0"):
        malformed = _context_rows()
        malformed[0]["metrics"]["rmse_3d_mm"] = invalid
        with pytest.raises(GPUOutageGBDTV4Error, match="JSON real number"):
            score_gbdt_v4_candidates(malformed, _config())


def test_context_replay_binds_full_identity_truth_rows_and_predictions() -> None:
    expected = pd.DataFrame(
        {
            "trial_uid": ["trial_a", "trial_b"],
            "segment_id": [36, 36],
            "target_id": ["T00", "T01"],
            "true_x_mm": [1.0, 2.0],
            "true_y_mm": [3.0, 4.0],
            "true_z_mm": [5.0, 6.0],
            "pred_x_mm": [7.0, 8.0],
            "pred_y_mm": [9.0, 10.0],
            "pred_z_mm": [11.0, 12.0],
        }
    )
    ensemble = expected.loc[
        :, ["pred_x_mm", "pred_y_mm", "pred_z_mm"]
    ].to_numpy(dtype=np.float64)
    v4_experiment._validate_context_replay_frame(
        expected.copy(), expected, ensemble, expected_rows=2
    )
    for field, value, message in (
        ("trial_uid", "different_trial", "identity/truth"),
        ("target_id", "different_target", "identity/truth"),
        ("true_x_mm", 999.0, "identity/truth"),
        ("pred_x_mm", 999.0, "prediction"),
    ):
        malformed = expected.copy()
        malformed.loc[0, field] = value
        with pytest.raises(GPUOutageGBDTV4Error, match=message):
            v4_experiment._validate_context_replay_frame(
                malformed, expected, ensemble, expected_rows=2
            )
    with pytest.raises(GPUOutageGBDTV4Error, match="identity/truth"):
        v4_experiment._validate_context_replay_frame(
            expected.iloc[:1].copy(), expected, ensemble, expected_rows=2
        )


def test_v4_score_matches_v3_score_semantics() -> None:
    v4_rows = _context_rows(
        rmse_by_candidate={
            "v3_incumbent": 13.0,
            "xgboost_mask_replicated": 12.0,
            "equal_blend_incumbent_xgboost": 11.0,
        }
    )
    v4 = score_gbdt_v4_candidates(v4_rows, _config())
    v3_config = load_outage_gpu_config(V3_CONFIG_PATH)
    v3_ids = [candidate["candidate_id"] for candidate in v3_config["candidates"]]
    v3_rows: list[dict] = []
    for priority, candidate in enumerate(v3_ids):
        source_id = GBDT_V4_CANDIDATE_IDS[priority] if priority < 3 else None
        source = next(
            row for row in v4_rows if row["candidate_id"] == source_id
        ) if source_id is not None else None
        for segment in (36, 37, 38, 39):
            for outage in range(4):
                if source_id is None:
                    rmse, p95, dispersion = 100.0 + priority, 110.0 + priority, 1.0
                else:
                    row = next(
                        item
                        for item in v4_rows
                        if item["candidate_id"] == source_id
                        and item["held_out_segment"] == segment
                        and item["forced_outage_index"] == outage
                    )
                    rmse = row["metrics"]["rmse_3d_mm"]
                    p95 = row["metrics"]["p95_3d_error_mm"]
                    dispersion = row["seed_rmse_population_std_mm"]
                v3_rows.append(
                    {
                        "candidate_id": candidate,
                        "held_out_segment": segment,
                        "forced_outage_index": outage,
                        "metrics": {"rmse_3d_mm": rmse, "p95_3d_error_mm": p95},
                        "seed_rmse_population_std_mm": dispersion,
                    }
                )
    v3 = score_outage_candidates(v3_rows, v3_config)
    for index in range(3):
        v4_trace = v4["trace"][index]
        v3_trace = v3["trace"][index]
        for key in (
            "mean_rmse_over_16_contexts_mm",
            "worst_segment_mean_rmse_mm",
            "worst_sensor_mean_rmse_mm",
            "outage_robust_j_mm",
            "mean_p95_3d_mm_over_16_contexts",
            "mean_seed_rmse_population_std_mm_over_16_contexts",
        ):
            assert v4_trace[key] == pytest.approx(v3_trace[key])


def test_task_execution_receipt_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "development" / "task_execution.json"
    _exclusive_write_json(path, {"status": "worker_claimed"})
    with pytest.raises(GPUOutageGBDTV4Error, match="already claimed"):
        _exclusive_write_json(path, {"status": "second"})
    assert "worker_claimed" in path.read_text(encoding="utf-8")


def test_prepare_refuses_open_v3_seal_before_creating_output(tmp_path: Path) -> None:
    v3 = tmp_path / "v3"
    marker = v3 / "challenge" / "label_access_1.started.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "v4"
    with pytest.raises(GPUOutageGBDTV4Error, match="sealed development-only"):
        prepare_gbdt_v4_run(_config(), v3_run_dir=v3, run_dir=output)
    assert not output.exists()


def test_prepare_paths_reject_equality_ancestors_and_symlink_aliases(
    tmp_path: Path,
) -> None:
    v3 = tmp_path / "v3"
    v3.mkdir()
    v4 = tmp_path / "v4"
    assert _validate_prepare_run_paths(v3, v4) == (v3.resolve(), v4.resolve())
    for output in (v3, v3 / "child", tmp_path):
        with pytest.raises(GPUOutageGBDTV4Error, match="non-ancestor"):
            _validate_prepare_run_paths(v3, output)
    alias = tmp_path / "v3_alias"
    alias.symlink_to(v3, target_is_directory=True)
    with pytest.raises(GPUOutageGBDTV4Error, match="symlink aliases"):
        _validate_prepare_run_paths(alias, v4)
    lexical_alias = tmp_path / "lexical_alias"
    lexical_alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(GPUOutageGBDTV4Error, match="symlink aliases"):
        _validate_prepare_run_paths(lexical_alias / ".." / "v3", v4)


def test_prepare_stability_detects_concurrent_marker_and_manifest_drift(
    tmp_path: Path,
) -> None:
    v3 = tmp_path / "v3"
    atomic_write_json(
        v3 / "development_manifest.json",
        {"stage": "development_complete_challenge_labels_sealed"},
    )
    atomic_write_json(v3 / "development" / "selection.json", {"winner": "a"})
    before = v4_experiment.manifest_files(v3)

    atomic_write_json(
        v3 / "challenge" / "label_access_1.started.json", {"started": True}
    )
    with pytest.raises(GPUOutageGBDTV4Error, match="marker appeared"):
        v4_experiment._assert_v3_prepare_stability(v3, before)
    (v3 / "challenge" / "label_access_1.started.json").unlink()

    atomic_write_json(
        v3 / "development_manifest.json",
        {"stage": "concurrently_changed"},
    )
    with pytest.raises(GPUOutageGBDTV4Error, match="run changed"):
        v4_experiment._assert_v3_prepare_stability(v3, before)


def test_prepare_invalid_marker_is_o_excl_and_all_validators_reject_it(
    tmp_path: Path,
) -> None:
    v3 = tmp_path / "v3"
    atomic_write_json(v3 / "development_manifest.json", {"sealed": True})
    before = v4_experiment.manifest_files(v3)
    run = tmp_path / "v4"
    atomic_write_json(run / "incumbent.freeze.json", {"partially": "published"})
    atomic_write_json(run / "prepared_manifest.json", {"partially": "published"})
    marker = v4_experiment._claim_prepare_invalid(
        v3_run=v3,
        v4_run=run,
        before_manifest=before,
        stage="after_partial_publication",
        reason="first immutable reason",
    )
    first_bytes = marker.read_bytes()
    v4_experiment._claim_prepare_invalid(
        v3_run=v3,
        v4_run=run,
        before_manifest=before,
        stage="after_partial_publication",
        reason="must not overwrite",
    )
    assert marker.read_bytes() == first_bytes
    for validator in (
        v4_experiment.validate_gbdt_v4_prepared_run,
        v4_experiment.validate_gbdt_v4_development_run,
    ):
        with pytest.raises(GPUOutageGBDTV4Error, match="prepare_invalid"):
            validator(run)


def test_v3_sealed_validator_runs_only_in_subprocess_and_requires_finite_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "status": "ok",
        "stage": "development_complete_challenge_labels_sealed",
        "challenge_labels_sealed": True,
        "development_seed_fits": 240,
        "scored_contexts": 96,
    }
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": __import__("json").dumps(expected)},
        )()

    monkeypatch.setattr(v4_experiment.subprocess, "run", fake_run)
    assert _validate_sealed_v3_development(tmp_path / "v3") == expected
    assert calls[0][3] == "validate-gpu-outage-experiment"

    def nonfinite_run(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return type("Completed", (), {"returncode": 0, "stdout": '{"x":NaN}'})()

    monkeypatch.setattr(v4_experiment.subprocess, "run", nonfinite_run)
    with pytest.raises(GPUOutageGBDTV4Error, match="finite JSON"):
        _validate_sealed_v3_development(tmp_path / "v3")


def test_clean_v4_import_does_not_load_v3_label_loader_module() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import imu_vive_learning.gpu_outage_gbdt_v4_experiment; "
                "import imu_vive_learning.gpu_outage_gbdt_v4_training; "
                "assert 'imu_vive_learning.gpu_outage_data' not in sys.modules"
            ),
        ],
        cwd=WORKSPACE,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )
    assert completed.returncode == 0, completed.stderr


def test_prepare_requires_all_96_v3_scored_contexts() -> None:
    with pytest.raises(GPUOutageGBDTV4Error, match="sealed stage"):
        _require_sealed_v3_development_validation(
            {
                "status": "ok",
                "stage": "development_complete_challenge_labels_sealed",
                "challenge_labels_sealed": True,
                "development_seed_fits": 240,
                "scored_contexts": 95,
            }
        )


def test_run_requires_explicit_prepare_then_resume(tmp_path: Path) -> None:
    with pytest.raises(GPUOutageGBDTV4Error, match="prepare command"):
        run_gbdt_v4_experiment(_config(), tmp_path / "run", resume=False)


def test_incomplete_task_is_quarantined_without_deletion(tmp_path: Path) -> None:
    run = tmp_path / "run"
    task_id = "xgb__segment36__seed0"
    task_dir = run / "development" / "xgb_bundles" / task_id
    task_dir.mkdir(parents=True)
    partial = task_dir / "axis_x.partial"
    partial.write_bytes(b"partial-checkpoint-evidence")
    audit = _quarantine_task_directory(
        run_dir=run,
        task_dir=task_dir,
        phase="development",
        task_id=task_id,
        reason="synthetic_interrupted_write",
    )
    assert audit is not None
    assert not task_dir.exists()
    quarantined = run / audit["quarantined_path"]
    assert (quarantined / partial.name).read_bytes() == b"partial-checkpoint-evidence"
    validated = _validate_quarantine_phase(
        run, phase="development", allowed_task_ids=[task_id]
    )
    assert validated == [audit]

    (quarantined / partial.name).write_bytes(b"tampered")
    with pytest.raises(GPUOutageGBDTV4Error, match="audit changed"):
        _validate_quarantine_phase(
            run, phase="development", allowed_task_ids=[task_id]
        )


def test_orphaned_quarantine_move_recovers_audit_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    task_id = "xgb__segment36__seed0"
    task_dir = run / "development" / "xgb_bundles" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "partial.ubj").write_bytes(b"durable-before-crash")
    real_atomic_write_json = v4_experiment.atomic_write_json

    def crash_before_audit(path: Path, payload: dict) -> Path:
        if str(path).endswith(".audit.json"):
            raise RuntimeError("synthetic crash after move")
        return real_atomic_write_json(path, payload)

    monkeypatch.setattr(v4_experiment, "atomic_write_json", crash_before_audit)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        _quarantine_task_directory(
            run_dir=run,
            task_dir=task_dir,
            phase="development",
            task_id=task_id,
            reason="synthetic_move_audit_window",
        )
    recovery = run / "recovery" / "development"
    assert (recovery / f"{task_id}__attempt0001").is_dir()
    assert (recovery / f"{task_id}__attempt0001.intent.json").is_file()
    assert not (recovery / f"{task_id}__attempt0001.audit.json").exists()

    monkeypatch.setattr(v4_experiment, "atomic_write_json", real_atomic_write_json)
    assert (
        v4_experiment._resume_xgb_development_task(run, {"task_id": task_id})
        is None
    )
    assert not (recovery / f"{task_id}__attempt0001.intent.json").exists()
    assert (recovery / f"{task_id}__attempt0001.audit.json").is_file()
    assert (
        recovery / f"{task_id}__attempt0001" / "partial.ubj"
    ).read_bytes() == b"durable-before-crash"


def test_missing_aggregate_worker_ledger_is_recovered_from_durable_workers(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    environment = {"payload": {"cuda_devices": _fake_cuda_devices()}}
    atomic_write_json(run / "provenance" / "environment_snapshot.json", environment)
    for gpu_id, device in enumerate(_fake_cuda_devices()):
        runtime = {
            "physical_gpu_id": gpu_id,
            "visible_device": "cuda:0",
            "visible_cuda_device_uuid": device["uuid"],
            "visible_cuda_pci_bus_id": device["pci_bus_id"],
            "visible_cuda_name": device["name"],
            "xgboost_version": "3.3.0",
            "xgboost_use_cuda": True,
        }
        atomic_write_json(
            run / "provenance" / "worker_ready" / "development" / f"gpu{gpu_id}.json",
            {"phase": "development", "runtime": runtime},
        )
    aggregate = _validate_aggregate_worker_ledger(
        run,
        phase="development",
        kind="xgb",
        output_name="development_gpu_workers.json",
    )
    assert aggregate["worker_count"] == 4
    assert [row["physical_gpu_id"] for row in aggregate["workers"]] == [0, 1, 2, 3]
    assert (
        read_json_object(run / "provenance" / "development_gpu_workers.json")
        == aggregate
    )
    tampered = deepcopy(aggregate)
    tampered["worker_count"] = 4.0
    atomic_write_json(run / "provenance" / "development_gpu_workers.json", tampered)
    with pytest.raises(GPUOutageGBDTV4Error, match="worker_count"):
        _validate_aggregate_worker_ledger(
            run,
            phase="development",
            kind="xgb",
            output_name="development_gpu_workers.json",
        )


def test_worker_invariant_environment_tamper_fails_with_isolated_gpu_grid(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _write_synthetic_provenance(run)
    v4_experiment._validate_v4_source_provenance(
        run, validate_environment=False
    )
    snapshot_path = run / "provenance" / "environment_snapshot.json"
    snapshot = read_json_object(snapshot_path)
    snapshot["unexpected"] = "not-hash-covered"
    atomic_write_json(snapshot_path, snapshot)
    with pytest.raises(GPUOutageGBDTV4Error, match="snapshot self-hash"):
        v4_experiment._validate_v4_source_provenance(
            run, validate_environment=False
        )
    snapshot.pop("unexpected")
    snapshot["payload"]["numpy"] = "tampered-version"
    snapshot["fingerprint"] = stable_hash(snapshot["payload"])
    atomic_write_json(snapshot_path, snapshot)
    with pytest.raises(GPUOutageGBDTV4Error, match="invariant environment"):
        v4_experiment._validate_v4_source_provenance(
            run, validate_environment=False
        )


def test_complete_bundle_missing_result_is_reconstructed_without_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    config = _config()
    atomic_write_json(run / "config.resolved.json", public_gbdt_v4_config(config))
    _write_synthetic_provenance(run)
    data = _tiny_development_data()
    task = gbdt_v4_development_tasks(config)[0]
    monkeypatch.setitem(v4_experiment._DEVELOPMENT_SEGMENT_ROWS, 36, 3)
    monkeypatch.setattr(v4_experiment, "_load_development_replay", lambda _: data)
    result = _execute_xgb_development_task(
        data,
        task,
        config,
        run,
        gpu_id=0,
        allow_cpu_for_testing=True,
    )
    for field, value in (
        ("fit_count", True),
        ("evaluation_count", 4.0),
        ("gpu_id", "0"),
    ):
        malformed = deepcopy(result)
        malformed.pop("result_file")
        malformed[field] = value
        with pytest.raises(GPUOutageGBDTV4Error, match="exact integer"):
            v4_experiment._validate_xgb_result_shape(
                run_dir=run,
                result=malformed,
                config=config,
                task=task,
                phase="development",
            )
    assert v4_experiment._load_valid_xgb_result(run, task) is None
    bundle_dir = run / result["artifacts"]["bundle"]["path"]
    cpu_bundle = v4_training.load_xgb_bundle(
        bundle_dir,
        expected_held_out_segment=int(task["held_out_segment"]),
        expected_paired_seed=int(task["model_seed"]),
    )
    synthetic_cuda_bundle = replace(cpu_bundle, training_device="cuda:0")
    result_path = run / "development" / "xgb_bundles" / task["task_id"] / "result.json"
    result_path.unlink()
    with pytest.raises(GPUOutageGBDTV4Error, match="isolated cuda:0"):
        _reconstruct_xgb_development_result(run, task)
    preprocessor = v4_experiment._preprocessor_from_metadata(
        read_json_object(run / result["artifacts"]["preprocessing"]["path"])
    )
    scaled_predictions: dict[int, np.ndarray] = {}
    for outage_index in range(4):
        prediction = pd.read_parquet(
            run
            / "development"
            / "xgb_bundles"
            / task["task_id"]
            / f"predictions.drop{outage_index}.parquet"
        ).loc[:, ["pred_x_mm", "pred_y_mm", "pred_z_mm"]]
        scaled_predictions[outage_index] = preprocessor.transform_targets(
            prediction.to_numpy(dtype=np.float64)
        )
    model_audit = {
        key: value
        for key, value in result["bundle_save_audit"].items()
        if key not in {"checkpoint_directory", "metadata_sha256"}
    }

    monkeypatch.setattr(
        v4_training, "load_xgb_bundle", lambda *args, **kwargs: synthetic_cuda_bundle
    )
    monkeypatch.setattr(
        v4_training, "validate_xgb_training_replay", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        v4_training,
        "predict_xgb_bundle_forced_outage",
        lambda bundle, features, outage_index, *, device: (
            scaled_predictions[int(outage_index)]
            if bundle is synthetic_cuda_bundle and device == "cuda:0"
            else (_ for _ in ()).throw(AssertionError("unexpected synthetic replay"))
        ),
    )
    monkeypatch.setattr(
        v4_training, "validate_xgb_bundle_models", lambda bundle: model_audit
    )
    before = {
        path.name: (path.stat().st_mtime_ns, sha256_file(path))
        for path in bundle_dir.iterdir()
    }

    recovered = _reconstruct_xgb_development_result(run, task)
    assert recovered is not None
    assert recovered["result_recovery"] == {
        "mode": "authenticated_complete_artifacts_without_refit"
    }
    after = {
        path.name: (path.stat().st_mtime_ns, sha256_file(path))
        for path in bundle_dir.iterdir()
    }
    assert after == before

    raw = read_json_object(result_path)
    raw["undeclared"] = True
    atomic_write_json(result_path, raw)
    assert v4_experiment._load_valid_xgb_result(run, task) is None


def test_synthetic_development_lifecycle_stops_sealed_and_rejects_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "synthetic_development"
    config = _config()
    atomic_write_json(run / "config.resolved.json", public_gbdt_v4_config(config))
    selection = {
        "selected_candidate_id": "xgboost_mask_replicated",
        "selected_outage_robust_j_mm": 9.75,
        "challenge_used_for_selection": False,
    }
    monkeypatch.setattr(v4_experiment, "_validate_requested_gpu_ids", lambda *args: None)
    monkeypatch.setattr(
        v4_experiment,
        "validate_gbdt_v4_prepared_run",
        lambda *args, **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        v4_experiment,
        "validate_gbdt_v4_development_run",
        lambda *args, **kwargs: {"status": "ok"},
    )
    monkeypatch.setattr(
        v4_experiment,
        "_run_xgb_development_pool",
        lambda **kwargs: [{"synthetic_bundle": index} for index in range(12)],
    )
    monkeypatch.setattr(
        v4_experiment,
        "_validate_v4_source_provenance",
        lambda *args, **kwargs: (
            {"fingerprint": "d" * 64},
            {"fingerprint": "e" * 64},
        ),
    )

    def fake_aggregate(*args: object, **kwargs: object) -> tuple[list, dict]:
        atomic_write_json(
            run / "development" / "context_metrics.json",
            {"synthetic_contexts": 48},
        )
        atomic_write_json(run / "development" / "selection.json", selection)
        return [], selection

    monkeypatch.setattr(
        v4_experiment, "aggregate_gbdt_v4_development", fake_aggregate
    )
    development = run_gbdt_v4_experiment(
        config, run, gpu_ids=[0, 1, 2, 3], resume=True
    )
    assert development["stage"] == "development_complete_challenge_labels_sealed"
    assert development["challenge_labels_sealed"] is True
    assert development["counts"]["challenge_label_access_passes"] == 0
    assert not (run / "final").exists()
    assert not (run / "manifest.json").exists()
    assert not (run / "challenge" / "label_access_1.started.json").exists()
    before = {
        path.relative_to(run).as_posix(): sha256_file(path)
        for path in run.rglob("*")
        if path.is_file()
    }
    with pytest.raises(GPUOutageGBDTV4Error, match="development-only"):
        run_gbdt_v4_experiment(
            config,
            run,
            gpu_ids=[0, 1, 2, 3],
            resume=True,
            finalize_challenge=True,
        )
    after = {
        path.relative_to(run).as_posix(): sha256_file(path)
        for path in run.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_cli_and_module_expose_no_v4_challenge_label_path() -> None:
    parser = v4_experiment._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--config",
                str(CONFIG_PATH),
                "--output",
                "unused",
                "--resume",
                "--finalize-challenge",
            ]
        )
    module_source = Path(v4_experiment.__file__).read_text(encoding="utf-8")
    assert "load_challenge_labels_once" not in module_source
    assert "_open_challenge_once" not in module_source
    assert "from .gpu_outage_experiment import" not in module_source
    assert not hasattr(v4_experiment, "load_challenge_labels_once")
    assert not hasattr(v4_experiment, "_open_challenge_once")
    assert not hasattr(v4_experiment, "validate_outage_development_run")
    assert not hasattr(v4_experiment, "outage_development_tasks")


def test_run_rejects_any_challenge_directory_before_other_run_reads(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    (run / "challenge").mkdir(parents=True)
    (run / "challenge" / "undeclared.bin").write_bytes(b"must-not-be-consumed")
    with pytest.raises(GPUOutageGBDTV4Error, match="forbidden artifacts"):
        run_gbdt_v4_experiment(
            _config(), run, gpu_ids=[0, 1, 2, 3], resume=True
        )
