from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import imu_vive_learning.openaxes_external_model_v1_experiment as experiment_module

from imu_vive_learning.openaxes_external_model_v1_config import (
    OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
    OPENAXES_MODEL_METHOD_IDS,
    OPENAXES_MODEL_SEEDS,
    OPENAXES_MODEL_SHIFT_MS,
    load_openaxes_external_model_config,
)
from imu_vive_learning.openaxes_external_model_v1_experiment import (
    MODEL_COMPLETED_TASK_RESULT_MEMBERS,
    MODEL_SOURCE_CLOSURE,
    OpenAxesExternalModelV1Error,
    TaskSpec,
    _canonical_gpu_uuid,
    _closed_frame_sha256,
    _canonical_sha256,
    _context_metrics,
    _build_sensitivity_family_summaries,
    _build_loso_directional_gate,
    _execution_binding,
    _execution_launch_semantic_id,
    _execution_session_semantic_id,
    _freeze_or_load_execution_session,
    _freeze_or_load_prelaunch_session_recovery_receipt,
    _freeze_or_validate_full_signed_source_replay_receipt,
    _parent_source_authority_receipt,
    _prepare_synthetic_openaxes_external_model_run_for_test,
    _start_execution_launch,
    _strict_json_load,
    _task_attempt_semantic_id,
    _validate_execution_evidence_sets,
    _validate_child_visible_gpu,
    _validate_completed_task_result_member_set,
    _validate_latest_execution_launch_terminal,
    _validate_prepared_model_input_structure,
    _validate_execution_session,
    _validate_quarantine_evidence,
    _validate_signed_shift_reconstruction,
    _verify_execution_binding,
    _verify_reference,
    _verify_alignment_sensitivity_semantics,
    _verify_complete_semantic_hashes,
    _write_task_attempt_ledger,
    _write_worker_ready_receipt,
    _write_phase,
    aggregate_alignment_sensitivity,
    assign_tasks_to_gpu_slots,
    build_model_task_grid,
    classify_task_resume_state,
    doctor_openaxes_external_model_v1,
    make_synthetic_prepared_model_input,
    prepare_openaxes_external_model_run,
    quarantine_task_state_nondelete,
    run_spawn_process_probe,
    validate_gpu_inventory,
    validate_openaxes_external_model_development,
    validate_openaxes_external_model_prepared_run,
)


def _folds() -> list[dict]:
    scenarios = [
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    ]
    result: list[dict] = []
    for scenario in scenarios:
        result.append(
            {
                "fold_id": f"scenario_loso::{scenario}",
                "family": "scenario_loso",
                "role": "primary",
                "train_scenarios": [item for item in scenarios if item != scenario],
                "test_scenarios": [scenario],
                "inner_selection": "scenario_blocked_training_groups_only",
                "outer_test_fitting_allowed": False,
            }
        )
    for speed in (100, 150, 200, 250):
        test = [f"circle-{speed}", f"triangle-{speed}"]
        result.append(
            {
                "fold_id": f"speed_pair_out::{speed}",
                "family": "speed_pair_out",
                "role": "stress",
                "train_scenarios": [item for item in scenarios if item not in test],
                "test_scenarios": test,
                "inner_selection": "scenario_blocked_training_groups_only",
                "outer_test_fitting_allowed": False,
            }
        )
    for shape in ("circle", "triangle"):
        test = [item for item in scenarios if item.startswith(shape)]
        result.append(
            {
                "fold_id": f"shape_out::{shape}",
                "family": "shape_out",
                "role": "stress",
                "train_scenarios": [item for item in scenarios if item not in test],
                "test_scenarios": test,
                "inner_selection": "scenario_blocked_training_groups_only",
                "outer_test_fitting_allowed": False,
            }
        )
    return result


def _prepared() -> tuple[pd.DataFrame, dict, dict]:
    sensor_groups = {
        sensor: [f"{sensor}_channel_{index}" for index in range(9)]
        for sensor in ("humerus", "radius", "carpus", "digitus")
    }
    features = [column for columns in sensor_groups.values() for column in columns]
    scenarios = [
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    ]
    rows = len(scenarios) * 2
    frame = pd.DataFrame(
        {
            "row_uid": [f"uid-{index}" for index in range(rows)],
            "scenario_id": [scenario for scenario in scenarios for _ in range(2)],
            "sample_index": np.tile(np.arange(2, dtype=np.int64), len(scenarios)),
            "aligned_time_us": np.tile(np.asarray([0, 10_000], dtype=np.int64), len(scenarios)),
            **{
                column: np.linspace(index, index + 1, rows, dtype=np.float64)
                for index, column in enumerate(features)
            },
            "tcp_x_m": np.linspace(0.0, 0.1, rows, dtype=np.float64),
            "tcp_y_m": np.linspace(0.2, 0.3, rows, dtype=np.float64),
            "tcp_z_m": np.linspace(0.4, 0.5, rows, dtype=np.float64),
        }
    )
    manifest = {
        "contract": "openaxes.external.preparation.v1",
        "schema": {
            "metadata_columns_not_model_inputs": [
                "row_uid",
                "scenario_id",
                "sample_index",
                "aligned_time_us",
            ],
            "model_feature_columns": features,
            "model_feature_count": 36,
            "sensor_groups": sensor_groups,
            "target_columns": ["tcp_x_m", "tcp_y_m", "tcp_z_m"],
            "target_frame": "UR5_robot_base_TCP",
            "target_unit": "m",
            "forbidden_model_inputs": ["scenario_id", "tcp_x_m", "tcp_y_m", "tcp_z_m"],
            "raw_quaternion_columns_excluded_all_zero": [],
        },
        "preprocessing": {
            "sensitivity_recipe": {"signed_target_shifts_ms": list(OPENAXES_MODEL_SHIFT_MS)}
        },
        "folds": {
            "sensitivity_may_select_model_or_decide_claim_gate": False,
            "items": _folds(),
        },
        "limitations": {
            "observed_real_sensor_outage": False,
            "simulated_sensor_dropout_only": True,
        },
        "counts": {"prepared_active_rows": rows},
        "provenance": {
            "stage": {
                "contract_sources_committed_at_git_head": False,
                "downstream_model_results_blocked_by_uncommitted_contract": True,
            }
        },
    }
    validation = {
        "status": "ok",
        "contract": "openaxes.external.preparation.v1",
        "config_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "prepared_rows": rows,
        "scenario_rows": {scenario: 2 for scenario in scenarios},
        "feature_count": 36,
        "declared_outer_folds": 14,
        "aligned_robot_rows_unmatched": 492,
        "signed_sensitivity_shifts_ms": list(OPENAXES_MODEL_SHIFT_MS),
        "downstream_model_results_blocked_by_uncommitted_contract": True,
        "finite": True,
        "strict_100_hz_per_scenario": True,
        "semantic_replay": True,
    }
    return frame, manifest, validation


def _gpus() -> list[dict]:
    return [
        {
            "physical_index": index,
            "uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
            "pci_bus_id": f"00000000:{index + 1:02x}:00.0",
            "name": "Synthetic GPU",
        }
        for index in range(4)
    ]


def _synthetic_session_only_run(
    tmp_path: Path, name: str
) -> tuple[dict, Path, dict, tuple[TaskSpec, ...], dict]:
    frame, preparation_manifest, preparation_validation = _prepared()
    prepared = make_synthetic_prepared_model_input(
        frame, preparation_manifest, preparation_validation
    )
    config = load_openaxes_external_model_config(
        "configs/openaxes_external_model_v1.yaml"
    )
    root = tmp_path / name
    _prepare_synthetic_openaxes_external_model_run_for_test(
        config, prepared, root, _gpus()
    )
    manifest = _strict_json_load(root / "prepared_manifest.json")
    tasks = build_model_task_grid(preparation_manifest)
    pre_fit_validation = validate_openaxes_external_model_prepared_run(
        config, root, allow_synthetic_test_only=True
    )
    session = _freeze_or_load_execution_session(
        root,
        manifest,
        tasks,
        workers_per_gpu=2,
        resume=False,
        pre_fit_validation=pre_fit_validation,
    )
    return config, root, manifest, tasks, session


def test_prepared_input_builds_exact_336_closed_world_tasks() -> None:
    frame, manifest, validation = _prepared()
    prepared = make_synthetic_prepared_model_input(frame, manifest, validation)
    _validate_prepared_model_input_structure(prepared, production=False)
    tasks = build_model_task_grid(prepared.manifest)
    assert len(tasks) == 336
    assert sum(task.role == "primary" for task in tasks) == 192
    assert sum(task.role == "stress" for task in tasks) == 144
    assert {task.method_id for task in tasks} == set(OPENAXES_MODEL_METHOD_IDS)
    assert {task.seed for task in tasks} == set(OPENAXES_MODEL_SEEDS)
    assert len({task.spec_sha256 for task in tasks}) == 336


def test_synthetic_preparation_is_explicit_and_production_rejects_it() -> None:
    frame, manifest, validation = _prepared()
    prepared = make_synthetic_prepared_model_input(frame, manifest, validation)
    with pytest.raises(OpenAxesExternalModelV1Error, match="synthetic"):
        _validate_prepared_model_input_structure(prepared, production=True)
    attacked = deepcopy(validation)
    attacked["feature_count"] = 36.0
    with pytest.raises(OpenAxesExternalModelV1Error, match="integer"):
        make_synthetic_prepared_model_input(frame, manifest, attacked)


def test_fold_role_partition_and_feature_schema_attacks_fail_closed() -> None:
    frame, manifest, validation = _prepared()
    canonical_manifest = json.loads(json.dumps(manifest, sort_keys=True))
    assert list(canonical_manifest["schema"]["sensor_groups"]) != list(
        manifest["schema"]["sensor_groups"]
    )
    canonical_prepared = make_synthetic_prepared_model_input(
        frame, canonical_manifest, validation
    )
    _validate_prepared_model_input_structure(canonical_prepared, production=False)

    attacked = deepcopy(manifest)
    attacked["folds"]["items"][0]["role"] = "stress"
    with pytest.raises(OpenAxesExternalModelV1Error, match="family/role"):
        make_synthetic_prepared_model_input(frame, attacked, validation)
    attacked = deepcopy(manifest)
    attacked["schema"]["target_columns"][0] = "normalized_tcp_x_m"
    with pytest.raises(OpenAxesExternalModelV1Error, match="schema"):
        make_synthetic_prepared_model_input(frame, attacked, validation)
    attacked_frame = frame.copy()
    attacked_frame["shape"] = "circle"
    with pytest.raises(OpenAxesExternalModelV1Error, match="columns"):
        make_synthetic_prepared_model_input(attacked_frame, manifest, validation)

    attacked = deepcopy(manifest)
    del attacked["schema"]["sensor_groups"]["humerus"]
    with pytest.raises(OpenAxesExternalModelV1Error, match="4x9 sensor grouping"):
        make_synthetic_prepared_model_input(frame, attacked, validation)
    attacked = deepcopy(manifest)
    attacked["schema"]["sensor_groups"]["extra"] = [
        f"extra_channel_{index}" for index in range(9)
    ]
    with pytest.raises(OpenAxesExternalModelV1Error, match="4x9 sensor grouping"):
        make_synthetic_prepared_model_input(frame, attacked, validation)
    attacked = deepcopy(manifest)
    attacked["schema"]["sensor_groups"]["humerus"][0] = attacked["schema"][
        "sensor_groups"
    ]["radius"][0]
    with pytest.raises(OpenAxesExternalModelV1Error, match="feature order"):
        make_synthetic_prepared_model_input(frame, attacked, validation)
    attacked = deepcopy(manifest)
    attacked["schema"]["model_feature_columns"][0:2] = reversed(
        attacked["schema"]["model_feature_columns"][0:2]
    )
    with pytest.raises(OpenAxesExternalModelV1Error, match="feature order"):
        make_synthetic_prepared_model_input(frame, attacked, validation)


def test_gpu_inventory_and_slot_assignment_are_exact_and_nonduplicating() -> None:
    gpus = _gpus()
    assert len(validate_gpu_inventory(gpus)) == 4
    tasks = build_model_task_grid(_prepared()[1])
    for workers in (1, 2, 3):
        slots = assign_tasks_to_gpu_slots(tasks, gpus, workers)
        assert len(slots) == 4 * workers
        flattened = [task_id for slot in slots for task_id in slot["task_ids"]]
        assert len(flattened) == len(set(flattened)) == 336
    with pytest.raises(OpenAxesExternalModelV1Error, match="1..3"):
        assign_tasks_to_gpu_slots(tasks, gpus, True)
    attacked = deepcopy(gpus)
    attacked[0]["physical_index"] = "0"
    with pytest.raises(OpenAxesExternalModelV1Error, match="integer"):
        validate_gpu_inventory(attacked)
    attacked = deepcopy(gpus)
    attacked[0]["uuid"] = "GPU-not-a-canonical-uuid"
    with pytest.raises(OpenAxesExternalModelV1Error, match="GPU UUID"):
        validate_gpu_inventory(attacked)


@pytest.mark.parametrize(
    "property_uuid",
    (
        "5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb",
        "GPU-5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb",
        "gpu-5EBDC8BE-B2F3-B6E2-4DA1-AF79EF91F7CB",
    ),
)
def test_child_visible_gpu_accepts_only_equivalent_canonical_uuid_representations(
    property_uuid: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen_gpu = {
        "physical_index": 0,
        "uuid": "GPU-5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb",
        "pci_bus_id": "00000000:01:00.0",
        "name": "Synthetic GPU",
    }
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _index: SimpleNamespace(uuid=property_uuid),
        get_device_name=lambda _index: frozen_gpu["name"],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", frozen_gpu["uuid"])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(
        experiment_module,
        "_query_physical_gpu_rows",
        lambda: [
            {
                "index": "0",
                "uuid": frozen_gpu["uuid"],
                "pci_bus_id": frozen_gpu["pci_bus_id"],
                "name": frozen_gpu["name"],
            }
        ],
    )

    _validate_child_visible_gpu(frozen_gpu)
    assert _canonical_gpu_uuid(property_uuid, name="test GPU UUID") == frozen_gpu["uuid"]


@pytest.mark.parametrize(
    "property_uuid",
    (
        "5ebdc8be-b2f3-b6e2-4da1-af79ef91f7c",
        "GPU-5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb ",
        "MIG-5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb",
        "not-a-uuid",
    ),
)
def test_child_visible_gpu_rejects_malformed_uuid_representations(
    property_uuid: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen_gpu = {
        "physical_index": 0,
        "uuid": "GPU-5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb",
        "pci_bus_id": "00000000:01:00.0",
        "name": "Synthetic GPU",
    }
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _index: SimpleNamespace(uuid=property_uuid),
        get_device_name=lambda _index: frozen_gpu["name"],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", frozen_gpu["uuid"])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    with pytest.raises(OpenAxesExternalModelV1Error, match="visible CUDA UUID"):
        _validate_child_visible_gpu(frozen_gpu)


def test_child_visible_gpu_rejects_a_different_well_formed_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_gpu = {
        "physical_index": 0,
        "uuid": "GPU-5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb",
        "pci_bus_id": "00000000:01:00.0",
        "name": "Synthetic GPU",
    }
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _index: SimpleNamespace(
            uuid="5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cc"
        ),
        get_device_name=lambda _index: frozen_gpu["name"],
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", frozen_gpu["uuid"])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    with pytest.raises(OpenAxesExternalModelV1Error, match="visible CUDA UUID changed"):
        _validate_child_visible_gpu(frozen_gpu)


def test_synthetic_prepare_marks_artifact_and_normal_validator_rejects(
    tmp_path: Path,
) -> None:
    frame, manifest, validation = _prepared()
    prepared = make_synthetic_prepared_model_input(frame, manifest, validation)
    config = load_openaxes_external_model_config(
        "configs/openaxes_external_model_v1.yaml"
    )
    run = tmp_path / "model-run"
    result = _prepare_synthetic_openaxes_external_model_run_for_test(
        config,
        prepared,
        run,
        _gpus(),
    )
    assert result["synthetic_test_only"] is True
    validation_result = validate_openaxes_external_model_prepared_run(
        config, run, allow_synthetic_test_only=True
    )
    assert validation_result["scientific_fits"] == 336
    assert validation_result["synthetic_test_only"] is True

    marker_path = run / "PREPARED.json"
    original_marker = _strict_json_load(marker_path)
    prepared_manifest = _strict_json_load(run / "prepared_manifest.json")
    for key, replacement in (
        ("contract", "attacker.contract"),
        ("config_sha256", "0" * 64),
    ):
        attacked_marker = deepcopy(original_marker)
        attacked_marker[key] = replacement
        marker_path.write_text(
            json.dumps(attacked_marker, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(OpenAxesExternalModelV1Error, match="prepared marker"):
            validate_openaxes_external_model_prepared_run(
                config, run, allow_synthetic_test_only=True
            )

    canonical_references = {
        "prepared_manifest": prepared_manifest["task_grid"]["task_index"],
        "task_index": original_marker["prepared_manifest"],
    }
    for key, alternate_valid_reference in canonical_references.items():
        attacked_marker = deepcopy(original_marker)
        attacked_marker[key] = alternate_valid_reference
        marker_path.write_text(
            json.dumps(attacked_marker, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(OpenAxesExternalModelV1Error, match="canonical artifact"):
            validate_openaxes_external_model_prepared_run(
                config, run, allow_synthetic_test_only=True
            )
    for key, attacked_path in (
        ("prepared_manifest", "./prepared_manifest.json"),
        ("task_index", "tasks//./index.json"),
    ):
        attacked_marker = deepcopy(original_marker)
        attacked_marker[key]["path"] = attacked_path
        marker_path.write_text(
            json.dumps(attacked_marker, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(OpenAxesExternalModelV1Error, match="canonical POSIX"):
            validate_openaxes_external_model_prepared_run(
                config, run, allow_synthetic_test_only=True
            )
    marker_path.write_text(
        json.dumps(original_marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = run / "prepared_manifest.json"
    for mutate_runtime in (
        lambda value: value.__setitem__("attacker", True),
        lambda value: value["python"].__setitem__("version", "forged"),
    ):
        attacked_manifest = deepcopy(prepared_manifest)
        mutate_runtime(attacked_manifest["runtime"])
        manifest_path.write_text(
            json.dumps(attacked_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        attacked_marker = deepcopy(original_marker)
        attacked_marker["prepared_manifest"] = {
            "path": "prepared_manifest.json",
            "bytes": manifest_path.stat().st_size,
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
        attacked_marker["prepared_runtime_sha256"] = _canonical_sha256(
            attacked_manifest["runtime"]
        )
        marker_path.write_text(
            json.dumps(attacked_marker, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(OpenAxesExternalModelV1Error, match="runtime|Python"):
            validate_openaxes_external_model_prepared_run(
                config, run, allow_synthetic_test_only=True
            )
    attacked_manifest = deepcopy(prepared_manifest)
    attacked_manifest["execution"]["prelaunch_session_recovery"][
        "required_launch_ordinal"
    ] = 1
    manifest_path.write_text(
        json.dumps(attacked_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    attacked_marker = deepcopy(original_marker)
    attacked_marker["prepared_manifest"] = {
        "path": "prepared_manifest.json",
        "bytes": manifest_path.stat().st_size,
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    marker_path.write_text(
        json.dumps(attacked_marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OpenAxesExternalModelV1Error, match="prelaunch recovery"):
        validate_openaxes_external_model_prepared_run(
            config, run, allow_synthetic_test_only=True
        )
    manifest_path.write_text(
        json.dumps(prepared_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    marker_path.write_text(
        json.dumps(original_marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    extra_task_member = run / "tasks" / "extra.json"
    extra_task_member.write_text("{}\n", encoding="utf-8")
    with pytest.raises(OpenAxesExternalModelV1Error, match="task-index directory"):
        validate_openaxes_external_model_prepared_run(
            config, run, allow_synthetic_test_only=True
        )
    extra_task_member.unlink()
    quarantine_extra = run / "quarantine" / "extra.bin"
    quarantine_extra.write_bytes(b"attack")
    with pytest.raises(OpenAxesExternalModelV1Error, match="quarantine"):
        validate_openaxes_external_model_prepared_run(
            config, run, allow_synthetic_test_only=True
        )
    quarantine_extra.unlink()
    quarantine_symlink = run / "quarantine" / f"event_{'f' * 64}"
    quarantine_symlink.symlink_to(run / "tasks", target_is_directory=True)
    with pytest.raises(OpenAxesExternalModelV1Error, match="quarantine"):
        validate_openaxes_external_model_prepared_run(
            config, run, allow_synthetic_test_only=True
        )
    quarantine_symlink.unlink()
    with pytest.raises(OpenAxesExternalModelV1Error, match="synthetic"):
        validate_openaxes_external_model_prepared_run(config, run)


def test_production_prepare_is_paths_only_and_rejects_forged_or_missing_authority(
    tmp_path: Path,
) -> None:
    frame, manifest, validation = _prepared()
    forged = make_synthetic_prepared_model_input(frame, manifest, validation)
    config = load_openaxes_external_model_config(
        "configs/openaxes_external_model_v1.yaml"
    )
    with pytest.raises(OpenAxesExternalModelV1Error, match="paths|PreparedModelInput"):
        prepare_openaxes_external_model_run(
            config,
            forged,
            tmp_path / "not-a-config.yaml",
            tmp_path / "not-a-preparation",
            tmp_path / "output",
            _gpus(),
        )
    with pytest.raises(OpenAxesExternalModelV1Error, match="missing"):
        prepare_openaxes_external_model_run(
            config,
            tmp_path / "missing-source.zip",
            tmp_path / "missing-config.yaml",
            tmp_path / "missing-preparation",
            tmp_path / "output",
            _gpus(),
        )


def test_same_process_pre_spawn_then_final_validation_allows_expected_torch_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    frame, preparation_manifest, preparation_validation = _prepared()
    prepared = make_synthetic_prepared_model_input(
        frame, preparation_manifest, preparation_validation
    )
    config = load_openaxes_external_model_config(
        "configs/openaxes_external_model_v1.yaml"
    )
    root = tmp_path / "same-process-runtime"
    _prepare_synthetic_openaxes_external_model_run_for_test(
        config, prepared, root, _gpus()
    )

    # Before a session exists, a newly imported controller torch module is a
    # launch blocker, exactly like the production run entrypoint.
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(OpenAxesExternalModelV1Error, match="pre-spawn torch-import"):
        validate_openaxes_external_model_prepared_run(
            config, root, allow_synthetic_test_only=True
        )
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    pre_spawn = validate_openaxes_external_model_prepared_run(
        config, root, allow_synthetic_test_only=True
    )
    manifest = _strict_json_load(root / "prepared_manifest.json")
    tasks = build_model_task_grid(manifest["preparation"])
    session = _freeze_or_load_execution_session(
        root,
        manifest,
        tasks,
        workers_per_gpu=2,
        resume=False,
        pre_fit_validation=pre_spawn,
    )

    # Final semantic verification may legitimately import torch in this same
    # controller. Stable environment fields and every recorded hash stay exact;
    # only the current import-state comparison is phase-aware.
    monkeypatch.setitem(sys.modules, "torch", object())
    assert validate_openaxes_external_model_prepared_run(
        config, root, allow_synthetic_test_only=True
    )["status"] == "ok"
    assert _validate_execution_session(
        root, manifest, tasks, session, runtime_phase="post_spawn"
    ) == session


def test_parent_and_later_model_git_heads_are_separate_with_exact_unchanged_blobs(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    workspace = repo / "09_SO101_IMU_VIVE_LEARNING"
    parent_relatives = (
        "configs/openaxes_external_v1.yaml",
        "docs/OPENAXES_EXTERNAL_V1_CONTRACT.md",
        "pyproject.toml",
        "src/imu_vive_learning/openaxes_external_v1.py",
        "src/imu_vive_learning/openaxes_external_v1_cli.py",
        "src/imu_vive_learning/openaxes_external_v1_config.py",
        "tests/test_openaxes_external_v1.py",
    )
    for index, relative in enumerate(parent_relatives):
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"parent-source-{index}\n".encode())
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Synthetic Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "parent"], check=True
    )
    parent_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    (repo / "model-only.txt").write_text("model\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "model-only.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "model"], check=True
    )
    model_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    recorded_files = {
        relative: {
            "bytes": (workspace / relative).stat().st_size,
            "sha256": hashlib.sha256((workspace / relative).read_bytes()).hexdigest(),
        }
        for relative in parent_relatives
    }
    canonical = lambda value: json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    repo_paths = {
        relative: f"09_SO101_IMU_VIVE_LEARNING/{relative}"
        for relative in parent_relatives
    }
    manifest = {
        "provenance": {
            "repository": {
                "git_head": parent_head,
                "contract_source_repo_paths": repo_paths,
            },
            "contract_and_lock_sources": {
                "files": recorded_files,
                "source_set_sha256": hashlib.sha256(
                    canonical(recorded_files)
                ).hexdigest(),
            },
        }
    }
    receipt = _parent_source_authority_receipt(
        preparation_manifest=manifest,
        preparation_config_path=workspace / parent_relatives[0],
    )
    assert receipt["parent_preparation_git_head"] == parent_head
    assert receipt["model_contract_git_head"] == model_head
    assert receipt["parent_sources_unchanged_at_model_head"] is True
    (workspace / parent_relatives[1]).write_text("working-tree attack\n", encoding="utf-8")
    with pytest.raises(OpenAxesExternalModelV1Error, match="Git boundaries"):
        _parent_source_authority_receipt(
            preparation_manifest=manifest,
            preparation_config_path=workspace / parent_relatives[0],
        )


def test_reference_validator_rejects_type_coercion_extra_fields_and_tamper(
    tmp_path: Path,
) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"abc")
    reference = {
        "path": "value.bin",
        "bytes": 3,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
    }
    assert _verify_reference(tmp_path, reference, name="value") == path
    for attack in (
        {**reference, "bytes": "3"},
        {**reference, "sha256": reference["sha256"].upper()},
        {**reference, "attacker": True},
    ):
        with pytest.raises(OpenAxesExternalModelV1Error):
            _verify_reference(tmp_path, attack, name="value")
    path.write_bytes(b"abd")
    with pytest.raises(OpenAxesExternalModelV1Error, match="changed"):
        _verify_reference(tmp_path, reference, name="value")


def _dummy_task() -> TaskSpec:
    payload = {
        "ordinal": 0,
        "method": "masked_erm",
        "fold": "scenario_loso::circle-100",
        "seed": 1701,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return TaskSpec(
        task_id=f"task_0000_{digest[:12]}",
        ordinal=0,
        method_id="masked_erm",
        fold_id="scenario_loso::circle-100",
        family="scenario_loso",
        role="primary",
        train_scenarios=(
            "circle-150",
            "circle-200",
            "circle-250",
            "triangle-100",
            "triangle-150",
            "triangle-200",
            "triangle-250",
        ),
        test_scenarios=("circle-100",),
        seed=1701,
        spec_sha256=digest,
    )


def test_phase_chain_resume_fails_closed_on_ambiguous_fit_and_outer_eval(
    tmp_path: Path,
) -> None:
    task = _dummy_task()
    attempt = "a" * 64
    assert classify_task_resume_state(tmp_path, task) == "fresh_fit"
    _write_phase(tmp_path, task, "execution.started", attempt_id=attempt, payload={})
    with pytest.raises(OpenAxesExternalModelV1Error, match="ambiguous"):
        classify_task_resume_state(tmp_path, task)
    _write_phase(tmp_path, task, "training.completed", attempt_id=attempt, payload={})
    assert classify_task_resume_state(tmp_path, task) == "predict_without_refit"
    _write_phase(tmp_path, task, "outer_prediction.completed", attempt_id=attempt, payload={})
    assert classify_task_resume_state(tmp_path, task) == "evaluate_once"
    _write_phase(tmp_path, task, "outer_eval.started", attempt_id=attempt, payload={})
    with pytest.raises(OpenAxesExternalModelV1Error, match="evaluation started"):
        classify_task_resume_state(tmp_path, task)
    _write_phase(tmp_path, task, "outer_eval.completed", attempt_id=attempt, payload={})
    assert classify_task_resume_state(tmp_path, task) == "bundle_without_reevaluation"


def test_quarantine_never_restores_fresh_fit_for_any_prefix_or_corrupt_state(
    tmp_path: Path,
) -> None:
    task = _dummy_task()
    attempt = "a" * 64
    phases = (
        "execution.started",
        "training.completed",
        "outer_prediction.completed",
        "outer_eval.started",
        "outer_eval.completed",
        "bundle.completed",
        "result.completed",
    )
    for prefix_length in range(1, len(phases) + 1):
        root = tmp_path / f"prefix-{prefix_length}"
        for phase in phases[:prefix_length]:
            _write_phase(root, task, phase, attempt_id=attempt, payload={})
        quarantine_task_state_nondelete(root, task, reason="synthetic prefix audit")
        with pytest.raises(OpenAxesExternalModelV1Error, match="persistent prior attempt"):
            classify_task_resume_state(root, task)

    corrupt_root = tmp_path / "corrupt-prefix"
    corrupt_ledger = corrupt_root / "runtime" / "ledgers" / task.task_id
    corrupt_ledger.mkdir(parents=True)
    (corrupt_ledger / "execution.started.json").write_text("{", encoding="utf-8")
    quarantine_task_state_nondelete(
        corrupt_root, task, reason="synthetic corrupt prefix audit"
    )
    _validate_quarantine_evidence(corrupt_root, (task,), require_empty=False)
    with pytest.raises(OpenAxesExternalModelV1Error, match="empty quarantine"):
        _validate_quarantine_evidence(corrupt_root, (task,), require_empty=True)
    event_dir = next((corrupt_root / "quarantine").iterdir())
    extra = event_dir / "extra.bin"
    extra.write_bytes(b"attack")
    with pytest.raises(OpenAxesExternalModelV1Error, match="quarantine event"):
        _validate_quarantine_evidence(corrupt_root, (task,), require_empty=False)
    extra.unlink()
    moved_result_symlink = event_dir / "result"
    moved_result_symlink.symlink_to(event_dir / "ledger", target_is_directory=True)
    with pytest.raises(OpenAxesExternalModelV1Error, match="quarantine event"):
        _validate_quarantine_evidence(corrupt_root, (task,), require_empty=False)
    moved_result_symlink.unlink()
    with pytest.raises(OpenAxesExternalModelV1Error, match="persistent prior attempt"):
        classify_task_resume_state(corrupt_root, task)


def test_completed_task_result_directory_is_exact_and_symlink_free(
    tmp_path: Path,
) -> None:
    task = _dummy_task()
    directory = tmp_path / "results" / task.task_id
    directory.mkdir(parents=True)
    for name in MODEL_COMPLETED_TASK_RESULT_MEMBERS:
        (directory / name).write_bytes(b"fixture")
    _validate_completed_task_result_member_set(tmp_path, task)

    extra = directory / "extra.bin"
    extra.write_bytes(b"attack")
    with pytest.raises(OpenAxesExternalModelV1Error, match="member set"):
        _validate_completed_task_result_member_set(tmp_path, task)
    extra.unlink()

    checkpoint = directory / "checkpoint.pt"
    checkpoint.unlink()
    checkpoint.symlink_to(directory / "history.json")
    with pytest.raises(OpenAxesExternalModelV1Error, match="member set"):
        _validate_completed_task_result_member_set(tmp_path, task)


def test_authenticated_session_only_resume_creates_exact_first_launch(
    tmp_path: Path,
) -> None:
    config, root, _manifest, _tasks, session = _synthetic_session_only_run(
        tmp_path, "session-only-recovery"
    )
    launches_root = root / "runtime" / "execution_launches"
    recovery_path = root / "runtime" / "prelaunch_session_recovery.json"
    assert not any(launches_root.iterdir())
    assert not recovery_path.exists()

    # Simulate a controller crash after the immutable session and recovery
    # authority are durable, but before the first launch directory is created.
    recovery = _freeze_or_load_prelaunch_session_recovery_receipt(root, session)
    assert recovery_path.is_file()
    assert recovery[
        "local_scientific_attempt_evidence_observed_at_exclusive_creation"
    ] is False
    assert recovery["authorized_action_scope"] == (
        "deterministic_authenticated_ordinal_zero_launch_creation_only"
    )
    assert recovery["recovery_receipt_is_attempt_authority"] is False
    assert recovery[
        "authorizes_scientific_refit_under_declared_non_total_erasure_threat_model"
    ] is False
    assert recovery["external_append_only_or_signature_authority_present"] is False
    assert recovery["total_erasure_prior_execution_provable"] is False
    assert "cannot_prove_whether_execution_previously_occurred" in recovery[
        "total_erasure_limitation"
    ]
    assert "authorizes_scientific_refit" not in recovery
    assert not any(launches_root.iterdir())
    assert validate_openaxes_external_model_prepared_run(
        config, root, allow_synthetic_test_only=True
    )["status"] == "ok"

    attacked_recovery = deepcopy(recovery)
    attacked_recovery["total_erasure_prior_execution_provable"] = True
    attacked_recovery["receipt_payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in attacked_recovery.items()
            if key != "receipt_payload_sha256"
        }
    )
    recovery_path.write_text(
        json.dumps(attacked_recovery, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OpenAxesExternalModelV1Error, match="receipt changed"):
        validate_openaxes_external_model_prepared_run(
            config, root, allow_synthetic_test_only=True
        )
    recovery_path.write_text(
        json.dumps(recovery, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(OpenAxesExternalModelV1Error, match="fresh first launch"):
        _start_execution_launch(root, session, resume=False)

    # A persisted receipt never masks subsequently observed partial evidence.
    partial = root / "runtime" / "attempt_roster" / "partial.json"
    partial.write_text("{}\n", encoding="utf-8")
    with pytest.raises(OpenAxesExternalModelV1Error, match="observed execution evidence"):
        _start_execution_launch(root, session, resume=True)
    partial.unlink()

    launch = _start_execution_launch(root, session, resume=True)
    assert launch["launch_ordinal"] == 0
    assert launch["resume"] is True
    assert launch["launch_mode"] == "authenticated_prelaunch_session_recovery"
    assert launch["previous_launch"] is None
    assert launch["prelaunch_session_recovery"]["path"] == (
        "runtime/prelaunch_session_recovery.json"
    )
    assert launch["prelaunch_session_recovery_sha256"] == _canonical_sha256(
        recovery
    )
    assert launch["launch_id"] == _execution_launch_semantic_id(
        session,
        launch_ordinal=0,
        previous_launch=None,
        launch_mode="authenticated_prelaunch_session_recovery",
    )
    assert validate_openaxes_external_model_prepared_run(
        config, root, allow_synthetic_test_only=True
    )["status"] == "ok"


def test_session_only_resume_rejects_every_partial_execution_authority(
    tmp_path: Path,
) -> None:
    _config, root, _manifest, tasks, session = _synthetic_session_only_run(
        tmp_path, "session-only-partial-evidence"
    )
    recovery_path = root / "runtime" / "prelaunch_session_recovery.json"

    forged_session = deepcopy(session)
    forged_session["controller_source_sha256"] = "0" * 64
    forged_session["session_payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in forged_session.items()
            if key != "session_payload_sha256"
        }
    )
    with pytest.raises(OpenAxesExternalModelV1Error, match="authenticated authority"):
        _start_execution_launch(root, forged_session, resume=True)
    assert not recovery_path.exists()

    task_id = tasks[0].task_id
    partial_paths = (
        (root / "runtime" / "execution_launches" / f"launch_{'a' * 64}", True),
        (root / "runtime" / "worker_ready" / "orphan", True),
        (root / "runtime" / "attempt_roster" / f"{task_id}.json", False),
        (root / "runtime" / "attempt_ledger" / f"{task_id}.json", False),
        (root / "runtime" / "ledgers" / task_id, True),
        (root / "results" / task_id, True),
        (root / "quarantine" / "event_orphan", True),
    )
    for path, is_directory in partial_paths:
        if is_directory:
            path.mkdir()
        else:
            path.write_text("{}\n", encoding="utf-8")
        with pytest.raises(OpenAxesExternalModelV1Error):
            _start_execution_launch(root, session, resume=True)
        assert not recovery_path.exists()
        if is_directory:
            path.rmdir()
        else:
            path.unlink()


def test_immutable_execution_session_ready_binding_and_resume_mapping(
    tmp_path: Path,
) -> None:
    frame, preparation_manifest, validation = _prepared()
    prepared = make_synthetic_prepared_model_input(
        frame, preparation_manifest, validation
    )
    config = load_openaxes_external_model_config(
        "configs/openaxes_external_model_v1.yaml"
    )
    root = tmp_path / "session-run"
    _prepare_synthetic_openaxes_external_model_run_for_test(
        config, prepared, root, _gpus()
    )
    manifest = _strict_json_load(root / "prepared_manifest.json")
    tasks = build_model_task_grid(preparation_manifest)
    pre_fit_validation = validate_openaxes_external_model_prepared_run(
        config, root, allow_synthetic_test_only=True
    )
    session = _freeze_or_load_execution_session(
        root,
        manifest,
        tasks,
        workers_per_gpu=2,
        resume=False,
        pre_fit_validation=pre_fit_validation,
    )
    assert session["workers_per_gpu"] == 2
    assert len(session["slots"]) == 8
    assert len(
        {
            task_id
            for slot in session["slots"]
            for task_id in slot["task_ids"]
        }
    ) == 336
    with pytest.raises(OpenAxesExternalModelV1Error, match="explicit resume"):
        _freeze_or_load_execution_session(
            root,
            manifest,
            tasks,
            workers_per_gpu=2,
            resume=False,
            pre_fit_validation=pre_fit_validation,
        )
    assert _freeze_or_load_execution_session(
        root,
        manifest,
        tasks,
        workers_per_gpu=2,
        resume=True,
        pre_fit_validation=pre_fit_validation,
    )["task_assignment_sha256"] == session["task_assignment_sha256"]
    with pytest.raises(OpenAxesExternalModelV1Error, match="workers/GPU"):
        _freeze_or_load_execution_session(
            root,
            manifest,
            tasks,
            workers_per_gpu=3,
            resume=True,
            pre_fit_validation=pre_fit_validation,
        )

    session_path = root / "runtime" / "execution_session.json"
    original_session = _strict_json_load(session_path)
    attacked_session = deepcopy(original_session)
    attacked_session["execution_session_id"] = "f" * 64
    attacked_session["session_payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in attacked_session.items()
            if key != "session_payload_sha256"
        }
    )
    with pytest.raises(OpenAxesExternalModelV1Error, match="semantic id"):
        _validate_execution_session(root, manifest, tasks, attacked_session)

    # Even a coherent rewrite of worker count, slots, assignment, semantic ID,
    # and session seal cannot override the source-closed prepared authority.
    attacked_session = deepcopy(original_session)
    attacked_slots = assign_tasks_to_gpu_slots(
        tasks, manifest["gpu_inventory"], workers_per_gpu=3
    )
    attacked_session["workers_per_gpu"] = 3
    attacked_session["slots"] = list(attacked_slots)
    attacked_session["task_assignment_sha256"] = _canonical_sha256(
        attacked_slots
    )
    attacked_session["execution_session_id"] = _execution_session_semantic_id(
        root,
        manifest,
        workers_per_gpu=3,
        slots=attacked_slots,
        pre_fit_receipt=_strict_json_load(
            root / "runtime" / "pre_fit_validation.json"
        ),
    )
    attacked_session["session_payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in attacked_session.items()
            if key != "session_payload_sha256"
        }
    )
    with pytest.raises(OpenAxesExternalModelV1Error, match="config-frozen"):
        _validate_execution_session(root, manifest, tasks, attacked_session)

    attacked_session = deepcopy(original_session)
    attacked_session["pre_fit_validation_receipt_sha256"] = "0" * 64
    attacked_session["session_payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in attacked_session.items()
            if key != "session_payload_sha256"
        }
    )
    session_path.write_text(
        json.dumps(attacked_session, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OpenAxesExternalModelV1Error, match="pre-fit prepared-run"):
        _validate_execution_session(root, manifest, tasks, attacked_session)
    session_path.write_text(
        json.dumps(original_session, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pre_fit_path = root / "runtime" / "pre_fit_validation.json"
    original_pre_fit = _strict_json_load(pre_fit_path)
    attacked_pre_fit = deepcopy(original_pre_fit)
    attacked_pre_fit["gpu_inventory_sha256"] = "f" * 64
    attacked_pre_fit["receipt_payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in attacked_pre_fit.items()
            if key != "receipt_payload_sha256"
        }
    )
    pre_fit_path.write_text(
        json.dumps(attacked_pre_fit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    attacked_session = deepcopy(original_session)
    attacked_session["pre_fit_validation"] = {
        "relative_path": "runtime/pre_fit_validation.json",
        "bytes": pre_fit_path.stat().st_size,
        "sha256": hashlib.sha256(pre_fit_path.read_bytes()).hexdigest(),
    }
    attacked_session["pre_fit_validation_receipt_sha256"] = _canonical_sha256(
        attacked_pre_fit
    )
    attacked_session["session_payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in attacked_session.items()
            if key != "session_payload_sha256"
        }
    )
    session_path.write_text(
        json.dumps(attacked_session, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OpenAxesExternalModelV1Error, match="pre-fit validation receipt changed"):
        _validate_execution_session(root, manifest, tasks, attacked_session)
    pre_fit_path.write_text(
        json.dumps(original_pre_fit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    session_path.write_text(
        json.dumps(original_session, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    launch = _start_execution_launch(root, session, resume=False)
    assert launch["launch_mode"] == "fresh_first_launch"
    assert launch["prelaunch_session_recovery"] is None
    assert launch["prelaunch_session_recovery_sha256"] is None
    assert launch["launch_id"] == _execution_launch_semantic_id(
        session, launch_ordinal=0, previous_launch=None
    )
    launch_directory = (
        root
        / "runtime"
        / "execution_launches"
        / f"launch_{launch['launch_id']}"
    )
    forged_launch_id = "e" * 64
    forged_directory = launch_directory.with_name(f"launch_{forged_launch_id}")
    launch_directory.rename(forged_directory)
    forged_launch = deepcopy(launch)
    forged_launch["launch_id"] = forged_launch_id
    forged_launch["launch_payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in forged_launch.items()
            if key != "launch_payload_sha256"
        }
    )
    (forged_directory / "launch.started.json").write_text(
        json.dumps(forged_launch, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OpenAxesExternalModelV1Error, match="semantic id"):
        _start_execution_launch(root, session, resume=True)
    (forged_directory / "launch.started.json").write_text(
        json.dumps(launch, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    forged_directory.rename(launch_directory)

    observations_path = launch_directory / "controller_ready_observations.json"
    observations_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(OpenAxesExternalModelV1Error, match="worker-ready directory"):
        validate_openaxes_external_model_prepared_run(
            config, root, allow_synthetic_test_only=True
        )
    empty_ready_directory = (
        root / "runtime" / "worker_ready" / f"launch_{launch['launch_id']}"
    )
    empty_ready_directory.mkdir()
    with pytest.raises(OpenAxesExternalModelV1Error, match="empty worker-ready"):
        validate_openaxes_external_model_prepared_run(
            config, root, allow_synthetic_test_only=True
        )
    empty_ready_directory.rmdir()
    observations_path.unlink()
    slot = session["slots"][0]
    ready = _write_worker_ready_receipt(root, session, launch, slot)
    assert ready["gpu"] == slot["gpu"]
    binding = _execution_binding(root, session, launch, slot)
    assert binding["controller_ready_observations"] is None
    assert binding["controller_observations_sha256"] is None
    task = next(item for item in tasks if item.task_id == slot["task_ids"][0])
    assert _verify_execution_binding(root, task, binding) == binding
    launch_path = (
        root
        / "runtime"
        / "execution_launches"
        / f"launch_{launch['launch_id']}"
        / "launch.started.json"
    )
    original_launch = _strict_json_load(launch_path)
    launch_attacks = (
        ("resume", "forged"),
        ("launch_mode", "authenticated_prelaunch_session_recovery"),
        ("prelaunch_session_recovery", {}),
        ("prelaunch_session_recovery_sha256", "0" * 64),
        ("workers_per_gpu", 99),
        ("slot_count", 999),
        ("task_assignment_sha256", "0" * 64),
        ("controller_source_sha256", "0" * 64),
        ("model_source_set_sha256", "0" * 64),
        ("parent_public_validation_sha256", "0" * 64),
        ("pre_fit_validation_receipt_sha256", "0" * 64),
        ("session", {}),
    )
    for key, replacement in launch_attacks:
        attacked_launch = deepcopy(original_launch)
        attacked_launch[key] = replacement
        attacked_launch["launch_payload_sha256"] = _canonical_sha256(
            {
                field: value
                for field, value in attacked_launch.items()
                if field != "launch_payload_sha256"
            }
        )
        launch_path.write_text(
            json.dumps(attacked_launch, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(OpenAxesExternalModelV1Error, match="launch|session"):
            _verify_execution_binding(root, task, binding)
        launch_path.write_text(
            json.dumps(original_launch, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    attacked_binding = deepcopy(binding)
    attacked_binding["launch_payload_sha256"] = "0" * 64
    with pytest.raises(OpenAxesExternalModelV1Error, match="execution binding"):
        _verify_execution_binding(root, task, attacked_binding)
    attempt = _task_attempt_semantic_id(task, binding)
    _write_task_attempt_ledger(
        root, task, attempt_id=attempt, binding=binding
    )
    phase = _write_phase(
        root,
        task,
        "execution.started",
        attempt_id=attempt,
        payload=binding,
    )
    assert phase["execution_binding"]["launch_payload_sha256"] == launch[
        "launch_payload_sha256"
    ]
    phase_path = (
        root
        / "runtime"
        / "ledgers"
        / task.task_id
        / "execution.started.json"
    )
    original_phase = _strict_json_load(phase_path)
    attacked_phase = deepcopy(original_phase)
    attacked_phase["execution_binding"]["launch_payload_sha256"] = "0" * 64
    phase_path.write_text(
        json.dumps(attacked_phase, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OpenAxesExternalModelV1Error, match="execution binding"):
        classify_task_resume_state(root, task)
    phase_path.write_text(
        json.dumps(original_phase, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    attempt_path = root / "runtime" / "attempt_ledger" / f"{task.task_id}.json"
    attempt_bytes = attempt_path.read_bytes()
    attempt_path.unlink()
    with pytest.raises(OpenAxesExternalModelV1Error, match="lacks.*attempt ledger"):
        validate_openaxes_external_model_prepared_run(
            config, root, allow_synthetic_test_only=True
        )
    attempt_path.write_bytes(attempt_bytes)
    with pytest.raises(OpenAxesExternalModelV1Error, match="ambiguous"):
        classify_task_resume_state(root, task)

    # The monotonic roster is durably created before the ordinary attempt
    # ledger.  A crash (or deletion) in that narrow window must never make a
    # second scientific fit look fresh.
    second_task = next(
        item for item in tasks if item.task_id == slot["task_ids"][1]
    )
    _write_task_attempt_ledger(
        root,
        second_task,
        attempt_id=_task_attempt_semantic_id(second_task, binding),
        binding=binding,
    )
    second_attempt_path = (
        root / "runtime" / "attempt_ledger" / f"{second_task.task_id}.json"
    )
    second_attempt_path.unlink()
    with pytest.raises(OpenAxesExternalModelV1Error, match="persistent prior attempt"):
        classify_task_resume_state(root, second_task)

    # Two ready receipts emitted by the same process cannot masquerade as two
    # independently spawned workers.
    _write_worker_ready_receipt(root, session, launch, session["slots"][1])
    with pytest.raises(OpenAxesExternalModelV1Error, match="identities are not unique"):
        _validate_execution_evidence_sets(root, session)

    resumed_launch = _start_execution_launch(root, session, resume=True)
    _write_worker_ready_receipt(root, session, resumed_launch, slot)
    resumed_binding = _execution_binding(root, session, resumed_launch, slot)
    with pytest.raises(OpenAxesExternalModelV1Error, match="terminal-complete"):
        _validate_latest_execution_launch_terminal(root, session)

    # A resumed launch may continue a task, but phase evidence cannot jump
    # from launch A to B and then back to A under a coherently resealed phase.
    _write_phase(
        root,
        task,
        "training.completed",
        attempt_id=attempt,
        payload={},
        execution_binding=resumed_binding,
    )
    _write_phase(
        root,
        task,
        "outer_prediction.completed",
        attempt_id=attempt,
        payload={},
        execution_binding=binding,
    )
    with pytest.raises(OpenAxesExternalModelV1Error, match="moved backwards"):
        classify_task_resume_state(root, task)


def test_spawn_processes_set_visibility_before_torch_and_isolate_rng_and_init() -> None:
    slots = [
        [
            {"item_id": "slow", "seed": 1701, "duration_ms": 20, "initial_state_probe": True},
            {"item_id": "after", "seed": 2603, "duration_ms": 0, "initial_state_probe": False},
        ],
        [
            {"item_id": "fast", "seed": 1701, "duration_ms": 0, "initial_state_probe": True},
        ],
    ]
    results = run_spawn_process_probe(slots, _private_test_only_mode=True)
    assert len(results) == 3
    assert all(item["synthetic_test_only"] is True for item in results)
    assert all(item["torch_imported_before_cuda_visible_devices"] is False for item in results)
    by_id = {item["item_id"]: item for item in results}
    assert by_id["slow"]["initial_state_sha256"] == by_id["fast"]["initial_state_sha256"]
    assert by_id["slow"]["rng_value"] == by_id["fast"]["rng_value"]
    slot_zero = [item["item_id"] for item in results if item["slot_index"] == 0]
    assert slot_zero == ["slow", "after"]
    with pytest.raises(OpenAxesExternalModelV1Error, match="private-test-only"):
        run_spawn_process_probe(slots, _private_test_only_mode=False)


def _ensemble_frames(tasks: tuple[TaskSpec, ...]) -> dict[tuple[str, str], pd.DataFrame]:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for task in tasks:
        if task.method_id not in {"masked_erm", "paired_pord"}:
            continue
        rows: list[dict] = []
        for scenario in task.test_scenarios:
            truth = np.array([0.1, 0.2, 0.3])
            for view_index, view_id in enumerate(
                ("full", "outage_humerus", "outage_radius", "outage_carpus", "outage_digitus")
            ):
                error = 0.001 * view_index + (0.0005 if task.method_id == "masked_erm" else 0.0)
                rows.append(
                    {
                        "row_uid": f"uid::{scenario}",
                        "scenario_id": scenario,
                        "sample_index": 0,
                        "aligned_time_us": 100_000,
                        "view_id": view_id,
                        "tcp_x_m": truth[0],
                        "tcp_y_m": truth[1],
                        "tcp_z_m": truth[2],
                        "prediction_x_m": truth[0] + error,
                        "prediction_y_m": truth[1],
                        "prediction_z_m": truth[2],
                        "seed_ensemble_size": 3,
                    }
                )
        frames[(task.method_id, task.fold_id)] = pd.DataFrame(rows)
    return frames


def _shift_truth() -> tuple[pd.DataFrame, dict, dict]:
    scenarios = [
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    ]
    rows: list[dict] = []
    for scenario in scenarios:
        for shift in OPENAXES_MODEL_SHIFT_MS:
            rows.append(
                {
                    "row_uid": f"uid::{scenario}",
                    "scenario_id": scenario,
                    "base_aligned_time_us": 100_000,
                    "shift_ms": shift,
                    "shifted_target_time_us": 100_000 + shift * 1000,
                    "shifted_tcp_x_m": 0.1 + shift * 1e-5,
                    "shifted_tcp_y_m": 0.2,
                    "shifted_tcp_z_m": 0.3,
                }
            )
    truth = pd.DataFrame(rows)
    receipt = {
        "contract": "openaxes.external.signed_shift_truth.v1",
        "preparation_contract": "openaxes.external.preparation.v1",
        "config_sha256": "c" * 64,
        "scope": {
            "read_only_truth_reconstruction": True,
            "features_included": False,
            "model_fit_performed": False,
            "metrics_computed": False,
            "selection_performed": False,
            "claim_gate_decided": False,
            "artifact_written": False,
            "semantic_validator_required_before_model_use": True,
        },
        "source": {},
        "prepared": {
            "base_rows": 8,
            "base_exact_source_replay": True,
            "stage": {"downstream_model_results_blocked_by_uncommitted_contract": False},
        },
        "output": {
            "closed_schema": list(truth.columns),
            "row_order": "scenario_then_signed_shift_then_base_time",
            "target_columns_only_with_pairing_metadata": True,
            "rows": len(truth),
            "rows_per_shift": {
                str(shift): int((truth["shift_ms"] == shift).sum())
                for shift in OPENAXES_MODEL_SHIFT_MS
            },
            "dropped_edge_rows_by_scenario_and_shift": {
                scenario: {
                    str(shift): 0 for shift in OPENAXES_MODEL_SHIFT_MS
                }
                for scenario in scenarios
            },
            "long_table_sha256": _closed_frame_sha256(
                truth,
                (
                    ("row_uid", "utf8"),
                    ("scenario_id", "utf8"),
                    ("base_aligned_time_us", "int64"),
                    ("shift_ms", "int64"),
                    ("shifted_target_time_us", "int64"),
                    ("shifted_tcp_x_m", "float64"),
                    ("shifted_tcp_y_m", "float64"),
                    ("shifted_tcp_z_m", "float64"),
                ),
            ),
        },
        "recipe": {},
        "deterministic_receipt": True,
    }
    prep_manifest = {"counts": {"prepared_active_rows": 8}}
    return truth, receipt, prep_manifest


def _reseal_sensitivity(value: dict) -> None:
    records = value["fold_records"]
    value["fold_record_identity_sha256"] = _canonical_sha256(
        [
            {
                "method_id": record["method_id"],
                "fold_id": record["fold_id"],
                "family": record["family"],
                "shift_ms": record["shift_ms"],
                "base_uid_rows": record["base_uid_rows"],
                "base_uids": record["base_uids"],
                "base_uid_sha256": record["base_uid_sha256"],
                "outage_context_count": record["outage_context_count"],
            }
            for record in records
        ]
    )
    value["family_summaries"] = _build_sensitivity_family_summaries(records)
    value["family_summary_sha256"] = _canonical_sha256(
        value["family_summaries"]
    )
    value["sensitivity_semantic_sha256"] = _canonical_sha256(
        {
            key: item
            for key, item in value.items()
            if key != "sensitivity_semantic_sha256"
        }
    )


def test_context_score_and_signed_sensitivity_are_report_only_and_same_uid() -> None:
    manifest = _prepared()[1]
    tasks = build_model_task_grid(manifest)
    one = _ensemble_frames(tasks)[("paired_pord", "scenario_loso::circle-100")]
    metrics = _context_metrics(one, include_full=False)
    assert metrics["context_count"] == 4
    truth, receipt, prep_manifest = _shift_truth()
    _validate_signed_shift_reconstruction(truth, receipt, prep_manifest)
    dropped_attack = deepcopy(receipt)
    dropped_attack["output"]["dropped_edge_rows_by_scenario_and_shift"][
        "circle-100"
    ]["-50"] = 1
    with pytest.raises(OpenAxesExternalModelV1Error, match="dropped edge"):
        _validate_signed_shift_reconstruction(
            truth, dropped_attack, prep_manifest
        )
    sensitivity = aggregate_alignment_sensitivity(
        _ensemble_frames(tasks), tasks, truth, receipt
    )
    assert sensitivity["selection_use_allowed"] is False
    assert sensitivity["claim_gate_use_allowed"] is False
    assert len(sensitivity["fold_records"]) == 2 * 14 * 6
    assert sensitivity["shifted_truth_persisted_or_redistributed"] is False
    _verify_alignment_sensitivity_semantics(
        sensitivity, tasks, _ensemble_frames(tasks)
    )
    for mutator, match in (
        (
            lambda value: value["fold_records"][0].__setitem__(
                "family", "shape_out"
            ),
            "family",
        ),
        (
            lambda value: value["fold_records"][0].__setitem__(
                "delta_rmse3d_m_shift_minus_zero", 0.0
            ),
            "delta",
        ),
        (
            lambda value: value["family_summaries"][0].__setitem__(
                "folds", 999
            ),
            "summary",
        ),
        (
            lambda value: value.__setitem__(
                "fold_record_identity_sha256", "0" * 64
            ),
            "identity",
        ),
    ):
        tampered = deepcopy(sensitivity)
        mutator(tampered)
        with pytest.raises(OpenAxesExternalModelV1Error, match=match):
            _verify_alignment_sensitivity_semantics(
                tampered, tasks, _ensemble_frames(tasks)
            )
    forged_support = deepcopy(sensitivity)
    forged_support["fold_records"][0]["base_uid_rows"] = 999999
    forged_support["fold_records"][0]["base_uid_sha256"] = "f" * 64
    _reseal_sensitivity(forged_support)
    with pytest.raises(OpenAxesExternalModelV1Error, match="UID support"):
        _verify_alignment_sensitivity_semantics(
            forged_support, tasks, _ensemble_frames(tasks)
        )

    # Recompute every report-level seal while replacing both methods with the
    # same invented UID.  Cross-method agreement is not source authority.
    invented_support = deepcopy(sensitivity)
    target = invented_support["fold_records"][0]
    for record in invented_support["fold_records"]:
        if (
            record["fold_id"] == target["fold_id"]
            and record["shift_ms"] == target["shift_ms"]
        ):
            record["base_uids"] = ["uid::invented"]
            record["base_uid_rows"] = 1
            record["base_uid_sha256"] = _canonical_sha256(
                record["base_uids"]
            )
    _reseal_sensitivity(invented_support)
    with pytest.raises(OpenAxesExternalModelV1Error, match="UID support"):
        _verify_alignment_sensitivity_semantics(
            invented_support, tasks, _ensemble_frames(tasks)
        )

    # A valid strict subset is equally invalid, even when both methods and all
    # summaries are coherently resealed around that subset.
    subset_support = deepcopy(sensitivity)
    subset_records = [
        record
        for record in subset_support["fold_records"]
        if record["fold_id"] == "speed_pair_out::100"
        and record["shift_ms"] == OPENAXES_MODEL_SHIFT_MS[0]
    ]
    assert len(subset_records) == 2
    assert len(subset_records[0]["base_uids"]) > 1
    subset_uid = subset_records[0]["base_uids"][0]
    for record in subset_records:
        record["base_uids"] = [subset_uid]
        record["base_uid_rows"] = 1
        record["base_uid_sha256"] = _canonical_sha256(record["base_uids"])
    _reseal_sensitivity(subset_support)
    with pytest.raises(OpenAxesExternalModelV1Error, match="UID support"):
        _verify_alignment_sensitivity_semantics(
            subset_support, tasks, _ensemble_frames(tasks)
        )

    forged_zero_metric = deepcopy(sensitivity)
    forged_record = forged_zero_metric["fold_records"][0]
    forged_record["zero_same_uid_rmse3d_m"] += 0.25
    forged_record["delta_rmse3d_m_shift_minus_zero"] = (
        forged_record["rmse3d_m"] - forged_record["zero_same_uid_rmse3d_m"]
    )
    _reseal_sensitivity(forged_zero_metric)
    with pytest.raises(OpenAxesExternalModelV1Error, match="zero-support metric"):
        _verify_alignment_sensitivity_semantics(
            forged_zero_metric, tasks, _ensemble_frames(tasks)
        )
    attacked = truth.copy()
    attacked.loc[0, "row_uid"] = "invented"
    # The public receipt still binds row accounting; fold pairing catches the
    # invented UID when metrics are constructed.
    with pytest.raises(OpenAxesExternalModelV1Error, match="intersection"):
        aggregate_alignment_sensitivity(_ensemble_frames(tasks), tasks, attacked, receipt)


def test_loso_directional_gate_is_strict_context_matched_and_six_of_eight() -> None:
    context_hash = "c" * 64
    scores = [
        {
            "method_id": method_id,
            "robust_J_m": 1.0 if method_id == "paired_pord" else 2.0,
            "primary_context_identity_sha256": context_hash,
        }
        for method_id in OPENAXES_MODEL_METHOD_IDS
    ]
    scenarios = [
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    ]
    fold_records = []
    for method_id in ("masked_erm", "paired_pord"):
        for index, scenario in enumerate(scenarios):
            score = 2.0
            if method_id == "paired_pord":
                score = 1.0 if index < 6 else 3.0
            fold_records.append(
                {
                    "method_id": method_id,
                    "fold_id": f"scenario_loso::{scenario}",
                    "family": "scenario_loso",
                    "test_scenarios": [scenario],
                    "outage": {"A_mean_context_rmse3d_m": score},
                }
            )
    gate = _build_loso_directional_gate(scores, fold_records)
    assert gate["passes"] is True
    assert gate["strict_scenario_wins"] == 6
    equality = deepcopy(scores)
    next(
        item for item in equality if item["method_id"] == "masked_erm"
    )["robust_J_m"] = 1.0
    assert _build_loso_directional_gate(equality, fold_records)["passes"] is False
    mismatched = deepcopy(scores)
    mismatched[0]["primary_context_identity_sha256"] = "d" * 64
    with pytest.raises(OpenAxesExternalModelV1Error, match="context identity"):
        _build_loso_directional_gate(mismatched, fold_records)


def test_complete_marker_hashes_gate_zero_shift_and_sensitivity_against_tamper() -> None:
    development = {
        "zero_shift": {
            "selection": {"selected_method_id": "paired_pord"},
            "directional_gate": {"passes": True, "gate_sha256": "a" * 64},
            "method_scores": [],
        },
        "alignment_sensitivity_report_only": {
            "selection_use_allowed": False,
            "sensitivity_semantic_sha256": "b" * 64,
        },
    }
    complete = {
        "selected_method_id": "paired_pord",
        "selection_sha256": _canonical_sha256(
            development["zero_shift"]["selection"]
        ),
        "zero_shift_sha256": _canonical_sha256(development["zero_shift"]),
        "directional_gate_passes": True,
        "directional_gate_sha256": _canonical_sha256(
            development["zero_shift"]["directional_gate"]
        ),
        "alignment_sensitivity_sha256": _canonical_sha256(
            development["alignment_sensitivity_report_only"]
        ),
    }
    _verify_complete_semantic_hashes(complete, development)
    for key, replacement in (
        ("selected_method_id", "masked_erm"),
        ("selection_sha256", "0" * 64),
        ("zero_shift_sha256", "0" * 64),
        ("directional_gate_passes", False),
        ("directional_gate_sha256", "0" * 64),
        ("alignment_sensitivity_sha256", "0" * 64),
    ):
        attacked = deepcopy(complete)
        attacked[key] = replacement
        with pytest.raises(OpenAxesExternalModelV1Error, match="complete"):
            _verify_complete_semantic_hashes(attacked, development)


def test_production_complete_requires_signed_source_replay(tmp_path: Path) -> None:
    replay_parameter = inspect.signature(
        validate_openaxes_external_model_development
    ).parameters["replay_signed_source"]
    assert replay_parameter.default is True
    with pytest.raises(OpenAxesExternalModelV1Error, match="requires signed-source"):
        validate_openaxes_external_model_development(
            {}, tmp_path, replay_signed_source=False
        )


def test_full_signed_source_replay_receipt_is_immutable(tmp_path: Path) -> None:
    receipt = {
        "contract": "openaxes.external.model_benchmark.v1",
        "receipt_kind": "immutable_full_signed_source_semantic_replay",
        "signed_report_sha256": "a" * 64,
        "scenario_shift_support_identity_sha256": "b" * 64,
        "validator_source_set_sha256": "c" * 64,
        "full_signed_source_replay": True,
    }
    receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
    frozen = _freeze_or_validate_full_signed_source_replay_receipt(
        tmp_path, receipt, allow_create=True
    )
    assert frozen == receipt
    path = tmp_path / "runtime" / "full_signed_source_replay.json"
    attacked = deepcopy(receipt)
    attacked["signed_report_sha256"] = "f" * 64
    attacked["receipt_payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in attacked.items()
            if key != "receipt_payload_sha256"
        }
    )
    path.write_text(
        json.dumps(attacked, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OpenAxesExternalModelV1Error, match="receipt changed"):
        _freeze_or_validate_full_signed_source_replay_receipt(
            tmp_path, receipt, allow_create=False
        )


def test_model_source_uses_public_shift_api_and_contains_no_challenge_route() -> None:
    source = Path(
        "src/imu_vive_learning/openaxes_external_model_v1_experiment.py"
    ).read_text(encoding="utf-8")
    assert "from .io_utils import sha256_file, write_json, write_parquet" in source
    assert "reconstruct_openaxes_signed_shift_truth_v1" in source
    assert "_open_authenticated_source" not in source
    assert "_scan_nested_alignment" not in source
    assert "finalize_challenge" not in source
    assert tuple(inspect.signature(prepare_openaxes_external_model_run).parameters) == (
        "model_config",
        "source_path",
        "preparation_config_path",
        "preparation_run_dir",
        "output_dir",
        "gpu_inventory",
    )
    assert source.count("_run_parent_public_validator_at_recorded_git_state(") >= 4
    assert "validate_openaxes_external_model_prepared_run(\n        model_config, run_dir" in source
    assert tuple(MODEL_SOURCE_CLOSURE) == (
        "configs/openaxes_external_model_v1.yaml",
        "docs/OPENAXES_EXTERNAL_MODEL_V1_CONTRACT.md",
        "src/imu_vive_learning/io_utils.py",
        "src/imu_vive_learning/openaxes_external_model_v1_config.py",
        "src/imu_vive_learning/openaxes_external_model_v1_training.py",
        "src/imu_vive_learning/openaxes_external_model_v1_experiment.py",
        "tests/test_openaxes_external_model_v1_config.py",
        "tests/test_openaxes_external_model_v1_training.py",
        "tests/test_openaxes_external_model_v1_experiment.py",
    )


def test_doctor_checks_source_closure_without_data_cuda_or_metrics() -> None:
    result = doctor_openaxes_external_model_v1(
        load_openaxes_external_model_config(
            "configs/openaxes_external_model_v1.yaml"
        )
    )
    assert result["status"] == "ok"
    assert result["source_files"] == 9
    assert result["public_signed_shift_api"] is True
    assert result["private_preparation_helpers_imported"] is False
    assert result["challenge_route_present"] is False
    assert result["data_read"] is False
    assert result["cuda_used"] is False
    assert result["metrics_computed"] is False


def test_contract_document_freezes_honest_limits_and_counts() -> None:
    text = Path("docs/OPENAXES_EXTERNAL_MODEL_V1_CONTRACT.md").read_text(
        encoding="utf-8"
    )
    required = (
        "not an independent confirmation",
        "synthetic intervention",
        "label-assisted",
        "offline",
        "transductive",
        "acausal",
        "336",
        "192 primary",
        "144 stress",
        "nine-file model source closure",
        "total-erasure history is unprovable",
        "[-50,-20,-10,+10,+20,+50]",
        "Private preparation helpers are forbidden",
        "There is intentionally no challenge",
        "openaxes-v1-sealed-001",
        "ce09392a287f00447c5451471eae854d6aa726e6",
        "e032c62b899efd7e90647748f6c087b05d271ae1e312fc8ec64f390cc1bc20e3",
        "key-order-only",
        "fresh run",
    )
    assert all(value in text for value in required)
    assert OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256 == (
        "ce6a828e6d92118e29b7c67ec166cc3a555709e78bb41ef0838a79e982e80a3e"
    )
