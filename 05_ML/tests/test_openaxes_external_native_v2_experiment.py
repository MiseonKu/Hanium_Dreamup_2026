from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imu_vive_learning.openaxes_external_native_v2_experiment import (
    ATTEMPT_ORIGIN_KEYS,
    CANDIDATE_IDS,
    EXECUTOR_BINDING_KEYS,
    FIT_COUNTS_BY_GPU,
    FIT_COUNTS_BY_SLOT,
    GPU_INVENTORY,
    METADATA_COLUMNS,
    PHASES,
    PHASE_ACTIONS,
    PHASE_PAYLOAD_KEYS,
    TARGET_COLUMNS,
    TaskSpec,
    VIEW_IDS,
    OpenAxesExternalNativeV2Error,
    RUN_ID,
    _canonical_file,
    _fd_from_argument,
    _normalize_torch_cuda_uuid,
    _run_fresh_task_children,
    _strict_json,
    build_native_v2_task_grid,
    build_slot_assignments,
    build_slot_assignments_receipt,
    build_task_index_receipt,
    classify_resume_state,
    derive_primary_gate_inputs,
    evaluate_outer_predictions_once,
    load_outer_feature_batch_once,
    make_phase_receipt,
    mint_outer_label_capability,
    mint_outer_prediction_capability,
    project_feature_only_frame,
    validate_phase_chain,
    validate_slot_assignments_receipt,
    validate_task_index_receipt,
)


def test_run_id_and_torch_cuda_uuid_are_fresh_and_exact() -> None:
    assert RUN_ID == "openaxes-native-v2-sealed-003"

    cuda_uuid_type = type(
        "_CUuuid",
        (),
        {"__str__": lambda self: "462585f2-b4cc-3864-e27c-cbaae4ab6c9a"},
    )
    cuda_uuid_type.__module__ = "torch._C"
    value = cuda_uuid_type()
    assert (
        _normalize_torch_cuda_uuid(value)
        == "GPU-462585f2-b4cc-3864-e27c-cbaae4ab6c9a"
    )
    with pytest.raises(OpenAxesExternalNativeV2Error, match="runtime type changed"):
        _normalize_torch_cuda_uuid("462585f2-b4cc-3864-e27c-cbaae4ab6c9a")


def test_fresh_fit_children_require_inherited_dependency_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "imu_vive_learning.openaxes_external_native_v2_experiment.build_slot_assignments",
        lambda _tasks: [],
    )
    monkeypatch.setattr(
        "imu_vive_learning.openaxes_external_native_v2_experiment._environment_projection",
        lambda: {"environment": {}},
    )
    with pytest.raises(
        OpenAxesExternalNativeV2Error,
        match="dependency descriptor is not a retained directory",
    ):
        _run_fresh_task_children(3, 4, -1, -1, [], {}, {}, {})


def test_fresh_fit_children_forward_exact_dependency_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    task = TaskSpec(
        task_id="task_000",
        task_ordinal=0,
        context_ordinal=0,
        fold_id="fold",
        fold_ordinal=0,
        fold_family="scenario_loso",
        train_scenarios=(),
        test_scenarios=(),
        seed=0,
        seed_ordinal=0,
        candidate_id="A",
        candidate_ordinal=0,
        task_spec_sha256="0" * 64,
    )
    launches: list[tuple[list[str], dict[str, object]]] = []

    class _CompletedProcess:
        def wait(self) -> int:
            return 0

    def _popen(command: list[str], **kwargs: object) -> _CompletedProcess:
        launches.append((command, kwargs))
        return _CompletedProcess()

    try:
        monkeypatch.setattr(
            "imu_vive_learning.openaxes_external_native_v2_experiment.build_slot_assignments",
            lambda _tasks: [
                {
                    "ordered_task_ids": ["task_000"],
                    "gpu_reference": {"uuid": "GPU-exact"},
                }
            ],
        )
        monkeypatch.setattr(
            "imu_vive_learning.openaxes_external_native_v2_experiment._environment_projection",
            lambda: {"environment": {"PATH": "/usr/bin:/bin"}},
        )
        monkeypatch.setattr(
            "imu_vive_learning.openaxes_external_native_v2_experiment._mkdir_exclusive",
            lambda *_args: None,
        )
        monkeypatch.setattr(
            "imu_vive_learning.openaxes_external_native_v2_experiment._attempt_origin",
            lambda *_args: {},
        )
        monkeypatch.setattr(
            "imu_vive_learning.openaxes_external_native_v2_experiment._write_attempt_authority",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "imu_vive_learning.openaxes_external_native_v2_experiment.uuid.uuid4",
            lambda: "11111111-1111-4111-8111-111111111111",
        )
        monkeypatch.setattr(
            "imu_vive_learning.openaxes_external_native_v2_experiment.subprocess.Popen",
            _popen,
        )
        _run_fresh_task_children(
            3,
            4,
            5,
            descriptor,
            [task],
            {},
            {},
            {"controllers": [{"controller_uuid": "controller"}]},
        )
    finally:
        os.close(descriptor)

    assert len(launches) == 1
    command, kwargs = launches[0]
    dependency_root = f"/proc/self/fd/{descriptor}"
    assert command == [
        "/proc/self/exe",
        "-P",
        "-s",
        "-S",
        "-B",
        "-m",
        "imu_vive_learning.openaxes_external_native_v2_experiment",
        "internal-fit",
        "--workspace-root",
        "/proc/self/fd/3",
        "--run-dir",
        "/proc/self/fd/4",
        "--dependency-root",
        dependency_root,
        "--task-id",
        "task_000",
        "--attempt-id",
        "11111111-1111-4111-8111-111111111111",
        "--slot-child-ordinal",
        "0",
        "--parent-controller-uuid",
        "controller",
        "--controller-ready-path",
        "runtime/worker_ready/launch_000000/slot_00.json",
        "--launch-ready-path",
        "runtime/execution_launches/launch_000000/launch.ready.json",
    ]
    assert kwargs == {
        "env": {
            "PATH": "/usr/bin:/bin",
            "CUDA_VISIBLE_DEVICES": "GPU-exact",
            "PYTHONPATH": f"/proc/self/fd/5/src{os.pathsep}{dependency_root}",
        },
        "pass_fds": (3, 4, 5, descriptor),
        "close_fds": True,
    }


def _synthetic_manifest() -> dict[str, object]:
    scenarios = [
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    ]
    folds: list[dict[str, object]] = []
    for scenario in scenarios:
        folds.append(
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
        folds.append(
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
        test = [f"{shape}-{speed}" for speed in (100, 150, 200, 250)]
        folds.append(
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
    return {"contract": "openaxes.external.preparation.v1", "folds": {"items": folds}}


def _reference(path: str, marker: str = "0") -> dict[str, object]:
    return {"path": path, "bytes": 7, "sha256": marker * 64}


def _origin() -> dict[str, object]:
    values: dict[str, object] = {
        "execution_session_id": "00000000-0000-4000-8000-000000000001",
        "attempt_origin_launch_id": "launch_000000",
        "attempt_origin_launch_ordinal": 0,
        "transport_slot_index": 0,
        "local_slot": 0,
        "gpu_reference": dict(GPU_INVENTORY[0]),
        "scientific_fit_attempt": 1,
    }
    for key in ATTEMPT_ORIGIN_KEYS:
        if key not in values:
            values[key] = "1" * 64
    return values


def _executor(phase: str) -> dict[str, object]:
    values: dict[str, object] = {
        "execution_session_id": "00000000-0000-4000-8000-000000000001",
        "launch_id": "launch_000000",
        "launch_ordinal": 0,
        "transport_slot_index": 0,
        "local_slot": 0,
        "gpu_reference": dict(GPU_INVENTORY[0]),
        "phase_action_class": PHASE_ACTIONS[phase],
        "launch_ready_reference": _reference("runtime/launch.ready.json", "2"),
        "controller_ready_reference": _reference("runtime/controller.ready.json", "3"),
        "child_ready_reference": _reference("runtime/child.ready.json", "4"),
        "child_uuid": "00000000-0000-4000-8000-000000000002",
        "child_pid": 123,
    }
    for key in EXECUTOR_BINDING_KEYS:
        if key not in values:
            values[key] = (
                "1" * 64
                if key
                in {
                    "controller_source_sha256",
                    "environment_sha256",
                    "source_set_sha256",
                    "task_assignment_sha256",
                    "slot_assignment_sha256",
                    "comparator_bundle_sha256",
                    "pre_fit_validation_sha256",
                }
                else "5" * 64
            )
    return values


def _phase_payload(
    phase: str, task: object, executor: dict[str, object]
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for index, key in enumerate(PHASE_PAYLOAD_KEYS[phase]):
        if key.endswith("_reference"):
            payload[key] = _reference(f"runtime/synthetic/{phase}/{key}.json", str(index % 10))
        else:
            payload[key] = 0
    result_base = f"results/{task.task_id}"
    ledger_base = f"runtime/ledgers/{task.task_id}"
    paths = {
        "execution.started": {
            "attempt_roster_reference": f"runtime/attempt_roster/{task.task_id}.json",
            "attempt_ledger_reference": f"runtime/attempt_ledger/{task.task_id}.json",
        },
        "training.completed": {
            "checkpoint_reference": f"{result_base}/checkpoint.pt",
            "transform_reference": f"{result_base}/transform.json",
            "history_reference": f"{result_base}/history.json",
            "training_receipt_reference": f"{result_base}/training_receipt.json",
        },
        "outer_prediction.completed": {
            "prediction_claim_reference": f"runtime/outer_prediction_claims/{task.task_id}.json",
            "checkpoint_reference": f"{result_base}/checkpoint.pt",
            "outer_predictions_reference": f"{result_base}/outer_predictions.parquet",
        },
        "outer_eval.started": {
            "label_claim_reference": f"runtime/outer_label_claims/{task.task_id}.json",
            "outer_prediction_phase_reference": f"{ledger_base}/outer_prediction.completed.json",
            "outer_predictions_reference": f"{result_base}/outer_predictions.parquet",
        },
        "outer_eval.completed": {
            "label_claim_reference": f"runtime/outer_label_claims/{task.task_id}.json",
            "outer_predictions_reference": f"{result_base}/outer_predictions.parquet",
            "outer_evaluation_rows_reference": f"{result_base}/outer_evaluation_rows.parquet",
            "outer_evaluation_reference": f"{result_base}/outer_evaluation.json",
        },
        "bundle.completed": {
            "bundle_reference": f"{result_base}/bundle.json",
            "outer_evaluation_rows_reference": f"{result_base}/outer_evaluation_rows.parquet",
            "outer_evaluation_reference": f"{result_base}/outer_evaluation.json",
        },
        "result.completed": {
            "result_reference": f"{result_base}/result.json",
            "bundle_reference": f"{result_base}/bundle.json",
        },
    }[phase]
    for index, (key, path) in enumerate(paths.items()):
        payload[key] = _reference(path, str((index + 6) % 10))
    if phase == "execution.started":
        payload["fit_child_ready_reference"] = deepcopy(
            executor["child_ready_reference"]
        )
        payload["launch_ready_reference"] = deepcopy(
            executor["launch_ready_reference"]
        )
    if phase == "training.completed":
        payload.update(
            {
                "selected_update": 17,
                "optimizer_updates": 400,
                "training_model_forward_calls": 400,
                "checkpoint_model_forward_calls": 400,
                "parameter_gradient_traversals": 2000
                if task.candidate_id == CANDIDATE_IDS[2]
                else 400,
                "cagrad_solver_calls": 400
                if task.candidate_id == CANDIDATE_IDS[2]
                else 0,
            }
        )
    elif phase == "outer_prediction.completed":
        payload.update(
            {
                "outer_feature_loader_calls": 1,
                "outer_model_forward_calls": 1,
                "outer_view_equivalent_forwards": 5,
                "outer_label_reads": 0,
                "task_evaluator_calls": 0,
            }
        )
    elif phase == "outer_eval.completed":
        payload.update(
            {"outer_label_reads": 1, "task_evaluator_calls": 1, "views_evaluated": 5}
        )
    return payload


def test_exact_task_grid_triplets_and_interleaved_slot_accounting() -> None:
    tasks = build_native_v2_task_grid(_synthetic_manifest())
    assert len(tasks) == 126
    assert [task.task_id for task in tasks] == [f"task_{index:03d}" for index in range(126)]
    for task in tasks:
        assert task.task_ordinal == task.context_ordinal * 3 + task.candidate_ordinal
        assert task.context_ordinal == task.fold_ordinal * 3 + task.seed_ordinal
    for context in range(42):
        triplet = tasks[context * 3 : context * 3 + 3]
        assert [task.candidate_id for task in triplet] == list(CANDIDATE_IDS)
        assert len({task.context_ordinal % 12 for task in triplet}) == 1
    slots = build_slot_assignments(tasks)
    assert tuple(len(row["ordered_task_ids"]) for row in slots) == FIT_COUNTS_BY_SLOT
    gpu_counts = tuple(
        sum(len(slots[slot]["ordered_task_ids"]) for slot in range(gpu, 12, 4))
        for gpu in range(4)
    )
    assert gpu_counts == FIT_COUNTS_BY_GPU == (33, 33, 30, 30)


def test_task_index_and_slot_receipts_round_trip_and_reject_tamper() -> None:
    tasks = build_native_v2_task_grid(_synthetic_manifest())
    references = {
        task.task_id: _reference(f"tasks/{task.task_id}.json", "a") for task in tasks
    }
    index = build_task_index_receipt(tasks, references)
    assert validate_task_index_receipt(index, tasks) == index
    slots = build_slot_assignments_receipt(tasks, index["task_assignment_sha256"])
    assert validate_slot_assignments_receipt(
        slots, tasks, index["task_assignment_sha256"]
    ) == slots
    changed = deepcopy(index)
    changed["task_assignment_rows"][0]["transport_slot_index"] = 1
    with pytest.raises(OpenAxesExternalNativeV2Error):
        validate_task_index_receipt(changed, tasks)
    changed_slots = deepcopy(slots)
    changed_slots["slots"][0]["ordered_task_ids"].reverse()
    with pytest.raises(OpenAxesExternalNativeV2Error):
        validate_slot_assignments_receipt(
            changed_slots, tasks, index["task_assignment_sha256"]
        )


def test_all_seven_phase_receipts_form_one_tamper_evident_prefix() -> None:
    task = build_native_v2_task_grid(_synthetic_manifest())[2]
    attempt_id = "00000000-0000-4000-8000-000000000003"
    receipts: list[dict[str, object]] = []
    previous: dict[str, object] | None = None
    for phase in PHASES:
        executor = _executor(phase)
        receipt = make_phase_receipt(
            task,
            phase,
            attempt_id=attempt_id,
            previous_phase=previous,
            attempt_origin_binding=_origin(),
            executor_binding=executor,
            payload=_phase_payload(phase, task, executor),
        )
        receipts.append(receipt)
        raw = _canonical_file(receipt)
        previous = {
            "path": f"runtime/ledgers/{task.task_id}/{phase}.json",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    assert len(validate_phase_chain(receipts, task)) == 7
    changed = deepcopy(receipts)
    changed[3]["payload"]["prior_outer_label_reads"] = 1
    with pytest.raises(OpenAxesExternalNativeV2Error):
        validate_phase_chain(changed, task)
    changed_origin = deepcopy(receipts)
    changed_origin[5]["attempt_origin_binding"]["environment_sha256"] = "f" * 64
    changed_origin[5]["phase_receipt_sha256"] = "f" * 64
    with pytest.raises(OpenAxesExternalNativeV2Error):
        validate_phase_chain(changed_origin, task)


def test_single_use_feature_and_label_capabilities_enforce_durable_order() -> None:
    task_id = "task_000"
    attempt_id = "00000000-0000-4000-8000-000000000004"
    prediction_claim = _reference(
        f"runtime/outer_prediction_claims/{task_id}.json", "1"
    )
    prediction_capability = mint_outer_prediction_capability(
        task_id=task_id,
        attempt_id=attempt_id,
        claim_reference=prediction_claim,
    )
    rows = 4
    feature_frame = pd.DataFrame(
        {
            "row_uid": [f"uid-{index}" for index in range(rows)],
            "scenario_id": ["circle-100"] * rows,
            "sample_index": list(range(rows)),
            "aligned_time_us": [index * 10000 for index in range(rows)],
            **{
                f"sensor_{column:02d}": np.arange(rows, dtype=np.float64) + column
                for column in range(36)
            },
        }
    )
    feature_frame.columns = [
        *METADATA_COLUMNS,
        *[
            f"{sensor}_{channel}"
            for sensor in ("humerus", "radius", "carpus", "digitus")
            for channel in (
                "accel_adxl355_x_mps2",
                "accel_adxl355_y_mps2",
                "accel_adxl355_z_mps2",
                "accel_bmi160_x_mps2",
                "accel_bmi160_y_mps2",
                "accel_bmi160_z_mps2",
                "gyro_x_rad_s",
                "gyro_y_rad_s",
                "gyro_z_rad_s",
            )
        ],
    ]
    opens = {"features": 0, "truth": 0}

    def feature_opener() -> pd.DataFrame:
        opens["features"] += 1
        return feature_frame.copy()

    batch = load_outer_feature_batch_once(
        prediction_capability,
        task_id=task_id,
        attempt_id=attempt_id,
        claim_sha256=prediction_claim["sha256"],
        opener=feature_opener,
    )
    assert batch.continuous_features.shape == (4, 36)
    assert opens["features"] == 1
    with pytest.raises(OpenAxesExternalNativeV2Error, match="spent"):
        load_outer_feature_batch_once(
            prediction_capability,
            task_id=task_id,
            attempt_id=attempt_id,
            claim_sha256=prediction_claim["sha256"],
            opener=feature_opener,
        )
    prediction_reference = _reference(
        f"results/{task_id}/outer_predictions.parquet", "2"
    )
    started_reference = _reference(
        f"runtime/ledgers/{task_id}/outer_eval.started.json", "3"
    )
    label_claim = _reference(f"runtime/outer_label_claims/{task_id}.json", "4")
    label_capability = mint_outer_label_capability(
        task_id=task_id,
        attempt_id=attempt_id,
        prediction_reference=prediction_reference,
        outer_eval_started_reference=started_reference,
        claim_reference=label_claim,
    )
    truth_frame = feature_frame[list(METADATA_COLUMNS)].copy()
    truth_frame["tcp_x_m"] = 0.0
    truth_frame["tcp_y_m"] = 0.0
    truth_frame["tcp_z_m"] = 0.0

    def truth_opener() -> pd.DataFrame:
        opens["truth"] += 1
        return truth_frame.copy()

    evaluation, observed_truth = evaluate_outer_predictions_once(
        label_capability,
        task_id=task_id,
        attempt_id=attempt_id,
        prediction_sha256=prediction_reference["sha256"],
        outer_eval_started_sha256=started_reference["sha256"],
        claim_sha256=label_claim["sha256"],
        prediction_m=np.zeros((5, rows, 3), dtype=np.float64),
        truth_opener=truth_opener,
    )
    assert opens == {"features": 1, "truth": 1}
    assert list(observed_truth.columns) == [*METADATA_COLUMNS, *TARGET_COLUMNS]
    assert [row["view_id"] for row in evaluation["views"]] == list(VIEW_IDS)
    assert all(row["rmse3d_m"] == 0.0 for row in evaluation["views"])
    with pytest.raises(OpenAxesExternalNativeV2Error, match="spent"):
        evaluate_outer_predictions_once(
            label_capability,
            task_id=task_id,
            attempt_id=attempt_id,
            prediction_sha256=prediction_reference["sha256"],
            outer_eval_started_sha256=started_reference["sha256"],
            claim_sha256=label_claim["sha256"],
            prediction_m=np.zeros((5, rows, 3), dtype=np.float64),
            truth_opener=truth_opener,
        )
    assert opens["truth"] == 1


def test_resume_classifier_refuses_ambiguous_or_repeated_scientific_actions() -> None:
    common = {
        "prediction_permit_present": True,
        "prediction_claim_present": False,
        "label_permit_present": True,
        "label_claim_present": False,
    }
    assert classify_resume_state(
        attempt_authority_present=False, last_phase=None, **common
    ).startswith("fresh_fit_once")
    assert classify_resume_state(
        attempt_authority_present=True, last_phase="execution.started", **common
    ) == "fail_closed"
    assert classify_resume_state(
        attempt_authority_present=True, last_phase="training.completed", **common
    ) == "promote_prediction_permit_and_predict_once"
    assert classify_resume_state(
        attempt_authority_present=True,
        last_phase="outer_eval.started",
        prediction_permit_present=False,
        prediction_claim_present=True,
        label_permit_present=False,
        label_claim_present=True,
    ) == "fail_closed"
    assert classify_resume_state(
        attempt_authority_present=True,
        last_phase="outer_eval.completed",
        prediction_permit_present=False,
        prediction_claim_present=True,
        label_permit_present=False,
        label_claim_present=True,
    ) == "bundle_from_persisted_evaluation"


def test_fixed_gate_derivation_and_fd_root_parser(tmp_path: Path) -> None:
    outages = [
        {"rmse3d_m": float(index + 1)} for index in range(32)
    ]
    full = [{"rmse3d_m": float(index + 1)} for index in range(8)]
    per_seed = [
        {"seed": seed, "rmse3d_m": float(index + 1)}
        for seed in (1701, 2603, 3907)
        for index in range(32)
    ]
    derived = derive_primary_gate_inputs(outages, full, per_seed)
    assert list(derived) == [
        "A_mean_context_rmse3d_m",
        "WS_worst_scenario_mean_rmse3d_m",
        "WM_worst_mask_mean_rmse3d_m",
        "robust_J_m",
        "fold_outage_scores_m",
        "per_seed_J_m",
        "full_F_m",
    ]
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        duplicate = _fd_from_argument(
            f"/proc/self/fd/{descriptor}", name="synthetic root"
        )
        try:
            assert os.fstat(duplicate).st_ino == os.fstat(descriptor).st_ino
        finally:
            os.close(duplicate)
    finally:
        os.close(descriptor)
    with pytest.raises(OpenAxesExternalNativeV2Error):
        _fd_from_argument(str(tmp_path), name="synthetic root")


def test_strict_json_and_runtime_source_negative_guard() -> None:
    canonical = b'{"a":1,"b":2}\n'
    assert _strict_json(canonical, name="synthetic") == {"a": 1, "b": 2}
    with pytest.raises(OpenAxesExternalNativeV2Error, match="duplicate"):
        _strict_json(b'{"a":1,"a":2}\n', name="synthetic")
    with pytest.raises(OpenAxesExternalNativeV2Error, match="canonical"):
        _strict_json(b'{"b":2, "a":1}\n', name="synthetic")
    forbidden_roots = {
        "builtins",
        "ctypes",
        "gc",
        "importlib",
        "inspect",
        "marshal",
        "runpy",
        "site",
        "sys",
    }
    forbidden_attributes = {
        "__getattribute__",
        "abspath",
        "readlink",
        "realpath",
        "resolve",
    }
    source_paths = [
        Path("src/imu_vive_learning/openaxes_external_native_v2_config.py"),
        Path("src/imu_vive_learning/openaxes_external_native_v2_training.py"),
        Path("src/imu_vive_learning/openaxes_external_native_v2_experiment.py"),
    ]
    for source_path in source_paths:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = {
            alias.asname or alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name != "*"
        }

        def stored_names(node: ast.AST) -> set[str]:
            return {
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, (ast.Store, ast.Del))
            }

        shared = set(imported)

        def direct_shared(node: ast.AST | None) -> bool:
            if isinstance(node, ast.Name):
                return isinstance(node.ctx, ast.Load) and node.id in shared
            if isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
                return direct_shared(node.value)
            return False

        def contains_shared(node: ast.AST | None) -> bool:
            if direct_shared(node):
                return True
            if isinstance(node, ast.Call):
                return False
            return node is not None and any(
                contains_shared(child) for child in ast.iter_child_nodes(node)
            )

        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
        ]
        flows = [
            node
            for node in ast.walk(tree)
            if isinstance(
                node,
                (ast.For, ast.AsyncFor, ast.comprehension, ast.With, ast.AsyncWith),
            )
        ]
        changed = True
        while changed:
            changed = False
            for assignment in assignments:
                if isinstance(assignment, ast.Assign):
                    value = assignment.value
                    targets = assignment.targets
                else:
                    value = assignment.value
                    targets = [assignment.target]
                if contains_shared(value):
                    additions = set().union(
                        *(stored_names(target) for target in targets)
                    )
                    if not additions <= shared:
                        shared.update(additions)
                        changed = True
            for flow in flows:
                if isinstance(flow, (ast.For, ast.AsyncFor, ast.comprehension)):
                    sources = [flow.iter]
                    targets = [flow.target]
                else:
                    sources = [item.context_expr for item in flow.items]
                    targets = [
                        item.optional_vars
                        for item in flow.items
                        if item.optional_vars is not None
                    ]
                if any(contains_shared(source) for source in sources):
                    additions = set().union(
                        *(stored_names(target) for target in targets)
                    )
                    if not additions <= shared:
                        shared.update(additions)
                        changed = True

        def mutation_root(node: ast.AST) -> str | None:
            cursor = node
            while isinstance(cursor, (ast.Attribute, ast.Subscript)):
                cursor = cursor.value
            return cursor.id if isinstance(cursor, ast.Name) else None

        shared_mutators = {
            "__delitem__",
            "__iadd__",
            "__iand__",
            "__ior__",
            "__isub__",
            "__ixor__",
            "__setitem__",
            "add",
            "append",
            "chdir",
            "clear",
            "discard",
            "extend",
            "insert",
            "pop",
            "popitem",
            "putenv",
            "remove",
            "reverse",
            "setdefault",
            "sort",
            "umask",
            "unsetenv",
            "update",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                assert node.id not in imported
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".", 1)[0] not in forbidden_roots for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] not in forbidden_roots
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_attributes
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not (
                    node.func.attr == "relative_file_manifest"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "io_utils"
                )
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                assert mutation_root(node) not in shared
            if isinstance(node, ast.AugAssign):
                assert mutation_root(node.target) not in shared
            if isinstance(node, ast.Call):
                called = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else None
                )
                if isinstance(node.func, ast.Attribute) and node.func.attr in shared_mutators:
                    assert mutation_root(node.func) not in shared
                if called not in {"isinstance", "issubclass"}:
                    assert not any(
                        contains_shared(argument)
                        for argument in [
                            *node.args,
                            *(keyword.value for keyword in node.keywords),
                        ]
                    )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                defaults = [*node.args.defaults, *node.args.kw_defaults]
                assert not any(contains_shared(default) for default in defaults)
            if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
                assert not contains_shared(node.value)
    experiment_tree = ast.parse(source_paths[2].read_text(encoding="utf-8"))
    relative_project_imports = {
        alias.name
        for node in ast.walk(experiment_tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
        for alias in node.names
    }
    assert relative_project_imports == {
        "io_utils",
        "openaxes_external_native_v2_config",
        "openaxes_external_native_v2_training",
    }
