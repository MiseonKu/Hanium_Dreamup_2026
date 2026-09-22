from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from imu_vive_learning.experiment_io import (
    atomic_write_json,
    atomic_write_parquet,
    stable_hash,
)
from imu_vive_learning.experiment_metrics import regression_metrics
from imu_vive_learning.features import FUSION_COLUMNS
from imu_vive_learning.gpu_outage_data import AVAILABILITY_NAMES
from imu_vive_learning.gpu_outage_pord_v5_config import (
    PORD_V5_CANDIDATE_IDS,
    PORD_V5_PRIMARY_BACKBONE_ID,
    PORD_V5_PRIMARY_CANDIDATE_IDS,
    PORD_V5_TRANSFER_CANDIDATE_IDS,
    load_pord_v5_config,
)
from imu_vive_learning.gpu_outage_pord_v5_training import state_dict_sha256
import imu_vive_learning.gpu_outage_pord_v5_experiment as exp


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "endpoint_regression_gpu_outage_v5_pord.yaml"
TRUE_COLUMNS = ("true_x_mm", "true_y_mm", "true_z_mm")
PRED_COLUMNS = ("pred_x_mm", "pred_y_mm", "pred_z_mm")
SEGMENT_COUNTS = {36: 1672, 37: 1597, 38: 1707, 39: 1659}
CHALLENGE_COUNTS = {40: 1672, 41: 1623}


def _filesystem_tree_sha256(root: Path) -> str:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            rows.append({"path": relative, "kind": "directory"})
        elif path.is_file():
            rows.append(
                {
                    "path": relative,
                    "kind": "file",
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        else:
            rows.append({"path": relative, "kind": "other"})
    return stable_hash(rows)


def _config() -> dict[str, Any]:
    return load_pord_v5_config(CONFIG_PATH)


def _gpu_devices() -> list[dict[str, Any]]:
    return [
        {
            "physical_gpu_id": index,
            "uuid": f"00000000-0000-0000-0000-{index + 1:012x}",
            "pci_bus_id": f"0000:{index + 1:02x}:00.0",
            "name": "synthetic-cuda-device",
        }
        for index in range(4)
    ]


def _spawn_initial_hash_worker(output_queue: Any, seed: int) -> None:
    preloaded = "torch" in sys.modules
    from imu_vive_learning.gpu_outage_pord_v5_training import (
        initial_pord_state_sha256,
    )

    output_queue.put(
        {
            "torch_preloaded": preloaded,
            "sha256": initial_pord_state_sha256(
                backbone_id="masked_mlp_64x64_relu", model_seed=seed
            ),
            "process_id": os.getpid(),
        }
    )


def _crashing_task_executor(**_: Any) -> Mapping[str, Any]:
    raise RuntimeError("intentional_spawn_worker_crash")


def _metric_contexts(errors: Mapping[str, float]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate,
            "held_out_segment": segment,
            "forced_outage_index": outage,
            "metrics": {
                "rmse_3d_mm": float(error + 0.01 * (segment - 36) + 0.001 * outage),
                "p95_3d_error_mm": float(error + 0.1),
            },
            "seed_rmse_population_std_mm": 0.01,
        }
        for candidate, error in errors.items()
        for segment in (36, 37, 38, 39)
        for outage in range(4)
    ]


def test_exact_168_fit_and_224_56_context_arithmetic() -> None:
    tasks = exp.pord_v5_development_tasks(_config())
    assert len(tasks) == 168
    assert len({task["task_id"] for task in tasks}) == 168
    assert sum(task["candidate_id"] in PORD_V5_PRIMARY_CANDIDATE_IDS for task in tasks) == 144
    assert sum(task["candidate_id"] in PORD_V5_TRANSFER_CANDIDATE_IDS for task in tasks) == 24
    assert 12 * 4 * 4 == 192
    assert 2 * 4 * 4 == 32
    assert 192 + 32 == 224
    assert 14 * 4 == 56


@pytest.mark.parametrize("workers_per_gpu", [1, 2, 3])
def test_workers_per_gpu_changes_only_runtime_assignment(workers_per_gpu: int) -> None:
    tasks = exp.pord_v5_development_tasks(_config())
    assigned = exp.assign_pord_v5_tasks(
        tasks, gpu_devices=_gpu_devices(), workers_per_gpu=workers_per_gpu
    )
    assert [row["task"]["task_id"] for row in assigned] == [
        task["task_id"] for task in tasks
    ]
    assert len({row["task"]["task_id"] for row in assigned}) == 168
    assert len({row["worker"]["logical_worker_id"] for row in assigned}) == 4 * workers_per_gpu


@pytest.mark.parametrize("bad", [0, 4, True, 1.0, "1"])
def test_workers_per_gpu_and_gpu_ids_reject_coercion(bad: Any) -> None:
    with pytest.raises(exp.GPUOutagePORDV5Error):
        exp.assign_pord_v5_tasks(
            exp.pord_v5_development_tasks(_config()),
            gpu_devices=_gpu_devices(),
            workers_per_gpu=bad,
        )
    if bad in (1.0, "1", True):
        with pytest.raises(exp.GPUOutagePORDV5Error, match="exact integers"):
            exp.pord_v5_environment_payload([bad, 1, 2, 3])


def test_module_import_is_torch_free_and_spawn_pairing_avoids_thread_rng_race() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    checked = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import imu_vive_learning.gpu_outage_pord_v5_experiment; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr
    source = (ROOT / "src/imu_vive_learning/gpu_outage_pord_v5_experiment.py").read_text()
    assert "ThreadPoolExecutor" not in source
    assert "concurrent.futures" not in source

    context = mp.get_context("spawn")
    output_queue = context.Queue()
    processes = [
        context.Process(target=_spawn_initial_hash_worker, args=(output_queue, 1729))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    rows = [output_queue.get(timeout=30.0) for _ in processes]
    for process in processes:
        process.join(timeout=30.0)
    output_queue.close()
    assert all(process.exitcode == 0 for process in processes)
    assert all(row["torch_preloaded"] is False for row in rows)
    assert len({row["process_id"] for row in rows}) == 2
    assert len({row["sha256"] for row in rows}) == 1


def test_cuda_uuid_selector_is_canonical_and_distinct_from_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = {
        **_gpu_devices()[0],
        "logical_worker_id": "gpu0__worker0",
        "worker_index_on_gpu": 0,
    }
    bare_uuid = worker["uuid"]
    selector = exp._canonical_cuda_uuid_selector(bare_uuid)
    assert selector == f"GPU-{bare_uuid}"
    assert selector != bare_uuid
    for malformed in (
        selector,
        "ABCDEFAB-0000-0000-0000-000000000001",
        bare_uuid.replace("-", ""),
        "GPU-SYNTHETIC-0",
        "",
        None,
        True,
    ):
        with pytest.raises(exp.GPUOutagePORDV5Error, match="bare lowercase UUID"):
            exp._canonical_cuda_uuid_selector(malformed)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", selector)
    receipt = exp._process_runtime_gpu_receipt(
        worker, synthetic_test_only=True
    )
    assert receipt["uuid"] == bare_uuid
    assert receipt["cuda_visible_devices"] == selector
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", bare_uuid)
    with pytest.raises(exp.GPUOutagePORDV5Error, match="canonical frozen-UUID"):
        exp._process_runtime_gpu_receipt(worker, synthetic_test_only=True)


def test_canonical_selector_exposes_current_gpu_in_torch_fresh_subprocess() -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        pytest.skip("read-only NVIDIA probe is unavailable")
    queried = subprocess.run(
        [nvidia_smi, "--query-gpu=uuid", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    selectors = [line.strip() for line in queried.stdout.splitlines() if line.strip()]
    if queried.returncode != 0 or not selectors:
        pytest.skip("read-only NVIDIA UUID probe found no visible GPU")
    selector = selectors[0]
    if not selector.startswith("GPU-"):
        pytest.fail("nvidia-smi returned a noncanonical GPU UUID selector")
    bare_uuid = selector.removeprefix("GPU-")
    assert exp._canonical_cuda_uuid_selector(bare_uuid) == selector

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["CUDA_VISIBLE_DEVICES"] = selector
    script = """
import json
import sys
assert 'torch' not in sys.modules
import imu_vive_learning.gpu_outage_pord_v5_experiment as experiment
module_import_was_torch_free = 'torch' not in sys.modules
visible = experiment._cuda_device_identity(0)
worker = {**visible, 'logical_worker_id': 'production-probe', 'worker_index_on_gpu': 0}
receipt = experiment._process_runtime_gpu_receipt(worker, synthetic_test_only=False)
import torch
print(json.dumps({
    'module_import_was_torch_free': module_import_was_torch_free,
    'device_count': torch.cuda.device_count(),
    'visible_uuid': visible['uuid'],
    'receipt': receipt,
}))
"""
    checked = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert checked.returncode == 0, checked.stderr
    payload = json.loads(checked.stdout)
    assert payload["module_import_was_torch_free"] is True
    assert payload["device_count"] == 1
    assert payload["visible_uuid"] == bare_uuid
    assert payload["receipt"]["uuid"] == bare_uuid
    assert payload["receipt"]["cuda_visible_devices"] == selector


def test_spawn_controller_detects_child_crash_and_terminates_slots(tmp_path: Path) -> None:
    run, _, _ = _prepare_synthetic_run(tmp_path)
    config = _config()
    assignments = exp.assign_pord_v5_tasks(
        exp.pord_v5_development_tasks(config),
        gpu_devices=_gpu_devices(),
        workers_per_gpu=1,
    )
    one_per_slot: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for assignment in assignments:
        worker_id = str(assignment["worker"]["logical_worker_id"])
        if worker_id not in seen:
            one_per_slot.append(assignment)
            seen.add(worker_id)
    assert len(one_per_slot) == 4
    with pytest.raises(exp.GPUOutagePORDV5Error, match="spawn worker failed|crashed"):
        exp._run_spawn_slot_assignments(
            run_dir=run,
            config=config,
            data=exp._load_development_snapshot(run),
            assignments=one_per_slot,
            workers_per_gpu=1,
            task_executor=_crashing_task_executor,
            resume=False,
            synthetic_test_only=True,
        )


def test_exact_v3_j_semantics_and_missing_nonfinite_fail_closed() -> None:
    contexts = _metric_contexts({"a": 2.0, "b": 1.0})
    scored = exp.score_pord_v5_candidates(contexts, candidate_ids=("a", "b"))
    assert scored["selected_candidate_id"] == "b"
    row = scored["trace"][1]
    expected_j = (
        row["mean_rmse_3d_mm"]
        + row["worst_segment_mean_rmse_3d_mm"]
        + row["worst_sensor_mean_rmse_3d_mm"]
    ) / 3.0
    assert row["outage_robust_j_mm"] == pytest.approx(expected_j)
    with pytest.raises(exp.GPUOutagePORDV5Error, match="missing"):
        exp.score_pord_v5_candidates(contexts[:-1], candidate_ids=("a", "b"))
    attacked = [dict(row) for row in contexts]
    attacked[0] = {**attacked[0], "metrics": {**attacked[0]["metrics"], "rmse_3d_mm": np.nan}}
    with pytest.raises(exp.GPUOutagePORDV5Error, match="finite"):
        exp.score_pord_v5_candidates(attacked, candidate_ids=("a", "b"))


def test_score_and_artifact_references_reject_type_coercion_and_open_schema(
    tmp_path: Path,
) -> None:
    artifact = atomic_write_json(tmp_path / "artifact.json", {"ok": True})
    reference = exp._artifact_reference(artifact, tmp_path)
    assert exp._verify_reference(tmp_path, reference) == artifact
    for attacked in (
        {**reference, "bytes": str(reference["bytes"])},
        {**reference, "sha256": reference["sha256"].upper()},
        {**reference, "unexpected": 1},
    ):
        with pytest.raises(exp.GPUOutagePORDV5Error):
            exp._verify_reference(tmp_path, attacked)

    contexts = _metric_contexts({"a": 1.0})
    attacks = (
        {**contexts[0], "candidate_id": 7},
        {**contexts[0], "held_out_segment": "36"},
        {**contexts[0], "forced_outage_index": "0"},
        {
            **contexts[0],
            "metrics": {**contexts[0]["metrics"], "rmse_3d_mm": "1.0"},
        },
        {**contexts[0], "unexpected": False},
    )
    for attack in attacks:
        attacked = [dict(row) for row in contexts]
        attacked[0] = attack
        with pytest.raises(exp.GPUOutagePORDV5Error):
            exp.score_pord_v5_candidates(attacked, candidate_ids=("a",))


def test_v3_parent_checkpoint_reference_adapter_is_strict_and_version_scoped(
    tmp_path: Path,
) -> None:
    v3 = tmp_path / "v3"
    preprocessing = atomic_write_json(
        v3 / "development" / "preprocessing.json", {"frozen": True}
    )
    preprocessing_reference = exp._artifact_reference(preprocessing, v3)
    assert (
        exp._verify_v3_parent_preprocessing_reference(
            v3, preprocessing_reference
        )
        == preprocessing
    )
    with pytest.raises(exp.GPUOutagePORDV5Error, match="preprocessing reference"):
        exp._verify_v3_parent_preprocessing_reference(
            v3, {**preprocessing_reference, "rows": 1}
        )

    checkpoint = exp._save_checkpoint(
        v3 / "development" / "checkpoint.npz",
        {
            "layer.weight": np.ones((2, 2), dtype=np.float32),
            "layer.bias": np.zeros(2, dtype=np.float32),
        },
    )
    reference = exp._artifact_reference(checkpoint, v3, tensor_count=2)
    assert exp._verify_v3_parent_checkpoint_reference(v3, reference) == checkpoint
    result = {
        "artifacts": {
            "preprocessing": preprocessing_reference,
            "checkpoint": reference,
        }
    }
    assert exp._verify_v3_parent_model_artifacts(v3, result) == (
        preprocessing,
        checkpoint,
    )
    for consumer in (
        exp._snapshot_v3_full_baselines,
        exp._predict_v3_incumbent_challenge,
    ):
        source = inspect.getsource(consumer)
        assert "_verify_v3_parent_model_artifacts" in source
        assert "_verify_reference(v3_run" not in source
    # Enriched parent receipts never broaden the generic v5 reference schema.
    with pytest.raises(exp.GPUOutagePORDV5Error, match="reference schema"):
        exp._verify_reference(v3, reference)

    attacks = (
        {key: value for key, value in reference.items() if key != "tensor_count"},
        {**reference, "tensor_count": 1},
        {**reference, "tensor_count": True},
        {**reference, "unexpected": 1},
        {**reference, "rows": 1},
    )
    for attacked in attacks:
        with pytest.raises(exp.GPUOutagePORDV5Error):
            exp._verify_v3_parent_checkpoint_reference(v3, attacked)

    original_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(original_bytes + b"tamper")
    with pytest.raises(exp.GPUOutagePORDV5Error, match="reference changed"):
        exp._verify_v3_parent_checkpoint_reference(v3, reference)
    checkpoint.write_bytes(original_bytes)

    # Even a hash/size-rebound archive must retain valid tensor names/values.
    np.savez_compressed(checkpoint, **{"": np.ones(1, dtype=np.float32)})
    empty_name_reference = exp._artifact_reference(
        checkpoint, v3, tensor_count=1
    )
    with pytest.raises(exp.GPUOutagePORDV5Error, match="tensor names"):
        exp._verify_v3_parent_checkpoint_reference(v3, empty_name_reference)
    np.savez_compressed(
        checkpoint, invalid=np.asarray([np.nan], dtype=np.float32)
    )
    nonfinite_reference = exp._artifact_reference(
        checkpoint, v3, tensor_count=1
    )
    with pytest.raises(exp.GPUOutagePORDV5Error, match="NPZ is invalid"):
        exp._verify_v3_parent_checkpoint_reference(v3, nonfinite_reference)
    np.savez_compressed(
        checkpoint, invalid=np.asarray([True], dtype=np.bool_)
    )
    boolean_reference = exp._artifact_reference(
        checkpoint, v3, tensor_count=1
    )
    with pytest.raises(exp.GPUOutagePORDV5Error, match="tensor names"):
        exp._verify_v3_parent_checkpoint_reference(v3, boolean_reference)


def test_prepared_manifest_is_closed_and_count_types_are_exact(tmp_path: Path) -> None:
    run, _, _ = _prepare_synthetic_run(tmp_path)
    manifest_path = run / "prepared_manifest.json"
    original = json.loads(manifest_path.read_text())
    for attacked in (
        {**original, "unexpected": True},
        {**original, "fit_count": float(original["fit_count"])},
        {**original, "fit_count": True},
    ):
        atomic_write_json(manifest_path, attacked)
        with pytest.raises(exp.GPUOutagePORDV5Error, match="prepared manifest"):
            exp.validate_pord_v5_prepared_run(
                run,
                verify_current_source=False,
                allow_synthetic_test_only=True,
                verify_live_environment=False,
            )
    atomic_write_json(manifest_path, original)

    task = exp.pord_v5_development_tasks(_config())[0]
    worker = {
        **_gpu_devices()[0],
        "logical_worker_id": "gpu0_worker0",
        "worker_slot": 0,
    }
    runtime = {
        "execution_kind": "synthetic_test_only_spawn_transport",
        "transport_bound": True,
        "logical_worker_id": worker["logical_worker_id"],
        "process_id": 12345,
        "physical_gpu_id": worker["physical_gpu_id"],
        "uuid": worker["uuid"],
        "pci_bus_id": worker["pci_bus_id"],
        "name": worker["name"],
        "cuda_visible_devices": exp._canonical_cuda_uuid_selector(worker["uuid"]),
        "visible_device_index": 0,
    }
    receipt = exp._task_execution_payload(
        run_dir=run,
        config=_config(),
        task=task,
        worker=worker,
        workers_per_gpu=1,
        runtime_gpu_receipt=runtime,
    )
    assert receipt["worker_development_label_authority"] == (
        "full_snapshot_in_process_not_capability_isolated"
    )
    assert receipt["trainer_outer_array_surface"] == "absent"
    assert receipt["outer_nonuse_enforcement"] == (
        "code_order_and_durable_phase_receipts"
    )


def test_exact_parquet_reader_rejects_arrow_dtype_narrowing(
    tmp_path: Path,
) -> None:
    columns = [
        "trial_uid",
        "segment_id",
        FUSION_COLUMNS[0],
        AVAILABILITY_NAMES[0],
    ]
    canonical = pd.DataFrame(
        {
            "trial_uid": ["row-0", "row-1"],
            "segment_id": np.asarray([36, 37], dtype=np.int64),
            FUSION_COLUMNS[0]: np.asarray([0.0, 1.0], dtype=np.float64),
            AVAILABILITY_NAMES[0]: np.asarray([1, 0], dtype=np.int64),
        }
    )
    canonical_path = atomic_write_parquet(tmp_path / "canonical.parquet", canonical)
    assert exp._read_exact_parquet(canonical_path, columns).equals(canonical)

    attacks = {
        "int32_segment": ("segment_id", np.int32),
        "float32_feature": (FUSION_COLUMNS[0], np.float32),
        "int8_availability": (AVAILABILITY_NAMES[0], np.int8),
    }
    for name, (column, dtype) in attacks.items():
        attacked = canonical.copy()
        attacked[column] = attacked[column].astype(dtype)
        path = atomic_write_parquet(tmp_path / f"{name}.parquet", attacked)
        with pytest.raises(exp.GPUOutagePORDV5Error, match="Arrow dtype"):
            exp._read_exact_parquet(path, columns)


def test_development_snapshot_rejects_resigned_nullability_and_metadata_drift(
    tmp_path: Path,
) -> None:
    base_run, _, _ = _prepare_synthetic_run(tmp_path)
    for attack in (
        "nullable_false",
        "extra_schema_metadata",
        "mutated_pandas_metadata",
        "reformatted_pandas_metadata_bytes",
    ):
        run = tmp_path / f"resigned_{attack}"
        shutil.copytree(base_run, run)
        receipt_path = run / "inputs" / "development_replay.json"
        receipt = json.loads(receipt_path.read_text())
        parquet_path = exp._verify_reference(run, receipt["input_file"])
        table = pq.read_table(parquet_path)
        fields = list(table.schema)
        metadata = dict(table.schema.metadata or {})
        if attack == "nullable_false":
            original = fields[0]
            fields[0] = pa.field(
                original.name,
                original.type,
                nullable=False,
                metadata=original.metadata,
            )
        elif attack == "extra_schema_metadata":
            metadata[b"attacker"] = b"arbitrary-re-signed-metadata"
        else:
            pandas_metadata = json.loads(metadata[b"pandas"].decode("utf-8"))
            if attack == "mutated_pandas_metadata":
                pandas_metadata["pandas_version"] = "attacker-version"
                metadata[b"pandas"] = json.dumps(pandas_metadata).encode("utf-8")
            else:
                metadata[b"pandas"] = json.dumps(
                    pandas_metadata, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
        attacked_schema = pa.schema(fields, metadata=metadata)
        attacked_table = pa.Table.from_arrays(
            [table.column(index) for index in range(table.num_columns)],
            schema=attacked_schema,
        )
        pq.write_table(attacked_table, parquet_path)
        receipt["input_file"]["bytes"] = parquet_path.stat().st_size
        receipt["input_file"]["sha256"] = hashlib.sha256(
            parquet_path.read_bytes()
        ).hexdigest()
        atomic_write_json(receipt_path, receipt)

        with pytest.raises(
            exp.GPUOutagePORDV5Error, match="nullability|metadata"
        ):
            exp._load_development_snapshot(run)


def test_default_executor_spy_fixes_checkpoint_before_one_outer_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = exp.pord_v5_development_tasks(_config())[0]
    worker = {
        **_gpu_devices()[0],
        "logical_worker_id": "gpu0_worker0",
        "worker_slot": 0,
    }
    segments = np.repeat(np.asarray([36, 37, 38, 39]), 2)
    rows = len(segments)
    data = {
        "X": np.arange(rows * 43, dtype=np.float64).reshape(rows, 43) / 100.0,
        "y": np.arange(rows * 3, dtype=np.float64).reshape(rows, 3),
        "trial_uid": np.asarray([f"spy-{index}" for index in range(rows)]),
        "segment_id": segments,
        "target_id": np.asarray([f"target-{index}" for index in range(rows)]),
    }
    events: list[str] = []
    state = {"weight": np.asarray([1.0], dtype=np.float32)}

    def trainer_spy(**kwargs: Any) -> Mapping[str, Any]:
        events.append("train_checkpoint_fixed")
        assert not {
            "X_holdout_scaled",
            "y_holdout_scaled",
            "holdout_trial_uids",
            "X_outer_scaled",
            "y_outer_raw_mm",
        }.intersection(kwargs)
        return {
            "backbone_id": task["backbone_id"],
            "method_id": task["method_id"],
            "epoch_budget": 400,
            "optimizer_update_count": 400,
            "contract_epoch_budget_complete": True,
            "initial_state_sha256": "a" * 64,
            "selected_state_sha256": state_dict_sha256(state),
                "result_sha256": "b" * 64,
                "training_bundle_sha256": "c" * 64,
                "base_train_view_bundle_sha256": "d" * 64,
                "train_view_bundle_sha256": "d" * 64,
                "availability_semantics": "physical_mask_state",
                "gradient_view_semantics": "full_view_gradient_only_outage_selected_checkpoint",
                "pre_registered_wrong_pair_map_sha256": "1" * 64,
                "pairing_mode": "none",
                "full_reference_detached": True,
            "target_transform_sha256": "e" * 64,
            "target_mean_mm": kwargs["target_mean_mm"].tolist(),
            "target_scale_mm": kwargs["target_scale_mm"].tolist(),
            "best_epoch": 1,
            "best_checkpoint_score": 1.0,
            "checkpoint_selection_scope": "training_segments_only",
            "checkpoint_criterion": exp.PORD_CHECKPOINT_CRITERION,
            "same_checkpoint_for_all_four_outages": True,
            "history": [
                {
                    "epoch": epoch,
                    "checkpoint_selection_scope": "training_segments_only",
                    "outer_eval_count": 0,
                    "outer_labels_accessed": False,
                }
                for epoch in range(1, 401)
            ],
            "state_dict": state,
        }

    outer_calls = 0

    def prediction_spy(**kwargs: Any) -> Mapping[str, Any]:
        assert events == ["train_checkpoint_fixed"]
        events.append("outer_predicted_without_labels")
        count = len(kwargs["outer_trial_uids"])
        return {
                "receipt": {
                    "base_outer_view_bundle_sha256": "f" * 64,
                    "outer_view_bundle_sha256": "f" * 64,
                    "availability_semantics": "physical_mask_state",
                "outer_labels_accessed": False,
            },
            "trial_uids": list(kwargs["outer_trial_uids"]),
            "predictions_raw_mm": np.zeros((5, count, 3), dtype=np.float64),
        }

    def outer_spy(**kwargs: Any) -> Mapping[str, Any]:
        nonlocal outer_calls
        assert events == [
            "train_checkpoint_fixed",
            "outer_predicted_without_labels",
        ]
        outer_calls += 1
        events.append("outer_evaluated")
        truth = np.asarray(kwargs["y_outer_raw_mm"], dtype=np.float64)
        predictions = np.asarray(
            kwargs["outer_prediction"]["predictions_raw_mm"], dtype=np.float64
        )
        return {
                "receipt": {
                    "base_outer_view_bundle_sha256": "f" * 64,
                    "outer_view_bundle_sha256": "f" * 64,
                    "availability_semantics": "physical_mask_state",
                "outer_evaluation_sha256": "0" * 64,
                "checkpoint_fixed_before_outer_labels": True,
                "outer_eval_count": 1,
            },
            "truth_raw_mm": truth,
            "predictions_raw_mm": predictions,
        }

    monkeypatch.setattr(exp, "train_pord_fold_seed", trainer_spy)
    monkeypatch.setattr(exp, "predict_pord_checkpoint_outer", prediction_spy)
    monkeypatch.setattr(exp, "evaluate_pord_outer_predictions_once", outer_spy)
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES", exp._canonical_cuda_uuid_selector(worker["uuid"])
    )
    output = exp._default_task_executor(
        task=task,
        data=data,
        worker=worker,
        config=_config(),
    )
    assert events == [
        "train_checkpoint_fixed",
        "outer_predicted_without_labels",
        "outer_evaluated",
    ]
    assert outer_calls == 1
    assert output["training"]["outer_eval_count"] == 1
    assert output["training"]["checkpoint_selection_scope"] == (
        "training_segments_only"
    )


def test_resume_writes_outer_start_before_any_y_access_then_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = exp.pord_v5_development_tasks(_config())[0]
    run = tmp_path / "marker_before_y"
    task_dir = exp._task_dir(run, task)
    task_dir.mkdir(parents=True)
    execution_path = atomic_write_json(
        task_dir / "execution.started.json", {"phase": "execution.started"}
    )
    core = {
        "history": [],
        "state_dict": {"weight": np.asarray([1.0], dtype=np.float32)},
        "result_sha256": "a" * 64,
        "training_bundle_sha256": "b" * 64,
        "selected_state_sha256": "c" * 64,
        "initial_state_sha256": "d" * 64,
        "checkpoint_selection_scope": "training_segments_only",
        "outer_eval_count": 0,
        "outer_labels_accessed": False,
    }
    exp._production_task_phase_callback(
        run_dir=run,
        task=task,
        execution_path=execution_path,
        runtime_gpu_receipt={},
    )("training.completed", {"core": core, "preprocessing": {}})

    class FrozenPreprocessor:
        target_mean_ = np.zeros(3, dtype=np.float64)
        target_scale_ = 1.0

        @staticmethod
        def transform_features(values: Any) -> np.ndarray:
            return np.asarray(values, dtype=np.float64)

    started_path = task_dir / "outer_eval.started.json"
    events: list[str] = []

    def callback_factory(**_: Any) -> Any:
        def callback(phase: str, __: Mapping[str, Any]) -> None:
            events.append(phase)
            if phase == "outer_eval.started":
                assert data.y_access_count == 0
                exp._exclusive_write_json(started_path, {"phase": phase})

        return callback

    class MarkerGuardedData(dict[str, Any]):
        y_access_count = 0

        def __getitem__(self, key: str) -> Any:
            if key == "y":
                self.y_access_count += 1
                assert started_path.is_file()
                raise RuntimeError("simulated crash after marker before evaluation")
            return super().__getitem__(key)

    segments = np.asarray([36, 37, 38, 39], dtype=np.int64)
    data = MarkerGuardedData(
        X=np.zeros((4, 43), dtype=np.float64),
        segment_id=segments,
        trial_uid=np.asarray([f"uid-{index}" for index in range(4)]),
        target_id=np.asarray([f"target-{index}" for index in range(4)]),
    )
    monkeypatch.setattr(exp, "verify_pord_result_hash", lambda _: None)
    monkeypatch.setattr(
        exp, "_preprocessor_from_metadata", lambda _: FrozenPreprocessor()
    )
    monkeypatch.setattr(exp, "_production_task_phase_callback", callback_factory)
    monkeypatch.setattr(
        exp,
        "predict_pord_checkpoint_outer",
        lambda **kwargs: {
            "receipt": {"outer_labels_accessed": False},
            "trial_uids": list(kwargs["outer_trial_uids"]),
            "predictions_raw_mm": np.zeros(
                (5, len(kwargs["outer_trial_uids"]), 3), dtype=np.float64
            ),
        },
    )

    with pytest.raises(RuntimeError, match="simulated crash"):
        exp._resume_production_task_phases(
            run_dir=run,
            task=task,
            data=data,
            execution_path=execution_path,
            runtime_gpu_receipt={},
        )
    assert events == ["outer_prediction.completed", "outer_eval.started"]
    assert data.y_access_count == 1
    with pytest.raises(
        exp.GPUOutagePORDV5Error,
        match="outer_eval.started without authenticated completion",
    ):
        exp._resume_production_task_phases(
            run_dir=run,
            task=task,
            data=data,
            execution_path=execution_path,
            runtime_gpu_receipt={},
        )
    assert data.y_access_count == 1


def test_completed_outer_phase_resume_never_reads_truth_or_calls_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = exp.pord_v5_development_tasks(_config())[0]
    held_out = task["held_out_segment"]
    segments = np.repeat(np.asarray([36, 37, 38, 39], dtype=np.int64), 2)
    rows = len(segments)
    data = {
        "X": (
            np.arange(rows * 43, dtype=np.float64).reshape(rows, 43) / 97.0
        ),
        "y": np.arange(rows * 3, dtype=np.float64).reshape(rows, 3) / 11.0,
        "trial_uid": np.asarray([f"resume-{index}" for index in range(rows)]),
        "segment_id": segments,
        "target_id": np.asarray([f"target-{index}" for index in range(rows)]),
    }
    eval_rows = segments == held_out
    train_rows = ~eval_rows
    preprocessor = exp._fit_v5_preprocessor(data, held_out_segment=held_out)
    target_mean = np.asarray(preprocessor.target_mean_, dtype=np.float64)
    target_scale = np.repeat(float(preprocessor.target_scale_), 3)
    core = exp.train_pord_fold_seed(
        X_train_scaled=preprocessor.transform_features(data["X"][train_rows]).astype(
            np.float32
        ),
        y_train_scaled=preprocessor.transform_targets(data["y"][train_rows]).astype(
            np.float32
        ),
        train_segment_ids=segments[train_rows],
        train_trial_uids=data["trial_uid"][train_rows].tolist(),
        held_out_segment=held_out,
        target_mean_mm=target_mean,
        target_scale_mm=target_scale,
        method_id=task["method_id"],
        model_seed=task["model_seed"],
        device="cpu",
        backbone_id=task["backbone_id"],
        epoch_budget=400,
    )
    run = tmp_path / "phase_resume"
    task_dir = exp._task_dir(run, task)
    task_dir.mkdir(parents=True)
    execution_path = atomic_write_json(
        task_dir / "execution.started.json", {"phase": "execution.started"}
    )
    callback = exp._production_task_phase_callback(
        run_dir=run,
        task=task,
        execution_path=execution_path,
    )
    callback(
        "training.completed",
        {"core": core, "preprocessing": preprocessor.metadata()},
    )
    outer_uids = data["trial_uid"][eval_rows].tolist()
    prediction = exp.predict_pord_checkpoint_outer(
        training_result=core,
        X_outer_scaled=preprocessor.transform_features(data["X"][eval_rows]).astype(
            np.float32
        ),
        outer_segment_ids=segments[eval_rows],
        outer_trial_uids=outer_uids,
        held_out_segment=held_out,
        target_mean_mm=target_mean,
        target_scale_mm=target_scale,
        device="cpu",
    )
    callback(
        "outer_prediction.completed",
        {
            "outer_prediction": prediction,
            "trial_uid": outer_uids,
            "segment_id": segments[eval_rows].tolist(),
            "target_id": data["target_id"][eval_rows].tolist(),
        },
    )
    callback("outer_eval.started", {"outer_prediction_receipt": prediction["receipt"]})
    evaluated = exp.evaluate_pord_outer_predictions_once(
        training_result=core,
        outer_prediction=prediction,
        y_outer_raw_mm=data["y"][eval_rows],
    )
    callback(
        "outer_eval.completed",
        {
            "outer_evaluation": evaluated["receipt"],
            "truth_raw_mm": evaluated["truth_raw_mm"],
            "predictions_raw_mm": evaluated["predictions_raw_mm"],
        },
    )

    def forbidden_evaluator(**_: Any) -> Mapping[str, Any]:
        raise AssertionError("completed resume must not evaluate or read held-out truth")

    monkeypatch.setattr(
        exp, "evaluate_pord_outer_predictions_once", forbidden_evaluator
    )
    recovered = exp._resume_production_task_phases(
        run_dir=run,
        task=task,
        data={},
        execution_path=execution_path,
        runtime_gpu_receipt={},
    )
    assert recovered["training"]["outer_eval_count"] == 1
    assert np.array_equal(recovered["truth_mm"], evaluated["truth_raw_mm"])
    assert np.array_equal(
        recovered["predictions_mm"], evaluated["predictions_raw_mm"]
    )


def test_row_positive_excess_is_paired_mean_p95_max_fraction() -> None:
    truth = np.zeros((4, 3), dtype=np.float64)
    full = np.asarray([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=np.float64)
    outage = np.asarray([[2, 0, 0], [1, 0, 0], [5, 0, 0], [4, 0, 0]], dtype=np.float64)
    summary = exp.row_positive_excess_summary(truth, full, outage)
    positive = np.asarray([1.0, 0.0, 2.0, 0.0])
    assert summary == {
        "mean_mm": pytest.approx(float(positive.mean())),
        "p95_mm": pytest.approx(float(np.percentile(positive, 95))),
        "max_mm": pytest.approx(2.0),
        "positive_fraction": pytest.approx(0.5),
    }


def _claim_score_inputs(
    overrides: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    errors = {
        "masked_mlp_64x64_relu__full_gradient_only_mask_selected_control": 2.5,
        "masked_mlp_64x64_relu__masked_erm": 2.0,
        "masked_mlp_64x64_relu__constant_availability_erm": 2.1,
        "masked_mlp_64x64_relu__group_mean_pord": 1.3,
        "masked_mlp_64x64_relu__paired_tail": 1.2,
        "masked_mlp_64x64_relu__paired_interaction": 1.25,
        "masked_mlp_64x64_relu__paired_pord": 1.0,
        "masked_mlp_64x64_relu__paired_wrong_pair": 1.4,
        "masked_mlp_64x64_relu__paired_no_stopgrad": 1.15,
        "masked_mlp_64x64_relu__groupdro": 1.4,
        "masked_mlp_64x64_relu__vrex": 1.5,
        "masked_mlp_64x64_relu__mofe": 1.6,
    }
    if overrides is not None:
        errors.update(overrides)
    score = exp.score_pord_v5_candidates(
        _metric_contexts(errors), candidate_ids=tuple(errors)
    )
    seeds = {
        seed: exp.score_pord_v5_candidates(
            _metric_contexts(errors), candidate_ids=tuple(errors)
        )["trace"]
        for seed in range(3)
    }
    return score["trace"], seeds


def test_all_preregistered_claim_gates_and_both_full_gates() -> None:
    trace, seeds = _claim_score_inputs()
    record = exp.evaluate_pord_v5_claim_gates(
        score_trace=trace,
        seed_score_traces=seeds,
        full_fold_rmse_by_candidate={
            "masked_mlp_64x64_relu__paired_pord": [0.5] * 4,
            "masked_mlp_64x64_relu__masked_erm": [0.6] * 4,
        },
        frozen_v3_masked_full_fold_rmse=[0.5] * 4,
    )
    assert record["all_claim_gates_passed"] is True
    assert len(record["gates"]) == 15
    attacked = exp.evaluate_pord_v5_claim_gates(
        score_trace=trace,
        seed_score_traces=seeds,
        full_fold_rmse_by_candidate={
            "masked_mlp_64x64_relu__paired_pord": [0.7] * 4,
            "masked_mlp_64x64_relu__masked_erm": [0.6] * 4,
        },
        frozen_v3_masked_full_fold_rmse=[0.5] * 4,
    )
    assert attacked["all_claim_gates_passed"] is False
    assert attacked["gates"]["same_v5_masked_erm_full_gate"] is False


@pytest.mark.parametrize(
    ("candidate_id", "gate"),
    [
        (
            "masked_mlp_64x64_relu__paired_tail",
            "strict_j_lower_than_paired_tail",
        ),
        (
            "masked_mlp_64x64_relu__paired_interaction",
            "strict_j_lower_than_paired_interaction",
        ),
        (
            "masked_mlp_64x64_relu__paired_no_stopgrad",
            "strict_j_lower_than_paired_no_stopgrad",
        ),
    ],
)
def test_combined_objective_and_stop_gradient_gates_are_strict(
    candidate_id: str, gate: str
) -> None:
    trace, seeds = _claim_score_inputs({candidate_id: 0.9})
    record = exp.evaluate_pord_v5_claim_gates(
        score_trace=trace,
        seed_score_traces=seeds,
        full_fold_rmse_by_candidate={
            "masked_mlp_64x64_relu__paired_pord": [0.5] * 4,
            "masked_mlp_64x64_relu__masked_erm": [0.6] * 4,
        },
        frozen_v3_masked_full_fold_rmse=[0.5] * 4,
    )
    assert record["gates"][gate] is False
    assert record["all_claim_gates_passed"] is False


def _selection_trace(candidate_id: str, j: float) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "outage_robust_j_mm": j,
        "mean_p95_3d_mm": j + 0.1,
        "mean_seed_rmse_population_std_mm": 0.01,
    }


def test_selection_claim_refusal_novelty_failure_and_eligible_tie_policy() -> None:
    incumbent = _selection_trace("frozen_v4_incumbent", 1.2)
    pord = _selection_trace("masked_mlp_64x64_relu__paired_pord", 1.0)
    blend = _selection_trace("fixed_half_v4_incumbent_half_pord", 1.1)
    refused = exp.select_pord_v5_novel_candidate(
        claim_gate_record={"all_claim_gates_passed": False},
        incumbent_trace=incumbent,
        pord_trace=pord,
        blend_trace=blend,
    )
    assert refused["challenge_finalize_allowed"] is False
    failed = exp.select_pord_v5_novel_candidate(
        claim_gate_record={"all_claim_gates_passed": True},
        incumbent_trace=incumbent,
        pord_trace={**pord, "outage_robust_j_mm": 1.3},
        blend_trace={**blend, "outage_robust_j_mm": 1.4},
    )
    assert failed["stage"] == "novelty_gate_failed_challenge_sealed"
    selected = exp.select_pord_v5_novel_candidate(
        claim_gate_record={"all_claim_gates_passed": True},
        incumbent_trace=incumbent,
        pord_trace=pord,
        blend_trace=blend,
    )
    assert selected["selected_candidate_id"] == "masked_mlp_64x64_relu__paired_pord"
    assert selected["challenge_used_for_selection"] is False


def _development_frame() -> pd.DataFrame:
    segments = np.concatenate(
        [np.repeat(segment, count) for segment, count in SEGMENT_COUNTS.items()]
    ).astype(np.int64)
    rows = len(segments)
    frame = pd.DataFrame(
        {
            "trial_uid": [f"development_uid_{index:05d}" for index in range(rows)],
            "segment_id": segments,
            "target_id": [f"target_{index % 7}" for index in range(rows)],
        }
    )
    base = np.arange(rows, dtype=np.float64)
    for index, name in enumerate(FUSION_COLUMNS):
        frame[name] = (base % (index + 5)) * 0.001
    frame["true_x_mm"] = (base % 17) * 0.01
    frame["true_y_mm"] = (base % 13) * 0.01
    frame["true_z_mm"] = (base % 11) * 0.01
    return frame


def _challenge_frame() -> pd.DataFrame:
    segments = np.concatenate(
        [np.repeat(segment, count) for segment, count in CHALLENGE_COUNTS.items()]
    ).astype(np.int64)
    rows = len(segments)
    frame = pd.DataFrame(
        {
            "trial_uid": [f"challenge_uid_{index:05d}" for index in range(rows)],
            "segment_id": segments,
            "target_id": [f"target_{index % 7}" for index in range(rows)],
        }
    )
    for index, name in enumerate(FUSION_COLUMNS):
        frame[name] = np.full(rows, index * 0.001, dtype=np.float64)
    availability = np.tile(np.asarray((1, 0, 1, 1), dtype=np.int64), (rows, 1))
    for index, name in enumerate(AVAILABILITY_NAMES):
        frame[name] = availability[:, index]
    return frame


def _parent_snapshotter_factory(
    development_frame: pd.DataFrame, challenge_frame: pd.DataFrame
):
    def snapshotter(
        *, v3_run: Path, v4_run: Path, v5_run: Path, config: Mapping[str, Any]
    ) -> dict[str, Any]:
        del v3_run, v4_run, config
        development_path = atomic_write_parquet(
            v5_run / "inputs" / "development_replay.parquet", development_frame
        )
        uid_hash = hashlib.sha256(
            "\n".join(sorted(development_frame["trial_uid"].tolist())).encode()
        ).hexdigest()
        development_meta = {
            "role": "v4_authenticated_development_replay_snapshot",
            "rows": 6635,
            "segments": [36, 37, 38, 39],
            "segment_rows": [1672, 1597, 1707, 1659],
            "feature_names": list(FUSION_COLUMNS),
            "label_names": list(TRUE_COLUMNS),
            "sorted_trial_uid_sha256": uid_hash,
            "input_file": exp._artifact_reference(development_path, v5_run, rows=6635),
        }
        development_meta_path = atomic_write_json(
            v5_run / "inputs" / "development_replay.json", development_meta
        )
        challenge_path = atomic_write_parquet(
            v5_run / "inputs" / "challenge_identity_inputs.parquet", challenge_frame
        )
        challenge_uid_hash = hashlib.sha256(
            "\n".join(challenge_frame["trial_uid"].tolist()).encode()
        ).hexdigest()
        challenge_meta = {
            "role": "v4_label_blind_challenge_identity_snapshot",
            "rows": 3295,
            "segments": [40, 41],
            "segment_rows": [1672, 1623],
            "feature_names": list(FUSION_COLUMNS),
            "availability_names": list(AVAILABILITY_NAMES),
            "labels_projected": False,
            "recorded_imu1_projected": False,
            "ordered_trial_uid_sha256": challenge_uid_hash,
            "identity_file": exp._artifact_reference(challenge_path, v5_run, rows=3295),
        }
        challenge_meta_path = atomic_write_json(
            v5_run / "inputs" / "challenge_identity.json", challenge_meta
        )
        snapshots: list[dict[str, Any]] = []
        contexts: list[dict[str, Any]] = []
        for segment in (36, 37, 38, 39):
            rows = development_frame["segment_id"].eq(segment)
            truth = development_frame.loc[rows, list(TRUE_COLUMNS)].to_numpy(dtype=np.float64)
            for outage in range(4):
                seed_predictions = []
                for seed in range(3):
                    prediction = truth.copy()
                    prediction[:, 0] += 1.2
                    frame = development_frame.loc[
                        rows, ["trial_uid", "segment_id", "target_id", *TRUE_COLUMNS]
                    ].copy()
                    for axis, name in enumerate(PRED_COLUMNS):
                        frame[name] = prediction[:, axis]
                    path = atomic_write_parquet(
                        v5_run
                        / "parents"
                        / "v4_incumbent_seed_predictions"
                        / f"segment{segment}__drop{outage}__seed{seed}.parquet",
                        frame,
                    )
                    snapshots.append(
                        {
                            "held_out_segment": segment,
                            "forced_outage_index": outage,
                            "paired_seed_index": seed,
                            "prediction_file": exp._artifact_reference(path, v5_run, rows=len(frame)),
                        }
                    )
                    seed_predictions.append(prediction)
                ensemble = np.mean(np.stack(seed_predictions), axis=0)
                contexts.append(
                    {
                        "candidate_id": "frozen_v4_incumbent",
                        "held_out_segment": segment,
                        "forced_outage_index": outage,
                        "metrics": regression_metrics(truth, ensemble),
                        "seed_rmse_population_std_mm": 0.0,
                    }
                )
        return {
            "inputs": {
                "development_metadata": exp._artifact_reference(development_meta_path, v5_run),
                "challenge_identity_metadata": exp._artifact_reference(challenge_meta_path, v5_run),
            },
            "v4_selected_candidate_id": "xgboost_mask_replicated",
            "v4_incumbent_seed_prediction_count": 48,
            "v4_incumbent_seed_predictions": snapshots,
            "v4_incumbent_contexts": contexts,
            "v3_same_backbone_full_baselines": {
                "masked_mlp_64x64_relu": {
                    "source_forced_outage_index": 1,
                    "fold_rmse_3d_mm": [0.75] * 4,
                },
                "chain_endpoint_only": {
                    "source_forced_outage_index": 1,
                    "fold_rmse_3d_mm": [1.5] * 4,
                },
            },
            "component_recipe": {"synthetic_test": True},
        }

    return snapshotter


def _validators() -> tuple[Any, Any]:
    def v3(_: Path) -> dict[str, Any]:
        return {
            "status": "ok",
            "stage": "development_complete_challenge_labels_sealed",
            "challenge_labels_sealed": True,
            "development_seed_fits": 240,
            "scored_contexts": 96,
        }

    def v4(_: Path) -> dict[str, Any]:
        return {
            "status": "ok",
            "stage": "development_complete_challenge_labels_sealed",
            "challenge_labels_sealed": True,
            "xgb_unique_bundle_fits": 12,
            "scored_contexts": 48,
            "challenge_label_access_passes": 0,
        }

    return v3, v4


def _prepare_synthetic_run(tmp_path: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    development = _development_frame()
    challenge = _challenge_frame()
    v3_run = tmp_path / "parent_v3"
    v4_run = tmp_path / "parent_v4"
    run = tmp_path / "outage_v5"
    for parent in (v3_run, v4_run):
        parent.mkdir(parents=True)
        atomic_write_json(
            parent / "development_manifest.json",
            {"stage": "development_complete_challenge_labels_sealed"},
        )
    v3_validator, v4_validator = _validators()
    source = {"files": [], "fingerprint": stable_hash([])}
    environment_payload = {
        "python_implementation": "cpython",
        "python_version": "3.12.synthetic",
        "cuda_devices": _gpu_devices(),
    }
    exp.prepare_pord_v5_run(
        _config(),
        v3_run_dir=v3_run,
        v4_run_dir=v4_run,
        run_dir=run,
        v3_validator=v3_validator,
        v4_validator=v4_validator,
        parent_snapshotter=_parent_snapshotter_factory(development, challenge),
        source_payload=source,
        environment_payload=environment_payload,
        _private_test_only_mode=True,
    )
    return run, development, challenge


OUTAGE_ERROR = {
    "masked_mlp_64x64_relu__full_gradient_only_mask_selected_control": 2.5,
    "masked_mlp_64x64_relu__masked_erm": 2.0,
    "masked_mlp_64x64_relu__constant_availability_erm": 2.1,
    "masked_mlp_64x64_relu__group_mean_pord": 1.3,
    "masked_mlp_64x64_relu__paired_tail": 1.2,
    "masked_mlp_64x64_relu__paired_interaction": 1.25,
    "masked_mlp_64x64_relu__paired_pord": 1.0,
    "masked_mlp_64x64_relu__paired_wrong_pair": 1.4,
    "masked_mlp_64x64_relu__paired_no_stopgrad": 1.15,
    "masked_mlp_64x64_relu__groupdro": 1.5,
    "masked_mlp_64x64_relu__vrex": 1.6,
    "masked_mlp_64x64_relu__mofe": 1.7,
    "chain_endpoint_only__masked_erm": 2.4,
    "chain_endpoint_only__paired_pord": 2.0,
}
FULL_ERROR = {
    "masked_mlp_64x64_relu__full_gradient_only_mask_selected_control": 0.65,
    "masked_mlp_64x64_relu__masked_erm": 0.6,
    "masked_mlp_64x64_relu__constant_availability_erm": 0.62,
    "masked_mlp_64x64_relu__group_mean_pord": 0.55,
    "masked_mlp_64x64_relu__paired_tail": 0.55,
    "masked_mlp_64x64_relu__paired_interaction": 0.55,
    "masked_mlp_64x64_relu__paired_pord": 0.5,
    "masked_mlp_64x64_relu__paired_wrong_pair": 0.56,
    "masked_mlp_64x64_relu__paired_no_stopgrad": 0.54,
    "masked_mlp_64x64_relu__groupdro": 0.55,
    "masked_mlp_64x64_relu__vrex": 0.55,
    "masked_mlp_64x64_relu__mofe": 0.55,
    "chain_endpoint_only__masked_erm": 1.2,
    "chain_endpoint_only__paired_pord": 1.1,
}


def _synthetic_task_executor(
    *, task: Mapping[str, Any], data: Mapping[str, Any], worker: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    del config
    # Deliberately heterogeneous completion order exercises thread scheduling
    # without changing the ordered scientific task/result grid.
    time.sleep((int(task["paired_seed_index"]) + int(task["held_out_segment"])) % 3 * 0.0005)
    segment = int(task["held_out_segment"])
    rows = np.asarray(data["segment_id"]) == segment
    truth = np.asarray(data["y"], dtype=np.float64)[rows]
    predictions = np.repeat(truth[None, :, :], 5, axis=0)
    predictions[0, :, 0] += FULL_ERROR[str(task["candidate_id"])]
    predictions[1:, :, 0] += OUTAGE_ERROR[str(task["candidate_id"])]
    checkpoint_state = {
        "weight": np.asarray(
            [OUTAGE_ERROR[str(task["candidate_id"])], task["paired_seed_index"]],
            dtype=np.float32,
        )
    }
    pairing = {
        "backbone": task["backbone_id"],
        "segment": segment,
        "seed": task["paired_seed_index"],
    }
    uids = np.asarray(data["trial_uid"])[rows].astype(str).tolist()
    training_core_result_sha256 = stable_hash({**pairing, "kind": "training_core"})
    method = str(task["method_id"])
    availability_semantics = (
        "constant_all_available_negative_control"
        if method == "constant_availability_erm"
        else "physical_mask_state"
    )
    base_train_view_bundle_sha256 = stable_hash({**pairing, "kind": "base_train_views"})
    train_view_bundle_sha256 = (
        stable_hash({**pairing, "kind": "constant_train_views"})
        if method == "constant_availability_erm"
        else base_train_view_bundle_sha256
    )
    base_outer_view_bundle_sha256 = stable_hash({**pairing, "kind": "base_outer_views"})
    outer_view_bundle_sha256 = (
        stable_hash({**pairing, "kind": "constant_outer_views"})
        if method == "constant_availability_erm"
        else base_outer_view_bundle_sha256
    )
    target_transform_sha256 = stable_hash(
        {"segment": segment, "target": "isotropic"}
    )
    outer_evaluation = {
        "contract": "alphaf.endpoint_regression.gpu_outage_pord.v5.outer_evaluation",
        "training_result_sha256": training_core_result_sha256,
        "selected_state_sha256": state_dict_sha256(checkpoint_state),
        "backbone_id": task["backbone_id"],
        "held_out_segment": str(segment),
        "checkpoint_selection_scope": "training_segments_only",
        "checkpoint_fixed_before_outer_labels": True,
        "outer_eval_count": 1,
        "base_outer_view_bundle_sha256": base_outer_view_bundle_sha256,
        "outer_view_bundle_sha256": outer_view_bundle_sha256,
        "availability_semantics": availability_semantics,
        "outer_trial_uid_sha256": stable_hash(uids),
        "outer_truth_raw_mm_sha256": exp.array_sha256(truth),
        "outer_prediction_raw_mm_sha256": exp.array_sha256(predictions),
        "target_transform_sha256": target_transform_sha256,
        "evaluated_view_names": [
            "full",
            "drop_imu0",
            "drop_imu1",
            "drop_imu2",
            "drop_imu3",
        ],
        "same_checkpoint_for_all_four_outages": True,
        "outer_full_rmse_3d_mm": regression_metrics(
            truth, predictions[0]
        )["rmse_3d_mm"],
        "outer_outage_rmse_3d_mm": [
            regression_metrics(truth, predictions[index])["rmse_3d_mm"]
            for index in range(1, 5)
        ],
    }
    outer_evaluation["outer_evaluation_sha256"] = stable_hash(
        {
            key: outer_evaluation[key]
            for key in exp._OUTER_EVALUATION_HASH_KEYS
        }
    )
    training = {
        "backbone_id": task["backbone_id"],
        "method_id": task["method_id"],
        "epoch_budget": 400,
        "optimizer_update_count": 400,
        "contract_epoch_budget_complete": True,
        "initial_state_sha256": stable_hash({**pairing, "kind": "initial"}),
        "selected_state_sha256": state_dict_sha256(checkpoint_state),
        "training_core_result_sha256": training_core_result_sha256,
        "training_bundle_sha256": stable_hash({**pairing, "kind": "bundle"}),
        "base_train_view_bundle_sha256": base_train_view_bundle_sha256,
        "train_view_bundle_sha256": train_view_bundle_sha256,
        "availability_semantics": availability_semantics,
        "gradient_view_semantics": (
            "full_view_gradient_only_outage_selected_checkpoint"
            if method == "full_gradient_only_mask_selected_control"
            else "five_view_mask_weighted_gradient"
        ),
        "pre_registered_wrong_pair_map_sha256": stable_hash(
            {**pairing, "kind": "wrong_pair_map"}
        ),
        "pairing_mode": {
            "group_mean_pord": "group_mean_difference_not_pair_specific",
            "paired_tail": "correct_uid_stop_gradient",
            "paired_interaction": "correct_uid_stop_gradient",
            "paired_pord": "correct_uid_stop_gradient",
            "paired_wrong_pair": "sorted_uid_within_segment_circular_derangement",
            "paired_no_stopgrad": "correct_uid_no_stop_gradient",
        }.get(method, "none"),
        "full_reference_detached": method != "paired_no_stopgrad",
        "base_outer_view_bundle_sha256": base_outer_view_bundle_sha256,
        "outer_view_bundle_sha256": outer_view_bundle_sha256,
        "outer_evaluation_sha256": outer_evaluation[
            "outer_evaluation_sha256"
        ],
        "target_transform_sha256": target_transform_sha256,
        "target_mean_mm": [0.0, 0.0, 0.0],
        "target_scale_mm": [1.0, 1.0, 1.0],
        "best_epoch": 3,
        "best_checkpoint_score": OUTAGE_ERROR[str(task["candidate_id"])],
        "checkpoint_selection_scope": "training_segments_only",
        "checkpoint_fixed_before_outer_labels": True,
        "outer_eval_count": 1,
        "same_checkpoint_for_all_four_outages": True,
        "checkpoint_criterion": (
            "0.5*(mean4_training_outage_rmse_3d_mm+"
            "max4_training_outage_rmse_3d_mm)"
        ),
        "exact_five_views": True,
    }
    preprocessing = {
        "feature_names": list(FUSION_COLUMNS),
        "feature_mean": [0.0] * 43,
        "feature_scale": [1.0] * 43,
        "target_mean": [0.0, 0.0, 0.0],
        "target_scale": 1.0,
        "leakage_audit": {
            "split_context": f"outage_v5_pord_segment{segment}",
            "fit_scope": "real_train_only",
            "forbidden_uid_overlap_count": 0,
        },
    }
    return {
        "preprocessing": preprocessing,
        "history": [
            {
                "epoch": epoch,
                "checkpoint_selection_scope": "training_segments_only",
                "outer_eval_count": 0,
                "outer_labels_accessed": False,
            }
            for epoch in range(1, 401)
        ],
        "checkpoint_state": checkpoint_state,
        "training": training,
        "outer_evaluation": outer_evaluation,
        "trial_uid": uids,
        "segment_id": np.asarray(data["segment_id"])[rows].astype(int).tolist(),
        "target_id": np.asarray(data["target_id"])[rows].astype(str).tolist(),
        "truth_mm": truth,
        "predictions_mm": predictions,
    }


@pytest.fixture(scope="module")
def completed_synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("pord_v5_e2e")
    run, development, challenge = _prepare_synthetic_run(root)
    result = exp.run_pord_v5_development(
        run,
        workers_per_gpu=3,
        resume=False,
        task_executor=_synthetic_task_executor,
        parallel=True,
        verify_current_source=False,
        _private_test_only_mode=True,
    )
    return {
        "run": run,
        "development": development,
        "challenge": challenge,
        "result": result,
    }


def test_synthetic_e2e_exact_counts_gates_selection_and_runtime_sentinel(
    completed_synthetic_run: Mapping[str, Any],
) -> None:
    run = Path(completed_synthetic_run["run"])
    with pytest.raises(exp.GPUOutagePORDV5Error, match="synthetic"):
        exp.validate_pord_v5_development_run(
            run,
            verify_current_source=False,
            verify_live_environment=False,
        )
    validation = exp.validate_pord_v5_development_run(
        run,
        verify_current_source=False,
        allow_synthetic_test_only=True,
        verify_live_environment=False,
    )
    assert validation["controlled_fit_count"] == 168
    assert validation["primary_outage_contexts"] == 192
    assert validation["transfer_outage_contexts"] == 32
    assert validation["outage_contexts_total"] == 224
    assert validation["full_contexts_total"] == 56
    assert validation["claim_gates_passed"] is True
    assert validation["selected_candidate_id"] == "masked_mlp_64x64_relu__paired_pord"
    selection = json.loads((run / "development" / "selection.json").read_text())
    assert selection["challenge_finalize_allowed"] is True
    assert selection["challenge_values_can_trigger_reselection"] is False
    assert (
        run / "development" / "all_tasks_authenticated.completed.json"
    ).is_file()
    sentinel = json.loads(
        (run / "development" / "all_tasks_authenticated.completed.json").read_text()
    )
    assert sentinel["sentinel_created_after_all_authenticated_completions"] is True
    assert sentinel["scientific_task_count"] == 168
    excess = json.loads(
        (run / "development" / "row_positive_excess_summaries.json").read_text()
    )
    assert excess["summary_count"] == 672
    scheduler = json.loads(
        (run / "runtime_auth" / "spawn_process_workers.json").read_text()
    )
    assert scheduler["start_method"] == "spawn"
    assert scheduler["logical_slot_count"] == 12
    assert scheduler["max_processes_per_physical_gpu"] == 3
    assert scheduler["fixed_slot_fifo_serial"] is True
    assert scheduler["result_order_reconstructed_by_assignment_index"] is True
    assert len({row["process_id"] for row in scheduler["workers"]}) == 12
    assert all(
        row["torch_preloaded_before_worker_bootstrap"] is False
        for row in scheduler["workers"]
    )
    assert {
        row["cuda_visible_devices"] for row in scheduler["workers"]
    } == {
        exp._canonical_cuda_uuid_selector(device["uuid"])
        for device in _gpu_devices()
    }
    assert all(
        row["cuda_visible_devices"]
        == exp._canonical_cuda_uuid_selector(
            next(
                device["uuid"]
                for device in _gpu_devices()
                if row["logical_worker_id"].startswith(
                    f"gpu{device['physical_gpu_id']}__"
                )
            )
        )
        and row["cuda_visible_devices"]
        not in {device["uuid"] for device in _gpu_devices()}
        for row in scheduler["workers"]
    )


def test_resultless_completed_bundle_and_missing_aggregate_ledger_recover_without_refit(
    completed_synthetic_run: Mapping[str, Any],
) -> None:
    run = Path(completed_synthetic_run["run"])
    task = exp.pord_v5_development_tasks(_config())[0]
    result_path = exp._task_dir(run, task) / "result.json"
    original = result_path.read_bytes()
    result_path.unlink()
    aggregate_ledger = run / "runtime_auth" / "development_gpu_workers.json"
    aggregate_original = aggregate_ledger.read_bytes()
    aggregate_ledger.unlink()

    def forbidden_executor(**_: Any) -> Mapping[str, Any]:
        raise AssertionError("completed recovery must not refit")

    recovered = exp.run_pord_v5_development(
        run,
        workers_per_gpu=3,
        resume=True,
        task_executor=forbidden_executor,
        parallel=False,
        verify_current_source=False,
        _private_test_only_mode=True,
    )
    assert recovered["status"] == "ok"
    assert result_path.read_bytes() == original
    assert aggregate_ledger.read_bytes() == aggregate_original


def test_started_only_fit_fails_closed_and_quarantine_intent_reconciles(
    tmp_path: Path,
) -> None:
    run, _, _ = _prepare_synthetic_run(tmp_path)
    config = _config()
    data = exp._load_development_snapshot(run)
    task = exp.pord_v5_development_tasks(config)[0]
    worker = exp.assign_pord_v5_tasks(
        exp.pord_v5_development_tasks(config),
        gpu_devices=_gpu_devices(),
        workers_per_gpu=1,
    )[0]["worker"]
    task_dir = exp._task_dir(run, task)
    task_dir.mkdir(parents=True)
    atomic_write_json(task_dir / "execution.started.json", {"incomplete": True})
    runtime_receipt = {
        "execution_kind": "synthetic_test_only_spawn_transport",
        "transport_bound": True,
        "logical_worker_id": worker["logical_worker_id"],
        "process_id": 12345,
        "physical_gpu_id": worker["physical_gpu_id"],
        "uuid": worker["uuid"],
        "pci_bus_id": worker["pci_bus_id"],
        "name": worker["name"],
        "cuda_visible_devices": exp._canonical_cuda_uuid_selector(worker["uuid"]),
        "visible_device_index": 0,
    }
    with pytest.raises(exp.GPUOutagePORDV5Error, match="ambiguous|fail closed"):
        exp._execute_or_resume_task(
            run_dir=run,
            config=config,
            data=data,
            task=task,
            worker=worker,
            workers_per_gpu=1,
            task_executor=_synthetic_task_executor,
            resume=True,
            runtime_gpu_receipt=runtime_receipt,
        )
    assert task_dir.is_dir()

    eval_started_task = exp.pord_v5_development_tasks(config)[2]
    eval_started_dir = exp._task_dir(run, eval_started_task)
    eval_started_dir.mkdir(parents=True)
    execution_path = atomic_write_json(
        eval_started_dir / "execution.started.json", {"phase": "execution.started"}
    )
    atomic_write_json(
        eval_started_dir / "training.completed.json", {"phase": "training.completed"}
    )
    atomic_write_json(
        eval_started_dir / "outer_eval.started.json", {"phase": "outer_eval.started"}
    )
    with pytest.raises(
        exp.GPUOutagePORDV5Error,
        match="outer_eval.started without authenticated completion",
    ):
        exp._resume_production_task_phases(
            run_dir=run,
            task=eval_started_task,
            data=data,
            execution_path=execution_path,
            runtime_gpu_receipt=runtime_receipt,
        )

    second = exp.pord_v5_development_tasks(config)[1]
    second_dir = exp._task_dir(run, second)
    second_dir.mkdir(parents=True)
    atomic_write_json(second_dir / "execution.started.json", {"crash": "before_move"})
    destination = run / "recovery" / "development_tasks" / f"{second['task_id']}__attempt1"
    intent_payload = {
        "task_id": second["task_id"],
        "attempt": 1,
        "source": exp._relative(second_dir, run),
        "destination": exp._relative(destination, run),
        "reason": "synthetic_crash_before_move",
        "operation": "nondelete_atomic_directory_move",
        "status": "intent_durable_before_move",
    }
    exp._exclusive_write_json(
        run
        / "recovery"
        / "quarantine_intents"
        / f"{second['task_id']}__attempt1.json",
        intent_payload,
    )
    exp._reconcile_quarantine_intents(run, second["task_id"])
    assert destination.is_dir()
    assert (
        run
        / "recovery"
        / "quarantine_audits"
        / f"{second['task_id']}__attempt1.json"
    ).is_file()


def test_selected_only_feature_freeze_and_exactly_once_challenge_recovery(
    completed_synthetic_run: Mapping[str, Any], tmp_path: Path,
) -> None:
    run = Path(completed_synthetic_run["run"])
    calls: list[str] = []

    def predictor(
        *, component_id: str, identity: Mapping[str, Any], **_: Any
    ) -> np.ndarray:
        calls.append(component_id)
        values = np.zeros((len(identity["trial_uid"]), 3), dtype=np.float64)
        values[:, 0] = 0.25
        return values

    attack_run = tmp_path / "challenge_toctou_attack"
    shutil.copytree(run, attack_run)
    exp.freeze_pord_v5_challenge_predictions(
        attack_run,
        component_predictor=predictor,
        verify_current_source=False,
        _private_test_only_mode=True,
    )
    attack_label_calls = 0

    def mutating_labels(identity: Mapping[str, Any], _: Path) -> Mapping[str, Any]:
        nonlocal attack_label_calls
        attack_label_calls += 1
        freeze = json.loads(
            (attack_run / "challenge" / "freeze.before_label_access.json").read_text()
        )
        selected_path = exp._verify_reference(
            attack_run, freeze["selected_prediction_file"]
        )
        frame = pd.read_parquet(selected_path)
        frame.loc[0, "pred_x_mm"] += 1.0
        atomic_write_parquet(selected_path, frame)
        return {
            "trial_uid": identity["trial_uid"],
            "segment_id": identity["segment_id"],
            "target_id": identity["target_id"],
            "y_mm": np.zeros((3295, 3), dtype=np.float64),
        }

    with pytest.raises(exp.GPUOutagePORDV5Error, match="changed|hash|reference"):
        exp.finalize_pord_v5_challenge(
            attack_run,
            resume=True,
            finalize_challenge=True,
            label_loader=mutating_labels,
            verify_current_source=False,
            _private_test_only_mode=True,
        )
    assert attack_label_calls == 1

    for attack_name, relative_path in (
        ("parent", Path("parent.freeze.json")),
        ("identity", Path("inputs/challenge_identity.json")),
    ):
        authority_attack = tmp_path / f"challenge_authority_{attack_name}_attack"
        shutil.copytree(run, authority_attack)
        exp.freeze_pord_v5_challenge_predictions(
            authority_attack,
            component_predictor=predictor,
            verify_current_source=False,
            _private_test_only_mode=True,
        )

        def mutating_authority_labels(
            identity: Mapping[str, Any], _: Path, *, target: Path = relative_path
        ) -> Mapping[str, Any]:
            path = authority_attack / target
            payload = json.loads(path.read_text())
            payload["post_marker_attack"] = attack_name
            atomic_write_json(path, payload)
            return {
                "trial_uid": identity["trial_uid"],
                "segment_id": identity["segment_id"],
                "target_id": identity["target_id"],
                "y_mm": np.zeros((3295, 3), dtype=np.float64),
            }

        with pytest.raises(
            exp.GPUOutagePORDV5Error, match="changed|reference|authority"
        ):
            exp.finalize_pord_v5_challenge(
                authority_attack,
                resume=True,
                finalize_challenge=True,
                label_loader=mutating_authority_labels,
                verify_current_source=False,
                _private_test_only_mode=True,
            )

    marker_attack = tmp_path / "challenge_parent_marker_race_attack"
    shutil.copytree(run, marker_attack)
    exp.freeze_pord_v5_challenge_predictions(
        marker_attack,
        component_predictor=predictor,
        verify_current_source=False,
        _private_test_only_mode=True,
    )
    parent = json.loads((marker_attack / "parent.freeze.json").read_text())
    raced_parent_marker = (
        Path(parent["v3_run_origin"])
        / "challenge"
        / "label_access_1.started.json"
    )

    def parent_marker_race_labels(
        identity: Mapping[str, Any], _: Path
    ) -> Mapping[str, Any]:
        atomic_write_json(raced_parent_marker, {"concurrent_parent": True})
        return {
            "trial_uid": identity["trial_uid"],
            "segment_id": identity["segment_id"],
            "target_id": identity["target_id"],
            "y_mm": np.zeros((3295, 3), dtype=np.float64),
        }

    try:
        with pytest.raises(
            exp.GPUOutagePORDV5Error,
            match="prior v3/v4 label-access marker",
        ):
            exp.finalize_pord_v5_challenge(
                marker_attack,
                resume=True,
                finalize_challenge=True,
                label_loader=parent_marker_race_labels,
                verify_current_source=False,
                _private_test_only_mode=True,
            )
    finally:
        raced_parent_marker.unlink(missing_ok=True)
    calls.clear()

    frozen = exp.freeze_pord_v5_challenge_predictions(
        run,
        component_predictor=predictor,
        verify_current_source=False,
        _private_test_only_mode=True,
    )
    assert frozen["selected_only_component_count"] == 1
    assert frozen["selected_only_component_ids"] == [
        "masked_mlp_64x64_relu__paired_pord"
    ]
    assert calls == [
        "masked_mlp_64x64_relu__paired_pord",
        "masked_mlp_64x64_relu__paired_pord",
    ]
    access_calls = 0

    def labels(identity: Mapping[str, Any], _: Path) -> Mapping[str, Any]:
        nonlocal access_calls
        access_calls += 1
        return {
            "trial_uid": identity["trial_uid"],
            "segment_id": identity["segment_id"],
            "target_id": identity["target_id"],
            "y_mm": np.zeros((3295, 3), dtype=np.float64),
        }

    completed = exp.finalize_pord_v5_challenge(
        run,
        resume=True,
        finalize_challenge=True,
        label_loader=labels,
        verify_current_source=False,
        _private_test_only_mode=True,
    )
    assert completed["stage"] == "challenge_complete_no_reselection"
    assert access_calls == 1
    assert completed["report"]["primary_segment"] == 41
    assert completed["report"]["secondary_segment"] == 40
    assert completed["report"]["challenge_used_for_reselection"] is False

    def forbidden_labels(_: Mapping[str, Any], __: Path) -> Any:
        raise AssertionError("completed recovery must not reread labels")

    (run / "manifest.json").unlink()
    recovered = exp.finalize_pord_v5_challenge(
        run,
        resume=True,
        finalize_challenge=True,
        label_loader=forbidden_labels,
        verify_current_source=False,
        _private_test_only_mode=True,
    )
    assert recovered["recovered_without_source_reread"] is True
    assert (run / "manifest.json").is_file()
    assert access_calls == 1
    tree_before_validation = _filesystem_tree_sha256(run)
    final = exp.validate_pord_v5_final_run(
        run,
        verify_current_source=False,
        allow_synthetic_test_only=True,
        verify_live_environment=False,
    )
    assert final["challenge_label_access_passes"] == 1
    assert _filesystem_tree_sha256(run) == tree_before_validation

    worker_ledger_path = next(
        (run / "runtime_auth" / "worker_ledgers").rglob("*.json")
    )
    worker_ledger_bytes = worker_ledger_path.read_bytes()
    worker_ledger_path.unlink()
    missing_ledger_tree = _filesystem_tree_sha256(run)
    with pytest.raises(exp.GPUOutagePORDV5Error, match="worker ledger"):
        exp.validate_pord_v5_final_run(
            run,
            verify_current_source=False,
            allow_synthetic_test_only=True,
            verify_live_environment=False,
        )
    assert _filesystem_tree_sha256(run) == missing_ledger_tree
    worker_ledger_path.write_bytes(worker_ledger_bytes)

    development_manifest_path = run / "development_manifest.json"
    development_manifest = json.loads(development_manifest_path.read_text())
    attacked_development = {
        **development_manifest,
        "unexpected": True,
        "counts": {
            **development_manifest["counts"],
            "controlled_fit_count": 168.0,
        },
    }
    atomic_write_json(development_manifest_path, attacked_development)
    with pytest.raises(exp.GPUOutagePORDV5Error, match="development manifest"):
        exp.validate_pord_v5_development_run(
            run,
            verify_current_source=False,
            allow_completed_challenge=True,
            allow_synthetic_test_only=True,
            verify_live_environment=False,
        )
    atomic_write_json(development_manifest_path, development_manifest)

    final_manifest_path = run / "manifest.json"
    final_manifest = json.loads(final_manifest_path.read_text())
    atomic_write_json(
        final_manifest_path,
        {**final_manifest, "unexpected": True, "primary_segment": 41.0},
    )
    with pytest.raises(exp.GPUOutagePORDV5Error, match="final v5 manifest"):
        exp.validate_pord_v5_final_run(
            run,
            verify_current_source=False,
            allow_synthetic_test_only=True,
            verify_live_environment=False,
        )
    atomic_write_json(final_manifest_path, final_manifest)


def test_challenge_lifecycle_and_label_gate_negative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(exp.GPUOutagePORDV5Error, match="both"):
        exp.finalize_pord_v5_challenge(
            tmp_path, resume=False, finalize_challenge=True
        )
    started_run = tmp_path / "started_only"
    (started_run / "challenge").mkdir(parents=True)
    atomic_write_json(
        started_run / "challenge" / "label_access_1.started.json", {"started": True}
    )
    with pytest.raises(exp.GPUOutagePORDV5Error, match="fail closed"):
        exp.finalize_pord_v5_challenge(
            started_run, resume=True, finalize_challenge=True
        )

    refused_run = tmp_path / "refused"
    refused_run.mkdir()
    monkeypatch.setattr(
        exp,
        "validate_pord_v5_development_run",
        lambda *args, **kwargs: {"challenge_finalize_allowed": False},
    )
    monkeypatch.setattr(
        exp,
        "_require_mode_authority",
        lambda *args, **kwargs: exp.PORD_V5_PRODUCTION_MODE,
    )
    with pytest.raises(exp.GPUOutagePORDV5Error, match="claim/novelty"):
        exp.freeze_pord_v5_challenge_predictions(refused_run, device="cuda:0")


def test_prepare_rejects_parent_marker_and_bidirectional_path_overlap(tmp_path: Path) -> None:
    config = _config()
    v3 = tmp_path / "v3"
    v4 = tmp_path / "v4"
    v3.mkdir()
    v4.mkdir()
    atomic_write_json(v3 / "development_manifest.json", {})
    atomic_write_json(v4 / "development_manifest.json", {})
    (v3 / "challenge").mkdir()
    atomic_write_json(v3 / "challenge" / "label_access_1.started.json", {})
    with pytest.raises(exp.GPUOutagePORDV5Error, match="marker"):
        exp.prepare_pord_v5_run(
            config,
            v3_run_dir=v3,
            v4_run_dir=v4,
            run_dir=tmp_path / "out",
        )
    with pytest.raises(exp.GPUOutagePORDV5Error, match="disjoint"):
        exp._require_bidirectionally_disjoint_paths(v3, v4, v3 / "nested")
