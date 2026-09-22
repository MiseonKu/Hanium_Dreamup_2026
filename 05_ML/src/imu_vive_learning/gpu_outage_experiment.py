"""Resumable four-GPU runner for the prospective outage-v3 experiment.

The challenge boundary is deliberately asymmetric: identity and inputs are
materialised without Vive coordinates before development fitting, while the
single label read is allowed only after selection, final refit, checkpoints,
preprocessing, and label-blind predictions have durable SHA-256 references.
The semantic validator at the bottom of this module reads only those durable
artifacts; it never reopens either source archive.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import queue
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .experiment_io import (
    atomic_write_json,
    atomic_write_parquet,
    manifest_files,
    prepare_experiment_directory,
    read_json_object,
    sha256_file,
    stable_hash,
    verify_file_manifest,
)
from .experiment_metrics import regression_metrics
from .experiment_preprocessing import TrainOnlyPreprocessor
from .features import FUSION_COLUMNS
from .gpu_outage_data import (
    AVAILABILITY_NAMES,
    ChallengeIdentityData,
    DevelopmentData,
    load_challenge_identity,
    load_challenge_labels_once,
    load_development_data,
    ordered_trial_uid_sha256,
)
from .gpu_outage_experiment_config import (
    OUTAGE_CANDIDATE_IDS,
    OUTAGE_GPU_EXPERIMENT_CONTRACT,
    TRAINABLE_OUTAGE_CANDIDATE_IDS,
    outage_candidate_grid,
    outage_gpu_config_sha256,
    public_outage_gpu_config,
    resolve_outage_development_release_path,
    resolve_outage_source_path,
    validate_outage_gpu_config,
)
from .gpu_v3_backbones import IMU_SENSOR_INDEX_MAP


OUTAGE_RUN_MANIFEST_CONTRACT = "alphaf.endpoint_regression.gpu_outage_run.v3"
OUTAGE_SEED_RESULT_CONTRACT = "alphaf.endpoint_regression.gpu_outage_seed_result.v3"
OUTAGE_CHALLENGE_FREEZE_CONTRACT = "alphaf.endpoint_regression.gpu_outage_challenge_freeze.v3"

OUTAGE_SOURCE_FILES = (
    "configs/endpoint_regression_gpu_outage_v3.yaml",
    "requirements-gpu.lock",
    "requirements-eval.lock",
    "pyproject.toml",
    "src/imu_vive_learning/__init__.py",
    "src/imu_vive_learning/gpu_outage_experiment.py",
    "src/imu_vive_learning/gpu_outage_experiment_config.py",
    "src/imu_vive_learning/gpu_outage_data.py",
    "src/imu_vive_learning/gpu_outage_training.py",
    "src/imu_vive_learning/gpu_v3_backbones.py",
    "src/imu_vive_learning/gpu_robust_training.py",
    "src/imu_vive_learning/gpu_robust_experiment_config.py",
    "src/imu_vive_learning/gpu_experiment.py",
    "src/imu_vive_learning/gpu_experiment_config.py",
    "src/imu_vive_learning/gpu_inference.py",
    "src/imu_vive_learning/experiment_models.py",
    "src/imu_vive_learning/experiment_preprocessing.py",
    "src/imu_vive_learning/experiment_metrics.py",
    "src/imu_vive_learning/experiment_io.py",
    "src/imu_vive_learning/experiment_release.py",
    "src/imu_vive_learning/experiment_splits.py",
    "src/imu_vive_learning/features.py",
    "src/imu_vive_learning/config.py",
    "src/imu_vive_learning/source.py",
    "src/imu_vive_learning/io_utils.py",
    "src/imu_vive_learning/cleaning.py",
    "src/imu_vive_learning/splitting.py",
    "src/imu_vive_learning/cli.py",
)

_TRUE_COLUMNS = ("true_x_mm", "true_y_mm", "true_z_mm")
_PRED_COLUMNS = ("pred_x_mm", "pred_y_mm", "pred_z_mm")
_OUTAGE_NAME_TO_INDEX = {
    "imu0_base": 0,
    "imu1_link_root": 1,
    "imu2_link_mid": 2,
    "imu3_tip": 3,
}


class GPUOutageExperimentError(RuntimeError):
    """Raised when execution or durable evidence violates outage-v3."""


def _require_python_312() -> None:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise GPUOutageExperimentError("outage-v3 requires CPython 3.12.x")


def _read_resolved_outage_config(path: str | Path) -> dict[str, Any]:
    """Restore YAML integer map keys after the JSON durability round-trip."""

    config = read_json_object(path)
    try:
        roles = config["row_roles"]
        for role_name in ("development", "challenge"):
            raw = roles[role_name]["expected_segment_rows"]
            roles[role_name]["expected_segment_rows"] = {
                int(key): int(value) for key, value in raw.items()
            }
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise GPUOutageExperimentError(
            "resolved outage-v3 config has invalid segment-count mappings"
        ) from exc
    validate_outage_gpu_config(config)
    return config


def _relative(path: str | Path, run_dir: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(run_dir).resolve()).as_posix()
    except ValueError as exc:
        raise GPUOutageExperimentError("artifact path escapes the run directory") from exc


def _safe_member(run_dir: str | Path, relative: str) -> Path:
    member = PurePosixPath(str(relative))
    if member.is_absolute() or ".." in member.parts:
        raise GPUOutageExperimentError(f"unsafe run artifact path: {relative!r}")
    path = (Path(run_dir).resolve() / Path(*member.parts)).resolve()
    try:
        path.relative_to(Path(run_dir).resolve())
    except ValueError as exc:
        raise GPUOutageExperimentError(f"run artifact escapes directory: {relative!r}") from exc
    return path


def _artifact_reference(path: str | Path, run_dir: str | Path, **extra: Any) -> dict[str, Any]:
    artifact = Path(path)
    payload: dict[str, Any] = {
        "path": _relative(artifact, run_dir),
        "bytes": int(artifact.stat().st_size),
        "sha256": sha256_file(artifact),
    }
    payload.update(extra)
    return payload


def _exclusive_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Durably create a JSON claim exactly once without a replace race."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise GPUOutageExperimentError(
            "challenge label access is already claimed; fail closed rather than "
            "opening labels a second time"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # Never remove a partial claim.  Its presence keeps retries fail-closed.
        raise
    return output


def _verify_reference(run_dir: str | Path, reference: Mapping[str, Any]) -> Path:
    path = _safe_member(run_dir, str(reference.get("path", "")))
    if not path.is_file():
        raise GPUOutageExperimentError(f"referenced artifact is missing: {path}")
    if int(reference.get("bytes", -1)) != int(path.stat().st_size):
        raise GPUOutageExperimentError(f"referenced artifact size changed: {path}")
    if str(reference.get("sha256", "")) != sha256_file(path):
        raise GPUOutageExperimentError(f"referenced artifact SHA-256 changed: {path}")
    return path


def _require_canonical_reference(
    run_dir: str | Path,
    reference: Mapping[str, Any],
    canonical_path: str | Path,
    **extra: Any,
) -> Path:
    expected = _artifact_reference(canonical_path, run_dir, **extra)
    if dict(reference) != expected:
        raise GPUOutageExperimentError(
            f"artifact reference is not bound to canonical path {expected['path']}"
        )
    return _verify_reference(run_dir, reference)


def _read_exact_parquet_projection(
    path: str | Path, expected_columns: Sequence[str]
) -> pd.DataFrame:
    """Check the Arrow schema before decoding only the approved columns."""

    source = Path(path)
    columns = [str(value) for value in expected_columns]
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    if schema.names != columns or len(schema) != len(columns):
        raise GPUOutageExperimentError(
            f"Parquet schema changed before projection: {source}"
        )
    for field in schema:
        if field.name in {"trial_uid", "target_id"}:
            valid_type = pa.types.is_string(field.type)
        elif field.name == "segment_id":
            valid_type = pa.types.is_int64(field.type)
        else:
            valid_type = pa.types.is_float64(field.type)
        if not valid_type:
            raise GPUOutageExperimentError(
                f"Parquet field type changed for {field.name}: {field.type}"
            )
    return pd.read_parquet(source, columns=columns)


def _atomic_write_npz(path: str | Path, state: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    arrays: dict[str, np.ndarray] = {}
    for name, value in state.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise GPUOutageExperimentError(f"checkpoint tensor is non-finite: {name}")
        arrays[str(name)] = array
    if not arrays:
        raise GPUOutageExperimentError("checkpoint state is empty")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)
    return output


def load_outage_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    """Load a trusted runner checkpoint without pickle."""

    with np.load(Path(path), allow_pickle=False) as archive:
        state = {str(name): np.array(archive[name], copy=True) for name in archive.files}
    if not state or any(not np.isfinite(value).all() for value in state.values()):
        raise GPUOutageExperimentError("checkpoint is empty or non-finite")
    return state


def outage_training_source_payload(workspace_root: str | Path) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    files: list[dict[str, Any]] = []
    for relative in OUTAGE_SOURCE_FILES:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise GPUOutageExperimentError(f"source path escapes workspace: {relative}") from exc
        if not path.is_file():
            raise GPUOutageExperimentError(f"outage-v3 source file is missing: {relative}")
        files.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {"files": files, "fingerprint": stable_hash(files)}


def _runner_workspace_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "configs" / "endpoint_regression_gpu_outage_v3.yaml").is_file():
        raise GPUOutageExperimentError("cannot locate the outage-v3 source workspace")
    return root


def _resolved_config_payload(
    config: Mapping[str, Any], source_fingerprint: str
) -> dict[str, Any]:
    payload = public_outage_gpu_config(dict(config))
    payload["_training_source_fingerprint"] = str(source_fingerprint)
    return payload


def _run_identity_payload(
    config: Mapping[str, Any], source_fingerprint: str
) -> dict[str, Any]:
    return {
        "contract": OUTAGE_GPU_EXPERIMENT_CONTRACT,
        "config_sha256": outage_gpu_config_sha256(dict(config)),
        "development_release_sha256": str(
            config["development_release"]["archive_sha256"]
        ),
        "source_archive_sha256": str(config["source"]["archive_sha256"]),
        "source_member_sha256": str(config["source"]["member_sha256"]),
        "challenge_ordered_trial_uid_sha256": str(
            config["row_roles"]["challenge"]["ordered_trial_uid_sha256"]
        ),
        "training_source_fingerprint": str(source_fingerprint),
    }


def _validate_source_provenance(
    run_dir: str | Path,
    config: Mapping[str, Any],
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Bind the run identity and stored snapshot to the current source closure."""

    run_path = Path(run_dir).resolve()
    workspace = (
        _runner_workspace_root()
        if workspace_root is None
        else Path(workspace_root).resolve()
    )
    current = outage_training_source_payload(workspace)
    stored = read_json_object(run_path / "provenance" / "source_snapshot.json")
    if stable_hash(stored.get("files")) != stored.get("fingerprint"):
        raise GPUOutageExperimentError("durable source snapshot self-hash changed")
    if stored != current:
        raise GPUOutageExperimentError(
            "current outage-v3 source closure differs from the frozen snapshot"
        )
    fingerprint = str(current["fingerprint"])
    if config.get("_training_source_fingerprint") != fingerprint:
        raise GPUOutageExperimentError(
            "resolved config training-source fingerprint changed"
        )
    run_identity = read_json_object(run_path / "_state" / "run_identity.json")
    if run_identity != _run_identity_payload(config, fingerprint):
        raise GPUOutageExperimentError("run identity/source fingerprint binding changed")
    return current


def _validate_gpu_ids(gpu_ids: Sequence[int], expected_count: int = 4) -> tuple[int, ...]:
    values = tuple(int(value) for value in gpu_ids)
    if len(values) != expected_count or len(set(values)) != expected_count:
        raise GPUOutageExperimentError(
            f"outage-v3 requires exactly {expected_count} distinct GPU IDs"
        )
    if any(value < 0 for value in values):
        raise GPUOutageExperimentError("GPU IDs must be non-negative")
    return values


def _candidate_map(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate["candidate_id"]): dict(candidate)
        for candidate in outage_candidate_grid(dict(config))
    }


def _forced_outages(config: Mapping[str, Any]) -> tuple[str, ...]:
    values = tuple(str(value) for value in config["splits"]["forced_validation_outages"])
    if values != tuple(_OUTAGE_NAME_TO_INDEX):
        raise GPUOutageExperimentError("forced outage order changed")
    return values


def outage_development_tasks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact frozen 240 one-candidate/context/seed tasks."""

    validate_outage_gpu_config(dict(config))
    candidates = _candidate_map(config)
    segments = tuple(int(value) for value in config["splits"]["development_segments"])
    outages = _forced_outages(config)
    seeds = tuple(int(value) for value in config["training"]["paired_seed_values"])
    tasks: list[dict[str, Any]] = []
    for candidate_id in TRAINABLE_OUTAGE_CANDIDATE_IDS:
        for held_out_segment in segments:
            for outage_name in outages:
                for seed_index, model_seed in enumerate(seeds):
                    task_id = (
                        f"development__{candidate_id}__segment{held_out_segment}__"
                        f"drop{_OUTAGE_NAME_TO_INDEX[outage_name]}__seed{seed_index}"
                    )
                    tasks.append(
                        {
                            "task_id": task_id,
                            "phase": "development",
                            "candidate_id": candidate_id,
                            "candidate": deepcopy(candidates[candidate_id]),
                            "held_out_segment": held_out_segment,
                            "forced_outage_name": outage_name,
                            "forced_outage_index": _OUTAGE_NAME_TO_INDEX[outage_name],
                            "paired_seed_index": seed_index,
                            "model_seed": model_seed,
                        }
                    )
    expected = int(config["selection"]["trainable_seeded_fit_count"])
    if len(tasks) != expected or len({row["task_id"] for row in tasks}) != expected:
        raise GPUOutageExperimentError(
            f"development task grid changed: expected {expected}, received {len(tasks)}"
        )
    return tasks


def _round_median(values: Sequence[int]) -> int:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise GPUOutageExperimentError("cannot derive final epoch from invalid values")
    return max(1, int(math.floor(float(np.median(array)) + 0.5)))


def _fit_preprocessor(
    data: DevelopmentData,
    train_rows: np.ndarray,
    *,
    forbidden_rows: np.ndarray,
    split_context: str,
) -> TrainOnlyPreprocessor:
    preprocessor = TrainOnlyPreprocessor(
        feature_names=tuple(data.feature_names),
        scale_features=True,
        split_context=split_context,
    )
    return preprocessor.fit_real_train(
        data.X[train_rows],
        data.y[train_rows],
        trial_uids=data.trial_uids[train_rows].tolist(),
        forbidden_uids=data.trial_uids[forbidden_rows].tolist(),
    )


def _development_task_dir(run_dir: str | Path, task: Mapping[str, Any]) -> Path:
    return Path(run_dir) / "development" / "seed_tasks" / str(task["task_id"])


def _development_result_path(run_dir: str | Path, task: Mapping[str, Any]) -> Path:
    return _development_task_dir(run_dir, task) / "result.json"


def _result_without_large_arrays(result: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"state_dict", "eval_prediction_raw_mm", "train_prediction_raw_mm", "history"}
    return {key: value for key, value in result.items() if key not in excluded}


def _execute_development_task(
    data: DevelopmentData,
    task: Mapping[str, Any],
    config: Mapping[str, Any],
    run_dir: str | Path,
    *,
    gpu_id: int,
    train_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute one seed fit; ``train_fn`` is an explicit synthetic-test seam."""

    if train_fn is None:
        from .gpu_outage_training import train_outage_seed

        train_fn = train_outage_seed
    held_out = int(task["held_out_segment"])
    eval_rows = np.asarray(data.segment_ids == held_out, dtype=bool)
    train_rows = ~eval_rows
    if not train_rows.any() or not eval_rows.any():
        raise GPUOutageExperimentError("development task has an empty train/eval role")
    context = (
        f"outage_v3_segment{held_out}_drop{int(task['forced_outage_index'])}_"
        f"seed{int(task['paired_seed_index'])}"
    )
    preprocessor = _fit_preprocessor(
        data, train_rows, forbidden_rows=eval_rows, split_context=context
    )
    X_train = preprocessor.transform_features(data.X[train_rows]).astype(np.float32)
    y_train = preprocessor.transform_targets(data.y[train_rows]).astype(np.float32)
    X_eval = preprocessor.transform_features(data.X[eval_rows]).astype(np.float32)
    result = dict(
        train_fn(
            X_train=X_train,
            y_train=y_train,
            segment_ids=data.segment_ids[train_rows],
            trial_uids=data.trial_uids[train_rows],
            X_eval=X_eval,
            y_eval_raw=data.y[eval_rows],
            preprocessor=preprocessor,
            held_out_segment=held_out,
            forced_outage=int(task["forced_outage_index"]),
            paired_seed_index=int(task["paired_seed_index"]),
            candidate=task["candidate"],
            config=config,
            model_seed=int(task["model_seed"]),
            device="cuda:0",
        )
    )
    prediction = np.asarray(result["eval_prediction_raw_mm"], dtype=np.float64)
    truth = np.asarray(data.y[eval_rows], dtype=np.float64)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise GPUOutageExperimentError("development prediction shape/value is invalid")
    recomputed = regression_metrics(truth, prediction)
    if not math.isclose(
        float(recomputed["rmse_3d_mm"]),
        float(result["eval_metrics"]["rmse_3d_mm"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise GPUOutageExperimentError("development task metric/prediction mismatch")

    task_dir = _development_task_dir(run_dir, task)
    prep_path = atomic_write_json(task_dir / "preprocessing.json", preprocessor.metadata())
    history = pd.DataFrame(list(result["history"]))
    if len(history) != int(result["trained_epochs"]):
        raise GPUOutageExperimentError("development history length changed")
    history_path = atomic_write_parquet(task_dir / "history.parquet", history)
    predictions = pd.DataFrame(
        {
            "trial_uid": data.trial_uids[eval_rows],
            "segment_id": data.segment_ids[eval_rows],
            "target_id": data.target_ids[eval_rows],
            **{name: truth[:, index] for index, name in enumerate(_TRUE_COLUMNS)},
            **{name: prediction[:, index] for index, name in enumerate(_PRED_COLUMNS)},
        }
    )
    prediction_path = atomic_write_parquet(task_dir / "predictions.parquet", predictions)
    checkpoint_path = _atomic_write_npz(task_dir / "checkpoint.npz", result["state_dict"])
    artifacts = {
        "preprocessing": _artifact_reference(prep_path, run_dir),
        "history": _artifact_reference(history_path, run_dir, rows=len(history)),
        "predictions": _artifact_reference(prediction_path, run_dir, rows=len(predictions)),
        "checkpoint": _artifact_reference(
            checkpoint_path, run_dir, tensor_count=len(result["state_dict"])
        ),
    }
    payload = {
        "contract": OUTAGE_SEED_RESULT_CONTRACT,
        "config_sha256": outage_gpu_config_sha256(dict(config)),
        "task": dict(task),
        "gpu_id": int(gpu_id),
        "summary": _result_without_large_arrays(result),
        "metrics": recomputed,
        "artifacts": artifacts,
    }
    result_path = atomic_write_json(task_dir / "result.json", payload)
    payload["result_file"] = _artifact_reference(result_path, run_dir)
    return payload


def _load_valid_task_result(
    run_dir: str | Path, task: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _development_result_path(run_dir, task)
    if not path.is_file():
        return None
    try:
        result = read_json_object(path)
        if result.get("contract") != OUTAGE_SEED_RESULT_CONTRACT:
            return None
        if result.get("task") != dict(task):
            return None
        for reference in result.get("artifacts", {}).values():
            _verify_reference(run_dir, reference)
    except (KeyError, OSError, TypeError, ValueError, GPUOutageExperimentError):
        return None
    result["result_file"] = _artifact_reference(path, run_dir)
    return result


def _read_prediction_reference(
    run_dir: str | Path, result: Mapping[str, Any]
) -> pd.DataFrame:
    reference = result["artifacts"]["predictions"]
    return _read_exact_parquet_projection(
        _verify_reference(run_dir, reference),
        ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS, *_PRED_COLUMNS],
    )


def _context_key(candidate_id: str, segment: int, outage_index: int) -> tuple[str, int, int]:
    return str(candidate_id), int(segment), int(outage_index)


def _aggregate_context(
    *,
    candidate_id: str,
    held_out_segment: int,
    forced_outage_index: int,
    seed_results: Sequence[Mapping[str, Any]],
    run_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    ordered = sorted(seed_results, key=lambda row: int(row["task"]["paired_seed_index"]))
    if [int(row["task"]["paired_seed_index"]) for row in ordered] != [0, 1, 2]:
        raise GPUOutageExperimentError("context lacks exactly paired seeds 0,1,2")
    frames = [_read_prediction_reference(run_dir, row) for row in ordered]
    identity_columns = ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS]
    reference_identity = frames[0][identity_columns]
    for frame in frames[1:]:
        if not frame[identity_columns].equals(reference_identity):
            raise GPUOutageExperimentError("seed predictions are not row-aligned")
    truth = frames[0].loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
    seed_predictions = [
        frame.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64) for frame in frames
    ]
    ensemble = np.mean(np.stack(seed_predictions, axis=0), axis=0)
    metrics = regression_metrics(truth, ensemble)
    seed_rmses = [float(regression_metrics(truth, value)["rmse_3d_mm"]) for value in seed_predictions]
    output = reference_identity.copy()
    for index, name in enumerate(_PRED_COLUMNS):
        output[name] = ensemble[:, index]
    path = atomic_write_parquet(output_path, output)
    return {
        "candidate_id": candidate_id,
        "held_out_segment": int(held_out_segment),
        "forced_outage_index": int(forced_outage_index),
        "context_id": f"segment{held_out_segment}__drop{forced_outage_index}",
        "seed_count": 3,
        "metrics": metrics,
        "seed_rmse_3d_mm": seed_rmses,
        "seed_rmse_population_std_mm": float(np.std(seed_rmses, ddof=0)),
        "prediction_file": _artifact_reference(path, run_dir, rows=len(output)),
    }


def _aggregate_blend_context(
    *,
    held_out_segment: int,
    forced_outage_index: int,
    constituent_results: Mapping[str, Sequence[Mapping[str, Any]]],
    run_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    component_ids = ("tabm_batchensemble", "chain_masked_reconstruction")
    component_frames: dict[str, list[pd.DataFrame]] = {}
    for component in component_ids:
        rows = sorted(
            constituent_results[component],
            key=lambda row: int(row["task"]["paired_seed_index"]),
        )
        if [int(row["task"]["paired_seed_index"]) for row in rows] != [0, 1, 2]:
            raise GPUOutageExperimentError("blend constituent lacks three paired seeds")
        component_frames[component] = [_read_prediction_reference(run_dir, row) for row in rows]
    identity_columns = ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS]
    base = component_frames[component_ids[0]][0]
    for frames in component_frames.values():
        for frame in frames:
            if not frame[identity_columns].equals(base[identity_columns]):
                raise GPUOutageExperimentError("blend constituent predictions are misaligned")
    truth = base.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
    paired_seed_predictions: list[np.ndarray] = []
    for seed_index in range(3):
        pair = [
            component_frames[component][seed_index]
            .loc[:, list(_PRED_COLUMNS)]
            .to_numpy(dtype=np.float64)
            for component in component_ids
        ]
        paired_seed_predictions.append(0.5 * pair[0] + 0.5 * pair[1])
    ensemble = np.mean(np.stack(paired_seed_predictions, axis=0), axis=0)
    metrics = regression_metrics(truth, ensemble)
    seed_rmses = [
        float(regression_metrics(truth, value)["rmse_3d_mm"])
        for value in paired_seed_predictions
    ]
    output = base[identity_columns].copy()
    for index, name in enumerate(_PRED_COLUMNS):
        output[name] = ensemble[:, index]
    path = atomic_write_parquet(output_path, output)
    return {
        "candidate_id": "equal_blend_tabm_chain_reconstruction",
        "held_out_segment": int(held_out_segment),
        "forced_outage_index": int(forced_outage_index),
        "context_id": f"segment{held_out_segment}__drop{forced_outage_index}",
        "seed_count": 3,
        "metrics": metrics,
        "seed_rmse_3d_mm": seed_rmses,
        "seed_rmse_population_std_mm": float(np.std(seed_rmses, ddof=0)),
        "blend_weights": {
            "tabm_batchensemble": 0.5,
            "chain_masked_reconstruction": 0.5,
        },
        "prediction_file": _artifact_reference(path, run_dir, rows=len(output)),
    }


def score_outage_candidates(
    context_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute frozen A/WS/WM/J and tolerance-aware tie-breaks."""

    selection_config = config["selection"]
    if selection_config.get("nonfinite_context_score") != "positive_infinity":
        raise GPUOutageExperimentError("non-finite context scoring policy changed")
    if selection_config.get("missing_context_action") != "fail_closed_before_scoring":
        raise GPUOutageExperimentError("missing-context scoring policy changed")
    segments = tuple(int(value) for value in config["splits"]["development_segments"])
    outages = tuple(range(4))
    trace: list[dict[str, Any]] = []
    expected_keys = {
        _context_key(candidate_id, segment, outage)
        for candidate_id in OUTAGE_CANDIDATE_IDS
        for segment in segments
        for outage in outages
    }
    by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in context_rows:
        key = _context_key(
            str(row["candidate_id"]),
            int(row["held_out_segment"]),
            int(row["forced_outage_index"]),
        )
        if key not in expected_keys:
            raise GPUOutageExperimentError(f"unexpected scored context: {key}")
        if key in by_key:
            raise GPUOutageExperimentError(f"duplicate scored context: {key}")
        by_key[key] = row
    if set(by_key) != expected_keys or len(context_rows) != len(expected_keys):
        missing = sorted(expected_keys.difference(by_key))
        raise GPUOutageExperimentError(
            f"scored context grid must contain exactly 96 unique keys; missing {missing[:5]}"
        )
    for priority, candidate_id in enumerate(OUTAGE_CANDIDATE_IDS):
        rows: list[Mapping[str, Any]] = []
        missing: list[str] = []
        for segment in segments:
            for outage in outages:
                row = by_key.get(_context_key(candidate_id, segment, outage))
                if row is None:
                    missing.append(f"segment{segment}/drop{outage}")
                else:
                    rows.append(row)
        finite = len(rows) == 16 and all(
            math.isfinite(float(row["metrics"]["rmse_3d_mm"]))
            and math.isfinite(float(row["metrics"]["p95_3d_error_mm"]))
            and math.isfinite(float(row["seed_rmse_population_std_mm"]))
            for row in rows
        )
        if not finite:
            trace.append(
                {
                    "candidate_id": candidate_id,
                    "listed_priority": priority,
                    "context_count": len(rows),
                    "missing_contexts": missing,
                    "score_status": "positive_infinity",
                }
            )
            continue
        rmses = [float(row["metrics"]["rmse_3d_mm"]) for row in rows]
        p95s = [float(row["metrics"]["p95_3d_error_mm"]) for row in rows]
        per_segment = [
            {
                "segment_id": segment,
                "mean_rmse_3d_mm": float(
                    np.mean(
                        [
                            float(by_key[_context_key(candidate_id, segment, outage)]["metrics"]["rmse_3d_mm"])
                            for outage in outages
                        ]
                    )
                ),
            }
            for segment in segments
        ]
        per_sensor = [
            {
                "forced_outage_index": outage,
                "mean_rmse_3d_mm": float(
                    np.mean(
                        [
                            float(by_key[_context_key(candidate_id, segment, outage)]["metrics"]["rmse_3d_mm"])
                            for segment in segments
                        ]
                    )
                ),
            }
            for outage in outages
        ]
        mean_rmse = float(np.mean(rmses))
        worst_segment = max(per_segment, key=lambda row: float(row["mean_rmse_3d_mm"]))
        worst_sensor = max(per_sensor, key=lambda row: float(row["mean_rmse_3d_mm"]))
        robust_j = float(
            (
                mean_rmse
                + float(worst_segment["mean_rmse_3d_mm"])
                + float(worst_sensor["mean_rmse_3d_mm"])
            )
            / 3.0
        )
        trace.append(
            {
                "candidate_id": candidate_id,
                "listed_priority": priority,
                "context_count": 16,
                "score_status": "finite",
                "mean_rmse_over_16_contexts_mm": mean_rmse,
                "worst_segment_mean_rmse_mm": float(worst_segment["mean_rmse_3d_mm"]),
                "worst_sensor_mean_rmse_mm": float(worst_sensor["mean_rmse_3d_mm"]),
                "outage_robust_j_mm": robust_j,
                "mean_p95_3d_mm_over_16_contexts": float(np.mean(p95s)),
                "mean_seed_rmse_population_std_mm_over_16_contexts": float(
                    np.mean([float(row["seed_rmse_population_std_mm"]) for row in rows])
                ),
                "per_segment": per_segment,
                "per_sensor": per_sensor,
            }
        )
    finite_trace = [row for row in trace if row["score_status"] == "finite"]
    if not finite_trace:
        raise GPUOutageExperimentError("all outage-v3 candidates have infinite scores")
    tolerance = float(config["selection"]["numeric_equality_abs_tolerance_mm"])
    winner = finite_trace[0]
    metric_names = (
        "outage_robust_j_mm",
        "mean_p95_3d_mm_over_16_contexts",
        "mean_seed_rmse_population_std_mm_over_16_contexts",
    )
    for challenger in finite_trace[1:]:
        replace = False
        for name in metric_names:
            left = float(challenger[name])
            right = float(winner[name])
            if left < right - tolerance:
                replace = True
                break
            if left > right + tolerance:
                break
        if replace:
            winner = challenger
    return {
        "score_name": "outage_robust_j_mm",
        "context_count_per_candidate": 16,
        "trainable_seeded_fit_count": 240,
        "trace": trace,
        "selected_candidate_id": str(winner["candidate_id"]),
        "selected_outage_robust_j_mm": float(winner["outage_robust_j_mm"]),
        "challenge_used_for_selection": False,
    }


def _identity_frame(identity: ChallengeIdentityData) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_uid": identity.trial_uids,
            "segment_id": identity.segment_ids,
            "target_id": identity.target_ids,
            **{
                name: identity.X[:, index]
                for index, name in enumerate(identity.feature_names)
            },
            **{
                name: identity.availability[:, index]
                for index, name in enumerate(AVAILABILITY_NAMES)
            },
        }
    )


def _write_challenge_identity(
    identity: ChallengeIdentityData, run_dir: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    path = atomic_write_parquet(
        Path(run_dir) / "challenge" / "identity_inputs.parquet",
        _identity_frame(identity),
    )
    counts = {
        str(segment): int(np.count_nonzero(identity.segment_ids == segment))
        for segment in sorted(set(int(value) for value in identity.segment_ids))
    }
    payload = {
        "contract": OUTAGE_CHALLENGE_FREEZE_CONTRACT,
        "phase": "label_blind_identity",
        "config_sha256": outage_gpu_config_sha256(dict(config)),
        "rows": identity.rows,
        "segment_rows": counts,
        "ordered_trial_uid_sha256": identity.ordered_trial_uid_sha256,
        "feature_names": list(identity.feature_names),
        "availability_names": list(AVAILABILITY_NAMES),
        "labels_projected": False,
        "recorded_imu1_projected": False,
        "label_seal_scope": str(config["challenge"]["label_seal_scope"]),
        "parent_release_transiently_deserializes_historical_segment_40_labels": bool(
            config["challenge"][
                "parent_release_transiently_deserializes_historical_segment_40_labels"
            ]
        ),
        "identity_file": _artifact_reference(path, run_dir, rows=identity.rows),
    }
    metadata_path = atomic_write_json(Path(run_dir) / "challenge" / "identity.json", payload)
    payload["metadata_file"] = _artifact_reference(metadata_path, run_dir)
    return payload


def _load_durable_challenge_identity(run_dir: str | Path) -> ChallengeIdentityData:
    metadata = read_json_object(Path(run_dir) / "challenge" / "identity.json")
    feature_names = tuple(str(value) for value in metadata["feature_names"])
    availability_names = tuple(str(value) for value in metadata["availability_names"])
    if feature_names != FUSION_COLUMNS or availability_names != AVAILABILITY_NAMES:
        raise GPUOutageExperimentError("durable challenge input layout changed")
    frame = _read_exact_parquet_projection(
        _verify_reference(run_dir, metadata["identity_file"]),
        ["trial_uid", "segment_id", "target_id", *feature_names, *availability_names],
    )
    identity = ChallengeIdentityData(
        feature_names=feature_names,
        X=frame.loc[:, list(feature_names)].to_numpy(dtype=np.float64),
        availability=frame.loc[:, list(availability_names)].to_numpy(dtype=np.float64),
        trial_uids=frame["trial_uid"].astype(str).to_numpy(),
        segment_ids=frame["segment_id"].to_numpy(dtype=np.int64),
        target_ids=frame["target_id"].astype(str).to_numpy(),
    )
    if identity.rows != int(metadata["rows"]):
        raise GPUOutageExperimentError("durable challenge identity row count changed")
    if identity.ordered_trial_uid_sha256 != str(metadata["ordered_trial_uid_sha256"]):
        raise GPUOutageExperimentError("durable challenge identity digest changed")
    return identity


def _validate_challenge_identity_contract(
    run_dir: str | Path,
    identity: ChallengeIdentityData,
    config: Mapping[str, Any],
) -> None:
    """Validate the complete label-free identity projection against its contract."""

    run_path = Path(run_dir).resolve()
    metadata_path = run_path / "challenge" / "identity.json"
    identity_path = run_path / "challenge" / "identity_inputs.parquet"
    metadata = read_json_object(metadata_path)
    role = config["row_roles"]["challenge"]
    segments = tuple(int(value) for value in role["segments"])
    expected_counts = {
        int(key): int(value) for key, value in role["expected_segment_rows"].items()
    }
    expected_rows = int(role["expected_rows"])
    expected_segment_rows = {
        str(segment): expected_counts[segment] for segment in segments
    }
    expected_identity_reference = _artifact_reference(
        identity_path, run_path, rows=expected_rows
    )
    expected_metadata = {
        "contract": OUTAGE_CHALLENGE_FREEZE_CONTRACT,
        "phase": "label_blind_identity",
        "config_sha256": outage_gpu_config_sha256(dict(config)),
        "rows": expected_rows,
        "segment_rows": expected_segment_rows,
        "ordered_trial_uid_sha256": str(role["ordered_trial_uid_sha256"]),
        "feature_names": list(FUSION_COLUMNS),
        "availability_names": list(AVAILABILITY_NAMES),
        "labels_projected": False,
        "recorded_imu1_projected": False,
        "label_seal_scope": str(config["challenge"]["label_seal_scope"]),
        "parent_release_transiently_deserializes_historical_segment_40_labels": bool(
            config["challenge"][
                "parent_release_transiently_deserializes_historical_segment_40_labels"
            ]
        ),
        "identity_file": expected_identity_reference,
    }
    if metadata != expected_metadata:
        raise GPUOutageExperimentError("challenge identity metadata/config binding changed")
    _require_canonical_reference(
        run_path,
        metadata["identity_file"],
        identity_path,
        rows=expected_rows,
    )
    expected_columns = [
        "trial_uid",
        "segment_id",
        "target_id",
        *FUSION_COLUMNS,
        *AVAILABILITY_NAMES,
    ]
    frame = _read_exact_parquet_projection(identity_path, expected_columns)
    if len(frame) != expected_rows:
        raise GPUOutageExperimentError("challenge identity schema/row count changed")
    if (
        frame["trial_uid"].astype(str).tolist() != identity.trial_uids.tolist()
        or not np.array_equal(
            frame["segment_id"].to_numpy(dtype=np.int64), identity.segment_ids
        )
        or frame["target_id"].astype(str).tolist() != identity.target_ids.tolist()
        or not np.array_equal(
            frame.loc[:, list(FUSION_COLUMNS)].to_numpy(dtype=np.float64),
            identity.X,
            equal_nan=True,
        )
        or not np.array_equal(
            frame.loc[:, list(AVAILABILITY_NAMES)].to_numpy(dtype=np.float64),
            identity.availability,
        )
    ):
        raise GPUOutageExperimentError(
            "challenge durable frame differs from validated in-memory identity"
        )
    if identity.rows != expected_rows:
        raise GPUOutageExperimentError("challenge identity in-memory row count changed")
    expected_segments = np.concatenate(
        [
            np.full(expected_counts[segment], segment, dtype=np.int64)
            for segment in segments
        ]
    )
    if not np.array_equal(identity.segment_ids, expected_segments):
        raise GPUOutageExperimentError(
            "challenge ordered segment-to-UID mapping changed"
        )
    if identity.ordered_trial_uid_sha256 != str(role["ordered_trial_uid_sha256"]):
        raise GPUOutageExperimentError("challenge ordered UID identity changed")
    expected_availability = np.tile(
        np.asarray(config["masking"]["challenge_availability"], dtype=np.float64),
        (expected_rows, 1),
    )
    if not np.array_equal(identity.availability, expected_availability) or not np.array_equal(
        expected_availability[0], np.asarray((1.0, 0.0, 1.0, 1.0))
    ):
        raise GPUOutageExperimentError("challenge forced IMU1 availability changed")
    imu1_indices = np.asarray(IMU_SENSOR_INDEX_MAP[1], dtype=np.int64)
    if not np.isnan(identity.X[:, imu1_indices]).all():
        raise GPUOutageExperimentError("challenge identity contains recorded IMU1 values")


def _write_development_inputs(
    data: DevelopmentData, run_dir: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze replay inputs so pre-label validation need not trust live sources."""

    frame = pd.DataFrame(
        {
            "trial_uid": data.trial_uids,
            "segment_id": data.segment_ids,
            "target_id": data.target_ids,
            **{
                name: data.X[:, index]
                for index, name in enumerate(data.feature_names)
            },
            **{name: data.y[:, index] for index, name in enumerate(_TRUE_COLUMNS)},
        }
    )
    path = atomic_write_parquet(
        Path(run_dir) / "development" / "replay_inputs.parquet", frame
    )
    sorted_uid_sha256 = hashlib.sha256(
        "\n".join(sorted(data.trial_uids.astype(str).tolist())).encode("utf-8")
    ).hexdigest()
    segment_rows = {
        str(segment): int(np.count_nonzero(data.segment_ids == segment))
        for segment in (36, 37, 38, 39)
    }
    payload = {
        "config_sha256": outage_gpu_config_sha256(dict(config)),
        "role": "development_replay_only",
        "rows": data.rows,
        "segments": [36, 37, 38, 39],
        "segment_rows": segment_rows,
        "sorted_trial_uid_sha256": sorted_uid_sha256,
        "feature_names": list(data.feature_names),
        "label_names": list(_TRUE_COLUMNS),
        "input_file": _artifact_reference(path, run_dir, rows=data.rows),
    }
    metadata_path = atomic_write_json(
        Path(run_dir) / "development" / "replay_inputs.json", payload
    )
    payload["metadata_file"] = _artifact_reference(metadata_path, run_dir)
    return payload


def _load_durable_development_inputs(run_dir: str | Path) -> DevelopmentData:
    metadata = read_json_object(Path(run_dir) / "development" / "replay_inputs.json")
    feature_names = tuple(str(value) for value in metadata["feature_names"])
    if feature_names != FUSION_COLUMNS or tuple(metadata["label_names"]) != _TRUE_COLUMNS:
        raise GPUOutageExperimentError("development replay layout changed")
    frame = _read_exact_parquet_projection(
        _verify_reference(run_dir, metadata["input_file"]),
        ["trial_uid", "segment_id", "target_id", *feature_names, *_TRUE_COLUMNS],
    )
    data = DevelopmentData(
        feature_names=feature_names,
        X=frame.loc[:, list(feature_names)].to_numpy(dtype=np.float64),
        y=frame.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64),
        availability=np.ones((len(frame), 4), dtype=np.float64),
        trial_uids=frame["trial_uid"].astype(str).to_numpy(),
        segment_ids=frame["segment_id"].to_numpy(dtype=np.int64),
        target_ids=frame["target_id"].astype(str).to_numpy(),
    )
    if data.rows != int(metadata["rows"]):
        raise GPUOutageExperimentError("development replay row count changed")
    actual_uid_hash = hashlib.sha256(
        "\n".join(sorted(data.trial_uids.astype(str).tolist())).encode("utf-8")
    ).hexdigest()
    if actual_uid_hash != str(metadata["sorted_trial_uid_sha256"]):
        raise GPUOutageExperimentError("development replay UID digest changed")
    return data


def _validate_development_identity_contract(
    data: DevelopmentData,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    run_dir: str | Path | None = None,
) -> None:
    role = config["row_roles"]["development"]
    role_segments = tuple(int(value) for value in role["segments"])
    counts = {
        int(segment): int(np.count_nonzero(data.segment_ids == int(segment)))
        for segment in role_segments
    }
    expected_counts = {
        int(key): int(value) for key, value in role["expected_segment_rows"].items()
    }
    uid_hash = hashlib.sha256(
        "\n".join(sorted(data.trial_uids.astype(str).tolist())).encode("utf-8")
    ).hexdigest()
    if run_dir is not None:
        input_path = Path(run_dir) / "development" / "replay_inputs.parquet"
        expected_metadata = {
            "config_sha256": outage_gpu_config_sha256(dict(config)),
            "role": "development_replay_only",
            "rows": int(role["expected_rows"]),
            "segments": [int(value) for value in role["segments"]],
            "segment_rows": {
                str(segment): expected_counts[segment]
                for segment in role_segments
            },
            "sorted_trial_uid_sha256": str(role["sorted_trial_uid_sha256"]),
            "feature_names": list(FUSION_COLUMNS),
            "label_names": list(_TRUE_COLUMNS),
            "input_file": _artifact_reference(
                input_path, run_dir, rows=int(role["expected_rows"])
            ),
        }
        if dict(metadata) != expected_metadata:
            raise GPUOutageExperimentError(
                "development replay metadata/config binding changed"
            )
        _require_canonical_reference(
            run_dir,
            metadata["input_file"],
            input_path,
            rows=int(role["expected_rows"]),
        )
    if (
        data.rows != int(role["expected_rows"])
        or counts != expected_counts
        or uid_hash != str(role["sorted_trial_uid_sha256"])
        or metadata.get("sorted_trial_uid_sha256") != uid_hash
        or {int(key): int(value) for key, value in metadata.get("segment_rows", {}).items()}
        != expected_counts
    ):
        raise GPUOutageExperimentError("development replay population identity changed")


def _standardize_challenge_features(
    identity: ChallengeIdentityData, preprocessor: TrainOnlyPreprocessor
) -> np.ndarray:
    """Scale available values with development state, then zero unavailable nodes."""

    from .gpu_outage_training import transform_outage_features

    return transform_outage_features(
        preprocessor=preprocessor,
        X_raw=identity.X,
        availability=identity.availability,
    )


def aggregate_outage_development(
    config: Mapping[str, Any],
    seed_results: Sequence[Mapping[str, Any]],
    run_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialise 80 trainable plus 16 virtual-blend context predictions."""

    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for result in seed_results:
        task = result["task"]
        key = _context_key(
            str(task["candidate_id"]),
            int(task["held_out_segment"]),
            int(task["forced_outage_index"]),
        )
        grouped.setdefault(key, []).append(result)
    contexts: list[dict[str, Any]] = []
    segments = tuple(int(value) for value in config["splits"]["development_segments"])
    for candidate_id in TRAINABLE_OUTAGE_CANDIDATE_IDS:
        for segment in segments:
            for outage_index in range(4):
                key = _context_key(candidate_id, segment, outage_index)
                rows = grouped.get(key, [])
                if len(rows) != 3:
                    raise GPUOutageExperimentError(f"context {key} lacks three seed fits")
                path = (
                    Path(run_dir)
                    / "development"
                    / "contexts"
                    / candidate_id
                    / f"segment{segment}__drop{outage_index}.parquet"
                )
                contexts.append(
                    _aggregate_context(
                        candidate_id=candidate_id,
                        held_out_segment=segment,
                        forced_outage_index=outage_index,
                        seed_results=rows,
                        run_dir=run_dir,
                        output_path=path,
                    )
                )
    blend_id = "equal_blend_tabm_chain_reconstruction"
    for segment in segments:
        for outage_index in range(4):
            components = {
                component: grouped[_context_key(component, segment, outage_index)]
                for component in ("tabm_batchensemble", "chain_masked_reconstruction")
            }
            path = (
                Path(run_dir)
                / "development"
                / "contexts"
                / blend_id
                / f"segment{segment}__drop{outage_index}.parquet"
            )
            contexts.append(
                _aggregate_blend_context(
                    held_out_segment=segment,
                    forced_outage_index=outage_index,
                    constituent_results=components,
                    run_dir=run_dir,
                    output_path=path,
                )
            )
    if len(contexts) != 96:
        raise GPUOutageExperimentError("outage-v3 must produce 96 scored contexts")
    context_path = atomic_write_json(
        Path(run_dir) / "development" / "context_metrics.json",
        {
            "config_sha256": outage_gpu_config_sha256(dict(config)),
            "candidate_count": 6,
            "context_count_per_candidate": 16,
            "contexts": contexts,
        },
    )
    selection = score_outage_candidates(contexts, config)
    selection["candidate"] = _candidate_map(config)[selection["selected_candidate_id"]]
    selection["context_metrics_file"] = _artifact_reference(context_path, run_dir)
    selection_path = atomic_write_json(
        Path(run_dir) / "development" / "selection.json", selection
    )
    selection["selection_file"] = _artifact_reference(selection_path, run_dir)
    return contexts, selection


def _selected_components(
    config: Mapping[str, Any], selected_candidate_id: str
) -> tuple[str, ...]:
    if selected_candidate_id != "equal_blend_tabm_chain_reconstruction":
        if selected_candidate_id not in TRAINABLE_OUTAGE_CANDIDATE_IDS:
            raise GPUOutageExperimentError("selected candidate is unknown")
        return (selected_candidate_id,)
    virtual = _candidate_map(config)[selected_candidate_id]
    components = tuple(str(row["candidate_id"]) for row in virtual["components"])
    if components != ("tabm_batchensemble", "chain_masked_reconstruction"):
        raise GPUOutageExperimentError("frozen blend components changed")
    return components


def outage_final_tasks(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    development_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if config["final_refit"]["epoch_median_scope"] != (
        "per_candidate_per_paired_seed_over_16_contexts"
    ) or config["final_refit"]["epoch_rounding"] != "half_up":
        raise GPUOutageExperimentError("final epoch median policy changed")
    candidates = _candidate_map(config)
    selected_id = str(selection["selected_candidate_id"])
    tasks: list[dict[str, Any]] = []
    for component in _selected_components(config, selected_id):
        for seed_index, model_seed in enumerate(config["training"]["paired_seed_values"]):
            epochs = [
                int(row["summary"]["best_epoch"])
                for row in development_results
                if row["task"]["candidate_id"] == component
                and int(row["task"]["paired_seed_index"]) == seed_index
            ]
            if len(epochs) != 16:
                raise GPUOutageExperimentError(
                    f"final component {component}/seed{seed_index} lacks 16 best epochs"
                )
            final_epochs = _round_median(epochs)
            task_id = f"final__{component}__seed{seed_index}__epochs{final_epochs}"
            tasks.append(
                {
                    "task_id": task_id,
                    "phase": "final",
                    "selected_candidate_id": selected_id,
                    "candidate_id": component,
                    "candidate": deepcopy(candidates[component]),
                    "paired_seed_index": seed_index,
                    "model_seed": int(model_seed),
                    "epochs": final_epochs,
                    "development_best_epochs": epochs,
                    "mask_held_out_segment": str(
                        config["final_refit"]["mask_draw_held_out_segment_token"]
                    ),
                    "mask_forced_outage_name": str(
                        config["final_refit"]["mask_draw_forced_outage_sensor"]
                    ),
                    "mask_forced_outage_index": _OUTAGE_NAME_TO_INDEX[
                        str(config["final_refit"]["mask_draw_forced_outage_sensor"])
                    ],
                }
            )
    return tasks


def _final_task_dir(run_dir: str | Path, task: Mapping[str, Any]) -> Path:
    return Path(run_dir) / "final" / "seed_tasks" / str(task["task_id"])


def _final_result_path(run_dir: str | Path, task: Mapping[str, Any]) -> Path:
    return _final_task_dir(run_dir, task) / "result.json"


def _execute_final_task(
    data: DevelopmentData,
    identity: ChallengeIdentityData,
    task: Mapping[str, Any],
    config: Mapping[str, Any],
    run_dir: str | Path,
    *,
    gpu_id: int,
    fit_fn: Callable[..., Mapping[str, Any]] | None = None,
    predict_fn: Callable[..., np.ndarray] | None = None,
) -> dict[str, Any]:
    """Exact-epoch all-development refit and label-free challenge prediction."""

    if fit_fn is None or predict_fn is None:
        from .gpu_outage_training import (
            fit_outage_seed_fixed_epochs,
            predict_outage_state,
        )

        fit_fn = fit_fn or fit_outage_seed_fixed_epochs
        predict_fn = predict_fn or predict_outage_state
    all_rows = np.ones(data.rows, dtype=bool)
    no_rows = np.zeros(data.rows, dtype=bool)
    preprocessor = _fit_preprocessor(
        data,
        all_rows,
        forbidden_rows=no_rows,
        split_context="outage_v3_final_all_development",
    )
    global_preprocessing = read_json_object(Path(run_dir) / "final" / "preprocessing.json")
    if preprocessor.metadata() != global_preprocessing:
        raise GPUOutageExperimentError("final worker preprocessing differs from frozen state")
    result = dict(
        fit_fn(
            X_train=preprocessor.transform_features(data.X).astype(np.float32),
            y_train=preprocessor.transform_targets(data.y).astype(np.float32),
            segment_ids=data.segment_ids,
            trial_uids=data.trial_uids,
            preprocessor=preprocessor,
            candidate=task["candidate"],
            config=config,
            model_seed=int(task["model_seed"]),
            paired_seed_index=int(task["paired_seed_index"]),
            epochs=int(task["epochs"]),
            mask_held_out_segment=task["mask_held_out_segment"],
            mask_forced_outage=int(task["mask_forced_outage_index"]),
            device="cuda:0",
        )
    )
    if int(result["trained_epochs"]) != int(task["epochs"]):
        raise GPUOutageExperimentError("final refit did not stop at the exact frozen epoch")
    challenge_scaled = _standardize_challenge_features(identity, preprocessor)
    prediction = np.asarray(
        predict_fn(
            state_dict=result["state_dict"],
            X_scaled=challenge_scaled,
            availability=identity.availability,
            preprocessor=preprocessor,
            candidate=task["candidate"],
            device="cuda:0",
        ),
        dtype=np.float64,
    )
    if prediction.shape != (identity.rows, 3) or not np.isfinite(prediction).all():
        raise GPUOutageExperimentError("final label-blind prediction is invalid")
    task_dir = _final_task_dir(run_dir, task)
    history = pd.DataFrame(list(result["history"]))
    history_path = atomic_write_parquet(task_dir / "history.parquet", history)
    checkpoint_path = _atomic_write_npz(task_dir / "checkpoint.npz", result["state_dict"])
    prediction_frame = pd.DataFrame(
        {
            "trial_uid": identity.trial_uids,
            "segment_id": identity.segment_ids,
            "target_id": identity.target_ids,
            **{name: prediction[:, index] for index, name in enumerate(_PRED_COLUMNS)},
        }
    )
    prediction_path = atomic_write_parquet(
        task_dir / "challenge_predictions.parquet", prediction_frame
    )
    artifacts = {
        "history": _artifact_reference(history_path, run_dir, rows=len(history)),
        "checkpoint": _artifact_reference(
            checkpoint_path, run_dir, tensor_count=len(result["state_dict"])
        ),
        "challenge_predictions": _artifact_reference(
            prediction_path, run_dir, rows=len(prediction_frame)
        ),
        "preprocessing": _artifact_reference(
            Path(run_dir) / "final" / "preprocessing.json", run_dir
        ),
    }
    payload = {
        "contract": OUTAGE_SEED_RESULT_CONTRACT,
        "config_sha256": outage_gpu_config_sha256(dict(config)),
        "task": dict(task),
        "gpu_id": int(gpu_id),
        "summary": _result_without_large_arrays(result),
        "challenge_labels_accessed": False,
        "artifacts": artifacts,
    }
    result_path = atomic_write_json(task_dir / "result.json", payload)
    payload["result_file"] = _artifact_reference(result_path, run_dir)
    return payload


def _load_valid_final_result(
    run_dir: str | Path, task: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _final_result_path(run_dir, task)
    if not path.is_file():
        return None
    try:
        result = read_json_object(path)
        if result.get("contract") != OUTAGE_SEED_RESULT_CONTRACT:
            return None
        if result.get("task") != dict(task) or result.get("challenge_labels_accessed") is not False:
            return None
        for reference in result.get("artifacts", {}).values():
            _verify_reference(run_dir, reference)
    except (KeyError, OSError, TypeError, ValueError, GPUOutageExperimentError):
        return None
    result["result_file"] = _artifact_reference(path, run_dir)
    return result


def _validate_final_seed_result(
    run_dir: str | Path,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
) -> pd.DataFrame:
    task = result["task"]
    summary = result["summary"]
    if result.get("config_sha256") != outage_gpu_config_sha256(dict(config)):
        raise GPUOutageExperimentError("final seed config identity changed")
    expected = {
        "candidate_id": task["candidate_id"],
        "fit_kind": "all_development_exact_fixed_epochs",
        "model_seed": task["model_seed"],
        "paired_seed_index": task["paired_seed_index"],
        "mask_held_out_segment": task["mask_held_out_segment"],
        "mask_forced_outage": task["mask_forced_outage_index"],
        "trained_epochs": task["epochs"],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise GPUOutageExperimentError(f"final seed summary field changed: {key}")
    history = pd.read_parquet(_verify_reference(run_dir, result["artifacts"]["history"]))
    if (
        len(history) != int(task["epochs"])
        or history["epoch"].astype(int).tolist() != list(range(1, int(task["epochs"]) + 1))
        or not np.isfinite(
            history[
                [
                    "scaled_objective_loss",
                    "scaled_endpoint_member_mse",
                    "scaled_masked_reconstruction_mse",
                ]
            ].to_numpy(dtype=np.float64)
        ).all()
    ):
        raise GPUOutageExperimentError("final seed exact-epoch history changed")
    if result.get("challenge_labels_accessed") is not False:
        raise GPUOutageExperimentError("final seed artifact claims challenge label access")
    global_preprocessing = _artifact_reference(
        Path(run_dir) / "final" / "preprocessing.json", run_dir
    )
    if result["artifacts"]["preprocessing"] != global_preprocessing:
        raise GPUOutageExperimentError("final seed preprocessing binding changed")
    return history


def _worker_runtime(gpu_id: int, config: Mapping[str, Any]) -> dict[str, Any]:
    _require_python_312()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(
        config["runtime"]["cublas_workspace_config"]
    )
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise GPUOutageExperimentError(
            f"worker for physical GPU {gpu_id} must see exactly one CUDA device"
        )
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    properties = torch.cuda.get_device_properties(0)
    return {
        "physical_gpu_id": int(gpu_id),
        "visible_gpu_id": 0,
        "name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda),
    }


def _worker_loop(
    gpu_id: int,
    config: dict[str, Any],
    development_release_path: str,
    run_dir: str,
    phase: str,
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        runtime = _worker_runtime(gpu_id, config)
        workspace = Path(str(config["_workspace_root"])).resolve()
        actual_source = outage_training_source_payload(workspace)["fingerprint"]
        if actual_source != str(config["_training_source_fingerprint"]):
            raise GPUOutageExperimentError("training source changed after run freeze")
        data = load_development_data(
            development_release_path,
            expected_hash=str(config["development_release"]["archive_sha256"]),
        )
        identity = (
            _load_durable_challenge_identity(run_dir) if phase == "final" else None
        )
        result_queue.put(
            {"event": "worker_ready", "gpu_id": int(gpu_id), "runtime": runtime}
        )
        while True:
            task = task_queue.get()
            if task is None:
                break
            try:
                if phase == "development":
                    result = _execute_development_task(
                        data, task, config, run_dir, gpu_id=gpu_id
                    )
                elif phase == "final":
                    assert identity is not None
                    result = _execute_final_task(
                        data, identity, task, config, run_dir, gpu_id=gpu_id
                    )
                else:
                    raise GPUOutageExperimentError(f"unknown worker phase {phase!r}")
                result_queue.put(
                    {
                        "event": "task_complete",
                        "gpu_id": int(gpu_id),
                        "task_id": str(task["task_id"]),
                        "result_file": result["result_file"],
                    }
                )
            except Exception as exc:  # pragma: no cover - child-process boundary
                result_queue.put(
                    {
                        "event": "task_error",
                        "gpu_id": int(gpu_id),
                        "task_id": str(task.get("task_id", "unknown")),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                return
    except Exception as exc:  # pragma: no cover - child-process boundary
        result_queue.put(
            {
                "event": "worker_error",
                "gpu_id": int(gpu_id),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _run_gpu_task_pool(
    *,
    config: dict[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    phase: str,
    development_release_path: str | Path,
    run_dir: str | Path,
    gpu_ids: Sequence[int],
) -> list[dict[str, Any]]:
    loader = _load_valid_task_result if phase == "development" else _load_valid_final_result
    completed: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for task in tasks:
        result = loader(run_dir, task)
        if result is None:
            pending.append(dict(task))
        else:
            completed[str(task["task_id"])] = result
    if not pending:
        return [completed[str(task["task_id"])] for task in tasks]

    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    for task in pending:
        task_queue.put(task)
    for _ in gpu_ids:
        task_queue.put(None)
    processes = [
        context.Process(
            target=_worker_loop,
            args=(
                int(gpu_id),
                config,
                str(Path(development_release_path).resolve()),
                str(Path(run_dir).resolve()),
                phase,
                task_queue,
                result_queue,
            ),
            name=f"outage-v3-{phase}-gpu{gpu_id}",
        )
        for gpu_id in gpu_ids
    ]
    for process in processes:
        process.start()
    ready: dict[int, dict[str, Any]] = {}
    remaining = {str(task["task_id"]) for task in pending}
    try:
        while remaining or len(ready) < len(processes):
            try:
                event = result_queue.get(timeout=1.0)
            except queue.Empty:
                dead = [process for process in processes if not process.is_alive()]
                if dead and (remaining or len(ready) < len(processes)):
                    codes = {process.name: process.exitcode for process in dead}
                    raise GPUOutageExperimentError(
                        f"GPU worker exited before completion: {codes}"
                    )
                continue
            kind = str(event.get("event"))
            if kind == "worker_ready":
                ready[int(event["gpu_id"])] = dict(event["runtime"])
            elif kind == "task_complete":
                task_id = str(event["task_id"])
                remaining.discard(task_id)
            elif kind in {"task_error", "worker_error"}:
                raise GPUOutageExperimentError(
                    f"{kind} on GPU {event.get('gpu_id')}: {event.get('error')}\n"
                    f"{event.get('traceback', '')}"
                )
            else:
                raise GPUOutageExperimentError(f"unknown worker event: {event}")
    finally:
        if remaining:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join(timeout=10.0)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
    bad_exit_codes = {
        process.name: process.exitcode
        for process in processes
        if process.exitcode not in (0, None)
    }
    if bad_exit_codes and not remaining:
        raise GPUOutageExperimentError(
            f"GPU workers exited non-zero after task processing: {bad_exit_codes}"
        )
    if remaining:
        raise GPUOutageExperimentError(
            f"{phase} pool stopped with {len(remaining)} unfinished tasks"
        )
    runtime_path = Path(run_dir) / "provenance" / f"{phase}_gpu_workers.json"
    atomic_write_json(
        runtime_path,
        {
            "phase": phase,
            "worker_count": len(ready),
            "workers": [ready[key] for key in sorted(ready)],
        },
    )
    results: list[dict[str, Any]] = []
    for task in tasks:
        result = loader(run_dir, task)
        if result is None:
            raise GPUOutageExperimentError(
                f"worker did not leave a valid result for {task['task_id']}"
            )
        results.append(result)
    return results


def _validate_worker_provenance(
    run_dir: str | Path, *, phase: str, expected_workers: int
) -> None:
    payload = read_json_object(
        Path(run_dir) / "provenance" / f"{phase}_gpu_workers.json"
    )
    workers = payload.get("workers")
    if (
        payload.get("phase") != phase
        or int(payload.get("worker_count", -1)) != expected_workers
        or not isinstance(workers, list)
        or len(workers) != expected_workers
        or len({int(row["physical_gpu_id"]) for row in workers}) != expected_workers
        or any(int(row.get("visible_gpu_id", -1)) != 0 for row in workers)
        or any(int(row.get("total_memory_bytes", 0)) <= 0 for row in workers)
    ):
        raise GPUOutageExperimentError(
            f"{phase} did not record exactly {expected_workers} isolated GPU workers"
        )


def _aggregate_final_predictions(
    *,
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    final_results: Sequence[Mapping[str, Any]],
    identity: ChallengeIdentityData,
    run_dir: str | Path,
) -> dict[str, Any]:
    selected_id = str(selection["selected_candidate_id"])
    components = _selected_components(config, selected_id)
    by_component_seed: dict[tuple[str, int], pd.DataFrame] = {}
    for result in final_results:
        task = result["task"]
        frame = _read_exact_parquet_projection(
            _verify_reference(run_dir, result["artifacts"]["challenge_predictions"]),
            ["trial_uid", "segment_id", "target_id", *_PRED_COLUMNS],
        )
        key = (str(task["candidate_id"]), int(task["paired_seed_index"]))
        if key in by_component_seed:
            raise GPUOutageExperimentError("duplicate final component/seed prediction")
        by_component_seed[key] = frame
    identity_columns = ["trial_uid", "segment_id", "target_id"]
    expected_identity = _identity_frame(identity)[identity_columns]
    component_seed_values: dict[tuple[str, int], np.ndarray] = {}
    for component in components:
        for seed_index in range(3):
            key = (component, seed_index)
            frame = by_component_seed.get(key)
            if frame is None or not frame[identity_columns].equals(expected_identity):
                raise GPUOutageExperimentError("final component prediction identity changed")
            component_seed_values[key] = frame.loc[:, list(_PRED_COLUMNS)].to_numpy(
                dtype=np.float64
            )
    selected_seed_values: list[np.ndarray] = []
    for seed_index in range(3):
        if len(components) == 1:
            value = component_seed_values[(components[0], seed_index)]
        else:
            value = 0.5 * component_seed_values[(components[0], seed_index)] + 0.5 * component_seed_values[(components[1], seed_index)]
        selected_seed_values.append(value)
    ensemble = np.mean(np.stack(selected_seed_values, axis=0), axis=0)
    output = expected_identity.copy()
    for seed_index, values in enumerate(selected_seed_values):
        for axis_index, axis in enumerate(("x", "y", "z")):
            output[f"seed{seed_index}_pred_{axis}_mm"] = values[:, axis_index]
    for axis_index, name in enumerate(_PRED_COLUMNS):
        output[name] = ensemble[:, axis_index]
    prediction_path = atomic_write_parquet(
        Path(run_dir) / "challenge" / "predictions.label_blind.parquet", output
    )
    checkpoint_references = [
        result["artifacts"]["checkpoint"]
        for result in sorted(final_results, key=lambda row: str(row["task"]["task_id"]))
    ]
    preprocessing_reference = _artifact_reference(
        Path(run_dir) / "final" / "preprocessing.json", run_dir
    )
    identity_reference = _artifact_reference(
        Path(run_dir) / "challenge" / "identity_inputs.parquet", run_dir
    )
    return {
        "selected_candidate_id": selected_id,
        "selected_components": list(components),
        "selected_component_weights": (
            {components[0]: 1.0}
            if len(components) == 1
            else {components[0]: 0.5, components[1]: 0.5}
        ),
        "seed_count": 3,
        "checkpoint_files": checkpoint_references,
        "model_bundle_sha256": stable_hash(checkpoint_references),
        "preprocessing_file": preprocessing_reference,
        "identity_file": identity_reference,
        "prediction_file": _artifact_reference(
            prediction_path, run_dir, rows=len(output)
        ),
        "ordered_trial_uid_sha256": ordered_trial_uid_sha256(
            output["trial_uid"].astype(str).tolist()
        ),
        "challenge_labels_accessed": False,
    }


def _challenge_report(
    *,
    config: Mapping[str, Any],
    selected_candidate_id: str,
    evaluation: pd.DataFrame,
) -> dict[str, Any]:
    truth = evaluation.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
    ensemble = evaluation.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64)
    pooled = regression_metrics(truth, ensemble)
    per_segment: list[dict[str, Any]] = []
    for segment in (40, 41):
        mask = evaluation["segment_id"].to_numpy(dtype=np.int64) == segment
        if not mask.any():
            raise GPUOutageExperimentError(f"challenge segment {segment} is empty")
        per_segment.append(
            {"segment_id": segment, "metrics": regression_metrics(truth[mask], ensemble[mask])}
        )
    seed_metrics: list[dict[str, Any]] = []
    segment_seed_rmses: dict[int, list[float]] = {40: [], 41: []}
    for seed_index in range(3):
        columns = [f"seed{seed_index}_pred_{axis}_mm" for axis in ("x", "y", "z")]
        seed_prediction = evaluation.loc[:, columns].to_numpy(dtype=np.float64)
        seed_pooled = regression_metrics(truth, seed_prediction)
        seed_segments: list[dict[str, Any]] = []
        for segment in (40, 41):
            mask = evaluation["segment_id"].to_numpy(dtype=np.int64) == segment
            metrics = regression_metrics(truth[mask], seed_prediction[mask])
            segment_seed_rmses[segment].append(float(metrics["rmse_3d_mm"]))
            seed_segments.append({"segment_id": segment, "metrics": metrics})
        seed_metrics.append(
            {"paired_seed_index": seed_index, "pooled_metrics": seed_pooled, "per_segment": seed_segments}
        )
    pooled_seed_rmses = [
        float(row["pooled_metrics"]["rmse_3d_mm"]) for row in seed_metrics
    ]
    return {
        "contract": OUTAGE_RUN_MANIFEST_CONTRACT,
        "report_kind": "selected_only_one_time_challenge",
        "config_sha256": outage_gpu_config_sha256(dict(config)),
        "selected_candidate_id": selected_candidate_id,
        "challenge_designation": str(config["challenge"]["designation"]),
        "forced_outage_sensor": str(config["challenge"]["forced_outage_sensor"]),
        "label_seal_scope": str(config["challenge"]["label_seal_scope"]),
        "parent_release_transiently_deserializes_historical_segment_40_labels": bool(
            config["challenge"][
                "parent_release_transiently_deserializes_historical_segment_40_labels"
            ]
        ),
        "development_api_returns_segment_40_rows_or_labels": bool(
            config["challenge"]["development_api_returns_segment_40_rows_or_labels"]
        ),
        "scientific_interpretation": {
            "primary_previously_unused_segment": int(
                config["challenge"]["primary_previously_unused_segment"]
            ),
            "secondary_previously_observed_segment": int(
                config["challenge"]["secondary_previously_observed_segment"]
            ),
            "combined_two_segment_metrics_are_descriptive_only": bool(
                config["challenge"]["combined_two_segment_metrics_are_descriptive_only"]
            ),
        },
        "row_count": int(len(evaluation)),
        "pooled_metrics": pooled,
        "per_segment": per_segment,
        "equal_segment_macro_rmse_3d_mm": float(
            np.mean([float(row["metrics"]["rmse_3d_mm"]) for row in per_segment])
        ),
        "equal_segment_macro_p95_3d_mm": float(
            np.mean([float(row["metrics"]["p95_3d_error_mm"]) for row in per_segment])
        ),
        "seed_metrics": seed_metrics,
        "seed_dispersion": {
            "pooled_rmse_3d_population_std_mm": float(np.std(pooled_seed_rmses, ddof=0)),
            "per_segment_rmse_3d_population_std_mm": {
                str(segment): float(np.std(values, ddof=0))
                for segment, values in segment_seed_rmses.items()
            },
        },
        "acceptance_gate": "none_diagnostic_only",
        "challenge_used_for_tuning_or_reselection": False,
        "claim_boundaries": {
            "independent_session": False,
            "normal_four_imu": False,
            "robot_base_or_tcp_calibrated": False,
            "dynamic_or_control_ready": False,
        },
    }


def _open_challenge_once(
    *,
    config: dict[str, Any],
    source_config: Mapping[str, Any],
    identity: ChallengeIdentityData,
    selection: Mapping[str, Any],
    freeze: Mapping[str, Any],
    run_dir: str | Path,
    label_loader: Callable[..., Any] = load_challenge_labels_once,
) -> dict[str, Any]:
    challenge_dir = Path(run_dir) / "challenge"
    freeze_path = challenge_dir / "freeze.before_label_access.json"
    started_path = challenge_dir / "label_access_1.started.json"
    completed_path = challenge_dir / "label_access_1.completed.json"
    if completed_path.exists():
        return read_json_object(completed_path)
    if started_path.exists():
        raise GPUOutageExperimentError(
            "challenge label access previously started without durable completion; "
            "fail closed rather than opening labels a second time"
        )
    prelabel_validation_path = challenge_dir / "prelabel_semantic_validation.json"
    prelabel_validation = read_json_object(prelabel_validation_path)
    if (
        prelabel_validation.get("status") != "ok"
        or prelabel_validation.get("challenge_labels_accessed") is not False
        or prelabel_validation.get("selected_candidate_id")
        != str(selection["selected_candidate_id"])
    ):
        raise GPUOutageExperimentError("durable pre-label semantic validation is invalid")
    durable_freeze = read_json_object(freeze_path)
    supplied_freeze = dict(freeze)
    supplied_freeze.pop("freeze_file", None)
    if supplied_freeze != durable_freeze:
        raise GPUOutageExperimentError(
            "in-memory challenge freeze differs from durable frozen evidence"
        )
    durable_selection = read_json_object(
        Path(run_dir) / "development" / "selection.json"
    )
    supplied_selection = dict(selection)
    supplied_selection.pop("selection_file", None)
    if supplied_selection != durable_selection:
        raise GPUOutageExperimentError(
            "in-memory selection differs from durable development selection"
        )
    freeze_reference = _artifact_reference(freeze_path, run_dir)
    for reference in (
        freeze["prediction_file"],
        freeze["preprocessing_file"],
        freeze["identity_file"],
        *freeze["checkpoint_files"],
    ):
        _verify_reference(run_dir, reference)
    started = {
        "contract": OUTAGE_CHALLENGE_FREEZE_CONTRACT,
        "access_index": 1,
        "status": "started_fail_closed",
        "maximum_source_label_access_passes": 1,
        "freeze_file": freeze_reference,
        "prelabel_semantic_validation_file": _artifact_reference(
            prelabel_validation_path, run_dir
        ),
        "selected_candidate_id": str(selection["selected_candidate_id"]),
        "ordered_trial_uid_sha256": identity.ordered_trial_uid_sha256,
        "requested_label_columns": list(config["task"]["source_label_columns"]),
        "source_member_sha256": str(config["source"]["member_sha256"]),
    }
    _exclusive_write_json(started_path, started)

    challenge = label_loader(source_config, config, identity)
    if not np.array_equal(challenge.trial_uids, identity.trial_uids):
        raise GPUOutageExperimentError("unsealed challenge labels are not identity-aligned")
    predictions = _read_exact_parquet_projection(
        _verify_reference(run_dir, freeze["prediction_file"]),
        [
            "trial_uid",
            "segment_id",
            "target_id",
            *[
                f"seed{seed}_pred_{axis}_mm"
                for seed in range(3)
                for axis in ("x", "y", "z")
            ],
            *_PRED_COLUMNS,
        ],
    )
    if predictions["trial_uid"].astype(str).tolist() != identity.trial_uids.tolist():
        raise GPUOutageExperimentError("frozen predictions are not challenge-aligned")
    evaluation = predictions.copy()
    for index, name in enumerate(_TRUE_COLUMNS):
        evaluation[name] = challenge.y[:, index]
    ordered_columns = [
        "trial_uid",
        "segment_id",
        "target_id",
        *_TRUE_COLUMNS,
        *[f"seed{seed}_pred_{axis}_mm" for seed in range(3) for axis in ("x", "y", "z")],
        *_PRED_COLUMNS,
    ]
    evaluation = evaluation.loc[:, ordered_columns]
    evaluation_path = atomic_write_parquet(
        challenge_dir / "evaluation.selected_only.parquet", evaluation
    )
    report = _challenge_report(
        config=config,
        selected_candidate_id=str(selection["selected_candidate_id"]),
        evaluation=evaluation,
    )
    report_path = atomic_write_json(challenge_dir / "report.json", report)
    completion = {
        "contract": OUTAGE_CHALLENGE_FREEZE_CONTRACT,
        "access_index": 1,
        "status": "completed",
        "source_label_access_pass_count": 1,
        "freeze_file": freeze_reference,
        "started_file": _artifact_reference(started_path, run_dir),
        "evaluation_file": _artifact_reference(
            evaluation_path, run_dir, rows=len(evaluation)
        ),
        "report_file": _artifact_reference(report_path, run_dir),
        "selected_candidate_id": str(selection["selected_candidate_id"]),
        "candidate_specific_challenge_reports": 1,
        "reselection_after_access": False,
    }
    atomic_write_json(completed_path, completion)
    access_log = {
        "contract": OUTAGE_CHALLENGE_FREEZE_CONTRACT,
        "maximum_source_label_access_passes": 1,
        "source_label_access_pass_count": 1,
        "events": [
            _artifact_reference(started_path, run_dir),
            _artifact_reference(completed_path, run_dir),
        ],
        "status": "completed",
    }
    atomic_write_json(challenge_dir / "label_access_log.json", access_log)
    completion["completion_file"] = _artifact_reference(completed_path, run_dir)
    return completion


def _write_manifest(run_dir: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    selection = read_json_object(Path(run_dir) / "development" / "selection.json")
    freeze = read_json_object(Path(run_dir) / "challenge" / "freeze.before_label_access.json")
    report = read_json_object(Path(run_dir) / "challenge" / "report.json")
    access_log = read_json_object(Path(run_dir) / "challenge" / "label_access_log.json")
    manifest = {
        "contract": OUTAGE_RUN_MANIFEST_CONTRACT,
        "experiment_contract": OUTAGE_GPU_EXPERIMENT_CONTRACT,
        "config_sha256": outage_gpu_config_sha256(dict(config)),
        "status": "complete",
        "counts": {
            "development_rows": int(config["row_roles"]["development"]["expected_rows"]),
            "challenge_rows": int(config["row_roles"]["challenge"]["expected_rows"]),
            "trainable_candidates": 5,
            "scored_candidates": 6,
            "contexts_per_candidate": 16,
            "development_seed_fits": 240,
            "challenge_label_access_passes": 1,
        },
        "selection": {
            "selected_candidate_id": selection["selected_candidate_id"],
            "selected_outage_robust_j_mm": selection["selected_outage_robust_j_mm"],
            "challenge_used_for_selection": False,
        },
        "final_freeze": {
            "model_bundle_sha256": freeze["model_bundle_sha256"],
            "preprocessing_sha256": freeze["preprocessing_file"]["sha256"],
            "prediction_sha256": freeze["prediction_file"]["sha256"],
            "identity_sha256": freeze["identity_file"]["sha256"],
        },
        "challenge": {
            "designation": config["challenge"]["designation"],
            "selected_candidate_id": report["selected_candidate_id"],
            "acceptance_gate": "none_diagnostic_only",
            "source_label_access_pass_count": access_log["source_label_access_pass_count"],
            "independent_session": False,
            "normal_four_imu": False,
        },
        "files": manifest_files(run_dir),
    }
    atomic_write_json(Path(run_dir) / "manifest.json", manifest)
    return manifest


def _write_development_manifest(
    run_dir: str | Path, config: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    files = _development_manifest_files(run_dir)
    manifest = {
        "contract": OUTAGE_RUN_MANIFEST_CONTRACT,
        "experiment_contract": OUTAGE_GPU_EXPERIMENT_CONTRACT,
        "stage": "development_complete_challenge_labels_sealed",
        "config_sha256": outage_gpu_config_sha256(dict(config)),
        "counts": {
            "development_seed_fits": 240,
            "scored_contexts": 96,
            "challenge_label_access_passes": 0,
        },
        "selection": {
            "selected_candidate_id": selection["selected_candidate_id"],
            "selected_outage_robust_j_mm": selection["selected_outage_robust_j_mm"],
            "challenge_used_for_selection": False,
        },
        "challenge_labels_sealed": True,
        "files": files,
    }
    atomic_write_json(Path(run_dir) / "development_manifest.json", manifest)
    return manifest


def _development_manifest_files(run_dir: str | Path) -> list[dict[str, Any]]:
    allowed_challenge = {"challenge/identity.json", "challenge/identity_inputs.parquet"}
    return [
        row
        for row in manifest_files(run_dir)
        if row["path"] != "development_manifest.json"
        and not row["path"].startswith("final/")
        and (
            not row["path"].startswith("challenge/")
            or row["path"] in allowed_challenge
        )
        and row["path"] != "provenance/final_gpu_workers.json"
    ]


def _development_stage_snapshot(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    run_path = Path(run_dir).resolve()
    return {
        name: _artifact_reference(path, run_path)
        for name, path in {
            "context_metrics": run_path / "development" / "context_metrics.json",
            "selection": run_path / "development" / "selection.json",
            "manifest": run_path / "development_manifest.json",
        }.items()
    }


def _require_unchanged_development_stage(
    run_dir: str | Path, snapshot: Mapping[str, Mapping[str, Any]]
) -> None:
    current = _development_stage_snapshot(run_dir)
    if current != {key: dict(value) for key, value in snapshot.items()}:
        raise GPUOutageExperimentError(
            "validated development contexts/selection/manifest changed before final refit"
        )


def _resolved_source_config(config: Mapping[str, Any]) -> dict[str, Any]:
    source = deepcopy(dict(config["source"]))
    source["archive"] = str(resolve_outage_source_path(dict(config)))
    return source


def run_outage_gpu_experiment(
    config: dict[str, Any],
    run_dir: str | Path,
    *,
    gpu_ids: Sequence[int] = (0, 1, 2, 3),
    resume: bool = False,
    finalize_challenge: bool = False,
) -> dict[str, Any]:
    """Run the frozen 240-fit development study and one-time challenge."""

    _require_python_312()
    validate_outage_gpu_config(config)
    gpu_ids = _validate_gpu_ids(gpu_ids, int(config["runtime"]["gpu_count"]))
    if finalize_challenge and not resume:
        raise GPUOutageExperimentError(
            "challenge finalization requires a prior sealed development run and "
            "explicit --resume --finalize-challenge"
        )
    run_path = Path(run_dir).resolve()
    manifest_path = run_path / "manifest.json"
    if resume and manifest_path.is_file():
        return validate_outage_gpu_run(run_path)
    development_manifest_path = run_path / "development_manifest.json"
    if finalize_challenge and not development_manifest_path.is_file():
        raise GPUOutageExperimentError(
            "challenge finalization requires an existing development_manifest.json"
        )
    if (run_path / "challenge" / "label_access_1.started.json").exists() and not (
        run_path / "challenge" / "label_access_1.completed.json"
    ).exists():
        raise GPUOutageExperimentError(
            "prior challenge label access is incomplete; a second pass is forbidden"
        )

    workspace = Path(str(config["_workspace_root"])).resolve()
    source_snapshot = outage_training_source_payload(workspace)
    config["_training_source_fingerprint"] = source_snapshot["fingerprint"]
    config_sha = outage_gpu_config_sha256(config)
    identity_payload = _run_identity_payload(config, source_snapshot["fingerprint"])
    completed_access = run_path / "challenge" / "label_access_1.completed.json"
    staged_snapshot: dict[str, dict[str, Any]] | None = None
    if finalize_challenge and not completed_access.is_file():
        validate_outage_development_run(run_path, _allow_final_artifacts=True)
        staged_snapshot = _development_stage_snapshot(run_path)
    elif resume and development_manifest_path.is_file() and not completed_access.is_file():
        validate_outage_development_run(run_path)
        return read_json_object(development_manifest_path)

    run_path = prepare_experiment_directory(run_path, identity=identity_payload, resume=resume)
    resolved_config_path = run_path / "config.resolved.json"
    expected_resolved = _resolved_config_payload(
        config, source_snapshot["fingerprint"]
    )
    if resolved_config_path.is_file():
        existing_resolved = _read_resolved_outage_config(resolved_config_path)
        if (
            public_outage_gpu_config(existing_resolved)
            != public_outage_gpu_config(config)
            or existing_resolved.get("_training_source_fingerprint")
            != source_snapshot["fingerprint"]
        ):
            raise GPUOutageExperimentError("resumed resolved config/provenance changed")
    else:
        atomic_write_json(resolved_config_path, expected_resolved)
    source_snapshot_path = run_path / "provenance" / "source_snapshot.json"
    if source_snapshot_path.is_file():
        if read_json_object(source_snapshot_path) != source_snapshot:
            raise GPUOutageExperimentError("resumed source snapshot changed")
    else:
        atomic_write_json(source_snapshot_path, source_snapshot)

    if completed_access.is_file():
        access_log_path = run_path / "challenge" / "label_access_log.json"
        if not access_log_path.is_file():
            completed_payload = read_json_object(completed_access)
            if (
                completed_payload.get("status") != "completed"
                or int(completed_payload.get("source_label_access_pass_count", -1)) != 1
            ):
                raise GPUOutageExperimentError(
                    "completed label access evidence is invalid"
                )
            atomic_write_json(
                access_log_path,
                {
                    "contract": OUTAGE_CHALLENGE_FREEZE_CONTRACT,
                    "maximum_source_label_access_passes": 1,
                    "source_label_access_pass_count": 1,
                    "events": [
                        _artifact_reference(
                            run_path / "challenge" / "label_access_1.started.json",
                            run_path,
                        ),
                        _artifact_reference(completed_access, run_path),
                    ],
                    "status": "completed",
                },
            )
        manifest = _write_manifest(run_path, config)
        validate_outage_gpu_run(run_path)
        return manifest

    identity_metadata_path = run_path / "challenge" / "identity.json"
    if identity_metadata_path.is_file():
        challenge_identity = _load_durable_challenge_identity(run_path)
    else:
        challenge_identity = load_challenge_identity(_resolved_source_config(config), config)
        _write_challenge_identity(challenge_identity, run_path, config)

    development_release = resolve_outage_development_release_path(config)
    replay_metadata_path = run_path / "development" / "replay_inputs.json"
    if replay_metadata_path.is_file():
        development_data = _load_durable_development_inputs(run_path)
    else:
        development_data = load_development_data(
            development_release,
            expected_hash=str(config["development_release"]["archive_sha256"]),
        )
        _write_development_inputs(development_data, run_path, config)
    development_tasks = outage_development_tasks(config)
    if finalize_challenge:
        development_results = []
        for task in development_tasks:
            result = _load_valid_task_result(run_path, task)
            if result is None:
                raise GPUOutageExperimentError(
                    f"validated staged task disappeared: {task['task_id']}"
                )
            development_results.append(result)
        selection = read_json_object(run_path / "development" / "selection.json")
        assert staged_snapshot is not None
        _require_unchanged_development_stage(run_path, staged_snapshot)
    else:
        development_results = _run_gpu_task_pool(
            config=config,
            tasks=development_tasks,
            phase="development",
            development_release_path=development_release,
            run_dir=run_path,
            gpu_ids=gpu_ids,
        )
        _, selection = aggregate_outage_development(
            config, development_results, run_path
        )
        development_manifest = _write_development_manifest(
            run_path, config, selection
        )
        validate_outage_development_run(run_path)
        return development_manifest

    all_rows = np.ones(development_data.rows, dtype=bool)
    final_preprocessor = _fit_preprocessor(
        development_data,
        all_rows,
        forbidden_rows=np.zeros(development_data.rows, dtype=bool),
        split_context="outage_v3_final_all_development",
    )
    preprocessing_path = run_path / "final" / "preprocessing.json"
    if preprocessing_path.is_file():
        if read_json_object(preprocessing_path) != final_preprocessor.metadata():
            raise GPUOutageExperimentError("resumed final preprocessing state changed")
    else:
        atomic_write_json(preprocessing_path, final_preprocessor.metadata())

    final_tasks = outage_final_tasks(config, selection, development_results)
    final_results = _run_gpu_task_pool(
        config=config,
        tasks=final_tasks,
        phase="final",
        development_release_path=development_release,
        run_dir=run_path,
        gpu_ids=gpu_ids,
    )
    final_prediction = _aggregate_final_predictions(
        config=config,
        selection=selection,
        final_results=final_results,
        identity=challenge_identity,
        run_dir=run_path,
    )
    freeze = {
        "contract": OUTAGE_CHALLENGE_FREEZE_CONTRACT,
        "config_sha256": config_sha,
        "status": "frozen_before_label_access",
        "selection_file": _artifact_reference(
            run_path / "development" / "selection.json", run_path
        ),
        "selection_challenge_metrics_used": False,
        "final_refit_uses_development_segments_only": [36, 37, 38, 39],
        **final_prediction,
    }
    freeze_path = atomic_write_json(
        run_path / "challenge" / "freeze.before_label_access.json", freeze
    )
    freeze["freeze_file"] = _artifact_reference(freeze_path, run_path)
    _prelabel_gate_then_open(
        run_dir=run_path,
        config=config,
        source_config=_resolved_source_config(config),
        identity=challenge_identity,
        selection=selection,
        freeze=freeze,
    )
    manifest = _write_manifest(run_path, config)
    validate_outage_gpu_run(run_path)
    return manifest


def _assert_json_close(actual: Any, expected: Any, *, path: str = "root") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise GPUOutageExperimentError(f"JSON mapping changed at {path}")
        for key, value in expected.items():
            _assert_json_close(actual[key], value, path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise GPUOutageExperimentError(f"JSON list changed at {path}")
        for index, value in enumerate(expected):
            _assert_json_close(actual[index], value, path=f"{path}[{index}]")
        return
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        if not math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-9
        ):
            raise GPUOutageExperimentError(
                f"numeric evidence changed at {path}: {actual} != {expected}"
            )
        return
    if actual != expected:
        raise GPUOutageExperimentError(
            f"JSON evidence changed at {path}: {actual!r} != {expected!r}"
        )


def _validate_development_seed_result(
    run_dir: str | Path,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
) -> pd.DataFrame:
    task = result["task"]
    if result["config_sha256"] != outage_gpu_config_sha256(dict(config)):
        raise GPUOutageExperimentError("development seed config identity changed")
    summary = result["summary"]
    expected_fields = {
        "candidate_id": task["candidate_id"],
        "model_seed": task["model_seed"],
        "paired_seed_index": task["paired_seed_index"],
        "held_out_segment": str(task["held_out_segment"]),
        "forced_outage": task["forced_outage_index"],
        "trained_epochs": int(config["training"]["epoch_budget"]),
    }
    for key, expected in expected_fields.items():
        if summary.get(key) != expected:
            raise GPUOutageExperimentError(
                f"development seed summary field changed: {key}"
            )
    history = pd.read_parquet(_verify_reference(run_dir, result["artifacts"]["history"]))
    if len(history) != int(config["training"]["epoch_budget"]):
        raise GPUOutageExperimentError("development history does not contain 400 epochs")
    scores = history["validation_rmse_3d_mm"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise GPUOutageExperimentError("development history contains non-finite RMSE")
    running_best = math.inf
    running_best_epoch = 0
    for index, score in enumerate(scores):
        improved = float(score) < running_best
        if improved:
            running_best = float(score)
            running_best_epoch = index + 1
        row = history.iloc[index]
        if (
            bool(row["checkpoint_improved"]) is not improved
            or int(row["best_epoch_so_far"]) != running_best_epoch
            or not math.isclose(
                float(row["best_validation_rmse_3d_mm"]),
                running_best,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise GPUOutageExperimentError(
                f"development checkpoint history trajectory changed at epoch {index + 1}"
            )
    earliest_best = int(np.flatnonzero(scores == np.min(scores))[0]) + 1
    if int(summary["best_epoch"]) != earliest_best:
        raise GPUOutageExperimentError("development checkpoint is not earliest minimum RMSE")
    if not math.isclose(
        float(summary["best_validation_rmse_3d_mm"]),
        float(scores[earliest_best - 1]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise GPUOutageExperimentError("development best RMSE/history mismatch")
    predictions = _read_prediction_reference(run_dir, result)
    required = {"trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS, *_PRED_COLUMNS}
    if set(predictions.columns) != required:
        raise GPUOutageExperimentError("development seed prediction schema changed")
    if set(predictions["segment_id"].astype(int)) != {int(task["held_out_segment"])}:
        raise GPUOutageExperimentError("development prediction contains wrong segment")
    metrics = regression_metrics(
        predictions.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64),
        predictions.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64),
    )
    _assert_json_close(result["metrics"], metrics, path="development.metrics")
    _assert_json_close(
        summary["eval_metrics"], metrics, path="development.summary.eval_metrics"
    )
    if not math.isclose(
        float(metrics["rmse_3d_mm"]),
        float(summary["best_validation_rmse_3d_mm"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise GPUOutageExperimentError(
            "development selected prediction RMSE differs from history best"
        )
    preprocessing = read_json_object(
        _verify_reference(run_dir, result["artifacts"]["preprocessing"])
    )
    audit = preprocessing.get("leakage_audit", {})
    if audit.get("fit_scope") != "real_train_only" or int(
        audit.get("forbidden_uid_overlap_count", -1)
    ) != 0:
        raise GPUOutageExperimentError("development preprocessing leakage audit failed")
    expected_context = _development_split_context(task)
    if audit.get("split_context") != expected_context:
        raise GPUOutageExperimentError("development preprocessing context changed")
    load_outage_checkpoint(_verify_reference(run_dir, result["artifacts"]["checkpoint"]))
    return history


def _validate_context_evidence(
    run_dir: str | Path,
    contexts: Sequence[Mapping[str, Any]],
    seed_results: Sequence[Mapping[str, Any]] | None = None,
    development_data: DevelopmentData | None = None,
) -> None:
    if len(contexts) != 96:
        raise GPUOutageExperimentError("durable development context count is not 96")
    seen: set[tuple[str, int, int]] = set()
    for row in contexts:
        key = _context_key(
            str(row["candidate_id"]),
            int(row["held_out_segment"]),
            int(row["forced_outage_index"]),
        )
        if key in seen:
            raise GPUOutageExperimentError(f"duplicate durable context {key}")
        seen.add(key)
        frame = _read_exact_parquet_projection(
            _verify_reference(run_dir, row["prediction_file"]),
            ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS, *_PRED_COLUMNS],
        )
        if development_data is not None:
            _validate_development_prediction_rows(
                frame, development_data, int(row["held_out_segment"])
            )
        metrics = regression_metrics(
            frame.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64),
            frame.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64),
        )
        _assert_json_close(row["metrics"], metrics, path=f"context.{key}.metrics")
        seed_scores = np.asarray(row["seed_rmse_3d_mm"], dtype=np.float64)
        if seed_scores.shape != (3,) or not np.isfinite(seed_scores).all():
            raise GPUOutageExperimentError(f"context {key} seed dispersion is invalid")
        if not math.isclose(
            float(row["seed_rmse_population_std_mm"]),
            float(np.std(seed_scores, ddof=0)),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise GPUOutageExperimentError(f"context {key} seed dispersion changed")
    if seed_results is None:
        return
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for result in seed_results:
        task = result["task"]
        grouped.setdefault(
            _context_key(
                str(task["candidate_id"]),
                int(task["held_out_segment"]),
                int(task["forced_outage_index"]),
            ),
            [],
        ).append(result)
    context_map = {
        _context_key(
            str(row["candidate_id"]),
            int(row["held_out_segment"]),
            int(row["forced_outage_index"]),
        ): row
        for row in contexts
    }
    for candidate_id in TRAINABLE_OUTAGE_CANDIDATE_IDS:
        for segment in (36, 37, 38, 39):
            for outage in range(4):
                key = _context_key(candidate_id, segment, outage)
                seeds = sorted(
                    grouped[key],
                    key=lambda row: int(row["task"]["paired_seed_index"]),
                )
                frames = [_read_prediction_reference(run_dir, row) for row in seeds]
                expected = np.mean(
                    np.stack(
                        [
                            frame.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64)
                            for frame in frames
                        ],
                        axis=0,
                    ),
                    axis=0,
                )
                truth = frames[0].loc[:, list(_TRUE_COLUMNS)].to_numpy(
                    dtype=np.float64
                )
                expected_seed_scores = [
                    float(
                        regression_metrics(
                            truth,
                            frame.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64),
                        )["rmse_3d_mm"]
                    )
                    for frame in frames
                ]
                aggregate = _read_exact_parquet_projection(
                    _verify_reference(run_dir, context_map[key]["prediction_file"]),
                    [
                        "trial_uid",
                        "segment_id",
                        "target_id",
                        *_TRUE_COLUMNS,
                        *_PRED_COLUMNS,
                    ],
                )
                if frames[0][["trial_uid", *_TRUE_COLUMNS]].equals(
                    aggregate[["trial_uid", *_TRUE_COLUMNS]]
                ) is False or not np.allclose(
                    aggregate.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64),
                    expected,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise GPUOutageExperimentError(
                        f"context aggregate is not the three-seed mean: {key}"
                    )
                if not np.allclose(
                    np.asarray(context_map[key]["seed_rmse_3d_mm"], dtype=np.float64),
                    np.asarray(expected_seed_scores, dtype=np.float64),
                    rtol=0.0,
                    atol=1e-9,
                ):
                    raise GPUOutageExperimentError(
                        f"context seed RMSE values do not match predictions: {key}"
                    )
    blend_id = "equal_blend_tabm_chain_reconstruction"
    for segment in (36, 37, 38, 39):
        for outage in range(4):
            paired: list[np.ndarray] = []
            reference_frame: pd.DataFrame | None = None
            for seed_index in range(3):
                values = []
                for component in ("tabm_batchensemble", "chain_masked_reconstruction"):
                    matches = [
                        row
                        for row in grouped[_context_key(component, segment, outage)]
                        if int(row["task"]["paired_seed_index"]) == seed_index
                    ]
                    if len(matches) != 1:
                        raise GPUOutageExperimentError("blend seed pairing evidence changed")
                    frame = _read_prediction_reference(run_dir, matches[0])
                    if reference_frame is None:
                        reference_frame = frame
                    values.append(
                        frame.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64)
                    )
                paired.append(0.5 * values[0] + 0.5 * values[1])
            expected = np.mean(np.stack(paired, axis=0), axis=0)
            row = context_map[_context_key(blend_id, segment, outage)]
            aggregate = _read_exact_parquet_projection(
                _verify_reference(run_dir, row["prediction_file"]),
                [
                    "trial_uid",
                    "segment_id",
                    "target_id",
                    *_TRUE_COLUMNS,
                    *_PRED_COLUMNS,
                ],
            )
            assert reference_frame is not None
            blend_truth = reference_frame.loc[:, list(_TRUE_COLUMNS)].to_numpy(
                dtype=np.float64
            )
            expected_seed_scores = [
                float(regression_metrics(blend_truth, value)["rmse_3d_mm"])
                for value in paired
            ]
            if not reference_frame[["trial_uid", *_TRUE_COLUMNS]].equals(
                aggregate[["trial_uid", *_TRUE_COLUMNS]]
            ) or not np.allclose(
                aggregate.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64),
                expected,
                rtol=0.0,
                atol=1e-12,
            ):
                raise GPUOutageExperimentError(
                    "virtual blend is not paired 0.5/0.5 before seed aggregation"
                )
            if not np.allclose(
                np.asarray(row["seed_rmse_3d_mm"], dtype=np.float64),
                np.asarray(expected_seed_scores, dtype=np.float64),
                rtol=0.0,
                atol=1e-9,
            ):
                raise GPUOutageExperimentError(
                    "virtual blend seed dispersion does not use paired seed predictions"
                )


def _validate_selection_evidence(
    run_dir: str | Path,
    config: Mapping[str, Any],
    context_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_sha = outage_gpu_config_sha256(dict(config))
    if (
        set(context_payload)
        != {
            "config_sha256",
            "candidate_count",
            "context_count_per_candidate",
            "contexts",
        }
        or context_payload.get("config_sha256") != config_sha
        or int(context_payload.get("candidate_count", -1)) != 6
        or int(context_payload.get("context_count_per_candidate", -1)) != 16
        or not isinstance(context_payload.get("contexts"), list)
    ):
        raise GPUOutageExperimentError("development context payload identity changed")
    contexts = [dict(value) for value in context_payload["contexts"]]
    expected_selection = score_outage_candidates(contexts, config)
    expected_selection["candidate"] = _candidate_map(config)[
        expected_selection["selected_candidate_id"]
    ]
    expected_selection["context_metrics_file"] = _artifact_reference(
        Path(run_dir) / "development" / "context_metrics.json", run_dir
    )
    selection = read_json_object(Path(run_dir) / "development" / "selection.json")
    if selection != expected_selection:
        raise GPUOutageExperimentError(
            "durable selection is not the exact frozen scoring result"
        )
    return contexts, selection


def _validate_checkpoint_candidate(
    checkpoint_path: str | Path, candidate: Mapping[str, Any]
) -> None:
    """Bind a pickle-free checkpoint tensor schema to its declared backbone."""

    import torch

    from .gpu_outage_training import _build_candidate

    state = load_outage_checkpoint(checkpoint_path)
    model = _build_candidate(torch, candidate)
    try:
        model.load_state_dict(
            {name: torch.as_tensor(value) for name, value in state.items()}, strict=True
        )
    except RuntimeError as exc:
        raise GPUOutageExperimentError(
            f"checkpoint is incompatible with {candidate['candidate_id']}"
        ) from exc
    del model


def _preprocessor_from_metadata(metadata: Mapping[str, Any]) -> TrainOnlyPreprocessor:
    preprocessor = TrainOnlyPreprocessor(
        feature_names=tuple(str(value) for value in metadata["feature_names"]),
        scale_features=True,
        split_context=str(metadata["leakage_audit"]["split_context"]),
    )
    preprocessor.feature_mean_ = np.asarray(metadata["feature_mean"], dtype=np.float64)
    preprocessor.feature_scale_ = np.asarray(metadata["feature_scale"], dtype=np.float64)
    preprocessor.target_mean_ = np.asarray(metadata["target_mean"], dtype=np.float64)
    preprocessor.target_scale_ = float(metadata["target_scale"])
    preprocessor.leakage_audit_ = dict(metadata["leakage_audit"])
    preprocessor._check_fitted()
    return preprocessor


def _development_split_context(task: Mapping[str, Any]) -> str:
    return (
        f"outage_v3_segment{int(task['held_out_segment'])}_"
        f"drop{int(task['forced_outage_index'])}_"
        f"seed{int(task['paired_seed_index'])}"
    )


def _refit_and_validate_development_preprocessor(
    data: DevelopmentData,
    task: Mapping[str, Any],
    stored_metadata: Mapping[str, Any],
) -> TrainOnlyPreprocessor:
    eval_rows = np.asarray(
        data.segment_ids == int(task["held_out_segment"]), dtype=bool
    )
    expected = _fit_preprocessor(
        data,
        ~eval_rows,
        forbidden_rows=eval_rows,
        split_context=_development_split_context(task),
    )
    expected_metadata = expected.metadata()
    if dict(stored_metadata) != expected_metadata or stable_hash(
        dict(stored_metadata)
    ) != stable_hash(expected_metadata):
        raise GPUOutageExperimentError(
            f"development preprocessor does not reproduce durable fold: {task['task_id']}"
        )
    return expected


def _refit_and_validate_final_preprocessor(
    data: DevelopmentData, stored_metadata: Mapping[str, Any]
) -> TrainOnlyPreprocessor:
    all_rows = np.ones(data.rows, dtype=bool)
    expected = _fit_preprocessor(
        data,
        all_rows,
        forbidden_rows=np.zeros(data.rows, dtype=bool),
        split_context="outage_v3_final_all_development",
    )
    expected_metadata = expected.metadata()
    if dict(stored_metadata) != expected_metadata or stable_hash(
        dict(stored_metadata)
    ) != stable_hash(expected_metadata):
        raise GPUOutageExperimentError(
            "final preprocessor does not reproduce all durable development rows"
        )
    return expected


def _validate_development_prediction_rows(
    stored: pd.DataFrame,
    data: DevelopmentData,
    held_out_segment: int,
) -> np.ndarray:
    eval_rows = np.asarray(data.segment_ids == int(held_out_segment), dtype=bool)
    if len(stored) != int(np.count_nonzero(eval_rows)):
        raise GPUOutageExperimentError("development prediction row count changed")
    if (
        stored["trial_uid"].astype(str).tolist()
        != data.trial_uids[eval_rows].astype(str).tolist()
        or not np.array_equal(
            stored["segment_id"].to_numpy(dtype=np.int64),
            data.segment_ids[eval_rows],
        )
        or stored["target_id"].astype(str).tolist()
        != data.target_ids[eval_rows].astype(str).tolist()
        or not np.array_equal(
            stored.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64),
            data.y[eval_rows],
        )
    ):
        raise GPUOutageExperimentError(
            "development prediction identity/truth differs from durable replay inputs"
        )
    return eval_rows


def _validate_development_best_rmse_binding(
    *,
    result: Mapping[str, Any],
    durable_truth: np.ndarray,
    stored_prediction: np.ndarray,
    replay_prediction: np.ndarray,
) -> None:
    stored_rmse = float(
        regression_metrics(durable_truth, stored_prediction)["rmse_3d_mm"]
    )
    replay_rmse = float(
        regression_metrics(durable_truth, replay_prediction)["rmse_3d_mm"]
    )
    summary_rmse = float(result["summary"]["best_validation_rmse_3d_mm"])
    if not math.isclose(stored_rmse, summary_rmse, rel_tol=0.0, abs_tol=1e-9):
        raise GPUOutageExperimentError(
            "stored development prediction RMSE differs from selected history best"
        )
    if not math.isclose(replay_rmse, summary_rmse, rel_tol=1e-6, abs_tol=1e-4):
        raise GPUOutageExperimentError(
            "replayed development checkpoint RMSE differs from selected history best"
        )


def _validate_mask_schedule_history(
    history: pd.DataFrame,
    trial_uids: Sequence[str],
    *,
    base_seed: int,
    held_out_segment: Any,
    forced_outage: int,
    paired_seed_index: int,
    cache: dict[tuple[Any, ...], list[dict[str, Any]]] | None = None,
) -> None:
    """Recompute each persisted epoch mask hash/count from durable train UIDs."""

    from .gpu_outage_training import (
        MASK_FULL_STATE,
        _mask_states_from_uid_keys,
        _uid_mask_keys,
    )

    uids = tuple(str(value) for value in trial_uids)
    epochs = tuple(int(value) for value in history["epoch"].tolist())
    if epochs != tuple(range(1, len(history) + 1)):
        raise GPUOutageExperimentError("mask schedule history epoch order changed")
    key = (
        hashlib.sha256("\n".join(uids).encode("utf-8")).hexdigest(),
        int(base_seed),
        str(held_out_segment),
        int(forced_outage),
        int(paired_seed_index),
        len(history),
    )
    expected_rows = None if cache is None else cache.get(key)
    if expected_rows is None:
        row_keys = _uid_mask_keys(
            uids,
            base_seed=int(base_seed),
            held_out_segment=held_out_segment,
            forced_outage=int(forced_outage),
            paired_seed_index=int(paired_seed_index),
        )
        expected_rows = []
        for epoch in epochs:
            states = _mask_states_from_uid_keys(row_keys, epoch=epoch)
            counts = {
                state: int(np.count_nonzero(states == state))
                for state in (MASK_FULL_STATE, 0, 1, 2, 3)
            }
            expected_rows.append(
                {
                    "mask_schedule_sha256": hashlib.sha256(
                        states.tobytes()
                    ).hexdigest(),
                    "mask_full_row_count": counts[MASK_FULL_STATE],
                    "mask_drop_imu0_row_count": counts[0],
                    "mask_drop_imu1_row_count": counts[1],
                    "mask_drop_imu2_row_count": counts[2],
                    "mask_drop_imu3_row_count": counts[3],
                }
            )
        if cache is not None:
            cache[key] = expected_rows
    fields = tuple(expected_rows[0]) if expected_rows else ()
    for index, expected in enumerate(expected_rows):
        actual = history.iloc[index]
        if any(
            (
                str(actual[field]) != str(value)
                if field == "mask_schedule_sha256"
                else int(actual[field]) != int(value)
            )
            for field, value in expected.items()
        ) or sum(int(actual[field]) for field in fields if field.endswith("row_count")) != len(
            uids
        ):
            raise GPUOutageExperimentError(
                f"mask schedule evidence changed at epoch {index + 1}"
            )


def _validate_development_uid_scopes(
    run_dir: str | Path, results: Sequence[Mapping[str, Any]], expected_rows: int
) -> None:
    reference_by_segment: dict[int, tuple[str, ...]] = {}
    for segment in (36, 37, 38, 39):
        matches = [
            row
            for row in results
            if row["task"]["candidate_id"] == "masked_mlp"
            and int(row["task"]["held_out_segment"]) == segment
            and int(row["task"]["forced_outage_index"]) == 0
            and int(row["task"]["paired_seed_index"]) == 0
        ]
        if len(matches) != 1:
            raise GPUOutageExperimentError("cannot establish development UID universe")
        frame = _read_prediction_reference(run_dir, matches[0])
        reference_by_segment[segment] = tuple(frame["trial_uid"].astype(str).tolist())
    all_uids = set().union(*(set(values) for values in reference_by_segment.values()))
    if len(all_uids) != expected_rows or sum(map(len, reference_by_segment.values())) != expected_rows:
        raise GPUOutageExperimentError("development validation UID roles overlap or drifted")
    for result in results:
        segment = int(result["task"]["held_out_segment"])
        frame = _read_prediction_reference(run_dir, result)
        eval_uids = tuple(frame["trial_uid"].astype(str).tolist())
        if eval_uids != reference_by_segment[segment]:
            raise GPUOutageExperimentError("paired development tasks use different held-out rows")
        expected_train = sorted(all_uids.difference(eval_uids))
        expected_hash = hashlib.sha256("\n".join(expected_train).encode("utf-8")).hexdigest()
        preprocessing = read_json_object(
            _verify_reference(run_dir, result["artifacts"]["preprocessing"])
        )
        audit = preprocessing["leakage_audit"]
        if (
            int(audit.get("fit_row_count", -1)) != len(expected_train)
            or int(audit.get("forbidden_uid_count", -1)) != len(eval_uids)
            or audit.get("fit_trial_uid_sha256") != expected_hash
        ):
            raise GPUOutageExperimentError("development preprocessor UID scope changed")


def _validate_development_manifest_binding(
    run_dir: str | Path,
    config: Mapping[str, Any],
    *,
    allow_final_artifacts: bool,
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    config_sha = outage_gpu_config_sha256(dict(config))
    manifest = read_json_object(run_path / "development_manifest.json")
    selection = read_json_object(run_path / "development" / "selection.json")
    expected_selection = {
        "selected_candidate_id": selection["selected_candidate_id"],
        "selected_outage_robust_j_mm": selection["selected_outage_robust_j_mm"],
        "challenge_used_for_selection": False,
    }
    if (
        manifest.get("contract") != OUTAGE_RUN_MANIFEST_CONTRACT
        or manifest.get("experiment_contract") != OUTAGE_GPU_EXPERIMENT_CONTRACT
        or manifest.get("stage")
        != "development_complete_challenge_labels_sealed"
        or manifest.get("config_sha256") != config_sha
        or manifest.get("selection") != expected_selection
        or manifest.get("challenge_labels_sealed") is not True
        or int(manifest.get("counts", {}).get("development_seed_fits", -1)) != 240
        or int(manifest.get("counts", {}).get("scored_contexts", -1)) != 96
        or int(
            manifest.get("counts", {}).get("challenge_label_access_passes", -1)
        )
        != 0
    ):
        raise GPUOutageExperimentError("development manifest identity/status changed")
    if not allow_final_artifacts and (
        run_path / "final" / "preprocessing.json"
    ).exists():
        raise GPUOutageExperimentError(
            "development-only run already contains final refit state"
        )
    if _development_manifest_files(run_path) != manifest.get("files"):
        raise GPUOutageExperimentError("development-stage artifact manifest changed")
    return manifest


def validate_outage_prelabel_run(
    run_dir: str | Path, *, _allow_opened: bool = False
) -> dict[str, Any]:
    """Validate every decision/model artifact before irreversible label access."""

    _require_python_312()
    run_path = Path(run_dir).resolve()
    if (
        not _allow_opened
        and (run_path / "challenge" / "label_access_1.started.json").exists()
    ):
        raise GPUOutageExperimentError("pre-label validation must precede label access")
    config = _read_resolved_outage_config(run_path / "config.resolved.json")
    config_sha = outage_gpu_config_sha256(config)
    _validate_source_provenance(run_path, config)
    identity = _load_durable_challenge_identity(run_path)
    _validate_challenge_identity_contract(run_path, identity, config)
    development_data = _load_durable_development_inputs(run_path)
    replay_metadata = read_json_object(run_path / "development" / "replay_inputs.json")
    _validate_development_identity_contract(
        development_data, replay_metadata, config, run_dir=run_path
    )
    _validate_development_manifest_binding(
        run_path, config, allow_final_artifacts=True
    )
    if (
        replay_metadata.get("config_sha256") != config_sha
    ):
        raise GPUOutageExperimentError("pre-label development replay identity changed")

    from .gpu_outage_training import predict_outage_state
    import torch

    replay_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _validate_worker_provenance(
        run_path, phase="development", expected_workers=int(config["runtime"]["gpu_count"])
    )

    tasks = outage_development_tasks(config)
    development_results: list[dict[str, Any]] = []
    mask_schedule_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for task in tasks:
        result = _load_valid_task_result(run_path, task)
        if result is None:
            raise GPUOutageExperimentError(
                f"pre-label development task invalid: {task['task_id']}"
            )
        history = _validate_development_seed_result(run_path, config, result)
        _validate_checkpoint_candidate(
            _verify_reference(run_path, result["artifacts"]["checkpoint"]),
            task["candidate"],
        )
        preprocessing_metadata = read_json_object(
            _verify_reference(run_path, result["artifacts"]["preprocessing"])
        )
        preprocessor = _refit_and_validate_development_preprocessor(
            development_data, task, preprocessing_metadata
        )
        stored_prediction = _read_prediction_reference(run_path, result)
        eval_rows = _validate_development_prediction_rows(
            stored_prediction, development_data, int(task["held_out_segment"])
        )
        _validate_mask_schedule_history(
            history,
            development_data.trial_uids[~eval_rows],
            base_seed=int(config["contract"]["base_seed"]),
            held_out_segment=task["held_out_segment"],
            forced_outage=int(task["forced_outage_index"]),
            paired_seed_index=int(task["paired_seed_index"]),
            cache=mask_schedule_cache,
        )
        X_scaled = preprocessor.transform_features(development_data.X[eval_rows]).astype(
            np.float32
        )
        outage_index = int(task["forced_outage_index"])
        X_scaled[:, np.asarray(IMU_SENSOR_INDEX_MAP[outage_index], dtype=np.int64)] = 0.0
        availability = np.ones((int(np.count_nonzero(eval_rows)), 4), dtype=np.float32)
        availability[:, outage_index] = 0.0
        checkpoint_path = _verify_reference(
            run_path, result["artifacts"]["checkpoint"]
        )
        state = {
            name: torch.as_tensor(value)
            for name, value in load_outage_checkpoint(checkpoint_path).items()
        }
        replay_prediction = predict_outage_state(
            state_dict=state,
            X_scaled=X_scaled,
            availability=availability,
            preprocessor=preprocessor,
            candidate=task["candidate"],
            device=replay_device,
        )
        if not np.allclose(
            stored_prediction.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64),
            replay_prediction,
            rtol=1e-5,
            atol=1e-4,
        ):
            raise GPUOutageExperimentError(
                f"development checkpoint does not reproduce prediction: {task['task_id']}"
            )
        _validate_development_best_rmse_binding(
            result=result,
            durable_truth=development_data.y[eval_rows],
            stored_prediction=stored_prediction.loc[
                :, list(_PRED_COLUMNS)
            ].to_numpy(dtype=np.float64),
            replay_prediction=np.asarray(replay_prediction, dtype=np.float64),
        )
        development_results.append(result)
    _validate_development_uid_scopes(
        run_path,
        development_results,
        int(config["row_roles"]["development"]["expected_rows"]),
    )

    context_payload = read_json_object(run_path / "development" / "context_metrics.json")
    contexts, selection = _validate_selection_evidence(
        run_path, config, context_payload
    )
    _validate_context_evidence(
        run_path, contexts, development_results, development_data
    )

    preprocessing_metadata = read_json_object(run_path / "final" / "preprocessing.json")
    preprocessor = _refit_and_validate_final_preprocessor(
        development_data, preprocessing_metadata
    )
    challenge_scaled = _standardize_challenge_features(identity, preprocessor)
    final_tasks = outage_final_tasks(config, selection, development_results)
    _validate_worker_provenance(
        run_path, phase="final", expected_workers=int(config["runtime"]["gpu_count"])
    )
    final_results: list[dict[str, Any]] = []
    recomputed_seed_predictions: dict[tuple[str, int], np.ndarray] = {}
    for task in final_tasks:
        result = _load_valid_final_result(run_path, task)
        if result is None:
            raise GPUOutageExperimentError(f"pre-label final task invalid: {task['task_id']}")
        history = _validate_final_seed_result(run_path, config, result)
        _validate_mask_schedule_history(
            history,
            development_data.trial_uids,
            base_seed=int(config["contract"]["base_seed"]),
            held_out_segment=task["mask_held_out_segment"],
            forced_outage=int(task["mask_forced_outage_index"]),
            paired_seed_index=int(task["paired_seed_index"]),
            cache=mask_schedule_cache,
        )
        checkpoint_path = _verify_reference(
            run_path, result["artifacts"]["checkpoint"]
        )
        _validate_checkpoint_candidate(checkpoint_path, task["candidate"])
        state = {
            name: torch.as_tensor(value)
            for name, value in load_outage_checkpoint(checkpoint_path).items()
        }
        prediction = predict_outage_state(
            state_dict=state,
            X_scaled=challenge_scaled,
            availability=identity.availability,
            preprocessor=preprocessor,
            candidate=task["candidate"],
            device=replay_device,
        )
        stored = _read_exact_parquet_projection(
            _verify_reference(run_path, result["artifacts"]["challenge_predictions"]),
            ["trial_uid", "segment_id", "target_id", *_PRED_COLUMNS],
        )
        expected_identity = _identity_frame(identity)[
            ["trial_uid", "segment_id", "target_id"]
        ]
        if not stored[["trial_uid", "segment_id", "target_id"]].equals(
            expected_identity
        ):
            raise GPUOutageExperimentError("final seed prediction identity changed")
        stored_values = stored.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.allclose(stored_values, prediction, rtol=1e-5, atol=1e-4):
            raise GPUOutageExperimentError(
                f"final checkpoint does not reproduce frozen seed prediction: {task['task_id']}"
            )
        recomputed_seed_predictions[
            (str(task["candidate_id"]), int(task["paired_seed_index"]))
        ] = np.asarray(prediction, dtype=np.float64)
        final_results.append(result)

    freeze_path = run_path / "challenge" / "freeze.before_label_access.json"
    freeze = read_json_object(freeze_path)
    if freeze.get("config_sha256") != config_sha or freeze.get("status") != "frozen_before_label_access":
        raise GPUOutageExperimentError("pre-label freeze identity/status changed")
    if freeze.get("selected_candidate_id") != selection["selected_candidate_id"]:
        raise GPUOutageExperimentError("pre-label freeze selection changed")
    _require_canonical_reference(
        run_path,
        freeze["selection_file"],
        run_path / "development" / "selection.json",
    )
    _require_canonical_reference(
        run_path,
        freeze["preprocessing_file"],
        run_path / "final" / "preprocessing.json",
    )
    _require_canonical_reference(
        run_path,
        freeze["identity_file"],
        run_path / "challenge" / "identity_inputs.parquet",
    )
    _require_canonical_reference(
        run_path,
        freeze["prediction_file"],
        run_path / "challenge" / "predictions.label_blind.parquet",
        rows=identity.rows,
    )
    for reference in freeze["checkpoint_files"]:
        _verify_reference(run_path, reference)
    expected_checkpoint_refs = [
        row["artifacts"]["checkpoint"]
        for row in sorted(final_results, key=lambda value: str(value["task"]["task_id"]))
    ]
    if freeze["checkpoint_files"] != expected_checkpoint_refs or freeze[
        "model_bundle_sha256"
    ] != stable_hash(expected_checkpoint_refs):
        raise GPUOutageExperimentError("pre-label model bundle binding changed")
    components = _selected_components(config, str(selection["selected_candidate_id"]))
    expected_seeds: list[np.ndarray] = []
    for seed_index in range(3):
        if len(components) == 1:
            value = recomputed_seed_predictions[(components[0], seed_index)]
        else:
            value = 0.5 * recomputed_seed_predictions[(components[0], seed_index)] + 0.5 * recomputed_seed_predictions[(components[1], seed_index)]
        expected_seeds.append(value)
    expected_ensemble = np.mean(np.stack(expected_seeds, axis=0), axis=0)
    frozen = _read_exact_parquet_projection(
        _verify_reference(run_path, freeze["prediction_file"]),
        [
            "trial_uid",
            "segment_id",
            "target_id",
            *[
                f"seed{seed}_pred_{axis}_mm"
                for seed in range(3)
                for axis in ("x", "y", "z")
            ],
            *_PRED_COLUMNS,
        ],
    )
    if not frozen[["trial_uid", "segment_id", "target_id"]].equals(
        _identity_frame(identity)[["trial_uid", "segment_id", "target_id"]]
    ):
        raise GPUOutageExperimentError("pre-label frozen prediction identity changed")
    for seed_index, expected in enumerate(expected_seeds):
        columns = [f"seed{seed_index}_pred_{axis}_mm" for axis in ("x", "y", "z")]
        if not np.allclose(
            frozen.loc[:, columns].to_numpy(dtype=np.float64),
            expected,
            rtol=1e-5,
            atol=1e-4,
        ):
            raise GPUOutageExperimentError("pre-label paired seed prediction changed")
    if not np.allclose(
        frozen.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64),
        expected_ensemble,
        rtol=1e-5,
        atol=1e-4,
    ):
        raise GPUOutageExperimentError("pre-label selected ensemble prediction changed")
    return {
        "status": "ok",
        "config_sha256": config_sha,
        "development_seed_fits": 240,
        "scored_contexts": 96,
        "final_seed_fits": len(final_results),
        "selected_candidate_id": str(selection["selected_candidate_id"]),
        "challenge_rows": identity.rows,
        "challenge_labels_accessed": False,
    }


def _prelabel_gate_then_open(
    *,
    run_dir: str | Path,
    validate_fn: Callable[[str | Path], Mapping[str, Any]] = validate_outage_prelabel_run,
    open_fn: Callable[..., dict[str, Any]] = _open_challenge_once,
    **open_kwargs: Any,
) -> dict[str, Any]:
    """Small injectable gate: validation must return before any label loader path."""

    validation = validate_fn(run_dir)
    if validation.get("status") != "ok" or validation.get("challenge_labels_accessed") is not False:
        raise GPUOutageExperimentError("pre-label semantic validation did not pass closed")
    atomic_write_json(
        Path(run_dir) / "challenge" / "prelabel_semantic_validation.json",
        dict(validation),
    )
    return open_fn(run_dir=run_dir, **open_kwargs)


def validate_outage_development_run(
    run_dir: str | Path, *, _allow_final_artifacts: bool = False
) -> dict[str, Any]:
    """Validate the 240-fit/96-context stage while challenge labels stay sealed."""

    _require_python_312()
    run_path = Path(run_dir).resolve()
    if (run_path / "challenge" / "label_access_1.started.json").exists():
        raise GPUOutageExperimentError(
            "development-only validation refuses a run whose label seal was opened"
        )
    config = _read_resolved_outage_config(run_path / "config.resolved.json")
    config_sha = outage_gpu_config_sha256(config)
    _validate_source_provenance(run_path, config)
    identity = _load_durable_challenge_identity(run_path)
    _validate_challenge_identity_contract(run_path, identity, config)
    development_data = _load_durable_development_inputs(run_path)
    replay_metadata = read_json_object(run_path / "development" / "replay_inputs.json")
    _validate_development_identity_contract(
        development_data, replay_metadata, config, run_dir=run_path
    )
    if replay_metadata.get("config_sha256") != config_sha:
        raise GPUOutageExperimentError("development-stage durable input identity changed")

    from .gpu_outage_training import predict_outage_state
    import torch

    replay_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _validate_worker_provenance(
        run_path, phase="development", expected_workers=int(config["runtime"]["gpu_count"])
    )
    results: list[dict[str, Any]] = []
    mask_schedule_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for task in outage_development_tasks(config):
        result = _load_valid_task_result(run_path, task)
        if result is None:
            raise GPUOutageExperimentError(
                f"development-stage task invalid: {task['task_id']}"
            )
        history = _validate_development_seed_result(run_path, config, result)
        checkpoint_path = _verify_reference(
            run_path, result["artifacts"]["checkpoint"]
        )
        _validate_checkpoint_candidate(checkpoint_path, task["candidate"])
        preprocessing_metadata = read_json_object(
            _verify_reference(run_path, result["artifacts"]["preprocessing"])
        )
        preprocessor = _refit_and_validate_development_preprocessor(
            development_data, task, preprocessing_metadata
        )
        stored = _read_prediction_reference(run_path, result)
        eval_rows = _validate_development_prediction_rows(
            stored, development_data, int(task["held_out_segment"])
        )
        _validate_mask_schedule_history(
            history,
            development_data.trial_uids[~eval_rows],
            base_seed=int(config["contract"]["base_seed"]),
            held_out_segment=task["held_out_segment"],
            forced_outage=int(task["forced_outage_index"]),
            paired_seed_index=int(task["paired_seed_index"]),
            cache=mask_schedule_cache,
        )
        X_scaled = preprocessor.transform_features(development_data.X[eval_rows]).astype(
            np.float32
        )
        outage_index = int(task["forced_outage_index"])
        X_scaled[:, np.asarray(IMU_SENSOR_INDEX_MAP[outage_index], dtype=np.int64)] = 0.0
        availability = np.ones((int(np.count_nonzero(eval_rows)), 4), dtype=np.float32)
        availability[:, outage_index] = 0.0
        state = {
            name: torch.as_tensor(value)
            for name, value in load_outage_checkpoint(checkpoint_path).items()
        }
        replay = predict_outage_state(
            state_dict=state,
            X_scaled=X_scaled,
            availability=availability,
            preprocessor=preprocessor,
            candidate=task["candidate"],
            device=replay_device,
        )
        if not np.allclose(
            stored.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64),
            replay,
            rtol=1e-5,
            atol=1e-4,
        ):
            raise GPUOutageExperimentError(
                f"development checkpoint/prediction binding failed: {task['task_id']}"
            )
        _validate_development_best_rmse_binding(
            result=result,
            durable_truth=development_data.y[eval_rows],
            stored_prediction=stored.loc[:, list(_PRED_COLUMNS)].to_numpy(
                dtype=np.float64
            ),
            replay_prediction=np.asarray(replay, dtype=np.float64),
        )
        results.append(result)
    _validate_development_uid_scopes(
        run_path,
        results,
        int(config["row_roles"]["development"]["expected_rows"]),
    )
    context_payload = read_json_object(run_path / "development" / "context_metrics.json")
    contexts, selection = _validate_selection_evidence(
        run_path, config, context_payload
    )
    _validate_context_evidence(run_path, contexts, results, development_data)

    _validate_development_manifest_binding(
        run_path, config, allow_final_artifacts=_allow_final_artifacts
    )
    return {
        "status": "ok",
        "contract": OUTAGE_RUN_MANIFEST_CONTRACT,
        "stage": "development_complete_challenge_labels_sealed",
        "config_sha256": config_sha,
        "development_seed_fits": 240,
        "scored_contexts": 96,
        "selected_candidate_id": str(selection["selected_candidate_id"]),
        "selected_outage_robust_j_mm": float(selection["selected_outage_robust_j_mm"]),
        "challenge_rows_identity_only": identity.rows,
        "challenge_label_access_passes": 0,
        "challenge_labels_sealed": True,
    }


def validate_outage_gpu_run(run_dir: str | Path) -> dict[str, Any]:
    """Validate durable outage-v3 evidence without reopening either source ZIP."""

    _require_python_312()
    run_path = Path(run_dir).resolve()
    config = _read_resolved_outage_config(run_path / "config.resolved.json")
    config_sha = outage_gpu_config_sha256(config)
    manifest = read_json_object(run_path / "manifest.json")
    if manifest.get("contract") != OUTAGE_RUN_MANIFEST_CONTRACT:
        raise GPUOutageExperimentError("outage-v3 run manifest contract changed")
    if manifest.get("experiment_contract") != OUTAGE_GPU_EXPERIMENT_CONTRACT:
        raise GPUOutageExperimentError("outage-v3 experiment contract changed")
    if manifest.get("config_sha256") != config_sha or manifest.get("status") != "complete":
        raise GPUOutageExperimentError("outage-v3 manifest identity/status changed")
    verify_file_manifest(run_path, manifest["files"])
    replayed_prelabel_validation = validate_outage_prelabel_run(
        run_path, _allow_opened=True
    )

    snapshot = read_json_object(run_path / "provenance" / "source_snapshot.json")
    if stable_hash(snapshot.get("files")) != snapshot.get("fingerprint"):
        raise GPUOutageExperimentError("durable source snapshot fingerprint changed")
    identity_metadata = read_json_object(run_path / "challenge" / "identity.json")
    if identity_metadata.get("labels_projected") is not False or identity_metadata.get(
        "recorded_imu1_projected"
    ) is not False:
        raise GPUOutageExperimentError("challenge identity seal claims were weakened")
    identity = _load_durable_challenge_identity(run_path)
    expected_challenge = config["row_roles"]["challenge"]
    if identity.rows != int(expected_challenge["expected_rows"]):
        raise GPUOutageExperimentError("challenge durable row count changed")
    if identity.ordered_trial_uid_sha256 != str(
        expected_challenge["ordered_trial_uid_sha256"]
    ):
        raise GPUOutageExperimentError("challenge durable UID seal changed")
    if not np.array_equal(
        identity.availability,
        np.tile(np.asarray((1.0, 0.0, 1.0, 1.0)), (identity.rows, 1)),
    ):
        raise GPUOutageExperimentError("challenge availability is not forced IMU1 outage")
    imu1_indices = np.asarray(IMU_SENSOR_INDEX_MAP[1], dtype=np.int64)
    if not np.isnan(identity.X[:, imu1_indices]).all():
        raise GPUOutageExperimentError("durable challenge identity contains recorded IMU1")

    development_tasks = outage_development_tasks(config)
    development_results: list[dict[str, Any]] = []
    for task in development_tasks:
        result = _load_valid_task_result(run_path, task)
        if result is None:
            raise GPUOutageExperimentError(f"invalid development task {task['task_id']}")
        _validate_development_seed_result(run_path, config, result)
        development_results.append(result)
    if len(development_results) != 240:
        raise GPUOutageExperimentError("durable development seeded-fit count changed")

    context_payload = read_json_object(run_path / "development" / "context_metrics.json")
    contexts, selection = _validate_selection_evidence(
        run_path, config, context_payload
    )
    _validate_context_evidence(run_path, contexts)

    preprocessing = read_json_object(run_path / "final" / "preprocessing.json")
    audit = preprocessing.get("leakage_audit", {})
    if (
        audit.get("fit_scope") != "real_train_only"
        or int(audit.get("fit_row_count", -1)) != int(config["row_roles"]["development"]["expected_rows"])
        or int(audit.get("forbidden_uid_count", -1)) != 0
        or int(audit.get("forbidden_uid_overlap_count", -1)) != 0
    ):
        raise GPUOutageExperimentError("final preprocessing scope is not all development only")
    final_tasks = outage_final_tasks(config, selection, development_results)
    final_results: list[dict[str, Any]] = []
    for task in final_tasks:
        result = _load_valid_final_result(run_path, task)
        if result is None:
            raise GPUOutageExperimentError(f"invalid final task {task['task_id']}")
        _validate_final_seed_result(run_path, config, result)
        if int(result["summary"]["trained_epochs"]) != int(task["epochs"]):
            raise GPUOutageExperimentError("final task epoch count changed")
        history = pd.read_parquet(_verify_reference(run_path, result["artifacts"]["history"]))
        if len(history) != int(task["epochs"]):
            raise GPUOutageExperimentError("final task history length changed")
        prediction = _read_exact_parquet_projection(
            _verify_reference(run_path, result["artifacts"]["challenge_predictions"]),
            ["trial_uid", "segment_id", "target_id", *_PRED_COLUMNS],
        )
        if any(column in prediction.columns for column in _TRUE_COLUMNS):
            raise GPUOutageExperimentError("final label-blind seed prediction contains labels")
        if ordered_trial_uid_sha256(prediction["trial_uid"].astype(str).tolist()) != identity.ordered_trial_uid_sha256:
            raise GPUOutageExperimentError("final seed prediction identity changed")
        load_outage_checkpoint(_verify_reference(run_path, result["artifacts"]["checkpoint"]))
        final_results.append(result)

    freeze_path = run_path / "challenge" / "freeze.before_label_access.json"
    freeze = read_json_object(freeze_path)
    if freeze.get("status") != "frozen_before_label_access" or freeze.get(
        "config_sha256"
    ) != config_sha:
        raise GPUOutageExperimentError("challenge freeze identity/status changed")
    if freeze.get("selected_candidate_id") != selection["selected_candidate_id"]:
        raise GPUOutageExperimentError("challenge freeze selected candidate changed")
    if freeze.get("challenge_labels_accessed") is not False:
        raise GPUOutageExperimentError("pre-access freeze claims labels were accessed")
    for reference in (
        freeze["selection_file"],
        freeze["prediction_file"],
        freeze["preprocessing_file"],
        freeze["identity_file"],
        *freeze["checkpoint_files"],
    ):
        _verify_reference(run_path, reference)
    if stable_hash(freeze["checkpoint_files"]) != freeze["model_bundle_sha256"]:
        raise GPUOutageExperimentError("frozen model bundle digest changed")
    frozen_prediction = _read_exact_parquet_projection(
        _verify_reference(run_path, freeze["prediction_file"]),
        [
            "trial_uid",
            "segment_id",
            "target_id",
            *[
                f"seed{seed}_pred_{axis}_mm"
                for seed in range(3)
                for axis in ("x", "y", "z")
            ],
            *_PRED_COLUMNS,
        ],
    )
    if any(column in frozen_prediction.columns for column in _TRUE_COLUMNS):
        raise GPUOutageExperimentError("pre-access challenge prediction contains labels")
    if ordered_trial_uid_sha256(frozen_prediction["trial_uid"].astype(str).tolist()) != identity.ordered_trial_uid_sha256:
        raise GPUOutageExperimentError("frozen challenge prediction identity changed")

    started_path = run_path / "challenge" / "label_access_1.started.json"
    completed_path = run_path / "challenge" / "label_access_1.completed.json"
    access_log_path = run_path / "challenge" / "label_access_log.json"
    started = read_json_object(started_path)
    completed = read_json_object(completed_path)
    access_log = read_json_object(access_log_path)
    if (
        int(started.get("access_index", -1)) != 1
        or started.get("status") != "started_fail_closed"
        or int(completed.get("source_label_access_pass_count", -1)) != 1
        or completed.get("status") != "completed"
        or int(access_log.get("source_label_access_pass_count", -1)) != 1
        or int(access_log.get("maximum_source_label_access_passes", -1)) != 1
        or access_log.get("status") != "completed"
    ):
        raise GPUOutageExperimentError("challenge one-pass access evidence changed")
    prelabel_validation = read_json_object(
        _verify_reference(run_path, started["prelabel_semantic_validation_file"])
    )
    if (
        prelabel_validation.get("status") != "ok"
        or prelabel_validation.get("challenge_labels_accessed") is not False
        or int(prelabel_validation.get("development_seed_fits", -1)) != 240
        or int(prelabel_validation.get("scored_contexts", -1)) != 96
        or prelabel_validation.get("selected_candidate_id")
        != selection["selected_candidate_id"]
    ):
        raise GPUOutageExperimentError("durable pre-label semantic validation changed")
    _assert_json_close(
        prelabel_validation,
        replayed_prelabel_validation,
        path="challenge.prelabel_semantic_validation",
    )
    if _verify_reference(run_path, started["freeze_file"]) != freeze_path:
        raise GPUOutageExperimentError("label access did not reference the frozen evidence")
    evaluation = _read_exact_parquet_projection(
        _verify_reference(run_path, completed["evaluation_file"]),
        [
            "trial_uid",
            "segment_id",
            "target_id",
            *_TRUE_COLUMNS,
            *[
                f"seed{seed}_pred_{axis}_mm"
                for seed in range(3)
                for axis in ("x", "y", "z")
            ],
            *_PRED_COLUMNS,
        ],
    )
    if evaluation["trial_uid"].astype(str).tolist() != identity.trial_uids.tolist():
        raise GPUOutageExperimentError("durable challenge evaluation identity changed")
    if len(evaluation) != identity.rows:
        raise GPUOutageExperimentError("durable challenge evaluation row count changed")
    if not np.isfinite(
        evaluation.loc[:, [*_TRUE_COLUMNS, *_PRED_COLUMNS]].to_numpy(dtype=np.float64)
    ).all():
        raise GPUOutageExperimentError("durable challenge evaluation is non-finite")
    recomputed_report = _challenge_report(
        config=config,
        selected_candidate_id=str(selection["selected_candidate_id"]),
        evaluation=evaluation,
    )
    report = read_json_object(_verify_reference(run_path, completed["report_file"]))
    _assert_json_close(report, recomputed_report, path="challenge.report")
    if int(completed.get("candidate_specific_challenge_reports", -1)) != 1 or completed.get(
        "reselection_after_access"
    ) is not False:
        raise GPUOutageExperimentError("challenge selected-only/reselection evidence changed")

    if manifest["selection"]["selected_candidate_id"] != selection["selected_candidate_id"]:
        raise GPUOutageExperimentError("manifest selection differs from durable selection")
    if int(manifest["counts"].get("development_seed_fits", -1)) != 240 or int(
        manifest["counts"].get("challenge_label_access_passes", -1)
    ) != 1:
        raise GPUOutageExperimentError("manifest fit/access counts changed")
    return {
        "status": "ok",
        "contract": OUTAGE_RUN_MANIFEST_CONTRACT,
        "config_sha256": config_sha,
        "development_seed_fits": 240,
        "scored_contexts": 96,
        "selected_candidate_id": str(selection["selected_candidate_id"]),
        "selected_outage_robust_j_mm": float(selection["selected_outage_robust_j_mm"]),
        "challenge_rows": identity.rows,
        "challenge_label_access_passes": 1,
        "raw_source_reopened_by_validator": False,
    }


__all__ = [
    "GPUOutageExperimentError",
    "OUTAGE_CHALLENGE_FREEZE_CONTRACT",
    "OUTAGE_RUN_MANIFEST_CONTRACT",
    "OUTAGE_SEED_RESULT_CONTRACT",
    "aggregate_outage_development",
    "load_outage_checkpoint",
    "outage_development_tasks",
    "outage_final_tasks",
    "outage_training_source_payload",
    "run_outage_gpu_experiment",
    "score_outage_candidates",
    "validate_outage_development_run",
    "validate_outage_gpu_run",
    "validate_outage_prelabel_run",
]
