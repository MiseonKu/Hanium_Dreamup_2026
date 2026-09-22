"""Four-GPU runner for the post-v1 segment-robust development experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

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
from .features import LABEL_COLUMNS
from .gpu_experiment import (
    GPUExperimentError,
    MODEL_NAMES,
    PREDICTION_COLUMNS,
    TRUE_COLUMNS,
    _WorkerData,
    _all_real_arrays,
    _assert_json_close,
    _execute_task,
    _expected_final_model_manifest,
    _final_model_manifest,
    _git_payload,
    _load_worker_data,
    _relative,
    _result_path,
    _round_median,
    _safe_run_member,
    _validate_gpu_ids,
    _validate_task_results,
    _validation_workspace_root,
    _worker_runtime,
    gpu_configuration_id,
    gpu_environment,
)
from .gpu_robust_experiment_config import (
    HISTORICAL_V1_ROBUST_J_MM,
    ROBUST_GPU_EXPERIMENT_CONTRACT,
    public_robust_gpu_config,
    resolve_robust_gpu_release_path,
    robust_candidate_grid,
    robust_gpu_config_sha256,
    validate_robust_gpu_config,
)


ROBUST_MODEL_TRAINING_CONTRACT = "alphaf.endpoint_regression.gpu_training.robust_v2"
ROBUST_GPU_MODEL_CONTRACT = "alphaf.endpoint_regression.gpu_model.robust_v2"
ROBUST_SOURCE_FILES = (
    "configs/endpoint_regression_gpu_robust_v2.yaml",
    "configs/endpoint_regression_gpu_v1.yaml",
    "requirements-gpu.lock",
    "requirements-eval.lock",
    "src/imu_vive_learning/gpu_robust_experiment.py",
    "src/imu_vive_learning/gpu_robust_experiment_config.py",
    "src/imu_vive_learning/gpu_robust_training.py",
    "src/imu_vive_learning/gpu_experiment.py",
    "src/imu_vive_learning/gpu_experiment_config.py",
    "src/imu_vive_learning/gpu_inference.py",
    "src/imu_vive_learning/cli.py",
    "src/imu_vive_learning/experiment_io.py",
    "src/imu_vive_learning/experiment_metrics.py",
    "src/imu_vive_learning/experiment_models.py",
    "src/imu_vive_learning/experiment_preprocessing.py",
    "src/imu_vive_learning/experiment_release.py",
    "src/imu_vive_learning/experiment_splits.py",
    "src/imu_vive_learning/features.py",
    "src/imu_vive_learning/config.py",
    "src/imu_vive_learning/io_utils.py",
    "src/imu_vive_learning/cleaning.py",
    "src/imu_vive_learning/splitting.py",
)


class GPURobustExperimentError(GPUExperimentError):
    """Raised when robust-v2 execution or evidence violates its contract."""


def robust_training_source_payload(workspace_root: str | Path) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    files: list[dict[str, Any]] = []
    for relative in ROBUST_SOURCE_FILES:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise GPURobustExperimentError(
                f"robust training source escapes workspace: {relative}"
            ) from exc
        if not path.is_file():
            raise GPURobustExperimentError(
                f"robust training source is missing: {relative}"
            )
        files.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {"files": files, "fingerprint": stable_hash(files)}


def _verify_robust_source(config: Mapping[str, Any]) -> None:
    expected = config.get("_source_fingerprint")
    if expected is None:
        return
    workspace = config.get("_workspace_root")
    if not isinstance(workspace, str):
        raise GPURobustExperimentError("robust source verification lacks workspace root")
    actual = robust_training_source_payload(workspace)["fingerprint"]
    if actual != expected:
        raise GPURobustExperimentError(
            f"robust training source changed after freeze: {actual} != {expected}"
        )


def robust_candidate_config_id(
    candidate: Mapping[str, Any], model_params: Mapping[str, Any]
) -> str:
    return stable_hash(
        {"model": "torch_mlp", "params": dict(model_params), "candidate": dict(candidate)}
    )[:16]


def _model_params(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["models"]["torch_mlp"]
    return {
        "hidden_layer_sizes": list(model["hidden_layer_sizes"]),
        "weight_decay": float(model["weight_decay"]),
    }


def _robust_task_id(
    phase: str,
    *,
    outer_fold: int,
    inner_fold: int,
    candidate_id: str,
    candidate_config_id: str,
    seed_index: int | None = None,
) -> str:
    parts = [
        phase,
        "segment_loso",
        f"outer{int(outer_fold)}",
        f"inner{int(inner_fold)}",
        candidate_id,
        candidate_config_id,
    ]
    if seed_index is not None:
        parts.append(f"seed{int(seed_index)}")
    return "__".join(parts)


def _task(
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    phase: str,
    outer_fold: int,
    inner_fold: int,
    epochs_per_seed: Mapping[str, int] | None = None,
    seed_index: int | None = None,
    epoch_budget_override: int | None = None,
) -> dict[str, Any]:
    params = _model_params(config)
    export_config_id = gpu_configuration_id("torch_mlp", params)
    candidate_config_id = robust_candidate_config_id(candidate, params)
    result: dict[str, Any] = {
        "phase": phase,
        "scheme": "segment_loso",
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "model": "torch_mlp",
        "params": params,
        "config_id": export_config_id,
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_config_id": candidate_config_id,
        "candidate": dict(candidate),
        "training_recipe": dict(candidate),
    }
    if epochs_per_seed is not None:
        result["epochs_per_seed"] = {
            str(key): int(value) for key, value in epochs_per_seed.items()
        }
    if seed_index is not None:
        result["seed_indices"] = [int(seed_index)]
    if epoch_budget_override is not None:
        result["epoch_budget_override"] = int(epoch_budget_override)
        result["smoke_contract_eligible"] = False
    result["task_id"] = _robust_task_id(
        phase,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        candidate_id=str(candidate["candidate_id"]),
        candidate_config_id=candidate_config_id,
        seed_index=seed_index,
    )
    return result


def robust_inner_tasks(
    config: dict[str, Any], *, smoke: bool = False
) -> list[dict[str, Any]]:
    validate_robust_gpu_config(config)
    groups = [int(value) for value in config["splits"]["primary"]["groups"]]
    tasks: list[dict[str, Any]] = []
    for outer_fold in (groups[:1] if smoke else groups):
        inner_groups = [group for group in groups if group != outer_fold]
        for inner_fold in (inner_groups[:1] if smoke else inner_groups):
            for candidate in robust_candidate_grid(config):
                tasks.append(
                    _task(
                        config,
                        candidate,
                        phase="smoke" if smoke else "inner",
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                        epoch_budget_override=50 if smoke else None,
                    )
                )
    expected = 6 if smoke else 120
    if len(tasks) != expected or len({task["task_id"] for task in tasks}) != expected:
        raise GPURobustExperimentError(
            f"robust inner task count/identity changed: {len(tasks)} != {expected}"
        )
    return tasks


def _candidate_rows(
    results: Sequence[dict[str, Any]], candidate_id: str, *, outer_fold: int | None = None
) -> list[dict[str, Any]]:
    rows = [
        result
        for result in results
        if result["task"]["candidate_id"] == candidate_id
        and (outer_fold is None or int(result["task"]["outer_fold"]) == outer_fold)
    ]
    return sorted(
        rows,
        key=lambda row: (
            int(row["task"]["outer_fold"]),
            int(row["task"]["inner_fold"]),
        ),
    )


def _refit_epochs(
    results: Sequence[dict[str, Any]], candidate_id: str, *, outer_fold: int | None
) -> dict[str, int]:
    rows = _candidate_rows(results, candidate_id, outer_fold=outer_fold)
    expected = 4 if outer_fold is not None else 20
    if len(rows) != expected:
        raise GPURobustExperimentError(
            f"robust refit context count changed: {len(rows)} != {expected}"
        )
    output: dict[str, int] = {}
    for seed_index in range(3):
        epochs: list[int] = []
        for row in rows:
            matches = [
                seed
                for seed in row["seed_summaries"]
                if int(seed["seed_index"]) == seed_index
            ]
            if len(matches) != 1:
                raise GPURobustExperimentError("robust inner seed evidence is incomplete")
            epochs.append(int(matches[0]["selected_epoch"]))
        output[str(seed_index)] = _round_median(epochs)
    return output


def robust_selection_and_outer_tasks(
    config: dict[str, Any], inner_results: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = [int(value) for value in config["splits"]["primary"]["groups"]]
    trace: list[dict[str, Any]] = []
    candidates = robust_candidate_grid(config)
    for priority, candidate in enumerate(candidates):
        candidate_id = str(candidate["candidate_id"])
        rows = _candidate_rows(inner_results, candidate_id)
        if len(rows) != 20:
            raise GPURobustExperimentError(
                f"candidate {candidate_id} has {len(rows)} validation contexts, not 20"
            )
        rmses = [float(row["metrics"]["rmse_3d_mm"]) for row in rows]
        p95s = [float(row["metrics"]["p95_3d_error_mm"]) for row in rows]
        if not np.isfinite([*rmses, *p95s]).all():
            raise GPURobustExperimentError("robust selection contains non-finite metrics")
        per_validation_segment = []
        for segment in groups:
            selected = [
                float(row["metrics"]["rmse_3d_mm"])
                for row in rows
                if int(row["task"]["inner_fold"]) == segment
            ]
            if len(selected) != 4:
                raise GPURobustExperimentError(
                    f"candidate {candidate_id} validation segment {segment} lacks four contexts"
                )
            per_validation_segment.append(
                {"segment_id": segment, "context_count": 4, "mean_rmse_3d_mm": float(np.mean(selected))}
            )
        mean_rmse = float(np.mean(rmses))
        worst_segment = max(
            per_validation_segment, key=lambda row: float(row["mean_rmse_3d_mm"])
        )
        robust_j = 0.5 * (mean_rmse + float(worst_segment["mean_rmse_3d_mm"]))
        seed_stds = []
        for row in rows:
            seed_scores = [
                float(seed["evaluation_rmse_3d_mm"])
                for seed in row["seed_summaries"]
            ]
            if len(seed_scores) != 3 or not np.isfinite(seed_scores).all():
                raise GPURobustExperimentError("robust seed dispersion evidence is incomplete")
            seed_stds.append(float(np.std(seed_scores, ddof=0)))
        trace.append(
            {
                "candidate_id": candidate_id,
                "candidate": dict(candidate),
                "listed_priority": priority,
                "context_count": len(rows),
                "mean_rmse_3d_mm": mean_rmse,
                "mean_p95_3d_error_mm": float(np.mean(p95s)),
                "mean_seed_rmse_std_mm": float(np.mean(seed_stds)),
                "per_validation_segment": per_validation_segment,
                "worst_validation_segment": worst_segment,
                "robust_j_mm": robust_j,
            }
        )
    winner = min(
        trace,
        key=lambda row: (
            float(row["robust_j_mm"]),
            float(row["mean_p95_3d_error_mm"]),
            float(row["mean_seed_rmse_std_mm"]),
            int(row["listed_priority"]),
        ),
    )
    selected_candidate = dict(winner["candidate"])
    selected_id = str(winner["candidate_id"])
    selected_j = float(winner["robust_j_mm"])
    promotion_eligible = selected_j < HISTORICAL_V1_ROBUST_J_MM and not math.isclose(
        selected_j,
        HISTORICAL_V1_ROBUST_J_MM,
        rel_tol=0.0,
        abs_tol=float(config["selection"]["numeric_equality_abs_tolerance_mm"]),
    )
    global_epochs = _refit_epochs(inner_results, selected_id, outer_fold=None)
    model_params = _model_params(config)
    export_config_id = gpu_configuration_id("torch_mlp", model_params)
    compatible_selection = {
        "scheme": "segment_loso",
        "outer_fold": None,
        "model": "torch_mlp",
        "selection_metric": "robust_j_mm",
        "selected_config_id": export_config_id,
        "selected_params": model_params,
        "selected_score_mm": selected_j,
        "refit_epochs_per_seed": global_epochs,
    }
    selection = {
        "outer_diagnostic_used_for_selection": False,
        "post_v1_outer_observed": True,
        "context_count_per_candidate": 20,
        "seed_level_validation_fits_per_candidate": 60,
        "trace": trace,
        "selected_candidate_id": selected_id,
        "selected_candidate": selected_candidate,
        "selected_robust_j_mm": selected_j,
        "historical_v1_robust_j_mm": HISTORICAL_V1_ROBUST_J_MM,
        "promotion_eligible": promotion_eligible,
        "promotion_action": (
            "promote_robust_v2"
            if promotion_eligible
            else "retain_v1"
        ),
        "final_refit_epochs_per_seed": global_epochs,
        "final_global_segment_inner_selection": compatible_selection,
    }
    outer_tasks = [
        _task(
            config,
            selected_candidate,
            phase="outer",
            outer_fold=outer_fold,
            inner_fold=-1,
            epochs_per_seed=_refit_epochs(
                inner_results, selected_id, outer_fold=outer_fold
            ),
        )
        for outer_fold in groups
    ]
    return selection, outer_tasks


def robust_final_tasks(
    config: Mapping[str, Any], selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    candidate = selection["selected_candidate"]
    epochs = selection["final_refit_epochs_per_seed"]
    return [
        _task(
            config,
            candidate,
            phase="final",
            outer_fold=-1,
            inner_fold=-1,
            epochs_per_seed={str(seed): int(epochs[str(seed)])},
            seed_index=seed,
        )
        for seed in range(3)
    ]


def _runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(config["runtime"])
    runtime["xgboost_version"] = "3.3.0"
    return {"runtime": runtime, "models": config["models"]}


def _robust_task_cost(task: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    epochs = float(task.get("epoch_budget_override", config["models"]["torch_mlp"]["epoch_budget"]))
    if "epochs_per_seed" in task:
        epochs = float(sum(int(value) for value in task["epochs_per_seed"].values()))
        seeds = 1.0 if "seed_indices" in task else 1.0
    else:
        seeds = float(len(task.get("seed_indices", [])) or 3)
    sam_multiplier = 2.0 if bool(task["candidate"].get("sam", False)) else 1.0
    return epochs * seeds * sam_multiplier


def _balanced_task_bins(
    tasks: Sequence[dict[str, Any]], count: int, config: Mapping[str, Any]
) -> list[list[dict[str, Any]]]:
    bins = [[] for _ in range(count)]
    loads = [0.0] * count
    for task in sorted(
        tasks,
        key=lambda item: (-_robust_task_cost(item, config), str(item["task_id"])),
    ):
        index = min(range(count), key=lambda value: (loads[value], value))
        bins[index].append(task)
        loads[index] += _robust_task_cost(task, config)
    return bins


def _robust_worker_loop(
    gpu_id: int,
    config: dict[str, Any],
    release_path: str,
    run_dir: str,
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=1)
        runtime = _worker_runtime(gpu_id, _runtime_config(config))
        _verify_robust_source(config)
        data = _load_worker_data(config, Path(release_path))
        result_queue.put({"event": "worker_ready", "gpu": gpu_id, "runtime": runtime})
        while True:
            task = task_queue.get()
            if task is None:
                break
            try:
                result = _execute_task(
                    data,
                    task=task,
                    config=config,
                    run_dir=Path(run_dir),
                    gpu_id=gpu_id,
                )
                result_queue.put(
                    {
                        "event": "task_complete",
                        "gpu": gpu_id,
                        "task_id": task["task_id"],
                        "candidate_id": task["candidate_id"],
                        "phase": task["phase"],
                        "outer_fold": task["outer_fold"],
                        "inner_fold": task["inner_fold"],
                        "rmse_3d_mm": (
                            None
                            if result["metrics"] is None
                            else result["metrics"]["rmse_3d_mm"]
                        ),
                        "seconds": result["elapsed_seconds"],
                    }
                )
            except BaseException as exc:
                result_queue.put(
                    {
                        "event": "task_error",
                        "gpu": gpu_id,
                        "task_id": task.get("task_id"),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                return
        result_queue.put({"event": "worker_done", "gpu": gpu_id})
        limiter.restore_original_limits()
    except BaseException as exc:
        result_queue.put(
            {
                "event": "worker_error",
                "gpu": gpu_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _read_completed(
    root: Path, task: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _result_path(root, task)
    if not path.is_file():
        return None
    result = read_json_object(path)
    if (
        result.get("contract") != ROBUST_GPU_EXPERIMENT_CONTRACT
        or result.get("config_sha256") != robust_gpu_config_sha256(dict(config))
        or result.get("source_fingerprint") != config.get("_source_fingerprint")
        or result.get("task") != dict(task)
        or result.get("task_hash") != stable_hash(task)
    ):
        raise GPURobustExperimentError(f"robust resume identity changed: {task['task_id']}")
    expected = {
        "prediction_file": (
            None
            if task["phase"] == "final"
            else f"{task['phase']}/predictions/{task['task_id']}.parquet"
        ),
        "history_file": f"{task['phase']}/history/{task['task_id']}.parquet",
        "group_history_file": f"{task['phase']}/group_history/{task['task_id']}.parquet",
        "checkpoint_file": (
            f"final/checkpoints/torch_mlp_seed{int(task['seed_indices'][0])}.npz"
            if task["phase"] == "final"
            else None
        ),
    }
    for key, relative in expected.items():
        entry = result.get(key)
        if relative is None:
            if entry is not None:
                raise GPURobustExperimentError(
                    f"unexpected robust resume artifact: {task['task_id']}/{key}"
                )
            continue
        if not isinstance(entry, dict):
            raise GPURobustExperimentError(
                f"missing robust resume artifact: {task['task_id']}/{key}"
            )
        artifact = _safe_run_member(root, entry.get("path"), expected=relative)
        if sha256_file(artifact) != entry.get("sha256"):
            raise GPURobustExperimentError(
                f"changed robust resume artifact: {task['task_id']}/{key}"
            )
    return result


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True), flush=True)


def _task_monitor_payload(root: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    task = result["task"]
    summaries = result["seed_summaries"]
    trained_epochs = [int(row["trained_epochs"]) for row in summaries]
    selected_epochs = [int(row["selected_epoch"]) for row in summaries]
    payload: dict[str, Any] = {
        "event": "robust_task_monitor",
        "task_id": task["task_id"],
        "phase": task["phase"],
        "candidate_id": task["candidate_id"],
        "outer_fold": task["outer_fold"],
        "inner_fold": task["inner_fold"],
        "selected_epochs": selected_epochs,
        "trained_epochs": trained_epochs,
        "checkpoint_staleness_epochs": [
            trained - selected
            for trained, selected in zip(trained_epochs, selected_epochs, strict=True)
        ],
        "seed_train_rmse_3d_mm": [
            float(row["selected_train_rmse_3d_mm"]) for row in summaries
        ],
    }
    if result.get("metrics") is not None:
        seed_rmse = [float(row["evaluation_rmse_3d_mm"]) for row in summaries]
        train_rmse = payload["seed_train_rmse_3d_mm"]
        payload.update(
            {
                "ensemble_rmse_3d_mm": float(result["metrics"]["rmse_3d_mm"]),
                "ensemble_p95_3d_error_mm": float(
                    result["metrics"]["p95_3d_error_mm"]
                ),
                "seed_rmse_3d_mm": seed_rmse,
                "seed_rmse_std_mm": float(np.std(seed_rmse, ddof=0)),
                "train_validation_gap_mm": [
                    evaluation - train
                    for evaluation, train in zip(seed_rmse, train_rmse, strict=True)
                ],
            }
        )
        prediction = pd.read_parquet(root / result["prediction_file"]["path"])
        per_segment = []
        for segment_id, rows in prediction.groupby("segment_id", sort=True):
            truth = rows.loc[:, list(TRUE_COLUMNS)].to_numpy(dtype=np.float64)
            predicted = rows.loc[:, list(PREDICTION_COLUMNS)].to_numpy(dtype=np.float64)
            metrics = regression_metrics(truth, predicted)
            per_segment.append(
                {
                    "segment_id": int(segment_id),
                    "rows": int(len(rows)),
                    "rmse_3d_mm": float(metrics["rmse_3d_mm"]),
                    "p95_3d_error_mm": float(metrics["p95_3d_error_mm"]),
                }
            )
        payload["per_segment"] = per_segment
    group_history = pd.read_parquet(root / result["group_history_file"]["path"])
    latest = group_history.sort_values("epoch", kind="mergesort").groupby(
        ["seed_index", "segment_id"], sort=True, as_index=False
    ).tail(1)
    payload["latest_scaled_group_mse_range"] = [
        float(latest["scaled_group_mse_loss"].min()),
        float(latest["scaled_group_mse_loss"].max()),
    ]
    if task["candidate"]["objective"] == "group_dro":
        payload["latest_group_dro_q_range"] = [
            float(latest["group_dro_q"].min()),
            float(latest["group_dro_q"].max()),
        ]
    return payload


def _run_batch(
    tasks: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    release_path: Path,
    run_dir: Path,
    gpu_ids: Sequence[int],
) -> list[dict[str, Any]]:
    _verify_robust_source(config)
    results: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for task in tasks:
        result = _read_completed(run_dir, task, config)
        if result is None:
            pending.append(task)
        else:
            results[str(task["task_id"])] = result
    if not pending:
        return [results[str(task["task_id"])] for task in tasks]

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    task_queues = [context.Queue() for _ in gpu_ids]
    workers = [
        context.Process(
            target=_robust_worker_loop,
            args=(
                int(gpu_id),
                config,
                str(release_path),
                str(run_dir),
                task_queues[index],
                result_queue,
            ),
            name=f"gpu-robust-v2-{gpu_id}",
        )
        for index, gpu_id in enumerate(gpu_ids)
    ]
    bins = _balanced_task_bins(pending, len(gpu_ids), config)
    for worker in workers:
        worker.start()
    for task_queue, assigned in zip(task_queues, bins, strict=True):
        for task in assigned:
            task_queue.put(task)
        task_queue.put(None)

    completed = 0
    failure: dict[str, Any] | None = None
    last_progress = time.monotonic()
    try:
        while completed < len(pending) and failure is None:
            try:
                event = result_queue.get(timeout=2.0)
            except queue.Empty:
                if all(not worker.is_alive() for worker in workers):
                    failure = {
                        "event": "workers_exited_before_all_results",
                        "completed": completed,
                        "expected": len(pending),
                    }
                elif time.monotonic() - last_progress >= 30.0:
                    _emit(
                        {
                            "event": "progress",
                            "completed": completed,
                            "pending": len(pending) - completed,
                            "total": len(pending),
                        }
                    )
                    last_progress = time.monotonic()
                continue
            name = str(event.get("event"))
            if name == "task_complete":
                completed += 1
                task = next(
                    value for value in pending if value["task_id"] == event["task_id"]
                )
                result = _read_completed(run_dir, task, config)
                if result is None:
                    raise GPURobustExperimentError(
                        f"worker reported missing result: {task['task_id']}"
                    )
                results[str(task["task_id"])] = result
                _emit(_task_monitor_payload(run_dir, result))
                if completed == 1 or completed % 6 == 0 or completed == len(pending):
                    _emit({**event, "completed": completed, "total": len(pending)})
                last_progress = time.monotonic()
            elif name == "worker_ready":
                _emit(event)
            elif name == "worker_done":
                continue
            elif name in {"worker_error", "task_error"}:
                failure = event
                _emit(event)
            else:
                failure = {"event": "unknown_worker_event", "payload": event}
        if failure is not None:
            raise GPURobustExperimentError(json.dumps(failure, ensure_ascii=False))
    except BaseException:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        raise
    finally:
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
    if completed != len(pending):
        raise GPURobustExperimentError(
            f"robust task batch incomplete: {completed}/{len(pending)}"
        )
    _verify_robust_source(config)
    return [results[str(task["task_id"])] for task in tasks]


def _outer_summary(
    results: Sequence[dict[str, Any]], *, run_dir: Path, verify_existing: bool = False
) -> dict[str, Any]:
    rows = sorted(results, key=lambda row: int(row["task"]["outer_fold"]))
    if [int(row["task"]["outer_fold"]) for row in rows] != [36, 37, 38, 39, 40]:
        raise GPURobustExperimentError("robust outer fold set changed")
    frames = [
        pd.read_parquet(run_dir / str(row["prediction_file"]["path"])) for row in rows
    ]
    oof = pd.concat(frames, ignore_index=True)
    oof.sort_values("trial_uid", kind="mergesort", inplace=True)
    oof.reset_index(drop=True, inplace=True)
    if len(oof) != 7702 or oof["trial_uid"].astype(str).duplicated().any():
        raise GPURobustExperimentError("robust OOF coverage changed")
    truth = oof.loc[:, list(TRUE_COLUMNS)].to_numpy(dtype=np.float64)
    prediction = oof.loc[:, list(PREDICTION_COLUMNS)].to_numpy(dtype=np.float64)
    pooled = regression_metrics(truth, prediction)
    metric_names = (
        "rmse_3d_mm",
        "mean_3d_error_mm",
        "median_3d_error_mm",
        "p90_3d_error_mm",
        "p95_3d_error_mm",
    )
    macro = {
        name: float(np.mean([float(row["metrics"][name]) for row in rows]))
        for name in metric_names
    }
    per_group = []
    for row in rows:
        seed_scores = [
            float(seed["evaluation_rmse_3d_mm"])
            for seed in row["seed_summaries"]
        ]
        per_group.append(
            {
                "outer_fold": int(row["task"]["outer_fold"]),
                "candidate_id": str(row["task"]["candidate_id"]),
                "row_count": int(row["evaluation_rows"]),
                **{name: float(row["metrics"][name]) for name in metric_names},
                "seed_rmse_3d_mm": seed_scores,
                "seed_rmse_std_mm": float(np.std(seed_scores, ddof=0)),
            }
        )
    oof_path = run_dir / "evaluation" / "segment_loso__robust_v2__oof.parquet"
    if verify_existing:
        recorded = pd.read_parquet(oof_path)
        if not recorded.equals(oof):
            raise GPURobustExperimentError("robust OOF differs from fold predictions")
    else:
        atomic_write_parquet(oof_path, oof)
    return {
        "designation": "reused_holdout_development_diagnostic",
        "independent_test": False,
        "used_for_reselection": False,
        "candidate_id": str(rows[0]["task"]["candidate_id"]),
        "outer_group_count": 5,
        "oof_rows": len(oof),
        "macro_equal_group": macro,
        "pooled": pooled,
        "per_group": per_group,
        "oof_prediction_file": {
            "path": _relative(oof_path, run_dir),
            "sha256": sha256_file(oof_path),
        },
    }


def _prepare_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    config["_result_contract"] = ROBUST_GPU_EXPERIMENT_CONTRACT
    config["_result_config_sha256"] = robust_gpu_config_sha256(config)
    return config


def _training_recipe_payload(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    source_fingerprint: str,
) -> dict[str, Any]:
    return {
        "contract": ROBUST_MODEL_TRAINING_CONTRACT,
        "experiment_contract": ROBUST_GPU_EXPERIMENT_CONTRACT,
        "robust_config_sha256": robust_gpu_config_sha256(dict(config)),
        "source_fingerprint": source_fingerprint,
        "candidate_id": selection["selected_candidate_id"],
        "candidate": selection["selected_candidate"],
        "candidate_config_id": robust_candidate_config_id(
            selection["selected_candidate"], _model_params(config)
        ),
        "optimization_metadata_only": "segment_id",
        "optimization_metadata_at_inference": False,
        "inference_feature_count": 43,
        "promotion_eligible": selection["promotion_eligible"],
        "promotion_action": selection["promotion_action"],
        "confirmatory_independent_restart_sessions_required": 10,
    }


def _summary_payload(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    final_model: Mapping[str, Any],
    training_recipe_reference: Mapping[str, Any],
    *,
    status: str,
    validation_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "contract": ROBUST_GPU_EXPERIMENT_CONTRACT,
        "status": status,
        "objective": "segment-robust Vive-free static endpoint development",
        "selection": dict(selection),
        "evaluation": dict(evaluation),
        "final_model": dict(final_model),
        "training_recipe": dict(training_recipe_reference),
        "claim_boundaries": config["claim_boundaries"],
    }
    if validation_reference is not None:
        summary["validation"] = dict(validation_reference)
    return summary


def _robust_model_manifest_payload(
    base_model: Mapping[str, Any],
    selection: Mapping[str, Any],
    training_recipe_reference: Mapping[str, Any],
) -> dict[str, Any]:
    model = dict(base_model)
    model.update(
        {
            "contract": ROBUST_GPU_MODEL_CONTRACT,
            "candidate_id": selection["selected_candidate_id"],
            "candidate": selection["selected_candidate"],
            "promotion_eligible": selection["promotion_eligible"],
            "promotion_action": selection["promotion_action"],
            "training_recipe": dict(training_recipe_reference),
        }
    )
    return model


def run_robust_gpu_experiment(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    gpu_ids: Sequence[int] = (0, 1, 2, 3),
    resume: bool = False,
) -> dict[str, Any]:
    validate_robust_gpu_config(config)
    gpu_ids = _validate_gpu_ids(gpu_ids, int(config["runtime"]["gpu_count"]))
    release_path = resolve_robust_gpu_release_path(config)
    release = _load_worker_data(config, release_path).release
    if len(release.eligible_real) != 7702:
        raise GPURobustExperimentError("robust verified release row count changed")
    workspace = Path(str(config["_workspace_root"])).resolve()
    source = robust_training_source_payload(workspace)
    config["_source_fingerprint"] = source["fingerprint"]
    config["_allowed_gpu_ids"] = list(gpu_ids)
    _prepare_runtime_config(config)
    environment = gpu_environment(gpu_ids, _runtime_config(config))
    environment["git"] = _git_payload(workspace)
    environment["training_source"] = source
    environment["resolved_release_path"] = str(release_path)
    environment["training_workspace_root"] = str(workspace)
    identity = {
        "contract": ROBUST_GPU_EXPERIMENT_CONTRACT,
        "config_sha256": robust_gpu_config_sha256(config),
        "release_sha256": str(config["release"]["archive_sha256"]),
        "source_fingerprint": source["fingerprint"],
        "run_kind": "full",
        "gpu_ids": list(gpu_ids),
    }
    run_dir = prepare_experiment_directory(output_dir, identity=identity, resume=resume)
    config_path = run_dir / "config.resolved.json"
    environment_path = run_dir / "environment.json"
    if resume:
        if read_json_object(config_path) != public_robust_gpu_config(config):
            raise GPURobustExperimentError("robust resume config changed")
        if read_json_object(environment_path) != environment:
            raise GPURobustExperimentError("robust resume environment/source changed")
    else:
        atomic_write_json(config_path, public_robust_gpu_config(config))
        atomic_write_json(environment_path, environment)

    inner_tasks = robust_inner_tasks(config)
    _emit({"event": "robust_run_started", "inner_tasks": len(inner_tasks), "gpus": list(gpu_ids)})
    inner_results = _run_batch(
        inner_tasks,
        config=config,
        release_path=release_path,
        run_dir=run_dir,
        gpu_ids=gpu_ids,
    )
    selection, outer_tasks = robust_selection_and_outer_tasks(config, inner_results)
    selection_path = run_dir / "selection" / "inner_selection.json"
    atomic_write_json(selection_path, selection)
    for row in selection["trace"]:
        _emit(
            {
                "event": "robust_candidate_direction",
                "candidate_id": row["candidate_id"],
                "mean_rmse_3d_mm": row["mean_rmse_3d_mm"],
                "worst_validation_segment": row["worst_validation_segment"],
                "mean_p95_3d_error_mm": row["mean_p95_3d_error_mm"],
                "mean_seed_rmse_std_mm": row["mean_seed_rmse_std_mm"],
                "robust_j_mm": row["robust_j_mm"],
            }
        )
    _emit(
        {
            "event": "robust_inner_selection_complete",
            "candidate_id": selection["selected_candidate_id"],
            "robust_j_mm": selection["selected_robust_j_mm"],
            "promotion_action": selection["promotion_action"],
        }
    )

    outer_results = _run_batch(
        outer_tasks,
        config=config,
        release_path=release_path,
        run_dir=run_dir,
        gpu_ids=gpu_ids,
    )
    evaluation = _outer_summary(outer_results, run_dir=run_dir)
    atomic_write_json(run_dir / "evaluation" / "summary.json", evaluation)

    final_tasks = robust_final_tasks(config, selection)
    final_results = _run_batch(
        final_tasks,
        config=config,
        release_path=release_path,
        run_dir=run_dir,
        gpu_ids=gpu_ids,
    )
    training_recipe = _training_recipe_payload(
        config, selection, source_fingerprint=source["fingerprint"]
    )
    recipe_path = run_dir / "final" / "training_recipe.json"
    atomic_write_json(recipe_path, training_recipe)
    recipe_reference = {
        "path": _relative(recipe_path, run_dir),
        "sha256": sha256_file(recipe_path),
    }
    base_model = _final_model_manifest(
        config, selection, final_results, run_dir=run_dir
    )
    final_model = _robust_model_manifest_payload(
        base_model, selection, recipe_reference
    )
    atomic_write_json(run_dir / "final" / "model_manifest.json", final_model)
    summary = _summary_payload(
        config,
        selection,
        evaluation,
        final_model,
        recipe_reference,
        status="awaiting_validation",
    )
    atomic_write_json(run_dir / "summary.json", summary)
    _verify_robust_source(config)
    manifest = {
        "contract": ROBUST_GPU_EXPERIMENT_CONTRACT,
        "status": "awaiting_validation",
        "config_sha256": robust_gpu_config_sha256(config),
        "release_sha256": str(config["release"]["archive_sha256"]),
        "source_binding": "task_embedded",
        "source_fingerprint": source["fingerprint"],
        "counts": {
            "eligible_real_rows": 7702,
            "candidates": 6,
            "validation_contexts_per_candidate": 20,
            "seed_level_validation_fits": 360,
            "inner_tasks": len(inner_results),
            "outer_tasks": len(outer_results),
            "final_seed_tasks": len(final_results),
            "gpu_count": len(gpu_ids),
        },
        "selection": {
            "path": _relative(selection_path, run_dir),
            "sha256": sha256_file(selection_path),
        },
        "training_recipe": recipe_reference,
        "final_model": final_model,
        "files": manifest_files(run_dir),
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    try:
        initial_validation = validate_robust_gpu_run(
            run_dir, _expected_manifest_status="awaiting_validation"
        )
        validation_path = run_dir / "validation.json"
        atomic_write_json(validation_path, initial_validation)
        validation_reference = {
            "path": _relative(validation_path, run_dir),
            "sha256": sha256_file(validation_path),
        }
        summary = _summary_payload(
            config,
            selection,
            evaluation,
            final_model,
            recipe_reference,
            status="complete",
            validation_reference=validation_reference,
        )
        atomic_write_json(run_dir / "summary.json", summary)
        manifest["status"] = "finalizing_validation"
        manifest["validation"] = validation_reference
        manifest["files"] = manifest_files(run_dir)
        atomic_write_json(run_dir / "manifest.json", manifest)
        validate_robust_gpu_run(
            run_dir, _expected_manifest_status="finalizing_validation"
        )
        manifest["status"] = "complete"
        atomic_write_json(run_dir / "manifest.json", manifest)
        validation = validate_robust_gpu_run(run_dir)
    except BaseException as exc:
        failure_path = run_dir / "validation_failure.json"
        atomic_write_json(
            failure_path,
            {
                "status": "validation_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        summary["status"] = "validation_failed"
        summary["validation_failure"] = {
            "path": _relative(failure_path, run_dir),
            "sha256": sha256_file(failure_path),
        }
        atomic_write_json(run_dir / "summary.json", summary)
        manifest["status"] = "validation_failed"
        manifest["validation_failure"] = summary["validation_failure"]
        manifest["files"] = manifest_files(run_dir)
        atomic_write_json(run_dir / "manifest.json", manifest)
        raise
    _emit({"event": "robust_run_complete", "validation": validation})
    return manifest


def run_robust_gpu_smoke(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    gpu_ids: Sequence[int] = (0, 1, 2, 3),
    resume: bool = False,
) -> dict[str, Any]:
    validate_robust_gpu_config(config)
    gpu_ids = _validate_gpu_ids(gpu_ids, 4)
    workspace = Path(str(config["_workspace_root"])).resolve()
    source = robust_training_source_payload(workspace)
    config["_source_fingerprint"] = source["fingerprint"]
    config["_allowed_gpu_ids"] = list(gpu_ids)
    _prepare_runtime_config(config)
    release_path = resolve_robust_gpu_release_path(config)
    identity = {
        "contract": ROBUST_GPU_EXPERIMENT_CONTRACT,
        "config_sha256": robust_gpu_config_sha256(config),
        "release_sha256": str(config["release"]["archive_sha256"]),
        "source_fingerprint": source["fingerprint"],
        "run_kind": "smoke_non_contract",
        "gpu_ids": list(gpu_ids),
    }
    run_dir = prepare_experiment_directory(output_dir, identity=identity, resume=resume)
    atomic_write_json(run_dir / "config.resolved.json", public_robust_gpu_config(config))
    tasks = robust_inner_tasks(config, smoke=True)
    results = _run_batch(
        tasks,
        config=config,
        release_path=release_path,
        run_dir=run_dir,
        gpu_ids=gpu_ids,
    )
    smoke_data = _load_worker_data(config, release_path)
    smoke_array_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    validated_results = _validate_task_results(
        run_dir,
        config,
        smoke_data,
        tasks,
        phase="smoke",
        source_binding="task_embedded",
        source_fingerprint=source["fingerprint"],
        array_cache=smoke_array_cache,
    )
    _validate_group_history(
        run_dir, validated_results, smoke_array_cache
    )
    summary = {
        "status": "smoke_ok",
        "contract_eligible": False,
        "candidate_count": 6,
        "task_count": len(results),
        "seed_level_fits": len(results) * 3,
        "finite_metrics": all(
            np.isfinite(float(row["metrics"]["rmse_3d_mm"])) for row in results
        ),
        "semantic_task_validation": True,
        "candidate_metrics": [
            {
                "candidate_id": row["task"]["candidate_id"],
                "rmse_3d_mm": float(row["metrics"]["rmse_3d_mm"]),
            }
            for row in results
        ],
    }
    atomic_write_json(run_dir / "smoke_summary.json", summary)
    return summary


def _validate_group_history(
    root: Path,
    results: Sequence[dict[str, Any]],
    array_cache: Mapping[tuple[Any, ...], dict[str, Any]],
) -> None:
    for result in results:
        task = result["task"]
        reference = result.get("group_history_file")
        if not isinstance(reference, dict):
            raise GPURobustExperimentError(
                f"robust group history missing: {task['task_id']}"
            )
        relative = f"{task['phase']}/group_history/{task['task_id']}.parquet"
        path = _safe_run_member(root, reference.get("path"), expected=relative)
        if sha256_file(path) != reference.get("sha256"):
            raise GPURobustExperimentError(
                f"robust group history digest changed: {task['task_id']}"
            )
        frame = pd.read_parquet(path)
        if len(frame) != reference.get("rows") or frame.empty:
            raise GPURobustExperimentError(
                f"robust group history row count changed: {task['task_id']}"
            )
        if set(frame["task_id"].astype(str)) != {str(task["task_id"])}:
            raise GPURobustExperimentError(
                f"robust group history task identity changed: {task['task_id']}"
            )
        cache_key = (
            ("final",)
            if task["phase"] == "final"
            else (
                str(task["scheme"]),
                int(task["outer_fold"]),
                int(task["inner_fold"]),
            )
        )
        expected_groups = {
            str(value) for value in array_cache[cache_key]["train"]["segment_id"].unique()
        }
        if set(frame["segment_id"].astype(str)) != expected_groups:
            raise GPURobustExperimentError(
                f"robust group history consumed held-out metadata: {task['task_id']}"
            )
        if set(frame["candidate_id"].astype(str)) != {str(task["candidate_id"])}:
            raise GPURobustExperimentError(
                f"robust group history candidate changed: {task['task_id']}"
            )
        if set(frame["objective"].astype(str)) != {
            str(task["candidate"]["objective"])
        } or set(frame["sam"].astype(bool)) != {
            bool(task["candidate"].get("sam", False))
        }:
            raise GPURobustExperimentError(
                f"robust group history objective changed: {task['task_id']}"
            )
        always_numeric = frame[
            [
                "seed_index",
                "model_seed",
                "epoch",
                "scaled_group_mse_loss",
                "empirical_train_mass",
                "balanced_batch_mass",
                "v_rex_beta",
            ]
        ].to_numpy(dtype=np.float64)
        if not np.isfinite(always_numeric).all():
            raise GPURobustExperimentError(
                f"robust group history contains non-finite evidence: {task['task_id']}"
            )
        expected_mass = 1.0 / len(expected_groups)
        if not np.allclose(
            frame["balanced_batch_mass"].to_numpy(dtype=np.float64),
            expected_mass,
            rtol=0.0,
            atol=1e-12,
        ):
            raise GPURobustExperimentError(
                f"robust balanced objective mass changed: {task['task_id']}"
            )
        train_segments = array_cache[cache_key]["train"]["segment_id"].astype(str)
        empirical = train_segments.value_counts(normalize=True).to_dict()
        for segment_id, rows in frame.groupby("segment_id", sort=False):
            if not np.allclose(
                rows["empirical_train_mass"].to_numpy(dtype=np.float64),
                float(empirical[str(segment_id)]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise GPURobustExperimentError(
                    f"robust empirical group mass changed: {task['task_id']}"
                )
        if task["candidate"]["objective"] == "v_rex":
            warmup = int(task["candidate"]["warmup_epochs"])
            beta = float(task["candidate"]["beta"])
            expected_beta = np.where(
                frame["epoch"].to_numpy(dtype=np.int64) <= warmup, 0.0, beta
            )
        else:
            expected_beta = np.zeros(len(frame), dtype=np.float64)
        if not np.array_equal(
            frame["v_rex_beta"].to_numpy(dtype=np.float64), expected_beta
        ):
            raise GPURobustExperimentError(
                f"robust V-REx warm-up schedule changed: {task['task_id']}"
            )
        if task["candidate"]["objective"] == "group_dro":
            q = frame["group_dro_q"].to_numpy(dtype=np.float64)
            if not np.isfinite(q).all() or np.any(q <= 0.0) or np.any(q >= 1.0):
                raise GPURobustExperimentError(
                    f"robust GroupDRO weights are invalid: {task['task_id']}"
                )
            for _, group in frame.groupby(["seed_index", "epoch"], sort=False):
                if not math.isclose(float(group["group_dro_q"].sum()), 1.0, abs_tol=1e-9):
                    raise GPURobustExperimentError(
                        f"robust GroupDRO weights do not sum to one: {task['task_id']}"
                    )
        elif frame["group_dro_q"].notna().any():
            raise GPURobustExperimentError(
                f"non-GroupDRO candidate emitted q weights: {task['task_id']}"
            )


def validate_robust_gpu_run(
    run_dir: str | Path,
    *,
    _expected_manifest_status: str = "complete",
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    manifest = read_json_object(root / "manifest.json")
    if manifest.get("contract") != ROBUST_GPU_EXPERIMENT_CONTRACT:
        raise GPURobustExperimentError("robust manifest contract changed")
    if _expected_manifest_status not in {
        "awaiting_validation",
        "finalizing_validation",
        "complete",
    }:
        raise GPURobustExperimentError("invalid internal robust validation state")
    if manifest.get("status") != _expected_manifest_status:
        raise GPURobustExperimentError(
            "robust run is not in the required durable validation state"
        )
    config = read_json_object(root / "config.resolved.json")
    validate_robust_gpu_config(config)
    config_sha = robust_gpu_config_sha256(config)
    if manifest.get("config_sha256") != config_sha:
        raise GPURobustExperimentError("robust manifest/config hash mismatch")
    expected_counts = {
        "eligible_real_rows": 7702,
        "candidates": 6,
        "validation_contexts_per_candidate": 20,
        "seed_level_validation_fits": 360,
        "inner_tasks": 120,
        "outer_tasks": 5,
        "final_seed_tasks": 3,
        "gpu_count": 4,
    }
    if manifest.get("counts") != expected_counts:
        raise GPURobustExperimentError("robust run counts changed")
    verify_file_manifest(root, manifest["files"])

    environment = read_json_object(root / "environment.json")
    source = environment.get("training_source")
    if (
        not isinstance(source, dict)
        or source.get("fingerprint") != manifest.get("source_fingerprint")
        or stable_hash(source.get("files")) != source.get("fingerprint")
        or [row.get("path") for row in source.get("files", [])] != list(ROBUST_SOURCE_FILES)
    ):
        raise GPURobustExperimentError("robust task-embedded source binding changed")
    recorded_workspace = environment.get("training_workspace_root")
    if isinstance(recorded_workspace, str) and Path(recorded_workspace).is_dir():
        validation_workspace = Path(recorded_workspace).resolve()
    else:
        validation_workspace = _validation_workspace_root(root, config)
    if robust_training_source_payload(validation_workspace) != source:
        raise GPURobustExperimentError(
            "robust validation workspace differs from the task-bound source ledger"
        )
    config["_workspace_root"] = str(validation_workspace)
    recorded_release = environment.get("resolved_release_path")
    if isinstance(recorded_release, str) and Path(recorded_release).is_file():
        config["_release_override"] = str(Path(recorded_release).resolve())
    selected_devices = environment.get("selected_devices")
    if not isinstance(selected_devices, list) or len(selected_devices) != 4:
        raise GPURobustExperimentError("robust selected GPU evidence changed")
    config["_allowed_gpu_ids"] = [int(row["id"]) for row in selected_devices]
    config["_source_fingerprint"] = source["fingerprint"]
    _prepare_runtime_config(config)
    release_path = resolve_robust_gpu_release_path(config)
    data = _load_worker_data(config, release_path)
    array_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    inner_tasks = robust_inner_tasks(config)
    inner_results = _validate_task_results(
        root,
        config,
        data,
        inner_tasks,
        phase="inner",
        source_binding="task_embedded",
        source_fingerprint=source["fingerprint"],
        array_cache=array_cache,
    )
    expected_selection, outer_tasks = robust_selection_and_outer_tasks(
        config, inner_results
    )
    selection_reference = manifest.get("selection")
    if not isinstance(selection_reference, dict):
        raise GPURobustExperimentError("robust selection reference missing")
    selection_path = _safe_run_member(
        root, selection_reference.get("path"), expected="selection/inner_selection.json"
    )
    if sha256_file(selection_path) != selection_reference.get("sha256"):
        raise GPURobustExperimentError("robust selection digest changed")
    _assert_json_close(
        read_json_object(selection_path), expected_selection, context="robust_selection"
    )

    outer_results = _validate_task_results(
        root,
        config,
        data,
        outer_tasks,
        phase="outer",
        source_binding="task_embedded",
        source_fingerprint=source["fingerprint"],
        array_cache=array_cache,
    )
    expected_evaluation = _outer_summary(
        outer_results, run_dir=root, verify_existing=True
    )
    _assert_json_close(
        read_json_object(root / "evaluation" / "summary.json"),
        expected_evaluation,
        context="robust_evaluation",
    )

    final_tasks = robust_final_tasks(config, expected_selection)
    final_results = _validate_task_results(
        root,
        config,
        data,
        final_tasks,
        phase="final",
        source_binding="task_embedded",
        source_fingerprint=source["fingerprint"],
        array_cache=array_cache,
    )
    _validate_group_history(
        root, [*inner_results, *outer_results, *final_results], array_cache
    )
    recipe_reference = manifest.get("training_recipe")
    if not isinstance(recipe_reference, dict):
        raise GPURobustExperimentError("robust training recipe reference missing")
    recipe_path = _safe_run_member(
        root, recipe_reference.get("path"), expected="final/training_recipe.json"
    )
    if sha256_file(recipe_path) != recipe_reference.get("sha256"):
        raise GPURobustExperimentError("robust training recipe digest changed")
    recipe = read_json_object(recipe_path)
    expected_recipe = _training_recipe_payload(
        config, expected_selection, source_fingerprint=source["fingerprint"]
    )
    _assert_json_close(recipe, expected_recipe, context="robust_training_recipe")

    expected_base_model = _expected_final_model_manifest(
        config, expected_selection, final_results, root=root
    )
    expected_model = _robust_model_manifest_payload(
        expected_base_model, expected_selection, recipe_reference
    )
    model_manifest = read_json_object(root / "final" / "model_manifest.json")
    _assert_json_close(model_manifest, expected_model, context="robust_final_model")
    _assert_json_close(
        manifest.get("final_model"), expected_model, context="manifest.robust_final_model"
    )

    validation_reference = manifest.get("validation")
    if _expected_manifest_status == "awaiting_validation":
        if validation_reference is not None:
            raise GPURobustExperimentError(
                "awaiting robust run unexpectedly contains validation evidence"
            )
        expected_summary_status = "awaiting_validation"
    else:
        if not isinstance(validation_reference, dict):
            raise GPURobustExperimentError(
                "finalized robust validation reference missing"
            )
        expected_summary_status = "complete"
    expected_summary = _summary_payload(
        config,
        expected_selection,
        expected_evaluation,
        expected_model,
        recipe_reference,
        status=expected_summary_status,
        validation_reference=(
            validation_reference
            if isinstance(validation_reference, dict)
            else None
        ),
    )
    _assert_json_close(
        read_json_object(root / "summary.json"),
        expected_summary,
        context="robust_summary",
    )

    from .gpu_inference import GPUInferenceError, load_gpu_endpoint_model

    model = load_gpu_endpoint_model(
        root, device="cpu", allow_unpromoted_diagnostic=True
    )
    sample = data.eligible.iloc[:2].loc[:, list(data.feature_names)].to_numpy(dtype=np.float64)
    prediction = model.predict(sample, batch_size=2)
    if prediction.shape != (2, 3) or not np.isfinite(prediction).all():
        raise GPURobustExperimentError("robust final CPU inference smoke failed")
    if expected_selection["promotion_action"] == "retain_v1":
        try:
            load_gpu_endpoint_model(root, device="cpu")
        except GPUInferenceError as exc:
            if "retain_v1 forbids inference" not in str(exc):
                raise GPURobustExperimentError(
                    "unpromoted robust inference failed for the wrong reason"
                ) from exc
        else:
            raise GPURobustExperimentError(
                "unpromoted robust checkpoint bypassed the default inference gate"
            )
    elif _expected_manifest_status == "complete":
        promoted_model = load_gpu_endpoint_model(root, device="cpu")
        if promoted_model.model_fingerprint != model.model_fingerprint:
            raise GPURobustExperimentError(
                "promoted robust default/diagnostic model fingerprints differ"
            )
    gates = {
        "manifest_hash_tree": True,
        "source_bound_in_every_task": True,
        "release_authenticated": True,
        "feature_allowlist_43": True,
        "real_train_only_preprocessing": True,
        "distinct_validation_contexts_20": True,
        "seed_level_validation_fits_360": True,
        "training_only_segment_metadata": True,
        "group_objective_mass": True,
        "prediction_truth_and_metrics_recomputed": True,
        "selection_recomputed_without_outer": True,
        "oof_coverage_7702": True,
        "outer_designated_development_only": True,
        "final_checkpoint_hashes": True,
        "diagnostic_cpu_load_with_explicit_override": True,
        "promotion_inference_policy_enforced": True,
        "confirmatory_sessions_required_10": True,
    }
    result = {
        "status": "ok",
        "validation_level": "semantic_recomputation",
        "contract": ROBUST_GPU_EXPERIMENT_CONTRACT,
        "config_sha256": config_sha,
        "release_sha256": data.release.archive_sha256,
        "source_fingerprint": source["fingerprint"],
        "validated_result_counts": {
            "inner": len(inner_results),
            "outer": len(outer_results),
            "final": len(final_results),
        },
        "distinct_validation_contexts_per_candidate": 20,
        "seed_level_validation_fits": 360,
        "gate_count": len(gates),
        "gates": gates,
        "selected_candidate_id": expected_selection["selected_candidate_id"],
        "robust_j_mm": expected_selection["selected_robust_j_mm"],
        "promotion_action": expected_selection["promotion_action"],
        "model_fingerprint": model.model_fingerprint,
    }
    if _expected_manifest_status != "awaiting_validation":
        assert isinstance(validation_reference, dict)
        validation_path = _safe_run_member(
            root, validation_reference.get("path"), expected="validation.json"
        )
        if sha256_file(validation_path) != validation_reference.get("sha256"):
            raise GPURobustExperimentError("robust validation report digest changed")
        _assert_json_close(
            read_json_object(validation_path),
            result,
            context="robust_validation_report",
        )
    return result


__all__ = [
    "GPURobustExperimentError",
    "ROBUST_MODEL_TRAINING_CONTRACT",
    "ROBUST_SOURCE_FILES",
    "robust_candidate_config_id",
    "robust_final_tasks",
    "robust_inner_tasks",
    "robust_selection_and_outer_tasks",
    "robust_training_source_payload",
    "run_robust_gpu_experiment",
    "run_robust_gpu_smoke",
    "validate_robust_gpu_run",
]
