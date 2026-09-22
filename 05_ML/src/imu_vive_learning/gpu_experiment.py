"""Four-GPU nested group-CV runner for endpoint regression.

The runner keeps inner selection and outer evaluation as separate task phases.
Outer-test tasks do not exist until every required inner result has completed,
which makes the no-test-peeking boundary auditable rather than conventional.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import platform
import queue
import subprocess
import sys
import tarfile
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
from .experiment_models import derive_context_seed
from .experiment_preprocessing import TrainOnlyPreprocessor
from .experiment_release import ReleaseData, verify_and_load_release
from .experiment_splits import build_experiment_assignments, context_assignments
from .features import LABEL_COLUMNS, validate_feature_views
from .gpu_experiment_config import (
    GPU_EXPERIMENT_CONTRACT,
    gpu_config_sha256,
    gpu_model_grid,
    public_gpu_config,
    resolve_gpu_release_path,
    validate_gpu_config,
)


MODEL_NAMES = ("torch_mlp", "xgboost_cuda")
TRAINING_SOURCE_FILES = (
    "configs/endpoint_regression_gpu_v1.yaml",
    "requirements-gpu.lock",
    "requirements-eval.lock",
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
TRUE_COLUMNS = (
    "true_vive_world_x_mm",
    "true_vive_world_y_mm",
    "true_vive_world_z_mm",
)
PREDICTION_COLUMNS = (
    "pred_vive_world_x_mm",
    "pred_vive_world_y_mm",
    "pred_vive_world_z_mm",
)


class GPUExperimentError(RuntimeError):
    """Raised when a GPU experiment violates its execution contract."""


@dataclass(frozen=True)
class _WorkerData:
    release: ReleaseData
    assignments: pd.DataFrame
    eligible: pd.DataFrame
    feature_names: tuple[str, ...]


def gpu_configuration_id(model_name: str, params: Mapping[str, Any]) -> str:
    return stable_hash({"model": str(model_name), "params": dict(params)})[:16]


def _result_contract(config: Mapping[str, Any]) -> str:
    return str(config.get("_result_contract", GPU_EXPERIMENT_CONTRACT))


def _result_config_sha256(config: Mapping[str, Any]) -> str:
    override = config.get("_result_config_sha256")
    return str(override) if override is not None else gpu_config_sha256(dict(config))


def _task_id(
    phase: str,
    scheme: str,
    outer_fold: int,
    inner_fold: int,
    model_name: str,
    config_id: str,
    seed_index: int | None = None,
) -> str:
    parts = [
        phase,
        scheme,
        f"outer{int(outer_fold)}",
        f"inner{int(inner_fold)}",
        model_name,
        config_id,
    ]
    if seed_index is not None:
        parts.append(f"seed{int(seed_index)}")
    return "__".join(parts)


def _round_median(values: Sequence[int]) -> int:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise GPUExperimentError("cannot derive a refit epoch from empty/non-finite values")
    return max(1, int(math.floor(float(np.median(array)) + 0.5)))


def _load_worker_data(config: dict[str, Any], release_path: Path) -> _WorkerData:
    release = verify_and_load_release(
        release_path,
        expected_sha256=str(config["release"]["archive_sha256"]),
    )
    assignments = build_experiment_assignments(release.canonical, release.assignments)
    eligible = release.eligible_real
    feature_names = tuple(release.feature_sets[str(config["task"]["input_view"])])
    if len(feature_names) != int(config["task"]["expected_feature_count"]):
        raise GPUExperimentError("verified release feature count differs from GPU contract")
    validate_feature_views(eligible, {str(config["task"]["input_view"]): feature_names})
    forbidden_tokens = tuple(str(value).lower() for value in config["task"]["inference_forbidden_tokens"])
    leaked = [
        name
        for name in feature_names
        if any(token in name.lower() for token in forbidden_tokens)
    ]
    if leaked:
        raise GPUExperimentError(f"GPU feature allow-list contains forbidden fields: {leaked}")
    return _WorkerData(
        release=release,
        assignments=assignments,
        eligible=eligible.set_index("trial_uid", drop=False),
        feature_names=feature_names,
    )


def _context_arrays(
    data: _WorkerData,
    *,
    scheme: str,
    outer_fold: int,
    inner_fold: int,
) -> dict[str, Any]:
    context = context_assignments(
        data.assignments,
        scheme=scheme,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    evaluation_role = "validation" if int(inner_fold) != -1 else "test"
    train_ids = context.loc[context["role"].eq("train"), "trial_uid"].astype(str).tolist()
    evaluation_ids = context.loc[
        context["role"].eq(evaluation_role), "trial_uid"
    ].astype(str).tolist()
    forbidden_ids = context.loc[~context["role"].eq("train"), "trial_uid"].astype(str).tolist()
    if not train_ids or not evaluation_ids:
        raise GPUExperimentError("split context lacks train or evaluation rows")
    train = data.eligible.loc[train_ids].copy()
    evaluation = data.eligible.loc[evaluation_ids].copy()
    if set(train_ids) & set(forbidden_ids):
        raise GPUExperimentError("held-out ID leaked into a GPU training context")

    X_train_raw = train.loc[:, list(data.feature_names)].to_numpy(dtype=np.float64)
    y_train_raw = train.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
    X_eval_raw = evaluation.loc[:, list(data.feature_names)].to_numpy(dtype=np.float64)
    y_eval_raw = evaluation.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
    split_context = str(context["split_context"].iloc[0])
    preprocessor = TrainOnlyPreprocessor(
        data.feature_names,
        scale_features=True,
        split_context=split_context,
    ).fit_real_train(
        X_train_raw,
        y_train_raw,
        trial_uids=train_ids,
        forbidden_uids=forbidden_ids,
    )
    return {
        "split_context": split_context,
        "train": train,
        "evaluation": evaluation,
        "evaluation_role": evaluation_role,
        "X_train": preprocessor.transform_features(X_train_raw),
        "y_train": preprocessor.transform_targets(y_train_raw),
        "X_eval": preprocessor.transform_features(X_eval_raw),
        "y_eval_raw": y_eval_raw,
        "preprocessor": preprocessor,
        "forbidden_uid_count": len(forbidden_ids),
    }


def _all_real_arrays(data: _WorkerData) -> dict[str, Any]:
    real = (
        data.eligible.reset_index(drop=True)
        .sort_values("trial_uid", kind="mergesort")
        .copy()
    )
    trial_ids = real["trial_uid"].astype(str).tolist()
    X_raw = real.loc[:, list(data.feature_names)].to_numpy(dtype=np.float64)
    y_raw = real.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
    preprocessor = TrainOnlyPreprocessor(
        data.feature_names,
        scale_features=True,
        split_context="all_eligible_real",
    ).fit_real_train(X_raw, y_raw, trial_uids=trial_ids)
    return {
        "split_context": "all_eligible_real",
        "train": real,
        "X_train": preprocessor.transform_features(X_raw),
        "y_train": preprocessor.transform_targets(y_raw),
        "preprocessor": preprocessor,
    }


def _prediction_frame(
    evaluation: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    task: Mapping[str, Any],
) -> pd.DataFrame:
    if y_true.shape != y_pred.shape or y_true.shape[1:] != (3,):
        raise GPUExperimentError("prediction and truth matrices are not aligned XYZ")
    result = evaluation.loc[:, ["trial_uid", "segment_id", "target_id"]].reset_index(drop=True)
    result["scheme"] = str(task["scheme"])
    result["outer_fold"] = int(task["outer_fold"])
    result["inner_fold"] = int(task["inner_fold"])
    result["model"] = str(task["model"])
    result["config_id"] = str(task["config_id"])
    for index, name in enumerate(TRUE_COLUMNS):
        result[name] = y_true[:, index]
    for index, name in enumerate(PREDICTION_COLUMNS):
        result[name] = y_pred[:, index]
    result["error_3d_mm"] = np.linalg.norm(y_pred - y_true, axis=1)
    if not np.isfinite(result[[*TRUE_COLUMNS, *PREDICTION_COLUMNS, "error_3d_mm"]].to_numpy(dtype=np.float64)).all():
        raise GPUExperimentError("prediction evidence contains non-finite values")
    return result


def _build_torch_model(torch: Any, input_count: int, hidden: Sequence[int]) -> Any:
    layers: list[Any] = []
    previous = int(input_count)
    for width in hidden:
        layers.append(torch.nn.Linear(previous, int(width)))
        layers.append(torch.nn.ReLU())
        previous = int(width)
    layers.append(torch.nn.Linear(previous, 3))
    return torch.nn.Sequential(*layers)


def _raw_prediction(
    model: Any,
    X: Any,
    *,
    torch: Any,
    preprocessor: TrainOnlyPreprocessor,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        scaled = model(X).detach().cpu().numpy().astype(np.float64)
    result = preprocessor.inverse_targets(scaled)
    if not np.isfinite(result).all():
        raise GPUExperimentError("PyTorch prediction contains non-finite values")
    return result


def _train_torch(
    arrays: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    config: Mapping[str, Any],
    return_state: bool = False,
) -> dict[str, Any]:
    import torch

    model_config = config["models"]["torch_mlp"]
    params = dict(task["params"])
    hidden = tuple(int(value) for value in params["hidden_layer_sizes"])
    batch_size = int(model_config["batch_size"])
    learning_rate = float(model_config["learning_rate"])
    history_interval = int(config["monitoring"]["history_interval_epochs"])
    default_epochs = int(task.get("epoch_budget_override", model_config["epoch_budget"]))
    seed_count = int(task.get("seed_count_override", model_config["ensemble_seed_count"]))
    seed_indices = [int(value) for value in task.get("seed_indices", range(seed_count))]
    select_checkpoint = str(task["phase"]) in {"inner", "smoke"}

    device = torch.device("cuda:0")
    X_train = torch.as_tensor(np.asarray(arrays["X_train"], dtype=np.float32), device=device)
    y_train = torch.as_tensor(np.asarray(arrays["y_train"], dtype=np.float32), device=device)
    X_eval = None
    if "X_eval" in arrays:
        X_eval = torch.as_tensor(np.asarray(arrays["X_eval"], dtype=np.float32), device=device)
    preprocessor = arrays["preprocessor"]
    train_truth_raw = preprocessor.inverse_targets(np.asarray(arrays["y_train"], dtype=np.float64))

    predictions: list[np.ndarray] = []
    seed_summaries: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    states: dict[int, dict[str, np.ndarray]] = {}
    for seed_index in seed_indices:
        epochs = int(task.get("epochs_per_seed", {}).get(str(seed_index), default_epochs))
        if epochs <= 0 or epochs > int(model_config["epoch_budget"]):
            raise GPUExperimentError(f"invalid MLP epoch budget for seed {seed_index}: {epochs}")
        model_seed = derive_context_seed(
            int(config["contract"]["base_seed"]),
            "gpu_model_seed",
            str(arrays["split_context"]),
            seed_index,
        )
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)
        model = _build_torch_model(torch, X_train.shape[1], hidden).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(params["weight_decay"]),
        )
        order_generator = torch.Generator(device="cpu")
        order_generator.manual_seed(
            derive_context_seed(model_seed, "epoch_shuffle")
        )
        best_score = math.inf
        best_epoch = 0
        best_state: dict[str, Any] | None = None
        final_loss = math.nan
        started = time.perf_counter()

        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(X_train), generator=order_generator)
            loss_mass = 0.0
            row_mass = 0
            for start in range(0, len(X_train), batch_size):
                indices = order[start : start + batch_size].to(device=device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(X_train[indices])
                loss = torch.mean(torch.square(prediction - y_train[indices]))
                if not bool(torch.isfinite(loss).item()):
                    raise GPUExperimentError("non-finite PyTorch training loss")
                loss.backward()
                optimizer.step()
                count = int(len(indices))
                loss_mass += float(loss.detach().item()) * count
                row_mass += count
            final_loss = loss_mass / row_mass

            validation_prediction: np.ndarray | None = None
            validation_score: float | None = None
            if select_checkpoint:
                if X_eval is None:
                    raise GPUExperimentError("inner checkpoint selection lacks validation data")
                validation_prediction = _raw_prediction(
                    model, X_eval, torch=torch, preprocessor=preprocessor
                )
                validation_score = float(
                    regression_metrics(arrays["y_eval_raw"], validation_prediction)[
                        "rmse_3d_mm"
                    ]
                )
                if validation_score < best_score:
                    best_score = validation_score
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }

            if epoch == 1 or epoch % history_interval == 0 or epoch == epochs:
                train_prediction = _raw_prediction(
                    model, X_train, torch=torch, preprocessor=preprocessor
                )
                train_score = float(
                    regression_metrics(train_truth_raw, train_prediction)["rmse_3d_mm"]
                )
                row: dict[str, Any] = {
                    "task_id": str(task["task_id"]),
                    "seed_index": seed_index,
                    "model_seed": model_seed,
                    "epoch": epoch,
                    "scaled_mse_loss": float(final_loss),
                    "train_rmse_3d_mm": train_score,
                }
                if select_checkpoint:
                    assert validation_score is not None
                    row["validation_rmse_3d_mm"] = validation_score
                    row["train_validation_gap_mm"] = validation_score - train_score
                    row["best_validation_rmse_3d_mm"] = best_score
                    row["best_epoch_so_far"] = best_epoch
                history_rows.append(row)

        if select_checkpoint:
            if best_state is None or best_epoch <= 0:
                raise GPUExperimentError("MLP checkpoint selection produced no finite state")
            model.load_state_dict(best_state)
        evaluation_prediction: np.ndarray | None = None
        if X_eval is not None:
            evaluation_prediction = _raw_prediction(
                model, X_eval, torch=torch, preprocessor=preprocessor
            )
            predictions.append(evaluation_prediction)
        final_train_prediction = _raw_prediction(
            model, X_train, torch=torch, preprocessor=preprocessor
        )
        summary: dict[str, Any] = {
            "seed_index": seed_index,
            "model_seed": model_seed,
            "trained_epochs": epochs,
            "selected_epoch": best_epoch if select_checkpoint else epochs,
            "best_validation_rmse_3d_mm": best_score if select_checkpoint else None,
            "final_scaled_mse_loss": float(final_loss),
            "selected_train_rmse_3d_mm": float(
                regression_metrics(train_truth_raw, final_train_prediction)["rmse_3d_mm"]
            ),
            "fit_seconds": float(time.perf_counter() - started),
        }
        if evaluation_prediction is not None:
            evaluation_metrics = regression_metrics(
                arrays["y_eval_raw"], evaluation_prediction
            )
            summary["evaluation_rmse_3d_mm"] = float(
                evaluation_metrics["rmse_3d_mm"]
            )
            summary["evaluation_p95_3d_error_mm"] = float(
                evaluation_metrics["p95_3d_error_mm"]
            )
        seed_summaries.append(summary)
        if return_state:
            states[seed_index] = {
                name: value.detach().cpu().numpy()
                for name, value in model.state_dict().items()
            }
        del model, optimizer
        torch.cuda.empty_cache()

    ensemble = None
    metrics = None
    if predictions:
        ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
        metrics = regression_metrics(arrays["y_eval_raw"], ensemble)
    return {
        "ensemble_prediction": ensemble,
        "metrics": metrics,
        "seed_summaries": seed_summaries,
        "history": history_rows,
        "states": states,
    }


def _train_xgboost(
    arrays: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from xgboost import XGBRegressor

    model_config = config["models"]["xgboost_cuda"]
    params = dict(task["params"])
    model_seed = derive_context_seed(
        int(config["contract"]["base_seed"]),
        "gpu_model_seed",
        str(arrays["split_context"]),
        0,
    )
    started = time.perf_counter()
    predictions: list[np.ndarray] = []
    for axis in range(3):
        estimator = XGBRegressor(
            n_estimators=int(model_config["n_estimators"]),
            learning_rate=float(model_config["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            reg_lambda=float(params["reg_lambda"]),
            subsample=float(model_config["subsample"]),
            colsample_bytree=float(model_config["colsample_bytree"]),
            objective="reg:squarederror",
            tree_method=str(model_config["tree_method"]),
            device="cuda:0",
            n_jobs=int(model_config["n_jobs"]),
            random_state=derive_context_seed(model_seed, "xgb_axis", axis),
            verbosity=0,
        )
        estimator.fit(arrays["X_train"], arrays["y_train"][:, axis])
        predictions.append(estimator.predict(arrays["X_eval"]))
    scaled_prediction = np.column_stack(predictions).astype(np.float64)
    prediction = arrays["preprocessor"].inverse_targets(scaled_prediction)
    metrics = regression_metrics(arrays["y_eval_raw"], prediction)
    return {
        "ensemble_prediction": prediction,
        "metrics": metrics,
        "seed_summaries": [
            {
                "seed_index": 0,
                "model_seed": model_seed,
                "trained_epochs": int(model_config["n_estimators"]),
                "selected_epoch": int(model_config["n_estimators"]),
                "best_validation_rmse_3d_mm": float(metrics["rmse_3d_mm"]),
                "fit_seconds": float(time.perf_counter() - started),
            }
        ],
        "history": [],
        "states": {},
    }


def _result_path(run_dir: Path, task: Mapping[str, Any]) -> Path:
    return run_dir / str(task["phase"]) / "results" / f"{task['task_id']}.json"


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **dict(arrays))
    os.replace(temporary, path)
    return path


def _execute_task(
    data: _WorkerData,
    *,
    task: dict[str, Any],
    config: dict[str, Any],
    run_dir: Path,
    gpu_id: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    phase = str(task["phase"])
    if phase == "final":
        arrays = _all_real_arrays(data)
    else:
        arrays = _context_arrays(
            data,
            scheme=str(task["scheme"]),
            outer_fold=int(task["outer_fold"]),
            inner_fold=int(task["inner_fold"]),
        )

    if str(task["model"]) == "torch_mlp":
        if "training_recipe" in task:
            from .gpu_robust_training import train_robust_torch

            trained = train_robust_torch(
                arrays,
                task=task,
                config=config,
                return_state=phase == "final",
            )
        else:
            trained = _train_torch(
                arrays,
                task=task,
                config=config,
                return_state=phase == "final",
            )
    elif str(task["model"]) == "xgboost_cuda":
        if phase == "final":
            raise GPUExperimentError("the reference XGBoost family is not a final model")
        trained = _train_xgboost(arrays, task=task, config=config)
    else:
        raise GPUExperimentError(f"unknown GPU task model: {task['model']!r}")

    result: dict[str, Any] = {
        "contract": _result_contract(config),
        "config_sha256": _result_config_sha256(config),
        "source_fingerprint": config.get("_source_fingerprint"),
        "task": task,
        "task_hash": stable_hash(task),
        "task_id": str(task["task_id"]),
        "phase": phase,
        "gpu_physical_id": int(gpu_id),
        "gpu_visible_device": 0,
        "split_context": str(arrays["split_context"]),
        "feature_count": len(data.feature_names),
        "feature_names": list(data.feature_names),
        "real_train_rows": int(len(arrays["train"])),
        "evaluation_rows": int(len(arrays.get("evaluation", []))),
        "evaluation_role": arrays.get("evaluation_role"),
        "preprocessor": arrays["preprocessor"].metadata(),
        "metrics": trained["metrics"],
        "seed_summaries": trained["seed_summaries"],
        "elapsed_seconds": float(time.perf_counter() - started),
    }

    if trained["ensemble_prediction"] is not None:
        prediction_frame = _prediction_frame(
            arrays["evaluation"],
            arrays["y_eval_raw"],
            trained["ensemble_prediction"],
            task=task,
        )
        prediction_path = run_dir / phase / "predictions" / f"{task['task_id']}.parquet"
        atomic_write_parquet(prediction_path, prediction_frame)
        result["prediction_file"] = {
            "path": _relative(prediction_path, run_dir),
            "sha256": sha256_file(prediction_path),
            "rows": len(prediction_frame),
        }

    history = trained["history"]
    if history:
        history_frame = pd.DataFrame(history)
        history_path = run_dir / phase / "history" / f"{task['task_id']}.parquet"
        atomic_write_parquet(history_path, history_frame)
        result["history_file"] = {
            "path": _relative(history_path, run_dir),
            "sha256": sha256_file(history_path),
            "rows": len(history_frame),
        }

    group_history = trained.get("group_history", [])
    if group_history:
        group_history_frame = pd.DataFrame(group_history)
        group_history_path = (
            run_dir / phase / "group_history" / f"{task['task_id']}.parquet"
        )
        atomic_write_parquet(group_history_path, group_history_frame)
        result["group_history_file"] = {
            "path": _relative(group_history_path, run_dir),
            "sha256": sha256_file(group_history_path),
            "rows": len(group_history_frame),
        }

    states: dict[int, dict[str, np.ndarray]] = trained["states"]
    if states:
        if phase != "final" or len(states) != 1:
            raise GPUExperimentError("only one final seed state may be emitted per task")
        seed_index, state = next(iter(states.items()))
        checkpoint_path = run_dir / "final" / "checkpoints" / f"torch_mlp_seed{seed_index}.npz"
        _atomic_write_npz(checkpoint_path, state)
        result["checkpoint_file"] = {
            "path": _relative(checkpoint_path, run_dir),
            "sha256": sha256_file(checkpoint_path),
            "tensor_count": len(state),
        }

    result_path = _result_path(run_dir, task)
    atomic_write_json(result_path, result)
    return result


def _worker_runtime(gpu_id: int, config: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(
        config["runtime"]["cublas_workspace_config"]
    )
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import torch
    import xgboost

    if torch.__version__ != str(config["runtime"]["torch_version"]):
        raise GPUExperimentError(
            f"worker torch version changed: {torch.__version__!r}"
        )
    if torch.version.cuda != str(config["runtime"]["cuda_runtime"]):
        raise GPUExperimentError(
            f"worker CUDA runtime changed: {torch.version.cuda!r}"
        )
    if xgboost.__version__ != str(config["runtime"]["xgboost_version"]):
        raise GPUExperimentError(
            f"worker XGBoost version changed: {xgboost.__version__!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise GPUExperimentError(
            f"worker {gpu_id} must see exactly one CUDA device"
        )
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(
        bool(config["models"]["torch_mlp"]["deterministic_algorithms"])
    )
    properties = torch.cuda.get_device_properties(0)
    return {
        "physical_id": int(gpu_id),
        "visible_id": 0,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "xgboost": xgboost.__version__,
    }


def _worker_loop(
    gpu_id: int,
    config: dict[str, Any],
    release_path: str,
    run_dir: str,
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        from threadpoolctl import threadpool_limits

        native_thread_limiter = threadpool_limits(limits=1)
        runtime = _worker_runtime(gpu_id, config)
        _verify_training_source_fingerprint(config)
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
                        "phase": task["phase"],
                        "model": task["model"],
                        "scheme": task["scheme"],
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
        native_thread_limiter.restore_original_limits()
    except BaseException as exc:
        result_queue.put(
            {
                "event": "worker_error",
                "gpu": gpu_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _task_cost(task: Mapping[str, Any]) -> float:
    if task["model"] == "xgboost_cuda":
        return 160.0
    hidden = task["params"]["hidden_layer_sizes"]
    width_factor = 1.0 if hidden == [64, 64] else 1.35
    epochs = float(task.get("epoch_budget_override", 400))
    seed_count = len(task.get("seed_indices", [])) or int(
        task.get("seed_count_override", 3)
    )
    if "epochs_per_seed" in task:
        epochs = float(sum(int(value) for value in task["epochs_per_seed"].values()))
        seed_count = 1
    return width_factor * epochs * seed_count


def _balanced_bins(tasks: Sequence[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    bins: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for task in sorted(tasks, key=lambda item: (-_task_cost(item), str(item["task_id"]))):
        index = min(range(count), key=lambda value: (loads[value], value))
        bins[index].append(task)
        loads[index] += _task_cost(task)
    return bins


def _read_completed_task(
    run_dir: Path,
    task: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = _result_path(run_dir, task)
    if not path.is_file():
        return None
    result = read_json_object(path)
    if result.get("contract") != _result_contract(config):
        raise GPUExperimentError(f"resume contract changed: {task['task_id']}")
    if result.get("task") != dict(task) or result.get("task_hash") != stable_hash(task):
        raise GPUExperimentError(f"resume task identity changed: {task['task_id']}")
    if result.get("config_sha256") != _result_config_sha256(config):
        raise GPUExperimentError(f"resume config changed: {task['task_id']}")
    expected_source = config.get("_source_fingerprint")
    if expected_source is not None and result.get("source_fingerprint") != expected_source:
        raise GPUExperimentError(f"resume source fingerprint changed: {task['task_id']}")
    expected_artifacts = {
        "prediction_file": (
            None
            if task["phase"] == "final"
            else f"{task['phase']}/predictions/{task['task_id']}.parquet"
        ),
        "history_file": (
            f"{task['phase']}/history/{task['task_id']}.parquet"
            if task["model"] == "torch_mlp"
            else None
        ),
        "group_history_file": (
            f"{task['phase']}/group_history/{task['task_id']}.parquet"
            if "training_recipe" in task
            else None
        ),
        "checkpoint_file": (
            f"final/checkpoints/torch_mlp_seed{int(task['seed_indices'][0])}.npz"
            if task["phase"] == "final"
            else None
        ),
    }
    for key, expected_relative in expected_artifacts.items():
        entry = result.get(key)
        if expected_relative is None:
            if entry is not None:
                raise GPUExperimentError(
                    f"resume contains an unexpected artifact for {task['task_id']}: {key}"
                )
            continue
        if not isinstance(entry, dict):
            raise GPUExperimentError(
                f"resume artifact is missing for {task['task_id']}: {key}"
            )
        artifact = _safe_run_member(
            run_dir, entry.get("path"), expected=expected_relative
        )
        if sha256_file(artifact) != str(entry.get("sha256")):
            raise GPUExperimentError(
                f"resume artifact is missing or changed for {task['task_id']}: {key}"
            )
    return result


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True), flush=True)


def _training_direction_warnings(
    result: Mapping[str, Any],
    *,
    run_dir: Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate the frozen validation-regression/staleness warning rules."""

    if result.get("phase") not in {"inner", "smoke"} or result.get("history_file") is None:
        return []
    reference = result["history_file"]
    history = pd.read_parquet(run_dir / str(reference["path"]))
    ratio_limit = float(
        config["monitoring"]["warning_current_to_best_validation_ratio"]
    )
    stale_limit = int(config["monitoring"]["warning_stale_epochs"])
    warnings: list[dict[str, Any]] = []
    for seed_index, rows in history.groupby("seed_index", sort=True):
        last = rows.sort_values("epoch", kind="mergesort").iloc[-1]
        current = float(last["validation_rmse_3d_mm"])
        best = float(last["best_validation_rmse_3d_mm"])
        epoch = int(last["epoch"])
        best_epoch = int(last["best_epoch_so_far"])
        ratio = current / best
        if ratio > ratio_limit:
            warnings.append(
                {
                    "kind": "validation_regression",
                    "seed_index": int(seed_index),
                    "epoch": epoch,
                    "current_to_best_ratio": ratio,
                    "limit": ratio_limit,
                }
            )
        stale_epochs = epoch - best_epoch
        if stale_epochs >= stale_limit:
            warnings.append(
                {
                    "kind": "stale_validation_best",
                    "seed_index": int(seed_index),
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs,
                    "limit": stale_limit,
                }
            )
    return warnings


def _run_task_batch(
    tasks: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    release_path: Path,
    run_dir: Path,
    gpu_ids: Sequence[int],
) -> list[dict[str, Any]]:
    _verify_training_source_fingerprint(config)
    results: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for task in tasks:
        completed = _read_completed_task(run_dir, task, config=config)
        if completed is None:
            pending.append(task)
        else:
            results[str(task["task_id"])] = completed
    if not pending:
        _emit({"event": "phase_resume_complete", "tasks": len(tasks)})
        return [results[str(task["task_id"])] for task in tasks]

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    task_queues = [context.Queue() for _ in gpu_ids]
    workers = [
        context.Process(
            target=_worker_loop,
            args=(
                int(gpu_id),
                config,
                str(release_path),
                str(run_dir),
                task_queues[index],
                result_queue,
            ),
            name=f"gpu-experiment-{gpu_id}",
        )
        for index, gpu_id in enumerate(gpu_ids)
    ]
    bins = _balanced_bins(pending, len(gpu_ids))
    for worker in workers:
        worker.start()
    for task_queue, assigned in zip(task_queues, bins, strict=True):
        for task in assigned:
            task_queue.put(task)
        task_queue.put(None)

    completed_count = 0
    ready_count = 0
    done_count = 0
    failure: dict[str, Any] | None = None
    last_progress = time.monotonic()
    try:
        while completed_count < len(pending) and failure is None:
            try:
                event = result_queue.get(timeout=2.0)
            except queue.Empty:
                dead = [
                    worker
                    for worker in workers
                    if not worker.is_alive() and worker.exitcode not in (None, 0)
                ]
                if dead:
                    failure = {
                        "event": "worker_process_exit",
                        "workers": [(worker.name, worker.exitcode) for worker in dead],
                    }
                elif all(not worker.is_alive() for worker in workers):
                    failure = {
                        "event": "workers_exited_before_all_task_events",
                        "completed": completed_count,
                        "expected": len(pending),
                    }
                elif time.monotonic() - last_progress >= 30.0:
                    _emit(
                        {
                            "event": "progress",
                            "completed": completed_count,
                            "pending": len(pending) - completed_count,
                            "total": len(pending),
                        }
                    )
                    last_progress = time.monotonic()
                continue
            event_name = str(event.get("event"))
            if event_name == "worker_ready":
                ready_count += 1
                _emit(event)
            elif event_name == "task_complete":
                completed_count += 1
                task_id = str(event["task_id"])
                task = next(item for item in pending if item["task_id"] == task_id)
                completed = _read_completed_task(run_dir, task, config=config)
                if completed is None:
                    raise GPUExperimentError(f"worker reported an incomplete task: {task_id}")
                results[task_id] = completed
                direction_warnings = _training_direction_warnings(
                    completed, run_dir=run_dir, config=config
                )
                should_report_progress = (
                    completed_count == 1
                    or completed_count % 8 == 0
                    or completed_count == len(pending)
                )
                if direction_warnings or should_report_progress:
                    _emit(
                        {
                            **event,
                            "completed": completed_count,
                            "total": len(pending),
                            "direction_warnings": direction_warnings,
                        }
                    )
                last_progress = time.monotonic()
            elif event_name == "worker_done":
                done_count += 1
            elif event_name in {"worker_error", "task_error"}:
                failure = event
                _emit(event)
            else:
                failure = {"event": "unknown_worker_event", "payload": event}
        if failure is not None:
            raise GPUExperimentError(json.dumps(failure, ensure_ascii=False))
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

    if completed_count != len(pending):
        raise GPUExperimentError(
            f"GPU task batch incomplete: {completed_count}/{len(pending)}"
        )
    _verify_training_source_fingerprint(config)
    return [results[str(task["task_id"])] for task in tasks]


def _inner_tasks(config: dict[str, Any], *, smoke: bool = False) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    split_sections = (
        [config["splits"]["primary"]]
        if smoke
        else [config["splits"]["primary"], config["splits"]["secondary"]]
    )
    for split in split_sections:
        scheme = str(split["name"])
        outer_groups = [int(value) for value in split["groups"]]
        selected_outer = outer_groups[:1] if smoke else outer_groups
        for outer_fold in selected_outer:
            selected_inner = [value for value in outer_groups if value != outer_fold]
            if smoke:
                selected_inner = selected_inner[:1]
            for inner_fold in selected_inner:
                for model_name in MODEL_NAMES:
                    for params in gpu_model_grid(config, model_name):
                        config_id = gpu_configuration_id(model_name, params)
                        phase = "smoke" if smoke else "inner"
                        task: dict[str, Any] = {
                            "phase": phase,
                            "scheme": scheme,
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "model": model_name,
                            "params": params,
                            "config_id": config_id,
                        }
                        if smoke:
                            task["epoch_budget_override"] = 20
                            task["seed_count_override"] = 1
                            task["smoke_contract_eligible"] = False
                        task["task_id"] = _task_id(
                            phase,
                            scheme,
                            outer_fold,
                            inner_fold,
                            model_name,
                            config_id,
                        )
                        tasks.append(task)
    expected = 12 if smoke else 480
    if len(tasks) != expected:
        raise GPUExperimentError(
            f"inner GPU task count changed: expected {expected}, received {len(tasks)}"
        )
    return tasks


def _group_results(
    results: Sequence[dict[str, Any]],
    *,
    scheme: str,
    outer_fold: int | None,
    model_name: str,
    config_id: str,
) -> list[dict[str, Any]]:
    selected = [
        result
        for result in results
        if result["task"]["scheme"] == scheme
        and (outer_fold is None or int(result["task"]["outer_fold"]) == int(outer_fold))
        and result["task"]["model"] == model_name
        and result["task"]["config_id"] == config_id
    ]
    return sorted(
        selected,
        key=lambda result: (
            int(result["task"]["outer_fold"]),
            int(result["task"]["inner_fold"]),
        ),
    )


def _select_config(
    config: dict[str, Any],
    results: Sequence[dict[str, Any]],
    *,
    scheme: str,
    outer_fold: int | None,
    model_name: str,
    expected_contexts: int,
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    for priority, params in enumerate(gpu_model_grid(config, model_name)):
        config_id = gpu_configuration_id(model_name, params)
        rows = _group_results(
            results,
            scheme=scheme,
            outer_fold=outer_fold,
            model_name=model_name,
            config_id=config_id,
        )
        if len(rows) != expected_contexts:
            raise GPUExperimentError(
                f"selection context count changed for {scheme}/{outer_fold}/{model_name}/{config_id}: "
                f"{len(rows)} != {expected_contexts}"
            )
        scores = [float(row["metrics"]["rmse_3d_mm"]) for row in rows]
        if not np.isfinite(scores).all():
            raise GPUExperimentError("inner selection contains non-finite scores")
        trace.append(
            {
                "model": model_name,
                "config_id": config_id,
                "params": params,
                "tie_priority": priority,
                "context_count": len(rows),
                "inner_rmse_3d_mm": scores,
                "mean_inner_rmse_3d_mm": float(np.mean(scores)),
                "max_inner_rmse_3d_mm": float(np.max(scores)),
            }
        )
    selected = min(
        trace,
        key=lambda row: (float(row["mean_inner_rmse_3d_mm"]), int(row["tie_priority"])),
    )
    return {
        "scheme": scheme,
        "outer_fold": outer_fold,
        "model": model_name,
        "selection_metric": "equal_validation_group_mean_rmse_3d_mm",
        "selected_config_id": selected["config_id"],
        "selected_params": selected["params"],
        "selected_score_mm": selected["mean_inner_rmse_3d_mm"],
        "trace": trace,
    }


def _refit_epochs(
    selected: Mapping[str, Any],
    inner_results: Sequence[dict[str, Any]],
    *,
    seed_count: int,
) -> dict[str, int]:
    rows = _group_results(
        inner_results,
        scheme=str(selected["scheme"]),
        outer_fold=(
            None if selected["outer_fold"] is None else int(selected["outer_fold"])
        ),
        model_name="torch_mlp",
        config_id=str(selected["selected_config_id"]),
    )
    result: dict[str, int] = {}
    for seed_index in range(seed_count):
        epochs: list[int] = []
        for row in rows:
            matches = [
                summary
                for summary in row["seed_summaries"]
                if int(summary["seed_index"]) == seed_index
            ]
            if len(matches) != 1:
                raise GPUExperimentError("inner MLP seed evidence is incomplete")
            epochs.append(int(matches[0]["selected_epoch"]))
        result[str(seed_index)] = _round_median(epochs)
    return result


def _selection_and_outer_tasks(
    config: dict[str, Any], inner_results: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outer_selections: list[dict[str, Any]] = []
    outer_tasks: list[dict[str, Any]] = []
    seed_count = int(config["models"]["torch_mlp"]["ensemble_seed_count"])
    for split in (config["splits"]["primary"], config["splits"]["secondary"]):
        scheme = str(split["name"])
        for outer_fold in [int(value) for value in split["groups"]]:
            for model_name in MODEL_NAMES:
                selected = _select_config(
                    config,
                    inner_results,
                    scheme=scheme,
                    outer_fold=outer_fold,
                    model_name=model_name,
                    expected_contexts=4,
                )
                if model_name == "torch_mlp":
                    selected["refit_epochs_per_seed"] = _refit_epochs(
                        selected, inner_results, seed_count=seed_count
                    )
                outer_selections.append(selected)
                config_id = str(selected["selected_config_id"])
                task: dict[str, Any] = {
                    "phase": "outer",
                    "scheme": scheme,
                    "outer_fold": outer_fold,
                    "inner_fold": -1,
                    "model": model_name,
                    "params": selected["selected_params"],
                    "config_id": config_id,
                }
                if model_name == "torch_mlp":
                    task["epochs_per_seed"] = selected["refit_epochs_per_seed"]
                task["task_id"] = _task_id(
                    "outer",
                    scheme,
                    outer_fold,
                    -1,
                    model_name,
                    config_id,
                )
                outer_tasks.append(task)

    global_selection = _select_config(
        config,
        inner_results,
        scheme=str(config["splits"]["primary"]["name"]),
        outer_fold=None,
        model_name="torch_mlp",
        expected_contexts=20,
    )
    global_selection["refit_epochs_per_seed"] = _refit_epochs(
        global_selection, inner_results, seed_count=seed_count
    )
    selection = {
        "outer_test_used": False,
        "outer_selections": outer_selections,
        "final_global_segment_inner_selection": global_selection,
    }
    if len(outer_tasks) != 20:
        raise GPUExperimentError(f"outer task count changed: {len(outer_tasks)} != 20")
    return selection, outer_tasks


def _final_tasks(config: dict[str, Any], selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = selection["final_global_segment_inner_selection"]
    seed_count = int(config["models"]["torch_mlp"]["ensemble_seed_count"])
    tasks: list[dict[str, Any]] = []
    for seed_index in range(seed_count):
        config_id = str(selected["selected_config_id"])
        task = {
            "phase": "final",
            "scheme": "segment_loso",
            "outer_fold": -1,
            "inner_fold": -1,
            "model": "torch_mlp",
            "params": selected["selected_params"],
            "config_id": config_id,
            "seed_indices": [seed_index],
            "epochs_per_seed": {
                str(seed_index): int(selected["refit_epochs_per_seed"][str(seed_index)])
            },
        }
        task["task_id"] = _task_id(
            "final",
            "segment_loso",
            -1,
            -1,
            "torch_mlp",
            config_id,
            seed_index,
        )
        tasks.append(task)
    return tasks


def training_source_payload(workspace_root: str | Path) -> dict[str, Any]:
    """Hash the exact source bytes that can affect a future GPU run."""

    root = Path(workspace_root).resolve()
    files: list[dict[str, Any]] = []
    for relative in TRAINING_SOURCE_FILES:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise GPUExperimentError(f"training source escapes workspace: {relative}") from exc
        if not path.is_file():
            raise GPUExperimentError(f"training source file is missing: {relative}")
        files.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {"files": files, "fingerprint": stable_hash(files)}


def _verify_training_source_fingerprint(config: Mapping[str, Any]) -> None:
    """Fail if source bytes change between parent/worker/phase boundaries."""

    expected = config.get("_source_fingerprint")
    if expected is None:
        return
    workspace_root = config.get("_workspace_root")
    if not isinstance(workspace_root, str):
        raise GPUExperimentError("training-source verification lacks workspace root")
    actual = training_source_payload(workspace_root)["fingerprint"]
    if actual != expected:
        raise GPUExperimentError(
            "training source changed after run identity was frozen: "
            f"{actual} != {expected}"
        )


def _git_payload(workspace_root: Path) -> dict[str, Any]:
    def run(arguments: list[str]) -> str:
        try:
            completed = subprocess.run(
                arguments,
                cwd=workspace_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GPUExperimentError("cannot record Git provenance") from exc
        return completed.stdout.strip()

    status = run(["git", "status", "--porcelain", "--untracked-files=all"])
    return {
        "revision": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_lines": status.splitlines(),
        "diff_sha256": stable_hash(
            {
                "tracked_diff": run(["git", "diff", "--no-ext-diff"]),
                "staged_diff": run(["git", "diff", "--cached", "--no-ext-diff"]),
                "status": status,
            }
        ),
    }


def gpu_environment(gpu_ids: Sequence[int], config: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    import xgboost

    if torch.__version__ != str(config["runtime"]["torch_version"]):
        raise GPUExperimentError(
            f"torch version differs from contract: {torch.__version__!r}"
        )
    if torch.version.cuda != str(config["runtime"]["cuda_runtime"]):
        raise GPUExperimentError(
            f"torch CUDA runtime differs from contract: {torch.version.cuda!r}"
        )
    if xgboost.__version__ != str(config["runtime"]["xgboost_version"]):
        raise GPUExperimentError(
            f"XGBoost version differs from contract: {xgboost.__version__!r}"
        )
    if not torch.cuda.is_available():
        raise GPUExperimentError("PyTorch cannot access CUDA")
    device_count = torch.cuda.device_count()
    devices: list[dict[str, Any]] = []
    for gpu_id in gpu_ids:
        if int(gpu_id) < 0 or int(gpu_id) >= device_count:
            raise GPUExperimentError(
                f"requested GPU {gpu_id} is outside visible range 0..{device_count - 1}"
            )
        properties = torch.cuda.get_device_properties(int(gpu_id))
        devices.append(
            {
                "id": int(gpu_id),
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [int(properties.major), int(properties.minor)],
            }
        )
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "xgboost": xgboost.__version__,
        "cuda_available": True,
        "visible_device_count": device_count,
        "selected_devices": devices,
    }


def _outer_evaluation_summary(
    config: dict[str, Any],
    outer_results: Sequence[dict[str, Any]],
    *,
    run_dir: Path,
    verify_existing: bool = False,
) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    metric_names = (
        "rmse_3d_mm",
        "mean_3d_error_mm",
        "median_3d_error_mm",
        "p90_3d_error_mm",
        "p95_3d_error_mm",
    )
    for split in (config["splits"]["primary"], config["splits"]["secondary"]):
        scheme = str(split["name"])
        expected_groups = [int(value) for value in split["groups"]]
        for model_name in MODEL_NAMES:
            rows = sorted(
                [
                    result
                    for result in outer_results
                    if result["task"]["scheme"] == scheme
                    and result["task"]["model"] == model_name
                ],
                key=lambda result: int(result["task"]["outer_fold"]),
            )
            actual_groups = [int(row["task"]["outer_fold"]) for row in rows]
            if actual_groups != expected_groups:
                raise GPUExperimentError(
                    f"outer result groups changed for {scheme}/{model_name}: {actual_groups}"
                )
            prediction_frames = [
                pd.read_parquet(run_dir / str(row["prediction_file"]["path"]))
                for row in rows
            ]
            oof = pd.concat(prediction_frames, ignore_index=True)
            oof.sort_values("trial_uid", kind="mergesort", inplace=True)
            oof.reset_index(drop=True, inplace=True)
            if len(oof) != 7702 or oof["trial_uid"].astype(str).duplicated().any():
                raise GPUExperimentError(
                    f"outer OOF coverage changed for {scheme}/{model_name}"
                )
            truth = oof.loc[:, list(TRUE_COLUMNS)].to_numpy(dtype=np.float64)
            prediction = oof.loc[:, list(PREDICTION_COLUMNS)].to_numpy(dtype=np.float64)
            pooled = regression_metrics(truth, prediction)
            macro = {
                name: float(np.mean([float(row["metrics"][name]) for row in rows]))
                for name in metric_names
            }
            per_group: list[dict[str, Any]] = []
            for row in rows:
                seed_scores = [
                    float(seed["evaluation_rmse_3d_mm"])
                    for seed in row["seed_summaries"]
                    if seed.get("evaluation_rmse_3d_mm") is not None
                ]
                per_group.append(
                    {
                        "outer_fold": int(row["task"]["outer_fold"]),
                        "config_id": str(row["task"]["config_id"]),
                        "params": row["task"]["params"],
                        "row_count": int(row["evaluation_rows"]),
                        **{name: float(row["metrics"][name]) for name in metric_names},
                        "seed_rmse_3d_mm": seed_scores,
                        "seed_rmse_std_mm": (
                            0.0 if len(seed_scores) <= 1 else float(np.std(seed_scores, ddof=0))
                        ),
                    }
                )
            oof_path = run_dir / "evaluation" / f"{scheme}__{model_name}__oof.parquet"
            if verify_existing:
                if not oof_path.is_file():
                    raise GPUExperimentError(
                        f"outer OOF artifact is missing for {scheme}/{model_name}"
                    )
                recorded_oof = pd.read_parquet(oof_path)
                if not recorded_oof.equals(oof):
                    raise GPUExperimentError(
                        f"outer OOF content differs from fold predictions for "
                        f"{scheme}/{model_name}"
                    )
            else:
                atomic_write_parquet(oof_path, oof)
            evaluations.append(
                {
                    "scheme": scheme,
                    "role": "primary" if split is config["splits"]["primary"] else "secondary",
                    "model": model_name,
                    "outer_group_count": len(rows),
                    "oof_rows": len(oof),
                    "macro_equal_group": macro,
                    "pooled": pooled,
                    "per_group": per_group,
                    "oof_prediction_file": {
                        "path": _relative(oof_path, run_dir),
                        "sha256": sha256_file(oof_path),
                    },
                }
            )

    def evaluation(scheme: str, model: str) -> dict[str, Any]:
        matches = [
            row
            for row in evaluations
            if row["scheme"] == scheme and row["model"] == model
        ]
        if len(matches) != 1:
            raise GPUExperimentError("evaluation summary lookup is ambiguous")
        return matches[0]

    primary_scheme = str(config["splits"]["primary"]["name"])
    secondary_scheme = str(config["splits"]["secondary"]["name"])
    direction: dict[str, Any] = {}
    for scheme in (primary_scheme, secondary_scheme):
        torch_row = evaluation(scheme, "torch_mlp")
        xgb_row = evaluation(scheme, "xgboost_cuda")
        torch_macro = torch_row["macro_equal_group"]
        xgb_macro = xgb_row["macro_equal_group"]
        direction[scheme] = {
            "primary_torch_minus_xgboost_macro_rmse_3d_mm": float(
                torch_macro["rmse_3d_mm"] - xgb_macro["rmse_3d_mm"]
            ),
            "primary_torch_minus_xgboost_macro_p95_3d_error_mm": float(
                torch_macro["p95_3d_error_mm"] - xgb_macro["p95_3d_error_mm"]
            ),
            "torch_worst_group_rmse_3d_mm": float(
                max(row["rmse_3d_mm"] for row in torch_row["per_group"])
            ),
            "torch_mean_seed_rmse_std_mm": float(
                np.mean([row["seed_rmse_std_mm"] for row in torch_row["per_group"]])
            ),
        }
    return {
        "outer_test_accessed_after_inner_selection": True,
        "outer_test_used_for_final_config": False,
        "evaluations": evaluations,
        "direction": direction,
    }


def _final_model_manifest(
    config: dict[str, Any],
    selection: Mapping[str, Any],
    final_results: Sequence[dict[str, Any]],
    *,
    run_dir: Path,
) -> dict[str, Any]:
    if len(final_results) != int(config["models"]["torch_mlp"]["ensemble_seed_count"]):
        raise GPUExperimentError("final ensemble seed count changed")
    preprocessors = [result["preprocessor"] for result in final_results]
    if any(value != preprocessors[0] for value in preprocessors[1:]):
        raise GPUExperimentError("final seed preprocessors differ")
    preprocessing_path = run_dir / "final" / "preprocessing.json"
    atomic_write_json(preprocessing_path, preprocessors[0])
    checkpoints: list[dict[str, Any]] = []
    for result in sorted(
        final_results, key=lambda row: int(row["seed_summaries"][0]["seed_index"])
    ):
        summary = result["seed_summaries"][0]
        checkpoints.append(
            {
                "seed_index": int(summary["seed_index"]),
                "model_seed": int(summary["model_seed"]),
                "trained_epochs": int(summary["trained_epochs"]),
                "train_rmse_3d_mm": float(summary["selected_train_rmse_3d_mm"]),
                **result["checkpoint_file"],
            }
        )
    selected = selection["final_global_segment_inner_selection"]
    manifest = {
        "contract": "alphaf.endpoint_regression.gpu_model.v1",
        "model_family": "torch_mlp",
        "ensemble_kind": "mean_raw_vive_world_xyz",
        "ensemble_seed_count": len(checkpoints),
        "feature_view": str(config["task"]["input_view"]),
        "feature_names": preprocessors[0]["feature_names"],
        "label_columns": list(LABEL_COLUMNS),
        "label_frame": str(config["contract"]["label_frame"]),
        "label_unit": str(config["contract"]["label_unit"]),
        "selected_config_id": str(selected["selected_config_id"]),
        "selected_params": selected["selected_params"],
        "selection_source": "20_segment_inner_validation_contexts",
        "selection_score_mm": float(selected["selected_score_mm"]),
        "independent_test_score_available": False,
        "preprocessing": {
            "path": _relative(preprocessing_path, run_dir),
            "sha256": sha256_file(preprocessing_path),
        },
        "checkpoints": checkpoints,
    }
    path = run_dir / "final" / "model_manifest.json"
    atomic_write_json(path, manifest)
    return manifest


def _validate_gpu_ids(gpu_ids: Sequence[int], expected_count: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in gpu_ids)
    if len(values) != expected_count or len(set(values)) != expected_count:
        raise GPUExperimentError(
            f"full GPU experiment requires {expected_count} distinct GPU IDs"
        )
    return values


def run_gpu_experiment(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    gpu_ids: Sequence[int] = (0, 1, 2, 3),
    resume: bool = False,
) -> dict[str, Any]:
    """Run the frozen inner-selection, outer-evaluation, and final-refit phases."""

    validate_gpu_config(config)
    gpu_ids = _validate_gpu_ids(gpu_ids, int(config["runtime"]["gpu_count"]))
    release_path = resolve_gpu_release_path(config)
    # Authenticate once before creating worker processes; workers independently
    # repeat this check before any model receives data.
    release = verify_and_load_release(
        release_path,
        expected_sha256=str(config["release"]["archive_sha256"]),
    )
    if len(release.eligible_real) != int(config["release"]["expected_train_eligible_rows"]):
        raise GPUExperimentError("verified eligible-row count changed")
    environment = gpu_environment(gpu_ids, config)
    workspace_root = Path(str(config["_workspace_root"])).resolve()
    source = training_source_payload(workspace_root)
    config["_source_fingerprint"] = source["fingerprint"]
    identity = {
        "contract": GPU_EXPERIMENT_CONTRACT,
        "config_sha256": gpu_config_sha256(config),
        "release_sha256": str(config["release"]["archive_sha256"]),
        "run_kind": "full",
        "gpu_ids": list(gpu_ids),
        "source_fingerprint": source["fingerprint"],
    }
    run_dir = prepare_experiment_directory(output_dir, identity=identity, resume=resume)
    atomic_write_json(run_dir / "config.resolved.json", public_gpu_config(config))
    environment["git"] = _git_payload(workspace_root)
    environment["training_source"] = source
    environment_path = run_dir / "environment.json"
    if resume:
        if read_json_object(environment_path) != environment:
            raise GPUExperimentError("resume environment/source provenance changed")
    else:
        atomic_write_json(environment_path, environment)
    _emit(
        {
            "event": "run_started",
            "run_dir": str(run_dir),
            "inner_tasks": 480,
            "gpus": list(gpu_ids),
        }
    )

    inner_tasks = _inner_tasks(config)
    inner_results = _run_task_batch(
        inner_tasks,
        config=config,
        release_path=release_path,
        run_dir=run_dir,
        gpu_ids=gpu_ids,
    )
    selection, outer_tasks = _selection_and_outer_tasks(config, inner_results)
    selection_path = run_dir / "selection" / "inner_selection.json"
    atomic_write_json(selection_path, selection)
    _emit(
        {
            "event": "inner_selection_complete",
            "outer_test_accessed": False,
            "global_torch_config": selection["final_global_segment_inner_selection"][
                "selected_config_id"
            ],
            "global_inner_rmse_3d_mm": selection[
                "final_global_segment_inner_selection"
            ]["selected_score_mm"],
        }
    )

    outer_results = _run_task_batch(
        outer_tasks,
        config=config,
        release_path=release_path,
        run_dir=run_dir,
        gpu_ids=gpu_ids,
    )
    evaluation = _outer_evaluation_summary(
        config, outer_results, run_dir=run_dir
    )
    atomic_write_json(run_dir / "evaluation" / "summary.json", evaluation)
    _emit({"event": "outer_evaluation_complete", "direction": evaluation["direction"]})

    final_tasks = _final_tasks(config, selection)
    final_results = _run_task_batch(
        final_tasks,
        config=config,
        release_path=release_path,
        run_dir=run_dir,
        gpu_ids=gpu_ids,
    )
    final_model = _final_model_manifest(
        config, selection, final_results, run_dir=run_dir
    )
    summary = {
        "contract": GPU_EXPERIMENT_CONTRACT,
        "status": "complete",
        "objective": "Vive-free static endpoint estimation from encoders and four IMUs",
        "input_view": str(config["task"]["input_view"]),
        "label_frame": str(config["contract"]["label_frame"]),
        "evaluation": evaluation,
        "final_model": final_model,
        "claim_boundaries": config["claim_boundaries"],
    }
    atomic_write_json(run_dir / "summary.json", summary)
    manifest = {
        "contract": GPU_EXPERIMENT_CONTRACT,
        "config_sha256": gpu_config_sha256(config),
        "release_sha256": str(config["release"]["archive_sha256"]),
        "source_binding": "task_embedded",
        "source_fingerprint": source["fingerprint"],
        "counts": {
            "eligible_real_rows": 7702,
            "inner_tasks": len(inner_results),
            "outer_tasks": len(outer_results),
            "final_seed_tasks": len(final_results),
            "gpu_count": len(gpu_ids),
        },
        "selection": {
            "path": _relative(selection_path, run_dir),
            "sha256": sha256_file(selection_path),
        },
        "final_model": final_model,
        "files": manifest_files(run_dir),
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    validation = validate_gpu_run(run_dir)
    _emit({"event": "run_complete", "validation": validation})
    return manifest


def run_gpu_smoke(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    gpu_ids: Sequence[int] = (0, 1, 2, 3),
    resume: bool = False,
) -> dict[str, Any]:
    """Exercise all GPU model configurations with a non-claimable 20-epoch pilot."""

    validate_gpu_config(config)
    gpu_ids = _validate_gpu_ids(gpu_ids, int(config["runtime"]["gpu_count"]))
    release_path = resolve_gpu_release_path(config)
    environment = gpu_environment(gpu_ids, config)
    workspace_root = Path(str(config["_workspace_root"])).resolve()
    source = training_source_payload(workspace_root)
    config["_source_fingerprint"] = source["fingerprint"]
    identity = {
        "contract": GPU_EXPERIMENT_CONTRACT,
        "config_sha256": gpu_config_sha256(config),
        "release_sha256": str(config["release"]["archive_sha256"]),
        "run_kind": "smoke_non_contract",
        "gpu_ids": list(gpu_ids),
        "source_fingerprint": source["fingerprint"],
    }
    run_dir = prepare_experiment_directory(output_dir, identity=identity, resume=resume)
    atomic_write_json(run_dir / "config.resolved.json", public_gpu_config(config))
    environment["git"] = _git_payload(workspace_root)
    environment["training_source"] = source
    environment_path = run_dir / "environment.json"
    if resume:
        if read_json_object(environment_path) != environment:
            raise GPUExperimentError("resume environment/source provenance changed")
    else:
        atomic_write_json(environment_path, environment)
    tasks = _inner_tasks(config, smoke=True)
    results = _run_task_batch(
        tasks,
        config=config,
        release_path=release_path,
        run_dir=run_dir,
        gpu_ids=gpu_ids,
    )
    torch_results = [row for row in results if row["task"]["model"] == "torch_mlp"]
    directions: list[dict[str, Any]] = []
    for result in torch_results:
        history_path = run_dir / str(result["history_file"]["path"])
        history = pd.read_parquet(history_path).sort_values("epoch")
        first = float(history.iloc[0]["validation_rmse_3d_mm"])
        last = float(history.iloc[-1]["validation_rmse_3d_mm"])
        directions.append(
            {
                "config_id": result["task"]["config_id"],
                "first_validation_rmse_3d_mm": first,
                "last_validation_rmse_3d_mm": last,
                "improved": bool(last < first),
            }
        )
    if not all(row["improved"] for row in directions):
        raise GPUExperimentError("one or more MLP smoke configurations failed to improve")
    best_torch = min(
        torch_results, key=lambda row: float(row["metrics"]["rmse_3d_mm"])
    )
    best_xgb = min(
        (row for row in results if row["task"]["model"] == "xgboost_cuda"),
        key=lambda row: float(row["metrics"]["rmse_3d_mm"]),
    )
    integration_tasks: list[dict[str, Any]] = []
    for selected in (best_torch, best_xgb):
        model_name = str(selected["task"]["model"])
        task: dict[str, Any] = {
            "phase": "outer",
            "scheme": "segment_loso",
            "outer_fold": 36,
            "inner_fold": -1,
            "model": model_name,
            "params": selected["task"]["params"],
            "config_id": selected["task"]["config_id"],
            "smoke_contract_eligible": False,
        }
        if model_name == "torch_mlp":
            task["epochs_per_seed"] = {"0": 20, "1": 20, "2": 20}
        task["task_id"] = _task_id(
            "outer", "segment_loso", 36, -1, model_name, task["config_id"]
        )
        integration_tasks.append(task)
    for seed_index in range(3):
        task = {
            "phase": "final",
            "scheme": "segment_loso",
            "outer_fold": -1,
            "inner_fold": -1,
            "model": "torch_mlp",
            "params": best_torch["task"]["params"],
            "config_id": best_torch["task"]["config_id"],
            "seed_indices": [seed_index],
            "epochs_per_seed": {str(seed_index): 20},
            "smoke_contract_eligible": False,
        }
        task["task_id"] = _task_id(
            "final",
            "segment_loso",
            -1,
            -1,
            "torch_mlp",
            task["config_id"],
            seed_index,
        )
        integration_tasks.append(task)
    integration_results = _run_task_batch(
        integration_tasks,
        config=config,
        release_path=release_path,
        run_dir=run_dir,
        gpu_ids=gpu_ids,
    )
    if not all(
        (row["metrics"] is None or np.isfinite(float(row["metrics"]["rmse_3d_mm"])))
        and (
            row["phase"] != "final"
            or (run_dir / str(row["checkpoint_file"]["path"])).is_file()
        )
        for row in integration_results
    ):
        raise GPUExperimentError("GPU smoke outer/final integration evidence failed")
    summary = {
        "contract": GPU_EXPERIMENT_CONTRACT,
        "status": "smoke_ok",
        "contract_eligible": False,
        "reason": "20 epochs and one seed are execution diagnostics only",
        "task_count": len(results),
        "integration_task_count": len(integration_results),
        "outer_and_final_paths_ok": True,
        "torch_direction": directions,
        "finite_metrics": all(
            row["metrics"] is not None
            and np.isfinite(float(row["metrics"]["rmse_3d_mm"]))
            for row in results
        ),
    }
    atomic_write_json(run_dir / "smoke_summary.json", summary)
    return summary


def _assert_json_close(actual: Any, expected: Any, *, context: str) -> None:
    """Compare JSON-like evidence while allowing only floating-point round-off."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise GPUExperimentError(f"{context} mapping keys changed")
        for key in expected:
            _assert_json_close(
                actual[key], expected[key], context=f"{context}.{key}"
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise GPUExperimentError(f"{context} list shape changed")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_json_close(
                actual_item, expected_item, context=f"{context}[{index}]"
            )
        return
    if type(expected) is float:
        if type(actual) not in (float, int) or not math.isfinite(float(actual)):
            raise GPUExperimentError(f"{context} is not a finite number")
        if not math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-10
        ):
            raise GPUExperimentError(
                f"{context} changed: {actual!r} != {expected!r}"
            )
        return
    if type(actual) is not type(expected) or actual != expected:
        raise GPUExperimentError(f"{context} changed: {actual!r} != {expected!r}")


def _safe_run_member(root: Path, recorded: Any, *, expected: str) -> Path:
    if not isinstance(recorded, str) or not recorded:
        raise GPUExperimentError(f"invalid recorded run member for {expected}")
    relative = PurePosixPath(recorded)
    if (
        relative.is_absolute()
        or relative.as_posix() != recorded
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise GPUExperimentError(f"unsafe recorded run member: {recorded!r}")
    if recorded != expected:
        raise GPUExperimentError(
            f"run member path changed: {recorded!r} != {expected!r}"
        )
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise GPUExperimentError(f"run member escapes artifact root: {recorded}") from exc
    if not path.is_file():
        raise GPUExperimentError(f"run member is missing: {recorded}")
    return path


def _validation_workspace_root(root: Path, config: Mapping[str, Any]) -> Path:
    configured = Path(str(config["release"]["archive"]))
    for candidate in (root, *root.parents):
        release_path = (
            configured.resolve()
            if configured.is_absolute()
            else (candidate / configured).resolve()
        )
        if (
            release_path.is_file()
            and (candidate / "src" / "imu_vive_learning").is_dir()
        ):
            return candidate
    raise GPUExperimentError(
        "cannot locate the pinned preparation release relative to the GPU artifact"
    )


def _validate_source_provenance(
    root: Path, manifest: Mapping[str, Any]
) -> str | None:
    binding = manifest.get("source_binding")
    fingerprint = manifest.get("source_fingerprint")
    if binding == "task_embedded":
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise GPUExperimentError("task-embedded source fingerprint is invalid")
        environment = read_json_object(root / "environment.json")
        source = environment.get("training_source")
        if not isinstance(source, dict) or source.get("fingerprint") != fingerprint:
            raise GPUExperimentError("environment/source fingerprint mismatch")
        files = source.get("files")
        if (
            not isinstance(files, list)
            or [row.get("path") if isinstance(row, dict) else None for row in files]
            != list(TRAINING_SOURCE_FILES)
            or stable_hash(files) != fingerprint
        ):
            raise GPUExperimentError("task-embedded training-source ledger changed")
        return fingerprint
    if binding != "post_run_uninterrupted_capture":
        raise GPUExperimentError("GPU run has no auditable training-source binding")
    reference = manifest.get("training_source_provenance")
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise GPUExperimentError("post-run source provenance reference is invalid")
    provenance_path = _safe_run_member(
        root,
        reference.get("path"),
        expected="training_source_provenance.json",
    )
    if sha256_file(provenance_path) != reference.get("sha256"):
        raise GPUExperimentError("training-source provenance digest changed")
    provenance = read_json_object(provenance_path)
    if (
        provenance.get("contract") != GPU_EXPERIMENT_CONTRACT
        or provenance.get("run_was_uninterrupted_without_resume") is not True
    ):
        raise GPUExperimentError("post-run source provenance claim changed")
    source_files = provenance.get("source_files")
    if not isinstance(source_files, list):
        raise GPUExperimentError("training-source ledger is missing")
    ledger_paths = [
        row.get("path") if isinstance(row, dict) else None for row in source_files
    ]
    historical_training_paths = set(TRAINING_SOURCE_FILES) - {
        "src/imu_vive_learning/gpu_inference.py"
    }
    if (
        len(ledger_paths) != len(historical_training_paths)
        or len(ledger_paths) != len(set(ledger_paths))
        or set(ledger_paths) != historical_training_paths
    ):
        raise GPUExperimentError("training-source ledger path set changed")
    ledger: dict[str, str] = {}
    for row in source_files:
        if not isinstance(row, dict) or set(row) - {"path", "sha256", "bytes"}:
            raise GPUExperimentError("training-source ledger entry changed")
        digest = row.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(value not in "0123456789abcdef" for value in digest)
        ):
            raise GPUExperimentError("training-source ledger digest is invalid")
        ledger[str(row["path"])] = digest
    ledger_fingerprint = stable_hash(source_files)
    if provenance.get("source_fingerprint") != ledger_fingerprint:
        raise GPUExperimentError("training-source ledger fingerprint changed")
    if fingerprint != ledger_fingerprint:
        raise GPUExperimentError("manifest/source provenance fingerprint mismatch")

    snapshot = provenance.get("exact_source_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("authoritative") is not True:
        raise GPUExperimentError("authoritative exact source snapshot is missing")
    snapshot_path = _safe_run_member(
        root,
        snapshot.get("path"),
        expected="training_source_snapshot_exact.tar",
    )
    if (
        snapshot.get("sha256") != sha256_file(snapshot_path)
        or type(snapshot.get("bytes")) is not int
        or int(snapshot["bytes"]) != snapshot_path.stat().st_size
    ):
        raise GPUExperimentError("exact source snapshot digest/size changed")
    try:
        with tarfile.open(snapshot_path, mode="r:") as archive:
            members = archive.getmembers()
            member_names = [member.name for member in members]
            if (
                len(members) != len(historical_training_paths)
                or any(not member.isfile() for member in members)
                or len(member_names) != len(set(member_names))
                or set(member_names) != historical_training_paths
            ):
                raise GPUExperimentError("exact source snapshot member set changed")
            for member in members:
                handle = archive.extractfile(member)
                if handle is None:
                    raise GPUExperimentError(
                        f"cannot read source snapshot member: {member.name}"
                    )
                digest = sha256(handle.read()).hexdigest()
                if digest != ledger[member.name]:
                    raise GPUExperimentError(
                        f"source snapshot member digest changed: {member.name}"
                    )
    except (OSError, tarfile.TarError) as exc:
        raise GPUExperimentError("cannot authenticate exact source snapshot") from exc
    return ledger_fingerprint


def _validate_seed_and_history(
    root: Path,
    result: Mapping[str, Any],
    task: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    summaries = result.get("seed_summaries")
    if not isinstance(summaries, list):
        raise GPUExperimentError(f"seed summaries missing: {task['task_id']}")
    split_context = str(result["split_context"])
    if task["model"] == "torch_mlp":
        expected_indices = [int(value) for value in task.get("seed_indices", range(3))]
    else:
        expected_indices = [0]
    if [row.get("seed_index") for row in summaries if isinstance(row, dict)] != expected_indices:
        raise GPUExperimentError(f"seed index set changed: {task['task_id']}")
    for summary, seed_index in zip(summaries, expected_indices, strict=True):
        if not isinstance(summary, dict):
            raise GPUExperimentError(f"malformed seed summary: {task['task_id']}")
        expected_seed = derive_context_seed(
            int(config["contract"]["base_seed"]),
            "gpu_model_seed",
            split_context,
            seed_index,
        )
        if summary.get("model_seed") != expected_seed:
            raise GPUExperimentError(f"model seed changed: {task['task_id']}")
        if task["model"] == "torch_mlp":
            expected_epochs = int(
                task.get("epochs_per_seed", {}).get(
                    str(seed_index),
                    task.get(
                        "epoch_budget_override",
                        config["models"]["torch_mlp"]["epoch_budget"],
                    ),
                )
            )
        else:
            expected_epochs = int(config["models"]["xgboost_cuda"]["n_estimators"])
        if summary.get("trained_epochs") != expected_epochs:
            raise GPUExperimentError(f"trained epoch count changed: {task['task_id']}")
        selected_epoch = summary.get("selected_epoch")
        if type(selected_epoch) is not int or not 1 <= selected_epoch <= expected_epochs:
            raise GPUExperimentError(f"selected epoch is invalid: {task['task_id']}")
        if task["phase"] not in {"inner", "smoke"} and selected_epoch != expected_epochs:
            raise GPUExperimentError(f"refit epoch evidence changed: {task['task_id']}")
        required_numeric = {"fit_seconds"}
        if task["model"] == "torch_mlp":
            required_numeric.update(
                {"final_scaled_mse_loss", "selected_train_rmse_3d_mm"}
            )
        if task["phase"] != "final" and task["model"] == "torch_mlp":
            required_numeric.update(
                {
                    "evaluation_rmse_3d_mm",
                    "evaluation_p95_3d_error_mm",
                }
            )
        if task["phase"] in {"inner", "smoke"} and task["model"] == "torch_mlp":
            required_numeric.add("best_validation_rmse_3d_mm")
        if task["model"] == "xgboost_cuda":
            required_numeric.add("best_validation_rmse_3d_mm")
        for key in required_numeric:
            value = summary.get(key)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise GPUExperimentError(
                    f"seed summary is non-finite: {task['task_id']}/{key}"
                )

    history_reference = result.get("history_file")
    if task["model"] != "torch_mlp":
        if history_reference is not None:
            raise GPUExperimentError(f"unexpected XGBoost history: {task['task_id']}")
        return
    if not isinstance(history_reference, dict):
        raise GPUExperimentError(f"MLP history reference is missing: {task['task_id']}")
    expected_relative = f"{task['phase']}/history/{task['task_id']}.parquet"
    history_path = _safe_run_member(
        root, history_reference.get("path"), expected=expected_relative
    )
    if sha256_file(history_path) != history_reference.get("sha256"):
        raise GPUExperimentError(f"MLP history digest changed: {task['task_id']}")
    history = pd.read_parquet(history_path)
    if type(history_reference.get("rows")) is not int or len(history) != int(
        history_reference["rows"]
    ):
        raise GPUExperimentError(f"MLP history row count changed: {task['task_id']}")
    if history.empty or set(history["task_id"].astype(str)) != {str(task["task_id"])}:
        raise GPUExperimentError(f"MLP history task identity changed: {task['task_id']}")
    numeric = history.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if numeric.size == 0 or not np.isfinite(numeric).all():
        raise GPUExperimentError(f"MLP history contains non-finite values: {task['task_id']}")
    for summary in summaries:
        seed_index = int(summary["seed_index"])
        rows = history.loc[history["seed_index"].astype(int).eq(seed_index)].sort_values(
            "epoch", kind="mergesort"
        )
        epochs = rows["epoch"].astype(int).tolist()
        if (
            not epochs
            or epochs[0] != 1
            or epochs[-1] != int(summary["trained_epochs"])
            or len(epochs) != len(set(epochs))
        ):
            raise GPUExperimentError(f"MLP history cadence changed: {task['task_id']}")
        if set(rows["model_seed"].astype(int)) != {int(summary["model_seed"])}:
            raise GPUExperimentError(f"MLP history seed changed: {task['task_id']}")
        if task["phase"] in {"inner", "smoke"}:
            last = rows.iloc[-1]
            if int(last["best_epoch_so_far"]) != int(summary["selected_epoch"]):
                raise GPUExperimentError(
                    f"MLP selected checkpoint/history mismatch: {task['task_id']}"
                )
            if not math.isclose(
                float(last["best_validation_rmse_3d_mm"]),
                float(summary["best_validation_rmse_3d_mm"]),
                rel_tol=1e-12,
                abs_tol=1e-10,
            ):
                raise GPUExperimentError(
                    f"MLP best score/history mismatch: {task['task_id']}"
                )


def _validate_task_results(
    root: Path,
    config: dict[str, Any],
    data: _WorkerData,
    expected_tasks: Sequence[dict[str, Any]],
    *,
    phase: str,
    source_binding: str,
    source_fingerprint: str | None,
    array_cache: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    result_dir = root / phase / "results"
    paths = sorted(result_dir.glob("*.json"))
    expected_by_id = {str(task["task_id"]): task for task in expected_tasks}
    if len(expected_by_id) != len(expected_tasks):
        raise GPUExperimentError(f"duplicate expected {phase} task IDs")
    actual_ids = {path.stem for path in paths}
    if actual_ids != set(expected_by_id):
        raise GPUExperimentError(
            f"GPU {phase} task set changed; missing={sorted(set(expected_by_id) - actual_ids)}, "
            f"unexpected={sorted(actual_ids - set(expected_by_id))}"
        )
    results: list[dict[str, Any]] = []
    expected_feature_names = list(data.feature_names)
    for expected_task in expected_tasks:
        task_id = str(expected_task["task_id"])
        path = result_dir / f"{task_id}.json"
        result = read_json_object(path)
        if result.get("contract") != _result_contract(config):
            raise GPUExperimentError(f"GPU result contract changed: {task_id}")
        if result.get("task") != expected_task or result.get("task_hash") != stable_hash(
            expected_task
        ):
            raise GPUExperimentError(f"GPU task identity changed: {task_id}")
        if (
            result.get("task_id") != task_id
            or result.get("phase") != phase
            or result.get("config_sha256") != _result_config_sha256(config)
        ):
            raise GPUExperimentError(f"GPU result identity changed: {task_id}")
        if source_binding == "task_embedded" and result.get(
            "source_fingerprint"
        ) != source_fingerprint:
            raise GPUExperimentError(f"GPU task source fingerprint changed: {task_id}")
        if (
            result.get("feature_count") != len(expected_feature_names)
            or result.get("feature_names") != expected_feature_names
            or type(result.get("gpu_physical_id")) is not int
            or result.get("gpu_physical_id")
            not in {
                int(value)
                for value in config.get("_allowed_gpu_ids", (0, 1, 2, 3))
            }
            or result.get("gpu_visible_device") != 0
        ):
            raise GPUExperimentError(f"GPU feature/device evidence changed: {task_id}")
        elapsed = result.get("elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(float(elapsed)) or elapsed <= 0:
            raise GPUExperimentError(f"GPU elapsed time is invalid: {task_id}")

        if phase == "final":
            cache_key = ("final",)
            arrays = array_cache.setdefault(cache_key, _all_real_arrays(data))
            expected_role = None
            expected_eval_rows = 0
            expected_train_rows = len(arrays["train"])
        else:
            cache_key = (
                str(expected_task["scheme"]),
                int(expected_task["outer_fold"]),
                int(expected_task["inner_fold"]),
            )
            if cache_key not in array_cache:
                array_cache[cache_key] = _context_arrays(
                    data,
                    scheme=str(expected_task["scheme"]),
                    outer_fold=int(expected_task["outer_fold"]),
                    inner_fold=int(expected_task["inner_fold"]),
                )
            arrays = array_cache[cache_key]
            expected_role = "validation" if phase in {"inner", "smoke"} else "test"
            expected_eval_rows = len(arrays["evaluation"])
            expected_train_rows = len(arrays["train"])
        if (
            result.get("split_context") != arrays["split_context"]
            or result.get("evaluation_role") != expected_role
            or result.get("evaluation_rows") != expected_eval_rows
            or result.get("real_train_rows") != expected_train_rows
        ):
            raise GPUExperimentError(f"GPU split row evidence changed: {task_id}")
        _assert_json_close(
            result.get("preprocessor"),
            arrays["preprocessor"].metadata(),
            context=f"{task_id}.preprocessor",
        )
        _validate_seed_and_history(root, result, expected_task, config)

        prediction_reference = result.get("prediction_file")
        if phase == "final":
            if prediction_reference is not None or result.get("metrics") is not None:
                raise GPUExperimentError(f"final task contains evaluation evidence: {task_id}")
        else:
            if not isinstance(prediction_reference, dict):
                raise GPUExperimentError(f"prediction reference is missing: {task_id}")
            expected_relative = f"{phase}/predictions/{task_id}.parquet"
            prediction_path = _safe_run_member(
                root, prediction_reference.get("path"), expected=expected_relative
            )
            if sha256_file(prediction_path) != prediction_reference.get("sha256"):
                raise GPUExperimentError(f"prediction digest changed: {task_id}")
            frame = pd.read_parquet(prediction_path)
            expected_columns = [
                "trial_uid",
                "segment_id",
                "target_id",
                "scheme",
                "outer_fold",
                "inner_fold",
                "model",
                "config_id",
                *TRUE_COLUMNS,
                *PREDICTION_COLUMNS,
                "error_3d_mm",
            ]
            if list(frame.columns) != expected_columns:
                raise GPUExperimentError(f"prediction schema changed: {task_id}")
            if (
                type(prediction_reference.get("rows")) is not int
                or len(frame) != prediction_reference["rows"]
                or len(frame) != expected_eval_rows
            ):
                raise GPUExperimentError(f"prediction row count changed: {task_id}")
            evaluation = arrays["evaluation"]
            expected_ids = evaluation["trial_uid"].astype(str).tolist()
            if frame["trial_uid"].astype(str).tolist() != expected_ids:
                raise GPUExperimentError(f"prediction split membership changed: {task_id}")
            if (
                frame["segment_id"].astype(int).tolist()
                != evaluation["segment_id"].astype(int).tolist()
                or frame["target_id"].astype(str).tolist()
                != evaluation["target_id"].astype(str).tolist()
            ):
                raise GPUExperimentError(f"prediction grouping evidence changed: {task_id}")
            for column, expected_value in (
                ("scheme", str(expected_task["scheme"])),
                ("outer_fold", int(expected_task["outer_fold"])),
                ("inner_fold", int(expected_task["inner_fold"])),
                ("model", str(expected_task["model"])),
                ("config_id", str(expected_task["config_id"])),
            ):
                if set(frame[column].tolist()) != {expected_value}:
                    raise GPUExperimentError(
                        f"prediction task column changed: {task_id}/{column}"
                    )
            truth = frame.loc[:, list(TRUE_COLUMNS)].to_numpy(dtype=np.float64)
            prediction = frame.loc[:, list(PREDICTION_COLUMNS)].to_numpy(dtype=np.float64)
            recorded_error = frame["error_3d_mm"].to_numpy(dtype=np.float64)
            if (
                not np.isfinite(prediction).all()
                or not np.array_equal(truth, arrays["y_eval_raw"])
                or not np.allclose(
                    recorded_error,
                    np.linalg.norm(prediction - truth, axis=1),
                    rtol=1e-13,
                    atol=1e-12,
                )
            ):
                raise GPUExperimentError(f"prediction truth/error changed: {task_id}")
            _assert_json_close(
                result.get("metrics"),
                regression_metrics(truth, prediction),
                context=f"{task_id}.metrics",
            )

        checkpoint_reference = result.get("checkpoint_file")
        if phase != "final":
            if checkpoint_reference is not None:
                raise GPUExperimentError(f"non-final task emitted checkpoint: {task_id}")
        else:
            seed_index = int(expected_task["seed_indices"][0])
            if not isinstance(checkpoint_reference, dict):
                raise GPUExperimentError(f"final checkpoint reference is missing: {task_id}")
            expected_relative = f"final/checkpoints/torch_mlp_seed{seed_index}.npz"
            checkpoint_path = _safe_run_member(
                root, checkpoint_reference.get("path"), expected=expected_relative
            )
            if sha256_file(checkpoint_path) != checkpoint_reference.get("sha256"):
                raise GPUExperimentError(f"final checkpoint digest changed: {task_id}")
        results.append(result)
    return results


def _expected_final_model_manifest(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    final_results: Sequence[Mapping[str, Any]],
    *,
    root: Path,
) -> dict[str, Any]:
    preprocessors = [result["preprocessor"] for result in final_results]
    if not preprocessors or any(value != preprocessors[0] for value in preprocessors[1:]):
        raise GPUExperimentError("final seed preprocessors differ")
    preprocessing_path = _safe_run_member(
        root, "final/preprocessing.json", expected="final/preprocessing.json"
    )
    _assert_json_close(
        read_json_object(preprocessing_path),
        preprocessors[0],
        context="final.preprocessing",
    )
    checkpoints: list[dict[str, Any]] = []
    for result in sorted(
        final_results, key=lambda row: int(row["seed_summaries"][0]["seed_index"])
    ):
        summary = result["seed_summaries"][0]
        checkpoints.append(
            {
                "seed_index": int(summary["seed_index"]),
                "model_seed": int(summary["model_seed"]),
                "trained_epochs": int(summary["trained_epochs"]),
                "train_rmse_3d_mm": float(summary["selected_train_rmse_3d_mm"]),
                **result["checkpoint_file"],
            }
        )
    selected = selection["final_global_segment_inner_selection"]
    return {
        "contract": "alphaf.endpoint_regression.gpu_model.v1",
        "model_family": "torch_mlp",
        "ensemble_kind": "mean_raw_vive_world_xyz",
        "ensemble_seed_count": len(checkpoints),
        "feature_view": str(config["task"]["input_view"]),
        "feature_names": preprocessors[0]["feature_names"],
        "label_columns": list(LABEL_COLUMNS),
        "label_frame": str(config["contract"]["label_frame"]),
        "label_unit": str(config["contract"]["label_unit"]),
        "selected_config_id": str(selected["selected_config_id"]),
        "selected_params": selected["selected_params"],
        "selection_source": "20_segment_inner_validation_contexts",
        "selection_score_mm": float(selected["selected_score_mm"]),
        "independent_test_score_available": False,
        "preprocessing": {
            "path": "final/preprocessing.json",
            "sha256": sha256_file(preprocessing_path),
        },
        "checkpoints": checkpoints,
    }


def validate_gpu_run(run_dir: str | Path) -> dict[str, Any]:
    """Semantically validate a completed GPU run without requiring CUDA."""

    root = Path(run_dir).resolve()
    manifest = read_json_object(root / "manifest.json")
    if manifest.get("contract") != GPU_EXPERIMENT_CONTRACT:
        raise GPUExperimentError("GPU run manifest contract changed")
    config = read_json_object(root / "config.resolved.json")
    validate_gpu_config(config)
    config_sha256 = gpu_config_sha256(config)
    if manifest.get("config_sha256") != config_sha256:
        raise GPUExperimentError("GPU run manifest/config hash mismatch")
    expected_counts = {
        "eligible_real_rows": 7702,
        "inner_tasks": 480,
        "outer_tasks": 20,
        "final_seed_tasks": 3,
        "gpu_count": 4,
    }
    if manifest.get("counts") != expected_counts:
        raise GPUExperimentError(
            f"GPU run counts changed: {manifest.get('counts')!r}"
        )
    verify_file_manifest(root, manifest["files"])
    source_fingerprint = _validate_source_provenance(root, manifest)
    source_binding = str(manifest["source_binding"])

    workspace_root = _validation_workspace_root(root, config)
    config["_workspace_root"] = str(workspace_root)
    release_path = resolve_gpu_release_path(config)
    data = _load_worker_data(config, release_path)
    if len(data.eligible) != expected_counts["eligible_real_rows"]:
        raise GPUExperimentError("verified release eligible-row count changed")
    array_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    inner_tasks = _inner_tasks(config)
    inner_results = _validate_task_results(
        root,
        config,
        data,
        inner_tasks,
        phase="inner",
        source_binding=source_binding,
        source_fingerprint=source_fingerprint,
        array_cache=array_cache,
    )
    expected_selection, outer_tasks = _selection_and_outer_tasks(config, inner_results)
    selection_reference = manifest.get("selection")
    if not isinstance(selection_reference, dict) or set(selection_reference) != {
        "path",
        "sha256",
    }:
        raise GPUExperimentError("inner-selection manifest reference changed")
    selection_path = _safe_run_member(
        root,
        selection_reference.get("path"),
        expected="selection/inner_selection.json",
    )
    if sha256_file(selection_path) != selection_reference.get("sha256"):
        raise GPUExperimentError("inner-selection digest changed")
    recorded_selection = read_json_object(selection_path)
    _assert_json_close(
        recorded_selection, expected_selection, context="inner_selection"
    )

    outer_results = _validate_task_results(
        root,
        config,
        data,
        outer_tasks,
        phase="outer",
        source_binding=source_binding,
        source_fingerprint=source_fingerprint,
        array_cache=array_cache,
    )
    expected_evaluation = _outer_evaluation_summary(
        config, outer_results, run_dir=root, verify_existing=True
    )
    recorded_evaluation = read_json_object(root / "evaluation" / "summary.json")
    _assert_json_close(
        recorded_evaluation, expected_evaluation, context="outer_evaluation"
    )

    final_tasks = _final_tasks(config, expected_selection)
    final_results = _validate_task_results(
        root,
        config,
        data,
        final_tasks,
        phase="final",
        source_binding=source_binding,
        source_fingerprint=source_fingerprint,
        array_cache=array_cache,
    )
    expected_model = _expected_final_model_manifest(
        config, expected_selection, final_results, root=root
    )
    model_manifest = read_json_object(root / "final" / "model_manifest.json")
    _assert_json_close(model_manifest, expected_model, context="final_model")
    _assert_json_close(
        manifest.get("final_model"), expected_model, context="manifest.final_model"
    )

    expected_summary = {
        "contract": GPU_EXPERIMENT_CONTRACT,
        "status": "complete",
        "objective": "Vive-free static endpoint estimation from encoders and four IMUs",
        "input_view": str(config["task"]["input_view"]),
        "label_frame": str(config["contract"]["label_frame"]),
        "evaluation": expected_evaluation,
        "final_model": expected_model,
        "claim_boundaries": config["claim_boundaries"],
    }
    _assert_json_close(
        read_json_object(root / "summary.json"),
        expected_summary,
        context="run_summary",
    )

    from .gpu_inference import load_gpu_endpoint_model

    model = load_gpu_endpoint_model(root, device="cpu")
    sample = data.eligible.iloc[:2].loc[:, list(data.feature_names)].to_numpy(
        dtype=np.float64
    )
    smoke_prediction = model.predict(sample, batch_size=2)
    if smoke_prediction.shape != (2, 3) or not np.isfinite(smoke_prediction).all():
        raise GPUExperimentError("final CPU inference smoke produced invalid predictions")
    import torch

    all_real = (
        data.eligible.reset_index(drop=True)
        .sort_values("trial_uid", kind="mergesort")
        .reset_index(drop=True)
    )
    all_features = all_real.loc[:, list(data.feature_names)].to_numpy(dtype=np.float64)
    all_truth = all_real.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
    scaled_features = model.preprocessing.transform_features(all_features)
    final_train_rmse_differences: list[float] = []
    with torch.inference_mode():
        for network, checkpoint in zip(
            model.models, model_manifest["checkpoints"], strict=True
        ):
            chunks: list[np.ndarray] = []
            for start in range(0, len(scaled_features), 4096):
                scaled_prediction = network(
                    torch.from_numpy(scaled_features[start : start + 4096])
                )
                chunks.append(
                    scaled_prediction.detach().cpu().numpy().astype(np.float64)
                )
            raw_prediction = model.preprocessing.inverse_targets(
                np.concatenate(chunks, axis=0)
            )
            recomputed_rmse = float(
                regression_metrics(all_truth, raw_prediction)["rmse_3d_mm"]
            )
            difference = abs(recomputed_rmse - float(checkpoint["train_rmse_3d_mm"]))
            if difference > 1e-5:
                raise GPUExperimentError("final checkpoint train RMSE changed")
            final_train_rmse_differences.append(difference)
    return {
        "status": "ok",
        "validation_level": "semantic_recomputation",
        "contract": GPU_EXPERIMENT_CONTRACT,
        "config_sha256": config_sha256,
        "release_sha256": data.release.archive_sha256,
        "source_binding": source_binding,
        "source_fingerprint": source_fingerprint,
        "counts": expected_counts,
        "validated_result_counts": {
            "inner": len(inner_results),
            "outer": len(outer_results),
            "final": len(final_results),
        },
        "validated_split_contexts": len(array_cache),
        "recomputed_selection": True,
        "recomputed_prediction_metrics": True,
        "recomputed_oof": True,
        "recomputed_final_seed_train_rmse": True,
        "max_final_seed_train_rmse_recompute_difference_mm": max(
            final_train_rmse_differences
        ),
        "hash_bound_attestations": [
            "inner_best_epoch_between_recorded_history_cadences",
            "outer_per_seed_prediction_metrics",
        ],
        "cpu_inference_smoke": True,
        "model_fingerprint": model.model_fingerprint,
        "final_model_config_id": model_manifest["selected_config_id"],
    }


__all__ = [
    "GPUExperimentError",
    "gpu_configuration_id",
    "gpu_environment",
    "run_gpu_experiment",
    "run_gpu_smoke",
    "validate_gpu_run",
]
