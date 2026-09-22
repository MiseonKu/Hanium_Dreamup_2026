"""Frozen configuration for the exploratory four-GPU endpoint experiment."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import yaml

from .config import canonical_json_bytes
from .experiment_release import PINNED_RELEASE_SHA256, PREPARATION_CONTRACT
from .features import FUSION_COLUMNS, LABEL_COLUMNS


GPU_EXPERIMENT_CONTRACT = "alphaf.endpoint_regression.gpu_experiment.v1"
PINNED_GPU_CONFIG_SHA256 = (
    "9bda3280b91351bd00f67faece111454c9807d277463683d5a3c409747dc4402"
)


class GPUExperimentConfigError(ValueError):
    """Raised when the pre-registered GPU experiment contract has drifted."""


def public_gpu_config(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy({key: value for key, value in config.items() if not key.startswith("_")})


def gpu_config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(public_gpu_config(config))).hexdigest()


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise GPUExperimentConfigError(f"missing mapping section: {key}")
    return value


def gpu_model_grid(config: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    """Return the frozen grid in its simple-first tie order."""

    models = _mapping(config, "models")
    if model_name == "torch_mlp":
        model = _mapping(models, model_name)
        return [
            {"hidden_layer_sizes": list(hidden), "weight_decay": float(decay)}
            for hidden in model["hidden_layer_sizes"]
            for decay in model["weight_decay"]
        ]
    if model_name == "xgboost_cuda":
        model = _mapping(models, model_name)
        return [
            {
                "max_depth": int(depth),
                "min_child_weight": float(child),
                "reg_lambda": float(reg_lambda),
            }
            for depth in model["max_depth"]
            for child in model["min_child_weight"]
            for reg_lambda in model["reg_lambda"]
        ]
    raise KeyError(f"unknown GPU model family: {model_name!r}")


def validate_gpu_config(config: dict[str, Any]) -> None:
    contract = _mapping(config, "contract")
    release = _mapping(config, "release")
    task = _mapping(config, "task")
    splits = _mapping(config, "splits")
    models = _mapping(config, "models")
    preprocessing = _mapping(config, "preprocessing")
    selection = _mapping(config, "selection")
    monitoring = _mapping(config, "monitoring")
    runtime = _mapping(config, "runtime")
    boundaries = _mapping(config, "claim_boundaries")

    if contract.get("name") != GPU_EXPERIMENT_CONTRACT:
        raise GPUExperimentConfigError("GPU experiment contract name changed")
    if contract.get("parent_contract") != PREPARATION_CONTRACT:
        raise GPUExperimentConfigError("GPU experiment parent contract changed")
    if int(contract.get("base_seed", -1)) != 20_260_818:
        raise GPUExperimentConfigError("GPU experiment base seed changed")
    if contract.get("research_status") != "exploratory_gpu_v1":
        raise GPUExperimentConfigError("GPU experiment must remain exploratory")
    if contract.get("label_frame") != "vive_world" or contract.get("label_unit") != "mm":
        raise GPUExperimentConfigError("GPU experiment label frame/unit changed")
    if contract.get("independent_test_available") is not False:
        raise GPUExperimentConfigError("GPU experiment cannot claim an independent test")
    if contract.get("pilot_observed_before_freeze") is not True:
        raise GPUExperimentConfigError("the disclosed pre-freeze pilot cannot be removed")

    if release.get("archive_sha256") != PINNED_RELEASE_SHA256:
        raise GPUExperimentConfigError("preparation release digest changed")
    if release.get("expected_contract") != PREPARATION_CONTRACT:
        raise GPUExperimentConfigError("preparation release contract changed")
    if (int(release.get("expected_canonical_rows", -1)), int(release.get("expected_train_eligible_rows", -1))) != (7707, 7702):
        raise GPUExperimentConfigError("preparation release counts changed")

    if task.get("input_view") != "fusion_v1" or int(task.get("expected_feature_count", -1)) != len(FUSION_COLUMNS):
        raise GPUExperimentConfigError("GPU input view changed")
    if tuple(task.get("labels", [])) != LABEL_COLUMNS:
        raise GPUExperimentConfigError("GPU labels changed")
    if task.get("recipe") != "real_only":
        raise GPUExperimentConfigError("GPU v1 must remain real-only")
    forbidden = {str(value) for value in task.get("inference_forbidden_tokens", [])}
    if forbidden != {"vive", "tracker", "planned", "target_id", "segment", "timestamp", "qc", "provenance"}:
        raise GPUExperimentConfigError("inference forbidden-token set changed")

    primary = _mapping(splits, "primary")
    secondary = _mapping(splits, "secondary")
    if primary != {"name": "segment_loso", "groups": [36, 37, 38, 39, 40]}:
        raise GPUExperimentConfigError("primary split changed")
    if secondary != {"name": "target_group", "groups": [0, 1, 2, 3, 4]}:
        raise GPUExperimentConfigError("secondary split changed")
    if splits.get("preprocessing_fit_scope") != "real_train_only" or splits.get("validation_and_test_real_only") is not True:
        raise GPUExperimentConfigError("real-only split boundary changed")
    if splits.get("outer_test_access") != "after_inner_selection_only":
        raise GPUExperimentConfigError("outer-test access policy changed")

    if set(models) != {"torch_mlp", "xgboost_cuda"}:
        raise GPUExperimentConfigError("GPU model family set changed")
    torch_model = _mapping(models, "torch_mlp")
    if torch_model.get("role") != "primary" or torch_model.get("hidden_layer_sizes") != [[64, 64], [128, 64]]:
        raise GPUExperimentConfigError("primary MLP role or architecture grid changed")
    if [float(value) for value in torch_model.get("weight_decay", [])] != [0.001, 0.0001]:
        raise GPUExperimentConfigError("MLP weight-decay grid changed")
    if (int(torch_model.get("epoch_budget", -1)), int(torch_model.get("ensemble_seed_count", -1)), int(torch_model.get("batch_size", -1))) != (400, 3, 128):
        raise GPUExperimentConfigError("MLP budget/ensemble/batch contract changed")
    if torch_model.get("deterministic_algorithms") is not True or torch_model.get("dtype") != "float32":
        raise GPUExperimentConfigError("MLP determinism or precision changed")
    xgb = _mapping(models, "xgboost_cuda")
    if xgb.get("role") != "reference_baseline" or xgb.get("device") != "cuda" or xgb.get("tree_method") != "hist":
        raise GPUExperimentConfigError("CUDA XGBoost role/backend changed")
    if int(xgb.get("n_estimators", -1)) != 500 or float(xgb.get("learning_rate", -1)) != 0.05:
        raise GPUExperimentConfigError("CUDA XGBoost budget changed")
    if len(gpu_model_grid(config, "torch_mlp")) != 4 or len(gpu_model_grid(config, "xgboost_cuda")) != 8:
        raise GPUExperimentConfigError("GPU hyperparameter grid cardinality changed")

    if preprocessing.get("target_scale") != "one_isotropic_scale" or preprocessing.get("preserve_axis_weighting") is not True:
        raise GPUExperimentConfigError("target preprocessing changed")
    if selection.get("metric") != "rmse_3d_mm" or selection.get("inner_aggregate") != "equal_validation_group_mean":
        raise GPUExperimentConfigError("inner selection metric changed")
    if selection.get("final_model_family") != "torch_mlp" or selection.get("outer_scores_not_used_for_final_config") is not True:
        raise GPUExperimentConfigError("final family or outer-score boundary changed")
    if int(monitoring.get("history_interval_epochs", -1)) != 10 or monitoring.get("require_finite_loss_and_predictions") is not True:
        raise GPUExperimentConfigError("monitoring cadence or finite-value gate changed")
    if int(runtime.get("gpu_count", -1)) != 4 or runtime.get("artificial_model_scaling_for_utilization") is not False:
        raise GPUExperimentConfigError("GPU scheduling contract changed")
    if runtime.get("torch_version") != "2.12.1+cu132" or runtime.get("xgboost_version") != "3.3.0":
        raise GPUExperimentConfigError("GPU package lock changed")
    if not all(bool(boundaries.get(key)) for key in (
        "not_robot_base_or_tcp_calibrated",
        "not_dynamic_state_estimation",
        "not_shapefk",
        "not_control_authority",
        "deployment_requires_independent_sessions_calibration_uncertainty_ood",
    )):
        raise GPUExperimentConfigError("claim boundary was weakened")

    digest = gpu_config_sha256(config)
    if digest != PINNED_GPU_CONFIG_SHA256:
        raise GPUExperimentConfigError(
            "resolved GPU experiment config differs from pinned canonical SHA-256 "
            f"{PINNED_GPU_CONFIG_SHA256}; received {digest}"
        )


def load_gpu_config(path: str | Path, *, release_override: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GPUExperimentConfigError(f"GPU config does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise GPUExperimentConfigError(f"invalid GPU config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GPUExperimentConfigError("GPU config root must be a mapping")
    payload["_config_path"] = str(config_path)
    payload["_workspace_root"] = str(config_path.parent.parent)
    if release_override is not None:
        payload["_release_override"] = str(Path(release_override).resolve())
    validate_gpu_config(payload)
    return payload


def resolve_gpu_release_path(config: dict[str, Any]) -> Path:
    if "_release_override" in config:
        return Path(str(config["_release_override"])).resolve()
    configured = Path(str(config["release"]["archive"]))
    if configured.is_absolute():
        return configured.resolve()
    return (Path(str(config["_workspace_root"])) / configured).resolve()


__all__ = [
    "GPU_EXPERIMENT_CONTRACT",
    "GPUExperimentConfigError",
    "PINNED_GPU_CONFIG_SHA256",
    "gpu_config_sha256",
    "gpu_model_grid",
    "load_gpu_config",
    "public_gpu_config",
    "resolve_gpu_release_path",
    "validate_gpu_config",
]
