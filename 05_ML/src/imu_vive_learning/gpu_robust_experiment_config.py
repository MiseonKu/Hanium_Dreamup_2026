"""Frozen configuration loader for the post-v1 robust GPU experiment."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import yaml

from .config import canonical_json_bytes
from .experiment_release import PINNED_RELEASE_SHA256, PREPARATION_CONTRACT
from .features import FUSION_COLUMNS, LABEL_COLUMNS
from .gpu_experiment_config import GPU_EXPERIMENT_CONTRACT, PINNED_GPU_CONFIG_SHA256


ROBUST_GPU_EXPERIMENT_CONTRACT = "alphaf.endpoint_regression.gpu_experiment.robust_v2"
GPU_ROBUST_EXPERIMENT_CONTRACT = ROBUST_GPU_EXPERIMENT_CONTRACT
PINNED_ROBUST_GPU_CONFIG_SHA256 = (
    "ed28b16cde0be391b1c893003a58bd26c4de57df144e6e1cdf64c7a1dca8339a"
)
PINNED_GPU_ROBUST_CONFIG_SHA256 = PINNED_ROBUST_GPU_CONFIG_SHA256
FUSION_V1_ALLOWLIST_SHA256 = (
    "1756b0aafb8ba4c9a740683d0a0d12236d8ee6e0065cfbe9fc6fb9d656345b6f"
)
HISTORICAL_V1_ROBUST_J_MM = 16.3801

_EXPECTED_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"candidate_id": "balanced_erm", "objective": "balanced_erm"},
    {
        "candidate_id": "group_dro_eta_0_01",
        "objective": "group_dro",
        "eta": 0.01,
    },
    {
        "candidate_id": "group_dro_eta_0_1",
        "objective": "group_dro",
        "eta": 0.1,
    },
    {
        "candidate_id": "v_rex_beta_10",
        "objective": "v_rex",
        "beta": 10.0,
        "warmup_epochs": 40,
    },
    {
        "candidate_id": "v_rex_beta_100",
        "objective": "v_rex",
        "beta": 100.0,
        "warmup_epochs": 40,
    },
    {
        "candidate_id": "balanced_erm_sam_rho_0_05",
        "objective": "balanced_erm",
        "sam": True,
        "rho": 0.05,
    },
)


class GPURobustExperimentConfigError(ValueError):
    """Raised when the frozen robust-v2 experiment contract has drifted."""


RobustGPUExperimentConfigError = GPURobustExperimentConfigError


def public_robust_gpu_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return only the material, hash-covered part of a resolved config."""

    return deepcopy({key: value for key, value in config.items() if not key.startswith("_")})


def robust_gpu_config_sha256(config: dict[str, Any]) -> str:
    """Return the canonical JSON SHA-256 used as the robust-v2 run identity."""

    return hashlib.sha256(canonical_json_bytes(public_robust_gpu_config(config))).hexdigest()


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise GPURobustExperimentConfigError(f"missing mapping section: {key}")
    return value


def robust_candidate_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the six candidates in their frozen final tie-break order."""

    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not all(
        isinstance(candidate, dict) for candidate in candidates
    ):
        raise GPURobustExperimentConfigError("candidates must be an ordered list of mappings")
    return deepcopy(candidates)


def validate_robust_gpu_config(config: dict[str, Any]) -> None:
    """Reject semantic or byte-independent material drift from robust v2."""

    contract = _mapping(config, "contract")
    lineage = _mapping(config, "lineage")
    release = _mapping(config, "release")
    task = _mapping(config, "task")
    splits = _mapping(config, "splits")
    models = _mapping(config, "models")
    training_protocol = _mapping(config, "training_protocol")
    preprocessing = _mapping(config, "preprocessing")
    selection = _mapping(config, "selection")
    evaluation = _mapping(config, "evaluation")
    confirmatory = _mapping(config, "confirmatory")
    monitoring = _mapping(config, "monitoring")
    runtime = _mapping(config, "runtime")
    boundaries = _mapping(config, "claim_boundaries")

    if contract.get("name") != ROBUST_GPU_EXPERIMENT_CONTRACT:
        raise GPURobustExperimentConfigError("robust GPU experiment contract name changed")
    if contract.get("parent_contract") != GPU_EXPERIMENT_CONTRACT:
        raise GPURobustExperimentConfigError("robust GPU parent contract changed")
    if int(contract.get("base_seed", -1)) != 20_260_818:
        raise GPURobustExperimentConfigError("robust GPU base seed changed")
    if contract.get("research_status") != "post_v1_robustness_development":
        raise GPURobustExperimentConfigError("robust v2 must remain a development study")
    if contract.get("label_frame") != "vive_world" or contract.get("label_unit") != "mm":
        raise GPURobustExperimentConfigError("robust GPU label frame/unit changed")
    if contract.get("post_v1_outer_observed") is not True:
        raise GPURobustExperimentConfigError("prior observation of v1 outer folds must be disclosed")
    if contract.get("independent_test_available") is not False:
        raise GPURobustExperimentConfigError("robust v2 cannot claim an independent test")

    if lineage.get("base_v1_config_sha256") != PINNED_GPU_CONFIG_SHA256:
        raise GPURobustExperimentConfigError("base v1 config digest changed")
    if lineage.get("preparation_release_sha256") != PINNED_RELEASE_SHA256:
        raise GPURobustExperimentConfigError("lineage preparation release digest changed")
    if release.get("archive_sha256") != PINNED_RELEASE_SHA256:
        raise GPURobustExperimentConfigError("preparation release digest changed")
    if release.get("expected_contract") != PREPARATION_CONTRACT:
        raise GPURobustExperimentConfigError("preparation release contract changed")
    if (
        int(release.get("expected_canonical_rows", -1)),
        int(release.get("expected_train_eligible_rows", -1)),
    ) != (7707, 7702):
        raise GPURobustExperimentConfigError("preparation release row counts changed")

    allowlist_digest = hashlib.sha256(
        canonical_json_bytes(list(FUSION_COLUMNS))
    ).hexdigest()
    if allowlist_digest != FUSION_V1_ALLOWLIST_SHA256:
        raise GPURobustExperimentConfigError("runtime fusion_v1 feature allow-list changed")
    if task.get("input_view") != "fusion_v1":
        raise GPURobustExperimentConfigError("robust GPU input view changed")
    if int(task.get("expected_feature_count", -1)) != len(FUSION_COLUMNS) or len(
        FUSION_COLUMNS
    ) != 43:
        raise GPURobustExperimentConfigError("robust GPU feature count changed")
    if task.get("feature_allowlist_sha256") != FUSION_V1_ALLOWLIST_SHA256:
        raise GPURobustExperimentConfigError("configured fusion_v1 allow-list digest changed")
    if tuple(task.get("labels", [])) != LABEL_COLUMNS:
        raise GPURobustExperimentConfigError("robust GPU labels changed")
    if task.get("recipe") != "real_only":
        raise GPURobustExperimentConfigError("robust v2 must remain real-only")
    forbidden = {str(value) for value in task.get("inference_forbidden_tokens", [])}
    if forbidden != {
        "vive",
        "tracker",
        "planned",
        "target_id",
        "segment",
        "timestamp",
        "qc",
        "provenance",
    }:
        raise GPURobustExperimentConfigError("inference forbidden-token set changed")

    if _mapping(splits, "primary") != {
        "name": "segment_loso",
        "groups": [36, 37, 38, 39, 40],
    }:
        raise GPURobustExperimentConfigError("primary segment LOSO split changed")
    if splits.get("nested_policy") != "each_non_test_group_once_as_validation":
        raise GPURobustExperimentConfigError("nested split policy changed")
    if int(splits.get("expected_inner_contexts_per_candidate", -1)) != 20:
        raise GPURobustExperimentConfigError("candidate inner-context count changed")
    if splits.get("optimization_group_key") != "segment_id":
        raise GPURobustExperimentConfigError("robust optimization group key changed")
    if splits.get("preprocessing_fit_scope") != "real_train_only":
        raise GPURobustExperimentConfigError("preprocessing fit scope changed")
    if splits.get("validation_and_diagnostic_real_only") is not True:
        raise GPURobustExperimentConfigError("validation/diagnostic real-only boundary changed")
    if splits.get("diagnostic_access") != "after_inner_selection_only":
        raise GPURobustExperimentConfigError("diagnostic access boundary changed")

    if set(models) != {"torch_mlp"}:
        raise GPURobustExperimentConfigError("robust v2 model family set changed")
    mlp = _mapping(models, "torch_mlp")
    expected_mlp = {
        "role": "sole_candidate_family",
        "output_strategy": "shared_xyz",
        "hidden_layer_sizes": [64, 64],
        "weight_decay": 0.001,
        "activation": "relu",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "batch_size": 128,
        "epoch_budget": 400,
        "ensemble_seed_count": 3,
        "dtype": "float32",
        "deterministic_algorithms": True,
        "checkpoint_rule": "minimum_inner_validation_rmse_3d_mm",
        "refit_epoch_rule": "rounded_median_inner_best_epoch_per_seed",
    }
    if mlp != expected_mlp:
        raise GPURobustExperimentConfigError("common robust MLP architecture or budget changed")

    if training_protocol != {
        "paired_candidate_model_seeds": True,
        "balanced_batch_rule": "floor_max_batch_rows_divided_by_training_group_count",
        "group_dro_update": "exponentiated_gradient_per_batch",
        "group_dro_weight_floor": 1.0e-12,
        "v_rex_schedule": "zero_beta_through_warmup_epochs_then_full_beta",
        "sam_variant": "nonadaptive_l2_two_pass",
        "inner_checkpoint_evaluation_cadence": "every_epoch",
        "unpromoted_candidate_default_inference": "forbidden",
    }:
        raise GPURobustExperimentConfigError("robust training protocol changed")

    candidates = robust_candidate_grid(config)
    if candidates != list(_EXPECTED_CANDIDATES):
        raise GPURobustExperimentConfigError("robust candidate values or listed order changed")
    if len({candidate["candidate_id"] for candidate in candidates}) != 6:
        raise GPURobustExperimentConfigError("robust candidate identifiers are not unique")

    if preprocessing != {
        "feature_center_scale": "train_mean_and_population_std",
        "target_center": "train_xyz_mean",
        "target_scale": "one_isotropic_scale",
        "preserve_axis_weighting": True,
    }:
        raise GPURobustExperimentConfigError("robust preprocessing changed")

    if int(selection.get("context_count_per_candidate", -1)) != 20:
        raise GPURobustExperimentConfigError("selection context count changed")
    if selection.get("score_name") != "robust_j_mm":
        raise GPURobustExperimentConfigError("robust selection score name changed")
    if _mapping(selection, "mean_component") != {
        "metric": "rmse_3d_mm",
        "aggregation": "mean_over_20_inner_contexts",
    }:
        raise GPURobustExperimentConfigError("selection mean component changed")
    if _mapping(selection, "worst_segment_component") != {
        "metric": "rmse_3d_mm",
        "within_segment_aggregation": "mean_over_4_inner_contexts",
        "across_segment_aggregation": "maximum",
    }:
        raise GPURobustExperimentConfigError("selection worst-segment component changed")
    if selection.get("tie_break_order") != [
        "mean_p95_3d_mm",
        "seed_dispersion_rmse_3d_mm",
        "listed_order",
    ]:
        raise GPURobustExperimentConfigError("selection tie-break order changed")
    if float(selection.get("historical_v1_j_mm", float("nan"))) != HISTORICAL_V1_ROBUST_J_MM:
        raise GPURobustExperimentConfigError("historical v1 robust J threshold changed")
    if float(selection.get("numeric_equality_abs_tolerance_mm", float("nan"))) != 1.0e-12:
        raise GPURobustExperimentConfigError("robust J numeric equality tolerance changed")
    if selection.get("promotion_rule") != "strict_candidate_j_less_than_historical_v1_j":
        raise GPURobustExperimentConfigError("candidate promotion rule changed")
    if selection.get("no_improvement_action") != "retain_v1":
        raise GPURobustExperimentConfigError("v1 retention fallback changed")
    if selection.get("diagnostic_scores_can_trigger_reselection") is not False:
        raise GPURobustExperimentConfigError("diagnostic reselection boundary was weakened")

    if evaluation != {
        "outer_designation": "reused_holdout_development_diagnostic",
        "outer_scores_are_independent_test": False,
        "selection_must_be_frozen_before_access": True,
        "reselection_after_outer_diagnostic": "forbidden",
    }:
        raise GPURobustExperimentConfigError("reused outer diagnostic policy changed")
    if confirmatory != {
        "occurs_after_candidate_and_model_freeze": True,
        "data_requirement": "independent_restart_sessions",
        "minimum_session_count": 10,
        "no_tuning_or_reselection_on_confirmatory_data": True,
        "required_before_confirmatory_or_deployment_claim": True,
    }:
        raise GPURobustExperimentConfigError("independent confirmatory-session policy changed")

    if int(monitoring.get("history_interval_epochs", -1)) != 10:
        raise GPURobustExperimentConfigError("monitoring cadence changed")
    if not all(
        monitoring.get(key) is True
        for key in (
            "require_finite_loss_and_predictions",
            "report_train_validation_gap",
            "report_per_segment_rmse_and_p95",
            "report_seed_dispersion",
            "require_durable_semantic_validation",
        )
    ):
        raise GPURobustExperimentConfigError("robust monitoring was weakened")
    if int(runtime.get("gpu_count", -1)) != 4:
        raise GPURobustExperimentConfigError("robust GPU count changed")
    if runtime.get("artificial_model_scaling_for_utilization") is not False:
        raise GPURobustExperimentConfigError("artificial model scaling cannot be enabled")
    if runtime.get("torch_version") != "2.12.1+cu132":
        raise GPURobustExperimentConfigError("robust PyTorch lock changed")

    if not all(
        boundaries.get(key) is True
        for key in (
            "not_robot_base_or_tcp_calibrated",
            "not_dynamic_state_estimation",
            "not_shapefk",
            "not_control_authority",
            "development_diagnostic_not_independent_confirmation",
            "deployment_requires_independent_sessions_calibration_uncertainty_ood",
        )
    ):
        raise GPURobustExperimentConfigError("robust-v2 claim boundary was weakened")

    digest = robust_gpu_config_sha256(config)
    if digest != PINNED_ROBUST_GPU_CONFIG_SHA256:
        raise GPURobustExperimentConfigError(
            "resolved robust GPU experiment config differs from pinned canonical SHA-256 "
            f"{PINNED_ROBUST_GPU_CONFIG_SHA256}; received {digest}"
        )


def load_robust_gpu_config(
    path: str | Path,
    *,
    release_override: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate the pinned robust-v2 YAML contract."""

    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GPURobustExperimentConfigError(
            f"robust GPU config does not exist: {config_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise GPURobustExperimentConfigError(
            f"invalid robust GPU config {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise GPURobustExperimentConfigError("robust GPU config root must be a mapping")
    payload["_config_path"] = str(config_path)
    payload["_workspace_root"] = str(config_path.parent.parent)
    if release_override is not None:
        payload["_release_override"] = str(Path(release_override).resolve())
    validate_robust_gpu_config(payload)
    return payload


def resolve_robust_gpu_release_path(config: dict[str, Any]) -> Path:
    """Resolve the preparation release path, honoring a non-material override."""

    if "_release_override" in config:
        return Path(str(config["_release_override"])).resolve()
    configured = Path(str(config["release"]["archive"]))
    if configured.is_absolute():
        return configured.resolve()
    return (Path(str(config["_workspace_root"])) / configured).resolve()


# Natural aliases for callers that put ``gpu`` before ``robust``.
gpu_robust_config_sha256 = robust_gpu_config_sha256
validate_gpu_robust_config = validate_robust_gpu_config
load_gpu_robust_config = load_robust_gpu_config
resolve_gpu_robust_release_path = resolve_robust_gpu_release_path


__all__ = [
    "FUSION_V1_ALLOWLIST_SHA256",
    "GPU_ROBUST_EXPERIMENT_CONTRACT",
    "GPURobustExperimentConfigError",
    "HISTORICAL_V1_ROBUST_J_MM",
    "PINNED_GPU_ROBUST_CONFIG_SHA256",
    "PINNED_ROBUST_GPU_CONFIG_SHA256",
    "ROBUST_GPU_EXPERIMENT_CONTRACT",
    "RobustGPUExperimentConfigError",
    "gpu_robust_config_sha256",
    "load_gpu_robust_config",
    "load_robust_gpu_config",
    "public_robust_gpu_config",
    "resolve_gpu_robust_release_path",
    "resolve_robust_gpu_release_path",
    "robust_candidate_grid",
    "robust_gpu_config_sha256",
    "validate_gpu_robust_config",
    "validate_robust_gpu_config",
]
