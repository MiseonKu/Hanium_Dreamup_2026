"""Frozen configuration loader for the prospective outage-v3 GPU study."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any

import yaml

from .config import canonical_json_bytes
from .experiment_release import PINNED_RELEASE_SHA256, PREPARATION_CONTRACT
from .features import FUSION_COLUMNS, LABEL_COLUMNS
from .gpu_robust_experiment_config import (
    FUSION_V1_ALLOWLIST_SHA256,
    PINNED_ROBUST_GPU_CONFIG_SHA256,
    ROBUST_GPU_EXPERIMENT_CONTRACT,
)


OUTAGE_GPU_EXPERIMENT_CONTRACT = "alphaf.endpoint_regression.gpu_experiment.outage_v3"
PINNED_OUTAGE_GPU_CONFIG_SHA256 = (
    "4113628abf72f0e0ba911c100bdf3290543a723d9d4d580c3f3397affb61b6a7"
)
PINNED_SOURCE_ARCHIVE_SHA256 = (
    "ceadf164a15a3ce302f8b2ffdde450df0f6d3a91c3b0c258a28acb5de9b92d64"
)
PINNED_SOURCE_MEMBER_SHA256 = (
    "1b0a821a1822f237adadb1c978ea73da02b0e4bb7f099ce085a63931e9792b54"
)
PINNED_CHALLENGE_UID_SHA256 = (
    "fa24e77ba5ce5f333bc4bfc5d7b75dd567d098651493ec89cda8dfbe9f12c0a1"
)
AVAILABILITY_FEATURES = (
    "imu0_base_available",
    "imu1_link_root_available",
    "imu2_link_mid_available",
    "imu3_tip_available",
)
OUTAGE_CANDIDATE_IDS = (
    "masked_mlp",
    "residual_mlp",
    "tabm_batchensemble",
    "chain_endpoint_only",
    "chain_masked_reconstruction",
    "equal_blend_tabm_chain_reconstruction",
)
TRAINABLE_OUTAGE_CANDIDATE_IDS = OUTAGE_CANDIDATE_IDS[:-1]


class GPUOutageExperimentConfigError(ValueError):
    """Raised when the pre-result outage-v3 contract has drifted."""


def public_outage_gpu_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the hash-covered material configuration."""

    return deepcopy({key: value for key, value in config.items() if not key.startswith("_")})


def outage_gpu_config_sha256(config: dict[str, Any]) -> str:
    """Return the canonical JSON configuration identity."""

    return hashlib.sha256(canonical_json_bytes(public_outage_gpu_config(config))).hexdigest()


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise GPUOutageExperimentConfigError(f"missing mapping section: {key}")
    return value


def outage_candidate_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidates in their frozen tie-break order."""

    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not all(
        isinstance(candidate, dict) for candidate in candidates
    ):
        raise GPUOutageExperimentConfigError("candidates must be an ordered list of mappings")
    return deepcopy(candidates)


def validate_outage_gpu_config(config: dict[str, Any]) -> None:
    """Reject semantic drift before a development fit or challenge access."""

    contract = _mapping(config, "contract")
    lineage = _mapping(config, "lineage")
    development_release = _mapping(config, "development_release")
    source = _mapping(config, "source")
    row_roles = _mapping(config, "row_roles")
    eligibility = _mapping(config, "label_blind_eligibility")
    task = _mapping(config, "task")
    layout = _mapping(config, "sensor_layout")
    preprocessing = _mapping(config, "preprocessing")
    masking = _mapping(config, "masking")
    splits = _mapping(config, "splits")
    training = _mapping(config, "training")
    selection = _mapping(config, "selection")
    final_refit = _mapping(config, "final_refit")
    challenge = _mapping(config, "challenge")
    runtime = _mapping(config, "runtime")
    boundaries = _mapping(config, "claim_boundaries")

    if contract.get("name") != OUTAGE_GPU_EXPERIMENT_CONTRACT:
        raise GPUOutageExperimentConfigError("outage-v3 contract name changed")
    if contract.get("parent_contract") != ROBUST_GPU_EXPERIMENT_CONTRACT:
        raise GPUOutageExperimentConfigError("outage-v3 parent contract changed")
    if int(contract.get("base_seed", -1)) != 20_260_824:
        raise GPUOutageExperimentConfigError("outage-v3 base seed changed")
    if contract.get("label_frame") != "vive_world" or contract.get("label_unit") != "mm":
        raise GPUOutageExperimentConfigError("outage-v3 label frame/unit changed")
    if contract.get("frozen_before_challenge_label_access") is not True:
        raise GPUOutageExperimentConfigError("challenge label seal was weakened")
    if contract.get("confirmatory_claim_allowed") is not False:
        raise GPUOutageExperimentConfigError("outage-v3 cannot permit a confirmatory claim")

    if lineage.get("robust_v2_config_sha256") != PINNED_ROBUST_GPU_CONFIG_SHA256:
        raise GPUOutageExperimentConfigError("robust-v2 lineage digest changed")
    if lineage.get("preparation_release_sha256") != PINNED_RELEASE_SHA256:
        raise GPUOutageExperimentConfigError("preparation lineage digest changed")
    if development_release.get("archive_sha256") != PINNED_RELEASE_SHA256:
        raise GPUOutageExperimentConfigError("development release digest changed")
    if development_release.get("expected_contract") != PREPARATION_CONTRACT:
        raise GPUOutageExperimentConfigError("development release contract changed")
    if (
        int(development_release.get("expected_canonical_rows", -1)),
        int(development_release.get("expected_train_eligible_rows", -1)),
    ) != (7707, 7702):
        raise GPUOutageExperimentConfigError("development release row counts changed")

    if source.get("archive_sha256") != PINNED_SOURCE_ARCHIVE_SHA256:
        raise GPUOutageExperimentConfigError("challenge source archive digest changed")
    if source.get("member_sha256") != PINNED_SOURCE_MEMBER_SHA256:
        raise GPUOutageExperimentConfigError("challenge source member digest changed")
    if int(source.get("expected_rows", -1)) != 10_055:
        raise GPUOutageExperimentConfigError("challenge source row count changed")
    if source.get("immutable_read_only") is not True:
        raise GPUOutageExperimentConfigError("challenge source must remain read-only")

    development = _mapping(row_roles, "development")
    frozen_challenge = _mapping(row_roles, "challenge")
    if development.get("segments") != [36, 37, 38, 39] or int(
        development.get("expected_rows", -1)
    ) != 6635:
        raise GPUOutageExperimentConfigError("development population changed")
    if development.get("expected_segment_rows") != {
        36: 1672,
        37: 1597,
        38: 1707,
        39: 1659,
    }:
        raise GPUOutageExperimentConfigError("development segment counts changed")
    if development.get("sorted_trial_uid_hash_algorithm") != (
        "sha256_of_newline_joined_lexicographically_sorted_strings"
    ) or development.get("sorted_trial_uid_sha256") != (
        "f8ef6f2ea62aa4e3591e8ea10fb195ade5fc0804342a5925a420039b2fd33ad0"
    ):
        raise GPUOutageExperimentConfigError("development trial UID identity changed")
    if frozen_challenge.get("segments") != [40, 41]:
        raise GPUOutageExperimentConfigError("challenge segments changed")
    if int(frozen_challenge.get("expected_rows", -1)) != 3295:
        raise GPUOutageExperimentConfigError("challenge row count changed")
    if frozen_challenge.get("expected_segment_rows") != {40: 1672, 41: 1623}:
        raise GPUOutageExperimentConfigError("challenge segment counts changed")
    if frozen_challenge.get("ordered_trial_uid_sha256") != PINNED_CHALLENGE_UID_SHA256:
        raise GPUOutageExperimentConfigError("challenge identity digest changed")
    if frozen_challenge.get("force_outage_sensor") != "imu1_link_root":
        raise GPUOutageExperimentConfigError("challenge outage sensor changed")
    if row_roles.get("no_synthetic_rows") is not True:
        raise GPUOutageExperimentConfigError("outage-v3 must remain measured-real only")

    if eligibility.get("duration_s") != [2.8, 3.2]:
        raise GPUOutageExperimentConfigError("duration eligibility changed")
    if eligibility.get("encoder_active_ranges") != [
        [1160.0, 3083.0],
        [1202.0, 3186.0],
        [1937.0, 3968.0],
    ]:
        raise GPUOutageExperimentConfigError("encoder active ranges changed")
    if eligibility.get("challenge_exempts_only") != [
        "imu1_link_root_feature_finiteness",
        "imu1_link_root_sample_count",
    ]:
        raise GPUOutageExperimentConfigError("challenge eligibility exemptions changed")

    runtime_fusion_digest = hashlib.sha256(canonical_json_bytes(list(FUSION_COLUMNS))).hexdigest()
    if runtime_fusion_digest != FUSION_V1_ALLOWLIST_SHA256:
        raise GPUOutageExperimentConfigError("runtime fusion_v1 allow-list changed")
    model_allowlist = list(FUSION_COLUMNS) + list(AVAILABILITY_FEATURES)
    model_allowlist_digest = hashlib.sha256(canonical_json_bytes(model_allowlist)).hexdigest()
    if task.get("sensor_feature_allowlist_sha256") != runtime_fusion_digest:
        raise GPUOutageExperimentConfigError("configured sensor allow-list digest changed")
    if task.get("model_input_allowlist_sha256") != model_allowlist_digest:
        raise GPUOutageExperimentConfigError("configured model allow-list digest changed")
    if int(task.get("sensor_feature_count", -1)) != 43 or int(
        task.get("model_input_feature_count", -1)
    ) != 47:
        raise GPUOutageExperimentConfigError("outage-v3 input width changed")
    if tuple(task.get("availability_features", [])) != AVAILABILITY_FEATURES:
        raise GPUOutageExperimentConfigError("availability feature order changed")
    if tuple(task.get("labels", [])) != LABEL_COLUMNS:
        raise GPUOutageExperimentConfigError("outage-v3 labels changed")
    forbidden = {str(value) for value in task.get("inference_forbidden_tokens", [])}
    if forbidden != {
        "vive", "tracker", "planned", "target_id", "segment", "timestamp", "qc", "provenance"
    }:
        raise GPUOutageExperimentConfigError("inference forbidden-token set changed")

    if layout.get("ordered_sensors") != [
        "imu0_base", "imu1_link_root", "imu2_link_mid", "imu3_tip"
    ] or int(layout.get("per_sensor_feature_count", -1)) != 10:
        raise GPUOutageExperimentConfigError("physical sensor layout changed")
    if layout.get("sensor_major_reshape_of_fusion_v1_forbidden") is not True:
        raise GPUOutageExperimentConfigError("unsafe sensor-major reshape was enabled")
    if preprocessing.get("unavailable_sensor_imputation") != "exact_zero_after_standardization":
        raise GPUOutageExperimentConfigError("outage imputation rule changed")
    if preprocessing.get("challenge_statistics_forbidden") is not True:
        raise GPUOutageExperimentConfigError("challenge preprocessing boundary weakened")

    states = masking.get("states")
    if not isinstance(states, list) or len(states) != 5:
        raise GPUOutageExperimentConfigError("masking state grid changed")
    probabilities = [float(state.get("probability", math.nan)) for state in states]
    if not all(math.isfinite(value) and value > 0 for value in probabilities) or not math.isclose(
        sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-15
    ):
        raise GPUOutageExperimentConfigError("mask probabilities must be positive and sum to one")
    if probabilities != [0.25, 0.1875, 0.1875, 0.1875, 0.1875]:
        raise GPUOutageExperimentConfigError("mask probabilities changed")
    if masking.get("challenge_availability") != [1, 0, 1, 1]:
        raise GPUOutageExperimentConfigError("challenge availability changed")
    if masking.get("paired_mask_schedule_across_trainable_candidates") is not True:
        raise GPUOutageExperimentConfigError("paired mask schedule was disabled")

    if splits.get("development_segments") != [36, 37, 38, 39]:
        raise GPUOutageExperimentConfigError("development split groups changed")
    if int(splits.get("expected_contexts_per_candidate", -1)) != 16:
        raise GPUOutageExperimentConfigError("development context count changed")
    if splits.get("challenge_access_after_selection_only") is not True:
        raise GPUOutageExperimentConfigError("challenge access boundary weakened")

    if training.get("optimizer") != "AdamW" or int(training.get("ensemble_seed_count", -1)) != 3:
        raise GPUOutageExperimentConfigError("training optimizer or seed count changed")
    if training.get("paired_seed_values") != [20260824, 20260825, 20260826]:
        raise GPUOutageExperimentConfigError("paired model seeds changed")
    if training.get("model_seed_policy") != "use_paired_seed_value_directly":
        raise GPUOutageExperimentConfigError("model seed policy changed")
    if (
        float(training.get("learning_rate", math.nan)),
        float(training.get("weight_decay", math.nan)),
        int(training.get("batch_size", -1)),
        int(training.get("epoch_budget", -1)),
    ) != (0.001, 0.001, 128, 400):
        raise GPUOutageExperimentConfigError("common training budget changed")
    expected_training_policies = {
        "sampling_policy": "deterministic_segment_balanced_largest_group_epoch",
        "rows_per_segment_per_batch": (
            "floor_batch_size_divided_by_active_segment_count"
        ),
        "smaller_segment_policy": (
            "deterministic_cycle_to_largest_group_epoch_length"
        ),
        "endpoint_segment_weighting": "equal_mean_over_active_training_segments",
        "equal_checkpoint_rmse_tie": "earliest_epoch",
    }
    if any(training.get(key) != value for key, value in expected_training_policies.items()):
        raise GPUOutageExperimentConfigError("training sampling/checkpoint policy changed")

    candidates = outage_candidate_grid(config)
    candidate_ids = tuple(str(candidate.get("candidate_id")) for candidate in candidates)
    if candidate_ids != OUTAGE_CANDIDATE_IDS:
        raise GPUOutageExperimentConfigError("candidate identities or tie order changed")
    if tuple(
        str(candidate["candidate_id"])
        for candidate in candidates
        if candidate.get("kind") == "trainable"
    ) != TRAINABLE_OUTAGE_CANDIDATE_IDS:
        raise GPUOutageExperimentConfigError("trainable candidate set changed")
    if float(candidates[4].get("masked_reconstruction_loss_weight", math.nan)) != 0.1:
        raise GPUOutageExperimentConfigError("masked reconstruction weight changed")
    if candidates[5].get("components") != [
        {"candidate_id": "tabm_batchensemble", "weight": 0.5},
        {"candidate_id": "chain_masked_reconstruction", "weight": 0.5},
    ]:
        raise GPUOutageExperimentConfigError("fixed blend definition changed")

    if (
        int(selection.get("scored_candidate_count", -1)),
        int(selection.get("trained_candidate_count", -1)),
        int(selection.get("context_count_per_candidate", -1)),
        int(selection.get("trainable_seeded_fit_count", -1)),
    ) != (6, 5, 16, 240):
        raise GPUOutageExperimentConfigError("selection task counts changed")
    if selection.get("score_name") != "outage_robust_j_mm":
        raise GPUOutageExperimentConfigError("selection score changed")
    if selection.get("context_prediction_space") != (
        "inverse_transformed_vive_world_xyz_mm"
    ):
        raise GPUOutageExperimentConfigError("context prediction space changed")
    if selection.get("virtual_blend_seed_pairing") != (
        "same_paired_seed_index_before_seed_dispersion"
    ):
        raise GPUOutageExperimentConfigError("virtual blend seed pairing changed")
    if selection.get("nonfinite_context_score") != "positive_infinity":
        raise GPUOutageExperimentConfigError("non-finite context score policy changed")
    if selection.get("missing_context_action") != "fail_closed_before_scoring":
        raise GPUOutageExperimentConfigError("missing context action changed")
    if selection.get("challenge_scores_can_trigger_selection_or_reselection") is not False:
        raise GPUOutageExperimentConfigError("challenge reselection was enabled")
    if final_refit.get("challenge_rows_in_training") != "forbidden":
        raise GPUOutageExperimentConfigError("challenge rows may not enter final training")
    if final_refit.get("epoch_median_scope") != (
        "per_candidate_per_paired_seed_over_16_contexts"
    ):
        raise GPUOutageExperimentConfigError("final-refit epoch median scope changed")
    if final_refit.get("epoch_rounding") != "half_up":
        raise GPUOutageExperimentConfigError("final-refit epoch rounding changed")
    if final_refit.get("mask_draw_held_out_segment_token") != (
        "final_refit_all_development"
    ):
        raise GPUOutageExperimentConfigError("final-refit mask context token changed")
    if final_refit.get("mask_draw_forced_outage_sensor") != "imu1_link_root":
        raise GPUOutageExperimentConfigError("final-refit mask outage key changed")
    if final_refit.get("mask_draw_reuses_training_key_recipe") is not True:
        raise GPUOutageExperimentConfigError("final-refit mask key recipe changed")
    if final_refit.get("exact_rounded_median_epoch_final_state") is not True:
        raise GPUOutageExperimentConfigError("final-refit final-epoch rule changed")
    if challenge.get("maximum_source_label_access_passes") != 1:
        raise GPUOutageExperimentConfigError("challenge label access count changed")
    if challenge.get("evaluate_selected_frozen_model_only") is not True:
        raise GPUOutageExperimentConfigError("non-selected challenge evaluation was enabled")
    if (
        int(challenge.get("primary_previously_unused_segment", -1)),
        int(challenge.get("secondary_previously_observed_segment", -1)),
    ) != (41, 40):
        raise GPUOutageExperimentConfigError("challenge segment interpretation changed")
    if challenge.get("combined_two_segment_metrics_are_descriptive_only") is not True:
        raise GPUOutageExperimentConfigError("combined challenge metrics claim changed")
    if challenge.get("freshness_claim_applies_only_to_segment_41") is not True:
        raise GPUOutageExperimentConfigError("challenge freshness boundary changed")
    if challenge.get("label_seal_scope") != (
        "current_v3_raw_source_coordinate_projection_pass"
    ):
        raise GPUOutageExperimentConfigError("challenge label seal scope changed")
    if challenge.get(
        "parent_release_transiently_deserializes_historical_segment_40_labels"
    ) is not True or challenge.get(
        "development_api_returns_segment_40_rows_or_labels"
    ) is not False:
        raise GPUOutageExperimentConfigError("historical segment-40 boundary changed")
    if (
        runtime.get("gpu_count") != 4
        or runtime.get("python") != "3.12"
        or runtime.get("artificial_model_scaling_for_utilization") is not False
    ):
        raise GPUOutageExperimentConfigError("runtime GPU or model-scaling contract changed")
    if not all(
        boundaries.get(key) is True
        for key in (
            "not_independent_session_confirmation",
            "not_four_imu_confirmation",
            "not_robot_base_or_tcp_calibrated",
            "not_dynamic_state_estimation",
            "not_shapefk_or_formal_imu_only",
            "not_deployment_ready",
            "not_control_authority",
        )
    ):
        raise GPUOutageExperimentConfigError("outage-v3 claim boundary was weakened")

    digest = outage_gpu_config_sha256(config)
    if digest != PINNED_OUTAGE_GPU_CONFIG_SHA256:
        raise GPUOutageExperimentConfigError(
            "resolved outage-v3 config differs from pinned canonical SHA-256 "
            f"{PINNED_OUTAGE_GPU_CONFIG_SHA256}; received {digest}"
        )


def load_outage_gpu_config(
    path: str | Path,
    *,
    development_release_override: str | Path | None = None,
    source_override: str | Path | None = None,
) -> dict[str, Any]:
    """Load the pre-result frozen outage-v3 YAML contract."""

    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GPUOutageExperimentConfigError(f"outage-v3 config does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise GPUOutageExperimentConfigError(f"invalid outage-v3 config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GPUOutageExperimentConfigError("outage-v3 config root must be a mapping")
    payload["_config_path"] = str(config_path)
    payload["_workspace_root"] = str(config_path.parent.parent)
    if development_release_override is not None:
        payload["_development_release_override"] = str(Path(development_release_override).resolve())
    if source_override is not None:
        payload["_source_override"] = str(Path(source_override).resolve())
    validate_outage_gpu_config(payload)
    return payload


def _resolve_path(config: dict[str, Any], section: str, override: str) -> Path:
    if override in config:
        return Path(str(config[override])).resolve()
    configured = Path(str(config[section]["archive"]))
    if configured.is_absolute():
        return configured.resolve()
    return (Path(str(config["_workspace_root"])) / configured).resolve()


def resolve_outage_development_release_path(config: dict[str, Any]) -> Path:
    return _resolve_path(config, "development_release", "_development_release_override")


def resolve_outage_source_path(config: dict[str, Any]) -> Path:
    return _resolve_path(config, "source", "_source_override")


__all__ = [
    "AVAILABILITY_FEATURES",
    "GPUOutageExperimentConfigError",
    "OUTAGE_CANDIDATE_IDS",
    "OUTAGE_GPU_EXPERIMENT_CONTRACT",
    "PINNED_CHALLENGE_UID_SHA256",
    "PINNED_OUTAGE_GPU_CONFIG_SHA256",
    "PINNED_SOURCE_ARCHIVE_SHA256",
    "PINNED_SOURCE_MEMBER_SHA256",
    "TRAINABLE_OUTAGE_CANDIDATE_IDS",
    "load_outage_gpu_config",
    "outage_candidate_grid",
    "outage_gpu_config_sha256",
    "public_outage_gpu_config",
    "resolve_outage_development_release_path",
    "resolve_outage_source_path",
    "validate_outage_gpu_config",
]
