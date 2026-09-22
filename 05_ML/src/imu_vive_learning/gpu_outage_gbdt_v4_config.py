"""Configuration boundary for the post-v3 GBDT outage follow-up.

Unlike outage-v3, the static v4 YAML intentionally does not contain a guessed
incumbent identity.  ``prepare`` materialises those result-dependent hashes
only after a complete, still-sealed v3 development run has validated.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import canonical_json_bytes


GBDT_V4_CONTRACT = "alphaf.endpoint_regression.gpu_experiment.outage_v4_gbdt"
GBDT_V4_PARENT_CONTRACT = "alphaf.endpoint_regression.gpu_experiment.outage_v3"
GBDT_V4_CANDIDATE_IDS = (
    "v3_incumbent",
    "xgboost_mask_replicated",
    "equal_blend_incumbent_xgboost",
)
GBDT_V4_PAIRED_SEEDS = (20_260_824, 20_260_825, 20_260_826)
GBDT_V4_DEVELOPMENT_SEGMENTS = (36, 37, 38, 39)
GBDT_V4_FORCED_OUTAGES = (
    "imu0_base",
    "imu1_link_root",
    "imu2_link_mid",
    "imu3_tip",
)
_INTERNAL_CONFIG_KEYS = frozenset({"_config_path", "_workspace_root"})


class GPUOutageGBDTV4ConfigError(ValueError):
    """Raised when the v4 static contract has semantic drift."""


def public_gbdt_v4_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only hash-covered static configuration fields."""

    if any(type(key) is not str for key in config):
        raise GPUOutageGBDTV4ConfigError("v4 config keys must be exact strings")
    private = {key for key in config if key.startswith("_")}
    if private and private != _INTERNAL_CONFIG_KEYS:
        raise GPUOutageGBDTV4ConfigError("v4 internal config keys changed")
    if private:
        if any(type(config[key]) is not str for key in _INTERNAL_CONFIG_KEYS):
            raise GPUOutageGBDTV4ConfigError(
                "v4 internal config paths must be exact strings"
            )
        config_path = Path(config["_config_path"]).resolve()
        workspace_root = Path(config["_workspace_root"]).resolve()
        if workspace_root != config_path.parent.parent:
            raise GPUOutageGBDTV4ConfigError(
                "v4 internal config/workspace path binding changed"
            )
    return deepcopy({key: value for key, value in config.items() if key not in private})


def gbdt_v4_config_sha256(config: Mapping[str, Any]) -> str:
    """Return the canonical JSON identity of the static v4 contract."""

    return hashlib.sha256(canonical_json_bytes(public_gbdt_v4_config(config))).hexdigest()


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise GPUOutageGBDTV4ConfigError(f"missing mapping section: {key}")
    return value


def _require_exact(value: Any, expected: Any, *, name: str) -> None:
    """Require a type-exact, recursively closed-world YAML value.

    Plain Python equality is not sufficient for a frozen experiment contract:
    ``True == 1`` and ``1 == 1.0``.  Reject those substitutions as well as
    missing/extra mapping keys and reordered list members.
    """

    def matches(actual: Any, frozen: Any) -> bool:
        if isinstance(frozen, Mapping):
            if not isinstance(actual, Mapping) or len(actual) != len(frozen):
                return False
            actual_keys = tuple(actual.keys())
            for frozen_key, frozen_value in frozen.items():
                exact_keys = [
                    key
                    for key in actual_keys
                    if type(key) is type(frozen_key) and key == frozen_key
                ]
                if len(exact_keys) != 1 or not matches(
                    actual[exact_keys[0]], frozen_value
                ):
                    return False
            return True
        if isinstance(frozen, list):
            return (
                isinstance(actual, list)
                and len(actual) == len(frozen)
                and all(matches(left, right) for left, right in zip(actual, frozen))
            )
        return type(actual) is type(frozen) and actual == frozen

    if not matches(value, expected):
        raise GPUOutageGBDTV4ConfigError(f"v4 {name} contract changed")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys before validation."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    seen: list[Any] = []
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = any(existing == key for existing in seen)
        except Exception as exc:  # pragma: no cover - malformed exotic YAML key
            raise GPUOutageGBDTV4ConfigError(
                "v4 config contains an unsupported mapping key"
            ) from exc
        if duplicate:
            raise GPUOutageGBDTV4ConfigError(
                f"v4 config contains duplicate mapping key: {key!r}"
            )
        try:
            result[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as exc:
            raise GPUOutageGBDTV4ConfigError(
                "v4 config contains an unhashable mapping key"
            ) from exc
        seen.append(key)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def gbdt_v4_candidate_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not all(
        isinstance(candidate, Mapping) for candidate in candidates
    ):
        raise GPUOutageGBDTV4ConfigError("candidates must be an ordered list of mappings")
    return [deepcopy(dict(candidate)) for candidate in candidates]


def validate_gbdt_v4_config(config: Mapping[str, Any]) -> None:
    """Reject changes to the predeclared v4 comparison and label boundary."""

    public = public_gbdt_v4_config(config)
    expected_root_keys = {
        "contract",
        "incumbent_freeze",
        "data_roles",
        "task",
        "mask_replication",
        "xgboost",
        "splits",
        "candidates",
        "selection",
        "final_refit",
        "execution",
        "provenance",
        "challenge_report",
        "claim_boundaries",
    }
    if set(public) != expected_root_keys:
        raise GPUOutageGBDTV4ConfigError("v4 config root keys changed")

    contract = _mapping(config, "contract")
    incumbent = _mapping(config, "incumbent_freeze")
    roles = _mapping(config, "data_roles")
    task = _mapping(config, "task")
    replication = _mapping(config, "mask_replication")
    xgb = _mapping(config, "xgboost")
    splits = _mapping(config, "splits")
    selection = _mapping(config, "selection")
    final_refit = _mapping(config, "final_refit")
    execution = _mapping(config, "execution")
    provenance = _mapping(config, "provenance")
    report = _mapping(config, "challenge_report")
    boundaries = _mapping(config, "claim_boundaries")

    _require_exact(
        dict(contract),
        {
            "name": GBDT_V4_CONTRACT,
            "parent_contract": GBDT_V4_PARENT_CONTRACT,
            "base_seed": GBDT_V4_PAIRED_SEEDS[0],
            "research_status": "post_v3_development_gbdt_followup_on_reused_same_session",
            "label_frame": "vive_world",
            "label_unit": "mm",
            "frozen_before_challenge_label_access": True,
            "independent_session_available": False,
            "confirmatory_claim_allowed": False,
            "development_only": True,
            "challenge_gate_owner": "outage_v5_separate_sole_owner_contract",
        },
        name="contract section",
    )

    if contract.get("name") != GBDT_V4_CONTRACT:
        raise GPUOutageGBDTV4ConfigError("v4 contract name changed")
    if contract.get("parent_contract") != GBDT_V4_PARENT_CONTRACT:
        raise GPUOutageGBDTV4ConfigError("v4 parent contract changed")
    if int(contract.get("base_seed", -1)) != GBDT_V4_PAIRED_SEEDS[0]:
        raise GPUOutageGBDTV4ConfigError("v4 base seed changed")
    if contract.get("label_frame") != "vive_world" or contract.get("label_unit") != "mm":
        raise GPUOutageGBDTV4ConfigError("v4 label frame/unit changed")
    if contract.get("frozen_before_challenge_label_access") is not True:
        raise GPUOutageGBDTV4ConfigError("v4 challenge label freeze was weakened")
    if contract.get("confirmatory_claim_allowed") is not False:
        raise GPUOutageGBDTV4ConfigError("v4 cannot permit a confirmatory claim")
    if (
        contract.get("development_only") is not True
        or contract.get("challenge_gate_owner")
        != "outage_v5_separate_sole_owner_contract"
    ):
        raise GPUOutageGBDTV4ConfigError("v4 challenge-gate ownership changed")

    expected_bindings = [
        "v3_resolved_config_sha256",
        "v3_development_manifest_sha256",
        "v3_validator_payload_sha256",
        "v3_selection_sha256",
        "v3_selected_candidate_id",
        "v3_240_task_prediction_tree_sha256",
    ]
    _require_exact(
        dict(incumbent),
        {
            "prepare_command_required": True,
            "v3_required_stage": "development_complete_challenge_labels_sealed",
            "v3_challenge_label_access_marker_must_be_absent": True,
            "bindings_materialized_after_v3_completion": expected_bindings,
            "task_prediction_tree_expected_tasks": 240,
            "selected_incumbent_seed_predictions_snapshotted": True,
            "development_replay_snapshotted": True,
            "label_blind_challenge_identity_snapshotted": True,
            "v3_run_remains_unmodified": True,
            "deferred_values_must_not_be_guessed_in_static_config": True,
        },
        name="incumbent freeze section",
    )
    if incumbent.get("prepare_command_required") is not True:
        raise GPUOutageGBDTV4ConfigError("v3 incumbent prepare/freeze was disabled")
    if incumbent.get("v3_required_stage") != "development_complete_challenge_labels_sealed":
        raise GPUOutageGBDTV4ConfigError("v3 required stage changed")
    if incumbent.get("bindings_materialized_after_v3_completion") != expected_bindings:
        raise GPUOutageGBDTV4ConfigError("v3 incumbent binding set/order changed")
    if int(incumbent.get("task_prediction_tree_expected_tasks", -1)) != 240:
        raise GPUOutageGBDTV4ConfigError("v3 task/prediction tree count changed")
    if not all(
        incumbent.get(key) is True
        for key in (
            "v3_challenge_label_access_marker_must_be_absent",
            "selected_incumbent_seed_predictions_snapshotted",
            "development_replay_snapshotted",
            "label_blind_challenge_identity_snapshotted",
            "v3_run_remains_unmodified",
            "deferred_values_must_not_be_guessed_in_static_config",
        )
    ):
        raise GPUOutageGBDTV4ConfigError("v3 incumbent immutability boundary weakened")

    normalized_roles = deepcopy(dict(roles))
    for key in ("development_segment_rows", "challenge_segment_rows"):
        raw_rows = _mapping(roles, key)
        try:
            normalized_rows = {
                int(segment): rows for segment, rows in raw_rows.items()
            }
        except (TypeError, ValueError) as exc:
            raise GPUOutageGBDTV4ConfigError(
                f"v4 {key} keys must be integer segment IDs"
            ) from exc
        if len(normalized_rows) != len(raw_rows) or any(
            not (
                type(segment) is int
                or (isinstance(segment, str) and segment == str(int(segment)))
            )
            for segment in raw_rows
        ):
            raise GPUOutageGBDTV4ConfigError(
                f"v4 {key} keys changed"
            )
        normalized_roles[key] = normalized_rows
    _require_exact(
        normalized_roles,
        {
            "development_segments": [36, 37, 38, 39],
            "development_rows": 6635,
            "development_segment_rows": {36: 1672, 37: 1597, 38: 1707, 39: 1659},
            "challenge_segments": [40, 41],
            "challenge_rows": 3295,
            "challenge_segment_rows": {40: 1672, 41: 1623},
            "challenge_forced_outage_sensor": "imu1_link_root",
            "challenge_availability": [1, 0, 1, 1],
            "segment_41_role": "primary_previously_unused_same_session_diagnostic",
            "segment_40_role": "secondary_previously_observed_same_session_diagnostic",
            "no_synthetic_rows": True,
        },
        name="data roles section",
    )

    if tuple(int(value) for value in roles.get("development_segments", ())) != (
        GBDT_V4_DEVELOPMENT_SEGMENTS
    ):
        raise GPUOutageGBDTV4ConfigError("v4 development segments changed")
    if int(roles.get("development_rows", -1)) != 6635 or {
        int(key): int(value)
        for key, value in _mapping(roles, "development_segment_rows").items()
    } != {36: 1672, 37: 1597, 38: 1707, 39: 1659}:
        raise GPUOutageGBDTV4ConfigError("v4 development population changed")
    if tuple(int(value) for value in roles.get("challenge_segments", ())) != (40, 41):
        raise GPUOutageGBDTV4ConfigError("v4 challenge segments changed")
    if int(roles.get("challenge_rows", -1)) != 3295 or {
        int(key): int(value)
        for key, value in _mapping(roles, "challenge_segment_rows").items()
    } != {40: 1672, 41: 1623}:
        raise GPUOutageGBDTV4ConfigError("v4 challenge population changed")
    if roles.get("challenge_forced_outage_sensor") != "imu1_link_root" or roles.get(
        "challenge_availability"
    ) != [1, 0, 1, 1]:
        raise GPUOutageGBDTV4ConfigError("v4 challenge outage changed")
    if roles.get("segment_41_role") != "primary_previously_unused_same_session_diagnostic":
        raise GPUOutageGBDTV4ConfigError("segment 41 primary interpretation changed")
    if roles.get("segment_40_role") != "secondary_previously_observed_same_session_diagnostic":
        raise GPUOutageGBDTV4ConfigError("segment 40 secondary interpretation changed")
    if roles.get("no_synthetic_rows") is not True:
        raise GPUOutageGBDTV4ConfigError("v4 must remain measured-real only")

    _require_exact(
        dict(task),
        {
            "sensor_feature_view": "fusion_v1",
            "sensor_feature_count": 43,
            "availability_feature_count": 4,
            "model_input_feature_count": 47,
            "unavailable_sensor_imputation": "exact_zero_after_train_only_standardization",
            "availability_features_are_raw_binary": True,
            "target_transform": "train_xyz_mean_and_one_isotropic_scale",
            "prediction_space": "inverse_transformed_vive_world_xyz_mm",
            "inference_forbidden_tokens": [
                "vive",
                "tracker",
                "planned",
                "target_id",
                "segment",
                "timestamp",
                "qc",
                "provenance",
            ],
        },
        name="task section",
    )

    if (
        task.get("sensor_feature_view") != "fusion_v1"
        or int(task.get("sensor_feature_count", -1)) != 43
        or int(task.get("availability_feature_count", -1)) != 4
        or int(task.get("model_input_feature_count", -1)) != 47
    ):
        raise GPUOutageGBDTV4ConfigError("v4 model input layout changed")
    if task.get("unavailable_sensor_imputation") != (
        "exact_zero_after_train_only_standardization"
    ):
        raise GPUOutageGBDTV4ConfigError("v4 outage imputation changed")
    if task.get("target_transform") != "train_xyz_mean_and_one_isotropic_scale":
        raise GPUOutageGBDTV4ConfigError("v4 target transform changed")
    forbidden = {str(value) for value in task.get("inference_forbidden_tokens", ())}
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
        raise GPUOutageGBDTV4ConfigError("v4 inference forbidden-token set changed")

    states = replication.get("states")
    if not isinstance(states, list) or len(states) != 5:
        raise GPUOutageGBDTV4ConfigError("v4 mask replication states changed")
    probabilities = [float(state.get("probability")) for state in states]
    if probabilities != [0.25, 0.1875, 0.1875, 0.1875, 0.1875] or not math.isclose(
        sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-15
    ):
        raise GPUOutageGBDTV4ConfigError("v4 mask replication probabilities changed")
    expected_availability = [
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ]
    _require_exact(
        dict(replication),
        {
            "states": [
                {"state": "full", "probability": 0.25, "availability": [1, 1, 1, 1]},
                {
                    "state": "drop_imu0_base",
                    "probability": 0.1875,
                    "availability": [0, 1, 1, 1],
                },
                {
                    "state": "drop_imu1_link_root",
                    "probability": 0.1875,
                    "availability": [1, 0, 1, 1],
                },
                {
                    "state": "drop_imu2_link_mid",
                    "probability": 0.1875,
                    "availability": [1, 1, 0, 1],
                },
                {
                    "state": "drop_imu3_tip",
                    "probability": 0.1875,
                    "availability": [1, 1, 1, 0],
                },
            ],
            "materialization": "each_complete_training_uid_replicated_once_per_state",
            "replicated_rows_per_training_uid": 5,
            "segment_balance_weight_formula": "N_train / (K * n_segment) * q_mask",
            "total_effective_weight": "N_train",
            "fit_preprocessor_on_complete_unreplicated_train_rows_only": True,
        },
        name="mask replication section",
    )
    if [state.get("availability") for state in states] != expected_availability:
        raise GPUOutageGBDTV4ConfigError("v4 mask availability states changed")
    if int(replication.get("replicated_rows_per_training_uid", -1)) != 5:
        raise GPUOutageGBDTV4ConfigError("v4 replication width changed")
    if replication.get("segment_balance_weight_formula") != "N_train / (K * n_segment) * q_mask":
        raise GPUOutageGBDTV4ConfigError("v4 segment/mask weight formula changed")
    if replication.get("fit_preprocessor_on_complete_unreplicated_train_rows_only") is not True:
        raise GPUOutageGBDTV4ConfigError("v4 train-only preprocessing boundary changed")

    exact_xgb = {
        "bundle_axes": ["x", "y", "z"],
        "scalar_boosters_per_bundle": 3,
        "development_unique_bundle_count": 12,
        "development_evaluation_record_count": 48,
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "num_boost_round": 1000,
        "eta": 0.03,
        "max_depth": 6,
        "min_child_weight": 5.0,
        "subsample": 1.0,
        "sampling_method": "uniform",
        "colsample_bytree": 0.8,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
        "max_bin": 256,
        "grow_policy": "depthwise",
        "num_parallel_tree": 1,
        "seed_per_iteration": False,
        "nthread": 1,
        "early_stopping": False,
        "device": "supplied_per_isolated_gpu_worker_as_cuda_0",
        "base_score": "exact_train_only_weighted_mean_per_axis",
        "axis_seed_recipe": "uint32_big_endian_first4_sha256(canonical_json([paired_seed,axis_index]))",
    }
    _require_exact(dict(xgb), exact_xgb, name="XGBoost section")

    _require_exact(
        dict(splits),
        {
            "held_out_segment_policy": "leave_one_development_segment_out",
            "forced_validation_outages": list(GBDT_V4_FORCED_OUTAGES),
            "paired_seed_values": list(GBDT_V4_PAIRED_SEEDS),
            "xgb_unique_bundles": "four_held_out_segments_by_three_paired_seeds",
            "xgb_outage_evaluations_reuse_bundle": "four_forced_outages_per_bundle",
        },
        name="splits section",
    )

    if tuple(str(value) for value in splits.get("forced_validation_outages", ())) != (
        GBDT_V4_FORCED_OUTAGES
    ):
        raise GPUOutageGBDTV4ConfigError("v4 validation outage order changed")
    if tuple(int(value) for value in splits.get("paired_seed_values", ())) != (
        GBDT_V4_PAIRED_SEEDS
    ):
        raise GPUOutageGBDTV4ConfigError("v4 paired seeds changed")
    if splits.get("xgb_unique_bundles") != "four_held_out_segments_by_three_paired_seeds":
        raise GPUOutageGBDTV4ConfigError("v4 XGB bundle grid changed")
    if splits.get("xgb_outage_evaluations_reuse_bundle") != (
        "four_forced_outages_per_bundle"
    ):
        raise GPUOutageGBDTV4ConfigError("v4 bundle reuse rule changed")

    candidates = gbdt_v4_candidate_grid(config)
    _require_exact(
        candidates,
        [
            {
                "candidate_id": "v3_incumbent",
                "kind": "immutable_v3_development_winner",
            },
            {
                "candidate_id": "xgboost_mask_replicated",
                "kind": "trainable_three_axis_gbdt_bundle",
            },
            {
                "candidate_id": "equal_blend_incumbent_xgboost",
                "kind": "fixed_virtual_blend",
                "components": [
                    {"candidate_id": "v3_incumbent", "weight": 0.5},
                    {"candidate_id": "xgboost_mask_replicated", "weight": 0.5},
                ],
                "blend_space": "inverse_transformed_vive_world_xyz_mm",
                "pair_same_seed_index_before_ensemble_and_dispersion": True,
                "learned_blend_weights": False,
            },
        ],
        name="candidate records",
    )
    if tuple(str(row.get("candidate_id")) for row in candidates) != GBDT_V4_CANDIDATE_IDS:
        raise GPUOutageGBDTV4ConfigError("v4 candidate identities/tie order changed")
    if candidates[2].get("components") != [
        {"candidate_id": "v3_incumbent", "weight": 0.5},
        {"candidate_id": "xgboost_mask_replicated", "weight": 0.5},
    ]:
        raise GPUOutageGBDTV4ConfigError("v4 fixed blend definition changed")

    _require_exact(
        dict(selection),
        {
            "candidate_order_is_tie_order": True,
            "scored_candidate_count": 3,
            "context_count_per_candidate": 16,
            "context_prediction": "arithmetic_mean_of_three_paired_seed_predictions",
            "score_name": "outage_robust_j_mm",
            "score_formula": "(mean_rmse_over_16_contexts_mm + worst_segment_mean_rmse_over_4_outages_mm + worst_sensor_mean_rmse_over_4_segments_mm) / 3",
            "tie_break_order": [
                "mean_p95_3d_mm_over_16_contexts",
                "mean_seed_rmse_population_std_mm_over_16_contexts",
                "listed_order",
            ],
            "numeric_equality_abs_tolerance_mm": 1e-12,
            "nonfinite_context_score": "positive_infinity",
            "missing_context_action": "fail_closed_before_scoring",
            "exact_v3_scoring_semantics": True,
            "challenge_scores_can_trigger_selection_or_reselection": False,
        },
        name="selection section",
    )

    if (
        int(selection.get("scored_candidate_count", -1)),
        int(selection.get("context_count_per_candidate", -1)),
    ) != (3, 16):
        raise GPUOutageGBDTV4ConfigError("v4 selection grid changed")
    if selection.get("score_name") != "outage_robust_j_mm":
        raise GPUOutageGBDTV4ConfigError("v4 score changed")
    if selection.get("tie_break_order") != [
        "mean_p95_3d_mm_over_16_contexts",
        "mean_seed_rmse_population_std_mm_over_16_contexts",
        "listed_order",
    ]:
        raise GPUOutageGBDTV4ConfigError("v4 tie-break order changed")
    if float(selection.get("numeric_equality_abs_tolerance_mm", math.nan)) != 1e-12:
        raise GPUOutageGBDTV4ConfigError("v4 numeric tie tolerance changed")
    if selection.get("nonfinite_context_score") != "positive_infinity" or selection.get(
        "missing_context_action"
    ) != "fail_closed_before_scoring":
        raise GPUOutageGBDTV4ConfigError("v4 fail-closed score policy changed")
    if selection.get("exact_v3_scoring_semantics") is not True or selection.get(
        "challenge_scores_can_trigger_selection_or_reselection"
    ) is not False:
        raise GPUOutageGBDTV4ConfigError("v4 selection/challenge boundary changed")

    _require_exact(
        dict(final_refit),
        {
            "enabled": False,
            "owner": "outage_v5_separate_sole_owner_contract",
            "v4_final_refit_forbidden": True,
            "v4_challenge_prediction_forbidden": True,
            "v4_challenge_label_access_forbidden": True,
        },
        name="final refit section",
    )

    _require_exact(
        dict(execution),
        {
            "fresh_run_stops_after_development_selection": True,
            "development_only": True,
            "finalize_challenge_argument_forbidden": True,
            "challenge_label_loader_import_forbidden": True,
            "source_archive_access_forbidden": True,
            "maximum_source_label_access_passes": 0,
            "v4_reselection_after_development_forbidden": True,
            "resultless_complete_bundle_resume": "authenticate_and_reconstruct_without_refit",
            "incomplete_bundle_resume": "quarantine_without_deletion_then_refit",
            "missing_aggregate_worker_ledger_resume": "reconstruct_from_durable_per_worker_ledgers",
            "gpu_count": 4,
            "python": "3.12",
        },
        name="execution section",
    )

    if not all(
        execution.get(key) is True
        for key in (
            "fresh_run_stops_after_development_selection",
            "development_only",
            "finalize_challenge_argument_forbidden",
            "challenge_label_loader_import_forbidden",
            "source_archive_access_forbidden",
            "v4_reselection_after_development_forbidden",
        )
    ):
        raise GPUOutageGBDTV4ConfigError("v4 development-only boundary changed")
    if int(execution.get("maximum_source_label_access_passes", -1)) != 0:
        raise GPUOutageGBDTV4ConfigError("v4 label-access pass count changed")
    if int(execution.get("gpu_count", -1)) != 4 or execution.get("python") != "3.12":
        raise GPUOutageGBDTV4ConfigError("v4 runtime changed")

    _require_exact(
        dict(provenance),
        {
            "source_closure_required_before_any_gbdt_metric": True,
            "source_closure_includes_all_v4_and_imported_v3_files": True,
            "current_source_must_match_prepare_snapshot": True,
            "environment_snapshot_required": True,
            "xgboost_version": "3.3.0",
            "xgboost_cuda_build_required": True,
            "loaded_libxgboost_sha256_required": True,
            "fixed_xgb_parameter_sha256_required": True,
            "exact_four_gpu_identity_required": True,
            "parent_gpu_uuid_and_pci_required": True,
            "worker_visible_cuda_0_uuid_pci_exact_physical_mapping_required": True,
        },
        name="provenance section",
    )

    if not all(
        provenance.get(key) is True
        for key in (
            "source_closure_required_before_any_gbdt_metric",
            "source_closure_includes_all_v4_and_imported_v3_files",
            "current_source_must_match_prepare_snapshot",
            "environment_snapshot_required",
            "xgboost_cuda_build_required",
            "loaded_libxgboost_sha256_required",
            "fixed_xgb_parameter_sha256_required",
            "exact_four_gpu_identity_required",
            "parent_gpu_uuid_and_pci_required",
            "worker_visible_cuda_0_uuid_pci_exact_physical_mapping_required",
        )
    ) or str(provenance.get("xgboost_version")) != "3.3.0":
        raise GPUOutageGBDTV4ConfigError("v4 source/environment provenance boundary changed")

    _require_exact(
        dict(report),
        {
            "enabled": False,
            "owner": "outage_v5_separate_sole_owner_contract",
            "v4_may_materialize_challenge_metrics": False,
            "v4_handoff_is_development_selection_only": True,
        },
        name="challenge report section",
    )

    if (
        report.get("enabled") is not False
        or report.get("owner") != "outage_v5_separate_sole_owner_contract"
        or report.get("v4_may_materialize_challenge_metrics") is not False
        or report.get("v4_handoff_is_development_selection_only") is not True
    ):
        raise GPUOutageGBDTV4ConfigError("v4 challenge interpretation changed")
    _require_exact(
        dict(boundaries),
        {
            "current_output": "static_trial_endpoint_in_raw_vive_world_mm_under_forced_single_imu_outage",
            "same_collection_session_as_development": True,
            "not_independent_session_confirmation": True,
            "not_normal_four_imu_confirmation": True,
            "not_robot_base_or_tcp_calibrated": True,
            "not_dynamic_state_estimation": True,
            "not_shapefk_or_formal_imu_only": True,
            "not_deployment_ready": True,
            "not_control_authority": True,
            "method_combination_novelty_not_claimed": True,
        },
        name="claim boundaries section",
    )
    if not all(
        boundaries.get(key) is True
        for key in (
            "same_collection_session_as_development",
            "not_independent_session_confirmation",
            "not_normal_four_imu_confirmation",
            "not_robot_base_or_tcp_calibrated",
            "not_dynamic_state_estimation",
            "not_shapefk_or_formal_imu_only",
            "not_deployment_ready",
            "not_control_authority",
            "method_combination_novelty_not_claimed",
        )
    ):
        raise GPUOutageGBDTV4ConfigError("v4 claim boundary was weakened")


def load_gbdt_v4_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the static v4 YAML contract."""

    config_path = Path(path).resolve()
    try:
        payload = yaml.load(
            config_path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader
        )
    except FileNotFoundError as exc:
        raise GPUOutageGBDTV4ConfigError(f"v4 config does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise GPUOutageGBDTV4ConfigError(f"invalid v4 config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GPUOutageGBDTV4ConfigError("v4 config root must be a mapping")
    payload["_config_path"] = str(config_path)
    payload["_workspace_root"] = str(config_path.parent.parent)
    validate_gbdt_v4_config(payload)
    return payload


__all__ = [
    "GBDT_V4_CANDIDATE_IDS",
    "GBDT_V4_CONTRACT",
    "GBDT_V4_DEVELOPMENT_SEGMENTS",
    "GBDT_V4_FORCED_OUTAGES",
    "GBDT_V4_PAIRED_SEEDS",
    "GBDT_V4_PARENT_CONTRACT",
    "GPUOutageGBDTV4ConfigError",
    "gbdt_v4_candidate_grid",
    "gbdt_v4_config_sha256",
    "load_gbdt_v4_config",
    "public_gbdt_v4_config",
    "validate_gbdt_v4_config",
]
