"""Strict configuration boundary for the OpenAXES external model benchmark.

This module deliberately imports neither the preparation implementation nor
PyTorch.  Configuration inspection is therefore label-, source-, and CUDA-
free and is safe in a process controller before GPU workers are spawned.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


OPENAXES_EXTERNAL_MODEL_V1_CONTRACT = "openaxes.external.model_benchmark.v1"
OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256 = (
    "ce6a828e6d92118e29b7c67ec166cc3a555709e78bb41ef0838a79e982e80a3e"
)
OPENAXES_EXTERNAL_PREPARATION_V1_CONTRACT = "openaxes.external.preparation.v1"
OPENAXES_MODEL_SENSOR_GROUPS = ("humerus", "radius", "carpus", "digitus")
OPENAXES_MODEL_VIEW_IDS = (
    "full",
    "outage_humerus",
    "outage_radius",
    "outage_carpus",
    "outage_digitus",
)
OPENAXES_MODEL_METHOD_IDS = (
    "full_gradient_only_mask_selected_control",
    "masked_erm",
    "group_mean_pord",
    "paired_pord",
    "paired_wrong_pair",
    "groupdro",
    "vrex",
    "mofe",
)
OPENAXES_MODEL_SEEDS = (1701, 2603, 3907)
OPENAXES_MODEL_SHIFT_MS = (-50, -20, -10, 10, 20, 50)
OPENAXES_MODEL_PHASES = (
    "execution.started",
    "training.completed",
    "outer_prediction.completed",
    "outer_eval.started",
    "outer_eval.completed",
    "bundle.completed",
    "result.completed",
)
OPENAXES_PRELAUNCH_RECOVERY_PATH = "runtime/prelaunch_session_recovery.json"
OPENAXES_PRELAUNCH_RECOVERY_MODE = "authenticated_prelaunch_session_recovery"
OPENAXES_PRELAUNCH_EXACT_ROOT_MEMBERS = (
    "prepared_manifest.json",
    "PREPARED.json",
    "tasks",
    "runtime",
    "results",
    "quarantine",
)
OPENAXES_PRELAUNCH_EXACT_RUNTIME_MEMBERS_BEFORE_RECEIPT = (
    "ledgers",
    "execution_launches",
    "worker_ready",
    "attempt_ledger",
    "attempt_roster",
    "pre_fit_validation.json",
    "execution_session.json",
)
OPENAXES_PRELAUNCH_EXACT_RUNTIME_MEMBERS_AFTER_RECEIPT = (
    *OPENAXES_PRELAUNCH_EXACT_RUNTIME_MEMBERS_BEFORE_RECEIPT,
    "prelaunch_session_recovery.json",
)
OPENAXES_PRELAUNCH_EXACT_EMPTY_DIRECTORIES = (
    "runtime/execution_launches",
    "runtime/worker_ready",
    "runtime/attempt_ledger",
    "runtime/attempt_roster",
    "runtime/ledgers",
    "results",
    "quarantine",
)
OPENAXES_PRELAUNCH_AUTHORIZED_ACTION_SCOPE = (
    "deterministic_authenticated_ordinal_zero_launch_creation_only"
)
OPENAXES_PRELAUNCH_FORBIDDEN_ACTION_SCOPES = (
    "scientific_refit_authorization_under_declared_non_total_erasure_threat_model",
    "task_attempt_creation_from_recovery_receipt_alone",
    "evidence_deletion_or_quarantine_as_freshness",
    "worker_gpu_or_task_remapping",
    "later_launch_creation_without_authenticated_predecessor",
)
OPENAXES_NO_REFIT_THREAT_MODEL = (
    "crash_corrupt_prefix_quarantine_and_ordinary_partial_deletion_with_any_local_"
    "attempt_authority_remaining"
)
OPENAXES_TOTAL_ERASURE_LIMITATION = (
    "simultaneous_total_erasure_of_all_local_attempt_authorities_cannot_prove_"
    "whether_execution_previously_occurred_without_external_append_only_or_"
    "signature_authority"
)


class OpenAxesExternalModelV1ConfigError(ValueError):
    """Raised when the frozen external-model contract drifts."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    seen: list[Any] = []
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if any(type(old) is type(key) and old == key for old in seen):
                raise OpenAxesExternalModelV1ConfigError(
                    f"duplicate OpenAXES model config key: {key!r}"
                )
            result[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as exc:
            raise OpenAxesExternalModelV1ConfigError(
                "OpenAXES model config has an unsupported mapping key"
            ) from exc
        seen.append(key)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OpenAxesExternalModelV1ConfigError(
            "OpenAXES model config is not finite canonical JSON"
        ) from exc


def openaxes_external_model_config_sha256(config: Mapping[str, Any]) -> str:
    public = {key: value for key, value in config.items() if key != "_config_path"}
    return hashlib.sha256(_canonical_json(public)).hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise OpenAxesExternalModelV1ConfigError(
            f"{name} must be a string-keyed mapping"
        )
    return value


def _keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise OpenAxesExternalModelV1ConfigError(f"{name} keys changed")


def _exact_bool(value: Any, expected: bool, *, name: str) -> None:
    if type(value) is not bool or value is not expected:
        raise OpenAxesExternalModelV1ConfigError(f"{name} must be {expected!r}")


def _exact_int(value: Any, expected: int | None = None, *, name: str) -> int:
    if type(value) is not int or (expected is not None and value != expected):
        suffix = "an exact integer" if expected is None else f"exactly {expected}"
        raise OpenAxesExternalModelV1ConfigError(f"{name} must be {suffix}")
    return value


def _exact_float(
    value: Any, expected: float | None = None, *, name: str
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise OpenAxesExternalModelV1ConfigError(
            f"{name} must be a finite exact float"
        )
    if expected is not None and value != expected:
        raise OpenAxesExternalModelV1ConfigError(
            f"{name} must be exactly {expected}"
        )
    return value


def _exact_str(value: Any, expected: str | None = None, *, name: str) -> str:
    if type(value) is not str or not value or (
        expected is not None and value != expected
    ):
        raise OpenAxesExternalModelV1ConfigError(f"{name} is not exact text")
    return value


def _exact_str_list(
    value: Any, expected: Sequence[str], *, name: str
) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise OpenAxesExternalModelV1ConfigError(
            f"{name} must be an exact string list"
        )
    result = tuple(value)
    if result != tuple(expected):
        raise OpenAxesExternalModelV1ConfigError(f"{name} order changed")
    return result


def validate_openaxes_external_model_config(config: Mapping[str, Any]) -> None:
    """Validate the immutable external-model development benchmark config."""

    public = {key: value for key, value in config.items() if key != "_config_path"}
    _keys(
        public,
        {
            "contract",
            "schema",
            "preprocessing",
            "model",
            "methods",
            "optimization",
            "folds",
            "selection",
            "alignment_sensitivity",
            "execution",
            "artifacts",
            "limitations",
        },
        name="root",
    )

    contract = _mapping(public["contract"], name="contract")
    _keys(
        contract,
        {
            "name",
            "prepared_input_contract",
            "research_role",
            "prepared_before_external_model_results",
            "independent_so101_confirmation",
            "so101_data_or_weights_allowed",
            "challenge_access_allowed",
            "actual_sensor_outage_observed",
            "measured_complete_streams_with_synthetic_one_imu_masks",
            "first_causal_or_provable_claim_allowed",
        },
        name="contract",
    )
    _exact_str(
        contract["name"], OPENAXES_EXTERNAL_MODEL_V1_CONTRACT, name="contract name"
    )
    _exact_str(
        contract["prepared_input_contract"],
        OPENAXES_EXTERNAL_PREPARATION_V1_CONTRACT,
        name="prepared contract",
    )
    _exact_str(
        contract["research_role"],
        "secondary_objective_portability_development_benchmark",
        name="research role",
    )
    for key in (
        "prepared_before_external_model_results",
        "measured_complete_streams_with_synthetic_one_imu_masks",
    ):
        _exact_bool(contract[key], True, name=f"contract.{key}")
    for key in (
        "independent_so101_confirmation",
        "so101_data_or_weights_allowed",
        "challenge_access_allowed",
        "actual_sensor_outage_observed",
        "first_causal_or_provable_claim_allowed",
    ):
        _exact_bool(contract[key], False, name=f"contract.{key}")

    schema = _mapping(public["schema"], name="schema")
    _keys(
        schema,
        {
            "sensor_groups",
            "continuous_channels_per_sensor",
            "prepared_continuous_feature_count",
            "availability_bit_count",
            "model_input_count",
            "target_columns",
            "target_role",
            "target_frame",
            "target_unit",
            "metadata_not_model_inputs",
            "shape_or_speed_as_model_input_allowed",
        },
        name="schema",
    )
    _exact_str_list(schema["sensor_groups"], OPENAXES_MODEL_SENSOR_GROUPS, name="sensor groups")
    for key, expected in (
        ("continuous_channels_per_sensor", 9),
        ("prepared_continuous_feature_count", 36),
        ("availability_bit_count", 4),
        ("model_input_count", 40),
    ):
        _exact_int(schema[key], expected, name=f"schema.{key}")
    _exact_str_list(
        schema["target_columns"],
        ("tcp_x_m", "tcp_y_m", "tcp_z_m"),
        name="target columns",
    )
    _exact_str(schema["target_role"], "label_only", name="target role")
    _exact_str(schema["target_frame"], "UR5_robot_base_TCP", name="target frame")
    _exact_str(schema["target_unit"], "m", name="target unit")
    _exact_str_list(
        schema["metadata_not_model_inputs"],
        ("row_uid", "scenario_id", "sample_index", "aligned_time_us"),
        name="metadata columns",
    )
    _exact_bool(
        schema["shape_or_speed_as_model_input_allowed"],
        False,
        name="shape/speed feature boundary",
    )

    preprocessing = _mapping(public["preprocessing"], name="preprocessing")
    _keys(
        preprocessing,
        {
            "feature_transform",
            "feature_zero_variance_policy",
            "target_transform",
            "target_per_axis_scale_allowed",
            "outer_rows_in_transform_fit_allowed",
            "mask_after_feature_standardization",
            "continuous_mask_value",
            "view_order",
            "availability_vectors",
            "mask_law",
            "exact_five_views_per_training_uid",
            "same_uid_full_missing_pairing",
            "scenario_id_used_only_for_blocking_and_training_group_atoms",
        },
        name="preprocessing",
    )
    _exact_str(
        preprocessing["feature_transform"],
        "train_only_population_mean_and_std",
        name="feature transform",
    )
    _exact_str(
        preprocessing["feature_zero_variance_policy"],
        "fail_closed",
        name="zero variance policy",
    )
    _exact_str(
        preprocessing["target_transform"],
        "train_only_three_vector_mean_and_single_isotropic_population_scale",
        name="target transform",
    )
    for key in ("target_per_axis_scale_allowed", "outer_rows_in_transform_fit_allowed"):
        _exact_bool(preprocessing[key], False, name=f"preprocessing.{key}")
    for key in (
        "mask_after_feature_standardization",
        "exact_five_views_per_training_uid",
        "same_uid_full_missing_pairing",
        "scenario_id_used_only_for_blocking_and_training_group_atoms",
    ):
        _exact_bool(preprocessing[key], True, name=f"preprocessing.{key}")
    _exact_float(
        preprocessing["continuous_mask_value"], 0.0, name="continuous mask value"
    )
    _exact_str_list(preprocessing["view_order"], OPENAXES_MODEL_VIEW_IDS, name="view order")
    availability = _mapping(
        preprocessing["availability_vectors"], name="availability vectors"
    )
    mask_law = _mapping(preprocessing["mask_law"], name="mask law")
    if tuple(availability) != OPENAXES_MODEL_VIEW_IDS or tuple(mask_law) != OPENAXES_MODEL_VIEW_IDS:
        raise OpenAxesExternalModelV1ConfigError("five-view mapping order changed")
    expected_availability = (
        (1.0, 1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0, 0.0),
    )
    expected_law = (0.25, 0.1875, 0.1875, 0.1875, 0.1875)
    for view, bits, probability in zip(
        OPENAXES_MODEL_VIEW_IDS, expected_availability, expected_law
    ):
        value = availability[view]
        if type(value) is not list or tuple(value) != bits or any(
            type(item) is not float for item in value
        ):
            raise OpenAxesExternalModelV1ConfigError(
                f"availability vector changed: {view}"
            )
        _exact_float(mask_law[view], probability, name=f"mask law {view}")
    if math.fsum(mask_law.values()) != 1.0:
        raise OpenAxesExternalModelV1ConfigError("mask law does not sum to one")

    model = _mapping(public["model"], name="model")
    _keys(
        model,
        {"backbone", "hidden_widths", "activation", "output_count", "reconstruction_head"},
        name="model",
    )
    _exact_str(model["backbone"], "masked_mlp_64x64_relu", name="backbone")
    if type(model["hidden_widths"]) is not list or model["hidden_widths"] != [64, 64] or any(
        type(item) is not int for item in model["hidden_widths"]
    ):
        raise OpenAxesExternalModelV1ConfigError("hidden widths changed")
    _exact_str(model["activation"], "ReLU", name="activation")
    _exact_int(model["output_count"], 3, name="output count")
    _exact_bool(model["reconstruction_head"], False, name="reconstruction head")

    methods = _mapping(public["methods"], name="methods")
    _keys(
        methods,
        {
            "ordered_ids",
            "frozen_comparator",
            "frozen_objective",
            "paired_outage_degradation_name_only",
            "regret_name_or_oracle_comparator_allowed",
            "upper_tail",
            "interaction",
            "objectives",
        },
        name="methods",
    )
    _exact_str_list(methods["ordered_ids"], OPENAXES_MODEL_METHOD_IDS, name="method ids")
    _exact_str(methods["frozen_comparator"], "masked_erm", name="comparator")
    _exact_str(methods["frozen_objective"], "paired_pord", name="objective")
    _exact_bool(
        methods["paired_outage_degradation_name_only"], True, name="method naming"
    )
    _exact_bool(
        methods["regret_name_or_oracle_comparator_allowed"],
        False,
        name="regret boundary",
    )
    tail = _mapping(methods["upper_tail"], name="upper tail")
    _keys(
        tail,
        {"kind", "standard_cvar_theorem_claim_allowed", "atom", "count_rule", "cutoff_tie_subgradient"},
        name="upper tail",
    )
    _exact_str(tail["kind"], "fixed_atom_top_quarter_surrogate", name="tail kind")
    _exact_bool(tail["standard_cvar_theorem_claim_allowed"], False, name="CVaR claim")
    _exact_str(
        tail["atom"], "training_scenario_group_by_forced_outage", name="tail atom"
    )
    _exact_str(
        tail["count_rule"], "ceil_one_quarter_of_available_atoms", name="tail count"
    )
    _exact_str(tail["cutoff_tie_subgradient"], "symmetric", name="tie gradient")
    interaction = _mapping(methods["interaction"], name="interaction")
    _keys(
        interaction,
        {"kind", "coefficient", "absolute_zero_subgradient"},
        name="interaction",
    )
    _exact_str(
        interaction["kind"],
        "two_way_double_centered_outage_degradation_atoms",
        name="interaction kind",
    )
    _exact_float(interaction["coefficient"], 1.0, name="interaction coefficient")
    _exact_float(
        interaction["absolute_zero_subgradient"], 0.0, name="abs zero gradient"
    )
    objectives = _mapping(methods["objectives"], name="objectives")
    if tuple(objectives) != OPENAXES_MODEL_METHOD_IDS or any(
        not isinstance(value, Mapping) for value in objectives.values()
    ):
        raise OpenAxesExternalModelV1ConfigError("method objective table changed")
    if objectives["paired_wrong_pair"].get("pairing") != (
        "within_training_scenario_sorted_uid_circular_derangement_by_one"
    ):
        raise OpenAxesExternalModelV1ConfigError("wrong-pair control changed")
    if objectives["groupdro"].get("eta") != 0.01:
        raise OpenAxesExternalModelV1ConfigError("GroupDRO eta changed")
    if objectives["vrex"].get("beta") != 1.0:
        raise OpenAxesExternalModelV1ConfigError("V-REx beta changed")
    if objectives["mofe"].get("lambda") != 0.1:
        raise OpenAxesExternalModelV1ConfigError("MoFe lambda changed")

    optimization = _mapping(public["optimization"], name="optimization")
    _keys(
        optimization,
        {
            "optimizer",
            "learning_rate",
            "weight_decay",
            "full_data_updates",
            "batch_policy",
            "checkpoint_selection_scope",
            "checkpoint_score",
            "checkpoint_tie_rule",
            "outer_labels_visible_to_trainer",
            "outer_evaluation_count_per_fit",
            "identical_initial_state_views_scaler_budget_and_update_count",
        },
        name="optimization",
    )
    _exact_str(optimization["optimizer"], "AdamW", name="optimizer")
    _exact_float(optimization["learning_rate"], 0.001, name="learning rate")
    _exact_float(optimization["weight_decay"], 0.001, name="weight decay")
    _exact_int(optimization["full_data_updates"], 400, name="updates")
    _exact_str(
        optimization["batch_policy"],
        "full_outer_training_partition_every_update",
        name="batch policy",
    )
    _exact_str(
        optimization["checkpoint_selection_scope"],
        "training_scenarios_only",
        name="checkpoint scope",
    )
    _exact_str(
        optimization["checkpoint_score"],
        "half_mean_four_outage_rmse_plus_half_max_four_outage_rmse",
        name="checkpoint score",
    )
    _exact_str(
        optimization["checkpoint_tie_rule"],
        "earliest_strict_minimum",
        name="checkpoint tie",
    )
    _exact_bool(
        optimization["outer_labels_visible_to_trainer"],
        False,
        name="outer label trainer boundary",
    )
    _exact_int(
        optimization["outer_evaluation_count_per_fit"], 1, name="outer eval count"
    )
    _exact_bool(
        optimization["identical_initial_state_views_scaler_budget_and_update_count"],
        True,
        name="paired controls",
    )

    folds = _mapping(public["folds"], name="folds")
    _keys(
        folds,
        {
            "ordered_families",
            "family_counts",
            "primary_selection_family",
            "report_only_families",
            "outer_test_fitting_allowed",
            "row_random_split_allowed",
            "seeds",
            "method_count",
            "total_outer_folds",
            "primary_scientific_fits",
            "stress_scientific_fits",
            "total_scientific_fits",
            "zero_shift_fit_level_outage_contexts",
            "zero_shift_fit_level_full_contexts",
            "primary_seed_ensemble_outage_contexts",
            "stress_seed_ensemble_outage_contexts",
        },
        name="folds",
    )
    _exact_str_list(
        folds["ordered_families"],
        ("scenario_loso", "speed_pair_out", "shape_out"),
        name="fold families",
    )
    family_counts = _mapping(folds["family_counts"], name="family counts")
    if dict(family_counts) != {"scenario_loso": 8, "speed_pair_out": 4, "shape_out": 2} or any(
        type(value) is not int for value in family_counts.values()
    ):
        raise OpenAxesExternalModelV1ConfigError("fold family counts changed")
    _exact_str(
        folds["primary_selection_family"], "scenario_loso", name="primary family"
    )
    _exact_str_list(
        folds["report_only_families"],
        ("speed_pair_out", "shape_out"),
        name="stress families",
    )
    _exact_bool(folds["outer_test_fitting_allowed"], False, name="outer fitting")
    _exact_bool(folds["row_random_split_allowed"], False, name="random split")
    if type(folds["seeds"]) is not list or tuple(folds["seeds"]) != OPENAXES_MODEL_SEEDS or any(
        type(item) is not int for item in folds["seeds"]
    ):
        raise OpenAxesExternalModelV1ConfigError("paired seeds changed")
    expected_counts = {
        "method_count": 8,
        "total_outer_folds": 14,
        "primary_scientific_fits": 192,
        "stress_scientific_fits": 144,
        "total_scientific_fits": 336,
        "zero_shift_fit_level_outage_contexts": 1344,
        "zero_shift_fit_level_full_contexts": 336,
        "primary_seed_ensemble_outage_contexts": 256,
        "stress_seed_ensemble_outage_contexts": 192,
    }
    for key, expected in expected_counts.items():
        _exact_int(folds[key], expected, name=f"folds.{key}")
    if folds["method_count"] * folds["total_outer_folds"] * len(OPENAXES_MODEL_SEEDS) != folds["total_scientific_fits"]:
        raise OpenAxesExternalModelV1ConfigError("336-fit accounting is not closed")

    selection = _mapping(public["selection"], name="selection")
    _keys(
        selection,
        {
            "source_family",
            "seed_aggregation",
            "score_name",
            "score_formula",
            "score_contexts_per_method",
            "lower_is_better",
            "tie_rule",
            "full_view_metric_role",
            "speed_or_shape_stress_may_select",
            "alignment_sensitivity_may_select",
            "alignment_sensitivity_may_decide_directional_gate",
            "three_seeds_are_independent_experiments",
            "overlapping_outer_folds_are_independent_replications",
            "directional_gate",
        },
        name="selection",
    )
    _exact_str(selection["source_family"], "scenario_loso", name="selection family")
    _exact_str(
        selection["seed_aggregation"],
        "raw_prediction_mean_across_three_paired_seeds",
        name="seed aggregation",
    )
    _exact_str(selection["score_name"], "robust_J_m", name="score name")
    _exact_int(selection["score_contexts_per_method"], 32, name="score contexts")
    _exact_bool(selection["lower_is_better"], True, name="score direction")
    for key in (
        "speed_or_shape_stress_may_select",
        "alignment_sensitivity_may_select",
        "alignment_sensitivity_may_decide_directional_gate",
        "three_seeds_are_independent_experiments",
        "overlapping_outer_folds_are_independent_replications",
    ):
        _exact_bool(selection[key], False, name=f"selection.{key}")
    directional_gate = _mapping(
        selection["directional_gate"], name="directional gate"
    )
    _keys(
        directional_gate,
        {
            "objective_method_id",
            "strict_robust_J_better_than",
            "scenario_comparator_method_id",
            "scenario_score",
            "scenario_count",
            "minimum_strict_scenario_wins",
            "equality_passes",
            "exact_method_context_identity_hash_required",
        },
        name="directional gate",
    )
    _exact_str(
        directional_gate["objective_method_id"],
        "paired_pord",
        name="directional objective",
    )
    _exact_str_list(
        directional_gate["strict_robust_J_better_than"],
        ("masked_erm", "group_mean_pord", "paired_wrong_pair"),
        name="directional comparators",
    )
    _exact_str(
        directional_gate["scenario_comparator_method_id"],
        "masked_erm",
        name="directional scenario comparator",
    )
    _exact_str(
        directional_gate["scenario_score"],
        "mean_four_outage_context_rmse3d_m",
        name="directional scenario score",
    )
    _exact_int(directional_gate["scenario_count"], 8, name="directional scenarios")
    _exact_int(
        directional_gate["minimum_strict_scenario_wins"],
        6,
        name="directional minimum wins",
    )
    _exact_bool(
        directional_gate["equality_passes"], False, name="directional equality"
    )
    _exact_bool(
        directional_gate["exact_method_context_identity_hash_required"],
        True,
        name="directional context identity",
    )

    sensitivity = _mapping(
        public["alignment_sensitivity"], name="alignment sensitivity"
    )
    _keys(
        sensitivity,
        {
            "required_signed_target_shifts_ms",
            "zero_shift_required",
            "reconstruction_api_owner",
            "private_preparation_helper_import_allowed",
            "target_lookup_rule",
            "no_extrapolation",
            "edge_policy",
            "fixed_source_requires_zero_dropped_edges",
            "pairing_key",
            "same_base_uid_pairing_required",
            "methods_reported",
            "fold_families_reported",
            "metrics",
            "selection_or_gate_use_allowed",
            "production_complete_requires_signed_source_replay",
        },
        name="alignment sensitivity",
    )
    if type(sensitivity["required_signed_target_shifts_ms"]) is not list or tuple(
        sensitivity["required_signed_target_shifts_ms"]
    ) != OPENAXES_MODEL_SHIFT_MS or any(
        type(item) is not int for item in sensitivity["required_signed_target_shifts_ms"]
    ):
        raise OpenAxesExternalModelV1ConfigError("signed shift set changed")
    _exact_bool(sensitivity["zero_shift_required"], True, name="zero shift")
    _exact_str(
        sensitivity["reconstruction_api_owner"],
        OPENAXES_EXTERNAL_PREPARATION_V1_CONTRACT,
        name="shift API owner",
    )
    _exact_bool(
        sensitivity["private_preparation_helper_import_allowed"],
        False,
        name="private helper boundary",
    )
    _exact_bool(sensitivity["no_extrapolation"], True, name="shift extrapolation")
    _exact_bool(
        sensitivity["fixed_source_requires_zero_dropped_edges"],
        True,
        name="fixed-source dropped edges",
    )
    _exact_str(sensitivity["pairing_key"], "row_uid", name="shift pairing key")
    _exact_bool(
        sensitivity["same_base_uid_pairing_required"], True, name="shift UID pairing"
    )
    _exact_str_list(
        sensitivity["methods_reported"],
        ("masked_erm", "paired_pord"),
        name="shift methods",
    )
    _exact_str_list(
        sensitivity["fold_families_reported"],
        ("scenario_loso", "speed_pair_out", "shape_out"),
        name="shift families",
    )
    _exact_bool(
        sensitivity["selection_or_gate_use_allowed"], False, name="shift gate boundary"
    )
    _exact_bool(
        sensitivity["production_complete_requires_signed_source_replay"],
        True,
        name="production COMPLETE signed-source replay",
    )

    execution = _mapping(public["execution"], name="execution")
    _keys(
        execution,
        {
            "production_cuda_required",
            "required_physical_gpu_count",
            "workers_per_gpu_min",
            "workers_per_gpu_max",
            "workers_per_gpu_exact",
            "scheduler",
            "child_sets_cuda_visible_devices_before_torch_import",
            "child_visible_device",
            "worker_identity_from_process_transport_only",
            "duplicate_scientific_task_allowed",
            "result_order",
            "phases",
            "started_only_training_recovery",
            "completed_training_refit_allowed",
            "completed_outer_evaluation_replay_allowed",
            "resultless_recovery",
            "immutable_execution_session_required",
            "exact_resume_session_lineage_required",
            "child_gpu_ready_receipts_required",
            "persistent_attempt_ledger_survives_quarantine",
            "prelaunch_session_recovery",
        },
        name="execution",
    )
    _exact_bool(execution["production_cuda_required"], True, name="CUDA requirement")
    _exact_int(execution["required_physical_gpu_count"], 4, name="GPU count")
    _exact_int(execution["workers_per_gpu_min"], 1, name="minimum workers/GPU")
    _exact_int(execution["workers_per_gpu_max"], 3, name="maximum workers/GPU")
    _exact_int(execution["workers_per_gpu_exact"], 2, name="frozen workers/GPU")
    _exact_str(
        execution["scheduler"],
        "spawn_persistent_process_per_logical_gpu_slot_fifo_serial",
        name="scheduler",
    )
    _exact_bool(
        execution["child_sets_cuda_visible_devices_before_torch_import"],
        True,
        name="child CUDA isolation",
    )
    _exact_str(execution["child_visible_device"], "cuda:0", name="child device")
    _exact_bool(
        execution["worker_identity_from_process_transport_only"],
        True,
        name="worker identity",
    )
    _exact_bool(
        execution["duplicate_scientific_task_allowed"], False, name="task duplicate"
    )
    _exact_str(execution["result_order"], "canonical_task_id", name="result order")
    _exact_str_list(execution["phases"], OPENAXES_MODEL_PHASES, name="phase chain")
    _exact_str(
        execution["started_only_training_recovery"],
        "fail_closed_ambiguous_no_refit",
        name="started recovery",
    )
    _exact_bool(
        execution["completed_training_refit_allowed"], False, name="completed refit"
    )
    _exact_bool(
        execution["completed_outer_evaluation_replay_allowed"],
        False,
        name="outer eval replay",
    )
    for key in (
        "immutable_execution_session_required",
        "exact_resume_session_lineage_required",
        "child_gpu_ready_receipts_required",
        "persistent_attempt_ledger_survives_quarantine",
    ):
        _exact_bool(execution[key], True, name=f"execution.{key}")
    prelaunch = _mapping(
        execution["prelaunch_session_recovery"],
        name="prelaunch session recovery",
    )
    _keys(
        prelaunch,
        {
            "receipt_path",
            "required_resume_value",
            "required_launch_ordinal",
            "required_launch_mode",
            "exact_prepared_root_members",
            "exact_runtime_members_before_receipt",
            "exact_runtime_members_after_receipt",
            "exact_empty_directories",
            "authorized_action_scope",
            "forbidden_action_scopes",
            "partial_or_ambiguous_evidence_action",
            "recovery_receipt_is_attempt_authority",
            "scientific_refit_authorized_under_declared_non_total_erasure_threat_model",
            "no_refit_guarantee_threat_model",
            "external_append_only_or_signature_authority_present",
            "total_erasure_prior_execution_provable",
            "total_erasure_limitation",
        },
        name="prelaunch session recovery",
    )
    _exact_str(
        prelaunch["receipt_path"],
        OPENAXES_PRELAUNCH_RECOVERY_PATH,
        name="prelaunch recovery path",
    )
    _exact_bool(
        prelaunch["required_resume_value"],
        True,
        name="prelaunch recovery resume marker",
    )
    _exact_int(
        prelaunch["required_launch_ordinal"],
        0,
        name="prelaunch recovery launch ordinal",
    )
    _exact_str(
        prelaunch["required_launch_mode"],
        OPENAXES_PRELAUNCH_RECOVERY_MODE,
        name="prelaunch recovery launch mode",
    )
    for key, expected in (
        ("exact_prepared_root_members", OPENAXES_PRELAUNCH_EXACT_ROOT_MEMBERS),
        (
            "exact_runtime_members_before_receipt",
            OPENAXES_PRELAUNCH_EXACT_RUNTIME_MEMBERS_BEFORE_RECEIPT,
        ),
        (
            "exact_runtime_members_after_receipt",
            OPENAXES_PRELAUNCH_EXACT_RUNTIME_MEMBERS_AFTER_RECEIPT,
        ),
        ("exact_empty_directories", OPENAXES_PRELAUNCH_EXACT_EMPTY_DIRECTORIES),
        ("forbidden_action_scopes", OPENAXES_PRELAUNCH_FORBIDDEN_ACTION_SCOPES),
    ):
        _exact_str_list(prelaunch[key], expected, name=f"prelaunch recovery {key}")
    _exact_str(
        prelaunch["authorized_action_scope"],
        OPENAXES_PRELAUNCH_AUTHORIZED_ACTION_SCOPE,
        name="prelaunch authorized action",
    )
    _exact_str(
        prelaunch["partial_or_ambiguous_evidence_action"],
        "fail_closed",
        name="prelaunch partial evidence action",
    )
    for key in (
        "recovery_receipt_is_attempt_authority",
        "scientific_refit_authorized_under_declared_non_total_erasure_threat_model",
        "external_append_only_or_signature_authority_present",
        "total_erasure_prior_execution_provable",
    ):
        _exact_bool(prelaunch[key], False, name=f"prelaunch recovery {key}")
    _exact_str(
        prelaunch["no_refit_guarantee_threat_model"],
        OPENAXES_NO_REFIT_THREAT_MODEL,
        name="prelaunch no-refit threat model",
    )
    _exact_str(
        prelaunch["total_erasure_limitation"],
        OPENAXES_TOTAL_ERASURE_LIMITATION,
        name="prelaunch total-erasure limitation",
    )

    artifacts = _mapping(public["artifacts"], name="artifacts")
    _keys(
        artifacts,
        {
            "prepared_model_manifest",
            "task_directory",
            "ledger_directory",
            "result_directory",
            "development_manifest",
            "complete_marker",
            "immutable_nondelete_quarantine",
            "source_receipts_required",
            "config_source_environment_gpu_task_checkpoint_history_prediction_hashes_required",
            "synthetic_test_only_artifacts_accepted_in_production",
            "pre_fit_validation_receipt",
            "prelaunch_session_recovery_receipt",
            "full_signed_source_replay_receipt",
            "execution_session",
            "execution_launch_directory",
            "worker_ready_directory",
            "persistent_attempt_ledger_directory",
            "monotonic_attempt_roster_directory",
        },
        name="artifacts",
    )
    for key in (
        "immutable_nondelete_quarantine",
        "source_receipts_required",
        "config_source_environment_gpu_task_checkpoint_history_prediction_hashes_required",
    ):
        _exact_bool(artifacts[key], True, name=f"artifacts.{key}")
    _exact_bool(
        artifacts["synthetic_test_only_artifacts_accepted_in_production"],
        False,
        name="synthetic production boundary",
    )
    for key, expected in (
        ("pre_fit_validation_receipt", "runtime/pre_fit_validation.json"),
        (
            "prelaunch_session_recovery_receipt",
            OPENAXES_PRELAUNCH_RECOVERY_PATH,
        ),
        ("full_signed_source_replay_receipt", "runtime/full_signed_source_replay.json"),
        ("execution_session", "runtime/execution_session.json"),
        ("execution_launch_directory", "runtime/execution_launches"),
        ("worker_ready_directory", "runtime/worker_ready"),
        ("persistent_attempt_ledger_directory", "runtime/attempt_ledger"),
        ("monotonic_attempt_roster_directory", "runtime/attempt_roster"),
    ):
        _exact_str(artifacts[key], expected, name=f"artifacts.{key}")
    if artifacts["prelaunch_session_recovery_receipt"] != prelaunch["receipt_path"]:
        raise OpenAxesExternalModelV1ConfigError(
            "prelaunch recovery artifact/config paths disagree"
        )

    limitations = _mapping(public["limitations"], name="limitations")
    required_true = (
        "external_development_benchmark_only",
        "independent_so101_confirmation",
        "same_public_recording_session",
        "effective_scenario_clusters",
        "measured_complete_streams",
        "actual_sensor_failures_observed",
        "synthetic_single_imu_mask_only",
        "label_assisted_alignment",
        "heldout_input_clock_fit_offline_transductive",
        "acausal_alignment",
        "causal_streaming_or_deployment_claim_allowed",
        "pord_first_causal_provable_or_regret_claim_allowed",
        "statistical_significance_claim_allowed",
        "source_license_conflict_must_be_resolved_before_redistribution",
        "no_refit_guarantee_threat_model",
        "external_append_only_or_signature_attempt_authority_present",
        "total_erasure_prior_execution_provable",
        "total_erasure_resistance_claim_allowed",
        "total_erasure_limitation",
    )
    _keys(limitations, set(required_true), name="limitations")
    for key in (
        "external_development_benchmark_only",
        "same_public_recording_session",
        "measured_complete_streams",
        "synthetic_single_imu_mask_only",
        "label_assisted_alignment",
        "heldout_input_clock_fit_offline_transductive",
        "acausal_alignment",
        "source_license_conflict_must_be_resolved_before_redistribution",
    ):
        _exact_bool(limitations[key], True, name=f"limitations.{key}")
    _exact_int(
        limitations["effective_scenario_clusters"], 8, name="scenario clusters"
    )
    for key in (
        "independent_so101_confirmation",
        "actual_sensor_failures_observed",
        "causal_streaming_or_deployment_claim_allowed",
        "pord_first_causal_provable_or_regret_claim_allowed",
        "statistical_significance_claim_allowed",
        "external_append_only_or_signature_attempt_authority_present",
        "total_erasure_prior_execution_provable",
        "total_erasure_resistance_claim_allowed",
    ):
        _exact_bool(limitations[key], False, name=f"limitations.{key}")
    _exact_str(
        limitations["no_refit_guarantee_threat_model"],
        OPENAXES_NO_REFIT_THREAT_MODEL,
        name="limitations no-refit threat model",
    )
    _exact_str(
        limitations["total_erasure_limitation"],
        OPENAXES_TOTAL_ERASURE_LIMITATION,
        name="limitations total-erasure limitation",
    )

    if openaxes_external_model_config_sha256(config) != OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256:
        raise OpenAxesExternalModelV1ConfigError(
            "OpenAXES external-model config canonical SHA-256 changed"
        )


def load_openaxes_external_model_config(path: str | Path) -> dict[str, Any]:
    """Load only the canonical external-model YAML with duplicate rejection."""

    config_path = Path(path).resolve()
    try:
        payload = yaml.load(
            config_path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise OpenAxesExternalModelV1ConfigError(
            f"cannot load OpenAXES model config: {config_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise OpenAxesExternalModelV1ConfigError(
            "OpenAXES model config root must be a mapping"
        )
    config = deepcopy(dict(payload))
    config["_config_path"] = str(config_path)
    validate_openaxes_external_model_config(config)
    return config
