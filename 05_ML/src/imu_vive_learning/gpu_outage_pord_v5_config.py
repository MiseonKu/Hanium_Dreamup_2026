"""Strict static configuration boundary for outage PORD-v5.

The YAML is a preregistration, not a parameter suggestion.  Loading rejects
duplicate YAML keys, non-string mapping keys, missing/extra fields, scalar
type drift (including ``bool``/``int`` and ``int``/``float`` substitutions),
and every semantic change through a pinned canonical-JSON digest.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any

import yaml

from .config import canonical_json_bytes


PORD_V5_CONTRACT = "alphaf.endpoint_regression.gpu_experiment.outage_v5_pord"
PORD_V5_PARENT_V3_CONTRACT = "alphaf.endpoint_regression.gpu_experiment.outage_v3"
PORD_V5_PARENT_V4_CONTRACT = (
    "alphaf.endpoint_regression.gpu_experiment.outage_v4_gbdt"
)
PORD_V5_CONFIG_SHA256 = (
    "c035cfde58adec69892e9abe34d5dc011ed564cb5c8de7a238b09fec7fe903a2"
)
PORD_V5_DEVELOPMENT_SEGMENTS = (36, 37, 38, 39)
PORD_V5_CHALLENGE_SEGMENTS = (40, 41)
PORD_V5_PAIRED_SEEDS = (20_260_824, 20_260_825, 20_260_826)
PORD_V5_OUTAGE_NAMES = (
    "imu0_base",
    "imu1_link_root",
    "imu2_link_mid",
    "imu3_tip",
)
PORD_V5_PRIMARY_BACKBONE_ID = "masked_mlp_64x64_relu"
PORD_V5_TRANSFER_BACKBONE_ID = "chain_endpoint_only"
PORD_V5_PRIMARY_METHOD_IDS = (
    "full_gradient_only_mask_selected_control",
    "masked_erm",
    "constant_availability_erm",
    "group_mean_pord",
    "paired_tail",
    "paired_interaction",
    "paired_pord",
    "paired_wrong_pair",
    "paired_no_stopgrad",
    "groupdro",
    "vrex",
    "mofe",
)
PORD_V5_TRANSFER_METHOD_IDS = ("masked_erm", "paired_pord")
PORD_V5_PRIMARY_CANDIDATE_IDS = tuple(
    f"{PORD_V5_PRIMARY_BACKBONE_ID}__{method}"
    for method in PORD_V5_PRIMARY_METHOD_IDS
)
PORD_V5_TRANSFER_CANDIDATE_IDS = tuple(
    f"{PORD_V5_TRANSFER_BACKBONE_ID}__{method}"
    for method in PORD_V5_TRANSFER_METHOD_IDS
)
PORD_V5_CANDIDATE_IDS = (
    *PORD_V5_PRIMARY_CANDIDATE_IDS,
    *PORD_V5_TRANSFER_CANDIDATE_IDS,
)
PORD_V5_FIT_COUNT = 168
PORD_V5_PRIMARY_OUTAGE_CONTEXT_COUNT = 192
PORD_V5_TRANSFER_OUTAGE_CONTEXT_COUNT = 32
PORD_V5_OUTAGE_CONTEXT_COUNT = 224
PORD_V5_FULL_CONTEXT_COUNT = 56
PORD_V5_ROW_EXCESS_SUMMARY_COUNT = 672

_INTERNAL_CONFIG_KEYS = frozenset({"_config_path", "_workspace_root"})
_ROOT_KEYS = frozenset(
    {
        "contract",
        "parent_freeze",
        "data_roles",
        "task",
        "views",
        "backbones",
        "methods",
        "splits",
        "training",
        "execution",
        "aggregation",
        "claim_gates",
        "selection",
        "challenge",
        "provenance",
        "claim_boundaries",
    }
)


class GPUOutagePORDV5ConfigError(ValueError):
    """Raised when the frozen PORD-v5 static contract has drifted."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader which refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    loader.flatten_mapping(node)
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if type(key) is not str:
            raise GPUOutagePORDV5ConfigError(
                "PORD-v5 YAML mapping keys must be exact strings"
            )
        if key in result:
            raise GPUOutagePORDV5ConfigError(
                f"PORD-v5 config contains duplicate mapping key: {key!r}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def public_pord_v5_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the hash-covered portion of a loaded or resolved config."""

    if not isinstance(config, Mapping) or any(type(key) is not str for key in config):
        raise GPUOutagePORDV5ConfigError("PORD-v5 config must be a string-key mapping")
    private = {key for key in config if key.startswith("_")}
    if private and private != _INTERNAL_CONFIG_KEYS:
        raise GPUOutagePORDV5ConfigError("PORD-v5 internal config keys changed")
    if private:
        if any(type(config[key]) is not str for key in _INTERNAL_CONFIG_KEYS):
            raise GPUOutagePORDV5ConfigError(
                "PORD-v5 internal paths must be exact strings"
            )
        config_path = Path(config["_config_path"]).resolve()
        workspace_root = Path(config["_workspace_root"]).resolve()
        if workspace_root != config_path.parent.parent:
            raise GPUOutagePORDV5ConfigError(
                "PORD-v5 config/workspace path binding changed"
            )
    return deepcopy({key: value for key, value in config.items() if key not in private})


def pord_v5_config_sha256(config: Mapping[str, Any]) -> str:
    """Return the canonical semantic identity of the static contract."""

    try:
        payload = canonical_json_bytes(public_pord_v5_config(config))
    except (TypeError, ValueError) as exc:
        raise GPUOutagePORDV5ConfigError(
            "PORD-v5 config is not finite canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _walk_json(value: Any, *, path: str = "root") -> None:
    """Enforce exact JSON-compatible scalar families before digest checking."""

    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise GPUOutagePORDV5ConfigError(f"non-string key at {path}")
        for key, child in value.items():
            _walk_json(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _walk_json(child, path=f"{path}[{index}]")
        return
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float and math.isfinite(value):
        return
    raise GPUOutagePORDV5ConfigError(
        f"unsupported or non-finite PORD-v5 value at {path}"
    )


def pord_v5_candidate_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact twelve-primary plus two-transfer controlled rows."""

    validate_pord_v5_config(config)
    rows = [
        {
            "candidate_id": f"{PORD_V5_PRIMARY_BACKBONE_ID}__{method}",
            "backbone_id": PORD_V5_PRIMARY_BACKBONE_ID,
            "method_id": method,
            "table_role": "primary_controlled",
        }
        for method in PORD_V5_PRIMARY_METHOD_IDS
    ]
    rows.extend(
        {
            "candidate_id": f"{PORD_V5_TRANSFER_BACKBONE_ID}__{method}",
            "backbone_id": PORD_V5_TRANSFER_BACKBONE_ID,
            "method_id": method,
            "table_role": "chain_transfer",
        }
        for method in PORD_V5_TRANSFER_METHOD_IDS
    )
    if tuple(row["candidate_id"] for row in rows) != PORD_V5_CANDIDATE_IDS:
        raise GPUOutagePORDV5ConfigError("PORD-v5 candidate grid changed")
    return rows


def validate_pord_v5_config(config: Mapping[str, Any]) -> None:
    """Validate the closed-world, type-exact, pinned PORD-v5 contract."""

    public = public_pord_v5_config(config)
    if frozenset(public) != _ROOT_KEYS:
        raise GPUOutagePORDV5ConfigError("PORD-v5 config root keys changed")
    if any(not isinstance(public[key], Mapping) for key in _ROOT_KEYS):
        raise GPUOutagePORDV5ConfigError(
            "every PORD-v5 root section must remain a mapping"
        )
    _walk_json(public)
    digest = pord_v5_config_sha256(public)
    if digest != PORD_V5_CONFIG_SHA256:
        raise GPUOutagePORDV5ConfigError(
            "PORD-v5 static contract changed (value, type, key, or ordering drift)"
        )

    contract = public["contract"]
    splits = public["splits"]
    backbones = public["backbones"]
    task = public["task"]
    training = public["training"]
    execution = public["execution"]
    methods = public["methods"]
    claim_gates = public["claim_gates"]
    claim_boundaries = public["claim_boundaries"]
    parent_freeze = public["parent_freeze"]
    if (
        contract.get("name") != PORD_V5_CONTRACT
        or contract.get("parent_v3_contract") != PORD_V5_PARENT_V3_CONTRACT
        or contract.get("parent_v4_contract") != PORD_V5_PARENT_V4_CONTRACT
    ):
        raise GPUOutagePORDV5ConfigError("PORD-v5 parent/contract identity changed")
    if (
        tuple(splits.get("held_out_segments", ())) != PORD_V5_DEVELOPMENT_SEGMENTS
        or tuple(splits.get("paired_seed_values", ())) != PORD_V5_PAIRED_SEEDS
        or splits.get("exact_fit_count") != PORD_V5_FIT_COUNT
        or splits.get("outage_contexts_primary")
        != PORD_V5_PRIMARY_OUTAGE_CONTEXT_COUNT
        or splits.get("outage_contexts_transfer")
        != PORD_V5_TRANSFER_OUTAGE_CONTEXT_COUNT
        or splits.get("outage_contexts_total") != PORD_V5_OUTAGE_CONTEXT_COUNT
        or splits.get("full_contexts_total") != PORD_V5_FULL_CONTEXT_COUNT
        or splits.get("row_positive_excess_summaries")
        != PORD_V5_ROW_EXCESS_SUMMARY_COUNT
    ):
        raise GPUOutagePORDV5ConfigError("PORD-v5 fit/context grid changed")
    if (
        backbones.get("primary", {}).get("backbone_id")
        != PORD_V5_PRIMARY_BACKBONE_ID
        or tuple(backbones.get("primary", {}).get("methods", ()))
        != PORD_V5_PRIMARY_METHOD_IDS
        or backbones.get("transfer", {}).get("backbone_id")
        != PORD_V5_TRANSFER_BACKBONE_ID
        or tuple(backbones.get("transfer", {}).get("methods", ()))
        != PORD_V5_TRANSFER_METHOD_IDS
    ):
        raise GPUOutagePORDV5ConfigError("PORD-v5 backbone/method allocation changed")
    if (
        task.get("sensor_feature_count") != 43
        or task.get("availability_feature_count") != 4
        or task.get("model_input_feature_count") != 47
        or task.get("segment_is_model_input") is not False
    ):
        raise GPUOutagePORDV5ConfigError("PORD-v5 47D input boundary changed")
    if (
        training.get("checkpoint_selection_scope") != "training_segments_only"
        or training.get("outer_labels_read_before_checkpoint_fixed") is not False
        or training.get("outer_eval_count_per_fit") != 1
        or training.get("outer_evaluation_timing")
        != "exactly_once_after_checkpoint_fixed"
    ):
        raise GPUOutagePORDV5ConfigError(
            "PORD-v5 train-only checkpoint/outer-once boundary changed"
        )
    if (
        contract.get("challenge_gate_protocol_owner")
        != "v5_under_single_controller_assumption"
        or contract.get("machine_enforced_cross_parent_finalizer_mutex") is not False
        or contract.get(
            "single_controller_no_concurrent_parent_finalizer_assumption"
        )
        is not True
        or execution.get("worker_development_label_authority")
        != "full_snapshot_in_process_not_capability_isolated"
        or execution.get("trainer_outer_array_surface") != "absent"
        or execution.get("outer_nonuse_enforcement")
        != "code_order_and_durable_phase_receipts"
        or execution.get("cuda_identity_uuid_format")
        != "bare_lowercase_8_4_4_4_12_hex"
        or execution.get("cuda_visible_devices_selector")
        != "GPU_dash_plus_frozen_bare_uuid_exact"
        or execution.get(
            "cuda_visible_devices_set_to_canonical_selector_before_child_module_import"
        )
        is not True
        or execution.get("standalone_validators_are_pure_nonmutating") is not True
        or execution.get("recovery_mutation_route")
        != "explicit_resume_execution_only"
        or execution.get(
            "single_controller_no_concurrent_parent_finalizer_assumption"
        )
        is not True
        or execution.get("machine_enforced_cross_parent_finalizer_mutex") is not False
        or execution.get("parent_challenge_markers_checked_before_and_after_label_load")
        is not True
        or execution.get("parquet_arrow_dtype_contract")
        != "string_int64_float64_exact_no_narrowing_or_coercion"
        or execution.get("parquet_arrow_field_nullability")
        != "nullable_true_exact"
        or execution.get("parquet_schema_metadata")
        != "exact_single_canonical_pandas_metadata_bytes"
        or execution.get("parquet_metadata_version_policy")
        != "exact_locked_runtime_pyarrow_and_pandas_versions"
    ):
        raise GPUOutagePORDV5ConfigError(
            "PORD-v5 authority/recovery/schema boundary changed"
        )
    expected_strict_competitors = tuple(
        f"{PORD_V5_PRIMARY_BACKBONE_ID}__{method}"
        for method in (
            "group_mean_pord",
            "paired_tail",
            "paired_interaction",
            "paired_no_stopgrad",
            "paired_wrong_pair",
            "groupdro",
            "vrex",
            "mofe",
        )
    )
    if (
        methods.get("wrong_pair_evidence_is_bounded_to_one_preregistered_map")
        is not True
        or methods.get("no_general_claim_over_all_derangements") is not True
        or tuple(claim_gates.get("strict_j_lower_than", ()))
        != expected_strict_competitors
        or claim_gates.get("combined_objective_and_stop_gradient_gates_required")
        is not True
        or claim_gates.get("chain_transfer_is_separate_honest_report") is not True
        or claim_gates.get("chain_transfer_primary_eligibility_gate") is not False
        or claim_gates.get("chain_failure_interpretation")
        != "primary_claim_is_masked_mlp_architecture_specific"
        or claim_boundaries.get(
            "wrong_pair_evidence_is_bounded_to_one_preregistered_map"
        )
        is not True
        or claim_boundaries.get("no_general_claim_over_all_derangements") is not True
    ):
        raise GPUOutagePORDV5ConfigError(
            "PORD-v5 ablation/pairing claim gates changed"
        )
    if (
        parent_freeze.get("v3_preprocessing_reference_schema")
        != "path_bytes_sha256_exact"
        or parent_freeze.get("v3_checkpoint_reference_schema")
        != "path_bytes_sha256_tensor_count_exact"
        or parent_freeze.get("v3_checkpoint_npz_validation")
        != "positive_exact_tensor_count_nonempty_names_numeric_finite"
    ):
        raise GPUOutagePORDV5ConfigError(
            "PORD-v5 v3 parent artifact adapter contract changed"
        )


def load_pord_v5_config(path: str | Path) -> dict[str, Any]:
    """Load a v5 YAML file with duplicate-key and pinned-semantic checks."""

    config_path = Path(path).resolve()
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GPUOutagePORDV5ConfigError(
            f"cannot read PORD-v5 config: {config_path}"
        ) from exc
    try:
        loaded = yaml.load(raw, Loader=_UniqueKeySafeLoader)
    except GPUOutagePORDV5ConfigError:
        raise
    except yaml.YAMLError as exc:
        raise GPUOutagePORDV5ConfigError("invalid PORD-v5 YAML") from exc
    if not isinstance(loaded, Mapping):
        raise GPUOutagePORDV5ConfigError("PORD-v5 YAML root must be a mapping")
    config = dict(loaded)
    config["_config_path"] = str(config_path)
    config["_workspace_root"] = str(config_path.parent.parent.resolve())
    validate_pord_v5_config(config)
    return config


__all__ = [
    "GPUOutagePORDV5ConfigError",
    "PORD_V5_CANDIDATE_IDS",
    "PORD_V5_CHALLENGE_SEGMENTS",
    "PORD_V5_CONFIG_SHA256",
    "PORD_V5_CONTRACT",
    "PORD_V5_DEVELOPMENT_SEGMENTS",
    "PORD_V5_FIT_COUNT",
    "PORD_V5_OUTAGE_CONTEXT_COUNT",
    "PORD_V5_OUTAGE_NAMES",
    "PORD_V5_PAIRED_SEEDS",
    "PORD_V5_PARENT_V3_CONTRACT",
    "PORD_V5_PARENT_V4_CONTRACT",
    "PORD_V5_PRIMARY_BACKBONE_ID",
    "PORD_V5_PRIMARY_CANDIDATE_IDS",
    "PORD_V5_PRIMARY_METHOD_IDS",
    "PORD_V5_PRIMARY_OUTAGE_CONTEXT_COUNT",
    "PORD_V5_ROW_EXCESS_SUMMARY_COUNT",
    "PORD_V5_TRANSFER_BACKBONE_ID",
    "PORD_V5_TRANSFER_CANDIDATE_IDS",
    "PORD_V5_TRANSFER_METHOD_IDS",
    "PORD_V5_TRANSFER_OUTAGE_CONTEXT_COUNT",
    "PORD_V5_FULL_CONTEXT_COUNT",
    "load_pord_v5_config",
    "pord_v5_candidate_rows",
    "pord_v5_config_sha256",
    "public_pord_v5_config",
    "validate_pord_v5_config",
]
