from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from imu_vive_learning.gpu_outage_pord_v5_config import (
    GPUOutagePORDV5ConfigError,
    PORD_V5_CANDIDATE_IDS,
    PORD_V5_CONFIG_SHA256,
    PORD_V5_FIT_COUNT,
    PORD_V5_FULL_CONTEXT_COUNT,
    PORD_V5_OUTAGE_CONTEXT_COUNT,
    PORD_V5_PRIMARY_OUTAGE_CONTEXT_COUNT,
    PORD_V5_PRIMARY_CANDIDATE_IDS,
    PORD_V5_TRANSFER_CANDIDATE_IDS,
    PORD_V5_TRANSFER_OUTAGE_CONTEXT_COUNT,
    PORD_V5_ROW_EXCESS_SUMMARY_COUNT,
    load_pord_v5_config,
    pord_v5_candidate_rows,
    pord_v5_config_sha256,
    public_pord_v5_config,
    validate_pord_v5_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "endpoint_regression_gpu_outage_v5_pord.yaml"


def test_pinned_config_and_exact_primary_transfer_grid() -> None:
    config = load_pord_v5_config(CONFIG_PATH)
    assert pord_v5_config_sha256(config) == PORD_V5_CONFIG_SHA256
    rows = pord_v5_candidate_rows(config)
    assert len(rows) == 14
    assert tuple(row["candidate_id"] for row in rows) == PORD_V5_CANDIDATE_IDS
    assert tuple(row["candidate_id"] for row in rows[:12]) == PORD_V5_PRIMARY_CANDIDATE_IDS
    assert tuple(row["candidate_id"] for row in rows[12:]) == PORD_V5_TRANSFER_CANDIDATE_IDS
    assert PORD_V5_FIT_COUNT == 168
    assert PORD_V5_PRIMARY_OUTAGE_CONTEXT_COUNT == 192
    assert PORD_V5_TRANSFER_OUTAGE_CONTEXT_COUNT == 32
    assert PORD_V5_OUTAGE_CONTEXT_COUNT == 224
    assert PORD_V5_FULL_CONTEXT_COUNT == 56
    assert PORD_V5_ROW_EXCESS_SUMMARY_COUNT == 672
    assert config["training"]["checkpoint_selection_scope"] == (
        "training_segments_only"
    )
    assert config["training"]["outer_labels_read_before_checkpoint_fixed"] is False
    assert config["training"]["outer_eval_count_per_fit"] == 1
    assert config["contract"]["challenge_gate_protocol_owner"] == (
        "v5_under_single_controller_assumption"
    )
    assert config["contract"]["machine_enforced_cross_parent_finalizer_mutex"] is False
    assert config["contract"][
        "single_controller_no_concurrent_parent_finalizer_assumption"
    ] is True
    assert config["execution"]["worker_development_label_authority"] == (
        "full_snapshot_in_process_not_capability_isolated"
    )
    assert config["execution"]["cuda_identity_uuid_format"] == (
        "bare_lowercase_8_4_4_4_12_hex"
    )
    assert config["execution"]["cuda_visible_devices_selector"] == (
        "GPU_dash_plus_frozen_bare_uuid_exact"
    )
    assert config["execution"][
        "cuda_visible_devices_set_to_canonical_selector_before_child_module_import"
    ] is True
    assert config["execution"]["standalone_validators_are_pure_nonmutating"] is True
    assert config["execution"]["recovery_mutation_route"] == (
        "explicit_resume_execution_only"
    )
    assert config["execution"]["parquet_arrow_dtype_contract"] == (
        "string_int64_float64_exact_no_narrowing_or_coercion"
    )
    assert config["execution"]["parquet_arrow_field_nullability"] == (
        "nullable_true_exact"
    )
    assert config["execution"]["parquet_schema_metadata"] == (
        "exact_single_canonical_pandas_metadata_bytes"
    )
    assert config["execution"]["parquet_metadata_version_policy"] == (
        "exact_locked_runtime_pyarrow_and_pandas_versions"
    )
    assert tuple(config["claim_gates"]["strict_j_lower_than"])[1:4] == (
        "masked_mlp_64x64_relu__paired_tail",
        "masked_mlp_64x64_relu__paired_interaction",
        "masked_mlp_64x64_relu__paired_no_stopgrad",
    )
    assert config["claim_gates"][
        "combined_objective_and_stop_gradient_gates_required"
    ] is True
    assert config["claim_gates"]["chain_transfer_primary_eligibility_gate"] is False
    assert config["methods"][
        "wrong_pair_evidence_is_bounded_to_one_preregistered_map"
    ] is True
    assert config["claim_boundaries"][
        "no_general_claim_over_all_derangements"
    ] is True
    assert config["parent_freeze"]["v3_preprocessing_reference_schema"] == (
        "path_bytes_sha256_exact"
    )
    assert config["parent_freeze"]["v3_checkpoint_reference_schema"] == (
        "path_bytes_sha256_tensor_count_exact"
    )
    assert config["parent_freeze"]["v3_checkpoint_npz_validation"] == (
        "positive_exact_tensor_count_nonempty_names_numeric_finite"
    )


@pytest.mark.parametrize(
    "duplicate",
    [
        "contract:\n  name: first\ncontract:\n  name: second\n",
        "contract:\n  name: first\n  name: second\n",
    ],
)
def test_loader_rejects_duplicate_keys_before_semantic_validation(
    tmp_path: Path, duplicate: str
) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(GPUOutagePORDV5ConfigError, match="duplicate"):
        load_pord_v5_config(path)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("splits", "exact_fit_count", 168.0),
        ("training", "update_budget", "400"),
        ("execution", "gpu_count", True),
        ("claim_gates", "same_v5_masked_erm_full_tolerance_mm", 0),
        ("task", "segment_is_model_input", 0),
    ],
)
def test_type_drift_attacks_fail_closed(section: str, key: str, value: object) -> None:
    config = load_pord_v5_config(CONFIG_PATH)
    attacked = public_pord_v5_config(config)
    attacked[section][key] = value
    with pytest.raises(GPUOutagePORDV5ConfigError, match="static contract changed"):
        validate_pord_v5_config(attacked)


def test_missing_extra_and_nonfinite_attacks_fail_closed() -> None:
    config = public_pord_v5_config(load_pord_v5_config(CONFIG_PATH))
    missing = deepcopy(config)
    del missing["training"]["update_budget"]
    extra = deepcopy(config)
    extra["training"]["new_knob"] = 1
    nonfinite = deepcopy(config)
    nonfinite["training"]["learning_rate"] = float("nan")
    for attacked in (missing, extra, nonfinite):
        with pytest.raises(GPUOutagePORDV5ConfigError):
            validate_pord_v5_config(attacked)


def test_internal_config_paths_are_both_present_or_absent_and_bound() -> None:
    config = load_pord_v5_config(CONFIG_PATH)
    public = public_pord_v5_config(config)
    validate_pord_v5_config(public)
    attacked = dict(config)
    attacked["_workspace_root"] = str(CONFIG_PATH.parent)
    with pytest.raises(GPUOutagePORDV5ConfigError, match="path binding"):
        public_pord_v5_config(attacked)
