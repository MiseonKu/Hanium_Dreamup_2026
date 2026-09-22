from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from imu_vive_learning.openaxes_external_model_v1_config import (
    OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
    OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
    OPENAXES_MODEL_METHOD_IDS,
    OPENAXES_MODEL_PHASES,
    OpenAxesExternalModelV1ConfigError,
    load_openaxes_external_model_config,
    openaxes_external_model_config_sha256,
    validate_openaxes_external_model_config,
)


CONFIG = Path("configs/openaxes_external_model_v1.yaml")


def _config() -> dict:
    return load_openaxes_external_model_config(CONFIG)


def test_canonical_config_digest_and_closed_336_fit_grid() -> None:
    config = _config()
    assert config["contract"]["name"] == OPENAXES_EXTERNAL_MODEL_V1_CONTRACT
    assert openaxes_external_model_config_sha256(config) == OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256
    assert tuple(config["methods"]["ordered_ids"]) == OPENAXES_MODEL_METHOD_IDS
    assert tuple(config["execution"]["phases"]) == OPENAXES_MODEL_PHASES
    assert config["schema"]["model_input_count"] == 40
    assert config["folds"]["total_scientific_fits"] == 336
    assert config["folds"]["primary_scientific_fits"] == 192
    assert config["folds"]["stress_scientific_fits"] == 144
    assert config["folds"]["zero_shift_fit_level_outage_contexts"] == 1344
    assert config["folds"]["zero_shift_fit_level_full_contexts"] == 336
    assert config["execution"]["workers_per_gpu_exact"] == 2
    recovery = config["execution"]["prelaunch_session_recovery"]
    assert recovery["receipt_path"] == "runtime/prelaunch_session_recovery.json"
    assert recovery["required_resume_value"] is True
    assert recovery["required_launch_ordinal"] == 0
    assert recovery["required_launch_mode"] == (
        "authenticated_prelaunch_session_recovery"
    )
    assert recovery["exact_prepared_root_members"] == [
        "prepared_manifest.json",
        "PREPARED.json",
        "tasks",
        "runtime",
        "results",
        "quarantine",
    ]
    assert recovery["exact_runtime_members_before_receipt"] == [
        "ledgers",
        "execution_launches",
        "worker_ready",
        "attempt_ledger",
        "attempt_roster",
        "pre_fit_validation.json",
        "execution_session.json",
    ]
    assert recovery["exact_runtime_members_after_receipt"] == [
        *recovery["exact_runtime_members_before_receipt"],
        "prelaunch_session_recovery.json",
    ]
    assert recovery["exact_empty_directories"] == [
        "runtime/execution_launches",
        "runtime/worker_ready",
        "runtime/attempt_ledger",
        "runtime/attempt_roster",
        "runtime/ledgers",
        "results",
        "quarantine",
    ]
    assert recovery["authorized_action_scope"] == (
        "deterministic_authenticated_ordinal_zero_launch_creation_only"
    )
    assert recovery["forbidden_action_scopes"] == [
        "scientific_refit_authorization_under_declared_non_total_erasure_threat_model",
        "task_attempt_creation_from_recovery_receipt_alone",
        "evidence_deletion_or_quarantine_as_freshness",
        "worker_gpu_or_task_remapping",
        "later_launch_creation_without_authenticated_predecessor",
    ]
    assert recovery["partial_or_ambiguous_evidence_action"] == "fail_closed"
    assert recovery["recovery_receipt_is_attempt_authority"] is False
    assert recovery[
        "scientific_refit_authorized_under_declared_non_total_erasure_threat_model"
    ] is False
    assert recovery["no_refit_guarantee_threat_model"] == (
        "crash_corrupt_prefix_quarantine_and_ordinary_partial_deletion_with_any_local_"
        "attempt_authority_remaining"
    )
    assert recovery["external_append_only_or_signature_authority_present"] is False
    assert recovery["total_erasure_prior_execution_provable"] is False
    assert recovery["total_erasure_limitation"] == (
        "simultaneous_total_erasure_of_all_local_attempt_authorities_cannot_prove_"
        "whether_execution_previously_occurred_without_external_append_only_or_"
        "signature_authority"
    )


def test_config_freezes_synthetic_masks_primary_selection_and_report_only_stress() -> None:
    config = _config()
    assert sum(config["preprocessing"]["mask_law"].values()) == 1.0
    assert config["preprocessing"]["exact_five_views_per_training_uid"] is True
    assert config["folds"]["primary_selection_family"] == "scenario_loso"
    assert config["folds"]["report_only_families"] == ["speed_pair_out", "shape_out"]
    assert config["selection"]["speed_or_shape_stress_may_select"] is False
    assert config["selection"]["alignment_sensitivity_may_select"] is False
    gate = config["selection"]["directional_gate"]
    assert gate["objective_method_id"] == "paired_pord"
    assert gate["strict_robust_J_better_than"] == [
        "masked_erm",
        "group_mean_pord",
        "paired_wrong_pair",
    ]
    assert gate["minimum_strict_scenario_wins"] == 6
    assert gate["equality_passes"] is False
    assert config["alignment_sensitivity"]["required_signed_target_shifts_ms"] == [
        -50,
        -20,
        -10,
        10,
        20,
        50,
    ]
    assert config["alignment_sensitivity"][
        "fixed_source_requires_zero_dropped_edges"
    ] is True
    assert config["alignment_sensitivity"]["private_preparation_helper_import_allowed"] is False
    assert config["alignment_sensitivity"][
        "production_complete_requires_signed_source_replay"
    ] is True
    assert config["artifacts"]["pre_fit_validation_receipt"] == (
        "runtime/pre_fit_validation.json"
    )
    assert config["artifacts"]["prelaunch_session_recovery_receipt"] == (
        "runtime/prelaunch_session_recovery.json"
    )
    assert config["artifacts"]["full_signed_source_replay_receipt"] == (
        "runtime/full_signed_source_replay.json"
    )
    assert config["artifacts"]["monotonic_attempt_roster_directory"] == (
        "runtime/attempt_roster"
    )


def test_config_freezes_claim_and_alignment_limitations() -> None:
    config = _config()
    assert config["contract"]["independent_so101_confirmation"] is False
    assert config["contract"]["actual_sensor_outage_observed"] is False
    assert config["limitations"]["synthetic_single_imu_mask_only"] is True
    assert config["limitations"]["label_assisted_alignment"] is True
    assert config["limitations"]["heldout_input_clock_fit_offline_transductive"] is True
    assert config["limitations"]["acausal_alignment"] is True
    assert config["limitations"]["pord_first_causal_provable_or_regret_claim_allowed"] is False
    assert config["limitations"][
        "external_append_only_or_signature_attempt_authority_present"
    ] is False
    assert config["limitations"]["total_erasure_prior_execution_provable"] is False
    assert config["limitations"]["total_erasure_resistance_claim_allowed"] is False
    assert "cannot_prove_whether_execution_previously_occurred" in config[
        "limitations"
    ]["total_erasure_limitation"]


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        (lambda value: value["folds"].__setitem__("total_scientific_fits", 336.0), "exactly"),
        (lambda value: value["schema"].__setitem__("model_input_count", True), "exactly"),
        (lambda value: value["execution"].__setitem__("workers_per_gpu_max", 4), "SHA-256|exact"),
        (
            lambda value: value["execution"]["prelaunch_session_recovery"].__setitem__(
                "receipt_path", "runtime/forged.json"
            ),
            "exact text",
        ),
        (
            lambda value: value["execution"]["prelaunch_session_recovery"][
                "exact_empty_directories"
            ].pop(),
            "order",
        ),
        (
            lambda value: value["execution"]["prelaunch_session_recovery"].__setitem__(
                "authorized_action_scope", "scientific_refit"
            ),
            "exact text",
        ),
        (
            lambda value: value["execution"]["prelaunch_session_recovery"][
                "forbidden_action_scopes"
            ].pop(),
            "order",
        ),
        (
            lambda value: value["execution"]["prelaunch_session_recovery"].__setitem__(
                "total_erasure_limitation", "total_erasure_is_safe"
            ),
            "exact text",
        ),
        (
            lambda value: value["limitations"].__setitem__(
                "total_erasure_prior_execution_provable", True
            ),
            "False",
        ),
        (lambda value: value["limitations"].__setitem__("actual_sensor_failures_observed", True), "False"),
        (lambda value: value["selection"].__setitem__("alignment_sensitivity_may_select", True), "False"),
        (lambda value: value["selection"]["directional_gate"].__setitem__("equality_passes", True), "False"),
        (
            lambda value: value["alignment_sensitivity"].__setitem__(
                "production_complete_requires_signed_source_replay", False
            ),
            "True",
        ),
        (lambda value: value["methods"]["ordered_ids"].reverse(), "order"),
    ),
)
def test_type_direction_and_closed_grid_drift_fail_closed(mutator, match: str) -> None:
    config = deepcopy(_config())
    mutator(config)
    with pytest.raises(OpenAxesExternalModelV1ConfigError, match=match):
        validate_openaxes_external_model_config(config)


def test_duplicate_yaml_key_and_nonfinite_value_fail_closed(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("contract:\n  name: x\n  name: y\n", encoding="utf-8")
    with pytest.raises(OpenAxesExternalModelV1ConfigError, match="duplicate"):
        load_openaxes_external_model_config(duplicate)

    nonfinite = tmp_path / "nonfinite.yaml"
    text = CONFIG.read_text(encoding="utf-8").replace(
        "learning_rate: 0.001", "learning_rate: .nan"
    )
    nonfinite.write_text(text, encoding="utf-8")
    with pytest.raises(OpenAxesExternalModelV1ConfigError, match="finite|SHA-256"):
        load_openaxes_external_model_config(nonfinite)


def test_extra_root_or_nested_key_fails_closed() -> None:
    config = deepcopy(_config())
    config["attacker"] = True
    with pytest.raises(OpenAxesExternalModelV1ConfigError, match="root keys"):
        validate_openaxes_external_model_config(config)
    config = deepcopy(_config())
    config["contract"]["attacker"] = False
    with pytest.raises(OpenAxesExternalModelV1ConfigError, match="contract keys"):
        validate_openaxes_external_model_config(config)
