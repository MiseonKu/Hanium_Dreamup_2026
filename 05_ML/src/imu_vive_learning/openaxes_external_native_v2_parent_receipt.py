"""Read-only parent evidence binder for the preregistered OpenAXES native v2.

This module deliberately uses only the Python standard library.  It never
imports the v1 experiment, preparation, model, evaluator, NumPy, PyTorch, or a
network client.  The production command consumes already-persisted JSON
receipts and hashes opaque files; it writes only three new files below a fresh
``<v2-run>/parents`` directory.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_DIRECTORY = "09_SO101_IMU_VIVE_LEARNING"
AUDITOR_IDENTITY = "openaxes_external_native_v2_parent_receipt.py"

SOURCE_AUDIT_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.parent_receipt_source_audit.erratum2.v1"
)
TERMINAL_CONTRACT = "openaxes.external.model_benchmark.v1.terminal_for_native_v2.v1"
BUNDLE_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.masked_erm_parent_bundle.v1"
)
V1_CONTRACT = "openaxes.external.model_benchmark.v1"
PREPARATION_CONTRACT = "openaxes.external.preparation.v1"
PRELAUNCH_RECOVERY_MODE = "authenticated_prelaunch_session_recovery"
PRELAUNCH_AUTHORIZED_ACTION_SCOPE = (
    "deterministic_authenticated_ordinal_zero_launch_creation_only"
)
PRELAUNCH_FORBIDDEN_ACTION_SCOPES = (
    "scientific_refit_authorization_under_declared_non_total_erasure_threat_model",
    "task_attempt_creation_from_recovery_receipt_alone",
    "evidence_deletion_or_quarantine_as_freshness",
    "worker_gpu_or_task_remapping",
    "later_launch_creation_without_authenticated_predecessor",
)
NO_REFIT_GUARANTEE_THREAT_MODEL = (
    "crash_corrupt_prefix_quarantine_and_ordinary_partial_deletion_with_any_local_"
    "attempt_authority_remaining"
)
TOTAL_ERASURE_LIMITATION = (
    "simultaneous_total_erasure_of_all_local_attempt_authorities_cannot_prove_"
    "whether_execution_previously_occurred_without_external_append_only_or_"
    "signature_authority"
)

ACTIVE_SOURCE_AUDIT_RECEIPT_PATH = (
    "receipts/openaxes_external_native_v2_parent_receipt_source_audit_erratum2.json"
)
ORIGINAL_SOURCE_AUDIT_RECEIPT_PATH = (
    "receipts/openaxes_external_native_v2_parent_receipt_source_audit.json"
)
ORIGINAL_AUDITED_SOURCE_COMMIT = "49d9cb64406d6f5c9457e210623c45324a0880ac"
ORIGINAL_SOURCE_AUDIT_EVIDENCE_COMMIT = (
    "ce09392a287f00447c5451471eae854d6aa726e6"
)
ERRATUM1_SOURCE_AUDIT_RECEIPT_PATH = (
    "receipts/openaxes_external_native_v2_parent_receipt_source_audit_erratum1.json"
)
ERRATUM1_AUDITED_SOURCE_COMMIT = "6f5b1226cff400b708f0c9d167973ce505c48938"
ERRATUM1_SOURCE_AUDIT_EVIDENCE_COMMIT = (
    "26795e6c34a7ac11995875134d8ff81758758622"
)
# Backward-compatible names retained for the original immutable authority.
RETIRED_SOURCE_AUDIT_RECEIPT_PATH = ORIGINAL_SOURCE_AUDIT_RECEIPT_PATH
RETIRED_AUDITED_SOURCE_COMMIT = ORIGINAL_AUDITED_SOURCE_COMMIT
RETIRED_SOURCE_AUDIT_EVIDENCE_COMMIT = ORIGINAL_SOURCE_AUDIT_EVIDENCE_COMMIT
HISTORICAL_AUDITED_SOURCE_COMMITS = frozenset(
    {ORIGINAL_AUDITED_SOURCE_COMMIT, ERRATUM1_AUDITED_SOURCE_COMMIT}
)
HISTORICAL_SOURCE_AUDIT_EVIDENCE_COMMITS = frozenset(
    {
        ORIGINAL_SOURCE_AUDIT_EVIDENCE_COMMIT,
        ERRATUM1_SOURCE_AUDIT_EVIDENCE_COMMIT,
    }
)
SOURCE_AUDIT_ERRATUM_ID = "openaxes_external_native_v2_parent_receipt_erratum2"
FAILED_V1_RUN_001_ID = "openaxes-v1-sealed-001"
FAILED_V1_RUN_002_ID = "openaxes-v1-sealed-002"
SUCCESSFUL_V1_RUN_ID = "openaxes-v1-sealed-003"
SUCCESSFUL_V1_RUN_PROJECT_RELATIVE_PATH = (
    "artifacts/openaxes_external_model_v1/run_openaxes-v1-sealed-003"
)
SUCCESSFUL_PREPARATION_RUN_PROJECT_RELATIVE_PATH = (
    "artifacts/openaxes_external_v1/prepared_openaxes-v1-sealed-003"
)
SUCCESSFUL_SUPERVISOR_RELATIVE_PATH = (
    "openaxes-v1-runs/openaxes-v1-sealed-003"
)
FAILED_001_SUPERVISOR_TERMINAL_RELATIVE_PATH = (
    "openaxes-v1-runs/openaxes-v1-sealed-001/LATEST_TERMINAL.json"
)
FAILED_001_SUPERVISOR_TERMINAL_SHA256 = (
    "e032c62b899efd7e90647748f6c087b05d271ae1e312fc8ec64f390cc1bc20e3"
)
FAILED_001_PERSISTED_SENSOR_GROUPS_MAPPING_SHA256 = (
    "30deb94cc249ec6c2d17ac141b5450a30e4c526c4f96378a8a57b0c94443762a"
)
FAILED_002_RUN_SPEC_SHA256 = (
    "e86c303be806cc2e076676a4d680f8b20ef28adbf03dfea24b0df28a7d7c8e16"
)
FAILED_002_RUN_SPEC_PAYLOAD_SHA256 = (
    "a21554298dc0b1683755bd6068687a55b4102699aafe642f8ce22e1355ab3dac"
)
FAILED_002_CLEANUP_BLOCKED_SHA256 = (
    "063644f72ad6d83eb3890d5ea0693ef3d42b0cb8f25831b2743fde87938b25e0"
)
FAILED_002_SUPERVISOR_LOG_SHA256 = (
    "4cad8de028568efd84c48b334b1285d38c8759318c751607c9455214007e33b8"
)
FAILED_002_LAUNCH_LOG_SHA256 = (
    "056bc9f64d882e16cf1b139fd383ecc5224927c1e5e55a6d88f6923218a71d48"
)
FAILED_002_PREPARATION_MANIFEST_SHA256 = (
    "1afd49e9f0b111e3a54d724668a894a9522ad039231746b7992e29d7b1117673"
)
FAILED_002_PREPARATION_PARQUET_SHA256 = (
    "98747e51fd95506b91facfeb273be23c454d07da563deea13a6f2192ded48703"
)
FAILED_002_PREPARATION_COMPLETE_SHA256 = (
    "129f08a63991a87a51755100a6b01a3b31daefac0aa28ba2560811b720bec7ea"
)
FAILED_002_PUBLIC_PREPARE_STDOUT_SHA256 = (
    "84fb2181e279b5fffc5ad89249e31529e2f4cc64ed93cc17118b0c5749166451"
)
FAILED_002_PUBLIC_VALIDATE_STDOUT_SHA256 = (
    "4d3f793511dd55a6df0fe5ba7404ca71e487aafcbfcfb7d19b54f2b4b5e3ba94"
)
FAILED_002_MODEL_PREPARED_SHA256 = (
    "6e2b6a55b66e379e34bf7ade7e96e961bc2a9b039987eae2140a8773b5aa6026"
)
FAILED_002_MODEL_PREPARED_MANIFEST_SHA256 = (
    "42e0cc5063e2a32b21e5aee25804b603690f2bd73930c81d326c4b8258a46498"
)
FAILED_002_TASK_INDEX_SHA256 = (
    "2fa8051f788ae9dc0ed9bb6eaae3e9a2c1a2395ca82d5ae751670583f114f416"
)
FAILED_002_EXECUTION_SESSION_SHA256 = (
    "836aff56ad4b8341bb4e72c8fc73d614e6479bd614ca9b6dab0d3b464bf7bc7c"
)
FAILED_002_PRE_FIT_VALIDATION_SHA256 = (
    "2940e0a25c6afbd48b99c1d17f20faee5777fdbe4598b5e8bfedbdaa911d0265"
)
FAILED_002_LAUNCH_STARTED_SHA256 = (
    "0a05882663931f38d9836a48632b9e227d934b3d555b5a11a387b4eeebbefdb9"
)
ERRATUM_FAILURE_CLASSIFICATION = (
    "outcome_free_pytorch_cuda_uuid_representation_bug"
)
ERRATUM_TIMING_SCOPE = (
    "committed_and_source_audited_before_terminal_bound_successful_"
    "openaxes_v1_sealed_003_preparation"
)
PRE_ERRATUM_V1_MODEL_AUTHORITY_COMMIT = ORIGINAL_SOURCE_AUDIT_EVIDENCE_COMMIT
PRE_ERRATUM2_V1_MODEL_AUTHORITY_COMMIT = ERRATUM1_SOURCE_AUDIT_EVIDENCE_COMMIT

V1_CONFIG_SHA256 = "ce6a828e6d92118e29b7c67ec166cc3a555709e78bb41ef0838a79e982e80a3e"
V1_SOURCE_SET_SHA256 = "dd6418badc644f7d7576c836d17074ee2914b4ae0cb555b26315eae8669b8488"
PREPARATION_CONFIG_SHA256 = (
    "05c3f44b58a16cebb8c0cef1338a42ae55a0fd84dd9f87db09196a7b740b32c6"
)
V1_TERMINAL_STAGE = "development_complete_before_any_external_outcome_reinterpretation"

SOURCE_CLOSURE = (
    "configs/openaxes_external_native_v2.yaml",
    "docs/OPENAXES_EXTERNAL_NATIVE_V2_CONTRACT.md",
    "docs/OPENAXES_EXTERNAL_NATIVE_V2_PARENT_RECEIPT_CONTRACT.md",
    "requirements-eval.lock",
    "src/imu_vive_learning/__init__.py",
    "src/imu_vive_learning/io_utils.py",
    "src/imu_vive_learning/openaxes_external_native_v2_parent_receipt.py",
    "tests/test_openaxes_external_native_v2_parent_receipt.py",
)
V2_PREREGISTRATION_PATHS = SOURCE_CLOSURE[:2]
V1_SOURCE_FILE_COUNT = 9
V1_TASK_COUNT = 336
MASKED_PRIMARY_TASK_COUNT = 24
V1_SOURCE_CLOSURE = (
    "configs/openaxes_external_model_v1.yaml",
    "docs/OPENAXES_EXTERNAL_MODEL_V1_CONTRACT.md",
    "src/imu_vive_learning/io_utils.py",
    "src/imu_vive_learning/openaxes_external_model_v1_config.py",
    "src/imu_vive_learning/openaxes_external_model_v1_training.py",
    "src/imu_vive_learning/openaxes_external_model_v1_experiment.py",
    "tests/test_openaxes_external_model_v1_config.py",
    "tests/test_openaxes_external_model_v1_training.py",
    "tests/test_openaxes_external_model_v1_experiment.py",
)
V1_SOURCE_REFERENCE_ORACLE = {
    "configs/openaxes_external_model_v1.yaml": {"path": "configs/openaxes_external_model_v1.yaml", "bytes": 12099, "sha256": "848a19cc270401f6c37d391b399832996324792382a4a926f93b1584d7186c51"},
    "docs/OPENAXES_EXTERNAL_MODEL_V1_CONTRACT.md": {"path": "docs/OPENAXES_EXTERNAL_MODEL_V1_CONTRACT.md", "bytes": 27939, "sha256": "1a9cf072567f11498f82f2b5cea96a02971e1eb67785c59dfcfe99e7e6e325ac"},
    "src/imu_vive_learning/io_utils.py": {"path": "src/imu_vive_learning/io_utils.py", "bytes": 2693, "sha256": "7c4371e7fc7098e77b3d0668edda68901e16f485e2d9a6ff4fb287951ec7362a"},
    "src/imu_vive_learning/openaxes_external_model_v1_config.py": {"path": "src/imu_vive_learning/openaxes_external_model_v1_config.py", "bytes": 39905, "sha256": "d9120b1e91f0635503978c9f3317219ac20a2e082ce491e7fab927171532647f"},
    "src/imu_vive_learning/openaxes_external_model_v1_training.py": {"path": "src/imu_vive_learning/openaxes_external_model_v1_training.py", "bytes": 28615, "sha256": "f727a36548a68cf5c4b7a5029698077b47c65f29e83dbc806c956b22ad776e07"},
    "src/imu_vive_learning/openaxes_external_model_v1_experiment.py": {"path": "src/imu_vive_learning/openaxes_external_model_v1_experiment.py", "bytes": 317879, "sha256": "a33f884347dc5470a4cb6604a1e692e340b8c0136e0ab5e390421e6e434da297"},
    "tests/test_openaxes_external_model_v1_config.py": {"path": "tests/test_openaxes_external_model_v1_config.py", "bytes": 10620, "sha256": "7d514c9fb152071aecca654ca459daa7acd12a94e56f33a886631ba227805c55"},
    "tests/test_openaxes_external_model_v1_training.py": {"path": "tests/test_openaxes_external_model_v1_training.py", "bytes": 12103, "sha256": "64903b2fde52a808f383fa0116028739273a3b2891fbd0688c095c1342140730"},
    "tests/test_openaxes_external_model_v1_experiment.py": {"path": "tests/test_openaxes_external_model_v1_experiment.py", "bytes": 76262, "sha256": "7774ddd5c8d37addd12a100482f81c3b98a6c1812cb5e78745bd5a19f5761862"},
}
V1_SOURCE_BLOB_OID_ORACLE = {
    "configs/openaxes_external_model_v1.yaml": "1ba7385381617295dbd36534aa6cd8e3d9bdeac5",
    "docs/OPENAXES_EXTERNAL_MODEL_V1_CONTRACT.md": "1f8b3d384d077217f5e0183e621a1dee4c3619be",
    "src/imu_vive_learning/io_utils.py": "333fd380c828b7a22fe5c4a4275b60ee0a7a4448",
    "src/imu_vive_learning/openaxes_external_model_v1_config.py": "8d59672b28c2a0f789ea4ff9dc9043e0b3a2f8ca",
    "src/imu_vive_learning/openaxes_external_model_v1_training.py": "631bc9c774a2852805685dcd1bc1f271173a5bcd",
    "src/imu_vive_learning/openaxes_external_model_v1_experiment.py": "a66b5a29e946946939aad326e9565b9fc33771ec",
    "tests/test_openaxes_external_model_v1_config.py": "55c5657d830e3419193ac4e5232d7a6b2f6e089c",
    "tests/test_openaxes_external_model_v1_training.py": "37624983131948046418329137bc7470e112b37a",
    "tests/test_openaxes_external_model_v1_experiment.py": "710cf53757ca5dcfedfbbcd4a29b72cbde4d7935",
}

METHOD_ORDER = (
    "full_gradient_only_mask_selected_control",
    "masked_erm",
    "group_mean_pord",
    "paired_pord",
    "paired_wrong_pair",
    "groupdro",
    "vrex",
    "mofe",
)
SEED_ORDER = (1701, 2603, 3907)
VIEW_ORDER = (
    "full",
    "outage_humerus",
    "outage_radius",
    "outage_carpus",
    "outage_digitus",
)
OUTAGE_VIEW_ORDER = VIEW_ORDER[1:]
SCENARIO_ORDER = tuple(
    sorted(
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    )
)
SIGNED_SHIFT_ORDER_MS = (-50, -20, -10, 10, 20, 50)
V1_PHASE_ORDER = (
    "execution.started",
    "training.completed",
    "outer_prediction.completed",
    "outer_eval.started",
    "outer_eval.completed",
    "bundle.completed",
    "result.completed",
)

SOURCE_AUDIT_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "created_at_utc",
        "audited_git_commit",
        "source_closure",
        "source_set_sha256",
        "erratum",
        "test_receipt",
        "assertions",
        "payload_sha256",
    }
)
SOURCE_AUDIT_ERRATUM_KEYS = frozenset(
    {
        "erratum_id",
        "active_receipt_path",
        "historical_source_audit_authorities",
        "sealed_001_evidence",
        "sealed_002_evidence",
        "terminal_bound_successful_run_id",
        "terminal_bound_successful_v1_run_project_relative_path",
        "terminal_bound_successful_preparation_run_project_relative_path",
        "terminal_bound_successful_supervisor_relative_path",
        "failure_classification",
        "torch_uuid_observation",
        "sealed_002_stage_counts",
        "no_v1_model_result_or_metric_artifact_existed_or_was_accessed",
        "scientific_projection_unchanged",
    }
)
HISTORICAL_SOURCE_AUDIT_AUTHORITY_KEYS = frozenset(
    {"authority_id", "receipt_path", "audited_git_commit", "evidence_git_commit"}
)
SEALED_001_EVIDENCE_KEYS = frozenset(
    {
        "run_id",
        "supervisor_terminal_relative_path",
        "supervisor_terminal_sha256",
        "public_preparation_manifest_sha256",
        "public_preparation_parquet_sha256",
        "public_preparation_complete_sha256",
        "public_validator_stdout_sha256",
        "persisted_sensor_groups_mapping_sha256",
        "failure_classification",
        "stage_counts",
    }
)
SEALED_001_STAGE_COUNT_KEYS = frozenset(
    {
        "model_prepared",
        "execution_session",
        "scientific_fit",
        "prediction",
        "evaluation",
        "result",
    }
)
SEALED_002_EVIDENCE_KEYS = frozenset(
    {
        "run_id",
        "audited_git_head",
        "supervisor_run_spec_relative_path",
        "supervisor_run_spec_sha256",
        "supervisor_run_spec_payload_sha256",
        "cleanup_blocked_current_relative_path",
        "cleanup_blocked_current_sha256",
        "cleanup_blocked_marker_relative_path",
        "cleanup_blocked_marker_sha256",
        "supervisor_log_relative_path",
        "supervisor_log_sha256",
        "launch_log_relative_path",
        "launch_log_sha256",
        "public_preparation_manifest_sha256",
        "public_preparation_parquet_sha256",
        "public_preparation_complete_sha256",
        "public_prepare_stdout_sha256",
        "public_validator_stdout_sha256",
        "public_preparation_valid_despite_supervisor_progress_flag",
        "model_prepared_sha256",
        "model_prepared_manifest_sha256",
        "task_index_sha256",
        "execution_session_sha256",
        "pre_fit_validation_sha256",
        "launch_started_sha256",
        "failure_status",
        "failure_message",
        "failure_stage",
        "failed_terminal_written",
        "automatic_deletion_or_refit_performed",
        "original_evidence_preserved",
        "marked_processes_after_verified_natural_exit",
        "manual_signal_performed",
        "cleanup_refused_survivor_role",
        "cleanup_refusal_classification",
        "cleanup_survivor_self_exited",
        "hardware_mapping_unchanged",
        "resume_scientifically_forbidden",
        "fresh_run_id_required",
    }
)
SEALED_002_STAGE_COUNT_KEYS = frozenset(
    {
        "public_preparation",
        "model_prepared",
        "execution_session",
        "launch_started",
        "worker_ready",
        "task_ledger",
        "attempt_ledger",
        "attempt_roster",
        "task_result_directory",
        "quarantine_member",
        "scientific_fit",
        "prediction",
        "evaluation",
        "result",
    }
)
TORCH_UUID_OBSERVATION_KEYS = frozenset(
    {
        "assigned_nvidia_uuid",
        "pytorch_version",
        "pytorch_property_type_name",
        "pytorch_property_string",
        "literal_gpu_prefix_is_only_representation_difference",
    }
)
SOURCE_TEST_KEYS = frozenset(
    {
        "python_launch_path_absolute",
        "python_real_executable_absolute",
        "python_real_executable_sha256",
        "python_version",
        "sys_prefix_absolute",
        "venv_site_packages_absolute",
        "working_directory_absolute",
        "argv_after_executable",
        "environment",
        "source_audit_bootstrap",
        "pyvenv_cfg_reference",
        "pyvenv_cfg_fields",
        "requirements_eval_lock_reference",
        "dependency_resolution",
        "pinned_distributions",
        "installed_distribution_inventory",
        "dependency_file_closure_before_tests",
        "dependency_file_closure_after_tests",
        "dependency_file_closure_unchanged",
        "runner_module_files",
        "source_import_boundary",
        "audited_git_commit",
        "source_set_sha256",
        "exit_code",
        "passed",
        "failed",
        "skipped",
        "xfailed",
        "stdout_utf8",
        "stdout_bytes",
        "stdout_sha256",
        "stderr_utf8",
        "stderr_bytes",
        "stderr_sha256",
        "report_bytes",
        "report_sha256",
    }
)
SOURCE_ASSERTION_KEYS = frozenset(
    {
        "source_written_without_v1_result_or_metric_access",
        "exact_eight_file_closure",
        "all_mandatory_tests_passed",
        "no_forbidden_import_or_action",
        "no_challenge_access",
    }
)
EXACT_TEST_ARGV = (
    "-P",
    "-S",
    "-m",
    "pytest",
    "-c",
    "/dev/null",
    "--noconftest",
    "-p",
    "no:cacheprovider",
    "-q",
    "tests/test_openaxes_external_native_v2_parent_receipt.py",
)
STATIC_TEST_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "PATH": "/usr/bin:/bin",
}

AUDITOR_MODULE_PATH = "src/imu_vive_learning/openaxes_external_native_v2_parent_receipt.py"
AUDITOR_TEST_PATH = "tests/test_openaxes_external_native_v2_parent_receipt.py"
REQUIREMENTS_LOCK_PATH = "requirements-eval.lock"
PINNED_TEST_DISTRIBUTIONS = ("pytest", "pluggy", "iniconfig")
TEST_PASS_COUNT = 12
PYTEST_EXACT_PASS_SUMMARY_RE = re.compile(
    rf"(?m)^{TEST_PASS_COUNT} passed in [0-9]+(?:\.[0-9]+)?s"
    r"(?: \((?:0|[1-9][0-9]*):[0-5][0-9]:[0-5][0-9]\))?$"
)
REQUIREMENTS_EVAL_PINS = (
    ("colorama", "0.4.6"),
    ("contourpy", "1.3.3"),
    ("cycler", "0.12.1"),
    ("fonttools", "4.63.0"),
    ("iniconfig", "2.3.0"),
    ("joblib", "1.5.3"),
    ("kiwisolver", "1.5.0"),
    ("matplotlib", "3.11.1"),
    ("narwhals", "2.24.0"),
    ("numpy", "2.3.5"),
    ("packaging", "26.3"),
    ("pandas", "2.3.3"),
    ("pillow", "12.3.0"),
    ("pluggy", "1.6.0"),
    ("pyarrow", "21.0.0"),
    ("pygments", "2.21.0"),
    ("pyparsing", "3.3.2"),
    ("pytest", "9.0.2"),
    ("python-dateutil", "2.9.0.post0"),
    ("pytz", "2026.3.post1"),
    ("PyYAML", "6.0.3"),
    ("scikit-learn", "1.9.0"),
    ("scipy", "1.17.1"),
    ("six", "1.17.0"),
    ("threadpoolctl", "3.6.0"),
    ("tzdata", "2026.3"),
    ("xgboost", "3.3.0"),
)


def _parse_exact_pytest_outcome_report(report: str) -> dict[str, int]:
    """Derive and validate the one closed pytest outcome grammar."""

    if type(report) is not str:
        _fail("exact source-audit report is not text")
    counts: dict[str, int] = {}
    for label in ("passed", "failed", "skipped", "xfailed"):
        matches = re.findall(rf"(?<![0-9])([0-9]+) {label}(?![A-Za-z])", report)
        counts[label] = sum(int(value) for value in matches)
    if (
        counts != {"passed": TEST_PASS_COUNT, "failed": 0, "skipped": 0, "xfailed": 0}
        or len(PYTEST_EXACT_PASS_SUMMARY_RE.findall(report)) != 1
        or any(
            token in report.lower()
            for token in ("error", "deselected", "xpassed", "warning")
        )
    ):
        _fail(
            "exact source-audit report does not derive exactly 12 passed and "
            "the exact accepted outcome counts"
        )
    return counts

REFERENCE_KEYS = frozenset({"path", "bytes", "sha256"})
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
GIT_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
RFC3339_UTC = re.compile(
    r"(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z\Z"
)
TASK_ID = re.compile(r"task_[0-9]{4}_[0-9a-f]{12}\Z")

TERMINAL_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "created_at_utc",
        "reference_roots",
        "auditor_source_audit",
        "git_lineage",
        "v1_lineage",
        "preparation_lineage",
        "references",
        "comparator_bundle_reference",
        "assertions",
        "payload_sha256",
    }
)
ROOT_KEYS = frozenset(
    {"workspace_root", "project_root", "v1_run_root", "preparation_run_root", "v2_run_root"}
)
AUDITOR_BINDING_KEYS = frozenset(
    {
        "auditor_identity",
        "git_commit",
        "source_set_sha256",
        "source_audit_reference",
        "source_audit_git_commit",
        "source_audit_git_blob_oid",
        "source_audit_receipt_path",
        "source_audit_erratum_id",
        "timing_scope",
        "terminal_bound_successful_run_id",
        "audited_source_ancestor_check_command",
        "audited_source_ancestor_check_exit_code",
        "source_audit_evidence_ancestor_check_command",
        "source_audit_evidence_ancestor_check_exit_code",
        "committed_before_v1_prepare",
        "source_audited_before_v1_prepare",
    }
)
GIT_LINEAGE_KEYS = frozenset(
    {
        "repository_object_format",
        "v2_preregistration_git_commit",
        "v2_yaml_git_blob_oid",
        "v2_markdown_git_blob_oid",
        "v2_yaml_git_blob_reference",
        "v2_markdown_git_blob_reference",
        "v1_model_contract_git_head",
        "ancestor_check_command",
        "ancestor_check_exit_code",
    }
)
V1_LINEAGE_KEYS = frozenset(
    {
        "contract",
        "canonical_config_sha256",
        "terminal_source_set_file_count",
        "terminal_source_set_sha256",
        "terminal_stage",
        "exact_task_count",
        "comparator_method_id",
    }
)
PREPARATION_LINEAGE_KEYS = frozenset(
    {
        "contract",
        "canonical_config_sha256",
        "manifest_sha256",
        "prepared_table_and_fold_identity_sha256",
    }
)
TERMINAL_REFERENCE_KEYS = frozenset(
    {
        "v1_complete_marker",
        "v1_development_manifest",
        "v1_task_index",
        "v1_prepared_manifest",
        "preparation_manifest",
        "preparation_table",
    }
)
TERMINAL_ASSERTION_KEYS = frozenset(
    {
        "v2_preregistration_blobs_verified_from_git_objects",
        "v2_preregistration_commit_is_ancestor_of_v1_model_contract_git_head",
        "parent_auditor_committed_and_source_audited_before_v1_prepare",
        "parent_auditor_source_identity_equals_copied_source_audit_receipt",
        "parent_auditor_audited_source_commit_is_ancestor_of_v1_model_contract_git_head",
        "parent_auditor_source_audit_receipt_commit_is_ancestor_of_v1_model_contract_git_head",
        "parent_auditor_source_audit_raw_bytes_equal_committed_git_blob",
        "v1_outcomes_did_not_modify_v2_preregistration",
        "no_v1_refit",
        "no_v1_source_truth_reread",
        "no_v1_outer_evaluator_replay",
        "masked_erm_only_bundle",
        "no_challenge_access",
    }
)

BUNDLE_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "method_id",
        "seed_order",
        "primary_fold_order",
        "view_order",
        "initial_state_by_seed",
        "primary_context_identity",
        "primary_seed_ensemble_outage_contexts",
        "primary_seed_ensemble_full_contexts",
        "primary_per_seed_outage_contexts",
        "derived_gate_inputs",
        "source_references",
        "payload_sha256",
    }
)
CONTEXT_IDENTITY_KEYS = frozenset({"context_count", "ordered_contexts", "identity_sha256"})
CONTEXT_KEYS = frozenset(
    {"fold_id", "scenario_id", "view_id", "rows", "identity_sha256", "rmse3d_m"}
)
PER_SEED_CONTEXT_KEYS = frozenset({"seed", *CONTEXT_KEYS})
DERIVED_KEYS = frozenset(
    {
        "A_mean_context_rmse3d_m",
        "WS_worst_scenario_mean_rmse3d_m",
        "WM_worst_mask_mean_rmse3d_m",
        "robust_J_m",
        "fold_outage_scores_m",
        "per_seed_J_m",
        "full_F_m",
    }
)
SOURCE_REFERENCE_KEYS = frozenset(
    {
        "v1_development_manifest",
        "masked_erm_primary_training_receipts",
        "masked_erm_primary_outer_metric_receipts",
    }
)


class OpenAxesNativeV2ParentReceiptError(RuntimeError):
    """Raised when a parent receipt input fails the frozen contract."""


def _fail(message: str) -> None:
    raise OpenAxesNativeV2ParentReceiptError(message)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        _fail(f"{name} must be an exact JSON object")
    return value


def _keys(value: Mapping[str, Any], expected: frozenset[str], *, name: str) -> None:
    if frozenset(value) != expected:
        missing = sorted(expected.difference(value))
        extra = sorted(set(value).difference(expected))
        _fail(f"{name} has a non-closed schema; missing={missing}, extra={extra}")


def _exact_str(value: Any, *, name: str, expected: str | None = None) -> str:
    if type(value) is not str or not value or (expected is not None and value != expected):
        _fail(f"{name} is not the required nonempty string")
    return value


def _exact_int(value: Any, *, name: str, expected: int | None = None) -> int:
    if type(value) is not int or (expected is not None and value != expected):
        _fail(f"{name} is not the required exact integer")
    return value


def _exact_bool(value: Any, *, name: str, expected: bool) -> bool:
    if type(value) is not bool or value is not expected:
        _fail(f"{name} is not exactly {expected}")
    return value


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        _fail(f"{name} is not a finite nonnegative binary64-origin JSON number")
    return value


def _hex64(value: Any, *, name: str) -> str:
    text = _exact_str(value, name=name)
    if HEX64.fullmatch(text) is None:
        _fail(f"{name} is not lowercase SHA-256")
    return text


def _git_sha1(value: Any, *, name: str) -> str:
    text = _exact_str(value, name=name)
    if GIT_SHA1.fullmatch(text) is None:
        _fail(f"{name} is not a lowercase SHA-1 Git object ID")
    return text


def _timestamp(value: Any, *, name: str = "created_at_utc") -> str:
    text = _exact_str(value, name=name)
    if RFC3339_UTC.fullmatch(text) is None:
        _fail(f"{name} is not canonical second-resolution UTC RFC3339")
    try:
        datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise OpenAxesNativeV2ParentReceiptError(f"{name} is not a real UTC date") from exc
    return text


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_json_value(value: Any, *, name: str = "JSON") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"{name} contains a non-finite number")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, name=f"{name}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail(f"{name} contains a non-string key")
            _validate_json_value(item, name=f"{name}.{key}")
        return
    _fail(f"{name} contains a non-JSON value")


def canonical_json_bytes(value: Any, *, final_newline: bool = False) -> bytes:
    _validate_json_value(value)
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise OpenAxesNativeV2ParentReceiptError("cannot encode canonical JSON") from exc
    return raw + (b"\n" if final_newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def strict_json_bytes(raw: bytes, *, name: str, require_canonical: bool = False) -> Any:
    if type(raw) is not bytes:
        _fail(f"{name} raw input is not bytes")

    def reject_constant(token: str) -> None:
        _fail(f"{name} contains forbidden JSON constant {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if type(key) is not str or key in result:
                _fail(f"{name} contains a duplicate or non-string key")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OpenAxesNativeV2ParentReceiptError(f"{name} is not strict UTF-8 JSON") from exc
    _validate_json_value(value, name=name)
    if require_canonical and raw != canonical_json_bytes(value, final_newline=True):
        _fail(f"{name} is not canonical JSON with exactly one final LF")
    return value


def _canonical_relative_path(value: Any, *, name: str) -> str:
    path = _exact_str(value, name=name)
    if (
        path.startswith("/")
        or path.endswith("/")
        or "\\" in path
        or "//" in path
        or path in {".", ".."}
    ):
        _fail(f"{name} is not a canonical relative POSIX path")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _fail(f"{name} contains a forbidden path component")
    if PurePosixPath(path).as_posix() != path:
        _fail(f"{name} is not canonical POSIX syntax")
    return path


def _absolute_root(value: str | os.PathLike[str], *, name: str) -> Path:
    raw = os.fspath(value)
    candidate = Path(raw)
    if not candidate.is_absolute() or os.path.normpath(raw) != raw:
        _fail(f"{name} must be canonical absolute syntax")
    resolved = Path(os.path.realpath(raw))
    if str(resolved) != raw:
        _fail(f"{name} must equal its canonical realpath")
    current = Path(resolved.anchor)
    for component in resolved.parts[1:]:
        current = current / component
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise OpenAxesNativeV2ParentReceiptError(f"{name} does not exist") from exc
        if stat.S_ISLNK(info.st_mode):
            _fail(f"{name} contains a symbolic-link component")
    if not resolved.is_dir():
        _fail(f"{name} is not an existing directory")
    return resolved


def _directory_identity(info: os.stat_result) -> tuple[int, int, int]:
    """Identity of a directory entry, excluding mutable directory metadata."""

    return (info.st_dev, info.st_ino, info.st_mode)


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Identity and stability fields for one regular-file read."""

    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


@dataclass
class _PinnedRoot:
    """An absolute directory identity held componentwise from the filesystem root."""

    path: Path
    descriptors: list[int]
    components: list[str]
    identities: list[tuple[int, int, int]]

    @property
    def descriptor(self) -> int:
        return self.descriptors[-1]

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    def __truediv__(self, other: str) -> Path:
        return self.path / other

    def revalidate(self, *, name: str) -> None:
        if len(self.descriptors) != len(self.identities):
            _fail(f"{name} pinned-root bookkeeping changed")
        for index, descriptor in enumerate(self.descriptors):
            if _directory_identity(os.fstat(descriptor)) != self.identities[index]:
                _fail(f"{name} pinned directory identity changed")
            if index:
                observed = os.stat(
                    self.components[index - 1],
                    dir_fd=self.descriptors[index - 1],
                    follow_symlinks=False,
                )
                if _directory_identity(observed) != self.identities[index]:
                    _fail(f"{name} parent entry no longer names the pinned directory")

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.descriptors.clear()


def _pin_root(value: str | os.PathLike[str], *, name: str) -> _PinnedRoot:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        _fail("pinned-root issuance requires Linux O_DIRECTORY and O_NOFOLLOW")
    path = _absolute_root(value, name=name)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    identities: list[tuple[int, int, int]] = []
    components = list(path.parts[1:])
    try:
        descriptor = os.open("/", flags)
        descriptors.append(descriptor)
        identities.append(_directory_identity(os.fstat(descriptor)))
        for component in components:
            descriptor = os.open(component, flags, dir_fd=descriptor)
            descriptors.append(descriptor)
            info = os.fstat(descriptor)
            if not stat.S_ISDIR(info.st_mode):
                _fail(f"{name} component is not a directory")
            identities.append(_directory_identity(info))
        pinned = _PinnedRoot(path, descriptors, components, identities)
        pinned.revalidate(name=name)
        return pinned
    except Exception:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _read_nofollow(root: Path | _PinnedRoot, relative: str, *, name: str) -> bytes:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        _fail("no-follow reads require Linux O_DIRECTORY and O_NOFOLLOW")
    path = _canonical_relative_path(relative, name=f"{name}.path")
    root_path = root.path if isinstance(root, _PinnedRoot) else root
    expected = root_path.joinpath(*path.split("/"))
    if os.path.commonpath((str(root_path), str(expected))) != str(root_path):
        _fail(f"{name} escapes its reference root")
    flags_dir = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags_file = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    directory_identities: list[tuple[int, int, int]] = []
    try:
        descriptor = (
            os.dup(root.descriptor)
            if isinstance(root, _PinnedRoot)
            else os.open(root_path, flags_dir)
        )
        descriptors.append(descriptor)
        directory_identities.append(_directory_identity(os.fstat(descriptor)))
        parts = path.split("/")
        for component in parts[:-1]:
            descriptor = os.open(component, flags_dir, dir_fd=descriptor)
            descriptors.append(descriptor)
            directory_identities.append(_directory_identity(os.fstat(descriptor)))
        file_descriptor = os.open(parts[-1], flags_file, dir_fd=descriptor)
        descriptors.append(file_descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{name} final object is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
        identity_before = _file_identity(before)
        identity_after = _file_identity(after)
        if identity_before != identity_after:
            _fail(f"{name} changed during its raw read")
        if _file_identity(
            os.stat(parts[-1], dir_fd=descriptors[-2], follow_symlinks=False)
        ) != identity_before:
            _fail(f"{name} parent entry no longer names the opened regular file")
        for index, identity in enumerate(directory_identities):
            if _directory_identity(os.fstat(descriptors[index])) != identity:
                _fail(f"{name} parent directory identity changed during raw read")
            if index:
                observed_parent_entry = os.stat(
                    parts[index - 1],
                    dir_fd=descriptors[index - 1],
                    follow_symlinks=False,
                )
                if _directory_identity(observed_parent_entry) != identity:
                    _fail(f"{name} parent component was replaced during raw read")
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            _fail(f"{name} byte count changed during read")
        return raw
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError(f"cannot no-follow read {name}") from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _reference(path: str, raw: bytes) -> dict[str, Any]:
    canonical = _canonical_relative_path(path, name="reference.path")
    return {"path": canonical, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _validate_reference_shape(value: Any, *, name: str, expected_path: str | None = None) -> dict[str, Any]:
    reference = _mapping(value, name=name)
    _keys(reference, REFERENCE_KEYS, name=name)
    path = _canonical_relative_path(reference["path"], name=f"{name}.path")
    if expected_path is not None and path != expected_path:
        _fail(f"{name} path differs from the frozen path")
    count = _exact_int(reference["bytes"], name=f"{name}.bytes")
    if count < 0:
        _fail(f"{name}.bytes is negative")
    digest = _hex64(reference["sha256"], name=f"{name}.sha256")
    return {"path": path, "bytes": count, "sha256": digest}


def _verify_reference(root: Path, value: Any, *, name: str, expected_path: str | None = None) -> tuple[dict[str, Any], bytes]:
    reference = _validate_reference_shape(value, name=name, expected_path=expected_path)
    raw = _read_nofollow(root, reference["path"], name=name)
    observed = _reference(reference["path"], raw)
    if observed != reference:
        _fail(f"{name} raw bytes do not equal their reference")
    return observed, raw


def _raw_reference(root: Path, relative: str, *, name: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_nofollow(root, relative, name=name)
    return _reference(relative, raw), raw


def _git_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }


def _validate_git_argv(arguments: Sequence[str], *, name: str) -> list[str]:
    argv = list(arguments)
    if not argv or any(type(item) is not str or not item for item in argv):
        _fail(f"{name} Git argv is malformed")

    def sha(value: str) -> bool:
        return GIT_SHA1.fullmatch(value) is not None

    def commit_expr(value: str) -> bool:
        return len(value) == 49 and value.endswith("^{commit}") and sha(value[:-9])

    def tree_expr(value: str) -> bool:
        if len(value) <= 41 or value[40] != ":" or not sha(value[:40]):
            return False
        try:
            logical = value[41:]
            if not logical.startswith(f"{PROJECT_DIRECTORY}/"):
                return False
            _canonical_relative_path(logical[len(PROJECT_DIRECTORY) + 1 :], name="Git tree path")
        except OpenAxesNativeV2ParentReceiptError:
            return False
        return True

    allowed = False
    if argv in (["rev-parse", "--show-toplevel"], ["rev-parse", "--show-object-format"]):
        allowed = True
    elif len(argv) == 3 and argv[:2] == ["rev-parse", "--verify"] and commit_expr(argv[2]):
        allowed = True
    elif len(argv) == 2 and argv[0] == "rev-parse" and tree_expr(argv[1]):
        allowed = True
    elif len(argv) == 3 and argv[:2] == ["cat-file", "-t"] and sha(argv[2]):
        allowed = True
    elif len(argv) == 3 and argv[:2] == ["cat-file", "blob"] and tree_expr(argv[2]):
        allowed = True
    elif argv == ["hash-object", "--stdin"]:
        allowed = True
    elif len(argv) == 4 and argv[:2] == ["merge-base", "--is-ancestor"] and sha(argv[2]) and sha(argv[3]):
        allowed = True
    elif argv == ["for-each-ref", "--format=%(refname)", "refs/replace"]:
        allowed = True
    elif (
        len(argv) == 4
        and argv[0] == "ls-tree"
        and sha(argv[1])
        and argv[2] == "--"
        and argv[3].startswith(f"{PROJECT_DIRECTORY}/")
    ):
        try:
            _canonical_relative_path(
                argv[3][len(PROJECT_DIRECTORY) + 1 :], name="Git ls-tree path"
            )
            allowed = True
        except OpenAxesNativeV2ParentReceiptError:
            allowed = False
    if not allowed:
        _fail(f"{name} Git argv is outside the fixed allowlist")
    return argv


def _git_bytes(repo: Path, arguments: Sequence[str], *, name: str, stdin: bytes | None = None) -> bytes:
    argv = _validate_git_argv(arguments, name=name)
    try:
        result = subprocess.run(
            ["/usr/bin/git", "-C", str(repo), *argv],
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=_git_environment(),
        )
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError(f"cannot execute fixed Git command for {name}") from exc
    if result.returncode != 0:
        _fail(f"fixed Git command failed closed for {name} with exit {result.returncode}")
    return result.stdout


def _git_text(repo: Path, arguments: Sequence[str], *, name: str) -> str:
    try:
        value = _git_bytes(repo, arguments, name=name).decode("ascii").strip()
    except UnicodeError as exc:
        raise OpenAxesNativeV2ParentReceiptError(f"{name} Git output is not ASCII") from exc
    if not value:
        _fail(f"{name} Git output is empty")
    return value


def _repo_path(relative: str) -> str:
    logical = _canonical_relative_path(relative, name="Git logical path")
    return f"{PROJECT_DIRECTORY}/{logical}"


def _verify_commit(repo: Path, commit: str, *, name: str) -> str:
    expected = _git_sha1(commit, name=name)
    observed = _git_text(repo, ["rev-parse", "--verify", f"{expected}^{{commit}}"], name=name)
    if observed != expected:
        _fail(f"{name} does not resolve to the exact commit")
    return expected


def _git_blob(repo: Path, commit: str, relative: str, *, name: str) -> tuple[str, dict[str, Any], bytes]:
    commit_id = _verify_commit(repo, commit, name=f"{name} commit")
    repo_path = _repo_path(relative)
    entry_raw = _git_bytes(
        repo, ["ls-tree", commit_id, "--", repo_path], name=f"{name} tree entry"
    )
    try:
        entry = entry_raw.decode("utf-8").rstrip("\n")
    except UnicodeError as exc:
        raise OpenAxesNativeV2ParentReceiptError(f"{name} tree entry is not UTF-8") from exc
    match = re.fullmatch(r"(100644|100755) blob ([0-9a-f]{40})\t(.+)", entry)
    if match is None or match.group(3) != repo_path:
        _fail(f"{name} Git tree entry is not one exact regular-file mode")
    object_id = _git_text(repo, ["rev-parse", f"{commit_id}:{repo_path}"], name=f"{name} OID")
    _git_sha1(object_id, name=f"{name} OID")
    if match.group(2) != object_id:
        _fail(f"{name} ls-tree OID differs from rev-parse OID")
    object_type = _git_text(repo, ["cat-file", "-t", object_id], name=f"{name} type")
    if object_type != "blob":
        _fail(f"{name} is not a regular Git blob")
    raw = _git_bytes(repo, ["cat-file", "blob", f"{commit_id}:{repo_path}"], name=f"{name} bytes")
    # hash-object needs the exact blob bytes on stdin; keeping it in the sole Git wrapper
    # makes the subprocess surface statically auditable.
    hashed_raw = _git_bytes(repo, ["hash-object", "--stdin"], name=f"{name} hash-object", stdin=raw)
    try:
        hashed_oid = hashed_raw.decode("ascii").strip()
    except UnicodeError as exc:
        raise OpenAxesNativeV2ParentReceiptError(f"{name} hash-object output is invalid") from exc
    if hashed_oid != object_id:
        _fail(f"{name} OID differs from git hash-object of exact bytes")
    return object_id, _reference(relative, raw), raw


def _git_ancestor(repo: Path, ancestor: str, descendant: str, *, name: str) -> str:
    old = _verify_commit(repo, ancestor, name=f"{name} ancestor")
    new = _verify_commit(repo, descendant, name=f"{name} descendant")
    _git_bytes(repo, ["merge-base", "--is-ancestor", old, new], name=name)
    return f"git merge-base --is-ancestor {old} {new}"


def _repository(workspace: Path) -> Path:
    root = _git_text(workspace, ["rev-parse", "--show-toplevel"], name="workspace repository root")
    if root != str(workspace):
        _fail("workspace_root does not equal git rev-parse --show-toplevel")
    object_format = _git_text(workspace, ["rev-parse", "--show-object-format"], name="repository object format")
    if object_format != "sha1":
        _fail("repository object format is not the preregistered sha1")
    try:
        git_directory_info = os.lstat(workspace / ".git")
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError("cannot bind repository .git directory") from exc
    if stat.S_ISLNK(git_directory_info.st_mode) or not stat.S_ISDIR(git_directory_info.st_mode):
        _fail("repository .git is not a nonsymlink directory")
    replace_refs = _git_bytes(
        workspace,
        ["for-each-ref", "--format=%(refname)", "refs/replace"],
        name="repository replace refs",
    )
    if replace_refs:
        _fail("repository contains forbidden replace refs")
    grafts = workspace / ".git/info/grafts"
    try:
        graft_info = os.lstat(grafts)
    except FileNotFoundError:
        graft_info = None
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError("cannot inspect repository grafts") from exc
    if graft_info is not None:
        _fail("repository contains forbidden legacy grafts")
    return workspace


def _payload_hash(value: Mapping[str, Any], *, key: str, name: str) -> None:
    recorded = _hex64(value[key], name=f"{name}.{key}")
    expected = canonical_sha256({field: item for field, item in value.items() if field != key})
    if recorded != expected:
        _fail(f"{name} payload hash mismatch")


def audit_python_source_safety(raw: bytes) -> dict[str, bool]:
    """Apply defense-in-depth AST checks to an authenticated auditor blob.

    This is deliberately not claimed as a proof against a malicious auditor
    rewriting its own checker.  Git blob authentication and review remain the
    authority; this check rejects accidental or obvious capability expansion.
    """

    try:
        source = raw.decode("utf-8")
        tree = ast.parse(source)
    except (UnicodeError, SyntaxError) as exc:
        raise OpenAxesNativeV2ParentReceiptError("auditor source is not valid UTF-8 Python") from exc
    allowed_imports = {
        "__future__",
        "argparse",
        "ast",
        "base64",
        "csv",
        "dataclasses",
        "datetime",
        "hashlib",
        "importlib",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "stat",
        "subprocess",
        "sys",
        "typing",
    }
    forbidden_roots = {
        "torch",
        "cuda",
        "numpy",
        "pandas",
        "pyarrow",
        "requests",
        "urllib",
        "http",
        "socket",
        "ftplib",
        "openaxes_external_model_v1_experiment",
        "openaxes_external_model_v1_training",
        "openaxes_external_v1",
    }
    imported: set[str] = set()
    import_statements: set[tuple[str, tuple[str, ...]]] = set()
    subprocess_calls: list[str] = []
    forbidden_calls: list[str] = []
    direct_write_calls: list[str] = []
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
        if isinstance(node, ast.Import):
            if any(alias.asname is not None or alias.name == "*" for alias in node.names):
                forbidden_calls.append("aliased_import")
            imported.update(alias.name.split(".")[0] for alias in node.names)
            import_statements.add(("import", tuple(alias.name for alias in node.names)))
        elif isinstance(node, ast.ImportFrom):
            if node.level or any(alias.asname is not None or alias.name == "*" for alias in node.names):
                forbidden_calls.append("relative_import")
            elif node.module:
                imported.add(node.module.split(".")[0])
            import_statements.add(
                (f"from:{node.module or ''}", tuple(alias.name for alias in node.names))
            )
        elif isinstance(node, ast.Call):
            def enclosing_function() -> str:
                current: ast.AST | None = node
                while current in parents:
                    current = parents[current]
                    if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        return current.name
                return ""

            def dotted(target: ast.AST) -> str:
                if isinstance(target, ast.Name):
                    return target.id
                if isinstance(target, ast.Attribute):
                    prefix = dotted(target.value)
                    return f"{prefix}.{target.attr}" if prefix else target.attr
                return ""

            call_name = dotted(node.func)
            if call_name == "subprocess.run":
                subprocess_calls.append(enclosing_function())
            if call_name.startswith("subprocess.") and call_name != "subprocess.run":
                forbidden_calls.append(call_name)
            if call_name == "getattr":
                valid_getattr = (
                    len(node.args) == 3
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "os"
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in {"O_DIRECTORY", "O_NOFOLLOW"}
                    and isinstance(node.args[2], ast.Constant)
                    and node.args[2].value == 0
                )
                if not valid_getattr:
                    forbidden_calls.append("dynamic_getattr")
            if call_name in {
                "eval",
                "exec",
                "compile",
                "__import__",
                "importlib.import_module",
                "os.system",
                "os.popen",
                "os.fork",
                "os.forkpty",
                "os.posix_spawn",
                "os.posix_spawnp",
                "os.spawnl",
                "os.spawnle",
                "os.spawnlp",
                "os.spawnlpe",
                "os.spawnv",
                "os.spawnve",
                "os.spawnvp",
                "os.spawnvpe",
                "os.execv",
                "os.execve",
                "os.execl",
                "os.execle",
                "os.execlp",
                "os.execlpe",
                "os.execvp",
                "os.execvpe",
            }:
                forbidden_calls.append(call_name)
            lowered = call_name.lower()
            if any(
                token in lowered
                for token in (
                    "urlopen",
                    "connect",
                    "network",
                    "train_model",
                    "model_train",
                    "backward",
                    "optimizer.step",
                    "predict_outer",
                    "model_predict",
                    "evaluate_outer",
                    "evaluator",
                    "source_truth",
                    "truth_loader",
                    "reconstruct_openaxes",
                    "challenge",
                )
            ):
                forbidden_calls.append(call_name)
            if call_name in {
                "open",
                "Path.write_bytes",
                "Path.write_text",
                "write_bytes",
                "write_text",
                "unlink",
                "os.remove",
                "os.rmdir",
                "os.rename",
                "os.replace",
                "shutil.rmtree",
            } or call_name.endswith((".write_bytes", ".write_text")) or (
                call_name.endswith(".unlink") and call_name != "os.unlink"
            ):
                direct_write_calls.append(call_name)
            if call_name in {"os.write", "os.unlink"} and enclosing_function() != "_write_exclusive":
                direct_write_calls.append(call_name)
            if call_name == "os.mkdir" and enclosing_function() not in {
                "_issue_parent_receipts_pinned",
                "_write_new_path",
            }:
                direct_write_calls.append(call_name)
            if call_name == "os.fsync" and enclosing_function() not in {
                "_write_exclusive",
                "_issue_parent_receipts_pinned",
                "_write_new_path",
            }:
                direct_write_calls.append(call_name)
            if call_name == "os.open":
                writes = any(
                    isinstance(argument, ast.Attribute)
                    and isinstance(argument.value, ast.Name)
                    and argument.value.id == "os"
                    and argument.attr in {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"}
                    for argument in ast.walk(node)
                )
                if writes and enclosing_function() != "_write_exclusive":
                    direct_write_calls.append("os.open(write)")
    expected_import_statements = {
        ("from:__future__", ("annotations",)),
        ("import", ("argparse",)),
        ("import", ("ast",)),
        ("import", ("base64",)),
        ("import", ("csv",)),
        ("from:dataclasses", ("dataclass",)),
        ("from:datetime", ("datetime", "timezone")),
        ("import", ("hashlib",)),
        ("import", ("importlib.metadata",)),
        ("import", ("importlib.util",)),
        ("import", ("json",)),
        ("import", ("math",)),
        ("import", ("os",)),
        ("from:pathlib", ("Path", "PurePosixPath")),
        ("import", ("re",)),
        ("import", ("stat",)),
        ("import", ("subprocess",)),
        ("import", ("sys",)),
        ("from:typing", ("Any", "Mapping", "Sequence")),
    }
    if (
        import_statements != expected_import_statements
        or not imported.issubset(allowed_imports)
        or imported.intersection(forbidden_roots)
    ):
        _fail("auditor source imports a forbidden or undeclared module")
    if sorted(subprocess_calls) != ["_git_bytes", "_run_exact_source_tests"]:
        _fail("subprocess.run is not confined to the two fixed process wrappers")
    if forbidden_calls:
        _fail(f"auditor source has a forbidden call surface: {sorted(forbidden_calls)}")
    if direct_write_calls:
        _fail(
            "auditor source has an unconfined direct filesystem mutation surface: "
            f"{sorted(direct_write_calls)}"
        )
    return {
        "stdlib_only_imports": True,
        "subprocess_confined_to_fixed_git_and_test_wrappers": True,
        "no_training_prediction_truth_evaluator_cuda_or_network_call": True,
    }


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _parse_requirements_lock(raw: bytes) -> list[tuple[str, str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise OpenAxesNativeV2ParentReceiptError("requirements-eval.lock is not UTF-8") from exc
    pins: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "--hash=" in line or " ; " in line or line.count("==") != 1:
            _fail(f"requirements-eval.lock line {line_number} is not one plain version pin")
        distribution, version = line.split("==", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", distribution) or not re.fullmatch(
            r"[A-Za-z0-9_.+!-]+", version
        ):
            _fail(f"requirements-eval.lock line {line_number} is malformed")
        normalized = _normalized_distribution_name(distribution)
        if normalized in seen:
            _fail("requirements-eval.lock contains a duplicate normalized distribution")
        seen.add(normalized)
        pins.append((distribution, version))
    if not pins:
        _fail("requirements-eval.lock contains no version pins")
    return pins


def _exact_test_environment(site_packages: Path) -> dict[str, str]:
    environment = dict(STATIC_TEST_ENVIRONMENT)
    environment["PYTHONPATH"] = f"src:{site_packages}"
    return environment


def _file_binding(path: Path, *, root: Path, name: str) -> dict[str, Any]:
    try:
        canonical = Path(os.path.realpath(path))
        info = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError(f"cannot bind {name}") from exc
    if canonical != path or stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _fail(f"{name} is not one canonical nonsymlink regular file")
    if os.path.commonpath((str(root), str(path))) != str(root):
        _fail(f"{name} is outside its frozen root")
    return {
        "path_absolute": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _source_audit_bootstrap_evidence(project: Path) -> dict[str, Any]:
    """Fail closed unless this source-audit process itself is hermetic."""

    launch = project / ".venv312/bin/python"
    site_packages = project / ".venv312/lib/python3.12/site-packages"
    expected_environment = _exact_test_environment(site_packages)
    module = project / AUDITOR_MODULE_PATH
    observed_module = Path(__file__).absolute()
    if (
        Path(sys.executable).absolute() != launch
        or Path(os.path.realpath(sys.executable)) != Path("/usr/bin/python3.12")
        or sys.flags.safe_path is not True
        or sys.flags.no_site != 1
        or sys.flags.no_user_site != 1
        or sys.flags.dont_write_bytecode != 1
        or dict(os.environ) != expected_environment
        or observed_module != module
        or Path(os.path.realpath(observed_module)) != observed_module
    ):
        _fail(
            "production source-audit CLI must itself run via env -i, project .venv312, -P -S, and exact PYTHONPATH"
        )
    expected_path = [
        str(project / "src"),
        str(site_packages),
        "/usr/lib/python312.zip",
        "/usr/lib/python3.12",
        "/usr/lib/python3.12/lib-dynload",
    ]
    if sys.path != expected_path:
        _fail("source-audit bootstrap sys.path differs from the frozen -P -S closure")
    return {
        "safe_path": True,
        "no_site": 1,
        "no_user_site": 1,
        "dont_write_bytecode": 1,
        "sys_executable_absolute": str(launch),
        "sys_real_executable_absolute": "/usr/bin/python3.12",
        "module_file_absolute": str(module),
        "sys_path_exact": expected_path,
        "environment_exact": expected_environment,
    }


def _parse_pyvenv_cfg(raw: bytes, project: Path) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise OpenAxesNativeV2ParentReceiptError("pyvenv.cfg is not UTF-8") from exc
    fields: dict[str, str] = {}
    for line in lines:
        if line.count(" = ") != 1:
            _fail("pyvenv.cfg does not use the exact key/value syntax")
        key, value = line.split(" = ", 1)
        if key in fields or not key or not value:
            _fail("pyvenv.cfg contains a duplicate or empty field")
        fields[key] = value
    expected = {
        "home": "/usr/bin",
        "include-system-site-packages": "false",
        "version": "3.12.3",
        "executable": "/usr/bin/python3.12",
        "command": f"/usr/bin/python3.12 -m venv {project}/.venv312",
    }
    if fields != expected:
        _fail("pyvenv.cfg fields differ from the exact project venv")
    return fields


def _distribution_file_closure(
    *,
    venv_root: Path,
    site_packages: Path,
    owner: str,
    expected_version: str,
    distribution: importlib.metadata.Distribution,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dist_info = Path(distribution._path)  # type: ignore[attr-defined]
    if (
        Path(os.path.realpath(dist_info)) != dist_info
        or dist_info.parent != site_packages
        or not dist_info.name.endswith(".dist-info")
    ):
        _fail(f"pinned distribution {owner} has an alternate dist-info authority")
    metadata_path = dist_info / "METADATA"
    record_path = dist_info / "RECORD"
    metadata_relative = metadata_path.relative_to(venv_root).as_posix()
    record_relative = record_path.relative_to(venv_root).as_posix()
    metadata_raw = _read_nofollow(
        venv_root, metadata_relative, name=f"{owner} dist-info METADATA"
    )
    record_raw = _read_nofollow(
        venv_root, record_relative, name=f"{owner} dist-info RECORD"
    )
    try:
        metadata_text = metadata_raw.decode("utf-8")
        record_text = record_raw.decode("utf-8")
        rows = list(csv.reader(record_text.splitlines()))
    except (UnicodeError, csv.Error) as exc:
        raise OpenAxesNativeV2ParentReceiptError(
            f"{owner} dist-info METADATA or RECORD is malformed"
        ) from exc
    metadata_names = distribution.metadata.get_all("Name")
    metadata_versions = distribution.metadata.get_all("Version")
    if (
        metadata_names is None
        or len(metadata_names) != 1
        or _normalized_distribution_name(metadata_names[0]) != owner
        or metadata_versions != [expected_version]
    ):
        _fail(f"{owner} METADATA Name/Version differs from the exact pin")
    distribution_files = distribution.files
    if distribution_files is None:
        _fail(f"{owner} importlib Distribution.files is unavailable")
    distribution_file_paths = [str(path).replace("\\", "/") for path in distribution_files]
    entries: list[dict[str, Any]] = []
    seen_record_paths: set[str] = set()
    seen_resolved_paths: set[str] = set()
    record_self_path = f"{dist_info.name}/RECORD"
    generated_pyc = re.compile(
        r"(?:[^/]+/)*__pycache__/[A-Za-z0-9_.-]+\.cpython-312(?:\.opt-[12])?\.pyc"
    )
    for row in rows:
        if len(row) != 3 or not row[0]:
            _fail(f"{owner} dist-info RECORD row is malformed")
        record_path = _exact_str(row[0], name=f"{owner} RECORD path")
        if (
            record_path.startswith("/")
            or record_path.endswith("/")
            or "\\" in record_path
            or "//" in record_path
        ):
            _fail(f"{owner} RECORD path is not canonical POSIX syntax")
        components = record_path.split("/")
        if any(component in {"", "."} for component in components):
            _fail(f"{owner} RECORD path contains an empty or dot component")
        parent_prefix = 0
        while parent_prefix < len(components) and components[parent_prefix] == "..":
            parent_prefix += 1
        if parent_prefix not in (0, 3) or ".." in components[parent_prefix:]:
            _fail(f"{owner} RECORD path has a noncanonical parent traversal")
        if parent_prefix == len(components):
            _fail(f"{owner} RECORD path resolves to the venv root")
        if record_path in seen_record_paths:
            _fail(f"{owner} dist-info RECORD contains a duplicate raw path")
        seen_record_paths.add(record_path)
        absolute = Path(os.path.normpath(site_packages / record_path))
        if os.path.commonpath((str(venv_root), str(absolute))) != str(venv_root):
            _fail(f"{owner} installed file escapes the venv")
        relative = absolute.relative_to(venv_root).as_posix()
        _canonical_relative_path(relative, name=f"{owner} installed file path")
        if relative in seen_resolved_paths:
            _fail(f"{owner} dist-info RECORD aliases one installed file")
        seen_resolved_paths.add(relative)
        raw = _read_nofollow(
            venv_root, relative, name=f"{owner} installed file {relative}"
        )
        record_hash, record_size = row[1], row[2]
        if bool(record_hash) != bool(record_size):
            _fail(f"{owner} RECORD hash and size must be both present or both empty")
        if not record_hash:
            if record_path != record_self_path and generated_pyc.fullmatch(record_path) is None:
                _fail(f"{owner} RECORD permits empty integrity only for RECORD or generated CPython 3.12 pyc")
        else:
            if re.fullmatch(r"sha256=[A-Za-z0-9_-]{43}", record_hash) is None:
                _fail(f"{owner} RECORD hash is not canonical urlsafe-base64 SHA-256")
            encoded = record_hash.removeprefix("sha256=")
            try:
                recorded_digest = base64.urlsafe_b64decode(encoded + "=")
            except ValueError as exc:
                raise OpenAxesNativeV2ParentReceiptError(
                    f"{owner} RECORD hash is not decodable SHA-256"
                ) from exc
            if (
                len(recorded_digest) != 32
                or base64.urlsafe_b64encode(recorded_digest).rstrip(b"=").decode("ascii")
                != encoded
                or recorded_digest != hashlib.sha256(raw).digest()
            ):
                _fail(f"{owner} RECORD hash differs from installed raw bytes")
            if re.fullmatch(r"0|[1-9][0-9]*", record_size) is None:
                _fail(f"{owner} RECORD size is not canonical decimal")
            if int(record_size) != len(raw):
                _fail(f"{owner} RECORD size differs from installed raw bytes")
        entries.append(
            {
                "path": relative,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "owner": owner,
            }
        )
    if len(set(distribution_file_paths)) != len(distribution_file_paths):
        _fail(f"{owner} importlib Distribution.files contains duplicates")
    if set(distribution_file_paths) != seen_record_paths:
        _fail(f"{owner} Distribution.files differs from the exact RECORD path set")
    entries.sort(key=lambda item: item["path"])
    summary = {
        "dist_info_directory": dist_info.relative_to(venv_root).as_posix(),
        "metadata_reference": _reference(metadata_relative, metadata_raw),
        "record_reference": _reference(record_relative, record_raw),
        "installed_file_count": len(entries),
        "installed_total_bytes": sum(item["bytes"] for item in entries),
        "installed_file_set_sha256": canonical_sha256(entries),
    }
    return summary, entries


def _dependency_evidence(project: Path, lock_raw: bytes) -> dict[str, Any]:
    venv_root = project / ".venv312"
    site_packages = venv_root / "lib/python3.12/site-packages"
    if not site_packages.is_dir() or Path(os.path.realpath(site_packages)) != site_packages:
        _fail("exact venv site-packages directory is absent or noncanonical")
    pins = _parse_requirements_lock(lock_raw)
    if tuple(pins) != REQUIREMENTS_EVAL_PINS:
        _fail("requirements-eval.lock differs from the frozen version-pin set/order")
    pyvenv_raw = _read_nofollow(venv_root, "pyvenv.cfg", name="project pyvenv.cfg")
    pyvenv_reference = _reference(".venv312/pyvenv.cfg", pyvenv_raw)
    pyvenv_fields = _parse_pyvenv_cfg(pyvenv_raw, project)
    installed: dict[str, importlib.metadata.Distribution] = {}
    inventory: list[dict[str, str]] = []
    for distribution in importlib.metadata.distributions(path=[str(site_packages)]):
        metadata_name = distribution.metadata.get("Name")
        if not metadata_name:
            _fail("installed distribution lacks a metadata Name")
        normalized = _normalized_distribution_name(metadata_name)
        if normalized in installed:
            _fail("venv has duplicate normalized distribution authorities")
        installed[normalized] = distribution
        inventory.append(
            {
                "name": metadata_name,
                "normalized_name": normalized,
                "version": distribution.version,
            }
        )
    inventory.sort(key=lambda item: item["normalized_name"])
    evidence: list[dict[str, Any]] = []
    union_entries: list[dict[str, Any]] = []
    for name, version in pins:
        normalized = _normalized_distribution_name(name)
        distribution = installed.get(normalized)
        if distribution is None or distribution.version != version:
            _fail(f"pinned distribution {name}=={version} is not installed exactly")
        closure, entries = _distribution_file_closure(
            venv_root=venv_root,
            site_packages=site_packages,
            owner=normalized,
            expected_version=version,
            distribution=distribution,
        )
        evidence.append(
            {
                "name": name,
                "normalized_name": normalized,
                "pinned_version": version,
                "installed_version": distribution.version,
                **closure,
            }
        )
        union_entries.extend(entries)
    union_entries.sort(key=lambda item: (item["path"], item["owner"]))
    if len({(item["path"], item["owner"]) for item in union_entries}) != len(
        union_entries
    ):
        _fail("pinned distribution file closure has duplicate owner/path entries")
    runner_files: dict[str, Any] = {}
    for name in PINNED_TEST_DISTRIBUTIONS:
        module_file = site_packages / name / "__init__.py"
        runner_files[name] = _file_binding(
            module_file, root=site_packages, name=f"{name} runner module"
        )
    return {
        "site_packages": site_packages,
        "pyvenv_cfg_reference": pyvenv_reference,
        "pyvenv_cfg_fields": pyvenv_fields,
        "pinned_distributions": evidence,
        "installed_distribution_inventory": inventory,
        "dependency_file_closure": {
            "file_count": len(union_entries),
            "total_bytes": sum(item["bytes"] for item in union_entries),
            "manifest_sha256": canonical_sha256(union_entries),
        },
        "runner_module_files": runner_files,
    }


def _source_import_boundary(project: Path, closure: Mapping[str, Any]) -> dict[str, Any]:
    source_root = project / "src"
    try:
        entries = sorted(source_root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError("cannot inspect source import boundary") from exc
    direct_files = [item.name for item in entries if item.is_file() or item.is_symlink()]
    directories = [item.name for item in entries if item.is_dir() and not item.is_symlink()]
    if direct_files:
        _fail("src contains a forbidden top-level file, symlink, or shadow module")
    if any(name != "imu_vive_learning" and not name.endswith(".egg-info") for name in directories):
        _fail("src contains an undeclared top-level import directory")
    package_files = {
        relative: closure[relative]
        for relative in (
            "src/imu_vive_learning/__init__.py",
            AUDITOR_MODULE_PATH,
        )
    }
    return {
        "direct_top_level_files": direct_files,
        "top_level_directories": directories,
        "imported_package_files": package_files,
    }


def _run_exact_source_tests(
    project: Path, audited_git_commit: str, source_set_sha256: str, closure: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the sole preregistered test command and bind its raw process report."""

    lock_raw = _read_nofollow(project, REQUIREMENTS_LOCK_PATH, name="requirements eval lock")
    lock_reference = _reference(REQUIREMENTS_LOCK_PATH, lock_raw)
    if lock_reference != closure[REQUIREMENTS_LOCK_PATH]:
        _fail("test dependency lock differs from the audited Git blob")
    bootstrap = _source_audit_bootstrap_evidence(project)
    dependencies_before = _dependency_evidence(project, lock_raw)
    site_packages = dependencies_before["site_packages"]
    import_boundary_before = _source_import_boundary(project, closure)
    launch = project / ".venv312/bin/python"
    real_executable = Path(os.path.realpath(launch))
    try:
        real_info = real_executable.lstat()
        real_raw = real_executable.read_bytes()
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError("cannot bind real Python executable") from exc
    if not stat.S_ISREG(real_info.st_mode) or stat.S_ISLNK(real_info.st_mode):
        _fail("real Python executable is not a nonsymlink regular file")
    environment = _exact_test_environment(site_packages)
    try:
        completed = subprocess.run(
            [str(launch), *EXACT_TEST_ARGV],
            cwd=project,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OpenAxesNativeV2ParentReceiptError("exact source-audit tests could not complete") from exc
    dependencies_after = _dependency_evidence(project, lock_raw)
    import_boundary_after = _source_import_boundary(project, closure)
    comparable_dependency_keys = {
        "pyvenv_cfg_reference",
        "pyvenv_cfg_fields",
        "pinned_distributions",
        "installed_distribution_inventory",
        "dependency_file_closure",
        "runner_module_files",
    }
    if any(
        dependencies_before[key] != dependencies_after[key]
        for key in comparable_dependency_keys
    ):
        _fail("venv dependency/file closure changed across exact source tests")
    if import_boundary_before != import_boundary_after:
        _fail("source import boundary changed across exact source tests")
    try:
        stdout = completed.stdout.decode("utf-8")
        stderr = completed.stderr.decode("utf-8")
    except UnicodeError as exc:
        raise OpenAxesNativeV2ParentReceiptError("exact test report is not UTF-8") from exc
    summary = stdout + "\n" + stderr
    counts = _parse_exact_pytest_outcome_report(summary)
    if completed.returncode != 0:
        _fail("exact source-audit test command did not yield exactly 12 passed and zero other outcomes")
    report_raw = canonical_json_bytes(
        {"exit_code": completed.returncode, "stderr_utf8": stderr, "stdout_utf8": stdout}
    )
    return {
        "python_launch_path_absolute": str(launch),
        "python_real_executable_absolute": str(real_executable),
        "python_real_executable_sha256": hashlib.sha256(real_raw).hexdigest(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        # With -S, CPython intentionally does not apply pyvenv.cfg and the
        # child test process reports base_prefix.  Bind that honest value while
        # separately binding the launch venv and its explicit site-packages.
        "sys_prefix_absolute": str(Path(sys.base_prefix)),
        "venv_site_packages_absolute": str(site_packages),
        "working_directory_absolute": str(project),
        "argv_after_executable": list(EXACT_TEST_ARGV),
        "environment": environment,
        "source_audit_bootstrap": bootstrap,
        "pyvenv_cfg_reference": dependencies_before["pyvenv_cfg_reference"],
        "pyvenv_cfg_fields": dependencies_before["pyvenv_cfg_fields"],
        "requirements_eval_lock_reference": lock_reference,
        "dependency_resolution": "version_pinned_without_wheel_artifact_hashes;linux_installed_file_closure_verified;visible_extras_inventoried",
        "pinned_distributions": dependencies_before["pinned_distributions"],
        "installed_distribution_inventory": dependencies_before[
            "installed_distribution_inventory"
        ],
        "dependency_file_closure_before_tests": dependencies_before[
            "dependency_file_closure"
        ],
        "dependency_file_closure_after_tests": dependencies_after[
            "dependency_file_closure"
        ],
        "dependency_file_closure_unchanged": True,
        "runner_module_files": dependencies_before["runner_module_files"],
        "source_import_boundary": import_boundary_after,
        "audited_git_commit": audited_git_commit,
        "source_set_sha256": source_set_sha256,
        "exit_code": completed.returncode,
        **counts,
        "stdout_utf8": stdout,
        "stdout_bytes": len(completed.stdout),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_utf8": stderr,
        "stderr_bytes": len(completed.stderr),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "report_bytes": len(report_raw),
        "report_sha256": hashlib.sha256(report_raw).hexdigest(),
    }


def _validate_test_receipt(value: Any) -> dict[str, Any]:
    receipt = _mapping(value, name="source-audit test receipt")
    _keys(receipt, SOURCE_TEST_KEYS, name="source-audit test receipt")
    launch_text = _exact_str(receipt["python_launch_path_absolute"], name="test Python launch path")
    launch = Path(launch_text)
    if (
        not launch.is_absolute()
        or os.path.normpath(launch_text) != launch_text
        or launch.name != "python"
        or launch.parent.name != "bin"
        or launch.parents[1].name != ".venv312"
    ):
        _fail("test Python launch path is not the exact absolute venv form")
    executable_text = _exact_str(
        receipt["python_real_executable_absolute"], name="test real Python executable"
    )
    executable = Path(executable_text)
    if not executable.is_absolute() or Path(os.path.realpath(launch)) != executable:
        _fail("test real Python executable does not resolve from the launch path")
    try:
        info = executable.lstat()
        raw = executable.read_bytes()
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError("cannot bind test real Python executable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _fail("test real Python executable is not a regular nonsymlink file")
    if hashlib.sha256(raw).hexdigest() != _hex64(
        receipt["python_real_executable_sha256"], name="test real Python executable SHA-256"
    ):
        _fail("test real Python executable bytes changed")
    _exact_str(receipt["python_version"], name="test Python version", expected="3.12.3")
    working_text = _exact_str(receipt["working_directory_absolute"], name="test working directory")
    working = Path(working_text)
    if (
        not working.is_absolute()
        or str(working.resolve()) != working_text
    ):
        _fail("test working directory is not canonical absolute")
    if receipt["argv_after_executable"] != list(EXACT_TEST_ARGV):
        _fail("source-audit test argv differs from preregistration")
    sys_prefix = Path(_exact_str(receipt["sys_prefix_absolute"], name="test sys.prefix"))
    site_packages = Path(
        _exact_str(receipt["venv_site_packages_absolute"], name="test venv site-packages")
    )
    if sys_prefix != executable.parents[1]:
        _fail("test -S sys.prefix differs from the real interpreter base prefix")
    if site_packages != launch.parents[1] / "lib/python3.12/site-packages":
        _fail("test site-packages binding differs from the launch venv")
    environment = _mapping(receipt["environment"], name="source-audit test environment")
    if dict(environment) != _exact_test_environment(site_packages):
        _fail("source-audit test environment differs from preregistration")
    bootstrap = _mapping(receipt["source_audit_bootstrap"], name="source-audit bootstrap")
    _keys(
        bootstrap,
        frozenset(
            {
                "safe_path",
                "no_site",
                "no_user_site",
                "dont_write_bytecode",
                "sys_executable_absolute",
                "sys_real_executable_absolute",
                "module_file_absolute",
                "sys_path_exact",
                "environment_exact",
            }
        ),
        name="source-audit bootstrap",
    )
    _exact_bool(bootstrap["safe_path"], name="bootstrap safe-path", expected=True)
    for key in ("no_site", "no_user_site", "dont_write_bytecode"):
        _exact_int(bootstrap[key], name=f"bootstrap {key}", expected=1)
    _exact_str(
        bootstrap["sys_executable_absolute"],
        name="bootstrap executable",
        expected=str(launch),
    )
    _exact_str(
        bootstrap["sys_real_executable_absolute"],
        name="bootstrap real executable",
        expected=str(executable),
    )
    _exact_str(
        bootstrap["module_file_absolute"],
        name="bootstrap module file",
        expected=str(working / AUDITOR_MODULE_PATH),
    )
    expected_sys_path = [
        str(working / "src"),
        str(site_packages),
        "/usr/lib/python312.zip",
        "/usr/lib/python3.12",
        "/usr/lib/python3.12/lib-dynload",
    ]
    if bootstrap["sys_path_exact"] != expected_sys_path or bootstrap[
        "environment_exact"
    ] != environment:
        _fail("source-audit bootstrap path/environment binding changed")
    _validate_reference_shape(
        receipt["pyvenv_cfg_reference"],
        name="pyvenv.cfg reference",
        expected_path=".venv312/pyvenv.cfg",
    )
    pyvenv_fields = _mapping(receipt["pyvenv_cfg_fields"], name="pyvenv.cfg fields")
    expected_pyvenv = {
        "home": "/usr/bin",
        "include-system-site-packages": "false",
        "version": "3.12.3",
        "executable": "/usr/bin/python3.12",
        "command": f"/usr/bin/python3.12 -m venv {working}/.venv312",
    }
    if dict(pyvenv_fields) != expected_pyvenv:
        _fail("pyvenv.cfg semantic fields changed")
    _validate_reference_shape(
        receipt["requirements_eval_lock_reference"],
        name="requirements-eval lock reference",
        expected_path=REQUIREMENTS_LOCK_PATH,
    )
    _exact_str(
        receipt["dependency_resolution"],
        name="dependency resolution",
        expected="version_pinned_without_wheel_artifact_hashes;linux_installed_file_closure_verified;visible_extras_inventoried",
    )
    pins = receipt["pinned_distributions"]
    if type(pins) is not list or len(pins) != len(REQUIREMENTS_EVAL_PINS):
        _fail("pinned distribution evidence is empty")
    pin_keys = frozenset(
        {
            "name",
            "normalized_name",
            "pinned_version",
            "installed_version",
            "dist_info_directory",
            "metadata_reference",
            "record_reference",
            "installed_file_count",
            "installed_total_bytes",
            "installed_file_set_sha256",
        }
    )
    for index, pin in enumerate(pins):
        item = _mapping(pin, name=f"pinned distribution {index}")
        _keys(item, pin_keys, name=f"pinned distribution {index}")
        if item["installed_version"] != item["pinned_version"]:
            _fail("installed dependency differs from the version pin")
        if _normalized_distribution_name(_exact_str(item["name"], name="distribution name")) != item["normalized_name"]:
            _fail("normalized dependency name changed")
        expected_name, expected_version = REQUIREMENTS_EVAL_PINS[index]
        if item["name"] != expected_name or item["pinned_version"] != expected_version:
            _fail("dependency receipt differs from the frozen lock pin/order")
        dist_info = _canonical_relative_path(
            item["dist_info_directory"], name="distribution dist-info directory"
        )
        expected_prefix = "lib/python3.12/site-packages/"
        if not dist_info.startswith(expected_prefix) or not dist_info.endswith(".dist-info"):
            _fail("distribution dist-info path escapes exact venv site-packages")
        for key, filename in (("metadata_reference", "METADATA"), ("record_reference", "RECORD")):
            _validate_reference_shape(
                item[key],
                name=f"distribution {key}",
                expected_path=f"{dist_info}/{filename}",
            )
        if _exact_int(item["installed_file_count"], name="installed file count") < 1:
            _fail("pinned distribution has an empty installed-file closure")
        if _exact_int(item["installed_total_bytes"], name="installed total bytes") < 1:
            _fail("pinned distribution has an empty installed-byte closure")
        _hex64(item["installed_file_set_sha256"], name="installed file-set hash")
    inventory = receipt["installed_distribution_inventory"]
    if type(inventory) is not list or not inventory:
        _fail("installed distribution inventory is empty")
    inventory_keys = frozenset({"name", "normalized_name", "version"})
    normalized_inventory: list[str] = []
    inventory_versions: dict[str, str] = {}
    for index, raw_item in enumerate(inventory):
        item = _mapping(raw_item, name=f"installed distribution inventory {index}")
        _keys(item, inventory_keys, name=f"installed distribution inventory {index}")
        normalized = _exact_str(item["normalized_name"], name="inventory normalized name")
        if normalized != _normalized_distribution_name(
            _exact_str(item["name"], name="inventory distribution name")
        ):
            _fail("installed distribution inventory normalization changed")
        version = _exact_str(item["version"], name="inventory distribution version")
        normalized_inventory.append(normalized)
        inventory_versions[normalized] = version
    if normalized_inventory != sorted(set(normalized_inventory)):
        _fail("installed distribution inventory is not unique canonical order")
    for name, version in REQUIREMENTS_EVAL_PINS:
        if inventory_versions.get(_normalized_distribution_name(name)) != version:
            _fail("installed inventory differs from a frozen version pin")
    closure_keys = frozenset({"file_count", "total_bytes", "manifest_sha256"})
    before_closure = _mapping(
        receipt["dependency_file_closure_before_tests"],
        name="dependency closure before tests",
    )
    after_closure = _mapping(
        receipt["dependency_file_closure_after_tests"],
        name="dependency closure after tests",
    )
    for label, closure in (("before", before_closure), ("after", after_closure)):
        _keys(closure, closure_keys, name=f"dependency closure {label}")
        if _exact_int(closure["file_count"], name=f"dependency {label} file count") < 1:
            _fail("dependency file closure is empty")
        if _exact_int(closure["total_bytes"], name=f"dependency {label} bytes") < 1:
            _fail("dependency byte closure is empty")
        _hex64(closure["manifest_sha256"], name=f"dependency {label} manifest hash")
    _exact_bool(
        receipt["dependency_file_closure_unchanged"],
        name="dependency closure unchanged",
        expected=True,
    )
    if before_closure != after_closure:
        _fail("dependency file closure changed across exact tests")
    runner_files = _mapping(receipt["runner_module_files"], name="runner module files")
    if set(runner_files) != set(PINNED_TEST_DISTRIBUTIONS):
        _fail("runner module file set changed")
    binding_keys = frozenset({"path_absolute", "bytes", "sha256"})
    for name in PINNED_TEST_DISTRIBUTIONS:
        binding = _mapping(runner_files[name], name=f"{name} runner binding")
        _keys(binding, binding_keys, name=f"{name} runner binding")
        path = Path(_exact_str(binding["path_absolute"], name=f"{name} runner path"))
        if os.path.commonpath((str(site_packages), str(path))) != str(site_packages):
            _fail("runner module path escapes exact venv site-packages")
        _exact_int(binding["bytes"], name=f"{name} runner bytes")
        _hex64(binding["sha256"], name=f"{name} runner hash")
    boundary = _mapping(receipt["source_import_boundary"], name="source import boundary")
    _keys(
        boundary,
        frozenset({"direct_top_level_files", "top_level_directories", "imported_package_files"}),
        name="source import boundary",
    )
    directories = boundary["top_level_directories"]
    if (
        boundary["direct_top_level_files"] != []
        or type(directories) is not list
        or directories != sorted(set(directories))
        or "imu_vive_learning" not in directories
        or any(
            type(item) is not str
            or (item != "imu_vive_learning" and not item.endswith(".egg-info"))
            for item in directories
        )
    ):
        _fail("source import boundary admits top-level shadow files")
    imported_files = _mapping(boundary["imported_package_files"], name="imported package files")
    if set(imported_files) != {"src/imu_vive_learning/__init__.py", AUDITOR_MODULE_PATH}:
        _fail("source package import closure changed")
    for path, binding in imported_files.items():
        _validate_reference_shape(binding, name=f"imported package file {path}", expected_path=path)
    _git_sha1(receipt["audited_git_commit"], name="test audited Git commit")
    _hex64(receipt["source_set_sha256"], name="test source-set hash")
    _exact_int(receipt["exit_code"], name="test exit code", expected=0)
    stored_counts: dict[str, int] = {}
    passed = _exact_int(receipt["passed"], name="test passed count")
    stored_counts["passed"] = passed
    if passed != TEST_PASS_COUNT:
        _fail("source-audit test suite did not pass exactly 12 tests")
    for key in ("failed", "skipped", "xfailed"):
        stored_counts[key] = _exact_int(
            receipt[key], name=f"test {key} count", expected=0
        )
    stdout = receipt["stdout_utf8"]
    stderr = receipt["stderr_utf8"]
    if type(stdout) is not str or type(stderr) is not str:
        _fail("test stdout/stderr are not exact UTF-8 strings")
    combined_report = stdout + "\n" + stderr
    derived_counts = _parse_exact_pytest_outcome_report(combined_report)
    if stored_counts != derived_counts:
        _fail("stored test outcomes differ from raw stdout/stderr-derived outcomes")
    for label, text in (("stdout", stdout), ("stderr", stderr)):
        raw_text = text.encode("utf-8")
        _exact_int(receipt[f"{label}_bytes"], name=f"test {label} bytes", expected=len(raw_text))
        if _hex64(receipt[f"{label}_sha256"], name=f"test {label} SHA-256") != hashlib.sha256(raw_text).hexdigest():
            _fail(f"test {label} raw binding changed")
    report_raw = canonical_json_bytes({"exit_code": 0, "stderr_utf8": stderr, "stdout_utf8": stdout})
    _exact_int(receipt["report_bytes"], name="test report bytes", expected=len(report_raw))
    if _hex64(receipt["report_sha256"], name="test report SHA-256") != hashlib.sha256(report_raw).hexdigest():
        _fail("test report raw binding changed")
    return dict(receipt)


def _expected_source_audit_erratum() -> dict[str, Any]:
    return {
        "erratum_id": SOURCE_AUDIT_ERRATUM_ID,
        "active_receipt_path": ACTIVE_SOURCE_AUDIT_RECEIPT_PATH,
        "historical_source_audit_authorities": [
            {
                "authority_id": "original",
                "receipt_path": ORIGINAL_SOURCE_AUDIT_RECEIPT_PATH,
                "audited_git_commit": ORIGINAL_AUDITED_SOURCE_COMMIT,
                "evidence_git_commit": ORIGINAL_SOURCE_AUDIT_EVIDENCE_COMMIT,
            },
            {
                "authority_id": "erratum1",
                "receipt_path": ERRATUM1_SOURCE_AUDIT_RECEIPT_PATH,
                "audited_git_commit": ERRATUM1_AUDITED_SOURCE_COMMIT,
                "evidence_git_commit": ERRATUM1_SOURCE_AUDIT_EVIDENCE_COMMIT,
            },
        ],
        "sealed_001_evidence": {
            "run_id": FAILED_V1_RUN_001_ID,
            "supervisor_terminal_relative_path": (
                FAILED_001_SUPERVISOR_TERMINAL_RELATIVE_PATH
            ),
            "supervisor_terminal_sha256": FAILED_001_SUPERVISOR_TERMINAL_SHA256,
            "public_preparation_manifest_sha256": (
                "dcffda6bb3e8c0c3af52ec2b7a2a98e9eaee5a788352de32a38b915d82153a17"
            ),
            "public_preparation_parquet_sha256": (
                "98747e51fd95506b91facfeb273be23c454d07da563deea13a6f2192ded48703"
            ),
            "public_preparation_complete_sha256": (
                "1a9f4cc4e5a3528e0a986f27aebc060fb6fee749458b43ad4154250623968b12"
            ),
            "public_validator_stdout_sha256": (
                "4d3f793511dd55a6df0fe5ba7404ca71e487aafcbfcfb7d19b54f2b4b5e3ba94"
            ),
            "persisted_sensor_groups_mapping_sha256": (
                FAILED_001_PERSISTED_SENSOR_GROUPS_MAPPING_SHA256
            ),
            "failure_classification": (
                "outcome_free_canonical_json_key_order_consumer_bug"
            ),
            "stage_counts": {
                key: 0 for key in sorted(SEALED_001_STAGE_COUNT_KEYS)
            },
        },
        "sealed_002_evidence": {
            "run_id": FAILED_V1_RUN_002_ID,
            "audited_git_head": ERRATUM1_SOURCE_AUDIT_EVIDENCE_COMMIT,
            "supervisor_run_spec_relative_path": (
                "openaxes-v1-runs/openaxes-v1-sealed-002/RUN_SPEC.json"
            ),
            "supervisor_run_spec_sha256": FAILED_002_RUN_SPEC_SHA256,
            "supervisor_run_spec_payload_sha256": (
                FAILED_002_RUN_SPEC_PAYLOAD_SHA256
            ),
            "cleanup_blocked_current_relative_path": (
                "openaxes-v1-runs/openaxes-v1-sealed-002/CURRENT.json"
            ),
            "cleanup_blocked_current_sha256": FAILED_002_CLEANUP_BLOCKED_SHA256,
            "cleanup_blocked_marker_relative_path": (
                "openaxes-v1-runs/openaxes-v1-sealed-002/attempts/attempt_0001/"
                "CLEANUP_BLOCKED.json"
            ),
            "cleanup_blocked_marker_sha256": FAILED_002_CLEANUP_BLOCKED_SHA256,
            "supervisor_log_relative_path": (
                "openaxes-v1-runs/openaxes-v1-sealed-002/attempts/attempt_0001/"
                "supervisor.log.jsonl"
            ),
            "supervisor_log_sha256": FAILED_002_SUPERVISOR_LOG_SHA256,
            "launch_log_relative_path": "openaxes-v1-sealed-002.launch.log",
            "launch_log_sha256": FAILED_002_LAUNCH_LOG_SHA256,
            "public_preparation_manifest_sha256": (
                FAILED_002_PREPARATION_MANIFEST_SHA256
            ),
            "public_preparation_parquet_sha256": (
                FAILED_002_PREPARATION_PARQUET_SHA256
            ),
            "public_preparation_complete_sha256": (
                FAILED_002_PREPARATION_COMPLETE_SHA256
            ),
            "public_prepare_stdout_sha256": FAILED_002_PUBLIC_PREPARE_STDOUT_SHA256,
            "public_validator_stdout_sha256": (
                FAILED_002_PUBLIC_VALIDATE_STDOUT_SHA256
            ),
            "public_preparation_valid_despite_supervisor_progress_flag": True,
            "model_prepared_sha256": FAILED_002_MODEL_PREPARED_SHA256,
            "model_prepared_manifest_sha256": (
                FAILED_002_MODEL_PREPARED_MANIFEST_SHA256
            ),
            "task_index_sha256": FAILED_002_TASK_INDEX_SHA256,
            "execution_session_sha256": FAILED_002_EXECUTION_SESSION_SHA256,
            "pre_fit_validation_sha256": FAILED_002_PRE_FIT_VALIDATION_SHA256,
            "launch_started_sha256": FAILED_002_LAUNCH_STARTED_SHA256,
            "failure_status": "cleanup_blocked_nonterminal",
            "failure_message": (
                "GPU slot 6 failed: OpenAxesExternalModelV1Error: "
                "visible CUDA UUID changed"
            ),
            "failure_stage": "worker_identity_bootstrap_before_worker_ready_before_fit",
            "failed_terminal_written": False,
            "automatic_deletion_or_refit_performed": False,
            "original_evidence_preserved": True,
            "marked_processes_after_verified_natural_exit": 0,
            "manual_signal_performed": False,
            "cleanup_refused_survivor_role": "multiprocessing.resource_tracker",
            "cleanup_refusal_classification": "supervisor_argv_allowlist_reporting_bug",
            "cleanup_survivor_self_exited": True,
            "hardware_mapping_unchanged": True,
            "resume_scientifically_forbidden": True,
            "fresh_run_id_required": SUCCESSFUL_V1_RUN_ID,
        },
        "terminal_bound_successful_run_id": SUCCESSFUL_V1_RUN_ID,
        "terminal_bound_successful_v1_run_project_relative_path": (
            SUCCESSFUL_V1_RUN_PROJECT_RELATIVE_PATH
        ),
        "terminal_bound_successful_preparation_run_project_relative_path": (
            SUCCESSFUL_PREPARATION_RUN_PROJECT_RELATIVE_PATH
        ),
        "terminal_bound_successful_supervisor_relative_path": (
            SUCCESSFUL_SUPERVISOR_RELATIVE_PATH
        ),
        "failure_classification": ERRATUM_FAILURE_CLASSIFICATION,
        "torch_uuid_observation": {
            "assigned_nvidia_uuid": "GPU-5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb",
            "pytorch_version": "2.12.1+cu132",
            "pytorch_property_type_name": "_CUuuid",
            "pytorch_property_string": "5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb",
            "literal_gpu_prefix_is_only_representation_difference": True,
        },
        "sealed_002_stage_counts": {
            "public_preparation": 1,
            "model_prepared": 1,
            "execution_session": 1,
            "launch_started": 1,
            "worker_ready": 0,
            "task_ledger": 0,
            "attempt_ledger": 0,
            "attempt_roster": 0,
            "task_result_directory": 0,
            "quarantine_member": 0,
            "scientific_fit": 0,
            "prediction": 0,
            "evaluation": 0,
            "result": 0,
        },
        "no_v1_model_result_or_metric_artifact_existed_or_was_accessed": True,
        "scientific_projection_unchanged": True,
    }


def _validate_source_audit_erratum(value: Any) -> dict[str, Any]:
    erratum = _mapping(value, name="source-audit erratum")
    _keys(erratum, SOURCE_AUDIT_ERRATUM_KEYS, name="source-audit erratum")
    authorities_value = erratum["historical_source_audit_authorities"]
    if type(authorities_value) is not list:
        _fail("source-audit historical authorities is not an exact list")
    authorities = authorities_value
    if len(authorities) != 2:
        _fail("source-audit erratum must retain exactly two historical authorities")
    for index, authority_value in enumerate(authorities):
        authority = _mapping(
            authority_value, name=f"source-audit historical authority {index}"
        )
        _keys(
            authority,
            HISTORICAL_SOURCE_AUDIT_AUTHORITY_KEYS,
            name=f"source-audit historical authority {index}",
        )
        _git_sha1(
            authority["audited_git_commit"],
            name=f"source-audit historical authority {index} audited commit",
        )
        _git_sha1(
            authority["evidence_git_commit"],
            name=f"source-audit historical authority {index} evidence commit",
        )
    sealed_001 = _mapping(erratum["sealed_001_evidence"], name="sealed-001 evidence")
    _keys(sealed_001, SEALED_001_EVIDENCE_KEYS, name="sealed-001 evidence")
    sealed_001_counts = _mapping(
        sealed_001["stage_counts"], name="sealed-001 stage counts"
    )
    _keys(
        sealed_001_counts,
        SEALED_001_STAGE_COUNT_KEYS,
        name="sealed-001 stage counts",
    )
    for key in SEALED_001_STAGE_COUNT_KEYS:
        _exact_int(sealed_001_counts[key], name=f"sealed-001 {key} count", expected=0)
    sealed_002 = _mapping(erratum["sealed_002_evidence"], name="sealed-002 evidence")
    _keys(sealed_002, SEALED_002_EVIDENCE_KEYS, name="sealed-002 evidence")
    for key, item in sealed_002.items():
        if key.endswith("_sha256"):
            _hex64(item, name=f"sealed-002 {key}")
    counts = _mapping(erratum["sealed_002_stage_counts"], name="sealed-002 stage counts")
    _keys(counts, SEALED_002_STAGE_COUNT_KEYS, name="sealed-002 stage counts")
    for key in SEALED_002_STAGE_COUNT_KEYS:
        expected_count = 1 if key in {
            "public_preparation", "model_prepared", "execution_session", "launch_started"
        } else 0
        _exact_int(counts[key], name=f"sealed-002 {key} count", expected=expected_count)
    torch_uuid = _mapping(erratum["torch_uuid_observation"], name="torch UUID observation")
    _keys(torch_uuid, TORCH_UUID_OBSERVATION_KEYS, name="torch UUID observation")
    for key in (
        "no_v1_model_result_or_metric_artifact_existed_or_was_accessed",
        "scientific_projection_unchanged",
    ):
        _exact_bool(erratum[key], name=f"source-audit erratum {key}", expected=True)
    expected = _expected_source_audit_erratum()
    if dict(erratum) != expected:
        _fail("source-audit erratum differs from the fixed amendment authority")
    return dict(erratum)


def validate_source_audit_receipt(value: Any) -> dict[str, Any]:
    receipt = _mapping(value, name="parent-auditor source-audit receipt")
    _keys(receipt, SOURCE_AUDIT_KEYS, name="parent-auditor source-audit receipt")
    _exact_str(receipt["contract"], name="source-audit contract", expected=SOURCE_AUDIT_CONTRACT)
    _exact_int(receipt["schema_version"], name="source-audit schema version", expected=1)
    _timestamp(receipt["created_at_utc"])
    audited_commit = _git_sha1(
        receipt["audited_git_commit"], name="audited source Git commit"
    )
    if audited_commit in HISTORICAL_AUDITED_SOURCE_COMMITS:
        _fail("historical source-audit commit cannot be active erratum2 authority")
    closure = _mapping(receipt["source_closure"], name="source-audit closure")
    if set(closure) != set(SOURCE_CLOSURE) or len(closure) != 8:
        _fail("source-audit closure is not the exact eight-file set")
    normalized: dict[str, Any] = {}
    for path in SOURCE_CLOSURE:
        normalized[path] = _validate_reference_shape(
            closure[path], name=f"source-audit closure {path}", expected_path=path
        )
    if canonical_sha256(normalized) != _hex64(
        receipt["source_set_sha256"], name="source-audit source-set SHA-256"
    ):
        _fail("source-audit source-set digest mismatch")
    _validate_source_audit_erratum(receipt["erratum"])
    validated_test = _validate_test_receipt(receipt["test_receipt"])
    if validated_test["audited_git_commit"] != receipt["audited_git_commit"]:
        _fail("test report is not bound to the audited Git commit")
    if validated_test["source_set_sha256"] != receipt["source_set_sha256"]:
        _fail("test report is not bound to the audited source set")
    if validated_test["requirements_eval_lock_reference"] != normalized[REQUIREMENTS_LOCK_PATH]:
        _fail("test report dependency lock differs from audited source closure")
    imported = validated_test["source_import_boundary"]["imported_package_files"]
    for path in ("src/imu_vive_learning/__init__.py", AUDITOR_MODULE_PATH):
        if imported[path] != normalized[path]:
            _fail("test report imported package file differs from audited source closure")
    assertions = _mapping(receipt["assertions"], name="source-audit assertions")
    _keys(assertions, SOURCE_ASSERTION_KEYS, name="source-audit assertions")
    for key in SOURCE_ASSERTION_KEYS:
        _exact_bool(assertions[key], name=f"source-audit assertion {key}", expected=True)
    _payload_hash(receipt, key="payload_sha256", name="source-audit receipt")
    return dict(receipt)


def _source_closure_from_git(repo: Path, commit: str) -> dict[str, Any]:
    closure: dict[str, Any] = {}
    for relative in SOURCE_CLOSURE:
        _oid, reference, raw = _git_blob(repo, commit, relative, name=f"auditor source {relative}")
        if relative == AUDITOR_MODULE_PATH:
            audit_python_source_safety(raw)
        closure[relative] = reference
    return closure


def _verify_worktree_source_closure(
    project_root: Path, closure: Mapping[str, Any]
) -> None:
    if set(closure) != set(SOURCE_CLOSURE):
        _fail("worktree verification did not receive the exact source closure")
    for relative in SOURCE_CLOSURE:
        raw = _read_nofollow(
            project_root, relative, name=f"audited worktree source {relative}"
        )
        if _reference(relative, raw) != closure[relative]:
            _fail(f"audited worktree source differs from Git blob: {relative}")


def build_source_audit_receipt(
    workspace_root: str | os.PathLike[str],
    audited_git_commit: str,
) -> dict[str, Any]:
    """Build, but do not write, the pre-v1 source-audit receipt."""

    workspace = _absolute_root(workspace_root, name="workspace_root")
    repo = _repository(workspace)
    commit = _verify_commit(repo, audited_git_commit, name="audited source commit")
    if commit in HISTORICAL_AUDITED_SOURCE_COMMITS:
        _fail("historical source commit cannot issue the active erratum2 receipt")
    closure = _source_closure_from_git(repo, commit)
    project = workspace / PROJECT_DIRECTORY
    _verify_worktree_source_closure(project, closure)
    source_set_sha256 = canonical_sha256(closure)
    validated_test = _validate_test_receipt(
        _run_exact_source_tests(project, commit, source_set_sha256, closure)
    )
    if validated_test["working_directory_absolute"] != str(project):
        _fail("direct test working directory differs from the audited project root")
    _verify_worktree_source_closure(project, closure)
    if _source_closure_from_git(repo, commit) != closure:
        _fail("audited Git source closure changed across direct test execution")
    receipt: dict[str, Any] = {
        "contract": SOURCE_AUDIT_CONTRACT,
        "schema_version": 1,
        "created_at_utc": _timestamp(_now_utc()),
        "audited_git_commit": commit,
        "source_closure": closure,
        "source_set_sha256": source_set_sha256,
        "erratum": _expected_source_audit_erratum(),
        "test_receipt": validated_test,
        "assertions": {
            "source_written_without_v1_result_or_metric_access": True,
            "exact_eight_file_closure": True,
            "all_mandatory_tests_passed": True,
            "no_forbidden_import_or_action": True,
            "no_challenge_access": True,
        },
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    return validate_source_audit_receipt(receipt)


def _verify_source_audit_git_evidence(
    repo: Path,
    raw: bytes,
    receipt: Mapping[str, Any],
    evidence_commit: str,
) -> tuple[str, str]:
    audited_commit = _git_sha1(receipt["audited_git_commit"], name="audited source commit")
    if audited_commit in HISTORICAL_AUDITED_SOURCE_COMMITS:
        _fail("historical audited source cannot authorize a successful terminal lineage")
    if receipt["test_receipt"]["working_directory_absolute"] != str(
        repo / PROJECT_DIRECTORY
    ):
        _fail("source-audit test cwd differs from the evidence repository project root")
    observed = _source_closure_from_git(repo, audited_commit)
    if observed != receipt["source_closure"] or canonical_sha256(observed) != receipt["source_set_sha256"]:
        _fail("copied source audit does not equal exact Git eight-file closure")
    evidence = _verify_commit(repo, evidence_commit, name="source-audit evidence commit")
    if evidence in HISTORICAL_SOURCE_AUDIT_EVIDENCE_COMMITS:
        _fail("historical source-audit evidence cannot authorize a successful terminal lineage")
    if evidence == audited_commit:
        _fail("source-audit evidence commit must follow the audited source commit")
    _git_ancestor(repo, audited_commit, evidence, name="audited-source/evidence ancestry")
    oid, _reference_value, blob = _git_blob(
        repo,
        evidence,
        ACTIVE_SOURCE_AUDIT_RECEIPT_PATH,
        name="source-audit evidence",
    )
    if blob != raw:
        _fail("source-audit raw bytes differ from committed Git evidence blob")
    _verify_live_source_audit_environment(
        repo / PROJECT_DIRECTORY, receipt["test_receipt"]
    )
    return evidence, oid


def _verify_live_source_audit_environment(
    project: Path, test_receipt: Mapping[str, Any]
) -> None:
    """Recompute the persisted venv/installed-file closure at parent validation."""

    lock_raw = _read_nofollow(project, REQUIREMENTS_LOCK_PATH, name="live requirements lock")
    if _reference(REQUIREMENTS_LOCK_PATH, lock_raw) != test_receipt[
        "requirements_eval_lock_reference"
    ]:
        _fail("live requirements lock differs from source-audit receipt")
    evidence = _dependency_evidence(project, lock_raw)
    expected = {
        "pyvenv_cfg_reference": evidence["pyvenv_cfg_reference"],
        "pyvenv_cfg_fields": evidence["pyvenv_cfg_fields"],
        "pinned_distributions": evidence["pinned_distributions"],
        "installed_distribution_inventory": evidence[
            "installed_distribution_inventory"
        ],
        "dependency_file_closure_before_tests": evidence[
            "dependency_file_closure"
        ],
        "dependency_file_closure_after_tests": evidence[
            "dependency_file_closure"
        ],
        "runner_module_files": evidence["runner_module_files"],
    }
    for key, expected_value in expected.items():
        if test_receipt[key] != expected_value:
            _fail(f"live source-audit environment changed: {key}")


def _verify_v1_sources_at_model_head(repo: Path, model_head: str) -> dict[str, str]:
    """Cross-check the frozen nine-file v1 oracle against exact Git blobs."""

    head = _verify_commit(repo, model_head, name="v1 model source head")
    observed_oids: dict[str, str] = {}
    for relative in V1_SOURCE_CLOSURE:
        oid, reference, _raw = _git_blob(
            repo, head, relative, name=f"v1 model source {relative}"
        )
        if reference != V1_SOURCE_REFERENCE_ORACLE[relative]:
            _fail(f"v1 model source Git bytes differ from frozen oracle: {relative}")
        if oid != V1_SOURCE_BLOB_OID_ORACLE[relative]:
            _fail(f"v1 model source Git blob OID differs from frozen oracle: {relative}")
        observed_oids[relative] = oid
    if observed_oids != V1_SOURCE_BLOB_OID_ORACLE:
        _fail("v1 model source Git blob OID map changed")
    return observed_oids


def _fixed_sum(values: Sequence[float], *, name: str) -> float:
    if not values:
        _fail(f"{name} cannot reduce an empty vector")
    level = [_finite_nonnegative(value, name=f"{name} input") for value in values]
    while len(level) > 1:
        next_level: list[float] = []
        index = 0
        while index + 1 < len(level):
            value = level[index] + level[index + 1]
            if not math.isfinite(value):
                _fail(f"{name} overflowed")
            next_level.append(value)
            index += 2
        if index < len(level):
            next_level.append(level[index])
        level = next_level
    return level[0]


def _fixed_mean(values: Sequence[float], *, name: str) -> float:
    return _fixed_sum(values, name=name) / len(values)


def _robust_inputs(rows: Sequence[Mapping[str, Any]], fold_order: Sequence[str]) -> tuple[float, float, float, float, list[float]]:
    lookup = {(row["fold_id"], row["view_id"]): row["rmse3d_m"] for row in rows}
    if len(lookup) != 32:
        _fail("robust input contexts are duplicated or incomplete")
    fold_scores = [
        _fixed_mean(
            [lookup[(fold_id, view_id)] for view_id in OUTAGE_VIEW_ORDER],
            name=f"fold outage score {fold_id}",
        )
        for fold_id in fold_order
    ]
    all_values = [lookup[(fold_id, view_id)] for fold_id in fold_order for view_id in OUTAGE_VIEW_ORDER]
    mask_scores = [
        _fixed_mean(
            [lookup[(fold_id, view_id)] for fold_id in fold_order],
            name=f"mask score {view_id}",
        )
        for view_id in OUTAGE_VIEW_ORDER
    ]
    a_value = _fixed_mean(all_values, name="A mean contexts")
    ws_value = max(fold_scores)
    wm_value = max(mask_scores)
    robust = _fixed_mean([a_value, ws_value, wm_value], name="robust J")
    return a_value, ws_value, wm_value, robust, fold_scores


def _validate_context_row(value: Any, *, name: str, per_seed: bool = False) -> dict[str, Any]:
    row = _mapping(value, name=name)
    _keys(row, PER_SEED_CONTEXT_KEYS if per_seed else CONTEXT_KEYS, name=name)
    result: dict[str, Any] = {}
    if per_seed:
        seed = _exact_int(row["seed"], name=f"{name}.seed")
        if seed not in SEED_ORDER:
            _fail(f"{name}.seed is outside the frozen seed order")
        result["seed"] = seed
    result.update(
        {
            "fold_id": _exact_str(row["fold_id"], name=f"{name}.fold_id"),
            "scenario_id": _exact_str(row["scenario_id"], name=f"{name}.scenario_id"),
            "view_id": _exact_str(row["view_id"], name=f"{name}.view_id"),
            "rows": _exact_int(row["rows"], name=f"{name}.rows"),
            "identity_sha256": _hex64(row["identity_sha256"], name=f"{name}.identity_sha256"),
            "rmse3d_m": _finite_nonnegative(row["rmse3d_m"], name=f"{name}.rmse3d_m"),
        }
    )
    if result["rows"] < 1:
        _fail(f"{name}.rows must be positive")
    return result


def validate_comparator_bundle(value: Any) -> dict[str, Any]:
    bundle = _mapping(value, name="masked_erm comparator bundle")
    _keys(bundle, BUNDLE_KEYS, name="masked_erm comparator bundle")
    _exact_str(bundle["contract"], name="bundle contract", expected=BUNDLE_CONTRACT)
    _exact_int(bundle["schema_version"], name="bundle schema version", expected=1)
    _exact_str(bundle["method_id"], name="bundle method", expected="masked_erm")
    if bundle["seed_order"] != list(SEED_ORDER) or bundle["view_order"] != list(VIEW_ORDER):
        _fail("bundle seed or view order changed")
    fold_order = bundle["primary_fold_order"]
    if type(fold_order) is not list or len(fold_order) != 8 or any(type(item) is not str or not item for item in fold_order) or len(set(fold_order)) != 8:
        _fail("bundle primary fold order is not eight unique strings")
    if any(not fold_id.startswith("scenario_loso::") for fold_id in fold_order):
        _fail("bundle primary fold IDs are not scenario_loso identities")
    initial = _mapping(bundle["initial_state_by_seed"], name="initial states")
    if set(initial) != {str(seed) for seed in SEED_ORDER}:
        _fail("bundle initial-state seed keys changed")
    for seed in SEED_ORDER:
        _hex64(initial[str(seed)], name=f"initial state seed {seed}")

    identity = _mapping(bundle["primary_context_identity"], name="primary context identity")
    _keys(identity, CONTEXT_IDENTITY_KEYS, name="primary context identity")
    _exact_int(identity["context_count"], name="primary context count", expected=32)
    identity_rows = identity["ordered_contexts"]
    if type(identity_rows) is not list or len(identity_rows) != 32:
        _fail("primary identity must contain exactly 32 rows")
    identity_keys = frozenset({"fold_id", "scenario_id", "view_id", "rows", "identity_sha256"})
    normalized_identity: list[dict[str, Any]] = []
    for index, raw in enumerate(identity_rows):
        row = _mapping(raw, name=f"primary identity row {index}")
        _keys(row, identity_keys, name=f"primary identity row {index}")
        normalized_identity.append(
            {
                "fold_id": _exact_str(row["fold_id"], name="identity fold"),
                "scenario_id": _exact_str(row["scenario_id"], name="identity scenario"),
                "view_id": _exact_str(row["view_id"], name="identity view"),
                "rows": _exact_int(row["rows"], name="identity rows"),
                "identity_sha256": _hex64(row["identity_sha256"], name="identity hash"),
            }
        )
    expected_pairs = [(fold, view) for fold in fold_order for view in OUTAGE_VIEW_ORDER]
    if [(row["fold_id"], row["view_id"]) for row in normalized_identity] != expected_pairs:
        _fail("primary identity order differs from fold then outage-view order")
    if any(row["rows"] < 1 for row in normalized_identity):
        _fail("primary identity contains a nonpositive row count")
    scenario_by_fold: dict[str, str] = {}
    for row in normalized_identity:
        expected_scenario = row["fold_id"].removeprefix("scenario_loso::")
        if row["scenario_id"] != expected_scenario:
            _fail("primary context scenario differs from its LOSO fold identity")
        scenario_by_fold[row["fold_id"]] = row["scenario_id"]
    if len(set(scenario_by_fold.values())) != 8:
        _fail("primary contexts do not contain eight distinct held-out scenarios")
    if canonical_sha256(normalized_identity) != _hex64(identity["identity_sha256"], name="primary context identity hash"):
        _fail("primary context identity digest mismatch")

    ensemble_outage_raw = bundle["primary_seed_ensemble_outage_contexts"]
    ensemble_full_raw = bundle["primary_seed_ensemble_full_contexts"]
    per_seed_raw = bundle["primary_per_seed_outage_contexts"]
    if type(ensemble_outage_raw) is not list or len(ensemble_outage_raw) != 32:
        _fail("bundle does not contain exactly 32 ensemble outage contexts")
    if type(ensemble_full_raw) is not list or len(ensemble_full_raw) != 8:
        _fail("bundle does not contain exactly eight ensemble full contexts")
    if type(per_seed_raw) is not list or len(per_seed_raw) != 96:
        _fail("bundle does not contain exactly 96 per-seed outage contexts")
    ensemble_outage = [
        _validate_context_row(item, name=f"ensemble outage {index}")
        for index, item in enumerate(ensemble_outage_raw)
    ]
    ensemble_full = [
        _validate_context_row(item, name=f"ensemble full {index}")
        for index, item in enumerate(ensemble_full_raw)
    ]
    per_seed = [
        _validate_context_row(item, name=f"per-seed outage {index}", per_seed=True)
        for index, item in enumerate(per_seed_raw)
    ]
    if [(row["fold_id"], row["view_id"]) for row in ensemble_outage] != expected_pairs:
        _fail("ensemble outage order changed")
    if [(row["fold_id"], row["view_id"]) for row in ensemble_full] != [(fold, "full") for fold in fold_order]:
        _fail("ensemble full order changed")
    if any(
        row["scenario_id"] != scenario_by_fold[row["fold_id"]]
        for row in ensemble_full
    ):
        _fail("ensemble full scenario differs from its LOSO fold identity")
    expected_seed_pairs = [
        (seed, fold, view)
        for seed in SEED_ORDER
        for fold in fold_order
        for view in OUTAGE_VIEW_ORDER
    ]
    if [(row["seed"], row["fold_id"], row["view_id"]) for row in per_seed] != expected_seed_pairs:
        _fail("per-seed outage order changed")
    identity_lookup = {(row["fold_id"], row["view_id"]): row for row in normalized_identity}
    for row in ensemble_outage:
        base = identity_lookup[(row["fold_id"], row["view_id"])]
        if any(row[key] != base[key] for key in ("scenario_id", "rows", "identity_sha256")):
            _fail("ensemble outage identity differs from closed identity")
    for row in per_seed:
        base = identity_lookup[(row["fold_id"], row["view_id"])]
        if any(row[key] != base[key] for key in ("scenario_id", "rows", "identity_sha256")):
            _fail("per-seed outage identity differs from closed identity")

    derived = _mapping(bundle["derived_gate_inputs"], name="derived gate inputs")
    _keys(derived, DERIVED_KEYS, name="derived gate inputs")
    a_value, ws_value, wm_value, robust, fold_scores = _robust_inputs(ensemble_outage, fold_order)
    expected_per_seed: list[float] = []
    for seed in SEED_ORDER:
        seed_rows = [row for row in per_seed if row["seed"] == seed]
        expected_per_seed.append(_robust_inputs(seed_rows, fold_order)[3])
    full_values = [row["rmse3d_m"] for row in ensemble_full]
    full_f = 0.5 * _fixed_mean(full_values, name="full mean") + 0.5 * max(full_values)
    exact_scalars = {
        "A_mean_context_rmse3d_m": a_value,
        "WS_worst_scenario_mean_rmse3d_m": ws_value,
        "WM_worst_mask_mean_rmse3d_m": wm_value,
        "robust_J_m": robust,
        "full_F_m": full_f,
    }
    for key, expected in exact_scalars.items():
        if _finite_nonnegative(derived[key], name=f"derived {key}") != expected:
            _fail(f"derived {key} does not equal the fixed reduction")
    fold_values = derived["fold_outage_scores_m"]
    seed_values = derived["per_seed_J_m"]
    if (
        type(fold_values) is not list
        or len(fold_values) != 8
        or type(seed_values) is not list
        or len(seed_values) != 3
    ):
        _fail("derived fold or per-seed values have the wrong closed cardinality")
    for index, value in enumerate(fold_values):
        _finite_nonnegative(value, name=f"derived fold score {index}")
    for index, value in enumerate(seed_values):
        _finite_nonnegative(value, name=f"derived per-seed J {index}")
    if fold_values != fold_scores or seed_values != expected_per_seed:
        _fail("derived fold or per-seed values do not equal fixed reductions")

    sources = _mapping(bundle["source_references"], name="bundle source references")
    _keys(sources, SOURCE_REFERENCE_KEYS, name="bundle source references")
    _validate_reference_shape(sources["v1_development_manifest"], name="bundle development reference", expected_path="development_manifest.json")
    for key, expected_count, suffix in (
        ("masked_erm_primary_training_receipts", 24, "/training_receipt.json"),
        ("masked_erm_primary_outer_metric_receipts", 24, "/outer_evaluation.json"),
    ):
        references = sources[key]
        if type(references) is not list or len(references) != expected_count:
            _fail(f"bundle {key} must contain exactly {expected_count} references")
        paths: list[str] = []
        for index, reference in enumerate(references):
            normalized = _validate_reference_shape(reference, name=f"bundle {key}[{index}]")
            if not normalized["path"].startswith("results/task_") or not normalized["path"].endswith(suffix):
                _fail(f"bundle {key} contains a forbidden artifact path")
            lowered = normalized["path"].lower()
            if any(token in lowered for token in ("prediction", "evaluation_rows", "truth", "stress", "challenge")):
                _fail(f"bundle {key} contains forbidden outcome or truth reference")
            paths.append(normalized["path"])
        if len(set(paths)) != expected_count:
            _fail(f"bundle {key} contains duplicate references")
    _payload_hash(bundle, key="payload_sha256", name="comparator bundle")
    return dict(bundle)


def _expected_tasks(preparation_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    folds = _mapping(preparation_manifest.get("folds"), name="preparation folds")
    items = folds.get("items")
    if type(items) is not list or len(items) != 14:
        _fail("preparation manifest does not contain the exact 14 folds")
    family_roles = {"scenario_loso": "primary", "speed_pair_out": "stress", "shape_out": "stress"}
    expected_scenarios = [
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    ]
    expected_folds: list[dict[str, Any]] = []
    for scenario in expected_scenarios:
        expected_folds.append(
            {
                "fold_id": f"scenario_loso::{scenario}",
                "family": "scenario_loso",
                "role": "primary",
                "train_scenarios": [item for item in expected_scenarios if item != scenario],
                "test_scenarios": [scenario],
                "inner_selection": "scenario_blocked_training_groups_only",
                "outer_test_fitting_allowed": False,
            }
        )
    for speed in (100, 150, 200, 250):
        held_out = [f"circle-{speed}", f"triangle-{speed}"]
        expected_folds.append(
            {
                "fold_id": f"speed_pair_out::{speed}",
                "family": "speed_pair_out",
                "role": "stress",
                "train_scenarios": [item for item in expected_scenarios if item not in held_out],
                "test_scenarios": held_out,
                "inner_selection": "scenario_blocked_training_groups_only",
                "outer_test_fitting_allowed": False,
            }
        )
    for shape in ("circle", "triangle"):
        held_out = [item for item in expected_scenarios if item.startswith(f"{shape}-")]
        expected_folds.append(
            {
                "fold_id": f"shape_out::{shape}",
                "family": "shape_out",
                "role": "stress",
                "train_scenarios": [item for item in expected_scenarios if item not in held_out],
                "test_scenarios": held_out,
                "inner_selection": "scenario_blocked_training_groups_only",
                "outer_test_fitting_allowed": False,
            }
        )
    normalized_folds: list[dict[str, Any]] = []
    for index, value in enumerate(items):
        fold = _mapping(value, name=f"preparation fold {index}")
        required = frozenset(
            {"fold_id", "family", "role", "train_scenarios", "test_scenarios", "inner_selection", "outer_test_fitting_allowed"}
        )
        _keys(fold, required, name=f"preparation fold {index}")
        family = _exact_str(fold["family"], name="fold family")
        role = _exact_str(fold["role"], name="fold role")
        if family not in family_roles or role != family_roles[family]:
            _fail("preparation fold family/role changed")
        fold_id = _exact_str(fold["fold_id"], name="fold ID")
        train = fold["train_scenarios"]
        test = fold["test_scenarios"]
        if type(train) is not list or type(test) is not list or any(type(item) is not str or not item for item in train + test):
            _fail("preparation fold scenario lists are malformed")
        if set(train).intersection(test) or len(set(train)) != len(train) or len(set(test)) != len(test):
            _fail("preparation fold scenarios overlap or duplicate")
        _exact_str(fold["inner_selection"], name="fold inner selection", expected="scenario_blocked_training_groups_only")
        _exact_bool(fold["outer_test_fitting_allowed"], name="fold outer fitting", expected=False)
        normalized_folds.append(dict(fold))
    if normalized_folds != expected_folds:
        _fail("preparation fold declarations or their order differ from the frozen 8+4+2 grid")
    tasks: list[dict[str, Any]] = []
    ordinal = 0
    for method_id in METHOD_ORDER:
        for fold in normalized_folds:
            for seed in SEED_ORDER:
                base = {
                    "contract": V1_CONTRACT,
                    "ordinal": ordinal,
                    "method_id": method_id,
                    "fold_id": fold["fold_id"],
                    "family": fold["family"],
                    "role": fold["role"],
                    "train_scenarios": list(fold["train_scenarios"]),
                    "test_scenarios": list(fold["test_scenarios"]),
                    "seed": seed,
                    "checkpoint_selection_scope": "training_scenarios_only",
                    "outer_eval_count": 1,
                }
                digest = canonical_sha256(base)
                tasks.append({**base, "task_id": f"task_{ordinal:04d}_{digest[:12]}", "spec_sha256": digest})
                ordinal += 1
    return tasks


def _validate_task_index(value: Any, expected_tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = _mapping(value, name="v1 task index")
    _keys(index, frozenset({"contract", "task_count", "primary_task_count", "stress_task_count", "tasks"}), name="v1 task index")
    _exact_str(index["contract"], name="task-index contract", expected=V1_CONTRACT)
    _exact_int(index["task_count"], name="task-index count", expected=336)
    _exact_int(index["primary_task_count"], name="task-index primary count", expected=192)
    _exact_int(index["stress_task_count"], name="task-index stress count", expected=144)
    tasks = index["tasks"]
    if type(tasks) is not list or tasks != list(expected_tasks):
        _fail("v1 task index differs from the deterministic frozen 336-task grid")
    return [dict(task) for task in tasks]


def _validate_training_receipt(value: Any, task: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _mapping(value, name="masked_erm training receipt")
    keys = frozenset(
        {
            "method_id",
            "seed",
            "updates",
            "checkpoint_update",
            "checkpoint_score_m",
            "checkpoint_selection_scope",
            "outer_eval_count",
            "initial_state_sha256",
            "checkpoint_state_sha256",
            "view_sha256",
        }
    )
    _keys(receipt, keys, name="masked_erm training receipt")
    _exact_str(receipt["method_id"], name="training method", expected=task["method_id"])
    _exact_int(receipt["seed"], name="training seed", expected=task["seed"])
    _exact_int(receipt["updates"], name="training updates", expected=400)
    checkpoint = _exact_int(receipt["checkpoint_update"], name="checkpoint update")
    if not 1 <= checkpoint <= 400:
        _fail("checkpoint update is outside the frozen budget")
    _finite_nonnegative(receipt["checkpoint_score_m"], name="checkpoint score")
    _exact_str(receipt["checkpoint_selection_scope"], name="checkpoint scope", expected="training_scenarios_only")
    _exact_int(receipt["outer_eval_count"], name="training outer count", expected=0)
    for key in ("initial_state_sha256", "checkpoint_state_sha256", "view_sha256"):
        _hex64(receipt[key], name=f"training {key}")
    return dict(receipt)


def _validate_outer_metric(value: Any, task: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(value, name="masked_erm outer metric")
    keys = frozenset(
        {"task_id", "method_id", "fold_id", "family", "role", "seed", "checkpoint_selection_scope", "outer_eval_count", "metrics"}
    )
    _keys(payload, keys, name="masked_erm outer metric")
    for key in ("task_id", "method_id", "fold_id", "family", "role"):
        _exact_str(payload[key], name=f"outer metric {key}", expected=task[key])
    _exact_int(payload["seed"], name="outer metric seed", expected=task["seed"])
    _exact_str(payload["checkpoint_selection_scope"], name="outer metric scope", expected="training_scenarios_only")
    _exact_int(payload["outer_eval_count"], name="outer metric count", expected=1)
    metrics = _mapping(payload["metrics"], name="outer metric body")
    _keys(metrics, frozenset({"outer_eval_count", "views", "finite"}), name="outer metric body")
    _exact_int(metrics["outer_eval_count"], name="metric evaluator count", expected=1)
    _exact_bool(metrics["finite"], name="outer metric finite", expected=True)
    views = metrics["views"]
    if type(views) is not list or len(views) != 5:
        _fail("outer metric does not contain exactly five views")
    view_keys = frozenset({"view_id", "rows", "rmse3d_m", "p95_distance_m", "mean_distance_m"})
    for index, raw in enumerate(views):
        view = _mapping(raw, name=f"outer metric view {index}")
        _keys(view, view_keys, name=f"outer metric view {index}")
        _exact_str(view["view_id"], name="outer metric view ID", expected=VIEW_ORDER[index])
        rows = _exact_int(view["rows"], name="outer metric rows")
        if rows < 1:
            _fail("outer metric rows must be positive")
        for key in ("rmse3d_m", "p95_distance_m", "mean_distance_m"):
            _finite_nonnegative(view[key], name=f"outer metric {key}")
    return dict(payload)


def _validate_gpu_identity(value: Any, *, name: str) -> dict[str, Any]:
    gpu = _mapping(value, name=name)
    _keys(
        gpu,
        frozenset({"physical_index", "uuid", "pci_bus_id", "name"}),
        name=name,
    )
    physical_index = _exact_int(gpu["physical_index"], name=f"{name}.physical_index")
    if physical_index not in range(4):
        _fail(f"{name}.physical_index is outside 0..3")
    if not _exact_str(gpu["uuid"], name=f"{name}.uuid").startswith("GPU-"):
        _fail(f"{name}.uuid is not an NVIDIA UUID")
    if re.fullmatch(
        r"[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]",
        _exact_str(gpu["pci_bus_id"], name=f"{name}.pci_bus_id"),
    ) is None:
        _fail(f"{name}.pci_bus_id is malformed")
    _exact_str(gpu["name"], name=f"{name}.name")
    return dict(gpu)


def _validate_execution_binding(value: Any, *, name: str) -> dict[str, Any]:
    binding = _mapping(value, name=name)
    keys = frozenset(
        {
            "execution_session_id",
            "launch_id",
            "launch_started",
            "launch_payload_sha256",
            "transport_slot_index",
            "local_slot",
            "gpu",
            "controller_source_sha256",
            "task_assignment_sha256",
            "slot_assignment_sha256",
            "worker_ready",
            "controller_ready_observations",
            "controller_observations_sha256",
            "worker_identity_source",
            "scientific_fit_attempt",
        }
    )
    _keys(binding, keys, name=name)
    _hex64(binding["execution_session_id"], name=f"{name}.execution_session_id")
    launch_id = _hex64(binding["launch_id"], name=f"{name}.launch_id")
    for key in (
        "launch_payload_sha256",
        "controller_source_sha256",
        "task_assignment_sha256",
        "slot_assignment_sha256",
        "controller_observations_sha256",
    ):
        _hex64(binding[key], name=f"{name}.{key}")
    for key in ("transport_slot_index", "local_slot"):
        if _exact_int(binding[key], name=f"{name}.{key}") < 0:
            _fail(f"{name}.{key} is negative")
    _validate_gpu_identity(binding["gpu"], name=f"{name}.gpu")
    launch_directory = f"runtime/execution_launches/launch_{launch_id}"
    _validate_reference_shape(
        binding["launch_started"],
        name=f"{name}.launch_started",
        expected_path=f"{launch_directory}/launch.started.json",
    )
    _validate_reference_shape(
        binding["controller_ready_observations"],
        name=f"{name}.controller_ready_observations",
        expected_path=f"{launch_directory}/controller_ready_observations.json",
    )
    slot_index = binding["transport_slot_index"]
    _validate_reference_shape(
        binding["worker_ready"],
        name=f"{name}.worker_ready",
        expected_path=f"runtime/worker_ready/launch_{launch_id}/slot_{slot_index:04d}.json",
    )
    _exact_str(
        binding["worker_identity_source"],
        name=f"{name}.worker_identity_source",
        expected="fixed_process_transport_and_ready_receipt",
    )
    _exact_int(binding["scientific_fit_attempt"], name=f"{name}.scientific_fit_attempt", expected=1)
    return dict(binding)


def _validate_launch_started_authority(
    v1_root: Path | _PinnedRoot,
    binding: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    session_reference: Mapping[str, Any],
    recovery_authority: Mapping[str, Any] | None,
    name: str,
) -> tuple[int, dict[str, Any]]:
    launch_id = binding["launch_id"]
    expected_path = f"runtime/execution_launches/launch_{launch_id}/launch.started.json"
    _launch_ref, launch_raw = _verify_reference(
        v1_root,
        binding["launch_started"],
        name=f"{name} launch-start authority",
        expected_path=expected_path,
    )
    launch = _mapping(
        strict_json_bytes(launch_raw, name=f"{name} launch-start receipt"),
        name=f"{name} launch-start receipt",
    )
    launch_keys = frozenset(
        {
            "contract",
            "execution_session_id",
            "launch_id",
            "launch_payload_sha256",
            "launch_ordinal",
            "launch_mode",
            "previous_launch",
            "prelaunch_session_recovery",
            "prelaunch_session_recovery_sha256",
            "resume",
            "session",
            "workers_per_gpu",
            "slot_count",
            "task_assignment_sha256",
            "controller_source_sha256",
            "model_source_set_sha256",
            "parent_public_validation_sha256",
            "pre_fit_validation_receipt_sha256",
        }
    )
    _keys(launch, launch_keys, name=f"{name} launch-start receipt")
    _exact_str(launch["contract"], name=f"{name} launch contract", expected=V1_CONTRACT)
    _exact_str(
        launch["execution_session_id"],
        name=f"{name} launch session",
        expected=session["execution_session_id"],
    )
    if binding["execution_session_id"] != session["execution_session_id"]:
        _fail(f"{name} binding differs from authenticated execution session")
    _exact_str(launch["launch_id"], name=f"{name} launch ID", expected=launch_id)
    ordinal = _exact_int(launch["launch_ordinal"], name=f"{name} launch ordinal")
    if ordinal < 0:
        _fail(f"{name} launch ordinal is negative")
    resume = launch["resume"]
    if type(resume) is not bool:
        _fail(f"{name} launch resume marker is not exact")
    if ordinal == 0:
        expected_modes = (
            {PRELAUNCH_RECOVERY_MODE}
            if resume
            else {"fresh_first_launch"}
        )
        if launch["previous_launch"] is not None:
            _fail(f"{name} first launch has a predecessor")
    else:
        _exact_bool(resume, name=f"{name} resumed launch marker", expected=True)
        expected_modes = {"resume_after_prior_launch"}
        _validate_reference_shape(
            launch["previous_launch"], name=f"{name} previous launch"
        )
    mode = _exact_str(launch["launch_mode"], name=f"{name} launch mode")
    if mode not in expected_modes:
        _fail(f"{name} launch mode disagrees with its ordinal/resume marker")
    expected_recovery_ref = (
        None if recovery_authority is None else recovery_authority["reference"]
    )
    expected_recovery_sha = (
        None if recovery_authority is None else recovery_authority["semantic_sha256"]
    )
    if mode == PRELAUNCH_RECOVERY_MODE and recovery_authority is None:
        _fail(f"{name} recovered first launch lacks authenticated recovery authority")
    if mode == "fresh_first_launch" and recovery_authority is not None:
        _fail(f"{name} fresh first launch conflicts with persisted recovery authority")
    if (
        launch["prelaunch_session_recovery"] != expected_recovery_ref
        or launch["prelaunch_session_recovery_sha256"] != expected_recovery_sha
    ):
        _fail(f"{name} launch recovery raw/semantic authority changed")
    if launch["session"] != session_reference:
        _fail(f"{name} launch session raw reference changed")
    _exact_int(launch["workers_per_gpu"], name=f"{name} workers/GPU", expected=2)
    _exact_int(launch["slot_count"], name=f"{name} slot count", expected=8)
    for key in (
        "task_assignment_sha256",
        "controller_source_sha256",
        "model_source_set_sha256",
        "parent_public_validation_sha256",
        "pre_fit_validation_receipt_sha256",
    ):
        _hex64(launch[key], name=f"{name} launch {key}")
    immutable_session_values = {
        "task_assignment_sha256": session["task_assignment_sha256"],
        "controller_source_sha256": session["controller_source_sha256"],
        "model_source_set_sha256": session["model_source_set_sha256"],
        "parent_public_validation_sha256": session[
            "parent_public_validation_sha256"
        ],
        "pre_fit_validation_receipt_sha256": session[
            "pre_fit_validation_receipt_sha256"
        ],
    }
    if any(launch[key] != value for key, value in immutable_session_values.items()):
        _fail(f"{name} launch receipt differs from immutable execution session")
    if (
        launch["task_assignment_sha256"] != binding["task_assignment_sha256"]
        or launch["controller_source_sha256"] != binding["controller_source_sha256"]
        or launch["launch_payload_sha256"] != binding["launch_payload_sha256"]
    ):
        _fail(f"{name} launch receipt differs from execution binding")
    expected_launch_id = canonical_sha256(
        {
            "contract": V1_CONTRACT,
            "execution_session_id": binding["execution_session_id"],
            "launch_ordinal": ordinal,
            "previous_launch": launch["previous_launch"],
            "launch_mode": mode,
        }
    )
    if launch_id != expected_launch_id:
        _fail(f"{name} launch ID is not the canonical semantic ID")
    _payload_hash(launch, key="launch_payload_sha256", name=f"{name} launch-start")
    return ordinal, dict(launch)


def _validate_worker_ready_authority(
    v1_root: Path | _PinnedRoot,
    reference: Any,
    *,
    launch: Mapping[str, Any],
    launch_reference: Mapping[str, Any],
    expected_slot_index: int,
    name: str,
    session: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_path = (
        f"runtime/worker_ready/launch_{launch['launch_id']}/"
        f"slot_{expected_slot_index:04d}.json"
    )
    _ready_ref, ready_raw = _verify_reference(
        v1_root, reference, name=name, expected_path=expected_path
    )
    ready = _mapping(
        strict_json_bytes(ready_raw, name=name), name=name
    )
    _keys(
        ready,
        frozenset(
            {
                "contract",
                "execution_session_id",
                "launch_id",
                "launch_started",
                "launch_payload_sha256",
                "slot_index",
                "local_slot",
                "gpu",
                "visible_child_device",
                "cuda_visible_devices",
                "controller_source_sha256",
                "task_assignment_sha256",
                "slot_assignment_sha256",
                "task_ids_sha256",
                "child_process_id",
                "ready_after_exact_gpu_validation",
                "ready_receipt_sha256",
            }
        ),
        name=name,
    )
    _exact_str(ready["contract"], name=f"{name} contract", expected=V1_CONTRACT)
    for key in ("execution_session_id", "launch_id", "launch_payload_sha256"):
        _exact_str(ready[key], name=f"{name} {key}", expected=launch[key])
    if ready["launch_started"] != launch_reference:
        _fail(f"{name} launch-start raw reference changed")
    _exact_int(ready["slot_index"], name=f"{name} slot", expected=expected_slot_index)
    local_slot = _exact_int(ready["local_slot"], name=f"{name} local slot")
    if local_slot not in (0, 1):
        _fail(f"{name} local slot is outside 0..1")
    gpu = _validate_gpu_identity(ready["gpu"], name=f"{name} GPU")
    _exact_str(ready["visible_child_device"], name=f"{name} child device", expected="cuda:0")
    _exact_str(
        ready["cuda_visible_devices"],
        name=f"{name} CUDA visibility",
        expected=gpu["uuid"],
    )
    for key in (
        "controller_source_sha256",
        "task_assignment_sha256",
        "slot_assignment_sha256",
        "task_ids_sha256",
    ):
        _hex64(ready[key], name=f"{name} {key}")
    if (
        ready["controller_source_sha256"] != launch["controller_source_sha256"]
        or ready["task_assignment_sha256"] != launch["task_assignment_sha256"]
    ):
        _fail(f"{name} immutable launch authority changed")
    if session is not None:
        session_slot = session["slots"][expected_slot_index]
        if (
            ready["local_slot"] != session_slot["local_slot"]
            or ready["gpu"] != session_slot["gpu"]
            or ready["slot_assignment_sha256"] != session_slot["assignment_sha256"]
            or ready["task_ids_sha256"] != canonical_sha256(session_slot["task_ids"])
        ):
            _fail(f"{name} worker-ready receipt differs from immutable session slot")
    if _exact_int(ready["child_process_id"], name=f"{name} process ID") <= 0:
        _fail(f"{name} process ID is not positive")
    _exact_bool(
        ready["ready_after_exact_gpu_validation"],
        name=f"{name} exact GPU marker",
        expected=True,
    )
    _payload_hash(ready, key="ready_receipt_sha256", name=name)
    return dict(ready)


def _validate_controller_authority(
    v1_root: Path | _PinnedRoot,
    binding: Mapping[str, Any],
    launch: Mapping[str, Any],
    *,
    name: str,
    session: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _controller_ref, controller_raw = _verify_reference(
        v1_root,
        binding["controller_ready_observations"],
        name=f"{name} controller-ready authority",
        expected_path=(
            f"runtime/execution_launches/launch_{launch['launch_id']}/"
            "controller_ready_observations.json"
        ),
    )
    controller = _mapping(
        strict_json_bytes(controller_raw, name=f"{name} controller-ready receipt"),
        name=f"{name} controller-ready receipt",
    )
    _keys(
        controller,
        frozenset(
            {
                "contract",
                "execution_session_id",
                "launch_id",
                "launch_started",
                "launch_payload_sha256",
                "controller_source_sha256",
                "task_assignment_sha256",
                "slot_count",
                "observations",
                "controller_observed_spawn_process_identity",
                "controller_observations_sha256",
            }
        ),
        name=f"{name} controller-ready receipt",
    )
    for key, expected in (
        ("contract", V1_CONTRACT),
        ("execution_session_id", launch["execution_session_id"]),
        ("launch_id", launch["launch_id"]),
        ("launch_payload_sha256", launch["launch_payload_sha256"]),
        ("controller_source_sha256", launch["controller_source_sha256"]),
        ("task_assignment_sha256", launch["task_assignment_sha256"]),
    ):
        _exact_str(controller[key], name=f"{name} controller {key}", expected=expected)
    if controller["launch_started"] != binding["launch_started"]:
        _fail(f"{name} controller launch-start reference changed")
    _exact_int(controller["slot_count"], name=f"{name} controller slot count", expected=8)
    _exact_bool(
        controller["controller_observed_spawn_process_identity"],
        name=f"{name} controller process marker",
        expected=True,
    )
    observations = controller["observations"]
    if type(observations) is not list or len(observations) != 8:
        _fail(f"{name} controller observations are not the exact eight slots")
    process_ids: set[int] = set()
    selected_ready: dict[str, Any] | None = None
    for slot_index, raw in enumerate(observations):
        observation = _mapping(raw, name=f"{name} controller observation {slot_index}")
        _keys(
            observation,
            frozenset(
                {
                    "slot_index",
                    "child_process_id",
                    "child_process_name",
                    "gpu",
                    "slot_assignment_sha256",
                    "ready_receipt",
                    "ready_receipt_sha256",
                }
            ),
            name=f"{name} controller observation {slot_index}",
        )
        _exact_int(
            observation["slot_index"],
            name=f"{name} observed slot",
            expected=slot_index,
        )
        process_id = _exact_int(
            observation["child_process_id"], name=f"{name} observed process ID"
        )
        if process_id <= 0 or process_id in process_ids:
            _fail(f"{name} observed process identities are not positive and unique")
        process_ids.add(process_id)
        _exact_str(
            observation["child_process_name"],
            name=f"{name} observed process name",
            expected=f"openaxes-model-slot-{slot_index}",
        )
        _validate_gpu_identity(observation["gpu"], name=f"{name} observed GPU {slot_index}")
        _hex64(
            observation["slot_assignment_sha256"],
            name=f"{name} observed slot-assignment hash",
        )
        ready = _validate_worker_ready_authority(
            v1_root,
            observation["ready_receipt"],
            launch=launch,
            launch_reference=binding["launch_started"],
            expected_slot_index=slot_index,
            name=f"{name} worker-ready slot {slot_index}",
            session=session,
        )
        if (
            observation["child_process_id"] != ready["child_process_id"]
            or observation["gpu"] != ready["gpu"]
            or observation["slot_assignment_sha256"] != ready["slot_assignment_sha256"]
            or observation["ready_receipt_sha256"] != ready["ready_receipt_sha256"]
        ):
            _fail(f"{name} controller observation differs from worker-ready receipt")
        if slot_index == binding["transport_slot_index"]:
            selected_ready = ready
            if observation["ready_receipt"] != binding["worker_ready"]:
                _fail(f"{name} binding worker-ready reference changed")
    if selected_ready is None:
        _fail(f"{name} binding slot is absent from controller observations")
    if (
        selected_ready["local_slot"] != binding["local_slot"]
        or selected_ready["gpu"] != binding["gpu"]
        or selected_ready["slot_assignment_sha256"] != binding["slot_assignment_sha256"]
    ):
        _fail(f"{name} binding slot authority differs from worker-ready receipt")
    _payload_hash(
        controller,
        key="controller_observations_sha256",
        name=f"{name} controller-ready receipt",
    )
    if controller["controller_observations_sha256"] != binding[
        "controller_observations_sha256"
    ]:
        _fail(f"{name} controller self-hash differs from execution binding")
    return dict(controller)


def _validate_binding_authorities(
    v1_root: Path | _PinnedRoot,
    value: Any,
    *,
    session: Mapping[str, Any],
    session_reference: Mapping[str, Any],
    recovery_authority: Mapping[str, Any] | None,
    task_id: str | None = None,
    name: str,
) -> tuple[dict[str, Any], int]:
    binding = _validate_execution_binding(value, name=name)
    ordinal, launch = _validate_launch_started_authority(
        v1_root,
        binding,
        session=session,
        session_reference=session_reference,
        recovery_authority=recovery_authority,
        name=name,
    )
    _validate_controller_authority(
        v1_root, binding, launch, name=name, session=session
    )
    if task_id is not None:
        slot_index = binding["transport_slot_index"]
        slots = session["slots"]
        if slot_index >= len(slots) or task_id not in slots[slot_index]["task_ids"]:
            _fail(f"{name} task is not assigned to its authenticated transport slot")
    return binding, ordinal


def _verify_exact_result_directory(root: Path | _PinnedRoot, task_id: str) -> None:
    expected = {
        "checkpoint.pt",
        "transform.json",
        "history.json",
        "training_receipt.json",
        "outer_predictions.parquet",
        "outer_evaluation_rows.parquet",
        "outer_evaluation.json",
        "bundle.json",
        "result.json",
    }
    base_descriptor = root.descriptor if isinstance(root, _PinnedRoot) else None
    descriptors: list[int] = []
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.dup(base_descriptor) if base_descriptor is not None else os.open(root, flags)
        descriptors.append(descriptor)
        for component in ("results", task_id):
            descriptor = os.open(component, flags, dir_fd=descriptor)
            descriptors.append(descriptor)
        names = set(os.listdir(descriptor))
        if names != expected:
            _fail(f"sealed result directory for {task_id} is not the exact nine-file set")
        for filename in expected:
            info = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                _fail(f"sealed result object is not regular: {task_id}/{filename}")
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError(
            f"cannot inspect sealed result directory for {task_id}"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_exact_phase_directory(root: Path | _PinnedRoot, task_id: str) -> None:
    expected = {f"{phase}.json" for phase in V1_PHASE_ORDER}
    base_descriptor = root.descriptor if isinstance(root, _PinnedRoot) else None
    descriptors: list[int] = []
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.dup(base_descriptor) if base_descriptor is not None else os.open(root, flags)
        descriptors.append(descriptor)
        for component in ("runtime", "ledgers", task_id):
            descriptor = os.open(component, flags, dir_fd=descriptor)
            descriptors.append(descriptor)
        if set(os.listdir(descriptor)) != expected:
            _fail(f"phase directory for {task_id} is not the exact seven-file set")
        for filename in expected:
            if not stat.S_ISREG(
                os.stat(filename, dir_fd=descriptor, follow_symlinks=False).st_mode
            ):
                _fail(f"phase receipt is not regular: {task_id}/{filename}")
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError(
            f"cannot inspect phase directory for {task_id}"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_exact_directory_members(
    root: Path | _PinnedRoot,
    relative: str,
    *,
    expected_files: set[str],
    expected_directories: set[str],
    name: str,
) -> None:
    """Close one directory member set and reject links/special objects."""

    path = (
        ""
        if relative == "."
        else _canonical_relative_path(relative, name=f"{name}.path")
    )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        descriptor = (
            os.dup(root.descriptor)
            if isinstance(root, _PinnedRoot)
            else os.open(root, flags)
        )
        descriptors.append(descriptor)
        if path:
            for component in path.split("/"):
                descriptor = os.open(component, flags, dir_fd=descriptor)
                descriptors.append(descriptor)
        names = set(os.listdir(descriptor))
        expected = expected_files | expected_directories
        if names != expected:
            _fail(f"{name} member set changed")
        for member in expected_files:
            if not stat.S_ISREG(
                os.stat(member, dir_fd=descriptor, follow_symlinks=False).st_mode
            ):
                _fail(f"{name} expected regular file changed: {member}")
        for member in expected_directories:
            if not stat.S_ISDIR(
                os.stat(member, dir_fd=descriptor, follow_symlinks=False).st_mode
            ):
                _fail(f"{name} expected directory changed: {member}")
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError(f"cannot inspect {name}") from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_complete_task_directory_sets(
    v1_root: Path | _PinnedRoot, task_ids: set[str]
) -> None:
    if len(task_ids) != V1_TASK_COUNT or any(
        re.fullmatch(r"task_[0-9]{4}_[0-9a-f]{12}", task_id) is None
        for task_id in task_ids
    ):
        _fail("complete task identity set is not the exact 336 canonical IDs")
    _verify_exact_directory_members(
        v1_root,
        "tasks",
        expected_files={"index.json"},
        expected_directories=set(),
        name="complete tasks directory",
    )
    _verify_exact_directory_members(
        v1_root,
        "results",
        expected_files=set(),
        expected_directories=task_ids,
        name="complete results directory",
    )
    _verify_exact_directory_members(
        v1_root,
        "runtime/ledgers",
        expected_files=set(),
        expected_directories=task_ids,
        name="complete phase-ledger directory",
    )
    expected_receipts = {f"{task_id}.json" for task_id in task_ids}
    for relative, label in (
        ("runtime/attempt_roster", "complete attempt-roster directory"),
        ("runtime/attempt_ledger", "complete attempt-ledger directory"),
    ):
        _verify_exact_directory_members(
            v1_root,
            relative,
            expected_files=expected_receipts,
            expected_directories=set(),
            name=label,
        )
    _verify_exact_directory_members(
        v1_root,
        "quarantine",
        expected_files=set(),
        expected_directories=set(),
        name="complete empty quarantine directory",
    )


def _verify_complete_root_and_runtime_sets(v1_root: Path | _PinnedRoot) -> bool:
    _verify_exact_directory_members(
        v1_root,
        ".",
        expected_files={
            "prepared_manifest.json",
            "PREPARED.json",
            "development_manifest.json",
            "COMPLETE.json",
        },
        expected_directories={"tasks", "runtime", "results", "quarantine"},
        name="complete v1 root",
    )
    required_runtime_files = {
        "pre_fit_validation.json",
        "execution_session.json",
        "full_signed_source_replay.json",
    }
    required_runtime_directories = {
        "ledgers",
        "execution_launches",
        "worker_ready",
        "attempt_ledger",
        "attempt_roster",
    }
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        descriptor = (
            os.dup(v1_root.descriptor)
            if isinstance(v1_root, _PinnedRoot)
            else os.open(v1_root, flags)
        )
        descriptors.append(descriptor)
        descriptor = os.open("runtime", flags, dir_fd=descriptor)
        descriptors.append(descriptor)
        observed = set(os.listdir(descriptor))
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError("cannot inspect complete runtime") from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
    allowed = (
        required_runtime_files | required_runtime_directories,
        required_runtime_files
        | required_runtime_directories
        | {"prelaunch_session_recovery.json"},
    )
    if observed not in allowed:
        _fail("complete runtime member set changed")
    _verify_exact_directory_members(
        v1_root,
        "runtime",
        expected_files=observed - required_runtime_directories,
        expected_directories=required_runtime_directories,
        name="complete runtime",
    )
    return "prelaunch_session_recovery.json" in observed


def _validate_attempt_receipts(
    v1_root: Path | _PinnedRoot,
    task: Mapping[str, Any],
    *,
    session: Mapping[str, Any],
    session_reference: Mapping[str, Any],
    recovery_authority: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    task_id = task["task_id"]
    roster_path = f"runtime/attempt_roster/{task_id}.json"
    ledger_path = f"runtime/attempt_ledger/{task_id}.json"
    _roster_ref, roster_raw = _raw_reference(v1_root, roster_path, name=f"attempt roster {task_id}")
    _ledger_ref, ledger_raw = _raw_reference(v1_root, ledger_path, name=f"attempt ledger {task_id}")
    roster = _mapping(strict_json_bytes(roster_raw, name=f"attempt roster {task_id}"), name="attempt roster")
    ledger = _mapping(strict_json_bytes(ledger_raw, name=f"attempt ledger {task_id}"), name="attempt ledger")
    common = {
        "contract",
        "task_id",
        "task_spec_sha256",
        "attempt_id",
        "origin",
        "binding",
        "attempt_started_or_state_observed",
    }
    _keys(
        roster,
        frozenset(
            common
            | {
                "monotonic_first_attempt_roster",
                "survives_ledger_deletion_and_quarantine",
                "attempt_roster_sha256",
            }
        ),
        name="attempt roster",
    )
    _keys(
        ledger,
        frozenset(common | {"survives_quarantine", "attempt_ledger_sha256"}),
        name="attempt ledger",
    )
    for receipt, label in ((roster, "roster"), (ledger, "ledger")):
        _exact_str(receipt["contract"], name=f"attempt {label} contract", expected=V1_CONTRACT)
        _exact_str(receipt["task_id"], name=f"attempt {label} task", expected=task_id)
        _exact_str(
            receipt["task_spec_sha256"],
            name=f"attempt {label} task spec",
            expected=task["spec_sha256"],
        )
        _exact_str(receipt["origin"], name=f"attempt {label} origin", expected="production_execution")
        _exact_bool(
            receipt["attempt_started_or_state_observed"],
            name=f"attempt {label} started marker",
            expected=True,
        )
    attempt_id = _hex64(roster["attempt_id"], name="attempt ID")
    if ledger["attempt_id"] != attempt_id or ledger["binding"] != roster["binding"]:
        _fail("attempt roster and ledger identity/binding disagree")
    binding, _launch_ordinal = _validate_binding_authorities(
        v1_root,
        roster["binding"],
        session=session,
        session_reference=session_reference,
        recovery_authority=recovery_authority,
        task_id=task_id,
        name="attempt execution binding",
    )
    expected_attempt_id = canonical_sha256(
        {
            "contract": V1_CONTRACT,
            "task_id": task_id,
            "task_spec_sha256": task["spec_sha256"],
            "first_execution_binding": binding,
            "scientific_fit_attempt": 1,
        }
    )
    if attempt_id != expected_attempt_id:
        _fail("attempt ID is not the canonical first-binding semantic identity")
    _exact_bool(
        roster["monotonic_first_attempt_roster"], name="monotonic attempt roster", expected=True
    )
    _exact_bool(
        roster["survives_ledger_deletion_and_quarantine"],
        name="roster durability marker",
        expected=True,
    )
    _exact_bool(ledger["survives_quarantine"], name="ledger quarantine marker", expected=True)
    _payload_hash(roster, key="attempt_roster_sha256", name="attempt roster")
    _payload_hash(ledger, key="attempt_ledger_sha256", name="attempt ledger")
    return attempt_id, binding


def _validate_task_result_lineage(
    v1_root: Path | _PinnedRoot,
    task: Mapping[str, Any],
    indexed_result: Any,
    *,
    session: Mapping[str, Any],
    session_reference: Mapping[str, Any],
    recovery_authority: Mapping[str, Any] | None,
    validate_metric: bool,
) -> dict[str, Any]:
    task_id = task["task_id"]
    _verify_exact_result_directory(v1_root, task_id)
    _verify_exact_phase_directory(v1_root, task_id)
    attempt_id, binding = _validate_attempt_receipts(
        v1_root,
        task,
        session=session,
        session_reference=session_reference,
        recovery_authority=recovery_authority,
    )
    phase_values: list[dict[str, Any]] = []
    phase_refs: list[dict[str, Any]] = []
    phase_keys = frozenset(
        {
            "contract",
            "task_id",
            "task_spec_sha256",
            "phase",
            "attempt_id",
            "previous_phase",
            "execution_binding",
            "payload",
        }
    )
    previous_binding: dict[str, Any] | None = None
    previous_launch_ordinal: int | None = None
    stable_binding_keys = (
        "execution_session_id",
        "transport_slot_index",
        "local_slot",
        "gpu",
        "controller_source_sha256",
        "task_assignment_sha256",
        "slot_assignment_sha256",
        "worker_identity_source",
        "scientific_fit_attempt",
    )
    for index, phase_name in enumerate(V1_PHASE_ORDER):
        path = f"runtime/ledgers/{task_id}/{phase_name}.json"
        phase_ref, phase_raw = _raw_reference(v1_root, path, name=f"phase {task_id}/{phase_name}")
        phase = _mapping(strict_json_bytes(phase_raw, name=f"phase {task_id}/{phase_name}"), name="phase receipt")
        _keys(phase, phase_keys, name="phase receipt")
        _exact_str(phase["contract"], name="phase contract", expected=V1_CONTRACT)
        _exact_str(phase["task_id"], name="phase task", expected=task_id)
        _exact_str(
            phase["task_spec_sha256"], name="phase task spec", expected=task["spec_sha256"]
        )
        _exact_str(phase["phase"], name="phase name", expected=phase_name)
        _exact_str(phase["attempt_id"], name="phase attempt", expected=attempt_id)
        expected_previous = None if index == 0 else phase_refs[index - 1]
        if phase["previous_phase"] != expected_previous:
            _fail("phase previous receipt does not bind the exact prior raw phase")
        phase_binding, launch_ordinal = _validate_binding_authorities(
            v1_root,
            phase["execution_binding"],
            session=session,
            session_reference=session_reference,
            recovery_authority=recovery_authority,
            task_id=task_id,
            name=f"phase {task_id}/{phase_name} execution binding",
        )
        if index == 0 and phase_binding != binding:
            _fail("execution-start binding differs from first-attempt authority")
        if previous_binding is not None:
            for key in stable_binding_keys:
                if phase_binding[key] != previous_binding[key]:
                    _fail(f"phase execution lineage changed stable field: {key}")
            if previous_launch_ordinal is None or launch_ordinal < previous_launch_ordinal:
                _fail("phase execution launch ordinal moved backwards")
            if launch_ordinal == previous_launch_ordinal and phase_binding != previous_binding:
                _fail("same-ordinal phase execution binding changed")
        previous_binding = phase_binding
        previous_launch_ordinal = launch_ordinal
        _mapping(phase["payload"], name="phase payload")
        phase_values.append(dict(phase))
        phase_refs.append(phase_ref)

    if phase_values[0]["payload"] != binding:
        _fail("execution.started payload differs from the exact execution binding")
    training = _mapping(phase_values[1]["payload"], name="training.completed payload")
    training_keys = frozenset(
        {
            "method_id",
            "seed",
            "updates",
            "checkpoint_update",
            "checkpoint_score_m",
            "checkpoint_selection_scope",
            "outer_eval_count",
            "initial_state_sha256",
            "checkpoint_state_sha256",
            "view_sha256",
            "artifacts",
        }
    )
    _keys(training, training_keys, name="training.completed payload")
    artifacts = _mapping(training["artifacts"], name="training artifact references")
    _keys(
        artifacts,
        frozenset({"checkpoint", "transform", "history", "training_receipt"}),
        name="training artifact references",
    )
    expected_artifact_paths = {
        "checkpoint": f"results/{task_id}/checkpoint.pt",
        "transform": f"results/{task_id}/transform.json",
        "history": f"results/{task_id}/history.json",
        "training_receipt": f"results/{task_id}/training_receipt.json",
    }
    verified_artifact_refs: dict[str, dict[str, Any]] = {}
    artifact_raw: dict[str, bytes] = {}
    for key, path in expected_artifact_paths.items():
        verified_artifact_refs[key], artifact_raw[key] = _verify_reference(
            v1_root,
            artifacts[key],
            name=f"sealed training artifact {task_id}/{key}",
            expected_path=path,
        )
    training_plain = {key: value for key, value in training.items() if key != "artifacts"}
    _validate_training_receipt(training_plain, task)
    verified_training_ref = verified_artifact_refs["training_receipt"]
    training_raw = artifact_raw["training_receipt"]
    if strict_json_bytes(training_raw, name=f"training receipt {task_id}") != training_plain:
        _fail("training.completed payload differs from sealed training receipt")

    prediction = _mapping(phase_values[2]["payload"], name="outer_prediction.completed payload")
    _keys(
        prediction,
        frozenset({"rows", "base_rows", "views", "finite", "prediction"}),
        name="outer_prediction.completed payload",
    )
    base_rows = _exact_int(prediction["base_rows"], name="prediction base rows")
    if base_rows < 1 or prediction["rows"] != base_rows * 5:
        _fail("outer prediction row multiplicity changed")
    _exact_int(prediction["views"], name="prediction views", expected=5)
    _exact_bool(prediction["finite"], name="prediction finite marker", expected=True)
    prediction_ref, _opaque_prediction_raw = _verify_reference(
        v1_root,
        prediction["prediction"],
        name="opaque outer prediction reference",
        expected_path=f"results/{task_id}/outer_predictions.parquet",
    )
    started = _mapping(phase_values[3]["payload"], name="outer_eval.started payload")
    _keys(
        started,
        frozenset({"prediction", "truth_read_count_before", "evaluator_call_count_before"}),
        name="outer_eval.started payload",
    )
    if started["prediction"] != prediction_ref:
        _fail("outer-eval start does not bind the sealed prediction reference")
    _exact_int(started["truth_read_count_before"], name="truth read count before", expected=0)
    _exact_int(started["evaluator_call_count_before"], name="evaluator count before", expected=0)
    evaluation = _mapping(phase_values[4]["payload"], name="outer_eval.completed payload")
    _keys(
        evaluation,
        frozenset(
            {
                "outer_eval_count",
                "checkpoint_selection_scope",
                "evaluation_rows",
                "evaluation_metrics",
            }
        ),
        name="outer_eval.completed payload",
    )
    _exact_int(evaluation["outer_eval_count"], name="evaluation count", expected=1)
    _exact_str(
        evaluation["checkpoint_selection_scope"],
        name="evaluation checkpoint scope",
        expected="training_scenarios_only",
    )
    evaluation_rows_ref, _opaque_evaluation_rows_raw = _verify_reference(
        v1_root,
        evaluation["evaluation_rows"],
        name="opaque evaluation rows reference",
        expected_path=f"results/{task_id}/outer_evaluation_rows.parquet",
    )
    metric_ref, metric_raw = _verify_reference(
        v1_root,
        evaluation["evaluation_metrics"],
        name="outer metric reference",
        expected_path=f"results/{task_id}/outer_evaluation.json",
    )

    bundle_ref, bundle_raw = _raw_reference(
        v1_root, f"results/{task_id}/bundle.json", name=f"sealed bundle {task_id}"
    )
    bundle = _mapping(strict_json_bytes(bundle_raw, name=f"sealed bundle {task_id}"), name="sealed task bundle")
    _keys(
        bundle,
        frozenset({"contract", "task", "training", "evaluation", "phase_lineage_sha256"}),
        name="sealed task bundle",
    )
    _exact_str(bundle["contract"], name="task bundle contract", expected=V1_CONTRACT)
    if bundle["task"] != task or bundle["training"] != training or bundle["evaluation"] != evaluation:
        _fail("sealed task bundle payload differs from task/phase lineage")
    if bundle["phase_lineage_sha256"] != canonical_sha256(phase_values[:5]):
        _fail("sealed task bundle phase-lineage hash changed")
    bundle_phase_payload = _mapping(phase_values[5]["payload"], name="bundle.completed payload")
    _keys(bundle_phase_payload, frozenset({"bundle"}), name="bundle.completed payload")
    if bundle_phase_payload["bundle"] != bundle_ref:
        _fail("bundle.completed phase does not bind the exact bundle raw bytes")

    result_ref, result_raw = _verify_reference(
        v1_root,
        indexed_result,
        name=f"sealed result {task_id}",
        expected_path=f"results/{task_id}/result.json",
    )
    result = _mapping(strict_json_bytes(result_raw, name=f"sealed result {task_id}"), name="sealed task result")
    _keys(
        result,
        frozenset(
            {
                "contract",
                "task",
                "bundle",
                "checkpoint_selection_scope",
                "outer_eval_count",
                "scientific_fit_count",
                "synthetic_test_only",
            }
        ),
        name="sealed task result",
    )
    _exact_str(result["contract"], name="task result contract", expected=V1_CONTRACT)
    if result["task"] != task or result["bundle"] != bundle_ref:
        _fail("sealed task result does not bind its exact task/bundle")
    _exact_str(
        result["checkpoint_selection_scope"],
        name="task result checkpoint scope",
        expected="training_scenarios_only",
    )
    _exact_int(result["outer_eval_count"], name="task result evaluation count", expected=1)
    _exact_int(result["scientific_fit_count"], name="task result fit count", expected=1)
    _exact_bool(result["synthetic_test_only"], name="task result synthetic marker", expected=False)
    result_phase_payload = _mapping(phase_values[6]["payload"], name="result.completed payload")
    _keys(result_phase_payload, frozenset({"result"}), name="result.completed payload")
    if result_phase_payload["result"] != result_ref:
        _fail("result.completed phase does not bind the exact result raw bytes")

    metric: dict[str, Any] | None = None
    if validate_metric:
        metric = _validate_outer_metric(
            strict_json_bytes(metric_raw, name=f"masked primary outer metric {task_id}"), task
        )
    return {
        "training_reference": verified_training_ref,
        "training": training_plain,
        "metric_reference": metric_ref,
        "metric": metric,
        "prediction_reference": prediction_ref,
        "evaluation_rows_reference": evaluation_rows_ref,
        "result_reference": result_ref,
    }


def _extract_model_head(prepared: Mapping[str, Any]) -> str:
    preparation = _mapping(prepared.get("preparation"), name="v1 preparation snapshot")
    authority = _mapping(preparation.get("parent_git_authority"), name="v1 parent Git authority")
    authority_head = _git_sha1(authority.get("model_contract_git_head"), name="v1 model contract Git HEAD")
    repository = _mapping(prepared.get("repository"), name="v1 repository receipt")
    repository_head = _git_sha1(repository.get("model_contract_git_head"), name="v1 repository Git HEAD")
    if authority_head != repository_head:
        _fail("v1 model contract Git HEAD disagrees across immutable receipts")
    _exact_bool(repository.get("model_contract_sources_committed_at_git_head"), name="v1 source committed marker", expected=True)
    _exact_bool(repository.get("downstream_results_blocked_by_uncommitted_contract"), name="v1 downstream block marker", expected=False)
    return authority_head


def _verify_external_file_reference(
    value: Any,
    *,
    name: str,
    expected_absolute: Path | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Hash one absolute external reference through a pinned parent dirfd."""

    reference = _mapping(value, name=name)
    _keys(
        reference,
        frozenset({"absolute_path", "bytes", "sha256"}),
        name=name,
    )
    absolute_text = _exact_str(reference["absolute_path"], name=f"{name}.absolute_path")
    absolute = Path(absolute_text)
    if (
        not absolute.is_absolute()
        or os.path.normpath(absolute_text) != absolute_text
        or Path(os.path.realpath(absolute)) != absolute
        or absolute.name in {"", ".", ".."}
    ):
        _fail(f"{name} absolute path is not canonical and symlink-free")
    if expected_absolute is not None and absolute != expected_absolute:
        _fail(f"{name} absolute path differs from its designated authority")
    parent = _pin_root(absolute.parent, name=f"{name} parent")
    try:
        raw = _read_nofollow(parent, absolute.name, name=name)
        parent.revalidate(name=f"{name} parent")
    finally:
        parent.close()
    if (
        _exact_int(reference["bytes"], name=f"{name}.bytes") != len(raw)
        or _hex64(reference["sha256"], name=f"{name}.sha256")
        != hashlib.sha256(raw).hexdigest()
    ):
        _fail(f"{name} raw external-file binding changed")
    return dict(reference), raw


def _validate_execution_session_authority(
    v1_root: Path | _PinnedRoot,
    prepared: Mapping[str, Any],
    prepared_ref: Mapping[str, Any],
    task_index_ref: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _session_ref, session_raw = _raw_reference(
        v1_root, "runtime/execution_session.json", name="execution session"
    )
    session = _mapping(
        strict_json_bytes(session_raw, name="execution session"),
        name="execution session",
    )
    _keys(
        session,
        frozenset(
            {
                "contract",
                "execution_session_id",
                "prepared_manifest",
                "model_contract_git_head",
                "controller_source_sha256",
                "model_source_set_sha256",
                "prepared_runtime_sha256",
                "parent_public_validation_sha256",
                "pre_fit_validation",
                "pre_fit_validation_receipt_sha256",
                "workers_per_gpu",
                "gpu_inventory",
                "task_index_sha256",
                "task_assignment_sha256",
                "slots",
                "immutable",
                "session_payload_sha256",
            }
        ),
        name="execution session",
    )
    _exact_str(session["contract"], name="execution session contract", expected=V1_CONTRACT)
    _hex64(session["execution_session_id"], name="execution session ID")
    if session["prepared_manifest"] != prepared_ref:
        _fail("execution session prepared-manifest reference changed")
    model_head = _extract_model_head(prepared)
    _exact_str(
        session["model_contract_git_head"],
        name="execution session model head",
        expected=model_head,
    )
    source_closure = _mapping(prepared["source_closure"], name="execution source closure")
    source_files = _mapping(source_closure["files"], name="execution source files")
    controller_sha = source_files[
        "src/imu_vive_learning/openaxes_external_model_v1_experiment.py"
    ]["sha256"]
    _exact_str(
        session["controller_source_sha256"],
        name="execution controller source",
        expected=controller_sha,
    )
    _exact_str(
        session["model_source_set_sha256"],
        name="execution model source set",
        expected=source_closure["set_sha256"],
    )
    expected_runtime_sha = canonical_sha256(
        _mapping(prepared["runtime"], name="execution prepared runtime")
    )
    _exact_str(
        session["prepared_runtime_sha256"],
        name="execution prepared runtime hash",
        expected=expected_runtime_sha,
    )
    preparation = _mapping(prepared["preparation"], name="execution preparation")
    validation = _mapping(preparation["validation"], name="execution parent validation")
    expected_parent_validation = canonical_sha256(validation)
    _exact_str(
        session["parent_public_validation_sha256"],
        name="execution parent validation hash",
        expected=expected_parent_validation,
    )
    _pre_fit_ref, pre_fit_raw = _verify_reference(
        v1_root,
        session["pre_fit_validation"],
        name="execution pre-fit validation",
        expected_path="runtime/pre_fit_validation.json",
    )
    pre_fit = _mapping(
        strict_json_bytes(pre_fit_raw, name="execution pre-fit validation"),
        name="execution pre-fit validation",
    )
    expected_pre_fit_sha = canonical_sha256(pre_fit)
    _exact_str(
        session["pre_fit_validation_receipt_sha256"],
        name="execution pre-fit semantic hash",
        expected=expected_pre_fit_sha,
    )
    _exact_int(session["workers_per_gpu"], name="execution workers/GPU", expected=2)
    gpu_inventory = session["gpu_inventory"]
    if type(gpu_inventory) is not list or len(gpu_inventory) != 4:
        _fail("execution GPU inventory is not the exact four devices")
    normalized_gpus = [
        _validate_gpu_identity(raw, name=f"execution GPU {index}")
        for index, raw in enumerate(gpu_inventory)
    ]
    if [gpu["physical_index"] for gpu in normalized_gpus] != list(range(4)):
        _fail("execution GPU physical indices changed")
    if len({gpu["uuid"] for gpu in normalized_gpus}) != 4 or len(
        {gpu["pci_bus_id"].lower() for gpu in normalized_gpus}
    ) != 4:
        _fail("execution GPU UUID/PCI identities are not unique")
    if prepared["gpu_inventory"] != normalized_gpus:
        _fail("execution GPU inventory differs from prepared authority")
    _exact_str(
        session["task_index_sha256"],
        name="execution task-index hash",
        expected=task_index_ref["sha256"],
    )
    slots = session["slots"]
    if type(slots) is not list or len(slots) != 8:
        _fail("execution slot roster is not the exact eight slots")
    for slot_index, raw in enumerate(slots):
        slot = _mapping(raw, name=f"execution slot {slot_index}")
        _keys(
            slot,
            frozenset({"slot_index", "local_slot", "gpu", "task_ids", "assignment_sha256"}),
            name=f"execution slot {slot_index}",
        )
        _exact_int(slot["slot_index"], name="execution slot index", expected=slot_index)
        _exact_int(slot["local_slot"], name="execution local slot", expected=slot_index % 2)
        if slot["gpu"] != normalized_gpus[slot_index // 2]:
            _fail("execution slot GPU assignment changed")
        expected_task_ids = [task["task_id"] for task in tasks[slot_index::8]]
        if slot["task_ids"] != expected_task_ids or len(expected_task_ids) != 42:
            _fail("execution slot task assignment changed")
        expected_assignment = canonical_sha256(
            {
                "slot_index": slot_index,
                "local_slot": slot_index % 2,
                "gpu": normalized_gpus[slot_index // 2],
                "task_ids": expected_task_ids,
            }
        )
        _exact_str(
            slot["assignment_sha256"],
            name="execution slot assignment hash",
            expected=expected_assignment,
        )
    expected_task_assignment = canonical_sha256(slots)
    _exact_str(
        session["task_assignment_sha256"],
        name="execution task assignment hash",
        expected=expected_task_assignment,
    )
    authority = _mapping(
        preparation["parent_git_authority"], name="execution parent Git authority"
    )
    expected_session_id = canonical_sha256(
        {
            "contract": V1_CONTRACT,
            "config_sha256": V1_CONFIG_SHA256,
            "prepared_manifest": prepared_ref,
            "model_contract_git_head": model_head,
            "parent_git_authority_sha256": canonical_sha256(authority),
            "parent_public_validation_sha256": expected_parent_validation,
            "model_source_set_sha256": source_closure["set_sha256"],
            "prepared_runtime_sha256": expected_runtime_sha,
            "pre_fit_validation_receipt_sha256": expected_pre_fit_sha,
            "task_index_sha256": task_index_ref["sha256"],
            "gpu_inventory_sha256": canonical_sha256(normalized_gpus),
            "workers_per_gpu": 2,
            "task_assignment_sha256": expected_task_assignment,
        }
    )
    _exact_str(
        session["execution_session_id"],
        name="execution session semantic ID",
        expected=expected_session_id,
    )
    _exact_bool(session["immutable"], name="execution session immutable", expected=True)
    _payload_hash(session, key="session_payload_sha256", name="execution session")
    return dict(session)


def _validated_execution_session_reference(
    v1_root: Path | _PinnedRoot, session: Mapping[str, Any]
) -> dict[str, Any]:
    session_ref, session_raw = _raw_reference(
        v1_root, "runtime/execution_session.json", name="authenticated execution session"
    )
    if strict_json_bytes(session_raw, name="authenticated execution session") != session:
        _fail("execution session raw reference differs from authenticated authority")
    return session_ref


def _validate_prelaunch_session_recovery_authority(
    v1_root: Path | _PinnedRoot,
    prepared: Mapping[str, Any],
    session: Mapping[str, Any],
    session_reference: Mapping[str, Any],
    *,
    present: bool,
) -> dict[str, Any] | None:
    """Recompute the official immutable pre-launch recovery receipt exactly."""

    if not present:
        return None
    recovery_ref, recovery_raw = _raw_reference(
        v1_root,
        "runtime/prelaunch_session_recovery.json",
        name="pre-launch session recovery authority",
    )
    recovery = _mapping(
        strict_json_bytes(recovery_raw, name="pre-launch session recovery authority"),
        name="pre-launch session recovery authority",
    )
    recovery_keys = frozenset(
        {
            "contract",
            "receipt_kind",
            "config_sha256",
            "prelaunch_recovery_contract_sha256",
            "execution_session_id",
            "execution_session",
            "session_payload_sha256",
            "pre_fit_validation",
            "pre_fit_validation_receipt_sha256",
            "task_assignment_sha256",
            "empty_evidence_at_exclusive_creation",
            "local_scientific_attempt_evidence_observed_at_exclusive_creation",
            "authorized_action_scope",
            "forbidden_action_scopes",
            "recovery_receipt_is_attempt_authority",
            "authorizes_scientific_refit_under_declared_non_total_erasure_threat_model",
            "no_refit_guarantee_threat_model",
            "external_append_only_or_signature_authority_present",
            "total_erasure_prior_execution_provable",
            "total_erasure_limitation",
            "receipt_payload_sha256",
        }
    )
    _keys(recovery, recovery_keys, name="pre-launch session recovery authority")
    pre_fit_ref, _pre_fit_raw = _raw_reference(
        v1_root, "runtime/pre_fit_validation.json", name="recovery pre-fit validation"
    )
    prepared_execution = _mapping(
        prepared["execution"], name="pre-launch recovery prepared execution"
    )
    frozen_contract = _mapping(
        prepared_execution.get("prelaunch_session_recovery"),
        name="pre-launch recovery frozen contract",
    )
    empty_evidence = _mapping(
        recovery["empty_evidence_at_exclusive_creation"],
        name="pre-launch recovery empty evidence",
    )
    empty_keys = frozenset(
        {
            "execution_launch_directories",
            "worker_ready_directories",
            "attempt_rosters",
            "attempt_ledgers",
            "task_phase_directories",
            "result_directories",
            "quarantine_events",
        }
    )
    _keys(empty_evidence, empty_keys, name="pre-launch recovery empty evidence")
    for key in empty_keys:
        _exact_int(
            empty_evidence[key], name=f"pre-launch recovery empty evidence {key}", expected=0
        )
    for key in (
        "local_scientific_attempt_evidence_observed_at_exclusive_creation",
        "recovery_receipt_is_attempt_authority",
        "authorizes_scientific_refit_under_declared_non_total_erasure_threat_model",
        "external_append_only_or_signature_authority_present",
        "total_erasure_prior_execution_provable",
    ):
        _exact_bool(recovery[key], name=f"pre-launch recovery {key}", expected=False)
    expected = {
        "contract": V1_CONTRACT,
        "receipt_kind": "authenticated_session_only_prelaunch_recovery",
        "config_sha256": prepared["config_sha256"],
        "prelaunch_recovery_contract_sha256": canonical_sha256(frozen_contract),
        "execution_session_id": session["execution_session_id"],
        "execution_session": session_reference,
        "session_payload_sha256": session["session_payload_sha256"],
        "pre_fit_validation": pre_fit_ref,
        "pre_fit_validation_receipt_sha256": session[
            "pre_fit_validation_receipt_sha256"
        ],
        "task_assignment_sha256": session["task_assignment_sha256"],
        "empty_evidence_at_exclusive_creation": {key: 0 for key in empty_keys},
        "local_scientific_attempt_evidence_observed_at_exclusive_creation": False,
        "authorized_action_scope": PRELAUNCH_AUTHORIZED_ACTION_SCOPE,
        "forbidden_action_scopes": list(PRELAUNCH_FORBIDDEN_ACTION_SCOPES),
        "recovery_receipt_is_attempt_authority": False,
        "authorizes_scientific_refit_under_declared_non_total_erasure_threat_model": False,
        "no_refit_guarantee_threat_model": NO_REFIT_GUARANTEE_THREAT_MODEL,
        "external_append_only_or_signature_authority_present": False,
        "total_erasure_prior_execution_provable": False,
        "total_erasure_limitation": TOTAL_ERASURE_LIMITATION,
    }
    expected["receipt_payload_sha256"] = canonical_sha256(expected)
    if recovery != expected:
        _fail("pre-launch session recovery receipt differs from official semantics")
    _payload_hash(
        recovery,
        key="receipt_payload_sha256",
        name="pre-launch session recovery authority",
    )
    return {
        "reference": recovery_ref,
        "semantic_sha256": canonical_sha256(recovery),
        "receipt": dict(recovery),
    }


def _validate_launch_completed_authority(
    v1_root: Path | _PinnedRoot,
    session: Mapping[str, Any],
    launch: Mapping[str, Any],
    launch_ref: Mapping[str, Any],
    controller_ref: Mapping[str, Any],
    controller: Mapping[str, Any],
) -> dict[str, Any]:
    launch_id = launch["launch_id"]
    path = f"runtime/execution_launches/launch_{launch_id}/launch.completed.json"
    _completed_ref, raw = _raw_reference(v1_root, path, name="execution launch completion")
    completed = _mapping(
        strict_json_bytes(raw, name="execution launch completion"),
        name="execution launch completion",
    )
    _keys(
        completed,
        frozenset(
            {
                "contract",
                "execution_session_id",
                "launch_id",
                "launch_started",
                "launch_payload_sha256",
                "launch_ordinal",
                "resume",
                "slot_count",
                "task_count",
                "task_assignment_sha256",
                "controller_source_sha256",
                "model_source_set_sha256",
                "parent_public_validation_sha256",
                "pre_fit_validation_receipt_sha256",
                "controller_ready_observations",
                "controller_observations_sha256",
                "ready_receipts",
                "all_tasks_authenticated_complete",
                "launch_completed_sha256",
            }
        ),
        name="execution launch completion",
    )
    expected = {
        "contract": V1_CONTRACT,
        "execution_session_id": session["execution_session_id"],
        "launch_id": launch_id,
        "launch_started": launch_ref,
        "launch_payload_sha256": launch["launch_payload_sha256"],
        "launch_ordinal": launch["launch_ordinal"],
        "resume": launch["resume"],
        "slot_count": 8,
        "task_count": V1_TASK_COUNT,
        "task_assignment_sha256": session["task_assignment_sha256"],
        "controller_source_sha256": session["controller_source_sha256"],
        "model_source_set_sha256": session["model_source_set_sha256"],
        "parent_public_validation_sha256": session[
            "parent_public_validation_sha256"
        ],
        "pre_fit_validation_receipt_sha256": session[
            "pre_fit_validation_receipt_sha256"
        ],
        "controller_ready_observations": controller_ref,
        "controller_observations_sha256": controller[
            "controller_observations_sha256"
        ],
        "all_tasks_authenticated_complete": True,
    }
    for key, expected_value in expected.items():
        if completed[key] != expected_value:
            _fail(f"execution launch completion changed: {key}")
    expected_ready = [
        observation["ready_receipt"] for observation in controller["observations"]
    ]
    if completed["ready_receipts"] != expected_ready:
        _fail("execution launch completion worker-ready reference order changed")
    _payload_hash(
        completed, key="launch_completed_sha256", name="execution launch completion"
    )
    return dict(completed)


def _validate_worker_ready_evidence_sets(
    v1_root: Path | _PinnedRoot,
    session: Mapping[str, Any],
    launch_entries: Sequence[
        tuple[int, dict[str, Any], dict[str, Any], set[str], dict[str, Any] | None]
    ],
    controller_launch_ids: set[str],
) -> None:
    """Accept authenticated partial pre-controller worker-ready evidence."""

    launch_authorities = {
        launch["launch_id"]: (launch, launch_reference)
        for _ordinal, launch, launch_reference, _members, _controller in launch_entries
    }
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        descriptor = (
            os.dup(v1_root.descriptor)
            if isinstance(v1_root, _PinnedRoot)
            else os.open(v1_root, flags)
        )
        descriptors.append(descriptor)
        for component in ("runtime", "worker_ready"):
            descriptor = os.open(component, flags, dir_fd=descriptor)
            descriptors.append(descriptor)
        ready_directories = set(os.listdir(descriptor))
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError(
            "cannot inspect worker-ready evidence root"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
    _verify_exact_directory_members(
        v1_root,
        "runtime/worker_ready",
        expected_files=set(),
        expected_directories=ready_directories,
        name="worker-ready evidence root",
    )
    observed_launch_ids: set[str] = set()
    for directory in ready_directories:
        match = re.fullmatch(r"launch_([0-9a-f]{64})", directory)
        if match is None or match.group(1) not in launch_authorities:
            _fail("worker-ready receipt has no authenticated launch")
        launch_id = match.group(1)
        observed_launch_ids.add(launch_id)
        relative = f"runtime/worker_ready/{directory}"
        member_descriptors: list[int] = []
        try:
            descriptor = (
                os.dup(v1_root.descriptor)
                if isinstance(v1_root, _PinnedRoot)
                else os.open(v1_root, flags)
            )
            member_descriptors.append(descriptor)
            for component in relative.split("/"):
                descriptor = os.open(component, flags, dir_fd=descriptor)
                member_descriptors.append(descriptor)
            members = set(os.listdir(descriptor))
        except OSError as exc:
            raise OpenAxesNativeV2ParentReceiptError(
                "cannot inspect worker-ready launch directory"
            ) from exc
        finally:
            for descriptor in reversed(member_descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if not members:
            _fail("empty worker-ready launch directory is not evidence")
        slots: set[int] = set()
        for member in members:
            slot_match = re.fullmatch(r"slot_([0-9]{4})\.json", member)
            if slot_match is None:
                _fail("worker-ready receipt member set changed")
            slot_index = int(slot_match.group(1))
            if slot_index in slots or not 0 <= slot_index < len(session["slots"]):
                _fail("worker-ready slot identity changed")
            slots.add(slot_index)
        _verify_exact_directory_members(
            v1_root,
            relative,
            expected_files=members,
            expected_directories=set(),
            name=f"worker-ready launch {launch_id}",
        )
        launch, launch_reference = launch_authorities[launch_id]
        process_ids: set[int] = set()
        for slot_index in sorted(slots):
            ready_path = f"{relative}/slot_{slot_index:04d}.json"
            ready_reference, _ready_raw = _raw_reference(
                v1_root, ready_path, name=f"worker-ready evidence {launch_id}/{slot_index}"
            )
            ready = _validate_worker_ready_authority(
                v1_root,
                ready_reference,
                launch=launch,
                launch_reference=launch_reference,
                expected_slot_index=slot_index,
                name=f"worker-ready evidence {launch_id}/{slot_index}",
                session=session,
            )
            process_id = ready["child_process_id"]
            if process_id in process_ids:
                _fail("worker-ready child process identities are not unique")
            process_ids.add(process_id)
        if launch_id in controller_launch_ids and slots != set(range(8)):
            _fail("controller observations lack their exact eight worker-ready receipts")
    if not controller_launch_ids.issubset(observed_launch_ids):
        _fail("controller observations lack their worker-ready directory")


def _validate_complete_launch_authority(
    v1_root: Path | _PinnedRoot,
    session: Mapping[str, Any],
    session_reference: Mapping[str, Any],
    recovery_authority: Mapping[str, Any] | None,
) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        descriptor = (
            os.dup(v1_root.descriptor)
            if isinstance(v1_root, _PinnedRoot)
            else os.open(v1_root, flags)
        )
        descriptors.append(descriptor)
        for component in ("runtime", "execution_launches"):
            descriptor = os.open(component, flags, dir_fd=descriptor)
            descriptors.append(descriptor)
        launch_directories = set(os.listdir(descriptor))
    except OSError as exc:
        raise OpenAxesNativeV2ParentReceiptError(
            "cannot inspect execution launch directory"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
    if not launch_directories:
        _fail("complete execution has no launch authority")
    launch_entries: list[
        tuple[int, dict[str, Any], dict[str, Any], set[str], dict[str, Any] | None]
    ] = []
    controller_launch_ids: set[str] = set()
    for directory in launch_directories:
        match = re.fullmatch(r"launch_([0-9a-f]{64})", directory)
        if match is None:
            _fail("execution launch directory name is unsafe")
        launch_id = match.group(1)
        relative_directory = f"runtime/execution_launches/{directory}"
        member_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        member_descriptors: list[int] = []
        try:
            descriptor = (
                os.dup(v1_root.descriptor)
                if isinstance(v1_root, _PinnedRoot)
                else os.open(v1_root, member_flags)
            )
            member_descriptors.append(descriptor)
            for component in relative_directory.split("/"):
                descriptor = os.open(component, member_flags, dir_fd=descriptor)
                member_descriptors.append(descriptor)
            members = set(os.listdir(descriptor))
        except OSError as exc:
            raise OpenAxesNativeV2ParentReceiptError(
                "cannot inspect one execution launch"
            ) from exc
        finally:
            for descriptor in reversed(member_descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        allowed_members = (
            {"launch.started.json"},
            {"launch.started.json", "controller_ready_observations.json"},
            {
                "launch.started.json",
                "controller_ready_observations.json",
                "launch.completed.json",
            },
        )
        if members not in allowed_members:
            _fail("execution launch member set changed")
        _verify_exact_directory_members(
            v1_root,
            relative_directory,
            expected_files=members,
            expected_directories=set(),
            name=f"execution launch {launch_id}",
        )
        launch_path = f"{relative_directory}/launch.started.json"
        launch_ref, launch_raw = _raw_reference(
            v1_root, launch_path, name=f"execution launch start {launch_id}"
        )
        untrusted = _mapping(
            strict_json_bytes(launch_raw, name=f"execution launch start {launch_id}"),
            name=f"execution launch start {launch_id}",
        )
        partial_binding = {
            "execution_session_id": session["execution_session_id"],
            "launch_id": launch_id,
            "launch_started": launch_ref,
            "launch_payload_sha256": untrusted.get("launch_payload_sha256"),
            "task_assignment_sha256": session["task_assignment_sha256"],
            "controller_source_sha256": session["controller_source_sha256"],
        }
        ordinal, launch = _validate_launch_started_authority(
            v1_root,
            partial_binding,
            session=session,
            session_reference=session_reference,
            recovery_authority=recovery_authority,
            name=f"execution launch {launch_id}",
        )
        if (
            launch["session"] != session_reference
            or launch["model_source_set_sha256"] != session["model_source_set_sha256"]
            or launch["parent_public_validation_sha256"]
            != session["parent_public_validation_sha256"]
            or launch["pre_fit_validation_receipt_sha256"]
            != session["pre_fit_validation_receipt_sha256"]
        ):
            _fail("execution launch differs from immutable execution session")
        controller: dict[str, Any] | None = None
        if "controller_ready_observations.json" in members:
            controller_launch_ids.add(launch_id)
            controller_path = f"{relative_directory}/controller_ready_observations.json"
            controller_ref, controller_raw = _raw_reference(
                v1_root, controller_path, name=f"execution controller {launch_id}"
            )
            controller_untrusted = _mapping(
                strict_json_bytes(controller_raw, name=f"execution controller {launch_id}"),
                name=f"execution controller {launch_id}",
            )
            slot = session["slots"][0]
            full_binding = {
                "execution_session_id": session["execution_session_id"],
                "launch_id": launch_id,
                "launch_started": launch_ref,
                "launch_payload_sha256": launch["launch_payload_sha256"],
                "transport_slot_index": 0,
                "local_slot": slot["local_slot"],
                "gpu": slot["gpu"],
                "controller_source_sha256": session["controller_source_sha256"],
                "task_assignment_sha256": session["task_assignment_sha256"],
                "slot_assignment_sha256": slot["assignment_sha256"],
                "worker_ready": controller_untrusted["observations"][0]["ready_receipt"],
                "controller_ready_observations": controller_ref,
                "controller_observations_sha256": controller_untrusted[
                    "controller_observations_sha256"
                ],
                "worker_identity_source": "fixed_process_transport_and_ready_receipt",
                "scientific_fit_attempt": 1,
            }
            _validated_binding, observed_ordinal = _validate_binding_authorities(
                v1_root,
                full_binding,
                session=session,
                session_reference=session_reference,
                recovery_authority=recovery_authority,
                name=f"execution launch {launch_id} authority",
            )
            if observed_ordinal != ordinal:
                _fail("execution launch ordinal changed across authorities")
            controller = _validate_controller_authority(
                v1_root,
                full_binding,
                launch,
                name=f"execution launch {launch_id} authority",
                session=session,
            )
            ready_files = {f"slot_{slot_index:04d}.json" for slot_index in range(8)}
            _verify_exact_directory_members(
                v1_root,
                f"runtime/worker_ready/launch_{launch_id}",
                expected_files=ready_files,
                expected_directories=set(),
                name=f"worker-ready launch {launch_id}",
            )
        launch_entries.append((ordinal, launch, launch_ref, members, controller))
    launch_entries.sort(key=lambda item: item[0])
    if [item[0] for item in launch_entries] != list(range(len(launch_entries))):
        _fail("execution launch ordinals are not contiguous")
    for index, (_ordinal, launch, _ref_value, _members, _controller) in enumerate(
        launch_entries
    ):
        expected_previous = None if index == 0 else launch_entries[index - 1][2]
        if launch["previous_launch"] != expected_previous:
            _fail("execution launch predecessor raw reference changed")
    _validate_worker_ready_evidence_sets(
        v1_root,
        session,
        launch_entries,
        controller_launch_ids,
    )
    _ordinal, latest, latest_ref, latest_members, latest_controller = launch_entries[-1]
    if "launch.completed.json" not in latest_members or latest_controller is None:
        _fail("latest execution launch is not terminal-complete")
    latest_controller_ref = _reference(
        f"runtime/execution_launches/launch_{latest['launch_id']}/controller_ready_observations.json",
        _read_nofollow(
            v1_root,
            f"runtime/execution_launches/launch_{latest['launch_id']}/controller_ready_observations.json",
            name="latest execution controller",
        ),
    )
    _validate_launch_completed_authority(
        v1_root,
        session,
        latest,
        latest_ref,
        latest_controller_ref,
        latest_controller,
    )


def _validate_full_signed_source_replay(
    value: Any,
    *,
    prepared: Mapping[str, Any],
    prepared_ref: Mapping[str, Any],
    development: Mapping[str, Any],
    development_ref: Mapping[str, Any],
) -> dict[str, Any]:
    replay = _mapping(value, name="v1 full signed-source replay")
    keys = frozenset(
        {
            "contract",
            "receipt_kind",
            "config_sha256",
            "prepared_manifest",
            "development_manifest",
            "preparation_artifacts",
            "parent_public_validation_sha256",
            "parent_git_authority_sha256",
            "reconstruction_receipt_sha256",
            "scenario_shift_support_identity",
            "scenario_shift_support_identity_sha256",
            "signed_report_sha256",
            "validator_source_sha256",
            "validator_source_set_sha256",
            "full_signed_source_replay",
            "semantic_replay",
            "receipt_payload_sha256",
        }
    )
    _keys(replay, keys, name="v1 full signed-source replay")
    _exact_str(replay["contract"], name="replay contract", expected=V1_CONTRACT)
    _exact_str(
        replay["receipt_kind"],
        name="replay receipt kind",
        expected="immutable_full_signed_source_semantic_replay",
    )
    _exact_str(replay["config_sha256"], name="replay config", expected=V1_CONFIG_SHA256)
    if replay["prepared_manifest"] != prepared_ref or replay["development_manifest"] != development_ref:
        _fail("signed-source replay manifest references changed")
    preparation = _mapping(prepared["preparation"], name="replay preparation snapshot")
    preparation_artifacts = _mapping(
        preparation.get("artifacts"), name="prepared preparation artifacts"
    )
    _keys(
        preparation_artifacts,
        frozenset({"synthetic_test_only", "source", "config", "manifest", "parquet", "complete"}),
        name="prepared preparation artifacts",
    )
    replay_artifacts = _mapping(
        replay["preparation_artifacts"], name="replay preparation artifacts"
    )
    projection_keys = ("source", "config", "manifest", "parquet", "complete")
    _keys(replay_artifacts, frozenset(projection_keys), name="replay preparation artifacts")
    expected_artifacts = {key: preparation_artifacts[key] for key in projection_keys}
    if replay_artifacts != expected_artifacts:
        _fail("signed-source replay preparation-artifact binding changed")
    external_reference_keys = frozenset({"absolute_path", "bytes", "sha256"})
    for key in projection_keys:
        external = _mapping(replay_artifacts[key], name=f"replay preparation {key}")
        _keys(external, external_reference_keys, name=f"replay preparation {key}")
        absolute = _exact_str(
            external["absolute_path"], name=f"replay preparation {key} absolute path"
        )
        if not Path(absolute).is_absolute() or os.path.normpath(absolute) != absolute:
            _fail(f"replay preparation {key} absolute path is noncanonical")
        if _exact_int(external["bytes"], name=f"replay preparation {key} bytes") < 0:
            _fail(f"replay preparation {key} byte count is negative")
        _hex64(external["sha256"], name=f"replay preparation {key} hash")
    validation = _mapping(preparation.get("validation"), name="preparation public validation")
    authority = _mapping(preparation.get("parent_git_authority"), name="preparation Git authority")
    if replay["parent_public_validation_sha256"] != canonical_sha256(validation):
        _fail("signed-source replay public-validation hash changed")
    if replay["parent_git_authority_sha256"] != canonical_sha256(authority):
        _fail("signed-source replay Git-authority hash changed")
    _hex64(replay["reconstruction_receipt_sha256"], name="reconstruction receipt hash")
    support = replay["scenario_shift_support_identity"]
    if type(support) is not list or len(support) != len(SCENARIO_ORDER) * len(
        SIGNED_SHIFT_ORDER_MS
    ):
        _fail("scenario-shift support identity is not the exact 48-item list")
    expected_pairs = [
        (scenario_id, shift_ms)
        for scenario_id in SCENARIO_ORDER
        for shift_ms in SIGNED_SHIFT_ORDER_MS
    ]
    for index, (raw, (expected_scenario, expected_shift)) in enumerate(
        zip(support, expected_pairs, strict=True)
    ):
        item = _mapping(raw, name=f"scenario-shift support identity {index}")
        _keys(
            item,
            frozenset(
                {"scenario_id", "shift_ms", "base_uid_rows", "base_uid_sha256"}
            ),
            name=f"scenario-shift support identity {index}",
        )
        _exact_str(
            item["scenario_id"],
            name=f"scenario-shift support scenario {index}",
            expected=expected_scenario,
        )
        _exact_int(
            item["shift_ms"],
            name=f"scenario-shift support shift {index}",
            expected=expected_shift,
        )
        if _exact_int(
            item["base_uid_rows"], name=f"scenario-shift support rows {index}"
        ) < 1:
            _fail("scenario-shift support row count is not positive")
        _hex64(
            item["base_uid_sha256"],
            name=f"scenario-shift support UID hash {index}",
        )
    if replay["scenario_shift_support_identity_sha256"] != canonical_sha256(support):
        _fail("scenario-shift support identity hash changed")
    if replay["signed_report_sha256"] != canonical_sha256(
        development["alignment_sensitivity_report_only"]
    ):
        _fail("signed-source replay report hash changed")
    if replay["validator_source_sha256"] != V1_SOURCE_REFERENCE_ORACLE[
        "src/imu_vive_learning/openaxes_external_model_v1_experiment.py"
    ]["sha256"]:
        _fail("signed-source replay validator source changed")
    _exact_str(
        replay["validator_source_set_sha256"],
        name="replay validator source set",
        expected=V1_SOURCE_SET_SHA256,
    )
    _exact_bool(replay["full_signed_source_replay"], name="full replay marker", expected=True)
    _exact_bool(replay["semantic_replay"], name="semantic replay marker", expected=True)
    _payload_hash(replay, key="receipt_payload_sha256", name="full signed-source replay")
    return dict(replay)


def _validate_v1_envelopes(
    v1_root: Path | _PinnedRoot,
    preparation_root: Path | _PinnedRoot,
    project_root: Path | _PinnedRoot,
) -> dict[str, Any]:
    recovery_present = _verify_complete_root_and_runtime_sets(v1_root)
    prepared_ref, prepared_raw = _raw_reference(v1_root, "prepared_manifest.json", name="v1 prepared manifest")
    prepared = _mapping(strict_json_bytes(prepared_raw, name="v1 prepared manifest"), name="v1 prepared manifest")
    prepared_keys = frozenset(
        {
            "contract",
            "config_sha256",
            "stage",
            "synthetic_test_only",
            "preparation",
            "task_grid",
            "gpu_inventory",
            "execution",
            "source_closure",
            "repository",
            "runtime",
            "results_started",
            "challenge_route_present",
        }
    )
    _keys(prepared, prepared_keys, name="v1 prepared manifest")
    _exact_str(prepared["contract"], name="v1 prepared contract", expected=V1_CONTRACT)
    _exact_str(prepared["config_sha256"], name="v1 config", expected=V1_CONFIG_SHA256)
    _exact_str(prepared["stage"], name="v1 prepared stage", expected="prepared_before_external_model_results")
    _exact_bool(prepared["synthetic_test_only"], name="v1 synthetic marker", expected=False)
    _exact_bool(prepared["results_started"], name="v1 prepared results marker", expected=False)
    _exact_bool(prepared["challenge_route_present"], name="v1 challenge route", expected=False)
    source_closure = _mapping(prepared["source_closure"], name="v1 source closure")
    _keys(source_closure, frozenset({"files", "set_sha256"}), name="v1 source closure")
    files = _mapping(source_closure["files"], name="v1 source files")
    if set(files) != set(V1_SOURCE_CLOSURE) or len(files) != 9:
        _fail("v1 terminal source closure paths changed")
    normalized_v1_sources: dict[str, Any] = {}
    for relative in V1_SOURCE_CLOSURE:
        normalized_v1_sources[relative] = _validate_reference_shape(
            files[relative], name=f"v1 source file {relative}", expected_path=relative
        )
    if (
        normalized_v1_sources != V1_SOURCE_REFERENCE_ORACLE
        or canonical_sha256(normalized_v1_sources) != V1_SOURCE_SET_SHA256
        or source_closure["set_sha256"] != V1_SOURCE_SET_SHA256
    ):
        _fail("v1 terminal source closure identity changed")
    model_head = _extract_model_head(prepared)

    preparation_snapshot = _mapping(prepared["preparation"], name="v1 preparation snapshot")
    artifacts = _mapping(preparation_snapshot.get("artifacts"), name="v1 preparation artifacts")
    _keys(
        artifacts,
        frozenset({"synthetic_test_only", "source", "config", "manifest", "parquet", "complete"}),
        name="v1 preparation artifacts",
    )
    _exact_bool(artifacts.get("synthetic_test_only"), name="preparation synthetic marker", expected=False)
    preparation_path = (
        preparation_root.path if isinstance(preparation_root, _PinnedRoot) else preparation_root
    )
    project_path = project_root.path if isinstance(project_root, _PinnedRoot) else project_root
    external_artifacts: dict[str, dict[str, Any]] = {}
    for key, expected_absolute in (
        ("source", None),
        ("config", project_path / "configs/openaxes_external_v1.yaml"),
        ("manifest", preparation_path / "manifest.json"),
        ("parquet", preparation_path / "prepared.parquet"),
        ("complete", preparation_path / "COMPLETE.json"),
    ):
        external_artifacts[key], _external_raw = _verify_external_file_reference(
            artifacts[key],
            name=f"external preparation {key}",
            expected_absolute=expected_absolute,
        )
    source_path = Path(external_artifacts["source"]["absolute_path"])
    if source_path.suffix.lower() != ".zip":
        _fail("external preparation source is not the designated ZIP input")
    manifest_external = external_artifacts["manifest"]
    parquet_external = external_artifacts["parquet"]
    preparation_manifest_ref, preparation_manifest_raw = _raw_reference(
        preparation_root, "manifest.json", name="preparation manifest"
    )
    preparation_table_ref, _opaque_table_raw = _raw_reference(
        preparation_root, "prepared.parquet", name="opaque preparation table"
    )
    for external, observed, name in (
        (manifest_external, preparation_manifest_ref, "preparation manifest"),
        (parquet_external, preparation_table_ref, "preparation table"),
    ):
        if external["bytes"] != observed["bytes"] or external["sha256"] != observed["sha256"]:
            _fail(f"v1 external {name} reference changed")
    preparation_manifest = _mapping(
        strict_json_bytes(preparation_manifest_raw, name="preparation manifest"),
        name="preparation manifest",
    )
    _exact_str(preparation_manifest.get("contract"), name="preparation contract", expected=PREPARATION_CONTRACT)
    _exact_str(preparation_manifest.get("config_sha256"), name="preparation config", expected=PREPARATION_CONFIG_SHA256)
    prep_files = _mapping(preparation_manifest.get("files"), name="preparation file receipts")
    if set(prep_files) != {"prepared.parquet"}:
        _fail("preparation manifest table receipt set changed")
    table_receipt = _mapping(prep_files["prepared.parquet"], name="preparation table receipt")
    if set(table_receipt) != {"path", "bytes", "sha256", "rows"}:
        _fail("preparation table receipt schema changed")
    if any(table_receipt[key] != preparation_table_ref[key] for key in ("path", "bytes", "sha256")):
        _fail("preparation manifest table receipt changed")

    expected_tasks = _expected_tasks(preparation_manifest)
    task_grid = _mapping(prepared["task_grid"], name="v1 task grid")
    task_index_ref = _validate_reference_shape(task_grid.get("task_index"), name="v1 task index reference", expected_path="tasks/index.json")
    verified_task_ref, task_raw = _verify_reference(v1_root, task_index_ref, name="v1 task index", expected_path="tasks/index.json")
    tasks = _validate_task_index(strict_json_bytes(task_raw, name="v1 task index"), expected_tasks)
    execution_session = _validate_execution_session_authority(
        v1_root, prepared, prepared_ref, verified_task_ref, tasks
    )
    execution_session_ref = _validated_execution_session_reference(
        v1_root, execution_session
    )
    recovery_authority = _validate_prelaunch_session_recovery_authority(
        v1_root,
        prepared,
        execution_session,
        execution_session_ref,
        present=recovery_present,
    )
    _validate_complete_launch_authority(
        v1_root,
        execution_session,
        execution_session_ref,
        recovery_authority,
    )
    _prepared_marker_ref, prepared_marker_raw = _raw_reference(
        v1_root, "PREPARED.json", name="v1 PREPARED marker"
    )
    prepared_marker = _mapping(
        strict_json_bytes(prepared_marker_raw, name="v1 PREPARED marker"),
        name="v1 PREPARED marker",
    )
    _keys(
        prepared_marker,
        frozenset(
            {
                "contract",
                "config_sha256",
                "prepared_manifest",
                "task_index",
                "prepared_runtime_sha256",
                "synthetic_test_only",
                "model_contract_sources_committed_at_git_head",
                "downstream_results_blocked_by_uncommitted_contract",
                "immutable",
            }
        ),
        name="v1 PREPARED marker",
    )
    _exact_str(prepared_marker["contract"], name="PREPARED contract", expected=V1_CONTRACT)
    _exact_str(prepared_marker["config_sha256"], name="PREPARED config", expected=V1_CONFIG_SHA256)
    if prepared_marker["prepared_manifest"] != prepared_ref or prepared_marker["task_index"] != verified_task_ref:
        _fail("PREPARED marker manifest/task-index raw references changed")
    runtime_receipt = _mapping(prepared["runtime"], name="prepared runtime receipt")
    if _hex64(
        prepared_marker["prepared_runtime_sha256"], name="PREPARED runtime hash"
    ) != canonical_sha256(runtime_receipt):
        _fail("PREPARED runtime hash differs from canonical prepared runtime receipt")
    for key, expected in (
        ("synthetic_test_only", False),
        ("model_contract_sources_committed_at_git_head", True),
        ("downstream_results_blocked_by_uncommitted_contract", False),
        ("immutable", True),
    ):
        _exact_bool(prepared_marker[key], name=f"PREPARED {key}", expected=expected)

    development_ref, development_raw = _raw_reference(v1_root, "development_manifest.json", name="v1 development manifest")
    development = _mapping(strict_json_bytes(development_raw, name="v1 development manifest"), name="v1 development manifest")
    development_keys = frozenset(
        {
            "contract",
            "config_sha256",
            "stage",
            "synthetic_test_only",
            "prepared_manifest",
            "scientific_fit_count",
            "outer_evaluation_count",
            "result_index",
            "paired_controls",
            "zero_shift",
            "alignment_sensitivity_report_only",
            "claim_boundary",
        }
    )
    _keys(development, development_keys, name="v1 development manifest")
    _exact_str(development["contract"], name="development contract", expected=V1_CONTRACT)
    _exact_str(development["config_sha256"], name="development config", expected=V1_CONFIG_SHA256)
    _exact_str(development["stage"], name="development stage", expected=V1_TERMINAL_STAGE)
    _exact_bool(development["synthetic_test_only"], name="development synthetic marker", expected=False)
    _exact_int(development["scientific_fit_count"], name="development fits", expected=336)
    _exact_int(development["outer_evaluation_count"], name="development evaluations", expected=336)
    _verify_reference(v1_root, development["prepared_manifest"], name="development prepared reference", expected_path="prepared_manifest.json")
    claim = _mapping(development["claim_boundary"], name="development claim boundary")
    _exact_bool(claim.get("challenge_access"), name="development challenge access", expected=False)

    result_index = _mapping(development["result_index"], name="development result index")
    task_ids = {task["task_id"] for task in tasks}
    if set(result_index) != task_ids or len(result_index) != 336:
        _fail("development result index does not bind all 336 task identities")
    _verify_complete_task_directory_sets(v1_root, task_ids)
    task_authority: dict[str, Any] = {}
    for task in tasks:
        masked_primary = (
            task["method_id"] == "masked_erm"
            and task["family"] == "scenario_loso"
            and task["role"] == "primary"
        )
        task_authority[task["task_id"]] = _validate_task_result_lineage(
            v1_root,
            task,
            result_index[task["task_id"]],
            session=execution_session,
            session_reference=execution_session_ref,
            recovery_authority=recovery_authority,
            validate_metric=masked_primary,
        )
    if sum(item["metric"] is not None for item in task_authority.values()) != MASKED_PRIMARY_TASK_COUNT:
        _fail("authoritative masked_erm primary metric set is not exactly 24 tasks")

    complete_ref, complete_raw = _raw_reference(v1_root, "COMPLETE.json", name="v1 complete marker")
    complete = _mapping(strict_json_bytes(complete_raw, name="v1 complete marker"), name="v1 complete marker")
    complete_keys = frozenset(
        {
            "contract",
            "config_sha256",
            "prepared_manifest",
            "development_manifest",
            "scientific_fits",
            "outer_evaluations",
            "selected_method_id",
            "selection_sha256",
            "zero_shift_sha256",
            "directional_gate_passes",
            "directional_gate_sha256",
            "alignment_sensitivity_sha256",
            "full_signed_source_replay",
            "full_signed_source_replay_sha256",
            "full_signed_source_replay_performed",
            "production_validation_requires_signed_source_replay",
            "stress_or_sensitivity_used_for_selection",
            "challenge_access_performed",
            "immutable",
        }
    )
    _keys(complete, complete_keys, name="v1 complete marker")
    _exact_str(complete["contract"], name="complete contract", expected=V1_CONTRACT)
    _exact_str(complete["config_sha256"], name="complete config", expected=V1_CONFIG_SHA256)
    _exact_int(complete["scientific_fits"], name="complete fits", expected=336)
    _exact_int(complete["outer_evaluations"], name="complete evals", expected=336)
    _verify_reference(v1_root, complete["prepared_manifest"], name="complete prepared reference", expected_path="prepared_manifest.json")
    _verify_reference(v1_root, complete["development_manifest"], name="complete development reference", expected_path="development_manifest.json")
    if _hex64(complete["zero_shift_sha256"], name="complete zero-shift hash") != canonical_sha256(development["zero_shift"]):
        _fail("complete marker zero-shift hash changed")
    _hex64(complete["selection_sha256"], name="complete selection hash")
    _hex64(complete["directional_gate_sha256"], name="complete gate hash")
    _hex64(complete["alignment_sensitivity_sha256"], name="complete sensitivity hash")
    for key, expected in (
        ("full_signed_source_replay_performed", True),
        ("production_validation_requires_signed_source_replay", True),
        ("stress_or_sensitivity_used_for_selection", False),
        ("challenge_access_performed", False),
        ("immutable", True),
    ):
        _exact_bool(complete[key], name=f"complete {key}", expected=expected)
    _full_ref, full_replay_raw = _verify_reference(
        v1_root,
        complete["full_signed_source_replay"],
        name="v1 signed-source replay receipt",
        expected_path="runtime/full_signed_source_replay.json",
    )
    full_replay = _mapping(
        strict_json_bytes(full_replay_raw, name="v1 signed-source replay receipt"),
        name="v1 signed-source replay receipt",
    )
    _validate_full_signed_source_replay(
        full_replay,
        prepared=prepared,
        prepared_ref=prepared_ref,
        development=development,
        development_ref=development_ref,
    )
    zero = _mapping(development["zero_shift"], name="development zero shift")
    selection = _mapping(zero.get("selection"), name="development selection")
    directional_gate = _mapping(zero.get("directional_gate"), name="development directional gate")
    if complete["selected_method_id"] != selection.get("selected_method_id"):
        _fail("complete selected method differs from sealed development selection")
    if complete["selection_sha256"] != canonical_sha256(selection):
        _fail("complete selection hash differs from sealed development selection")
    if complete["directional_gate_passes"] is not directional_gate.get("passes"):
        _fail("complete directional-gate outcome differs from development")
    if complete["directional_gate_sha256"] != canonical_sha256(directional_gate):
        _fail("complete directional-gate hash differs from development")
    if complete["alignment_sensitivity_sha256"] != canonical_sha256(
        development["alignment_sensitivity_report_only"]
    ):
        _fail("complete sensitivity hash differs from development")
    if complete["full_signed_source_replay_sha256"] != canonical_sha256(full_replay):
        _fail("complete signed-source replay semantic hash changed")

    return {
        "prepared": prepared,
        "prepared_ref": prepared_ref,
        "development": development,
        "development_ref": development_ref,
        "complete": complete,
        "complete_ref": complete_ref,
        "preparation_manifest": preparation_manifest,
        "preparation_manifest_ref": preparation_manifest_ref,
        "preparation_table_ref": preparation_table_ref,
        "task_index_ref": verified_task_ref,
        "tasks": tasks,
        "task_authority": task_authority,
        "execution_session": execution_session,
        "model_head": model_head,
    }


def _masked_fold_records(development: Mapping[str, Any], fold_order: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    zero = _mapping(development["zero_shift"], name="v1 zero-shift development")
    zero_keys = frozenset({"selection", "method_scores", "directional_gate", "fold_records", "stress_report_only", "counts"})
    _keys(zero, zero_keys, name="v1 zero-shift development")
    records = zero["fold_records"]
    if type(records) is not list or len(records) != 112:
        _fail("v1 zero-shift fold-record grid is not exactly 8 by 14")
    selected: dict[str, Mapping[str, Any]] = {}
    for value in records:
        record = _mapping(value, name="v1 fold record")
        if record.get("method_id") != "masked_erm" or record.get("fold_id") not in fold_order:
            continue
        expected_keys = frozenset(
            {"method_id", "fold_id", "family", "role", "test_scenarios", "seed_ensemble_size", "outage", "full_diagnostic"}
        )
        _keys(record, expected_keys, name="masked_erm primary fold record")
        _exact_str(record["family"], name="masked fold family", expected="scenario_loso")
        _exact_str(record["role"], name="masked fold role", expected="primary")
        _exact_int(record["seed_ensemble_size"], name="masked ensemble size", expected=3)
        if record["fold_id"] in selected:
            _fail("duplicate masked_erm primary fold record")
        selected[record["fold_id"]] = record
    if list(selected) != list(fold_order):
        # v1 aggregation sorts records; ordering in the manifest is not authority.
        if set(selected) != set(fold_order):
            _fail("masked_erm primary fold record set changed")
    return selected


def _ensemble_contexts(
    records: Mapping[str, Mapping[str, Any]], fold_order: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    outage_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    context_input_keys = frozenset({"scenario_id", "view_id", "rows", "rmse3d_m", "p95_distance_m", "identity_sha256"})
    aggregate_keys = frozenset(
        {"context_count", "A_mean_context_rmse3d_m", "WS_worst_scenario_mean_rmse3d_m", "WM_worst_mask_mean_rmse3d_m", "robust_J_m", "contexts"}
    )
    for fold_id in fold_order:
        record = records[fold_id]
        scenarios = record["test_scenarios"]
        if type(scenarios) is not list or len(scenarios) != 1 or type(scenarios[0]) is not str:
            _fail("primary LOSO fold does not hold exactly one scenario")
        scenario = scenarios[0]
        outage = _mapping(record["outage"], name="masked outage aggregate")
        full = _mapping(record["full_diagnostic"], name="masked full aggregate")
        _keys(outage, aggregate_keys, name="masked outage aggregate")
        _keys(full, aggregate_keys, name="masked full aggregate")
        _exact_int(outage["context_count"], name="masked outage context count", expected=4)
        _exact_int(full["context_count"], name="masked full context count", expected=1)
        outage_by_view: dict[str, Mapping[str, Any]] = {}
        for raw in outage["contexts"]:
            context = _mapping(raw, name="masked outage context")
            _keys(context, context_input_keys, name="masked outage context")
            outage_by_view[_exact_str(context["view_id"], name="masked outage view")] = context
        if set(outage_by_view) != set(OUTAGE_VIEW_ORDER):
            _fail("masked outage view set changed")
        full_contexts = full["contexts"]
        if type(full_contexts) is not list or len(full_contexts) != 1:
            _fail("masked full context set changed")
        full_context = _mapping(full_contexts[0], name="masked full context")
        _keys(full_context, context_input_keys, name="masked full context")
        if full_context["view_id"] != "full":
            _fail("masked full context view changed")
        for view_id in OUTAGE_VIEW_ORDER:
            context = outage_by_view[view_id]
            if context["scenario_id"] != scenario:
                _fail("masked outage scenario identity changed")
            row = {
                "fold_id": fold_id,
                "scenario_id": scenario,
                "view_id": view_id,
                "rows": _exact_int(context["rows"], name="ensemble context rows"),
                "identity_sha256": _hex64(context["identity_sha256"], name="ensemble context identity"),
                "rmse3d_m": _finite_nonnegative(context["rmse3d_m"], name="ensemble context RMSE"),
            }
            outage_rows.append(row)
            identity_rows.append({key: row[key] for key in ("fold_id", "scenario_id", "view_id", "rows", "identity_sha256")})
        if full_context["scenario_id"] != scenario:
            _fail("masked full scenario identity changed")
        full_rows.append(
            {
                "fold_id": fold_id,
                "scenario_id": scenario,
                "view_id": "full",
                "rows": _exact_int(full_context["rows"], name="full context rows"),
                "identity_sha256": _hex64(full_context["identity_sha256"], name="full context identity"),
                "rmse3d_m": _finite_nonnegative(full_context["rmse3d_m"], name="full context RMSE"),
            }
        )
    return outage_rows, full_rows, identity_rows


def _build_comparator_bundle(
    v1_root: Path | _PinnedRoot, state: Mapping[str, Any]
) -> dict[str, Any]:
    tasks = state["tasks"]
    primary_tasks = [
        task for task in tasks if task["method_id"] == "masked_erm" and task["family"] == "scenario_loso" and task["role"] == "primary"
    ]
    if len(primary_tasks) != 24:
        _fail("masked_erm primary task set is not exactly 24")
    preparation_folds = state["preparation_manifest"]["folds"]["items"]
    fold_order = [fold["fold_id"] for fold in preparation_folds if fold["family"] == "scenario_loso"]
    records = _masked_fold_records(state["development"], fold_order)
    outage_rows, full_rows, identity_rows = _ensemble_contexts(records, fold_order)
    identity_lookup = {(row["fold_id"], row["view_id"]): row for row in identity_rows}

    initial_by_seed: dict[int, set[str]] = {seed: set() for seed in SEED_ORDER}
    training_refs: list[dict[str, Any]] = []
    metric_refs: list[dict[str, Any]] = []
    metrics: dict[tuple[int, str], Mapping[str, Any]] = {}
    task_lookup = {(task["fold_id"], task["seed"]): task for task in primary_tasks}
    for fold_id in fold_order:
        for seed in SEED_ORDER:
            task = task_lookup.get((fold_id, seed))
            if task is None:
                _fail("masked_erm primary fold/seed grid is incomplete")
            authority = state["task_authority"][task["task_id"]]
            training = authority["training"]
            initial_by_seed[seed].add(training["initial_state_sha256"])
            training_refs.append(authority["training_reference"])
            metric = authority["metric"]
            if metric is None:
                _fail("masked_erm primary task lacks authoritative sealed metric")
            metrics[(seed, fold_id)] = metric
            metric_refs.append(authority["metric_reference"])
    if any(len(values) != 1 for values in initial_by_seed.values()):
        _fail("masked_erm initial state differs across primary folds for a seed")

    per_seed_rows: list[dict[str, Any]] = []
    for seed in SEED_ORDER:
        for fold_id in fold_order:
            metric = metrics[(seed, fold_id)]
            views = {view["view_id"]: view for view in metric["metrics"]["views"]}
            for view_id in OUTAGE_VIEW_ORDER:
                identity = identity_lookup[(fold_id, view_id)]
                metric_view = views[view_id]
                if metric_view["rows"] != identity["rows"]:
                    _fail("per-seed metric rows differ from ensemble context identity")
                per_seed_rows.append(
                    {
                        "seed": seed,
                        **identity,
                        "rmse3d_m": metric_view["rmse3d_m"],
                    }
                )
    a_value, ws_value, wm_value, robust, fold_scores = _robust_inputs(outage_rows, fold_order)
    per_seed_scores = [
        _robust_inputs([row for row in per_seed_rows if row["seed"] == seed], fold_order)[3]
        for seed in SEED_ORDER
    ]
    full_values = [row["rmse3d_m"] for row in full_rows]
    full_f = 0.5 * _fixed_mean(full_values, name="full F mean") + 0.5 * max(full_values)
    bundle: dict[str, Any] = {
        "contract": BUNDLE_CONTRACT,
        "schema_version": 1,
        "method_id": "masked_erm",
        "seed_order": list(SEED_ORDER),
        "primary_fold_order": fold_order,
        "view_order": list(VIEW_ORDER),
        "initial_state_by_seed": {str(seed): next(iter(initial_by_seed[seed])) for seed in SEED_ORDER},
        "primary_context_identity": {
            "context_count": 32,
            "ordered_contexts": identity_rows,
            "identity_sha256": canonical_sha256(identity_rows),
        },
        "primary_seed_ensemble_outage_contexts": outage_rows,
        "primary_seed_ensemble_full_contexts": full_rows,
        "primary_per_seed_outage_contexts": per_seed_rows,
        "derived_gate_inputs": {
            "A_mean_context_rmse3d_m": a_value,
            "WS_worst_scenario_mean_rmse3d_m": ws_value,
            "WM_worst_mask_mean_rmse3d_m": wm_value,
            "robust_J_m": robust,
            "fold_outage_scores_m": fold_scores,
            "per_seed_J_m": per_seed_scores,
            "full_F_m": full_f,
        },
        "source_references": {
            "v1_development_manifest": state["development_ref"],
            "masked_erm_primary_training_receipts": training_refs,
            "masked_erm_primary_outer_metric_receipts": metric_refs,
        },
    }
    bundle["payload_sha256"] = canonical_sha256(bundle)
    return validate_comparator_bundle(bundle)


def _write_exclusive(directory_descriptor: int, name: str, raw: bytes) -> None:
    if "/" in name or name in {"", ".", ".."}:
        _fail("exclusive output name is not a single canonical component")
    if not hasattr(os, "O_NOFOLLOW"):
        _fail("exclusive output writes require Linux O_NOFOLLOW")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
    complete = False
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail("exclusive output write made no progress")
            offset += written
        os.fsync(descriptor)
        complete = True
    finally:
        os.close(descriptor)
        if not complete:
            try:
                os.unlink(name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass


def _validate_root_relationships(
    workspace: Path, project: Path, v1: Path, preparation: Path, v2: Path
) -> None:
    if project != workspace / PROJECT_DIRECTORY:
        _fail("project_root is not the exact declared child of workspace_root")
    expected_v1 = project.joinpath(*SUCCESSFUL_V1_RUN_PROJECT_RELATIVE_PATH.split("/"))
    expected_preparation = project.joinpath(
        *SUCCESSFUL_PREPARATION_RUN_PROJECT_RELATIVE_PATH.split("/")
    )
    if v1 != expected_v1:
        _fail(
            "v1_run_root is not the exact terminal-bound successful "
            "openaxes-v1-sealed-003 project path"
        )
    if preparation != expected_preparation:
        _fail(
            "preparation_run_root is not the exact terminal-bound successful "
            "openaxes-v1-sealed-003 project path"
        )
    if not (
        v1.name == f"run_{SUCCESSFUL_V1_RUN_ID}"
        and preparation.name == f"prepared_{SUCCESSFUL_V1_RUN_ID}"
        and v1.name.removeprefix("run_")
        == preparation.name.removeprefix("prepared_")
        == SUCCESSFUL_V1_RUN_ID
    ):
        _fail("v1 and preparation roots do not share the frozen sealed-003 suffix")
    scoped = (v1, preparation, v2)
    if len(set(scoped)) != 3:
        _fail("v1, preparation, and v2 run roots must be distinct")
    for index, left in enumerate(scoped):
        for right in scoped[index + 1 :]:
            if left in right.parents or right in left.parents:
                _fail("v1, preparation, and v2 run roots may not contain one another")


def _issue_parent_receipts_pinned(
    *,
    workspace: _PinnedRoot,
    project: _PinnedRoot,
    v1_root: _PinnedRoot,
    preparation_root: _PinnedRoot,
    v2_root: _PinnedRoot,
    source_audit_evidence_git_commit: str,
) -> dict[str, Any]:
    """Issue the immutable terminal envelope and masked-only comparator bundle."""

    _validate_root_relationships(
        workspace.path, project.path, v1_root.path, preparation_root.path, v2_root.path
    )
    repo = _repository(workspace.path)

    evidence_commit = _verify_commit(
        repo, source_audit_evidence_git_commit, name="source-audit evidence commit"
    )
    _oid, _evidence_ref, source_audit_raw = _git_blob(
        repo,
        evidence_commit,
        ACTIVE_SOURCE_AUDIT_RECEIPT_PATH,
        name="source-audit evidence",
    )
    source_audit = validate_source_audit_receipt(
        strict_json_bytes(source_audit_raw, name="source-audit evidence", require_canonical=True)
    )
    evidence_commit, evidence_oid = _verify_source_audit_git_evidence(
        repo, source_audit_raw, source_audit, evidence_commit
    )
    _verify_worktree_source_closure(project, source_audit["source_closure"])
    audited_commit = source_audit["audited_git_commit"]

    state = _validate_v1_envelopes(v1_root, preparation_root, project)
    model_head = state["model_head"]
    _verify_v1_sources_at_model_head(repo, model_head)
    prereg_ancestry_command = _git_ancestor(
        repo, audited_commit, model_head, name="v2 preregistration/v1 ancestry"
    )
    audited_ancestry_command = _git_ancestor(
        repo, audited_commit, model_head, name="audited source/v1 ancestry"
    )
    evidence_ancestry_command = _git_ancestor(
        repo, evidence_commit, model_head, name="source-audit evidence/v1 ancestry"
    )
    yaml_oid, yaml_reference, _yaml_raw = _git_blob(
        repo, audited_commit, V2_PREREGISTRATION_PATHS[0], name="native-v2 YAML preregistration"
    )
    markdown_oid, markdown_reference, _markdown_raw = _git_blob(
        repo, audited_commit, V2_PREREGISTRATION_PATHS[1], name="native-v2 Markdown preregistration"
    )
    bundle = _build_comparator_bundle(v1_root, state)
    timestamp = _timestamp(_now_utc())
    source_audit_copy_path = "parents/parent_receipt_auditor_source_audit.json"
    bundle_path = "parents/v1_masked_erm_comparator_bundle.json"
    terminal_path = "parents/v1_terminal_for_native_v2.json"
    source_audit_reference = _reference(source_audit_copy_path, source_audit_raw)
    bundle_raw = canonical_json_bytes(bundle, final_newline=True)
    bundle_reference = _reference(bundle_path, bundle_raw)

    fold_items = state["preparation_manifest"]["folds"]["items"]
    table_fold_identity = canonical_sha256(
        {"prepared_table": state["preparation_table_ref"], "folds": fold_items}
    )
    terminal: dict[str, Any] = {
        "contract": TERMINAL_CONTRACT,
        "schema_version": 1,
        "created_at_utc": timestamp,
        "reference_roots": {
            "workspace_root": str(workspace),
            "project_root": str(project),
            "v1_run_root": str(v1_root),
            "preparation_run_root": str(preparation_root),
            "v2_run_root": str(v2_root),
        },
        "auditor_source_audit": {
            "auditor_identity": AUDITOR_IDENTITY,
            "git_commit": audited_commit,
            "source_set_sha256": source_audit["source_set_sha256"],
            "source_audit_reference": source_audit_reference,
            "source_audit_git_commit": evidence_commit,
            "source_audit_git_blob_oid": evidence_oid,
            "source_audit_receipt_path": ACTIVE_SOURCE_AUDIT_RECEIPT_PATH,
            "source_audit_erratum_id": SOURCE_AUDIT_ERRATUM_ID,
            "timing_scope": ERRATUM_TIMING_SCOPE,
            "terminal_bound_successful_run_id": SUCCESSFUL_V1_RUN_ID,
            "audited_source_ancestor_check_command": audited_ancestry_command,
            "audited_source_ancestor_check_exit_code": 0,
            "source_audit_evidence_ancestor_check_command": evidence_ancestry_command,
            "source_audit_evidence_ancestor_check_exit_code": 0,
            "committed_before_v1_prepare": True,
            "source_audited_before_v1_prepare": True,
        },
        "git_lineage": {
            "repository_object_format": "sha1",
            "v2_preregistration_git_commit": audited_commit,
            "v2_yaml_git_blob_oid": yaml_oid,
            "v2_markdown_git_blob_oid": markdown_oid,
            "v2_yaml_git_blob_reference": yaml_reference,
            "v2_markdown_git_blob_reference": markdown_reference,
            "v1_model_contract_git_head": model_head,
            "ancestor_check_command": prereg_ancestry_command,
            "ancestor_check_exit_code": 0,
        },
        "v1_lineage": {
            "contract": V1_CONTRACT,
            "canonical_config_sha256": V1_CONFIG_SHA256,
            "terminal_source_set_file_count": V1_SOURCE_FILE_COUNT,
            "terminal_source_set_sha256": V1_SOURCE_SET_SHA256,
            "terminal_stage": V1_TERMINAL_STAGE,
            "exact_task_count": V1_TASK_COUNT,
            "comparator_method_id": "masked_erm",
        },
        "preparation_lineage": {
            "contract": PREPARATION_CONTRACT,
            "canonical_config_sha256": PREPARATION_CONFIG_SHA256,
            "manifest_sha256": state["preparation_manifest_ref"]["sha256"],
            "prepared_table_and_fold_identity_sha256": table_fold_identity,
        },
        "references": {
            "v1_complete_marker": state["complete_ref"],
            "v1_development_manifest": state["development_ref"],
            "v1_task_index": state["task_index_ref"],
            "v1_prepared_manifest": state["prepared_ref"],
            "preparation_manifest": state["preparation_manifest_ref"],
            "preparation_table": state["preparation_table_ref"],
        },
        "comparator_bundle_reference": bundle_reference,
        "assertions": {key: True for key in sorted(TERMINAL_ASSERTION_KEYS)},
    }
    terminal["payload_sha256"] = canonical_sha256(terminal)
    _validate_terminal_receipt_structure(
        terminal,
        comparator_bundle=bundle,
        copied_source_audit=source_audit,
        expected_roots={
            "workspace_root": workspace,
            "project_root": project,
            "v1_run_root": v1_root,
            "preparation_run_root": preparation_root,
            "v2_run_root": v2_root,
        },
    )
    terminal_raw = canonical_json_bytes(terminal, final_newline=True)

    v2_descriptor = os.dup(v2_root.descriptor)
    try:
        try:
            os.mkdir("parents", 0o700, dir_fd=v2_descriptor)
        except FileExistsError as exc:
            raise OpenAxesNativeV2ParentReceiptError(
                "v2 parents directory must be fresh and absent"
            ) from exc
        parent_descriptor = os.open(
            "parents",
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=v2_descriptor,
        )
        try:
            parent_dev_ino_mode = _directory_identity(os.fstat(parent_descriptor))
            _write_exclusive(parent_descriptor, "parent_receipt_auditor_source_audit.json", source_audit_raw)
            _write_exclusive(parent_descriptor, "v1_masked_erm_comparator_bundle.json", bundle_raw)
            _write_exclusive(parent_descriptor, "v1_terminal_for_native_v2.json", terminal_raw)
            os.fsync(parent_descriptor)
            if _directory_identity(os.fstat(parent_descriptor)) != parent_dev_ino_mode:
                _fail("parents output directory identity changed during exclusive writes")
            parent_entry = os.stat("parents", dir_fd=v2_descriptor, follow_symlinks=False)
            if _directory_identity(parent_entry) != parent_dev_ino_mode:
                _fail("v2 root no longer names the pinned parents output directory")
        finally:
            os.close(parent_descriptor)
        os.fsync(v2_descriptor)
    finally:
        os.close(v2_descriptor)
    return terminal


def issue_parent_receipts(
    *,
    workspace_root: str | os.PathLike[str],
    v1_run_dir: str | os.PathLike[str],
    preparation_run_dir: str | os.PathLike[str],
    v2_run_dir: str | os.PathLike[str],
    source_audit_evidence_git_commit: str,
) -> dict[str, Any]:
    """Issue from componentwise pinned Linux directory handles."""

    pinned: list[_PinnedRoot] = []
    terminal: dict[str, Any] | None = None
    try:
        workspace = _pin_root(workspace_root, name="workspace_root")
        pinned.append(workspace)
        project = _pin_root(workspace.path / PROJECT_DIRECTORY, name="project_root")
        pinned.append(project)
        v1_root = _pin_root(v1_run_dir, name="v1_run_root")
        pinned.append(v1_root)
        preparation_root = _pin_root(preparation_run_dir, name="preparation_run_root")
        pinned.append(preparation_root)
        v2_root = _pin_root(v2_run_dir, name="v2_run_root")
        pinned.append(v2_root)
        terminal = _issue_parent_receipts_pinned(
            workspace=workspace,
            project=project,
            v1_root=v1_root,
            preparation_root=preparation_root,
            v2_root=v2_root,
            source_audit_evidence_git_commit=source_audit_evidence_git_commit,
        )
        for name, root in (
            ("workspace_root", workspace),
            ("project_root", project),
            ("v1_run_root", v1_root),
            ("preparation_run_root", preparation_root),
            ("v2_run_root", v2_root),
        ):
            root.revalidate(name=name)
    finally:
        for root in reversed(pinned):
            root.close()
    if terminal is None:
        _fail("parent receipt issuance did not produce a terminal envelope")
    # Success is not returned until the three persisted files have been
    # reopened and the independent public authority path has recomputed them.
    return validate_terminal_receipt(
        terminal,
        workspace_root=workspace_root,
        v1_run_dir=v1_run_dir,
        preparation_run_dir=preparation_run_dir,
        v2_run_dir=v2_run_dir,
        source_audit_evidence_git_commit=source_audit_evidence_git_commit,
    )


def _validate_terminal_receipt_structure(
    value: Any,
    *,
    comparator_bundle: Mapping[str, Any],
    copied_source_audit: Mapping[str, Any],
    expected_roots: Mapping[str, Path | _PinnedRoot] | None = None,
) -> dict[str, Any]:
    terminal = _mapping(value, name="native-v2 parent terminal receipt")
    _keys(terminal, TERMINAL_KEYS, name="native-v2 parent terminal receipt")
    _exact_str(terminal["contract"], name="terminal contract", expected=TERMINAL_CONTRACT)
    _exact_int(terminal["schema_version"], name="terminal schema version", expected=1)
    _timestamp(terminal["created_at_utc"])
    roots = _mapping(terminal["reference_roots"], name="terminal reference roots")
    _keys(roots, ROOT_KEYS, name="terminal reference roots")
    if expected_roots is None:
        resolved_roots: dict[str, Path | _PinnedRoot] = {
            key: _absolute_root(
                _exact_str(roots[key], name=f"terminal root {key}"),
                name=f"terminal root {key}",
            )
            for key in ROOT_KEYS
        }
    else:
        if set(expected_roots) != set(ROOT_KEYS):
            _fail("internal terminal validator did not receive all pinned roots")
        resolved_roots = dict(expected_roots)
        for key, resolved in resolved_roots.items():
            expected_path = resolved.path if isinstance(resolved, _PinnedRoot) else resolved
            if roots[key] != str(expected_path):
                _fail(f"terminal root {key} differs from the caller-pinned authority")
    root_paths = {
        key: value.path if isinstance(value, _PinnedRoot) else value
        for key, value in resolved_roots.items()
    }
    _validate_root_relationships(
        root_paths["workspace_root"],
        root_paths["project_root"],
        root_paths["v1_run_root"],
        root_paths["preparation_run_root"],
        root_paths["v2_run_root"],
    )
    auditor = _mapping(terminal["auditor_source_audit"], name="terminal auditor source audit")
    _keys(auditor, AUDITOR_BINDING_KEYS, name="terminal auditor source audit")
    _exact_str(auditor["auditor_identity"], name="auditor identity", expected=AUDITOR_IDENTITY)
    _git_sha1(auditor["git_commit"], name="auditor Git commit")
    _hex64(auditor["source_set_sha256"], name="auditor source-set hash")
    _validate_reference_shape(
        auditor["source_audit_reference"],
        name="copied source-audit reference",
        expected_path="parents/parent_receipt_auditor_source_audit.json",
    )
    _git_sha1(auditor["source_audit_git_commit"], name="source-audit evidence commit")
    _git_sha1(auditor["source_audit_git_blob_oid"], name="source-audit evidence OID")
    _exact_str(
        auditor["source_audit_receipt_path"],
        name="active source-audit receipt path",
        expected=ACTIVE_SOURCE_AUDIT_RECEIPT_PATH,
    )
    _exact_str(
        auditor["source_audit_erratum_id"],
        name="source-audit erratum identity",
        expected=SOURCE_AUDIT_ERRATUM_ID,
    )
    _exact_str(
        auditor["timing_scope"],
        name="source-audit timing scope",
        expected=ERRATUM_TIMING_SCOPE,
    )
    _exact_str(
        auditor["terminal_bound_successful_run_id"],
        name="terminal-bound successful run identity",
        expected=SUCCESSFUL_V1_RUN_ID,
    )
    for key in ("audited_source_ancestor_check_command", "source_audit_evidence_ancestor_check_command"):
        command = _exact_str(auditor[key], name=key)
        if not command.startswith("git merge-base --is-ancestor "):
            _fail(f"{key} is not the frozen command")
    for key in ("audited_source_ancestor_check_exit_code", "source_audit_evidence_ancestor_check_exit_code"):
        _exact_int(auditor[key], name=key, expected=0)
    for key in ("committed_before_v1_prepare", "source_audited_before_v1_prepare"):
        _exact_bool(auditor[key], name=key, expected=True)

    source_audit = validate_source_audit_receipt(dict(copied_source_audit))
    if auditor["git_commit"] != source_audit["audited_git_commit"] or auditor["source_set_sha256"] != source_audit["source_set_sha256"]:
        _fail("terminal auditor identity differs from copied source-audit receipt")
    source_raw = canonical_json_bytes(source_audit, final_newline=True)
    if auditor["source_audit_reference"] != _reference(
        "parents/parent_receipt_auditor_source_audit.json", source_raw
    ):
        _fail("terminal copied source-audit raw reference changed")

    git_lineage = _mapping(terminal["git_lineage"], name="terminal Git lineage")
    _keys(git_lineage, GIT_LINEAGE_KEYS, name="terminal Git lineage")
    _exact_str(git_lineage["repository_object_format"], name="Git object format", expected="sha1")
    prereg_commit = _git_sha1(git_lineage["v2_preregistration_git_commit"], name="v2 preregistration commit")
    if prereg_commit != auditor["git_commit"]:
        _fail("v2 preregistration commit differs from audited source commit")
    for key in ("v2_yaml_git_blob_oid", "v2_markdown_git_blob_oid", "v1_model_contract_git_head"):
        _git_sha1(git_lineage[key], name=key)
    _validate_reference_shape(
        git_lineage["v2_yaml_git_blob_reference"],
        name="v2 YAML Git reference",
        expected_path=V2_PREREGISTRATION_PATHS[0],
    )
    _validate_reference_shape(
        git_lineage["v2_markdown_git_blob_reference"],
        name="v2 Markdown Git reference",
        expected_path=V2_PREREGISTRATION_PATHS[1],
    )
    _exact_str(git_lineage["ancestor_check_command"], name="preregistration ancestor command")
    _exact_int(git_lineage["ancestor_check_exit_code"], name="preregistration ancestor exit", expected=0)

    v1_lineage = _mapping(terminal["v1_lineage"], name="terminal v1 lineage")
    _keys(v1_lineage, V1_LINEAGE_KEYS, name="terminal v1 lineage")
    expected_v1 = {
        "contract": V1_CONTRACT,
        "canonical_config_sha256": V1_CONFIG_SHA256,
        "terminal_source_set_file_count": 9,
        "terminal_source_set_sha256": V1_SOURCE_SET_SHA256,
        "terminal_stage": V1_TERMINAL_STAGE,
        "exact_task_count": 336,
        "comparator_method_id": "masked_erm",
    }
    if dict(v1_lineage) != expected_v1:
        _fail("terminal v1 lineage differs from preregistration")
    preparation_lineage = _mapping(terminal["preparation_lineage"], name="terminal preparation lineage")
    _keys(preparation_lineage, PREPARATION_LINEAGE_KEYS, name="terminal preparation lineage")
    _exact_str(preparation_lineage["contract"], name="preparation lineage contract", expected=PREPARATION_CONTRACT)
    _exact_str(preparation_lineage["canonical_config_sha256"], name="preparation lineage config", expected=PREPARATION_CONFIG_SHA256)
    _hex64(preparation_lineage["manifest_sha256"], name="preparation manifest hash")
    _hex64(preparation_lineage["prepared_table_and_fold_identity_sha256"], name="preparation table/fold identity")

    references = _mapping(terminal["references"], name="terminal references")
    _keys(references, TERMINAL_REFERENCE_KEYS, name="terminal references")
    expected_paths = {
        "v1_complete_marker": "COMPLETE.json",
        "v1_development_manifest": "development_manifest.json",
        "v1_task_index": "tasks/index.json",
        "v1_prepared_manifest": "prepared_manifest.json",
        "preparation_manifest": "manifest.json",
        "preparation_table": "prepared.parquet",
    }
    for key, path in expected_paths.items():
        root = (
            resolved_roots["preparation_run_root"]
            if key.startswith("preparation_")
            else resolved_roots["v1_run_root"]
        )
        _verify_reference(
            root,
            references[key],
            name=f"terminal reference {key}",
            expected_path=path,
        )
    comparator = validate_comparator_bundle(dict(comparator_bundle))
    comparator_raw = canonical_json_bytes(comparator, final_newline=True)
    expected_comparator_ref = _reference("parents/v1_masked_erm_comparator_bundle.json", comparator_raw)
    if terminal["comparator_bundle_reference"] != expected_comparator_ref:
        _fail("terminal comparator-bundle reference changed")
    assertions = _mapping(terminal["assertions"], name="terminal assertions")
    _keys(assertions, TERMINAL_ASSERTION_KEYS, name="terminal assertions")
    for key in TERMINAL_ASSERTION_KEYS:
        _exact_bool(assertions[key], name=f"terminal assertion {key}", expected=True)
    _payload_hash(terminal, key="payload_sha256", name="terminal receipt")
    return dict(terminal)


def validate_terminal_receipt(
    value: Any,
    *,
    workspace_root: str | os.PathLike[str],
    v1_run_dir: str | os.PathLike[str],
    preparation_run_dir: str | os.PathLike[str],
    v2_run_dir: str | os.PathLike[str],
    source_audit_evidence_git_commit: str,
) -> dict[str, Any]:
    """Authoritatively reopen and recompute a persisted terminal envelope.

    Caller-supplied comparator/audit dictionaries are intentionally not an
    authority.  This entry point pins all roots, reads the three persisted
    parent files itself, and recomputes Git, v1, preparation, and comparator
    bindings from the declared immutable evidence.
    """

    claimed = _mapping(value, name="claimed terminal receipt")
    pinned: list[_PinnedRoot] = []
    try:
        workspace = _pin_root(workspace_root, name="workspace_root")
        pinned.append(workspace)
        project = _pin_root(workspace.path / PROJECT_DIRECTORY, name="project_root")
        pinned.append(project)
        v1_root = _pin_root(v1_run_dir, name="v1_run_root")
        pinned.append(v1_root)
        preparation_root = _pin_root(preparation_run_dir, name="preparation_run_root")
        pinned.append(preparation_root)
        v2_root = _pin_root(v2_run_dir, name="v2_run_root")
        pinned.append(v2_root)
        _validate_root_relationships(
            workspace.path, project.path, v1_root.path, preparation_root.path, v2_root.path
        )
        _verify_exact_directory_members(
            v2_root,
            "parents",
            expected_files={
                "parent_receipt_auditor_source_audit.json",
                "v1_masked_erm_comparator_bundle.json",
                "v1_terminal_for_native_v2.json",
            },
            expected_directories=set(),
            name="persisted native-v2 parent receipt directory",
        )
        _terminal_ref, terminal_raw = _raw_reference(
            v2_root,
            "parents/v1_terminal_for_native_v2.json",
            name="persisted native-v2 terminal",
        )
        persisted_terminal = _mapping(
            strict_json_bytes(
                terminal_raw, name="persisted native-v2 terminal", require_canonical=True
            ),
            name="persisted native-v2 terminal",
        )
        if persisted_terminal != claimed:
            _fail("claimed terminal differs from persisted canonical terminal bytes")
        _audit_ref, audit_raw = _raw_reference(
            v2_root,
            "parents/parent_receipt_auditor_source_audit.json",
            name="persisted copied source audit",
        )
        copied_audit = validate_source_audit_receipt(
            strict_json_bytes(
                audit_raw, name="persisted copied source audit", require_canonical=True
            )
        )
        _bundle_ref, bundle_raw = _raw_reference(
            v2_root,
            "parents/v1_masked_erm_comparator_bundle.json",
            name="persisted comparator bundle",
        )
        persisted_bundle = validate_comparator_bundle(
            strict_json_bytes(
                bundle_raw, name="persisted comparator bundle", require_canonical=True
            )
        )
        validated = _validate_terminal_receipt_structure(
            persisted_terminal,
            comparator_bundle=persisted_bundle,
            copied_source_audit=copied_audit,
            expected_roots={
                "workspace_root": workspace,
                "project_root": project,
                "v1_run_root": v1_root,
                "preparation_run_root": preparation_root,
                "v2_run_root": v2_root,
            },
        )
        repo = _repository(workspace.path)
        evidence = _verify_commit(
            repo,
            source_audit_evidence_git_commit,
            name="public source-audit evidence commit",
        )
        if validated["auditor_source_audit"]["source_audit_git_commit"] != evidence:
            _fail("terminal source-audit evidence commit differs from caller-pinned authority")
        evidence_commit, evidence_oid = _verify_source_audit_git_evidence(
            repo, audit_raw, copied_audit, evidence
        )
        _verify_worktree_source_closure(project, copied_audit["source_closure"])
        state = _validate_v1_envelopes(v1_root, preparation_root, project)
        model_head = state["model_head"]
        _verify_v1_sources_at_model_head(repo, model_head)
        recomputed_bundle = _build_comparator_bundle(v1_root, state)
        if recomputed_bundle != persisted_bundle:
            _fail("persisted comparator bundle differs from authoritative sealed v1 evidence")
        audited_commit = copied_audit["audited_git_commit"]
        audited_command = _git_ancestor(
            repo, audited_commit, model_head, name="public audited-source/v1 ancestry"
        )
        evidence_command = _git_ancestor(
            repo, evidence_commit, model_head, name="public source-audit-evidence/v1 ancestry"
        )
        yaml_oid, yaml_ref, _yaml_raw = _git_blob(
            repo,
            audited_commit,
            V2_PREREGISTRATION_PATHS[0],
            name="public native-v2 YAML preregistration",
        )
        markdown_oid, markdown_ref, _markdown_raw = _git_blob(
            repo,
            audited_commit,
            V2_PREREGISTRATION_PATHS[1],
            name="public native-v2 Markdown preregistration",
        )
        auditor = validated["auditor_source_audit"]
        if (
            auditor["source_audit_git_blob_oid"] != evidence_oid
            or auditor["audited_source_ancestor_check_command"] != audited_command
            or auditor["source_audit_evidence_ancestor_check_command"] != evidence_command
        ):
            _fail("terminal auditor Git authority differs from public recomputation")
        lineage = validated["git_lineage"]
        if (
            lineage["v2_yaml_git_blob_oid"] != yaml_oid
            or lineage["v2_markdown_git_blob_oid"] != markdown_oid
            or lineage["v2_yaml_git_blob_reference"] != yaml_ref
            or lineage["v2_markdown_git_blob_reference"] != markdown_ref
            or lineage["v1_model_contract_git_head"] != model_head
            or lineage["ancestor_check_command"] != audited_command
        ):
            _fail("terminal Git lineage differs from public recomputation")
        fold_items = state["preparation_manifest"]["folds"]["items"]
        expected_table_fold = canonical_sha256(
            {"prepared_table": state["preparation_table_ref"], "folds": fold_items}
        )
        preparation_lineage = validated["preparation_lineage"]
        if (
            preparation_lineage["manifest_sha256"]
            != state["preparation_manifest_ref"]["sha256"]
            or preparation_lineage["prepared_table_and_fold_identity_sha256"]
            != expected_table_fold
        ):
            _fail("terminal preparation lineage differs from public recomputation")
        expected_refs = {
            "v1_complete_marker": state["complete_ref"],
            "v1_development_manifest": state["development_ref"],
            "v1_task_index": state["task_index_ref"],
            "v1_prepared_manifest": state["prepared_ref"],
            "preparation_manifest": state["preparation_manifest_ref"],
            "preparation_table": state["preparation_table_ref"],
        }
        if validated["references"] != expected_refs:
            _fail("terminal raw evidence references differ from public recomputation")
        for name, root in (
            ("workspace_root", workspace),
            ("project_root", project),
            ("v1_run_root", v1_root),
            ("preparation_run_root", preparation_root),
            ("v2_run_root", v2_root),
        ):
            root.revalidate(name=name)
        return validated
    finally:
        for root in reversed(pinned):
            root.close()


def _write_new_path(path: Path, raw: bytes) -> None:
    if path.name in {"", ".", ".."} or os.path.lexists(path):
        _fail("source-audit output must be a new canonical path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent = path.parent
    if not parent.exists():
        if parent.name != "receipts":
            _fail("only the frozen receipts directory may be created for source audit")
        grandparent = _absolute_root(parent.parent, name="source-audit project root")
        grandparent_descriptor = os.open(grandparent, flags)
        try:
            try:
                os.mkdir("receipts", 0o700, dir_fd=grandparent_descriptor)
                os.fsync(grandparent_descriptor)
            except FileExistsError:
                pass
            descriptor = os.open("receipts", flags, dir_fd=grandparent_descriptor)
        finally:
            os.close(grandparent_descriptor)
    else:
        canonical_parent = _absolute_root(parent, name="output parent")
        if parent != canonical_parent:
            _fail("source-audit output parent changed")
        descriptor = os.open(canonical_parent, flags)
    try:
        _write_exclusive(descriptor, path.name, raw)
        os.fsync(descriptor)
    except Exception:
        # A newly created empty receipts directory is harmless and retained;
        # any partial output is removed by _write_exclusive.  A failed source
        # audit must be rerun from a fresh output path.
        raise
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("source-audit", help="build the pre-v1 source-audit receipt")
    source.add_argument("--workspace-root", required=True)
    source.add_argument("--audited-git-commit", required=True)
    source.add_argument("--output", required=True)
    issue = subparsers.add_parser("issue", help="issue future post-v1 parent receipts read-only")
    issue.add_argument("--workspace-root", required=True)
    issue.add_argument("--v1-run-dir", required=True)
    issue.add_argument("--preparation-run-dir", required=True)
    issue.add_argument("--v2-run-dir", required=True)
    issue.add_argument("--source-audit-evidence-git-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "source-audit":
        workspace = _absolute_root(args.workspace_root, name="workspace_root")
        receipt = build_source_audit_receipt(
            workspace,
            args.audited_git_commit,
        )
        expected_output = (
            workspace
            / PROJECT_DIRECTORY
            / ACTIVE_SOURCE_AUDIT_RECEIPT_PATH
        )
        if Path(args.output) != expected_output:
            _fail("source-audit output path differs from the frozen receipt path")
        _write_new_path(expected_output, canonical_json_bytes(receipt, final_newline=True))
        return 0
    if args.command == "issue":
        issue_parent_receipts(
            workspace_root=args.workspace_root,
            v1_run_dir=args.v1_run_dir,
            preparation_run_dir=args.preparation_run_dir,
            v2_run_dir=args.v2_run_dir,
            source_audit_evidence_git_commit=args.source_audit_evidence_git_commit,
        )
        return 0
    _fail("unknown command")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OpenAxesNativeV2ParentReceiptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
