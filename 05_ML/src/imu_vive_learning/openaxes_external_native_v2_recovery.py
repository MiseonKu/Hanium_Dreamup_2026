"""Outcome-neutral compound finalization recovery for native-V2 RUN003.

This module is deliberately separate from the frozen RUN003 execution source.
It can only aggregate the already completed task artifacts, seal seven terminal
files outside the run, and append those exact bytes with ``O_EXCL``.  It has no
training, prediction, label-loading, or task-evaluator entry point.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import stat
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from . import openaxes_external_native_v2_experiment as frozen
from . import openaxes_external_native_v2_recovery_authority as recovery_authority
from . import openaxes_external_native_v2_training as native_training


RUN_ID = "openaxes-native-v2-sealed-003"
SOURCE_AUDIT_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.compound_recovery.source_audit.v1"
)
STAGE_SEAL_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.compound_recovery.stage_seal.v1"
)
TERMINAL_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.compound_recovery.terminal.v1"
)
SIGNED_REPORT_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.signed_shift_report.v1"
)
RECOVERY_SOURCE_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/"
    "openaxes_external_native_v2_recovery.py"
)
RECOVERY_TEST_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/tests/"
    "test_openaxes_external_native_v2_recovery.py"
)
RECOVERY_SOURCE_AUDIT_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/receipts/"
    "openaxes_external_native_v2_compound_recovery_source_audit.json"
)
ERRATUM_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/docs/"
    "OPENAXES_EXTERNAL_NATIVE_V2_RUN003_RECOVERY_ERRATUM.md"
)
DIAGNOSTIC_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/docs/"
    "OPENAXES_EXTERNAL_NATIVE_V2_SIGNED_DIAGNOSTIC_RECEIPT.md"
)
V1_CONFIG_PATH = "09_SO101_IMU_VIVE_LEARNING/configs/openaxes_external_v1.yaml"
V1_SOURCE_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/artifacts/openaxes_external_v1/source/"
    "openaxes-example-robot-dataset.zip"
)
V1_PREPARATION_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/artifacts/openaxes_external_v1/"
    "prepared_openaxes-v1-sealed-003"
)

ORIGINAL_CONFIG_SHA256 = frozen.CONFIG_SHA256
ORIGINAL_EXECUTION_SOURCE_SET_SHA256 = (
    "3cc3133cd925fc3ebeacd53a4f746090a99d8b52d92ce2157d1d0b36576beb99"
)
ORIGINAL_CONTROLLER_SHA256 = (
    "83a49e943494061bf0797d4acebe55a303aeb80e397db00f3a31d925d4ec9e59"
)
FAILED_TREE_SHA256 = (
    "e42472225abdaaa081da7a33dfdbe6d3543a607b579654a8ef7172aeb944b08b"
)
FAILED_REGULAR_FILE_COUNT = 2799
FAILED_REGULAR_FILE_BYTES = 318_851_615
FAILED_SOURCE_SET_RAW_SHA256 = (
    "330e2a4643186242f74a3b5ebee627a93fc7c4383601b202a273ee66f42eeb6c"
)
FAILED_SESSION_RAW_SHA256 = (
    "9500a36414e1408e998502b320015b9eba07140a0919302a7673274e845bdea7"
)
FAILED_TASK_INDEX_RAW_SHA256 = (
    "544a0ba786c9708047d0689124597a5d94a76d790fd544bc82fd71b80fe5975b"
)
ERRATUM_SHA256 = "0bca94bfb7ac930a504ccd01fa557261da16b312f6eec74779a33101604aab6a"
ERRATUM_BYTES = 5356
DIAGNOSTIC_SHA256 = (
    "bbabad0c4bbfa63f4beb0d01d476cba86148bf362fb2d892faf596fb41e0a38f"
)
DIAGNOSTIC_BYTES = 2027

OUTPUT_PATHS = (
    "reports/zero_shift.json",
    "reports/primary_gates.json",
    "reports/stress_report.json",
    "reports/signed_shift_report.json",
    "reports/development_summary.json",
    "development_manifest.json",
    "COMPLETE.json",
)
REPORT_NAMES = tuple(path.removeprefix("reports/") for path in OUTPUT_PATHS[:5])
STAGE_SEAL_PATH = "STAGE_SEAL.json"
TERMINAL_PATH = "COMPOUND_TERMINAL.json"
SHIFT_ORDER = (-50, -20, -10, 10, 20, 50)
SIGNED_TRUTH_COLUMNS = (
    "row_uid",
    "scenario_id",
    "base_aligned_time_us",
    "shift_ms",
    "shifted_target_time_us",
    "shifted_tcp_x_m",
    "shifted_tcp_y_m",
    "shifted_tcp_z_m",
)
V1_PUBLIC_SOURCE_PATHS = (
    "09_SO101_IMU_VIVE_LEARNING/configs/openaxes_external_v1.yaml",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/openaxes_external_v1_config.py",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/openaxes_external_v1.py",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/openaxes_external_v1_cli.py",
    "09_SO101_IMU_VIVE_LEARNING/tests/test_openaxes_external_v1.py",
    "09_SO101_IMU_VIVE_LEARNING/docs/OPENAXES_EXTERNAL_V1_CONTRACT.md",
    "09_SO101_IMU_VIVE_LEARNING/pyproject.toml",
)
RECOVERY_CLOSURE_PATHS = recovery_authority.RECOVERY_SOURCE_CLOSURE

ACTION_ACCOUNTING = {
    "external_diagnostic_calls": 2,
    "authoritative_recovery_calls": 1,
    "original_failed_in_memory_gate_computations": 1,
    "recovery_gate_replays": 1,
    "persisted_authoritative_gate_decisions": 1,
    "original_noncanonical_signed_preparation_reads": 1,
    "recovery_canonical_public_signed_reconstructions": 1,
    "recovery_scientific_fits": 0,
    "recovery_optimizer_updates": 0,
    "recovery_checkpoint_selections": 0,
    "recovery_model_forwards": 0,
    "recovery_outer_feature_loads": 0,
    "recovery_outer_label_reads": 0,
    "recovery_task_evaluator_calls": 0,
}
ORIGINAL_SCIENCE_AUTHORITY_KEYS = {
    "run_id",
    "canonical_config_sha256",
    "execution_source_set_sha256",
    "controller_source_sha256",
    "source_set_reference",
    "execution_session_reference",
    "task_index_reference",
    "result_completed_reference_set_sha256",
    "result_completed_count",
}
FAILED_PRESTATE_KEYS = {
    "canonical_tree_sha256",
    "regular_file_count",
    "regular_file_bytes",
    "top_level_members",
    "reports_members",
    "development_manifest_absent",
    "complete_absent",
    "result_completed_count",
}
SEMANTIC_VALIDATION_KEYS = {
    "original_science_authority_valid",
    "exact_126_phase_chains_valid",
    "failed_prestate_valid",
    "zero_shift_report_valid",
    "primary_gate_report_valid",
    "stress_report_valid",
    "signed_report_valid",
    "development_summary_valid",
    "development_manifest_valid",
    "complete_valid",
    "staged_outputs_equal_published_outputs",
    "action_accounting_valid",
    "source_authorities_separate",
}


class OpenAxesNativeV2RecoveryError(RuntimeError):
    """Raised when compound recovery authority or state is ambiguous."""


def _fail(message: str) -> None:
    raise OpenAxesNativeV2RecoveryError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise OpenAxesNativeV2RecoveryError("value is not canonical JSON") from exc


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _payload(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    output["payload_sha256"] = _canonical_hash(output)
    return output


def _validate_payload(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        _fail(f"{name} is not a string-keyed mapping")
    output = dict(value)
    observed = output.pop("payload_sha256", None)
    if not isinstance(observed, str) or observed != _canonical_hash(output):
        _fail(f"{name} payload hash changed")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        _fail(f"{name} keys changed")


def _hex64(value: Any, *, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail(f"{name} is not lowercase SHA-256")
    return value


def _reference_for_raw(path: str, raw: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _validate_reference(value: Any, *, expected_path: str, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} is not a reference")
    _exact_keys(value, {"path", "bytes", "sha256"}, name=name)
    if value.get("path") != expected_path or type(value.get("bytes")) is not int:
        _fail(f"{name} path/size changed")
    _hex64(value.get("sha256"), name=f"{name} SHA-256")
    return dict(value)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _regular_fd_argument(value: str, *, name: str) -> int:
    match = re.fullmatch(r"/proc/self/fd/([0-9]+)", value)
    if match is None:
        _fail(f"{name} must be an inherited /proc/self/fd descriptor")
    inherited = int(match.group(1))
    if inherited < 3:
        _fail(f"{name} aliases a standard stream")
    try:
        descriptor = os.dup(inherited)
        info = os.fstat(descriptor)
    except OSError as exc:
        raise OpenAxesNativeV2RecoveryError(
            f"{name} descriptor is not open"
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        _fail(f"{name} is not a retained regular descriptor")
    return descriptor


def _read_regular_fd(descriptor: int, *, name: str) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        _fail(f"{name} is not regular")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        _fail(f"{name} changed while read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        _fail(f"{name} short read")
    return raw


def _read_fd_path(root_fd: int, relative: str, *, name: str) -> bytes:
    try:
        return frozen._read_bytes(root_fd, relative, name=name)
    except frozen.OpenAxesExternalNativeV2Error as exc:
        raise OpenAxesNativeV2RecoveryError(str(exc)) from exc


def _raw_reference(root_fd: int, relative: str, *, name: str) -> dict[str, Any]:
    return _reference_for_raw(relative, _read_fd_path(root_fd, relative, name=name))


def _tree_projection(
    root_fd: int, *, excluded_files: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Hash regular files as project-relative path/size/raw-hash records."""

    rows: list[tuple[str, int, str]] = []

    def visit(directory_fd: int, prefix: str) -> None:
        for name in sorted(os.listdir(directory_fd), key=os.fsencode):
            relative = f"{prefix}/{name}" if prefix else name
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise OpenAxesNativeV2RecoveryError(
                    f"cannot stat recovery tree member: {relative}"
                ) from exc
            if stat.S_ISLNK(info.st_mode) or not (
                stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
            ):
                _fail(f"recovery tree has symlink/special member: {relative}")
            if stat.S_ISDIR(info.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    visit(child, relative)
                finally:
                    os.close(child)
                continue
            if relative in excluded_files:
                continue
            raw = _read_fd_path(root_fd, relative, name=f"tree file {relative}")
            rows.append((relative, len(raw), hashlib.sha256(raw).hexdigest()))

    visit(root_fd, "")
    records = b"".join(
        f"{path}\t{size}\t{digest}\n".encode("utf-8")
        for path, size, digest in rows
    )
    return {
        "canonical_tree_sha256": hashlib.sha256(records).hexdigest(),
        "regular_file_count": len(rows),
        "regular_file_bytes": sum(size for _path, size, _digest in rows),
    }


def _expected_failed_prestate() -> dict[str, Any]:
    return {
        "canonical_tree_sha256": FAILED_TREE_SHA256,
        "regular_file_count": FAILED_REGULAR_FILE_COUNT,
        "regular_file_bytes": FAILED_REGULAR_FILE_BYTES,
        "top_level_members": [
            "PREPARED.json",
            "parents",
            "prepared_manifest.json",
            "quarantine",
            "reports",
            "results",
            "runtime",
            "task_index.json",
            "tasks",
        ],
        "reports_members": [],
        "development_manifest_absent": True,
        "complete_absent": True,
        "result_completed_count": 126,
    }


def _expected_action_accounting() -> dict[str, int]:
    return dict(ACTION_ACCOUNTING)


def _external_reference_from_fd(
    descriptor: int, *, expected_path: str, name: str
) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_fd(descriptor, name=name)
    return _reference_for_raw(expected_path, raw), raw


def validate_recovery_source_audit(
    value: Any,
    *,
    workspace_fd: int,
    erratum_reference: Mapping[str, Any],
    diagnostic_reference: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        audit = recovery_authority.validate_source_audit_receipt(value)
    except recovery_authority.OpenAxesNativeV2RecoveryAuthorityError as exc:
        raise OpenAxesNativeV2RecoveryError(str(exc)) from exc
    if audit["preissued_erratum_reference"] != dict(erratum_reference):
        _fail("recovery source audit erratum reference changed")
    if audit["diagnostic_receipt_reference"] != dict(diagnostic_reference):
        _fail("recovery source audit diagnostic reference changed")
    repository = audit["repository"]
    if not isinstance(repository, Mapping):
        _fail("recovery source-audit repository is malformed")
    _exact_keys(repository, set(recovery_authority.REPOSITORY_KEYS), name="recovery source-audit repository")
    if (
        repository["object_format"] != "sha1"
        or repository["project_directory"] != frozen.PROJECT_DIRECTORY
        or repository["source_audit_receipt_path"] != RECOVERY_SOURCE_AUDIT_PATH
        or repository["source_audit_evidence_commit_required"] is not True
        or repository["implementation_commit_must_be_ancestor_of_evidence_commit"]
        is not True
    ):
        _fail("recovery source-audit Git object format changed")
    if type(repository["implementation_git_commit"]) is not str or re.fullmatch(
        r"[0-9a-f]{40}", repository["implementation_git_commit"]
    ) is None:
        _fail("recovery source-audit implementation commit is malformed")
    closure = audit["source_closure"]
    if not isinstance(closure, Mapping):
        _fail("recovery source closure is malformed")
    _exact_keys(
        closure,
        set(recovery_authority.SOURCE_CLOSURE_KEYS),
        name="recovery source closure",
    )
    if closure["path_order"] != list(RECOVERY_CLOSURE_PATHS):
        _fail("recovery source closure path order changed")
    files = closure["files"]
    if (
        not isinstance(files, Mapping)
        or set(files) != set(RECOVERY_CLOSURE_PATHS)
        or closure["file_count"] != len(RECOVERY_CLOSURE_PATHS)
    ):
        _fail("recovery source closure file roster changed")
    normalized: list[dict[str, Any]] = []
    for path in RECOVERY_CLOSURE_PATHS:
        row = files[path]
        if not isinstance(row, Mapping):
            _fail(f"recovery source row is malformed: {path}")
        _exact_keys(row, set(recovery_authority.SOURCE_FILE_KEYS), name=path)
        if (
            row["path"] != path
            or row["git_mode"] not in ("100644", "100755")
            or type(row["git_blob_oid"]) is not str
            or re.fullmatch(r"[0-9a-f]{40}", row["git_blob_oid"]) is None
        ):
            _fail(f"recovery source Git identity changed: {path}")
        reference = {
            key: row[key] for key in ("path", "bytes", "sha256")
        }
        _validate_reference(reference, expected_path=path, name=path)
        observed = _raw_reference(workspace_fd, path, name=f"recovery source {path}")
        if reference != observed:
            _fail(f"recovery source differs from source audit: {path}")
        normalized.append(dict(row))
    if closure["set_sha256"] != _canonical_hash(normalized):
        _fail("recovery source-set hash changed")
    authority = audit["original_science_authority"]
    if not isinstance(authority, Mapping):
        _fail("source-audit original science authority is malformed")
    _exact_keys(
        authority,
        ORIGINAL_SCIENCE_AUTHORITY_KEYS,
        name="source-audit original science authority",
    )
    if (
        authority["run_id"] != RUN_ID
        or authority["canonical_config_sha256"] != ORIGINAL_CONFIG_SHA256
        or authority["execution_source_set_sha256"]
        != ORIGINAL_EXECUTION_SOURCE_SET_SHA256
        or authority["controller_source_sha256"] != ORIGINAL_CONTROLLER_SHA256
        or authority["result_completed_count"] != 126
    ):
        _fail("source-audit original science authority changed")
    prestate = audit["failed_prestate"]
    if not isinstance(prestate, Mapping):
        _fail("source-audit failed prestate is malformed")
    _exact_keys(prestate, FAILED_PRESTATE_KEYS, name="source-audit failed prestate")
    if dict(prestate) != _expected_failed_prestate():
        _fail("source-audit failed prestate changed")
    if audit["assertions"] != recovery_authority.SOURCE_AUDIT_ASSERTIONS:
        _fail("recovery source-audit assertions changed")
    return audit


def _validate_failed_tree_boundary(run_fd: int) -> None:
    expected = _expected_failed_prestate()
    if os.listdir(run_fd) != expected["top_level_members"]:
        # The filesystem order is not authority; compare sorted byte names.
        if sorted(os.listdir(run_fd), key=os.fsencode) != sorted(
            expected["top_level_members"], key=os.fsencode
        ):
            _fail("RUN003 failed top-level member set changed")
    reports_fd = frozen._open_dir(run_fd, "reports")
    try:
        if sorted(os.listdir(reports_fd), key=os.fsencode):
            _fail("RUN003 reports directory is not empty")
    finally:
        os.close(reports_fd)
    projection = _tree_projection(run_fd)
    if projection != {
        key: expected[key]
        for key in ("canonical_tree_sha256", "regular_file_count", "regular_file_bytes")
    }:
        _fail("RUN003 failed canonical tree changed")


def _validate_base_tree_after_staging(run_fd: int) -> None:
    projection = _tree_projection(run_fd, excluded_files=frozenset(OUTPUT_PATHS))
    expected = _expected_failed_prestate()
    if projection != {
        key: expected[key]
        for key in ("canonical_tree_sha256", "regular_file_count", "regular_file_bytes")
    }:
        _fail("RUN003 prestate changed outside authorized terminal files")


def _authenticate_original_science(
    workspace_fd: int,
    run_fd: int,
    *,
    original_source_evidence_commit: str,
) -> tuple[list[frozen.TaskSpec], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate the exact failed science world without requiring reports."""

    _validate_failed_tree_boundary(run_fd)
    source_set_reference = _raw_reference(
        run_fd, "runtime/source_set.json", name="RUN003 source set"
    )
    session_reference = _raw_reference(
        run_fd, "runtime/execution_session.json", name="RUN003 session"
    )
    task_index_reference = _raw_reference(
        run_fd, "task_index.json", name="RUN003 task index"
    )
    if (
        source_set_reference["sha256"] != FAILED_SOURCE_SET_RAW_SHA256
        or session_reference["sha256"] != FAILED_SESSION_RAW_SHA256
        or task_index_reference["sha256"] != FAILED_TASK_INDEX_RAW_SHA256
    ):
        _fail("RUN003 source/session/task raw authority changed")
    source_set = frozen._strict_json_at(
        run_fd, "runtime/source_set.json", name="RUN003 source set"
    )
    frozen._payload_hash(source_set, "payload_sha256", name="RUN003 source set")
    if (
        source_set.get("source_set_sha256")
        != ORIGINAL_EXECUTION_SOURCE_SET_SHA256
        or source_set.get("canonical_config_sha256") != ORIGINAL_CONFIG_SHA256
    ):
        _fail("RUN003 source-set semantic authority changed")
    _old_audit, old_set_sha256, _old_rows = frozen._load_source_authority(
        workspace_fd, original_source_evidence_commit
    )
    if old_set_sha256 != ORIGINAL_EXECUTION_SOURCE_SET_SHA256:
        _fail("RUN003 live old source authority changed")
    terminal, comparator, _parents = frozen._load_parents(run_fd)
    prepared = frozen._strict_json_at(
        run_fd, "prepared_manifest.json", name="RUN003 prepared manifest"
    )
    tasks = frozen.build_native_v2_task_grid(prepared)
    expected_task_names = [f"task_{ordinal:03d}.json" for ordinal in range(126)]
    if frozen._names(run_fd, "tasks") != expected_task_names:
        _fail("RUN003 task roster changed")
    task_references: dict[str, dict[str, Any]] = {}
    for task in tasks:
        path = f"tasks/{task.task_id}.json"
        frozen.verify_task_payload(
            frozen._strict_json_at(run_fd, path, name="RUN003 task spec"), task
        )
        task_references[task.task_id] = frozen._reference(
            run_fd, path, name="RUN003 task spec"
        )
    task_index = frozen.validate_task_index_receipt(
        frozen._strict_json_at(run_fd, "task_index.json", name="RUN003 task index"),
        tasks,
    )
    if task_index["task_spec_references"] != task_references:
        _fail("RUN003 task-index references changed")
    if frozen._names(run_fd, "runtime") != sorted(frozen.RUNTIME_MEMBERS):
        _fail("RUN003 runtime closed world changed")
    slot_assignments = frozen.validate_slot_assignments_receipt(
        frozen._strict_json_at(
            run_fd, "runtime/slot_assignments.json", name="RUN003 slots"
        ),
        tasks,
        task_index["task_assignment_sha256"],
    )
    session = frozen.validate_execution_session_receipt(
        frozen._strict_json_at(
            run_fd, "runtime/execution_session.json", name="RUN003 session"
        )
    )
    if (
        session["source_set_reference"] != source_set_reference
        or session["source_set_sha256"] != ORIGINAL_EXECUTION_SOURCE_SET_SHA256
        or session["controller_source_sha256"] != ORIGINAL_CONTROLLER_SHA256
    ):
        _fail("RUN003 execution-session source binding changed")
    prefit = frozen.validate_pre_fit_validation_receipt(
        frozen._strict_json_at(
            run_fd, "runtime/pre_fit_validation.json", name="RUN003 prefit"
        )
    )
    frozen._validate_launch_zero_world(run_fd, session, prefit)
    expected_task_files = [f"task_{ordinal:03d}.json" for ordinal in range(126)]
    if (
        frozen._names(run_fd, "runtime/outer_prediction_permits")
        or frozen._names(run_fd, "runtime/outer_label_permits")
        or frozen._names(run_fd, "quarantine")
    ):
        _fail("RUN003 permits/quarantine are not terminal")
    for relative in (
        "runtime/attempt_roster",
        "runtime/attempt_ledger",
        "runtime/outer_prediction_claims",
        "runtime/outer_label_claims",
    ):
        if frozen._names(run_fd, relative) != expected_task_files:
            _fail("RUN003 task authority universe changed")
    if frozen._names(run_fd, "results") != [task.task_id for task in tasks]:
        _fail("RUN003 result directory universe changed")
    if frozen._names(run_fd, "runtime/ledgers") != [task.task_id for task in tasks]:
        _fail("RUN003 ledger directory universe changed")
    result_completed_references: list[dict[str, Any]] = []
    for task in tasks:
        if frozen._names(run_fd, f"results/{task.task_id}") != sorted(
            frozen.RESULT_MEMBERS
        ):
            _fail("RUN003 completed result member set changed")
        if frozen._names(run_fd, f"runtime/ledgers/{task.task_id}") != sorted(
            f"{phase}.json" for phase in frozen.PHASES
        ):
            _fail("RUN003 seven-phase ledger member set changed")
        phases = [
            frozen._strict_json_at(
                run_fd,
                f"runtime/ledgers/{task.task_id}/{phase}.json",
                name="RUN003 phase",
            )
            for phase in frozen.PHASES
        ]
        frozen.validate_phase_chain(phases, task)
        for phase in phases:
            for key, reference in phase["payload"].items():
                if key.endswith("_reference"):
                    shaped = frozen._validate_reference_shape(
                        reference, name=f"RUN003 phase {key}"
                    )
                    frozen._verify_reference(
                        run_fd,
                        shaped,
                        expected_path=shaped["path"],
                        name=f"RUN003 phase {key}",
                    )
        result_completed_references.append(
            frozen._reference(
                run_fd,
                f"runtime/ledgers/{task.task_id}/result.completed.json",
                name="RUN003 result completed",
            )
        )
    frozen._validate_child_ready_world(run_fd, tasks, slot_assignments)
    authority = {
        "run_id": RUN_ID,
        "canonical_config_sha256": ORIGINAL_CONFIG_SHA256,
        "execution_source_set_sha256": ORIGINAL_EXECUTION_SOURCE_SET_SHA256,
        "controller_source_sha256": ORIGINAL_CONTROLLER_SHA256,
        "source_set_reference": source_set_reference,
        "execution_session_reference": session_reference,
        "task_index_reference": task_index_reference,
        "result_completed_reference_set_sha256": _canonical_hash(
            result_completed_references
        ),
        "result_completed_count": len(result_completed_references),
    }
    return tasks, terminal, comparator, authority


def _validate_preparation_public_source_set(
    workspace_fd: int, preparation_manifest: Mapping[str, Any]
) -> str:
    provenance = preparation_manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        _fail("preparation provenance is malformed")
    source_lock = provenance.get("contract_and_lock_sources")
    if not isinstance(source_lock, Mapping):
        _fail("preparation public source lock is malformed")
    _exact_keys(
        source_lock,
        {"files", "dependency_lock_source", "exact_dependency_lock_file_present", "source_set_sha256"},
        name="preparation public source lock",
    )
    files = source_lock["files"]
    relative_paths = tuple(
        path.removeprefix("09_SO101_IMU_VIVE_LEARNING/")
        for path in V1_PUBLIC_SOURCE_PATHS
    )
    if not isinstance(files, Mapping) or set(files) != set(relative_paths):
        _fail("preparation public source roster changed")
    normalized: dict[str, dict[str, Any]] = {}
    for relative, workspace_path in zip(
        relative_paths, V1_PUBLIC_SOURCE_PATHS, strict=True
    ):
        value = files[relative]
        if not isinstance(value, Mapping):
            _fail(f"preparation public source row is malformed: {relative}")
        _exact_keys(value, {"bytes", "sha256"}, name=f"public source {relative}")
        raw = _read_fd_path(
            workspace_fd, workspace_path, name=f"public source {relative}"
        )
        expected = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if dict(value) != expected:
            _fail(f"preparation public source bytes changed: {relative}")
        normalized[relative] = expected
    observed = _canonical_hash(normalized)
    if source_lock["source_set_sha256"] != observed:
        _fail("preparation public source-set hash changed")
    return observed


def _path_for_workspace_fd(workspace_fd: int, relative: str) -> str:
    if workspace_fd < 3 or relative.startswith("/") or ".." in relative.split("/"):
        _fail("public API path authority is malformed")
    return f"/proc/self/fd/{workspace_fd}/{relative}"


def _call_canonical_signed_reconstruction_once(
    workspace_fd: int,
    *,
    expected_public_source_set_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    """Delayed, exact one config load and one public reconstruction call."""

    from . import openaxes_external_v1 as public_api
    from . import openaxes_external_v1_config as public_config

    if not expected_public_source_set_sha256:
        _fail("public source-set authority is absent")
    config = public_config.load_openaxes_config(
        _path_for_workspace_fd(workspace_fd, V1_CONFIG_PATH)
    )
    truth, receipt = public_api.reconstruct_openaxes_signed_shift_truth_v1(
        _path_for_workspace_fd(workspace_fd, V1_SOURCE_PATH),
        config,
        _path_for_workspace_fd(workspace_fd, V1_PREPARATION_PATH),
    )
    if not isinstance(receipt, Mapping):
        _fail("canonical signed reconstruction receipt is malformed")
    source = receipt.get("source")
    prepared = receipt.get("prepared")
    if not isinstance(source, Mapping) or not isinstance(prepared, Mapping):
        _fail("canonical signed reconstruction authority is malformed")
    _exact_keys(
        source,
        {
            "filename",
            "size_bytes",
            "sha256",
            "zip_entries",
            "zip_payload_bytes",
            "crc_verified",
            "single_no_follow_fd_bound",
            "same_handle_size_sha_crc_and_reads",
            "pre_post_fstat_stable",
            "post_path_binding_verified",
        },
        name="canonical signed source receipt",
    )
    configured_source = config["source"]
    if source != {
        "filename": configured_source["filename"],
        "size_bytes": configured_source["expected_size_bytes"],
        "sha256": configured_source["expected_sha256"],
        "zip_entries": configured_source["expected_zip_entries"],
        "zip_payload_bytes": configured_source["expected_zip_payload_bytes"],
        "crc_verified": True,
        "single_no_follow_fd_bound": True,
        "same_handle_size_sha_crc_and_reads": True,
        "pre_post_fstat_stable": True,
        "post_path_binding_verified": True,
    }:
        _fail("canonical signed source receipt differs from loaded config")
    if receipt.get("recipe") != config["alignment"]["sensitivity_recipe"]:
        _fail("canonical signed recipe differs from loaded config")
    return truth, dict(receipt)


def _closed_frame_sha256(frame: Any, schema: Sequence[tuple[str, str]]) -> str:
    if tuple(frame.columns) != tuple(name for name, _kind in schema):
        _fail("canonical signed truth column schema changed")
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(frame)))
    for name, kind in schema:
        encoded_name = name.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(kind.encode("ascii"))
        if kind == "utf8":
            for value in frame[name].tolist():
                if type(value) is not str:
                    _fail(f"canonical signed truth {name} is not exact text")
                encoded = value.encode("utf-8")
                digest.update(struct.pack("<I", len(encoded)))
                digest.update(encoded)
        elif kind == "int64":
            values = frame[name].to_numpy()
            if not np.issubdtype(values.dtype, np.integer):
                _fail(f"canonical signed truth {name} is not integer")
            digest.update(np.asarray(values, dtype="<i8").tobytes(order="C"))
        elif kind == "float64":
            values = frame[name].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                _fail(f"canonical signed truth {name} is not finite")
            digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
        else:
            _fail("canonical signed truth has unknown closed-table kind")
    return digest.hexdigest()


def _base_identity_from_ensembles(
    ensembles: Mapping[tuple[str, str], Any]
) -> Any:
    frames: list[Any] = []
    for frame in ensembles.values():
        required = {
            "row_uid",
            "scenario_id",
            "sample_index",
            "aligned_time_us",
            "view_id",
        }
        if not required.issubset(frame.columns):
            _fail("seed ensemble identity columns changed")
        selected = frame.loc[
            frame["view_id"] == frozen.VIEW_IDS[0],
            ["row_uid", "scenario_id", "sample_index", "aligned_time_us"],
        ]
        frames.append(selected)
    if not frames:
        _fail("seed ensemble world is empty")
    combined = frozen.io_utils.pd.concat(frames, ignore_index=True)
    grouped = combined.drop_duplicates().copy()
    if grouped["row_uid"].duplicated().any():
        _fail("seed ensemble row_uid has conflicting base identity")
    scenario_order = {
        f"{shape}-{speed}": ordinal
        for ordinal, (shape, speed) in enumerate(
            (shape, speed)
            for shape in ("circle", "triangle")
            for speed in (100, 150, 200, 250)
        )
    }
    if not set(grouped["scenario_id"]).issubset(scenario_order):
        _fail("seed ensemble scenario identity changed")
    grouped["_scenario_ordinal"] = grouped["scenario_id"].map(scenario_order)
    return grouped.sort_values(
        ["_scenario_ordinal", "sample_index"], kind="stable"
    ).drop(columns=["_scenario_ordinal"]).reset_index(drop=True)


def validate_canonical_signed_truth(
    truth: Any,
    receipt: Any,
    *,
    base_identity: Any,
    expected_public_source_set_sha256: str,
) -> dict[str, Any]:
    """Semantically validate the public truth and return support authority."""

    if tuple(truth.columns) != SIGNED_TRUTH_COLUMNS:
        _fail("canonical signed truth schema changed")
    if len(truth) < 1 or truth.duplicated(["row_uid", "shift_ms"]).any():
        _fail("canonical signed truth row_uid/shift identity changed")
    if set(truth["shift_ms"].tolist()) != set(SHIFT_ORDER):
        _fail("canonical signed truth shift set changed")
    if not np.array_equal(
        truth["shifted_target_time_us"].to_numpy(dtype=np.int64),
        truth["base_aligned_time_us"].to_numpy(dtype=np.int64)
        + truth["shift_ms"].to_numpy(dtype=np.int64) * 1000,
    ):
        _fail("canonical signed truth shift-time identity changed")
    if (
        list(base_identity.columns)
        != ["row_uid", "scenario_id", "sample_index", "aligned_time_us"]
        or base_identity["row_uid"].duplicated().any()
    ):
        _fail("zero-shift base identity is malformed")
    base_by_uid = base_identity.set_index("row_uid", verify_integrity=True)
    if not set(truth["row_uid"]).issubset(set(base_by_uid.index)):
        _fail("canonical signed truth invented a row_uid")
    joined = truth[["row_uid", "scenario_id", "base_aligned_time_us"]].join(
        base_by_uid,
        on="row_uid",
        how="left",
        rsuffix="_base",
        validate="many_to_one",
    )
    if (
        joined["scenario_id"].tolist() != joined["scenario_id_base"].tolist()
        or joined["base_aligned_time_us"].tolist()
        != joined["aligned_time_us"].tolist()
    ):
        _fail("canonical signed truth base identity changed")
    value = receipt
    if not isinstance(value, Mapping):
        _fail("canonical signed receipt is malformed")
    _exact_keys(
        value,
        {
            "contract",
            "preparation_contract",
            "config_sha256",
            "scope",
            "source",
            "prepared",
            "recipe",
            "output",
            "deterministic_receipt",
        },
        name="canonical signed receipt",
    )
    if (
        value["contract"] != "openaxes.external.signed_shift_truth.v1"
        or value["preparation_contract"] != "openaxes.external.preparation.v1"
        or value["config_sha256"]
        != "05c3f44b58a16cebb8c0cef1338a42ae55a0fd84dd9f87db09196a7b740b32c6"
        or value["deterministic_receipt"] is not True
    ):
        _fail("canonical signed receipt identity changed")
    expected_scope = {
        "read_only_truth_reconstruction": True,
        "features_included": False,
        "model_fit_performed": False,
        "metrics_computed": False,
        "selection_performed": False,
        "claim_gate_decided": False,
        "artifact_written": False,
        "semantic_validator_required_before_model_use": True,
    }
    if value["scope"] != expected_scope:
        _fail("canonical signed receipt scope changed")
    source = value["source"]
    if not isinstance(source, Mapping):
        _fail("canonical signed source receipt is malformed")
    _exact_keys(
        source,
        {
            "filename",
            "size_bytes",
            "sha256",
            "zip_entries",
            "zip_payload_bytes",
            "crc_verified",
            "single_no_follow_fd_bound",
            "same_handle_size_sha_crc_and_reads",
            "pre_post_fstat_stable",
            "post_path_binding_verified",
        },
        name="canonical signed source receipt",
    )
    if (
        type(source["filename"]) is not str
        or not source["filename"]
        or type(source["size_bytes"]) is not int
        or source["size_bytes"] < 1
        or type(source["zip_entries"]) is not int
        or source["zip_entries"] < 1
        or type(source["zip_payload_bytes"]) is not int
        or source["zip_payload_bytes"] < 1
        or any(
            source[key] is not True
            for key in (
                "crc_verified",
                "single_no_follow_fd_bound",
                "same_handle_size_sha_crc_and_reads",
                "pre_post_fstat_stable",
                "post_path_binding_verified",
            )
        )
    ):
        _fail("canonical signed source receipt semantics changed")
    _hex64(source["sha256"], name="canonical signed source hash")
    prepared = value["prepared"]
    if not isinstance(prepared, Mapping):
        _fail("canonical signed prepared receipt is malformed")
    _exact_keys(
        prepared,
        {
            "manifest_sha256",
            "parquet_sha256",
            "complete_sha256",
            "complete_manifest_sha256",
            "complete_parquet_sha256",
            "base_rows",
            "base_identity_sha256",
            "base_exact_source_replay",
            "stage",
        },
        name="canonical signed prepared receipt",
    )
    for key in (
        "manifest_sha256",
        "parquet_sha256",
        "complete_sha256",
        "complete_manifest_sha256",
        "complete_parquet_sha256",
        "base_identity_sha256",
    ):
        _hex64(prepared[key], name=f"canonical signed prepared {key}")
    if (
        prepared["complete_manifest_sha256"] != prepared["manifest_sha256"]
        or prepared["complete_parquet_sha256"] != prepared["parquet_sha256"]
        or not isinstance(prepared["stage"], Mapping)
        or not prepared["stage"]
        or prepared["base_exact_source_replay"] is not True
    ):
        _fail("canonical signed prepared receipt changed")
    if prepared.get("base_rows") != len(base_identity):
        _fail("canonical signed base row count changed")
    base_identity_schema = (
        ("row_uid", "utf8"),
        ("scenario_id", "utf8"),
        ("sample_index", "int64"),
        ("aligned_time_us", "int64"),
    )
    if prepared.get("base_identity_sha256") != _closed_frame_sha256(
        base_identity, base_identity_schema
    ):
        _fail("canonical signed base identity hash changed")
    recipe = value["recipe"]
    if not isinstance(recipe, Mapping):
        _fail("canonical signed recipe is malformed")
    if (
        recipe.get("signed_target_shifts_ms") != list(SHIFT_ORDER)
        or recipe.get("no_extrapolation") is not True
        or recipe.get("edge_policy")
        != "drop_base_rows_without_shifted_target_support"
        or recipe.get("pairing_key") != "row_uid"
        or recipe.get("same_base_uid_pairing_required") is not True
        or recipe.get("model_selection_use_allowed") is not False
        or recipe.get("claim_gate_use_allowed") is not False
    ):
        _fail("canonical signed recipe semantics changed")
    output = value["output"]
    if not isinstance(output, Mapping):
        _fail("canonical signed output receipt is malformed")
    _exact_keys(
        output,
        {
            "closed_schema",
            "row_order",
            "target_columns_only_with_pairing_metadata",
            "rows",
            "rows_per_shift",
            "dropped_edge_rows_by_scenario_and_shift",
            "long_table_sha256",
        },
        name="canonical signed output receipt",
    )
    if (
        output["closed_schema"] != list(SIGNED_TRUTH_COLUMNS)
        or output["row_order"] != "scenario_then_signed_shift_then_base_time"
        or output["target_columns_only_with_pairing_metadata"] is not True
        or output["rows"] != len(truth)
    ):
        _fail("canonical signed output identity changed")
    truth_schema = tuple(
        zip(
            SIGNED_TRUTH_COLUMNS,
            ("utf8", "utf8", "int64", "int64", "int64", "float64", "float64", "float64"),
            strict=True,
        )
    )
    if output["long_table_sha256"] != _closed_frame_sha256(truth, truth_schema):
        _fail("canonical signed long-table hash changed")
    rows_per_shift = {
        str(shift): int((truth["shift_ms"] == shift).sum()) for shift in SHIFT_ORDER
    }
    if output["rows_per_shift"] != rows_per_shift:
        _fail("canonical signed per-shift row counts changed")
    base_counts = {
        str(key): int(value)
        for key, value in base_identity.groupby("scenario_id", sort=False).size().items()
    }
    observed_drops: dict[str, dict[str, int]] = {}
    for scenario_id, base_rows in base_counts.items():
        observed_drops[scenario_id] = {}
        for shift in SHIFT_ORDER:
            supported = int(
                (
                    (truth["scenario_id"] == scenario_id)
                    & (truth["shift_ms"] == shift)
                ).sum()
            )
            observed_drops[scenario_id][str(shift)] = base_rows - supported
    if output["dropped_edge_rows_by_scenario_and_shift"] != observed_drops:
        _fail("canonical signed dropped-edge counts changed")
    if not expected_public_source_set_sha256:
        _fail("canonical signed public source-set binding is absent")
    return {
        "receipt_sha256": _canonical_hash(value),
        "long_table_sha256": output["long_table_sha256"],
        "public_source_set_sha256": expected_public_source_set_sha256,
        "dropped_edge_rows_by_scenario_and_shift": observed_drops,
        "total_dropped_edges": sum(
            count for row in observed_drops.values() for count in row.values()
        ),
        "rows": len(truth),
    }


def aggregate_signed_shift_recovery(
    tasks: Sequence[frozen.TaskSpec],
    ensembles: Mapping[tuple[str, str], Any],
    truth: Any,
    receipt: Mapping[str, Any],
    *,
    expected_public_source_set_sha256: str,
) -> dict[str, Any]:
    base_identity = _base_identity_from_ensembles(ensembles)
    authority = validate_canonical_signed_truth(
        truth,
        receipt,
        base_identity=base_identity,
        expected_public_source_set_sha256=expected_public_source_set_sha256,
    )
    fold_order = [
        task.fold_id
        for task in tasks
        if task.candidate_ordinal == 0 and task.seed_ordinal == 0
    ]
    if len(fold_order) != 14 or len(set(fold_order)) != 14:
        _fail("signed recovery fold order changed")
    lookup = truth.set_index(["row_uid", "shift_ms"], verify_integrity=True)
    support_cells: list[dict[str, Any]] = []
    for candidate_id in frozen.CANDIDATE_IDS:
        for fold_id in fold_order:
            frame = ensembles[(candidate_id, fold_id)]
            base_uids = list(
                dict.fromkeys(
                    frame.loc[frame["view_id"] == frozen.VIEW_IDS[0], "row_uid"].tolist()
                )
            )
            if not base_uids:
                _fail("signed recovery fold support is empty")
            for shift_ms in SHIFT_ORDER:
                supported_uids = [
                    uid for uid in base_uids if (uid, shift_ms) in lookup.index
                ]
                support_cells.append(
                    {
                        "candidate_id": candidate_id,
                        "fold_id": fold_id,
                        "shift_ms": shift_ms,
                        "base_rows": len(base_uids),
                        "supported_rows": len(supported_uids),
                        "dropped_edges": len(base_uids) - len(supported_uids),
                        "base_uid_support_sha256": _canonical_hash(base_uids),
                        "signed_uid_support_sha256": _canonical_hash(supported_uids),
                    }
                )
    if len(support_cells) != 252:
        _fail("signed recovery support cell grid changed")
    common = {
        "contract": SIGNED_REPORT_CONTRACT,
        "shift_order_ms": list(SHIFT_ORDER),
        "support_cells": support_cells,
        "canonical_truth_authority": authority,
        "candidate_fold_shift_support_cells": 252,
        "canonical_public_truth_reconstruction_calls": 1,
        "scientific_fits": 0,
        "model_forward_calls": 0,
        "task_evaluator_calls": 0,
        "may_select_or_change_primary_gate": False,
    }
    if authority["total_dropped_edges"] or any(
        cell["dropped_edges"] for cell in support_cells
    ):
        return {
            **common,
            "status": "incomplete",
            "incomplete_reason": "nonzero_dropped_edge_support",
            "rows": [],
            "primary_robust_J": [],
            "candidate_fold_shift_ensemble_cells": 0,
            "candidate_fold_shift_view_metric_cells": 0,
        }
    rows: list[dict[str, Any]] = []
    for candidate_id in frozen.CANDIDATE_IDS:
        for fold_id in fold_order:
            frame = ensembles[(candidate_id, fold_id)]
            for shift_ms in SHIFT_ORDER:
                for view_id in frozen.VIEW_IDS:
                    selected = frame["view_id"] == view_id
                    selected_frame = frame.loc[selected]
                    uids = selected_frame["row_uid"].tolist()
                    shifted_rows = lookup.loc[
                        [(uid, shift_ms) for uid in uids]
                    ]
                    if (
                        shifted_rows["scenario_id"].tolist()
                        != selected_frame["scenario_id"].tolist()
                        or shifted_rows["base_aligned_time_us"].tolist()
                        != selected_frame["aligned_time_us"].tolist()
                    ):
                        _fail("signed recovery joined truth identity changed")
                    prediction = selected_frame[
                        ["prediction_x_m", "prediction_y_m", "prediction_z_m"]
                    ].to_numpy(dtype="float64", copy=True)
                    shifted = shifted_rows[
                        ["shifted_tcp_x_m", "shifted_tcp_y_m", "shifted_tcp_z_m"]
                    ].to_numpy(dtype="float64", copy=True)
                    zero = frozen._context_metric(
                        frame, fold_id=fold_id, view_id=view_id
                    )["rmse3d_m"]
                    metric = native_training.fixed_rmse3d_m(
                        prediction, shifted, name="signed recovery context"
                    )
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "fold_id": fold_id,
                            "shift_ms": shift_ms,
                            "view_id": view_id,
                            "rows": len(prediction),
                            "rmse3d_m": metric,
                            "shift_minus_zero_rmse3d_m": metric - zero,
                            "dropped_edges": 0,
                            "uid_support_sha256": _canonical_hash(uids),
                        }
                    )
    if len(rows) != 1260:
        _fail("signed recovery metric cell grid changed")
    primary_folds = fold_order[:8]
    primary_robust: list[dict[str, Any]] = []
    for candidate_id in frozen.CANDIDATE_IDS:
        for shift_ms in SHIFT_ORDER:
            metrics = [
                row["rmse3d_m"]
                for fold_id in primary_folds
                for view_id in frozen.OUTAGE_VIEW_IDS
                for row in rows
                if row["candidate_id"] == candidate_id
                and row["fold_id"] == fold_id
                and row["shift_ms"] == shift_ms
                and row["view_id"] == view_id
            ]
            if len(metrics) != 32:
                _fail("signed recovery robust context grid changed")
            fold_scores = [
                native_training.fixed_mean_float64(
                    metrics[offset : offset + 4], name="signed recovery fold"
                )
                for offset in range(0, 32, 4)
            ]
            mask_scores = [
                native_training.fixed_mean_float64(
                    metrics[index::4], name="signed recovery mask"
                )
                for index in range(4)
            ]
            primary_robust.append(
                {
                    "candidate_id": candidate_id,
                    "shift_ms": shift_ms,
                    "robust_J_m": native_training.fixed_mean_float64(
                        [
                            native_training.fixed_mean_float64(
                                metrics, name="signed recovery mean"
                            ),
                            max(fold_scores),
                            max(mask_scores),
                        ],
                        name="signed recovery robust J",
                    ),
                }
            )
    return {
        **common,
        "status": "complete",
        "incomplete_reason": None,
        "rows": rows,
        "primary_robust_J": primary_robust,
        "candidate_fold_shift_ensemble_cells": 252,
        "candidate_fold_shift_view_metric_cells": 1260,
    }


def validate_signed_shift_recovery_report(value: Any) -> dict[str, Any]:
    report = _validate_payload(value, name="signed recovery report")
    _exact_keys(
        report,
        {
            "contract",
            "status",
            "incomplete_reason",
            "shift_order_ms",
            "support_cells",
            "canonical_truth_authority",
            "rows",
            "primary_robust_J",
            "candidate_fold_shift_support_cells",
            "candidate_fold_shift_ensemble_cells",
            "candidate_fold_shift_view_metric_cells",
            "canonical_public_truth_reconstruction_calls",
            "scientific_fits",
            "model_forward_calls",
            "task_evaluator_calls",
            "may_select_or_change_primary_gate",
            "payload_sha256",
        },
        name="signed recovery report",
    )
    if (
        report["contract"] != SIGNED_REPORT_CONTRACT
        or report["shift_order_ms"] != list(SHIFT_ORDER)
        or report["candidate_fold_shift_support_cells"] != 252
        or report["canonical_public_truth_reconstruction_calls"] != 1
        or report["scientific_fits"] != 0
        or report["model_forward_calls"] != 0
        or report["task_evaluator_calls"] != 0
        or report["may_select_or_change_primary_gate"] is not False
    ):
        _fail("signed recovery report identity/accounting changed")
    support = report["support_cells"]
    if type(support) is not list or len(support) != 252:
        _fail("signed recovery support grid changed")
    identities: set[tuple[str, str, int]] = set()
    for cell in support:
        if not isinstance(cell, Mapping):
            _fail("signed recovery support cell is malformed")
        _exact_keys(
            cell,
            {
                "candidate_id",
                "fold_id",
                "shift_ms",
                "base_rows",
                "supported_rows",
                "dropped_edges",
                "base_uid_support_sha256",
                "signed_uid_support_sha256",
            },
            name="signed recovery support cell",
        )
        identity = (cell["candidate_id"], cell["fold_id"], cell["shift_ms"])
        if identity in identities:
            _fail("signed recovery support identity is duplicated")
        identities.add(identity)
        if (
            cell["candidate_id"] not in frozen.CANDIDATE_IDS
            or cell["shift_ms"] not in SHIFT_ORDER
            or type(cell["base_rows"]) is not int
            or type(cell["supported_rows"]) is not int
            or type(cell["dropped_edges"]) is not int
            or cell["base_rows"] < 1
            or cell["supported_rows"] < 0
            or cell["dropped_edges"] < 0
            or cell["supported_rows"] + cell["dropped_edges"] != cell["base_rows"]
        ):
            _fail("signed recovery support accounting changed")
        _hex64(cell["base_uid_support_sha256"], name="base UID support")
        _hex64(cell["signed_uid_support_sha256"], name="signed UID support")
    authority = report["canonical_truth_authority"]
    if not isinstance(authority, Mapping):
        _fail("signed canonical truth authority is malformed")
    _exact_keys(
        authority,
        {
            "receipt_sha256",
            "long_table_sha256",
            "public_source_set_sha256",
            "dropped_edge_rows_by_scenario_and_shift",
            "total_dropped_edges",
            "rows",
        },
        name="signed canonical truth authority",
    )
    for key in ("receipt_sha256", "long_table_sha256", "public_source_set_sha256"):
        _hex64(authority[key], name=f"signed truth {key}")
    dropped = authority["dropped_edge_rows_by_scenario_and_shift"]
    if (
        not isinstance(dropped, Mapping)
        or type(authority["total_dropped_edges"]) is not int
        or authority["total_dropped_edges"] < 0
        or type(authority["rows"]) is not int
        or authority["rows"] < 1
    ):
        _fail("signed canonical truth support authority changed")
    dropped_total = 0
    for scenario_id, by_shift in dropped.items():
        if type(scenario_id) is not str or not scenario_id or not isinstance(
            by_shift, Mapping
        ) or set(by_shift) != {str(shift) for shift in SHIFT_ORDER}:
            _fail("signed canonical truth dropped-edge roster changed")
        if any(type(count) is not int or count < 0 for count in by_shift.values()):
            _fail("signed canonical truth dropped-edge count changed")
        dropped_total += sum(by_shift.values())
    if dropped_total != authority["total_dropped_edges"]:
        _fail("signed canonical truth dropped-edge total changed")
    support_drops = sum(cell["dropped_edges"] for cell in support)
    status = report["status"]
    if status == "complete":
        if (
            report["incomplete_reason"] is not None
            or authority["total_dropped_edges"] != 0
            or support_drops != 0
            or type(report["rows"]) is not list
            or len(report["rows"]) != 1260
            or type(report["primary_robust_J"]) is not list
            or len(report["primary_robust_J"]) != 18
            or report["candidate_fold_shift_ensemble_cells"] != 252
            or report["candidate_fold_shift_view_metric_cells"] != 1260
        ):
            _fail("complete signed recovery report changed")
        metric_identities = {
            (row.get("candidate_id"), row.get("fold_id"), row.get("shift_ms"), row.get("view_id"))
            for row in report["rows"]
            if isinstance(row, Mapping)
        }
        if len(metric_identities) != 1260:
            _fail("complete signed metric identities changed")
        for row in report["rows"]:
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "candidate_id",
                    "fold_id",
                    "shift_ms",
                    "view_id",
                    "rows",
                    "rmse3d_m",
                    "shift_minus_zero_rmse3d_m",
                    "dropped_edges",
                    "uid_support_sha256",
                }
                or row.get("candidate_id") not in frozen.CANDIDATE_IDS
                or row.get("shift_ms") not in SHIFT_ORDER
                or row.get("view_id") not in frozen.VIEW_IDS
                or type(row.get("rows")) is not int
                or row["rows"] < 1
                or row.get("dropped_edges") != 0
                or type(row.get("rmse3d_m")) is not float
                or not np.isfinite(row["rmse3d_m"])
                or type(row.get("shift_minus_zero_rmse3d_m")) is not float
                or not np.isfinite(row["shift_minus_zero_rmse3d_m"])
            ):
                _fail("complete signed metric row changed")
            _hex64(row["uid_support_sha256"], name="signed metric UID support")
        robust_identities: set[tuple[str, int]] = set()
        for row in report["primary_robust_J"]:
            if (
                not isinstance(row, Mapping)
                or set(row) != {"candidate_id", "shift_ms", "robust_J_m"}
                or row.get("candidate_id") not in frozen.CANDIDATE_IDS
                or row.get("shift_ms") not in SHIFT_ORDER
                or type(row.get("robust_J_m")) is not float
                or not np.isfinite(row["robust_J_m"])
            ):
                _fail("complete signed robust row changed")
            robust_identities.add((row["candidate_id"], row["shift_ms"]))
        if robust_identities != {
            (candidate_id, shift)
            for candidate_id in frozen.CANDIDATE_IDS
            for shift in SHIFT_ORDER
        }:
            _fail("complete signed robust identities changed")
    elif status == "incomplete":
        if (
            report["incomplete_reason"] != "nonzero_dropped_edge_support"
            or authority["total_dropped_edges"] <= 0
            or support_drops <= 0
            or report["rows"] != []
            or report["primary_robust_J"] != []
            or report["candidate_fold_shift_ensemble_cells"] != 0
            or report["candidate_fold_shift_view_metric_cells"] != 0
        ):
            _fail("incomplete signed recovery report leaked partial metrics")
    else:
        _fail("signed recovery report status changed")
    return report


def _report_reference_from_value(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    return _reference_for_raw(f"reports/{name}", _canonical_bytes(value))


def _build_terminal_payloads(
    run_fd: int,
    tasks: Sequence[frozen.TaskSpec],
    zero: Mapping[str, Any],
    gates: Mapping[str, Any],
    stress: Mapping[str, Any],
    signed: Mapping[str, Any],
    *,
    recovery_source_audit_reference: Mapping[str, Any],
    diagnostic_reference: Mapping[str, Any],
) -> dict[str, bytes]:
    reports: dict[str, dict[str, Any]] = {
        "zero_shift.json": _payload(zero),
        "primary_gates.json": _payload(gates),
        "stress_report.json": _payload(stress),
        "signed_shift_report.json": _payload(signed),
    }
    report_references = {
        name: _report_reference_from_value(name, value)
        for name, value in reports.items()
    }
    summary = _payload(
        {
            "contract": (
                "openaxes.external.native_sensor_experts.v2.development_summary.v1"
            ),
            "run_id": RUN_ID,
            "task_count": 126,
            "candidate_count": 3,
            "fold_count": 14,
            "seed_count": 3,
            "optimizer_updates": 50_400,
            "all_model_forward_calls": 100_926,
            "outer_label_reads": 126,
            "task_evaluator_calls": 126,
            "signed_truth_reconstruction_calls": 1,
            "signed_report_status": signed["status"],
            "report_references": report_references,
            "recovery_source_audit_reference": dict(
                recovery_source_audit_reference
            ),
            "diagnostic_receipt_reference": dict(diagnostic_reference),
            "action_accounting": _expected_action_accounting(),
            "compound_terminal_required_for_publication": True,
        }
    )
    reports["development_summary.json"] = summary
    report_references["development_summary.json"] = _report_reference_from_value(
        "development_summary.json", summary
    )
    result_references = [
        frozen._reference(
            run_fd,
            f"runtime/ledgers/{task.task_id}/result.completed.json",
            name="RUN003 result completed",
        )
        for task in tasks
    ]
    development = _payload(
        {
            "contract": (
                "openaxes.external.native_sensor_experts.v2.development_manifest.v1"
            ),
            "schema_version": 1,
            "run_id": RUN_ID,
            "canonical_config_sha256": ORIGINAL_CONFIG_SHA256,
            "task_id_order": [task.task_id for task in tasks],
            "result_completed_phase_references": result_references,
            "result_completed_phase_set_sha256": _canonical_hash(result_references),
            "report_references": report_references,
            "source_set_reference": frozen._reference(
                run_fd, "runtime/source_set.json", name="RUN003 source set"
            ),
            "recovery_source_audit_reference": dict(
                recovery_source_audit_reference
            ),
            "diagnostic_receipt_reference": dict(diagnostic_reference),
            "closed_task_count": 126,
            "quarantine_member_count": 0,
            "compound_terminal_required_for_publication": True,
        }
    )
    development_reference = _reference_for_raw(
        "development_manifest.json", _canonical_bytes(development)
    )
    complete = _payload(
        {
            "contract": (
                "openaxes.external.native_sensor_experts.v2.complete.v1"
            ),
            "schema_version": 1,
            "run_id": RUN_ID,
            "created_at_utc": _now_utc(),
            "development_manifest_reference": development_reference,
            "source_set_reference": development["source_set_reference"],
            "recovery_source_audit_reference": dict(
                recovery_source_audit_reference
            ),
            "closed_task_count": 126,
            "quarantine_member_count": 0,
            "scientific_values_disclosed": False,
            "compound_terminal_required_for_publication": True,
        }
    )
    values: dict[str, Mapping[str, Any]] = {
        "reports/zero_shift.json": reports["zero_shift.json"],
        "reports/primary_gates.json": reports["primary_gates.json"],
        "reports/stress_report.json": reports["stress_report.json"],
        "reports/signed_shift_report.json": reports["signed_shift_report.json"],
        "reports/development_summary.json": summary,
        "development_manifest.json": development,
        "COMPLETE.json": complete,
    }
    return {path: _canonical_bytes(values[path]) for path in OUTPUT_PATHS}


def _validate_zero_report(value: Any) -> dict[str, Any]:
    report = _validate_payload(value, name="zero-shift report")
    _exact_keys(
        report,
        {
            "contract",
            "candidate_order",
            "fold_order",
            "candidate_fold_records",
            "candidate_primary",
            "scientific_fits",
            "outer_model_forward_calls",
            "outer_label_reads",
            "task_evaluator_calls",
            "payload_sha256",
        },
        name="zero-shift report",
    )
    if (
        report.get("contract")
        != "openaxes.external.native_sensor_experts.v2.zero_shift.v1"
        or report.get("candidate_order") != list(frozen.CANDIDATE_IDS)
        or type(report.get("fold_order")) is not list
        or len(report["fold_order"]) != 14
        or type(report.get("candidate_fold_records")) is not list
        or len(report["candidate_fold_records"]) != 42
        or type(report.get("candidate_primary")) is not list
        or len(report["candidate_primary"]) != 3
        or report.get("scientific_fits") != 126
        or report.get("outer_model_forward_calls") != 126
        or report.get("outer_label_reads") != 126
        or report.get("task_evaluator_calls") != 126
    ):
        _fail("zero-shift recovery report semantics changed")
    fold_order = report["fold_order"]
    if (
        len(set(fold_order)) != 14
        or any(type(fold_id) is not str or not fold_id for fold_id in fold_order)
    ):
        _fail("zero-shift fold authority changed")
    records = report["candidate_fold_records"]
    identities: set[tuple[str, str]] = set()
    family_by_fold: dict[str, str] = {}
    for row in records:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"candidate_id", "fold_id", "fold_family", "contexts"}
            or row.get("candidate_id") not in frozen.CANDIDATE_IDS
            or row.get("fold_id") not in fold_order
            or row.get("fold_family") not in frozen.FOLD_FAMILIES
            or type(row.get("contexts")) is not list
            or any(not isinstance(context, Mapping) for context in row["contexts"])
            or [context.get("view_id") for context in row["contexts"]]
            != list(frozen.VIEW_IDS)
        ):
            _fail("zero-shift candidate/fold record changed")
        identity = (row["candidate_id"], row["fold_id"])
        if identity in identities:
            _fail("zero-shift candidate/fold identity is duplicated")
        identities.add(identity)
        previous = family_by_fold.setdefault(row["fold_id"], row["fold_family"])
        if previous != row["fold_family"]:
            _fail("zero-shift fold family changed across candidates")
    if identities != {
        (candidate_id, fold_id)
        for candidate_id in frozen.CANDIDATE_IDS
        for fold_id in fold_order
    }:
        _fail("zero-shift candidate/fold grid changed")
    primary = report["candidate_primary"]
    if [row.get("candidate_id") for row in primary if isinstance(row, Mapping)] != list(
        frozen.CANDIDATE_IDS
    ):
        _fail("zero-shift candidate-primary order changed")
    return report


def _validate_gate_report(value: Any) -> dict[str, Any]:
    report = _validate_payload(value, name="primary gate report")
    _exact_keys(
        report,
        {
            "contract",
            "contrast_order",
            "contrasts",
            "all_three_contrasts_pass",
            "gate_evaluation_count",
            "payload_sha256",
        },
        name="primary gate report",
    )
    contrasts = report.get("contrasts")
    if (
        report.get("contract")
        != "openaxes.external.native_sensor_experts.v2.primary_gates.v1"
        or report.get("contrast_order")
        != ["A_vs_frozen_v1_masked_erm", "B_vs_A", "C_vs_B"]
        or type(contrasts) is not list
        or len(contrasts) != 3
        or report.get("gate_evaluation_count") != 1
        or report.get("all_three_contrasts_pass")
        is not all(row.get("passed") is True for row in contrasts)
    ):
        _fail("primary gate recovery report semantics changed")
    expected_pairs = (
        (
            "A_vs_frozen_v1_masked_erm",
            frozen.CANDIDATE_IDS[0],
            "receipt_bound_frozen_v1_masked_erm",
        ),
        ("B_vs_A", frozen.CANDIDATE_IDS[1], frozen.CANDIDATE_IDS[0]),
        ("C_vs_B", frozen.CANDIDATE_IDS[2], frozen.CANDIDATE_IDS[1]),
    )
    for row, expected_pair in zip(contrasts, expected_pairs, strict=True):
        requirements = row.get("requirements") if isinstance(row, Mapping) else None
        if (
            not isinstance(requirements, Mapping)
            or set(row)
            != {
                "contrast",
                "challenger",
                "comparator",
                "fold_outage_strict_wins",
                "per_seed_J_strict_wins",
                "requirements",
                "passed",
            }
            or (row["contrast"], row["challenger"], row["comparator"])
            != expected_pair
            or type(row["fold_outage_strict_wins"]) is not int
            or not 0 <= row["fold_outage_strict_wins"] <= 8
            or type(row["per_seed_J_strict_wins"]) is not int
            or not 0 <= row["per_seed_J_strict_wins"] <= 3
            or set(requirements)
            != {
                "robust_J_strict",
                "worst_mask_strict",
                "fold_outage_strict_wins_at_least_six",
                "per_seed_J_strict_wins_at_least_two",
                "full_F_nonharm",
            }
            or any(type(value) is not bool for value in requirements.values())
            or row.get("passed") is not all(requirements.values())
        ):
            _fail("primary gate contrast semantics changed")
    return report


def _validate_stress_report(value: Any) -> dict[str, Any]:
    report = _validate_payload(value, name="stress report")
    _exact_keys(
        report,
        {
            "contract",
            "families",
            "rows",
            "may_select_or_change_primary_gate",
            "payload_sha256",
        },
        name="stress report",
    )
    if (
        report.get("contract")
        != "openaxes.external.native_sensor_experts.v2.stress_report.v1"
        or report.get("families") != ["speed_pair_out", "shape_out"]
        or type(report.get("rows")) is not list
        or len(report["rows"]) != 90
        or report.get("may_select_or_change_primary_gate") is not False
    ):
        _fail("stress recovery report semantics changed")
    return report


def validate_staged_terminal_files(
    reader: Mapping[str, bytes],
    *,
    recovery_source_audit_reference: Mapping[str, Any],
    diagnostic_reference: Mapping[str, Any],
) -> dict[str, Any]:
    if set(reader) != set(OUTPUT_PATHS):
        _fail("staged terminal file roster changed")
    values = {
        path: frozen._strict_json(raw, name=f"staged {path}")
        for path, raw in reader.items()
    }
    zero = _validate_zero_report(values["reports/zero_shift.json"])
    gates = _validate_gate_report(values["reports/primary_gates.json"])
    stress = _validate_stress_report(values["reports/stress_report.json"])
    signed = validate_signed_shift_recovery_report(
        values["reports/signed_shift_report.json"]
    )
    fold_order = zero["fold_order"]
    stress_folds = {
        row["fold_id"]
        for row in zero["candidate_fold_records"]
        if row["fold_family"] in ("speed_pair_out", "shape_out")
    }
    stress_identities = {
        (row.get("candidate_id"), row.get("fold_id"), row.get("view_id"))
        for row in stress["rows"]
        if isinstance(row, Mapping)
    }
    if stress_identities != {
        (candidate_id, fold_id, view_id)
        for candidate_id in frozen.CANDIDATE_IDS
        for fold_id in stress_folds
        for view_id in frozen.VIEW_IDS
    }:
        _fail("stress grid differs from zero-shift fold authority")
    expected_support_identities = {
        (candidate_id, fold_id, shift)
        for candidate_id in frozen.CANDIDATE_IDS
        for fold_id in fold_order
        for shift in SHIFT_ORDER
    }
    if {
        (row["candidate_id"], row["fold_id"], row["shift_ms"])
        for row in signed["support_cells"]
    } != expected_support_identities:
        _fail("signed support grid differs from zero-shift fold authority")
    if signed["status"] == "complete" and {
        (row["candidate_id"], row["fold_id"], row["shift_ms"], row["view_id"])
        for row in signed["rows"]
    } != {
        (candidate_id, fold_id, shift, view_id)
        for candidate_id in frozen.CANDIDATE_IDS
        for fold_id in fold_order
        for shift in SHIFT_ORDER
        for view_id in frozen.VIEW_IDS
    }:
        _fail("signed metric grid differs from zero-shift fold authority")
    summary = _validate_payload(
        values["reports/development_summary.json"], name="development summary"
    )
    _exact_keys(
        summary,
        {
            "contract",
            "run_id",
            "task_count",
            "candidate_count",
            "fold_count",
            "seed_count",
            "optimizer_updates",
            "all_model_forward_calls",
            "outer_label_reads",
            "task_evaluator_calls",
            "signed_truth_reconstruction_calls",
            "signed_report_status",
            "report_references",
            "recovery_source_audit_reference",
            "diagnostic_receipt_reference",
            "action_accounting",
            "compound_terminal_required_for_publication",
            "payload_sha256",
        },
        name="development summary",
    )
    if (
        summary.get("contract")
        != "openaxes.external.native_sensor_experts.v2.development_summary.v1"
        or summary.get("run_id") != RUN_ID
        or summary.get("task_count") != 126
        or summary.get("signed_report_status") != signed["status"]
        or summary.get("action_accounting") != _expected_action_accounting()
        or summary.get("recovery_source_audit_reference")
        != dict(recovery_source_audit_reference)
        or summary.get("diagnostic_receipt_reference")
        != dict(diagnostic_reference)
        or summary.get("compound_terminal_required_for_publication") is not True
    ):
        _fail("compound development summary changed")
    expected_report_references = {
        name: _reference_for_raw(f"reports/{name}", reader[f"reports/{name}"])
        for name in REPORT_NAMES[:4]
    }
    if summary.get("report_references") != expected_report_references:
        _fail("development summary report references changed")
    development = _validate_payload(
        values["development_manifest.json"], name="development manifest"
    )
    _exact_keys(
        development,
        {
            "contract",
            "schema_version",
            "run_id",
            "canonical_config_sha256",
            "task_id_order",
            "result_completed_phase_references",
            "result_completed_phase_set_sha256",
            "report_references",
            "source_set_reference",
            "recovery_source_audit_reference",
            "diagnostic_receipt_reference",
            "closed_task_count",
            "quarantine_member_count",
            "compound_terminal_required_for_publication",
            "payload_sha256",
        },
        name="development manifest",
    )
    if (
        development.get("contract")
        != "openaxes.external.native_sensor_experts.v2.development_manifest.v1"
        or development.get("run_id") != RUN_ID
        or development.get("canonical_config_sha256") != ORIGINAL_CONFIG_SHA256
        or development.get("closed_task_count") != 126
        or development.get("quarantine_member_count") != 0
        or development.get("task_id_order")
        != [f"task_{ordinal:03d}" for ordinal in range(126)]
        or development.get("recovery_source_audit_reference")
        != dict(recovery_source_audit_reference)
        or development.get("diagnostic_receipt_reference")
        != dict(diagnostic_reference)
        or development.get("compound_terminal_required_for_publication") is not True
    ):
        _fail("compound development manifest changed")
    all_report_references = {
        name: _reference_for_raw(f"reports/{name}", reader[f"reports/{name}"])
        for name in REPORT_NAMES
    }
    if development.get("report_references") != all_report_references:
        _fail("development manifest report references changed")
    result_refs = development.get("result_completed_phase_references")
    if (
        type(result_refs) is not list
        or len(result_refs) != 126
        or development.get("result_completed_phase_set_sha256")
        != _canonical_hash(result_refs)
    ):
        _fail("development manifest result references changed")
    for ordinal, reference in enumerate(result_refs):
        _validate_reference(
            reference,
            expected_path=(
                f"runtime/ledgers/task_{ordinal:03d}/result.completed.json"
            ),
            name="development result-completed phase",
        )
    source_set_reference = _validate_reference(
        development.get("source_set_reference"),
        expected_path="runtime/source_set.json",
        name="development source set",
    )
    complete = _validate_payload(values["COMPLETE.json"], name="complete marker")
    _exact_keys(
        complete,
        {
            "contract",
            "schema_version",
            "run_id",
            "created_at_utc",
            "development_manifest_reference",
            "source_set_reference",
            "recovery_source_audit_reference",
            "closed_task_count",
            "quarantine_member_count",
            "scientific_values_disclosed",
            "compound_terminal_required_for_publication",
            "payload_sha256",
        },
        name="complete marker",
    )
    if (
        complete.get("contract")
        != "openaxes.external.native_sensor_experts.v2.complete.v1"
        or complete.get("run_id") != RUN_ID
        or complete.get("closed_task_count") != 126
        or complete.get("quarantine_member_count") != 0
        or complete.get("scientific_values_disclosed") is not False
        or complete.get("compound_terminal_required_for_publication") is not True
        or complete.get("recovery_source_audit_reference")
        != dict(recovery_source_audit_reference)
        or complete.get("source_set_reference") != source_set_reference
        or complete.get("development_manifest_reference")
        != _reference_for_raw("development_manifest.json", reader["development_manifest.json"])
    ):
        _fail("compound complete marker changed")
    if gates.get("gate_evaluation_count") != 1 or signed.get(
        "may_select_or_change_primary_gate"
    ) is not False:
        _fail("signed report changed primary gate authority")
    return {
        "signed_report_status": signed["status"],
        "zero": zero,
        "gates": gates,
        "stress": stress,
        "signed": signed,
        "summary": summary,
        "development": development,
        "complete": complete,
    }


def _stage_output_references(stage_fd: int) -> list[dict[str, Any]]:
    return [
        _raw_reference(stage_fd, path, name=f"staged output {path}")
        for path in OUTPUT_PATHS
    ]


def _read_stage_outputs(stage_fd: int) -> dict[str, bytes]:
    return {
        path: _read_fd_path(stage_fd, path, name=f"staged output {path}")
        for path in OUTPUT_PATHS
    }


def _stage_member_set_exact(stage_fd: int) -> None:
    if frozen._names(stage_fd) != sorted(
        ["COMPLETE.json", "STAGE_SEAL.json", "development_manifest.json", "reports"]
    ):
        _fail("sealed recovery stage root member set changed")
    if frozen._names(stage_fd, "reports") != sorted(REPORT_NAMES):
        _fail("sealed recovery stage report member set changed")


def _load_source_audit_from_raw(
    raw: bytes,
    *,
    workspace_fd: int,
    erratum_reference: Mapping[str, Any],
    diagnostic_reference: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        parsed = frozen._strict_json(raw, name="recovery source audit")
    except frozen.OpenAxesExternalNativeV2Error as exc:
        raise OpenAxesNativeV2RecoveryError(str(exc)) from exc
    return validate_recovery_source_audit(
        parsed,
        workspace_fd=workspace_fd,
        erratum_reference=erratum_reference,
        diagnostic_reference=diagnostic_reference,
    )


def _load_external_authorities(
    workspace_fd: int,
    run_fd: int,
    source_audit_fd: int,
    erratum_fd: int,
    diagnostic_fd: int,
    *,
    recovery_source_evidence_commit: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    erratum_reference, erratum_raw = _external_reference_from_fd(
        erratum_fd, expected_path=ERRATUM_PATH, name="compound recovery erratum"
    )
    diagnostic_reference, diagnostic_raw = _external_reference_from_fd(
        diagnostic_fd,
        expected_path=DIAGNOSTIC_PATH,
        name="signed diagnostic receipt",
    )
    if erratum_reference != {
        "path": ERRATUM_PATH,
        "bytes": ERRATUM_BYTES,
        "sha256": ERRATUM_SHA256,
    }:
        _fail("compound recovery erratum bytes changed")
    if diagnostic_reference != {
        "path": DIAGNOSTIC_PATH,
        "bytes": DIAGNOSTIC_BYTES,
        "sha256": DIAGNOSTIC_SHA256,
    }:
        _fail("signed diagnostic receipt bytes changed")
    if b"external_diagnostic_calls=2" not in diagnostic_raw or b"authoritative_recovery_calls=1" not in diagnostic_raw:
        _fail("signed diagnostic call-count disclosure changed")
    source_audit_reference, source_audit_raw = _external_reference_from_fd(
        source_audit_fd,
        expected_path=RECOVERY_SOURCE_AUDIT_PATH,
        name="compound recovery source audit",
    )
    source_audit = _load_source_audit_from_raw(
        source_audit_raw,
        workspace_fd=workspace_fd,
        erratum_reference=erratum_reference,
        diagnostic_reference=diagnostic_reference,
    )
    project_fd = frozen._open_dir(workspace_fd, frozen.PROJECT_DIRECTORY)
    try:
        recovery_authority.validate_source_audit_live(
            source_audit,
            receipt_raw=source_audit_raw,
            repo_fd=workspace_fd,
            project_fd=project_fd,
            run_fd=run_fd,
            evidence_git_commit=recovery_source_evidence_commit,
            allow_exact_compound_outputs=True,
        )
    except recovery_authority.OpenAxesNativeV2RecoveryAuthorityError as exc:
        raise OpenAxesNativeV2RecoveryError(str(exc)) from exc
    finally:
        os.close(project_fd)
    return (
        source_audit,
        source_audit_reference,
        erratum_reference,
        diagnostic_reference,
    )


def _build_and_seal_stage(
    workspace_fd: int,
    run_fd: int,
    stage_fd: int,
    *,
    source_audit: Mapping[str, Any],
    source_audit_reference: Mapping[str, Any],
    erratum_reference: Mapping[str, Any],
    diagnostic_reference: Mapping[str, Any],
    original_source_evidence_commit: str,
) -> dict[str, Any]:
    if frozen._names(stage_fd):
        _fail("unsealed recovery stage is not empty")
    tasks, terminal_parent, comparator, original_authority = (
        _authenticate_original_science(
            workspace_fd,
            run_fd,
            original_source_evidence_commit=original_source_evidence_commit,
        )
    )
    if source_audit["original_science_authority"] != original_authority:
        _fail("recovery source audit does not bind live original science")
    if source_audit["failed_prestate"] != _expected_failed_prestate():
        _fail("recovery source audit does not bind live failed prestate")
    zero, gates, stress, ensembles = frozen.aggregate_zero_shift(
        run_fd, tasks, comparator
    )
    preparation_manifest = frozen._load_preparation_manifest(
        workspace_fd, terminal_parent
    )
    public_source_set_sha256 = _validate_preparation_public_source_set(
        workspace_fd, preparation_manifest
    )
    truth, truth_receipt = _call_canonical_signed_reconstruction_once(
        workspace_fd,
        expected_public_source_set_sha256=public_source_set_sha256,
    )
    signed = aggregate_signed_shift_recovery(
        tasks,
        ensembles,
        truth,
        truth_receipt,
        expected_public_source_set_sha256=public_source_set_sha256,
    )
    outputs = _build_terminal_payloads(
        run_fd,
        tasks,
        zero,
        gates,
        stress,
        signed,
        recovery_source_audit_reference=source_audit_reference,
        diagnostic_reference=diagnostic_reference,
    )
    frozen._mkdir_exclusive(stage_fd, "reports")
    for path in OUTPUT_PATHS:
        frozen._write_exclusive(stage_fd, path, outputs[path])
    validate_staged_terminal_files(
        outputs,
        recovery_source_audit_reference=source_audit_reference,
        diagnostic_reference=diagnostic_reference,
    )
    staged_references = _stage_output_references(stage_fd)
    seal = _payload(
        {
            "contract": STAGE_SEAL_CONTRACT,
            "schema_version": 1,
            "run_id": RUN_ID,
            "created_at_utc": _now_utc(),
            "erratum_reference": dict(erratum_reference),
            "diagnostic_receipt_reference": dict(diagnostic_reference),
            "recovery_source_audit_reference": dict(source_audit_reference),
            "original_science_authority": original_authority,
            "failed_prestate": _expected_failed_prestate(),
            "staged_file_order": list(OUTPUT_PATHS),
            "staged_file_references": staged_references,
            "staged_file_set_sha256": _canonical_hash(staged_references),
            "signed_report_status": signed["status"],
            "action_accounting": _expected_action_accounting(),
        }
    )
    frozen._write_exclusive(stage_fd, STAGE_SEAL_PATH, _canonical_bytes(seal))
    return validate_stage_seal(
        stage_fd,
        source_audit=source_audit,
        source_audit_reference=source_audit_reference,
        erratum_reference=erratum_reference,
        diagnostic_reference=diagnostic_reference,
    )


def validate_stage_seal(
    stage_fd: int,
    *,
    source_audit: Mapping[str, Any],
    source_audit_reference: Mapping[str, Any],
    erratum_reference: Mapping[str, Any],
    diagnostic_reference: Mapping[str, Any],
) -> dict[str, Any]:
    _stage_member_set_exact(stage_fd)
    raw = _read_fd_path(stage_fd, STAGE_SEAL_PATH, name="stage seal")
    try:
        parsed = frozen._strict_json(raw, name="stage seal")
    except frozen.OpenAxesExternalNativeV2Error as exc:
        raise OpenAxesNativeV2RecoveryError(str(exc)) from exc
    seal = _validate_payload(parsed, name="stage seal")
    _exact_keys(
        seal,
        {
            "contract",
            "schema_version",
            "run_id",
            "created_at_utc",
            "erratum_reference",
            "diagnostic_receipt_reference",
            "recovery_source_audit_reference",
            "original_science_authority",
            "failed_prestate",
            "staged_file_order",
            "staged_file_references",
            "staged_file_set_sha256",
            "signed_report_status",
            "action_accounting",
            "payload_sha256",
        },
        name="stage seal",
    )
    if (
        seal["contract"] != STAGE_SEAL_CONTRACT
        or seal["schema_version"] != 1
        or type(seal["schema_version"]) is not int
        or seal["run_id"] != RUN_ID
        or seal["erratum_reference"] != dict(erratum_reference)
        or seal["diagnostic_receipt_reference"] != dict(diagnostic_reference)
        or seal["recovery_source_audit_reference"]
        != dict(source_audit_reference)
        or seal["original_science_authority"]
        != source_audit["original_science_authority"]
        or seal["failed_prestate"] != _expected_failed_prestate()
        or seal["staged_file_order"] != list(OUTPUT_PATHS)
        or seal["action_accounting"] != _expected_action_accounting()
    ):
        _fail("stage seal authority changed")
    staged_references = _stage_output_references(stage_fd)
    if (
        seal["staged_file_references"] != staged_references
        or seal["staged_file_set_sha256"] != _canonical_hash(staged_references)
    ):
        _fail("stage seal output references changed")
    semantics = validate_staged_terminal_files(
        _read_stage_outputs(stage_fd),
        recovery_source_audit_reference=source_audit_reference,
        diagnostic_reference=diagnostic_reference,
    )
    if seal["signed_report_status"] != semantics["signed_report_status"]:
        _fail("stage seal signed branch changed")
    try:
        recovery_authority.validate_stage_seal(seal, source_audit=source_audit)
    except recovery_authority.OpenAxesNativeV2RecoveryAuthorityError as exc:
        raise OpenAxesNativeV2RecoveryError(str(exc)) from exc
    return seal


def _try_read_output(run_fd: int, path: str) -> bytes | None:
    parts = frozen._safe_parts(path)
    parent = frozen._open_dir(run_fd, "/".join(parts[:-1])) if len(parts) > 1 else os.dup(run_fd)
    descriptor = -1
    try:
        try:
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        except FileNotFoundError:
            return None
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            _fail(f"published recovery output is non-regular: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        if len(raw) != info.st_size:
            _fail(f"published recovery output short read: {path}")
        return raw
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _validate_publish_boundary(run_fd: int, staged: Mapping[str, bytes]) -> int:
    allowed_root = set(_expected_failed_prestate()["top_level_members"]) | {
        "development_manifest.json",
        "COMPLETE.json",
    }
    if set(frozen._names(run_fd)) - allowed_root:
        _fail("RUN003 gained an unauthorized root member")
    if set(frozen._names(run_fd, "reports")) - set(REPORT_NAMES):
        _fail("RUN003 gained an unauthorized report member")
    _validate_base_tree_after_staging(run_fd)
    existing: list[bool] = []
    for path in OUTPUT_PATHS:
        raw = _try_read_output(run_fd, path)
        existing.append(raw is not None)
        if raw is not None and raw != staged[path]:
            _fail(f"published recovery output differs from stage: {path}")
    present_count = sum(existing)
    if existing != [index < present_count for index in range(len(OUTPUT_PATHS))]:
        _fail("published recovery outputs are not an exact ordered prefix")
    return present_count


def _semantic_validation_flags() -> dict[str, bool]:
    return {key: True for key in sorted(SEMANTIC_VALIDATION_KEYS)}


def _build_compound_terminal(
    run_fd: int,
    stage_fd: int,
    *,
    seal: Mapping[str, Any],
    source_audit_reference: Mapping[str, Any],
    erratum_reference: Mapping[str, Any],
    diagnostic_reference: Mapping[str, Any],
) -> dict[str, Any]:
    output_references = [
        _raw_reference(run_fd, path, name=f"published output {path}")
        for path in OUTPUT_PATHS
    ]
    if output_references != seal["staged_file_references"]:
        _fail("published output references differ from sealed stage")
    stage_seal_reference = _raw_reference(
        stage_fd, STAGE_SEAL_PATH, name="stage seal"
    )
    poststate = {
        "prestate_canonical_tree_sha256": FAILED_TREE_SHA256,
        "added_file_order": list(OUTPUT_PATHS),
        "added_file_references": output_references,
        "added_file_count": 7,
        "existing_prestate_unchanged": True,
        "complete_root_member_set_exact": frozen._names(run_fd)
        == sorted(frozen.ROOT_MEMBERS),
    }
    if poststate["complete_root_member_set_exact"] is not True:
        _fail("compound RUN003 complete root member set changed")
    return _payload(
        {
            "contract": TERMINAL_CONTRACT,
            "schema_version": 1,
            "run_id": RUN_ID,
            "created_at_utc": _now_utc(),
            "erratum_reference": dict(erratum_reference),
            "diagnostic_receipt_reference": dict(diagnostic_reference),
            "recovery_source_audit_reference": dict(source_audit_reference),
            "original_science_authority": seal["original_science_authority"],
            "failed_prestate": _expected_failed_prestate(),
            "stage_seal_reference": stage_seal_reference,
            "staged_file_references": seal["staged_file_references"],
            "output_file_references": output_references,
            "poststate": poststate,
            "existing_prestate_unchanged": True,
            "signed_report_status": seal["signed_report_status"],
            "action_accounting": _expected_action_accounting(),
            "semantic_validation": _semantic_validation_flags(),
            "compound_terminal_required_for_publication": True,
        }
    )


def validate_compound_terminal(
    value: Any,
    *,
    run_fd: int,
    stage_fd: int,
    source_audit: Mapping[str, Any],
    source_audit_reference: Mapping[str, Any],
    erratum_reference: Mapping[str, Any],
    diagnostic_reference: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = _validate_payload(value, name="compound terminal")
    _exact_keys(
        terminal,
        {
            "contract",
            "schema_version",
            "run_id",
            "created_at_utc",
            "erratum_reference",
            "diagnostic_receipt_reference",
            "recovery_source_audit_reference",
            "original_science_authority",
            "failed_prestate",
            "stage_seal_reference",
            "staged_file_references",
            "output_file_references",
            "poststate",
            "existing_prestate_unchanged",
            "signed_report_status",
            "action_accounting",
            "semantic_validation",
            "compound_terminal_required_for_publication",
            "payload_sha256",
        },
        name="compound terminal",
    )
    seal = validate_stage_seal(
        stage_fd,
        source_audit=source_audit,
        source_audit_reference=source_audit_reference,
        erratum_reference=erratum_reference,
        diagnostic_reference=diagnostic_reference,
    )
    _validate_publish_boundary(run_fd, _read_stage_outputs(stage_fd))
    expected = {
        "contract": TERMINAL_CONTRACT,
        "schema_version": 1,
        "run_id": RUN_ID,
        "erratum_reference": dict(erratum_reference),
        "diagnostic_receipt_reference": dict(diagnostic_reference),
        "recovery_source_audit_reference": dict(source_audit_reference),
        "original_science_authority": source_audit["original_science_authority"],
        "failed_prestate": _expected_failed_prestate(),
        "stage_seal_reference": _raw_reference(
            stage_fd, STAGE_SEAL_PATH, name="stage seal"
        ),
        "staged_file_references": seal["staged_file_references"],
        "output_file_references": seal["staged_file_references"],
        "existing_prestate_unchanged": True,
        "signed_report_status": seal["signed_report_status"],
        "action_accounting": _expected_action_accounting(),
        "semantic_validation": _semantic_validation_flags(),
        "compound_terminal_required_for_publication": True,
    }
    for key, expected_value in expected.items():
        if terminal.get(key) != expected_value:
            _fail(f"compound terminal {key} changed")
    poststate = terminal["poststate"]
    if not isinstance(poststate, Mapping):
        _fail("compound terminal poststate is malformed")
    _exact_keys(
        poststate,
        {
            "prestate_canonical_tree_sha256",
            "added_file_order",
            "added_file_references",
            "added_file_count",
            "existing_prestate_unchanged",
            "complete_root_member_set_exact",
        },
        name="compound terminal poststate",
    )
    if (
        poststate["prestate_canonical_tree_sha256"] != FAILED_TREE_SHA256
        or poststate["added_file_order"] != list(OUTPUT_PATHS)
        or poststate["added_file_references"] != seal["staged_file_references"]
        or poststate["added_file_count"] != 7
        or poststate["existing_prestate_unchanged"] is not True
        or poststate["complete_root_member_set_exact"] is not True
    ):
        _fail("compound terminal poststate changed")
    validate_staged_terminal_files(
        {
            path: _read_fd_path(run_fd, path, name=f"published output {path}")
            for path in OUTPUT_PATHS
        },
        recovery_source_audit_reference=source_audit_reference,
        diagnostic_reference=diagnostic_reference,
    )
    stage_seal_raw = _read_fd_path(
        stage_fd, STAGE_SEAL_PATH, name="stage seal"
    )
    try:
        recovery_authority.validate_compound_terminal_receipt(
            terminal,
            source_audit=source_audit,
            source_audit_raw=_canonical_bytes(source_audit),
            stage_seal=seal,
            stage_seal_raw=stage_seal_raw,
            run_fd=run_fd,
            stage_fd=stage_fd,
        )
    except recovery_authority.OpenAxesNativeV2RecoveryAuthorityError as exc:
        raise OpenAxesNativeV2RecoveryError(str(exc)) from exc
    return terminal


def stage_compound_recovery(
    workspace_fd: int,
    run_fd: int,
    stage_fd: int,
    source_audit_fd: int,
    erratum_fd: int,
    diagnostic_fd: int,
    *,
    original_source_evidence_commit: str,
    recovery_source_evidence_commit: str,
) -> dict[str, Any]:
    (
        source_audit,
        source_audit_reference,
        erratum_reference,
        diagnostic_reference,
    ) = _load_external_authorities(
        workspace_fd,
        run_fd,
        source_audit_fd,
        erratum_fd,
        diagnostic_fd,
        recovery_source_evidence_commit=recovery_source_evidence_commit,
    )
    names = frozen._names(stage_fd)
    if not names:
        return _build_and_seal_stage(
            workspace_fd,
            run_fd,
            stage_fd,
            source_audit=source_audit,
            source_audit_reference=source_audit_reference,
            erratum_reference=erratum_reference,
            diagnostic_reference=diagnostic_reference,
            original_source_evidence_commit=original_source_evidence_commit,
        )
    if STAGE_SEAL_PATH not in names:
        _fail("ambiguous unsealed recovery stage cannot resume")
    return validate_stage_seal(
        stage_fd,
        source_audit=source_audit,
        source_audit_reference=source_audit_reference,
        erratum_reference=erratum_reference,
        diagnostic_reference=diagnostic_reference,
    )


def publish_compound_recovery(
    workspace_fd: int,
    run_fd: int,
    stage_fd: int,
    authority_fd: int,
    source_audit_fd: int,
    erratum_fd: int,
    diagnostic_fd: int,
    *,
    recovery_source_evidence_commit: str,
) -> dict[str, Any]:
    (
        source_audit,
        source_audit_reference,
        erratum_reference,
        diagnostic_reference,
    ) = _load_external_authorities(
        workspace_fd,
        run_fd,
        source_audit_fd,
        erratum_fd,
        diagnostic_fd,
        recovery_source_evidence_commit=recovery_source_evidence_commit,
    )
    seal = validate_stage_seal(
        stage_fd,
        source_audit=source_audit,
        source_audit_reference=source_audit_reference,
        erratum_reference=erratum_reference,
        diagnostic_reference=diagnostic_reference,
    )
    staged = _read_stage_outputs(stage_fd)
    present_count = _validate_publish_boundary(run_fd, staged)
    for path in OUTPUT_PATHS[present_count:]:
        frozen._write_exclusive(run_fd, path, staged[path])
    _validate_publish_boundary(run_fd, staged)
    validate_staged_terminal_files(
        {
            path: _read_fd_path(run_fd, path, name=f"published output {path}")
            for path in OUTPUT_PATHS
        },
        recovery_source_audit_reference=source_audit_reference,
        diagnostic_reference=diagnostic_reference,
    )
    try:
        frozen.validate_openaxes_external_native_v2_run(run_fd)
    except frozen.OpenAxesExternalNativeV2Error as exc:
        raise OpenAxesNativeV2RecoveryError(
            f"frozen full RUN003 validator rejected recovery: {exc}"
        ) from exc
    terminal = _build_compound_terminal(
        run_fd,
        stage_fd,
        seal=seal,
        source_audit_reference=source_audit_reference,
        erratum_reference=erratum_reference,
        diagnostic_reference=diagnostic_reference,
    )
    existing = _try_read_output(authority_fd, TERMINAL_PATH)
    if existing is None:
        frozen._write_exclusive(
            authority_fd, TERMINAL_PATH, _canonical_bytes(terminal)
        )
    elif existing != _canonical_bytes(terminal):
        # A valid existing terminal may have an earlier creation timestamp.
        try:
            existing_value = frozen._strict_json(existing, name="compound terminal")
        except frozen.OpenAxesExternalNativeV2Error as exc:
            raise OpenAxesNativeV2RecoveryError(str(exc)) from exc
        return validate_compound_terminal(
            existing_value,
            run_fd=run_fd,
            stage_fd=stage_fd,
            source_audit=source_audit,
            source_audit_reference=source_audit_reference,
            erratum_reference=erratum_reference,
            diagnostic_reference=diagnostic_reference,
        )
    terminal_raw = _read_fd_path(
        authority_fd, TERMINAL_PATH, name="compound terminal"
    )
    try:
        terminal_value = frozen._strict_json(terminal_raw, name="compound terminal")
    except frozen.OpenAxesExternalNativeV2Error as exc:
        raise OpenAxesNativeV2RecoveryError(str(exc)) from exc
    return validate_compound_terminal(
        terminal_value,
        run_fd=run_fd,
        stage_fd=stage_fd,
        source_audit=source_audit,
        source_audit_reference=source_audit_reference,
        erratum_reference=erratum_reference,
        diagnostic_reference=diagnostic_reference,
    )


def validate_published_compound_recovery(
    workspace_fd: int,
    run_fd: int,
    stage_fd: int,
    authority_fd: int,
    source_audit_fd: int,
    erratum_fd: int,
    diagnostic_fd: int,
    *,
    recovery_source_evidence_commit: str,
) -> dict[str, Any]:
    (
        source_audit,
        source_audit_reference,
        erratum_reference,
        diagnostic_reference,
    ) = _load_external_authorities(
        workspace_fd,
        run_fd,
        source_audit_fd,
        erratum_fd,
        diagnostic_fd,
        recovery_source_evidence_commit=recovery_source_evidence_commit,
    )
    terminal_raw = _try_read_output(authority_fd, TERMINAL_PATH)
    if terminal_raw is None:
        _fail("compound terminal is absent")
    try:
        terminal = frozen._strict_json(terminal_raw, name="compound terminal")
    except frozen.OpenAxesExternalNativeV2Error as exc:
        raise OpenAxesNativeV2RecoveryError(str(exc)) from exc
    result = validate_compound_terminal(
        terminal,
        run_fd=run_fd,
        stage_fd=stage_fd,
        source_audit=source_audit,
        source_audit_reference=source_audit_reference,
        erratum_reference=erratum_reference,
        diagnostic_reference=diagnostic_reference,
    )
    try:
        frozen.validate_openaxes_external_native_v2_run(run_fd)
    except frozen.OpenAxesExternalNativeV2Error as exc:
        raise OpenAxesNativeV2RecoveryError(
            f"frozen full RUN003 validator rejected recovery: {exc}"
        ) from exc
    return result


def recover_compound_run003(
    workspace_fd: int,
    run_fd: int,
    stage_fd: int,
    authority_fd: int,
    source_audit_fd: int,
    erratum_fd: int,
    diagnostic_fd: int,
    *,
    original_source_evidence_commit: str,
    recovery_source_evidence_commit: str,
) -> dict[str, Any]:
    stage_compound_recovery(
        workspace_fd,
        run_fd,
        stage_fd,
        source_audit_fd,
        erratum_fd,
        diagnostic_fd,
        original_source_evidence_commit=original_source_evidence_commit,
        recovery_source_evidence_commit=recovery_source_evidence_commit,
    )
    return publish_compound_recovery(
        workspace_fd,
        run_fd,
        stage_fd,
        authority_fd,
        source_audit_fd,
        erratum_fd,
        diagnostic_fd,
        recovery_source_evidence_commit=recovery_source_evidence_commit,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("stage", "publish", "recover", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--workspace-root", required=True)
        command.add_argument("--run-dir", required=True)
        command.add_argument("--stage-dir", required=True)
        command.add_argument("--authority-dir", required=True)
        command.add_argument("--source-audit-fd", required=True)
        command.add_argument("--erratum-fd", required=True)
        command.add_argument("--diagnostic-fd", required=True)
        command.add_argument("--original-source-evidence-commit", required=True)
        command.add_argument("--recovery-source-evidence-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    descriptors: list[int] = []
    try:
        workspace_fd = frozen._fd_from_argument(
            arguments.workspace_root, name="workspace root"
        )
        descriptors.append(workspace_fd)
        run_fd = frozen._fd_from_argument(arguments.run_dir, name="RUN003 root")
        descriptors.append(run_fd)
        stage_fd = frozen._fd_from_argument(
            arguments.stage_dir, name="external recovery stage"
        )
        descriptors.append(stage_fd)
        authority_fd = frozen._fd_from_argument(
            arguments.authority_dir, name="external recovery authority"
        )
        descriptors.append(authority_fd)
        source_audit_fd = _regular_fd_argument(
            arguments.source_audit_fd, name="recovery source audit"
        )
        descriptors.append(source_audit_fd)
        erratum_fd = _regular_fd_argument(
            arguments.erratum_fd, name="compound recovery erratum"
        )
        descriptors.append(erratum_fd)
        diagnostic_fd = _regular_fd_argument(
            arguments.diagnostic_fd, name="signed diagnostic receipt"
        )
        descriptors.append(diagnostic_fd)
        if re.fullmatch(r"[0-9a-f]{40}", arguments.original_source_evidence_commit) is None:
            _fail("original source evidence commit is malformed")
        if re.fullmatch(r"[0-9a-f]{40}", arguments.recovery_source_evidence_commit) is None:
            _fail("recovery source evidence commit is malformed")
        if arguments.command == "stage":
            stage_compound_recovery(
                workspace_fd,
                run_fd,
                stage_fd,
                source_audit_fd,
                erratum_fd,
                diagnostic_fd,
                original_source_evidence_commit=arguments.original_source_evidence_commit,
                recovery_source_evidence_commit=arguments.recovery_source_evidence_commit,
            )
        elif arguments.command == "publish":
            publish_compound_recovery(
                workspace_fd,
                run_fd,
                stage_fd,
                authority_fd,
                source_audit_fd,
                erratum_fd,
                diagnostic_fd,
                recovery_source_evidence_commit=arguments.recovery_source_evidence_commit,
            )
        elif arguments.command == "recover":
            recover_compound_run003(
                workspace_fd,
                run_fd,
                stage_fd,
                authority_fd,
                source_audit_fd,
                erratum_fd,
                diagnostic_fd,
                original_source_evidence_commit=arguments.original_source_evidence_commit,
                recovery_source_evidence_commit=arguments.recovery_source_evidence_commit,
            )
        elif arguments.command == "validate":
            validate_published_compound_recovery(
                workspace_fd,
                run_fd,
                stage_fd,
                authority_fd,
                source_audit_fd,
                erratum_fd,
                diagnostic_fd,
                recovery_source_evidence_commit=arguments.recovery_source_evidence_commit,
            )
        else:
            _fail("unknown compound recovery command")
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
