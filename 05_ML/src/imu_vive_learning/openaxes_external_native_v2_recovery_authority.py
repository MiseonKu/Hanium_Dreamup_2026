"""Outcome-neutral authority for the native-V2 RUN003 compound recovery.

This module never imports the model implementation, NumPy, pandas, PyTorch, or
CUDA.  Its pre-execution operation hashes raw bytes and validates authority
metadata only.  The later terminal operation validates the compound recovery
envelope and the exact seven appended files; it does not decode a scientific
value.

Public CLI roots are inherited directory descriptors.  Other CLI inputs are
either inherited descriptors or exact project-relative paths opened beneath a
pinned project descriptor without following symlinks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


SOURCE_AUDIT_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.compound_recovery.source_audit.v1"
)
STAGE_SEAL_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.compound_recovery.stage_seal.v1"
)
TERMINAL_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.compound_recovery.terminal.v1"
)
VALIDATION_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.compound_recovery.validation.v1"
)
RUN_ID = "openaxes-native-v2-sealed-003"
PROJECT_DIRECTORY = "09_SO101_IMU_VIVE_LEARNING"
RUN_RELATIVE_PATH = (
    "artifacts/openaxes_external_native_v2/run_openaxes-native-v2-sealed-003"
)
SOURCE_AUDIT_RECEIPT_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/receipts/"
    "openaxes_external_native_v2_compound_recovery_source_audit.json"
)
RECOVERY_ERRATUM_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/docs/"
    "OPENAXES_EXTERNAL_NATIVE_V2_RUN003_RECOVERY_ERRATUM.md"
)
HISTORICAL_RUN004_ERRATUM_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/docs/"
    "OPENAXES_EXTERNAL_NATIVE_V2_RUN004_ERRATUM.md"
)
SIGNED_DIAGNOSTIC_RECEIPT_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/docs/"
    "OPENAXES_EXTERNAL_NATIVE_V2_SIGNED_DIAGNOSTIC_RECEIPT.md"
)

RECOVERY_SOURCE_CLOSURE = (
    "09_SO101_IMU_VIVE_LEARNING/configs/openaxes_external_native_v2.yaml",
    "09_SO101_IMU_VIVE_LEARNING/configs/openaxes_external_v1.yaml",
    "09_SO101_IMU_VIVE_LEARNING/docs/OPENAXES_EXTERNAL_NATIVE_V2_CONTRACT.md",
    "09_SO101_IMU_VIVE_LEARNING/docs/OPENAXES_EXTERNAL_NATIVE_V2_PARENT_RECEIPT_CONTRACT.md",
    RECOVERY_ERRATUM_PATH,
    HISTORICAL_RUN004_ERRATUM_PATH,
    SIGNED_DIAGNOSTIC_RECEIPT_PATH,
    "09_SO101_IMU_VIVE_LEARNING/docs/OPENAXES_EXTERNAL_V1_CONTRACT.md",
    "09_SO101_IMU_VIVE_LEARNING/pyproject.toml",
    "09_SO101_IMU_VIVE_LEARNING/requirements-eval.lock",
    "09_SO101_IMU_VIVE_LEARNING/requirements-gpu.lock",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/__init__.py",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/io_utils.py",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/openaxes_external_native_v2_config.py",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/openaxes_external_native_v2_experiment.py",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/openaxes_external_native_v2_parent_receipt.py",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/openaxes_external_native_v2_training.py",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/openaxes_external_native_v2_recovery.py",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/openaxes_external_native_v2_recovery_authority.py",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/openaxes_external_v1.py",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/openaxes_external_v1_config.py",
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/openaxes_external_v1_cli.py",
    "09_SO101_IMU_VIVE_LEARNING/tests/test_openaxes_external_native_v2_config.py",
    "09_SO101_IMU_VIVE_LEARNING/tests/test_openaxes_external_native_v2_experiment.py",
    "09_SO101_IMU_VIVE_LEARNING/tests/test_openaxes_external_native_v2_parent_receipt.py",
    "09_SO101_IMU_VIVE_LEARNING/tests/test_openaxes_external_native_v2_training.py",
    "09_SO101_IMU_VIVE_LEARNING/tests/test_openaxes_external_native_v2_recovery.py",
    "09_SO101_IMU_VIVE_LEARNING/tests/test_openaxes_external_native_v2_recovery_authority.py",
    "09_SO101_IMU_VIVE_LEARNING/tests/test_openaxes_external_v1.py",
)

STAGED_FILE_ORDER = (
    "reports/zero_shift.json",
    "reports/primary_gates.json",
    "reports/stress_report.json",
    "reports/signed_shift_report.json",
    "reports/development_summary.json",
    "development_manifest.json",
    "COMPLETE.json",
)
FAILED_TOP_LEVEL_MEMBERS = (
    "PREPARED.json",
    "parents",
    "prepared_manifest.json",
    "quarantine",
    "reports",
    "results",
    "runtime",
    "task_index.json",
    "tasks",
)
COMPLETE_TOP_LEVEL_MEMBERS = tuple(
    sorted((*FAILED_TOP_LEVEL_MEMBERS, "development_manifest.json", "COMPLETE.json"))
)

FIXED_REGULAR_FILE_COUNT = 2_799
FIXED_REGULAR_FILE_BYTES = 318_851_615
FIXED_TREE_SHA256 = "e42472225abdaaa081da7a33dfdbe6d3543a607b579654a8ef7172aeb944b08b"
FIXED_EXECUTION_SOURCE_SET_SHA256 = (
    "3cc3133cd925fc3ebeacd53a4f746090a99d8b52d92ce2157d1d0b36576beb99"
)
FIXED_CONTROLLER_SOURCE_SHA256 = (
    "83a49e943494061bf0797d4acebe55a303aeb80e397db00f3a31d925d4ec9e59"
)
FIXED_CONFIG_SHA256 = "832a7618f0b7ce94c2813d4b5659153bb88d878d1e67f6ccba3f8f1e3555d59a"
FIXED_SOURCE_SET_RAW_SHA256 = (
    "330e2a4643186242f74a3b5ebee627a93fc7c4383601b202a273ee66f42eeb6c"
)
FIXED_EXECUTION_SESSION_RAW_SHA256 = (
    "9500a36414e1408e998502b320015b9eba07140a0919302a7673274e845bdea7"
)
FIXED_TASK_INDEX_RAW_SHA256 = (
    "544a0ba786c9708047d0689124597a5d94a76d790fd544bc82fd71b80fe5975b"
)
FIXED_DIAGNOSTIC_BYTES = 2_027
FIXED_DIAGNOSTIC_SHA256 = (
    "bbabad0c4bbfa63f4beb0d01d476cba86148bf362fb2d892faf596fb41e0a38f"
)
HISTORICAL_RUN004_ERRATUM_GIT_COMMIT = "75f470bcf1006135e6a12f4d62d36626d80b9f03"
PREISSUED_RECOVERY_ERRATUM_GIT_COMMIT = "b6df0ddde04d35f6216c486b7ec8508a2911ddec"
SIGNED_DIAGNOSTIC_RECEIPT_GIT_COMMIT = "6e98fd2e5f3ea916fe01a90dbe100e03eeb3d474"

REFERENCE_KEYS = frozenset(("path", "bytes", "sha256"))
ORIGINAL_SCIENCE_AUTHORITY_KEYS = frozenset(
    (
        "run_id",
        "canonical_config_sha256",
        "execution_source_set_sha256",
        "controller_source_sha256",
        "source_set_reference",
        "execution_session_reference",
        "task_index_reference",
        "result_completed_reference_set_sha256",
        "result_completed_count",
    )
)
FAILED_PRESTATE_KEYS = frozenset(
    (
        "canonical_tree_sha256",
        "regular_file_count",
        "regular_file_bytes",
        "top_level_members",
        "reports_members",
        "development_manifest_absent",
        "complete_absent",
        "result_completed_count",
    )
)
POSTSTATE_KEYS = frozenset(
    (
        "prestate_canonical_tree_sha256",
        "added_file_order",
        "added_file_references",
        "added_file_count",
        "existing_prestate_unchanged",
        "complete_root_member_set_exact",
    )
)
SEMANTIC_VALIDATION_KEYS = frozenset(
    (
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
    )
)
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
SOURCE_AUDIT_ASSERTIONS = {
    "performance_payload_decode_allowed": False,
    "run003_mutation_allowed_during_source_audit": False,
    "scientific_fit_allowed": False,
    "model_prediction_allowed": False,
    "task_evaluator_allowed": False,
    "cuda_allowed": False,
    "network_allowed": False,
    "source_closure_must_match_implementation_commit": True,
    "implementation_commit_must_ancestor_of_evidence_commit": True,
    "recovery_and_original_science_authorities_must_remain_separate": True,
    "compound_terminal_required_for_publication": True,
}
SOURCE_AUDIT_KEYS = frozenset(
    (
        "contract",
        "schema_version",
        "run_id",
        "created_at_utc",
        "repository",
        "source_closure",
        "preissued_erratum_reference",
        "diagnostic_receipt_reference",
        "original_science_authority",
        "failed_prestate",
        "assertions",
        "payload_sha256",
    )
)
REPOSITORY_KEYS = frozenset(
    (
        "object_format",
        "project_directory",
        "implementation_git_commit",
        "historical_run004_erratum_git_commit",
        "preissued_recovery_erratum_git_commit",
        "signed_diagnostic_receipt_git_commit",
        "source_audit_receipt_path",
        "source_audit_evidence_commit_required",
        "implementation_commit_must_be_ancestor_of_evidence_commit",
        "preissued_authorities_must_be_ancestors_of_implementation_commit",
    )
)
SOURCE_CLOSURE_KEYS = frozenset(("path_order", "files", "file_count", "set_sha256"))
SOURCE_FILE_KEYS = frozenset(("path", "bytes", "sha256", "git_blob_oid", "git_mode"))
STAGE_SEAL_KEYS = frozenset(
    (
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
    )
)
TERMINAL_KEYS = frozenset(
    (
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
    )
)

HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
UTC_TIME = re.compile(r"20[0-9]{2}-[01][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]Z\Z")


class OpenAxesNativeV2RecoveryAuthorityError(RuntimeError):
    """Fail-closed recovery-authority error."""


def _fail(message: str) -> None:
    raise OpenAxesNativeV2RecoveryAuthorityError(message)


def canonical_json_bytes(value: Any, *, final_newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OpenAxesNativeV2RecoveryAuthorityError("value is not canonical JSON") from exc
    return raw + (b"\n" if final_newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json_bytes(raw: bytes, *, name: str, require_canonical: bool = True) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"{name} contains duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenAxesNativeV2RecoveryAuthorityError(f"{name} is not strict JSON") from exc
    if type(value) is not dict:
        _fail(f"{name} is not a JSON object")
    if require_canonical and raw not in (
        canonical_json_bytes(value),
        canonical_json_bytes(value, final_newline=True),
    ):
        _fail(f"{name} is not canonical JSON")
    return value


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{name} is not an exact object")
    return value


def _keys(value: Mapping[str, Any], expected: frozenset[str], *, name: str) -> None:
    if set(value) != set(expected):
        _fail(f"{name} keys changed")


def _sha(value: Any, *, name: str) -> str:
    if type(value) is not str or HEX64.fullmatch(value) is None:
        _fail(f"{name} is not lowercase SHA-256")
    return value


def _commit(value: Any, *, name: str) -> str:
    if type(value) is not str or HEX40.fullmatch(value) is None:
        _fail(f"{name} is not a lowercase SHA-1 commit")
    return value


def _timestamp(value: Any, *, name: str) -> str:
    if type(value) is not str or UTC_TIME.fullmatch(value) is None:
        _fail(f"{name} is not canonical UTC time")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise OpenAxesNativeV2RecoveryAuthorityError(
            f"{name} is not a real UTC timestamp"
        ) from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        _fail(f"{name} is not canonical UTC time")
    return value


def _relative_path(value: Any, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        _fail(f"{name} is not an exact relative path")
    if value.startswith("/") or value.endswith("/"):
        _fail(f"{name} is not an exact relative path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        _fail(f"{name} contains unsafe path component")
    return value


def _project_local_path(value: Any, *, name: str) -> str:
    path = _relative_path(value, name=name)
    prefix = f"{PROJECT_DIRECTORY}/"
    if not path.startswith(prefix):
        _fail(f"{name} is not repository-relative below the frozen project")
    return _relative_path(path.removeprefix(prefix), name=name)


def _dir_fd(value: int, *, name: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{name} is not an inherited descriptor")
    try:
        metadata = os.fstat(value)
    except OSError as exc:
        raise OpenAxesNativeV2RecoveryAuthorityError(f"{name} is not open") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        _fail(f"{name} is not a directory descriptor")
    return value


def _open_relative(root_fd: int, relative: str, *, directory: bool = False) -> int:
    _dir_fd(root_fd, name="root descriptor")
    path = _relative_path(relative, name="relative path")
    current = os.dup(root_fd)
    try:
        for index, part in enumerate(path.split("/")):
            final = index == len(path.split("/")) - 1
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            if not final or directory:
                flags |= os.O_DIRECTORY
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        metadata = os.fstat(current)
        if directory and not stat.S_ISDIR(metadata.st_mode):
            _fail("relative directory path is not a directory")
        if not directory and not stat.S_ISREG(metadata.st_mode):
            _fail("relative file path is not a regular file")
        return current
    except Exception:
        os.close(current)
        raise


def _read_raw_at(root_fd: int, relative: str, *, name: str) -> bytes:
    descriptor = _open_relative(root_fd, relative)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after):
        _fail(f"{name} changed while read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        _fail(f"{name} byte count changed while read")
    return raw


def _reference(path: str, raw: bytes) -> dict[str, Any]:
    return {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _reference_at(root_fd: int, relative: str, *, name: str) -> dict[str, Any]:
    return _reference(relative, _read_raw_at(root_fd, relative, name=name))


def _validate_reference(value: Any, *, name: str, expected_path: str | None = None) -> dict[str, Any]:
    reference = _mapping(value, name=name)
    _keys(reference, REFERENCE_KEYS, name=name)
    path = _relative_path(reference.get("path"), name=f"{name} path")
    if expected_path is not None and path != expected_path:
        _fail(f"{name} path changed")
    if type(reference.get("bytes")) is not int or reference["bytes"] < 0:
        _fail(f"{name} byte count changed")
    _sha(reference.get("sha256"), name=f"{name} hash")
    return reference


def _verify_reference_at(root_fd: int, value: Any, *, name: str, expected_path: str) -> dict[str, Any]:
    reference = _validate_reference(value, name=name, expected_path=expected_path)
    if reference != _reference_at(root_fd, expected_path, name=name):
        _fail(f"{name} raw reference changed")
    return reference


def _payload_hash(value: Mapping[str, Any], *, name: str) -> None:
    claimed = _sha(value.get("payload_sha256"), name=f"{name} payload hash")
    base = dict(value)
    base.pop("payload_sha256", None)
    if claimed != canonical_sha256(base):
        _fail(f"{name} payload hash changed")


def _git(repo_fd: int, args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    _dir_fd(repo_fd, name="repository descriptor")
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "TZ": "UTC",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "SSH_ASKPASS": "/bin/false",
        "GIT_ASKPASS": "/bin/false",
    }
    command = [
        "/usr/bin/git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "protocol.allow=never",
        *args,
    ]
    try:
        return subprocess.run(
            command,
            cwd=f"/proc/self/fd/{repo_fd}",
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OpenAxesNativeV2RecoveryAuthorityError("local Git authority command failed") from exc


def _verify_repo_project_binding(repo_fd: int, project_fd: int) -> None:
    child = _open_relative(repo_fd, PROJECT_DIRECTORY, directory=True)
    try:
        left = os.fstat(child)
        right = os.fstat(_dir_fd(project_fd, name="project descriptor"))
    finally:
        os.close(child)
    if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
        _fail("repository/project descriptor binding changed")
    object_format = _git(repo_fd, ("rev-parse", "--show-object-format")).stdout.strip()
    if object_format != b"sha1":
        _fail("repository object format is not sha1")


def _git_blob(repo_fd: int, commit: str, path: str) -> tuple[str, str, bytes]:
    commit = _commit(commit, name="implementation commit")
    project_path = _relative_path(path, name="source path")
    if not project_path.startswith(f"{PROJECT_DIRECTORY}/"):
        _fail("Git source path escaped the frozen project")
    listing = _git(repo_fd, ("ls-tree", "-z", commit, "--", project_path)).stdout
    records = [item for item in listing.split(b"\0") if item]
    if len(records) != 1:
        _fail("source path does not resolve to one Git blob")
    try:
        prefix, observed = records[0].split(b"\t", 1)
        mode, object_type, oid = prefix.split(b" ", 2)
        decoded_path = observed.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise OpenAxesNativeV2RecoveryAuthorityError("Git tree record is malformed") from exc
    if decoded_path != project_path or object_type != b"blob" or mode not in (b"100644", b"100755"):
        _fail("Git source blob identity changed")
    oid_text = oid.decode("ascii")
    if HEX40.fullmatch(oid_text) is None:
        _fail("Git blob OID changed")
    raw = _git(repo_fd, ("cat-file", "blob", oid_text)).stdout
    expected_oid = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    if oid_text != expected_oid:
        _fail("Git blob OID does not match raw bytes")
    return mode.decode("ascii"), oid_text, raw


def _is_ancestor(repo_fd: int, ancestor: str, descendant: str) -> None:
    ancestor = _commit(ancestor, name="ancestor commit")
    descendant = _commit(descendant, name="descendant commit")
    result = _git(repo_fd, ("merge-base", "--is-ancestor", ancestor, descendant), check=False)
    if result.returncode != 0 or result.stdout or result.stderr:
        _fail("required Git ancestry is absent")


def _closure_projection(repo_fd: int, project_fd: int, commit: str) -> dict[str, Any]:
    _verify_repo_project_binding(repo_fd, project_fd)
    commit = _commit(commit, name="implementation commit")
    if _git(repo_fd, ("cat-file", "-t", commit)).stdout != b"commit\n":
        _fail("implementation authority is not a Git commit")
    files: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for path in RECOVERY_SOURCE_CLOSURE:
        mode, oid, committed = _git_blob(repo_fd, commit, path)
        live = _read_raw_at(
            project_fd,
            _project_local_path(path, name="live source path"),
            name=f"live source {path}",
        )
        if live != committed:
            _fail(f"live source differs from implementation commit: {path}")
        row = {
            "path": path,
            "bytes": len(live),
            "sha256": hashlib.sha256(live).hexdigest(),
            "git_blob_oid": oid,
            "git_mode": mode,
        }
        files[path] = row
        rows.append(row)
    for authority_commit, authority_path in (
        (HISTORICAL_RUN004_ERRATUM_GIT_COMMIT, HISTORICAL_RUN004_ERRATUM_PATH),
        (PREISSUED_RECOVERY_ERRATUM_GIT_COMMIT, RECOVERY_ERRATUM_PATH),
        (SIGNED_DIAGNOSTIC_RECEIPT_GIT_COMMIT, SIGNED_DIAGNOSTIC_RECEIPT_PATH),
    ):
        _is_ancestor(repo_fd, authority_commit, commit)
        _mode, _oid, preissued_raw = _git_blob(repo_fd, authority_commit, authority_path)
        row = files[authority_path]
        if (
            row["bytes"] != len(preissued_raw)
            or row["sha256"] != hashlib.sha256(preissued_raw).hexdigest()
        ):
            _fail("preissued authority Git blob differs from implementation closure")
    return {
        "path_order": list(RECOVERY_SOURCE_CLOSURE),
        "files": files,
        "file_count": len(rows),
        "set_sha256": canonical_sha256(rows),
    }


def _scan_tree(root_fd: int, *, prefix: str = "") -> tuple[list[str], list[dict[str, Any]]]:
    _dir_fd(root_fd, name="tree root descriptor")
    directories: list[str] = []
    files: list[dict[str, Any]] = []

    def visit(directory_fd: int, relative: str) -> None:
        try:
            names = sorted(os.listdir(directory_fd), key=os.fsencode)
        except OSError as exc:
            raise OpenAxesNativeV2RecoveryAuthorityError("cannot enumerate authority tree") from exc
        for name in names:
            if type(name) is not str or name in (".", "..") or "/" in name or "\x00" in name:
                _fail("authority tree member name is unsafe")
            path = f"{relative}/{name}" if relative else name
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
                    dir_fd=directory_fd,
                )
                try:
                    directories.append(path)
                    visit(child, path)
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                raw = _read_raw_at(directory_fd, name, name=f"authority tree {path}")
                files.append(_reference(path, raw))
            else:
                _fail("authority tree contains symlink or special member")

    visit(root_fd, prefix)
    files.sort(key=lambda item: os.fsencode(item["path"]))
    directories.sort(key=os.fsencode)
    return directories, files


def _tree_sha256(files: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in files:
        path = _relative_path(row.get("path"), name="tree record path")
        size = row.get("bytes")
        if type(size) is not int or size < 0:
            _fail("tree record size changed")
        sha = _sha(row.get("sha256"), name="tree record hash")
        # The frozen preissue called these paths project-relative, but its
        # published digest was formed from paths relative to the RUN003 root.
        # Preserve those exact bytes; the separately bound run root supplies
        # the missing outer path identity.
        digest.update(f"{path}\t{size}\t{sha}\n".encode("utf-8"))
    return digest.hexdigest()


def project_failed_prestate(run_fd: int) -> dict[str, Any]:
    """Hash RUN003 raw bytes and structural authority without decoding outcomes."""

    run_fd = _dir_fd(run_fd, name="RUN003 descriptor")
    _directories, files = _scan_tree(run_fd)
    top = sorted(os.listdir(run_fd), key=os.fsencode)
    reports_fd = _open_relative(run_fd, "reports", directory=True)
    try:
        reports = sorted(os.listdir(reports_fd), key=os.fsencode)
    finally:
        os.close(reports_fd)
    result_completed = [
        row
        for row in files
        if re.fullmatch(r"runtime/ledgers/task_[0-9]{3}/result\.completed\.json", row["path"])
    ]
    return {
        "canonical_tree_sha256": _tree_sha256(files),
        "regular_file_count": len(files),
        "regular_file_bytes": sum(int(row["bytes"]) for row in files),
        "top_level_members": top,
        "reports_members": reports,
        "development_manifest_absent": "development_manifest.json" not in top,
        "complete_absent": "COMPLETE.json" not in top,
        "result_completed_count": len(result_completed),
    }


def _validate_failed_prestate(value: Any, *, fixed: bool) -> dict[str, Any]:
    prestate = _mapping(value, name="failed prestate")
    _keys(prestate, FAILED_PRESTATE_KEYS, name="failed prestate")
    _sha(prestate.get("canonical_tree_sha256"), name="failed prestate tree hash")
    for key in ("regular_file_count", "regular_file_bytes", "result_completed_count"):
        if type(prestate.get(key)) is not int or prestate[key] < 0:
            _fail(f"failed prestate {key} changed")
    top = prestate.get("top_level_members")
    if (
        type(top) is not list
        or any(type(item) is not str for item in top)
        or top != sorted(top, key=os.fsencode)
    ):
        _fail("failed prestate top-level members changed")
    if type(prestate.get("reports_members")) is not list or prestate.get("reports_members") != []:
        _fail("failed prestate reports are not empty")
    if prestate.get("development_manifest_absent") is not True or prestate.get("complete_absent") is not True:
        _fail("failed prestate terminal absence changed")
    if fixed:
        expected = {
            "canonical_tree_sha256": FIXED_TREE_SHA256,
            "regular_file_count": FIXED_REGULAR_FILE_COUNT,
            "regular_file_bytes": FIXED_REGULAR_FILE_BYTES,
            "top_level_members": sorted(FAILED_TOP_LEVEL_MEMBERS),
            "reports_members": [],
            "development_manifest_absent": True,
            "complete_absent": True,
            "result_completed_count": 126,
        }
        if prestate != expected:
            _fail("failed RUN003 prestate differs from preissued authority")
    return prestate


def _json_at(run_fd: int, relative: str, *, name: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_raw_at(run_fd, relative, name=name)
    return raw, _strict_json_bytes(raw, name=name)


def project_original_science_authority(run_fd: int) -> dict[str, Any]:
    """Project only source/session/task/result receipt identities from RUN003."""

    source_raw, source = _json_at(run_fd, "runtime/source_set.json", name="source set")
    session_raw, session = _json_at(
        run_fd, "runtime/execution_session.json", name="execution session"
    )
    index_raw, task_index = _json_at(run_fd, "task_index.json", name="task index")
    source_reference = _reference("runtime/source_set.json", source_raw)
    session_reference = _reference("runtime/execution_session.json", session_raw)
    index_reference = _reference("task_index.json", index_raw)
    if (
        source_reference["sha256"] != FIXED_SOURCE_SET_RAW_SHA256
        or session_reference["sha256"] != FIXED_EXECUTION_SESSION_RAW_SHA256
        or index_reference["sha256"] != FIXED_TASK_INDEX_RAW_SHA256
    ):
        _fail("RUN003 source/session/task raw authority changed")
    _payload_hash(source, name="source set")
    _payload_hash(task_index, name="task index")
    if source.get("source_set_sha256") != FIXED_EXECUTION_SOURCE_SET_SHA256:
        _fail("RUN003 execution source-set identity changed")
    if source.get("canonical_config_sha256") != FIXED_CONFIG_SHA256:
        _fail("RUN003 source-set config identity changed")
    if session.get("source_set_reference") != source_reference:
        _fail("execution session source-set raw reference changed")
    if (
        session.get("source_set_sha256") != FIXED_EXECUTION_SOURCE_SET_SHA256
        or session.get("controller_source_sha256") != FIXED_CONTROLLER_SOURCE_SHA256
        or session.get("canonical_config_sha256") != FIXED_CONFIG_SHA256
        or session.get("task_index_reference") != index_reference
    ):
        _fail("RUN003 execution session authority changed")
    task_ids = [f"task_{ordinal:03d}" for ordinal in range(126)]
    if task_index.get("task_id_order") != task_ids:
        _fail("RUN003 task order changed")
    task_refs = _mapping(task_index.get("task_spec_references"), name="task references")
    if list(task_refs) != task_ids:
        _fail("RUN003 task reference order changed")
    for task_id in task_ids:
        _verify_reference_at(
            run_fd,
            task_refs[task_id],
            name=f"task reference {task_id}",
            expected_path=f"tasks/{task_id}.json",
        )
    result_references: list[dict[str, Any]] = []
    for task_id in task_ids:
        path = f"runtime/ledgers/{task_id}/result.completed.json"
        raw, phase = _json_at(run_fd, path, name=f"result phase {task_id}")
        if (
            phase.get("contract")
            != "openaxes.external.native_sensor_experts.v2.phase_receipt.v1"
            or phase.get("schema_version") != 1
            or phase.get("phase") != "result.completed"
            or phase.get("task_id") != task_id
        ):
            _fail("RUN003 result-completed phase identity changed")
        origin = _mapping(phase.get("attempt_origin_binding"), name="attempt origin")
        if (
            origin.get("source_set_sha256") != FIXED_EXECUTION_SOURCE_SET_SHA256
            or origin.get("controller_source_sha256") != FIXED_CONTROLLER_SOURCE_SHA256
            or origin.get("scientific_fit_attempt") != 1
        ):
            _fail("RUN003 result phase source/attempt authority changed")
        phase_payload = _mapping(phase.get("payload"), name="result phase payload")
        if _sha(phase.get("payload_sha256"), name="result phase payload hash") != canonical_sha256(
            phase_payload
        ):
            _fail("RUN003 result phase payload hash changed")
        semantic = _sha(phase.get("phase_receipt_sha256"), name="phase receipt hash")
        base = dict(phase)
        base.pop("phase_receipt_sha256", None)
        if semantic != canonical_sha256(base):
            _fail("RUN003 result phase semantic hash changed")
        result_references.append(_reference(path, raw))
    return {
        "run_id": RUN_ID,
        "canonical_config_sha256": FIXED_CONFIG_SHA256,
        "execution_source_set_sha256": FIXED_EXECUTION_SOURCE_SET_SHA256,
        "controller_source_sha256": FIXED_CONTROLLER_SOURCE_SHA256,
        "source_set_reference": source_reference,
        "execution_session_reference": session_reference,
        "task_index_reference": index_reference,
        "result_completed_reference_set_sha256": canonical_sha256(result_references),
        "result_completed_count": len(result_references),
    }


def _validate_original_science_authority(value: Any) -> dict[str, Any]:
    authority = _mapping(value, name="original science authority")
    _keys(authority, ORIGINAL_SCIENCE_AUTHORITY_KEYS, name="original science authority")
    if (
        authority.get("run_id") != RUN_ID
        or authority.get("canonical_config_sha256") != FIXED_CONFIG_SHA256
        or authority.get("execution_source_set_sha256") != FIXED_EXECUTION_SOURCE_SET_SHA256
        or authority.get("controller_source_sha256") != FIXED_CONTROLLER_SOURCE_SHA256
        or authority.get("result_completed_count") != 126
    ):
        _fail("original science authority identity changed")
    _sha(
        authority.get("result_completed_reference_set_sha256"),
        name="result-completed reference-set hash",
    )
    for key, path in (
        ("source_set_reference", "runtime/source_set.json"),
        ("execution_session_reference", "runtime/execution_session.json"),
        ("task_index_reference", "task_index.json"),
    ):
        _validate_reference(authority.get(key), name=key, expected_path=path)
    if (
        authority["source_set_reference"]["sha256"] != FIXED_SOURCE_SET_RAW_SHA256
        or authority["execution_session_reference"]["sha256"]
        != FIXED_EXECUTION_SESSION_RAW_SHA256
        or authority["task_index_reference"]["sha256"] != FIXED_TASK_INDEX_RAW_SHA256
    ):
        _fail("original science raw references changed")
    return authority


def build_source_audit_receipt(
    repo_fd: int,
    project_fd: int,
    run_fd: int,
    *,
    implementation_git_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Build the outcome-neutral preexecution receipt without writing a file."""

    created_at_utc = _timestamp(created_at_utc, name="source-audit creation time")
    closure = _closure_projection(repo_fd, project_fd, implementation_git_commit)
    prestate = _validate_failed_prestate(project_failed_prestate(run_fd), fixed=True)
    authority = _validate_original_science_authority(project_original_science_authority(run_fd))
    erratum = _reference(
        RECOVERY_ERRATUM_PATH,
        _read_raw_at(
            project_fd,
            _project_local_path(RECOVERY_ERRATUM_PATH, name="recovery erratum path"),
            name="recovery erratum",
        ),
    )
    diagnostic = _reference(
        SIGNED_DIAGNOSTIC_RECEIPT_PATH,
        _read_raw_at(
            project_fd,
            _project_local_path(
                SIGNED_DIAGNOSTIC_RECEIPT_PATH, name="signed diagnostic receipt path"
            ),
            name="signed diagnostic receipt",
        ),
    )
    if diagnostic != {
        "path": SIGNED_DIAGNOSTIC_RECEIPT_PATH,
        "bytes": FIXED_DIAGNOSTIC_BYTES,
        "sha256": FIXED_DIAGNOSTIC_SHA256,
    }:
        _fail("signed diagnostic receipt differs from correction authority")
    receipt: dict[str, Any] = {
        "contract": SOURCE_AUDIT_CONTRACT,
        "schema_version": 1,
        "run_id": RUN_ID,
        "created_at_utc": created_at_utc,
        "repository": {
            "object_format": "sha1",
            "project_directory": PROJECT_DIRECTORY,
            "implementation_git_commit": implementation_git_commit,
            "historical_run004_erratum_git_commit": HISTORICAL_RUN004_ERRATUM_GIT_COMMIT,
            "preissued_recovery_erratum_git_commit": PREISSUED_RECOVERY_ERRATUM_GIT_COMMIT,
            "signed_diagnostic_receipt_git_commit": SIGNED_DIAGNOSTIC_RECEIPT_GIT_COMMIT,
            "source_audit_receipt_path": SOURCE_AUDIT_RECEIPT_PATH,
            "source_audit_evidence_commit_required": True,
            "implementation_commit_must_be_ancestor_of_evidence_commit": True,
            "preissued_authorities_must_be_ancestors_of_implementation_commit": True,
        },
        "source_closure": closure,
        "preissued_erratum_reference": erratum,
        "diagnostic_receipt_reference": diagnostic,
        "original_science_authority": authority,
        "failed_prestate": prestate,
        "assertions": dict(SOURCE_AUDIT_ASSERTIONS),
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    return validate_source_audit_receipt(receipt)


def validate_source_audit_receipt(value: Any) -> dict[str, Any]:
    receipt = _mapping(value, name="compound recovery source audit")
    _keys(receipt, SOURCE_AUDIT_KEYS, name="compound recovery source audit")
    if (
        receipt.get("contract") != SOURCE_AUDIT_CONTRACT
        or type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 1
        or receipt.get("run_id") != RUN_ID
    ):
        _fail("compound recovery source-audit identity changed")
    _timestamp(receipt.get("created_at_utc"), name="source-audit creation time")
    repository = _mapping(receipt.get("repository"), name="source-audit repository")
    _keys(repository, REPOSITORY_KEYS, name="source-audit repository")
    if (
        repository.get("object_format") != "sha1"
        or repository.get("project_directory") != PROJECT_DIRECTORY
        or repository.get("source_audit_receipt_path") != SOURCE_AUDIT_RECEIPT_PATH
        or repository.get("source_audit_evidence_commit_required") is not True
        or repository.get("implementation_commit_must_be_ancestor_of_evidence_commit") is not True
        or repository.get("preissued_authorities_must_be_ancestors_of_implementation_commit")
        is not True
        or repository.get("historical_run004_erratum_git_commit")
        != HISTORICAL_RUN004_ERRATUM_GIT_COMMIT
        or repository.get("preissued_recovery_erratum_git_commit")
        != PREISSUED_RECOVERY_ERRATUM_GIT_COMMIT
        or repository.get("signed_diagnostic_receipt_git_commit")
        != SIGNED_DIAGNOSTIC_RECEIPT_GIT_COMMIT
    ):
        _fail("source-audit repository authority changed")
    _commit(repository.get("implementation_git_commit"), name="implementation commit")
    closure = _mapping(receipt.get("source_closure"), name="source closure")
    _keys(closure, SOURCE_CLOSURE_KEYS, name="source closure")
    if closure.get("path_order") != list(RECOVERY_SOURCE_CLOSURE):
        _fail("source closure path order changed")
    files = _mapping(closure.get("files"), name="source closure files")
    if set(files) != set(RECOVERY_SOURCE_CLOSURE) or closure.get("file_count") != len(
        RECOVERY_SOURCE_CLOSURE
    ):
        _fail("source closure membership changed")
    rows: list[dict[str, Any]] = []
    for path in RECOVERY_SOURCE_CLOSURE:
        row = _mapping(files[path], name=f"source file {path}")
        _keys(row, SOURCE_FILE_KEYS, name=f"source file {path}")
        if row.get("path") != path or row.get("git_mode") not in ("100644", "100755"):
            _fail("source file path/mode changed")
        if type(row.get("bytes")) is not int or row["bytes"] < 0:
            _fail("source file byte count changed")
        _sha(row.get("sha256"), name="source file hash")
        _commit(row.get("git_blob_oid"), name="source Git blob OID")
        rows.append(row)
    if closure.get("set_sha256") != canonical_sha256(rows):
        _fail("source closure set hash changed")
    erratum = _validate_reference(
        receipt.get("preissued_erratum_reference"),
        name="preissued recovery erratum",
        expected_path=RECOVERY_ERRATUM_PATH,
    )
    diagnostic = _validate_reference(
        receipt.get("diagnostic_receipt_reference"),
        name="signed diagnostic receipt",
        expected_path=SIGNED_DIAGNOSTIC_RECEIPT_PATH,
    )
    for reference, path in ((erratum, RECOVERY_ERRATUM_PATH), (diagnostic, SIGNED_DIAGNOSTIC_RECEIPT_PATH)):
        row = files[path]
        if reference != {key: row[key] for key in ("path", "bytes", "sha256")}:
            _fail("source-audit document reference differs from closure")
    if diagnostic["bytes"] != FIXED_DIAGNOSTIC_BYTES or diagnostic["sha256"] != FIXED_DIAGNOSTIC_SHA256:
        _fail("signed diagnostic correction identity changed")
    _validate_original_science_authority(receipt.get("original_science_authority"))
    _validate_failed_prestate(receipt.get("failed_prestate"), fixed=True)
    if receipt.get("assertions") != SOURCE_AUDIT_ASSERTIONS:
        _fail("source-audit outcome-neutral assertions changed")
    _payload_hash(receipt, name="compound recovery source audit")
    return dict(receipt)


def validate_source_audit_live(
    value: Any,
    *,
    receipt_raw: bytes,
    repo_fd: int,
    project_fd: int,
    run_fd: int,
    evidence_git_commit: str,
    allow_exact_compound_outputs: bool = False,
) -> dict[str, Any]:
    receipt = validate_source_audit_receipt(value)
    if receipt_raw != canonical_json_bytes(receipt, final_newline=True):
        _fail("source-audit raw bytes are not canonical newline receipt bytes")
    expected_closure = _closure_projection(
        repo_fd, project_fd, receipt["repository"]["implementation_git_commit"]
    )
    if receipt["source_closure"] != expected_closure:
        _fail("live source closure differs from source audit")
    if allow_exact_compound_outputs:
        _directories, all_files = _scan_tree(run_fd)
        output_paths = set(STAGED_FILE_ORDER)
        existing = [row for row in all_files if row["path"] not in output_paths]
        output_members = {row["path"] for row in all_files if row["path"] in output_paths}
        present = [path in output_members for path in STAGED_FILE_ORDER]
        present_count = sum(present)
        if (
            present != [index < present_count for index in range(len(STAGED_FILE_ORDER))]
            or len(existing) != receipt["failed_prestate"]["regular_file_count"]
            or sum(int(row["bytes"]) for row in existing)
            != receipt["failed_prestate"]["regular_file_bytes"]
            or _tree_sha256(existing) != receipt["failed_prestate"]["canonical_tree_sha256"]
        ):
            _fail("live RUN003 base prestate differs below compound outputs")
    elif receipt["failed_prestate"] != project_failed_prestate(run_fd):
        _fail("live RUN003 prestate differs from source audit")
    if receipt["original_science_authority"] != project_original_science_authority(run_fd):
        _fail("live RUN003 science authority differs from source audit")
    evidence = _commit(evidence_git_commit, name="source-audit evidence commit")
    _is_ancestor(repo_fd, receipt["repository"]["implementation_git_commit"], evidence)
    _mode, _oid, committed_receipt = _git_blob(repo_fd, evidence, SOURCE_AUDIT_RECEIPT_PATH)
    if committed_receipt != receipt_raw:
        _fail("source-audit evidence Git blob differs from supplied raw receipt")
    return receipt


def _validate_action_accounting(value: Any) -> dict[str, Any]:
    accounting = _mapping(value, name="compound recovery action accounting")
    if any(type(item) is not int for item in accounting.values()) or accounting != ACTION_ACCOUNTING:
        _fail("compound recovery action accounting changed")
    return accounting


def _validate_staged_references(value: Any, *, name: str) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != len(STAGED_FILE_ORDER):
        _fail(f"{name} count changed")
    result: list[dict[str, Any]] = []
    for path, reference in zip(STAGED_FILE_ORDER, value, strict=True):
        result.append(_validate_reference(reference, name=f"{name} {path}", expected_path=path))
    return result


def validate_stage_seal(value: Any, *, source_audit: Mapping[str, Any]) -> dict[str, Any]:
    seal = _mapping(value, name="compound recovery stage seal")
    _keys(seal, STAGE_SEAL_KEYS, name="compound recovery stage seal")
    if (
        seal.get("contract") != STAGE_SEAL_CONTRACT
        or type(seal.get("schema_version")) is not int
        or seal.get("schema_version") != 1
        or seal.get("run_id") != RUN_ID
    ):
        _fail("compound recovery stage-seal identity changed")
    _timestamp(seal.get("created_at_utc"), name="stage-seal creation time")
    source_raw = canonical_json_bytes(source_audit, final_newline=True)
    if seal.get("recovery_source_audit_reference") != _reference(SOURCE_AUDIT_RECEIPT_PATH, source_raw):
        _fail("stage seal source-audit reference changed")
    if seal.get("erratum_reference") != source_audit["preissued_erratum_reference"]:
        _fail("stage seal erratum reference changed")
    if seal.get("diagnostic_receipt_reference") != source_audit["diagnostic_receipt_reference"]:
        _fail("stage seal diagnostic reference changed")
    if seal.get("original_science_authority") != source_audit["original_science_authority"]:
        _fail("stage seal original science authority changed")
    if seal.get("failed_prestate") != source_audit["failed_prestate"]:
        _fail("stage seal failed prestate changed")
    if seal.get("staged_file_order") != list(STAGED_FILE_ORDER):
        _fail("stage seal output order changed")
    references = _validate_staged_references(seal.get("staged_file_references"), name="staged files")
    if seal.get("staged_file_set_sha256") != canonical_sha256(references):
        _fail("stage seal staged-file set hash changed")
    if seal.get("signed_report_status") not in ("complete", "incomplete"):
        _fail("stage seal signed-report status changed")
    _validate_action_accounting(seal.get("action_accounting"))
    _payload_hash(seal, name="compound recovery stage seal")
    return dict(seal)


def _poststate_projection(run_fd: int, output_references: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _directories, files = _scan_tree(run_fd)
    output_paths = set(STAGED_FILE_ORDER)
    existing = [row for row in files if row["path"] not in output_paths]
    output_by_path = {row["path"]: row for row in files if row["path"] in output_paths}
    if set(output_by_path) != output_paths:
        _fail("published output member set changed")
    outputs = [output_by_path[path] for path in STAGED_FILE_ORDER]
    if outputs != list(output_references):
        _fail("published output references differ from terminal receipt")
    top = sorted(os.listdir(run_fd), key=os.fsencode)
    return {
        "prestate_canonical_tree_sha256": _tree_sha256(existing),
        "added_file_order": list(STAGED_FILE_ORDER),
        "added_file_references": list(output_references),
        "added_file_count": len(output_references),
        "existing_prestate_unchanged": (
            len(existing) == FIXED_REGULAR_FILE_COUNT
            and sum(int(row["bytes"]) for row in existing) == FIXED_REGULAR_FILE_BYTES
            and _tree_sha256(existing) == FIXED_TREE_SHA256
        ),
        "complete_root_member_set_exact": top == list(COMPLETE_TOP_LEVEL_MEMBERS),
    }


def validate_compound_terminal_receipt(
    value: Any,
    *,
    source_audit: Mapping[str, Any],
    source_audit_raw: bytes,
    stage_seal: Mapping[str, Any],
    stage_seal_raw: bytes,
    run_fd: int,
    stage_fd: int,
) -> dict[str, Any]:
    source = validate_source_audit_receipt(source_audit)
    seal = validate_stage_seal(stage_seal, source_audit=source)
    terminal = _mapping(value, name="compound recovery terminal")
    _keys(terminal, TERMINAL_KEYS, name="compound recovery terminal")
    if (
        terminal.get("contract") != TERMINAL_CONTRACT
        or type(terminal.get("schema_version")) is not int
        or terminal.get("schema_version") != 1
        or terminal.get("run_id") != RUN_ID
    ):
        _fail("compound recovery terminal identity changed")
    _timestamp(terminal.get("created_at_utc"), name="terminal creation time")
    if terminal.get("erratum_reference") != source["preissued_erratum_reference"]:
        _fail("terminal erratum reference changed")
    if terminal.get("diagnostic_receipt_reference") != source["diagnostic_receipt_reference"]:
        _fail("terminal diagnostic reference changed")
    if terminal.get("recovery_source_audit_reference") != _reference(
        SOURCE_AUDIT_RECEIPT_PATH, source_audit_raw
    ):
        _fail("terminal source-audit reference changed")
    if terminal.get("original_science_authority") != source["original_science_authority"]:
        _fail("terminal original science authority changed")
    if terminal.get("failed_prestate") != source["failed_prestate"]:
        _fail("terminal failed prestate changed")
    if terminal.get("stage_seal_reference") != _reference("STAGE_SEAL.json", stage_seal_raw):
        _fail("terminal stage-seal reference changed")
    staged = _validate_staged_references(terminal.get("staged_file_references"), name="terminal staged files")
    outputs = _validate_staged_references(terminal.get("output_file_references"), name="terminal output files")
    if staged != seal["staged_file_references"] or outputs != staged:
        _fail("terminal staged/published raw references changed")
    for path, reference in zip(STAGED_FILE_ORDER, staged, strict=True):
        if _verify_reference_at(stage_fd, reference, name=f"staged {path}", expected_path=path) != reference:
            _fail("staged recovery output changed")
        if _verify_reference_at(run_fd, reference, name=f"published {path}", expected_path=path) != reference:
            _fail("published recovery output changed")
    poststate = _mapping(terminal.get("poststate"), name="compound recovery poststate")
    _keys(poststate, POSTSTATE_KEYS, name="compound recovery poststate")
    expected_poststate = _poststate_projection(run_fd, outputs)
    if poststate != expected_poststate:
        _fail("compound recovery poststate changed")
    if terminal.get("existing_prestate_unchanged") is not True or not poststate["existing_prestate_unchanged"]:
        _fail("RUN003 prestate did not remain byte-identical")
    if terminal.get("signed_report_status") != seal["signed_report_status"]:
        _fail("terminal signed-report status changed")
    if terminal.get("action_accounting") != seal["action_accounting"]:
        _fail("terminal action accounting differs from stage seal")
    _validate_action_accounting(terminal.get("action_accounting"))
    semantic = _mapping(terminal.get("semantic_validation"), name="semantic validation")
    _keys(semantic, SEMANTIC_VALIDATION_KEYS, name="semantic validation")
    if any(value is not True for value in semantic.values()):
        _fail("compound recovery semantic validation is not all true")
    if terminal.get("compound_terminal_required_for_publication") is not True:
        _fail("compound terminal publication requirement changed")
    _payload_hash(terminal, name="compound recovery terminal")
    return dict(terminal)


def validate_compound_terminal_live(
    terminal_raw: bytes,
    *,
    source_audit_raw: bytes,
    stage_seal_raw: bytes,
    repo_fd: int,
    project_fd: int,
    run_fd: int,
    stage_fd: int,
    evidence_git_commit: str,
) -> dict[str, Any]:
    source = _strict_json_bytes(source_audit_raw, name="source-audit receipt")
    validate_source_audit_live(
        source,
        receipt_raw=source_audit_raw,
        repo_fd=repo_fd,
        project_fd=project_fd,
        run_fd=run_fd,
        evidence_git_commit=evidence_git_commit,
        allow_exact_compound_outputs=True,
    )
    seal = _strict_json_bytes(stage_seal_raw, name="stage seal")
    terminal = _strict_json_bytes(terminal_raw, name="compound terminal")
    if stage_seal_raw != canonical_json_bytes(seal, final_newline=True):
        _fail("stage-seal raw bytes are not canonical newline bytes")
    if terminal_raw != canonical_json_bytes(terminal, final_newline=True):
        _fail("compound-terminal raw bytes are not canonical newline bytes")
    return validate_compound_terminal_receipt(
        terminal,
        source_audit=source,
        source_audit_raw=source_audit_raw,
        stage_seal=seal,
        stage_seal_raw=stage_seal_raw,
        run_fd=run_fd,
        stage_fd=stage_fd,
    )


def _read_cli_input(
    root_fd: int,
    relative: str | None,
    descriptor: int | None,
    *,
    name: str,
    repository_relative_project_path: bool = False,
) -> bytes:
    if (relative is None) == (descriptor is None):
        _fail(f"{name} requires exactly one relative path or inherited descriptor")
    if relative is not None:
        path = (
            _project_local_path(relative, name=name)
            if repository_relative_project_path
            else _relative_path(relative, name=name)
        )
        return _read_raw_at(root_fd, path, name=name)
    assert descriptor is not None
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise OpenAxesNativeV2RecoveryAuthorityError(f"{name} descriptor is not open") from exc
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{name} descriptor is not a regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _write_cli_output(raw: bytes, descriptor: int | None) -> None:
    if descriptor is None:
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()
        return
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != 0 or os.lseek(
        descriptor, 0, os.SEEK_CUR
    ) != 0:
        _fail("output descriptor is not an empty O_EXCL regular file")
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            _fail("cannot write source-audit output descriptor")
        view = view[written:]
    os.fsync(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAXES native-V2 compound recovery authority")
    sub = parser.add_subparsers(dest="command", required=True)

    issue = sub.add_parser("issue-source-audit")
    issue.add_argument("--repository-fd", type=int, required=True)
    issue.add_argument("--project-fd", type=int, required=True)
    issue.add_argument("--run-fd", type=int, required=True)
    issue.add_argument("--implementation-git-commit", required=True)
    issue.add_argument("--created-at-utc", required=True)
    issue.add_argument("--output-fd", type=int)

    validate = sub.add_parser("validate-source-audit")
    validate.add_argument("--repository-fd", type=int, required=True)
    validate.add_argument("--project-fd", type=int, required=True)
    validate.add_argument("--run-fd", type=int, required=True)
    receipt = validate.add_mutually_exclusive_group(required=True)
    receipt.add_argument("--receipt-relative")
    receipt.add_argument("--receipt-fd", type=int)
    validate.add_argument("--evidence-git-commit", required=True)

    terminal = sub.add_parser("validate-terminal")
    terminal.add_argument("--repository-fd", type=int, required=True)
    terminal.add_argument("--project-fd", type=int, required=True)
    terminal.add_argument("--run-fd", type=int, required=True)
    terminal.add_argument("--stage-fd", type=int, required=True)
    terminal.add_argument("--authority-fd", type=int, required=True)
    for label in ("source-audit", "stage-seal", "terminal"):
        group = terminal.add_mutually_exclusive_group(required=True)
        group.add_argument(f"--{label}-relative")
        group.add_argument(f"--{label}-fd", type=int)
    terminal.add_argument("--evidence-git-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "issue-source-audit":
        receipt = build_source_audit_receipt(
            args.repository_fd,
            args.project_fd,
            args.run_fd,
            implementation_git_commit=args.implementation_git_commit,
            created_at_utc=args.created_at_utc,
        )
        _write_cli_output(canonical_json_bytes(receipt, final_newline=True), args.output_fd)
        return 0
    if args.command == "validate-source-audit":
        if args.receipt_relative is not None and args.receipt_relative != SOURCE_AUDIT_RECEIPT_PATH:
            _fail("source-audit CLI path differs from the frozen tracked receipt path")
        raw = _read_cli_input(
            args.project_fd,
            args.receipt_relative,
            args.receipt_fd,
            name="source-audit receipt",
            repository_relative_project_path=True,
        )
        receipt = _strict_json_bytes(raw, name="source-audit receipt")
        validate_source_audit_live(
            receipt,
            receipt_raw=raw,
            repo_fd=args.repository_fd,
            project_fd=args.project_fd,
            run_fd=args.run_fd,
            evidence_git_commit=args.evidence_git_commit,
        )
        return 0
    if (
        args.source_audit_relative is not None
        and args.source_audit_relative != SOURCE_AUDIT_RECEIPT_PATH
    ):
        _fail("terminal source-audit CLI path differs from the frozen tracked receipt path")
    if args.stage_seal_relative is not None and args.stage_seal_relative != "STAGE_SEAL.json":
        _fail("stage-seal CLI path differs from the frozen stage member")
    if args.terminal_relative is not None and args.terminal_relative != "COMPOUND_TERMINAL.json":
        _fail("compound-terminal CLI path differs from the frozen stage member")
    source_raw = _read_cli_input(
        args.project_fd,
        args.source_audit_relative,
        args.source_audit_fd,
        name="source-audit receipt",
        repository_relative_project_path=True,
    )
    stage_raw = _read_cli_input(
        args.stage_fd,
        args.stage_seal_relative,
        args.stage_seal_fd,
        name="stage seal",
    )
    terminal_raw = _read_cli_input(
        args.authority_fd,
        args.terminal_relative,
        args.terminal_fd,
        name="compound terminal",
    )
    validated = validate_compound_terminal_live(
        terminal_raw,
        source_audit_raw=source_raw,
        stage_seal_raw=stage_raw,
        repo_fd=args.repository_fd,
        project_fd=args.project_fd,
        run_fd=args.run_fd,
        stage_fd=args.stage_fd,
        evidence_git_commit=args.evidence_git_commit,
    )
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "contract": VALIDATION_CONTRACT,
                "run_id": RUN_ID,
                "compound_terminal_sha256": hashlib.sha256(terminal_raw).hexdigest(),
                "valid": True,
                "performance_values_disclosed": False,
            },
            final_newline=True,
        )
    )
    sys.stdout.buffer.flush()
    if validated["compound_terminal_required_for_publication"] is not True:
        _fail("compound publication authority changed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OpenAxesNativeV2RecoveryAuthorityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
