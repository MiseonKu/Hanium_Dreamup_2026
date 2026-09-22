"""No-recompute salvage of the native-V2 RUN003 failed recovery stage.

The first compound-recovery invocation consumed its one authorized gate replay
and canonical signed-truth reconstruction before failing on a metadata-only
aliasing defect.  This module authenticates those immutable seven staged
files, deletes exactly the surplus summary self-reference in memory, copies
the other six files byte-for-byte to a fresh stage, and emits the standard
seal consumed by the already-audited publisher.  It has no aggregation,
training, prediction, label, evaluator, or public reconstruction entry point.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import stat
from typing import Any, Mapping, Sequence

from . import openaxes_external_native_v2_experiment as frozen
from . import openaxes_external_native_v2_recovery as prior
from . import openaxes_external_native_v2_recovery_authority as prior_authority


RUN_ID = prior.RUN_ID
SOURCE_AUDIT_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.compound_recovery."
    "salvage.source_audit.v1"
)
TERMINAL_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.compound_recovery."
    "salvage.terminal.v1"
)
VALIDATION_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.compound_recovery."
    "salvage.validation.v1"
)
PREISSUE_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.compound_recovery."
    "salvage_preissue.v1"
)
PROJECT_DIRECTORY = frozen.PROJECT_DIRECTORY
PREISSUE_COMMIT = "6a953c9996cc7ff446a1a6a3aa7e7c29ccc7044d"
SUPPLEMENT_COMMIT = "4b71a1686f45638f9b4087c85401478626fc2ed7"
PRIOR_IMPLEMENTATION_COMMIT = "c77e8bb30a5cb1040194b3b0ade35cbd78aa3977"
PRIOR_EVIDENCE_COMMIT = "9cf2c36a47eb203ea07840b0410eec96c639fb43"
ERRATUM2_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/docs/"
    "OPENAXES_EXTERNAL_NATIVE_V2_RUN003_RECOVERY_ERRATUM2.md"
)
ERRATUM2_BYTES = 4390
ERRATUM2_SHA256 = "d053b8bed1aa7e427d06688b4978ad9bf6c48e105dd6b6da29a16bda682ae906"
PREISSUE_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/receipts/"
    "openaxes_external_native_v2_compound_recovery_salvage_preissue.json"
)
PREISSUE_BYTES = 5149
PREISSUE_SHA256 = "0b15daf0d47878d57a798a34bdc5df09092b4523dc1bd3070cf0653fe7a4f1b5"
SUPPLEMENT_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/receipts/"
    "openaxes_external_native_v2_compound_recovery_salvage_accounting_supplement.json"
)
SUPPLEMENT_BYTES = 2995
SUPPLEMENT_SHA256 = "4df21c681f15cbcfdad43a17304e4f5e23da9478aba77ea804b525d25a7122be"
SOURCE_AUDIT_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/receipts/"
    "openaxes_external_native_v2_compound_recovery_salvage_source_audit.json"
)
SALVAGE_SOURCE_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/"
    "openaxes_external_native_v2_recovery_salvage.py"
)
SALVAGE_TEST_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/tests/"
    "test_openaxes_external_native_v2_recovery_salvage.py"
)
SOURCE_CLOSURE = (
    ERRATUM2_PATH,
    PREISSUE_PATH,
    SUPPLEMENT_PATH,
    SALVAGE_SOURCE_PATH,
    SALVAGE_TEST_PATH,
)
FAILED_STAGE_ROOT = (
    "/home/shin-server/.codex-supervisors/capstone-code/"
    "openaxes-native-v2-run003-compound-recovery-001/stage"
)
FAILED_OUTER_ROOT = os.path.dirname(FAILED_STAGE_ROOT)
ORIGINAL_AUTHORITY_ROOT = (
    "/home/shin-server/.codex-supervisors/capstone-code/"
    "openaxes-native-v2-run003-compound-recovery-001/authority"
)
LIFECYCLE_LOCK_PATH = "LIFECYCLE.lock"
TERMINAL_PATH = "SALVAGE_TERMINAL.json"
OUTPUT_PATHS = prior.OUTPUT_PATHS
FIRST_FOUR = OUTPUT_PATHS[:4]
SUMMARY_PATH = OUTPUT_PATHS[4]
METADATA_PATHS = OUTPUT_PATHS[4:]

FAILED_REFERENCES = (
    {"bytes": 151211, "path": "reports/zero_shift.json", "sha256": "177185d615a1e3dd9ed0ec775e638ce386d858706dee55136178fc81e879f956"},
    {"bytes": 1476, "path": "reports/primary_gates.json", "sha256": "f91309d249a84342e2142f7364a49cea815685eeeb580d0d285845c07e751973"},
    {"bytes": 29629, "path": "reports/stress_report.json", "sha256": "5dac7a6052fd72f4ed47659f1b96f1c6eed44aa6e92e7be78541ac09b9e94ffc"},
    {"bytes": 521875, "path": "reports/signed_shift_report.json", "sha256": "00ab104b75f6610d93612489e5eb08e43eff85f85901e2fb26ed91204c5c66c8"},
    {"bytes": 2273, "path": "reports/development_summary.json", "sha256": "7d8dea342ca1f476edbb8f45d8dfa39c3fdaf75031507c9f5e734eea7b45efff"},
    {"bytes": 21875, "path": "development_manifest.json", "sha256": "c493baff0e4ec8f0cd2912f3e066d9eba077fd12d9327828f49093816d69e833"},
    {"bytes": 934, "path": "COMPLETE.json", "sha256": "318ec3c2ef1f3a8453819e8b0cbf18ec23193307e1927070c44a5b3ccbfa8299"},
)
CORRECTED_REFERENCES = (
    *FAILED_REFERENCES[:4],
    {"bytes": 2113, "path": "reports/development_summary.json", "sha256": "86e97674b466f8c229f5b93e1f8c7fafb07c9e378850c484ff6fcd8707b69d14"},
    *FAILED_REFERENCES[5:],
)
FAILED_REFERENCE_SET_SHA256 = "8c5974a3de927372c90bd395b760cc873c1f6bdbe308d9a9d7401853f54d00d3"
CORRECTED_REFERENCE_SET_SHA256 = "8f8987c8a0268b019fc4e8f1b884f705b1285ecbc09880ad0d57425bdebb6eb6"
PRIOR_SOURCE_AUDIT_REFERENCE = {
    "bytes": 15299,
    "path": prior.RECOVERY_SOURCE_AUDIT_PATH,
    "sha256": "887ba564fee0f701df2fa69e233600a0f7266df2cbb4ffde166bfb628ae7df4c",
}
PREISSUE_ACCOUNTING = {
    "failed_recovery_canonical_reconstructions": 1,
    "failed_recovery_gate_replays": 1,
    "failed_recovery_invocations": 1,
    "failed_recovery_stage_writes": 7,
    "salvage_aggregate_calls": 0,
    "salvage_canonical_reconstructions": 0,
    "salvage_checkpoint_selections": 0,
    "salvage_evaluator_calls": 0,
    "salvage_fits": 0,
    "salvage_gate_replays": 0,
    "salvage_model_forwards": 0,
    "salvage_optimizer_updates": 0,
    "salvage_outer_feature_loads": 0,
    "salvage_outer_label_reads": 0,
}
EXACT_ACTION_ACCOUNTING = {
    "corrected_stage_file_writes": 7,
    "corrected_stage_seal_writes": 1,
    "exact_development_summary_key_deletions": 1,
    "failed_stage_post_projection_passes": 1,
    "failed_stage_pre_projection_passes": 1,
    "original_compound_publisher_invocations": 1,
    "original_compound_terminal_writes": 1,
    "raw_byte_exact_stage_file_copies": 6,
    "run003_ordered_o_excl_file_copies": 7,
    "salvage_aggregate_calls": 0,
    "salvage_canonical_reconstructions": 0,
    "salvage_checkpoint_selections": 0,
    "salvage_evaluator_calls": 0,
    "salvage_fits": 0,
    "salvage_gate_replays": 0,
    "salvage_invocations": 1,
    "salvage_model_forwards": 0,
    "salvage_optimizer_updates": 0,
    "salvage_outer_feature_loads": 0,
    "salvage_outer_label_reads": 0,
    "salvage_scientific_payload_recomputations": 0,
    "salvage_terminal_writes": 1,
    "summary_metadata_canonicalizations": 1,
}
OBSERVED_ACTION_ACCOUNTING = {
    **EXACT_ACTION_ACCOUNTING,
    "original_compound_publisher_invocations": 0,
    "original_compound_terminal_writes": 0,
    "run003_ordered_o_excl_file_copies": 0,
}
AUTHORIZED_REMAINING_ACTION_ACCOUNTING = {
    key: (
        7
        if key == "run003_ordered_o_excl_file_copies"
        else 1
        if key in {"original_compound_publisher_invocations", "original_compound_terminal_writes"}
        else 0
    )
    for key in EXACT_ACTION_ACCOUNTING
}
RUN003_PRESTATE = {
    "canonical_tree_sha256": "e42472225abdaaa081da7a33dfdbe6d3543a607b579654a8ef7172aeb944b08b",
    "complete_absent": True,
    "development_manifest_absent": True,
    "regular_file_bytes": 318851615,
    "regular_file_count": 2799,
    "reports_empty": True,
}
PRIOR_ERRATUM_REFERENCE = {
    "bytes": 5356,
    "path": (
        "09_SO101_IMU_VIVE_LEARNING/docs/"
        "OPENAXES_EXTERNAL_NATIVE_V2_RUN003_RECOVERY_ERRATUM.md"
    ),
    "sha256": "0bca94bfb7ac930a504ccd01fa557261da16b312f6eec74779a33101604aab6a",
}
PRIOR_AUTHORITY = {
    "recovery_erratum_reference": PRIOR_ERRATUM_REFERENCE,
    "recovery_implementation_commit": PRIOR_IMPLEMENTATION_COMMIT,
    "recovery_source_audit_evidence_commit": PRIOR_EVIDENCE_COMMIT,
    "recovery_source_audit_reference": PRIOR_SOURCE_AUDIT_REFERENCE,
}
EXTERNAL_CLOSED_WORLD = {
    "failed_stage_root": FAILED_STAGE_ROOT,
    "failed_stage_root_exact_members": [
        "COMPLETE.json",
        "development_manifest.json",
        "reports",
    ],
    "failed_stage_reports_exact_members": [
        "development_summary.json",
        "primary_gates.json",
        "signed_shift_report.json",
        "stress_report.json",
        "zero_shift.json",
    ],
    "original_authority_root": ORIGINAL_AUTHORITY_ROOT,
    "original_authority_root_exact_pre_salvage_members": [],
    "pre_and_post_identity_fields": [
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
        "sha256",
    ],
}
PUBLICATION_VALIDATION = {
    "fresh_process_success_count": 10,
    "must_raw_bind_original_compound_terminal": True,
    "must_raw_bind_salvage_terminal": True,
    "must_run_original_recovery_full_validate": True,
    "must_run_salvage_semantic_validate": True,
    "ordered_record_set_sha256_required": True,
}


class OpenAxesNativeV2RecoverySalvageError(RuntimeError):
    """Raised when salvage authority, evidence, or bytes are ambiguous."""


def _fail(message: str) -> None:
    raise OpenAxesNativeV2RecoverySalvageError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OpenAxesNativeV2RecoverySalvageError(
            "salvage authority is not finite canonical JSON"
        ) from exc


def _canonical_bytes(value: Any) -> bytes:
    return _canonical(value) + b"\n"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "payload_sha256" in result:
        _fail("payload input already has its self hash")
    result["payload_sha256"] = _canonical_hash(result)
    return result


def _payload_valid(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        _fail(f"{name} is not a string-keyed mapping")
    result = dict(value)
    claimed = result.pop("payload_sha256", None)
    if type(claimed) is not str or claimed != _canonical_hash(result):
        _fail(f"{name} payload hash changed")
    return dict(value)


def _strict_json(raw: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = frozen._strict_json(raw, name=name)
    except frozen.OpenAxesExternalNativeV2Error as exc:
        raise OpenAxesNativeV2RecoverySalvageError(str(exc)) from exc
    if not isinstance(value, Mapping):
        _fail(f"{name} is not an object")
    return dict(value)


def _reference(path: str, raw: bytes) -> dict[str, Any]:
    return {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _read(root_fd: int, path: str, *, name: str) -> bytes:
    try:
        return frozen._read_bytes(root_fd, path, name=name)
    except frozen.OpenAxesExternalNativeV2Error as exc:
        raise OpenAxesNativeV2RecoverySalvageError(str(exc)) from exc


def _read_regular_fd(fd: int, *, name: str) -> bytes:
    if type(fd) is not int or fd < 3:
        _fail(f"{name} descriptor is not an inherited non-stdio descriptor")
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        _fail(f"{name} is not regular")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(fd)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after):
        _fail(f"{name} changed while read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        _fail(f"{name} short read")
    return raw


def _read_path_with_stat(
    root_fd: int, path: str, *, name: str
) -> tuple[bytes, os.stat_result]:
    parts = frozen._safe_parts(path)
    parent = (
        frozen._open_dir(root_fd, "/".join(parts[:-1]))
        if len(parts) > 1
        else os.dup(root_fd)
    )
    descriptor = -1
    try:
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent
        )
        raw = _read_regular_fd(descriptor, name=name)
        info = os.fstat(descriptor)
        return raw, info
    except OSError as exc:
        raise OpenAxesNativeV2RecoverySalvageError(
            f"cannot read retained {name}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _stat_row(info: os.stat_result, *, sha256: str) -> dict[str, Any]:
    return {
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_mode": info.st_mode,
        "st_uid": info.st_uid,
        "st_gid": info.st_gid,
        "st_nlink": info.st_nlink,
        "st_size": info.st_size,
        "st_mtime_ns": info.st_mtime_ns,
        "st_ctime_ns": info.st_ctime_ns,
        "sha256": sha256,
    }


def _retained_path(fd: int, *, expected: str, name: str) -> None:
    try:
        observed = os.readlink(f"/proc/self/fd/{fd}")
    except OSError as exc:
        raise OpenAxesNativeV2RecoverySalvageError(
            f"cannot resolve retained {name} descriptor"
        ) from exc
    if observed != expected:
        _fail(f"retained {name} path differs from preissue")


def _owned_private_directory(fd: int, *, name: str) -> os.stat_result:
    info = os.fstat(fd)
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or info.st_uid != os.geteuid()
        or info.st_gid != os.getegid()
    ):
        _fail(f"{name} is not an exact owned private 0700 directory")
    return info


def _require_empty_private_directory(fd: int, *, name: str) -> os.stat_result:
    info = _owned_private_directory(fd, name=name)
    if frozen._names(fd) or info.st_nlink != 2:
        _fail(f"{name} is not a fresh empty private directory")
    return info


def _directory_binding(info: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_mode": info.st_mode,
        "st_uid": info.st_uid,
        "st_gid": info.st_gid,
        "st_nlink": info.st_nlink,
    }


def _require_distinct_directories(
    descriptors: Mapping[str, int],
) -> None:
    seen: dict[tuple[int, int], str] = {}
    for name, descriptor in descriptors.items():
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            _fail(f"{name} is not a directory")
        identity = (info.st_dev, info.st_ino)
        if identity in seen:
            _fail(f"directory descriptors alias: {seen[identity]} and {name}")
        seen[identity] = name


def _stage_projection(
    stage_fd: int,
    *,
    require_failed_path: bool,
    sealed: bool = False,
) -> dict[str, Any]:
    root = os.fstat(stage_fd)
    if (
        not stat.S_ISDIR(root.st_mode)
        or stat.S_IMODE(root.st_mode) != 0o700
        or root.st_uid != os.geteuid()
        or root.st_gid != os.getegid()
    ):
        _fail("stage root is not an exact owned private 0700 directory")
    if require_failed_path:
        _retained_path(stage_fd, expected=FAILED_STAGE_ROOT, name="failed stage")
    expected_root = ["COMPLETE.json", "development_manifest.json", "reports"]
    if sealed:
        expected_root.append(prior.STAGE_SEAL_PATH)
    if frozen._names(stage_fd) != sorted(expected_root) or frozen._names(
        stage_fd, "reports"
    ) != sorted(prior.REPORT_NAMES):
        _fail("stage closed roster changed")
    raw_by_path: dict[str, bytes] = {}
    file_rows: list[dict[str, Any]] = []
    output_references: list[dict[str, Any]] = []
    paths = [*OUTPUT_PATHS, *([prior.STAGE_SEAL_PATH] if sealed else [])]
    for path in paths:
        raw, info = _read_path_with_stat(
            stage_fd, path, name=f"staged {path}"
        )
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()
            or info.st_gid != os.getegid()
            or info.st_nlink != 1
            or info.st_size != len(raw)
        ):
            _fail(f"stage file identity changed: {path}")
        reference = _reference(path, raw)
        raw_by_path[path] = raw
        if path in OUTPUT_PATHS:
            output_references.append(reference)
        file_rows.append(
            {
                "path": path,
                "kind": "regular",
                **_stat_row(info, sha256=reference["sha256"]),
            }
        )
    reports_fd = frozen._open_dir(stage_fd, "reports")
    try:
        reports_info = os.fstat(reports_fd)
    finally:
        os.close(reports_fd)
    if (
        stat.S_IMODE(reports_info.st_mode) != 0o700
        or reports_info.st_uid != os.geteuid()
        or reports_info.st_gid != os.getegid()
    ):
        _fail("stage reports-directory authority changed")
    report_child_refs = [
        _reference(path, raw_by_path[path]) for path in OUTPUT_PATHS[:5]
    ]
    reports_sha = _canonical_hash(report_child_refs)
    reports_row = {
        "path": "reports",
        "kind": "directory",
        **_stat_row(reports_info, sha256=reports_sha),
    }
    root_children = [
        {"path": "reports", "kind": "directory", "sha256": reports_sha},
        *[
            {"path": row["path"], "kind": "regular", "sha256": row["sha256"]}
            for row in file_rows
            if "/" not in row["path"]
        ],
    ]
    root_sha = _canonical_hash(sorted(root_children, key=lambda row: row["path"]))
    root_row = {"path": ".", "kind": "directory", **_stat_row(root, sha256=root_sha)}
    rows = [root_row, reports_row, *file_rows]
    return {
        "rows": rows,
        "file_references": output_references,
        "projection_sha256": _canonical_hash(rows),
    }


def _external_projection(
    failed_outer_fd: int,
    failed_stage_fd: int,
    original_authority_fd: int,
) -> dict[str, Any]:
    _retained_path(failed_outer_fd, expected=FAILED_OUTER_ROOT, name="failed outer root")
    _retained_path(failed_stage_fd, expected=FAILED_STAGE_ROOT, name="failed stage")
    _retained_path(
        original_authority_fd,
        expected=ORIGINAL_AUTHORITY_ROOT,
        name="original authority",
    )
    if frozen._names(original_authority_fd):
        _fail("original compound authority directory is not empty")
    authority_info = os.fstat(original_authority_fd)
    if (
        not stat.S_ISDIR(authority_info.st_mode)
        or stat.S_IMODE(authority_info.st_mode) != 0o700
        or authority_info.st_uid != os.geteuid()
        or authority_info.st_gid != os.getegid()
    ):
        _fail("original authority is not an exact private directory")
    if frozen._names(failed_outer_fd) != [
        LIFECYCLE_LOCK_PATH,
        "authority",
        "stage",
    ]:
        _fail("failed external root closed roster changed")
    stage_child = frozen._open_dir(failed_outer_fd, "stage")
    authority_child = frozen._open_dir(failed_outer_fd, "authority")
    try:
        outer_info = os.fstat(failed_outer_fd)
        stage_child_info = os.fstat(stage_child)
        authority_child_info = os.fstat(authority_child)
    finally:
        os.close(stage_child)
        os.close(authority_child)
    stage_info = os.fstat(failed_stage_fd)
    if (stage_child_info.st_dev, stage_child_info.st_ino) != (
        stage_info.st_dev,
        stage_info.st_ino,
    ) or (authority_child_info.st_dev, authority_child_info.st_ino) != (
        authority_info.st_dev,
        authority_info.st_ino,
    ):
        _fail("failed stage/authority descriptors are not exact outer-root children")
    if (
        not stat.S_ISDIR(outer_info.st_mode)
        or stat.S_IMODE(outer_info.st_mode) != 0o700
        or outer_info.st_uid != os.geteuid()
        or outer_info.st_gid != os.getegid()
    ):
        _fail("failed external root is not an exact private directory")
    lock_raw, lock_info = _read_path_with_stat(
        failed_outer_fd, LIFECYCLE_LOCK_PATH, name="failed lifecycle lock"
    )
    if (
        not stat.S_ISREG(lock_info.st_mode)
        or lock_info.st_uid != os.geteuid()
        or lock_info.st_gid != os.getegid()
        or lock_info.st_nlink != 1
        or lock_info.st_size != len(lock_raw)
    ):
        _fail("failed lifecycle lock identity changed")
    lock_row = {
        "path": LIFECYCLE_LOCK_PATH,
        "kind": "regular",
        **_stat_row(lock_info, sha256=hashlib.sha256(lock_raw).hexdigest()),
    }
    authority_row = {
        "path": "authority",
        "kind": "directory",
        **_stat_row(authority_info, sha256=_canonical_hash([])),
    }
    failed = _stage_projection(
        failed_stage_fd, require_failed_path=True, sealed=False
    )
    outer_children = [
        {"path": LIFECYCLE_LOCK_PATH, "kind": "regular", "sha256": lock_row["sha256"]},
        {"path": "authority", "kind": "directory", "sha256": authority_row["sha256"]},
        {"path": "stage", "kind": "directory", "sha256": failed["rows"][0]["sha256"]},
    ]
    outer_row = {
        "path": ".",
        "kind": "directory",
        **_stat_row(outer_info, sha256=_canonical_hash(outer_children)),
    }
    return {
        "external_root": outer_row,
        "lifecycle_lock": lock_row,
        "failed_stage": failed,
        "original_authority": authority_row,
        "projection_sha256": _canonical_hash(
            {
                "external_root": outer_row,
                "lifecycle_lock": lock_row,
                "failed_stage": failed,
                "original_authority": authority_row,
            }
        ),
    }


def _validate_stage_projection_value(
    value: Any,
    *,
    expected_references: Sequence[Mapping[str, Any]],
    sealed: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "rows", "file_references", "projection_sha256"
    }:
        _fail("stage projection schema changed")
    projection = dict(value)
    rows = projection["rows"]
    expected_paths = [
        ".",
        "reports",
        *OUTPUT_PATHS,
        *([prior.STAGE_SEAL_PATH] if sealed else []),
    ]
    if (
        not isinstance(rows, list)
        or len(rows) != len(expected_paths)
        or projection["file_references"] != list(expected_references)
        or projection["projection_sha256"] != _canonical_hash(rows)
    ):
        _fail("stage projection roster/hash changed")
    row_keys = {
        "path",
        "kind",
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
        "sha256",
    }
    by_path: dict[str, Mapping[str, Any]] = {}
    for expected_path, row in zip(expected_paths, rows, strict=True):
        expected_kind = "directory" if expected_path in {".", "reports"} else "regular"
        if (
            not isinstance(row, Mapping)
            or set(row) != row_keys
            or row.get("path") != expected_path
            or row.get("kind") != expected_kind
            or any(
                type(row.get(key)) is not int or row[key] < 0
                for key in (
                    "st_dev", "st_ino", "st_mode", "st_uid", "st_gid",
                    "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns",
                )
            )
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None
        ):
            _fail("stage projection identity row changed")
        by_path[expected_path] = row
    for reference in expected_references:
        row = by_path[reference["path"]]
        if row["sha256"] != reference["sha256"] or row["st_size"] != reference["bytes"]:
            _fail("stage projection file reference changed")
    report_sha = _canonical_hash(list(expected_references)[:5])
    if by_path["reports"]["sha256"] != report_sha:
        _fail("stage projection reports-directory hash changed")
    root_children = [
        {"path": "reports", "kind": "directory", "sha256": report_sha},
        *[
            {
                "path": path,
                "kind": "regular",
                "sha256": by_path[path]["sha256"],
            }
            for path in expected_paths
            if path not in {".", "reports"} and "/" not in path
        ],
    ]
    if by_path["."]["sha256"] != _canonical_hash(
        sorted(root_children, key=lambda row: row["path"])
    ):
        _fail("stage projection root hash changed")
    return projection


def _validate_external_projection_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "external_root",
        "lifecycle_lock",
        "failed_stage",
        "original_authority",
        "projection_sha256",
    }:
        _fail("failed external projection schema changed")
    result = dict(value)
    body = {
        key: result[key]
        for key in (
            "external_root",
            "lifecycle_lock",
            "failed_stage",
            "original_authority",
        )
    }
    if result["projection_sha256"] != _canonical_hash(body):
        _fail("failed external projection hash changed")
    _validate_stage_projection_value(
        result["failed_stage"],
        expected_references=FAILED_REFERENCES,
        sealed=False,
    )
    for key, expected_path, expected_kind in (
        ("external_root", ".", "directory"),
        ("lifecycle_lock", LIFECYCLE_LOCK_PATH, "regular"),
        ("original_authority", "authority", "directory"),
    ):
        row = result[key]
        if not isinstance(row, Mapping) or row.get("path") != expected_path or row.get(
            "kind"
        ) != expected_kind or set(row) != {
            "path",
            "kind",
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "sha256",
        } or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None:
            _fail("failed external identity row changed")
    return result


def _load_preissue(project_fd: int) -> tuple[dict[str, Any], dict[str, Any]]:
    local = PREISSUE_PATH.removeprefix(f"{PROJECT_DIRECTORY}/")
    raw = _read(project_fd, local, name="salvage preissue")
    reference = _reference(PREISSUE_PATH, raw)
    if reference != {"path": PREISSUE_PATH, "bytes": PREISSUE_BYTES, "sha256": PREISSUE_SHA256}:
        _fail("salvage preissue raw authority changed")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OpenAxesNativeV2RecoverySalvageError("salvage preissue is malformed") from exc
    receipt = _payload_valid(value, name="salvage preissue")
    if (
        receipt.get("contract") != PREISSUE_CONTRACT
        or receipt.get("schema_version") != 1
        or receipt.get("run_id") != RUN_ID
        or receipt.get("accounting") != PREISSUE_ACCOUNTING
        or receipt.get("failed_stage", {}).get("ordered_file_references") != list(FAILED_REFERENCES)
        or receipt.get("failed_stage", {}).get("ordered_reference_set_sha256") != FAILED_REFERENCE_SET_SHA256
        or receipt.get("corrected_stage", {}).get("ordered_file_references") != list(CORRECTED_REFERENCES)
        or receipt.get("corrected_stage", {}).get("ordered_reference_set_sha256") != CORRECTED_REFERENCE_SET_SHA256
        or receipt.get("known_defect", {}).get("removed_self_reference") != CORRECTED_REFERENCES[4]
        or receipt.get("prior_authority", {}).get("recovery_implementation_commit") != PRIOR_IMPLEMENTATION_COMMIT
        or receipt.get("prior_authority", {}).get("recovery_source_audit_evidence_commit") != PRIOR_EVIDENCE_COMMIT
        or receipt.get("prior_authority", {}).get("recovery_source_audit_reference") != PRIOR_SOURCE_AUDIT_REFERENCE
    ):
        _fail("salvage preissue semantics changed")
    return receipt, reference


def _load_erratum2(project_fd: int) -> dict[str, Any]:
    local = ERRATUM2_PATH.removeprefix(f"{PROJECT_DIRECTORY}/")
    raw = _read(project_fd, local, name="salvage erratum2")
    reference = _reference(ERRATUM2_PATH, raw)
    if reference != {"path": ERRATUM2_PATH, "bytes": ERRATUM2_BYTES, "sha256": ERRATUM2_SHA256}:
        _fail("salvage erratum2 raw authority changed")
    return reference


def _load_supplement(
    project_fd: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    local = SUPPLEMENT_PATH.removeprefix(f"{PROJECT_DIRECTORY}/")
    raw = _read(project_fd, local, name="salvage accounting supplement")
    reference = _reference(SUPPLEMENT_PATH, raw)
    if reference != {
        "path": SUPPLEMENT_PATH,
        "bytes": SUPPLEMENT_BYTES,
        "sha256": SUPPLEMENT_SHA256,
    }:
        _fail("salvage accounting supplement raw authority changed")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OpenAxesNativeV2RecoverySalvageError(
            "salvage accounting supplement is malformed"
        ) from exc
    supplement = _payload_valid(value, name="salvage accounting supplement")
    if (
        supplement.get("contract")
        != "openaxes.external.native_sensor_experts.v2.compound_recovery.salvage_accounting_supplement.v1"
        or supplement.get("schema_version") != 1
        or supplement.get("run_id") != RUN_ID
        or supplement.get("prior_preissue_reference")
        != {"path": PREISSUE_PATH, "bytes": PREISSUE_BYTES, "sha256": PREISSUE_SHA256}
        or supplement.get("exact_action_accounting") != EXACT_ACTION_ACCOUNTING
        or supplement.get("external_closed_world") != EXTERNAL_CLOSED_WORLD
        or supplement.get("publication_validation") != PUBLICATION_VALIDATION
    ):
        _fail("salvage accounting supplement semantics changed")
    return supplement, reference


def _validate_reference_set(references: Sequence[Mapping[str, Any]], *, corrected: bool) -> None:
    expected = CORRECTED_REFERENCES if corrected else FAILED_REFERENCES
    expected_hash = CORRECTED_REFERENCE_SET_SHA256 if corrected else FAILED_REFERENCE_SET_SHA256
    if list(references) != list(expected) or _canonical_hash(list(references)) != expected_hash:
        _fail("salvage ordered reference set changed")


def _correct_failed_outputs(stage_fd: int) -> tuple[dict[str, bytes], dict[str, Any]]:
    failed = {path: _read(stage_fd, path, name=f"failed-stage {path}") for path in OUTPUT_PATHS}
    references = [_reference(path, failed[path]) for path in OUTPUT_PATHS]
    _validate_reference_set(references, corrected=False)
    values = {path: _strict_json(raw, name=f"failed-stage {path}") for path, raw in failed.items()}
    # The four scientific reports must already be independently valid.
    prior._validate_zero_report(values[OUTPUT_PATHS[0]])
    prior._validate_gate_report(values[OUTPUT_PATHS[1]])
    prior._validate_stress_report(values[OUTPUT_PATHS[2]])
    prior.validate_signed_shift_recovery_report(values[OUTPUT_PATHS[3]])
    summary = values[SUMMARY_PATH]
    if set(summary) != {
        "contract", "run_id", "task_count", "candidate_count", "fold_count",
        "seed_count", "optimizer_updates", "all_model_forward_calls",
        "outer_label_reads", "task_evaluator_calls", "signed_truth_reconstruction_calls",
        "signed_report_status", "report_references", "recovery_source_audit_reference",
        "diagnostic_receipt_reference", "action_accounting",
        "compound_terminal_required_for_publication", "payload_sha256",
    }:
        _fail("failed summary closed schema changed")
    report_references = summary.get("report_references")
    if not isinstance(report_references, Mapping) or set(report_references) != set(prior.REPORT_NAMES):
        _fail("failed summary does not have the exact one surplus reference")
    removed = report_references.get("development_summary.json")
    corrected_summary = deepcopy(summary)
    corrected_summary["report_references"] = dict(report_references)
    del corrected_summary["report_references"]["development_summary.json"]
    corrected_summary_raw = _canonical_bytes(corrected_summary)
    if (
        removed != CORRECTED_REFERENCES[4]
        or _reference(SUMMARY_PATH, corrected_summary_raw) != CORRECTED_REFERENCES[4]
    ):
        _fail("failed summary self-reference is not the exact aliasing defect")
    _payload_valid(corrected_summary, name="corrected development summary")
    corrected = dict(failed)
    corrected[SUMMARY_PATH] = corrected_summary_raw
    corrected_references = [_reference(path, corrected[path]) for path in OUTPUT_PATHS]
    _validate_reference_set(corrected_references, corrected=True)
    if any(corrected[path] != failed[path] for path in (*FIRST_FOUR, *OUTPUT_PATHS[5:])):
        _fail("salvage changed a byte outside the development summary")
    return corrected, {"failed_references": references, "corrected_references": corrected_references}


def _source_file_projection(repo_fd: int, project_fd: int, commit: str) -> dict[str, Any]:
    prior_authority._verify_repo_project_binding(repo_fd, project_fd)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        _fail("salvage implementation commit is malformed")
    prior_authority._is_ancestor(repo_fd, PREISSUE_COMMIT, commit)
    prior_authority._is_ancestor(repo_fd, SUPPLEMENT_COMMIT, commit)
    files: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for path in SOURCE_CLOSURE:
        mode, oid, committed = prior_authority._git_blob(repo_fd, commit, path)
        local = path.removeprefix(f"{PROJECT_DIRECTORY}/")
        live = _read(project_fd, local, name=f"salvage source {path}")
        if live != committed:
            _fail(f"live salvage source differs from implementation commit: {path}")
        row = {"path": path, "bytes": len(live), "sha256": hashlib.sha256(live).hexdigest(), "git_blob_oid": oid, "git_mode": mode}
        files[path] = row
        rows.append(row)
    return {"path_order": list(SOURCE_CLOSURE), "files": files, "file_count": len(rows), "set_sha256": _canonical_hash(rows)}


def build_salvage_source_audit(
    repo_fd: int,
    project_fd: int,
    run_fd: int,
    failed_outer_fd: int,
    failed_stage_fd: int,
    original_authority_fd: int,
    *,
    implementation_git_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    _require_distinct_directories(
        {
            "repository": repo_fd,
            "project": project_fd,
            "RUN003": run_fd,
            "failed outer": failed_outer_fd,
            "failed stage": failed_stage_fd,
            "original authority": original_authority_fd,
        }
    )
    preissue, preissue_reference = _load_preissue(project_fd)
    erratum2_reference = _load_erratum2(project_fd)
    supplement, supplement_reference = _load_supplement(project_fd)
    projection = _external_projection(
        failed_outer_fd, failed_stage_fd, original_authority_fd
    )
    _validate_reference_set(
        projection["failed_stage"]["file_references"], corrected=False
    )
    observed_prestate = prior_authority.project_failed_prestate(run_fd)
    for key in (
        "canonical_tree_sha256",
        "regular_file_count",
        "regular_file_bytes",
        "development_manifest_absent",
        "complete_absent",
    ):
        if observed_prestate.get(key) != preissue["run003_prestate"].get(key):
            _fail("RUN003 prestate differs from salvage preissue")
    if observed_prestate.get("reports_members") != []:
        _fail("RUN003 reports are not empty before salvage")
    receipt = _payload(
        {
            "contract": SOURCE_AUDIT_CONTRACT,
            "schema_version": 1,
            "run_id": RUN_ID,
            "created_at_utc": created_at_utc,
            "repository": {
                "object_format": "sha1",
                "project_directory": PROJECT_DIRECTORY,
                "implementation_git_commit": implementation_git_commit,
                "preissue_git_commit": PREISSUE_COMMIT,
                "accounting_supplement_git_commit": SUPPLEMENT_COMMIT,
                "source_audit_receipt_path": SOURCE_AUDIT_PATH,
                "implementation_must_be_strict_ancestor_of_evidence": True,
            },
            "source_closure": _source_file_projection(repo_fd, project_fd, implementation_git_commit),
            "preissue_reference": preissue_reference,
            "accounting_supplement_reference": supplement_reference,
            "erratum2_reference": erratum2_reference,
            "prior_recovery_authority": preissue["prior_authority"],
            "run003_prestate": preissue["run003_prestate"],
            "failed_stage_references": list(FAILED_REFERENCES),
            "external_closed_world": supplement["external_closed_world"],
            "publication_validation": supplement["publication_validation"],
            "exact_action_accounting": supplement["exact_action_accounting"],
            "failed_external_projection": projection,
            "failed_external_projection_sha256": projection["projection_sha256"],
            "assertions": {
                "run003_mutation_allowed": False,
                "failed_stage_mutation_allowed": False,
                "scientific_recomputation_allowed": False,
                "fresh_corrected_stage_required": True,
                "separate_salvage_terminal_required": True,
            },
        }
    )
    return validate_salvage_source_audit_receipt(receipt)


def validate_salvage_source_audit_receipt(value: Any) -> dict[str, Any]:
    receipt = _payload_valid(value, name="salvage source audit")
    if set(receipt) != {
        "contract", "schema_version", "run_id", "created_at_utc", "repository",
        "source_closure", "preissue_reference", "accounting_supplement_reference", "erratum2_reference",
        "prior_recovery_authority", "run003_prestate", "failed_stage_references",
        "external_closed_world", "publication_validation", "exact_action_accounting",
        "failed_external_projection", "failed_external_projection_sha256",
        "assertions", "payload_sha256",
    } or receipt["contract"] != SOURCE_AUDIT_CONTRACT or type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1 or receipt["run_id"] != RUN_ID or not isinstance(receipt["created_at_utc"], str) or re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", receipt["created_at_utc"]) is None:
        _fail("salvage source-audit identity changed")
    repository = receipt["repository"]
    if not isinstance(repository, Mapping) or set(repository) != {
        "object_format", "project_directory", "implementation_git_commit",
        "preissue_git_commit", "accounting_supplement_git_commit",
        "source_audit_receipt_path",
        "implementation_must_be_strict_ancestor_of_evidence",
    } or repository.get("object_format") != "sha1" or repository.get("project_directory") != PROJECT_DIRECTORY or repository.get("preissue_git_commit") != PREISSUE_COMMIT or repository.get("accounting_supplement_git_commit") != SUPPLEMENT_COMMIT or repository.get("source_audit_receipt_path") != SOURCE_AUDIT_PATH or repository.get("implementation_must_be_strict_ancestor_of_evidence") is not True or re.fullmatch(r"[0-9a-f]{40}", str(repository.get("implementation_git_commit"))) is None:
        _fail("salvage source-audit repository authority changed")
    closure = receipt["source_closure"]
    if not isinstance(closure, Mapping) or set(closure) != {
        "path_order", "files", "file_count", "set_sha256"
    } or closure.get("path_order") != list(SOURCE_CLOSURE) or closure.get("file_count") != len(SOURCE_CLOSURE):
        _fail("salvage source-closure schema changed")
    files = closure.get("files")
    if not isinstance(files, Mapping) or set(files) != set(SOURCE_CLOSURE):
        _fail("salvage source-closure roster changed")
    rows: list[dict[str, Any]] = []
    for path in SOURCE_CLOSURE:
        row = files[path]
        if not isinstance(row, Mapping) or set(row) != {
            "path", "bytes", "sha256", "git_blob_oid", "git_mode"
        } or row.get("path") != path or type(row.get("bytes")) is not int or row["bytes"] < 1 or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None or re.fullmatch(r"[0-9a-f]{40}", str(row.get("git_blob_oid"))) is None or row.get("git_mode") != "100644":
            _fail("salvage source-closure file row changed")
        rows.append(dict(row))
    if closure.get("set_sha256") != _canonical_hash(rows):
        _fail("salvage source-closure set hash changed")
    if receipt["preissue_reference"] != {"path": PREISSUE_PATH, "bytes": PREISSUE_BYTES, "sha256": PREISSUE_SHA256} or receipt["accounting_supplement_reference"] != {"path": SUPPLEMENT_PATH, "bytes": SUPPLEMENT_BYTES, "sha256": SUPPLEMENT_SHA256} or receipt["erratum2_reference"] != {"path": ERRATUM2_PATH, "bytes": ERRATUM2_BYTES, "sha256": ERRATUM2_SHA256} or receipt["failed_stage_references"] != list(FAILED_REFERENCES):
        _fail("salvage source-audit preissued authority changed")
    if receipt["exact_action_accounting"] != EXACT_ACTION_ACCOUNTING:
        _fail("salvage source-audit action accounting changed")
    failed_projection = _validate_external_projection_value(
        receipt["failed_external_projection"]
    )
    if (
        receipt["prior_recovery_authority"] != PRIOR_AUTHORITY
        or receipt["run003_prestate"] != RUN003_PRESTATE
        or receipt["external_closed_world"] != EXTERNAL_CLOSED_WORLD
        or receipt["publication_validation"] != PUBLICATION_VALIDATION
        or re.fullmatch(r"[0-9a-f]{64}", str(receipt["failed_external_projection_sha256"])) is None
        or receipt["failed_external_projection_sha256"]
        != failed_projection["projection_sha256"]
    ):
        _fail("salvage source-audit nested authority changed")
    if receipt["assertions"] != {"run003_mutation_allowed": False, "failed_stage_mutation_allowed": False, "scientific_recomputation_allowed": False, "fresh_corrected_stage_required": True, "separate_salvage_terminal_required": True}:
        _fail("salvage source-audit assertions changed")
    return receipt


def validate_salvage_source_audit_live(
    value: Any,
    *,
    raw: bytes,
    repo_fd: int,
    project_fd: int,
    run_fd: int,
    failed_outer_fd: int,
    failed_stage_fd: int,
    original_authority_fd: int,
    evidence_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_distinct_directories(
        {
            "repository": repo_fd,
            "project": project_fd,
            "RUN003": run_fd,
            "failed outer": failed_outer_fd,
            "failed stage": failed_stage_fd,
            "original authority": original_authority_fd,
        }
    )
    receipt = validate_salvage_source_audit_receipt(value)
    if raw != _canonical_bytes(receipt):
        _fail("salvage source-audit raw bytes are not canonical newline bytes")
    implementation = receipt["repository"]["implementation_git_commit"]
    if receipt["source_closure"] != _source_file_projection(repo_fd, project_fd, implementation):
        _fail("live salvage source closure changed")
    if not re.fullmatch(r"[0-9a-f]{40}", evidence_commit) or evidence_commit == implementation:
        _fail("salvage evidence commit is not a strict later commit")
    prior_authority._is_ancestor(repo_fd, implementation, evidence_commit)
    _mode, _oid, committed = prior_authority._git_blob(repo_fd, evidence_commit, SOURCE_AUDIT_PATH)
    if committed != raw:
        _fail("salvage source-audit evidence blob changed")
    projection = _external_projection(
        failed_outer_fd, failed_stage_fd, original_authority_fd
    )
    if projection != receipt["failed_external_projection"] or projection["projection_sha256"] != receipt["failed_external_projection_sha256"] or projection["failed_stage"]["file_references"] != list(FAILED_REFERENCES):
        _fail("failed stage differs from salvage source audit")
    preissue, _reference_value = _load_preissue(project_fd)
    expected_prestate = dict(preissue["run003_prestate"])
    observed = prior_authority.project_failed_prestate(run_fd)
    for key in ("canonical_tree_sha256", "regular_file_count", "regular_file_bytes", "development_manifest_absent", "complete_absent"):
        if observed.get(key) != expected_prestate.get(key):
            _fail("RUN003 differs from salvage source audit")
    if observed.get("reports_members") != []:
        _fail("RUN003 reports are not empty before salvage publication")
    return receipt, projection


def _source_audit_from_fd(fd: int) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    raw = _read_regular_fd(fd, name="salvage source audit")
    receipt = _strict_json(raw, name="salvage source audit")
    return receipt, raw, _reference(SOURCE_AUDIT_PATH, raw)


def _read_salvage_terminal(
    salvage_authority_fd: int,
) -> tuple[dict[str, Any], bytes]:
    raw, info = _read_path_with_stat(
        salvage_authority_fd, TERMINAL_PATH, name="salvage terminal"
    )
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_uid != os.geteuid()
        or info.st_gid != os.getegid()
        or info.st_nlink != 1
        or info.st_size != len(raw)
    ):
        _fail("salvage terminal file identity changed")
    terminal = _strict_json(raw, name="salvage terminal")
    if raw != _canonical_bytes(terminal):
        _fail("salvage terminal is not canonical newline bytes")
    return terminal, raw


def _write_standard_seal(
    corrected_stage_fd: int,
    *,
    prior_source_audit: Mapping[str, Any],
    prior_source_audit_reference: Mapping[str, Any],
    prior_erratum_reference: Mapping[str, Any],
    diagnostic_reference: Mapping[str, Any],
    signed_status: str,
) -> dict[str, Any]:
    references = [_reference(path, _read(corrected_stage_fd, path, name=f"corrected {path}")) for path in OUTPUT_PATHS]
    seal = prior._payload(
        {
            "contract": prior.STAGE_SEAL_CONTRACT,
            "schema_version": 1,
            "run_id": RUN_ID,
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "erratum_reference": dict(prior_erratum_reference),
            "diagnostic_receipt_reference": dict(diagnostic_reference),
            "recovery_source_audit_reference": dict(prior_source_audit_reference),
            "original_science_authority": prior_source_audit["original_science_authority"],
            "failed_prestate": prior._expected_failed_prestate(),
            "staged_file_order": list(OUTPUT_PATHS),
            "staged_file_references": references,
            "staged_file_set_sha256": prior._canonical_hash(references),
            "signed_report_status": signed_status,
            "action_accounting": prior._expected_action_accounting(),
        }
    )
    frozen._write_exclusive(corrected_stage_fd, prior.STAGE_SEAL_PATH, prior._canonical_bytes(seal))
    return seal


def _build_salvage_terminal(
    *,
    failed_external_pre_projection: Mapping[str, Any],
    failed_external_post_projection: Mapping[str, Any],
    corrected_stage_fd: int,
    salvage_authority_binding: Mapping[str, Any],
    source_audit_reference: Mapping[str, Any],
    prior_source_audit_reference: Mapping[str, Any],
    preissue_reference: Mapping[str, Any],
    supplement_reference: Mapping[str, Any],
    erratum2_reference: Mapping[str, Any],
) -> dict[str, Any]:
    corrected_projection = _stage_projection(
        corrected_stage_fd, require_failed_path=False, sealed=True
    )
    corrected_refs = corrected_projection["file_references"]
    seal_ref = _reference(prior.STAGE_SEAL_PATH, _read(corrected_stage_fd, prior.STAGE_SEAL_PATH, name="corrected stage seal"))
    return _payload(
        {
            "contract": TERMINAL_CONTRACT,
            "schema_version": 1,
            "run_id": RUN_ID,
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "preissue_reference": dict(preissue_reference),
            "accounting_supplement_reference": dict(supplement_reference),
            "erratum2_reference": dict(erratum2_reference),
            "prior_recovery_authority": {
                "implementation_git_commit": PRIOR_IMPLEMENTATION_COMMIT,
                "source_audit_evidence_commit": PRIOR_EVIDENCE_COMMIT,
                "source_audit_reference": dict(prior_source_audit_reference),
            },
            "salvage_source_audit_reference": dict(source_audit_reference),
            "salvage_authority_binding": dict(salvage_authority_binding),
            "failed_stage": {
                "root": FAILED_STAGE_ROOT,
                "ordered_file_references": list(FAILED_REFERENCES),
                "pre_external_projection": dict(failed_external_pre_projection),
                "post_external_projection": dict(failed_external_post_projection),
                "pre_and_post_projections_equal": True,
                "stage_seal_absent": True,
            },
            "corrected_stage": {
                "ordered_file_references": corrected_refs,
                "stage_seal_reference": seal_ref,
                "only_changed_path": SUMMARY_PATH,
                "six_files_byte_identical": True,
                "sealed_projection": corrected_projection,
            },
            "observed_action_accounting": dict(OBSERVED_ACTION_ACCOUNTING),
            "authorized_remaining_action_accounting": dict(
                AUTHORIZED_REMAINING_ACTION_ACCOUNTING
            ),
            "total_authorized_action_accounting": dict(EXACT_ACTION_ACCOUNTING),
            "semantic_assertions": {
                "failed_stage_unchanged": True,
                "six_files_raw_byte_identical": True,
                "exact_summary_self_reference_deleted": True,
                "full_seven_file_semantic_validation_passed": True,
                "standard_stage_seal_validated": True,
                "scientific_actions_performed": 0,
            },
            "original_compound_terminal_required": True,
            "standard_and_salvage_terminals_jointly_required": True,
        }
    )


def validate_salvage_terminal(
    value: Any,
    *,
    failed_stage_fd: int,
    corrected_stage_fd: int,
    source_audit_reference: Mapping[str, Any],
    prior_source_audit_reference: Mapping[str, Any],
    preissue_reference: Mapping[str, Any],
    supplement_reference: Mapping[str, Any],
    erratum2_reference: Mapping[str, Any],
    expected_external_projection_sha256: str,
    salvage_authority_fd: int,
) -> dict[str, Any]:
    terminal = _payload_valid(value, name="salvage terminal")
    if set(terminal) != {
        "contract", "schema_version", "run_id", "created_at_utc", "preissue_reference", "accounting_supplement_reference",
        "erratum2_reference", "prior_recovery_authority", "salvage_source_audit_reference",
        "salvage_authority_binding",
        "failed_stage", "corrected_stage", "observed_action_accounting",
        "authorized_remaining_action_accounting", "total_authorized_action_accounting",
        "semantic_assertions",
        "original_compound_terminal_required", "standard_and_salvage_terminals_jointly_required",
        "payload_sha256",
    } or terminal["contract"] != TERMINAL_CONTRACT or type(terminal["schema_version"]) is not int or terminal["schema_version"] != 1 or terminal["run_id"] != RUN_ID or not isinstance(terminal["created_at_utc"], str) or re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", terminal["created_at_utc"]) is None:
        _fail("salvage terminal identity changed")
    failed_projection = _stage_projection(failed_stage_fd, require_failed_path=True)
    corrected_projection = _stage_projection(corrected_stage_fd, require_failed_path=False, sealed=True)
    authority_info = _owned_private_directory(
        salvage_authority_fd, name="salvage authority"
    )
    if terminal["preissue_reference"] != dict(preissue_reference) or terminal["accounting_supplement_reference"] != dict(supplement_reference) or terminal["erratum2_reference"] != dict(erratum2_reference) or terminal["salvage_source_audit_reference"] != dict(source_audit_reference) or terminal["prior_recovery_authority"] != {"implementation_git_commit": PRIOR_IMPLEMENTATION_COMMIT, "source_audit_evidence_commit": PRIOR_EVIDENCE_COMMIT, "source_audit_reference": dict(prior_source_audit_reference)}:
        _fail("salvage terminal source authority changed")
    failed_boundary = terminal["failed_stage"]
    if not isinstance(failed_boundary, Mapping) or set(failed_boundary) != {"root", "ordered_file_references", "pre_external_projection", "post_external_projection", "pre_and_post_projections_equal", "stage_seal_absent"}:
        _fail("salvage terminal failed-stage authority changed")
    pre_projection = _validate_external_projection_value(
        failed_boundary["pre_external_projection"]
    )
    post_projection = _validate_external_projection_value(
        failed_boundary["post_external_projection"]
    )
    if failed_boundary.get("root") != FAILED_STAGE_ROOT or failed_boundary.get("ordered_file_references") != list(FAILED_REFERENCES) or failed_boundary.get("pre_and_post_projections_equal") is not True or failed_boundary.get("stage_seal_absent") is not True or pre_projection != post_projection or pre_projection["projection_sha256"] != expected_external_projection_sha256 or pre_projection["failed_stage"]["projection_sha256"] != failed_projection["projection_sha256"]:
        _fail("salvage terminal failed-stage authority changed")
    corrected_refs = corrected_projection["file_references"]
    _validate_reference_set(corrected_refs, corrected=True)
    seal_ref = _reference(prior.STAGE_SEAL_PATH, _read(corrected_stage_fd, prior.STAGE_SEAL_PATH, name="corrected stage seal"))
    if terminal["salvage_authority_binding"] != _directory_binding(authority_info) or terminal["corrected_stage"] != {"ordered_file_references": corrected_refs, "stage_seal_reference": seal_ref, "only_changed_path": SUMMARY_PATH, "six_files_byte_identical": True, "sealed_projection": corrected_projection} or terminal["observed_action_accounting"] != OBSERVED_ACTION_ACCOUNTING or terminal["authorized_remaining_action_accounting"] != AUTHORIZED_REMAINING_ACTION_ACCOUNTING or terminal["total_authorized_action_accounting"] != EXACT_ACTION_ACCOUNTING or any(terminal["observed_action_accounting"][key] + terminal["authorized_remaining_action_accounting"][key] != EXACT_ACTION_ACCOUNTING[key] for key in EXACT_ACTION_ACCOUNTING) or terminal["semantic_assertions"] != {"failed_stage_unchanged": True, "six_files_raw_byte_identical": True, "exact_summary_self_reference_deleted": True, "full_seven_file_semantic_validation_passed": True, "standard_stage_seal_validated": True, "scientific_actions_performed": 0} or terminal["original_compound_terminal_required"] is not True or terminal["standard_and_salvage_terminals_jointly_required"] is not True:
        _fail("salvage terminal corrected-stage/accounting authority changed")
    return terminal


def salvage_failed_stage(
    workspace_fd: int,
    run_fd: int,
    failed_outer_fd: int,
    failed_stage_fd: int,
    corrected_stage_fd: int,
    original_authority_fd: int,
    salvage_authority_fd: int,
    prior_source_audit_fd: int,
    prior_erratum_fd: int,
    diagnostic_fd: int,
    salvage_source_audit_fd: int,
    *,
    original_source_evidence_commit: str,
    salvage_source_evidence_commit: str,
) -> dict[str, Any]:
    _require_distinct_directories(
        {
            "workspace": workspace_fd,
            "RUN003": run_fd,
            "failed outer": failed_outer_fd,
            "failed stage": failed_stage_fd,
            "corrected stage": corrected_stage_fd,
            "original authority": original_authority_fd,
            "salvage authority": salvage_authority_fd,
        }
    )
    project_fd = frozen._open_dir(workspace_fd, PROJECT_DIRECTORY)
    try:
        preissue, preissue_reference = _load_preissue(project_fd)
        erratum2_reference = _load_erratum2(project_fd)
        _supplement, supplement_reference = _load_supplement(project_fd)
        source_audit, source_raw, source_reference = _source_audit_from_fd(salvage_source_audit_fd)
        source_audit, before = validate_salvage_source_audit_live(
            source_audit,
            raw=source_raw,
            repo_fd=workspace_fd,
            project_fd=project_fd,
            run_fd=run_fd,
            failed_outer_fd=failed_outer_fd,
            failed_stage_fd=failed_stage_fd,
            original_authority_fd=original_authority_fd,
            evidence_commit=salvage_source_evidence_commit,
        )
    finally:
        os.close(project_fd)
    prior_audit, prior_audit_ref, prior_erratum_ref, diagnostic_ref = prior._load_external_authorities(workspace_fd, run_fd, prior_source_audit_fd, prior_erratum_fd, diagnostic_fd, recovery_source_evidence_commit=PRIOR_EVIDENCE_COMMIT)
    if prior_audit_ref != PRIOR_SOURCE_AUDIT_REFERENCE or frozen._names(original_authority_fd):
        _fail("prior recovery authority or original authority directory changed")
    prior._authenticate_original_science(workspace_fd, run_fd, original_source_evidence_commit=original_source_evidence_commit)
    if before["failed_stage"]["file_references"] != list(FAILED_REFERENCES):
        _fail("failed-stage bytes changed before salvage")
    stage_names = frozen._names(corrected_stage_fd)
    if not stage_names:
        _require_empty_private_directory(
            corrected_stage_fd, name="corrected salvage stage"
        )
        _require_empty_private_directory(
            salvage_authority_fd, name="salvage authority"
        )
        corrected, _proof = _correct_failed_outputs(failed_stage_fd)
        frozen._mkdir_exclusive(corrected_stage_fd, "reports")
        for path in OUTPUT_PATHS:
            frozen._write_exclusive(corrected_stage_fd, path, corrected[path])
        semantics = prior.validate_staged_terminal_files(corrected, recovery_source_audit_reference=prior_audit_ref, diagnostic_reference=diagnostic_ref)
        _write_standard_seal(corrected_stage_fd, prior_source_audit=prior_audit, prior_source_audit_reference=prior_audit_ref, prior_erratum_reference=prior_erratum_ref, diagnostic_reference=diagnostic_ref, signed_status=semantics["signed_report_status"])
    elif prior.STAGE_SEAL_PATH not in stage_names:
        _fail("ambiguous unsealed corrected stage cannot resume")
    else:
        _owned_private_directory(corrected_stage_fd, name="corrected salvage stage")
    prior.validate_stage_seal(corrected_stage_fd, source_audit=prior_audit, source_audit_reference=prior_audit_ref, erratum_reference=prior_erratum_ref, diagnostic_reference=diagnostic_ref)
    _stage_projection(corrected_stage_fd, require_failed_path=False, sealed=True)
    after = _external_projection(
        failed_outer_fd, failed_stage_fd, original_authority_fd
    )
    if after != before:
        _fail("failed external authority changed during salvage")
    authority_names = frozen._names(salvage_authority_fd)
    if not authority_names:
        authority_info = _require_empty_private_directory(
            salvage_authority_fd, name="salvage authority"
        )
        terminal = _build_salvage_terminal(failed_external_pre_projection=before, failed_external_post_projection=after, corrected_stage_fd=corrected_stage_fd, salvage_authority_binding=_directory_binding(authority_info), source_audit_reference=source_reference, prior_source_audit_reference=prior_audit_ref, preissue_reference=preissue_reference, supplement_reference=supplement_reference, erratum2_reference=erratum2_reference)
        frozen._write_exclusive(salvage_authority_fd, TERMINAL_PATH, _canonical_bytes(terminal))
        terminal, _terminal_raw = _read_salvage_terminal(salvage_authority_fd)
    else:
        _owned_private_directory(salvage_authority_fd, name="salvage authority")
        if authority_names != [TERMINAL_PATH]:
            _fail("salvage authority closed roster changed")
        terminal, _terminal_raw = _read_salvage_terminal(salvage_authority_fd)
    return validate_salvage_terminal(terminal, failed_stage_fd=failed_stage_fd, corrected_stage_fd=corrected_stage_fd, source_audit_reference=source_reference, prior_source_audit_reference=prior_audit_ref, preissue_reference=preissue_reference, supplement_reference=supplement_reference, erratum2_reference=erratum2_reference, expected_external_projection_sha256=source_audit["failed_external_projection_sha256"], salvage_authority_fd=salvage_authority_fd)


def validate_published_salvage(
    workspace_fd: int,
    run_fd: int,
    failed_stage_fd: int,
    corrected_stage_fd: int,
    original_authority_fd: int,
    salvage_authority_fd: int,
    prior_source_audit_fd: int,
    prior_erratum_fd: int,
    diagnostic_fd: int,
    salvage_source_audit_fd: int,
    *,
    salvage_source_evidence_commit: str,
    original_already_validated: bool,
) -> dict[str, Any]:
    _require_distinct_directories(
        {
            "workspace": workspace_fd,
            "RUN003": run_fd,
            "failed stage": failed_stage_fd,
            "corrected stage": corrected_stage_fd,
            "original authority": original_authority_fd,
            "salvage authority": salvage_authority_fd,
        }
    )
    if original_already_validated is not True:
        _fail("salvage validation requires one prior original full validation")
    project_fd = frozen._open_dir(workspace_fd, PROJECT_DIRECTORY)
    try:
        preissue, preissue_reference = _load_preissue(project_fd)
        erratum2_reference = _load_erratum2(project_fd)
        _supplement, supplement_reference = _load_supplement(project_fd)
        source_audit, source_raw, source_reference = _source_audit_from_fd(salvage_source_audit_fd)
        # The source audit was issued against the failed prestate.  Publication is
        # authenticated below by the original recovery validator; source/Git and
        # failed-stage validation remain independently exact here.
        receipt = validate_salvage_source_audit_receipt(source_audit)
        if source_raw != _canonical_bytes(receipt) or receipt["source_closure"] != _source_file_projection(workspace_fd, project_fd, receipt["repository"]["implementation_git_commit"]):
            _fail("published salvage source authority changed")
        implementation = receipt["repository"]["implementation_git_commit"]
        if salvage_source_evidence_commit == implementation:
            _fail("salvage evidence is not later than implementation")
        prior_authority._is_ancestor(workspace_fd, implementation, salvage_source_evidence_commit)
        _mode, _oid, committed = prior_authority._git_blob(workspace_fd, salvage_source_evidence_commit, SOURCE_AUDIT_PATH)
        if committed != source_raw:
            _fail("published salvage evidence blob changed")
    finally:
        os.close(project_fd)
    prior_audit, prior_audit_ref, prior_erratum_ref, diagnostic_ref = prior._load_external_authorities(workspace_fd, run_fd, prior_source_audit_fd, prior_erratum_fd, diagnostic_fd, recovery_source_evidence_commit=PRIOR_EVIDENCE_COMMIT)
    _owned_private_directory(salvage_authority_fd, name="salvage authority")
    if frozen._names(salvage_authority_fd) != [TERMINAL_PATH]:
        _fail("salvage authority closed roster changed")
    terminal, terminal_raw = _read_salvage_terminal(salvage_authority_fd)
    validate_salvage_terminal(terminal, failed_stage_fd=failed_stage_fd, corrected_stage_fd=corrected_stage_fd, source_audit_reference=source_reference, prior_source_audit_reference=prior_audit_ref, preissue_reference=preissue_reference, supplement_reference=supplement_reference, erratum2_reference=erratum2_reference, expected_external_projection_sha256=receipt["failed_external_projection_sha256"], salvage_authority_fd=salvage_authority_fd)
    compound_raw = _read(original_authority_fd, prior.TERMINAL_PATH, name="original compound terminal")
    seal_raw = _read(corrected_stage_fd, prior.STAGE_SEAL_PATH, name="corrected stage seal")
    complete_raw = _read(run_fd, "COMPLETE.json", name="RUN003 complete marker")
    development_raw = _read(run_fd, "development_manifest.json", name="RUN003 development manifest")
    return {
        "contract": VALIDATION_CONTRACT,
        "schema_version": 1,
        "run_id": RUN_ID,
        "original_recovery_source_audit_reference": prior_audit_ref,
        "salvage_source_audit_reference": source_reference,
        "v2_salvage_erratum2_reference": erratum2_reference,
        "v2_salvage_preissue_reference": preissue_reference,
        "v2_salvage_accounting_supplement_reference": supplement_reference,
        "corrected_stage_seal_reference": _reference(prior.STAGE_SEAL_PATH, seal_raw),
        "standard_compound_terminal_reference": _reference(prior.TERMINAL_PATH, compound_raw),
        "salvage_terminal_reference": _reference(TERMINAL_PATH, terminal_raw),
        "run003_complete_reference": _reference("COMPLETE.json", complete_raw),
        "run003_development_manifest_reference": _reference("development_manifest.json", development_raw),
        "semantic_assertions": {
            "original_recovery_full_validate_passed": True,
            "salvage_semantic_validate_passed": True,
            "both_terminal_raw_references_bound": True,
        },
        "performance_values_disclosed": False,
    }


def _run_original_validation_main_once(
    workspace_fd: int,
    run_fd: int,
    corrected_stage_fd: int,
    original_authority_fd: int,
    prior_source_audit_fd: int,
    prior_erratum_fd: int,
    diagnostic_fd: int,
    *,
    original_source_evidence_commit: str,
) -> None:
    result = prior.main(
        [
            "validate",
            "--workspace-root",
            f"/proc/self/fd/{workspace_fd}",
            "--run-dir",
            f"/proc/self/fd/{run_fd}",
            "--stage-dir",
            f"/proc/self/fd/{corrected_stage_fd}",
            "--authority-dir",
            f"/proc/self/fd/{original_authority_fd}",
            "--source-audit-fd",
            f"/proc/self/fd/{prior_source_audit_fd}",
            "--erratum-fd",
            f"/proc/self/fd/{prior_erratum_fd}",
            "--diagnostic-fd",
            f"/proc/self/fd/{diagnostic_fd}",
            "--original-source-evidence-commit",
            original_source_evidence_commit,
            "--recovery-source-evidence-commit",
            PRIOR_EVIDENCE_COMMIT,
        ]
    )
    if result != 0:
        _fail("original recovery full validate did not return zero")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    issue = sub.add_parser("issue-source-audit")
    for name in (
        "repository-fd", "project-fd", "run-fd", "failed-outer-fd",
        "failed-stage-fd", "original-authority-fd", "output-fd",
    ):
        issue.add_argument(f"--{name}", type=int, required=True)
    issue.add_argument("--implementation-git-commit", required=True)
    issue.add_argument("--created-at-utc", default=None)
    validate_source = sub.add_parser("validate-source-audit")
    for name in (
        "repository-fd", "project-fd", "run-fd", "failed-outer-fd",
        "failed-stage-fd", "original-authority-fd", "source-audit-fd",
    ):
        validate_source.add_argument(f"--{name}", type=int, required=True)
    validate_source.add_argument("--evidence-git-commit", required=True)
    for command_name in ("salvage", "validate"):
        command = sub.add_parser(command_name)
        for name in ("workspace-root", "run-dir", "failed-outer-dir", "failed-stage-dir", "corrected-stage-dir", "original-authority-dir", "salvage-authority-dir", "prior-source-audit-fd", "prior-erratum-fd", "diagnostic-fd", "salvage-source-audit-fd"):
            command.add_argument(f"--{name}", required=True)
        command.add_argument("--original-source-evidence-commit", required=True)
        command.add_argument("--salvage-source-evidence-commit", required=True)
    return parser


def _write_output(fd: int, raw: bytes) -> None:
    if type(fd) is not int or fd < 3:
        _fail("source-audit output is not an inherited non-stdio descriptor")
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.geteuid() or info.st_gid != os.getegid() or info.st_nlink != 1 or info.st_size != 0 or os.lseek(fd, 0, os.SEEK_CUR) != 0:
        _fail("source-audit output is not an empty O_EXCL regular file")
    offset = 0
    while offset < len(raw):
        written = os.write(fd, raw[offset:])
        if written <= 0:
            _fail("source-audit output short write")
        offset += written
    os.fsync(fd)
    after = os.fstat(fd)
    if after.st_size != len(raw) or (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino):
        _fail("source-audit output identity changed while writing")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "issue-source-audit":
        receipt = build_salvage_source_audit(args.repository_fd, args.project_fd, args.run_fd, args.failed_outer_fd, args.failed_stage_fd, args.original_authority_fd, implementation_git_commit=args.implementation_git_commit, created_at_utc=args.created_at_utc or _now())
        _write_output(args.output_fd, _canonical_bytes(receipt))
        return 0
    if args.command == "validate-source-audit":
        raw = _read_regular_fd(args.source_audit_fd, name="salvage source audit")
        validate_salvage_source_audit_live(_strict_json(raw, name="salvage source audit"), raw=raw, repo_fd=args.repository_fd, project_fd=args.project_fd, run_fd=args.run_fd, failed_outer_fd=args.failed_outer_fd, failed_stage_fd=args.failed_stage_fd, original_authority_fd=args.original_authority_fd, evidence_commit=args.evidence_git_commit)
        return 0
    descriptors: list[int] = []
    try:
        for attribute in ("workspace_root", "run_dir", "failed_outer_dir", "failed_stage_dir", "corrected_stage_dir", "original_authority_dir", "salvage_authority_dir"):
            descriptor = frozen._fd_from_argument(getattr(args, attribute), name=attribute.replace("_", " "))
            descriptors.append(descriptor)
        _require_distinct_directories(
            dict(
                zip(
                    (
                        "workspace", "RUN003", "failed outer", "failed stage",
                        "corrected stage", "original authority", "salvage authority",
                    ),
                    descriptors,
                    strict=True,
                )
            )
        )
        regulars: list[int] = []
        for attribute in ("prior_source_audit_fd", "prior_erratum_fd", "diagnostic_fd", "salvage_source_audit_fd"):
            descriptor = prior._regular_fd_argument(getattr(args, attribute), name=attribute.replace("_", " "))
            descriptors.append(descriptor)
            regulars.append(descriptor)
        workspace_fd, run_fd, failed_outer_fd, failed_stage_fd, corrected_stage_fd, original_authority_fd, salvage_authority_fd = descriptors[:7]
        prior_source_audit_fd, prior_erratum_fd, diagnostic_fd, salvage_source_audit_fd = regulars
        if args.command == "salvage":
            salvage_failed_stage(workspace_fd, run_fd, failed_outer_fd, failed_stage_fd, corrected_stage_fd, original_authority_fd, salvage_authority_fd, prior_source_audit_fd, prior_erratum_fd, diagnostic_fd, salvage_source_audit_fd, original_source_evidence_commit=args.original_source_evidence_commit, salvage_source_evidence_commit=args.salvage_source_evidence_commit)
        else:
            _run_original_validation_main_once(
                workspace_fd,
                run_fd,
                corrected_stage_fd,
                original_authority_fd,
                prior_source_audit_fd,
                prior_erratum_fd,
                diagnostic_fd,
                original_source_evidence_commit=args.original_source_evidence_commit,
            )
            validate_published_salvage(workspace_fd, run_fd, failed_stage_fd, corrected_stage_fd, original_authority_fd, salvage_authority_fd, prior_source_audit_fd, prior_erratum_fd, diagnostic_fd, salvage_source_audit_fd, salvage_source_evidence_commit=args.salvage_source_evidence_commit, original_already_validated=True)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OpenAxesNativeV2RecoverySalvageError as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        raise SystemExit(2) from None
