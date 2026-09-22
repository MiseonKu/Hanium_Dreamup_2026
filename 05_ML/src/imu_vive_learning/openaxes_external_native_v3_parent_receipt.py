"""Validate and copy the outcome-blind native-V2 salvage parent gate for V3.

Only authority records, raw references, and empty child streams are decoded.
No RUN003 report, prediction, metric, or scientific payload is opened here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence
import uuid

from .openaxes_external_native_v3_config import (
    OPENAXES_NATIVE_V3_SALVAGE_INPUT_REFERENCE_ORDER,
    OPENAXES_NATIVE_V3_SALVAGE_RECORD_KEYS,
    OPENAXES_NATIVE_V3_SALVAGE_RECORD_PATHS,
    OPENAXES_NATIVE_V3_SALVAGE_SEMANTIC_ASSERTIONS,
)


RUN_ID = "openaxes-native-v2-sealed-003"
RECORD_CONTRACT = (
    "openaxes.external.native_sensor_experts.v3."
    "parent_salvage_validation_record.v1"
)
SET_CONTRACT = (
    "openaxes.external.native_sensor_experts.v3."
    "parent_salvage_validation_receipt_set.v1"
)
RECORD_DIRECTORY = "parents/native_v2_compound_salvage_validations"
SET_RECEIPT_PATH = f"{RECORD_DIRECTORY}/receipt_set.json"
ORIGINAL_RECOVERY_IMPLEMENTATION_COMMIT = (
    "c77e8bb30a5cb1040194b3b0ade35cbd78aa3977"
)
ORIGINAL_RECOVERY_SOURCE_EVIDENCE_COMMIT = (
    "9cf2c36a47eb203ea07840b0410eec96c639fb43"
)
SALVAGE_IMPLEMENTATION_COMMIT = (
    "fa7a05e0d977032e9a32312a0cb022563665ae91"
)
SALVAGE_SOURCE_EVIDENCE_COMMIT = (
    "3596e04fe619ab43ff7cf6e4e0e6985d13d30f6f"
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
DEFAULT_RECORDS_ROOT = (
    "/home/shin-server/.codex-supervisors/capstone-code/"
    "openaxes-native-v2-run003-compound-salvage-001/validation_records"
)
DEFAULT_V3_RUN_ROOT = (
    "/home/shin-server/capstone-code/09_SO101_IMU_VIVE_LEARNING/artifacts/"
    "openaxes_external_native_v3/run_openaxes-native-v3-001"
)

STATIC_REFERENCES = {
    "original_recovery_source_audit_reference": {
        "path": (
            "09_SO101_IMU_VIVE_LEARNING/receipts/"
            "openaxes_external_native_v2_compound_recovery_source_audit.json"
        ),
        "bytes": 15299,
        "sha256": "887ba564fee0f701df2fa69e233600a0f7266df2cbb4ffde166bfb628ae7df4c",
    },
    "salvage_source_audit_reference": {
        "path": (
            "09_SO101_IMU_VIVE_LEARNING/receipts/"
            "openaxes_external_native_v2_compound_recovery_salvage_source_audit.json"
        ),
        "bytes": 12656,
        "sha256": "875d2068541b749a80eb1366bc28c68ba22504eab20efcb399ef3450832e0b3b",
    },
    "v2_salvage_erratum2_reference": {
        "path": (
            "09_SO101_IMU_VIVE_LEARNING/docs/"
            "OPENAXES_EXTERNAL_NATIVE_V2_RUN003_RECOVERY_ERRATUM2.md"
        ),
        "bytes": 4390,
        "sha256": "d053b8bed1aa7e427d06688b4978ad9bf6c48e105dd6b6da29a16bda682ae906",
    },
    "v2_salvage_preissue_reference": {
        "path": (
            "09_SO101_IMU_VIVE_LEARNING/receipts/"
            "openaxes_external_native_v2_compound_recovery_salvage_preissue.json"
        ),
        "bytes": 5149,
        "sha256": "0b15daf0d47878d57a798a34bdc5df09092b4523dc1bd3070cf0653fe7a4f1b5",
    },
    "v2_salvage_accounting_supplement_reference": {
        "path": (
            "09_SO101_IMU_VIVE_LEARNING/receipts/"
            "openaxes_external_native_v2_compound_recovery_"
            "salvage_accounting_supplement.json"
        ),
        "bytes": 2995,
        "sha256": "4df21c681f15cbcfdad43a17304e4f5e23da9478aba77ea804b525d25a7122be",
    },
}

SET_KEYS = {
    "contract", "schema_version", "run_id", "record_count",
    "salvage_validation_record_references",
    "salvage_validation_receipt_set_sha256",
    "original_recovery_source_audit_reference",
    "original_recovery_source_evidence_git_commit",
    "salvage_source_audit_reference", "salvage_source_evidence_git_commit",
    "v2_salvage_erratum2_reference", "v2_salvage_preissue_reference",
    "v2_salvage_accounting_supplement_reference",
    "corrected_stage_seal_reference", "standard_compound_terminal_reference",
    "salvage_terminal_reference", "run003_complete_reference",
    "run003_development_manifest_reference", "scientific_values_disclosed",
    "payload_sha256",
}


class OpenAxesExternalNativeV3ParentReceiptError(RuntimeError):
    """Raised when the composite salvage parent authority is not exact."""


@dataclass(frozen=True)
class SalvageValidationBundle:
    records: tuple[Mapping[str, Any], ...]
    record_raw_bytes: tuple[bytes, ...]
    record_references: tuple[Mapping[str, Any], ...]
    receipt_set: Mapping[str, Any]
    receipt_set_raw: bytes
    common_references: Mapping[str, Mapping[str, Any]]
    receipt_set_sha256: str

    def parent_fields(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "salvage_validation_record_references": [dict(row) for row in self.record_references],
            "salvage_validation_receipt_set_sha256": self.receipt_set_sha256,
            "original_recovery_source_evidence_git_commit": ORIGINAL_RECOVERY_SOURCE_EVIDENCE_COMMIT,
            "salvage_source_evidence_git_commit": SALVAGE_SOURCE_EVIDENCE_COMMIT,
        }
        output.update({key: dict(value) for key, value in self.common_references.items()})
        return output


def _fail(message: str) -> None:
    raise OpenAxesExternalNativeV3ParentReceiptError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OpenAxesExternalNativeV3ParentReceiptError(
            "parent authority is not finite canonical JSON"
        ) from exc


def _canonical_file(value: Any) -> bytes:
    return _canonical(value) + b"\n"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strict_json(raw: bytes, *, name: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in rows:
            if type(key) is not str or key in output:
                _fail(f"{name} has a duplicate or non-string key")
            output[key] = value
        return output

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OpenAxesExternalNativeV3ParentReceiptError(
            f"cannot parse {name}"
        ) from exc
    if not isinstance(value, Mapping):
        _fail(f"{name} is not a mapping")
    result = dict(value)
    if raw != _canonical_file(result):
        _fail(f"{name} is not canonical newline JSON")
    return result


def _validate_payload(value: Mapping[str, Any], *, name: str) -> None:
    payload = dict(value)
    observed = payload.pop("payload_sha256", None)
    if not isinstance(observed, str) or observed != _canonical_hash(payload):
        _fail(f"{name} payload SHA-256 changed")


def _raw_reference(path: str, raw: bytes) -> dict[str, Any]:
    return {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _validate_reference(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        _fail(f"{name} is not an exact raw reference")
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        type(path) is not str or not path or path.startswith("/")
        or "\\" in path or any(part in {"", ".", ".."} for part in path.split("/"))
        or type(size) is not int or size < 0
        or type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        _fail(f"{name} raw reference is malformed")
    return dict(value)


def _read_regular_at(root_fd: int, relative: str, *, name: str) -> bytes:
    parts = relative.split("/")
    if not relative or any(part in {"", ".", ".."} for part in parts):
        _fail(f"unsafe {name} path")
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            os.close(current)
            current = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                _fail(f"{name} is not a singly linked regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                _fail(f"{name} changed while read")
            raw = b"".join(chunks)
            if len(raw) != before.st_size:
                _fail(f"{name} read was incomplete")
            return raw
        finally:
            os.close(fd)
    except OSError as exc:
        raise OpenAxesExternalNativeV3ParentReceiptError(f"cannot read {name}") from exc
    finally:
        os.close(current)


def _open_root(path: str | Path) -> int:
    source = Path(path)
    try:
        info = source.stat(follow_symlinks=False)
        if source.is_symlink() or not stat.S_ISDIR(info.st_mode):
            _fail("validation root is not a nonsymlink directory")
        return os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise OpenAxesExternalNativeV3ParentReceiptError(
            "cannot open validation root"
        ) from exc


def _log_path(ordinal: int, stream: str) -> str:
    return f"{RECORD_DIRECTORY}/validation_{ordinal:02d}.{stream}.log"


def _validate_uuid(value: Any, *, ordinal: int) -> str:
    if type(value) is not str:
        _fail(f"validation {ordinal} UUID is not a string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise OpenAxesExternalNativeV3ParentReceiptError(
            f"validation {ordinal} UUID is malformed"
        ) from exc
    if parsed.version != 4 or str(parsed) != value or parsed.variant != uuid.RFC_4122:
        _fail(f"validation {ordinal} UUID is not canonical UUIDv4")
    return value


def _validate_record(
    root_fd: int, raw: bytes, *, ordinal: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    record = _strict_json(raw, name=f"salvage validation record {ordinal}")
    if set(record) != set(OPENAXES_NATIVE_V3_SALVAGE_RECORD_KEYS):
        _fail(f"validation {ordinal} exact keys changed")
    _validate_payload(record, name=f"validation {ordinal}")
    if (
        record["contract"] != RECORD_CONTRACT
        or record["schema_version"] != 1
        or type(record["schema_version"]) is not int
        or record["run_id"] != RUN_ID
        or record["validation_ordinal"] != ordinal
        or type(record["validation_ordinal"]) is not int
    ):
        _fail(f"validation {ordinal} identity changed")
    _validate_uuid(record["invocation_uuid_v4"], ordinal=ordinal)
    process = record["process_identity"]
    if (
        not isinstance(process, Mapping) or set(process) != {"pid", "proc_start_ticks"}
        or type(process["pid"]) is not int or process["pid"] <= 0
        or type(process["proc_start_ticks"]) is not int or process["proc_start_ticks"] <= 0
    ):
        _fail(f"validation {ordinal} process identity changed")
    commits = {
        "original_recovery_implementation_git_commit": ORIGINAL_RECOVERY_IMPLEMENTATION_COMMIT,
        "original_recovery_source_evidence_git_commit": ORIGINAL_RECOVERY_SOURCE_EVIDENCE_COMMIT,
        "salvage_implementation_git_commit": SALVAGE_IMPLEMENTATION_COMMIT,
        "salvage_source_evidence_git_commit": SALVAGE_SOURCE_EVIDENCE_COMMIT,
    }
    if any(record[key] != expected for key, expected in commits.items()):
        _fail(f"validation {ordinal} source lineage changed")
    references = {
        key: _validate_reference(record[key], name=f"validation {ordinal} {key}")
        for key in OPENAXES_NATIVE_V3_SALVAGE_INPUT_REFERENCE_ORDER
    }
    if any(references[key] != expected for key, expected in STATIC_REFERENCES.items()):
        _fail(f"validation {ordinal} static authority changed")
    if record["input_reference_set_sha256"] != _canonical_hash(
        [references[key] for key in OPENAXES_NATIVE_V3_SALVAGE_INPUT_REFERENCE_ORDER]
    ):
        _fail(f"validation {ordinal} input-reference set changed")
    if (
        record["original_recovery_full_validate_exit_code"] != 0
        or type(record["original_recovery_full_validate_exit_code"]) is not int
        or record["salvage_validate_exit_code"] != 0
        or type(record["salvage_validate_exit_code"]) is not int
        or record["semantic_assertions"] != OPENAXES_NATIVE_V3_SALVAGE_SEMANTIC_ASSERTIONS
        or record["scientific_values_disclosed"] is not False
    ):
        _fail(f"validation {ordinal} success assertions changed")
    for stream in ("stdout", "stderr"):
        expected_path = _log_path(ordinal, stream)
        reference = _validate_reference(
            record[f"{stream}_reference"], name=f"validation {ordinal} {stream}"
        )
        if reference != {"path": expected_path, "bytes": 0, "sha256": EMPTY_SHA256}:
            _fail(f"validation {ordinal} {stream} reference changed")
        stream_raw = _read_regular_at(root_fd, expected_path, name=f"validation {ordinal} {stream}")
        if stream_raw or _raw_reference(expected_path, stream_raw) != reference:
            _fail(f"validation {ordinal} {stream} is not the exact empty file")
    return record, references


def _validate_set_receipt(
    raw: bytes,
    *,
    record_references: Sequence[Mapping[str, Any]],
    common: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = _strict_json(raw, name="salvage validation receipt set")
    if set(receipt) != SET_KEYS:
        _fail("salvage validation receipt-set keys changed")
    _validate_payload(receipt, name="salvage validation receipt set")
    expected_hash = _canonical_hash(list(record_references))
    if (
        receipt["contract"] != SET_CONTRACT
        or receipt["schema_version"] != 1
        or type(receipt["schema_version"]) is not int
        or receipt["run_id"] != RUN_ID
        or receipt["record_count"] != 10
        or type(receipt["record_count"]) is not int
        or receipt["salvage_validation_record_references"] != list(record_references)
        or receipt["salvage_validation_receipt_set_sha256"] != expected_hash
        or receipt["original_recovery_source_evidence_git_commit"] != ORIGINAL_RECOVERY_SOURCE_EVIDENCE_COMMIT
        or receipt["salvage_source_evidence_git_commit"] != SALVAGE_SOURCE_EVIDENCE_COMMIT
        or receipt["scientific_values_disclosed"] is not False
    ):
        _fail("salvage validation receipt-set authority changed")
    if any(receipt[key] != common[key] for key in common):
        _fail("receipt-set common authority differs from validation records")
    return receipt


def validate_salvage_validation_bundle(
    records_root: str | Path,
) -> SalvageValidationBundle:
    """Validate the exact ten controller records without opening science."""

    root_fd = _open_root(records_root)
    try:
        record_directory_fd = os.open(
            RECORD_DIRECTORY,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        try:
            expected_members = {
                "receipt_set.json",
                *(f"validation_{ordinal:02d}.json" for ordinal in range(10)),
                *(f"validation_{ordinal:02d}.stdout.log" for ordinal in range(10)),
                *(f"validation_{ordinal:02d}.stderr.log" for ordinal in range(10)),
            }
            if set(os.listdir(record_directory_fd)) != expected_members:
                _fail("salvage validation directory is not the exact 31-file set")
        finally:
            os.close(record_directory_fd)
        records: list[Mapping[str, Any]] = []
        record_raw: list[bytes] = []
        record_references: list[Mapping[str, Any]] = []
        common: dict[str, Mapping[str, Any]] | None = None
        identities: set[tuple[int, int]] = set()
        uuids: set[str] = set()
        for ordinal, path in enumerate(OPENAXES_NATIVE_V3_SALVAGE_RECORD_PATHS):
            raw = _read_regular_at(root_fd, path, name=f"validation record {ordinal}")
            record, references = _validate_record(root_fd, raw, ordinal=ordinal)
            identity = (
                record["process_identity"]["pid"],
                record["process_identity"]["proc_start_ticks"],
            )
            invocation_uuid = record["invocation_uuid_v4"]
            if identity in identities or invocation_uuid in uuids:
                _fail("fresh process identity or UUID was reused")
            identities.add(identity)
            uuids.add(invocation_uuid)
            if common is None:
                common = references
            elif references != common:
                _fail("common raw references differ across the ten records")
            records.append(record)
            record_raw.append(raw)
            record_references.append(_raw_reference(path, raw))
        if common is None:
            _fail("no salvage validation records")
        set_raw = _read_regular_at(root_fd, SET_RECEIPT_PATH, name="receipt set")
        receipt = _validate_set_receipt(
            set_raw, record_references=record_references, common=common
        )
        return SalvageValidationBundle(
            records=tuple(records),
            record_raw_bytes=tuple(record_raw),
            record_references=tuple(record_references),
            receipt_set=receipt,
            receipt_set_raw=set_raw,
            common_references=common,
            receipt_set_sha256=str(receipt["salvage_validation_receipt_set_sha256"]),
        )
    except OSError as exc:
        raise OpenAxesExternalNativeV3ParentReceiptError(
            "cannot inspect salvage validation bundle"
        ) from exc
    finally:
        os.close(root_fd)


def _mkdir_private_at(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or info.st_gid != os.getegid():
        os.close(fd)
        _fail("destination parent directory ownership changed")
    return fd


def _write_exclusive(directory_fd: int, name: str, raw: bytes) -> None:
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        os.fchmod(fd, 0o600)
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                _fail("parent receipt copy short write")
            offset += written
        os.fsync(fd)
        if os.fstat(fd).st_size != len(raw):
            _fail("parent receipt copy size changed")
    finally:
        os.close(fd)


def copy_salvage_validation_bundle(
    records_root: str | Path,
    destination_root: str | Path,
) -> SalvageValidationBundle:
    """Validate then O_EXCL-copy the self-contained parent evidence bundle.

    The destination must not already contain the salvage-validation directory;
    a partial copy is deliberately non-resumable.
    """

    bundle = validate_salvage_validation_bundle(records_root)
    source_fd = _open_root(records_root)
    destination_fd = _open_root(destination_root)
    parents_fd = -1
    output_fd = -1
    try:
        parents_fd = _mkdir_private_at(destination_fd, "parents")
        try:
            os.mkdir("native_v2_compound_salvage_validations", 0o700, dir_fd=parents_fd)
        except FileExistsError as exc:
            raise OpenAxesExternalNativeV3ParentReceiptError(
                "salvage validation destination already exists; retry is forbidden"
            ) from exc
        output_fd = os.open(
            "native_v2_compound_salvage_validations",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parents_fd,
        )
        for ordinal, raw in enumerate(bundle.record_raw_bytes):
            _write_exclusive(output_fd, f"validation_{ordinal:02d}.json", raw)
            for stream in ("stdout", "stderr"):
                source_path = _log_path(ordinal, stream)
                stream_raw = _read_regular_at(source_fd, source_path, name=source_path)
                _write_exclusive(
                    output_fd, f"validation_{ordinal:02d}.{stream}.log", stream_raw
                )
        _write_exclusive(output_fd, "receipt_set.json", bundle.receipt_set_raw)
        os.fsync(output_fd)
        os.fsync(parents_fd)
        os.fsync(destination_fd)
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        if parents_fd >= 0:
            os.close(parents_fd)
        os.close(destination_fd)
        os.close(source_fd)
    copied = validate_salvage_validation_bundle(destination_root)
    if (
        copied.record_raw_bytes != bundle.record_raw_bytes
        or copied.receipt_set_raw != bundle.receipt_set_raw
    ):
        _fail("copied parent evidence differs from controller authority")
    return copied


def validate_and_copy_openaxes_external_native_v3_parent_gate(
    records_root: str | Path,
    destination_root: str | Path,
) -> dict[str, Any]:
    """Practical controller API returning the exact V3 parent receipt fields."""

    return copy_salvage_validation_bundle(records_root, destination_root).parent_fields()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--records-root", default=DEFAULT_RECORDS_ROOT)
    copy = commands.add_parser("copy")
    copy.add_argument("--records-root", default=DEFAULT_RECORDS_ROOT)
    copy.add_argument("--destination-root", default=DEFAULT_V3_RUN_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        fields = validate_salvage_validation_bundle(args.records_root).parent_fields()
    else:
        fields = validate_and_copy_openaxes_external_native_v3_parent_gate(
            args.records_root, args.destination_root
        )
    # This is authority-only JSON: it contains no predictions or metric values.
    os.write(1, _canonical_file(fields))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OpenAxesExternalNativeV3ParentReceiptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
