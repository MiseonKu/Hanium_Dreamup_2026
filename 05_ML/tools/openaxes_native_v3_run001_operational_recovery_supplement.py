#!/usr/bin/env python3
"""Issue and validate the append-only Run 001 operational-recovery supplement.

This tool records operational provenance only.  It never imports the V3
training, experiment, or analysis modules and never opens scientific values for
publication.  The issue command consumes raw, separately captured systemd and
journal outputs, authenticates the already-published recovery outputs, and
writes exactly one canonical JSON receipt with O_EXCL.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence


CONTRACT = (
    "openaxes.external.native_sensor_experts.v3."
    "run001.operational_recovery_supplement.v1"
)
SCHEMA_VERSION = 1
RUN_ID = "openaxes-native-v3-allmask-dmome-001"

PROJECT = Path("/home/shin-server/capstone-code/09_SO101_IMU_VIVE_LEARNING")
SUPERVISOR = Path("/home/shin-server/.codex-supervisors/capstone-code")
RUN_ROOT = PROJECT / "artifacts/openaxes_external_native_v3/run_openaxes-native-v3-001"
VALIDATION_ROOT = SUPERVISOR / "openaxes-native-v3-run001-mofe-final-validation-001"
RECOVERY_LOCK = SUPERVISOR / "openaxes-native-v3-run001-mofe-recovery-finalization-001"

INITIAL_ANALYSIS = PROJECT / "receipts/openaxes_external_native_v3_run001_deadline_primary8_analysis.json"
INITIAL_DOCUMENT = PROJECT / "receipts/openaxes_external_native_v3_run001_document_receipt.json"
EXPANDED_ANALYSIS = PROJECT / "receipts/openaxes_external_native_v3_run001_mofe_cagrad_primary8_analysis.json"
EXPANDED_DOCUMENT = PROJECT / "receipts/openaxes_external_native_v3_run001_mofe_document_receipt.json"
EVALUATION = RUN_ROOT / "reports/evaluation_partial_2fdc6a253782c3ac.json"
RUNTIME_STATUS = RUN_ROOT / "runtime/status.json"

RECOVERY_FINALIZER = SUPERVISOR / "openaxes_native_v3_run001_mofe_recovery_finalize.sh"
RECOVERY_DOCUMENT_FINALIZER = SUPERVISOR / "openaxes_native_v3_run001_mofe_recovery_document_finalize.sh"
EXPANDED_GENERATOR = SUPERVISOR / "openaxes_native_v3_run001_mofe_document_receipt.py"
VALIDATION_CONTROLLER = SUPERVISOR / "openaxes_native_v3_run001_validate10.sh"
BASE_GENERATOR = SUPERVISOR / "openaxes_native_v3_run001_document_receipt.py"
FOLLOWUP_LAUNCHER = SUPERVISOR / "openaxes_native_v3_run001_mofe_followup.sh"
FOLLOWUP_FINALIZER = SUPERVISOR / "openaxes_native_v3_run001_mofe_finalize.sh"
FOLLOWUP_DOCUMENT_FINALIZER = SUPERVISOR / "openaxes_native_v3_run001_mofe_document_finalize.sh"
EXPERIMENT_SOURCE = PROJECT / "src/imu_vive_learning/openaxes_external_native_v3_experiment.py"
ANALYSIS_SOURCE = PROJECT / "src/imu_vive_learning/openaxes_external_native_v3_analysis.py"

SUPPLEMENT_PATH = PROJECT / "receipts/openaxes_external_native_v3_run001_operational_recovery_supplement.json"

CANDIDATES = (
    "dense_6979__allmask_cardinality_balanced_erm",
    "dmome_6979__two_stage_allmask_cardinality_balanced_mofe",
    "dmome_6979__two_stage_allmask_cardinality_balanced_mofe_cagrad",
)
CANDIDATE_ORDINALS = (0, 4, 5)

FAILED_UNITS = (
    ("openaxes-v3-run001-mofe-followup.service", 41),
    ("openaxes-v3-run001-mofe-finalizer.service", 52),
    ("openaxes-v3-run001-mofe-document-finalizer.service", 60),
    ("openaxes-v3-run001-mofe-recovery-finalizer.service", 127),
    ("openaxes-v3-run001-mofe-recovery-document-finalizer.service", 80),
)
SYSTEMCTL_PROPERTIES = (
    "Id,LoadState,ActiveState,SubState,Result,ExecMainStatus,"
    "ExecMainStartTimestamp,ExecMainExitTimestamp,ExecStart,FragmentPath,UnitFileState"
)
SYSTEMCTL_COMMAND = (
    "systemctl --user show "
    + " ".join(unit for unit, _ in FAILED_UNITS)
    + " --property="
    + SYSTEMCTL_PROPERTIES
    + " --no-pager"
)
ORIGINAL_JOURNAL_COMMAND = (
    "journalctl --user "
    + " ".join(f"-u {unit}" for unit, _ in FAILED_UNITS)
    + " --since '2026-08-26 19:07:00' --until '2026-08-26 19:16:30' "
    "--no-pager -o short-iso-precise"
)
RECOVERY_JOURNAL_COMMAND = (
    "journalctl --user "
    "-u openaxes-v3-run001-mofe-recovery-finalizer-v2.service "
    "-u openaxes-v3-run001-mofe-recovery-document-finalizer-v2.service "
    "--since '2026-08-26 19:17:15' --until '2026-08-26 19:18:05' "
    "--no-pager -o short-iso-precise"
)

FAILED_RECOVERY_SHA256 = "cf168c9100de659b6365f2c1972299b6280e81886d49eddac5de03d820a434bd"
FAILED_RECOVERY_BYTES = 7770
FAILED_DOCUMENT_SHA256 = "c94ff16cc8c5683e150f4e8e779591820b64df6e93e32ff45fc750909cf8a8de"
FAILED_DOCUMENT_BYTES = 2031
FAILED_EXECUTION_STARTED_UTC = "2026-08-26T10:16:02Z"
SUCCESSFUL_RECOVERY_STARTED_UTC = "2026-08-26T10:17:15Z"
SUCCESSFUL_ANALYSIS_FINISHED_UTC = "2026-08-26T10:18:03Z"
SUCCESSFUL_DOCUMENT_FINISHED_UTC = "2026-08-26T10:18:04Z"

EXPECTED_SOURCE_IDENTITIES = {
    "recovery_finalizer": (8051, "3f40038d49dbbc41c5c77c3d926195e72626d9dc2f18049348c01498cfd4ac9a"),
    "recovery_document_finalizer": (2034, "8b14516828bc272aac46df30eba59fce2527271167131eacfe5edafe6ce0c3ed"),
    "expanded_document_generator": (34756, "8f51f426d99a21cfae1e609fc9902a8001432ae851e865ad3c20cfbb80645ebc"),
    "validation_controller": (5110, "832bce63e33ea5dfdae3ddd76877c95ceac0aded55104016aaa19acc45c8b735"),
    "reviewed_base_generator": (39345, "541ddde28f320b91be1c3961e357401ff922188cb90e66bdee930fcb10f92a7c"),
    "experiment_source": (56505, "b2a70ecd421ac2a43641904375808f8723a1a6494e612c91ae8bf52658bbebd7"),
    "analysis_source": (34082, "68eec470b63934fca370872b08ff066d0093d4cd7f21c59856fdcea50941bb30"),
    "followup_launcher": (5805, "f7511299fddf5e2ae727db989b8a000a2199ac4b3b4ce4756b5da09f74dcd34f"),
    "followup_finalizer": (4451, "12f1f8da1164151e65526fbf5d8cfc561b24115bb2b24d226efee42e6119673a"),
    "followup_document_finalizer": (1909, "98abe3142cbfc6bc620cefc583a81f45e0c7df52bd6d95c203c96005da617537"),
}

EXPECTED_VALIDATION_COMPLETE = b"10 validators and 1 evaluator exited 0\n"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
UUID4 = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)


class SupplementError(RuntimeError):
    """The operational authority is not the exact reviewed authority."""


def _fail(message: str) -> None:
    raise SupplementError(message)


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
        raise SupplementError("value is not finite canonical JSON") from exc


def _payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["payload_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _validate_payload(value: Mapping[str, Any], *, name: str) -> None:
    supplied = value.get("payload_sha256")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if not isinstance(supplied, str) or supplied != hashlib.sha256(_canonical(body)).hexdigest():
        _fail(f"{name} payload_sha256 is invalid")


def _iso_utc_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _stat_projection(info: os.stat_result) -> dict[str, Any]:
    if stat.S_ISREG(info.st_mode):
        kind = "regular"
    elif stat.S_ISDIR(info.st_mode):
        kind = "directory"
    else:
        kind = "other"
    return {
        "file_type": kind,
        "mode": format(stat.S_IMODE(info.st_mode), "04o"),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "nlink": info.st_nlink,
        "device": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "mtime_utc": _iso_utc_from_ns(info.st_mtime_ns),
        "ctime_ns": info.st_ctime_ns,
        "ctime_utc": _iso_utc_from_ns(info.st_ctime_ns),
    }


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    fields = (
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _read_regular(path: Path, *, name: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _fail(f"{name} is not a single-link regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if not _same_identity(before, after) or len(raw) != before.st_size:
            _fail(f"{name} changed while being read: {path}")
        return raw, before
    except OSError as exc:
        raise SupplementError(f"cannot read {name}: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _render_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT).as_posix()
    except ValueError:
        return path.as_posix()


def _file_projection(path: Path, *, name: str) -> dict[str, Any]:
    raw, info = _read_regular(path, name=name)
    return {
        "path": _render_path(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stat": _stat_projection(info),
    }


def _load_canonical_json(
    path: Path, *, name: str, require_payload: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, info = _read_regular(path, name=name)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SupplementError(f"{name} is not UTF-8 JSON") from exc
    if type(value) is not dict or raw != _canonical(value) + b"\n":
        _fail(f"{name} is not canonical object JSON with one final newline")
    if require_payload:
        _validate_payload(value, name=name)
    elif "payload_sha256" in value:
        _validate_payload(value, name=name)
    reference = {
        "path": _render_path(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stat": _stat_projection(info),
    }
    if "payload_sha256" in value:
        reference["payload_sha256"] = value["payload_sha256"]
    else:
        reference["payload_sha256_available"] = False
        reference["canonical_object_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value, reference


def _directory_snapshot(path: Path, *, name: str) -> tuple[list[str], dict[str, Any]]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise SupplementError(f"cannot stat {name}: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        _fail(f"{name} is not a no-symlink directory")
    names = sorted(os.listdir(path))
    after = os.lstat(path)
    if not _same_identity(before, after):
        _fail(f"{name} changed during membership snapshot")
    return names, _stat_projection(before)


def _raw_capture(path: Path, *, name: str, command: str) -> dict[str, Any]:
    raw, info = _read_regular(path, name=name)
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise SupplementError(f"{name} is not UTF-8") from exc
    if not text.endswith("\n"):
        _fail(f"{name} must end in one captured newline")
    return {
        "command": command,
        "raw_utf8": text,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "capture_file_stat": _stat_projection(info),
    }


def _parse_systemctl_capture(capture: Mapping[str, Any]) -> list[dict[str, Any]]:
    text = capture["raw_utf8"]
    blocks = [block for block in text.strip().split("\n\n") if block]
    if len(blocks) != len(FAILED_UNITS):
        _fail("systemctl capture is not the exact five-unit roster")
    parsed: list[dict[str, Any]] = []
    for block, (expected_id, expected_status) in zip(blocks, FAILED_UNITS):
        values: dict[str, str] = {}
        for line in block.splitlines():
            if "=" not in line:
                _fail("systemctl capture contains a malformed property")
            key, value = line.split("=", 1)
            if key in values:
                _fail("systemctl capture repeats a property")
            values[key] = value
        required = {
            "Id",
            "LoadState",
            "ActiveState",
            "SubState",
            "Result",
            "ExecMainStatus",
            "ExecMainStartTimestamp",
            "ExecMainExitTimestamp",
            "ExecStart",
            "FragmentPath",
            "UnitFileState",
        }
        if set(values) != required:
            _fail(f"systemctl property schema changed for {expected_id}")
        if (
            values["Id"] != expected_id
            or values["LoadState"] != "loaded"
            or values["ActiveState"] != "failed"
            or values["SubState"] != "failed"
            or values["Result"] != "exit-code"
            or values["ExecMainStatus"] != str(expected_status)
            or not values["ExecMainStartTimestamp"]
            or not values["ExecMainExitTimestamp"]
            or not values["ExecStart"]
        ):
            _fail(f"loaded failure-unit semantics changed for {expected_id}")
        parsed.append(values)
    return parsed


def _validate_journal_captures(
    original: Mapping[str, Any], recovery: Mapping[str, Any]
) -> dict[str, Any]:
    original_text = original["raw_utf8"]
    original_fragments = (
        "error: run contains failed, interrupted, or ambiguous tasks",
        "fail-closed: MoFe controllers exited 0/2",
        "status=41/n/a",
        "status=52/n/a",
        "status=60/n/a",
        "line 36: rg: command not found",
        "status=127/n/a",
        "fail-closed: recovery finalizer ended state=failed result=exit-code status=127",
        "status=80/n/a",
    )
    if any(fragment not in original_text for fragment in original_fragments):
        _fail("original/first-recovery journal capture lacks required evidence")
    recovery_text = recovery["raw_utf8"]
    recovery_fragments = (
        "Started openaxes-v3-run001-mofe-recovery-finalizer-v2.service",
        "recovery authority: 72 results, 72 unique single attempts, zero errors",
        "starting recovery exact ten-pass validation",
        "validation complete:",
        "starting recovery canonical three-candidate analysis",
        "recovery validation and analysis complete",
        "Started openaxes-v3-run001-mofe-recovery-document-finalizer-v2.service",
        "recovery document finalizer complete",
    )
    if any(fragment not in recovery_text for fragment in recovery_fragments):
        _fail("successful-recovery journal capture lacks required evidence")
    if "internal-fit" in recovery_text or "launching MoFe-only" in recovery_text:
        _fail("successful-recovery journal unexpectedly contains a fit action")
    main_pid_matches = set(
        re.findall(
            r"openaxes_native_v3_run001_mofe_recovery_finalize\.sh\[(\d+)\]: "
            r"2026-08-26T19:(?:17|18):",
            recovery_text,
        )
    )
    document_pid_matches = set(
        re.findall(
            r"openaxes_native_v3_run001_mofe_recovery_document_finalize\.sh\[(\d+)\]: "
            r"2026-08-26T19:(?:17|18):",
            recovery_text,
        )
    )
    if main_pid_matches != {"969050"} or document_pid_matches != {"969116"}:
        _fail("successful-recovery main process identities changed")
    exact_timeline = {
        "recovery_service_started": "2026-08-26T19:17:15.018609+09:00",
        "recovery_main_started": "2026-08-26T19:17:15+09:00",
        "validation_bundle_completed": "2026-08-26T19:17:17+09:00",
        "analysis_completed": "2026-08-26T19:18:03+09:00",
        "document_service_started": "2026-08-26T19:17:15.584641+09:00",
        "document_main_started": "2026-08-26T19:17:15+09:00",
        "document_completed": "2026-08-26T19:18:04+09:00",
    }
    for value in exact_timeline.values():
        if value not in recovery_text:
            _fail("successful-recovery exact timeline changed")
    return {
        "original_failure_fragments_all_present": True,
        "successful_recovery_fragments_all_present": True,
        "successful_recovery_journal_contains_internal_fit": False,
        "successful_recovery_journal_contains_training_launch": False,
        "successful_recovery_main_pid": 969050,
        "successful_document_main_pid": 969116,
        "successful_recovery_started_utc": SUCCESSFUL_RECOVERY_STARTED_UTC,
        "successful_analysis_finished_utc": SUCCESSFUL_ANALYSIS_FINISHED_UTC,
        "successful_document_finished_utc": SUCCESSFUL_DOCUMENT_FINISHED_UTC,
        "exact_journal_timeline": exact_timeline,
    }


def _reconstruct_failed_sources() -> dict[str, Any]:
    current_recovery, _ = _read_regular(RECOVERY_FINALIZER, name="current recovery finalizer")
    current_text = current_recovery.decode("utf-8")
    failed_text = current_text.replace(
        "rg_bin=/home/shin-server/.codex/packages/standalone/releases/0.149.0-x86_64-unknown-linux-musl/codex-path/rg\n",
        "",
        1,
    )
    failed_text = failed_text.replace(
        "require_failed_unit openaxes-v3-run001-mofe-recovery-finalizer.service 127\n"
        "require_failed_unit openaxes-v3-run001-mofe-recovery-document-finalizer.service 80\n",
        "",
        1,
    )
    failed_text = failed_text.replace('| "$rg_bin" -F', "| rg -F")
    failed_recovery = failed_text.encode("utf-8")
    if len(failed_recovery) != FAILED_RECOVERY_BYTES or hashlib.sha256(failed_recovery).hexdigest() != FAILED_RECOVERY_SHA256:
        _fail("failed recovery-finalizer bytes cannot be reconstructed exactly")

    current_document, _ = _read_regular(
        RECOVERY_DOCUMENT_FINALIZER, name="current recovery document finalizer"
    )
    failed_document = current_document.replace(
        b"upstream=openaxes-v3-run001-mofe-recovery-finalizer-v2.service",
        b"upstream=openaxes-v3-run001-mofe-recovery-finalizer.service",
        1,
    )
    if len(failed_document) != FAILED_DOCUMENT_BYTES or hashlib.sha256(failed_document).hexdigest() != FAILED_DOCUMENT_SHA256:
        _fail("failed recovery-document-finalizer bytes cannot be reconstructed exactly")
    common_stat = {
        "observation_source": "auditor live ls observation made before overwrite",
        "mode": "0664",
        "uid": 1000,
        "gid": 1000,
        "nlink": 1,
        "mtime_local_minute": "2026-08-26T19:15+09:00",
        "latest_possible_mtime_utc": "2026-08-26T10:15:59.999999Z",
        "execution_started_utc": FAILED_EXECUTION_STARTED_UTC,
        "mtime_precedes_execution": True,
        "device_inode_ctime_available": False,
    }
    return {
        "failed_recovery_finalizer": {
            "historical_path": RECOVERY_FINALIZER.as_posix(),
            "reconstructed_raw_utf8": failed_recovery.decode("utf-8"),
            "bytes": len(failed_recovery),
            "sha256": hashlib.sha256(failed_recovery).hexdigest(),
            "historical_stat_observation": common_stat,
        },
        "failed_recovery_document_finalizer": {
            "historical_path": RECOVERY_DOCUMENT_FINALIZER.as_posix(),
            "reconstructed_raw_utf8": failed_document.decode("utf-8"),
            "bytes": len(failed_document),
            "sha256": hashlib.sha256(failed_document).hexdigest(),
            "historical_stat_observation": common_stat,
        },
        "reconstruction_limitation": (
            "the failed files were overwritten after their fail-closed execution; exact bytes "
            "are reconstructed from the reviewed delta and independently match the hashes and "
            "sizes observed before overwrite, but their original device/inode/ctime are unavailable"
        ),
    }


def _successful_action_source_projection() -> dict[str, Any]:
    recovery_raw, _ = _read_regular(RECOVERY_FINALIZER, name="recovery action source")
    document_raw, _ = _read_regular(
        RECOVERY_DOCUMENT_FINALIZER, name="recovery document action source"
    )
    controller_raw, _ = _read_regular(VALIDATION_CONTROLLER, name="validation action source")
    generator_raw, _ = _read_regular(EXPANDED_GENERATOR, name="document generator source")
    recovery_text = recovery_raw.decode("utf-8")
    document_text = document_raw.decode("utf-8")
    controller_text = controller_raw.decode("utf-8")
    generator_text = generator_raw.decode("utf-8")
    normalized_recovery = re.sub(r"\s+", " ", recovery_text.replace("\\\n", " "))
    preflight_pattern = re.compile(
        r'-m imu_vive_learning\.openaxes_external_native_v3_experiment validate '
        r'--run-dir "\$run_root" --allow-partial --candidates "\$candidates" --contexts 0-23'
    )
    analysis_pattern = re.compile(
        r'-m imu_vive_learning\.openaxes_external_native_v3_analysis analyze '
        r'--run-dir "\$run_root" --prepared-root "\$prepared_root" '
        r'--candidates "\$candidates" --contexts 0-23 --output "\$analysis_receipt"'
    )
    controller_pattern = re.compile(
        r'"\$validation_script" --run-root "\$run_root" --validation-dir "\$validation_dir" '
        r'--candidates "\$candidates" --contexts 0-23'
    )
    forbidden_experiment_action = re.compile(
        r'-m imu_vive_learning\.openaxes_external_native_v3_experiment (?:run|internal-fit)(?: |$)'
    )
    preflight_calls = len(preflight_pattern.findall(normalized_recovery))
    analysis_calls = len(analysis_pattern.findall(normalized_recovery))
    controller_calls = len(controller_pattern.findall(normalized_recovery))
    forbidden_recovery_calls = len(forbidden_experiment_action.findall(normalized_recovery))
    if (preflight_calls, controller_calls, analysis_calls, forbidden_recovery_calls) != (1, 1, 1, 0):
        _fail("successful recovery action tokens changed")
    if document_text.count('python3 "$generator" \\\n') != 1:
        _fail("recovery document generator invocation count changed")
    if re.search(r"(?:internal-fit|openaxes_external_native_v3_experiment\s+run)", document_text):
        _fail("recovery document finalizer contains a fit action")
    controller_loop_count = controller_text.count("for ordinal in {0..9}; do")
    controller_validate_body_count = controller_text.count('run_one "$label" validate')
    controller_evaluate_count = controller_text.count("run_one evaluate-00 evaluate")
    if (controller_loop_count, controller_validate_body_count, controller_evaluate_count) != (1, 1, 1):
        _fail("validation controller exact ten-plus-one action tokens changed")
    if re.search(r"(?:internal-fit|\"\$mode\"\s+run)", controller_text):
        _fail("validation controller contains a fit action")
    if "subprocess" in generator_text or "internal-fit" in generator_text:
        _fail("document generator unexpectedly contains a process/fit action")
    projection = {
        "recovery_preflight_validate_invocations": preflight_calls,
        "recovery_validation_controller_invocations": controller_calls,
        "recovery_analysis_invocations": analysis_calls,
        "recovery_fit_or_internal_fit_invocations": forbidden_recovery_calls,
        "document_generator_invocations": 1,
        "document_fit_or_internal_fit_invocations": 0,
        "validation_controller_exact_loop_range": "0..9",
        "validation_controller_validate_loop_body_invocations": 1,
        "validation_controller_derived_validate_processes": 10,
        "validation_controller_evaluate_invocations": controller_evaluate_count,
        "validation_controller_fit_or_internal_fit_invocations": 0,
        "expanded_generator_process_or_internal_fit_invocations": 0,
    }
    return {**projection, "projection_sha256": hashlib.sha256(_canonical(projection)).hexdigest()}


def _source_authority() -> dict[str, Any]:
    execution_ns = {
        "recovery": int(datetime.fromisoformat(SUCCESSFUL_RECOVERY_STARTED_UTC.replace("Z", "+00:00")).timestamp() * 1_000_000_000),
        "document": int(datetime.fromisoformat(SUCCESSFUL_ANALYSIS_FINISHED_UTC.replace("Z", "+00:00")).timestamp() * 1_000_000_000),
    }
    specs = (
        ("recovery_finalizer", RECOVERY_FINALIZER, "recovery"),
        ("recovery_document_finalizer", RECOVERY_DOCUMENT_FINALIZER, "recovery"),
        ("expanded_document_generator", EXPANDED_GENERATOR, "document"),
        ("validation_controller", VALIDATION_CONTROLLER, "recovery"),
        ("reviewed_base_generator", BASE_GENERATOR, "document"),
        ("experiment_source", EXPERIMENT_SOURCE, "recovery"),
        ("analysis_source", ANALYSIS_SOURCE, "recovery"),
        ("followup_launcher", FOLLOWUP_LAUNCHER, "recovery"),
        ("followup_finalizer", FOLLOWUP_FINALIZER, "recovery"),
        ("followup_document_finalizer", FOLLOWUP_DOCUMENT_FINALIZER, "recovery"),
    )
    sources: dict[str, Any] = {}
    for key, path, boundary in specs:
        ref = _file_projection(path, name=key)
        expected_bytes, expected_sha256 = EXPECTED_SOURCE_IDENTITIES[key]
        if ref["bytes"] != expected_bytes or ref["sha256"] != expected_sha256:
            _fail(f"{key} does not match the reviewed source identity")
        ref["reviewed_expected_bytes"] = expected_bytes
        ref["reviewed_expected_sha256"] = expected_sha256
        ref["reviewed_identity_matches"] = True
        ref["mtime_not_after_execution_boundary"] = ref["stat"]["mtime_ns"] <= execution_ns[boundary]
        ref["execution_boundary_utc"] = (
            SUCCESSFUL_RECOVERY_STARTED_UTC if boundary == "recovery" else SUCCESSFUL_ANALYSIS_FINISHED_UTC
        )
        if not ref["mtime_not_after_execution_boundary"]:
            _fail(f"{key} mtime is later than its execution boundary")
        sources[key] = ref
    self_ref = _file_projection(Path(__file__).absolute(), name="supplement generator")
    return {
        "executed_chain_postexecution_source_references": sources,
        "successful_action_source_projection": _successful_action_source_projection(),
        "failed_first_recovery_source_reconstruction": _reconstruct_failed_sources(),
        "supplement_generator_source_reference": self_ref,
        "executed_byte_binding_limitation": (
            "systemd journal identifies the executed path and time, and each surviving source has "
            "a stable post-execution raw hash/stat with mtime before execution; no preissued source "
            "manifest or retained executable descriptor exists, so cryptographic identity between "
            "the bytes opened by the historical process and the later referenced bytes is not asserted"
        ),
    }


def _selected_task_map() -> dict[str, tuple[str, int]]:
    return {
        f"task_{context * 6 + ordinal:03d}": (candidate, context)
        for context in range(24)
        for candidate, ordinal in zip(CANDIDATES, CANDIDATE_ORDINALS)
    }


def _require_absent_at_dir_fd(directory_fd: int, name: str, *, label: str) -> None:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SupplementError(f"cannot verify absence of {label}") from exc
    _fail(f"{label} must be absent")


def _post_scientific_tree() -> dict[str, Any]:
    selected = _selected_task_map()
    root_fd = -1
    try:
        root_fd = os.open(
            RUN_ROOT,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        run_root_before = os.fstat(root_fd)
        if not stat.S_ISDIR(run_root_before.st_mode):
            _fail("run root descriptor is not a directory")
        _require_absent_at_dir_fd(root_fd, "errors", label="run-root errors directory (first pass)")

        attempt_names, attempt_dir_stat = _directory_snapshot(RUN_ROOT / "attempts", name="attempts")
        result_names, results_dir_stat = _directory_snapshot(RUN_ROOT / "results", name="results")
        if attempt_names != sorted(f"{task_id}.json" for task_id in selected):
            _fail("attempt directory is not the exact selected 72-file roster")
        if result_names != sorted(selected):
            _fail("results directory is not the exact selected 72-directory roster")

        rows: list[dict[str, Any]] = []
        seen_uuids: set[str] = set()
        candidate_counts = {candidate: 0 for candidate in CANDIDATES}
        max_scientific_mtime_ns = 0
        for task_id in sorted(selected):
            candidate, context = selected[task_id]
            attempt_path = RUN_ROOT / "attempts" / f"{task_id}.json"
            result_dir = RUN_ROOT / "results" / task_id
            result_path = result_dir / "result.json"
            prediction_path = result_dir / "outer_predictions.npy"
            checkpoint_dir = result_dir / "checkpoints"
            result_members, result_dir_stat = _directory_snapshot(result_dir, name=f"result directory {task_id}")
            if result_members != ["checkpoints", "outer_predictions.npy", "result.json"]:
                _fail(f"result directory {task_id} is not an exact three-member closed world")
            checkpoint_members, checkpoint_dir_stat = _directory_snapshot(
                checkpoint_dir, name=f"checkpoint directory {task_id}"
            )
            if len(checkpoint_members) != 1 or not checkpoint_members[0].endswith(".json"):
                _fail(f"checkpoint directory {task_id} is not an exact one-file closed world")
            checkpoint_path = checkpoint_dir / checkpoint_members[0]
            attempt, attempt_ref = _load_canonical_json(attempt_path, name=f"attempt {task_id}")
            result, result_ref = _load_canonical_json(result_path, name=f"result {task_id}")
            prediction_ref = _file_projection(prediction_path, name=f"prediction {task_id}")
            checkpoint_ref = _file_projection(checkpoint_path, name=f"checkpoint {task_id}")
            if (
                attempt.get("run_id") != RUN_ID
                or attempt.get("task_id") != task_id
                or attempt.get("scientific_fit_attempt") != 1
                or result.get("run_id") != RUN_ID
                or result.get("task_id") != task_id
                or result.get("candidate_id") != candidate
                or result.get("attempt_uuid_v4") != attempt.get("attempt_uuid_v4")
                or result.get("optimizer_updates") != 400
            ):
                _fail(f"attempt/result lineage changed for {task_id}")
            attempt_uuid = attempt.get("attempt_uuid_v4")
            if not isinstance(attempt_uuid, str) or UUID4.fullmatch(attempt_uuid) is None or attempt_uuid in seen_uuids:
                _fail(f"attempt UUID is malformed or duplicated for {task_id}")
            seen_uuids.add(attempt_uuid)
            training = result.get("training_result")
            if not isinstance(training, Mapping):
                _fail(f"training accounting is missing for {task_id}")
            expected_gradients = 1300 if candidate == CANDIDATES[2] else 400
            expected_solves = 300 if candidate == CANDIDATES[2] else 0
            stage_one_expected = candidate != CANDIDATES[0]
            if (
                training.get("updates") != 400
                or training.get("training_model_forward_calls") != 400
                or training.get("checkpoint_model_forward_calls") != 400
                or training.get("parameter_gradient_traversals") != expected_gradients
                or training.get("cagrad_solver_calls") != expected_solves
                or training.get("outer_feature_loads") != 1
                or training.get("outer_label_reads") != 1
                or training.get("outer_model_forward_calls") != 1
                or training.get("outer_evaluator_calls") != 1
                or (training.get("stage_one_model_state_sha256") is not None) != stage_one_expected
                or (training.get("stage_one_optimizer_state_sha256") is not None) != stage_one_expected
            ):
                _fail(f"fit accounting changed for {task_id}")
            expected_checkpoint_ref = {
                "path": f"checkpoints/{checkpoint_path.name}",
                "bytes": checkpoint_ref["bytes"],
                "sha256": checkpoint_ref["sha256"],
            }
            expected_prediction_ref = {
                "path": "outer_predictions.npy",
                "bytes": prediction_ref["bytes"],
                "sha256": prediction_ref["sha256"],
            }
            if training.get("checkpoint_reference") != expected_checkpoint_ref:
                _fail(f"result-declared checkpoint reference changed for {task_id}")
            if training.get("outer_predictions_reference") != expected_prediction_ref:
                _fail(f"result-declared prediction reference changed for {task_id}")
            for hash_key in ("stage_one_model_state_sha256", "stage_one_optimizer_state_sha256"):
                value = training.get(hash_key)
                if value is not None and (not isinstance(value, str) or HEX64.fullmatch(value) is None):
                    _fail(f"stage-one hash is malformed for {task_id}")
            candidate_counts[candidate] += 1
            max_scientific_mtime_ns = max(
                max_scientific_mtime_ns,
                attempt_ref["stat"]["mtime_ns"],
                result_ref["stat"]["mtime_ns"],
                prediction_ref["stat"]["mtime_ns"],
                checkpoint_ref["stat"]["mtime_ns"],
            )
            rows.append(
                {
                    "task_id": task_id,
                    "candidate_id": candidate,
                    "context_ordinal": context,
                    "attempt_uuid_v4": attempt_uuid,
                    "scientific_fit_attempt": 1,
                    "attempt_reference": attempt_ref,
                    "result_directory_stat": result_dir_stat,
                    "checkpoint_directory_stat": checkpoint_dir_stat,
                    "checkpoint_reference": checkpoint_ref,
                    "result_reference": result_ref,
                    "outer_predictions_reference": prediction_ref,
                }
            )
        _require_absent_at_dir_fd(root_fd, "errors", label="run-root errors directory (second pass)")
        run_root_after = os.fstat(root_fd)
        if not _same_identity(run_root_before, run_root_after):
            _fail("run root identity changed across the two absence passes")
    except OSError as exc:
        raise SupplementError("cannot retain the run-root directory authority") from exc
    finally:
        if root_fd >= 0:
            os.close(root_fd)
    if len(rows) != 72 or len(seen_uuids) != 72 or candidate_counts != {candidate: 24 for candidate in CANDIDATES}:
        _fail("poststate 72-fit cardinality changed")
    recovery_start_ns = int(
        datetime.fromisoformat(SUCCESSFUL_RECOVERY_STARTED_UTC.replace("Z", "+00:00")).timestamp()
        * 1_000_000_000
    )
    if max_scientific_mtime_ns >= recovery_start_ns:
        _fail("a selected scientific member mtime is not before recovery execution")
    return {
        "attempt_directory_stat": attempt_dir_stat,
        "results_directory_stat": results_dir_stat,
        "ordered_rows": rows,
        "ordered_row_count": len(rows),
        "ordered_set_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
        "candidate_fit_counts": candidate_counts,
        "unique_attempt_uuid_count": len(seen_uuids),
        "scientific_fit_attempt_values": [1],
        "error_json_count": 0,
        "error_absence_authority": {
            "retained_run_root_stat": _stat_projection(run_root_before),
            "errors_child_first_pass_lstat_enoent": True,
            "errors_child_second_pass_lstat_enoent": True,
            "result_directories_exact_closed_roster_excludes_error_json": True,
        },
        "single_link_regular_attempt_count": sum(row["attempt_reference"]["stat"]["nlink"] == 1 for row in rows),
        "single_link_regular_result_count": sum(row["result_reference"]["stat"]["nlink"] == 1 for row in rows),
        "single_link_regular_prediction_count": sum(row["outer_predictions_reference"]["stat"]["nlink"] == 1 for row in rows),
        "single_link_regular_checkpoint_count": sum(row["checkpoint_reference"]["stat"]["nlink"] == 1 for row in rows),
        "max_scientific_member_mtime_ns": max_scientific_mtime_ns,
        "max_scientific_member_mtime_utc": _iso_utc_from_ns(max_scientific_mtime_ns),
        "all_scientific_member_mtimes_precede_recovery": True,
        "creation_policy_evidence": {
            "experiment_source_reference": _file_projection(EXPERIMENT_SOURCE, name="experiment source"),
            "selected_attempts_declare_scientific_fit_attempt_one": True,
            "all_current_scientific_members_are_single_link_regular_files": True,
            "historical_o_excl_syscall_observed_directly": False,
            "limitation": (
                "the authenticated experiment source declares exclusive creation and every current "
                "selected member is single-link, but a past O_EXCL syscall is not retrospectively "
                "observable from filesystem state alone"
            ),
        },
    }


def _runtime_status_projection() -> dict[str, Any]:
    value, reference = _load_canonical_json(
        RUNTIME_STATUS, name="runtime status", require_payload=False
    )
    expected_counts = {
        "ambiguous": 0,
        "completed": 72,
        "failed": 0,
        "interrupted": 0,
        "pending": 180,
    }
    if (
        value.get("run_id") != RUN_ID
        or value.get("state_counts") != expected_counts
        or value.get("completed_by_gpu") != [18, 18, 18, 18]
        or value.get("valid_terminal_candidate") is not True
    ):
        _fail("runtime status is not the exact final 72-fit state")
    return {
        "reference": reference,
        "semantic_projection": {
            "state_counts": expected_counts,
            "completed_by_gpu": [18, 18, 18, 18],
            "valid_terminal_candidate": True,
        },
    }


def _validation_projection() -> dict[str, Any]:
    labels = [f"validate-{ordinal:02d}" for ordinal in range(10)] + ["evaluate-00"]
    suffixes = ("started_at_utc", "finished_at_utc", "exit_code", "stdout.log", "stderr.log")
    expected_names = sorted([f"{label}.{suffix}" for label in labels for suffix in suffixes] + ["COMPLETE"])
    names, root_stat = _directory_snapshot(VALIDATION_ROOT, name="sealed validation root")
    if names != expected_names:
        _fail("sealed validation root is not the exact 56-file roster")
    rows: list[dict[str, Any]] = []
    for label in labels:
        refs = {
            suffix: _file_projection(VALIDATION_ROOT / f"{label}.{suffix}", name=f"{label} {suffix}")
            for suffix in suffixes
        }
        exit_raw, _ = _read_regular(VALIDATION_ROOT / f"{label}.exit_code", name=f"{label} exit")
        stdout_raw, _ = _read_regular(VALIDATION_ROOT / f"{label}.stdout.log", name=f"{label} stdout")
        stderr_raw, _ = _read_regular(VALIDATION_ROOT / f"{label}.stderr.log", name=f"{label} stderr")
        if exit_raw != b"0\n" or stdout_raw or stderr_raw:
            _fail(f"{label} is not a silent exit-zero process record")
        rows.append({"label": label, "role": "validate" if label.startswith("validate") else "evaluate", "references": refs})
    complete_raw, _ = _read_regular(VALIDATION_ROOT / "COMPLETE", name="validation COMPLETE")
    if complete_raw != EXPECTED_VALIDATION_COMPLETE:
        _fail("validation COMPLETE changed")
    return {
        "root_path": VALIDATION_ROOT.as_posix(),
        "root_stat": root_stat,
        "member_count": len(names),
        "ordered_members": rows,
        "ordered_set_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
        "validator_process_count": 10,
        "evaluator_process_count": 1,
        "all_exit_zero": True,
        "all_stdout_stderr_empty": True,
        "empty_sha256": EMPTY_SHA256,
        "complete_reference": _file_projection(VALIDATION_ROOT / "COMPLETE", name="validation COMPLETE"),
    }


def _lock_projection() -> dict[str, Any]:
    names, lock_stat = _directory_snapshot(RECOVERY_LOCK, name="recovery lock")
    if names:
        _fail("recovery lock is not an empty directory")
    return {
        "path": RECOVERY_LOCK.as_posix(),
        "stat": lock_stat,
        "empty": True,
        "observed_mode": lock_stat["mode"],
        "gates_output_creation_after_preflight": True,
        "gates_preflight_validator_invocation": False,
        "one_shot_total_validation_claimed": False,
    }


def _output_authority() -> dict[str, Any]:
    initial_analysis, initial_analysis_ref = _load_canonical_json(INITIAL_ANALYSIS, name="initial analysis")
    initial_document, initial_document_ref = _load_canonical_json(INITIAL_DOCUMENT, name="initial document receipt")
    expanded_analysis, expanded_analysis_ref = _load_canonical_json(EXPANDED_ANALYSIS, name="expanded analysis")
    expanded_document, expanded_document_ref = _load_canonical_json(EXPANDED_DOCUMENT, name="expanded document receipt")
    evaluation, evaluation_ref = _load_canonical_json(EVALUATION, name="expanded evaluation")
    if (
        initial_analysis.get("run_id") != RUN_ID
        or initial_document.get("run_id") != RUN_ID
        or expanded_analysis.get("run_id") != RUN_ID
        or expanded_document.get("run_id") != RUN_ID
        or evaluation.get("run_id") != RUN_ID
    ):
        _fail("one or more bound outputs has the wrong run ID")
    if expanded_document.get("scope", {}).get("final_claim_eligible") is not False or expanded_document.get("scope", {}).get("claim_authorized") is not False:
        _fail("expanded document claim boundary changed")
    accounting = expanded_document.get("selected_execution", {}).get("accounting", {})
    if (
        accounting.get("selected_fit_count") != 72
        or accounting.get("selected_attempt_count") != 72
        or accounting.get("selected_result_count") != 72
        or accounting.get("error_json_count") != 0
    ):
        _fail("expanded document accounting changed")
    return {
        "initial_analysis_reference": initial_analysis_ref,
        "initial_document_receipt_reference": initial_document_ref,
        "expanded_analysis_reference": expanded_analysis_ref,
        "expanded_document_receipt_reference": expanded_document_ref,
        "expanded_evaluation_reference": evaluation_ref,
        "sealed_validation_bundle": _validation_projection(),
        "recovery_lock": _lock_projection(),
        "existing_outputs_modified_by_supplement": False,
    }


def _exact_action_accounting(source_projection: Mapping[str, Any]) -> dict[str, Any]:
    expected_projection = {
        "recovery_preflight_validate_invocations": 1,
        "recovery_validation_controller_invocations": 1,
        "recovery_analysis_invocations": 1,
        "recovery_fit_or_internal_fit_invocations": 0,
        "document_generator_invocations": 1,
        "document_fit_or_internal_fit_invocations": 0,
        "validation_controller_exact_loop_range": "0..9",
        "validation_controller_validate_loop_body_invocations": 1,
        "validation_controller_derived_validate_processes": 10,
        "validation_controller_evaluate_invocations": 1,
        "validation_controller_fit_or_internal_fit_invocations": 0,
        "expanded_generator_process_or_internal_fit_invocations": 0,
    }
    expected_projection_sha256 = hashlib.sha256(_canonical(expected_projection)).hexdigest()
    if source_projection != {**expected_projection, "projection_sha256": expected_projection_sha256}:
        _fail("action accounting source projection changed")
    return {
        "failed_first_recovery": {
            "preflight_validate_calls": 0,
            "sealed_bundle_validate_calls": 0,
            "total_validate_calls": 0,
            "evaluate_calls": 0,
            "analysis_calls": 0,
            "fit_calls": 0,
            "internal_fit_calls": 0,
            "failure_before_recovery_lock_or_output": True,
        },
        "successful_recovery": {
            "preflight_validate_calls": 1,
            "sealed_bundle_validate_calls": 10,
            "total_validate_calls": 11,
            "evaluate_calls": 1,
            "analysis_calls": 1,
            "fit_calls": 0,
            "internal_fit_calls": 0,
        },
        "sealed_bundle_validator_count_is_not_total_recovery_validator_count": True,
        "no_retry_claim_applies_to_sealed_ten_plus_one_bundle_only": True,
        "source_action_projection_sha256": expected_projection_sha256,
    }


def build_receipt(
    *,
    systemctl_capture_path: Path,
    original_journal_capture_path: Path,
    recovery_journal_capture_path: Path,
) -> dict[str, Any]:
    systemctl_capture = _raw_capture(
        systemctl_capture_path,
        name="systemctl five-failure-unit capture",
        command=SYSTEMCTL_COMMAND,
    )
    parsed_units = _parse_systemctl_capture(systemctl_capture)
    original_journal = _raw_capture(
        original_journal_capture_path,
        name="original failure journal capture",
        command=ORIGINAL_JOURNAL_COMMAND,
    )
    recovery_journal = _raw_capture(
        recovery_journal_capture_path,
        name="successful recovery journal capture",
        command=RECOVERY_JOURNAL_COMMAND,
    )
    journal_assertions = _validate_journal_captures(original_journal, recovery_journal)
    post_tree = _post_scientific_tree()
    source_authority = _source_authority()
    action_accounting = _exact_action_accounting(
        source_authority["successful_action_source_projection"]
    )
    return _payload(
        {
            "contract": CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "run_id": RUN_ID,
            "issued_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "scope": {
                "authority_kind": "append_only_operational_recovery_supplement",
                "scientific_outputs_recomputed": False,
                "existing_receipts_modified": False,
                "full_42_context_grid_complete": False,
                "final_claim_eligible": False,
                "claim_authorized": False,
            },
            "raw_operational_authority": {
                "five_loaded_failure_units_systemctl_capture": systemctl_capture,
                "five_loaded_failure_units_semantic_projection": parsed_units,
                "original_and_first_recovery_journal_capture": original_journal,
                "successful_v2_transient_recovery_journal_capture": recovery_journal,
                "journal_semantic_assertions": journal_assertions,
                "successful_transient_units_currently_loaded": False,
                "successful_execution_evidence_is_journal_not_live_unit_state": True,
            },
            "source_authority": source_authority,
            "post_recovery_scientific_tree": post_tree,
            "runtime_status": _runtime_status_projection(),
            "output_authority": _output_authority(),
            "exact_action_accounting": action_accounting,
            "claim_boundary": {
                "retraining_performed_field_asserted": False,
                "no_new_fit_evidence": {
                    "exact_selected_attempt_count": 72,
                    "exact_selected_result_count": 72,
                    "unique_attempt_uuid_count": 72,
                    "scientific_fit_attempt_values": [1],
                    "all_scientific_member_mtimes_precede_recovery": True,
                    "recovery_sources_contain_fit_or_internal_fit_invocation": (
                        source_authority["successful_action_source_projection"][
                            "recovery_fit_or_internal_fit_invocations"
                        ]
                        != 0
                    ),
                    "source_action_projection_sha256": source_authority[
                        "successful_action_source_projection"
                    ]["projection_sha256"],
                    "recovery_journal_contains_fit_or_internal_fit_invocation": False,
                },
                "no_retraining_conclusion": (
                    "the preserved poststate and reviewed recovery code/journal contain no evidence "
                    "of a new fit or internal-fit invocation; this is evidence, not a cryptographic "
                    "proof of every historical process action"
                ),
                "pre_post_scientific_byte_equality_available": False,
                "pre_post_scientific_byte_equality_claimed": False,
                "prestate_limitation": (
                    "no exact 72-member byte projection was sealed before recovery, so byte-for-byte "
                    "pre/post equality cannot be established retrospectively; only the exact current "
                    "72-member poststate, single-attempt lineage, pre-recovery mtimes, and recovery "
                    "source/journal evidence are bound"
                ),
                "executed_source_byte_identity_cryptographically_proven": False,
                "scientific_claims_extended": False,
                "adaptive_followup_remains_nonconfirmatory": True,
            },
        }
    )


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    output = path.absolute()
    if output.parent.is_symlink() or not output.parent.is_dir():
        _fail("supplement output parent must be an existing no-symlink directory")
    raw = _canonical(value) + b"\n"
    descriptor = -1
    created = False
    try:
        descriptor = os.open(
            output,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        created = True
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail("supplement write made no progress")
            offset += written
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if (
            info.st_size != len(raw)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or info.st_gid != os.getegid()
        ):
            _fail("supplement output stat is not private and exact")
    except FileExistsError as exc:
        raise SupplementError(f"supplement already exists: {output}") from exc
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
            descriptor = -1
        if created:
            try:
                output.unlink()
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_receipt_file(path: Path) -> dict[str, Any]:
    receipt, _ = _load_canonical_json(path, name="operational recovery supplement")
    expected_keys = {
        "contract",
        "schema_version",
        "run_id",
        "issued_at_utc",
        "scope",
        "raw_operational_authority",
        "source_authority",
        "post_recovery_scientific_tree",
        "runtime_status",
        "output_authority",
        "exact_action_accounting",
        "claim_boundary",
        "payload_sha256",
    }
    if set(receipt) != expected_keys:
        _fail("supplement top-level schema changed")
    if (
        receipt.get("contract") != CONTRACT
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("run_id") != RUN_ID
    ):
        _fail("supplement identity changed")
    action = receipt.get("exact_action_accounting")
    claims = receipt.get("claim_boundary")
    if (
        not isinstance(claims, Mapping)
        or claims.get("retraining_performed_field_asserted") is not False
        or claims.get("pre_post_scientific_byte_equality_available") is not False
        or claims.get("pre_post_scientific_byte_equality_claimed") is not False
        or claims.get("executed_source_byte_identity_cryptographically_proven") is not False
        or claims.get("scientific_claims_extended") is not False
    ):
        _fail("supplement claim boundary changed")
    raw_authority = receipt.get("raw_operational_authority")
    if not isinstance(raw_authority, Mapping):
        _fail("supplement raw authority is missing")
    _parse_systemctl_capture(raw_authority["five_loaded_failure_units_systemctl_capture"])
    _validate_journal_captures(
        raw_authority["original_and_first_recovery_journal_capture"],
        raw_authority["successful_v2_transient_recovery_journal_capture"],
    )
    if receipt.get("runtime_status") != _runtime_status_projection():
        _fail("live runtime status no longer matches supplement")
    if receipt.get("post_recovery_scientific_tree") != _post_scientific_tree():
        _fail("live 72-member scientific poststate no longer matches supplement")
    if receipt.get("output_authority") != _output_authority():
        _fail("live recovery outputs no longer match supplement")
    source = receipt.get("source_authority")
    if not isinstance(source, Mapping):
        _fail("supplement source authority is missing")
    expected_source = _source_authority()
    if source != expected_source:
        _fail("live recovery sources no longer match supplement")
    if not isinstance(action, Mapping) or action != _exact_action_accounting(
        expected_source["successful_action_source_projection"]
    ):
        _fail("supplement action accounting changed")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    issue = sub.add_parser("issue")
    issue.add_argument("--systemctl-capture", required=True)
    issue.add_argument("--original-journal-capture", required=True)
    issue.add_argument("--recovery-journal-capture", required=True)
    issue.add_argument("--output", default=SUPPLEMENT_PATH.as_posix())
    validate = sub.add_parser("validate")
    validate.add_argument("--receipt", default=SUPPLEMENT_PATH.as_posix())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "issue":
        receipt = build_receipt(
            systemctl_capture_path=Path(args.systemctl_capture).absolute(),
            original_journal_capture_path=Path(args.original_journal_capture).absolute(),
            recovery_journal_capture_path=Path(args.recovery_journal_capture).absolute(),
        )
        _write_exclusive(Path(args.output), receipt)
    elif args.command == "validate":
        _validate_receipt_file(Path(args.receipt).absolute())
    else:
        _fail("unknown command")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SupplementError as exc:
        raise SystemExit(f"error: {exc}") from exc
