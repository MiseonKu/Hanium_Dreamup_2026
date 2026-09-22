"""FD-rooted execution and validation for the frozen OpenAXES native-V2 grid.

The host authority calls :func:`main` with inherited ``/proc/self/fd/<n>``
workspace and run-root strings.  Those strings are parsed only to recover the
already-open descriptor numbers: neither root is canonicalized or reopened by
pathname.  Every controller and fit child inherits both descriptors and all
scientific files are opened componentwise relative to them.

The module is intentionally independent of the repository's central CLI and
of every V1/preparation implementation module.  The exact runtime project
closure is the package initializer, ``io_utils``, and the native config,
training, and experiment modules imported below.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import re
import stat
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np

from . import io_utils
from . import openaxes_external_native_v2_config as native_config
from . import openaxes_external_native_v2_training as native_training


PROJECT_DIRECTORY = "09_SO101_IMU_VIVE_LEARNING"
NATIVE_CONTRACT = "openaxes.external.native_sensor_experts.v2"
EXECUTION_CONTRACT = "openaxes.external.native_sensor_experts.v2.execution.v1"
CONFIG_SHA256 = "832a7618f0b7ce94c2813d4b5659153bb88d878d1e67f6ccba3f8f1e3555d59a"
PARENT_CONTRACT = "openaxes.external.model_benchmark.v1.terminal_for_native_v2.v1"
COMPARATOR_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.masked_erm_parent_bundle.v1"
)
SOURCE_AUDIT_CONTRACT = "openaxes.external.native_v2.execution_source_audit.v1"
PREPARATION_CONTRACT = "openaxes.external.preparation.v1"
RUN_ID = "openaxes-native-v2-sealed-003"
PREPARATION_BASE = (
    "09_SO101_IMU_VIVE_LEARNING/artifacts/openaxes_external_v1/"
    "prepared_openaxes-v1-sealed-003"
)
SOURCE_AUDIT_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/receipts/"
    "openaxes_external_native_v2_execution_source_audit.json"
)
CONFIG_PATH = "09_SO101_IMU_VIVE_LEARNING/configs/openaxes_external_native_v2.yaml"
CONTROLLER_SOURCE_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/"
    "openaxes_external_native_v2_experiment.py"
)

SENSOR_GROUPS = ("humerus", "radius", "carpus", "digitus")
CHANNELS = (
    "accel_adxl355_x_mps2",
    "accel_adxl355_y_mps2",
    "accel_adxl355_z_mps2",
    "accel_bmi160_x_mps2",
    "accel_bmi160_y_mps2",
    "accel_bmi160_z_mps2",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
)
FEATURE_COLUMNS = tuple(
    f"{sensor}_{channel}" for sensor in SENSOR_GROUPS for channel in CHANNELS
)
TARGET_COLUMNS = ("tcp_x_m", "tcp_y_m", "tcp_z_m")
METADATA_COLUMNS = ("row_uid", "scenario_id", "sample_index", "aligned_time_us")
VIEW_IDS = (
    "full",
    "outage_humerus",
    "outage_radius",
    "outage_carpus",
    "outage_digitus",
)
OUTAGE_VIEW_IDS = VIEW_IDS[1:]
CANDIDATE_IDS = (
    "dense_6979__scenario_balanced_masked_erm",
    "sensor_experts_6979__scenario_balanced_masked_erm",
    "sensor_experts_6979__scenario_balanced_cagrad",
)
SEEDS = (1701, 2603, 3907)
FOLD_FAMILIES = ("scenario_loso", "speed_pair_out", "shape_out")
PHASES = (
    "execution.started",
    "training.completed",
    "outer_prediction.completed",
    "outer_eval.started",
    "outer_eval.completed",
    "bundle.completed",
    "result.completed",
)
GPU_INVENTORY = (
    {
        "physical_index": 0,
        "uuid": "GPU-462585f2-b4cc-3864-e27c-cbaae4ab6c9a",
        "pci_bus_id": "00000000:01:00.0",
        "name": "NVIDIA CMP 170HX",
    },
    {
        "physical_index": 1,
        "uuid": "GPU-517dfd2b-2ade-0b3f-70d7-0ea2a78f868b",
        "pci_bus_id": "00000000:2D:00.0",
        "name": "NVIDIA CMP 170HX",
    },
    {
        "physical_index": 2,
        "uuid": "GPU-b3db5781-61ae-33bb-63a0-64b14dd9ae6b",
        "pci_bus_id": "00000000:41:00.0",
        "name": "NVIDIA CMP 170HX",
    },
    {
        "physical_index": 3,
        "uuid": "GPU-5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb",
        "pci_bus_id": "00000000:61:00.0",
        "name": "NVIDIA CMP 170HX",
    },
)
FIT_COUNTS_BY_SLOT = (12, 12, 12, 12, 12, 12, 9, 9, 9, 9, 9, 9)
FIT_COUNTS_BY_GPU = (33, 33, 30, 30)
SOURCE_CLOSURE = (
    "configs/openaxes_external_native_v2.yaml",
    "docs/OPENAXES_EXTERNAL_NATIVE_V2_CONTRACT.md",
    "docs/OPENAXES_EXTERNAL_NATIVE_V2_PARENT_RECEIPT_CONTRACT.md",
    "requirements-eval.lock",
    "requirements-gpu.lock",
    "src/imu_vive_learning/__init__.py",
    "src/imu_vive_learning/io_utils.py",
    "src/imu_vive_learning/openaxes_external_native_v2_config.py",
    "src/imu_vive_learning/openaxes_external_native_v2_experiment.py",
    "src/imu_vive_learning/openaxes_external_native_v2_parent_receipt.py",
    "src/imu_vive_learning/openaxes_external_native_v2_training.py",
    "tests/test_openaxes_external_native_v2_config.py",
    "tests/test_openaxes_external_native_v2_experiment.py",
    "tests/test_openaxes_external_native_v2_parent_receipt.py",
    "tests/test_openaxes_external_native_v2_training.py",
)
ROOT_MEMBERS = (
    "prepared_manifest.json",
    "PREPARED.json",
    "task_index.json",
    "parents",
    "tasks",
    "runtime",
    "results",
    "reports",
    "quarantine",
    "development_manifest.json",
    "COMPLETE.json",
)
PARENT_MEMBERS = (
    "parent_receipt_auditor_source_audit.json",
    "v1_masked_erm_comparator_bundle.json",
    "v1_terminal_for_native_v2.json",
)
RUNTIME_MEMBERS = (
    "source_set.json",
    "environment.json",
    "gpu_inventory.json",
    "pre_fit_validation.json",
    "execution_session.json",
    "slot_assignments.json",
    "execution_launches",
    "worker_ready",
    "child_ready",
    "attempt_roster",
    "attempt_ledger",
    "outer_prediction_permits",
    "outer_prediction_claims",
    "outer_label_permits",
    "outer_label_claims",
    "ledgers",
    "utilization.jsonl",
)
REPORT_MEMBERS = (
    "zero_shift.json",
    "primary_gates.json",
    "stress_report.json",
    "signed_shift_report.json",
    "development_summary.json",
)
RESULT_MEMBERS = (
    "checkpoint.pt",
    "transform.json",
    "history.json",
    "training_receipt.json",
    "outer_predictions.parquet",
    "outer_evaluation_rows.parquet",
    "outer_evaluation.json",
    "bundle.json",
    "result.json",
)
HEX40 = r"[0-9a-f]{40}\Z"
HEX64 = r"[0-9a-f]{64}\Z"
FD_PATH = r"/proc/self/fd/([0-9]+)\Z"
TASK_ID = r"task_(00[0-9]|0[1-9][0-9]|1[01][0-9]|12[0-5])\Z"
UUID4 = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"


class OpenAxesExternalNativeV2Error(RuntimeError):
    """Fail-closed native-V2 execution or validation error."""


def _fail(message: str) -> None:
    raise OpenAxesExternalNativeV2Error(message)


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
        raise OpenAxesExternalNativeV2Error(
            "receipt is not finite canonical JSON"
        ) from exc


def _canonical_file(value: Any) -> bytes:
    return _canonical(value) + b"\n"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _reject_constant(token: str) -> None:
    _fail(f"JSON contains a non-finite token: {token}")


def _unique_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(raw: bytes, *, name: str, require_canonical: bool = True) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OpenAxesExternalNativeV2Error(f"{name} is not strict UTF-8 JSON") from exc
    if require_canonical and raw != _canonical_file(value):
        _fail(f"{name} is not canonical JSON with one final LF")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        _fail(f"{name} must be a string-keyed mapping")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        _fail(f"{name} closed key set changed")


def _integer(value: Any, *, name: str, expected: int | None = None) -> int:
    if type(value) is not int or (expected is not None and value != expected):
        _fail(f"{name} is not the required exact integer")
    return value


def _text(value: Any, *, name: str, expected: str | None = None) -> str:
    if type(value) is not str or not value or (expected is not None and value != expected):
        _fail(f"{name} is not exact text")
    return value


def _sha(value: Any, *, name: str) -> str:
    text = _text(value, name=name)
    if re.fullmatch(HEX64, text) is None:
        _fail(f"{name} is not lowercase SHA-256")
    return text


def _uuid4(value: Any, *, name: str) -> str:
    text = _text(value, name=name)
    if re.fullmatch(UUID4, text) is None:
        _fail(f"{name} is not a canonical UUIDv4")
    return text


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_parts(relative: str) -> tuple[str, ...]:
    if (
        type(relative) is not str
        or not relative
        or relative.startswith("/")
        or relative.endswith("/")
        or "\\" in relative
        or "//" in relative
    ):
        _fail("relative path is not canonical POSIX")
    parts = tuple(relative.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        _fail("relative path has an unsafe component")
    return parts


def _fd_from_argument(value: str, *, name: str) -> int:
    if type(value) is not str:
        _fail(f"{name} is not an inherited descriptor path")
    match = re.fullmatch(FD_PATH, value)
    if match is None:
        _fail(f"{name} must be an inherited /proc/self/fd/<n> string")
    descriptor = int(match.group(1))
    if descriptor < 3:
        _fail(f"{name} aliases a standard stream")
    try:
        info = os.fstat(descriptor)
    except OSError as exc:
        raise OpenAxesExternalNativeV2Error(f"{name} descriptor is not open") from exc
    if not stat.S_ISDIR(info.st_mode):
        _fail(f"{name} descriptor is not a directory")
    return os.dup(descriptor)


def _snapshot_fd_from_module() -> int:
    marker = "/src/imu_vive_learning/openaxes_external_native_v2_experiment.py"
    value = __file__
    if not value.endswith(marker):
        _fail("runtime module did not execute through the inherited source snapshot")
    prefix = value[: -len(marker)]
    match = re.fullmatch(FD_PATH, prefix)
    if match is None:
        _fail("runtime source snapshot is not an inherited descriptor path")
    descriptor = int(match.group(1))
    if descriptor < 3 or not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        _fail("runtime source snapshot descriptor is invalid")
    return os.dup(descriptor)


def _open_dir(root_fd: int, relative: str = "", *, create: bool = False) -> int:
    current = os.dup(root_fd)
    if not relative:
        return current
    try:
        for component in _safe_parts(relative):
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=current)
                    os.fsync(current)
                except FileExistsError:
                    pass
            following = os.open(component, 196608, dir_fd=current)
            if not stat.S_ISDIR(os.fstat(following).st_mode):
                os.close(following)
                _fail("path component is not a directory")
            os.close(current)
            current = following
        return current
    except Exception:
        os.close(current)
        raise


def _mkdir_exclusive(root_fd: int, relative: str) -> None:
    parts = _safe_parts(relative)
    parent = _open_dir(root_fd, "/".join(parts[:-1])) if len(parts) > 1 else os.dup(root_fd)
    try:
        os.mkdir(parts[-1], 0o700, dir_fd=parent)
        os.fsync(parent)
    except OSError as exc:
        raise OpenAxesExternalNativeV2Error(
            f"cannot create fresh directory: {relative}"
        ) from exc
    finally:
        os.close(parent)


def _read_bytes(root_fd: int, relative: str, *, name: str) -> bytes:
    parts = _safe_parts(relative)
    parent = _open_dir(root_fd, "/".join(parts[:-1])) if len(parts) > 1 else os.dup(root_fd)
    descriptor = -1
    try:
        descriptor = os.open(parts[-1], 131072, dir_fd=parent)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{name} is not a regular nonsymlink file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        raw = b"".join(chunks)
        if identity_before != identity_after or len(raw) != before.st_size:
            _fail(f"{name} changed while read")
        return raw
    except OSError as exc:
        raise OpenAxesExternalNativeV2Error(f"cannot read {name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _exists(root_fd: int, relative: str) -> bool:
    parts = _safe_parts(relative)
    try:
        parent = _open_dir(root_fd, "/".join(parts[:-1])) if len(parts) > 1 else os.dup(root_fd)
    except (OSError, OpenAxesExternalNativeV2Error):
        return False
    try:
        try:
            os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        os.close(parent)


def _names(root_fd: int, relative: str = "") -> list[str]:
    directory = _open_dir(root_fd, relative)
    try:
        names = sorted(os.listdir(directory))
        for name in names:
            if type(name) is not str or name in {"", ".", ".."} or "/" in name:
                _fail("directory contains a noncanonical member")
            os.stat(name, dir_fd=directory, follow_symlinks=False)
        return names
    finally:
        os.close(directory)


def _write_exclusive(root_fd: int, relative: str, raw: bytes, *, mode: int = 0o600) -> None:
    if type(raw) is not bytes:
        _fail("exclusive writer requires bytes")
    parts = _safe_parts(relative)
    parent = _open_dir(root_fd, "/".join(parts[:-1])) if len(parts) > 1 else os.dup(root_fd)
    descriptor = -1
    try:
        descriptor = os.open(parts[-1], 131265, mode, dir_fd=parent)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail("exclusive write made no progress")
            offset += written
        os.fsync(descriptor)
        os.fsync(parent)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != len(raw):
            _fail("exclusive output identity changed")
    except OSError as exc:
        raise OpenAxesExternalNativeV2Error(
            f"cannot create exclusive output: {relative}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _write_json(root_fd: int, relative: str, value: Any) -> None:
    _write_exclusive(root_fd, relative, _canonical_file(value))


def _reference(root_fd: int, relative: str, *, name: str) -> dict[str, Any]:
    raw = _read_bytes(root_fd, relative, name=name)
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _verify_reference(
    root_fd: int,
    value: Any,
    *,
    expected_path: str,
    name: str,
) -> bytes:
    reference = _mapping(value, name=name)
    _keys(reference, {"path", "bytes", "sha256"}, name=name)
    if reference.get("path") != expected_path:
        _fail(f"{name} path changed")
    _integer(reference.get("bytes"), name=f"{name}.bytes")
    _sha(reference.get("sha256"), name=f"{name}.sha256")
    raw = _read_bytes(root_fd, expected_path, name=name)
    expected = {
        "path": expected_path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if dict(reference) != expected:
        _fail(f"{name} raw reference changed")
    return raw


def _strict_json_at(root_fd: int, relative: str, *, name: str) -> dict[str, Any]:
    value = _strict_json(_read_bytes(root_fd, relative, name=name), name=name)
    return dict(_mapping(value, name=name))


def _payload_hash(value: Mapping[str, Any], key: str, *, name: str) -> None:
    digest = _sha(value.get(key), name=f"{name}.{key}")
    payload = {item: member for item, member in value.items() if item != key}
    if _canonical_hash(payload) != digest:
        _fail(f"{name} self hash changed")


def _process_start_ticks(pid: int) -> int:
    descriptor = os.open(f"/proc/{pid}/stat", 0)
    try:
        raw = b""
        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(descriptor)
    closing = raw.rfind(b")")
    if closing < 1:
        _fail("process stat is malformed")
    fields = raw[closing + 2 :].split()
    if len(fields) < 20 or not fields[19].isdigit():
        _fail("process start ticks are malformed")
    ticks = int(fields[19])
    if ticks <= 0:
        _fail("process start ticks are not positive")
    return ticks


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_ordinal: int
    context_ordinal: int
    fold_id: str
    fold_ordinal: int
    fold_family: str
    train_scenarios: tuple[str, ...]
    test_scenarios: tuple[str, ...]
    seed: int
    seed_ordinal: int
    candidate_id: str
    candidate_ordinal: int
    task_spec_sha256: str

    def payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "contract": "openaxes.external.native_sensor_experts.v2.task_spec.v1",
            "schema_version": 1,
            "task_id": self.task_id,
            "task_ordinal": self.task_ordinal,
            "context_ordinal": self.context_ordinal,
            "fold_id": self.fold_id,
            "fold_ordinal": self.fold_ordinal,
            "fold_family": self.fold_family,
            "seed": self.seed,
            "seed_ordinal": self.seed_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_ordinal": self.candidate_ordinal,
        }
        if include_hash:
            value["task_spec_sha256"] = self.task_spec_sha256
        return value


def _fold_items(preparation_manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    folds = _mapping(preparation_manifest.get("folds"), name="preparation folds")
    raw_items = folds.get("items")
    if type(raw_items) is not list or len(raw_items) != 14:
        _fail("preparation manifest must declare exactly 14 folds")
    expected_scenarios = {
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    }
    family_roles = {
        "scenario_loso": "primary",
        "speed_pair_out": "stress",
        "shape_out": "stress",
    }
    expected_sizes = {
        "scenario_loso": (7, 1),
        "speed_pair_out": (6, 2),
        "shape_out": (4, 4),
    }
    items: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_items):
        fold = _mapping(raw, name=f"fold[{ordinal}]")
        _keys(
            fold,
            {
                "fold_id",
                "family",
                "role",
                "train_scenarios",
                "test_scenarios",
                "inner_selection",
                "outer_test_fitting_allowed",
            },
            name=f"fold[{ordinal}]",
        )
        fold_id = _text(fold["fold_id"], name="fold id")
        family = _text(fold["family"], name="fold family")
        role = _text(fold["role"], name="fold role")
        train = fold["train_scenarios"]
        test = fold["test_scenarios"]
        if (
            family not in family_roles
            or role != family_roles[family]
            or not fold_id.startswith(f"{family}::")
            or type(train) is not list
            or type(test) is not list
            or any(type(item) is not str or not item for item in train + test)
            or len(set(train)) != len(train)
            or len(set(test)) != len(test)
            or set(train).intersection(test)
            or set(train).union(test) != expected_scenarios
            or (len(train), len(test)) != expected_sizes[family]
            or fold["inner_selection"] != "scenario_blocked_training_groups_only"
            or fold["outer_test_fitting_allowed"] is not False
        ):
            _fail("preparation fold identity/partition changed")
        items.append(
            {
                "fold_id": fold_id,
                "family": family,
                "train_scenarios": list(train),
                "test_scenarios": list(test),
            }
        )
    families = [item["family"] for item in items]
    if families != ["scenario_loso"] * 8 + ["speed_pair_out"] * 4 + ["shape_out"] * 2:
        _fail("preparation fold order is not exact 8+4+2")
    if len({item["fold_id"] for item in items}) != 14:
        _fail("preparation fold IDs are not unique")
    return tuple(items)


def build_native_v2_task_grid(
    preparation_manifest: Mapping[str, Any],
) -> tuple[TaskSpec, ...]:
    folds = _fold_items(preparation_manifest)
    tasks: list[TaskSpec] = []
    for fold_ordinal, fold in enumerate(folds):
        for seed_ordinal, seed in enumerate(SEEDS):
            context_ordinal = fold_ordinal * 3 + seed_ordinal
            for candidate_ordinal, candidate_id in enumerate(CANDIDATE_IDS):
                task_ordinal = context_ordinal * 3 + candidate_ordinal
                task_id = f"task_{task_ordinal:03d}"
                base = {
                    "contract": "openaxes.external.native_sensor_experts.v2.task_spec.v1",
                    "schema_version": 1,
                    "task_id": task_id,
                    "task_ordinal": task_ordinal,
                    "context_ordinal": context_ordinal,
                    "fold_id": fold["fold_id"],
                    "fold_ordinal": fold_ordinal,
                    "fold_family": fold["family"],
                    "seed": seed,
                    "seed_ordinal": seed_ordinal,
                    "candidate_id": candidate_id,
                    "candidate_ordinal": candidate_ordinal,
                }
                tasks.append(
                    TaskSpec(
                        task_id=task_id,
                        task_ordinal=task_ordinal,
                        context_ordinal=context_ordinal,
                        fold_id=fold["fold_id"],
                        fold_ordinal=fold_ordinal,
                        fold_family=fold["family"],
                        train_scenarios=tuple(fold["train_scenarios"]),
                        test_scenarios=tuple(fold["test_scenarios"]),
                        seed=seed,
                        seed_ordinal=seed_ordinal,
                        candidate_id=candidate_id,
                        candidate_ordinal=candidate_ordinal,
                        task_spec_sha256=_canonical_hash(base),
                    )
                )
    if (
        len(tasks) != 126
        or [task.task_id for task in tasks] != [f"task_{index:03d}" for index in range(126)]
        or sum(task.fold_family == "scenario_loso" for task in tasks) != 72
        or sum(task.fold_family != "scenario_loso" for task in tasks) != 54
    ):
        _fail("exact 3 x 14 x 3 task grid did not close")
    return tuple(tasks)


def build_slot_assignments(tasks: Sequence[TaskSpec]) -> tuple[dict[str, Any], ...]:
    if len(tasks) != 126:
        _fail("slot assignment requires exactly 126 tasks")
    rows: list[dict[str, Any]] = []
    for slot in range(12):
        task_ids = [
            task.task_id
            for task in tasks
            if task.context_ordinal % 12 == slot
        ]
        rows.append(
            {
                "transport_slot_index": slot,
                "physical_index": slot % 4,
                "local_slot": slot // 4,
                "gpu_reference": dict(GPU_INVENTORY[slot % 4]),
                "ordered_task_ids": task_ids,
            }
        )
    counts = tuple(len(row["ordered_task_ids"]) for row in rows)
    gpu_counts = tuple(
        sum(len(rows[slot]["ordered_task_ids"]) for slot in range(gpu, 12, 4))
        for gpu in range(4)
    )
    flattened = [task_id for row in rows for task_id in row["ordered_task_ids"]]
    if (
        counts != FIT_COUNTS_BY_SLOT
        or gpu_counts != FIT_COUNTS_BY_GPU
        or sorted(flattened) != [f"task_{index:03d}" for index in range(126)]
        or len(flattened) != len(set(flattened))
    ):
        _fail("interleaved 12-slot assignment changed")
    return tuple(rows)


def verify_task_payload(value: Any, expected: TaskSpec) -> None:
    payload = _mapping(value, name="task spec")
    if dict(payload) != expected.payload():
        _fail("task spec payload changed")
    if expected.task_ordinal != expected.context_ordinal * 3 + expected.candidate_ordinal:
        _fail("task ordinal no longer equals context ordinal times three plus candidate")


def _validate_reference_shape(value: Any, *, name: str) -> dict[str, Any]:
    reference = _mapping(value, name=name)
    _keys(reference, {"path", "bytes", "sha256"}, name=name)
    _safe_parts(_text(reference["path"], name=f"{name}.path"))
    size = _integer(reference["bytes"], name=f"{name}.bytes")
    if size < 0:
        _fail(f"{name}.bytes is negative")
    _sha(reference["sha256"], name=f"{name}.sha256")
    return dict(reference)


def derive_primary_gate_inputs(
    outage_contexts: Sequence[Mapping[str, Any]],
    full_contexts: Sequence[Mapping[str, Any]],
    per_seed_outage_contexts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive the seven frozen primary-gate inputs with fixed reductions."""

    if len(outage_contexts) != 32 or len(full_contexts) != 8:
        _fail("primary context cardinality changed")
    outage_metrics = [float(row["rmse3d_m"]) for row in outage_contexts]
    full_metrics = [float(row["rmse3d_m"]) for row in full_contexts]
    if not bool(np.isfinite(outage_metrics).all()) or not bool(
        np.isfinite(full_metrics).all()
    ):
        _fail("primary context metric is non-finite")
    fold_scores = [
        native_training.fixed_mean_float64(
            outage_metrics[offset : offset + 4], name="fold outage score"
        )
        for offset in range(0, 32, 4)
    ]
    mask_scores = [
        native_training.fixed_mean_float64(
            outage_metrics[view_index::4], name="mask mean score"
        )
        for view_index in range(4)
    ]
    mean_context = native_training.fixed_mean_float64(
        outage_metrics, name="primary mean context"
    )
    worst_scenario = max(fold_scores)
    worst_mask = max(mask_scores)
    robust = native_training.fixed_mean_float64(
        [mean_context, worst_scenario, worst_mask], name="primary robust J"
    )
    full_score = 0.5 * (
        native_training.fixed_mean_float64(full_metrics, name="primary full mean")
        + max(full_metrics)
    )
    if len(per_seed_outage_contexts) != 96:
        _fail("per-seed primary context cardinality changed")
    per_seed: list[float] = []
    for seed_ordinal, seed in enumerate(SEEDS):
        rows = per_seed_outage_contexts[seed_ordinal * 32 : (seed_ordinal + 1) * 32]
        if any(row.get("seed") != seed for row in rows):
            _fail("per-seed primary context order changed")
        metrics = [float(row["rmse3d_m"]) for row in rows]
        seed_fold_scores = [
            native_training.fixed_mean_float64(
                metrics[offset : offset + 4], name="per-seed fold outage score"
            )
            for offset in range(0, 32, 4)
        ]
        seed_mask_scores = [
            native_training.fixed_mean_float64(
                metrics[view_index::4], name="per-seed mask mean score"
            )
            for view_index in range(4)
        ]
        per_seed.append(
            native_training.fixed_mean_float64(
                [
                    native_training.fixed_mean_float64(
                        metrics, name="per-seed mean context"
                    ),
                    max(seed_fold_scores),
                    max(seed_mask_scores),
                ],
                name="per-seed robust J",
            )
        )
    return {
        "A_mean_context_rmse3d_m": mean_context,
        "WS_worst_scenario_mean_rmse3d_m": worst_scenario,
        "WM_worst_mask_mean_rmse3d_m": worst_mask,
        "robust_J_m": robust,
        "fold_outage_scores_m": fold_scores,
        "per_seed_J_m": per_seed,
        "full_F_m": full_score,
    }


def validate_comparator_bundle(value: Any) -> dict[str, Any]:
    bundle = _mapping(value, name="comparator bundle")
    _keys(
        bundle,
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
        },
        name="comparator bundle",
    )
    if (
        bundle["contract"] != COMPARATOR_CONTRACT
        or bundle["schema_version"] != 1
        or type(bundle["schema_version"]) is not int
        or bundle["method_id"] != "masked_erm"
        or bundle["seed_order"] != list(SEEDS)
        or bundle["view_order"] != list(VIEW_IDS)
    ):
        _fail("comparator identity/order changed")
    folds = bundle["primary_fold_order"]
    if (
        type(folds) is not list
        or len(folds) != 8
        or len(set(folds)) != 8
        or any(type(item) is not str or not item.startswith("scenario_loso::") for item in folds)
    ):
        _fail("comparator primary fold order changed")
    initial = _mapping(bundle["initial_state_by_seed"], name="comparator initial states")
    if set(initial) != {str(seed) for seed in SEEDS}:
        _fail("comparator seed initial-state keys changed")
    for seed in SEEDS:
        _sha(initial[str(seed)], name="comparator initial state")
    identity = _mapping(bundle["primary_context_identity"], name="comparator identity")
    _keys(identity, {"context_count", "ordered_contexts", "identity_sha256"}, name="comparator identity")
    _integer(identity["context_count"], name="comparator context count", expected=32)
    contexts = identity["ordered_contexts"]
    if type(contexts) is not list or len(contexts) != 32:
        _fail("comparator identity row count changed")
    normalized: list[dict[str, Any]] = []
    for raw in contexts:
        row = _mapping(raw, name="comparator identity row")
        _keys(row, {"fold_id", "scenario_id", "view_id", "rows", "identity_sha256"}, name="comparator identity row")
        normalized.append(dict(row))
    expected_pairs = [(fold, view) for fold in folds for view in OUTAGE_VIEW_IDS]
    if [(row["fold_id"], row["view_id"]) for row in normalized] != expected_pairs:
        _fail("comparator context identity order changed")
    if _canonical_hash(normalized) != identity["identity_sha256"]:
        _fail("comparator context identity hash changed")
    row_specs = (
        ("primary_seed_ensemble_outage_contexts", 32, False),
        ("primary_seed_ensemble_full_contexts", 8, False),
        ("primary_per_seed_outage_contexts", 96, True),
    )
    for key, count, per_seed in row_specs:
        rows = bundle[key]
        if type(rows) is not list or len(rows) != count:
            _fail(f"comparator {key} count changed")
        expected_keys = {"fold_id", "scenario_id", "view_id", "rows", "identity_sha256", "rmse3d_m"}
        if per_seed:
            expected_keys = expected_keys | {"seed"}
        for raw in rows:
            row = _mapping(raw, name=f"comparator {key} row")
            _keys(row, expected_keys, name=f"comparator {key} row")
            _integer(row["rows"], name="comparator rows")
            _sha(row["identity_sha256"], name="comparator identity")
            metric = row["rmse3d_m"]
            if type(metric) is not float or not bool(np.isfinite(metric)) or metric < 0.0:
                _fail("comparator metric is not finite nonnegative binary64")
    outage_rows = bundle["primary_seed_ensemble_outage_contexts"]
    full_rows = bundle["primary_seed_ensemble_full_contexts"]
    per_seed_rows = bundle["primary_per_seed_outage_contexts"]
    if [
        (row["fold_id"], row["scenario_id"], row["view_id"], row["rows"], row["identity_sha256"])
        for row in outage_rows
    ] != [
        (row["fold_id"], row["scenario_id"], row["view_id"], row["rows"], row["identity_sha256"])
        for row in normalized
    ]:
        _fail("comparator outage context identity changed")
    if (
        [(row["fold_id"], row["view_id"]) for row in full_rows]
        != [(fold, "full") for fold in folds]
        or [
            (row["seed"], row["fold_id"], row["view_id"])
            for row in per_seed_rows
        ]
        != [
            (seed, fold, view)
            for seed in SEEDS
            for fold in folds
            for view in OUTAGE_VIEW_IDS
        ]
    ):
        _fail("comparator full/per-seed context order changed")
    derived = _mapping(bundle["derived_gate_inputs"], name="comparator derived gate inputs")
    expected_derived = derive_primary_gate_inputs(outage_rows, full_rows, per_seed_rows)
    if dict(derived) != expected_derived:
        _fail("comparator derived gate inputs do not recompute exactly")
    sources = _mapping(bundle["source_references"], name="comparator sources")
    _keys(
        sources,
        {
            "v1_development_manifest",
            "masked_erm_primary_training_receipts",
            "masked_erm_primary_outer_metric_receipts",
        },
        name="comparator sources",
    )
    _validate_reference_shape(sources["v1_development_manifest"], name="comparator development")
    for key, suffix in (
        ("masked_erm_primary_training_receipts", "/training_receipt.json"),
        ("masked_erm_primary_outer_metric_receipts", "/outer_evaluation.json"),
    ):
        references = sources[key]
        if type(references) is not list or len(references) != 24:
            _fail("comparator source reference count changed")
        paths: list[str] = []
        for raw in references:
            reference = _validate_reference_shape(raw, name="comparator source reference")
            lowered = reference["path"].lower()
            if (
                not reference["path"].startswith("results/task_")
                or not reference["path"].endswith(suffix)
                or any(token in lowered for token in ("prediction", "evaluation_rows", "truth", "stress", "challenge"))
            ):
                _fail("comparator source reference escapes masked primary receipts")
            paths.append(reference["path"])
        if len(paths) != len(set(paths)):
            _fail("comparator source references are duplicated")
    _payload_hash(bundle, "payload_sha256", name="comparator bundle")
    return dict(bundle)


def _load_parents(run_fd: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    if _names(run_fd, "parents") != sorted(PARENT_MEMBERS):
        _fail("parents directory is not exact-three closed world")
    references = {
        name: _reference(run_fd, f"parents/{name}", name=f"parent {name}")
        for name in PARENT_MEMBERS
    }
    terminal = _strict_json_at(
        run_fd,
        "parents/v1_terminal_for_native_v2.json",
        name="V1 terminal receipt",
    )
    comparator = _strict_json_at(
        run_fd,
        "parents/v1_masked_erm_comparator_bundle.json",
        name="V1 comparator bundle",
    )
    if terminal.get("contract") != PARENT_CONTRACT or terminal.get("schema_version") != 1:
        _fail("V1 terminal parent identity changed")
    _payload_hash(terminal, "payload_sha256", name="V1 terminal parent")
    assertions = _mapping(terminal.get("assertions"), name="V1 parent assertions")
    if not assertions or any(value is not True for value in assertions.values()):
        _fail("V1 terminal parent assertions are not all exact true")
    reference = _validate_reference_shape(
        terminal.get("comparator_bundle_reference"), name="terminal comparator reference"
    )
    if reference != references["v1_masked_erm_comparator_bundle.json"]:
        _fail("terminal comparator raw reference changed")
    validate_comparator_bundle(comparator)
    return terminal, comparator, references


def _project_fd(workspace_fd: int) -> int:
    return _open_dir(workspace_fd, PROJECT_DIRECTORY)


def _load_source_authority(
    workspace_fd: int,
    evidence_commit: str,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    if re.fullmatch(HEX40, evidence_commit) is None:
        _fail("source-audit evidence commit is not exact SHA-1")
    raw = _read_bytes(workspace_fd, SOURCE_AUDIT_PATH, name="execution source audit")
    receipt = _strict_json(raw, name="execution source audit")
    source = _mapping(receipt, name="execution source audit")
    if source.get("schema_version") != 1 or source.get("run_id") != RUN_ID:
        _fail("execution source-audit identity changed")
    _payload_hash(source, "payload_sha256", name="execution source audit")
    repository = _mapping(source.get("repository"), name="source-audit repository")
    if repository.get("receipt_path") != SOURCE_AUDIT_PATH:
        _fail("execution source-audit path changed")
    closure = _mapping(source.get("source_closure"), name="source-audit closure")
    if closure.get("file_count") != 15 or type(closure.get("file_count")) is not int:
        _fail("source-audit closure count changed")
    files = closure.get("files")
    if type(files) is not list or len(files) != 15:
        _fail("source-audit closure roster changed")
    if [row.get("path") if isinstance(row, Mapping) else None for row in files] != list(SOURCE_CLOSURE):
        _fail("source-audit closure path order changed")
    project = _project_fd(workspace_fd)
    try:
        normalized: list[dict[str, Any]] = []
        for raw_row, expected_path in zip(files, SOURCE_CLOSURE, strict=True):
            row = _mapping(raw_row, name="source closure row")
            if set(row) != {"path", "bytes", "sha256", "git_blob_oid"}:
                _fail("source closure row schema changed")
            _text(row["git_blob_oid"], name="source closure Git blob")
            reference = _reference(project, expected_path, name=f"source {expected_path}")
            if {key: row[key] for key in ("path", "bytes", "sha256")} != reference:
                _fail("runtime source differs from execution source audit")
            normalized.append(dict(row))
    finally:
        os.close(project)
    source_set_sha256 = _sha(closure.get("source_set_sha256"), name="source set")
    assertions = _mapping(source.get("assertions"), name="source-audit assertions")
    if any(value is not True for value in assertions.values()):
        _fail("source-audit assertions are not all exact true")
    if repository.get("implementation_git_commit") == evidence_commit:
        _fail("source-audit receipt commit must be a later evidence commit")
    return dict(source), source_set_sha256, normalized


def _preparation_reference_path(terminal: Mapping[str, Any], key: str) -> str:
    references = _mapping(terminal.get("references"), name="terminal references")
    reference = _validate_reference_shape(references.get(key), name=f"terminal {key}")
    return reference["path"]


def _read_preparation_file(
    workspace_fd: int,
    terminal: Mapping[str, Any],
    key: str,
    *,
    name: str,
) -> bytes:
    relative = _preparation_reference_path(terminal, key)
    reference = _mapping(terminal["references"][key], name=f"terminal {key}")
    path = f"{PREPARATION_BASE}/{relative}"
    raw = _read_bytes(workspace_fd, path, name=name)
    expected = {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if dict(reference) != expected:
        _fail(f"{name} differs from terminal receipt")
    return raw


def _load_preparation_manifest(
    workspace_fd: int, terminal: Mapping[str, Any]
) -> dict[str, Any]:
    raw = _read_preparation_file(
        workspace_fd,
        terminal,
        "preparation_manifest",
        name="preparation manifest",
    )
    value = _strict_json(raw, name="preparation manifest", require_canonical=False)
    manifest = _mapping(value, name="preparation manifest")
    if manifest.get("contract") != PREPARATION_CONTRACT:
        _fail("preparation contract changed")
    schema = _mapping(manifest.get("schema"), name="preparation schema")
    if (
        schema.get("model_feature_columns") != list(FEATURE_COLUMNS)
        or schema.get("model_feature_count") != 36
        or type(schema.get("model_feature_count")) is not int
        or schema.get("target_columns") != list(TARGET_COLUMNS)
        or schema.get("target_frame") != "UR5_robot_base_TCP"
        or schema.get("target_unit") != "m"
    ):
        _fail("preparation feature/target schema changed")
    sensor_groups = _mapping(schema.get("sensor_groups"), name="preparation sensor groups")
    if set(sensor_groups) != set(SENSOR_GROUPS):
        _fail("preparation sensor-group keys changed")
    flattened = [column for sensor in SENSOR_GROUPS for column in sensor_groups[sensor]]
    if flattened != list(FEATURE_COLUMNS):
        _fail("preparation sensor-group feature order changed")
    _fold_items(manifest)
    return dict(manifest)


def _query_gpu_inventory() -> list[dict[str, Any]]:
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8", "TZ": "UTC"}
    completed = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name",
            "--format=csv,noheader,nounits",
        ],
        env=environment,
        stdout=-1,
        stderr=-1,
        check=False,
    )
    if completed.returncode != 0 or completed.stderr:
        _fail("nvidia-smi inventory query failed")
    try:
        lines = completed.stdout.decode("utf-8").strip().splitlines()
    except UnicodeError as exc:
        raise OpenAxesExternalNativeV2Error("nvidia-smi output is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line in lines:
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4 or not fields[0].isdigit():
            _fail("nvidia-smi inventory row is malformed")
        rows.append(
            {
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "pci_bus_id": fields[2].upper(),
                "name": fields[3],
            }
        )
    expected = [dict(item) for item in GPU_INVENTORY]
    for row in expected:
        row["pci_bus_id"] = row["pci_bus_id"].upper()
    if rows != expected:
        _fail("physical GPU inventory changed")
    return rows


def _environment_projection() -> dict[str, Any]:
    required = {
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "PATH": "/usr/bin:/bin",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(item["uuid"] for item in GPU_INVENTORY),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    observed = {key: os.environ.get(key) for key in required}
    if observed != required:
        _fail("production environment differs from the frozen env-i mapping")
    return {
        "contract": "openaxes.external.native_sensor_experts.v2.environment.v1",
        "environment": required,
        "python_version": "3.12.3",
        "numpy_version": "2.3.5",
        "scipy_version": "1.17.1",
        "pyarrow_version": "21.0.0",
        "pandas_version": "2.3.3",
        "pyyaml_version": "6.0.3",
        "torch_version_checked_in_each_cuda_child": "2.12.1+cu132",
        "torch_cuda_runtime_checked_in_each_cuda_child": "13.2",
        "cudnn_runtime_checked_in_each_cuda_child": 92000,
    }


def _write_parquet(root_fd: int, relative: str, frame: Any) -> None:
    parts = _safe_parts(relative)
    parent = _open_dir(root_fd, "/".join(parts[:-1])) if len(parts) > 1 else os.dup(root_fd)
    descriptor = -1
    try:
        descriptor = os.open(parts[-1], 131265, 0o600, dir_fd=parent)
        handle = os.fdopen(descriptor, "wb", closefd=False)
        try:
            table = io_utils.pa.Table.from_pandas(frame, preserve_index=False)
            io_utils.pq.write_table(
                table,
                handle,
                compression="zstd",
                compression_level=9,
                use_dictionary=True,
                write_statistics=True,
                version="2.6",
                data_page_version="1.0",
            )
            handle.flush()
            os.fsync(descriptor)
        finally:
            handle.close()
        os.fsync(parent)
    except OSError as exc:
        raise OpenAxesExternalNativeV2Error(f"cannot write parquet: {relative}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _read_parquet(root_fd: int, relative: str, columns: Sequence[str]) -> Any:
    parts = _safe_parts(relative)
    parent = _open_dir(root_fd, "/".join(parts[:-1])) if len(parts) > 1 else os.dup(root_fd)
    descriptor = -1
    try:
        descriptor = os.open(parts[-1], 131072, dir_fd=parent)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            _fail("parquet member is not regular")
        handle = os.fdopen(descriptor, "rb", closefd=False)
        try:
            table = io_utils.pq.read_table(handle, columns=list(columns), use_threads=False)
        finally:
            handle.close()
        return table.to_pandas()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


class _PredictionCapability:
    __slots__ = ("task_id", "attempt_id", "claim_sha256", "spent")

    def __init__(self, task_id: str, attempt_id: str, claim_sha256: str) -> None:
        self.task_id = task_id
        self.attempt_id = attempt_id
        self.claim_sha256 = claim_sha256
        self.spent = False

    def consume(self, task_id: str, attempt_id: str, claim_sha256: str) -> None:
        if (
            self.spent
            or self.task_id != task_id
            or self.attempt_id != attempt_id
            or self.claim_sha256 != claim_sha256
        ):
            _fail("outer prediction capability is missing, mismatched, or spent")
        self.spent = True

    def __reduce__(self) -> None:
        _fail("outer prediction capability is not serializable")


class _LabelCapability:
    __slots__ = (
        "task_id",
        "attempt_id",
        "prediction_sha256",
        "started_sha256",
        "claim_sha256",
        "spent",
    )

    def __init__(
        self,
        task_id: str,
        attempt_id: str,
        prediction_sha256: str,
        started_sha256: str,
        claim_sha256: str,
    ) -> None:
        self.task_id = task_id
        self.attempt_id = attempt_id
        self.prediction_sha256 = prediction_sha256
        self.started_sha256 = started_sha256
        self.claim_sha256 = claim_sha256
        self.spent = False

    def consume(
        self,
        task_id: str,
        attempt_id: str,
        prediction_sha256: str,
        started_sha256: str,
        claim_sha256: str,
    ) -> None:
        if (
            self.spent
            or self.task_id != task_id
            or self.attempt_id != attempt_id
            or self.prediction_sha256 != prediction_sha256
            or self.started_sha256 != started_sha256
            or self.claim_sha256 != claim_sha256
        ):
            _fail("outer label capability is missing, mismatched, or spent")
        self.spent = True

    def __reduce__(self) -> None:
        _fail("outer label capability is not serializable")


def _promote_permit(
    run_fd: int,
    permit_path: str,
    claim_path: str,
) -> dict[str, Any]:
    permit_parts = _safe_parts(permit_path)
    claim_parts = _safe_parts(claim_path)
    permit_parent = _open_dir(run_fd, "/".join(permit_parts[:-1]))
    claim_parent = _open_dir(run_fd, "/".join(claim_parts[:-1]))
    try:
        if permit_parts[-1] != claim_parts[-1]:
            _fail("permit and claim task identities differ")
        try:
            os.link(
                permit_parts[-1],
                claim_parts[-1],
                src_dir_fd=permit_parent,
                dst_dir_fd=claim_parent,
                follow_symlinks=False,
            )
            os.fsync(claim_parent)
            os.unlink(permit_parts[-1], dir_fd=permit_parent)
            os.fsync(permit_parent)
        except OSError as exc:
            raise OpenAxesExternalNativeV2Error(
                "permit-to-claim promotion failed closed"
            ) from exc
    finally:
        os.close(permit_parent)
        os.close(claim_parent)
    if _exists(run_fd, permit_path) or not _exists(run_fd, claim_path):
        _fail("permit-to-claim state is not durable claim-only")
    return _reference(run_fd, claim_path, name="promoted claim")


def classify_resume_state(
    *,
    attempt_authority_present: bool,
    last_phase: str | None,
    prediction_permit_present: bool,
    prediction_claim_present: bool,
    label_permit_present: bool,
    label_claim_present: bool,
) -> str:
    values = (
        attempt_authority_present,
        prediction_permit_present,
        prediction_claim_present,
        label_permit_present,
        label_claim_present,
    )
    if any(type(value) is not bool for value in values):
        _fail("resume authority booleans are not exact")
    if last_phase is not None and last_phase not in PHASES:
        _fail("resume phase is outside the exact chain")
    if not attempt_authority_present and last_phase is None:
        if prediction_permit_present and not prediction_claim_present and label_permit_present and not label_claim_present:
            return "fresh_fit_once_only_if_no_attempt_authority_exists_anywhere"
        return "fail_closed"
    prediction_unused = prediction_permit_present and not prediction_claim_present
    prediction_consumed = prediction_claim_present and not prediction_permit_present
    label_unused = label_permit_present and not label_claim_present
    label_consumed = label_claim_present and not label_permit_present
    if last_phase in {None, "execution.started"}:
        return "fail_closed"
    if last_phase == "training.completed":
        if prediction_unused and label_unused:
            return "promote_prediction_permit_and_predict_once"
        return "fail_closed"
    if last_phase == "outer_prediction.completed":
        if prediction_consumed and label_unused:
            return "promote_label_permit_and_evaluate_once"
        return "fail_closed"
    if last_phase == "outer_eval.started":
        return "fail_closed"
    if last_phase == "outer_eval.completed":
        return "bundle_from_persisted_evaluation" if prediction_consumed and label_consumed else "fail_closed"
    if last_phase == "bundle.completed":
        return "result_from_persisted_bundle" if prediction_consumed and label_consumed else "fail_closed"
    if last_phase == "result.completed":
        return "verify_completed" if prediction_consumed and label_consumed else "fail_closed"
    return "fail_closed"


PHASE_PAYLOAD_KEYS = {
    "execution.started": (
        "attempt_roster_reference",
        "attempt_ledger_reference",
        "fit_child_ready_reference",
        "launch_ready_reference",
        "prior_optimizer_steps",
        "prior_outer_forwards",
        "prior_outer_label_reads",
        "prior_task_evaluator_calls",
    ),
    "training.completed": (
        "checkpoint_reference",
        "transform_reference",
        "history_reference",
        "training_receipt_reference",
        "selected_update",
        "optimizer_updates",
        "training_model_forward_calls",
        "checkpoint_model_forward_calls",
        "parameter_gradient_traversals",
        "cagrad_solver_calls",
    ),
    "outer_prediction.completed": (
        "prediction_claim_reference",
        "checkpoint_reference",
        "outer_predictions_reference",
        "outer_feature_loader_calls",
        "outer_model_forward_calls",
        "outer_view_equivalent_forwards",
        "outer_label_reads",
        "task_evaluator_calls",
    ),
    "outer_eval.started": (
        "label_claim_reference",
        "outer_prediction_phase_reference",
        "outer_predictions_reference",
        "prior_outer_label_reads",
        "prior_task_evaluator_calls",
    ),
    "outer_eval.completed": (
        "label_claim_reference",
        "outer_predictions_reference",
        "outer_evaluation_rows_reference",
        "outer_evaluation_reference",
        "outer_label_reads",
        "task_evaluator_calls",
        "views_evaluated",
    ),
    "bundle.completed": (
        "bundle_reference",
        "outer_evaluation_rows_reference",
        "outer_evaluation_reference",
        "source_truth_reopens_after_outer_eval",
        "task_evaluator_replays",
    ),
    "result.completed": (
        "result_reference",
        "bundle_reference",
        "source_truth_reopens_after_outer_eval",
        "task_evaluator_replays",
        "scientific_actions_after_bundle",
    ),
}

ATTEMPT_ORIGIN_KEYS = (
    "execution_session_id",
    "attempt_origin_launch_id",
    "attempt_origin_launch_ordinal",
    "transport_slot_index",
    "local_slot",
    "gpu_reference",
    "controller_source_sha256",
    "environment_sha256",
    "source_set_sha256",
    "task_assignment_sha256",
    "slot_assignment_sha256",
    "comparator_bundle_sha256",
    "pre_fit_validation_sha256",
    "scientific_fit_attempt",
)

EXECUTOR_BINDING_KEYS = (
    "execution_session_id",
    "launch_id",
    "launch_ordinal",
    "transport_slot_index",
    "local_slot",
    "gpu_reference",
    "phase_action_class",
    "launch_ready_reference",
    "controller_ready_reference",
    "child_ready_reference",
    "child_uuid",
    "child_pid",
    "controller_source_sha256",
    "environment_sha256",
    "source_set_sha256",
    "task_assignment_sha256",
    "slot_assignment_sha256",
    "comparator_bundle_sha256",
    "pre_fit_validation_sha256",
)

PHASE_ACTIONS = {
    "execution.started": "fresh_fit",
    "training.completed": "fresh_fit",
    "outer_prediction.completed": "outer_prediction",
    "outer_eval.started": "outer_evaluation",
    "outer_eval.completed": "outer_evaluation",
    "bundle.completed": "bundle_only",
    "result.completed": "result_only",
}


def _validate_phase_payload(
    phase: str,
    payload: Mapping[str, Any],
    task: TaskSpec,
    executor: Mapping[str, Any],
) -> None:
    if set(payload) != set(PHASE_PAYLOAD_KEYS[phase]):
        _fail("phase payload closed schema changed")
    for key, value in payload.items():
        if key.endswith("_reference"):
            _validate_reference_shape(value, name=f"{phase}.{key}")
    exact: dict[str, int] = {}
    if phase == "execution.started":
        exact = {
            "prior_optimizer_steps": 0,
            "prior_outer_forwards": 0,
            "prior_outer_label_reads": 0,
            "prior_task_evaluator_calls": 0,
        }
    elif phase == "training.completed":
        traversals = 2000 if task.candidate_id == CANDIDATE_IDS[2] else 400
        solves = 400 if task.candidate_id == CANDIDATE_IDS[2] else 0
        exact = {
            "optimizer_updates": 400,
            "training_model_forward_calls": 400,
            "checkpoint_model_forward_calls": 400,
            "parameter_gradient_traversals": traversals,
            "cagrad_solver_calls": solves,
        }
        selected = _integer(payload.get("selected_update"), name="selected update")
        if not 1 <= selected <= 400:
            _fail("selected checkpoint update is outside 1..400")
    elif phase == "outer_prediction.completed":
        exact = {
            "outer_feature_loader_calls": 1,
            "outer_model_forward_calls": 1,
            "outer_view_equivalent_forwards": 5,
            "outer_label_reads": 0,
            "task_evaluator_calls": 0,
        }
    elif phase == "outer_eval.started":
        exact = {"prior_outer_label_reads": 0, "prior_task_evaluator_calls": 0}
    elif phase == "outer_eval.completed":
        exact = {"outer_label_reads": 1, "task_evaluator_calls": 1, "views_evaluated": 5}
    elif phase == "bundle.completed":
        exact = {"source_truth_reopens_after_outer_eval": 0, "task_evaluator_replays": 0}
    elif phase == "result.completed":
        exact = {
            "source_truth_reopens_after_outer_eval": 0,
            "task_evaluator_replays": 0,
            "scientific_actions_after_bundle": 0,
        }
    for key, expected in exact.items():
        _integer(payload.get(key), name=f"{phase}.{key}", expected=expected)
    result_base = f"results/{task.task_id}"
    ledger_base = f"runtime/ledgers/{task.task_id}"
    paths = {
        "execution.started": {
            "attempt_roster_reference": f"runtime/attempt_roster/{task.task_id}.json",
            "attempt_ledger_reference": f"runtime/attempt_ledger/{task.task_id}.json",
            "fit_child_ready_reference": executor["child_ready_reference"]["path"],
            "launch_ready_reference": executor["launch_ready_reference"]["path"],
        },
        "training.completed": {
            "checkpoint_reference": f"{result_base}/checkpoint.pt",
            "transform_reference": f"{result_base}/transform.json",
            "history_reference": f"{result_base}/history.json",
            "training_receipt_reference": f"{result_base}/training_receipt.json",
        },
        "outer_prediction.completed": {
            "prediction_claim_reference": f"runtime/outer_prediction_claims/{task.task_id}.json",
            "checkpoint_reference": f"{result_base}/checkpoint.pt",
            "outer_predictions_reference": f"{result_base}/outer_predictions.parquet",
        },
        "outer_eval.started": {
            "label_claim_reference": f"runtime/outer_label_claims/{task.task_id}.json",
            "outer_prediction_phase_reference": f"{ledger_base}/outer_prediction.completed.json",
            "outer_predictions_reference": f"{result_base}/outer_predictions.parquet",
        },
        "outer_eval.completed": {
            "label_claim_reference": f"runtime/outer_label_claims/{task.task_id}.json",
            "outer_predictions_reference": f"{result_base}/outer_predictions.parquet",
            "outer_evaluation_rows_reference": f"{result_base}/outer_evaluation_rows.parquet",
            "outer_evaluation_reference": f"{result_base}/outer_evaluation.json",
        },
        "bundle.completed": {
            "bundle_reference": f"{result_base}/bundle.json",
            "outer_evaluation_rows_reference": f"{result_base}/outer_evaluation_rows.parquet",
            "outer_evaluation_reference": f"{result_base}/outer_evaluation.json",
        },
        "result.completed": {
            "result_reference": f"{result_base}/result.json",
            "bundle_reference": f"{result_base}/bundle.json",
        },
    }[phase]
    for key, expected_path in paths.items():
        if payload[key]["path"] != expected_path:
            _fail(f"{phase}.{key} path changed")
    if (
        phase == "execution.started"
        and (
            payload["fit_child_ready_reference"]
            != executor["child_ready_reference"]
            or payload["launch_ready_reference"]
            != executor["launch_ready_reference"]
        )
    ):
        _fail("execution-started child/launch raw reference changed")


def _validate_phase_bindings(
    task: TaskSpec,
    phase: str,
    origin: Mapping[str, Any],
    executor: Mapping[str, Any],
) -> None:
    slot = task.context_ordinal % 12
    gpu = dict(GPU_INVENTORY[slot % 4])
    if (
        origin["execution_session_id"] != executor["execution_session_id"]
        or origin["transport_slot_index"] != slot
        or origin["local_slot"] != slot // 4
        or origin["gpu_reference"] != gpu
        or executor["transport_slot_index"] != slot
        or executor["local_slot"] != slot // 4
        or executor["gpu_reference"] != gpu
        or origin["scientific_fit_attempt"] != 1
        or type(origin["scientific_fit_attempt"]) is not int
    ):
        _fail("phase task/slot/GPU attempt binding changed")
    _uuid4(origin["execution_session_id"], name="origin session ID")
    _uuid4(executor["child_uuid"], name="executor child UUID")
    child_pid = _integer(executor["child_pid"], name="executor child PID")
    launch_ordinal = _integer(executor["launch_ordinal"], name="executor launch ordinal")
    origin_ordinal = _integer(
        origin["attempt_origin_launch_ordinal"], name="origin launch ordinal"
    )
    if child_pid <= 0 or launch_ordinal < origin_ordinal:
        _fail("executor PID/launch ordering changed")
    if (
        origin["attempt_origin_launch_id"]
        != f"launch_{origin_ordinal:06d}"
        or executor["launch_id"] != f"launch_{launch_ordinal:06d}"
    ):
        _fail("phase launch identity changed")
    if phase in {"execution.started", "training.completed"} and (
        launch_ordinal != origin_ordinal
        or executor["launch_id"] != origin["attempt_origin_launch_id"]
    ):
        _fail("fresh fit phases escaped the attempt-origin launch")
    for key in (
        "controller_source_sha256",
        "environment_sha256",
        "source_set_sha256",
        "task_assignment_sha256",
        "slot_assignment_sha256",
        "comparator_bundle_sha256",
        "pre_fit_validation_sha256",
    ):
        _sha(origin[key], name=f"origin {key}")
        _sha(executor[key], name=f"executor {key}")
        if origin[key] != executor[key]:
            _fail("executor immutable provenance differs from attempt origin")
    for key in (
        "launch_ready_reference",
        "controller_ready_reference",
        "child_ready_reference",
    ):
        _validate_reference_shape(executor[key], name=f"executor {key}")


def make_phase_receipt(
    task: TaskSpec,
    phase: str,
    *,
    attempt_id: str,
    previous_phase: Mapping[str, Any] | None,
    attempt_origin_binding: Mapping[str, Any],
    executor_binding: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Create one exact, self-hashed member of the seven-phase ledger."""

    if phase not in PHASES:
        _fail("phase is outside the frozen seven")
    _uuid4(attempt_id, name="phase attempt ID")
    origin = _mapping(attempt_origin_binding, name="attempt origin")
    executor = _mapping(executor_binding, name="executor binding")
    body = _mapping(payload, name="phase payload")
    _keys(origin, set(ATTEMPT_ORIGIN_KEYS), name="attempt origin")
    _keys(executor, set(EXECUTOR_BINDING_KEYS), name="executor binding")
    if executor["phase_action_class"] != PHASE_ACTIONS[phase]:
        _fail("phase action class changed")
    _validate_phase_bindings(task, phase, origin, executor)
    if phase == PHASES[0]:
        if previous_phase is not None:
            _fail("execution.started must have null previous phase")
        previous: dict[str, Any] | None = None
    else:
        previous = _validate_reference_shape(previous_phase, name="previous phase")
        expected_previous_path = (
            f"runtime/ledgers/{task.task_id}/{PHASES[PHASES.index(phase) - 1]}.json"
        )
        if previous["path"] != expected_previous_path:
            _fail("previous phase path changed")
    _validate_phase_payload(phase, body, task, executor)
    phase_payload = dict(body)
    value = {
        "contract": "openaxes.external.native_sensor_experts.v2.phase_receipt.v1",
        "schema_version": 1,
        "task_id": task.task_id,
        "task_spec_sha256": task.task_spec_sha256,
        "phase": phase,
        "attempt_id": attempt_id,
        "previous_phase": previous,
        "attempt_origin_binding": dict(origin),
        "executor_binding": dict(executor),
        "payload": phase_payload,
        "payload_sha256": _canonical_hash(phase_payload),
    }
    value["phase_receipt_sha256"] = _canonical_hash(value)
    return value


def validate_phase_receipt(
    value: Any,
    task: TaskSpec,
    *,
    expected_phase: str,
    expected_previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    receipt = _mapping(value, name="phase receipt")
    _keys(
        receipt,
        {
            "contract",
            "schema_version",
            "task_id",
            "task_spec_sha256",
            "phase",
            "attempt_id",
            "previous_phase",
            "attempt_origin_binding",
            "executor_binding",
            "payload",
            "payload_sha256",
            "phase_receipt_sha256",
        },
        name="phase receipt",
    )
    expected = make_phase_receipt(
        task,
        expected_phase,
        attempt_id=_text(receipt["attempt_id"], name="phase attempt"),
        previous_phase=expected_previous,
        attempt_origin_binding=_mapping(receipt["attempt_origin_binding"], name="attempt origin"),
        executor_binding=_mapping(receipt["executor_binding"], name="executor binding"),
        payload=_mapping(receipt["payload"], name="phase payload"),
    )
    if dict(receipt) != expected:
        _fail("phase receipt semantic hash or closed schema changed")
    return dict(receipt)


def validate_phase_chain(
    values: Sequence[Mapping[str, Any]], task: TaskSpec
) -> tuple[dict[str, Any], ...]:
    if not 1 <= len(values) <= 7:
        _fail("phase chain is not a nonempty exact prefix")
    normalized: list[dict[str, Any]] = []
    previous_reference: dict[str, Any] | None = None
    stable_attempt: str | None = None
    stable_origin: Mapping[str, Any] | None = None
    previous_executor: Mapping[str, Any] | None = None
    previous_launch_ordinal = -1
    for ordinal, raw in enumerate(values):
        phase = PHASES[ordinal]
        receipt = validate_phase_receipt(
            raw,
            task,
            expected_phase=phase,
            expected_previous=previous_reference,
        )
        if stable_attempt is None:
            stable_attempt = receipt["attempt_id"]
            stable_origin = receipt["attempt_origin_binding"]
        elif (
            receipt["attempt_id"] != stable_attempt
            or receipt["attempt_origin_binding"] != stable_origin
        ):
            _fail("phase chain attempt origin changed")
        executor = receipt["executor_binding"]
        launch_ordinal = executor["launch_ordinal"]
        if launch_ordinal < previous_launch_ordinal:
            _fail("phase executor launch ordinal moved backwards")
        if previous_executor is not None and launch_ordinal == previous_launch_ordinal:
            left = {
                key: value
                for key, value in previous_executor.items()
                if key != "phase_action_class"
            }
            right = {
                key: value
                for key, value in executor.items()
                if key != "phase_action_class"
            }
            if left != right:
                _fail("same-launch phase executor identity changed")
        previous_executor = executor
        previous_launch_ordinal = launch_ordinal
        previous_reference = {
            "path": f"runtime/ledgers/{task.task_id}/{phase}.json",
            "bytes": len(_canonical_file(receipt)),
            "sha256": hashlib.sha256(_canonical_file(receipt)).hexdigest(),
        }
        normalized.append(receipt)
    return tuple(normalized)


def mint_outer_prediction_capability(
    *, task_id: str, attempt_id: str, claim_reference: Mapping[str, Any]
) -> _PredictionCapability:
    reference = _validate_reference_shape(claim_reference, name="prediction claim")
    if reference["path"] != f"runtime/outer_prediction_claims/{task_id}.json":
        _fail("prediction claim path changed")
    return _PredictionCapability(task_id, _uuid4(attempt_id, name="attempt ID"), reference["sha256"])


def mint_outer_label_capability(
    *,
    task_id: str,
    attempt_id: str,
    prediction_reference: Mapping[str, Any],
    outer_eval_started_reference: Mapping[str, Any],
    claim_reference: Mapping[str, Any],
) -> _LabelCapability:
    prediction = _validate_reference_shape(prediction_reference, name="outer prediction")
    started = _validate_reference_shape(outer_eval_started_reference, name="outer eval started")
    claim = _validate_reference_shape(claim_reference, name="label claim")
    if (
        prediction["path"] != f"results/{task_id}/outer_predictions.parquet"
        or started["path"] != f"runtime/ledgers/{task_id}/outer_eval.started.json"
        or claim["path"] != f"runtime/outer_label_claims/{task_id}.json"
    ):
        _fail("outer-label capability references changed")
    return _LabelCapability(
        task_id,
        _uuid4(attempt_id, name="attempt ID"),
        prediction["sha256"],
        started["sha256"],
        claim["sha256"],
    )


def project_feature_only_frame(frame: Any) -> native_training.FeatureOnlyBatch:
    expected = [*METADATA_COLUMNS, *FEATURE_COLUMNS]
    if not isinstance(frame, io_utils.pd.DataFrame) or list(frame.columns) != expected:
        _fail("outer feature loader projection changed or contains a target")
    if len(frame) < 1 or frame["row_uid"].duplicated().any():
        _fail("outer feature identity is empty or duplicated")
    matrix = frame[list(FEATURE_COLUMNS)].to_numpy(dtype="float64", copy=True)
    batch = native_training.FeatureOnlyBatch(
        continuous_features=matrix,
        row_uids=tuple(frame["row_uid"].tolist()),
        scenario_ids=tuple(frame["scenario_id"].tolist()),
        sample_indices=tuple(int(item) for item in frame["sample_index"].tolist()),
        aligned_time_us=tuple(int(item) for item in frame["aligned_time_us"].tolist()),
    )
    batch.validate()
    return batch


def load_outer_feature_batch_once(
    capability: _PredictionCapability,
    *,
    task_id: str,
    attempt_id: str,
    claim_sha256: str,
    opener: Callable[[], Any],
) -> native_training.FeatureOnlyBatch:
    if not isinstance(capability, _PredictionCapability) or not callable(opener):
        _fail("outer feature loader lacks its opaque capability/opener")
    capability.consume(task_id, attempt_id, claim_sha256)
    return project_feature_only_frame(opener())


def evaluate_outer_predictions_once(
    capability: _LabelCapability,
    *,
    task_id: str,
    attempt_id: str,
    prediction_sha256: str,
    outer_eval_started_sha256: str,
    claim_sha256: str,
    prediction_m: Any,
    truth_opener: Callable[[], Any],
) -> tuple[dict[str, Any], Any]:
    """Consume one label capability and evaluate all five views in one call."""

    if not isinstance(capability, _LabelCapability) or not callable(truth_opener):
        _fail("outer evaluator lacks its opaque capability/opener")
    capability.consume(
        task_id,
        attempt_id,
        prediction_sha256,
        outer_eval_started_sha256,
        claim_sha256,
    )
    truth_frame = truth_opener()
    expected = [*METADATA_COLUMNS, *TARGET_COLUMNS]
    if not isinstance(truth_frame, io_utils.pd.DataFrame) or list(truth_frame.columns) != expected:
        _fail("outer truth projection changed")
    if len(truth_frame) < 1 or truth_frame["row_uid"].duplicated().any():
        _fail("outer truth identity is empty or duplicated")
    predictions = np.asarray(prediction_m, dtype="float64")
    truth = truth_frame[list(TARGET_COLUMNS)].to_numpy(dtype="float64", copy=True)
    if predictions.shape != (5, len(truth), 3) or not bool(np.isfinite(predictions).all()):
        _fail("outer prediction/truth shape changed")
    views: list[dict[str, Any]] = []
    for view_index, view_id in enumerate(VIEW_IDS):
        views.append(
            {
                "view_id": view_id,
                "rows": len(truth),
                "rmse3d_m": native_training.fixed_rmse3d_m(
                    predictions[view_index], truth, name=f"outer {view_id}"
                ),
            }
        )
    return {"outer_label_reads": 1, "task_evaluator_calls": 1, "views": views}, truth_frame


def build_task_index_receipt(
    tasks: Sequence[TaskSpec],
    task_spec_references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if len(tasks) != 126 or [task.task_id for task in tasks] != [
        f"task_{ordinal:03d}" for ordinal in range(126)
    ]:
        _fail("task index requires the exact ordered 126-task grid")
    ordered_ids = [task.task_id for task in tasks]
    if list(task_spec_references) != ordered_ids:
        _fail("task-spec reference map order/universe changed")
    references: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for task in tasks:
        reference = _validate_reference_shape(
            task_spec_references[task.task_id], name=f"{task.task_id} task spec"
        )
        if reference["path"] != f"tasks/{task.task_id}.json":
            _fail("task-spec reference path changed")
        references[task.task_id] = reference
        rows.append(
            {
                "task_id": task.task_id,
                "task_spec_sha256": task.task_spec_sha256,
                "task_ordinal": task.task_ordinal,
                "context_ordinal": task.context_ordinal,
                "fold_id": task.fold_id,
                "fold_ordinal": task.fold_ordinal,
                "fold_family": task.fold_family,
                "seed": task.seed,
                "seed_ordinal": task.seed_ordinal,
                "candidate_id": task.candidate_id,
                "candidate_ordinal": task.candidate_ordinal,
                "transport_slot_index": task.context_ordinal % 12,
            }
        )
    task_assignment_sha256 = _canonical_hash(rows)
    value = {
        "contract": "openaxes.external.native_sensor_experts.v2.task_index.v1",
        "schema_version": 1,
        "task_id_order": ordered_ids,
        "task_spec_references": references,
        "task_assignment_rows": rows,
        "task_assignment_sha256": task_assignment_sha256,
    }
    value["payload_sha256"] = _canonical_hash(value)
    return value


def validate_task_index_receipt(
    value: Any, tasks: Sequence[TaskSpec]
) -> dict[str, Any]:
    receipt = _mapping(value, name="task index")
    _keys(
        receipt,
        {
            "contract",
            "schema_version",
            "task_id_order",
            "task_spec_references",
            "task_assignment_rows",
            "task_assignment_sha256",
            "payload_sha256",
        },
        name="task index",
    )
    references = _mapping(receipt["task_spec_references"], name="task references")
    expected = build_task_index_receipt(tasks, references)
    if dict(receipt) != expected:
        _fail("task index closed schema or hash changed")
    return dict(receipt)


def build_slot_assignments_receipt(
    tasks: Sequence[TaskSpec], task_assignment_sha256: str
) -> dict[str, Any]:
    assignment_hash = _sha(task_assignment_sha256, name="task assignment")
    slots = list(build_slot_assignments(tasks))
    value = {
        "contract": "openaxes.external.native_sensor_experts.v2.slot_assignments.v1",
        "schema_version": 1,
        "workers_per_physical_gpu": 3,
        "logical_slot_count": 12,
        "physical_gpu_inventory_sha256": _canonical_hash(
            [dict(item) for item in GPU_INVENTORY]
        ),
        "task_assignment_sha256": assignment_hash,
        "slots": slots,
        "slot_assignment_sha256": _canonical_hash(slots),
    }
    value["payload_sha256"] = _canonical_hash(value)
    return value


def validate_slot_assignments_receipt(
    value: Any, tasks: Sequence[TaskSpec], task_assignment_sha256: str
) -> dict[str, Any]:
    receipt = _mapping(value, name="slot assignments")
    _keys(
        receipt,
        {
            "contract",
            "schema_version",
            "workers_per_physical_gpu",
            "logical_slot_count",
            "physical_gpu_inventory_sha256",
            "task_assignment_sha256",
            "slots",
            "slot_assignment_sha256",
            "payload_sha256",
        },
        name="slot assignments",
    )
    expected = build_slot_assignments_receipt(tasks, task_assignment_sha256)
    if dict(receipt) != expected:
        _fail("slot assignments closed schema or hash changed")
    return dict(receipt)


def _source_set_receipt(
    source_audit: Mapping[str, Any],
    source_set_sha256: str,
    source_rows: Sequence[Mapping[str, Any]],
    parent_references: Mapping[str, Mapping[str, Any]],
    preparation_manifest_reference: Mapping[str, Any],
    preparation_table_reference: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "contract": "openaxes.external.native_sensor_experts.v2.source_set.v1",
        "schema_version": 1,
        "canonical_config_sha256": CONFIG_SHA256,
        "source_audit_payload_sha256": source_audit["payload_sha256"],
        "source_set_sha256": source_set_sha256,
        "files": [dict(row) for row in source_rows],
        "parent_references": {
            key: dict(parent_references[key]) for key in sorted(parent_references)
        },
        "preparation_manifest_reference": dict(preparation_manifest_reference),
        "preparation_table_reference": dict(preparation_table_reference),
    }
    value["payload_sha256"] = _canonical_hash(value)
    return value


def _permit_payload(
    task: TaskSpec,
    *,
    execution_session_id: str,
    task_assignment_sha256: str,
    kind: str,
) -> dict[str, Any]:
    if kind not in {"outer_prediction", "outer_label"}:
        _fail("permit kind changed")
    value = {
        "contract": f"openaxes.external.native_sensor_experts.v2.{kind}_permit.v1",
        "task_id": task.task_id,
        "task_spec_sha256": task.task_spec_sha256,
        "execution_session_id": execution_session_id,
        "task_assignment_sha256": task_assignment_sha256,
    }
    value["permit_sha256"] = _canonical_hash(value)
    return value


def _write_permit_set(
    run_fd: int,
    tasks: Sequence[TaskSpec],
    *,
    execution_session_id: str,
    task_assignment_sha256: str,
    kind: str,
) -> str:
    directory = "outer_prediction_permits" if kind == "outer_prediction" else "outer_label_permits"
    references: dict[str, dict[str, Any]] = {}
    for task in tasks:
        relative = f"runtime/{directory}/{task.task_id}.json"
        _write_json(
            run_fd,
            relative,
            _permit_payload(
                task,
                execution_session_id=execution_session_id,
                task_assignment_sha256=task_assignment_sha256,
                kind=kind,
            ),
        )
        references[task.task_id] = _reference(run_fd, relative, name=f"{kind} permit")
    return _canonical_hash(references)


def prepare_openaxes_external_native_v2_run(
    workspace_fd: int,
    run_fd: int,
    *,
    source_audit_evidence_git_commit: str,
) -> dict[str, Any]:
    """Freeze the exact pre-fit world below two inherited directory FDs."""

    if _names(run_fd) != ["parents"]:
        _fail("fresh native-V2 run root must initially contain only parents")
    native_config.load_openaxes_external_native_v2_config_at(workspace_fd, CONFIG_PATH)
    terminal, comparator, parent_references = _load_parents(run_fd)
    source_audit, source_set_sha256, source_rows = _load_source_authority(
        workspace_fd, source_audit_evidence_git_commit
    )
    preparation_manifest = _load_preparation_manifest(workspace_fd, terminal)
    preparation_manifest_raw = _read_preparation_file(
        workspace_fd, terminal, "preparation_manifest", name="preparation manifest"
    )
    preparation_table_raw = _read_preparation_file(
        workspace_fd, terminal, "preparation_table", name="preparation table"
    )
    preparation_manifest_reference = {
        "path": _preparation_reference_path(terminal, "preparation_manifest"),
        "bytes": len(preparation_manifest_raw),
        "sha256": hashlib.sha256(preparation_manifest_raw).hexdigest(),
    }
    preparation_table_reference = {
        "path": _preparation_reference_path(terminal, "preparation_table"),
        "bytes": len(preparation_table_raw),
        "sha256": hashlib.sha256(preparation_table_raw).hexdigest(),
    }
    environment = _environment_projection()
    gpu_inventory = _query_gpu_inventory()
    tasks = build_native_v2_task_grid(preparation_manifest)

    for relative in (
        "tasks",
        "runtime",
        "results",
        "reports",
        "quarantine",
        "runtime/execution_launches",
        "runtime/worker_ready",
        "runtime/child_ready",
        "runtime/attempt_roster",
        "runtime/attempt_ledger",
        "runtime/outer_prediction_permits",
        "runtime/outer_prediction_claims",
        "runtime/outer_label_permits",
        "runtime/outer_label_claims",
        "runtime/ledgers",
    ):
        _mkdir_exclusive(run_fd, relative)
    _write_exclusive(run_fd, "runtime/utilization.jsonl", b"")
    _write_json(run_fd, "prepared_manifest.json", preparation_manifest)
    prepared_manifest_reference = _reference(
        run_fd, "prepared_manifest.json", name="prepared manifest snapshot"
    )
    _write_json(
        run_fd,
        "PREPARED.json",
        {
            "contract": "openaxes.external.native_sensor_experts.v2.prepared.v1",
            "schema_version": 1,
            "canonical_config_sha256": CONFIG_SHA256,
            "prepared_manifest_reference": prepared_manifest_reference,
            "scientific_actions": 0,
        },
    )
    task_references: dict[str, dict[str, Any]] = {}
    for task in tasks:
        relative = f"tasks/{task.task_id}.json"
        _write_json(run_fd, relative, task.payload())
        task_references[task.task_id] = _reference(run_fd, relative, name="task spec")
    task_index = build_task_index_receipt(tasks, task_references)
    _write_json(run_fd, "task_index.json", task_index)
    slot_assignments = build_slot_assignments_receipt(
        tasks, task_index["task_assignment_sha256"]
    )
    _write_json(run_fd, "runtime/slot_assignments.json", slot_assignments)
    source_set = _source_set_receipt(
        source_audit,
        source_set_sha256,
        source_rows,
        parent_references,
        preparation_manifest_reference,
        preparation_table_reference,
    )
    _write_json(run_fd, "runtime/source_set.json", source_set)
    _write_json(run_fd, "runtime/environment.json", environment)
    _write_json(
        run_fd,
        "runtime/gpu_inventory.json",
        {
            "contract": "openaxes.external.native_sensor_experts.v2.gpu_inventory.v1",
            "schema_version": 1,
            "gpus": gpu_inventory,
            "gpu_inventory_sha256": _canonical_hash(gpu_inventory),
        },
    )
    execution_session_id = str(uuid.uuid4())
    controller_source_sha256 = next(
        row["sha256"]
        for row in source_rows
        if row["path"] == "src/imu_vive_learning/openaxes_external_native_v2_experiment.py"
    )
    session = {
        "contract": "openaxes.external.native_sensor_experts.v2.execution_session.v1",
        "schema_version": 1,
        "execution_session_id": execution_session_id,
        "created_at_utc": _now_utc(),
        "reference_roots": {
            "workspace_root": "inherited_workspace_directory_descriptor",
            "project_root": PROJECT_DIRECTORY,
            "v1_run_root": "receipt_bound_v1_run_root",
            "preparation_run_root": PREPARATION_BASE,
            "v2_run_root": "inherited_run_directory_descriptor",
        },
        "canonical_config_sha256": CONFIG_SHA256,
        "source_set_reference": _reference(run_fd, "runtime/source_set.json", name="source set"),
        "source_set_sha256": source_set_sha256,
        "v1_terminal_receipt_reference": parent_references["v1_terminal_for_native_v2.json"],
        "v1_terminal_receipt_sha256": parent_references["v1_terminal_for_native_v2.json"]["sha256"],
        "comparator_bundle_reference": parent_references["v1_masked_erm_comparator_bundle.json"],
        "comparator_bundle_sha256": parent_references["v1_masked_erm_comparator_bundle.json"]["sha256"],
        "prepared_manifest_reference": prepared_manifest_reference,
        "prepared_manifest_sha256": prepared_manifest_reference["sha256"],
        "prepared_table_reference": preparation_table_reference,
        "prepared_table_sha256": preparation_table_reference["sha256"],
        "environment_reference": _reference(run_fd, "runtime/environment.json", name="environment"),
        "environment_sha256": _canonical_hash(environment),
        "gpu_inventory_reference": _reference(run_fd, "runtime/gpu_inventory.json", name="GPU inventory"),
        "gpu_inventory_sha256": _canonical_hash(gpu_inventory),
        "task_index_reference": _reference(run_fd, "task_index.json", name="task index"),
        "task_assignment_sha256": task_index["task_assignment_sha256"],
        "slot_assignments_reference": _reference(run_fd, "runtime/slot_assignments.json", name="slot assignments"),
        "slot_assignment_sha256": slot_assignments["slot_assignment_sha256"],
        "controller_source_sha256": controller_source_sha256,
    }
    session["session_sha256"] = _canonical_hash(session)
    _write_json(run_fd, "runtime/execution_session.json", session)
    prediction_permit_set_sha256 = _write_permit_set(
        run_fd,
        tasks,
        execution_session_id=execution_session_id,
        task_assignment_sha256=task_index["task_assignment_sha256"],
        kind="outer_prediction",
    )
    label_permit_set_sha256 = _write_permit_set(
        run_fd,
        tasks,
        execution_session_id=execution_session_id,
        task_assignment_sha256=task_index["task_assignment_sha256"],
        kind="outer_label",
    )
    prefit = {
        "contract": "openaxes.external.native_sensor_experts.v2.pre_fit_validation.v1",
        "schema_version": 1,
        "created_at_utc": _now_utc(),
        "execution_session_reference": _reference(run_fd, "runtime/execution_session.json", name="execution session"),
        "execution_session_sha256": session["session_sha256"],
        "canonical_config_sha256": CONFIG_SHA256,
        "source_set_sha256": source_set_sha256,
        "v1_terminal_receipt_sha256": session["v1_terminal_receipt_sha256"],
        "comparator_bundle_sha256": session["comparator_bundle_sha256"],
        "prepared_manifest_sha256": session["prepared_manifest_sha256"],
        "prepared_table_sha256": session["prepared_table_sha256"],
        "environment_sha256": session["environment_sha256"],
        "gpu_inventory_sha256": session["gpu_inventory_sha256"],
        "task_assignment_sha256": session["task_assignment_sha256"],
        "slot_assignment_sha256": session["slot_assignment_sha256"],
        "prediction_permit_set_sha256": prediction_permit_set_sha256,
        "label_permit_set_sha256": label_permit_set_sha256,
        "assertions": {
            "parent_and_git_lineage_valid": True,
            "source_and_config_valid": True,
            "preparation_and_ordered_fold_identity_valid": True,
            "environment_and_locks_valid": True,
            "four_gpu_inventory_valid": True,
            "task_index_and_slot_assignments_valid": True,
            "exact_126_prediction_permits_valid": True,
            "exact_126_label_permits_valid": True,
            "zero_attempt_or_scientific_action_evidence_before_authorization": True,
            "all_mandatory_synthetic_source_tests_passed": True,
        },
    }
    prefit["payload_sha256"] = _canonical_hash(prefit)
    _write_json(run_fd, "runtime/pre_fit_validation.json", prefit)
    return {
        "tasks": tasks,
        "task_index": task_index,
        "slot_assignments": slot_assignments,
        "session": session,
        "prefit": prefit,
        "comparator": comparator,
    }


def _write_launch_zero(
    run_fd: int,
    tasks: Sequence[TaskSpec],
    session: Mapping[str, Any],
    prefit: Mapping[str, Any],
) -> dict[str, Any]:
    launch_id = "launch_000000"
    _mkdir_exclusive(run_fd, f"runtime/execution_launches/{launch_id}")
    _mkdir_exclusive(run_fd, f"runtime/worker_ready/{launch_id}")
    _mkdir_exclusive(run_fd, f"runtime/child_ready/{launch_id}")
    launch = {
        "contract": "openaxes.external.native_sensor_experts.v2.launch_started.v1",
        "schema_version": 1,
        "execution_session_id": session["execution_session_id"],
        "launch_id": launch_id,
        "launch_ordinal": 0,
        "resume_mode": "initial",
        "created_at_utc": _now_utc(),
        "execution_session_reference": _reference(run_fd, "runtime/execution_session.json", name="execution session"),
        "pre_fit_validation_reference": _reference(run_fd, "runtime/pre_fit_validation.json", name="pre-fit validation"),
        "canonical_config_sha256": CONFIG_SHA256,
        "controller_source_sha256": session["controller_source_sha256"],
        "environment_sha256": session["environment_sha256"],
        "source_set_sha256": session["source_set_sha256"],
        "gpu_inventory_sha256": session["gpu_inventory_sha256"],
        "task_assignment_sha256": session["task_assignment_sha256"],
        "slot_assignment_sha256": session["slot_assignment_sha256"],
        "comparator_bundle_sha256": session["comparator_bundle_sha256"],
        "recovery_snapshot_reference": None,
    }
    launch["payload_sha256"] = _canonical_hash(launch)
    launch_path = f"runtime/execution_launches/{launch_id}/launch.started.json"
    _write_json(run_fd, launch_path, launch)
    launch_reference = _reference(run_fd, launch_path, name="launch started")
    controller_pid = os.getpid()
    controller_ticks = _process_start_ticks(controller_pid)
    controller_rows: list[dict[str, Any]] = []
    controller_references: list[dict[str, Any]] = []
    for slot in range(12):
        controller_uuid = str(uuid.uuid4())
        gpu = dict(GPU_INVENTORY[slot % 4])
        controller = {
            "contract": "openaxes.external.native_sensor_experts.v2.controller_ready.v1",
            "schema_version": 1,
            "execution_session_id": session["execution_session_id"],
            "launch_id": launch_id,
            "launch_ordinal": 0,
            "transport_slot_index": slot,
            "local_slot": slot // 4,
            "controller_uuid": controller_uuid,
            "controller_pid": controller_pid,
            "controller_process_start_time_ticks": controller_ticks,
            "gpu_reference": gpu,
            "cuda_visible_devices": gpu["uuid"],
            "torch_imported": False,
            "execution_session_reference": launch["execution_session_reference"],
            "launch_started_reference": launch_reference,
            "controller_source_sha256": session["controller_source_sha256"],
            "environment_sha256": session["environment_sha256"],
            "source_set_sha256": session["source_set_sha256"],
            "task_assignment_sha256": session["task_assignment_sha256"],
            "slot_assignment_sha256": session["slot_assignment_sha256"],
            "pre_fit_validation_sha256": prefit["payload_sha256"],
            "created_at_utc": _now_utc(),
        }
        controller["payload_sha256"] = _canonical_hash(controller)
        path = f"runtime/worker_ready/{launch_id}/slot_{slot:02d}.json"
        _write_json(run_fd, path, controller)
        controller_rows.append(controller)
        controller_references.append(_reference(run_fd, path, name="controller ready"))
    launch_ready = {
        "contract": "openaxes.external.native_sensor_experts.v2.launch_ready.v1",
        "schema_version": 1,
        "execution_session_id": session["execution_session_id"],
        "launch_id": launch_id,
        "launch_ordinal": 0,
        "launch_started_reference": launch_reference,
        "ordered_controller_ready_references": controller_references,
        "controller_ready_set_sha256": _canonical_hash(controller_references),
    }
    launch_ready["payload_sha256"] = _canonical_hash(launch_ready)
    ready_path = f"runtime/execution_launches/{launch_id}/launch.ready.json"
    _write_json(run_fd, ready_path, launch_ready)
    return {
        "launch_id": launch_id,
        "launch_ready_reference": _reference(run_fd, ready_path, name="launch ready"),
        "controllers": controller_rows,
        "controller_references": controller_references,
        "tasks": tasks,
    }


def _attempt_origin(
    task: TaskSpec,
    session: Mapping[str, Any],
    prefit: Mapping[str, Any],
) -> dict[str, Any]:
    slot = task.context_ordinal % 12
    return {
        "execution_session_id": session["execution_session_id"],
        "attempt_origin_launch_id": "launch_000000",
        "attempt_origin_launch_ordinal": 0,
        "transport_slot_index": slot,
        "local_slot": slot // 4,
        "gpu_reference": dict(GPU_INVENTORY[slot % 4]),
        "controller_source_sha256": session["controller_source_sha256"],
        "environment_sha256": session["environment_sha256"],
        "source_set_sha256": session["source_set_sha256"],
        "task_assignment_sha256": session["task_assignment_sha256"],
        "slot_assignment_sha256": session["slot_assignment_sha256"],
        "comparator_bundle_sha256": session["comparator_bundle_sha256"],
        "pre_fit_validation_sha256": prefit["payload_sha256"],
        "scientific_fit_attempt": 1,
    }


def _write_attempt_authority(
    run_fd: int,
    task: TaskSpec,
    *,
    attempt_id: str,
    origin: Mapping[str, Any],
) -> None:
    roster = {
        "contract": "openaxes.external.native_sensor_experts.v2.attempt_roster.v1",
        "task_id": task.task_id,
        "task_spec_sha256": task.task_spec_sha256,
        "attempt_id": attempt_id,
        "attempt_origin_binding": dict(origin),
        "origin": "fresh_fit",
        "attempt_started_or_state_observed": True,
        "survives_quarantine": True,
    }
    roster["payload_sha256"] = _canonical_hash(roster)
    ledger = {
        "contract": "openaxes.external.native_sensor_experts.v2.attempt_ledger.v1",
        "task_id": task.task_id,
        "task_spec_sha256": task.task_spec_sha256,
        "attempt_id": attempt_id,
        "attempt_origin_binding": dict(origin),
        "scientific_fit_attempt": 1,
        "survives_quarantine": True,
    }
    ledger["payload_sha256"] = _canonical_hash(ledger)
    _write_json(run_fd, f"runtime/attempt_roster/{task.task_id}.json", roster)
    _write_json(run_fd, f"runtime/attempt_ledger/{task.task_id}.json", ledger)


def _read_preparation_partition(
    workspace_fd: int,
    terminal: Mapping[str, Any],
    scenarios: Sequence[str],
    columns: Sequence[str],
) -> Any:
    if not scenarios or any(type(item) is not str for item in scenarios):
        _fail("preparation partition scenario filter changed")
    relative = _preparation_reference_path(terminal, "preparation_table")
    path = f"{PREPARATION_BASE}/{relative}"
    parts = _safe_parts(path)
    parent = _open_dir(workspace_fd, "/".join(parts[:-1]))
    descriptor = -1
    try:
        descriptor = os.open(parts[-1], 131072, dir_fd=parent)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            _fail("prepared table is not regular")
        handle = os.fdopen(descriptor, "rb", closefd=False)
        try:
            table = io_utils.pq.read_table(
                handle,
                columns=list(columns),
                filters=[("scenario_id", "in", list(scenarios))],
                use_threads=False,
            )
        finally:
            handle.close()
        frame = table.to_pandas()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)
    if (
        list(frame.columns) != list(columns)
        or len(frame) < 1
        or set(frame["scenario_id"]) != set(scenarios)
        or frame["row_uid"].duplicated().any()
    ):
        _fail("prepared order-preserving partition changed")
    return frame


def _write_torch_checkpoint(run_fd: int, relative: str, state_dict: Mapping[str, Any]) -> None:
    import torch

    parts = _safe_parts(relative)
    parent = _open_dir(run_fd, "/".join(parts[:-1]))
    descriptor = -1
    try:
        descriptor = os.open(parts[-1], 131265, 0o600, dir_fd=parent)
        handle = os.fdopen(descriptor, "wb", closefd=False)
        try:
            torch.save(dict(state_dict), handle)
            handle.flush()
            os.fsync(descriptor)
        finally:
            handle.close()
        os.fsync(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _read_torch_checkpoint(run_fd: int, relative: str) -> Mapping[str, Any]:
    import torch

    parts = _safe_parts(relative)
    parent = _open_dir(run_fd, "/".join(parts[:-1]))
    descriptor = -1
    try:
        descriptor = os.open(parts[-1], 131072, dir_fd=parent)
        handle = os.fdopen(descriptor, "rb", closefd=False)
        try:
            value = torch.load(handle, map_location="cpu", weights_only=True)
        finally:
            handle.close()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)
    return _mapping(value, name="checkpoint state dictionary")


def _write_phase(
    run_fd: int,
    task: TaskSpec,
    phase: str,
    *,
    attempt_id: str,
    origin: Mapping[str, Any],
    executor: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    ordinal = PHASES.index(phase)
    previous = None
    if ordinal > 0:
        previous_path = f"runtime/ledgers/{task.task_id}/{PHASES[ordinal - 1]}.json"
        previous = _reference(run_fd, previous_path, name="previous phase")
    receipt = make_phase_receipt(
        task,
        phase,
        attempt_id=attempt_id,
        previous_phase=previous,
        attempt_origin_binding=origin,
        executor_binding=executor,
        payload=payload,
    )
    path = f"runtime/ledgers/{task.task_id}/{phase}.json"
    _write_json(run_fd, path, receipt)
    return receipt


def _normalize_torch_cuda_uuid(value: Any) -> str:
    value_type = type(value)
    if (
        value_type.__module__ != "torch._C"
        or value_type.__qualname__ != "_CUuuid"
    ):
        _fail("torch CUDA UUID runtime type changed")
    raw = str(value)
    if re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z",
        raw,
    ) is None:
        _fail("torch CUDA UUID text is malformed")
    return f"GPU-{raw}"


def _child_gpu_identity(expected_gpu: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.current_device() != 0
    ):
        _fail("fit child did not receive exactly one CUDA device at ordinal zero")
    properties = torch.cuda.get_device_properties(0)
    observed_uuid = _normalize_torch_cuda_uuid(properties.uuid)
    observed_name = torch.cuda.get_device_name(0)
    inventory = _query_gpu_inventory()
    matching = [row for row in inventory if row["uuid"] == observed_uuid]
    if len(matching) != 1:
        _fail("torch CUDA UUID is absent from authenticated nvidia-smi inventory")
    observed = matching[0]
    if (
        observed != dict(expected_gpu)
        or observed_name != expected_gpu["name"]
        or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu["uuid"]
    ):
        _fail("fit child CUDA UUID/PCI/name assignment changed")
    return {
        "backend": "torch.cuda",
        "logical_device_index": 0,
        "visible_device_count": 1,
        "uuid": observed_uuid,
        "pci_bus_id": observed["pci_bus_id"].upper(),
        "name": observed_name,
    }


def _executor_binding(
    *,
    phase: str,
    task: TaskSpec,
    child_uuid: str,
    child_pid: int,
    child_ready_reference: Mapping[str, Any],
    launch_ready_reference: Mapping[str, Any],
    controller_ready_reference: Mapping[str, Any],
    session: Mapping[str, Any],
    prefit: Mapping[str, Any],
) -> dict[str, Any]:
    slot = task.context_ordinal % 12
    return {
        "execution_session_id": session["execution_session_id"],
        "launch_id": "launch_000000",
        "launch_ordinal": 0,
        "transport_slot_index": slot,
        "local_slot": slot // 4,
        "gpu_reference": dict(GPU_INVENTORY[slot % 4]),
        "phase_action_class": PHASE_ACTIONS[phase],
        "launch_ready_reference": dict(launch_ready_reference),
        "controller_ready_reference": dict(controller_ready_reference),
        "child_ready_reference": dict(child_ready_reference),
        "child_uuid": child_uuid,
        "child_pid": child_pid,
        "controller_source_sha256": session["controller_source_sha256"],
        "environment_sha256": session["environment_sha256"],
        "source_set_sha256": session["source_set_sha256"],
        "task_assignment_sha256": session["task_assignment_sha256"],
        "slot_assignment_sha256": session["slot_assignment_sha256"],
        "comparator_bundle_sha256": session["comparator_bundle_sha256"],
        "pre_fit_validation_sha256": prefit["payload_sha256"],
    }


def _write_child_ready(
    run_fd: int,
    task: TaskSpec,
    *,
    attempt_id: str,
    slot_child_ordinal: int,
    parent_controller_uuid: str,
    controller_ready_reference: Mapping[str, Any],
    launch_ready_reference: Mapping[str, Any],
    session: Mapping[str, Any],
    prefit: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    slot = task.context_ordinal % 12
    gpu = dict(GPU_INVENTORY[slot % 4])
    probe = _child_gpu_identity(gpu)
    child_uuid = str(uuid.uuid4())
    child_pid = os.getpid()
    parent_pid = os.getppid()
    ready = {
        "contract": "openaxes.external.native_sensor_experts.v2.child_ready.v1",
        "schema_version": 1,
        "execution_session_id": session["execution_session_id"],
        "launch_id": "launch_000000",
        "launch_ordinal": 0,
        "task_id": task.task_id,
        "task_spec_sha256": task.task_spec_sha256,
        "attempt_id": attempt_id,
        "child_role": "fresh_fit_full_task",
        "slot_child_ordinal": slot_child_ordinal,
        "child_uuid": child_uuid,
        "child_pid": child_pid,
        "child_process_start_time_ticks": _process_start_ticks(child_pid),
        "parent_controller_uuid": parent_controller_uuid,
        "parent_controller_pid": parent_pid,
        "transport_slot_index": slot,
        "local_slot": slot // 4,
        "gpu_reference": gpu,
        "cuda_visible_devices": gpu["uuid"],
        "torch_imported_after_visible_device_binding": True,
        "torch_cuda_is_available": True,
        "torch_cuda_device_count": 1,
        "torch_cuda_current_device": 0,
        "observed_current_device_uuid": probe["uuid"],
        "observed_current_device_pci_bus_id": probe["pci_bus_id"],
        "observed_current_device_name": probe["name"],
        "cuda_identity_probe": probe,
        "launch_ready_reference": dict(launch_ready_reference),
        "controller_ready_reference": dict(controller_ready_reference),
        "controller_source_sha256": session["controller_source_sha256"],
        "environment_sha256": session["environment_sha256"],
        "source_set_sha256": session["source_set_sha256"],
        "task_assignment_sha256": session["task_assignment_sha256"],
        "slot_assignment_sha256": session["slot_assignment_sha256"],
        "comparator_bundle_sha256": session["comparator_bundle_sha256"],
        "pre_fit_validation_sha256": prefit["payload_sha256"],
        "created_at_utc": _now_utc(),
    }
    ready["payload_sha256"] = _canonical_hash(ready)
    path = f"runtime/child_ready/launch_000000/{child_uuid}.json"
    _write_json(run_fd, path, ready)
    return child_uuid, _reference(run_fd, path, name="child ready")


def _prediction_frame(
    batch: native_training.FeatureOnlyBatch, prediction: Any
) -> Any:
    values = np.asarray(prediction, dtype="float64")
    if values.shape != (5, len(batch.row_uids), 3):
        _fail("prediction tensor shape changed")
    pieces: list[Any] = []
    for view_index, view_id in enumerate(VIEW_IDS):
        piece = io_utils.pd.DataFrame(
            {
                "row_uid": list(batch.row_uids),
                "scenario_id": list(batch.scenario_ids),
                "sample_index": list(batch.sample_indices),
                "aligned_time_us": list(batch.aligned_time_us),
                "view_id": [view_id] * len(batch.row_uids),
                "prediction_x_m": values[view_index, :, 0],
                "prediction_y_m": values[view_index, :, 1],
                "prediction_z_m": values[view_index, :, 2],
            }
        )
        pieces.append(piece)
    frame = io_utils.pd.concat(pieces, ignore_index=True)
    if frame.duplicated(["row_uid", "view_id"]).any():
        _fail("outer prediction identity is duplicated")
    return frame


def _run_one_fresh_task(
    workspace_fd: int,
    run_fd: int,
    *,
    task_id: str,
    attempt_id: str,
    slot_child_ordinal: int,
    parent_controller_uuid: str,
    controller_ready_path: str,
    launch_ready_path: str,
) -> int:
    prepared_manifest = _strict_json_at(
        run_fd, "prepared_manifest.json", name="prepared manifest"
    )
    tasks = build_native_v2_task_grid(prepared_manifest)
    matches = [item for item in tasks if item.task_id == task_id]
    if len(matches) != 1:
        _fail("fit child task ID is outside exact task grid")
    task = matches[0]
    verify_task_payload(
        _strict_json_at(run_fd, f"tasks/{task.task_id}.json", name="task spec"), task
    )
    session = _strict_json_at(
        run_fd, "runtime/execution_session.json", name="execution session"
    )
    prefit = _strict_json_at(
        run_fd, "runtime/pre_fit_validation.json", name="pre-fit validation"
    )
    terminal, comparator, _parent_references = _load_parents(run_fd)
    launch_ready_reference = _reference(run_fd, launch_ready_path, name="launch ready")
    controller_ready_reference = _reference(
        run_fd, controller_ready_path, name="controller ready"
    )
    child_uuid, child_ready_reference = _write_child_ready(
        run_fd,
        task,
        attempt_id=attempt_id,
        slot_child_ordinal=slot_child_ordinal,
        parent_controller_uuid=parent_controller_uuid,
        controller_ready_reference=controller_ready_reference,
        launch_ready_reference=launch_ready_reference,
        session=session,
        prefit=prefit,
    )
    origin = _attempt_origin(task, session, prefit)
    common_executor = {
        phase: _executor_binding(
            phase=phase,
            task=task,
            child_uuid=child_uuid,
            child_pid=os.getpid(),
            child_ready_reference=child_ready_reference,
            launch_ready_reference=launch_ready_reference,
            controller_ready_reference=controller_ready_reference,
            session=session,
            prefit=prefit,
        )
        for phase in PHASES
    }
    _write_phase(
        run_fd,
        task,
        "execution.started",
        attempt_id=attempt_id,
        origin=origin,
        executor=common_executor["execution.started"],
        payload={
            "attempt_roster_reference": _reference(run_fd, f"runtime/attempt_roster/{task.task_id}.json", name="attempt roster"),
            "attempt_ledger_reference": _reference(run_fd, f"runtime/attempt_ledger/{task.task_id}.json", name="attempt ledger"),
            "fit_child_ready_reference": child_ready_reference,
            "launch_ready_reference": launch_ready_reference,
            "prior_optimizer_steps": 0,
            "prior_outer_forwards": 0,
            "prior_outer_label_reads": 0,
            "prior_task_evaluator_calls": 0,
        },
    )

    train_columns = [*METADATA_COLUMNS, *FEATURE_COLUMNS, *TARGET_COLUMNS]
    train_frame = _read_preparation_partition(
        workspace_fd, terminal, task.train_scenarios, train_columns
    )
    dense_expected = comparator["initial_state_by_seed"][str(task.seed)]
    expert_expected: str | None = None
    if task.candidate_id in CANDIDATE_IDS[1:]:
        expert_model = native_training.build_sensor_experts_6979(
            seed=task.seed, device="cpu"
        )
        expert_expected = native_training.state_dict_sha256(expert_model.state_dict())
    training = native_training.train_candidate_train_only(
        train_frame[list(FEATURE_COLUMNS)].to_numpy(dtype="float64", copy=True),
        train_frame[list(TARGET_COLUMNS)].to_numpy(dtype="float64", copy=True),
        train_frame["scenario_id"].tolist(),
        train_frame["row_uid"].tolist(),
        task.train_scenarios,
        candidate_id=task.candidate_id,
        seed=task.seed,
        updates=400,
        device="cuda:0",
        expected_dense_initial_state_sha256=dense_expected,
        expected_expert_initial_state_sha256=expert_expected,
    )
    checkpoint_path = f"results/{task.task_id}/checkpoint.pt"
    transform_path = f"results/{task.task_id}/transform.json"
    history_path = f"results/{task.task_id}/history.json"
    training_receipt_path = f"results/{task.task_id}/training_receipt.json"
    _write_torch_checkpoint(run_fd, checkpoint_path, training.state_dict)
    _write_json(run_fd, transform_path, training.transform.receipt())
    _write_json(
        run_fd,
        history_path,
        {
            "candidate_id": task.candidate_id,
            "seed": task.seed,
            "updates": 400,
            "records": list(training.history),
        },
    )
    training_receipt = {
        "contract": "openaxes.external.native_sensor_experts.v2.training_receipt.v1",
        "task_id": task.task_id,
        "candidate_id": task.candidate_id,
        "seed": task.seed,
        "updates": 400,
        "checkpoint_update": training.checkpoint_update,
        "checkpoint_score_m": training.checkpoint_score_m,
        "checkpoint_outage_components_m": list(
            training.checkpoint_outage_components_m
        ),
        "initial_state_sha256": training.initial_state_sha256,
        "checkpoint_state_sha256": training.checkpoint_state_sha256,
        "optimizer_initial_state_sha256": training.optimizer_initial_state_sha256,
        "training_model_forward_calls": training.training_model_forward_calls,
        "checkpoint_model_forward_calls": training.checkpoint_model_forward_calls,
        "parameter_gradient_traversals": training.parameter_gradient_traversals,
        "cagrad_solver_calls": training.cagrad_solver_calls,
    }
    training_receipt["payload_sha256"] = _canonical_hash(training_receipt)
    _write_json(run_fd, training_receipt_path, training_receipt)
    checkpoint_reference = _reference(run_fd, checkpoint_path, name="checkpoint")
    _write_phase(
        run_fd,
        task,
        "training.completed",
        attempt_id=attempt_id,
        origin=origin,
        executor=common_executor["training.completed"],
        payload={
            "checkpoint_reference": checkpoint_reference,
            "transform_reference": _reference(run_fd, transform_path, name="transform"),
            "history_reference": _reference(run_fd, history_path, name="history"),
            "training_receipt_reference": _reference(run_fd, training_receipt_path, name="training receipt"),
            "selected_update": training.checkpoint_update,
            "optimizer_updates": 400,
            "training_model_forward_calls": 400,
            "checkpoint_model_forward_calls": 400,
            "parameter_gradient_traversals": training.parameter_gradient_traversals,
            "cagrad_solver_calls": training.cagrad_solver_calls,
        },
    )

    prediction_claim = _promote_permit(
        run_fd,
        f"runtime/outer_prediction_permits/{task.task_id}.json",
        f"runtime/outer_prediction_claims/{task.task_id}.json",
    )
    prediction_capability = mint_outer_prediction_capability(
        task_id=task.task_id,
        attempt_id=attempt_id,
        claim_reference=prediction_claim,
    )

    def feature_opener() -> Any:
        return _read_preparation_partition(
            workspace_fd,
            terminal,
            task.test_scenarios,
            [*METADATA_COLUMNS, *FEATURE_COLUMNS],
        )

    feature_batch = load_outer_feature_batch_once(
        prediction_capability,
        task_id=task.task_id,
        attempt_id=attempt_id,
        claim_sha256=prediction_claim["sha256"],
        opener=feature_opener,
    )
    locked_state = _read_torch_checkpoint(run_fd, checkpoint_path)
    prediction = native_training.predict_feature_only_batch_once(
        candidate_id=task.candidate_id,
        seed=task.seed,
        state_dict=locked_state,
        transform=training.transform,
        feature_batch=feature_batch,
        device="cuda:0",
    )
    prediction_frame = _prediction_frame(feature_batch, prediction)
    prediction_path = f"results/{task.task_id}/outer_predictions.parquet"
    _write_parquet(run_fd, prediction_path, prediction_frame)
    prediction_reference = _reference(run_fd, prediction_path, name="outer predictions")
    _write_phase(
        run_fd,
        task,
        "outer_prediction.completed",
        attempt_id=attempt_id,
        origin=origin,
        executor=common_executor["outer_prediction.completed"],
        payload={
            "prediction_claim_reference": prediction_claim,
            "checkpoint_reference": checkpoint_reference,
            "outer_predictions_reference": prediction_reference,
            "outer_feature_loader_calls": 1,
            "outer_model_forward_calls": 1,
            "outer_view_equivalent_forwards": 5,
            "outer_label_reads": 0,
            "task_evaluator_calls": 0,
        },
    )

    label_claim = _promote_permit(
        run_fd,
        f"runtime/outer_label_permits/{task.task_id}.json",
        f"runtime/outer_label_claims/{task.task_id}.json",
    )
    outer_prediction_phase_reference = _reference(
        run_fd,
        f"runtime/ledgers/{task.task_id}/outer_prediction.completed.json",
        name="outer prediction phase",
    )
    _write_phase(
        run_fd,
        task,
        "outer_eval.started",
        attempt_id=attempt_id,
        origin=origin,
        executor=common_executor["outer_eval.started"],
        payload={
            "label_claim_reference": label_claim,
            "outer_prediction_phase_reference": outer_prediction_phase_reference,
            "outer_predictions_reference": prediction_reference,
            "prior_outer_label_reads": 0,
            "prior_task_evaluator_calls": 0,
        },
    )
    started_reference = _reference(
        run_fd,
        f"runtime/ledgers/{task.task_id}/outer_eval.started.json",
        name="outer eval started",
    )
    label_capability = mint_outer_label_capability(
        task_id=task.task_id,
        attempt_id=attempt_id,
        prediction_reference=prediction_reference,
        outer_eval_started_reference=started_reference,
        claim_reference=label_claim,
    )

    def truth_opener() -> Any:
        return _read_preparation_partition(
            workspace_fd,
            terminal,
            task.test_scenarios,
            [*METADATA_COLUMNS, *TARGET_COLUMNS],
        )

    evaluation, truth_frame = evaluate_outer_predictions_once(
        label_capability,
        task_id=task.task_id,
        attempt_id=attempt_id,
        prediction_sha256=prediction_reference["sha256"],
        outer_eval_started_sha256=started_reference["sha256"],
        claim_sha256=label_claim["sha256"],
        prediction_m=prediction,
        truth_opener=truth_opener,
    )
    if (
        tuple(truth_frame["row_uid"].tolist()) != feature_batch.row_uids
        or tuple(truth_frame["scenario_id"].tolist()) != feature_batch.scenario_ids
        or tuple(int(item) for item in truth_frame["sample_index"].tolist())
        != feature_batch.sample_indices
        or tuple(int(item) for item in truth_frame["aligned_time_us"].tolist())
        != feature_batch.aligned_time_us
    ):
        _fail("outer prediction/truth ordered identity changed")
    evaluation_rows = prediction_frame.merge(
        truth_frame[["row_uid", *TARGET_COLUMNS]],
        on="row_uid",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    evaluation_rows_path = f"results/{task.task_id}/outer_evaluation_rows.parquet"
    evaluation_path = f"results/{task.task_id}/outer_evaluation.json"
    _write_parquet(run_fd, evaluation_rows_path, evaluation_rows)
    evaluation_payload = {
        "contract": "openaxes.external.native_sensor_experts.v2.outer_evaluation.v1",
        "task_id": task.task_id,
        "candidate_id": task.candidate_id,
        "fold_id": task.fold_id,
        "fold_family": task.fold_family,
        "seed": task.seed,
        **evaluation,
    }
    evaluation_payload["payload_sha256"] = _canonical_hash(evaluation_payload)
    _write_json(run_fd, evaluation_path, evaluation_payload)
    evaluation_rows_reference = _reference(
        run_fd, evaluation_rows_path, name="outer evaluation rows"
    )
    evaluation_reference = _reference(
        run_fd, evaluation_path, name="outer evaluation"
    )
    _write_phase(
        run_fd,
        task,
        "outer_eval.completed",
        attempt_id=attempt_id,
        origin=origin,
        executor=common_executor["outer_eval.completed"],
        payload={
            "label_claim_reference": label_claim,
            "outer_predictions_reference": prediction_reference,
            "outer_evaluation_rows_reference": evaluation_rows_reference,
            "outer_evaluation_reference": evaluation_reference,
            "outer_label_reads": 1,
            "task_evaluator_calls": 1,
            "views_evaluated": 5,
        },
    )
    bundle_path = f"results/{task.task_id}/bundle.json"
    bundle = {
        "contract": "openaxes.external.native_sensor_experts.v2.task_bundle.v1",
        "task_id": task.task_id,
        "task_spec_sha256": task.task_spec_sha256,
        "attempt_id": attempt_id,
        "training_receipt_reference": _reference(run_fd, training_receipt_path, name="training receipt"),
        "outer_predictions_reference": prediction_reference,
        "outer_evaluation_rows_reference": evaluation_rows_reference,
        "outer_evaluation_reference": evaluation_reference,
        "outer_label_reads": 1,
        "task_evaluator_calls": 1,
    }
    bundle["payload_sha256"] = _canonical_hash(bundle)
    _write_json(run_fd, bundle_path, bundle)
    bundle_reference = _reference(run_fd, bundle_path, name="task bundle")
    _write_phase(
        run_fd,
        task,
        "bundle.completed",
        attempt_id=attempt_id,
        origin=origin,
        executor=common_executor["bundle.completed"],
        payload={
            "bundle_reference": bundle_reference,
            "outer_evaluation_rows_reference": evaluation_rows_reference,
            "outer_evaluation_reference": evaluation_reference,
            "source_truth_reopens_after_outer_eval": 0,
            "task_evaluator_replays": 0,
        },
    )
    result_path = f"results/{task.task_id}/result.json"
    result = {
        "contract": "openaxes.external.native_sensor_experts.v2.task_result.v1",
        "task_id": task.task_id,
        "task_spec_sha256": task.task_spec_sha256,
        "attempt_id": attempt_id,
        "bundle_reference": bundle_reference,
        "candidate_id": task.candidate_id,
        "fold_id": task.fold_id,
        "fold_family": task.fold_family,
        "seed": task.seed,
    }
    result["payload_sha256"] = _canonical_hash(result)
    _write_json(run_fd, result_path, result)
    _write_phase(
        run_fd,
        task,
        "result.completed",
        attempt_id=attempt_id,
        origin=origin,
        executor=common_executor["result.completed"],
        payload={
            "result_reference": _reference(run_fd, result_path, name="task result"),
            "bundle_reference": bundle_reference,
            "source_truth_reopens_after_outer_eval": 0,
            "task_evaluator_replays": 0,
            "scientific_actions_after_bundle": 0,
        },
    )
    return 0


def _run_fresh_task_children(
    workspace_fd: int,
    run_fd: int,
    snapshot_fd: int,
    dependency_fd: int,
    tasks: Sequence[TaskSpec],
    session: Mapping[str, Any],
    prefit: Mapping[str, Any],
    launch: Mapping[str, Any],
) -> None:
    slots = build_slot_assignments(tasks)
    by_id = {task.task_id: task for task in tasks}
    base_environment = dict(_environment_projection()["environment"])
    dependency_root = f"/proc/self/fd/{dependency_fd}"
    if dependency_fd < 3:
        _fail("fit-child dependency descriptor is not a retained directory")
    dependency_info = os.fstat(dependency_fd)
    if not stat.S_ISDIR(dependency_info.st_mode):
        _fail("fit-child dependency descriptor is not a retained directory")
    maximum = max(len(row["ordered_task_ids"]) for row in slots)
    for slot_child_ordinal in range(maximum):
        running: list[tuple[str, Any]] = []
        for slot, row in enumerate(slots):
            ordered = row["ordered_task_ids"]
            if slot_child_ordinal >= len(ordered):
                continue
            task = by_id[ordered[slot_child_ordinal]]
            _mkdir_exclusive(run_fd, f"results/{task.task_id}")
            _mkdir_exclusive(run_fd, f"runtime/ledgers/{task.task_id}")
            attempt_id = str(uuid.uuid4())
            origin = _attempt_origin(task, session, prefit)
            _write_attempt_authority(
                run_fd, task, attempt_id=attempt_id, origin=origin
            )
            controller = launch["controllers"][slot]
            controller_path = f"runtime/worker_ready/launch_000000/slot_{slot:02d}.json"
            launch_ready_path = "runtime/execution_launches/launch_000000/launch.ready.json"
            child_environment = dict(base_environment)
            child_environment["CUDA_VISIBLE_DEVICES"] = row["gpu_reference"]["uuid"]
            child_environment["PYTHONPATH"] = os.pathsep.join(
                (f"/proc/self/fd/{snapshot_fd}/src", dependency_root)
            )
            command = [
                "/proc/self/exe",
                "-P",
                "-s",
                "-S",
                "-B",
                "-m",
                "imu_vive_learning.openaxes_external_native_v2_experiment",
                "internal-fit",
                "--workspace-root",
                f"/proc/self/fd/{workspace_fd}",
                "--run-dir",
                f"/proc/self/fd/{run_fd}",
                "--dependency-root",
                dependency_root,
                "--task-id",
                task.task_id,
                "--attempt-id",
                attempt_id,
                "--slot-child-ordinal",
                str(slot_child_ordinal),
                "--parent-controller-uuid",
                controller["controller_uuid"],
                "--controller-ready-path",
                controller_path,
                "--launch-ready-path",
                launch_ready_path,
            ]
            process = subprocess.Popen(
                command,
                env=child_environment,
                pass_fds=(workspace_fd, run_fd, snapshot_fd, dependency_fd),
                close_fds=True,
            )
            running.append((task.task_id, process))
        for task_id, process in running:
            returncode = process.wait()
            if returncode != 0:
                _fail(f"fresh fit child failed closed for {task_id}")


def _identity_rows(frame: Any) -> list[dict[str, Any]]:
    columns = [
        "row_uid",
        "scenario_id",
        "sample_index",
        "aligned_time_us",
        "view_id",
        *TARGET_COLUMNS,
    ]
    ordered = frame[columns].sort_values(columns, kind="stable")
    return [
        {
            "row_uid": str(row["row_uid"]),
            "scenario_id": str(row["scenario_id"]),
            "sample_index": int(row["sample_index"]),
            "aligned_time_us": int(row["aligned_time_us"]),
            "view_id": str(row["view_id"]),
            "tcp_x_m": float(row["tcp_x_m"]),
            "tcp_y_m": float(row["tcp_y_m"]),
            "tcp_z_m": float(row["tcp_z_m"]),
        }
        for row in ordered.to_dict(orient="records")
    ]


def _context_metric(frame: Any, *, fold_id: str, view_id: str) -> dict[str, Any]:
    selected = frame[frame["view_id"] == view_id]
    scenarios = list(dict.fromkeys(selected["scenario_id"].tolist()))
    if len(selected) < 1 or not scenarios:
        _fail("metric context is empty")
    prediction = selected[
        ["prediction_x_m", "prediction_y_m", "prediction_z_m"]
    ].to_numpy(dtype="float64", copy=True)
    truth = selected[list(TARGET_COLUMNS)].to_numpy(dtype="float64", copy=True)
    return {
        "fold_id": fold_id,
        "scenario_id": scenarios[0] if len(scenarios) == 1 else "+".join(scenarios),
        "view_id": view_id,
        "rows": len(selected),
        "identity_sha256": _canonical_hash(_identity_rows(selected)),
        "rmse3d_m": native_training.fixed_rmse3d_m(
            prediction, truth, name="seed-ensemble context"
        ),
    }


def _task_evaluation_frame(run_fd: int, task: TaskSpec) -> Any:
    columns = [
        "row_uid",
        "scenario_id",
        "sample_index",
        "aligned_time_us",
        "view_id",
        "prediction_x_m",
        "prediction_y_m",
        "prediction_z_m",
        *TARGET_COLUMNS,
    ]
    frame = _read_parquet(
        run_fd, f"results/{task.task_id}/outer_evaluation_rows.parquet", columns
    )
    if (
        list(frame.columns) != columns
        or len(frame) < 5
        or frame.duplicated(["row_uid", "view_id"]).any()
        or list(dict.fromkeys(frame["view_id"].tolist())) != list(VIEW_IDS)
    ):
        _fail("persisted evaluation-row schema/order changed")
    return frame


def _ensemble_frame(run_fd: int, task_group: Sequence[TaskSpec]) -> Any:
    if (
        len(task_group) != 3
        or [task.seed for task in task_group] != list(SEEDS)
        or len({task.fold_id for task in task_group}) != 1
        or len({task.candidate_id for task in task_group}) != 1
    ):
        _fail("seed ensemble task identity changed")
    frames = [_task_evaluation_frame(run_fd, task) for task in task_group]
    identity_columns = [
        "row_uid",
        "scenario_id",
        "sample_index",
        "aligned_time_us",
        "view_id",
        *TARGET_COLUMNS,
    ]
    reference = frames[0][identity_columns].to_numpy()
    if any(
        not bool(np.array_equal(frame[identity_columns].to_numpy(), reference))
        for frame in frames[1:]
    ):
        _fail("seed ensemble outer identity/truth changed")
    output = frames[0][identity_columns].copy()
    for axis in ("x", "y", "z"):
        vectors = [
            frame[f"prediction_{axis}_m"].to_numpy(dtype="float64", copy=True)
            for frame in frames
        ]
        output[f"prediction_{axis}_m"] = (
            (vectors[0] + vectors[1]) + vectors[2]
        ) / 3.0
    output["seed_ensemble_size"] = 3
    return output


def _gate_contrast(
    name: str,
    challenger: Mapping[str, Any],
    comparator: Mapping[str, Any],
) -> dict[str, Any]:
    guard = 1.0e-12
    challenger_derived = challenger["derived_gate_inputs"]
    comparator_derived = comparator["derived_gate_inputs"]
    fold_wins = sum(
        left < right - guard
        for left, right in zip(
            challenger_derived["fold_outage_scores_m"],
            comparator_derived["fold_outage_scores_m"],
            strict=True,
        )
    )
    seed_wins = sum(
        left < right - guard
        for left, right in zip(
            challenger_derived["per_seed_J_m"],
            comparator_derived["per_seed_J_m"],
            strict=True,
        )
    )
    requirements = {
        "robust_J_strict": challenger_derived["robust_J_m"]
        < comparator_derived["robust_J_m"] - guard,
        "worst_mask_strict": challenger_derived["WM_worst_mask_mean_rmse3d_m"]
        < comparator_derived["WM_worst_mask_mean_rmse3d_m"] - guard,
        "fold_outage_strict_wins_at_least_six": fold_wins >= 6,
        "per_seed_J_strict_wins_at_least_two": seed_wins >= 2,
        "full_F_nonharm": challenger_derived["full_F_m"]
        <= comparator_derived["full_F_m"] + guard,
    }
    return {
        "contrast": name,
        "challenger": challenger["candidate_id"],
        "comparator": comparator["candidate_id"],
        "fold_outage_strict_wins": fold_wins,
        "per_seed_J_strict_wins": seed_wins,
        "requirements": requirements,
        "passed": all(requirements.values()),
    }


def aggregate_zero_shift(
    run_fd: int,
    tasks: Sequence[TaskSpec],
    comparator_bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[tuple[str, str], Any]]:
    """Aggregate persisted predictions once; never reopen preparation truth."""

    ensembles: dict[tuple[str, str], Any] = {}
    candidate_records: list[dict[str, Any]] = []
    candidate_primary: dict[str, dict[str, Any]] = {}
    stress_rows: list[dict[str, Any]] = []
    primary_folds = [
        task.fold_id
        for task in tasks
        if task.candidate_ordinal == 0
        and task.seed_ordinal == 0
        and task.fold_family == "scenario_loso"
    ]
    for candidate_id in CANDIDATE_IDS:
        outage_contexts: list[dict[str, Any]] = []
        full_contexts: list[dict[str, Any]] = []
        per_seed_by_seed: dict[int, list[dict[str, Any]]] = {
            seed: [] for seed in SEEDS
        }
        for fold_id in [
            task.fold_id
            for task in tasks
            if task.candidate_ordinal == 0 and task.seed_ordinal == 0
        ]:
            group = [
                task
                for task in tasks
                if task.candidate_id == candidate_id and task.fold_id == fold_id
            ]
            group.sort(key=lambda item: item.seed_ordinal)
            ensemble = _ensemble_frame(run_fd, group)
            ensembles[(candidate_id, fold_id)] = ensemble
            family = group[0].fold_family
            contexts = [
                _context_metric(ensemble, fold_id=fold_id, view_id=view_id)
                for view_id in VIEW_IDS
            ]
            candidate_records.append(
                {
                    "candidate_id": candidate_id,
                    "fold_id": fold_id,
                    "fold_family": family,
                    "contexts": contexts,
                }
            )
            if family == "scenario_loso":
                full_contexts.append(contexts[0])
                outage_contexts.extend(contexts[1:])
                for seed in SEEDS:
                    task = next(item for item in group if item.seed == seed)
                    seed_frame = _task_evaluation_frame(run_fd, task)
                    for view_id in OUTAGE_VIEW_IDS:
                        row = _context_metric(
                            seed_frame, fold_id=fold_id, view_id=view_id
                        )
                        per_seed_by_seed[seed].append({"seed": seed, **row})
            else:
                stress_rows.extend(
                    {"candidate_id": candidate_id, "fold_family": family, **row}
                    for row in contexts
                )
        if [row["fold_id"] for row in outage_contexts[::4]] != primary_folds:
            _fail("primary fold aggregation order changed")
        per_seed_contexts = [
            row for seed in SEEDS for row in per_seed_by_seed[seed]
        ]
        derived = derive_primary_gate_inputs(
            outage_contexts, full_contexts, per_seed_contexts
        )
        candidate_primary[candidate_id] = {
            "candidate_id": candidate_id,
            "primary_seed_ensemble_outage_contexts": outage_contexts,
            "primary_seed_ensemble_full_contexts": full_contexts,
            "primary_per_seed_outage_contexts": per_seed_contexts,
            "derived_gate_inputs": derived,
        }
    comparator_primary = {
        "candidate_id": "receipt_bound_frozen_v1_masked_erm",
        "primary_seed_ensemble_outage_contexts": comparator_bundle[
            "primary_seed_ensemble_outage_contexts"
        ],
        "primary_seed_ensemble_full_contexts": comparator_bundle[
            "primary_seed_ensemble_full_contexts"
        ],
        "primary_per_seed_outage_contexts": comparator_bundle[
            "primary_per_seed_outage_contexts"
        ],
        "derived_gate_inputs": comparator_bundle["derived_gate_inputs"],
    }
    comparator_identity = [
        (row["fold_id"], row["scenario_id"], row["view_id"], row["rows"], row["identity_sha256"])
        for row in comparator_primary["primary_seed_ensemble_outage_contexts"]
    ]
    comparator_full_identity = [
        (row["fold_id"], row["scenario_id"], row["view_id"], row["rows"], row["identity_sha256"])
        for row in comparator_primary["primary_seed_ensemble_full_contexts"]
    ]
    comparator_per_seed_identity = [
        (row["seed"], row["fold_id"], row["scenario_id"], row["view_id"], row["rows"], row["identity_sha256"])
        for row in comparator_primary["primary_per_seed_outage_contexts"]
    ]
    for candidate_id in CANDIDATE_IDS:
        identity = [
            (row["fold_id"], row["scenario_id"], row["view_id"], row["rows"], row["identity_sha256"])
            for row in candidate_primary[candidate_id]["primary_seed_ensemble_outage_contexts"]
        ]
        full_identity = [
            (row["fold_id"], row["scenario_id"], row["view_id"], row["rows"], row["identity_sha256"])
            for row in candidate_primary[candidate_id]["primary_seed_ensemble_full_contexts"]
        ]
        per_seed_identity = [
            (row["seed"], row["fold_id"], row["scenario_id"], row["view_id"], row["rows"], row["identity_sha256"])
            for row in candidate_primary[candidate_id]["primary_per_seed_outage_contexts"]
        ]
        if (
            identity != comparator_identity
            or full_identity != comparator_full_identity
            or per_seed_identity != comparator_per_seed_identity
        ):
            _fail("V2/V1 primary context identity changed")
    contrasts = [
        _gate_contrast(
            "A_vs_frozen_v1_masked_erm",
            candidate_primary[CANDIDATE_IDS[0]],
            comparator_primary,
        ),
        _gate_contrast(
            "B_vs_A",
            candidate_primary[CANDIDATE_IDS[1]],
            candidate_primary[CANDIDATE_IDS[0]],
        ),
        _gate_contrast(
            "C_vs_B",
            candidate_primary[CANDIDATE_IDS[2]],
            candidate_primary[CANDIDATE_IDS[1]],
        ),
    ]
    zero = {
        "contract": "openaxes.external.native_sensor_experts.v2.zero_shift.v1",
        "candidate_order": list(CANDIDATE_IDS),
        "fold_order": [
            task.fold_id
            for task in tasks
            if task.candidate_ordinal == 0 and task.seed_ordinal == 0
        ],
        "candidate_fold_records": candidate_records,
        "candidate_primary": [candidate_primary[item] for item in CANDIDATE_IDS],
        "scientific_fits": 126,
        "outer_model_forward_calls": 126,
        "outer_label_reads": 126,
        "task_evaluator_calls": 126,
    }
    gates = {
        "contract": "openaxes.external.native_sensor_experts.v2.primary_gates.v1",
        "contrast_order": [row["contrast"] for row in contrasts],
        "contrasts": contrasts,
        "all_three_contrasts_pass": all(row["passed"] for row in contrasts),
        "gate_evaluation_count": 1,
    }
    stress = {
        "contract": "openaxes.external.native_sensor_experts.v2.stress_report.v1",
        "families": ["speed_pair_out", "shape_out"],
        "rows": stress_rows,
        "may_select_or_change_primary_gate": False,
    }
    return zero, gates, stress, ensembles


def aggregate_signed_shift_once(
    workspace_fd: int,
    run_fd: int,
    tasks: Sequence[TaskSpec],
    terminal: Mapping[str, Any],
    ensembles: Mapping[tuple[str, str], Any],
) -> dict[str, Any]:
    scenarios = [
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    ]
    truth = _read_preparation_partition(
        workspace_fd,
        terminal,
        scenarios,
        [*METADATA_COLUMNS, *TARGET_COLUMNS],
    )
    lookup: dict[tuple[str, int], tuple[float, float, float]] = {}
    for row in truth.to_dict(orient="records"):
        key = (str(row["scenario_id"]), int(row["aligned_time_us"]))
        if key in lookup:
            _fail("signed truth lookup identity is duplicated")
        lookup[key] = (
            float(row["tcp_x_m"]),
            float(row["tcp_y_m"]),
            float(row["tcp_z_m"]),
        )
    fold_order = [
        task.fold_id
        for task in tasks
        if task.candidate_ordinal == 0 and task.seed_ordinal == 0
    ]
    rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        for fold_id in fold_order:
            frame = ensembles[(candidate_id, fold_id)]
            for shift_ms in (-50, -20, -10, 10, 20, 50):
                shifted_truth: list[tuple[float, float, float]] = []
                for row in frame.to_dict(orient="records"):
                    key = (
                        str(row["scenario_id"]),
                        int(row["aligned_time_us"]) + shift_ms * 1000,
                    )
                    if key not in lookup:
                        _fail("signed-shift reconstruction dropped an edge")
                    shifted_truth.append(lookup[key])
                shifted = np.asarray(shifted_truth, dtype="float64")
                for view_index, view_id in enumerate(VIEW_IDS):
                    selected = frame["view_id"] == view_id
                    prediction = frame.loc[
                        selected,
                        ["prediction_x_m", "prediction_y_m", "prediction_z_m"],
                    ].to_numpy(dtype="float64", copy=True)
                    shifted_view = shifted[np.asarray(selected, dtype="bool")]
                    zero = _context_metric(
                        frame, fold_id=fold_id, view_id=view_id
                    )["rmse3d_m"]
                    metric = native_training.fixed_rmse3d_m(
                        prediction,
                        shifted_view,
                        name="signed-shift context",
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
                            "uid_support_sha256": _canonical_hash(
                                frame.loc[selected, "row_uid"].tolist()
                            ),
                        }
                    )
    if len(rows) != 1260:
        _fail("signed-shift 3 x 14 x 6 x 5 cell grid changed")
    primary_folds = fold_order[:8]
    primary_robust: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        for shift_ms in (-50, -20, -10, 10, 20, 50):
            metrics = [
                row["rmse3d_m"]
                for fold_id in primary_folds
                for view_id in OUTAGE_VIEW_IDS
                for row in rows
                if row["candidate_id"] == candidate_id
                and row["fold_id"] == fold_id
                and row["shift_ms"] == shift_ms
                and row["view_id"] == view_id
            ]
            if len(metrics) != 32:
                _fail("signed primary robust context grid changed")
            fold_scores = [
                native_training.fixed_mean_float64(
                    metrics[offset : offset + 4], name="signed fold outage"
                )
                for offset in range(0, 32, 4)
            ]
            mask_scores = [
                native_training.fixed_mean_float64(
                    metrics[view_index::4], name="signed mask mean"
                )
                for view_index in range(4)
            ]
            primary_robust.append(
                {
                    "candidate_id": candidate_id,
                    "shift_ms": shift_ms,
                    "robust_J_m": native_training.fixed_mean_float64(
                        [
                            native_training.fixed_mean_float64(
                                metrics, name="signed mean context"
                            ),
                            max(fold_scores),
                            max(mask_scores),
                        ],
                        name="signed robust J",
                    ),
                }
            )
    return {
        "contract": "openaxes.external.native_sensor_experts.v2.signed_shift_report.v1",
        "shift_order_ms": [-50, -20, -10, 10, 20, 50],
        "rows": rows,
        "primary_robust_J": primary_robust,
        "candidate_fold_shift_ensemble_cells": 252,
        "candidate_fold_shift_view_metric_cells": 1260,
        "canonical_public_truth_reconstruction_calls": 1,
        "scientific_fits": 0,
        "model_forward_calls": 0,
        "task_evaluator_calls": 0,
        "may_select_or_change_primary_gate": False,
    }


def _write_payload_hashed_json(
    run_fd: int, relative: str, value: Mapping[str, Any]
) -> dict[str, Any]:
    payload = dict(value)
    payload["payload_sha256"] = _canonical_hash(payload)
    _write_json(run_fd, relative, payload)
    return payload


def finalize_openaxes_external_native_v2_run(
    workspace_fd: int,
    run_fd: int,
    tasks: Sequence[TaskSpec],
    comparator: Mapping[str, Any],
) -> None:
    zero, gates, stress, ensembles = aggregate_zero_shift(
        run_fd, tasks, comparator
    )
    terminal, _verified_comparator, _parent_references = _load_parents(run_fd)
    signed = aggregate_signed_shift_once(
        workspace_fd, run_fd, tasks, terminal, ensembles
    )
    reports = {
        "zero_shift.json": zero,
        "primary_gates.json": gates,
        "stress_report.json": stress,
        "signed_shift_report.json": signed,
    }
    report_references: dict[str, dict[str, Any]] = {}
    for name in (
        "zero_shift.json",
        "primary_gates.json",
        "stress_report.json",
        "signed_shift_report.json",
    ):
        _write_payload_hashed_json(run_fd, f"reports/{name}", reports[name])
        report_references[name] = _reference(
            run_fd, f"reports/{name}", name=f"report {name}"
        )
    summary = {
        "contract": "openaxes.external.native_sensor_experts.v2.development_summary.v1",
        "task_count": 126,
        "candidate_count": 3,
        "fold_count": 14,
        "seed_count": 3,
        "optimizer_updates": 50400,
        "all_model_forward_calls": 100926,
        "outer_label_reads": 126,
        "task_evaluator_calls": 126,
        "signed_truth_reconstruction_calls": 1,
        "report_references": report_references,
    }
    _write_payload_hashed_json(
        run_fd, "reports/development_summary.json", summary
    )
    report_references["development_summary.json"] = _reference(
        run_fd, "reports/development_summary.json", name="development summary"
    )
    result_phase_references = [
        _reference(
            run_fd,
            f"runtime/ledgers/{task.task_id}/result.completed.json",
            name="result completed phase",
        )
        for task in tasks
    ]
    development = {
        "contract": "openaxes.external.native_sensor_experts.v2.development_manifest.v1",
        "schema_version": 1,
        "run_id": RUN_ID,
        "canonical_config_sha256": CONFIG_SHA256,
        "task_id_order": [task.task_id for task in tasks],
        "result_completed_phase_references": result_phase_references,
        "result_completed_phase_set_sha256": _canonical_hash(result_phase_references),
        "report_references": {
            name: report_references[name] for name in REPORT_MEMBERS
        },
        "source_set_reference": _reference(
            run_fd, "runtime/source_set.json", name="source set"
        ),
        "quarantine_member_count": 0,
        "closed_task_count": 126,
    }
    development["payload_sha256"] = _canonical_hash(development)
    _write_json(run_fd, "development_manifest.json", development)
    complete = {
        "contract": "openaxes.external.native_sensor_experts.v2.complete.v1",
        "schema_version": 1,
        "run_id": RUN_ID,
        "created_at_utc": _now_utc(),
        "development_manifest_reference": _reference(
            run_fd, "development_manifest.json", name="development manifest"
        ),
        "source_set_reference": development["source_set_reference"],
        "closed_task_count": 126,
        "quarantine_member_count": 0,
        "scientific_values_disclosed": False,
    }
    complete["payload_sha256"] = _canonical_hash(complete)
    _write_json(run_fd, "COMPLETE.json", complete)


def _validate_persisted_evaluation(run_fd: int, task: TaskSpec) -> None:
    evaluation = _strict_json_at(
        run_fd,
        f"results/{task.task_id}/outer_evaluation.json",
        name="outer evaluation",
    )
    _payload_hash(evaluation, "payload_sha256", name="outer evaluation")
    if (
        evaluation.get("task_id") != task.task_id
        or evaluation.get("candidate_id") != task.candidate_id
        or evaluation.get("fold_id") != task.fold_id
        or evaluation.get("fold_family") != task.fold_family
        or evaluation.get("seed") != task.seed
        or evaluation.get("outer_label_reads") != 1
        or evaluation.get("task_evaluator_calls") != 1
    ):
        _fail("outer evaluation identity/accounting changed")
    frame = _task_evaluation_frame(run_fd, task)
    expected_views = [
        _context_metric(frame, fold_id=task.fold_id, view_id=view_id)
        for view_id in VIEW_IDS
    ]
    recorded = evaluation.get("views")
    if type(recorded) is not list or len(recorded) != 5:
        _fail("outer evaluation view roster changed")
    for expected, observed in zip(expected_views, recorded, strict=True):
        if observed != {
            "view_id": expected["view_id"],
            "rows": expected["rows"],
            "rmse3d_m": expected["rmse3d_m"],
        }:
            _fail("persisted outer metric does not recompute from evaluation rows")


def validate_execution_session_receipt(value: Any) -> dict[str, Any]:
    session = _mapping(value, name="execution session")
    _keys(
        session,
        {
            "contract",
            "schema_version",
            "execution_session_id",
            "created_at_utc",
            "reference_roots",
            "canonical_config_sha256",
            "source_set_reference",
            "source_set_sha256",
            "v1_terminal_receipt_reference",
            "v1_terminal_receipt_sha256",
            "comparator_bundle_reference",
            "comparator_bundle_sha256",
            "prepared_manifest_reference",
            "prepared_manifest_sha256",
            "prepared_table_reference",
            "prepared_table_sha256",
            "environment_reference",
            "environment_sha256",
            "gpu_inventory_reference",
            "gpu_inventory_sha256",
            "task_index_reference",
            "task_assignment_sha256",
            "slot_assignments_reference",
            "slot_assignment_sha256",
            "controller_source_sha256",
            "session_sha256",
        },
        name="execution session",
    )
    if (
        session["contract"]
        != "openaxes.external.native_sensor_experts.v2.execution_session.v1"
        or session["schema_version"] != 1
        or type(session["schema_version"]) is not int
        or session["canonical_config_sha256"] != CONFIG_SHA256
    ):
        _fail("execution session identity changed")
    _uuid4(session["execution_session_id"], name="execution session ID")
    roots = _mapping(session["reference_roots"], name="session reference roots")
    _keys(
        roots,
        {
            "workspace_root",
            "project_root",
            "v1_run_root",
            "preparation_run_root",
            "v2_run_root",
        },
        name="session reference roots",
    )
    for key in (
        "source_set_reference",
        "v1_terminal_receipt_reference",
        "comparator_bundle_reference",
        "prepared_manifest_reference",
        "prepared_table_reference",
        "environment_reference",
        "gpu_inventory_reference",
        "task_index_reference",
        "slot_assignments_reference",
    ):
        _validate_reference_shape(session[key], name=f"session {key}")
    for key in (
        "source_set_sha256",
        "v1_terminal_receipt_sha256",
        "comparator_bundle_sha256",
        "prepared_manifest_sha256",
        "prepared_table_sha256",
        "environment_sha256",
        "gpu_inventory_sha256",
        "task_assignment_sha256",
        "slot_assignment_sha256",
        "controller_source_sha256",
    ):
        _sha(session[key], name=f"session {key}")
    _payload_hash(session, "session_sha256", name="execution session")
    return dict(session)


def validate_pre_fit_validation_receipt(value: Any) -> dict[str, Any]:
    prefit = _mapping(value, name="pre-fit validation")
    _keys(
        prefit,
        {
            "contract",
            "schema_version",
            "created_at_utc",
            "execution_session_reference",
            "execution_session_sha256",
            "canonical_config_sha256",
            "source_set_sha256",
            "v1_terminal_receipt_sha256",
            "comparator_bundle_sha256",
            "prepared_manifest_sha256",
            "prepared_table_sha256",
            "environment_sha256",
            "gpu_inventory_sha256",
            "task_assignment_sha256",
            "slot_assignment_sha256",
            "prediction_permit_set_sha256",
            "label_permit_set_sha256",
            "assertions",
            "payload_sha256",
        },
        name="pre-fit validation",
    )
    if (
        prefit["contract"]
        != "openaxes.external.native_sensor_experts.v2.pre_fit_validation.v1"
        or prefit["schema_version"] != 1
        or type(prefit["schema_version"]) is not int
        or prefit["canonical_config_sha256"] != CONFIG_SHA256
    ):
        _fail("pre-fit validation identity changed")
    _validate_reference_shape(
        prefit["execution_session_reference"], name="pre-fit session reference"
    )
    for key in (
        "execution_session_sha256",
        "source_set_sha256",
        "v1_terminal_receipt_sha256",
        "comparator_bundle_sha256",
        "prepared_manifest_sha256",
        "prepared_table_sha256",
        "environment_sha256",
        "gpu_inventory_sha256",
        "task_assignment_sha256",
        "slot_assignment_sha256",
        "prediction_permit_set_sha256",
        "label_permit_set_sha256",
    ):
        _sha(prefit[key], name=f"pre-fit {key}")
    assertions = _mapping(prefit["assertions"], name="pre-fit assertions")
    if set(assertions) != {
        "parent_and_git_lineage_valid",
        "source_and_config_valid",
        "preparation_and_ordered_fold_identity_valid",
        "environment_and_locks_valid",
        "four_gpu_inventory_valid",
        "task_index_and_slot_assignments_valid",
        "exact_126_prediction_permits_valid",
        "exact_126_label_permits_valid",
        "zero_attempt_or_scientific_action_evidence_before_authorization",
        "all_mandatory_synthetic_source_tests_passed",
    } or any(item is not True for item in assertions.values()):
        _fail("pre-fit assertions changed")
    _payload_hash(prefit, "payload_sha256", name="pre-fit validation")
    return dict(prefit)


def _validate_launch_zero_world(
    run_fd: int,
    session: Mapping[str, Any],
    prefit: Mapping[str, Any],
) -> None:
    if (
        _names(run_fd, "runtime/execution_launches") != ["launch_000000"]
        or _names(run_fd, "runtime/execution_launches/launch_000000")
        != ["launch.ready.json", "launch.started.json"]
        or _names(run_fd, "runtime/worker_ready") != ["launch_000000"]
        or _names(run_fd, "runtime/worker_ready/launch_000000")
        != [f"slot_{slot:02d}.json" for slot in range(12)]
    ):
        _fail("launch-zero/controller closed world changed")
    started = _strict_json_at(
        run_fd,
        "runtime/execution_launches/launch_000000/launch.started.json",
        name="launch started",
    )
    _keys(
        started,
        {
            "contract",
            "schema_version",
            "execution_session_id",
            "launch_id",
            "launch_ordinal",
            "resume_mode",
            "created_at_utc",
            "execution_session_reference",
            "pre_fit_validation_reference",
            "canonical_config_sha256",
            "controller_source_sha256",
            "environment_sha256",
            "source_set_sha256",
            "gpu_inventory_sha256",
            "task_assignment_sha256",
            "slot_assignment_sha256",
            "comparator_bundle_sha256",
            "recovery_snapshot_reference",
            "payload_sha256",
        },
        name="launch started",
    )
    if (
        started["contract"]
        != "openaxes.external.native_sensor_experts.v2.launch_started.v1"
        or started["schema_version"] != 1
        or type(started["schema_version"]) is not int
        or started["execution_session_id"] != session["execution_session_id"]
        or started["launch_id"] != "launch_000000"
        or started["launch_ordinal"] != 0
        or type(started["launch_ordinal"]) is not int
        or started["resume_mode"] != "initial"
        or started["recovery_snapshot_reference"] is not None
    ):
        _fail("launch-zero identity changed")
    _payload_hash(started, "payload_sha256", name="launch started")
    started_reference = _reference(
        run_fd,
        "runtime/execution_launches/launch_000000/launch.started.json",
        name="launch started",
    )
    controller_references: list[dict[str, Any]] = []
    for slot in range(12):
        path = f"runtime/worker_ready/launch_000000/slot_{slot:02d}.json"
        controller = _strict_json_at(run_fd, path, name="controller ready")
        _keys(
            controller,
            {
                "contract",
                "schema_version",
                "execution_session_id",
                "launch_id",
                "launch_ordinal",
                "transport_slot_index",
                "local_slot",
                "controller_uuid",
                "controller_pid",
                "controller_process_start_time_ticks",
                "gpu_reference",
                "cuda_visible_devices",
                "torch_imported",
                "execution_session_reference",
                "launch_started_reference",
                "controller_source_sha256",
                "environment_sha256",
                "source_set_sha256",
                "task_assignment_sha256",
                "slot_assignment_sha256",
                "pre_fit_validation_sha256",
                "created_at_utc",
                "payload_sha256",
            },
            name="controller ready",
        )
        if (
            controller.get("contract")
            != "openaxes.external.native_sensor_experts.v2.controller_ready.v1"
            or controller.get("execution_session_id") != session["execution_session_id"]
            or controller.get("launch_id") != "launch_000000"
            or controller.get("launch_ordinal") != 0
            or controller.get("transport_slot_index") != slot
            or controller.get("local_slot") != slot // 4
            or controller.get("gpu_reference") != dict(GPU_INVENTORY[slot % 4])
            or controller.get("cuda_visible_devices") != GPU_INVENTORY[slot % 4]["uuid"]
            or controller.get("torch_imported") is not False
            or controller.get("launch_started_reference") != started_reference
            or controller.get("pre_fit_validation_sha256") != prefit["payload_sha256"]
        ):
            _fail("controller-ready slot/GPU lineage changed")
        _uuid4(controller.get("controller_uuid"), name="controller UUID")
        _payload_hash(controller, "payload_sha256", name="controller ready")
        controller_references.append(_reference(run_fd, path, name="controller ready"))
    ready = _strict_json_at(
        run_fd,
        "runtime/execution_launches/launch_000000/launch.ready.json",
        name="launch ready",
    )
    _keys(
        ready,
        {
            "contract",
            "schema_version",
            "execution_session_id",
            "launch_id",
            "launch_ordinal",
            "launch_started_reference",
            "ordered_controller_ready_references",
            "controller_ready_set_sha256",
            "payload_sha256",
        },
        name="launch ready",
    )
    if (
        ready.get("contract")
        != "openaxes.external.native_sensor_experts.v2.launch_ready.v1"
        or ready.get("schema_version") != 1
        or type(ready.get("schema_version")) is not int
        or ready.get("execution_session_id") != session["execution_session_id"]
        or ready.get("launch_id") != "launch_000000"
        or ready.get("launch_ordinal") != 0
        or ready.get("launch_started_reference") != started_reference
        or ready.get("ordered_controller_ready_references")
        != controller_references
        or ready.get("controller_ready_set_sha256")
        != _canonical_hash(controller_references)
    ):
        _fail("launch-ready controller set changed")
    _payload_hash(ready, "payload_sha256", name="launch ready")


def _validate_child_ready_world(
    run_fd: int,
    tasks: Sequence[TaskSpec],
    slot_assignments: Mapping[str, Any],
) -> None:
    launches = _names(run_fd, "runtime/child_ready")
    if launches != ["launch_000000"]:
        _fail("child-ready launch roster changed")
    names = _names(run_fd, "runtime/child_ready/launch_000000")
    if len(names) != 126:
        _fail("fresh child-ready count is not exact 126")
    task_by_id = {task.task_id: task for task in tasks}
    seen: set[str] = set()
    slot_counts = [0] * 12
    gpu_counts = [0] * 4
    exact_keys = {
        "contract",
        "schema_version",
        "execution_session_id",
        "launch_id",
        "launch_ordinal",
        "task_id",
        "task_spec_sha256",
        "attempt_id",
        "child_role",
        "slot_child_ordinal",
        "child_uuid",
        "child_pid",
        "child_process_start_time_ticks",
        "parent_controller_uuid",
        "parent_controller_pid",
        "transport_slot_index",
        "local_slot",
        "gpu_reference",
        "cuda_visible_devices",
        "torch_imported_after_visible_device_binding",
        "torch_cuda_is_available",
        "torch_cuda_device_count",
        "torch_cuda_current_device",
        "observed_current_device_uuid",
        "observed_current_device_pci_bus_id",
        "observed_current_device_name",
        "cuda_identity_probe",
        "launch_ready_reference",
        "controller_ready_reference",
        "controller_source_sha256",
        "environment_sha256",
        "source_set_sha256",
        "task_assignment_sha256",
        "slot_assignment_sha256",
        "comparator_bundle_sha256",
        "pre_fit_validation_sha256",
        "created_at_utc",
        "payload_sha256",
    }
    for name in names:
        if not name.endswith(".json"):
            _fail("child-ready member name changed")
        value = _strict_json_at(
            run_fd, f"runtime/child_ready/launch_000000/{name}", name="child ready"
        )
        _keys(value, exact_keys, name="child ready")
        _payload_hash(value, "payload_sha256", name="child ready")
        task_id = _text(value["task_id"], name="child task")
        if task_id not in task_by_id or task_id in seen:
            _fail("child-ready task universe is duplicated or unknown")
        task = task_by_id[task_id]
        slot = task.context_ordinal % 12
        gpu = dict(GPU_INVENTORY[slot % 4])
        _uuid4(value["execution_session_id"], name="child session")
        _uuid4(value["attempt_id"], name="child attempt")
        child_uuid = _uuid4(value["child_uuid"], name="child UUID")
        _uuid4(value["parent_controller_uuid"], name="controller UUID")
        if (
            name != f"{child_uuid}.json"
            or value["contract"] != "openaxes.external.native_sensor_experts.v2.child_ready.v1"
            or value["schema_version"] != 1
            or type(value["schema_version"]) is not int
            or value["launch_id"] != "launch_000000"
            or value["launch_ordinal"] != 0
            or type(value["launch_ordinal"]) is not int
            or value["task_spec_sha256"] != task.task_spec_sha256
            or value["child_role"] != "fresh_fit_full_task"
            or value["transport_slot_index"] != slot
            or value["local_slot"] != slot // 4
            or value["gpu_reference"] != gpu
            or value["cuda_visible_devices"] != gpu["uuid"]
            or value["torch_imported_after_visible_device_binding"] is not True
            or value["torch_cuda_is_available"] is not True
            or value["torch_cuda_device_count"] != 1
            or value["torch_cuda_current_device"] != 0
            or value["observed_current_device_uuid"] != gpu["uuid"]
            or value["observed_current_device_pci_bus_id"] != gpu["pci_bus_id"].upper()
            or value["observed_current_device_name"] != gpu["name"]
            or value["cuda_identity_probe"]
            != {
                "backend": "torch.cuda",
                "logical_device_index": 0,
                "visible_device_count": 1,
                "uuid": gpu["uuid"],
                "pci_bus_id": gpu["pci_bus_id"].upper(),
                "name": gpu["name"],
            }
            or value["slot_assignment_sha256"]
            != slot_assignments["slot_assignment_sha256"]
        ):
            _fail("child-ready lineage/GPU identity changed")
        _verify_reference(
            run_fd,
            value["launch_ready_reference"],
            expected_path="runtime/execution_launches/launch_000000/launch.ready.json",
            name="child launch ready",
        )
        _verify_reference(
            run_fd,
            value["controller_ready_reference"],
            expected_path=f"runtime/worker_ready/launch_000000/slot_{slot:02d}.json",
            name="child controller ready",
        )
        seen.add(task_id)
        slot_counts[slot] += 1
        gpu_counts[slot % 4] += 1
    if (
        seen != set(task_by_id)
        or tuple(slot_counts) != FIT_COUNTS_BY_SLOT
        or tuple(gpu_counts) != FIT_COUNTS_BY_GPU
    ):
        _fail("child-ready exact 126/slot/GPU accounting changed")


def validate_openaxes_external_native_v2_run(run_fd: int) -> dict[str, Any]:
    """Validate a completed run without reopening preparation/source truth."""

    if _names(run_fd) != sorted(ROOT_MEMBERS):
        _fail("completed native-V2 root member set changed")
    _terminal, comparator, _parent_references = _load_parents(run_fd)
    prepared_manifest = _strict_json_at(
        run_fd, "prepared_manifest.json", name="prepared manifest snapshot"
    )
    tasks = build_native_v2_task_grid(prepared_manifest)
    if _names(run_fd, "tasks") != [f"task_{ordinal:03d}.json" for ordinal in range(126)]:
        _fail("task-spec member set changed")
    task_references: dict[str, dict[str, Any]] = {}
    for task in tasks:
        path = f"tasks/{task.task_id}.json"
        verify_task_payload(_strict_json_at(run_fd, path, name="task spec"), task)
        task_references[task.task_id] = _reference(run_fd, path, name="task spec")
    task_index = validate_task_index_receipt(
        _strict_json_at(run_fd, "task_index.json", name="task index"), tasks
    )
    if task_index["task_spec_references"] != task_references:
        _fail("task index raw task-spec references changed")
    if _names(run_fd, "runtime") != sorted(RUNTIME_MEMBERS):
        _fail("runtime closed-world member set changed")
    slot_assignments = validate_slot_assignments_receipt(
        _strict_json_at(
            run_fd, "runtime/slot_assignments.json", name="slot assignments"
        ),
        tasks,
        task_index["task_assignment_sha256"],
    )
    session = validate_execution_session_receipt(
        _strict_json_at(
            run_fd, "runtime/execution_session.json", name="execution session"
        )
    )
    prefit = validate_pre_fit_validation_receipt(
        _strict_json_at(
            run_fd, "runtime/pre_fit_validation.json", name="pre-fit validation"
        )
    )
    _validate_launch_zero_world(run_fd, session, prefit)
    if (
        _names(run_fd, "runtime/outer_prediction_permits")
        or _names(run_fd, "runtime/outer_label_permits")
        or _names(run_fd, "quarantine")
    ):
        _fail("completed permits/quarantine are not empty")
    expected_task_files = [f"task_{ordinal:03d}.json" for ordinal in range(126)]
    for relative in (
        "runtime/attempt_roster",
        "runtime/attempt_ledger",
        "runtime/outer_prediction_claims",
        "runtime/outer_label_claims",
    ):
        if _names(run_fd, relative) != expected_task_files:
            _fail("task authority file universe changed")
    if _names(run_fd, "results") != [task.task_id for task in tasks]:
        _fail("result-directory universe changed")
    if _names(run_fd, "runtime/ledgers") != [task.task_id for task in tasks]:
        _fail("task-ledger universe changed")
    training_controls: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if _names(run_fd, f"results/{task.task_id}") != sorted(RESULT_MEMBERS):
            _fail("completed task result member set changed")
        if _names(run_fd, f"runtime/ledgers/{task.task_id}") != sorted(
            f"{phase}.json" for phase in PHASES
        ):
            _fail("task seven-phase ledger member set changed")
        phases = [
            _strict_json_at(
                run_fd,
                f"runtime/ledgers/{task.task_id}/{phase}.json",
                name="phase receipt",
            )
            for phase in PHASES
        ]
        validate_phase_chain(phases, task)
        for phase_receipt in phases:
            for key, reference in phase_receipt["payload"].items():
                if key.endswith("_reference"):
                    verified = _validate_reference_shape(
                        reference, name=f"phase {key}"
                    )
                    _verify_reference(
                        run_fd,
                        verified,
                        expected_path=verified["path"],
                        name=f"phase {key}",
                    )
        _validate_persisted_evaluation(run_fd, task)
        training_receipt = _strict_json_at(
            run_fd,
            f"results/{task.task_id}/training_receipt.json",
            name="training receipt",
        )
        _payload_hash(training_receipt, "payload_sha256", name="training receipt")
        if (
            training_receipt.get("task_id") != task.task_id
            or training_receipt.get("candidate_id") != task.candidate_id
            or training_receipt.get("seed") != task.seed
            or training_receipt.get("updates") != 400
        ):
            _fail("training receipt identity/update count changed")
        for key in (
            "initial_state_sha256",
            "checkpoint_state_sha256",
            "optimizer_initial_state_sha256",
        ):
            _sha(training_receipt.get(key), name=f"training {key}")
        training_controls[task.task_id] = training_receipt
        result = _strict_json_at(
            run_fd, f"results/{task.task_id}/result.json", name="task result"
        )
        _payload_hash(result, "payload_sha256", name="task result")
        if result.get("task_id") != task.task_id:
            _fail("task result identity changed")
    for context_ordinal in range(42):
        group = tasks[context_ordinal * 3 : context_ordinal * 3 + 3]
        controls = [training_controls[task.task_id] for task in group]
        if controls[0]["initial_state_sha256"] != comparator[
            "initial_state_by_seed"
        ][str(group[0].seed)]:
            _fail("candidate-A/V1 initial state pairing changed")
        if (
            controls[1]["initial_state_sha256"]
            != controls[2]["initial_state_sha256"]
            or controls[1]["optimizer_initial_state_sha256"]
            != controls[2]["optimizer_initial_state_sha256"]
        ):
            _fail("candidate-B/C paired initial state changed")
    _validate_child_ready_world(run_fd, tasks, slot_assignments)
    if _names(run_fd, "reports") != sorted(REPORT_MEMBERS):
        _fail("report closed-world member set changed")
    for name in REPORT_MEMBERS:
        report = _strict_json_at(run_fd, f"reports/{name}", name=f"report {name}")
        _payload_hash(report, "payload_sha256", name=f"report {name}")
    development = _strict_json_at(
        run_fd, "development_manifest.json", name="development manifest"
    )
    _payload_hash(development, "payload_sha256", name="development manifest")
    if (
        development.get("task_id_order") != [task.task_id for task in tasks]
        or development.get("closed_task_count") != 126
        or development.get("quarantine_member_count") != 0
    ):
        _fail("development manifest closed task roster changed")
    complete = _strict_json_at(run_fd, "COMPLETE.json", name="complete marker")
    _payload_hash(complete, "payload_sha256", name="complete marker")
    if (
        complete.get("contract")
        != "openaxes.external.native_sensor_experts.v2.complete.v1"
        or complete.get("run_id") != RUN_ID
        or complete.get("closed_task_count") != 126
        or complete.get("quarantine_member_count") != 0
        or complete.get("scientific_values_disclosed") is not False
    ):
        _fail("complete marker is not outcome-neutral exact-126 authority")
    _verify_reference(
        run_fd,
        complete["development_manifest_reference"],
        expected_path="development_manifest.json",
        name="complete development manifest",
    )
    validate_comparator_bundle(comparator)
    return {
        "contract": NATIVE_CONTRACT,
        "run_id": RUN_ID,
        "closed_task_count": 126,
        "quarantine_member_count": 0,
        "valid": True,
    }


def run_openaxes_external_native_v2(
    workspace_fd: int,
    run_fd: int,
    dependency_fd: int,
    *,
    source_audit_evidence_git_commit: str,
) -> None:
    prepared = prepare_openaxes_external_native_v2_run(
        workspace_fd,
        run_fd,
        source_audit_evidence_git_commit=source_audit_evidence_git_commit,
    )
    launch = _write_launch_zero(
        run_fd,
        prepared["tasks"],
        prepared["session"],
        prepared["prefit"],
    )
    snapshot_fd = _snapshot_fd_from_module()
    try:
        _run_fresh_task_children(
            workspace_fd,
            run_fd,
            snapshot_fd,
            dependency_fd,
            prepared["tasks"],
            prepared["session"],
            prepared["prefit"],
            launch,
        )
    finally:
        os.close(snapshot_fd)
    finalize_openaxes_external_native_v2_run(
        workspace_fd,
        run_fd,
        prepared["tasks"],
        prepared["comparator"],
    )
    validate_openaxes_external_native_v2_run(run_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate"):
        command = subparsers.add_parser(name)
        command.add_argument("--workspace-root", required=True)
        command.add_argument("--run-dir", required=True)
        command.add_argument("--dependency-root", required=True)
        command.add_argument("--source-audit-evidence-git-commit", required=True)
    internal = subparsers.add_parser("internal-fit")
    internal.add_argument("--workspace-root", required=True)
    internal.add_argument("--run-dir", required=True)
    internal.add_argument("--dependency-root", required=True)
    internal.add_argument("--task-id", required=True)
    internal.add_argument("--attempt-id", required=True)
    internal.add_argument("--slot-child-ordinal", required=True, type=int)
    internal.add_argument("--parent-controller-uuid", required=True)
    internal.add_argument("--controller-ready-path", required=True)
    internal.add_argument("--launch-ready-path", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    workspace_fd = _fd_from_argument(
        arguments.workspace_root, name="workspace root"
    )
    run_fd = _fd_from_argument(arguments.run_dir, name="run root")
    dependency_fd = _fd_from_argument(
        arguments.dependency_root, name="dependency root"
    )
    dependency_info = os.fstat(dependency_fd)
    if dependency_fd < 3 or not stat.S_ISDIR(dependency_info.st_mode):
        _fail("dependency root is not a retained directory descriptor")
    try:
        if arguments.command == "run":
            run_openaxes_external_native_v2(
                workspace_fd,
                run_fd,
                dependency_fd,
                source_audit_evidence_git_commit=arguments.source_audit_evidence_git_commit,
            )
        elif arguments.command == "validate":
            if re.fullmatch(HEX40, arguments.source_audit_evidence_git_commit) is None:
                _fail("validation evidence commit is malformed")
            validate_openaxes_external_native_v2_run(run_fd)
        elif arguments.command == "internal-fit":
            _uuid4(arguments.attempt_id, name="internal fit attempt")
            _uuid4(
                arguments.parent_controller_uuid,
                name="internal fit controller UUID",
            )
            if arguments.slot_child_ordinal < 0:
                _fail("internal fit child ordinal is negative")
            return _run_one_fresh_task(
                workspace_fd,
                run_fd,
                task_id=arguments.task_id,
                attempt_id=arguments.attempt_id,
                slot_child_ordinal=arguments.slot_child_ordinal,
                parent_controller_uuid=arguments.parent_controller_uuid,
                controller_ready_path=arguments.controller_ready_path,
                launch_ready_path=arguments.launch_ready_path,
            )
        else:
            _fail("unknown native-V2 command")
    finally:
        os.close(workspace_fd)
        os.close(run_fd)
        os.close(dependency_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
