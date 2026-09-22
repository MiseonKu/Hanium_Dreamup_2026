"""Read-only OpenAXES public-data preparation for an external benchmark.

The implementation authenticates and streams the official archive.  It never
extracts the archive, imports its Python files, executes its notebook, or uses
its publisher-generated TCP reconstructions.  Only raw IMU CSVs, raw robot TCP
truth, and the publisher-provided fixed aligned robot timestamps are consumed.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterator, Mapping, Sequence
import unicodedata
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy

from .io_utils import sha256_file, write_json, write_parquet
from .openaxes_external_v1_config import (
    OPENAXES_EXTERNAL_V1_CONFIG_SHA256,
    OPENAXES_EXTERNAL_V1_CONTRACT,
    OPENAXES_SCENARIOS,
    OPENAXES_SEGMENTS,
    OPENAXES_SENSITIVITY_SIGNED_SHIFTS_MS,
    model_feature_columns,
    openaxes_config_sha256,
    scenario_configs,
    validate_openaxes_config,
)


PREPARED_PARQUET = "prepared.parquet"
MANIFEST_JSON = "manifest.json"
COMPLETE_JSON = "COMPLETE.json"
PREPARED_FILES = frozenset({PREPARED_PARQUET, MANIFEST_JSON, COMPLETE_JSON})
_STRICT_INT = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_CONTRACT_SOURCE_RELATIVE_PATHS = (
    "configs/openaxes_external_v1.yaml",
    "src/imu_vive_learning/openaxes_external_v1_config.py",
    "src/imu_vive_learning/openaxes_external_v1.py",
    "src/imu_vive_learning/openaxes_external_v1_cli.py",
    "tests/test_openaxes_external_v1.py",
    "docs/OPENAXES_EXTERNAL_V1_CONTRACT.md",
    "pyproject.toml",
)
OPENAXES_SIGNED_SHIFT_TRUTH_V1_CONTRACT = (
    "openaxes.external.signed_shift_truth.v1"
)
OPENAXES_SIGNED_SHIFT_TRUTH_V1_COLUMNS = (
    "row_uid",
    "scenario_id",
    "base_aligned_time_us",
    "shift_ms",
    "shifted_target_time_us",
    "shifted_tcp_x_m",
    "shifted_tcp_y_m",
    "shifted_tcp_z_m",
)


class OpenAxesExternalV1Error(RuntimeError):
    """Raised when source or prepared evidence violates the frozen contract."""


@dataclass(frozen=True)
class _IMUStream:
    raw_time_us: np.ndarray
    fitted_time_us: np.ndarray
    index: np.ndarray
    corrected_features: np.ndarray
    receipt: dict[str, Any]


@dataclass(frozen=True)
class _RobotStream:
    timestamp_s: np.ndarray
    values: np.ndarray

    @property
    def tcp_xyz_m(self) -> np.ndarray:
        return self.values[:, 1:4]


@dataclass
class _BoundSource:
    path: Path
    handle: BinaryIO
    archive: zipfile.ZipFile
    identity: dict[str, int]
    receipt: dict[str, Any]


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise OpenAxesExternalV1Error(f"non-finite JSON constant: {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if type(key) is not str or key in result:
                raise OpenAxesExternalV1Error(
                    f"duplicate or non-string JSON key: {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpenAxesExternalV1Error(f"cannot decode strict JSON: {path}") from exc


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
        raise OpenAxesExternalV1Error("payload is not finite canonical JSON") from exc


def _strict_int(value: str, *, field: str) -> int:
    if not _STRICT_INT.fullmatch(value):
        raise OpenAxesExternalV1Error(f"{field} is not a canonical integer")
    return int(value)


def _strict_float(value: str, *, field: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise OpenAxesExternalV1Error(f"{field} is not numeric") from exc
    if not math.isfinite(result):
        raise OpenAxesExternalV1Error(f"{field} is not finite")
    return result


def _safe_member_name(name: str) -> bool:
    """Accept one canonical POSIX archive name, never a platform alias."""

    if (
        not name
        or "\\" in name
        or ":" in name
        or "\x00" in name
        or name.startswith("/")
        or PureWindowsPath(name).drive
    ):
        return False
    directory = name.endswith("/")
    core = name[:-1] if directory else name
    if not core or "//" in core:
        return False
    raw_parts = core.split("/")
    if any(
        part in {"", ".", ".."}
        or part != unicodedata.normalize("NFC", part)
        or part.rstrip(" .") != part
        or any(ord(character) < 32 for character in part)
        for part in raw_parts
    ):
        return False
    path = PurePosixPath(core)
    canonical = "/".join(path.parts) + ("/" if directory else "")
    return not path.is_absolute() and name == canonical


def _member_alias_key(name: str) -> str:
    if not _safe_member_name(name):
        raise OpenAxesExternalV1Error(f"unsafe archive member path: {name!r}")
    core = name[:-1] if name.endswith("/") else name
    return "/".join(part.casefold() for part in core.split("/"))


def _file_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _require_same_file_identity(
    expected: Mapping[str, int], observed: os.stat_result, *, stage: str
) -> dict[str, int]:
    actual = _file_identity(observed)
    if actual != dict(expected) or not stat.S_ISREG(observed.st_mode):
        raise OpenAxesExternalV1Error(
            f"OpenAXES source identity changed during {stage}"
        )
    return actual


def _sha256_handle(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


@contextmanager
def _open_authenticated_source(
    source: str | Path,
    *,
    expected_filename: str,
    expected_size: int,
    expected_sha256: str,
    expected_entries: int,
    expected_payload_bytes: int,
) -> Iterator[_BoundSource]:
    """Bind every source operation to one no-follow regular-file descriptor."""

    path = Path(os.path.abspath(os.fspath(source)))
    if path.name != expected_filename:
        raise OpenAxesExternalV1Error(
            f"OpenAXES source filename mismatch: {path.name!r}"
        )
    if not hasattr(os, "O_NOFOLLOW"):
        raise OpenAxesExternalV1Error("O_NOFOLLOW is required for OpenAXES source")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OpenAxesExternalV1Error(
            f"cannot open no-follow OpenAXES source: {path}"
        ) from exc
    try:
        handle = os.fdopen(descriptor, "rb", closefd=True)
    except Exception:
        os.close(descriptor)
        raise

    with handle:
        before_stat = os.fstat(handle.fileno())
        before = _file_identity(before_stat)
        if not stat.S_ISREG(before_stat.st_mode):
            raise OpenAxesExternalV1Error(
                "OpenAXES source must be one regular no-follow file"
            )
        if before["size_bytes"] != expected_size:
            raise OpenAxesExternalV1Error(
                "OpenAXES source size mismatch: "
                f"{before['size_bytes']} != {expected_size}"
            )
        actual_sha = _sha256_handle(handle)
        after_sha = _require_same_file_identity(
            before, os.fstat(handle.fileno()), stage="same-handle SHA-256"
        )
        if actual_sha != expected_sha256:
            raise OpenAxesExternalV1Error("OpenAXES source SHA-256 mismatch")
        try:
            with zipfile.ZipFile(handle) as archive:
                infos = archive.infolist()
                if len(infos) != expected_entries:
                    raise OpenAxesExternalV1Error(
                        "OpenAXES ZIP entry count mismatch"
                    )
                if sum(info.file_size for info in infos) != expected_payload_bytes:
                    raise OpenAxesExternalV1Error(
                        "OpenAXES ZIP payload size mismatch"
                    )
                names: set[str] = set()
                aliases: dict[str, str] = {}
                for info in infos:
                    name = info.filename
                    if name in names:
                        raise OpenAxesExternalV1Error(
                            f"OpenAXES ZIP duplicate member: {name!r}"
                        )
                    names.add(name)
                    try:
                        alias = _member_alias_key(name)
                    except OpenAxesExternalV1Error as exc:
                        raise OpenAxesExternalV1Error(
                            f"unsafe OpenAXES ZIP member: {name!r}"
                        ) from exc
                    previous = aliases.get(alias)
                    if previous is not None:
                        raise OpenAxesExternalV1Error(
                            "OpenAXES ZIP canonical alias collision: "
                            f"{previous!r} and {name!r}"
                        )
                    aliases[alias] = name
                    mode = (info.external_attr >> 16) & 0xFFFF
                    file_type = stat.S_IFMT(mode)
                    if stat.S_ISLNK(mode) or file_type not in {
                        0,
                        stat.S_IFREG,
                        stat.S_IFDIR,
                    }:
                        raise OpenAxesExternalV1Error(
                            f"unsafe OpenAXES ZIP member type: {name!r}"
                        )
                    if bool(info.flag_bits & 0x1):
                        raise OpenAxesExternalV1Error(
                            f"encrypted OpenAXES ZIP member: {name!r}"
                        )
                bad = archive.testzip()
                if bad is not None:
                    raise OpenAxesExternalV1Error(
                        f"OpenAXES ZIP CRC failure: {bad!r}"
                    )
                after_crc = _require_same_file_identity(
                    before,
                    os.fstat(handle.fileno()),
                    stage="same-handle ZIP CRC verification",
                )
                receipt: dict[str, Any] = {
                    "filename": path.name,
                    "size_bytes": before["size_bytes"],
                    "sha256": actual_sha,
                    "zip_entries": expected_entries,
                    "zip_payload_bytes": expected_payload_bytes,
                    "crc_verified": True,
                    "unsafe_paths": 0,
                    "canonical_alias_collisions": 0,
                    "duplicates": 0,
                    "encrypted_members": 0,
                    "single_no_follow_fd_bound": True,
                    "same_handle_size_sha_crc_and_reads": True,
                    "file_identity": {
                        "before": before,
                        "after_sha256": after_sha,
                        "after_crc": after_crc,
                    },
                }
                bound = _BoundSource(
                    path=path,
                    handle=handle,
                    archive=archive,
                    identity=before,
                    receipt=receipt,
                )
                try:
                    yield bound
                finally:
                    post_read = _require_same_file_identity(
                        before,
                        os.fstat(handle.fileno()),
                        stage="same-handle archive reads",
                    )
                    try:
                        path_stat = os.stat(path, follow_symlinks=False)
                    except OSError as exc:
                        raise OpenAxesExternalV1Error(
                            "OpenAXES source path was removed or replaced"
                        ) from exc
                    post_path = _require_same_file_identity(
                        before, path_stat, stage="post-read path binding"
                    )
                    receipt["file_identity"]["after_reads"] = post_read
                    receipt["file_identity"]["post_path_binding"] = post_path
                    receipt["pre_post_fstat_stable"] = True
                    receipt["post_path_binding_verified"] = True
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            if isinstance(exc, OpenAxesExternalV1Error):
                raise
            raise OpenAxesExternalV1Error(
                "cannot authenticate bound OpenAXES ZIP"
            ) from exc


def _audit_zip_envelope(
    source: Path,
    *,
    expected_filename: str,
    expected_size: int,
    expected_sha256: str,
    expected_entries: int,
    expected_payload_bytes: int,
) -> dict[str, Any]:
    """Authenticate a ZIP without reopening its pathname for any data read."""

    with _open_authenticated_source(
        source,
        expected_filename=expected_filename,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        expected_entries=expected_entries,
        expected_payload_bytes=expected_payload_bytes,
    ) as bound:
        receipt = bound.receipt
    return deepcopy(receipt)


def _csv_rows(payload: bytes, *, member: str, expected_header: Sequence[str]) -> list[list[str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OpenAxesExternalV1Error(f"non-UTF8 CSV member: {member}") from exc
    if "\x00" in text:
        raise OpenAxesExternalV1Error(f"NUL byte in CSV member: {member}")
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise OpenAxesExternalV1Error(f"empty CSV member: {member}") from exc
    if tuple(header) != tuple(expected_header):
        raise OpenAxesExternalV1Error(f"CSV header drift: {member}")
    rows: list[list[str]] = []
    for row_number, row in enumerate(reader, start=2):
        if len(row) != len(header):
            raise OpenAxesExternalV1Error(
                f"CSV width drift: {member}:{row_number}"
            )
        rows.append(row)
    if not rows:
        raise OpenAxesExternalV1Error(f"CSV has no data rows: {member}")
    return rows


def _robust_affine_time(
    index: np.ndarray,
    time_us: np.ndarray,
    *,
    iterations: int,
    slope_min: float,
    slope_max: float,
    residual_max: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit deterministic Huber IRLS to TIME~INDEX without any robot label."""

    if index.ndim != 1 or time_us.ndim != 1 or len(index) != len(time_us):
        raise OpenAxesExternalV1Error("invalid IMU clock arrays")
    x = index.astype(np.float64) - float(index[0])
    y = time_us.astype(np.float64) - float(time_us[0])
    design = np.column_stack((x, np.ones(len(x), dtype=np.float64)))
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    for _ in range(iterations):
        residual = y - design @ coefficient
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        scale = max(1.4826 * mad, 1.0)
        cutoff = 1.345 * scale
        absolute = np.abs(residual - center)
        weights = np.ones_like(absolute)
        outside = absolute > cutoff
        weights[outside] = cutoff / absolute[outside]
        root = np.sqrt(weights)
        coefficient, *_ = np.linalg.lstsq(
            design * root[:, None], y * root, rcond=None
        )
    slope, intercept = (float(coefficient[0]), float(coefficient[1]))
    fitted_relative = design @ coefficient
    residual = y - fitted_relative
    if not slope_min <= slope <= slope_max:
        raise OpenAxesExternalV1Error(f"IMU clock slope outside contract: {slope}")
    if float(np.max(np.abs(residual))) > residual_max:
        raise OpenAxesExternalV1Error("IMU robust clock residual exceeds contract")
    fitted = float(time_us[0]) + fitted_relative
    if not np.all(np.diff(fitted) > 0):
        raise OpenAxesExternalV1Error("fitted IMU clock is not strictly monotonic")
    quantiles = np.percentile(np.abs(residual), [50, 95, 99, 100])
    return fitted, {
        "method": "deterministic_huber_irls_time_on_index",
        "slope_us_per_index": slope,
        "fitted_time_at_first_index_us": float(time_us[0]) + intercept,
        "absolute_residual_us": {
            "p50": float(quantiles[0]),
            "p95": float(quantiles[1]),
            "p99": float(quantiles[2]),
            "max": float(quantiles[3]),
        },
    }


def _correct_quiet_prefix_gyro(
    features: np.ndarray,
    *,
    prefix_samples: int,
    max_axis_std_rad_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract a label-free quiet-prefix gyro mean from a 9-axis stream."""

    if (
        features.ndim != 2
        or features.shape[1] != 9
        or not np.isfinite(features).all()
        or type(prefix_samples) is not int
        or not 2 <= prefix_samples <= len(features)
        or type(max_axis_std_rad_s) is not float
        or not math.isfinite(max_axis_std_rad_s)
        or max_axis_std_rad_s <= 0.0
    ):
        raise OpenAxesExternalV1Error("invalid quiet-prefix gyro inputs")
    gyro = features[:, 6:9]
    quiet_mean = np.mean(gyro[:prefix_samples], axis=0)
    quiet_std = np.std(gyro[:prefix_samples], axis=0)
    if float(np.max(quiet_std)) > max_axis_std_rad_s:
        raise OpenAxesExternalV1Error("quiet gyro prefix is not quiet")
    corrected = features.copy()
    corrected[:, 6:9] -= quiet_mean
    return corrected, quiet_mean, quiet_std


def _read_imu_stream(
    archive: zipfile.ZipFile,
    *,
    scenario: Mapping[str, Any],
    segment: str,
    config: Mapping[str, Any],
) -> _IMUStream:
    scenario_id = str(scenario["id"])
    member = f"measure_raw-2022-09-15/imu-{scenario_id}/{segment}.csv"
    schema = config["schema"]
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise OpenAxesExternalV1Error(f"missing raw IMU member: {member}") from exc
    rows = _csv_rows(payload, member=member, expected_header=schema["imu_header"])
    expected_rows = scenario["imu_rows"][segment]
    if len(rows) != expected_rows:
        raise OpenAxesExternalV1Error(f"raw IMU row count mismatch: {member}")

    binding = schema["sensor_bindings"][segment]
    time_us = np.empty(len(rows), dtype=np.int64)
    index = np.empty(len(rows), dtype=np.int64)
    features = np.empty((len(rows), 9), dtype=np.float64)
    quaternion_nonzero = 0
    for row_index, row in enumerate(rows):
        if row[0] != binding["mac"] or row[1] != binding["imu_id"]:
            raise OpenAxesExternalV1Error(f"sensor identity drift: {member}")
        time_us[row_index] = _strict_int(row[2], field=f"{member}:TIME")
        index[row_index] = _strict_int(row[3], field=f"{member}:INDEX")
        quaternion = [
            _strict_float(value, field=f"{member}:quaternion") for value in row[4:8]
        ]
        quaternion_nonzero += int(any(value != 0.0 for value in quaternion))
        features[row_index] = [
            _strict_float(value, field=f"{member}:sensor") for value in row[8:17]
        ]
    if quaternion_nonzero != 0:
        raise OpenAxesExternalV1Error(
            f"raw quaternion columns are no longer all zero: {member}"
        )
    if not np.all(np.diff(time_us) > 0):
        raise OpenAxesExternalV1Error(f"raw IMU TIME is not strictly monotonic: {member}")
    if not np.all(np.diff(index) == 1):
        raise OpenAxesExternalV1Error(f"raw IMU INDEX has a gap: {member}")
    if not np.isfinite(features).all():
        raise OpenAxesExternalV1Error(f"raw IMU values are non-finite: {member}")

    prep = config["preprocessing"]
    prefix = prep["quiet_prefix_samples"]
    try:
        corrected, quiet_mean, quiet_std = _correct_quiet_prefix_gyro(
            features,
            prefix_samples=prefix,
            max_axis_std_rad_s=prep["quiet_prefix_max_axis_std_rad_s"],
        )
    except OpenAxesExternalV1Error as exc:
        raise OpenAxesExternalV1Error(
            f"quiet gyro prefix is invalid: {scenario_id}/{segment}"
        ) from exc
    fitted, clock_receipt = _robust_affine_time(
        index,
        time_us,
        iterations=prep["imu_clock_fit_iterations"],
        slope_min=prep["imu_clock_slope_min_us_per_index"],
        slope_max=prep["imu_clock_slope_max_us_per_index"],
        residual_max=prep["imu_clock_residual_max_us"],
    )
    return _IMUStream(
        raw_time_us=time_us,
        fitted_time_us=fitted,
        index=index,
        corrected_features=corrected,
        receipt={
            "rows": len(rows),
            "index_first": int(index[0]),
            "index_last": int(index[-1]),
            "raw_time_first_us": int(time_us[0]),
            "raw_time_last_us": int(time_us[-1]),
            "quaternion_nonzero_rows": quaternion_nonzero,
            "quiet_prefix_samples": prefix,
            "quiet_gyro_axis_mean_rad_s": [float(value) for value in quiet_mean],
            "quiet_gyro_axis_std_rad_s": [float(value) for value in quiet_std],
            "clock_fit": clock_receipt,
        },
    )


def _read_robot_stream(
    archive: zipfile.ZipFile,
    *,
    scenario: Mapping[str, Any],
    config: Mapping[str, Any],
) -> _RobotStream:
    scenario_id = str(scenario["id"])
    member = f"measure_raw-2022-09-15/robot/robot-{scenario_id}.csv"
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise OpenAxesExternalV1Error(f"missing robot member: {member}") from exc
    rows = _csv_rows(
        payload,
        member=member,
        expected_header=config["schema"]["robot_header"],
    )
    if len(rows) != scenario["robot_rows"]:
        raise OpenAxesExternalV1Error(f"robot row count mismatch: {member}")
    values = np.asarray(
        [
            [_strict_float(value, field=f"{member}:numeric") for value in row]
            for row in rows
        ],
        dtype=np.float64,
    )
    timestamp = values[:, 0]
    if not np.isfinite(values).all() or not np.all(np.diff(timestamp) > 0):
        raise OpenAxesExternalV1Error(f"robot values are invalid: {member}")
    if not np.allclose(np.diff(timestamp), 0.008, rtol=0.0, atol=1e-9):
        raise OpenAxesExternalV1Error(f"robot clock is not exact 125 Hz: {member}")
    return _RobotStream(timestamp_s=timestamp, values=values)


def _read_aligned_payload(
    payload: bytes,
    *,
    member: str,
    expected_header: Sequence[str],
    expected_rows: int,
) -> np.ndarray:
    rows = _csv_rows(payload, member=member, expected_header=expected_header)
    if len(rows) != expected_rows:
        raise OpenAxesExternalV1Error(f"aligned robot row count mismatch: {member}")
    time_us = np.empty(len(rows), dtype=np.int64)
    for row_index, row in enumerate(rows):
        time_us[row_index] = _strict_int(row[0], field=f"{member}:TIME")
        quaternion = [
            _strict_float(value, field=f"{member}:quaternion") for value in row[1:]
        ]
        if not all(math.isfinite(value) for value in quaternion):
            raise OpenAxesExternalV1Error(f"aligned quaternion is non-finite: {member}")
    if not np.all(np.diff(time_us) > 0):
        raise OpenAxesExternalV1Error(f"aligned TIME is not monotonic: {member}")
    return time_us


def _scan_nested_alignment(
    archive: zipfile.ZipFile,
    *,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = config["source"]
    wanted: dict[str, tuple[str, int]] = {}
    for scenario in scenario_configs(config):
        for segment in OPENAXES_SEGMENTS:
            member = (
                f"measure_raw-2022-09-15/robot/robot-{scenario['id']}/"
                f"{segment}-aligned.csv"
            )
            wanted[member] = (str(scenario["id"]), int(scenario["robot_rows"]))
    captured: dict[str, list[np.ndarray]] = {name: [] for name in OPENAXES_SCENARIOS}
    names: set[str] = set()
    aliases: dict[str, str] = {}
    entry_count = 0
    payload_bytes = 0
    try:
        with archive.open(source["nested_tar_member"]) as compressed:
            with tarfile.open(fileobj=compressed, mode="r|bz2") as nested:
                for member in nested:
                    entry_count += 1
                    payload_bytes += int(member.size)
                    if member.name in names:
                        raise OpenAxesExternalV1Error(
                            f"nested TAR duplicate member: {member.name!r}"
                        )
                    names.add(member.name)
                    try:
                        alias = _member_alias_key(member.name)
                    except OpenAxesExternalV1Error as exc:
                        raise OpenAxesExternalV1Error(
                            f"unsafe nested TAR member: {member.name!r}"
                        ) from exc
                    previous = aliases.get(alias)
                    if previous is not None:
                        raise OpenAxesExternalV1Error(
                            "nested TAR canonical alias collision: "
                            f"{previous!r} and {member.name!r}"
                        )
                    aliases[alias] = member.name
                    if (
                        member.issym()
                        or member.islnk()
                        or not (member.isfile() or member.isdir())
                    ):
                        raise OpenAxesExternalV1Error(
                            f"unsafe nested TAR member: {member.name!r}"
                        )
                    wanted_entry = wanted.get(member.name)
                    if wanted_entry is None:
                        continue
                    if not member.isfile():
                        raise OpenAxesExternalV1Error(
                            f"aligned member is not a file: {member.name}"
                        )
                    handle = nested.extractfile(member)
                    if handle is None:
                        raise OpenAxesExternalV1Error(
                            f"cannot stream aligned member: {member.name}"
                        )
                    scenario_id, rows = wanted_entry
                    captured[scenario_id].append(
                        _read_aligned_payload(
                            handle.read(),
                            member=member.name,
                            expected_header=config["schema"]["aligned_header"],
                            expected_rows=rows,
                        )
                    )
    except (KeyError, OSError, EOFError, tarfile.TarError) as exc:
        if isinstance(exc, OpenAxesExternalV1Error):
            raise
        raise OpenAxesExternalV1Error("cannot audit nested OpenAXES TAR") from exc
    if entry_count != source["expected_nested_tar_entries"]:
        raise OpenAxesExternalV1Error("nested TAR entry count mismatch")
    if payload_bytes != source["expected_nested_tar_payload_bytes"]:
        raise OpenAxesExternalV1Error("nested TAR payload size mismatch")
    if sum(len(value) for value in captured.values()) != len(wanted):
        raise OpenAxesExternalV1Error("nested TAR aligned member set is incomplete")
    aligned: dict[str, np.ndarray] = {}
    for scenario_id, arrays in captured.items():
        if len(arrays) != len(OPENAXES_SEGMENTS):
            raise OpenAxesExternalV1Error(
                f"aligned segment count mismatch: {scenario_id}"
            )
        reference = arrays[0]
        if any(not np.array_equal(reference, candidate) for candidate in arrays[1:]):
            raise OpenAxesExternalV1Error(
                f"aligned segment clocks disagree: {scenario_id}"
            )
        aligned[scenario_id] = reference
    return aligned, {
        "member": source["nested_tar_member"],
        "entries": entry_count,
        "payload_bytes": payload_bytes,
        "unsafe_paths_or_links": 0,
        "canonical_alias_collisions": 0,
        "duplicates": 0,
        "aligned_csv_members_consumed": len(wanted),
        "external_code_executed": False,
        "archive_extracted": False,
    }


def _validate_alignment_affine(
    robot: _RobotStream,
    aligned_time_us: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    alignment = config["alignment"]
    x = robot.timestamp_s - robot.timestamp_s[0]
    y = aligned_time_us.astype(np.float64) - float(aligned_time_us[0])
    design = np.column_stack((x, np.ones(len(x), dtype=np.float64)))
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope = float(coefficient[0])
    residual = y - design @ coefficient
    maximum = float(np.max(np.abs(residual)))
    if not alignment["aligned_clock_slope_min_us_per_robot_s"] <= slope <= alignment[
        "aligned_clock_slope_max_us_per_robot_s"
    ]:
        raise OpenAxesExternalV1Error("publisher aligned clock slope is invalid")
    if maximum > alignment["aligned_clock_fit_residual_max_us"]:
        raise OpenAxesExternalV1Error("publisher aligned clock is not affine")
    return {
        "label_assisted": True,
        "fixed_before_model_results": True,
        "source": alignment["source"],
        "slope_us_per_robot_s": slope,
        "fit_absolute_residual_max_us": maximum,
        "aligned_time_first_us": int(aligned_time_us[0]),
        "aligned_time_last_us": int(aligned_time_us[-1]),
    }


def _point_to_segment_distance(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    direction = end - start
    denominator = float(direction @ direction)
    if denominator <= 0.0:
        raise OpenAxesExternalV1Error("triangle geometry has a zero-length edge")
    position = np.clip(((points - start) @ direction) / denominator, 0.0, 1.0)
    projection = start + position[:, None] * direction
    return np.linalg.norm(points - projection, axis=1)


def _validate_active_geometry(
    robot_xyz_m: np.ndarray,
    *,
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    active = scenario["active_trace"]
    start = active["start_robot_row_inclusive"]
    end = active["end_robot_row_inclusive"]
    selected = robot_xyz_m[start : end + 1]
    geometry = active["geometry"]
    plane_error = np.abs(selected[:, 0] - float(geometry["plane_x_m"]))
    plane_max = float(np.max(plane_error))
    if plane_max > float(geometry["plane_tolerance_m"]):
        raise OpenAxesExternalV1Error(
            f"active trace leaves frozen plane: {scenario['id']}"
        )
    if geometry["kind"] == "circle_yz":
        center = np.asarray(
            [geometry["center_y_m"], geometry["center_z_m"]], dtype=np.float64
        )
        radius_error = np.abs(
            np.linalg.norm(selected[:, 1:3] - center, axis=1)
            - float(geometry["radius_m"])
        )
        path_max = float(np.max(radius_error))
    elif geometry["kind"] == "triangle_yz":
        vertices = np.asarray(geometry["vertices_yz_m"], dtype=np.float64)
        if vertices.shape != (3, 2) or not np.isfinite(vertices).all():
            raise OpenAxesExternalV1Error("invalid frozen triangle geometry")
        distances = np.vstack(
            [
                _point_to_segment_distance(
                    selected[:, 1:3], vertices[index], vertices[(index + 1) % 3]
                )
                for index in range(3)
            ]
        )
        path_max = float(np.max(np.min(distances, axis=0)))
    else:  # guarded by config validation
        raise OpenAxesExternalV1Error("unsupported active-trace geometry")
    if path_max > float(geometry["path_tolerance_m"]):
        raise OpenAxesExternalV1Error(
            f"active trace leaves frozen path: {scenario['id']}"
        )
    return {
        "source": "frozen_robot_truth_geometry_before_model_results",
        "start_robot_row_inclusive": int(start),
        "end_robot_row_inclusive": int(end),
        "source_robot_rows": int(end - start + 1),
        "geometry": deepcopy(dict(geometry)),
        "observed_plane_error_max_m": plane_max,
        "observed_path_error_max_m": path_max,
    }


def _declared_folds() -> list[dict[str, Any]]:
    scenarios = list(OPENAXES_SCENARIOS)
    folds: list[dict[str, Any]] = []
    for test in scenarios:
        folds.append(
            {
                "fold_id": f"scenario_loso::{test}",
                "family": "scenario_loso",
                "role": "primary",
                "train_scenarios": [item for item in scenarios if item != test],
                "test_scenarios": [test],
                "inner_selection": "scenario_blocked_training_groups_only",
                "outer_test_fitting_allowed": False,
            }
        )
    for speed in (100, 150, 200, 250):
        test = [f"circle-{speed}", f"triangle-{speed}"]
        folds.append(
            {
                "fold_id": f"speed_pair_out::{speed}",
                "family": "speed_pair_out",
                "role": "stress",
                "train_scenarios": [item for item in scenarios if item not in test],
                "test_scenarios": test,
                "inner_selection": "scenario_blocked_training_groups_only",
                "outer_test_fitting_allowed": False,
            }
        )
    for shape in ("circle", "triangle"):
        test = [item for item in scenarios if item.startswith(f"{shape}-")]
        folds.append(
            {
                "fold_id": f"shape_out::{shape}",
                "family": "shape_out",
                "role": "stress",
                "train_scenarios": [item for item in scenarios if item not in test],
                "test_scenarios": test,
                "inner_selection": "scenario_blocked_training_groups_only",
                "outer_test_fitting_allowed": False,
            }
        )
    if len(folds) != 14:
        raise AssertionError("internal OpenAXES fold construction drift")
    return folds


def _prepare_scenario(
    archive: zipfile.ZipFile,
    *,
    scenario: Mapping[str, Any],
    aligned_time_us: np.ndarray,
    config: Mapping[str, Any],
    source_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scenario_id = str(scenario["id"])
    robot = _read_robot_stream(archive, scenario=scenario, config=config)
    if len(aligned_time_us) != len(robot.timestamp_s):
        raise OpenAxesExternalV1Error(f"aligned/robot row mismatch: {scenario_id}")
    alignment_receipt = _validate_alignment_affine(
        robot, aligned_time_us, config=config
    )
    imu = {
        segment: _read_imu_stream(
            archive, scenario=scenario, segment=segment, config=config
        )
        for segment in OPENAXES_SEGMENTS
    }

    raw_common_start = max(stream.raw_time_us[0] for stream in imu.values())
    raw_common_end = min(stream.raw_time_us[-1] for stream in imu.values())
    common = (aligned_time_us >= raw_common_start) & (
        aligned_time_us <= raw_common_end
    )
    common_indices = np.flatnonzero(common)
    if not len(common_indices) or not np.all(
        np.diff(common_indices) == 1
    ):
        raise OpenAxesExternalV1Error(
            f"aligned robot/common IMU overlap is empty or fragmented: {scenario_id}"
        )
    dropped_head = int(common_indices[0])
    dropped_tail = int(len(common) - common_indices[-1] - 1)
    if (
        int(common.sum()) != scenario["common_aligned_robot_rows"]
        or dropped_head != scenario["dropped_robot_head_rows"]
        or dropped_tail != scenario["dropped_robot_tail_rows"]
    ):
        raise OpenAxesExternalV1Error(
            f"frozen common-overlap accounting changed: {scenario_id}"
        )

    active_receipt = _validate_active_geometry(
        robot.tcp_xyz_m, scenario=scenario
    )
    active_start = int(
        aligned_time_us[scenario["active_trace"]["start_robot_row_inclusive"]]
    )
    active_end = int(
        aligned_time_us[scenario["active_trace"]["end_robot_row_inclusive"]]
    )
    fitted_common_start = max(stream.fitted_time_us[0] for stream in imu.values())
    fitted_common_end = min(stream.fitted_time_us[-1] for stream in imu.values())
    if active_start < fitted_common_start or active_end > fitted_common_end:
        raise OpenAxesExternalV1Error(
            f"active trace is outside robust four-IMU common time: {scenario_id}"
        )
    interval_us = int(round(1_000_000 / config["preprocessing"]["grid_hz"]))
    grid_start = int(math.ceil(active_start / interval_us) * interval_us)
    grid_end = int(math.floor(active_end / interval_us) * interval_us)
    if grid_end <= grid_start:
        raise OpenAxesExternalV1Error(f"empty 100 Hz active grid: {scenario_id}")
    grid = np.arange(grid_start, grid_end + 1, interval_us, dtype=np.int64)
    if grid[0] < fitted_common_start or grid[-1] > fitted_common_end:
        raise OpenAxesExternalV1Error(f"100 Hz grid would extrapolate IMU: {scenario_id}")
    if grid[0] < aligned_time_us[0] or grid[-1] > aligned_time_us[-1]:
        raise OpenAxesExternalV1Error(f"100 Hz grid would extrapolate target: {scenario_id}")

    feature_parts = [
        np.column_stack(
            [
                np.interp(
                    grid.astype(np.float64),
                    imu[segment].fitted_time_us,
                    imu[segment].corrected_features[:, channel],
                )
                for channel in range(9)
            ]
        )
        for segment in OPENAXES_SEGMENTS
    ]
    features = np.column_stack(feature_parts)
    targets = np.column_stack(
        [
            np.interp(
                grid.astype(np.float64),
                aligned_time_us.astype(np.float64),
                robot.tcp_xyz_m[:, axis],
            )
            for axis in range(3)
        ]
    )
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise OpenAxesExternalV1Error(f"prepared values are non-finite: {scenario_id}")

    feature_names = model_feature_columns(config)
    target_names = tuple(config["schema"]["target_columns"])
    frame = pd.DataFrame(features, columns=feature_names)
    for target_index, target_name in enumerate(target_names):
        frame[target_name] = targets[:, target_index]
    frame.insert(0, "aligned_time_us", grid)
    frame.insert(0, "sample_index", np.arange(len(frame), dtype=np.int64))
    frame.insert(0, "scenario_id", scenario_id)
    frame.insert(
        0,
        "row_uid",
        [
            "openaxes_row_"
            + hashlib.sha256(
                (
                    f"{OPENAXES_EXTERNAL_V1_CONTRACT}|{source_sha256}|"
                    f"{scenario_id}|{int(time_us)}"
                ).encode("utf-8")
            ).hexdigest()
            for time_us in grid
        ],
    )
    expected_columns = (
        tuple(config["schema"]["metadata_columns"])
        + feature_names
        + target_names
    )
    if tuple(frame.columns) != expected_columns:
        raise OpenAxesExternalV1Error("prepared column order drift")

    receipt = {
        "scenario_id": scenario_id,
        "shape": scenario["shape"],
        "nominal_speed_mm_s": scenario["nominal_speed_mm_s"],
        "calibration_truth_complete": bool(
            scenario.get("calibration_truth_complete", True)
        ),
        "source_rows": {
            "robot": len(robot.timestamp_s),
            "imu": {segment: imu[segment].receipt["rows"] for segment in OPENAXES_SEGMENTS},
        },
        "common_overlap": {
            "raw_four_imu_start_us": int(raw_common_start),
            "raw_four_imu_end_us": int(raw_common_end),
            "aligned_robot_rows_inside": int(common.sum()),
            "aligned_robot_rows_dropped_head": dropped_head,
            "aligned_robot_rows_dropped_tail": dropped_tail,
        },
        "publisher_alignment": alignment_receipt,
        "imu_streams": {segment: imu[segment].receipt for segment in OPENAXES_SEGMENTS},
        "active_trace": {
            **active_receipt,
            "aligned_start_us": active_start,
            "aligned_end_us": active_end,
        },
        "prepared": {
            "rows": len(frame),
            "grid_hz": config["preprocessing"]["grid_hz"],
            "grid_start_us": int(grid[0]),
            "grid_end_us": int(grid[-1]),
            "strictly_monotonic": bool(np.all(np.diff(grid) == interval_us)),
            "finite_features": True,
            "finite_targets": True,
        },
    }
    return frame, receipt


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _run_git(repo: Path, arguments: Sequence[str]) -> bytes:
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except OSError as exc:
        raise OpenAxesExternalV1Error("git is required for provenance sealing") from exc
    if result.returncode != 0:
        raise OpenAxesExternalV1Error(
            "cannot capture required git provenance: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    return result.stdout


def _decode_git_path(value: bytes) -> str:
    try:
        decoded = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OpenAxesExternalV1Error("git path is not canonical UTF-8") from exc
    if not decoded or "\x00" in decoded or "\\" in decoded:
        raise OpenAxesExternalV1Error("git path is not canonical POSIX text")
    return decoded


def _parse_git_porcelain_z(payload: bytes) -> list[dict[str, str]]:
    parts = payload.split(b"\x00")
    if parts[-1] != b"":
        raise OpenAxesExternalV1Error("git porcelain snapshot is not NUL terminated")
    entries: list[dict[str, str]] = []
    index = 0
    while index < len(parts) - 1:
        item = parts[index]
        index += 1
        if len(item) < 4 or item[2:3] != b" ":
            raise OpenAxesExternalV1Error("git porcelain snapshot is malformed")
        try:
            status_code = item[:2].decode("ascii")
        except UnicodeDecodeError as exc:
            raise OpenAxesExternalV1Error("git status code is not ASCII") from exc
        entry = {"status": status_code, "path": _decode_git_path(item[3:])}
        if "R" in status_code or "C" in status_code:
            if index >= len(parts) - 1:
                raise OpenAxesExternalV1Error(
                    "git rename/copy snapshot is incomplete"
                )
            entry["source_path"] = _decode_git_path(parts[index])
            index += 1
        entries.append(entry)
    return entries


def _blas_receipt(package: Any) -> dict[str, Any]:
    config = getattr(package.__config__, "CONFIG", {})
    dependencies = config.get("Build Dependencies", {})
    result: dict[str, Any] = {}
    for kind in ("blas", "lapack"):
        source = dependencies.get(kind, {})
        result[kind] = {
            key: source.get(key)
            for key in (
                "name",
                "found",
                "version",
                "detection method",
                "openblas configuration",
                "has ilp64",
            )
        }
    return result


def _runtime_receipt() -> dict[str, Any]:
    arrow_runtime = pa.runtime_info()
    uname = platform.uname()
    return {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "cache_tag": sys.implementation.cache_tag,
            "byteorder": sys.byteorder,
        },
        "platform": {
            "system": uname.system,
            "node": uname.node,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
            "libc": list(platform.libc_ver()),
        },
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "pyarrow_cpp": pa.cpp_version,
            "scipy": scipy.__version__,
        },
        "pyarrow_runtime": {
            "simd_level": arrow_runtime.simd_level,
            "detected_simd_level": arrow_runtime.detected_simd_level,
        },
        "blas_runtime": {
            "numpy": _blas_receipt(np),
            "scipy": _blas_receipt(scipy),
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def _provenance_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    config_path = Path(str(config.get("_config_path", "")))
    if not config_path.is_absolute():
        raise OpenAxesExternalV1Error(
            "OpenAXES config path must be absolute for provenance sealing"
        )
    workspace = config_path.parent.parent
    expected_config = workspace / _CONTRACT_SOURCE_RELATIVE_PATHS[0]
    if config_path != expected_config:
        raise OpenAxesExternalV1Error(
            "OpenAXES config must be loaded from its canonical workspace path"
        )
    source_files: dict[str, dict[str, Any]] = {}
    for relative in _CONTRACT_SOURCE_RELATIVE_PATHS:
        path = workspace / relative
        if path.is_symlink() or not path.is_file():
            raise OpenAxesExternalV1Error(
                f"contract/lock source is missing or non-regular: {relative}"
            )
        source_files[relative] = {
            "bytes": path.stat(follow_symlinks=False).st_size,
            "sha256": sha256_file(path),
        }

    root_text = _run_git(workspace, ["rev-parse", "--show-toplevel"])
    try:
        repo = Path(root_text.decode("utf-8").strip())
    except UnicodeDecodeError as exc:
        raise OpenAxesExternalV1Error("git root is not UTF-8") from exc
    if not repo.is_absolute() or not repo.is_dir():
        raise OpenAxesExternalV1Error("git root provenance is invalid")
    head = _run_git(repo, ["rev-parse", "HEAD"]).decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", head):
        raise OpenAxesExternalV1Error("git HEAD provenance is invalid")

    repo_paths = {
        relative: (workspace / relative).relative_to(repo).as_posix()
        for relative in _CONTRACT_SOURCE_RELATIVE_PATHS
    }
    status_payload = _run_git(
        repo, ["status", "--porcelain=v1", "-z", "--untracked-files=all"]
    )
    dirty_entries = _parse_git_porcelain_z(status_payload)
    dirty_paths = sorted(
        {
            path
            for entry in dirty_entries
            for path in (entry["path"], entry.get("source_path"))
            if path is not None
        }
    )
    tracked_payload = _run_git(
        repo,
        ["ls-files", "-z", "--", *repo_paths.values()],
    )
    tracked_paths = {
        _decode_git_path(item)
        for item in tracked_payload.split(b"\x00")
        if item
    }
    required_repo_paths = set(repo_paths.values())
    contract_sources_committed = required_repo_paths.issubset(tracked_paths) and not (
        required_repo_paths.intersection(dirty_paths)
    )
    repository = {
        "git_head": head,
        "dirty": bool(dirty_entries),
        "dirty_path_snapshot": dirty_entries,
        "dirty_path_snapshot_sha256": hashlib.sha256(status_payload).hexdigest(),
        "contract_source_repo_paths": repo_paths,
        "tracked_contract_source_repo_paths": sorted(
            required_repo_paths.intersection(tracked_paths)
        ),
        "fixed_git_commands_only": True,
    }
    stage = {
        "name": "pre_model_result_contract_capture",
        "preparation_only": True,
        "contract_sources_committed_at_git_head": contract_sources_committed,
        "downstream_model_results_blocked_by_uncommitted_contract": (
            not contract_sources_committed
        ),
        "commit_required_before_model_results": True,
        "required_action": (
            "commit_contract_sources_then_reprepare_before_any_model_result"
        ),
    }
    return {
        "contract_and_lock_sources": {
            "files": source_files,
            "dependency_lock_source": "pyproject.toml",
            "exact_dependency_lock_file_present": False,
            "source_set_sha256": _canonical_sha256(source_files),
        },
        "repository": repository,
        "runtime": _runtime_receipt(),
        "stage": stage,
    }


def _complete_payload(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    parquet_path: Path,
) -> dict[str, Any]:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise OpenAxesExternalV1Error("manifest provenance receipt is missing")
    source_lock = provenance.get("contract_and_lock_sources")
    repository = provenance.get("repository")
    runtime = provenance.get("runtime")
    stage = provenance.get("stage")
    if not all(
        isinstance(value, Mapping)
        for value in (source_lock, repository, runtime, stage)
    ):
        raise OpenAxesExternalV1Error("manifest provenance receipt is malformed")
    files = source_lock.get("files")
    if not isinstance(files, Mapping):
        raise OpenAxesExternalV1Error("contract source hash receipt is malformed")
    return {
        "contract": OPENAXES_EXTERNAL_V1_CONTRACT,
        "config_sha256": OPENAXES_EXTERNAL_V1_CONFIG_SHA256,
        "source_sha256": config["source"]["expected_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "prepared_parquet_sha256": sha256_file(parquet_path),
        "contract_source_hashes": {
            path: receipt["sha256"] for path, receipt in files.items()
        },
        "contract_source_set_sha256": source_lock.get("source_set_sha256"),
        "git_head": repository.get("git_head"),
        "dirty_path_snapshot_sha256": repository.get(
            "dirty_path_snapshot_sha256"
        ),
        "runtime_receipt_sha256": _canonical_sha256(runtime),
        "stage": deepcopy(dict(stage)),
        "files": [PREPARED_PARQUET, MANIFEST_JSON],
        "immutable": True,
    }


def _prepare_in_memory(
    source_path: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_openaxes_config(config)
    source_config = config["source"]
    with _open_authenticated_source(
        source_path,
        expected_filename=source_config["filename"],
        expected_size=source_config["expected_size_bytes"],
        expected_sha256=source_config["expected_sha256"],
        expected_entries=source_config["expected_zip_entries"],
        expected_payload_bytes=source_config["expected_zip_payload_bytes"],
    ) as bound:
        archive = bound.archive
        source_receipt = bound.receipt
        try:
            license_text = archive.read("LICENSE.txt").decode("utf-8")
            readme_text = archive.read("README.md").decode("utf-8")
        except (KeyError, UnicodeDecodeError) as exc:
            raise OpenAxesExternalV1Error(
                "OpenAXES README/LICENSE member is missing or invalid"
            ) from exc
        if (
            "Files: measure_raw-2022-09-15/* measure_raw-2022-09-15.tar.bz2"
            not in license_text
            or "License: CC-BY-SA-4.0" not in license_text
        ):
            raise OpenAxesExternalV1Error(
                "OpenAXES member-level data license text changed"
            )
        normalized_readme = readme_text.replace("\u00a0", " ")
        if (
            "This results in a total of 8 different scenarios" not in normalized_readme
            or "100 Hz sample rate" not in normalized_readme
            or "not necessarily scientifically sound evidence" not in normalized_readme
        ):
            raise OpenAxesExternalV1Error("OpenAXES README evidence boundary changed")
        if source_config["portal_license"] == source_config["member_data_license"]:
            raise OpenAxesExternalV1Error("expected OpenAXES license conflict disappeared")
        aligned, nested_receipt = _scan_nested_alignment(archive, config=config)
        frames: list[pd.DataFrame] = []
        scenario_receipts: list[dict[str, Any]] = []
        for scenario in scenario_configs(config):
            frame, receipt = _prepare_scenario(
                archive,
                scenario=scenario,
                aligned_time_us=aligned[str(scenario["id"])],
                config=config,
                source_sha256=source_receipt["sha256"],
            )
            frames.append(frame)
            scenario_receipts.append(receipt)

    if not source_receipt.get("pre_post_fstat_stable"):
        raise OpenAxesExternalV1Error("bound OpenAXES source stability was not sealed")

    prepared = pd.concat(frames, ignore_index=True)
    if prepared["row_uid"].duplicated().any():
        raise OpenAxesExternalV1Error("prepared OpenAXES row_uid collision")
    for scenario_id, group in prepared.groupby("scenario_id", sort=False):
        if not np.all(np.diff(group["aligned_time_us"].to_numpy()) == 10_000):
            raise OpenAxesExternalV1Error(
                f"prepared time is not exact 100 Hz: {scenario_id}"
            )
    total_robot = sum(item["source_rows"]["robot"] for item in scenario_receipts)
    total_common = sum(
        item["common_overlap"]["aligned_robot_rows_inside"] for item in scenario_receipts
    )
    total_unmatched = sum(
        item["common_overlap"]["aligned_robot_rows_dropped_head"]
        + item["common_overlap"]["aligned_robot_rows_dropped_tail"]
        for item in scenario_receipts
    )
    if (total_robot, total_common, total_unmatched) != (39_793, 39_301, 492):
        raise OpenAxesExternalV1Error("global OpenAXES source accounting changed")

    features = list(model_feature_columns(config))
    forbidden = set(config["schema"]["forbidden_model_inputs"])
    if forbidden.intersection(features):
        raise OpenAxesExternalV1Error("forbidden value entered model feature schema")
    folds = _declared_folds()
    provenance = _provenance_receipt(config)
    manifest = {
        "contract": OPENAXES_EXTERNAL_V1_CONTRACT,
        "config_sha256": openaxes_config_sha256(config),
        "scope": {
            "preparation_only": True,
            "model_training_performed": False,
            "model_selection_performed": False,
            "challenge_access_performed": False,
            "so101_data_merged": False,
            "external_code_executed": False,
            "archive_extracted": False,
        },
        "source": {
            **source_receipt,
            "url": source_config["url"],
            "doi": source_config["doi"],
            "nested_tar": nested_receipt,
        },
        "license": {
            "portal": source_config["portal_license"],
            "member_data": source_config["member_data_license"],
            "conflict": True,
            "redistribution_allowed_by_this_contract": False,
            "attribution_required": True,
            "publication_action": "resolve_license_conflict_before_redistribution",
        },
        "schema": {
            "metadata_columns_not_model_inputs": list(
                config["schema"]["metadata_columns"]
            ),
            "model_feature_columns": features,
            "model_feature_count": len(features),
            "sensor_groups": {
                segment: [name for name in features if name.startswith(f"{segment}_")]
                for segment in OPENAXES_SEGMENTS
            },
            "target_columns": list(config["schema"]["target_columns"]),
            "target_frame": "UR5_robot_base_TCP",
            "target_unit": "m",
            "forbidden_model_inputs": list(config["schema"]["forbidden_model_inputs"]),
            "raw_quaternion_columns_excluded_all_zero": list(
                config["schema"]["raw_quaternion_columns"]
            ),
        },
        "preprocessing": {
            "grid_hz": 100,
            "active_trace_only": True,
            "imu_clock_fit": config["preprocessing"]["imu_clock_fit"],
            "gyro_correction": config["preprocessing"]["gyro_correction"],
            "allow_extrapolation": False,
            "publisher_alignment_label_assisted": True,
            "per_model_or_prediction_realign_allowed": False,
            "held_out_full_recording_clock_fit_is_offline_transductive": True,
            "held_out_full_recording_clock_fit_is_acausal": True,
            "sensitivity_recipe": deepcopy(
                dict(config["alignment"]["sensitivity_recipe"])
            ),
            "signed_shift_truth_api": {
                "contract": OPENAXES_SIGNED_SHIFT_TRUTH_V1_CONTRACT,
                "function": "reconstruct_openaxes_signed_shift_truth_v1",
                "closed_schema": list(OPENAXES_SIGNED_SHIFT_TRUTH_V1_COLUMNS),
                "returns": "dataframe_and_deterministic_receipt",
                "artifact_written": False,
                "semantic_validator_required_before_model_use": True,
            },
        },
        "counts": {
            "scenarios": 8,
            "physical_recording_sessions": 1,
            "effective_scenario_clusters": 8,
            "raw_imu_csv_files": 32,
            "raw_imu_rows": 166_106,
            "raw_robot_csv_files": 8,
            "raw_robot_rows": total_robot,
            "aligned_robot_rows_inside_four_imu_common": total_common,
            "aligned_robot_rows_unmatched": total_unmatched,
            "prepared_active_rows": len(prepared),
            "declared_outer_folds": len(folds),
        },
        "scenarios": scenario_receipts,
        "folds": {
            "declarations_only_no_fitting": True,
            "row_random_split_allowed": False,
            "repeated_loops_are_independent_folds": False,
            "overlapping_stress_folds_are_independent_replications": False,
            "sensitivity_reporting_required_on_primary_and_stress_folds": True,
            "sensitivity_may_select_model_or_decide_claim_gate": False,
            "items": folds,
        },
        "limitations": deepcopy(dict(config["limitations"])),
        "provenance": provenance,
    }
    if manifest["config_sha256"] != OPENAXES_EXTERNAL_V1_CONFIG_SHA256:
        raise OpenAxesExternalV1Error("prepared config identity drift")
    return prepared, manifest


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_openaxes_external_v1(
    source_path: str | Path,
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create one sealed prepared directory; existing outputs are never reused."""

    validate_openaxes_config(config)
    source = Path(os.path.abspath(os.fspath(source_path)))
    output = Path(output_dir).resolve()
    if os.path.lexists(output):
        raise OpenAxesExternalV1Error(f"prepared output already exists: {output}")
    try:
        source.relative_to(output)
    except ValueError:
        pass
    else:
        raise OpenAxesExternalV1Error("source ZIP must not be inside prepared output")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise OpenAxesExternalV1Error("prepared output parent must not be a symlink")

    frame, semantic_manifest = _prepare_in_memory(source, config)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        parquet_path = temporary / PREPARED_PARQUET
        write_parquet(parquet_path, frame)
        parquet_receipt = {
            "path": PREPARED_PARQUET,
            "bytes": parquet_path.stat().st_size,
            "sha256": sha256_file(parquet_path),
            "rows": len(frame),
        }
        manifest = deepcopy(semantic_manifest)
        manifest["files"] = {PREPARED_PARQUET: parquet_receipt}
        manifest_path = temporary / MANIFEST_JSON
        write_json(manifest_path, manifest)
        complete = _complete_payload(
            config=config,
            manifest=manifest,
            manifest_path=manifest_path,
            parquet_path=parquet_path,
        )
        complete_path = temporary / COMPLETE_JSON
        write_json(complete_path, complete)
        for path in (parquet_path, manifest_path, complete_path):
            _fsync_file(path)
        _fsync_dir(temporary)
        if os.path.lexists(output):
            raise OpenAxesExternalV1Error(
                f"prepared output appeared during build: {output}"
            )
        os.rename(temporary, output)
        _fsync_dir(output.parent)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def _load_prepared_envelope(
    config: Mapping[str, Any], run_dir: str | Path
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Path]]:
    root = Path(run_dir)
    if root.is_symlink() or not root.is_dir():
        raise OpenAxesExternalV1Error("prepared run must be a regular directory")
    paths = {path.name: path for path in root.iterdir()}
    if set(paths) != PREPARED_FILES or any(
        path.is_symlink() or not path.is_file() for path in paths.values()
    ):
        raise OpenAxesExternalV1Error("prepared run file set changed")
    manifest_path = paths[MANIFEST_JSON]
    parquet_path = paths[PREPARED_PARQUET]
    complete_path = paths[COMPLETE_JSON]
    manifest = _strict_json_load(manifest_path)
    complete = _strict_json_load(complete_path)
    if not isinstance(manifest, dict) or not isinstance(complete, dict):
        raise OpenAxesExternalV1Error("prepared JSON roots must be mappings")
    expected_complete = _complete_payload(
        config=config,
        manifest=manifest,
        manifest_path=manifest_path,
        parquet_path=parquet_path,
    )
    if _canonical(complete) != _canonical(expected_complete):
        raise OpenAxesExternalV1Error("prepared COMPLETE seal mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {PREPARED_PARQUET}:
        raise OpenAxesExternalV1Error("prepared manifest file receipt changed")
    receipt = files[PREPARED_PARQUET]
    if not isinstance(receipt, Mapping) or receipt != {
        "path": PREPARED_PARQUET,
        "bytes": parquet_path.stat().st_size,
        "sha256": sha256_file(parquet_path),
        "rows": manifest.get("counts", {}).get("prepared_active_rows"),
    }:
        raise OpenAxesExternalV1Error("prepared parquet receipt mismatch")
    try:
        frame = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise OpenAxesExternalV1Error("cannot decode prepared parquet") from exc
    return manifest, complete, frame, paths


def _closed_frame_sha256(
    frame: pd.DataFrame, schema: Sequence[tuple[str, str]]
) -> str:
    if tuple(frame.columns) != tuple(name for name, _kind in schema):
        raise OpenAxesExternalV1Error("closed-table column schema changed")
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
                    raise OpenAxesExternalV1Error(
                        f"closed-table {name} is not exact UTF-8 text"
                    )
                encoded = value.encode("utf-8")
                digest.update(struct.pack("<I", len(encoded)))
                digest.update(encoded)
        elif kind == "int64":
            values = frame[name].to_numpy()
            if not np.issubdtype(values.dtype, np.integer):
                raise OpenAxesExternalV1Error(
                    f"closed-table {name} is not an integer column"
                )
            digest.update(np.asarray(values, dtype="<i8").tobytes(order="C"))
        elif kind == "float64":
            values = frame[name].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise OpenAxesExternalV1Error(
                    f"closed-table {name} is not finite float64"
                )
            digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
        else:
            raise OpenAxesExternalV1Error(f"unknown closed-table kind: {kind}")
    return digest.hexdigest()


def reconstruct_openaxes_signed_shift_truth_v1(
    source_path: str | Path,
    config: Mapping[str, Any],
    run_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay the six frozen signed-shift truths without fitting or metrics."""

    validate_openaxes_config(config)
    recipe = config["alignment"]["sensitivity_recipe"]
    shifts = tuple(recipe["signed_target_shifts_ms"])
    if shifts != OPENAXES_SENSITIVITY_SIGNED_SHIFTS_MS:
        raise OpenAxesExternalV1Error("signed-shift recipe is not canonical")
    manifest, complete, observed_base, prepared_paths = _load_prepared_envelope(
        config, run_dir
    )
    manifest_source = manifest.get("source")
    manifest_preprocessing = manifest.get("preprocessing")
    manifest_schema = manifest.get("schema")
    manifest_provenance = manifest.get("provenance")
    if not all(
        isinstance(value, Mapping)
        for value in (
            manifest_source,
            manifest_preprocessing,
            manifest_schema,
            manifest_provenance,
        )
    ):
        raise OpenAxesExternalV1Error(
            "prepared manifest signed-shift receipts are malformed"
        )
    manifest_stage = manifest_provenance.get("stage")
    if not isinstance(manifest_stage, Mapping):
        raise OpenAxesExternalV1Error("prepared provenance stage is malformed")
    if (
        manifest.get("contract") != OPENAXES_EXTERNAL_V1_CONTRACT
        or manifest.get("config_sha256") != OPENAXES_EXTERNAL_V1_CONFIG_SHA256
        or manifest_source.get("sha256") != config["source"]["expected_sha256"]
        or manifest_preprocessing.get("sensitivity_recipe") != recipe
        or manifest_schema.get("model_feature_columns")
        != list(model_feature_columns(config))
        or manifest_schema.get("target_columns")
        != list(config["schema"]["target_columns"])
    ):
        raise OpenAxesExternalV1Error(
            "prepared manifest is incompatible with signed-shift truth replay"
        )

    source_config = config["source"]
    base_frames: list[pd.DataFrame] = []
    shifted_frames: list[pd.DataFrame] = []
    dropped_edges: dict[str, dict[str, int]] = {}
    with _open_authenticated_source(
        source_path,
        expected_filename=source_config["filename"],
        expected_size=source_config["expected_size_bytes"],
        expected_sha256=source_config["expected_sha256"],
        expected_entries=source_config["expected_zip_entries"],
        expected_payload_bytes=source_config["expected_zip_payload_bytes"],
    ) as bound:
        aligned, _nested_receipt = _scan_nested_alignment(
            bound.archive, config=config
        )
        for scenario in scenario_configs(config):
            scenario_id = str(scenario["id"])
            aligned_time_us = aligned[scenario_id]
            base, _scenario_receipt = _prepare_scenario(
                bound.archive,
                scenario=scenario,
                aligned_time_us=aligned_time_us,
                config=config,
                source_sha256=bound.receipt["sha256"],
            )
            base_frames.append(base)
            robot = _read_robot_stream(
                bound.archive, scenario=scenario, config=config
            )
            base_time = base["aligned_time_us"].to_numpy(dtype=np.int64)
            dropped_edges[scenario_id] = {}
            for shift_ms in shifts:
                query_time = base_time + int(shift_ms) * 1_000
                supported = (query_time >= int(aligned_time_us[0])) & (
                    query_time <= int(aligned_time_us[-1])
                )
                dropped_edges[scenario_id][str(shift_ms)] = int(
                    len(base_time) - int(supported.sum())
                )
                selected = base.loc[
                    supported,
                    ["row_uid", "scenario_id", "aligned_time_us"],
                ].reset_index(drop=True)
                shifted = pd.DataFrame(
                    {
                        "row_uid": selected["row_uid"].tolist(),
                        "scenario_id": selected["scenario_id"].tolist(),
                        "base_aligned_time_us": selected[
                            "aligned_time_us"
                        ].to_numpy(dtype=np.int64),
                        "shift_ms": np.full(
                            len(selected), int(shift_ms), dtype=np.int64
                        ),
                        "shifted_target_time_us": query_time[supported].astype(
                            np.int64, copy=False
                        ),
                    }
                )
                for axis, target in enumerate(("x", "y", "z")):
                    shifted[f"shifted_tcp_{target}_m"] = np.interp(
                        query_time[supported].astype(np.float64),
                        aligned_time_us.astype(np.float64),
                        robot.tcp_xyz_m[:, axis],
                    )
                shifted_frames.append(shifted)
        source_receipt = bound.receipt

    canonical_base = pd.concat(base_frames, ignore_index=True)
    try:
        pd.testing.assert_frame_equal(
            observed_base,
            canonical_base,
            check_exact=True,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise OpenAxesExternalV1Error(
            "prepared base row_uid/time/value identity replay mismatch"
        ) from exc
    truth = pd.concat(shifted_frames, ignore_index=True)
    truth_schema = tuple(
        zip(
            OPENAXES_SIGNED_SHIFT_TRUTH_V1_COLUMNS,
            (
                "utf8",
                "utf8",
                "int64",
                "int64",
                "int64",
                "float64",
                "float64",
                "float64",
            ),
            strict=True,
        )
    )
    if tuple(truth.columns) != tuple(name for name, _kind in truth_schema):
        raise OpenAxesExternalV1Error("signed-shift truth schema changed")
    if truth.duplicated(["row_uid", "shift_ms"]).any():
        raise OpenAxesExternalV1Error("signed-shift row_uid pairing is not unique")
    if set(truth["shift_ms"].unique()) != set(shifts):
        raise OpenAxesExternalV1Error("signed-shift truth set is incomplete")
    base_uids = set(canonical_base["row_uid"])
    if not set(truth["row_uid"]).issubset(base_uids):
        raise OpenAxesExternalV1Error("signed-shift truth invented a base row_uid")
    if any(
        not set(group["row_uid"]).issubset(base_uids)
        for _shift, group in truth.groupby("shift_ms", sort=False)
    ):
        raise OpenAxesExternalV1Error("signed-shift UID pairing changed")

    base_identity_schema = (
        ("row_uid", "utf8"),
        ("scenario_id", "utf8"),
        ("sample_index", "int64"),
        ("aligned_time_us", "int64"),
    )
    base_identity = canonical_base[
        [name for name, _kind in base_identity_schema]
    ]
    rows_per_shift = {
        str(shift): int((truth["shift_ms"] == shift).sum()) for shift in shifts
    }
    stable_source = {
        key: source_receipt[key]
        for key in (
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
        )
    }
    receipt = {
        "contract": OPENAXES_SIGNED_SHIFT_TRUTH_V1_CONTRACT,
        "preparation_contract": OPENAXES_EXTERNAL_V1_CONTRACT,
        "config_sha256": OPENAXES_EXTERNAL_V1_CONFIG_SHA256,
        "scope": {
            "read_only_truth_reconstruction": True,
            "features_included": False,
            "model_fit_performed": False,
            "metrics_computed": False,
            "selection_performed": False,
            "claim_gate_decided": False,
            "artifact_written": False,
            "semantic_validator_required_before_model_use": True,
        },
        "source": stable_source,
        "prepared": {
            "manifest_sha256": sha256_file(prepared_paths[MANIFEST_JSON]),
            "parquet_sha256": sha256_file(prepared_paths[PREPARED_PARQUET]),
            "complete_sha256": sha256_file(prepared_paths[COMPLETE_JSON]),
            "complete_manifest_sha256": complete["manifest_sha256"],
            "complete_parquet_sha256": complete["prepared_parquet_sha256"],
            "base_rows": len(canonical_base),
            "base_identity_sha256": _closed_frame_sha256(
                base_identity, base_identity_schema
            ),
            "base_exact_source_replay": True,
            "stage": deepcopy(dict(manifest_stage)),
        },
        "recipe": deepcopy(dict(recipe)),
        "output": {
            "closed_schema": [name for name, _kind in truth_schema],
            "row_order": "scenario_then_signed_shift_then_base_time",
            "target_columns_only_with_pairing_metadata": True,
            "rows": len(truth),
            "rows_per_shift": rows_per_shift,
            "dropped_edge_rows_by_scenario_and_shift": dropped_edges,
            "long_table_sha256": _closed_frame_sha256(truth, truth_schema),
        },
        "deterministic_receipt": True,
    }
    return truth, receipt


def validate_openaxes_external_v1(
    source_path: str | Path,
    config: Mapping[str, Any],
    run_dir: str | Path,
) -> dict[str, Any]:
    """Replay source preparation and authenticate every prepared value."""

    validate_openaxes_config(config)
    manifest, _complete, observed_frame, _paths = _load_prepared_envelope(
        config, run_dir
    )

    expected_frame, expected_semantic = _prepare_in_memory(source_path, config)
    observed_semantic = dict(manifest)
    observed_semantic.pop("files", None)
    if _canonical(observed_semantic) != _canonical(expected_semantic):
        raise OpenAxesExternalV1Error("prepared semantic manifest replay mismatch")
    try:
        pd.testing.assert_frame_equal(
            observed_frame,
            expected_frame,
            check_exact=True,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise OpenAxesExternalV1Error("prepared parquet replay mismatch") from exc
    return {
        "status": "ok",
        "contract": OPENAXES_EXTERNAL_V1_CONTRACT,
        "config_sha256": OPENAXES_EXTERNAL_V1_CONFIG_SHA256,
        "source_sha256": config["source"]["expected_sha256"],
        "prepared_rows": len(observed_frame),
        "scenario_rows": {
            scenario: int((observed_frame["scenario_id"] == scenario).sum())
            for scenario in OPENAXES_SCENARIOS
        },
        "feature_count": len(model_feature_columns(config)),
        "declared_outer_folds": 14,
        "aligned_robot_rows_unmatched": 492,
        "signed_sensitivity_shifts_ms": list(
            OPENAXES_SENSITIVITY_SIGNED_SHIFTS_MS
        ),
        "downstream_model_results_blocked_by_uncommitted_contract": bool(
            observed_semantic["provenance"]["stage"][
                "downstream_model_results_blocked_by_uncommitted_contract"
            ]
        ),
        "finite": bool(
            np.isfinite(
                observed_frame[
                    list(model_feature_columns(config))
                    + list(config["schema"]["target_columns"])
                ].to_numpy()
            ).all()
        ),
        "strict_100_hz_per_scenario": True,
        "semantic_replay": True,
    }
