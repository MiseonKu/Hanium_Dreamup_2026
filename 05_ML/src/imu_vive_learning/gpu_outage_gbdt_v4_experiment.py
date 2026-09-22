"""Standalone development-only runner for the post-v3 outage GBDT follow-up.

The module never changes outage-v3.  ``prepare`` first validates a completed,
still-sealed v3 development run and copies only durable development evidence
and label-blind challenge identity into a new v4 run. XGBoost then competes
with that immutable incumbent and a predeclared paired 0.5 blend. Every v4
invocation stops with challenge labels sealed. The separate outage-v5
contract is the sole owner of any future challenge gate; this module imports
no challenge-label loader and exposes no source-label access path.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
from pathlib import Path, PurePosixPath
import queue
import shutil
import subprocess
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .experiment_io import (
    atomic_write_json,
    atomic_write_parquet,
    manifest_files,
    prepare_experiment_directory,
    read_json_object,
    sha256_file,
    stable_hash,
)
from .experiment_metrics import regression_metrics
from .experiment_preprocessing import TrainOnlyPreprocessor
from .features import FUSION_COLUMNS
from .gpu_outage_experiment_config import (
    OUTAGE_GPU_EXPERIMENT_CONTRACT,
    TRAINABLE_OUTAGE_CANDIDATE_IDS,
    outage_candidate_grid,
    outage_gpu_config_sha256,
    validate_outage_gpu_config,
)
from .gpu_outage_gbdt_v4_config import (
    GBDT_V4_CANDIDATE_IDS,
    GBDT_V4_CONTRACT,
    GBDT_V4_DEVELOPMENT_SEGMENTS,
    GBDT_V4_PAIRED_SEEDS,
    gbdt_v4_config_sha256,
    load_gbdt_v4_config,
    public_gbdt_v4_config,
    validate_gbdt_v4_config,
)


GBDT_V4_RUN_CONTRACT = "alphaf.endpoint_regression.gpu_outage_gbdt_run.v4"
GBDT_V4_FREEZE_CONTRACT = "alphaf.endpoint_regression.gpu_outage_gbdt_incumbent_freeze.v4"
GBDT_V4_BUNDLE_RESULT_CONTRACT = "alphaf.endpoint_regression.gpu_outage_gbdt_bundle_result.v4"
GBDT_V4_TASK_EXECUTION_CONTRACT = (
    "alphaf.endpoint_regression.gpu_outage_gbdt_task_execution.v4"
)


def _read_v3_source_closure_without_importing_runner() -> tuple[str, ...]:
    """Read v3's literal closure without importing its challenge-capable runner."""

    source_path = Path(__file__).with_name("gpu_outage_experiment.py")
    module = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "OUTAGE_SOURCE_FILES"
            for target in statement.targets
        ):
            continue
        value = ast.literal_eval(statement.value)
        if (
            not isinstance(value, tuple)
            or not value
            or not all(type(path) is str and path for path in value)
        ):
            break
        return value
    raise RuntimeError("cannot read the frozen outage-v3 source closure")


_V3_SOURCE_FILES = _read_v3_source_closure_without_importing_runner()
GBDT_V4_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *_V3_SOURCE_FILES,
            "configs/endpoint_regression_gpu_outage_v4_gbdt.yaml",
            "docs/GPU_ENDPOINT_REGRESSION_OUTAGE_V4_GBDT_CONTRACT.md",
            "src/imu_vive_learning/gpu_outage_gbdt_v4_config.py",
            "src/imu_vive_learning/gpu_outage_gbdt_v4_experiment.py",
            "src/imu_vive_learning/gpu_outage_gbdt_v4_training.py",
            "tests/test_gpu_outage_gbdt_v4_experiment.py",
            "tests/test_gpu_outage_gbdt_v4_training.py",
        )
    )
)

_TRUE_COLUMNS = ("true_x_mm", "true_y_mm", "true_z_mm")
_PRED_COLUMNS = ("pred_x_mm", "pred_y_mm", "pred_z_mm")
AVAILABILITY_NAMES = (
    "imu0_base_available",
    "imu1_link_root_available",
    "imu2_link_mid_available",
    "imu3_tip_available",
)
_IMU_PREFIXES = ("imu0_base", "imu1_link_root", "imu2_link_mid", "imu3_tip")
_DEVELOPMENT_SEGMENT_ROWS = {36: 1672, 37: 1597, 38: 1707, 39: 1659}
_PREPARE_INVALID_RELATIVE_PATH = "_state/prepare_invalid.json"
_REFERENCE_COUNT_UNSET = object()
_XGB_BUNDLE_MEMBERS = {
    "axis_x.ubj": "file",
    "axis_y.ubj": "file",
    "axis_z.ubj": "file",
    "metadata.json": "file",
}


class GPUOutageGBDTV4Error(RuntimeError):
    """Raised when v4 evidence or execution violates the frozen contract."""


def _readonly_v4_array(value: Any, *, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_v4_input_arrays(
    *,
    feature_names: Sequence[str],
    X: Any,
    availability: Any,
    trial_uids: Any,
    segment_ids: Any,
    target_ids: Any,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate copied replay arrays without importing v3's label-loader module."""

    names = tuple(str(value) for value in feature_names)
    if names != FUSION_COLUMNS or len(names) != 43:
        raise GPUOutageGBDTV4Error(
            "v4 replay feature_names must be ordered 43-column fusion_v1"
        )
    features = _readonly_v4_array(X, dtype=np.float64)
    masks = _readonly_v4_array(availability, dtype=np.float64)
    uids = _readonly_v4_array(trial_uids, dtype=str)
    segments = _readonly_v4_array(segment_ids, dtype=np.int64)
    targets = _readonly_v4_array(target_ids, dtype=str)
    rows = int(features.shape[0]) if features.ndim == 2 else -1
    if features.ndim != 2 or features.shape[1] != 43:
        raise GPUOutageGBDTV4Error("v4 replay X must have shape [rows, 43]")
    if masks.shape != (rows, 4):
        raise GPUOutageGBDTV4Error(
            "v4 replay availability must have shape [rows, 4]"
        )
    if uids.shape != (rows,) or segments.shape != (rows,) or targets.shape != (rows,):
        raise GPUOutageGBDTV4Error(
            "v4 replay metadata arrays must each have shape [rows]"
        )
    if not np.isfinite(masks).all() or not np.isin(masks, (0.0, 1.0)).all():
        raise GPUOutageGBDTV4Error(
            "v4 replay availability must contain finite zero/one values"
        )
    if not np.isfinite(features[:, :3]).all():
        raise GPUOutageGBDTV4Error("v4 replay encoder inputs contain non-finite values")
    for sensor_index, prefix in enumerate(_IMU_PREFIXES):
        indices = np.asarray(
            [
                index
                for index, name in enumerate(FUSION_COLUMNS)
                if name.startswith(f"{prefix}_")
            ],
            dtype=np.int64,
        )
        values = features[:, indices]
        available_rows = masks[:, sensor_index] == 1.0
        unavailable_rows = ~available_rows
        if available_rows.any() and not np.isfinite(values[available_rows]).all():
            raise GPUOutageGBDTV4Error(
                f"v4 replay available IMU{sensor_index} inputs are non-finite"
            )
        if unavailable_rows.any() and not np.isnan(values[unavailable_rows]).all():
            raise GPUOutageGBDTV4Error(
                f"v4 replay unavailable IMU{sensor_index} inputs are not NaN sentinels"
            )
    uid_values = uids.tolist()
    if rows != len(set(uid_values)) or any(
        len(uid) != 70
        or not uid.startswith("trial_")
        or any(character not in "0123456789abcdef" for character in uid[6:])
        for uid in uid_values
    ):
        raise GPUOutageGBDTV4Error("v4 replay trial UID identity changed")
    if any(not value for value in targets.tolist()):
        raise GPUOutageGBDTV4Error("v4 replay target IDs cannot be empty")
    return names, features, masks, uids, segments, targets


@dataclass(frozen=True, slots=True, repr=False)
class ChallengeIdentityData:
    """Label-free challenge identity copied from sealed v3 evidence."""

    feature_names: tuple[str, ...]
    X: np.ndarray = field(repr=False)
    availability: np.ndarray = field(repr=False)
    trial_uids: np.ndarray = field(repr=False)
    segment_ids: np.ndarray = field(repr=False)
    target_ids: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        values = _validate_v4_input_arrays(
            feature_names=self.feature_names,
            X=self.X,
            availability=self.availability,
            trial_uids=self.trial_uids,
            segment_ids=self.segment_ids,
            target_ids=self.target_ids,
        )
        for name, value in zip(
            (
                "feature_names",
                "X",
                "availability",
                "trial_uids",
                "segment_ids",
                "target_ids",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)

    @property
    def rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def ordered_trial_uid_sha256(self) -> str:
        return stable_hash(self.trial_uids.astype(str).tolist())


@dataclass(frozen=True, slots=True, repr=False)
class DevelopmentData:
    """Authenticated development replay copied from sealed v3 evidence."""

    feature_names: tuple[str, ...]
    X: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    availability: np.ndarray = field(repr=False)
    trial_uids: np.ndarray = field(repr=False)
    segment_ids: np.ndarray = field(repr=False)
    target_ids: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        values = _validate_v4_input_arrays(
            feature_names=self.feature_names,
            X=self.X,
            availability=self.availability,
            trial_uids=self.trial_uids,
            segment_ids=self.segment_ids,
            target_ids=self.target_ids,
        )
        labels = _readonly_v4_array(self.y, dtype=np.float64)
        if labels.shape != (len(values[1]), 3) or not np.isfinite(labels).all():
            raise GPUOutageGBDTV4Error(
                "v4 development labels must be finite with shape [rows, 3]"
            )
        for name, value in zip(
            (
                "feature_names",
                "X",
                "availability",
                "trial_uids",
                "segment_ids",
                "target_ids",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "y", labels)

    @property
    def rows(self) -> int:
        return int(self.X.shape[0])


def _exact_int(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
    allowed: Sequence[int] | None = None,
) -> int:
    if type(value) is not int:
        raise GPUOutageGBDTV4Error(f"{name} must be an exact integer")
    if minimum is not None and value < minimum:
        raise GPUOutageGBDTV4Error(f"{name} is below its minimum")
    if allowed is not None and value not in allowed:
        raise GPUOutageGBDTV4Error(f"{name} is outside the frozen values")
    return value


def _metric_real(value: Any, *, name: str) -> float:
    """Accept JSON int/float metrics, but never bools or numeric strings."""

    if type(value) not in {int, float}:
        raise GPUOutageGBDTV4Error(f"{name} must be a JSON real number")
    return float(value)


def _type_exact_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(
                type(key) is str
                and _type_exact_equal(actual[key], expected[key])
                for key in expected
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _type_exact_equal(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    return type(actual) is type(expected) and actual == expected


def _validate_sealed_v3_development(run_dir: str | Path) -> Mapping[str, Any]:
    """Run v3's sealed validator in an isolated process and parse finite JSON."""

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        "-m",
        "imu_vive_learning.cli",
        "validate-gpu-outage-experiment",
        "--run-dir",
        str(Path(run_dir).resolve()),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=_runner_workspace_root(),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=300.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GPUOutageGBDTV4Error(
            "isolated v3 sealed-development validation failed to execute"
        ) from exc
    if completed.returncode != 0:
        raise GPUOutageGBDTV4Error(
            "isolated v3 sealed-development validator rejected the run"
        )

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        payload = json.loads(completed.stdout, parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GPUOutageGBDTV4Error(
            "isolated v3 validator did not return finite JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise GPUOutageGBDTV4Error("isolated v3 validator payload is not an object")
    return payload


def _require_sealed_v3_development_validation(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept only the complete 240-fit/96-context sealed v3 stage."""

    validation = dict(payload)
    if (
        validation.get("status") != "ok"
        or validation.get("stage")
        != "development_complete_challenge_labels_sealed"
        or validation.get("challenge_labels_sealed") is not True
        or _exact_int(
            validation.get("development_seed_fits"),
            name="v3 validator development_seed_fits",
        )
        != 240
        or _exact_int(
            validation.get("scored_contexts"),
            name="v3 validator scored_contexts",
        )
        != 96
    ):
        raise GPUOutageGBDTV4Error(
            "v3 development validator did not return the sealed stage"
        )
    return validation


def _require_python_312() -> None:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise GPUOutageGBDTV4Error("outage-v4 GBDT requires CPython 3.12.x")


def _path_has_symlink_component(path: str | Path) -> bool:
    lexical = Path(path)
    absolute = lexical if lexical.is_absolute() else Path.cwd() / lexical
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        if part in {"", "."}:
            continue
        if part == "..":
            current = current.parent
            continue
        current /= part
        if current.is_symlink():
            return True
    return False


def _validate_prepare_run_paths(
    v3_run_dir: str | Path, v4_run_dir: str | Path
) -> tuple[Path, Path]:
    """Require disjoint, non-aliased v3 input and v4 output trees."""

    if _path_has_symlink_component(v3_run_dir) or _path_has_symlink_component(
        v4_run_dir
    ):
        raise GPUOutageGBDTV4Error("v3/v4 run paths may not use symlink aliases")
    v3_run = Path(v3_run_dir).resolve()
    v4_run = Path(v4_run_dir).resolve()
    if (
        v3_run == v4_run
        or v3_run in v4_run.parents
        or v4_run in v3_run.parents
    ):
        raise GPUOutageGBDTV4Error(
            "v3 input and v4 output must be distinct non-ancestor paths"
        )
    return v3_run, v4_run


def _runner_workspace_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "configs" / "endpoint_regression_gpu_outage_v4_gbdt.yaml").is_file():
        raise GPUOutageGBDTV4Error("cannot locate the outage-v4 source workspace")
    return root


def gbdt_v4_source_payload(workspace_root: str | Path) -> dict[str, Any]:
    """Hash the complete executable/config/test closure used by v4."""

    root = Path(workspace_root).resolve()
    files: list[dict[str, Any]] = []
    for relative in GBDT_V4_SOURCE_FILES:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise GPUOutageGBDTV4Error(f"v4 source path escapes workspace: {relative}") from exc
        if not path.is_file():
            raise GPUOutageGBDTV4Error(f"v4 source closure file is missing: {relative}")
        files.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {"files": files, "fingerprint": stable_hash(files)}


def _cuda_device_identity(index: int) -> dict[str, Any]:
    """Return the stable UUID/PCI identity of one currently visible CUDA device."""

    import torch

    properties = torch.cuda.get_device_properties(int(index))
    uuid = str(getattr(properties, "uuid", ""))
    try:
        domain = int(properties.pci_domain_id)
        bus = int(properties.pci_bus_id)
        device = int(properties.pci_device_id)
    except (AttributeError, TypeError, ValueError) as exc:
        raise GPUOutageGBDTV4Error("CUDA device PCI identity is unavailable") from exc
    if not uuid:
        raise GPUOutageGBDTV4Error("CUDA device UUID is unavailable")
    return {
        "uuid": uuid,
        "pci_bus_id": f"{domain:08x}:{bus:02x}:{device:02x}.0",
        "name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
    }


def _validate_parent_cuda_device_grid(devices: Any) -> list[Mapping[str, Any]]:
    if not isinstance(devices, list) or len(devices) != 4:
        raise GPUOutageGBDTV4Error("v4 environment must expose exactly four CUDA GPUs")
    expected_keys = {
        "physical_gpu_id",
        "uuid",
        "pci_bus_id",
        "name",
        "total_memory_bytes",
        "compute_capability",
    }
    normalized: list[Mapping[str, Any]] = []
    for row in devices:
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise GPUOutageGBDTV4Error("v4 parent CUDA identity record changed")
        capability = row.get("compute_capability")
        physical_gpu_id = _exact_int(
            row.get("physical_gpu_id"),
            name="parent physical_gpu_id",
            allowed=(0, 1, 2, 3),
        )
        total_memory_bytes = _exact_int(
            row.get("total_memory_bytes"),
            name="parent total_memory_bytes",
            minimum=1,
        )
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(type(value) is not int or value < 0 for value in capability)
            or not isinstance(row.get("uuid"), str)
            or not row.get("uuid")
            or not isinstance(row.get("pci_bus_id"), str)
            or len(str(row.get("pci_bus_id"))) != 16
            or str(row.get("pci_bus_id"))[8] != ":"
            or str(row.get("pci_bus_id"))[11] != ":"
            or str(row.get("pci_bus_id"))[14] != "."
            or not isinstance(row.get("name"), str)
            or not row.get("name")
        ):
            raise GPUOutageGBDTV4Error("v4 parent CUDA UUID/PCI metadata changed")
        if physical_gpu_id != row["physical_gpu_id"] or total_memory_bytes != row[
            "total_memory_bytes"
        ]:
            raise GPUOutageGBDTV4Error("v4 parent CUDA integer identity changed")
        normalized.append(row)
    ids = [row["physical_gpu_id"] for row in normalized]
    uuids = [str(row["uuid"]) for row in normalized]
    pci_ids = [str(row["pci_bus_id"]) for row in normalized]
    if ids != [0, 1, 2, 3] or len(set(uuids)) != 4 or len(set(pci_ids)) != 4:
        raise GPUOutageGBDTV4Error("v4 parent CUDA UUID/PCI grid is not one-to-one")
    return normalized


def _environment_payload_without_cuda_devices() -> dict[str, Any]:
    """Capture worker-invariant runtime identity before CUDA visibility mapping."""

    _require_python_312()
    import pyarrow
    import torch
    import xgboost

    from .gpu_outage_gbdt_v4_training import fixed_xgb_parameter_sha256

    library = Path(str(xgboost.core._LIB._name)).resolve()
    if not library.is_file():
        raise GPUOutageGBDTV4Error("cannot locate the loaded libxgboost binary")
    build_info = dict(xgboost.build_info())
    if str(xgboost.__version__) != "3.3.0" or build_info.get("USE_CUDA") is not True:
        raise GPUOutageGBDTV4Error("v4 requires CUDA-enabled XGBoost 3.3.0")
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "torch": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda),
        "xgboost": str(xgboost.__version__),
        "xgboost_build_info": build_info,
        "libxgboost": {
            "path": str(library),
            "bytes": int(library.stat().st_size),
            "sha256": sha256_file(library),
        },
        "fixed_xgb_parameter_sha256": fixed_xgb_parameter_sha256(),
    }


def gbdt_v4_environment_payload() -> dict[str, Any]:
    """Bind worker-invariant runtime identity and the exact parent GPU grid."""

    import torch

    payload = _environment_payload_without_cuda_devices()
    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append(
                {
                    "physical_gpu_id": index,
                    **_cuda_device_identity(index),
                }
            )
    _validate_parent_cuda_device_grid(devices)
    payload["cuda_devices"] = devices
    return payload


def _validate_worker_cuda_mapping(
    runtime: Mapping[str, Any], environment_snapshot: Mapping[str, Any]
) -> None:
    """Fail closed unless visible ``cuda:0`` is the frozen requested physical GPU."""

    payload = environment_snapshot.get("payload", environment_snapshot)
    devices = _validate_parent_cuda_device_grid(
        payload.get("cuda_devices") if isinstance(payload, Mapping) else None
    )
    physical_gpu_id = _exact_int(
        runtime.get("physical_gpu_id"),
        name="worker physical_gpu_id",
        allowed=(0, 1, 2, 3),
    )
    matches = [
        row
        for row in devices
        if isinstance(row, Mapping)
        and row.get("physical_gpu_id") == physical_gpu_id
    ]
    if len(matches) != 1:
        raise GPUOutageGBDTV4Error("worker physical GPU is absent from parent freeze")
    expected = matches[0]
    if (
        runtime.get("visible_device") != "cuda:0"
        or runtime.get("visible_cuda_device_uuid") != expected.get("uuid")
        or runtime.get("visible_cuda_pci_bus_id") != expected.get("pci_bus_id")
        or runtime.get("visible_cuda_name") != expected.get("name")
    ):
        raise GPUOutageGBDTV4Error(
            "worker visible cuda:0 UUID/PCI does not map to its physical GPU"
        )


def _worker_cuda_runtime_identity(
    run_dir: str | Path, *, physical_gpu_id: int
) -> dict[str, Any]:
    """Capture and validate a subprocess after CUDA_VISIBLE_DEVICES isolation."""

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise GPUOutageGBDTV4Error("v4 worker must see exactly one CUDA device")
    visible = _cuda_device_identity(0)
    runtime = {
        "physical_gpu_id": _exact_int(
            physical_gpu_id,
            name="worker physical_gpu_id",
            allowed=(0, 1, 2, 3),
        ),
        "visible_device": "cuda:0",
        "visible_cuda_device_uuid": visible["uuid"],
        "visible_cuda_pci_bus_id": visible["pci_bus_id"],
        "visible_cuda_name": visible["name"],
    }
    environment = read_json_object(
        Path(run_dir) / "provenance" / "environment_snapshot.json"
    )
    _validate_worker_cuda_mapping(runtime, environment)
    return runtime


def _validate_v4_source_provenance(
    run_dir: str | Path,
    *,
    workspace_root: str | Path | None = None,
    validate_environment: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = Path(run_dir).resolve()
    workspace = _runner_workspace_root() if workspace_root is None else Path(workspace_root).resolve()
    current_source = gbdt_v4_source_payload(workspace)
    stored_source = read_json_object(run / "provenance" / "source_snapshot.json")
    if stable_hash(stored_source.get("files")) != stored_source.get("fingerprint"):
        raise GPUOutageGBDTV4Error("v4 durable source snapshot self-hash changed")
    if not _type_exact_equal(stored_source, current_source):
        raise GPUOutageGBDTV4Error("current v4 source closure differs from its freeze")
    stored_environment = read_json_object(
        run / "provenance" / "environment_snapshot.json"
    )
    if (
        set(stored_environment) != {"payload", "fingerprint"}
        or not isinstance(stored_environment.get("payload"), Mapping)
        or stable_hash(stored_environment.get("payload"))
        != stored_environment.get("fingerprint")
    ):
        raise GPUOutageGBDTV4Error("v4 durable environment snapshot self-hash changed")
    if validate_environment:
        current_environment = gbdt_v4_environment_payload()
        if not _type_exact_equal(
            stored_environment.get("payload"), current_environment
        ):
            raise GPUOutageGBDTV4Error("current v4/XGBoost environment differs from its freeze")
    else:
        frozen = stored_environment["payload"]
        if not isinstance(frozen, Mapping):
            raise GPUOutageGBDTV4Error("v4 worker environment freeze is invalid")
        _validate_parent_cuda_device_grid(frozen.get("cuda_devices"))
        frozen_invariant = {
            key: value for key, value in frozen.items() if key != "cuda_devices"
        }
        current_invariant = _environment_payload_without_cuda_devices()
        if not _type_exact_equal(frozen_invariant, current_invariant):
            raise GPUOutageGBDTV4Error(
                "v4 worker invariant environment identity changed"
            )
    return current_source, stored_environment


def _relative(path: str | Path, root: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError as exc:
        raise GPUOutageGBDTV4Error("artifact path escapes its run directory") from exc


def _safe_member(root: str | Path, relative: str) -> Path:
    member = PurePosixPath(str(relative))
    if member.is_absolute() or ".." in member.parts:
        raise GPUOutageGBDTV4Error(f"unsafe artifact member: {relative!r}")
    path = (Path(root).resolve() / Path(*member.parts)).resolve()
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise GPUOutageGBDTV4Error(f"artifact member escapes run: {relative!r}") from exc
    return path


def _assert_exact_directory_children(
    directory: str | Path,
    expected: Mapping[str, str],
    *,
    label: str,
) -> None:
    """Enforce a closed-world directory with exact file/directory member kinds."""

    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise GPUOutageGBDTV4Error(f"{label} directory is missing or unsafe")
    children = {child.name: child for child in root.iterdir()}
    if set(children) != set(expected):
        extra = sorted(set(children).difference(expected))
        missing = sorted(set(expected).difference(children))
        raise GPUOutageGBDTV4Error(
            f"{label} members changed; missing={missing}, extra={extra}"
        )
    for name, kind in expected.items():
        child = children[name]
        if child.is_symlink():
            raise GPUOutageGBDTV4Error(f"{label} member is a symlink: {name}")
        if kind == "file" and not child.is_file():
            raise GPUOutageGBDTV4Error(f"{label} member is not a file: {name}")
        if kind == "directory" and not child.is_dir():
            raise GPUOutageGBDTV4Error(f"{label} member is not a directory: {name}")
        if kind not in {"file", "directory"}:
            raise GPUOutageGBDTV4Error(f"invalid expected member kind: {kind}")


def _artifact_reference(
    path: str | Path, root: str | Path, **extra: Any
) -> dict[str, Any]:
    if set(extra).difference({"rows", "tasks"}):
        raise GPUOutageGBDTV4Error("artifact reference has an undeclared count")
    normalized_extra = {
        key: _exact_int(value, name=f"artifact reference {key}", minimum=0)
        for key, value in extra.items()
    }
    artifact = Path(path)
    payload: dict[str, Any] = {
        "path": _relative(artifact, root),
        "bytes": int(artifact.stat().st_size),
        "sha256": sha256_file(artifact),
    }
    payload.update(normalized_extra)
    return payload


def _assert_development_only_run(run_dir: str | Path) -> None:
    """Reject every artifact class reserved for the separate outage-v5 gate."""

    run = Path(run_dir)
    invalid_marker = run / _PREPARE_INVALID_RELATIVE_PATH
    if invalid_marker.exists() or invalid_marker.is_symlink():
        raise GPUOutageGBDTV4Error(
            "outage-v4 prepare_invalid marker rejects this run"
        )
    forbidden = [
        run / "final",
        run / "challenge",
        run / "manifest.json",
        run / "provenance" / "final_incumbent_gpu_workers.json",
        run / "provenance" / "final_xgb_gpu_workers.json",
        run / "provenance" / "worker_ready" / "final_incumbent",
        run / "provenance" / "worker_ready" / "final_xgb",
        run / "recovery" / "final_incumbent",
        run / "recovery" / "final_xgb",
    ]
    present = [_relative(path, run) for path in forbidden if path.exists()]
    if present:
        raise GPUOutageGBDTV4Error(
            f"outage-v4 is development-only; forbidden artifacts present: {present}"
        )


def _verify_reference(
    root: str | Path,
    reference: Mapping[str, Any],
    *,
    expected_rows: int | object = _REFERENCE_COUNT_UNSET,
    expected_tasks: int | object = _REFERENCE_COUNT_UNSET,
) -> Path:
    if not isinstance(reference, Mapping):
        raise GPUOutageGBDTV4Error("artifact reference must be an object")
    expected_counts: dict[str, int] = {}
    for key, value in (("rows", expected_rows), ("tasks", expected_tasks)):
        if value is not _REFERENCE_COUNT_UNSET:
            expected_counts[key] = _exact_int(
                value, name=f"expected artifact reference {key}", minimum=0
            )
    if set(reference) != {"path", "bytes", "sha256", *expected_counts}:
        raise GPUOutageGBDTV4Error("artifact reference schema changed")
    relative = reference.get("path")
    digest = reference.get("sha256")
    if type(relative) is not str or not relative:
        raise GPUOutageGBDTV4Error("artifact reference path must be an exact string")
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise GPUOutageGBDTV4Error(
            "artifact reference sha256 must be exact lowercase hexadecimal"
        )
    for key, expected in expected_counts.items():
        if _exact_int(
            reference.get(key), name=f"artifact reference {key}", minimum=0
        ) != expected:
            raise GPUOutageGBDTV4Error(f"artifact reference {key} changed")
    path = _safe_member(root, relative)
    if not path.is_file():
        raise GPUOutageGBDTV4Error(f"referenced artifact is missing: {path}")
    if _exact_int(
        reference.get("bytes"), name="artifact reference bytes", minimum=0
    ) != int(path.stat().st_size):
        raise GPUOutageGBDTV4Error(f"referenced artifact size changed: {path}")
    if digest != sha256_file(path):
        raise GPUOutageGBDTV4Error(f"referenced artifact SHA-256 changed: {path}")
    return path


def _copy_atomic(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    shutil.copyfile(source_path, temporary)
    os.replace(temporary, output)
    return output


def _exclusive_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Create a durable one-time claim; never remove a partial claim."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise GPUOutageGBDTV4Error(
            "exclusive v4 evidence path is already claimed; refusing overwrite"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        raise
    return output


def _task_execution_path(task_dir: str | Path) -> Path:
    return Path(task_dir) / "task_execution.json"


def _begin_xgb_task_execution(
    *,
    run_dir: str | Path,
    task_dir: str | Path,
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    phase: str,
    gpu_id: int,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Durably bind a task to its physical worker before fitting starts."""

    source_snapshot, environment_snapshot = _validate_v4_source_provenance(
        run_dir, validate_environment=False
    )
    physical_gpu_id = _exact_int(
        gpu_id, name="task execution physical_gpu_id", allowed=(0, 1, 2, 3)
    )
    payload = {
        "contract": GBDT_V4_TASK_EXECUTION_CONTRACT,
        "phase": phase,
        "config_sha256": gbdt_v4_config_sha256(config),
        "task": dict(task),
        "physical_gpu_id": physical_gpu_id,
        "training_source_fingerprint": source_snapshot["fingerprint"],
        "environment_fingerprint": environment_snapshot["fingerprint"],
        "challenge_labels_accessed": False,
    }
    path = _exclusive_write_json(_task_execution_path(task_dir), payload)
    return path, source_snapshot, environment_snapshot


def _validate_xgb_task_execution(
    *,
    run_dir: str | Path,
    reference_or_path: Mapping[str, Any] | str | Path,
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    phase: str,
) -> tuple[Path, dict[str, Any]]:
    """Authenticate the durable pre-fit receipt used for crash recovery."""

    path = (
        _verify_reference(run_dir, reference_or_path)
        if isinstance(reference_or_path, Mapping)
        else Path(reference_or_path)
    )
    payload = read_json_object(path)
    source = read_json_object(Path(run_dir) / "provenance" / "source_snapshot.json")
    environment = read_json_object(
        Path(run_dir) / "provenance" / "environment_snapshot.json"
    )
    expected_keys = {
        "contract",
        "phase",
        "config_sha256",
        "task",
        "physical_gpu_id",
        "training_source_fingerprint",
        "environment_fingerprint",
        "challenge_labels_accessed",
    }
    gpu_id = _exact_int(
        payload.get("physical_gpu_id"),
        name="task execution physical_gpu_id",
        allowed=(0, 1, 2, 3),
    )
    frozen_devices = _validate_parent_cuda_device_grid(
        environment.get("payload", {}).get("cuda_devices")
    )
    frozen_ids = {
        row["physical_gpu_id"]
        for row in frozen_devices
    }
    if (
        set(payload) != expected_keys
        or payload.get("contract") != GBDT_V4_TASK_EXECUTION_CONTRACT
        or payload.get("phase") != phase
        or payload.get("config_sha256") != gbdt_v4_config_sha256(config)
        or not _type_exact_equal(payload.get("task"), dict(task))
        or gpu_id not in frozen_ids
        or payload.get("training_source_fingerprint") != source.get("fingerprint")
        or payload.get("environment_fingerprint") != environment.get("fingerprint")
        or payload.get("challenge_labels_accessed") is not False
    ):
        raise GPUOutageGBDTV4Error("v4 task execution receipt changed")
    return path, payload


def _read_exact_parquet(path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    actual = parquet.schema_arrow.names
    expected = list(columns)
    if actual != expected:
        raise GPUOutageGBDTV4Error(
            f"Parquet schema/order changed for {path}: expected {expected}, received {actual}"
        )
    frame = parquet.read(columns=expected).to_pandas()
    if list(frame.columns) != expected:
        raise GPUOutageGBDTV4Error(f"Parquet projection order changed for {path}")
    return frame


def _normalise_v3_config(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        for role in ("development", "challenge"):
            raw = payload["row_roles"][role]["expected_segment_rows"]
            payload["row_roles"][role]["expected_segment_rows"] = {
                int(key): int(value) for key, value in raw.items()
            }
    except (KeyError, TypeError, ValueError) as exc:
        raise GPUOutageGBDTV4Error("invalid durable v3 resolved config") from exc
    validate_outage_gpu_config(payload)
    return payload


def _load_v3_config(path: str | Path) -> dict[str, Any]:
    return _normalise_v3_config(read_json_object(path))


def _preprocessor_from_metadata(metadata: Mapping[str, Any]) -> TrainOnlyPreprocessor:
    names = tuple(str(value) for value in metadata["feature_names"])
    preprocessor = TrainOnlyPreprocessor(
        feature_names=names,
        scale_features=True,
        split_context=str(metadata["leakage_audit"]["split_context"]),
    )
    preprocessor.feature_mean_ = np.asarray(metadata["feature_mean"], dtype=np.float64)
    preprocessor.feature_scale_ = np.asarray(metadata["feature_scale"], dtype=np.float64)
    preprocessor.target_mean_ = np.asarray(metadata["target_mean"], dtype=np.float64)
    preprocessor.target_scale_ = float(metadata["target_scale"])
    preprocessor.leakage_audit_ = dict(metadata["leakage_audit"])
    if names != FUSION_COLUMNS or not preprocessor.is_fitted:
        raise GPUOutageGBDTV4Error("durable preprocessor layout/state changed")
    return preprocessor


def _fit_preprocessor(
    data: DevelopmentData,
    train_rows: np.ndarray,
    *,
    forbidden_rows: np.ndarray,
    split_context: str,
) -> TrainOnlyPreprocessor:
    return TrainOnlyPreprocessor(
        feature_names=tuple(data.feature_names),
        scale_features=True,
        split_context=split_context,
    ).fit_real_train(
        data.X[train_rows],
        data.y[train_rows],
        trial_uids=data.trial_uids[train_rows].tolist(),
        forbidden_uids=data.trial_uids[forbidden_rows].tolist(),
    )


def _load_development_replay(run_dir: str | Path) -> DevelopmentData:
    run = Path(run_dir)
    metadata = read_json_object(run / "inputs" / "development_replay.json")
    replay_rows = _exact_int(
        metadata.get("rows"), name="development replay rows", minimum=1
    )
    if replay_rows != sum(_DEVELOPMENT_SEGMENT_ROWS.values()):
        raise GPUOutageGBDTV4Error("v4 development replay row count changed")
    feature_names = tuple(str(value) for value in metadata["feature_names"])
    if feature_names != FUSION_COLUMNS or tuple(metadata["label_names"]) != _TRUE_COLUMNS:
        raise GPUOutageGBDTV4Error("v4 development replay layout changed")
    frame = _read_exact_parquet(
        _verify_reference(
            run, metadata["input_file"], expected_rows=replay_rows
        ),
        ["trial_uid", "segment_id", "target_id", *feature_names, *_TRUE_COLUMNS],
    )
    data = DevelopmentData(
        feature_names=feature_names,
        X=frame.loc[:, list(feature_names)].to_numpy(dtype=np.float64),
        y=frame.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64),
        availability=np.ones((len(frame), 4), dtype=np.float64),
        trial_uids=frame["trial_uid"].astype(str).to_numpy(),
        segment_ids=frame["segment_id"].to_numpy(dtype=np.int64),
        target_ids=frame["target_id"].astype(str).to_numpy(),
    )
    if data.rows != replay_rows:
        raise GPUOutageGBDTV4Error("v4 development replay row count changed")
    counts = {
        int(segment): int(np.count_nonzero(data.segment_ids == segment))
        for segment in GBDT_V4_DEVELOPMENT_SEGMENTS
    }
    uid_hash = hashlib.sha256(
        "\n".join(sorted(data.trial_uids.astype(str).tolist())).encode("utf-8")
    ).hexdigest()
    if counts != _DEVELOPMENT_SEGMENT_ROWS or uid_hash != str(
        metadata["sorted_trial_uid_sha256"]
    ):
        raise GPUOutageGBDTV4Error("v4 development replay identity changed")
    return data


def _load_challenge_identity(run_dir: str | Path) -> ChallengeIdentityData:
    run = Path(run_dir)
    metadata = read_json_object(run / "inputs" / "challenge_identity.json")
    identity_rows = _exact_int(
        metadata.get("rows"), name="challenge identity rows", minimum=1
    )
    if identity_rows != 3295:
        raise GPUOutageGBDTV4Error("v4 challenge identity row count changed")
    feature_names = tuple(str(value) for value in metadata["feature_names"])
    availability_names = tuple(str(value) for value in metadata["availability_names"])
    if feature_names != FUSION_COLUMNS or availability_names != AVAILABILITY_NAMES:
        raise GPUOutageGBDTV4Error("v4 challenge identity layout changed")
    frame = _read_exact_parquet(
        _verify_reference(
            run, metadata["identity_file"], expected_rows=identity_rows
        ),
        ["trial_uid", "segment_id", "target_id", *feature_names, *availability_names],
    )
    identity = ChallengeIdentityData(
        feature_names=feature_names,
        X=frame.loc[:, list(feature_names)].to_numpy(dtype=np.float64),
        availability=frame.loc[:, list(availability_names)].to_numpy(dtype=np.float64),
        trial_uids=frame["trial_uid"].astype(str).to_numpy(),
        segment_ids=frame["segment_id"].to_numpy(dtype=np.int64),
        target_ids=frame["target_id"].astype(str).to_numpy(),
    )
    if identity.rows != identity_rows or identity.ordered_trial_uid_sha256 != str(
        metadata["ordered_trial_uid_sha256"]
    ):
        raise GPUOutageGBDTV4Error("v4 challenge identity changed")
    expected_availability = np.tile(
        np.asarray((1.0, 0.0, 1.0, 1.0), dtype=np.float64), (identity.rows, 1)
    )
    if not np.array_equal(identity.availability, expected_availability):
        raise GPUOutageGBDTV4Error("v4 challenge is not forced IMU1 outage")
    return identity


def _v3_result_prediction(
    v3_run: Path, result: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference = result["artifacts"]["predictions"]
    segment = _exact_int(
        result.get("task", {}).get("held_out_segment"),
        name="v3 prediction held_out_segment",
        allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
    )
    path = _verify_reference(
        v3_run,
        reference,
        expected_rows=_DEVELOPMENT_SEGMENT_ROWS[segment],
    )
    frame = _read_exact_parquet(
        path, ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS, *_PRED_COLUMNS]
    )
    return frame, {
        "path": _relative(path, v3_run),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
    }


def _v3_development_tasks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recreate v3's frozen development grid without importing its runner."""

    validate_outage_gpu_config(dict(config))
    candidates = {
        str(candidate["candidate_id"]): dict(candidate)
        for candidate in outage_candidate_grid(dict(config))
    }
    segments = tuple(int(value) for value in config["splits"]["development_segments"])
    outages = tuple(
        str(value) for value in config["splits"]["forced_validation_outages"]
    )
    expected_outages = ("imu0_base", "imu1_link_root", "imu2_link_mid", "imu3_tip")
    if outages != expected_outages:
        raise GPUOutageGBDTV4Error("v3 forced outage order changed")
    seeds = tuple(int(value) for value in config["training"]["paired_seed_values"])
    tasks: list[dict[str, Any]] = []
    for candidate_id in TRAINABLE_OUTAGE_CANDIDATE_IDS:
        for held_out_segment in segments:
            for forced_outage_index, outage_name in enumerate(outages):
                for seed_index, model_seed in enumerate(seeds):
                    task_id = (
                        f"development__{candidate_id}__segment{held_out_segment}__"
                        f"drop{forced_outage_index}__seed{seed_index}"
                    )
                    tasks.append(
                        {
                            "task_id": task_id,
                            "phase": "development",
                            "candidate_id": candidate_id,
                            "candidate": deepcopy(candidates[candidate_id]),
                            "held_out_segment": held_out_segment,
                            "forced_outage_name": outage_name,
                            "forced_outage_index": forced_outage_index,
                            "paired_seed_index": seed_index,
                            "model_seed": model_seed,
                        }
                    )
    expected = int(config["selection"]["trainable_seeded_fit_count"])
    if len(tasks) != expected or len({row["task_id"] for row in tasks}) != expected:
        raise GPUOutageGBDTV4Error(
            f"v3 development task grid changed: expected {expected}, received {len(tasks)}"
        )
    return tasks


def _read_v3_results(
    v3_run: Path, v3_config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    tree: list[dict[str, Any]] = []
    for task in _v3_development_tasks(v3_config):
        result_path = (
            v3_run / "development" / "seed_tasks" / str(task["task_id"]) / "result.json"
        )
        result = read_json_object(result_path)
        if not _type_exact_equal(result.get("task"), task):
            raise GPUOutageGBDTV4Error(f"v3 task identity changed: {task['task_id']}")
        frame, prediction_reference = _v3_result_prediction(v3_run, result)
        result_reference = {
            "path": _relative(result_path, v3_run),
            "bytes": int(result_path.stat().st_size),
            "sha256": sha256_file(result_path),
        }
        tree.append(
            {
                "task_id": str(task["task_id"]),
                "candidate_id": str(task["candidate_id"]),
                "held_out_segment": int(task["held_out_segment"]),
                "forced_outage_index": int(task["forced_outage_index"]),
                "paired_seed_index": int(task["paired_seed_index"]),
                "result": result_reference,
                "prediction": prediction_reference,
            }
        )
        if len(frame) == 0:
            raise GPUOutageGBDTV4Error("v3 task prediction is empty")
        results.append(result)
    if len(results) != 240 or len({row["task_id"] for row in tree}) != 240:
        raise GPUOutageGBDTV4Error("v3 task/prediction tree must contain exactly 240 tasks")
    return results, tree


def _selected_v3_components(
    v3_config: Mapping[str, Any], selected_candidate_id: str
) -> list[tuple[str, float]]:
    candidates = {
        str(row["candidate_id"]): row for row in outage_candidate_grid(dict(v3_config))
    }
    candidate = candidates.get(str(selected_candidate_id))
    if candidate is None:
        raise GPUOutageGBDTV4Error("v3 selection names an unknown candidate")
    if candidate.get("kind") == "trainable":
        return [(str(selected_candidate_id), 1.0)]
    components = candidate.get("components")
    if not isinstance(components, list) or not components:
        raise GPUOutageGBDTV4Error("v3 virtual incumbent lacks frozen components")
    parsed = [(str(row["candidate_id"]), float(row["weight"])) for row in components]
    if not math.isclose(sum(weight for _, weight in parsed), 1.0, abs_tol=1e-15):
        raise GPUOutageGBDTV4Error("v3 incumbent component weights changed")
    return parsed


def _snapshot_incumbent_predictions(
    *,
    v3_run: Path,
    v4_run: Path,
    v3_config: Mapping[str, Any],
    selection: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_id = str(selection["selected_candidate_id"])
    components = _selected_v3_components(v3_config, selected_id)
    by_key = {
        (
            str(row["task"]["candidate_id"]),
            int(row["task"]["held_out_segment"]),
            int(row["task"]["forced_outage_index"]),
            int(row["task"]["paired_seed_index"]),
        ): row
        for row in results
    }
    snapshots: list[dict[str, Any]] = []
    for segment in GBDT_V4_DEVELOPMENT_SEGMENTS:
        for outage_index in range(4):
            for seed_index in range(3):
                frames: list[tuple[pd.DataFrame, float]] = []
                origins: list[dict[str, Any]] = []
                for component, weight in components:
                    result = by_key.get((component, segment, outage_index, seed_index))
                    if result is None:
                        raise GPUOutageGBDTV4Error("v3 incumbent seed prediction is missing")
                    frame, origin = _v3_result_prediction(v3_run, result)
                    frames.append((frame, weight))
                    origins.append(
                        {
                            "candidate_id": component,
                            "weight": weight,
                            "prediction": origin,
                        }
                    )
                identity_columns = ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS]
                base = frames[0][0]
                if any(not frame[identity_columns].equals(base[identity_columns]) for frame, _ in frames):
                    raise GPUOutageGBDTV4Error("v3 incumbent components are row-misaligned")
                prediction = sum(
                    weight * frame.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64)
                    for frame, weight in frames
                )
                output = base[identity_columns].copy()
                for axis_index, name in enumerate(_PRED_COLUMNS):
                    output[name] = prediction[:, axis_index]
                path = atomic_write_parquet(
                    v4_run
                    / "incumbent"
                    / "development_predictions"
                    / f"segment{segment}__drop{outage_index}__seed{seed_index}.parquet",
                    output,
                )
                snapshots.append(
                    {
                        "held_out_segment": segment,
                        "forced_outage_index": outage_index,
                        "paired_seed_index": seed_index,
                        "selected_v3_candidate_id": selected_id,
                        "components": origins,
                        "snapshot": _artifact_reference(path, v4_run, rows=len(output)),
                    }
                )
    if len(snapshots) != 48:
        raise GPUOutageGBDTV4Error("v3 incumbent snapshot must contain 48 paired predictions")
    return snapshots


def _v3_prepare_forbidden_paths(v3_run: Path) -> tuple[Path, ...]:
    return (
        v3_run / "final",
        v3_run / "manifest.json",
        v3_run / "challenge" / "label_access_1.started.json",
        v3_run / "challenge" / "label_access_1.completed.json",
        v3_run / "challenge" / "label_access_log.json",
        v3_run / "challenge" / "evaluation.selected_only.parquet",
        v3_run / "challenge" / "report.json",
    )


def _assert_v3_prepare_stability(
    v3_run: Path, before_manifest: Sequence[Mapping[str, Any]]
) -> None:
    """Recheck both the sealed markers and exact parent tree at publication time."""

    if any(path.exists() for path in _v3_prepare_forbidden_paths(v3_run)):
        raise GPUOutageGBDTV4Error(
            "v3 challenge marker appeared while v4 prepare was reading it"
        )
    current_manifest = manifest_files(v3_run)
    if not _type_exact_equal(current_manifest, list(before_manifest)):
        raise GPUOutageGBDTV4Error("v3 run changed while v4 prepare was reading it")


def _claim_prepare_invalid(
    *,
    v3_run: Path,
    v4_run: Path,
    before_manifest: Sequence[Mapping[str, Any]],
    stage: str,
    reason: str,
) -> Path:
    """Permanently invalidate a partially published prepare without overwriting."""

    marker = v4_run / _PREPARE_INVALID_RELATIVE_PATH
    try:
        after_fingerprint = stable_hash(manifest_files(v3_run))
    except Exception:
        after_fingerprint = "unreadable"
    payload = {
        "contract": GBDT_V4_RUN_CONTRACT,
        "status": "prepare_invalid_fail_closed",
        "stage": str(stage),
        "reason": str(reason),
        "v3_manifest_before_sha256": stable_hash(list(before_manifest)),
        "v3_manifest_after_sha256": after_fingerprint,
    }
    try:
        return _exclusive_write_json(marker, payload)
    except GPUOutageGBDTV4Error:
        if marker.is_file():
            return marker
        raise


def prepare_gbdt_v4_run(
    config: dict[str, Any],
    *,
    v3_run_dir: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Validate and freeze a completed, sealed v3 development run without editing it."""

    _require_python_312()
    validate_gbdt_v4_config(config)
    v3_run, v4_run = _validate_prepare_run_paths(v3_run_dir, run_dir)
    if any(path.exists() for path in _v3_prepare_forbidden_paths(v3_run)):
        raise GPUOutageGBDTV4Error(
            "v3 is not at the sealed development-only handoff"
        )
    if not (v3_run / "development_manifest.json").is_file():
        raise GPUOutageGBDTV4Error("v3 development manifest is not complete")

    before_manifest = manifest_files(v3_run)
    validation = _require_sealed_v3_development_validation(
        _validate_sealed_v3_development(v3_run)
    )

    v3_config_path = v3_run / "config.resolved.json"
    v3_config = _load_v3_config(v3_config_path)
    v3_config_sha = outage_gpu_config_sha256(v3_config)
    v3_manifest_path = v3_run / "development_manifest.json"
    v3_manifest = read_json_object(v3_manifest_path)
    if (
        v3_manifest.get("experiment_contract") != OUTAGE_GPU_EXPERIMENT_CONTRACT
        or v3_manifest.get("stage") != "development_complete_challenge_labels_sealed"
        or v3_manifest.get("challenge_labels_sealed") is not True
        or v3_manifest.get("config_sha256") != v3_config_sha
        or _exact_int(
            v3_manifest.get("counts", {}).get("scored_contexts"),
            name="v3 manifest scored_contexts",
        )
        != 96
    ):
        raise GPUOutageGBDTV4Error("v3 development manifest/config binding changed")
    selection_path = v3_run / "development" / "selection.json"
    selection = read_json_object(selection_path)
    if selection.get("challenge_used_for_selection") is not False:
        raise GPUOutageGBDTV4Error("v3 incumbent selection used challenge evidence")
    if v3_manifest.get("selection", {}).get("selected_candidate_id") != selection.get(
        "selected_candidate_id"
    ):
        raise GPUOutageGBDTV4Error("v3 manifest/selection winner changed")

    results, task_tree = _read_v3_results(v3_run, v3_config)
    tree_digest = stable_hash(task_tree)
    workspace = Path(str(config.get("_workspace_root", _runner_workspace_root()))).resolve()
    source_snapshot = gbdt_v4_source_payload(workspace)
    environment_payload = gbdt_v4_environment_payload()
    environment_snapshot = {
        "payload": environment_payload,
        "fingerprint": stable_hash(environment_payload),
    }
    identity = {
        "contract": GBDT_V4_CONTRACT,
        "config_sha256": gbdt_v4_config_sha256(config),
        "v3_config_sha256": v3_config_sha,
        "v3_development_manifest_sha256": sha256_file(v3_manifest_path),
        "v3_selection_sha256": sha256_file(selection_path),
        "v3_task_prediction_tree_sha256": tree_digest,
        "training_source_fingerprint": source_snapshot["fingerprint"],
        "environment_fingerprint": environment_snapshot["fingerprint"],
    }
    prepare_experiment_directory(v4_run, identity=identity, resume=False)
    atomic_write_json(v4_run / "config.resolved.json", public_gbdt_v4_config(config))
    source_snapshot_path = atomic_write_json(
        v4_run / "provenance" / "source_snapshot.json", source_snapshot
    )
    environment_snapshot_path = atomic_write_json(
        v4_run / "provenance" / "environment_snapshot.json", environment_snapshot
    )

    copied_v3_config = _copy_atomic(v3_config_path, v4_run / "incumbent" / "v3_config.resolved.json")
    copied_v3_manifest = _copy_atomic(
        v3_manifest_path, v4_run / "incumbent" / "v3_development_manifest.json"
    )
    copied_v3_selection = _copy_atomic(
        selection_path, v4_run / "incumbent" / "v3_selection.json"
    )
    validation_path = atomic_write_json(
        v4_run / "incumbent" / "v3_validator_payload.json", validation
    )
    tree_path = atomic_write_json(
        v4_run / "incumbent" / "v3_task_prediction_tree.json",
        {"task_count": 240, "tree_sha256": tree_digest, "tasks": task_tree},
    )

    v3_replay_metadata = read_json_object(v3_run / "development" / "replay_inputs.json")
    replay_rows = _exact_int(
        v3_replay_metadata.get("rows"), name="v3 replay rows", minimum=1
    )
    v3_replay_path = _verify_reference(
        v3_run,
        v3_replay_metadata["input_file"],
        expected_rows=replay_rows,
    )
    replay_copy = _copy_atomic(v3_replay_path, v4_run / "inputs" / "development_replay.parquet")
    replay_metadata = {
        "role": "v3_authenticated_development_replay_snapshot",
        "rows": replay_rows,
        "segments": list(v3_replay_metadata["segments"]),
        "segment_rows": dict(v3_replay_metadata["segment_rows"]),
        "sorted_trial_uid_sha256": str(v3_replay_metadata["sorted_trial_uid_sha256"]),
        "feature_names": list(v3_replay_metadata["feature_names"]),
        "label_names": list(v3_replay_metadata["label_names"]),
        "v3_origin": {
            "path": _relative(v3_replay_path, v3_run),
            "bytes": int(v3_replay_path.stat().st_size),
            "sha256": sha256_file(v3_replay_path),
        },
        "input_file": _artifact_reference(replay_copy, v4_run, rows=replay_rows),
    }
    replay_metadata_path = atomic_write_json(
        v4_run / "inputs" / "development_replay.json", replay_metadata
    )

    v3_identity_metadata = read_json_object(v3_run / "challenge" / "identity.json")
    challenge_identity_rows = _exact_int(
        v3_identity_metadata.get("rows"), name="v3 challenge identity rows", minimum=1
    )
    if v3_identity_metadata.get("labels_projected") is not False or v3_identity_metadata.get(
        "recorded_imu1_projected"
    ) is not False:
        raise GPUOutageGBDTV4Error("v3 challenge identity is not label/IMU1 blind")
    v3_identity_path = _verify_reference(
        v3_run,
        v3_identity_metadata["identity_file"],
        expected_rows=challenge_identity_rows,
    )
    identity_copy = _copy_atomic(
        v3_identity_path, v4_run / "inputs" / "challenge_identity_inputs.parquet"
    )
    identity_metadata = {
        "role": "v3_label_blind_challenge_identity_snapshot",
        "rows": challenge_identity_rows,
        "segment_rows": dict(v3_identity_metadata["segment_rows"]),
        "ordered_trial_uid_sha256": str(v3_identity_metadata["ordered_trial_uid_sha256"]),
        "feature_names": list(v3_identity_metadata["feature_names"]),
        "availability_names": list(v3_identity_metadata["availability_names"]),
        "labels_projected": False,
        "recorded_imu1_projected": False,
        "v3_origin": {
            "path": _relative(v3_identity_path, v3_run),
            "bytes": int(v3_identity_path.stat().st_size),
            "sha256": sha256_file(v3_identity_path),
        },
        "identity_file": _artifact_reference(
            identity_copy, v4_run, rows=challenge_identity_rows
        ),
    }
    identity_metadata_path = atomic_write_json(
        v4_run / "inputs" / "challenge_identity.json", identity_metadata
    )

    incumbent_snapshots = _snapshot_incumbent_predictions(
        v3_run=v3_run,
        v4_run=v4_run,
        v3_config=v3_config,
        selection=selection,
        results=results,
    )
    freeze = {
        "contract": GBDT_V4_FREEZE_CONTRACT,
        "status": "prepared_from_validated_sealed_v3_development",
        "v4_config_sha256": gbdt_v4_config_sha256(config),
        "v3_origin_run": str(v3_run),
        "v3_resolved_config_sha256": v3_config_sha,
        "v3_development_manifest_sha256": sha256_file(v3_manifest_path),
        "v3_validator_payload_sha256": stable_hash(validation),
        "v3_selection_sha256": sha256_file(selection_path),
        "v3_selected_candidate_id": str(selection["selected_candidate_id"]),
        "v3_240_task_prediction_tree_sha256": tree_digest,
        "training_source_fingerprint": source_snapshot["fingerprint"],
        "environment_fingerprint": environment_snapshot["fingerprint"],
        "v3_task_count": 240,
        "v3_challenge_labels_sealed_at_prepare": True,
        "v3_run_modified_by_prepare": False,
        "references": {
            "v3_config": _artifact_reference(copied_v3_config, v4_run),
            "v3_development_manifest": _artifact_reference(copied_v3_manifest, v4_run),
            "v3_selection": _artifact_reference(copied_v3_selection, v4_run),
            "v3_validator_payload": _artifact_reference(validation_path, v4_run),
            "v3_task_prediction_tree": _artifact_reference(tree_path, v4_run, tasks=240),
            "development_replay_metadata": _artifact_reference(replay_metadata_path, v4_run),
            "challenge_identity_metadata": _artifact_reference(identity_metadata_path, v4_run),
            "source_snapshot": _artifact_reference(source_snapshot_path, v4_run),
            "environment_snapshot": _artifact_reference(
                environment_snapshot_path, v4_run
            ),
        },
        "incumbent_seed_prediction_count": len(incumbent_snapshots),
        "incumbent_seed_predictions": incumbent_snapshots,
    }
    try:
        _assert_v3_prepare_stability(v3_run, before_manifest)
    except Exception as exc:
        _claim_prepare_invalid(
            v3_run=v3_run,
            v4_run=v4_run,
            before_manifest=before_manifest,
            stage="before_publication",
            reason=str(exc),
        )
        raise

    published = False
    try:
        freeze_path = atomic_write_json(v4_run / "incumbent.freeze.json", freeze)
        published = True
        prepared = {
            "contract": GBDT_V4_RUN_CONTRACT,
            "stage": "prepared_v3_incumbent_frozen_challenge_labels_sealed",
            "config_sha256": gbdt_v4_config_sha256(config),
            "incumbent_freeze": _artifact_reference(freeze_path, v4_run),
            "challenge_labels_sealed": True,
            "files": manifest_files(v4_run),
        }
        atomic_write_json(v4_run / "prepared_manifest.json", prepared)
        _assert_v3_prepare_stability(v3_run, before_manifest)
        validate_gbdt_v4_prepared_run(v4_run)
        _assert_v3_prepare_stability(v3_run, before_manifest)
    except Exception as exc:
        if published or (v4_run / "prepared_manifest.json").exists():
            _claim_prepare_invalid(
                v3_run=v3_run,
                v4_run=v4_run,
                before_manifest=before_manifest,
                stage="after_partial_publication",
                reason=str(exc),
            )
        raise
    return prepared


def gbdt_v4_development_tasks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return 12 fit bundles; each bundle is evaluated under four outages."""

    validate_gbdt_v4_config(config)
    tasks: list[dict[str, Any]] = []
    for segment in GBDT_V4_DEVELOPMENT_SEGMENTS:
        for seed_index, model_seed in enumerate(GBDT_V4_PAIRED_SEEDS):
            tasks.append(
                {
                    "task_id": f"xgb__segment{segment}__seed{seed_index}",
                    "held_out_segment": segment,
                    "paired_seed_index": seed_index,
                    "model_seed": model_seed,
                    "forced_outage_evaluations": list(range(4)),
                }
            )
    if len(tasks) != 12 or len({row["task_id"] for row in tasks}) != 12:
        raise GPUOutageGBDTV4Error("v4 XGB bundle grid changed")
    return tasks


def score_gbdt_v4_candidates(
    context_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the exact outage-v3 A/WS/WM/J and tie-break semantics to v4."""

    validate_gbdt_v4_config(config)
    expected = {
        (candidate, segment, outage)
        for candidate in GBDT_V4_CANDIDATE_IDS
        for segment in GBDT_V4_DEVELOPMENT_SEGMENTS
        for outage in range(4)
    }
    by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in context_rows:
        key = (
            str(row["candidate_id"]),
            _exact_int(
                row.get("held_out_segment"),
                name="scored context held_out_segment",
                allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
            ),
            _exact_int(
                row.get("forced_outage_index"),
                name="scored context forced_outage_index",
                allowed=(0, 1, 2, 3),
            ),
        )
        if key not in expected or key in by_key:
            raise GPUOutageGBDTV4Error(f"unexpected or duplicate v4 context: {key}")
        by_key[key] = row
    if set(by_key) != expected or len(context_rows) != 48:
        missing = sorted(expected.difference(by_key))
        raise GPUOutageGBDTV4Error(
            f"v4 context grid must contain exactly 48 keys; missing {missing[:5]}"
        )

    trace: list[dict[str, Any]] = []
    for priority, candidate in enumerate(GBDT_V4_CANDIDATE_IDS):
        rows = [
            by_key[(candidate, segment, outage)]
            for segment in GBDT_V4_DEVELOPMENT_SEGMENTS
            for outage in range(4)
        ]
        finite = all(
            math.isfinite(
                _metric_real(
                    row["metrics"]["rmse_3d_mm"], name="context rmse_3d_mm"
                )
            )
            and math.isfinite(
                _metric_real(
                    row["metrics"]["p95_3d_error_mm"],
                    name="context p95_3d_error_mm",
                )
            )
            and math.isfinite(
                _metric_real(
                    row["seed_rmse_population_std_mm"],
                    name="context seed_rmse_population_std_mm",
                )
            )
            for row in rows
        )
        if not finite:
            trace.append(
                {
                    "candidate_id": candidate,
                    "listed_priority": priority,
                    "context_count": 16,
                    "missing_contexts": [],
                    "score_status": "positive_infinity",
                }
            )
            continue
        per_segment = [
            {
                "segment_id": segment,
                "mean_rmse_3d_mm": float(
                    np.mean(
                        [
                            _metric_real(
                                by_key[(candidate, segment, outage)]["metrics"][
                                    "rmse_3d_mm"
                                ],
                                name="context rmse_3d_mm",
                            )
                            for outage in range(4)
                        ]
                    )
                ),
            }
            for segment in GBDT_V4_DEVELOPMENT_SEGMENTS
        ]
        per_sensor = [
            {
                "forced_outage_index": outage,
                "mean_rmse_3d_mm": float(
                    np.mean(
                        [
                            _metric_real(
                                by_key[(candidate, segment, outage)]["metrics"][
                                    "rmse_3d_mm"
                                ],
                                name="context rmse_3d_mm",
                            )
                            for segment in GBDT_V4_DEVELOPMENT_SEGMENTS
                        ]
                    )
                ),
            }
            for outage in range(4)
        ]
        rmses = [
            _metric_real(row["metrics"]["rmse_3d_mm"], name="context rmse_3d_mm")
            for row in rows
        ]
        mean_rmse = float(np.mean(rmses))
        worst_segment = max(float(row["mean_rmse_3d_mm"]) for row in per_segment)
        worst_sensor = max(float(row["mean_rmse_3d_mm"]) for row in per_sensor)
        trace.append(
            {
                "candidate_id": candidate,
                "listed_priority": priority,
                "context_count": 16,
                "score_status": "finite",
                "mean_rmse_over_16_contexts_mm": mean_rmse,
                "worst_segment_mean_rmse_mm": worst_segment,
                "worst_sensor_mean_rmse_mm": worst_sensor,
                "outage_robust_j_mm": float((mean_rmse + worst_segment + worst_sensor) / 3.0),
                "mean_p95_3d_mm_over_16_contexts": float(
                    np.mean(
                        [
                            _metric_real(
                                row["metrics"]["p95_3d_error_mm"],
                                name="context p95_3d_error_mm",
                            )
                            for row in rows
                        ]
                    )
                ),
                "mean_seed_rmse_population_std_mm_over_16_contexts": float(
                    np.mean(
                        [
                            _metric_real(
                                row["seed_rmse_population_std_mm"],
                                name="context seed_rmse_population_std_mm",
                            )
                            for row in rows
                        ]
                    )
                ),
                "per_segment": per_segment,
                "per_sensor": per_sensor,
            }
        )
    finite_trace = [row for row in trace if row["score_status"] == "finite"]
    if not finite_trace:
        raise GPUOutageGBDTV4Error("all v4 candidates have non-finite scores")
    tolerance = float(config["selection"]["numeric_equality_abs_tolerance_mm"])
    winner = finite_trace[0]
    for challenger in finite_trace[1:]:
        replace = False
        for metric in (
            "outage_robust_j_mm",
            "mean_p95_3d_mm_over_16_contexts",
            "mean_seed_rmse_population_std_mm_over_16_contexts",
        ):
            left = float(challenger[metric])
            right = float(winner[metric])
            if left < right - tolerance:
                replace = True
                break
            if left > right + tolerance:
                break
        if replace:
            winner = challenger
    return {
        "score_name": "outage_robust_j_mm",
        "context_count_per_candidate": 16,
        "xgb_unique_bundle_count": 12,
        "xgb_evaluation_record_count": 48,
        "trace": trace,
        "selected_candidate_id": str(winner["candidate_id"]),
        "selected_outage_robust_j_mm": float(winner["outage_robust_j_mm"]),
        "challenge_used_for_selection": False,
    }


def _prepared_manifest_files(run_dir: str | Path) -> list[dict[str, Any]]:
    return [
        row
        for row in manifest_files(run_dir)
        if row["path"] != "prepared_manifest.json"
        and row["path"] != "development_manifest.json"
        and not row["path"].startswith("development/")
        and not row["path"].startswith("provenance/worker_ready/development/")
        and not row["path"].startswith("final/")
        and not row["path"].startswith("challenge/")
        and not row["path"].startswith("recovery/")
        and row["path"]
        not in {
            "provenance/development_gpu_workers.json",
            "provenance/final_incumbent_gpu_workers.json",
            "provenance/final_xgb_gpu_workers.json",
        }
        and row["path"] != "manifest.json"
    ]


def validate_gbdt_v4_prepared_run(run_dir: str | Path) -> dict[str, Any]:
    """Validate the frozen v3 incumbent using only copied durable artifacts."""

    _require_python_312()
    run = Path(run_dir).resolve()
    _assert_development_only_run(run)
    config = read_json_object(run / "config.resolved.json")
    validate_gbdt_v4_config(config)
    config_sha = gbdt_v4_config_sha256(config)
    current_source, stored_environment = _validate_v4_source_provenance(run)
    freeze_path = run / "incumbent.freeze.json"
    freeze = read_json_object(freeze_path)
    if (
        freeze.get("contract") != GBDT_V4_FREEZE_CONTRACT
        or freeze.get("status") != "prepared_from_validated_sealed_v3_development"
        or freeze.get("v4_config_sha256") != config_sha
        or freeze.get("v3_challenge_labels_sealed_at_prepare") is not True
        or freeze.get("v3_run_modified_by_prepare") is not False
    ):
        raise GPUOutageGBDTV4Error("v4 incumbent freeze identity/status changed")
    references = freeze.get("references")
    expected_reference_names = {
        "v3_config",
        "v3_development_manifest",
        "v3_selection",
        "v3_validator_payload",
        "v3_task_prediction_tree",
        "development_replay_metadata",
        "challenge_identity_metadata",
        "source_snapshot",
        "environment_snapshot",
    }
    if not isinstance(references, Mapping) or set(references) != expected_reference_names:
        raise GPUOutageGBDTV4Error("v4 incumbent freeze references are missing")
    resolved_paths = {
        key: _verify_reference(
            run,
            value,
            expected_tasks=(240 if key == "v3_task_prediction_tree" else _REFERENCE_COUNT_UNSET),
        )
        for key, value in references.items()
    }
    if (
        freeze.get("training_source_fingerprint") != current_source["fingerprint"]
        or freeze.get("environment_fingerprint") != stored_environment["fingerprint"]
    ):
        raise GPUOutageGBDTV4Error("v4 freeze source/environment binding changed")
    run_identity = read_json_object(run / "_state" / "run_identity.json")
    expected_identity = {
        "contract": GBDT_V4_CONTRACT,
        "config_sha256": config_sha,
        "v3_config_sha256": freeze["v3_resolved_config_sha256"],
        "v3_development_manifest_sha256": freeze["v3_development_manifest_sha256"],
        "v3_selection_sha256": freeze["v3_selection_sha256"],
        "v3_task_prediction_tree_sha256": freeze[
            "v3_240_task_prediction_tree_sha256"
        ],
        "training_source_fingerprint": current_source["fingerprint"],
        "environment_fingerprint": stored_environment["fingerprint"],
    }
    if run_identity != expected_identity:
        raise GPUOutageGBDTV4Error("v4 run identity/source/environment binding changed")
    v3_config = _load_v3_config(resolved_paths["v3_config"])
    if outage_gpu_config_sha256(v3_config) != str(freeze["v3_resolved_config_sha256"]):
        raise GPUOutageGBDTV4Error("copied v3 config semantic SHA changed")
    if sha256_file(resolved_paths["v3_development_manifest"]) != str(
        freeze["v3_development_manifest_sha256"]
    ):
        raise GPUOutageGBDTV4Error("copied v3 development manifest SHA changed")
    v3_manifest = read_json_object(resolved_paths["v3_development_manifest"])
    if (
        v3_manifest.get("stage") != "development_complete_challenge_labels_sealed"
        or v3_manifest.get("challenge_labels_sealed") is not True
        or v3_manifest.get("config_sha256") != freeze["v3_resolved_config_sha256"]
        or _exact_int(
            v3_manifest.get("counts", {}).get("scored_contexts"),
            name="copied v3 manifest scored_contexts",
        )
        != 96
    ):
        raise GPUOutageGBDTV4Error("copied v3 development manifest was not sealed")
    validator_payload = read_json_object(resolved_paths["v3_validator_payload"])
    if stable_hash(validator_payload) != str(freeze["v3_validator_payload_sha256"]):
        raise GPUOutageGBDTV4Error("copied v3 validator payload hash changed")
    if (
        validator_payload.get("status") != "ok"
        or validator_payload.get("challenge_labels_sealed") is not True
        or _exact_int(
            validator_payload.get("development_seed_fits"),
            name="copied v3 validator development_seed_fits",
        )
        != 240
        or _exact_int(
            validator_payload.get("scored_contexts"),
            name="copied v3 validator scored_contexts",
        )
        != 96
    ):
        raise GPUOutageGBDTV4Error("copied v3 validator payload is not sealed/complete")
    selection_path = resolved_paths["v3_selection"]
    if sha256_file(selection_path) != str(freeze["v3_selection_sha256"]):
        raise GPUOutageGBDTV4Error("copied v3 selection SHA changed")
    selection = read_json_object(selection_path)
    if (
        selection.get("selected_candidate_id") != freeze["v3_selected_candidate_id"]
        or selection.get("challenge_used_for_selection") is not False
    ):
        raise GPUOutageGBDTV4Error("copied v3 incumbent selection changed")
    tree_payload = read_json_object(resolved_paths["v3_task_prediction_tree"])
    tree = tree_payload.get("tasks")
    if (
        not isinstance(tree, list)
        or len(tree) != 240
        or _exact_int(
            tree_payload.get("task_count"), name="copied v3 task_count"
        )
        != 240
        or stable_hash(tree) != str(freeze["v3_240_task_prediction_tree_sha256"])
        or tree_payload.get("tree_sha256") != freeze["v3_240_task_prediction_tree_sha256"]
    ):
        raise GPUOutageGBDTV4Error("copied v3 240-task prediction tree changed")
    if len({str(row["task_id"]) for row in tree}) != 240:
        raise GPUOutageGBDTV4Error("copied v3 task tree contains duplicate IDs")

    snapshots = freeze.get("incumbent_seed_predictions")
    if (
        not isinstance(snapshots, list)
        or len(snapshots) != 48
        or _exact_int(
            freeze.get("incumbent_seed_prediction_count"),
            name="incumbent seed prediction count",
        )
        != 48
    ):
        raise GPUOutageGBDTV4Error("v4 incumbent snapshot grid changed")
    expected_keys = {
        (segment, outage, seed)
        for segment in GBDT_V4_DEVELOPMENT_SEGMENTS
        for outage in range(4)
        for seed in range(3)
    }
    seen: set[tuple[int, int, int]] = set()
    for snapshot in snapshots:
        key = (
            _exact_int(
                snapshot.get("held_out_segment"),
                name="incumbent snapshot held_out_segment",
                allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
            ),
            _exact_int(
                snapshot.get("forced_outage_index"),
                name="incumbent snapshot forced_outage_index",
                allowed=(0, 1, 2, 3),
            ),
            _exact_int(
                snapshot.get("paired_seed_index"),
                name="incumbent snapshot paired_seed_index",
                allowed=(0, 1, 2),
            ),
        )
        if key in seen or key not in expected_keys:
            raise GPUOutageGBDTV4Error("duplicate/unexpected incumbent seed snapshot")
        seen.add(key)
        frame = _read_exact_parquet(
            _verify_reference(
                run,
                snapshot["snapshot"],
                expected_rows=_DEVELOPMENT_SEGMENT_ROWS[key[0]],
            ),
            ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS, *_PRED_COLUMNS],
        )
        if (
            len(frame) != _DEVELOPMENT_SEGMENT_ROWS[key[0]]
            or not frame["segment_id"].eq(key[0]).all()
            or not np.isfinite(
                frame.loc[:, [*_TRUE_COLUMNS, *_PRED_COLUMNS]].to_numpy(dtype=np.float64)
            ).all()
        ):
            raise GPUOutageGBDTV4Error("incumbent seed snapshot content changed")
    if seen != expected_keys:
        raise GPUOutageGBDTV4Error("incumbent seed snapshot grid is incomplete")

    development = _load_development_replay(run)
    challenge = _load_challenge_identity(run)
    if development.rows != 6635 or challenge.rows != 3295:
        raise GPUOutageGBDTV4Error("v4 copied replay populations changed")
    prepared = read_json_object(run / "prepared_manifest.json")
    prepared_freeze = prepared.get("incumbent_freeze")
    verified_prepared_freeze = _verify_reference(run, prepared_freeze)
    if (
        set(prepared)
        != {
            "contract",
            "stage",
            "config_sha256",
            "incumbent_freeze",
            "challenge_labels_sealed",
            "files",
        }
        or prepared.get("contract") != GBDT_V4_RUN_CONTRACT
        or prepared.get("stage") != "prepared_v3_incumbent_frozen_challenge_labels_sealed"
        or prepared.get("config_sha256") != config_sha
        or prepared.get("challenge_labels_sealed") is not True
        or verified_prepared_freeze != freeze_path
        or not _type_exact_equal(
            prepared.get("incumbent_freeze"), _artifact_reference(freeze_path, run)
        )
        or not _type_exact_equal(prepared.get("files"), _prepared_manifest_files(run))
    ):
        raise GPUOutageGBDTV4Error("v4 prepared manifest changed")
    return {
        "status": "ok",
        "stage": "prepared_v3_incumbent_frozen_challenge_labels_sealed",
        "config_sha256": config_sha,
        "v3_selected_candidate_id": str(freeze["v3_selected_candidate_id"]),
        "v3_task_prediction_count": 240,
        "incumbent_seed_prediction_count": 48,
        "development_rows": development.rows,
        "challenge_rows_identity_only": challenge.rows,
        "challenge_labels_sealed": True,
        "source_archives_opened_by_validator": False,
    }


def _bundle_checkpoint_reference(directory: str | Path, run_dir: str | Path) -> dict[str, Any]:
    root = Path(directory)
    _assert_exact_directory_children(root, _XGB_BUNDLE_MEMBERS, label="XGB checkpoint")
    members = []
    for name in _XGB_BUNDLE_MEMBERS:
        path = root / name
        members.append(_artifact_reference(path, run_dir))
    return {
        "path": _relative(root, run_dir),
        "member_count": 4,
        "members": members,
        "tree_sha256": stable_hash(members),
    }


def _verify_bundle_checkpoint(
    run_dir: str | Path, reference: Mapping[str, Any]
) -> Path:
    root = _safe_member(run_dir, str(reference.get("path", "")))
    if not root.is_dir() or _exact_int(
        reference.get("member_count"), name="bundle member_count", minimum=0
    ) != 4:
        raise GPUOutageGBDTV4Error("XGB checkpoint directory/reference changed")
    expected = _bundle_checkpoint_reference(root, run_dir)
    if not _type_exact_equal(reference, expected):
        raise GPUOutageGBDTV4Error("XGB checkpoint tree hash changed")
    return root


def _validate_loaded_xgb_bundle_binding(
    bundle: Any,
    result: Mapping[str, Any],
    *,
    expected_config_sha256: str,
) -> None:
    """Bind a loaded checkpoint back to its exact immutable result record."""

    if (
        result.get("config_sha256") != expected_config_sha256
        or str(bundle.bundle_id) != result.get("bundle_id")
        or str(bundle.bundle_sha256) != result.get("bundle_sha256")
        or not _type_exact_equal(result.get("bundle_metadata"), bundle.metadata())
    ):
        raise GPUOutageGBDTV4Error("v4 XGB bundle/result/config binding changed")


def _expected_bundle_save_audit(bundle: Any, bundle_dir: Path) -> dict[str, Any]:
    from .gpu_outage_gbdt_v4_training import validate_xgb_bundle_models

    return {
        **validate_xgb_bundle_models(bundle),
        "checkpoint_directory": str(bundle_dir),
        "metadata_sha256": sha256_file(bundle_dir / "metadata.json"),
    }


def _validate_bundle_save_audit(
    bundle: Any, bundle_dir: Path, result: Mapping[str, Any]
) -> None:
    if not _type_exact_equal(
        result.get("bundle_save_audit"),
        _expected_bundle_save_audit(bundle, bundle_dir),
    ):
        raise GPUOutageGBDTV4Error("v4 XGB bundle save audit changed")


def _validate_xgb_result_shape(
    *,
    run_dir: str | Path,
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    """Validate the exact development result and its pre-fit worker receipt."""

    if phase != "development":
        raise GPUOutageGBDTV4Error(
            "outage-v4 only authenticates development XGB results"
        )

    common = {
        "contract",
        "config_sha256",
        "task",
        "gpu_id",
        "bundle_id",
        "bundle_sha256",
        "bundle_metadata",
        "bundle_save_audit",
        "training_source_fingerprint",
        "environment_fingerprint",
        "artifacts",
        "challenge_labels_accessed",
    }
    expected_keys = common | {
        "evaluations",
        "fit_count",
        "evaluation_count",
    }
    expected_artifacts = {"task_execution", "preprocessing", "bundle"}
    if "result_recovery" in result:
        expected_keys.add("result_recovery")
        if result.get("result_recovery") != {
            "mode": "authenticated_complete_artifacts_without_refit"
        }:
            raise GPUOutageGBDTV4Error("v4 XGB result recovery marker changed")
    artifacts = result.get("artifacts")
    source = read_json_object(Path(run_dir) / "provenance" / "source_snapshot.json")
    environment = read_json_object(
        Path(run_dir) / "provenance" / "environment_snapshot.json"
    )
    if (
        set(result) != expected_keys
        or result.get("contract") != GBDT_V4_BUNDLE_RESULT_CONTRACT
        or result.get("config_sha256") != gbdt_v4_config_sha256(config)
        or not _type_exact_equal(result.get("task"), dict(task))
        or result.get("challenge_labels_accessed") is not False
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != expected_artifacts
        or result.get("training_source_fingerprint") != source.get("fingerprint")
        or result.get("environment_fingerprint") != environment.get("fingerprint")
    ):
        raise GPUOutageGBDTV4Error("v4 XGB result envelope changed")
    if (
        _exact_int(result.get("fit_count"), name="result fit_count") != 1
        or _exact_int(
            result.get("evaluation_count"), name="result evaluation_count"
        )
        != 4
    ):
        raise GPUOutageGBDTV4Error("v4 XGB development result counts changed")
    execution_path, execution = _validate_xgb_task_execution(
        run_dir=run_dir,
        reference_or_path=artifacts["task_execution"],
        config=config,
        task=task,
        phase=phase,
    )
    result_gpu_id = _exact_int(
        result.get("gpu_id"), name="result gpu_id", allowed=(0, 1, 2, 3)
    )
    if result_gpu_id != execution["physical_gpu_id"]:
        raise GPUOutageGBDTV4Error("v4 XGB result/worker GPU binding changed")
    return {"execution_path": execution_path, "execution": execution}


def _xgb_task_dir(run_dir: str | Path, task: Mapping[str, Any]) -> Path:
    return Path(run_dir) / "development" / "xgb_bundles" / str(task["task_id"])


def _xgb_result_path(run_dir: str | Path, task: Mapping[str, Any]) -> Path:
    return _xgb_task_dir(run_dir, task) / "result.json"


def _assert_development_closed_world(
    run_dir: str | Path, config: Mapping[str, Any]
) -> None:
    """Reject undeclared development bundles, drops, contexts, tasks, and files."""

    run = Path(run_dir)
    _assert_exact_directory_children(
        run / "development",
        {
            "xgb_bundles": "directory",
            "contexts": "directory",
            "context_metrics.json": "file",
            "selection.json": "file",
        },
        label="v4 development",
    )
    tasks = gbdt_v4_development_tasks(config)
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    task_ids = [str(task["task_id"]) for task in tasks]
    _assert_exact_directory_children(
        run / "development" / "xgb_bundles",
        {task_id: "directory" for task_id in task_ids},
        label="v4 development XGB task grid",
    )
    task_members = {
        "bundle": "directory",
        "task_execution.json": "file",
        "preprocessing.json": "file",
        "result.json": "file",
        **{f"predictions.drop{index}.parquet": "file" for index in range(4)},
    }
    for task in tasks:
        _assert_exact_directory_children(
            _xgb_task_dir(run, task),
            task_members,
            label=f"v4 development task {task['task_id']}",
        )
    _assert_exact_directory_children(
        run / "development" / "contexts",
        {candidate: "directory" for candidate in GBDT_V4_CANDIDATE_IDS},
        label="v4 development context candidates",
    )
    expected_contexts = {
        f"segment{segment}__drop{outage}.parquet": "file"
        for segment in GBDT_V4_DEVELOPMENT_SEGMENTS
        for outage in range(4)
    }
    for candidate in GBDT_V4_CANDIDATE_IDS:
        _assert_exact_directory_children(
            run / "development" / "contexts" / candidate,
            expected_contexts,
            label=f"v4 development contexts {candidate}",
        )
    _validate_quarantine_phase(
        run,
        phase="development",
        allowed_task_ids=task_ids,
    )
    _validate_recovery_root(
        run,
        allowed_phases={"development": task_ids},
    )


def _execute_xgb_development_task(
    data: DevelopmentData,
    task: Mapping[str, Any],
    config: Mapping[str, Any],
    run_dir: str | Path,
    *,
    gpu_id: int,
    fit_fn: Callable[..., Any] | None = None,
    predict_fn: Callable[..., np.ndarray] | None = None,
    save_fn: Callable[..., Mapping[str, Any]] | None = None,
    allow_cpu_for_testing: bool = False,
) -> dict[str, Any]:
    """Fit one bundle once and materialize its four evaluation-only outages."""

    if fit_fn is None or predict_fn is None or save_fn is None:
        from .gpu_outage_gbdt_v4_training import (
            fit_xgb_bundle,
            predict_xgb_bundle_forced_outage,
            save_xgb_bundle,
        )

        fit_fn = fit_fn or fit_xgb_bundle
        predict_fn = predict_fn or predict_xgb_bundle_forced_outage
        save_fn = save_fn or save_xgb_bundle
    segment = _exact_int(
        task.get("held_out_segment"),
        name="development task held_out_segment",
        allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
    )
    paired_seed_index = _exact_int(
        task.get("paired_seed_index"),
        name="development task paired_seed_index",
        allowed=(0, 1, 2),
    )
    model_seed = _exact_int(
        task.get("model_seed"),
        name="development task model_seed",
        allowed=GBDT_V4_PAIRED_SEEDS,
    )
    physical_gpu_id = _exact_int(
        gpu_id, name="development task gpu_id", allowed=(0, 1, 2, 3)
    )
    eval_rows = np.asarray(data.segment_ids == segment, dtype=bool)
    train_rows = ~eval_rows
    if not train_rows.any() or not eval_rows.any():
        raise GPUOutageGBDTV4Error("XGB development task has an empty train/eval role")
    preprocessor = _fit_preprocessor(
        data,
        train_rows,
        forbidden_rows=eval_rows,
        split_context=f"outage_v4_gbdt_segment{segment}_seed{paired_seed_index}",
    )
    X_train = preprocessor.transform_features(data.X[train_rows]).astype(np.float32)
    y_train = preprocessor.transform_targets(data.y[train_rows]).astype(np.float32)
    X_eval = preprocessor.transform_features(data.X[eval_rows]).astype(np.float32)
    device = "cpu" if allow_cpu_for_testing else "cuda:0"
    task_dir = _xgb_task_dir(run_dir, task)
    execution_path, source_snapshot, environment_snapshot = _begin_xgb_task_execution(
        run_dir=run_dir,
        task_dir=task_dir,
        config=config,
        task=task,
        phase="development",
        gpu_id=physical_gpu_id,
    )
    preprocessor_path = atomic_write_json(
        task_dir / "preprocessing.json", preprocessor.metadata()
    )
    bundle = fit_fn(
        X_train,
        y_train,
        data.trial_uids[train_rows],
        data.segment_ids[train_rows],
        held_out_segment=segment,
        paired_seed=model_seed,
        device=device,
        feature_names=FUSION_COLUMNS,
        allow_cpu_for_testing=allow_cpu_for_testing,
    )
    checkpoint_dir = task_dir / "bundle"
    save_audit = dict(save_fn(bundle, checkpoint_dir))
    evaluations: list[dict[str, Any]] = []
    truth = data.y[eval_rows]
    for outage_index in range(4):
        prediction_scaled = np.asarray(
            predict_fn(bundle, X_eval, outage_index, device=device), dtype=np.float64
        )
        prediction = preprocessor.inverse_targets(prediction_scaled)
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise GPUOutageGBDTV4Error("XGB development prediction is invalid")
        metrics = regression_metrics(truth, prediction)
        frame = pd.DataFrame(
            {
                "trial_uid": data.trial_uids[eval_rows],
                "segment_id": data.segment_ids[eval_rows],
                "target_id": data.target_ids[eval_rows],
                **{name: truth[:, axis] for axis, name in enumerate(_TRUE_COLUMNS)},
                **{name: prediction[:, axis] for axis, name in enumerate(_PRED_COLUMNS)},
            }
        )
        path = atomic_write_parquet(
            task_dir / f"predictions.drop{outage_index}.parquet", frame
        )
        evaluations.append(
            {
                "forced_outage_index": outage_index,
                "metrics": metrics,
                "prediction_file": _artifact_reference(path, run_dir, rows=len(frame)),
            }
        )
    payload = {
        "contract": GBDT_V4_BUNDLE_RESULT_CONTRACT,
        "config_sha256": gbdt_v4_config_sha256(config),
        "task": dict(task),
        "gpu_id": physical_gpu_id,
        "bundle_id": str(bundle.bundle_id),
        "bundle_sha256": str(bundle.bundle_sha256),
        "bundle_metadata": bundle.metadata(),
        "bundle_save_audit": save_audit,
        "training_source_fingerprint": source_snapshot["fingerprint"],
        "environment_fingerprint": environment_snapshot["fingerprint"],
        "artifacts": {
            "task_execution": _artifact_reference(execution_path, run_dir),
            "preprocessing": _artifact_reference(preprocessor_path, run_dir),
            "bundle": _bundle_checkpoint_reference(checkpoint_dir, run_dir),
        },
        "evaluations": evaluations,
        "fit_count": 1,
        "evaluation_count": 4,
        "challenge_labels_accessed": False,
    }
    result_path = atomic_write_json(task_dir / "result.json", payload)
    payload["result_file"] = _artifact_reference(result_path, run_dir)
    return payload


def _load_valid_xgb_result(
    run_dir: str | Path, task: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _xgb_result_path(run_dir, task)
    if not path.is_file():
        return None
    try:
        result = read_json_object(path)
        config = read_json_object(Path(run_dir) / "config.resolved.json")
        _validate_xgb_result_shape(
            run_dir=run_dir,
            result=result,
            config=config,
            task=task,
            phase="development",
        )
        preprocessing_path = _verify_reference(
            run_dir, result["artifacts"]["preprocessing"]
        )
        preprocessing = _preprocessor_from_metadata(
            read_json_object(preprocessing_path)
        )
        bundle_dir = _verify_bundle_checkpoint(
            run_dir, result["artifacts"]["bundle"]
        )
        from .gpu_outage_gbdt_v4_training import load_xgb_bundle

        bundle = load_xgb_bundle(
            bundle_dir,
            expected_held_out_segment=_exact_int(
                task.get("held_out_segment"),
                name="result task held_out_segment",
                allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
            ),
            expected_paired_seed=_exact_int(
                task.get("model_seed"),
                name="result task model_seed",
                allowed=GBDT_V4_PAIRED_SEEDS,
            ),
            expected_source_sorted_trial_uid_sha256=preprocessing.leakage_audit_[
                "fit_trial_uid_sha256"
            ],
        )
        if bundle.training_device != "cuda:0":
            raise GPUOutageGBDTV4Error(
                "production v4 result requires an isolated cuda:0 bundle"
            )
        _validate_loaded_xgb_bundle_binding(
            bundle,
            result,
            expected_config_sha256=gbdt_v4_config_sha256(config),
        )
        _validate_bundle_save_audit(bundle, bundle_dir, result)
        evaluations = result.get("evaluations")
        if not isinstance(evaluations, list) or len(evaluations) != 4:
            return None
        for expected_outage, row in enumerate(evaluations):
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {"forced_outage_index", "metrics", "prediction_file"}
                or _exact_int(
                    row.get("forced_outage_index"),
                    name="result forced_outage_index",
                    allowed=(0, 1, 2, 3),
                )
                != expected_outage
            ):
                return None
            expected_segment = _exact_int(
                task.get("held_out_segment"),
                name="result prediction held_out_segment",
                allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
            )
            _verify_reference(
                run_dir,
                row["prediction_file"],
                expected_rows=_DEVELOPMENT_SEGMENT_ROWS[expected_segment],
            )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError, GPUOutageGBDTV4Error):
        return None
    result["result_file"] = _artifact_reference(path, run_dir)
    return result


def _quarantine_task_directory(
    *,
    run_dir: str | Path,
    task_dir: str | Path,
    phase: str,
    task_id: str,
    reason: str,
) -> dict[str, Any] | None:
    """Move an incomplete/invalid attempt aside without deleting evidence."""

    _recover_quarantine_intents(run_dir, phase=phase)
    source = Path(task_dir)
    if not source.exists():
        return None
    if not source.is_dir() or source.is_symlink():
        raise GPUOutageGBDTV4Error("unsafe v4 task path cannot be quarantined")
    before = manifest_files(source)
    root = Path(run_dir) / "recovery" / phase
    root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        name = f"{task_id}__attempt{attempt:04d}"
        destination = root / name
        audit_path = root / f"{name}.audit.json"
        intent_path = root / f"{name}.intent.json"
        if (
            not destination.exists()
            and not audit_path.exists()
            and not intent_path.exists()
        ):
            break
        attempt += 1
    intent = {
        "contract": GBDT_V4_TASK_EXECUTION_CONTRACT,
        "status": "prepared_before_quarantine_move",
        "phase": phase,
        "task_id": task_id,
        "attempt_index": attempt,
        "reason": str(reason),
        "source_path": _relative(source, run_dir),
        "quarantined_path": _relative(destination, run_dir),
        "audit_path": _relative(audit_path, run_dir),
        "member_count": len(before),
        "members": before,
        "tree_sha256": stable_hash(before),
    }
    _durable_atomic_write_json(intent_path, intent)
    return _finalize_quarantine_intent(run_dir, intent_path)


def _fsync_directory(path: str | Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_atomic_write_json(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    """Atomically write and fsync a quarantine journal/audit before state changes."""

    output = atomic_write_json(path, payload)
    descriptor = os.open(output, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(output.parent)
    return output


def _finalize_quarantine_intent(
    run_dir: str | Path, intent_path: str | Path
) -> dict[str, Any]:
    """Deterministically finish a journaled move and durable audit write."""

    run = Path(run_dir)
    intent_file = Path(intent_path)
    intent = read_json_object(intent_file)
    expected_keys = {
        "contract",
        "status",
        "phase",
        "task_id",
        "attempt_index",
        "reason",
        "source_path",
        "quarantined_path",
        "audit_path",
        "member_count",
        "members",
        "tree_sha256",
    }
    if (
        set(intent) != expected_keys
        or intent.get("contract") != GBDT_V4_TASK_EXECUTION_CONTRACT
        or intent.get("status") != "prepared_before_quarantine_move"
        or not isinstance(intent.get("phase"), str)
        or not isinstance(intent.get("task_id"), str)
        or not isinstance(intent.get("reason"), str)
        or not intent.get("reason")
        or not isinstance(intent.get("members"), list)
        or intent.get("tree_sha256") != stable_hash(intent.get("members"))
    ):
        raise GPUOutageGBDTV4Error("v4 quarantine intent changed")
    phase = str(intent["phase"])
    task_id = str(intent["task_id"])
    if (
        intent_file.parent.resolve().parent != (run / "recovery").resolve()
        or intent_file.parent.name != phase
    ):
        raise GPUOutageGBDTV4Error("v4 quarantine intent phase changed")
    attempt = _exact_int(
        intent.get("attempt_index"), name="quarantine attempt_index", minimum=1
    )
    member_count = _exact_int(
        intent.get("member_count"), name="quarantine member_count", minimum=0
    )
    members = intent["members"]
    if member_count != len(members):
        raise GPUOutageGBDTV4Error("v4 quarantine intent member count changed")
    name = f"{task_id}__attempt{attempt:04d}"
    expected_source = f"development/xgb_bundles/{task_id}"
    expected_destination = f"recovery/{phase}/{name}"
    expected_audit = f"recovery/{phase}/{name}.audit.json"
    if (
        intent_file.name != f"{name}.intent.json"
        or intent.get("source_path") != expected_source
        or intent.get("quarantined_path") != expected_destination
        or intent.get("audit_path") != expected_audit
    ):
        raise GPUOutageGBDTV4Error("v4 quarantine intent path identity changed")
    source = _safe_member(run, expected_source)
    destination = _safe_member(run, expected_destination)
    audit_path = _safe_member(run, expected_audit)
    source_exists = source.exists()
    destination_exists = destination.exists()
    if source_exists == destination_exists:
        raise GPUOutageGBDTV4Error("v4 quarantine intent move state is ambiguous")
    evidence = source if source_exists else destination
    if (
        not evidence.is_dir()
        or evidence.is_symlink()
        or not _type_exact_equal(manifest_files(evidence), members)
    ):
        raise GPUOutageGBDTV4Error("v4 quarantine intent evidence changed")
    if source_exists:
        os.replace(source, destination)
        _fsync_directory(source.parent)
        _fsync_directory(destination.parent)
    if not _type_exact_equal(manifest_files(destination), members):
        raise GPUOutageGBDTV4Error("v4 quarantine move changed task evidence")
    audit = {
        "contract": GBDT_V4_TASK_EXECUTION_CONTRACT,
        "status": "quarantined_without_deletion",
        "phase": phase,
        "task_id": task_id,
        "attempt_index": attempt,
        "reason": str(intent["reason"]),
        "quarantined_path": expected_destination,
        "member_count": member_count,
        "members": members,
        "tree_sha256": intent["tree_sha256"],
    }
    if audit_path.is_file():
        if not _type_exact_equal(read_json_object(audit_path), audit):
            raise GPUOutageGBDTV4Error("v4 quarantine audit changed")
    else:
        _durable_atomic_write_json(audit_path, audit)
    intent_file.unlink()
    _fsync_directory(intent_file.parent)
    return audit


def _recover_quarantine_intents(run_dir: str | Path, *, phase: str) -> None:
    root = Path(run_dir) / "recovery" / phase
    if not root.exists():
        return
    if not root.is_dir() or root.is_symlink():
        raise GPUOutageGBDTV4Error("v4 recovery phase directory is unsafe")
    intents = sorted(root.glob("*.intent.json"))
    if any(path.is_symlink() or not path.is_file() for path in intents):
        raise GPUOutageGBDTV4Error("v4 quarantine intent path is unsafe")
    for intent_path in intents:
        _finalize_quarantine_intent(run_dir, intent_path)


def _validate_quarantine_phase(
    run_dir: str | Path, *, phase: str, allowed_task_ids: Sequence[str]
) -> list[dict[str, Any]]:
    _recover_quarantine_intents(run_dir, phase=phase)
    root = Path(run_dir) / "recovery" / phase
    if not root.exists():
        return []
    if not root.is_dir() or root.is_symlink():
        raise GPUOutageGBDTV4Error("v4 recovery phase directory is unsafe")
    allowed = {str(value) for value in allowed_task_ids}
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    files = sorted(path for path in root.iterdir() if path.is_file())
    if any(path.is_symlink() for path in (*directories, *files)):
        raise GPUOutageGBDTV4Error("v4 recovery evidence contains a symlink")
    expected_files = {f"{path.name}.audit.json" for path in directories}
    if {path.name for path in files} != expected_files:
        raise GPUOutageGBDTV4Error("v4 recovery audit/directory pairs changed")
    audits: list[dict[str, Any]] = []
    seen_indices: dict[str, list[int]] = {}
    for directory in directories:
        prefix, separator, suffix = directory.name.rpartition("__attempt")
        if not separator or prefix not in allowed or len(suffix) != 4 or not suffix.isdigit():
            raise GPUOutageGBDTV4Error("v4 recovery task identity changed")
        attempt_index = int(suffix)
        if attempt_index < 1:
            raise GPUOutageGBDTV4Error("v4 recovery attempt index changed")
        members = manifest_files(directory)
        audit = read_json_object(root / f"{directory.name}.audit.json")
        if (
            set(audit)
            != {
                "contract",
                "status",
                "phase",
                "task_id",
                "attempt_index",
                "reason",
                "quarantined_path",
                "member_count",
                "members",
                "tree_sha256",
            }
            or audit.get("contract") != GBDT_V4_TASK_EXECUTION_CONTRACT
            or audit.get("status") != "quarantined_without_deletion"
            or audit.get("phase") != phase
            or audit.get("task_id") != prefix
            or _exact_int(
                audit.get("attempt_index"), name="quarantine audit attempt_index"
            )
            != attempt_index
            or not isinstance(audit.get("reason"), str)
            or not audit.get("reason")
            or audit.get("quarantined_path") != _relative(directory, run_dir)
            or _exact_int(
                audit.get("member_count"), name="quarantine audit member_count"
            )
            != len(members)
            or not _type_exact_equal(audit.get("members"), members)
            or audit.get("tree_sha256") != stable_hash(members)
        ):
            raise GPUOutageGBDTV4Error("v4 recovery audit changed")
        seen_indices.setdefault(prefix, []).append(attempt_index)
        audits.append(audit)
    if any(sorted(values) != list(range(1, max(values) + 1)) for values in seen_indices.values()):
        raise GPUOutageGBDTV4Error("v4 recovery attempt sequence has a gap")
    return audits


def _validate_recovery_root(
    run_dir: str | Path,
    *,
    allowed_phases: Mapping[str, Sequence[str]],
) -> None:
    root = Path(run_dir) / "recovery"
    if not root.exists():
        return
    if not root.is_dir() or root.is_symlink():
        raise GPUOutageGBDTV4Error("v4 recovery root is unsafe")
    children = {path.name: path for path in root.iterdir()}
    if any(path.is_symlink() or not path.is_dir() for path in children.values()):
        raise GPUOutageGBDTV4Error("v4 recovery root members changed")
    if not set(children).issubset(allowed_phases):
        raise GPUOutageGBDTV4Error("v4 recovery contains an undeclared phase")
    for phase in children:
        _validate_quarantine_phase(
            run_dir,
            phase=phase,
            allowed_task_ids=allowed_phases[phase],
        )


def _reconstruct_xgb_development_result(
    run_dir: str | Path, task: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Recover a fully written bundle/prediction set whose result write was lost."""

    task_dir = _xgb_task_dir(run_dir, task)
    result_path = task_dir / "result.json"
    if result_path.exists() or not task_dir.exists():
        return None
    expected_members = {
        "bundle": "directory",
        "task_execution.json": "file",
        "preprocessing.json": "file",
        **{f"predictions.drop{index}.parquet": "file" for index in range(4)},
    }
    _assert_exact_directory_children(
        task_dir,
        expected_members,
        label=f"recoverable v4 development task {task['task_id']}",
    )
    run = Path(run_dir)
    config = read_json_object(run / "config.resolved.json")
    execution_path, execution = _validate_xgb_task_execution(
        run_dir=run,
        reference_or_path=task_dir / "task_execution.json",
        config=config,
        task=task,
        phase="development",
    )
    data = _load_development_replay(run)
    segment = _exact_int(
        task.get("held_out_segment"),
        name="recovery task held_out_segment",
        allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
    )
    paired_seed_index = _exact_int(
        task.get("paired_seed_index"),
        name="recovery task paired_seed_index",
        allowed=(0, 1, 2),
    )
    model_seed = _exact_int(
        task.get("model_seed"),
        name="recovery task model_seed",
        allowed=GBDT_V4_PAIRED_SEEDS,
    )
    eval_rows = np.asarray(data.segment_ids == segment, dtype=bool)
    train_rows = ~eval_rows
    refit = _fit_preprocessor(
        data,
        train_rows,
        forbidden_rows=eval_rows,
        split_context=(
            f"outage_v4_gbdt_segment{segment}_seed"
            f"{paired_seed_index}"
        ),
    )
    preprocessing_path = task_dir / "preprocessing.json"
    if read_json_object(preprocessing_path) != refit.metadata():
        raise GPUOutageGBDTV4Error("recoverable v4 fold preprocessor changed")
    from .gpu_outage_gbdt_v4_training import (
        load_xgb_bundle,
        predict_xgb_bundle_forced_outage,
        validate_xgb_training_replay,
    )

    bundle_dir = task_dir / "bundle"
    bundle = load_xgb_bundle(
        bundle_dir,
        expected_held_out_segment=segment,
        expected_paired_seed=model_seed,
        expected_source_sorted_trial_uid_sha256=refit.leakage_audit_[
            "fit_trial_uid_sha256"
        ],
    )
    if bundle.training_device != "cuda:0":
        raise GPUOutageGBDTV4Error(
            "production v4 recovery requires an isolated cuda:0 bundle"
        )
    device = "cuda:0"
    validate_xgb_training_replay(
        bundle,
        refit.transform_features(data.X[train_rows]).astype(np.float32),
        refit.transform_targets(data.y[train_rows]).astype(np.float32),
        data.trial_uids[train_rows],
        data.segment_ids[train_rows],
        device=device,
    )
    X_eval = refit.transform_features(data.X[eval_rows]).astype(np.float32)
    truth = data.y[eval_rows]
    expected_identity = pd.DataFrame(
        {
            "trial_uid": data.trial_uids[eval_rows],
            "segment_id": data.segment_ids[eval_rows],
            "target_id": data.target_ids[eval_rows],
        }
    )
    evaluations: list[dict[str, Any]] = []
    for outage_index in range(4):
        prediction = refit.inverse_targets(
            predict_xgb_bundle_forced_outage(
                bundle, X_eval, outage_index, device=device
            )
        )
        prediction_path = task_dir / f"predictions.drop{outage_index}.parquet"
        stored = _read_exact_parquet(
            prediction_path,
            ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS, *_PRED_COLUMNS],
        )
        if (
            not stored[["trial_uid", "segment_id", "target_id"]].equals(
                expected_identity
            )
            or not np.array_equal(
                stored.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64), truth
            )
            or not np.array_equal(
                stored.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64),
                prediction,
            )
        ):
            raise GPUOutageGBDTV4Error(
                "recoverable v4 development prediction replay changed"
            )
        evaluations.append(
            {
                "forced_outage_index": outage_index,
                "metrics": regression_metrics(truth, prediction),
                "prediction_file": _artifact_reference(
                    prediction_path, run, rows=len(stored)
                ),
            }
        )
    source = read_json_object(run / "provenance" / "source_snapshot.json")
    environment = read_json_object(
        run / "provenance" / "environment_snapshot.json"
    )
    payload = {
        "contract": GBDT_V4_BUNDLE_RESULT_CONTRACT,
        "config_sha256": gbdt_v4_config_sha256(config),
        "task": dict(task),
        "gpu_id": execution["physical_gpu_id"],
        "bundle_id": str(bundle.bundle_id),
        "bundle_sha256": str(bundle.bundle_sha256),
        "bundle_metadata": bundle.metadata(),
        "bundle_save_audit": _expected_bundle_save_audit(bundle, bundle_dir),
        "training_source_fingerprint": source["fingerprint"],
        "environment_fingerprint": environment["fingerprint"],
        "artifacts": {
            "task_execution": _artifact_reference(execution_path, run),
            "preprocessing": _artifact_reference(preprocessing_path, run),
            "bundle": _bundle_checkpoint_reference(bundle_dir, run),
        },
        "evaluations": evaluations,
        "fit_count": 1,
        "evaluation_count": 4,
        "challenge_labels_accessed": False,
        "result_recovery": {
            "mode": "authenticated_complete_artifacts_without_refit"
        },
    }
    atomic_write_json(result_path, payload)
    recovered = _load_valid_xgb_result(run, task)
    if recovered is None:
        raise GPUOutageGBDTV4Error("reconstructed v4 development result is invalid")
    return recovered


def _resume_xgb_development_task(
    run_dir: str | Path, task: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return a valid/recovered result, or quarantine an unsafe attempt."""

    _recover_quarantine_intents(run_dir, phase="development")
    valid = _load_valid_xgb_result(run_dir, task)
    if valid is not None:
        return valid
    task_dir = _xgb_task_dir(run_dir, task)
    if not task_dir.exists():
        return None
    if not (task_dir / "result.json").exists():
        try:
            recovered = _reconstruct_xgb_development_result(run_dir, task)
            if recovered is not None:
                return recovered
        except Exception:
            # Recovery only authenticates existing evidence. Any malformed or
            # unreadable member is retained in quarantine instead of being
            # overwritten or deleted by resume.
            pass
    _quarantine_task_directory(
        run_dir=run_dir,
        task_dir=task_dir,
        phase="development",
        task_id=str(task["task_id"]),
        reason="incomplete_or_invalid_bundle_attempt",
    )
    return None


def _worker_ledger_directory(run_dir: str | Path, phase: str) -> Path:
    return Path(run_dir) / "provenance" / "worker_ready" / phase


def _record_worker_ready(
    run_dir: str | Path, *, phase: str, runtime: Mapping[str, Any]
) -> None:
    """Durably record worker routing before queue-only readiness can be lost."""

    gpu_id = _exact_int(
        runtime.get("physical_gpu_id"),
        name="worker ledger physical_gpu_id",
        allowed=(0, 1, 2, 3),
    )
    path = _worker_ledger_directory(run_dir, phase) / f"gpu{gpu_id}.json"
    payload = {"phase": phase, "runtime": dict(runtime)}
    if path.is_file():
        if read_json_object(path) != payload:
            raise GPUOutageGBDTV4Error("durable worker-ready ledger changed")
        return
    atomic_write_json(path, payload)


def _validate_worker_runtime_kind(runtime: Mapping[str, Any], *, kind: str) -> None:
    common = {
        "physical_gpu_id",
        "visible_device",
        "visible_cuda_device_uuid",
        "visible_cuda_pci_bus_id",
        "visible_cuda_name",
    }
    if kind != "xgb":
        raise GPUOutageGBDTV4Error(
            "outage-v4 only authenticates development XGB workers"
        )
    if set(runtime) != common | {"xgboost_version", "xgboost_use_cuda"} or (
        runtime.get("xgboost_version") != "3.3.0"
        or runtime.get("xgboost_use_cuda") is not True
    ):
        raise GPUOutageGBDTV4Error("v4 XGB worker runtime record changed")


def _materialize_worker_aggregate_ledger(
    run_dir: str | Path,
    *,
    phase: str,
    kind: str,
    gpu_ids: Sequence[int],
    output_name: str,
) -> dict[str, Any]:
    """Recover the aggregate ledger solely from four durable worker records."""

    expected_ids = tuple(sorted(_validate_gpu_ids(gpu_ids)))
    directory = _worker_ledger_directory(run_dir, phase)
    _assert_exact_directory_children(
        directory,
        {f"gpu{gpu_id}.json": "file" for gpu_id in expected_ids},
        label=f"v4 {phase} worker-ready ledger",
    )
    environment = read_json_object(
        Path(run_dir) / "provenance" / "environment_snapshot.json"
    )
    workers: list[dict[str, Any]] = []
    for gpu_id in expected_ids:
        durable = read_json_object(directory / f"gpu{gpu_id}.json")
        runtime = durable.get("runtime")
        if (
            set(durable) != {"phase", "runtime"}
            or durable.get("phase") != phase
            or not isinstance(runtime, Mapping)
            or _exact_int(
                runtime.get("physical_gpu_id"),
                name="durable worker physical_gpu_id",
                allowed=(0, 1, 2, 3),
            )
            != gpu_id
        ):
            raise GPUOutageGBDTV4Error("v4 durable worker-ready ledger changed")
        _validate_worker_cuda_mapping(runtime, environment)
        _validate_worker_runtime_kind(runtime, kind=kind)
        workers.append(dict(runtime))
    payload = {"phase": phase, "worker_count": 4, "workers": workers}
    output = Path(run_dir) / "provenance" / output_name
    if output.is_file():
        stored = read_json_object(output)
        _exact_int(stored.get("worker_count"), name="aggregate worker_count")
        if not _type_exact_equal(stored, payload):
            raise GPUOutageGBDTV4Error("v4 aggregate worker ledger changed")
    else:
        atomic_write_json(output, payload)
    return payload


def _validate_requested_gpu_ids(run_dir: str | Path, gpu_ids: Sequence[int]) -> None:
    requested = tuple(sorted(_validate_gpu_ids(gpu_ids)))
    environment = read_json_object(
        Path(run_dir) / "provenance" / "environment_snapshot.json"
    )
    devices = _validate_parent_cuda_device_grid(
        environment.get("payload", {}).get("cuda_devices")
    )
    expected = tuple(sorted(row["physical_gpu_id"] for row in devices))
    if requested != expected:
        raise GPUOutageGBDTV4Error(
            "requested GPU IDs differ from the exact four-GPU parent freeze"
        )


def _validate_aggregate_worker_ledger(
    run_dir: str | Path, *, phase: str, kind: str, output_name: str
) -> dict[str, Any]:
    """Validate or reconstruct the aggregate from the four durable ledgers."""

    environment = read_json_object(
        Path(run_dir) / "provenance" / "environment_snapshot.json"
    )
    devices = _validate_parent_cuda_device_grid(
        environment.get("payload", {}).get("cuda_devices")
    )
    gpu_ids = [row["physical_gpu_id"] for row in devices]
    return _materialize_worker_aggregate_ledger(
        run_dir,
        phase=phase,
        kind=kind,
        gpu_ids=gpu_ids,
        output_name=output_name,
    )


def _xgb_worker_loop(
    gpu_id: int,
    config: dict[str, Any],
    run_dir: str,
    task_queue: Any,
    result_queue: Any,
) -> None:
    reported_gpu_id = gpu_id if type(gpu_id) is int else None
    try:
        physical_gpu_id = _exact_int(
            gpu_id, name="worker physical_gpu_id", allowed=(0, 1, 2, 3)
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        _validate_v4_source_provenance(run_dir, validate_environment=False)
        data = _load_development_replay(run_dir)
        import xgboost

        if str(xgboost.__version__) != "3.3.0" or not bool(
            xgboost.build_info().get("USE_CUDA", False)
        ):
            raise GPUOutageGBDTV4Error("v4 worker requires CUDA XGBoost 3.3.0")
        runtime = {
            **_worker_cuda_runtime_identity(
                run_dir, physical_gpu_id=physical_gpu_id
            ),
            "xgboost_version": str(xgboost.__version__),
            "xgboost_use_cuda": True,
        }
        _record_worker_ready(run_dir, phase="development", runtime=runtime)
        result_queue.put(
            {
                "event": "worker_ready",
                "gpu_id": physical_gpu_id,
                "runtime": runtime,
            }
        )
        while True:
            task = task_queue.get()
            if task is None:
                return
            try:
                result = _execute_xgb_development_task(
                    data, task, config, run_dir, gpu_id=physical_gpu_id
                )
                result_queue.put(
                    {
                        "event": "task_complete",
                        "gpu_id": physical_gpu_id,
                        "task_id": str(task["task_id"]),
                        "result_file": result["result_file"],
                    }
                )
            except Exception as exc:  # pragma: no cover - subprocess boundary
                result_queue.put(
                    {
                        "event": "task_error",
                        "gpu_id": physical_gpu_id,
                        "task_id": str(task.get("task_id", "unknown")),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                return
    except Exception as exc:  # pragma: no cover - subprocess boundary
        result_queue.put(
            {
                "event": "worker_error",
                "gpu_id": reported_gpu_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _run_xgb_development_pool(
    *,
    config: dict[str, Any],
    run_dir: str | Path,
    gpu_ids: Sequence[int],
) -> list[dict[str, Any]]:
    gpu_ids = _validate_gpu_ids(gpu_ids)
    tasks = gbdt_v4_development_tasks(config)
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    completed: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for task in tasks:
        result = _resume_xgb_development_task(run_dir, task)
        if result is None:
            pending.append(task)
        else:
            completed[str(task["task_id"])] = result
    ledger_dir = _worker_ledger_directory(run_dir, "development")
    durable_worker_grid_complete = all(
        (ledger_dir / f"gpu{gpu_id}.json").is_file() for gpu_id in gpu_ids
    )
    if pending or not durable_worker_grid_complete:
        context = mp.get_context("spawn")
        task_queue = context.Queue()
        result_queue = context.Queue()
        for task in pending:
            task_queue.put(task)
        processes = [
            context.Process(
                target=_xgb_worker_loop,
                args=(gpu_id, dict(config), str(Path(run_dir).resolve()), task_queue, result_queue),
                name=f"outage-v4-gbdt-gpu{gpu_id}",
            )
            for gpu_id in gpu_ids
        ]
        for process in processes:
            process.start()
        remaining = {str(task["task_id"]) for task in pending}
        try:
            while remaining:
                try:
                    event = result_queue.get(timeout=1.0)
                except queue.Empty:
                    for task_id in list(remaining):
                        if _load_valid_xgb_result(run_dir, tasks_by_id[task_id]) is not None:
                            remaining.discard(task_id)
                    dead = [process for process in processes if not process.is_alive()]
                    if dead and remaining:
                        raise GPUOutageGBDTV4Error(
                            f"v4 XGB worker exited early: "
                            f"{ {process.name: process.exitcode for process in dead} }"
                        )
                    continue
                kind = str(event.get("event"))
                if kind == "worker_ready":
                    continue
                elif kind == "task_complete":
                    task_id = str(event["task_id"])
                    if (
                        task_id not in tasks_by_id
                        or _load_valid_xgb_result(run_dir, tasks_by_id[task_id]) is None
                    ):
                        raise GPUOutageGBDTV4Error(
                            "v4 worker completion event lacks an authenticated result"
                        )
                    remaining.discard(task_id)
                elif kind in {"task_error", "worker_error"}:
                    raise GPUOutageGBDTV4Error(
                        f"{kind} on GPU {event.get('gpu_id')}: {event.get('error')}\n"
                        f"{event.get('traceback', '')}"
                    )
                else:
                    raise GPUOutageGBDTV4Error(f"unknown v4 worker event: {event}")
            # A worker may finish much earlier than the worker holding the
            # final heterogeneous-duration task. Only release workers after
            # every pending result has been durably reloaded/authenticated.
            for _ in gpu_ids:
                task_queue.put(None)
        finally:
            if remaining:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
            for process in processes:
                process.join(timeout=10.0)
            for process in processes:
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5.0)
        if remaining:
            raise GPUOutageGBDTV4Error(
                f"v4 XGB pool stopped with {len(remaining)} unfinished bundles"
            )
    _materialize_worker_aggregate_ledger(
        run_dir,
        phase="development",
        kind="xgb",
        gpu_ids=gpu_ids,
        output_name="development_gpu_workers.json",
    )
    results: list[dict[str, Any]] = []
    for task in tasks:
        result = _load_valid_xgb_result(run_dir, task)
        if result is None:
            raise GPUOutageGBDTV4Error(f"missing XGB result: {task['task_id']}")
        results.append(result)
    return results


def _seed_prediction_frame(
    run_dir: str | Path,
    reference: Mapping[str, Any],
    *,
    expected_rows: int,
) -> pd.DataFrame:
    frame = _read_exact_parquet(
        _verify_reference(run_dir, reference, expected_rows=expected_rows),
        ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS, *_PRED_COLUMNS],
    )
    if len(frame) != expected_rows:
        raise GPUOutageGBDTV4Error("seed prediction row count changed")
    return frame


def _incumbent_seed_map(run_dir: str | Path) -> dict[tuple[int, int, int], pd.DataFrame]:
    freeze = read_json_object(Path(run_dir) / "incumbent.freeze.json")
    mapped: dict[tuple[int, int, int], pd.DataFrame] = {}
    for row in freeze["incumbent_seed_predictions"]:
        key = (
            _exact_int(
                row.get("held_out_segment"),
                name="incumbent map held_out_segment",
                allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
            ),
            _exact_int(
                row.get("forced_outage_index"),
                name="incumbent map forced_outage_index",
                allowed=(0, 1, 2, 3),
            ),
            _exact_int(
                row.get("paired_seed_index"),
                name="incumbent map paired_seed_index",
                allowed=(0, 1, 2),
            ),
        )
        if key in mapped:
            raise GPUOutageGBDTV4Error("duplicate incumbent seed evaluation")
        mapped[key] = _seed_prediction_frame(
            run_dir,
            row["snapshot"],
            expected_rows=_DEVELOPMENT_SEGMENT_ROWS[key[0]],
        )
    if len(mapped) != 48:
        raise GPUOutageGBDTV4Error("incumbent seed grid must contain 48 records")
    return mapped


def _xgb_seed_map(
    run_dir: str | Path, results: Sequence[Mapping[str, Any]]
) -> dict[tuple[int, int, int], pd.DataFrame]:
    mapped: dict[tuple[int, int, int], pd.DataFrame] = {}
    for result in results:
        task = result["task"]
        for row in result["evaluations"]:
            key = (
                _exact_int(
                    task.get("held_out_segment"),
                    name="XGB map held_out_segment",
                    allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
                ),
                _exact_int(
                    row.get("forced_outage_index"),
                    name="XGB map forced_outage_index",
                    allowed=(0, 1, 2, 3),
                ),
                _exact_int(
                    task.get("paired_seed_index"),
                    name="XGB map paired_seed_index",
                    allowed=(0, 1, 2),
                ),
            )
            if key in mapped:
                raise GPUOutageGBDTV4Error("duplicate XGB seed evaluation")
            mapped[key] = _seed_prediction_frame(
                run_dir,
                row["prediction_file"],
                expected_rows=_DEVELOPMENT_SEGMENT_ROWS[key[0]],
            )
    if len(mapped) != 48:
        raise GPUOutageGBDTV4Error("XGB seed evaluation grid must contain 48 records")
    return mapped


def _aggregate_v4_context(
    *,
    candidate_id: str,
    segment: int,
    outage_index: int,
    paired_seed_frames: Sequence[pd.DataFrame],
    run_dir: str | Path,
) -> dict[str, Any]:
    if len(paired_seed_frames) != 3:
        raise GPUOutageGBDTV4Error("v4 context requires exactly three paired seeds")
    identity_columns = ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS]
    base = paired_seed_frames[0]
    if any(not frame[identity_columns].equals(base[identity_columns]) for frame in paired_seed_frames[1:]):
        raise GPUOutageGBDTV4Error("v4 paired seed predictions are row-misaligned")
    truth = base.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
    seed_values = [
        frame.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64)
        for frame in paired_seed_frames
    ]
    ensemble = np.mean(np.stack(seed_values, axis=0), axis=0)
    seed_rmses = [float(regression_metrics(truth, value)["rmse_3d_mm"]) for value in seed_values]
    output = base[identity_columns].copy()
    for axis, name in enumerate(_PRED_COLUMNS):
        output[name] = ensemble[:, axis]
    path = atomic_write_parquet(
        Path(run_dir)
        / "development"
        / "contexts"
        / candidate_id
        / f"segment{segment}__drop{outage_index}.parquet",
        output,
    )
    return {
        "candidate_id": candidate_id,
        "held_out_segment": segment,
        "forced_outage_index": outage_index,
        "context_id": f"segment{segment}__drop{outage_index}",
        "seed_count": 3,
        "metrics": regression_metrics(truth, ensemble),
        "seed_rmse_3d_mm": seed_rmses,
        "seed_rmse_population_std_mm": float(np.std(seed_rmses, ddof=0)),
        "prediction_file": _artifact_reference(path, run_dir, rows=len(output)),
    }


def aggregate_gbdt_v4_development(
    config: Mapping[str, Any],
    xgb_results: Sequence[Mapping[str, Any]],
    run_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize 16 contexts each for incumbent, XGB, and paired 0.5 blend."""

    incumbent = _incumbent_seed_map(run_dir)
    xgb = _xgb_seed_map(run_dir, xgb_results)
    contexts: list[dict[str, Any]] = []
    for segment in GBDT_V4_DEVELOPMENT_SEGMENTS:
        for outage_index in range(4):
            incumbent_frames = [incumbent[(segment, outage_index, seed)] for seed in range(3)]
            xgb_frames = [xgb[(segment, outage_index, seed)] for seed in range(3)]
            contexts.append(
                _aggregate_v4_context(
                    candidate_id="v3_incumbent",
                    segment=segment,
                    outage_index=outage_index,
                    paired_seed_frames=incumbent_frames,
                    run_dir=run_dir,
                )
            )
            contexts.append(
                _aggregate_v4_context(
                    candidate_id="xgboost_mask_replicated",
                    segment=segment,
                    outage_index=outage_index,
                    paired_seed_frames=xgb_frames,
                    run_dir=run_dir,
                )
            )
            blended: list[pd.DataFrame] = []
            identity_columns = ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS]
            for incumbent_frame, xgb_frame in zip(incumbent_frames, xgb_frames, strict=True):
                if not incumbent_frame[identity_columns].equals(xgb_frame[identity_columns]):
                    raise GPUOutageGBDTV4Error("incumbent/XGB paired seed rows are misaligned")
                frame = incumbent_frame[identity_columns].copy()
                values = 0.5 * incumbent_frame.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64)
                values += 0.5 * xgb_frame.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64)
                for axis, name in enumerate(_PRED_COLUMNS):
                    frame[name] = values[:, axis]
                blended.append(frame)
            context = _aggregate_v4_context(
                candidate_id="equal_blend_incumbent_xgboost",
                segment=segment,
                outage_index=outage_index,
                paired_seed_frames=blended,
                run_dir=run_dir,
            )
            context["blend_weights"] = {
                "v3_incumbent": 0.5,
                "xgboost_mask_replicated": 0.5,
            }
            contexts.append(context)
    if len(contexts) != 48:
        raise GPUOutageGBDTV4Error("v4 must produce exactly 48 scored contexts")
    context_path = atomic_write_json(
        Path(run_dir) / "development" / "context_metrics.json",
        {
            "config_sha256": gbdt_v4_config_sha256(config),
            "candidate_count": 3,
            "context_count_per_candidate": 16,
            "contexts": contexts,
        },
    )
    selection = score_gbdt_v4_candidates(contexts, config)
    selection["context_metrics_file"] = _artifact_reference(context_path, run_dir)
    selection["v3_incumbent_selected_candidate_id"] = read_json_object(
        Path(run_dir) / "incumbent.freeze.json"
    )["v3_selected_candidate_id"]
    selection_path = atomic_write_json(
        Path(run_dir) / "development" / "selection.json", selection
    )
    selection["selection_file"] = _artifact_reference(selection_path, run_dir)
    return contexts, selection


def _development_manifest_files(run_dir: str | Path) -> list[dict[str, Any]]:
    return [
        row
        for row in manifest_files(run_dir)
        if row["path"] not in {"development_manifest.json", "manifest.json"}
        and not row["path"].startswith("final/")
        and not row["path"].startswith("challenge/")
        and not row["path"].startswith("provenance/final_")
        and not row["path"].startswith("recovery/final_")
    ]


def _write_development_manifest(
    run_dir: str | Path, config: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "contract": GBDT_V4_RUN_CONTRACT,
        "experiment_contract": GBDT_V4_CONTRACT,
        "stage": "development_complete_challenge_labels_sealed",
        "config_sha256": gbdt_v4_config_sha256(config),
        "counts": {
            "v3_incumbent_source_tasks_bound": 240,
            "xgb_unique_bundle_fits": 12,
            "xgb_axis_boosters": 36,
            "xgb_outage_evaluation_records": 48,
            "scored_contexts": 48,
            "challenge_label_access_passes": 0,
        },
        "selection": {
            "selected_candidate_id": str(selection["selected_candidate_id"]),
            "selected_outage_robust_j_mm": float(selection["selected_outage_robust_j_mm"]),
            "challenge_used_for_selection": False,
        },
        "challenge_labels_sealed": True,
        "files": _development_manifest_files(run_dir),
    }
    atomic_write_json(Path(run_dir) / "development_manifest.json", payload)
    return payload


def _assert_json_close(actual: Any, expected: Any, *, path: str = "root") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise GPUOutageGBDTV4Error(f"JSON mapping changed at {path}")
        for key in expected:
            _assert_json_close(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise GPUOutageGBDTV4Error(f"JSON list changed at {path}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_json_close(left, right, path=f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if type(actual) not in {int, float}:
            raise GPUOutageGBDTV4Error(f"JSON numeric value changed at {path}")
        value = float(actual)
        if not math.isfinite(value):
            raise GPUOutageGBDTV4Error(f"JSON numeric value changed at {path}")
        if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-10):
            raise GPUOutageGBDTV4Error(f"JSON numeric value changed at {path}")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise GPUOutageGBDTV4Error(f"JSON value changed at {path}")


def _recomputed_context(
    *,
    candidate_id: str,
    segment: int,
    outage_index: int,
    paired_seed_frames: Sequence[pd.DataFrame],
    prediction_reference: Mapping[str, Any],
    blend: bool = False,
) -> dict[str, Any]:
    identity_columns = ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS]
    base = paired_seed_frames[0]
    if len(paired_seed_frames) != 3 or any(
        not frame[identity_columns].equals(base[identity_columns])
        for frame in paired_seed_frames[1:]
    ):
        raise GPUOutageGBDTV4Error("replayed v4 context seed rows changed")
    truth = base.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
    seeds = [
        frame.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64)
        for frame in paired_seed_frames
    ]
    ensemble = np.mean(np.stack(seeds, axis=0), axis=0)
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "held_out_segment": segment,
        "forced_outage_index": outage_index,
        "context_id": f"segment{segment}__drop{outage_index}",
        "seed_count": 3,
        "metrics": regression_metrics(truth, ensemble),
        "seed_rmse_3d_mm": [
            float(regression_metrics(truth, value)["rmse_3d_mm"]) for value in seeds
        ],
        "seed_rmse_population_std_mm": float(
            np.std(
                [float(regression_metrics(truth, value)["rmse_3d_mm"]) for value in seeds],
                ddof=0,
            )
        ),
        "prediction_file": dict(prediction_reference),
    }
    if blend:
        payload["blend_weights"] = {
            "v3_incumbent": 0.5,
            "xgboost_mask_replicated": 0.5,
        }
    return payload


def _validate_context_replay_frame(
    context_file: pd.DataFrame,
    expected_seed_frame: pd.DataFrame,
    expected_ensemble: np.ndarray,
    *,
    expected_rows: int,
) -> None:
    """Bind a durable context to its complete identity, truth, and ensemble."""

    identity_columns = ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS]
    if (
        len(context_file) != expected_rows
        or len(expected_seed_frame) != expected_rows
        or not context_file.loc[:, identity_columns].equals(
            expected_seed_frame.loc[:, identity_columns]
        )
    ):
        raise GPUOutageGBDTV4Error(
            "v4 durable ensemble context identity/truth changed"
        )
    if expected_ensemble.shape != (expected_rows, 3) or not np.array_equal(
        context_file.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64),
        expected_ensemble,
    ):
        raise GPUOutageGBDTV4Error("v4 durable ensemble context prediction changed")


def validate_gbdt_v4_development_run(run_dir: str | Path) -> dict[str, Any]:
    """Replay all 12 bundles and 48 contexts without opening any source archive."""

    _require_python_312()
    run = Path(run_dir).resolve()
    _assert_development_only_run(run)
    prepared = validate_gbdt_v4_prepared_run(run)
    config = read_json_object(run / "config.resolved.json")
    config_sha = gbdt_v4_config_sha256(config)
    _assert_development_closed_world(run, config)
    data = _load_development_replay(run)
    _validate_aggregate_worker_ledger(
        run,
        phase="development",
        kind="xgb",
        output_name="development_gpu_workers.json",
    )

    from .gpu_outage_gbdt_v4_training import (
        load_xgb_bundle,
        predict_xgb_bundle_forced_outage,
        validate_development_bundle_coverage,
        validate_xgb_training_replay,
    )

    results: list[dict[str, Any]] = []
    loaded_bundles: list[Any] = []
    for task in gbdt_v4_development_tasks(config):
        result = _load_valid_xgb_result(run, task)
        if result is None or result.get("config_sha256") != config_sha:
            raise GPUOutageGBDTV4Error(f"invalid v4 XGB result: {task['task_id']}")
        segment = _exact_int(
            task.get("held_out_segment"),
            name="validation task held_out_segment",
            allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
        )
        paired_seed_index = _exact_int(
            task.get("paired_seed_index"),
            name="validation task paired_seed_index",
            allowed=(0, 1, 2),
        )
        model_seed = _exact_int(
            task.get("model_seed"),
            name="validation task model_seed",
            allowed=GBDT_V4_PAIRED_SEEDS,
        )
        eval_rows = np.asarray(data.segment_ids == segment, dtype=bool)
        train_rows = ~eval_rows
        preprocessor_metadata = read_json_object(
            _verify_reference(run, result["artifacts"]["preprocessing"])
        )
        refit = _fit_preprocessor(
            data,
            train_rows,
            forbidden_rows=eval_rows,
            split_context=f"outage_v4_gbdt_segment{segment}_seed{paired_seed_index}",
        )
        if preprocessor_metadata != refit.metadata():
            raise GPUOutageGBDTV4Error("v4 XGB fold preprocessor changed")
        bundle_dir = _verify_bundle_checkpoint(run, result["artifacts"]["bundle"])
        bundle = load_xgb_bundle(
            bundle_dir,
            expected_held_out_segment=segment,
            expected_paired_seed=model_seed,
            expected_source_sorted_trial_uid_sha256=refit.leakage_audit_[
                "fit_trial_uid_sha256"
            ],
        )
        if bundle.training_device != "cuda:0":
            raise GPUOutageGBDTV4Error(
                "production v4 validation requires an isolated cuda:0 bundle"
            )
        _validate_loaded_xgb_bundle_binding(
            bundle, result, expected_config_sha256=config_sha
        )
        loaded_bundles.append(bundle)
        X_train = refit.transform_features(data.X[train_rows]).astype(np.float32)
        y_train = refit.transform_targets(data.y[train_rows]).astype(np.float32)
        validate_xgb_training_replay(
            bundle,
            X_train,
            y_train,
            data.trial_uids[train_rows],
            data.segment_ids[train_rows],
            device="cuda:0",
        )
        X_eval = refit.transform_features(data.X[eval_rows]).astype(np.float32)
        truth = data.y[eval_rows]
        for evaluation in result["evaluations"]:
            outage_index = _exact_int(
                evaluation.get("forced_outage_index"),
                name="validation evaluation forced_outage_index",
                allowed=(0, 1, 2, 3),
            )
            replay_scaled = predict_xgb_bundle_forced_outage(
                bundle, X_eval, outage_index, device="cuda:0"
            )
            replay = refit.inverse_targets(replay_scaled)
            stored = _seed_prediction_frame(
                run,
                evaluation["prediction_file"],
                expected_rows=_DEVELOPMENT_SEGMENT_ROWS[segment],
            )
            if (
                stored["trial_uid"].astype(str).tolist()
                != data.trial_uids[eval_rows].astype(str).tolist()
                or not np.array_equal(
                    stored.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64), truth
                )
                or not np.array_equal(
                    stored.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64), replay
                )
            ):
                raise GPUOutageGBDTV4Error("v4 XGB bundle prediction replay changed")
            _assert_json_close(
                evaluation["metrics"], regression_metrics(truth, replay), path="xgb.metrics"
            )
        results.append(result)
    coverage = validate_development_bundle_coverage(
        loaded_bundles, require_cuda=True
    )
    if coverage != {
        "bundle_count": 12,
        "fold_count": 4,
        "forced_outage_fit_multiplier": 1,
        "paired_seed_count": 3,
        "scalar_booster_count": 36,
    }:
        raise GPUOutageGBDTV4Error("v4 CUDA bundle coverage audit changed")
    if len(results) != 12 or sum(len(row["evaluations"]) for row in results) != 48:
        raise GPUOutageGBDTV4Error("v4 XGB fit/evaluation counts changed")

    incumbent = _incumbent_seed_map(run)
    xgb = _xgb_seed_map(run, results)
    context_payload = read_json_object(run / "development" / "context_metrics.json")
    stored_contexts = context_payload.get("contexts")
    if (
        context_payload.get("config_sha256") != config_sha
        or _exact_int(
            context_payload.get("candidate_count"), name="context candidate_count"
        )
        != 3
        or _exact_int(
            context_payload.get("context_count_per_candidate"),
            name="context_count_per_candidate",
        )
        != 16
        or not isinstance(stored_contexts, list)
        or len(stored_contexts) != 48
    ):
        raise GPUOutageGBDTV4Error("v4 durable context payload changed")
    stored_by_key = {
        (
            str(row["candidate_id"]),
            _exact_int(
                row.get("held_out_segment"),
                name="context held_out_segment",
                allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
            ),
            _exact_int(
                row.get("forced_outage_index"),
                name="context forced_outage_index",
                allowed=(0, 1, 2, 3),
            ),
        ): row
        for row in stored_contexts
    }
    if len(stored_by_key) != 48:
        raise GPUOutageGBDTV4Error("v4 durable contexts contain duplicate keys")
    replayed_contexts: list[dict[str, Any]] = []
    for segment in GBDT_V4_DEVELOPMENT_SEGMENTS:
        for outage_index in range(4):
            incumbent_frames = [incumbent[(segment, outage_index, seed)] for seed in range(3)]
            xgb_frames = [xgb[(segment, outage_index, seed)] for seed in range(3)]
            paired: dict[str, list[pd.DataFrame]] = {
                "v3_incumbent": incumbent_frames,
                "xgboost_mask_replicated": xgb_frames,
            }
            blend_frames: list[pd.DataFrame] = []
            identity_columns = ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS]
            for left, right in zip(incumbent_frames, xgb_frames, strict=True):
                if not left[identity_columns].equals(right[identity_columns]):
                    raise GPUOutageGBDTV4Error("v4 replay blend rows changed")
                frame = left[identity_columns].copy()
                values = 0.5 * left.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64)
                values += 0.5 * right.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64)
                for axis, name in enumerate(_PRED_COLUMNS):
                    frame[name] = values[:, axis]
                blend_frames.append(frame)
            paired["equal_blend_incumbent_xgboost"] = blend_frames
            for candidate_id in GBDT_V4_CANDIDATE_IDS:
                stored = stored_by_key[(candidate_id, segment, outage_index)]
                context_file = _seed_prediction_frame(
                    run,
                    stored["prediction_file"],
                    expected_rows=_DEVELOPMENT_SEGMENT_ROWS[segment],
                )
                expected_ensemble = np.mean(
                    np.stack(
                        [
                            frame.loc[:, list(_PRED_COLUMNS)].to_numpy(dtype=np.float64)
                            for frame in paired[candidate_id]
                        ],
                        axis=0,
                    ),
                    axis=0,
                )
                _validate_context_replay_frame(
                    context_file,
                    paired[candidate_id][0],
                    expected_ensemble,
                    expected_rows=_DEVELOPMENT_SEGMENT_ROWS[segment],
                )
                replayed = _recomputed_context(
                    candidate_id=candidate_id,
                    segment=segment,
                    outage_index=outage_index,
                    paired_seed_frames=paired[candidate_id],
                    prediction_reference=stored["prediction_file"],
                    blend=candidate_id == "equal_blend_incumbent_xgboost",
                )
                _assert_json_close(stored, replayed, path=f"context.{candidate_id}")
                replayed_contexts.append(replayed)
    selection = read_json_object(run / "development" / "selection.json")
    recomputed_selection = score_gbdt_v4_candidates(replayed_contexts, config)
    expected_selection = {
        **recomputed_selection,
        "context_metrics_file": _artifact_reference(
            run / "development" / "context_metrics.json", run
        ),
        "v3_incumbent_selected_candidate_id": read_json_object(
            run / "incumbent.freeze.json"
        )["v3_selected_candidate_id"],
    }
    _assert_json_close(selection, expected_selection, path="selection")
    manifest = read_json_object(run / "development_manifest.json")
    expected_manifest = {
        "contract": GBDT_V4_RUN_CONTRACT,
        "experiment_contract": GBDT_V4_CONTRACT,
        "stage": "development_complete_challenge_labels_sealed",
        "config_sha256": config_sha,
        "counts": {
            "v3_incumbent_source_tasks_bound": 240,
            "xgb_unique_bundle_fits": 12,
            "xgb_axis_boosters": 36,
            "xgb_outage_evaluation_records": 48,
            "scored_contexts": 48,
            "challenge_label_access_passes": 0,
        },
        "selection": {
            "selected_candidate_id": selection["selected_candidate_id"],
            "selected_outage_robust_j_mm": selection["selected_outage_robust_j_mm"],
            "challenge_used_for_selection": False,
        },
        "challenge_labels_sealed": True,
        "files": _development_manifest_files(run),
    }
    _assert_json_close(manifest, expected_manifest, path="development_manifest")
    return {
        "status": "ok",
        "contract": GBDT_V4_RUN_CONTRACT,
        "stage": "development_complete_challenge_labels_sealed",
        "config_sha256": config_sha,
        "v3_incumbent_source_tasks_bound": 240,
        "xgb_unique_bundle_fits": 12,
        "xgb_outage_evaluation_records": 48,
        "scored_contexts": 48,
        "selected_candidate_id": str(selection["selected_candidate_id"]),
        "selected_outage_robust_j_mm": float(selection["selected_outage_robust_j_mm"]),
        "challenge_labels_sealed": True,
        "challenge_label_access_passes": 0,
        "source_archives_opened_by_validator": False,
        "prepared_stage": prepared["stage"],
    }




def _validate_gpu_ids(gpu_ids: Sequence[int]) -> tuple[int, ...]:
    values = tuple(
        _exact_int(value, name="requested GPU ID", allowed=(0, 1, 2, 3))
        for value in gpu_ids
    )
    if len(values) != 4 or len(set(values)) != 4:
        raise GPUOutageGBDTV4Error("v4 requires exactly four distinct non-negative GPU IDs")
    return values


def run_gbdt_v4_experiment(
    config: dict[str, Any],
    run_dir: str | Path,
    *,
    gpu_ids: Sequence[int] = (0, 1, 2, 3),
    resume: bool = False,
    finalize_challenge: bool = False,
) -> dict[str, Any]:
    """Run the development-only v4 comparison and keep challenge labels sealed."""

    _require_python_312()
    validate_gbdt_v4_config(config)
    if finalize_challenge:
        raise GPUOutageGBDTV4Error(
            "outage-v4 is development-only; outage-v5 exclusively owns the "
            "challenge gate"
        )
    gpu_ids = _validate_gpu_ids(gpu_ids)
    run = Path(run_dir).resolve()
    if not resume:
        raise GPUOutageGBDTV4Error(
            "v4 requires the explicit prepare command first, then run --resume"
        )
    _assert_development_only_run(run)
    _validate_requested_gpu_ids(run, gpu_ids)
    if (run / "development_manifest.json").is_file():
        return validate_gbdt_v4_development_run(run)
    validate_gbdt_v4_prepared_run(run)
    xgb_results = _run_xgb_development_pool(
        config=config, run_dir=run, gpu_ids=gpu_ids
    )
    _validate_v4_source_provenance(run)
    _, selection = aggregate_gbdt_v4_development(config, xgb_results, run)
    manifest = _write_development_manifest(run, config, selection)
    validate_gbdt_v4_development_run(run)
    return manifest




def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, run, and validate the standalone outage-v4 GBDT study."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="Freeze a completed, still-sealed v3 development run."
    )
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--v3-run", required=True)
    prepare.add_argument("--output", required=True)
    run = subparsers.add_parser("run", help="Run the sealed v4 development comparison.")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--gpus", nargs=4, type=int, default=[0, 1, 2, 3])
    run.add_argument("--resume", action="store_true")
    validate = subparsers.add_parser(
        "validate", help="Validate a prepared or development-complete v4 run."
    )
    validate.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        config = load_gbdt_v4_config(args.config)
        payload = prepare_gbdt_v4_run(
            config, v3_run_dir=args.v3_run, run_dir=args.output
        )
    elif args.command == "run":
        config = load_gbdt_v4_config(args.config)
        payload = run_gbdt_v4_experiment(
            config,
            args.output,
            gpu_ids=args.gpus,
            resume=bool(args.resume),
        )
    else:
        run = Path(args.run_dir)
        if (run / "manifest.json").is_file():
            raise GPUOutageGBDTV4Error(
                "outage-v4 has no complete/challenge stage; outage-v5 owns the gate"
            )
        elif (run / "development_manifest.json").is_file():
            payload = validate_gbdt_v4_development_run(run)
        else:
            payload = validate_gbdt_v4_prepared_run(run)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())


__all__ = [
    "GBDT_V4_BUNDLE_RESULT_CONTRACT",
    "GBDT_V4_FREEZE_CONTRACT",
    "GBDT_V4_RUN_CONTRACT",
    "GBDT_V4_SOURCE_FILES",
    "GPUOutageGBDTV4Error",
    "aggregate_gbdt_v4_development",
    "gbdt_v4_development_tasks",
    "gbdt_v4_environment_payload",
    "gbdt_v4_source_payload",
    "main",
    "prepare_gbdt_v4_run",
    "run_gbdt_v4_experiment",
    "score_gbdt_v4_candidates",
    "validate_gbdt_v4_development_run",
    "validate_gbdt_v4_prepared_run",
]
