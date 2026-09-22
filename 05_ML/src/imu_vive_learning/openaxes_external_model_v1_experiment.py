"""Closed-world orchestration for the OpenAXES external model benchmark.

The module owns development-only task planning, process isolation, durable
phase receipts, aggregation, and signed alignment-sensitivity consumption.  It
has no challenge route and imports no PyTorch at module import time.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import ast
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .io_utils import sha256_file, write_json, write_parquet
from .openaxes_external_model_v1_config import (
    OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
    OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
    OPENAXES_EXTERNAL_PREPARATION_V1_CONTRACT,
    OPENAXES_MODEL_METHOD_IDS,
    OPENAXES_NO_REFIT_THREAT_MODEL,
    OPENAXES_MODEL_PHASES,
    OPENAXES_MODEL_SEEDS,
    OPENAXES_MODEL_SENSOR_GROUPS,
    OPENAXES_MODEL_SHIFT_MS,
    OPENAXES_MODEL_VIEW_IDS,
    OPENAXES_PRELAUNCH_AUTHORIZED_ACTION_SCOPE,
    OPENAXES_PRELAUNCH_EXACT_EMPTY_DIRECTORIES,
    OPENAXES_PRELAUNCH_EXACT_ROOT_MEMBERS,
    OPENAXES_PRELAUNCH_EXACT_RUNTIME_MEMBERS_AFTER_RECEIPT,
    OPENAXES_PRELAUNCH_EXACT_RUNTIME_MEMBERS_BEFORE_RECEIPT,
    OPENAXES_PRELAUNCH_FORBIDDEN_ACTION_SCOPES,
    OPENAXES_PRELAUNCH_RECOVERY_MODE,
    OPENAXES_PRELAUNCH_RECOVERY_PATH,
    OPENAXES_TOTAL_ERASURE_LIMITATION,
    openaxes_external_model_config_sha256,
    validate_openaxes_external_model_config,
)
from .openaxes_external_model_v1_training import (
    OpenAxesExternalModelV1TrainingError,
    TrainOnlyTransform,
    TrainingResult,
    evaluate_outer_predictions_once,
    predict_outer_views_once,
    state_dict_sha256,
    train_model_train_only,
)


MODEL_PREPARED_MANIFEST = "prepared_manifest.json"
MODEL_PREPARED_MARKER = "PREPARED.json"
MODEL_DEVELOPMENT_MANIFEST = "development_manifest.json"
MODEL_COMPLETE_MARKER = "COMPLETE.json"
MODEL_TASKS_DIR = "tasks"
MODEL_LEDGERS_DIR = "runtime/ledgers"
MODEL_RESULTS_DIR = "results"
MODEL_QUARANTINE_DIR = "quarantine"
MODEL_EXECUTION_SESSION = "runtime/execution_session.json"
MODEL_PRE_FIT_VALIDATION = "runtime/pre_fit_validation.json"
MODEL_PRELAUNCH_SESSION_RECOVERY = OPENAXES_PRELAUNCH_RECOVERY_PATH
MODEL_FULL_SIGNED_SOURCE_REPLAY = "runtime/full_signed_source_replay.json"
MODEL_EXECUTION_LAUNCHES_DIR = "runtime/execution_launches"
MODEL_WORKER_READY_DIR = "runtime/worker_ready"
MODEL_ATTEMPT_LEDGER_DIR = "runtime/attempt_ledger"
MODEL_ATTEMPT_ROSTER_DIR = "runtime/attempt_roster"
MODEL_COMPLETED_TASK_RESULT_MEMBERS = frozenset(
    {
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
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
TASK_ID_PATTERN = re.compile(r"task_[0-9]{4}_[0-9a-f]{12}\Z")
GPU_UUID_PATTERN = re.compile(
    r"(?:GPU-)?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\Z",
    re.IGNORECASE | re.ASCII,
)

MODEL_SOURCE_CLOSURE = (
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

PARENT_SOURCE_CLOSURE = (
    "configs/openaxes_external_v1.yaml",
    "docs/OPENAXES_EXTERNAL_V1_CONTRACT.md",
    "pyproject.toml",
    "src/imu_vive_learning/openaxes_external_v1.py",
    "src/imu_vive_learning/openaxes_external_v1_cli.py",
    "src/imu_vive_learning/openaxes_external_v1_config.py",
    "tests/test_openaxes_external_v1.py",
)

PREPARATION_VALIDATOR_KEYS = {
    "status",
    "contract",
    "config_sha256",
    "source_sha256",
    "prepared_rows",
    "scenario_rows",
    "feature_count",
    "declared_outer_folds",
    "aligned_robot_rows_unmatched",
    "signed_sensitivity_shifts_ms",
    "downstream_model_results_blocked_by_uncommitted_contract",
    "finite",
    "strict_100_hz_per_scenario",
    "semantic_replay",
}


class OpenAxesExternalModelV1Error(RuntimeError):
    """Raised when model benchmark evidence violates the frozen contract."""


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise OpenAxesExternalModelV1Error(f"non-finite JSON constant: {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if type(key) is not str or key in result:
                raise OpenAxesExternalModelV1Error(
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
        raise OpenAxesExternalModelV1Error(f"cannot decode strict JSON: {path}") from exc


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
        raise OpenAxesExternalModelV1Error("payload is not finite canonical JSON") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _closed_frame_sha256(
    frame: pd.DataFrame, schema: Sequence[tuple[str, str]]
) -> str:
    """Independently reproduce the public preparation closed-table receipt."""

    if tuple(frame.columns) != tuple(name for name, _kind in schema):
        raise OpenAxesExternalModelV1Error("closed frame column schema changed")
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
                    raise OpenAxesExternalModelV1Error("closed frame text type changed")
                encoded = value.encode("utf-8")
                digest.update(struct.pack("<I", len(encoded)))
                digest.update(encoded)
        elif kind == "int64":
            values = frame[name].to_numpy()
            if not np.issubdtype(values.dtype, np.integer):
                raise OpenAxesExternalModelV1Error("closed frame integer type changed")
            digest.update(np.asarray(values, dtype="<i8").tobytes(order="C"))
        elif kind == "float64":
            values = frame[name].to_numpy()
            if values.dtype != np.dtype("float64") or not np.isfinite(values).all():
                raise OpenAxesExternalModelV1Error("closed frame float64 type changed")
            digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
        else:
            raise OpenAxesExternalModelV1Error("closed frame kind changed")
    return digest.hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise OpenAxesExternalModelV1Error(f"{name} must be a string-keyed mapping")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise OpenAxesExternalModelV1Error(f"{name} keys changed")


def _exact_str(value: Any, *, name: str, expected: str | None = None) -> str:
    if type(value) is not str or not value or (expected is not None and value != expected):
        raise OpenAxesExternalModelV1Error(f"{name} is not exact text")
    return value


def _canonical_gpu_uuid(value: Any, *, name: str) -> str:
    text = _exact_str(value, name=name)
    match = GPU_UUID_PATTERN.fullmatch(text)
    if match is None:
        raise OpenAxesExternalModelV1Error(
            f"{name} is not a canonical GPU UUID representation"
        )
    return f"GPU-{match.group(1).lower()}"


def _exact_int(value: Any, *, name: str, expected: int | None = None) -> int:
    if type(value) is not int or (expected is not None and value != expected):
        raise OpenAxesExternalModelV1Error(f"{name} is not an exact integer")
    return value


def _exact_float(value: Any, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise OpenAxesExternalModelV1Error(f"{name} is not a finite exact float")
    return value


def _exact_bool(value: Any, *, name: str, expected: bool) -> bool:
    if type(value) is not bool or value is not expected:
        raise OpenAxesExternalModelV1Error(f"{name} is not exactly {expected!r}")
    return value


def _hex64(value: Any, *, name: str) -> str:
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise OpenAxesExternalModelV1Error(f"{name} is not lowercase SHA-256")
    return value


def _reference(path: Path, *, relative: str) -> dict[str, Any]:
    if type(relative) is not str or not relative or path.is_symlink() or not path.is_file():
        raise OpenAxesExternalModelV1Error(f"cannot bind regular file: {relative}")
    return {
        "path": relative,
        "bytes": path.stat(follow_symlinks=False).st_size,
        "sha256": sha256_file(path),
    }


def _verify_reference(root: Path, value: Any, *, name: str) -> Path:
    reference = _mapping(value, name=name)
    _keys(reference, {"path", "bytes", "sha256"}, name=name)
    relative = _exact_str(reference["path"], name=f"{name}.path")
    parsed = Path(relative)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != relative
        or not parsed.parts
        or any(part in ("", ".", "..") for part in parsed.parts)
        or "\\" in relative
    ):
        raise OpenAxesExternalModelV1Error(
            f"{name} path is not canonical POSIX spelling"
        )
    _exact_int(reference["bytes"], name=f"{name}.bytes")
    _hex64(reference["sha256"], name=f"{name}.sha256")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise OpenAxesExternalModelV1Error(f"{name} target is missing")
    observed = _reference(path, relative=relative)
    if dict(reference) != observed:
        raise OpenAxesExternalModelV1Error(f"{name} reference changed")
    return path


def _atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    if exclusive:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _workspace_from_model_config(config: Mapping[str, Any]) -> Path:
    config_path = Path(_exact_str(config.get("_config_path"), name="model config path"))
    if not config_path.is_absolute() or config_path.name != "openaxes_external_model_v1.yaml":
        raise OpenAxesExternalModelV1Error("model config path is not canonical")
    workspace = config_path.parent.parent
    if config_path != workspace / MODEL_SOURCE_CLOSURE[0]:
        raise OpenAxesExternalModelV1Error("model config is outside its canonical workspace")
    return workspace


def _model_source_receipts(workspace: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in MODEL_SOURCE_CLOSURE:
        result[relative] = _reference(workspace / relative, relative=relative)
    return result


def _runtime_receipt_without_torch_import() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "pyarrow", "torch", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise OpenAxesExternalModelV1Error(f"required package is missing: {name}") from exc
    return {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "cache_tag": sys.implementation.cache_tag,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "torch_imported_in_controller": "torch" in sys.modules,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
    }


def _validate_prepared_runtime_receipt(
    value: Any, *, production: bool, current_phase: str
) -> dict[str, Any]:
    if type(production) is not bool or current_phase not in {
        "pre_spawn",
        "post_spawn",
    }:
        raise OpenAxesExternalModelV1Error(
            "runtime validation phase marker is not exact"
        )
    runtime = _mapping(value, name="prepared runtime receipt")
    _keys(
        runtime,
        {
            "python",
            "platform",
            "packages",
            "torch_imported_in_controller",
            "thread_environment",
        },
        name="prepared runtime receipt",
    )
    python = _mapping(runtime["python"], name="prepared Python runtime")
    _keys(
        python,
        {"version", "implementation", "executable", "cache_tag"},
        name="prepared Python runtime",
    )
    for key in ("version", "implementation", "executable", "cache_tag"):
        if not _exact_str(python[key], name=f"prepared Python {key}"):
            raise OpenAxesExternalModelV1Error(
                f"prepared Python {key} is empty"
            )
    platform_receipt = _mapping(runtime["platform"], name="prepared platform")
    _keys(
        platform_receipt,
        {"system", "release", "machine"},
        name="prepared platform",
    )
    for key in ("system", "release", "machine"):
        if not _exact_str(platform_receipt[key], name=f"prepared platform {key}"):
            raise OpenAxesExternalModelV1Error(
                f"prepared platform {key} is empty"
            )
    package_names = ("numpy", "pandas", "pyarrow", "torch", "PyYAML")
    packages = _mapping(runtime["packages"], name="prepared packages")
    _keys(packages, set(package_names), name="prepared packages")
    if tuple(packages) != package_names:
        raise OpenAxesExternalModelV1Error("prepared package order changed")
    for name in package_names:
        if not _exact_str(packages[name], name=f"prepared package {name}"):
            raise OpenAxesExternalModelV1Error(
                f"prepared package {name} version is empty"
            )
    torch_imported = runtime["torch_imported_in_controller"]
    if type(torch_imported) is not bool:
        raise OpenAxesExternalModelV1Error(
            "prepared controller torch-import state is not exact"
        )
    if production and torch_imported:
        raise OpenAxesExternalModelV1Error(
            "production prepared controller imported torch"
        )
    thread_names = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    threads = _mapping(
        runtime["thread_environment"], name="prepared thread environment"
    )
    _keys(threads, set(thread_names), name="prepared thread environment")
    if tuple(threads) != thread_names or any(
        value is not None and type(value) is not str for value in threads.values()
    ):
        raise OpenAxesExternalModelV1Error(
            "prepared thread environment changed schema"
        )
    current = _runtime_receipt_without_torch_import()
    stable_keys = ("python", "platform", "packages", "thread_environment")
    if _canonical({key: runtime[key] for key in stable_keys}) != _canonical(
        {key: current[key] for key in stable_keys}
    ):
        raise OpenAxesExternalModelV1Error(
            "prepared runtime differs from the current stable runtime"
        )
    if (
        current_phase == "pre_spawn"
        and runtime["torch_imported_in_controller"]
        is not current["torch_imported_in_controller"]
    ):
        raise OpenAxesExternalModelV1Error(
            "prepared runtime differs from the current pre-spawn torch-import state"
        )
    return deepcopy(dict(runtime))


@dataclass(frozen=True)
class PreparedModelInput:
    frame: pd.DataFrame
    manifest: Mapping[str, Any]
    preparation_validation: Mapping[str, Any]
    source_path: Path | None
    preparation_run_dir: Path | None
    preparation_config_path: Path | None
    synthetic_test_only: bool
    parent_authority: Mapping[str, Any] | None


def _validate_preparation_fold_items(value: Any) -> tuple[dict[str, Any], ...]:
    if type(value) is not list or len(value) != 14:
        raise OpenAxesExternalModelV1Error("preparation must declare exactly 14 folds")
    result: list[dict[str, Any]] = []
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
    for index, raw in enumerate(value):
        fold = _mapping(raw, name=f"fold[{index}]")
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
            name=f"fold[{index}]",
        )
        fold_id = _exact_str(fold["fold_id"], name=f"fold[{index}].fold_id")
        family = _exact_str(fold["family"], name=f"fold[{index}].family")
        role = _exact_str(fold["role"], name=f"fold[{index}].role")
        if family not in family_roles or role != family_roles[family] or not fold_id.startswith(f"{family}::"):
            raise OpenAxesExternalModelV1Error("fold family/role/id changed")
        train = fold["train_scenarios"]
        test = fold["test_scenarios"]
        if (
            type(train) is not list
            or type(test) is not list
            or any(type(item) is not str for item in train + test)
            or len(set(train)) != len(train)
            or len(set(test)) != len(test)
            or set(train).intersection(test)
            or set(train).union(test) != expected_scenarios
        ):
            raise OpenAxesExternalModelV1Error("fold scenario partition changed")
        expected_sizes = {
            "scenario_loso": (7, 1),
            "speed_pair_out": (6, 2),
            "shape_out": (4, 4),
        }[family]
        if (len(train), len(test)) != expected_sizes:
            raise OpenAxesExternalModelV1Error("fold train/test size changed")
        _exact_str(
            fold["inner_selection"],
            name="fold inner selection",
            expected="scenario_blocked_training_groups_only",
        )
        _exact_bool(
            fold["outer_test_fitting_allowed"],
            name="outer test fitting",
            expected=False,
        )
        result.append(deepcopy(dict(fold)))
    if [item["family"] for item in result].count("scenario_loso") != 8 or [
        item["family"] for item in result
    ].count("speed_pair_out") != 4 or [item["family"] for item in result].count("shape_out") != 2:
        raise OpenAxesExternalModelV1Error("8+4+2 fold family count changed")
    if len({item["fold_id"] for item in result}) != 14:
        raise OpenAxesExternalModelV1Error("fold ids are not unique")
    return tuple(result)


def _validate_prepared_model_input_structure(
    value: PreparedModelInput, *, production: bool
) -> None:
    """Validate an in-memory structure; this is not production authority."""
    if type(value.synthetic_test_only) is not bool:
        raise OpenAxesExternalModelV1Error("synthetic marker is not exact")
    if production and value.synthetic_test_only:
        raise OpenAxesExternalModelV1Error("synthetic test input cannot enter production")
    if value.synthetic_test_only:
        if value.parent_authority is not None:
            raise OpenAxesExternalModelV1Error(
                "synthetic preparation cannot carry production Git authority"
            )
    else:
        authority = _mapping(value.parent_authority, name="parent Git authority")
        _keys(
            authority,
            {
                "parent_preparation_git_head",
                "model_contract_git_head",
                "parent_source_set_sha256",
                "parent_source_files",
                "parent_sources_unchanged_at_model_head",
                "parent_is_ancestor_of_model_head",
                "working_tree_checked_separately",
                "git_object_proof",
                "receipt_sha256",
            },
            name="parent Git authority",
        )
        for key in ("parent_preparation_git_head", "model_contract_git_head"):
            head = _exact_str(authority[key], name=f"parent authority {key}")
            if re.fullmatch(r"[0-9a-f]{40,64}", head) is None:
                raise OpenAxesExternalModelV1Error("parent authority Git head changed")
        _hex64(authority["parent_source_set_sha256"], name="parent source set hash")
        files = _mapping(authority["parent_source_files"], name="parent source proof files")
        if tuple(files) != PARENT_SOURCE_CLOSURE:
            raise OpenAxesExternalModelV1Error("parent source proof path set changed")
        for relative, raw in files.items():
            proof = _mapping(raw, name=f"parent source proof {relative}")
            _keys(
                proof,
                {
                    "repo_path",
                    "bytes",
                    "sha256",
                    "parent_head_blob_sha256",
                    "model_head_blob_sha256",
                    "working_tree_sha256",
                },
                name=f"parent source proof {relative}",
            )
            recorded_sha = _hex64(proof["sha256"], name="parent source recorded hash")
            _exact_str(proof["repo_path"], name="parent source repository path")
            _exact_int(proof["bytes"], name="parent source byte count")
            for key in (
                "parent_head_blob_sha256",
                "model_head_blob_sha256",
                "working_tree_sha256",
            ):
                if _hex64(proof[key], name=f"parent source {key}") != recorded_sha:
                    raise OpenAxesExternalModelV1Error("parent source Git proof disagrees")
        _exact_bool(
            authority["parent_sources_unchanged_at_model_head"],
            name="parent sources unchanged proof",
            expected=True,
        )
        _exact_bool(
            authority["parent_is_ancestor_of_model_head"],
            name="parent/model ancestry proof",
            expected=True,
        )
        _exact_bool(
            authority["working_tree_checked_separately"],
            name="separate working-tree proof",
            expected=True,
        )
        _exact_str(
            authority["git_object_proof"],
            name="Git object proof mode",
            expected="cat-file_commit_and_show_explicit_revision_path",
        )
        expected_authority_hash = _canonical_sha256(
            {key: authority[key] for key in authority if key != "receipt_sha256"}
        )
        if _hex64(authority["receipt_sha256"], name="parent authority receipt hash") != expected_authority_hash:
            raise OpenAxesExternalModelV1Error("parent authority receipt hash changed")
    validation = _mapping(value.preparation_validation, name="preparation validation")
    _keys(validation, PREPARATION_VALIDATOR_KEYS, name="preparation validation")
    _exact_str(validation["status"], name="preparation status", expected="ok")
    _exact_str(
        validation["contract"],
        name="preparation contract",
        expected=OPENAXES_EXTERNAL_PREPARATION_V1_CONTRACT,
    )
    _hex64(validation["config_sha256"], name="preparation config hash")
    _hex64(validation["source_sha256"], name="preparation source hash")
    _exact_int(validation["prepared_rows"], name="prepared rows")
    scenario_rows = _mapping(validation["scenario_rows"], name="preparation scenario rows")
    expected_scenarios = {
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    }
    if set(scenario_rows) != expected_scenarios or any(
        type(count) is not int or count <= 0 for count in scenario_rows.values()
    ):
        raise OpenAxesExternalModelV1Error("preparation scenario row receipt changed")
    _exact_int(validation["feature_count"], name="feature count", expected=36)
    _exact_int(validation["declared_outer_folds"], name="fold count", expected=14)
    _exact_int(
        validation["aligned_robot_rows_unmatched"], name="unmatched rows", expected=492
    )
    shifts = validation["signed_sensitivity_shifts_ms"]
    if type(shifts) is not list or tuple(shifts) != OPENAXES_MODEL_SHIFT_MS or any(
        type(item) is not int for item in shifts
    ):
        raise OpenAxesExternalModelV1Error("preparation signed shifts changed")
    for key in ("finite", "strict_100_hz_per_scenario", "semantic_replay"):
        _exact_bool(validation[key], name=f"preparation {key}", expected=True)
    blocked = _exact_bool(
        validation["downstream_model_results_blocked_by_uncommitted_contract"],
        name="preparation downstream block",
        expected=False if production else validation[
            "downstream_model_results_blocked_by_uncommitted_contract"
        ],
    )
    if production and blocked:
        raise OpenAxesExternalModelV1Error(
            "preparation contract is uncommitted; downstream model results are blocked"
        )

    manifest = _mapping(value.manifest, name="preparation manifest")
    _exact_str(
        manifest.get("contract"),
        name="manifest contract",
        expected=OPENAXES_EXTERNAL_PREPARATION_V1_CONTRACT,
    )
    schema = _mapping(manifest.get("schema"), name="preparation schema")
    feature_columns = schema.get("model_feature_columns")
    targets = schema.get("target_columns")
    metadata = schema.get("metadata_columns_not_model_inputs")
    if (
        type(feature_columns) is not list
        or len(feature_columns) != 36
        or any(type(item) is not str for item in feature_columns)
        or type(targets) is not list
        or targets != ["tcp_x_m", "tcp_y_m", "tcp_z_m"]
        or type(metadata) is not list
        or metadata != ["row_uid", "scenario_id", "sample_index", "aligned_time_us"]
    ):
        raise OpenAxesExternalModelV1Error("preparation feature/target/metadata schema changed")
    sensor_groups = _mapping(schema.get("sensor_groups"), name="preparation sensor groups")
    if set(sensor_groups) != set(OPENAXES_MODEL_SENSOR_GROUPS) or any(
        type(columns) is not list
        or len(columns) != 9
        or any(type(column) is not str for column in columns)
        for columns in sensor_groups.values()
    ):
        raise OpenAxesExternalModelV1Error("preparation 4x9 sensor grouping changed")
    ordered_sensor_columns = [
        column
        for sensor in OPENAXES_MODEL_SENSOR_GROUPS
        for column in sensor_groups[sensor]
    ]
    if ordered_sensor_columns != feature_columns:
        raise OpenAxesExternalModelV1Error("preparation sensor groups do not match feature order")
    folds = _mapping(manifest.get("folds"), name="preparation folds")
    _validate_preparation_fold_items(folds.get("items"))
    _exact_bool(
        folds.get("sensitivity_may_select_model_or_decide_claim_gate"),
        name="preparation sensitivity selection boundary",
        expected=False,
    )
    provenance = _mapping(manifest.get("provenance"), name="preparation provenance")
    stage = _mapping(provenance.get("stage"), name="preparation stage")
    stage_blocked = stage.get("downstream_model_results_blocked_by_uncommitted_contract")
    if type(stage_blocked) is not bool or stage_blocked != validation[
        "downstream_model_results_blocked_by_uncommitted_contract"
    ]:
        raise OpenAxesExternalModelV1Error("preparation downstream block receipts disagree")
    if production:
        _exact_bool(
            stage.get("contract_sources_committed_at_git_head"),
            name="preparation committed sources",
            expected=True,
        )

    frame = value.frame
    expected_columns = metadata + feature_columns + targets
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != expected_columns:
        raise OpenAxesExternalModelV1Error("prepared dataframe columns changed")
    if len(frame) != validation["prepared_rows"] or len(frame) < 1:
        raise OpenAxesExternalModelV1Error("prepared dataframe row count changed")
    if frame["row_uid"].map(type).ne(str).any() or frame["row_uid"].duplicated().any():
        raise OpenAxesExternalModelV1Error("prepared row_uid is invalid")
    if frame["scenario_id"].map(type).ne(str).any():
        raise OpenAxesExternalModelV1Error("prepared scenario id is invalid")
    if (
        not np.issubdtype(frame["sample_index"].dtype, np.integer)
        or not np.issubdtype(frame["aligned_time_us"].dtype, np.integer)
        or (frame["sample_index"].to_numpy() < 0).any()
    ):
        raise OpenAxesExternalModelV1Error("prepared integer metadata is invalid")
    numeric = frame[feature_columns + targets].to_numpy()
    if numeric.dtype.kind not in "fc" or not np.isfinite(numeric).all():
        raise OpenAxesExternalModelV1Error("prepared model values are non-finite")
    if set(frame["scenario_id"]) != {
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    }:
        raise OpenAxesExternalModelV1Error("prepared scenario set changed")
    observed_scenario_rows = frame.groupby("scenario_id", sort=False).size().to_dict()
    if observed_scenario_rows != dict(scenario_rows):
        raise OpenAxesExternalModelV1Error("prepared scenario row counts disagree")


def _git_bytes(repo: Path, arguments: Sequence[str], *, name: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OpenAxesExternalModelV1Error(f"git authority proof failed: {name}") from exc
    return completed.stdout


def _git_commit_text(repo: Path, revision: str, *, name: str) -> str:
    raw = _git_bytes(repo, ["rev-parse", revision], name=name)
    try:
        value = raw.decode("ascii").strip()
    except UnicodeError as exc:
        raise OpenAxesExternalModelV1Error(f"git commit is not ASCII: {name}") from exc
    if re.fullmatch(r"[0-9a-f]{40,64}", value) is None:
        raise OpenAxesExternalModelV1Error(f"git commit is malformed: {name}")
    return value


def _parent_source_authority_receipt(
    *,
    preparation_manifest: Mapping[str, Any],
    preparation_config_path: Path,
) -> dict[str, Any]:
    """Prove parent blobs are identical at the parent and current model HEADs.

    Every comparison is against an explicit Git object.  Working-tree bytes are
    checked only as a third, separately named boundary and are never confused
    with either committed revision.
    """

    config_path = preparation_config_path.resolve()
    workspace = config_path.parent.parent
    if config_path != workspace / PARENT_SOURCE_CLOSURE[0]:
        raise OpenAxesExternalModelV1Error(
            "preparation config path is not the canonical parent contract path"
        )
    repo_raw = _git_bytes(
        workspace, ["rev-parse", "--show-toplevel"], name="repository root"
    )
    try:
        repo = Path(repo_raw.decode("utf-8").strip()).resolve()
    except UnicodeError as exc:
        raise OpenAxesExternalModelV1Error("git repository root is not UTF-8") from exc
    if not repo.is_dir() or workspace != (repo / workspace.relative_to(repo)).resolve():
        raise OpenAxesExternalModelV1Error("parent workspace is outside its Git repository")

    provenance = _mapping(
        preparation_manifest.get("provenance"), name="preparation provenance"
    )
    repository = _mapping(
        provenance.get("repository"), name="preparation repository provenance"
    )
    source_lock = _mapping(
        provenance.get("contract_and_lock_sources"),
        name="preparation source closure",
    )
    recorded_files = _mapping(
        source_lock.get("files"), name="recorded parent source files"
    )
    repo_paths = _mapping(
        repository.get("contract_source_repo_paths"),
        name="recorded parent repository paths",
    )
    if tuple(recorded_files) != PARENT_SOURCE_CLOSURE or tuple(repo_paths) != PARENT_SOURCE_CLOSURE:
        raise OpenAxesExternalModelV1Error("recorded parent source closure changed")
    parent_head = _exact_str(
        repository.get("git_head"), name="parent preparation Git HEAD"
    )
    if re.fullmatch(r"[0-9a-f]{40,64}", parent_head) is None:
        raise OpenAxesExternalModelV1Error("parent preparation Git HEAD is malformed")
    _git_bytes(
        repo,
        ["cat-file", "-e", f"{parent_head}^{{commit}}"],
        name="parent preparation commit object",
    )
    model_head = _git_commit_text(repo, "HEAD", name="model contract HEAD")
    _git_bytes(
        repo,
        ["cat-file", "-e", f"{model_head}^{{commit}}"],
        name="model contract commit object",
    )
    _git_bytes(
        repo,
        ["merge-base", "--is-ancestor", parent_head, model_head],
        name="parent preparation/model ancestry",
    )

    recorded_source_set = _hex64(
        source_lock.get("source_set_sha256"), name="parent source set hash"
    )
    if recorded_source_set != _canonical_sha256(recorded_files):
        raise OpenAxesExternalModelV1Error("recorded parent source set hash changed")

    files: dict[str, Any] = {}
    for relative in PARENT_SOURCE_CLOSURE:
        recorded = _mapping(
            recorded_files[relative], name=f"recorded parent source {relative}"
        )
        _keys(recorded, {"bytes", "sha256"}, name=f"recorded parent source {relative}")
        recorded_bytes = _exact_int(
            recorded["bytes"], name=f"recorded parent bytes {relative}"
        )
        recorded_sha = _hex64(
            recorded["sha256"], name=f"recorded parent SHA-256 {relative}"
        )
        repo_path = _exact_str(repo_paths[relative], name=f"parent repo path {relative}")
        candidate = Path(repo_path)
        if candidate.is_absolute() or ".." in candidate.parts or "\\" in repo_path:
            raise OpenAxesExternalModelV1Error("parent repository path is unsafe")
        parent_blob = _git_bytes(
            repo,
            ["show", f"{parent_head}:{repo_path}"],
            name=f"parent blob {relative}",
        )
        model_blob = _git_bytes(
            repo,
            ["show", f"{model_head}:{repo_path}"],
            name=f"model-head parent blob {relative}",
        )
        working_path = workspace / relative
        if working_path.is_symlink() or not working_path.is_file():
            raise OpenAxesExternalModelV1Error("parent working-tree source is missing")
        working_blob = working_path.read_bytes()
        observed = {
            "repo_path": repo_path,
            "bytes": recorded_bytes,
            "sha256": recorded_sha,
            "parent_head_blob_sha256": hashlib.sha256(parent_blob).hexdigest(),
            "model_head_blob_sha256": hashlib.sha256(model_blob).hexdigest(),
            "working_tree_sha256": hashlib.sha256(working_blob).hexdigest(),
        }
        if (
            len(parent_blob) != recorded_bytes
            or len(model_blob) != recorded_bytes
            or len(working_blob) != recorded_bytes
            or any(
                observed[key] != recorded_sha
                for key in (
                    "parent_head_blob_sha256",
                    "model_head_blob_sha256",
                    "working_tree_sha256",
                )
            )
        ):
            raise OpenAxesExternalModelV1Error(
                f"parent contract source changed across Git boundaries: {relative}"
            )
        files[relative] = observed
    receipt = {
        "parent_preparation_git_head": parent_head,
        "model_contract_git_head": model_head,
        "parent_source_set_sha256": _hex64(
            source_lock.get("source_set_sha256"), name="parent source set hash"
        ),
        "parent_source_files": files,
        "parent_sources_unchanged_at_model_head": True,
        "parent_is_ancestor_of_model_head": True,
        "working_tree_checked_separately": True,
        "git_object_proof": "cat-file_commit_and_show_explicit_revision_path",
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def _run_parent_public_validator_at_recorded_git_state(
    *,
    source_path: Path,
    preparation_config_path: Path,
    preparation_run_dir: Path,
    preparation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the public parent validator in its recorded Git/status context."""

    from .openaxes_external_v1 import validate_openaxes_external_v1
    from .openaxes_external_v1_config import load_openaxes_config

    workspace = preparation_config_path.resolve().parent.parent
    repo = Path(
        _git_bytes(
            workspace, ["rev-parse", "--show-toplevel"], name="parent replay root"
        )
        .decode("utf-8")
        .strip()
    ).resolve()
    provenance = _mapping(
        preparation_manifest.get("provenance"), name="preparation provenance"
    )
    repository = _mapping(
        provenance.get("repository"), name="preparation repository provenance"
    )
    parent_head = _exact_str(repository.get("git_head"), name="recorded parent HEAD")
    dirty = repository.get("dirty_path_snapshot")
    if type(dirty) is not list:
        raise OpenAxesExternalModelV1Error("recorded parent dirty snapshot is not a list")

    temporary_root = Path(tempfile.mkdtemp(prefix="openaxes-parent-validator-"))
    worktree = temporary_root / "worktree"
    added = False
    try:
        _git_bytes(
            repo,
            ["worktree", "add", "--detach", str(worktree), parent_head],
            name="recorded parent validation worktree",
        )
        added = True
        for index, raw in enumerate(dirty):
            entry = _mapping(raw, name=f"recorded dirty entry {index}")
            _keys(entry, {"path", "status"}, name=f"recorded dirty entry {index}")
            relative = _exact_str(entry["path"], name="recorded dirty path")
            _exact_str(entry["status"], name="recorded dirty status", expected="??")
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative:
                raise OpenAxesExternalModelV1Error("recorded dirty path is unsafe")
            placeholder = worktree / candidate
            placeholder.parent.mkdir(parents=True, exist_ok=True)
            placeholder.touch(exist_ok=False)
        parent_workspace = worktree / workspace.relative_to(repo)
        replay_config_path = parent_workspace / PARENT_SOURCE_CLOSURE[0]
        replay_config = load_openaxes_config(replay_config_path)
        validation = validate_openaxes_external_v1(
            source_path, replay_config, preparation_run_dir
        )
    finally:
        if added:
            try:
                _git_bytes(
                    repo,
                    ["worktree", "remove", "--force", str(worktree)],
                    name="recorded parent validation worktree cleanup",
                )
            finally:
                _git_bytes(repo, ["worktree", "prune"], name="worktree metadata cleanup")
        shutil.rmtree(temporary_root, ignore_errors=True)
    return deepcopy(dict(validation))


def authenticate_openaxes_prepared_input(
    source_path: str | Path,
    preparation_config_path: str | Path,
    preparation_run_dir: str | Path,
) -> PreparedModelInput:
    """Authenticate production input from paths through the public parent API."""

    if isinstance(source_path, PreparedModelInput) or isinstance(
        preparation_config_path, PreparedModelInput
    ):
        raise OpenAxesExternalModelV1Error(
            "production preparation authentication accepts paths, not prepared objects"
        )
    source = Path(source_path).resolve()
    config_path = Path(preparation_config_path).resolve()
    run = Path(preparation_run_dir).resolve()
    for path, name in (
        (source, "official source"),
        (config_path, "preparation config"),
        (run / "manifest.json", "preparation manifest"),
        (run / "prepared.parquet", "preparation parquet"),
        (run / "COMPLETE.json", "preparation COMPLETE"),
    ):
        if path.is_symlink() or not path.is_file():
            raise OpenAxesExternalModelV1Error(f"production {name} path is missing")
    manifest = _mapping(
        _strict_json_load(run / "manifest.json"), name="preparation manifest"
    )
    authority = _parent_source_authority_receipt(
        preparation_manifest=manifest,
        preparation_config_path=config_path,
    )
    validation = _run_parent_public_validator_at_recorded_git_state(
        source_path=source,
        preparation_config_path=config_path,
        preparation_run_dir=run,
        preparation_manifest=manifest,
    )
    try:
        frame = pd.read_parquet(run / "prepared.parquet")
    except Exception as exc:
        raise OpenAxesExternalModelV1Error("cannot load authenticated prepared parquet") from exc
    result = PreparedModelInput(
        frame=frame,
        manifest=manifest,
        preparation_validation=validation,
        source_path=source,
        preparation_run_dir=run,
        preparation_config_path=config_path,
        synthetic_test_only=False,
        parent_authority=authority,
    )
    _validate_prepared_model_input_structure(result, production=True)
    return result


def make_synthetic_prepared_model_input(
    frame: pd.DataFrame, manifest: Mapping[str, Any], validation: Mapping[str, Any]
) -> PreparedModelInput:
    """Explicit private-test seam; the marker can never validate as production."""

    result = PreparedModelInput(
        frame=frame.copy(),
        manifest=deepcopy(dict(manifest)),
        preparation_validation=deepcopy(dict(validation)),
        source_path=None,
        preparation_run_dir=None,
        preparation_config_path=None,
        synthetic_test_only=True,
        parent_authority=None,
    )
    _validate_prepared_model_input_structure(result, production=False)
    return result


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    ordinal: int
    method_id: str
    fold_id: str
    family: str
    role: str
    train_scenarios: tuple[str, ...]
    test_scenarios: tuple[str, ...]
    seed: int
    spec_sha256: str

    def payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
            "task_id": self.task_id,
            "ordinal": self.ordinal,
            "method_id": self.method_id,
            "fold_id": self.fold_id,
            "family": self.family,
            "role": self.role,
            "train_scenarios": list(self.train_scenarios),
            "test_scenarios": list(self.test_scenarios),
            "seed": self.seed,
            "checkpoint_selection_scope": "training_scenarios_only",
            "outer_eval_count": 1,
        }
        if include_hash:
            result["spec_sha256"] = self.spec_sha256
        return result


def build_model_task_grid(preparation_manifest: Mapping[str, Any]) -> tuple[TaskSpec, ...]:
    manifest = _mapping(preparation_manifest, name="preparation manifest")
    folds = _mapping(manifest.get("folds"), name="preparation folds")
    fold_items = _validate_preparation_fold_items(folds.get("items"))
    tasks: list[TaskSpec] = []
    ordinal = 0
    for method_id in OPENAXES_MODEL_METHOD_IDS:
        for fold in fold_items:
            for seed in OPENAXES_MODEL_SEEDS:
                base = {
                    "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
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
                digest = _canonical_sha256(base)
                task_id = f"task_{ordinal:04d}_{digest[:12]}"
                tasks.append(
                    TaskSpec(
                        task_id=task_id,
                        ordinal=ordinal,
                        method_id=method_id,
                        fold_id=fold["fold_id"],
                        family=fold["family"],
                        role=fold["role"],
                        train_scenarios=tuple(fold["train_scenarios"]),
                        test_scenarios=tuple(fold["test_scenarios"]),
                        seed=seed,
                        spec_sha256=digest,
                    )
                )
                ordinal += 1
    if len(tasks) != 336 or len({task.task_id for task in tasks}) != 336:
        raise OpenAxesExternalModelV1Error("exact 336-task grid did not close")
    if sum(task.role == "primary" for task in tasks) != 192 or sum(
        task.role == "stress" for task in tasks
    ) != 144:
        raise OpenAxesExternalModelV1Error("192 primary + 144 stress fit count changed")
    return tuple(tasks)


def verify_task_payload(value: Any, expected: TaskSpec) -> None:
    payload = _mapping(value, name="task payload")
    expected_payload = expected.payload()
    _keys(payload, set(expected_payload), name="task payload")
    if _canonical(payload) != _canonical(expected_payload):
        raise OpenAxesExternalModelV1Error("task payload changed")


def validate_gpu_inventory(value: Any) -> tuple[dict[str, Any], ...]:
    if type(value) is not list or len(value) != 4:
        raise OpenAxesExternalModelV1Error("exactly four physical GPUs are required")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        row = _mapping(raw, name=f"gpu[{index}]")
        _keys(row, {"physical_index", "uuid", "pci_bus_id", "name"}, name=f"gpu[{index}]")
        _exact_int(row["physical_index"], name="GPU physical index", expected=index)
        uuid = _exact_str(row["uuid"], name="GPU UUID")
        pci = _exact_str(row["pci_bus_id"], name="GPU PCI bus id")
        name = _exact_str(row["name"], name="GPU name")
        if not uuid.startswith("GPU-") or not re.fullmatch(r"[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]", pci):
            raise OpenAxesExternalModelV1Error("GPU UUID/PCI identity is malformed")
        _canonical_gpu_uuid(uuid, name="GPU UUID")
        rows.append({"physical_index": index, "uuid": uuid, "pci_bus_id": pci, "name": name})
    if len({_canonical_gpu_uuid(row["uuid"], name="GPU UUID") for row in rows}) != 4 or len({row["pci_bus_id"].lower() for row in rows}) != 4:
        raise OpenAxesExternalModelV1Error("GPU physical identities are not unique")
    return tuple(rows)


def assign_tasks_to_gpu_slots(
    tasks: Sequence[TaskSpec], gpu_inventory: Any, workers_per_gpu: int
) -> tuple[dict[str, Any], ...]:
    gpus = validate_gpu_inventory(gpu_inventory)
    if type(workers_per_gpu) is not int or not 1 <= workers_per_gpu <= 3:
        raise OpenAxesExternalModelV1Error("workers_per_gpu must be an exact integer 1..3")
    slots: list[dict[str, Any]] = []
    for gpu in gpus:
        for local_slot in range(workers_per_gpu):
            slot_index = len(slots)
            slots.append(
                {
                    "slot_index": slot_index,
                    "local_slot": local_slot,
                    "gpu": deepcopy(gpu),
                    "task_ids": [],
                }
            )
    for index, task in enumerate(tasks):
        slots[index % len(slots)]["task_ids"].append(task.task_id)
    for slot in slots:
        slot["assignment_sha256"] = _canonical_sha256(
            {
                "slot_index": slot["slot_index"],
                "local_slot": slot["local_slot"],
                "gpu": slot["gpu"],
                "task_ids": slot["task_ids"],
            }
        )
    flattened = [task_id for slot in slots for task_id in slot["task_ids"]]
    if len(flattened) != len(tasks) or set(flattened) != {task.task_id for task in tasks}:
        raise OpenAxesExternalModelV1Error("GPU slot assignment lost or duplicated a task")
    return tuple(slots)


def _git_contract_receipt(workspace: Path) -> dict[str, Any]:
    try:
        root_result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "--show-toplevel"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        head_result = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        status_result = subprocess.run(
            [
                "git",
                "-C",
                str(workspace),
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        tracked_result = subprocess.run(
            ["git", "-C", str(workspace), "ls-files", "-z", "--", *MODEL_SOURCE_CLOSURE],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OpenAxesExternalModelV1Error("cannot capture model contract git provenance") from exc
    try:
        repo = Path(root_result.stdout.decode("utf-8").strip())
        head = head_result.stdout.decode("ascii").strip()
        tracked_from_workspace = {
            item.decode("utf-8")
            for item in tracked_result.stdout.split(b"\x00")
            if item
        }
        status_text = status_result.stdout.decode("utf-8")
    except UnicodeError as exc:
        raise OpenAxesExternalModelV1Error("git provenance is not canonical text") from exc
    if not repo.is_absolute() or re.fullmatch(r"[0-9a-f]{40,64}", head) is None:
        raise OpenAxesExternalModelV1Error("git root/HEAD provenance is invalid")
    workspace_prefix = workspace.relative_to(repo).as_posix()
    tracked = {
        (
            f"{workspace_prefix}/{path}"
            if workspace_prefix != "." and not path.startswith(f"{workspace_prefix}/")
            else path
        )
        for path in tracked_from_workspace
    }
    required_repo_paths = {
        f"{workspace_prefix}/{relative}" if workspace_prefix != "." else relative
        for relative in MODEL_SOURCE_CLOSURE
    }
    dirty_paths: set[str] = set()
    parts = status_result.stdout.split(b"\x00")
    for raw in parts:
        if not raw:
            continue
        if len(raw) < 4 or raw[2:3] != b" ":
            raise OpenAxesExternalModelV1Error("git porcelain snapshot is malformed")
        try:
            dirty_paths.add(raw[3:].decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise OpenAxesExternalModelV1Error("git dirty path is not UTF-8") from exc
    committed = required_repo_paths.issubset(tracked) and not required_repo_paths.intersection(dirty_paths)
    head_source_files: dict[str, Any] = {}
    if committed:
        for relative in MODEL_SOURCE_CLOSURE:
            repo_path = (
                f"{workspace_prefix}/{relative}"
                if workspace_prefix != "."
                else relative
            )
            blob = _git_bytes(
                repo,
                ["show", f"{head}:{repo_path}"],
                name=f"model contract HEAD blob {relative}",
            )
            working = workspace / relative
            receipt = _reference(working, relative=relative)
            if (
                len(blob) != receipt["bytes"]
                or hashlib.sha256(blob).hexdigest() != receipt["sha256"]
            ):
                raise OpenAxesExternalModelV1Error(
                    f"model contract working tree differs from committed HEAD: {relative}"
                )
            head_source_files[relative] = {
                "repo_path": repo_path,
                "bytes": receipt["bytes"],
                "sha256": receipt["sha256"],
                "head_blob_sha256": hashlib.sha256(blob).hexdigest(),
            }
    return {
        "model_contract_git_head": head,
        "dirty": bool(status_text),
        "porcelain_sha256": hashlib.sha256(status_result.stdout).hexdigest(),
        "required_repo_paths": sorted(required_repo_paths),
        "tracked_required_repo_paths": sorted(required_repo_paths.intersection(tracked)),
        "dirty_required_repo_paths": sorted(required_repo_paths.intersection(dirty_paths)),
        "model_source_blobs_at_head": head_source_files,
        "model_source_blobs_at_head_sha256": _canonical_sha256(head_source_files),
        "model_contract_sources_committed_at_git_head": committed,
        "downstream_results_blocked_by_uncommitted_contract": not committed,
    }


def _external_reference(path: Path, *, name: str) -> dict[str, Any]:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise OpenAxesExternalModelV1Error(f"external authority is not a regular file: {name}")
    return {
        "absolute_path": str(resolved),
        "bytes": resolved.stat(follow_symlinks=False).st_size,
        "sha256": sha256_file(resolved),
    }


def _verify_external_reference(value: Any, *, name: str) -> Path:
    reference = _mapping(value, name=name)
    _keys(reference, {"absolute_path", "bytes", "sha256"}, name=name)
    raw = _exact_str(reference["absolute_path"], name=f"{name}.absolute_path")
    path = Path(raw)
    if not path.is_absolute() or str(path.resolve()) != raw:
        raise OpenAxesExternalModelV1Error(f"{name} path is not canonical absolute")
    _exact_int(reference["bytes"], name=f"{name}.bytes")
    _hex64(reference["sha256"], name=f"{name}.sha256")
    if _external_reference(path, name=name) != dict(reference):
        raise OpenAxesExternalModelV1Error(f"{name} changed")
    return path


def prepare_openaxes_external_model_run(
    model_config: Mapping[str, Any],
    source_path: str | Path,
    preparation_config_path: str | Path,
    preparation_run_dir: str | Path,
    output_dir: str | Path,
    gpu_inventory: Any,
) -> dict[str, Any]:
    """Production preparation: paths only, with a fresh public-parent replay."""

    if any(
        isinstance(value, PreparedModelInput)
        for value in (source_path, preparation_config_path, preparation_run_dir)
    ):
        raise OpenAxesExternalModelV1Error(
            "production preparation accepts source/config/run paths, not PreparedModelInput"
        )
    prepared_input = authenticate_openaxes_prepared_input(
        source_path, preparation_config_path, preparation_run_dir
    )
    return _prepare_openaxes_external_model_run_from_input(
        model_config,
        prepared_input,
        output_dir,
        gpu_inventory,
        _private_test_only_mode=False,
    )


def _prepare_synthetic_openaxes_external_model_run_for_test(
    model_config: Mapping[str, Any],
    prepared_input: PreparedModelInput,
    output_dir: str | Path,
    gpu_inventory: Any,
) -> dict[str, Any]:
    """Private synthetic fixture seam; it cannot construct production evidence."""

    return _prepare_openaxes_external_model_run_from_input(
        model_config,
        prepared_input,
        output_dir,
        gpu_inventory,
        _private_test_only_mode=True,
    )


def _prepare_openaxes_external_model_run_from_input(
    model_config: Mapping[str, Any],
    prepared_input: PreparedModelInput,
    output_dir: str | Path,
    gpu_inventory: Any,
    *,
    _private_test_only_mode: bool = False,
) -> dict[str, Any]:
    """Internal writer after either production authentication or test marking."""

    validate_openaxes_external_model_config(model_config)
    if type(_private_test_only_mode) is not bool:
        raise OpenAxesExternalModelV1Error("private test mode marker is not exact")
    production = not _private_test_only_mode
    if prepared_input.synthetic_test_only != _private_test_only_mode:
        raise OpenAxesExternalModelV1Error("synthetic preparation/test mode mismatch")
    _validate_prepared_model_input_structure(prepared_input, production=production)
    gpus = validate_gpu_inventory(gpu_inventory)
    workspace = _workspace_from_model_config(model_config)
    output = Path(output_dir).resolve()
    if os.path.lexists(output):
        raise OpenAxesExternalModelV1Error(f"model prepared output already exists: {output}")
    if prepared_input.preparation_run_dir is not None:
        preparation_root = prepared_input.preparation_run_dir.resolve()
        if output == preparation_root or output in preparation_root.parents or preparation_root in output.parents:
            raise OpenAxesExternalModelV1Error("model output and preparation run must be path-disjoint")
    tasks = build_model_task_grid(prepared_input.manifest)
    source_receipts = _model_source_receipts(workspace)
    git_receipt = _git_contract_receipt(workspace)
    runtime = _runtime_receipt_without_torch_import()
    if production and runtime["torch_imported_in_controller"]:
        raise OpenAxesExternalModelV1Error("production controller imported torch before spawn")
    if production and (
        not git_receipt["model_contract_sources_committed_at_git_head"]
        or git_receipt["downstream_results_blocked_by_uncommitted_contract"]
    ):
        raise OpenAxesExternalModelV1Error(
            "model contract sources must be committed unchanged at model HEAD"
        )

    preparation_artifacts: dict[str, Any]
    if prepared_input.preparation_run_dir is None:
        preparation_artifacts = {
            "synthetic_test_only": True,
            "manifest_sha256": _canonical_sha256(prepared_input.manifest),
            "frame_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(prepared_input.frame, index=True).values.tobytes()
            ).hexdigest(),
        }
    else:
        preparation_artifacts = {
            "synthetic_test_only": False,
            "source": _external_reference(prepared_input.source_path, name="official source"),
            "config": _external_reference(
                prepared_input.preparation_config_path, name="preparation config"
            ),
            "manifest": _external_reference(
                prepared_input.preparation_run_dir / "manifest.json", name="preparation manifest"
            ),
            "parquet": _external_reference(
                prepared_input.preparation_run_dir / "prepared.parquet", name="preparation parquet"
            ),
            "complete": _external_reference(
                prepared_input.preparation_run_dir / "COMPLETE.json", name="preparation COMPLETE"
            ),
        }
    preparation_snapshot = {
        "validation": deepcopy(dict(prepared_input.preparation_validation)),
        "artifacts": preparation_artifacts,
        "schema": deepcopy(dict(prepared_input.manifest["schema"])),
        "folds": deepcopy(dict(prepared_input.manifest["folds"])),
        "preprocessing": deepcopy(dict(prepared_input.manifest["preprocessing"])),
        "limitations": deepcopy(dict(prepared_input.manifest["limitations"])),
        "manifest_semantic_sha256": _canonical_sha256(prepared_input.manifest),
        "parent_git_authority": (
            None
            if prepared_input.parent_authority is None
            else deepcopy(dict(prepared_input.parent_authority))
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent)))
    try:
        task_dir = temporary / MODEL_TASKS_DIR
        task_dir.mkdir()
        task_index = task_dir / "index.json"
        _atomic_json(
            task_index,
            {
                "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
                "task_count": 336,
                "primary_task_count": 192,
                "stress_task_count": 144,
                "tasks": [task.payload() for task in tasks],
            },
        )
        manifest = {
            "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
            "config_sha256": OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
            "stage": "prepared_before_external_model_results",
            "synthetic_test_only": prepared_input.synthetic_test_only,
            "preparation": preparation_snapshot,
            "task_grid": {
                "methods": list(OPENAXES_MODEL_METHOD_IDS),
                "folds": 14,
                "seeds": list(OPENAXES_MODEL_SEEDS),
                "scientific_fits": 336,
                "primary_fits": 192,
                "stress_fits": 144,
                "task_index": _reference(task_index, relative=f"{MODEL_TASKS_DIR}/index.json"),
            },
            "gpu_inventory": list(gpus),
            "execution": {
                "workers_per_gpu_runtime_range": [1, 3],
                "workers_per_gpu_exact": model_config["execution"][
                    "workers_per_gpu_exact"
                ],
                "spawn_persistent_process_per_slot": True,
                "per_slot_fifo_serial": True,
                "visible_child_device": "cuda:0",
                "phase_chain": list(OPENAXES_MODEL_PHASES),
                "immutable_execution_session_required": True,
                "exact_resume_session_lineage_required": True,
                "child_gpu_ready_receipts_required": True,
                "persistent_attempt_ledger_survives_quarantine": True,
                "prelaunch_session_recovery": deepcopy(
                    dict(
                        _mapping(
                            model_config["execution"]["prelaunch_session_recovery"],
                            name="configured prelaunch session recovery",
                        )
                    )
                ),
            },
            "source_closure": {
                "files": source_receipts,
                "set_sha256": _canonical_sha256(source_receipts),
            },
            "repository": git_receipt,
            "runtime": runtime,
            "results_started": False,
            "challenge_route_present": False,
        }
        manifest_path = temporary / MODEL_PREPARED_MANIFEST
        _atomic_json(manifest_path, manifest)
        marker = {
            "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
            "config_sha256": OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
            "prepared_manifest": _reference(manifest_path, relative=MODEL_PREPARED_MANIFEST),
            "task_index": _reference(task_index, relative=f"{MODEL_TASKS_DIR}/index.json"),
            "prepared_runtime_sha256": _canonical_sha256(runtime),
            "synthetic_test_only": prepared_input.synthetic_test_only,
            "model_contract_sources_committed_at_git_head": git_receipt[
                "model_contract_sources_committed_at_git_head"
            ],
            "downstream_results_blocked_by_uncommitted_contract": git_receipt[
                "downstream_results_blocked_by_uncommitted_contract"
            ],
            "immutable": True,
        }
        _atomic_json(temporary / MODEL_PREPARED_MARKER, marker)
        (temporary / "runtime" / "ledgers").mkdir(parents=True)
        (temporary / MODEL_EXECUTION_LAUNCHES_DIR).mkdir(parents=True)
        (temporary / MODEL_WORKER_READY_DIR).mkdir(parents=True)
        (temporary / MODEL_ATTEMPT_LEDGER_DIR).mkdir(parents=True)
        (temporary / MODEL_ATTEMPT_ROSTER_DIR).mkdir(parents=True)
        (temporary / MODEL_RESULTS_DIR).mkdir()
        (temporary / MODEL_QUARANTINE_DIR).mkdir()
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def _load_task_index(root: Path, manifest: Mapping[str, Any]) -> tuple[TaskSpec, ...]:
    task_directory = root / MODEL_TASKS_DIR
    if task_directory.is_symlink() or not task_directory.is_dir():
        raise OpenAxesExternalModelV1Error("task-index directory changed")
    task_members = list(task_directory.iterdir())
    if (
        {path.name for path in task_members} != {"index.json"}
        or any(path.is_symlink() or not path.is_file() for path in task_members)
    ):
        raise OpenAxesExternalModelV1Error("task-index directory member set changed")
    task_grid = _mapping(manifest.get("task_grid"), name="task grid")
    index_path = _verify_reference(root, task_grid.get("task_index"), name="task index")
    if index_path != root / MODEL_TASKS_DIR / "index.json":
        raise OpenAxesExternalModelV1Error("task index path changed")
    index = _mapping(_strict_json_load(index_path), name="task index payload")
    _keys(
        index,
        {"contract", "task_count", "primary_task_count", "stress_task_count", "tasks"},
        name="task index payload",
    )
    _exact_str(index["contract"], name="task index contract", expected=OPENAXES_EXTERNAL_MODEL_V1_CONTRACT)
    _exact_int(index["task_count"], name="task count", expected=336)
    _exact_int(index["primary_task_count"], name="primary tasks", expected=192)
    _exact_int(index["stress_task_count"], name="stress tasks", expected=144)
    snapshot = _mapping(manifest.get("preparation"), name="preparation snapshot")
    folds = _mapping(snapshot.get("folds"), name="preparation fold snapshot")
    expected = build_model_task_grid({"folds": folds})
    raw_tasks = index["tasks"]
    if type(raw_tasks) is not list or len(raw_tasks) != 336:
        raise OpenAxesExternalModelV1Error("task index list changed")
    for payload, task in zip(raw_tasks, expected, strict=True):
        verify_task_payload(payload, task)
    return expected


def validate_openaxes_external_model_prepared_run(
    model_config: Mapping[str, Any],
    run_dir: str | Path,
    *,
    allow_synthetic_test_only: bool = False,
) -> dict[str, Any]:
    validate_openaxes_external_model_config(model_config)
    if type(allow_synthetic_test_only) is not bool:
        raise OpenAxesExternalModelV1Error("synthetic validation marker is not exact")
    root = Path(run_dir).resolve()
    if root.is_symlink() or not root.is_dir():
        raise OpenAxesExternalModelV1Error("model prepared run is not a regular directory")
    required = {
        MODEL_PREPARED_MANIFEST,
        MODEL_PREPARED_MARKER,
        MODEL_TASKS_DIR,
        "runtime",
        MODEL_RESULTS_DIR,
        MODEL_QUARANTINE_DIR,
    }
    observed_members = {path.name for path in root.iterdir()}
    allowed_members = (
        required,
        required | {MODEL_DEVELOPMENT_MANIFEST},
        required | {MODEL_DEVELOPMENT_MANIFEST, MODEL_COMPLETE_MARKER},
    )
    if observed_members not in allowed_members or any(path.is_symlink() for path in root.iterdir()):
        raise OpenAxesExternalModelV1Error("model prepared root member set changed")
    runtime_root = root / "runtime"
    if runtime_root.is_symlink() or not runtime_root.is_dir():
        raise OpenAxesExternalModelV1Error("model runtime root changed")
    runtime_required = {
        "ledgers",
        "execution_launches",
        "worker_ready",
        "attempt_ledger",
        "attempt_roster",
    }
    runtime_members = {path.name for path in runtime_root.iterdir()}
    allowed_runtime_members = (
        runtime_required,
        runtime_required | {Path(MODEL_PRE_FIT_VALIDATION).name},
        runtime_required
        | {
            Path(MODEL_PRE_FIT_VALIDATION).name,
            Path(MODEL_EXECUTION_SESSION).name,
        },
        runtime_required
        | {
            Path(MODEL_PRE_FIT_VALIDATION).name,
            Path(MODEL_EXECUTION_SESSION).name,
            Path(MODEL_PRELAUNCH_SESSION_RECOVERY).name,
        },
        runtime_required
        | {
            Path(MODEL_PRE_FIT_VALIDATION).name,
            Path(MODEL_EXECUTION_SESSION).name,
            Path(MODEL_FULL_SIGNED_SOURCE_REPLAY).name,
        },
        runtime_required
        | {
            Path(MODEL_PRE_FIT_VALIDATION).name,
            Path(MODEL_EXECUTION_SESSION).name,
            Path(MODEL_PRELAUNCH_SESSION_RECOVERY).name,
            Path(MODEL_FULL_SIGNED_SOURCE_REPLAY).name,
        },
    )
    if runtime_members not in allowed_runtime_members or any(
        path.is_symlink() for path in runtime_root.iterdir()
    ):
        raise OpenAxesExternalModelV1Error("model runtime member set changed")
    manifest = _mapping(
        _strict_json_load(root / MODEL_PREPARED_MANIFEST), name="model prepared manifest"
    )
    _keys(
        manifest,
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
        },
        name="model prepared manifest",
    )
    _exact_str(manifest["contract"], name="model contract", expected=OPENAXES_EXTERNAL_MODEL_V1_CONTRACT)
    _hex64(manifest["config_sha256"], name="model config hash")
    if manifest["config_sha256"] != OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256:
        raise OpenAxesExternalModelV1Error("model config hash changed")
    _exact_str(
        manifest["stage"], name="prepared stage", expected="prepared_before_external_model_results"
    )
    synthetic = manifest["synthetic_test_only"]
    if type(synthetic) is not bool or (synthetic and not allow_synthetic_test_only):
        raise OpenAxesExternalModelV1Error("synthetic model run is not production evidence")
    _exact_bool(manifest["results_started"], name="prepared results marker", expected=False)
    _exact_bool(manifest["challenge_route_present"], name="challenge route", expected=False)
    validate_gpu_inventory(manifest["gpu_inventory"])
    task_grid = _mapping(manifest["task_grid"], name="model task grid")
    _keys(
        task_grid,
        {
            "methods",
            "folds",
            "seeds",
            "scientific_fits",
            "primary_fits",
            "stress_fits",
            "task_index",
        },
        name="model task grid",
    )
    if task_grid["methods"] != list(OPENAXES_MODEL_METHOD_IDS) or task_grid[
        "seeds"
    ] != list(OPENAXES_MODEL_SEEDS):
        raise OpenAxesExternalModelV1Error("model task methods/seeds changed")
    for key, expected in (
        ("folds", 14),
        ("scientific_fits", 336),
        ("primary_fits", 192),
        ("stress_fits", 144),
    ):
        _exact_int(task_grid[key], name=f"task grid {key}", expected=expected)
    execution = _mapping(manifest["execution"], name="prepared execution")
    _keys(
        execution,
        {
            "workers_per_gpu_runtime_range",
            "workers_per_gpu_exact",
            "spawn_persistent_process_per_slot",
            "per_slot_fifo_serial",
            "visible_child_device",
            "phase_chain",
            "immutable_execution_session_required",
            "exact_resume_session_lineage_required",
            "child_gpu_ready_receipts_required",
            "persistent_attempt_ledger_survives_quarantine",
            "prelaunch_session_recovery",
        },
        name="prepared execution",
    )
    if execution["workers_per_gpu_runtime_range"] != [1, 3] or execution[
        "phase_chain"
    ] != list(OPENAXES_MODEL_PHASES):
        raise OpenAxesExternalModelV1Error("prepared execution range/phases changed")
    _exact_int(
        execution["workers_per_gpu_exact"],
        name="prepared frozen workers/GPU",
        expected=2,
    )
    for key in (
        "spawn_persistent_process_per_slot",
        "per_slot_fifo_serial",
        "immutable_execution_session_required",
        "exact_resume_session_lineage_required",
        "child_gpu_ready_receipts_required",
        "persistent_attempt_ledger_survives_quarantine",
    ):
        _exact_bool(execution[key], name=f"prepared execution {key}", expected=True)
    configured_prelaunch = _mapping(
        model_config["execution"]["prelaunch_session_recovery"],
        name="configured prelaunch session recovery",
    )
    prepared_prelaunch = _mapping(
        execution["prelaunch_session_recovery"],
        name="prepared prelaunch session recovery",
    )
    if _canonical(prepared_prelaunch) != _canonical(configured_prelaunch):
        raise OpenAxesExternalModelV1Error(
            "prepared prelaunch recovery contract changed"
        )
    _exact_str(
        execution["visible_child_device"], name="prepared child device", expected="cuda:0"
    )
    source_closure = _mapping(manifest["source_closure"], name="model source closure")
    _keys(source_closure, {"files", "set_sha256"}, name="model source closure")
    files = _mapping(source_closure["files"], name="model source files")
    if tuple(files) != MODEL_SOURCE_CLOSURE:
        raise OpenAxesExternalModelV1Error("model source closure path set/order changed")
    workspace = _workspace_from_model_config(model_config)
    observed_sources = _model_source_receipts(workspace)
    if _canonical(files) != _canonical(observed_sources) or source_closure["set_sha256"] != _canonical_sha256(observed_sources):
        raise OpenAxesExternalModelV1Error("model source closure changed")
    tasks = _load_task_index(root, manifest)
    marker = _mapping(_strict_json_load(root / MODEL_PREPARED_MARKER), name="prepared marker")
    _keys(
        marker,
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
        },
        name="prepared marker",
    )
    _exact_str(
        marker["contract"],
        name="prepared marker contract",
        expected=OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
    )
    _exact_str(
        marker["config_sha256"],
        name="prepared marker config hash",
        expected=OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
    )
    prepared_reference = _verify_reference(
        root, marker["prepared_manifest"], name="prepared manifest reference"
    )
    task_reference = _verify_reference(
        root, marker["task_index"], name="prepared task reference"
    )
    if (
        prepared_reference != root / MODEL_PREPARED_MANIFEST
        or task_reference != root / MODEL_TASKS_DIR / "index.json"
    ):
        raise OpenAxesExternalModelV1Error(
            "prepared marker canonical artifact path changed"
        )
    if marker["synthetic_test_only"] is not synthetic:
        raise OpenAxesExternalModelV1Error("prepared synthetic markers disagree")
    _exact_bool(marker["immutable"], name="prepared immutability", expected=True)
    runtime_phase = (
        "post_spawn"
        if (root / MODEL_EXECUTION_SESSION).is_file()
        else "pre_spawn"
    )
    runtime = _validate_prepared_runtime_receipt(
        manifest["runtime"],
        production=not synthetic,
        current_phase=runtime_phase,
    )
    _exact_str(
        marker["prepared_runtime_sha256"],
        name="prepared marker runtime hash",
        expected=_canonical_sha256(runtime),
    )
    repository = _mapping(manifest["repository"], name="repository receipt")
    _keys(
        repository,
        {
            "model_contract_git_head",
            "dirty",
            "porcelain_sha256",
            "required_repo_paths",
            "tracked_required_repo_paths",
            "dirty_required_repo_paths",
            "model_source_blobs_at_head",
            "model_source_blobs_at_head_sha256",
            "model_contract_sources_committed_at_git_head",
            "downstream_results_blocked_by_uncommitted_contract",
        },
        name="repository receipt",
    )
    model_head = _exact_str(
        repository["model_contract_git_head"], name="model contract Git HEAD"
    )
    if re.fullmatch(r"[0-9a-f]{40,64}", model_head) is None:
        raise OpenAxesExternalModelV1Error("model contract Git HEAD is malformed")
    _hex64(repository["porcelain_sha256"], name="model Git porcelain hash")
    _hex64(
        repository["model_source_blobs_at_head_sha256"],
        name="model HEAD source set hash",
    )
    committed = repository.get("model_contract_sources_committed_at_git_head")
    blocked = repository.get("downstream_results_blocked_by_uncommitted_contract")
    if type(committed) is not bool or type(blocked) is not bool or committed is blocked:
        raise OpenAxesExternalModelV1Error("repository result-block receipts disagree")
    if marker["model_contract_sources_committed_at_git_head"] is not committed or marker[
        "downstream_results_blocked_by_uncommitted_contract"
    ] is not blocked:
        raise OpenAxesExternalModelV1Error("prepared marker repository receipt changed")
    if not synthetic and (not committed or blocked):
        raise OpenAxesExternalModelV1Error("uncommitted model contract cannot produce results")
    if not synthetic:
        current_repository = _git_contract_receipt(workspace)
        for key in (
            "model_contract_git_head",
            "required_repo_paths",
            "tracked_required_repo_paths",
            "dirty_required_repo_paths",
            "model_source_blobs_at_head",
            "model_source_blobs_at_head_sha256",
            "model_contract_sources_committed_at_git_head",
            "downstream_results_blocked_by_uncommitted_contract",
        ):
            if _canonical(current_repository[key]) != _canonical(repository[key]):
                raise OpenAxesExternalModelV1Error(
                    f"model contract Git authority changed: {key}"
                )
    preparation = _mapping(manifest["preparation"], name="preparation snapshot")
    synthetic_marker = manifest.get("synthetic_test_only")
    if type(synthetic_marker) is not bool:
        raise OpenAxesExternalModelV1Error(
            "execution session model synthetic marker changed"
        )
    _keys(
        preparation,
        {
            "validation",
            "artifacts",
            "schema",
            "folds",
            "preprocessing",
            "limitations",
            "manifest_semantic_sha256",
            "parent_git_authority",
        },
        name="preparation snapshot",
    )
    _hex64(preparation["manifest_semantic_sha256"], name="preparation semantic hash")
    validation = _mapping(preparation.get("validation"), name="preparation validation snapshot")
    _keys(validation, PREPARATION_VALIDATOR_KEYS, name="preparation validation snapshot")
    artifacts = _mapping(preparation.get("artifacts"), name="preparation artifacts")
    if synthetic:
        if preparation["parent_git_authority"] is not None:
            raise OpenAxesExternalModelV1Error(
                "synthetic preparation carries parent Git authority"
            )
        _keys(
            artifacts,
            {"synthetic_test_only", "manifest_sha256", "frame_sha256"},
            name="synthetic preparation artifacts",
        )
        _exact_bool(artifacts.get("synthetic_test_only"), name="synthetic prep marker", expected=True)
        _hex64(artifacts["manifest_sha256"], name="synthetic manifest hash")
        _hex64(artifacts["frame_sha256"], name="synthetic frame hash")
    else:
        _keys(
            artifacts,
            {"synthetic_test_only", "source", "config", "manifest", "parquet", "complete"},
            name="preparation artifacts",
        )
        _exact_bool(artifacts["synthetic_test_only"], name="production prep marker", expected=False)
        external_paths = {
            key: _verify_external_reference(artifacts[key], name=f"preparation {key}")
            for key in ("source", "config", "manifest", "parquet", "complete")
        }
        external_manifest = _mapping(
            _strict_json_load(external_paths["manifest"]), name="external preparation manifest"
        )
        if _canonical_sha256(external_manifest) != preparation["manifest_semantic_sha256"]:
            raise OpenAxesExternalModelV1Error("external preparation manifest snapshot changed")
        for key in ("schema", "folds", "preprocessing", "limitations"):
            if _canonical(preparation[key]) != _canonical(external_manifest[key]):
                raise OpenAxesExternalModelV1Error(
                    f"preparation {key} snapshot disagrees with authenticated manifest"
                )
        observed_parent_authority = _parent_source_authority_receipt(
            preparation_manifest=external_manifest,
            preparation_config_path=external_paths["config"],
        )
        if _canonical(observed_parent_authority) != _canonical(
            preparation["parent_git_authority"]
        ):
            raise OpenAxesExternalModelV1Error(
                "parent preparation/model Git authority receipt changed"
            )
        public_validation = _run_parent_public_validator_at_recorded_git_state(
            source_path=external_paths["source"],
            preparation_config_path=external_paths["config"],
            preparation_run_dir=external_paths["manifest"].parent,
            preparation_manifest=external_manifest,
        )
        if _canonical(public_validation) != _canonical(validation):
            raise OpenAxesExternalModelV1Error(
                "fresh public preparation validator receipt changed"
            )
    session_path = _execution_session_path(root)
    pre_fit_path = _pre_fit_validation_path(root)
    attempt_dir = root / MODEL_ATTEMPT_LEDGER_DIR
    roster_dir = root / MODEL_ATTEMPT_ROSTER_DIR
    ledger_dir = root / MODEL_LEDGERS_DIR
    result_dir = root / MODEL_RESULTS_DIR
    if pre_fit_path.exists():
        _validate_pre_fit_validation_receipt(
            root, manifest, tasks, runtime_phase=runtime_phase
        )
    if session_path.exists():
        session = _validate_execution_session(
            root,
            manifest,
            tasks,
            _strict_json_load(session_path),
            runtime_phase=runtime_phase,
        )
        _validate_execution_evidence_sets(root, session)
        recovery_path = _prelaunch_session_recovery_path(root)
        if recovery_path.exists():
            _validate_prelaunch_session_recovery_receipt(root, session)
            if not any((root / MODEL_EXECUTION_LAUNCHES_DIR).iterdir()):
                _validate_exact_prelaunch_evidence_empty(root)
    elif (
        any(ledger_dir.iterdir())
        or any(attempt_dir.iterdir())
        or any(roster_dir.iterdir())
        or any(result_dir.iterdir())
    ):
        raise OpenAxesExternalModelV1Error(
            "execution evidence exists without its immutable session"
        )
    allowed_attempt_names = {f"{task.task_id}.json" for task in tasks}
    if any(
        path.is_symlink() or path.name not in allowed_attempt_names
        for path in attempt_dir.iterdir()
    ) or any(
        path.is_symlink() or path.name not in allowed_attempt_names
        for path in roster_dir.iterdir()
    ):
        raise OpenAxesExternalModelV1Error("persistent attempt-ledger member set changed")
    by_task = {task.task_id: task for task in tasks}
    for directory, name in (
        (ledger_dir, "task ledger"),
        (result_dir, "task result"),
    ):
        if any(
            path.is_symlink()
            or not path.is_dir()
            or path.name not in by_task
            for path in directory.iterdir()
        ):
            raise OpenAxesExternalModelV1Error(f"{name} member set changed")
    evidence_task_ids = {
        path.name for path in ledger_dir.iterdir()
    } | {path.name for path in result_dir.iterdir()}
    missing_attempts = sorted(
        task_id
        for task_id in evidence_task_ids
        if not _attempt_ledger_path(root, by_task[task_id]).is_file()
        or not _attempt_roster_path(root, by_task[task_id]).is_file()
    )
    if missing_attempts:
        raise OpenAxesExternalModelV1Error(
            "task phase/result evidence lacks its persistent attempt ledger"
        )
    for path in attempt_dir.iterdir():
        _verify_task_attempt_ledger(root, by_task[path.stem])
    for path in roster_dir.iterdir():
        _verify_task_attempt_roster(root, by_task[path.stem])
    _validate_quarantine_evidence(root, tasks, require_empty=False)
    return _prepared_run_validation_receipt(tasks, synthetic=synthetic)


def _prepared_run_validation_receipt(
    tasks: Sequence[TaskSpec], *, synthetic: bool
) -> dict[str, Any]:
    if type(synthetic) is not bool:
        raise OpenAxesExternalModelV1Error(
            "prepared-run validation synthetic marker is not exact"
        )
    return {
        "status": "ok",
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "config_sha256": OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
        "scientific_fits": len(tasks),
        "primary_fits": sum(task.role == "primary" for task in tasks),
        "stress_fits": sum(task.role == "stress" for task in tasks),
        "synthetic_test_only": synthetic,
        "challenge_route_present": False,
    }


def _execution_session_path(root: Path) -> Path:
    return root / MODEL_EXECUTION_SESSION


def _pre_fit_validation_path(root: Path) -> Path:
    return root / MODEL_PRE_FIT_VALIDATION


def _attempt_ledger_path(root: Path, task: TaskSpec) -> Path:
    return root / MODEL_ATTEMPT_LEDGER_DIR / f"{task.task_id}.json"


def _attempt_roster_path(root: Path, task: TaskSpec) -> Path:
    return root / MODEL_ATTEMPT_ROSTER_DIR / f"{task.task_id}.json"


def _expected_pre_fit_validation_receipt(
    root: Path,
    manifest: Mapping[str, Any],
    tasks: Sequence[TaskSpec],
    *,
    runtime_phase: str,
) -> dict[str, Any]:
    synthetic = manifest.get("synthetic_test_only")
    if type(synthetic) is not bool:
        raise OpenAxesExternalModelV1Error(
            "pre-fit synthetic authority marker changed"
        )
    preparation = _mapping(manifest.get("preparation"), name="pre-fit preparation")
    parent_authority = preparation.get("parent_git_authority")
    repository = _mapping(manifest.get("repository"), name="pre-fit repository")
    source_closure = _mapping(
        manifest.get("source_closure"), name="pre-fit model source closure"
    )
    source_files = _mapping(
        source_closure.get("files"), name="pre-fit model source files"
    )
    task_grid = _mapping(manifest.get("task_grid"), name="pre-fit task grid")
    task_index = _mapping(task_grid.get("task_index"), name="pre-fit task index")
    public_validation = _prepared_run_validation_receipt(
        tasks, synthetic=synthetic
    )
    runtime = _validate_prepared_runtime_receipt(
        manifest.get("runtime"),
        production=not synthetic,
        current_phase=runtime_phase,
    )
    if synthetic:
        if parent_authority is not None:
            raise OpenAxesExternalModelV1Error(
                "synthetic pre-fit receipt carries parent Git authority"
            )
        parent_head = None
        parent_ancestor = None
        parent_proof_hash = None
    else:
        authority = _mapping(parent_authority, name="pre-fit parent Git authority")
        parent_head = _exact_str(
            authority.get("parent_preparation_git_head"),
            name="pre-fit parent preparation Git HEAD",
        )
        parent_ancestor = authority.get("parent_is_ancestor_of_model_head")
        _exact_bool(
            parent_ancestor,
            name="pre-fit parent/model ancestry",
            expected=True,
        )
        parent_proof_hash = _hex64(
            authority.get("receipt_sha256"),
            name="pre-fit parent source-closure proof",
        )
        _exact_str(
            authority.get("model_contract_git_head"),
            name="pre-fit authority model Git HEAD",
            expected=repository["model_contract_git_head"],
        )
    receipt = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "receipt_kind": "immutable_pre_fit_public_authority",
        "config_sha256": OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
        "prepared_manifest": _reference(
            root / MODEL_PREPARED_MANIFEST, relative=MODEL_PREPARED_MANIFEST
        ),
        "prepared_run_validation": public_validation,
        "prepared_run_validation_sha256": _canonical_sha256(public_validation),
        "prepared_runtime_sha256": _canonical_sha256(runtime),
        "parent_public_validation_sha256": _canonical_sha256(
            preparation["validation"]
        ),
        "parent_git_authority_sha256": (
            None
            if parent_authority is None
            else _canonical_sha256(parent_authority)
        ),
        "parent_preparation_git_head": parent_head,
        "model_contract_git_head": repository["model_contract_git_head"],
        "parent_is_ancestor_of_model_head": parent_ancestor,
        "parent_source_closure_proof_sha256": parent_proof_hash,
        "model_source_set_sha256": source_closure["set_sha256"],
        "controller_source_sha256": source_files[
            "src/imu_vive_learning/openaxes_external_model_v1_experiment.py"
        ]["sha256"],
        "task_index": deepcopy(dict(task_index)),
        "task_index_sha256": task_index["sha256"],
        "gpu_inventory": deepcopy(list(manifest["gpu_inventory"])),
        "gpu_inventory_sha256": _canonical_sha256(manifest["gpu_inventory"]),
        "scientific_fits": len(tasks),
        "primary_fits": sum(task.role == "primary" for task in tasks),
        "stress_fits": sum(task.role == "stress" for task in tasks),
        "synthetic_test_only": synthetic,
        "challenge_route_present": False,
    }
    receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
    return receipt


def _validate_pre_fit_validation_receipt(
    root: Path,
    manifest: Mapping[str, Any],
    tasks: Sequence[TaskSpec],
    *,
    runtime_phase: str,
) -> dict[str, Any]:
    path = _pre_fit_validation_path(root)
    if path.is_symlink() or not path.is_file():
        raise OpenAxesExternalModelV1Error(
            "immutable pre-fit validation receipt is missing"
        )
    observed = _mapping(
        _strict_json_load(path), name="immutable pre-fit validation receipt"
    )
    expected = _expected_pre_fit_validation_receipt(
        root, manifest, tasks, runtime_phase=runtime_phase
    )
    if _canonical(observed) != _canonical(expected):
        raise OpenAxesExternalModelV1Error(
            "immutable pre-fit validation receipt changed"
        )
    return deepcopy(dict(observed))


def _freeze_or_load_pre_fit_validation_receipt(
    root: Path,
    manifest: Mapping[str, Any],
    tasks: Sequence[TaskSpec],
    pre_fit_validation: Mapping[str, Any],
) -> dict[str, Any]:
    expected_public = _prepared_run_validation_receipt(
        tasks, synthetic=manifest["synthetic_test_only"]
    )
    if _canonical(pre_fit_validation) != _canonical(expected_public):
        raise OpenAxesExternalModelV1Error(
            "caller pre-fit validation receipt is not public canonical truth"
        )
    path = _pre_fit_validation_path(root)
    if not path.exists():
        _atomic_json(
            path,
            _expected_pre_fit_validation_receipt(
                root, manifest, tasks, runtime_phase="pre_spawn"
            ),
            exclusive=True,
        )
    return _validate_pre_fit_validation_receipt(
        root, manifest, tasks, runtime_phase="pre_spawn"
    )


def _execution_session_semantic_id(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    workers_per_gpu: int,
    slots: Sequence[Mapping[str, Any]],
    pre_fit_receipt: Mapping[str, Any],
) -> str:
    preparation = _mapping(
        manifest["preparation"], name="execution session preparation authority"
    )
    return _canonical_sha256(
        {
            "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
            "config_sha256": OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
            "prepared_manifest": _reference(
                root / MODEL_PREPARED_MANIFEST, relative=MODEL_PREPARED_MANIFEST
            ),
            "model_contract_git_head": manifest["repository"][
                "model_contract_git_head"
            ],
            "parent_git_authority_sha256": (
                None
                if preparation["parent_git_authority"] is None
                else _canonical_sha256(preparation["parent_git_authority"])
            ),
            "parent_public_validation_sha256": _canonical_sha256(
                preparation["validation"]
            ),
            "model_source_set_sha256": manifest["source_closure"]["set_sha256"],
            "prepared_runtime_sha256": _canonical_sha256(manifest["runtime"]),
            "pre_fit_validation_receipt_sha256": _canonical_sha256(
                pre_fit_receipt
            ),
            "task_index_sha256": manifest["task_grid"]["task_index"]["sha256"],
            "gpu_inventory_sha256": _canonical_sha256(manifest["gpu_inventory"]),
            "workers_per_gpu": workers_per_gpu,
            "task_assignment_sha256": _canonical_sha256(slots),
        }
    )


def _validate_execution_session(
    root: Path,
    manifest: Mapping[str, Any],
    tasks: Sequence[TaskSpec],
    value: Any,
    *,
    runtime_phase: str = "post_spawn",
) -> dict[str, Any]:
    session = _mapping(value, name="execution session")
    synthetic_marker = manifest.get("synthetic_test_only")
    if type(synthetic_marker) is not bool:
        raise OpenAxesExternalModelV1Error(
            "execution session model synthetic marker changed"
        )
    _keys(
        session,
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
        },
        name="execution session",
    )
    _exact_str(
        session["contract"],
        name="execution session contract",
        expected=OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
    )
    _hex64(session["execution_session_id"], name="execution session id")
    prepared_manifest_path = _verify_reference(
        root, session["prepared_manifest"], name="execution prepared manifest"
    )
    if prepared_manifest_path != root / MODEL_PREPARED_MANIFEST:
        raise OpenAxesExternalModelV1Error(
            "execution prepared-manifest path changed"
        )
    repository = _mapping(manifest["repository"], name="model repository receipt")
    _exact_str(
        session["model_contract_git_head"],
        name="execution model Git HEAD",
        expected=repository["model_contract_git_head"],
    )
    source_closure = _mapping(manifest["source_closure"], name="model source closure")
    source_files = _mapping(source_closure["files"], name="model source files")
    controller = source_files[
        "src/imu_vive_learning/openaxes_external_model_v1_experiment.py"
    ]["sha256"]
    _exact_str(
        session["controller_source_sha256"],
        name="execution controller hash",
        expected=controller,
    )
    _exact_str(
        session["model_source_set_sha256"],
        name="execution source set hash",
        expected=source_closure["set_sha256"],
    )
    runtime = _validate_prepared_runtime_receipt(
        manifest["runtime"],
        production=not synthetic_marker,
        current_phase=runtime_phase,
    )
    _exact_str(
        session["prepared_runtime_sha256"],
        name="execution prepared runtime hash",
        expected=_canonical_sha256(runtime),
    )
    preparation = _mapping(manifest["preparation"], name="preparation snapshot")
    _exact_str(
        session["parent_public_validation_sha256"],
        name="pre-fit parent public validation hash",
        expected=_canonical_sha256(preparation["validation"]),
    )
    pre_fit = _validate_pre_fit_validation_receipt(
        root, manifest, tasks, runtime_phase=runtime_phase
    )
    pre_fit_path = _verify_reference(
        root,
        session["pre_fit_validation"],
        name="execution pre-fit validation receipt",
    )
    if pre_fit_path != _pre_fit_validation_path(root):
        raise OpenAxesExternalModelV1Error(
            "execution pre-fit validation receipt path changed"
        )
    _exact_str(
        session["pre_fit_validation_receipt_sha256"],
        name="pre-fit prepared-run validation hash",
        expected=_canonical_sha256(pre_fit),
    )
    workers = _exact_int(session["workers_per_gpu"], name="execution workers/GPU")
    frozen_workers = _exact_int(
        _mapping(manifest["execution"], name="execution session authority")[
            "workers_per_gpu_exact"
        ],
        name="config-frozen workers/GPU",
        expected=2,
    )
    if workers != frozen_workers:
        raise OpenAxesExternalModelV1Error(
            "execution workers/GPU differs from config-frozen authority"
        )
    gpus = validate_gpu_inventory(session["gpu_inventory"])
    if _canonical(gpus) != _canonical(manifest["gpu_inventory"]):
        raise OpenAxesExternalModelV1Error("execution GPU inventory changed")
    task_grid = _mapping(manifest["task_grid"], name="model task grid")
    task_index = _mapping(task_grid["task_index"], name="task index reference")
    _exact_str(
        session["task_index_sha256"],
        name="execution task index hash",
        expected=task_index["sha256"],
    )
    expected_slots = assign_tasks_to_gpu_slots(tasks, list(gpus), workers)
    if _canonical(session["slots"]) != _canonical(expected_slots):
        raise OpenAxesExternalModelV1Error("execution slot/GPU/task assignment changed")
    expected_assignment_hash = _canonical_sha256(expected_slots)
    _exact_str(
        session["task_assignment_sha256"],
        name="execution assignment hash",
        expected=expected_assignment_hash,
    )
    _exact_str(
        session["execution_session_id"],
        name="execution session semantic id",
        expected=_execution_session_semantic_id(
            root,
            manifest,
            workers_per_gpu=workers,
            slots=expected_slots,
            pre_fit_receipt=pre_fit,
        ),
    )
    _exact_bool(session["immutable"], name="execution session immutable", expected=True)
    base = {key: session[key] for key in session if key != "session_payload_sha256"}
    if _hex64(session["session_payload_sha256"], name="execution session hash") != _canonical_sha256(base):
        raise OpenAxesExternalModelV1Error("execution session payload hash changed")
    return deepcopy(dict(session))


def _freeze_or_load_execution_session(
    root: Path,
    manifest: Mapping[str, Any],
    tasks: Sequence[TaskSpec],
    *,
    workers_per_gpu: int,
    resume: bool,
    pre_fit_validation: Mapping[str, Any],
) -> dict[str, Any]:
    path = _execution_session_path(root)
    frozen_workers = _exact_int(
        _mapping(manifest["execution"], name="execution session authority")[
            "workers_per_gpu_exact"
        ],
        name="config-frozen workers/GPU",
        expected=2,
    )
    if type(workers_per_gpu) is not int or workers_per_gpu != frozen_workers:
        raise OpenAxesExternalModelV1Error(
            "requested workers/GPU differs from config-frozen authority"
        )
    frozen_pre_fit = _freeze_or_load_pre_fit_validation_receipt(
        root, manifest, tasks, pre_fit_validation
    )
    if path.exists():
        if not resume:
            raise OpenAxesExternalModelV1Error(
                "an immutable execution session already exists; explicit resume is required"
            )
        session = _validate_execution_session(
            root,
            manifest,
            tasks,
            _strict_json_load(path),
            runtime_phase="pre_spawn",
        )
        _exact_int(
            session["workers_per_gpu"],
            name="resume workers/GPU",
            expected=workers_per_gpu,
        )
        _exact_str(
            session["pre_fit_validation_receipt_sha256"],
            name="resume pre-fit validation receipt",
            expected=_canonical_sha256(frozen_pre_fit),
        )
        return session
    if resume:
        raise OpenAxesExternalModelV1Error(
            "resume cannot create a new execution session or remap workers"
        )
    if any((root / MODEL_LEDGERS_DIR).iterdir()) or any(
        (root / MODEL_ATTEMPT_LEDGER_DIR).iterdir()
    ) or any(
        (root / MODEL_ATTEMPT_ROSTER_DIR).iterdir()
    ):
        raise OpenAxesExternalModelV1Error(
            "task evidence exists without an immutable execution session"
        )
    slots = assign_tasks_to_gpu_slots(tasks, manifest["gpu_inventory"], workers_per_gpu)
    source_closure = _mapping(manifest["source_closure"], name="model source closure")
    files = _mapping(source_closure["files"], name="model source files")
    session_id = _execution_session_semantic_id(
        root,
        manifest,
        workers_per_gpu=workers_per_gpu,
        slots=slots,
        pre_fit_receipt=frozen_pre_fit,
    )
    session = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "execution_session_id": session_id,
        "prepared_manifest": _reference(
            root / MODEL_PREPARED_MANIFEST, relative=MODEL_PREPARED_MANIFEST
        ),
        "model_contract_git_head": manifest["repository"]["model_contract_git_head"],
        "controller_source_sha256": files[
            "src/imu_vive_learning/openaxes_external_model_v1_experiment.py"
        ]["sha256"],
        "model_source_set_sha256": source_closure["set_sha256"],
        "prepared_runtime_sha256": _canonical_sha256(manifest["runtime"]),
        "parent_public_validation_sha256": _canonical_sha256(
            manifest["preparation"]["validation"]
        ),
        "pre_fit_validation": _reference(
            _pre_fit_validation_path(root), relative=MODEL_PRE_FIT_VALIDATION
        ),
        "pre_fit_validation_receipt_sha256": _canonical_sha256(frozen_pre_fit),
        "workers_per_gpu": workers_per_gpu,
        "gpu_inventory": deepcopy(list(manifest["gpu_inventory"])),
        "task_index_sha256": manifest["task_grid"]["task_index"]["sha256"],
        "task_assignment_sha256": _canonical_sha256(slots),
        "slots": deepcopy(list(slots)),
        "immutable": True,
    }
    session["session_payload_sha256"] = _canonical_sha256(session)
    _atomic_json(path, session, exclusive=True)
    return _validate_execution_session(
        root, manifest, tasks, session, runtime_phase="pre_spawn"
    )


def _launch_started_path(root: Path, launch_id: str) -> Path:
    if HEX64.fullmatch(launch_id) is None:
        raise OpenAxesExternalModelV1Error("execution launch id is unsafe")
    return (
        root
        / MODEL_EXECUTION_LAUNCHES_DIR
        / f"launch_{launch_id}"
        / "launch.started.json"
    )


def _controller_ready_observations_path(root: Path, launch_id: str) -> Path:
    return _launch_started_path(root, launch_id).with_name(
        "controller_ready_observations.json"
    )


def _execution_launch_semantic_id(
    session: Mapping[str, Any],
    *,
    launch_ordinal: int,
    previous_launch: Mapping[str, Any] | None,
    launch_mode: str | None = None,
) -> str:
    ordinal = _exact_int(launch_ordinal, name="semantic launch ordinal")
    if ordinal < 0:
        raise OpenAxesExternalModelV1Error("semantic launch ordinal is negative")
    if ordinal == 0:
        if previous_launch is not None:
            raise OpenAxesExternalModelV1Error(
                "first semantic launch has a predecessor"
            )
        previous: Mapping[str, Any] | None = None
    else:
        previous = _mapping(
            previous_launch, name="semantic previous launch reference"
        )
    if launch_mode is None:
        mode = "fresh_first_launch" if ordinal == 0 else "resume_after_prior_launch"
    else:
        mode = _exact_str(launch_mode, name="semantic launch mode")
    allowed_modes = (
        {"fresh_first_launch", OPENAXES_PRELAUNCH_RECOVERY_MODE}
        if ordinal == 0
        else {"resume_after_prior_launch"}
    )
    if mode not in allowed_modes:
        raise OpenAxesExternalModelV1Error(
            "semantic launch mode disagrees with its ordinal"
        )
    return _canonical_sha256(
        {
            "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
            "execution_session_id": _hex64(
                session["execution_session_id"],
                name="semantic launch execution session",
            ),
            "launch_ordinal": ordinal,
            "previous_launch": previous,
            "launch_mode": mode,
        }
    )


def _prelaunch_session_recovery_path(root: Path) -> Path:
    return root / MODEL_PRELAUNCH_SESSION_RECOVERY


def _validate_exact_prelaunch_evidence_empty(root: Path) -> None:
    required_root_members = set(OPENAXES_PRELAUNCH_EXACT_ROOT_MEMBERS)
    if {path.name for path in root.iterdir()} != required_root_members or any(
        path.is_symlink() for path in root.iterdir()
    ):
        raise OpenAxesExternalModelV1Error(
            "pre-launch recovery root is not exact prepared state"
        )
    runtime = root / "runtime"
    required_runtime_members = set(
        OPENAXES_PRELAUNCH_EXACT_RUNTIME_MEMBERS_BEFORE_RECEIPT
    )
    observed_runtime_members = {path.name for path in runtime.iterdir()}
    if observed_runtime_members not in (
        required_runtime_members,
        set(OPENAXES_PRELAUNCH_EXACT_RUNTIME_MEMBERS_AFTER_RECEIPT),
    ) or any(path.is_symlink() for path in runtime.iterdir()):
        raise OpenAxesExternalModelV1Error(
            "pre-launch recovery runtime authority set changed"
        )
    for relative in OPENAXES_PRELAUNCH_EXACT_EMPTY_DIRECTORIES:
        directory = root / relative
        if directory.is_symlink() or not directory.is_dir() or any(
            directory.iterdir()
        ):
            raise OpenAxesExternalModelV1Error(
                f"pre-launch recovery observed execution evidence: {relative}"
            )


def _expected_prelaunch_session_recovery_receipt(
    root: Path, session: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = _mapping(
        _strict_json_load(root / MODEL_PREPARED_MANIFEST),
        name="pre-launch recovery prepared manifest",
    )
    prepared_execution = _mapping(
        manifest.get("execution"), name="pre-launch recovery prepared execution"
    )
    recovery_contract = _mapping(
        prepared_execution.get("prelaunch_session_recovery"),
        name="pre-launch recovery frozen contract",
    )
    receipt = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "receipt_kind": "authenticated_session_only_prelaunch_recovery",
        "config_sha256": manifest["config_sha256"],
        "prelaunch_recovery_contract_sha256": _canonical_sha256(
            recovery_contract
        ),
        "execution_session_id": session["execution_session_id"],
        "execution_session": _reference(
            _execution_session_path(root), relative=MODEL_EXECUTION_SESSION
        ),
        "session_payload_sha256": session["session_payload_sha256"],
        "pre_fit_validation": _reference(
            _pre_fit_validation_path(root), relative=MODEL_PRE_FIT_VALIDATION
        ),
        "pre_fit_validation_receipt_sha256": session[
            "pre_fit_validation_receipt_sha256"
        ],
        "task_assignment_sha256": session["task_assignment_sha256"],
        "empty_evidence_at_exclusive_creation": {
            "execution_launch_directories": 0,
            "worker_ready_directories": 0,
            "attempt_rosters": 0,
            "attempt_ledgers": 0,
            "task_phase_directories": 0,
            "result_directories": 0,
            "quarantine_events": 0,
        },
        "local_scientific_attempt_evidence_observed_at_exclusive_creation": False,
        "authorized_action_scope": OPENAXES_PRELAUNCH_AUTHORIZED_ACTION_SCOPE,
        "forbidden_action_scopes": list(
            OPENAXES_PRELAUNCH_FORBIDDEN_ACTION_SCOPES
        ),
        "recovery_receipt_is_attempt_authority": False,
        "authorizes_scientific_refit_under_declared_non_total_erasure_threat_model": False,
        "no_refit_guarantee_threat_model": OPENAXES_NO_REFIT_THREAT_MODEL,
        "external_append_only_or_signature_authority_present": False,
        "total_erasure_prior_execution_provable": False,
        "total_erasure_limitation": OPENAXES_TOTAL_ERASURE_LIMITATION,
    }
    receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
    return receipt


def _validate_prelaunch_session_recovery_receipt(
    root: Path, session: Mapping[str, Any]
) -> dict[str, Any]:
    path = _prelaunch_session_recovery_path(root)
    if path.is_symlink() or not path.is_file():
        raise OpenAxesExternalModelV1Error(
            "pre-launch session recovery receipt is missing"
        )
    observed = _mapping(
        _strict_json_load(path), name="pre-launch session recovery receipt"
    )
    expected = _expected_prelaunch_session_recovery_receipt(root, session)
    if _canonical(observed) != _canonical(expected):
        raise OpenAxesExternalModelV1Error(
            "pre-launch session recovery receipt changed"
        )
    return deepcopy(dict(observed))


def _freeze_or_load_prelaunch_session_recovery_receipt(
    root: Path, session: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_exact_prelaunch_evidence_empty(root)
    persisted_session = _mapping(
        _strict_json_load(_execution_session_path(root)),
        name="persisted pre-launch execution session",
    )
    manifest = _mapping(
        _strict_json_load(root / MODEL_PREPARED_MANIFEST),
        name="pre-launch prepared manifest",
    )
    tasks = _load_task_index(root, manifest)
    authenticated = _validate_execution_session(
        root,
        manifest,
        tasks,
        persisted_session,
        runtime_phase="pre_spawn",
    )
    if _canonical(authenticated) != _canonical(session):
        raise OpenAxesExternalModelV1Error(
            "pre-launch recovery session differs from authenticated authority"
        )
    path = _prelaunch_session_recovery_path(root)
    expected = _expected_prelaunch_session_recovery_receipt(root, authenticated)
    if not path.exists():
        _atomic_json(path, expected, exclusive=True)
    return _validate_prelaunch_session_recovery_receipt(root, authenticated)


def _validate_execution_launch_payload(
    root: Path, session: Mapping[str, Any], value: Any
) -> dict[str, Any]:
    launch = _mapping(value, name="execution launch")
    _keys(
        launch,
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
        },
        name="execution launch",
    )
    _exact_str(
        launch["contract"],
        name="execution launch contract",
        expected=OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
    )
    _exact_str(
        launch["execution_session_id"],
        name="launch execution session",
        expected=session["execution_session_id"],
    )
    _hex64(launch["launch_id"], name="execution launch id")
    ordinal = _exact_int(launch["launch_ordinal"], name="execution launch ordinal")
    if ordinal < 0:
        raise OpenAxesExternalModelV1Error("execution launch ordinal is negative")
    resume = launch["resume"]
    if type(resume) is not bool:
        raise OpenAxesExternalModelV1Error(
            "execution launch resume marker is not exact"
        )
    if ordinal == 0:
        expected_mode = (
            OPENAXES_PRELAUNCH_RECOVERY_MODE
            if resume
            else "fresh_first_launch"
        )
    else:
        _exact_bool(
            resume,
            name="later execution launch resume marker",
            expected=True,
        )
        expected_mode = "resume_after_prior_launch"
    launch_mode = _exact_str(
        launch["launch_mode"],
        name="execution launch mode",
        expected=expected_mode,
    )
    session_path = _verify_reference(
        root, launch["session"], name="execution launch session reference"
    )
    if session_path != _execution_session_path(root):
        raise OpenAxesExternalModelV1Error("execution launch session path changed")
    recovery_path = _prelaunch_session_recovery_path(root)
    if recovery_path.exists():
        recovery = _validate_prelaunch_session_recovery_receipt(root, session)
        expected_recovery_reference: Mapping[str, Any] | None = _reference(
            recovery_path, relative=MODEL_PRELAUNCH_SESSION_RECOVERY
        )
        expected_recovery_sha256: str | None = _canonical_sha256(recovery)
    else:
        expected_recovery_reference = None
        expected_recovery_sha256 = None
    if launch_mode == OPENAXES_PRELAUNCH_RECOVERY_MODE:
        if expected_recovery_reference is None:
            raise OpenAxesExternalModelV1Error(
                "recovered first launch lacks its authenticated pre-launch receipt"
            )
    elif launch_mode == "fresh_first_launch" and expected_recovery_reference is not None:
        raise OpenAxesExternalModelV1Error(
            "fresh first launch conflicts with a pre-launch recovery receipt"
        )
    if _canonical(launch["prelaunch_session_recovery"]) != _canonical(
        expected_recovery_reference
    ) or _canonical(launch["prelaunch_session_recovery_sha256"]) != _canonical(
        expected_recovery_sha256
    ):
        raise OpenAxesExternalModelV1Error(
            "execution launch pre-launch recovery authority changed"
        )
    expected = {
        "workers_per_gpu": session["workers_per_gpu"],
        "slot_count": len(session["slots"]),
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
    for key, expected_value in expected.items():
        if _canonical(launch[key]) != _canonical(expected_value):
            raise OpenAxesExternalModelV1Error(
                f"execution launch differs from immutable session: {key}"
            )
    previous = launch["previous_launch"]
    if ordinal == 0:
        if previous is not None:
            raise OpenAxesExternalModelV1Error("first execution launch has a predecessor")
    else:
        _mapping(previous, name="execution previous launch reference")
    _exact_str(
        launch["launch_id"],
        name="execution launch semantic id",
        expected=_execution_launch_semantic_id(
            session,
            launch_ordinal=ordinal,
            previous_launch=previous,
            launch_mode=launch_mode,
        ),
    )
    base = {key: launch[key] for key in launch if key != "launch_payload_sha256"}
    if _hex64(launch["launch_payload_sha256"], name="execution launch hash") != _canonical_sha256(base):
        raise OpenAxesExternalModelV1Error("execution launch payload hash changed")
    return deepcopy(dict(launch))


def _validate_execution_launch(
    root: Path, session: Mapping[str, Any], launch_id: str
) -> dict[str, Any]:
    requested_path = _launch_started_path(root, launch_id)
    launches_root = root / MODEL_EXECUTION_LAUNCHES_DIR
    if launches_root.is_symlink() or not launches_root.is_dir():
        raise OpenAxesExternalModelV1Error("execution launch root changed")
    entries: list[tuple[dict[str, Any], Path]] = []
    for directory in launches_root.iterdir():
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or not directory.name.startswith("launch_")
            or HEX64.fullmatch(directory.name.removeprefix("launch_")) is None
        ):
            raise OpenAxesExternalModelV1Error("execution launch directory set changed")
        members = {path.name for path in directory.iterdir()}
        if members not in (
            {"launch.started.json"},
            {"launch.started.json", "controller_ready_observations.json"},
            {
                "launch.started.json",
                "controller_ready_observations.json",
                "launch.completed.json",
            },
        ) or any(path.is_symlink() for path in directory.iterdir()):
            raise OpenAxesExternalModelV1Error(
                "execution launch member set changed"
            )
        path = directory / "launch.started.json"
        if path.is_symlink() or not path.is_file():
            raise OpenAxesExternalModelV1Error("execution launch start receipt is missing")
        payload = _validate_execution_launch_payload(
            root, session, _strict_json_load(path)
        )
        if payload["launch_id"] != directory.name.removeprefix("launch_"):
            raise OpenAxesExternalModelV1Error("execution launch directory/id changed")
        entries.append((payload, path))
    entries.sort(key=lambda item: item[0]["launch_ordinal"])
    if [item[0]["launch_ordinal"] for item in entries] != list(range(len(entries))):
        raise OpenAxesExternalModelV1Error("execution launch ordinals are not contiguous")
    for index, (payload, _path) in enumerate(entries):
        if index == 0:
            continue
        previous_path = entries[index - 1][1]
        expected_previous = _reference(
            previous_path, relative=previous_path.relative_to(root).as_posix()
        )
        if payload["previous_launch"] != expected_previous:
            raise OpenAxesExternalModelV1Error("execution launch predecessor changed")
    matches = [payload for payload, path in entries if path == requested_path]
    if len(matches) != 1:
        raise OpenAxesExternalModelV1Error("requested execution launch is not unique")
    return matches[0]


def _start_execution_launch(
    root: Path, session: Mapping[str, Any], *, resume: bool
) -> dict[str, Any]:
    if type(resume) is not bool:
        raise OpenAxesExternalModelV1Error("execution launch resume marker is not exact")
    launches_root = root / MODEL_EXECUTION_LAUNCHES_DIR
    if launches_root.is_symlink() or not launches_root.is_dir():
        raise OpenAxesExternalModelV1Error("execution launch root changed")
    existing_directories = list(launches_root.iterdir())
    existing: list[dict[str, Any]] = []
    if existing_directories:
        first_id = existing_directories[0].name.removeprefix("launch_")
        _validate_execution_launch(root, session, first_id)
        for directory in existing_directories:
            launch_id = directory.name.removeprefix("launch_")
            existing.append(_validate_execution_launch(root, session, launch_id))
    existing.sort(key=lambda item: item["launch_ordinal"])
    ordinal = len(existing)
    recovery: Mapping[str, Any] | None
    if ordinal == 0 and resume:
        launch_mode = OPENAXES_PRELAUNCH_RECOVERY_MODE
        recovery = _freeze_or_load_prelaunch_session_recovery_receipt(
            root, session
        )
    elif ordinal == 0:
        launch_mode = "fresh_first_launch"
        _validate_exact_prelaunch_evidence_empty(root)
        if _prelaunch_session_recovery_path(root).exists():
            raise OpenAxesExternalModelV1Error(
                "fresh first launch conflicts with a pre-launch recovery receipt"
            )
        recovery = None
    elif resume:
        launch_mode = "resume_after_prior_launch"
        recovery = (
            _validate_prelaunch_session_recovery_receipt(root, session)
            if _prelaunch_session_recovery_path(root).exists()
            else None
        )
    else:
        raise OpenAxesExternalModelV1Error(
            "execution launch resume marker disagrees with immutable launch lineage"
        )
    previous_path = (
        None
        if not existing
        else _launch_started_path(root, existing[-1]["launch_id"])
    )
    previous_reference = (
        None
        if previous_path is None
        else _reference(
            previous_path, relative=previous_path.relative_to(root).as_posix()
        )
    )
    launch_id = _execution_launch_semantic_id(
        session,
        launch_ordinal=ordinal,
        previous_launch=previous_reference,
        launch_mode=launch_mode,
    )
    launch_dir = root / MODEL_EXECUTION_LAUNCHES_DIR / f"launch_{launch_id}"
    launch_dir.mkdir(exist_ok=False)
    started = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "execution_session_id": session["execution_session_id"],
        "launch_id": launch_id,
        "launch_ordinal": ordinal,
        "launch_mode": launch_mode,
        "previous_launch": previous_reference,
        "prelaunch_session_recovery": (
            _reference(
                _prelaunch_session_recovery_path(root),
                relative=MODEL_PRELAUNCH_SESSION_RECOVERY,
            )
            if recovery is not None
            else None
        ),
        "prelaunch_session_recovery_sha256": (
            _canonical_sha256(recovery) if recovery is not None else None
        ),
        "resume": resume,
        "session": _reference(
            _execution_session_path(root), relative=MODEL_EXECUTION_SESSION
        ),
        "workers_per_gpu": session["workers_per_gpu"],
        "slot_count": len(session["slots"]),
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
    started["launch_payload_sha256"] = _canonical_sha256(started)
    _atomic_json(launch_dir / "launch.started.json", started, exclusive=True)
    return _validate_execution_launch(root, session, launch_id)


def _worker_ready_path(root: Path, launch_id: str, slot_index: int) -> Path:
    if HEX64.fullmatch(launch_id) is None or type(slot_index) is not int or slot_index < 0:
        raise OpenAxesExternalModelV1Error("worker-ready identity is unsafe")
    return (
        root
        / MODEL_WORKER_READY_DIR
        / f"launch_{launch_id}"
        / f"slot_{slot_index:04d}.json"
    )


def _write_worker_ready_receipt(
    root: Path,
    session: Mapping[str, Any],
    launch: Mapping[str, Any],
    slot: Mapping[str, Any],
) -> dict[str, Any]:
    path = _worker_ready_path(root, launch["launch_id"], slot["slot_index"])
    receipt = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "execution_session_id": session["execution_session_id"],
        "launch_id": launch["launch_id"],
        "launch_started": _reference(
            _launch_started_path(root, launch["launch_id"]),
            relative=_launch_started_path(root, launch["launch_id"])
            .relative_to(root)
            .as_posix(),
        ),
        "launch_payload_sha256": launch["launch_payload_sha256"],
        "slot_index": slot["slot_index"],
        "local_slot": slot["local_slot"],
        "gpu": deepcopy(dict(slot["gpu"])),
        "visible_child_device": "cuda:0",
        "cuda_visible_devices": slot["gpu"]["uuid"],
        "controller_source_sha256": session["controller_source_sha256"],
        "task_assignment_sha256": session["task_assignment_sha256"],
        "slot_assignment_sha256": slot["assignment_sha256"],
        "task_ids_sha256": _canonical_sha256(slot["task_ids"]),
        "child_process_id": os.getpid(),
        "ready_after_exact_gpu_validation": True,
    }
    receipt["ready_receipt_sha256"] = _canonical_sha256(receipt)
    _atomic_json(path, receipt, exclusive=True)
    return receipt


def _verify_worker_ready_receipt(
    root: Path,
    session: Mapping[str, Any],
    launch: Mapping[str, Any],
    slot: Mapping[str, Any],
) -> dict[str, Any]:
    path = _worker_ready_path(root, launch["launch_id"], slot["slot_index"])
    receipt = _mapping(_strict_json_load(path), name="worker ready receipt")
    expected_keys = {
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
    _keys(receipt, expected_keys, name="worker ready receipt")
    expected = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "execution_session_id": session["execution_session_id"],
        "launch_id": launch["launch_id"],
        "launch_started": _reference(
            _launch_started_path(root, launch["launch_id"]),
            relative=_launch_started_path(root, launch["launch_id"])
            .relative_to(root)
            .as_posix(),
        ),
        "launch_payload_sha256": launch["launch_payload_sha256"],
        "slot_index": slot["slot_index"],
        "local_slot": slot["local_slot"],
        "gpu": slot["gpu"],
        "visible_child_device": "cuda:0",
        "cuda_visible_devices": slot["gpu"]["uuid"],
        "controller_source_sha256": session["controller_source_sha256"],
        "task_assignment_sha256": session["task_assignment_sha256"],
        "slot_assignment_sha256": slot["assignment_sha256"],
        "task_ids_sha256": _canonical_sha256(slot["task_ids"]),
        "ready_after_exact_gpu_validation": True,
    }
    for key, expected_value in expected.items():
        if _canonical(receipt[key]) != _canonical(expected_value):
            raise OpenAxesExternalModelV1Error(f"worker ready receipt changed: {key}")
    process_id = _exact_int(receipt["child_process_id"], name="child process id")
    if process_id <= 0:
        raise OpenAxesExternalModelV1Error("child process id is invalid")
    base = {key: receipt[key] for key in receipt if key != "ready_receipt_sha256"}
    if _hex64(receipt["ready_receipt_sha256"], name="worker ready hash") != _canonical_sha256(base):
        raise OpenAxesExternalModelV1Error("worker ready receipt hash changed")
    return deepcopy(dict(receipt))


def _write_controller_ready_observations(
    root: Path,
    session: Mapping[str, Any],
    launch: Mapping[str, Any],
    slots: Sequence[Mapping[str, Any]],
    processes: Sequence[Any],
) -> dict[str, Any]:
    if len(slots) != len(processes):
        raise OpenAxesExternalModelV1Error(
            "controller process/slot cardinality changed"
        )
    observations: list[dict[str, Any]] = []
    observed_pids: set[int] = set()
    for slot, process in zip(slots, processes, strict=True):
        ready = _verify_worker_ready_receipt(root, session, launch, slot)
        process_id = _exact_int(process.pid, name="controller child process id")
        if process_id <= 0 or process_id in observed_pids:
            raise OpenAxesExternalModelV1Error(
                "controller child process identities are not unique"
            )
        observed_pids.add(process_id)
        _exact_int(
            ready["child_process_id"],
            name="ready/controller process id",
            expected=process_id,
        )
        expected_name = f"openaxes-model-slot-{slot['slot_index']}"
        _exact_str(
            process.name,
            name="controller child process name",
            expected=expected_name,
        )
        ready_path = _worker_ready_path(
            root, launch["launch_id"], slot["slot_index"]
        )
        observations.append(
            {
                "slot_index": slot["slot_index"],
                "child_process_id": process_id,
                "child_process_name": expected_name,
                "gpu": deepcopy(dict(slot["gpu"])),
                "slot_assignment_sha256": slot["assignment_sha256"],
                "ready_receipt": _reference(
                    ready_path, relative=ready_path.relative_to(root).as_posix()
                ),
                "ready_receipt_sha256": ready["ready_receipt_sha256"],
            }
        )
    receipt = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "execution_session_id": session["execution_session_id"],
        "launch_id": launch["launch_id"],
        "launch_started": _reference(
            _launch_started_path(root, launch["launch_id"]),
            relative=_launch_started_path(root, launch["launch_id"])
            .relative_to(root)
            .as_posix(),
        ),
        "launch_payload_sha256": launch["launch_payload_sha256"],
        "controller_source_sha256": session["controller_source_sha256"],
        "task_assignment_sha256": session["task_assignment_sha256"],
        "slot_count": len(slots),
        "observations": observations,
        "controller_observed_spawn_process_identity": True,
    }
    receipt["controller_observations_sha256"] = _canonical_sha256(receipt)
    _atomic_json(
        _controller_ready_observations_path(root, launch["launch_id"]),
        receipt,
        exclusive=True,
    )
    return receipt


def _verify_controller_ready_observations(
    root: Path,
    session: Mapping[str, Any],
    launch: Mapping[str, Any],
) -> dict[str, Any]:
    path = _controller_ready_observations_path(root, launch["launch_id"])
    if path.is_symlink() or not path.is_file():
        raise OpenAxesExternalModelV1Error(
            "durable controller ready observations are missing"
        )
    receipt = _mapping(
        _strict_json_load(path), name="controller ready observations"
    )
    _keys(
        receipt,
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
        },
        name="controller ready observations",
    )
    expected_header = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "execution_session_id": session["execution_session_id"],
        "launch_id": launch["launch_id"],
        "launch_started": _reference(
            _launch_started_path(root, launch["launch_id"]),
            relative=_launch_started_path(root, launch["launch_id"])
            .relative_to(root)
            .as_posix(),
        ),
        "launch_payload_sha256": launch["launch_payload_sha256"],
        "controller_source_sha256": session["controller_source_sha256"],
        "task_assignment_sha256": session["task_assignment_sha256"],
        "slot_count": len(session["slots"]),
        "controller_observed_spawn_process_identity": True,
    }
    for key, expected_value in expected_header.items():
        if _canonical(receipt[key]) != _canonical(expected_value):
            raise OpenAxesExternalModelV1Error(
                f"controller ready observation changed: {key}"
            )
    observations = receipt["observations"]
    if type(observations) is not list or len(observations) != len(session["slots"]):
        raise OpenAxesExternalModelV1Error(
            "controller ready observation cardinality changed"
        )
    process_ids: set[int] = set()
    for slot, raw in zip(session["slots"], observations, strict=True):
        observation = _mapping(raw, name="controller slot observation")
        _keys(
            observation,
            {
                "slot_index",
                "child_process_id",
                "child_process_name",
                "gpu",
                "slot_assignment_sha256",
                "ready_receipt",
                "ready_receipt_sha256",
            },
            name="controller slot observation",
        )
        ready = _verify_worker_ready_receipt(root, session, launch, slot)
        process_id = _exact_int(
            observation["child_process_id"], name="observed child process id"
        )
        if process_id <= 0 or process_id in process_ids:
            raise OpenAxesExternalModelV1Error(
                "controller-observed child process identities are not unique"
            )
        process_ids.add(process_id)
        ready_path = _verify_reference(
            root,
            observation["ready_receipt"],
            name="controller-observed ready receipt",
        )
        expected = {
            "slot_index": slot["slot_index"],
            "child_process_id": ready["child_process_id"],
            "child_process_name": f"openaxes-model-slot-{slot['slot_index']}",
            "gpu": slot["gpu"],
            "slot_assignment_sha256": slot["assignment_sha256"],
            "ready_receipt_sha256": ready["ready_receipt_sha256"],
        }
        for key, expected_value in expected.items():
            if _canonical(observation[key]) != _canonical(expected_value):
                raise OpenAxesExternalModelV1Error(
                    f"controller slot observation changed: {key}"
                )
        if ready_path != _worker_ready_path(
            root, launch["launch_id"], slot["slot_index"]
        ):
            raise OpenAxesExternalModelV1Error(
                "controller-observed ready path changed"
            )
    base = {
        key: receipt[key]
        for key in receipt
        if key != "controller_observations_sha256"
    }
    if _hex64(
        receipt["controller_observations_sha256"],
        name="controller observation hash",
    ) != _canonical_sha256(base):
        raise OpenAxesExternalModelV1Error(
            "controller ready observation hash changed"
        )
    return deepcopy(dict(receipt))


def _validate_execution_launch_completed(
    root: Path,
    session: Mapping[str, Any],
    launch: Mapping[str, Any],
) -> dict[str, Any]:
    path = (
        root
        / MODEL_EXECUTION_LAUNCHES_DIR
        / f"launch_{launch['launch_id']}"
        / "launch.completed.json"
    )
    value = _mapping(_strict_json_load(path), name="execution launch completion")
    _keys(
        value,
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
        },
        name="execution launch completion",
    )
    controller_observations = _verify_controller_ready_observations(
        root, session, launch
    )
    controller_path = _controller_ready_observations_path(
        root, launch["launch_id"]
    )
    ready_references = [
        _reference(
            _worker_ready_path(root, launch["launch_id"], slot["slot_index"]),
            relative=_worker_ready_path(
                root, launch["launch_id"], slot["slot_index"]
            )
            .relative_to(root)
            .as_posix(),
        )
        for slot in session["slots"]
    ]
    expected = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "execution_session_id": session["execution_session_id"],
        "launch_id": launch["launch_id"],
        "launch_started": _reference(
            _launch_started_path(root, launch["launch_id"]),
            relative=_launch_started_path(root, launch["launch_id"])
            .relative_to(root)
            .as_posix(),
        ),
        "launch_payload_sha256": launch["launch_payload_sha256"],
        "launch_ordinal": launch["launch_ordinal"],
        "resume": launch["resume"],
        "slot_count": len(session["slots"]),
        "task_count": 336,
        "task_assignment_sha256": session["task_assignment_sha256"],
        "controller_source_sha256": session["controller_source_sha256"],
        "model_source_set_sha256": session["model_source_set_sha256"],
        "parent_public_validation_sha256": session[
            "parent_public_validation_sha256"
        ],
        "pre_fit_validation_receipt_sha256": session[
            "pre_fit_validation_receipt_sha256"
        ],
        "controller_ready_observations": _reference(
            controller_path, relative=controller_path.relative_to(root).as_posix()
        ),
        "controller_observations_sha256": controller_observations[
            "controller_observations_sha256"
        ],
        "ready_receipts": ready_references,
        "all_tasks_authenticated_complete": True,
    }
    for key, expected_value in expected.items():
        if _canonical(value[key]) != _canonical(expected_value):
            raise OpenAxesExternalModelV1Error(
                f"execution launch completion changed: {key}"
            )
    base = {
        key: value[key]
        for key in value
        if key != "launch_completed_sha256"
    }
    if _hex64(
        value["launch_completed_sha256"], name="execution launch completion hash"
    ) != _canonical_sha256(base):
        raise OpenAxesExternalModelV1Error(
            "execution launch completion hash changed"
        )
    for slot in session["slots"]:
        _verify_worker_ready_receipt(root, session, launch, slot)
    return deepcopy(dict(value))


def _validate_execution_evidence_sets(
    root: Path,
    session: Mapping[str, Any],
) -> None:
    launches_root = root / MODEL_EXECUTION_LAUNCHES_DIR
    ready_root = root / MODEL_WORKER_READY_DIR
    if ready_root.is_symlink() or not ready_root.is_dir():
        raise OpenAxesExternalModelV1Error("worker-ready root changed")
    launches: dict[str, dict[str, Any]] = {}
    for directory in launches_root.iterdir():
        launch_id = directory.name.removeprefix("launch_")
        launch = _validate_execution_launch(root, session, launch_id)
        launches[launch_id] = launch
    observed_ready_launches: set[str] = set()
    for directory in ready_root.iterdir():
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or not directory.name.startswith("launch_")
        ):
            raise OpenAxesExternalModelV1Error(
                "worker-ready launch directory set changed"
            )
        launch_id = directory.name.removeprefix("launch_")
        if launch_id not in launches:
            raise OpenAxesExternalModelV1Error(
                "worker-ready receipt has no authenticated launch"
            )
        observed_ready_launches.add(launch_id)
        members = list(directory.iterdir())
        if not members:
            raise OpenAxesExternalModelV1Error(
                "empty worker-ready launch directory is not evidence"
            )
        observed_slots: set[int] = set()
        observed_process_ids: set[int] = set()
        for path in members:
            match = re.fullmatch(r"slot_([0-9]{4})\.json", path.name)
            if path.is_symlink() or not path.is_file() or match is None:
                raise OpenAxesExternalModelV1Error(
                    "worker-ready receipt member set changed"
                )
            slot_index = int(match.group(1))
            if (
                slot_index in observed_slots
                or not 0 <= slot_index < len(session["slots"])
            ):
                raise OpenAxesExternalModelV1Error(
                    "worker-ready slot identity changed"
                )
            observed_slots.add(slot_index)
            ready = _verify_worker_ready_receipt(
                root, session, launches[launch_id], session["slots"][slot_index]
            )
            process_id = ready["child_process_id"]
            if process_id in observed_process_ids:
                raise OpenAxesExternalModelV1Error(
                    "worker-ready child process identities are not unique"
                )
            observed_process_ids.add(process_id)
    for launch_id, launch in launches.items():
        observations_path = _controller_ready_observations_path(root, launch_id)
        if observations_path.exists():
            if launch_id not in observed_ready_launches:
                raise OpenAxesExternalModelV1Error(
                    "controller observations lack their exact worker-ready directory"
                )
            _verify_controller_ready_observations(root, session, launch)
        completed_path = (
            launches_root / f"launch_{launch_id}" / "launch.completed.json"
        )
        if completed_path.exists():
            if launch_id not in observed_ready_launches:
                raise OpenAxesExternalModelV1Error(
                    "completed launch lacks worker-ready receipts"
                )
            _validate_execution_launch_completed(root, session, launch)


def _validate_latest_execution_launch_terminal(
    root: Path, session: Mapping[str, Any]
) -> dict[str, Any]:
    launches_root = root / MODEL_EXECUTION_LAUNCHES_DIR
    if launches_root.is_symlink() or not launches_root.is_dir():
        raise OpenAxesExternalModelV1Error("execution launch root changed")
    launches = [
        _validate_execution_launch(
            root, session, directory.name.removeprefix("launch_")
        )
        for directory in launches_root.iterdir()
    ]
    if not launches:
        raise OpenAxesExternalModelV1Error(
            "COMPLETE has no authenticated completed execution launch"
        )
    latest = max(launches, key=lambda item: item["launch_ordinal"])
    latest_completed = (
        launches_root
        / f"launch_{latest['launch_id']}"
        / "launch.completed.json"
    )
    if latest_completed.is_symlink() or not latest_completed.is_file():
        raise OpenAxesExternalModelV1Error(
            "COMPLETE latest execution launch is not terminal-complete"
        )
    _validate_execution_launch_completed(root, session, latest)
    return latest


def _execution_binding(
    root: Path,
    session: Mapping[str, Any],
    launch: Mapping[str, Any],
    slot: Mapping[str, Any],
) -> dict[str, Any]:
    ready_path = _worker_ready_path(root, launch["launch_id"], slot["slot_index"])
    observations_path = _controller_ready_observations_path(
        root, launch["launch_id"]
    )
    if observations_path.is_file() and not observations_path.is_symlink():
        observations = _verify_controller_ready_observations(root, session, launch)
        observations_reference: Mapping[str, Any] | None = _reference(
            observations_path,
            relative=observations_path.relative_to(root).as_posix(),
        )
        observations_sha256: str | None = observations[
            "controller_observations_sha256"
        ]
    else:
        observations_reference = None
        observations_sha256 = None
    return {
        "execution_session_id": session["execution_session_id"],
        "launch_id": launch["launch_id"],
        "launch_started": _reference(
            _launch_started_path(root, launch["launch_id"]),
            relative=_launch_started_path(root, launch["launch_id"])
            .relative_to(root)
            .as_posix(),
        ),
        "launch_payload_sha256": launch["launch_payload_sha256"],
        "transport_slot_index": slot["slot_index"],
        "local_slot": slot["local_slot"],
        "gpu": deepcopy(dict(slot["gpu"])),
        "controller_source_sha256": session["controller_source_sha256"],
        "task_assignment_sha256": session["task_assignment_sha256"],
        "slot_assignment_sha256": slot["assignment_sha256"],
        "worker_ready": _reference(
            ready_path, relative=ready_path.relative_to(root).as_posix()
        ),
        "controller_ready_observations": observations_reference,
        "controller_observations_sha256": observations_sha256,
        "worker_identity_source": "fixed_process_transport_and_ready_receipt",
        "scientific_fit_attempt": 1,
    }


def _task_attempt_semantic_id(
    task: TaskSpec, binding: Mapping[str, Any]
) -> str:
    return _canonical_sha256(
        {
            "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
            "task_id": task.task_id,
            "task_spec_sha256": task.spec_sha256,
            "first_execution_binding": deepcopy(dict(binding)),
            "scientific_fit_attempt": 1,
        }
    )


def _write_task_attempt_roster(
    root: Path,
    task: TaskSpec,
    *,
    attempt_id: str,
    origin: str,
    binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if origin == "production_execution":
        if binding is None:
            raise OpenAxesExternalModelV1Error(
                "production attempt roster lacks an execution binding"
            )
        _exact_str(
            attempt_id,
            name="production attempt semantic id",
            expected=_task_attempt_semantic_id(task, binding),
        )
    payload = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "task_id": task.task_id,
        "task_spec_sha256": task.spec_sha256,
        "attempt_id": _hex64(attempt_id, name="attempt roster id"),
        "origin": origin,
        "binding": None if binding is None else deepcopy(dict(binding)),
        "attempt_started_or_state_observed": True,
        "monotonic_first_attempt_roster": True,
        "survives_ledger_deletion_and_quarantine": True,
    }
    payload["attempt_roster_sha256"] = _canonical_sha256(payload)
    _atomic_json(_attempt_roster_path(root, task), payload, exclusive=True)
    return payload


def _write_task_attempt_ledger(
    root: Path,
    task: TaskSpec,
    *,
    attempt_id: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    _write_task_attempt_roster(
        root,
        task,
        attempt_id=attempt_id,
        origin="production_execution",
        binding=binding,
    )
    payload = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "task_id": task.task_id,
        "task_spec_sha256": task.spec_sha256,
        "attempt_id": _hex64(attempt_id, name="attempt ledger id"),
        "origin": "production_execution",
        "binding": deepcopy(dict(binding)),
        "attempt_started_or_state_observed": True,
        "survives_quarantine": True,
    }
    payload["attempt_ledger_sha256"] = _canonical_sha256(payload)
    _atomic_json(_attempt_ledger_path(root, task), payload, exclusive=True)
    return payload


def _quarantine_tree_receipt(directory: Path, *, relative: str) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise OpenAxesExternalModelV1Error(
            f"quarantine moved {relative} is not a regular directory"
        )
    entries: list[dict[str, Any]] = []
    pending: list[tuple[Path, str]] = [(directory, "")]
    while pending:
        parent, prefix = pending.pop()
        for member in sorted(parent.iterdir(), key=lambda path: path.name, reverse=True):
            member_relative = f"{prefix}/{member.name}" if prefix else member.name
            if member.is_symlink():
                raise OpenAxesExternalModelV1Error(
                    "quarantine moved tree contains a symlink"
                )
            if member.is_dir():
                entries.append({"path": member_relative, "kind": "directory"})
                pending.append((member, member_relative))
            elif member.is_file():
                entries.append(
                    {
                        "path": member_relative,
                        "kind": "file",
                        "bytes": member.stat(follow_symlinks=False).st_size,
                        "sha256": sha256_file(member),
                    }
                )
            else:
                raise OpenAxesExternalModelV1Error(
                    "quarantine moved tree contains a special file"
                )
    entries.sort(key=lambda entry: entry["path"])
    return {
        "path": relative,
        "entries": entries,
        "tree_sha256": _canonical_sha256(entries),
    }


def _validate_quarantine_moved_directory(
    event_dir: Path,
    *,
    name: str,
    task: TaskSpec,
    attempt_id: str,
) -> dict[str, Any]:
    directory = event_dir / name
    receipt = _quarantine_tree_receipt(directory, relative=name)
    members = list(directory.iterdir())
    if name == "ledger":
        allowed = {f"{phase}.json" for phase in OPENAXES_MODEL_PHASES}
    elif name == "result":
        allowed = set(MODEL_COMPLETED_TASK_RESULT_MEMBERS)
    else:
        raise OpenAxesExternalModelV1Error("unknown quarantine moved directory")
    if any(
        member.name not in allowed
        or member.is_symlink()
        or not member.is_file()
        for member in members
    ):
        raise OpenAxesExternalModelV1Error(
            f"quarantine moved {name} member set changed"
        )
    json_members = members if name == "ledger" else [
        member for member in members if member.suffix == ".json"
    ]
    for member in json_members:
        try:
            raw = _strict_json_load(member)
        except OpenAxesExternalModelV1Error:
            # A quarantine event may preserve an undecodable corrupt prefix.
            continue
        payload = _mapping(raw, name=f"quarantine moved {name} payload")
        if name == "ledger":
            for key, expected in (
                ("task_id", task.task_id),
                ("task_spec_sha256", task.spec_sha256),
                ("attempt_id", attempt_id),
            ):
                _exact_str(
                    payload.get(key),
                    name=f"quarantine moved ledger {key}",
                    expected=expected,
                )
        else:
            for key, expected in (
                ("task_id", task.task_id),
                ("task_spec_sha256", task.spec_sha256),
                ("attempt_id", attempt_id),
            ):
                if key in payload:
                    _exact_str(
                        payload[key],
                        name=f"quarantine moved result {key}",
                        expected=expected,
                    )
    return receipt


def _validate_quarantine_evidence(
    root: Path,
    tasks: Sequence[TaskSpec],
    *,
    require_empty: bool,
) -> set[str]:
    if type(require_empty) is not bool:
        raise OpenAxesExternalModelV1Error(
            "quarantine validation mode is not exact"
        )
    quarantine = root / MODEL_QUARANTINE_DIR
    if quarantine.is_symlink() or not quarantine.is_dir():
        raise OpenAxesExternalModelV1Error("quarantine root changed")
    members = list(quarantine.iterdir())
    if require_empty and members:
        raise OpenAxesExternalModelV1Error("COMPLETE requires empty quarantine")
    by_task = {task.task_id: task for task in tasks}
    observed_tasks: set[str] = set()
    for event_dir in members:
        match = re.fullmatch(r"event_([0-9a-f]{64})", event_dir.name)
        if event_dir.is_symlink() or not event_dir.is_dir() or match is None:
            raise OpenAxesExternalModelV1Error("quarantine member set changed")
        event_id = match.group(1)
        event_members = {path.name: path for path in event_dir.iterdir()}
        allowed_event_members = {"intent.json", "audit.json", "ledger", "result"}
        if (
            "intent.json" not in event_members
            or not set(event_members).issubset(allowed_event_members)
            or any(path.is_symlink() for path in event_members.values())
            or any(
                (name in {"intent.json", "audit.json"} and not path.is_file())
                or (name in {"ledger", "result"} and not path.is_dir())
                for name, path in event_members.items()
            )
        ):
            raise OpenAxesExternalModelV1Error(
                "quarantine event member set changed"
            )
        intent_path = event_members["intent.json"]
        intent = _mapping(
            _strict_json_load(intent_path), name="quarantine intent"
        )
        _keys(
            intent,
            {
                "contract",
                "event_id",
                "task_id",
                "task_spec_sha256",
                "attempt_id",
                "reason",
                "sources",
                "nondelete",
            },
            name="quarantine intent",
        )
        _exact_str(
            intent["contract"],
            name="quarantine intent contract",
            expected=OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        )
        _exact_str(
            intent["event_id"],
            name="quarantine intent event",
            expected=event_id,
        )
        task_id = _exact_str(intent["task_id"], name="quarantine task")
        if task_id not in by_task:
            raise OpenAxesExternalModelV1Error(
                "quarantine event has no canonical task"
            )
        task = by_task[task_id]
        observed_tasks.add(task_id)
        _exact_str(
            intent["task_spec_sha256"],
            name="quarantine task hash",
            expected=task.spec_sha256,
        )
        attempt_id = _hex64(intent["attempt_id"], name="quarantine attempt")
        _exact_str(intent["reason"], name="quarantine reason")
        _exact_bool(intent["nondelete"], name="quarantine nondelete", expected=True)
        sources = _mapping(intent["sources"], name="quarantine sources")
        _keys(
            sources,
            {
                "ledger_exists",
                "result_exists",
                "execution_started_file_observed",
                "monotonic_attempt_roster",
                "persistent_attempt_ledger",
            },
            name="quarantine sources",
        )
        ledger_existed = _exact_bool(
            sources["ledger_exists"],
            name="quarantine source ledger",
            expected=sources["ledger_exists"],
        )
        result_existed = _exact_bool(
            sources["result_exists"],
            name="quarantine source result",
            expected=sources["result_exists"],
        )
        execution_started_observed = _exact_bool(
            sources["execution_started_file_observed"],
            name="quarantine source execution-start",
            expected=sources["execution_started_file_observed"],
        )
        if not ledger_existed and not result_existed:
            raise OpenAxesExternalModelV1Error(
                "quarantine intent names no source state"
            )
        roster_path = _verify_reference(
            root,
            sources["monotonic_attempt_roster"],
            name="quarantine attempt-roster reference",
        )
        attempt_path = _verify_reference(
            root,
            sources["persistent_attempt_ledger"],
            name="quarantine attempt-ledger reference",
        )
        if (
            roster_path != _attempt_roster_path(root, task)
            or attempt_path != _attempt_ledger_path(root, task)
        ):
            raise OpenAxesExternalModelV1Error(
                "quarantine persistent authority path changed"
            )
        roster = _verify_task_attempt_roster(root, task)
        attempt = _verify_task_attempt_ledger(root, task)
        if (
            roster is None
            or attempt is None
            or roster["attempt_id"] != attempt_id
            or attempt["attempt_id"] != attempt_id
        ):
            raise OpenAxesExternalModelV1Error(
                "quarantine task/attempt lineage changed"
            )
        moved_receipts: dict[str, dict[str, Any]] = {}
        source_directories = {
            "ledger": root / MODEL_LEDGERS_DIR / task.task_id,
            "result": _task_result_dir(root, task),
        }
        source_flags = {"ledger": ledger_existed, "result": result_existed}
        for name in ("ledger", "result"):
            moved = name in event_members
            original = source_directories[name]
            if moved and original.exists():
                raise OpenAxesExternalModelV1Error(
                    "quarantine duplicated moved source state"
                )
            if not source_flags[name] and moved:
                raise OpenAxesExternalModelV1Error(
                    "quarantine invented moved source state"
                )
            if source_flags[name] and not moved and not original.exists():
                raise OpenAxesExternalModelV1Error(
                    "quarantine source state disappeared"
                )
            if moved:
                moved_receipts[name] = _validate_quarantine_moved_directory(
                    event_dir,
                    name=name,
                    task=task,
                    attempt_id=attempt_id,
                )
        observed_execution_started = False
        if ledger_existed:
            ledger_authority = (
                event_dir / "ledger"
                if "ledger" in event_members
                else source_directories["ledger"]
            )
            observed_execution_started = (
                ledger_authority / "execution.started.json"
            ).exists()
        if execution_started_observed is not observed_execution_started:
            raise OpenAxesExternalModelV1Error(
                "quarantine execution-start observation changed"
            )
        audit_path = event_members.get("audit.json")
        if audit_path is not None:
            audit = _mapping(
                _strict_json_load(audit_path), name="quarantine audit"
            )
            _keys(
                audit,
                {
                    "contract",
                    "event_id",
                    "task_id",
                    "task_spec_sha256",
                    "attempt_id",
                    "moved",
                    "moved_receipts",
                    "intent",
                    "recoverable",
                    "fresh_refit_permanently_forbidden",
                },
                name="quarantine audit",
            )
            for key, expected in (
                ("contract", OPENAXES_EXTERNAL_MODEL_V1_CONTRACT),
                ("event_id", event_id),
                ("task_id", task.task_id),
                ("task_spec_sha256", task.spec_sha256),
                ("attempt_id", attempt_id),
            ):
                _exact_str(
                    audit[key], name=f"quarantine audit {key}", expected=expected
                )
            expected_moved = [
                name for name in ("ledger", "result") if name in event_members
            ]
            if audit["moved"] != expected_moved:
                raise OpenAxesExternalModelV1Error(
                    "quarantine audit moved set changed"
                )
            recorded_receipts = _mapping(
                audit["moved_receipts"], name="quarantine moved receipts"
            )
            if _canonical(recorded_receipts) != _canonical(moved_receipts):
                raise OpenAxesExternalModelV1Error(
                    "quarantine moved-tree receipt changed"
                )
            verified_intent_path = _verify_reference(
                event_dir, audit["intent"], name="quarantine intent reference"
            )
            if verified_intent_path != intent_path:
                raise OpenAxesExternalModelV1Error(
                    "quarantine intent path changed"
                )
            _exact_bool(
                audit["recoverable"],
                name="quarantine recoverability",
                expected=True,
            )
            _exact_bool(
                audit["fresh_refit_permanently_forbidden"],
                name="quarantine no-refit marker",
                expected=True,
            )
    return observed_tasks


def _task_has_quarantined_attempt(root: Path, task: TaskSpec) -> bool:
    quarantine = root / MODEL_QUARANTINE_DIR
    if not quarantine.exists():
        return False
    return task.task_id in _validate_quarantine_evidence(
        root, (task,), require_empty=False
    )


def _phase_path(root: Path, task: TaskSpec, phase: str) -> Path:
    if phase not in OPENAXES_MODEL_PHASES:
        raise OpenAxesExternalModelV1Error("phase is outside the closed chain")
    return root / MODEL_LEDGERS_DIR / task.task_id / f"{phase}.json"


def _task_result_dir(root: Path, task: TaskSpec) -> Path:
    if TASK_ID_PATTERN.fullmatch(task.task_id) is None:
        raise OpenAxesExternalModelV1Error("unsafe task id")
    return root / MODEL_RESULTS_DIR / task.task_id


def _validate_completed_task_result_member_set(
    root: Path, task: TaskSpec
) -> None:
    directory = _task_result_dir(root, task)
    if directory.is_symlink() or not directory.is_dir():
        raise OpenAxesExternalModelV1Error(
            "completed task result directory changed"
        )
    members = list(directory.iterdir())
    if {path.name for path in members} != MODEL_COMPLETED_TASK_RESULT_MEMBERS or any(
        path.is_symlink() or not path.is_file() for path in members
    ):
        raise OpenAxesExternalModelV1Error(
            "completed task result artifact member set changed"
        )


def _write_phase(
    root: Path,
    task: TaskSpec,
    phase: str,
    *,
    attempt_id: str,
    payload: Mapping[str, Any],
    execution_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    index = OPENAXES_MODEL_PHASES.index(phase)
    if execution_binding is None and index == 0 and (
        "launch_payload_sha256" in payload and "launch_id" in payload
    ):
        execution_binding = payload
    if _execution_session_path(root).is_file() and execution_binding is None:
        raise OpenAxesExternalModelV1Error(
            "production phase lacks its exact execution binding"
        )
    if not _execution_session_path(root).is_file() and execution_binding is not None:
        raise OpenAxesExternalModelV1Error(
            "execution binding exists without an immutable session"
        )
    previous: dict[str, Any] | None = None
    if index:
        previous_path = _phase_path(root, task, OPENAXES_MODEL_PHASES[index - 1])
        if not previous_path.is_file():
            raise OpenAxesExternalModelV1Error("phase chain predecessor is missing")
        previous = _reference(
            previous_path,
            relative=previous_path.relative_to(root).as_posix(),
        )
    receipt = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "task_id": task.task_id,
        "task_spec_sha256": task.spec_sha256,
        "phase": phase,
        "attempt_id": _hex64(attempt_id, name="attempt id"),
        "previous_phase": previous,
        "execution_binding": (
            None
            if execution_binding is None
            else deepcopy(dict(execution_binding))
        ),
        "payload": deepcopy(dict(payload)),
    }
    _atomic_json(_phase_path(root, task, phase), receipt, exclusive=True)
    return receipt


def _verify_phase_chain(root: Path, task: TaskSpec) -> tuple[dict[str, Any], ...]:
    ledger = root / MODEL_LEDGERS_DIR / task.task_id
    if not ledger.exists():
        return ()
    if ledger.is_symlink() or not ledger.is_dir():
        raise OpenAxesExternalModelV1Error("task ledger is not a regular directory")
    members = {path.name for path in ledger.iterdir()}
    allowed = {f"{phase}.json" for phase in OPENAXES_MODEL_PHASES}
    if not members.issubset(allowed) or any(path.is_symlink() for path in ledger.iterdir()):
        raise OpenAxesExternalModelV1Error("task ledger member set changed")
    present = [phase for phase in OPENAXES_MODEL_PHASES if f"{phase}.json" in members]
    if present != list(OPENAXES_MODEL_PHASES[: len(present)]):
        raise OpenAxesExternalModelV1Error("task phases are not a contiguous prefix")
    receipts: list[dict[str, Any]] = []
    attempt: str | None = None
    previous_execution_binding: dict[str, Any] | None = None
    previous_launch_ordinal: int | None = None
    for index, phase in enumerate(present):
        path = _phase_path(root, task, phase)
        receipt = _mapping(_strict_json_load(path), name=f"phase {phase}")
        _keys(
            receipt,
            {
                "contract",
                "task_id",
                "task_spec_sha256",
                "phase",
                "attempt_id",
                "previous_phase",
                "execution_binding",
                "payload",
            },
            name=f"phase {phase}",
        )
        _exact_str(receipt["contract"], name="phase contract", expected=OPENAXES_EXTERNAL_MODEL_V1_CONTRACT)
        _exact_str(receipt["task_id"], name="phase task id", expected=task.task_id)
        _exact_str(receipt["task_spec_sha256"], name="phase task hash", expected=task.spec_sha256)
        _exact_str(receipt["phase"], name="phase name", expected=phase)
        current_attempt = _hex64(receipt["attempt_id"], name="phase attempt")
        if attempt is None:
            attempt = current_attempt
        elif current_attempt != attempt:
            raise OpenAxesExternalModelV1Error("phase attempt lineage changed")
        payload = _mapping(receipt["payload"], name="phase payload")
        if _execution_session_path(root).is_file():
            binding = _verify_execution_binding(
                root, task, receipt["execution_binding"]
            )
            if index == 0 and _canonical(payload) != _canonical(binding):
                raise OpenAxesExternalModelV1Error(
                    "execution-start phase binding payload changed"
                )
            launch_value = _mapping(
                _strict_json_load(
                    _launch_started_path(root, binding["launch_id"])
                ),
                name="phase execution launch",
            )
            launch_ordinal = _exact_int(
                launch_value["launch_ordinal"], name="phase launch ordinal"
            )
            if previous_execution_binding is not None:
                for key in (
                    "execution_session_id",
                    "transport_slot_index",
                    "local_slot",
                    "gpu",
                    "controller_source_sha256",
                    "task_assignment_sha256",
                    "slot_assignment_sha256",
                    "worker_identity_source",
                    "scientific_fit_attempt",
                ):
                    if _canonical(binding[key]) != _canonical(
                        previous_execution_binding[key]
                    ):
                        raise OpenAxesExternalModelV1Error(
                            f"phase execution lineage changed: {key}"
                        )
                if previous_launch_ordinal is None or launch_ordinal < previous_launch_ordinal:
                    raise OpenAxesExternalModelV1Error(
                        "phase execution launch lineage moved backwards"
                    )
                if (
                    launch_ordinal == previous_launch_ordinal
                    and _canonical(binding) != _canonical(previous_execution_binding)
                ):
                    raise OpenAxesExternalModelV1Error(
                        "same-launch phase execution binding changed"
                    )
            previous_execution_binding = binding
            previous_launch_ordinal = launch_ordinal
        elif receipt["execution_binding"] is not None:
            raise OpenAxesExternalModelV1Error(
                "phase binding exists without an immutable session"
            )
        if index == 0:
            if receipt["previous_phase"] is not None:
                raise OpenAxesExternalModelV1Error("first phase has a predecessor")
        else:
            expected_reference = _reference(
                _phase_path(root, task, present[index - 1]),
                relative=_phase_path(root, task, present[index - 1]).relative_to(root).as_posix(),
            )
            if receipt["previous_phase"] != expected_reference:
                raise OpenAxesExternalModelV1Error("phase predecessor hash changed")
        receipts.append(deepcopy(dict(receipt)))
    if receipts and _execution_session_path(root).is_file():
        persistent_attempt = _verify_task_attempt_ledger(root, task)
        if persistent_attempt is None or (
            persistent_attempt["attempt_id"] != receipts[0]["attempt_id"]
            or _canonical(persistent_attempt["binding"])
            != _canonical(receipts[0]["execution_binding"])
        ):
            raise OpenAxesExternalModelV1Error(
                "phase chain lacks its first-attempt binding anchor"
            )
    return tuple(receipts)


def _verify_execution_binding(
    root: Path, task: TaskSpec, binding_value: Any
) -> dict[str, Any]:
    binding = _mapping(binding_value, name="task execution binding")
    _keys(
        binding,
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
        },
        name="task execution binding",
    )
    manifest = _mapping(
        _strict_json_load(root / MODEL_PREPARED_MANIFEST),
        name="model prepared manifest",
    )
    tasks = _load_task_index(root, manifest)
    session = _validate_execution_session(
        root, manifest, tasks, _strict_json_load(_execution_session_path(root))
    )
    _exact_str(
        binding["execution_session_id"],
        name="task execution session",
        expected=session["execution_session_id"],
    )
    launch_id = _hex64(binding["launch_id"], name="task launch id")
    launch = _validate_execution_launch(root, session, launch_id)
    controller_observations: Mapping[str, Any] | None = None
    if manifest["synthetic_test_only"] is False:
        controller_observations = _verify_controller_ready_observations(
            root, session, launch
        )
    launch_path = _verify_reference(
        root, binding["launch_started"], name="task launch-start reference"
    )
    if launch_path != _launch_started_path(root, launch_id):
        raise OpenAxesExternalModelV1Error("task launch-start path changed")
    slot_index = _exact_int(
        binding["transport_slot_index"], name="task transport slot"
    )
    if not 0 <= slot_index < len(session["slots"]):
        raise OpenAxesExternalModelV1Error("task transport slot is outside session")
    slot = session["slots"][slot_index]
    if task.task_id not in slot["task_ids"]:
        raise OpenAxesExternalModelV1Error("task is bound to another execution slot")
    expected = {
        "launch_payload_sha256": launch["launch_payload_sha256"],
        "local_slot": slot["local_slot"],
        "gpu": slot["gpu"],
        "controller_source_sha256": session["controller_source_sha256"],
        "task_assignment_sha256": session["task_assignment_sha256"],
        "slot_assignment_sha256": slot["assignment_sha256"],
        "worker_identity_source": "fixed_process_transport_and_ready_receipt",
        "scientific_fit_attempt": 1,
    }
    for key, expected_value in expected.items():
        if _canonical(binding[key]) != _canonical(expected_value):
            raise OpenAxesExternalModelV1Error(f"task execution binding changed: {key}")
    _verify_worker_ready_receipt(root, session, launch, slot)
    ready_path = _verify_reference(
        root, binding["worker_ready"], name="task worker-ready reference"
    )
    if ready_path != _worker_ready_path(root, launch_id, slot_index):
        raise OpenAxesExternalModelV1Error("task worker-ready path changed")
    if controller_observations is None:
        if (
            binding["controller_ready_observations"] is not None
            or binding["controller_observations_sha256"] is not None
        ):
            raise OpenAxesExternalModelV1Error(
                "synthetic task binding invented controller observations"
            )
    else:
        observations_path = _verify_reference(
            root,
            binding["controller_ready_observations"],
            name="task controller-ready observation reference",
        )
        if observations_path != _controller_ready_observations_path(
            root, launch_id
        ):
            raise OpenAxesExternalModelV1Error(
                "task controller-ready observation path changed"
            )
        _exact_str(
            binding["controller_observations_sha256"],
            name="task controller-ready observation hash",
            expected=controller_observations["controller_observations_sha256"],
        )
    return deepcopy(dict(binding))


def _verify_task_attempt_roster(root: Path, task: TaskSpec) -> dict[str, Any] | None:
    path = _attempt_roster_path(root, task)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise OpenAxesExternalModelV1Error("monotonic task attempt roster is invalid")
    roster = _mapping(_strict_json_load(path), name="monotonic task attempt roster")
    _keys(
        roster,
        {
            "contract",
            "task_id",
            "task_spec_sha256",
            "attempt_id",
            "origin",
            "binding",
            "attempt_started_or_state_observed",
            "monotonic_first_attempt_roster",
            "survives_ledger_deletion_and_quarantine",
            "attempt_roster_sha256",
        },
        name="monotonic task attempt roster",
    )
    _exact_str(
        roster["contract"],
        name="attempt roster contract",
        expected=OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
    )
    _exact_str(roster["task_id"], name="attempt roster task", expected=task.task_id)
    _exact_str(
        roster["task_spec_sha256"],
        name="attempt roster task hash",
        expected=task.spec_sha256,
    )
    _hex64(roster["attempt_id"], name="attempt roster attempt id")
    origin = _exact_str(roster["origin"], name="attempt roster origin")
    if origin == "production_execution":
        authenticated_binding = _verify_execution_binding(
            root, task, roster["binding"]
        )
        _exact_str(
            roster["attempt_id"],
            name="attempt roster semantic id",
            expected=_task_attempt_semantic_id(task, authenticated_binding),
        )
    elif origin == "quarantine_recovery":
        if roster["binding"] is not None:
            raise OpenAxesExternalModelV1Error(
                "quarantine attempt roster has a forged execution binding"
            )
    else:
        raise OpenAxesExternalModelV1Error("attempt roster origin changed")
    for key in (
        "attempt_started_or_state_observed",
        "monotonic_first_attempt_roster",
        "survives_ledger_deletion_and_quarantine",
    ):
        _exact_bool(roster[key], name=f"attempt roster {key}", expected=True)
    base = {
        key: roster[key]
        for key in roster
        if key != "attempt_roster_sha256"
    }
    if _hex64(
        roster["attempt_roster_sha256"], name="attempt roster hash"
    ) != _canonical_sha256(base):
        raise OpenAxesExternalModelV1Error("monotonic task attempt roster changed")
    return deepcopy(dict(roster))


def _verify_task_attempt_ledger(root: Path, task: TaskSpec) -> dict[str, Any] | None:
    path = _attempt_ledger_path(root, task)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise OpenAxesExternalModelV1Error("persistent task attempt ledger is invalid")
    ledger = _mapping(_strict_json_load(path), name="persistent task attempt ledger")
    _keys(
        ledger,
        {
            "contract",
            "task_id",
            "task_spec_sha256",
            "attempt_id",
            "origin",
            "binding",
            "attempt_started_or_state_observed",
            "survives_quarantine",
            "attempt_ledger_sha256",
        },
        name="persistent task attempt ledger",
    )
    _exact_str(
        ledger["contract"],
        name="attempt ledger contract",
        expected=OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
    )
    _exact_str(ledger["task_id"], name="attempt ledger task", expected=task.task_id)
    _exact_str(
        ledger["task_spec_sha256"],
        name="attempt ledger task hash",
        expected=task.spec_sha256,
    )
    _hex64(ledger["attempt_id"], name="attempt ledger attempt id")
    origin = _exact_str(ledger["origin"], name="attempt ledger origin")
    if origin == "production_execution":
        _verify_execution_binding(root, task, ledger["binding"])
    elif origin == "quarantine_recovery":
        if ledger["binding"] is not None:
            raise OpenAxesExternalModelV1Error(
                "quarantine-recovered attempt has a forged execution binding"
            )
    else:
        raise OpenAxesExternalModelV1Error("attempt ledger origin changed")
    for key in ("attempt_started_or_state_observed", "survives_quarantine"):
        _exact_bool(ledger[key], name=f"attempt ledger {key}", expected=True)
    base = {key: ledger[key] for key in ledger if key != "attempt_ledger_sha256"}
    if _hex64(ledger["attempt_ledger_sha256"], name="attempt ledger hash") != _canonical_sha256(base):
        raise OpenAxesExternalModelV1Error("persistent task attempt ledger hash changed")
    roster = _verify_task_attempt_roster(root, task)
    if roster is None or (
        roster["attempt_id"] != ledger["attempt_id"]
        or roster["origin"] != ledger["origin"]
        or _canonical(roster["binding"]) != _canonical(ledger["binding"])
    ):
        raise OpenAxesExternalModelV1Error(
            "persistent attempt ledger/roster lineage changed"
        )
    return deepcopy(dict(ledger))


def classify_task_resume_state(root: str | Path, task: TaskSpec) -> str:
    """Return the next safe action; ambiguous fit/eval windows fail closed."""

    resolved = Path(root).resolve()
    receipts = _verify_phase_chain(resolved, task)
    roster = _verify_task_attempt_roster(resolved, task)
    attempt = _verify_task_attempt_ledger(resolved, task)
    if not receipts:
        if (
            roster is not None
            or attempt is not None
            or _task_has_quarantined_attempt(resolved, task)
        ):
            raise OpenAxesExternalModelV1Error(
                "persistent prior attempt evidence forbids a fresh fit"
            )
        return "fresh_fit"
    if _execution_session_path(resolved).is_file():
        if attempt is None:
            raise OpenAxesExternalModelV1Error(
                "production phase state lacks its persistent attempt ledger"
            )
        if attempt["attempt_id"] != receipts[0]["attempt_id"]:
            raise OpenAxesExternalModelV1Error("phase and attempt-ledger lineage changed")
        binding = _verify_execution_binding(
            resolved, task, receipts[0]["payload"]
        )
        if _canonical(binding) != _canonical(attempt["binding"]):
            raise OpenAxesExternalModelV1Error(
                "phase and attempt-ledger execution binding changed"
            )
    last = receipts[-1]["phase"]
    if last == "execution.started":
        raise OpenAxesExternalModelV1Error(
            "ambiguous started-only training cannot be automatically refit"
        )
    if last == "training.completed":
        return "predict_without_refit"
    if last == "outer_prediction.completed":
        return "evaluate_once"
    if last == "outer_eval.started":
        raise OpenAxesExternalModelV1Error(
            "outer evaluation started without authenticated completion"
        )
    if last == "outer_eval.completed":
        return "bundle_without_reevaluation"
    if last == "bundle.completed":
        return "write_result_only"
    if last == "result.completed":
        return "complete"
    raise OpenAxesExternalModelV1Error("unknown resume state")


def _training_artifact_references(root: Path, task: TaskSpec) -> dict[str, Any]:
    directory = _task_result_dir(root, task)
    result = {
        "checkpoint": _reference(
            directory / "checkpoint.pt",
            relative=(directory / "checkpoint.pt").relative_to(root).as_posix(),
        ),
        "transform": _reference(
            directory / "transform.json",
            relative=(directory / "transform.json").relative_to(root).as_posix(),
        ),
        "history": _reference(
            directory / "history.json",
            relative=(directory / "history.json").relative_to(root).as_posix(),
        ),
        "training_receipt": _reference(
            directory / "training_receipt.json",
            relative=(directory / "training_receipt.json").relative_to(root).as_posix(),
        ),
    }
    return result


def _verify_training_completed(
    root: Path,
    task: TaskSpec,
    phase: Mapping[str, Any],
    *,
    verify_checkpoint_state: bool = False,
) -> dict[str, Any]:
    payload = _mapping(phase["payload"], name="training completed payload")
    _keys(
        payload,
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
        },
        name="training completed payload",
    )
    _exact_str(payload["method_id"], name="training method", expected=task.method_id)
    _exact_int(payload["seed"], name="training seed", expected=task.seed)
    _exact_int(payload["updates"], name="training updates", expected=400)
    checkpoint_update = _exact_int(payload["checkpoint_update"], name="checkpoint update")
    if not 1 <= checkpoint_update <= 400:
        raise OpenAxesExternalModelV1Error("checkpoint update is outside budget")
    score = _exact_float(payload["checkpoint_score_m"], name="checkpoint score")
    if score < 0.0:
        raise OpenAxesExternalModelV1Error("checkpoint score is negative")
    _exact_str(
        payload["checkpoint_selection_scope"],
        name="checkpoint scope",
        expected="training_scenarios_only",
    )
    _exact_int(payload["outer_eval_count"], name="training outer count", expected=0)
    _hex64(payload["initial_state_sha256"], name="initial state hash")
    _hex64(payload["checkpoint_state_sha256"], name="checkpoint state hash")
    _hex64(payload["view_sha256"], name="five-view hash")
    artifacts = _mapping(payload["artifacts"], name="training artifacts")
    _keys(artifacts, {"checkpoint", "transform", "history", "training_receipt"}, name="training artifacts")
    for key in artifacts:
        _verify_reference(root, artifacts[key], name=f"training artifact {key}")
    if dict(artifacts) != _training_artifact_references(root, task):
        raise OpenAxesExternalModelV1Error("training artifact reference set changed")
    receipt = _mapping(
        _strict_json_load(_task_result_dir(root, task) / "training_receipt.json"),
        name="training receipt",
    )
    if receipt != {key: payload[key] for key in payload if key != "artifacts"}:
        raise OpenAxesExternalModelV1Error("training receipt and phase disagree")
    transform_payload = _mapping(
        _strict_json_load(_task_result_dir(root, task) / "transform.json"),
        name="training transform receipt",
    )
    _keys(
        transform_payload,
        {
            "feature_mean",
            "feature_scale",
            "target_mean_m",
            "target_scale_m",
            "target_scale_isotropic",
            "train_rows",
            "train_uid_sha256",
            "outer_rows_used",
            "sha256",
        },
        name="training transform receipt",
    )
    transform_base = {
        key: transform_payload[key] for key in transform_payload if key != "sha256"
    }
    if _canonical_sha256(transform_base) != transform_payload["sha256"]:
        raise OpenAxesExternalModelV1Error("training transform receipt hash changed")
    vectors = (
        ("feature_mean", 36),
        ("feature_scale", 36),
        ("target_mean_m", 3),
    )
    for key, length in vectors:
        vector = transform_payload[key]
        if type(vector) is not list or len(vector) != length or any(
            type(item) is not float or not math.isfinite(item) for item in vector
        ):
            raise OpenAxesExternalModelV1Error("training transform vector changed")
    if any(value <= 0.0 for value in transform_payload["feature_scale"]):
        raise OpenAxesExternalModelV1Error("training feature scale is invalid")
    target_scale = _exact_float(
        transform_payload["target_scale_m"], name="training target scale"
    )
    if target_scale <= 0.0:
        raise OpenAxesExternalModelV1Error("training target scale is invalid")
    _exact_bool(
        transform_payload["target_scale_isotropic"],
        name="training isotropic target scale",
        expected=True,
    )
    train_rows = _exact_int(
        transform_payload["train_rows"], name="training transform rows"
    )
    if train_rows < 2:
        raise OpenAxesExternalModelV1Error("training transform has too few rows")
    _hex64(transform_payload["train_uid_sha256"], name="training UID hash")
    _exact_int(
        transform_payload["outer_rows_used"], name="training transform outer rows", expected=0
    )
    history_payload = _mapping(
        _strict_json_load(_task_result_dir(root, task) / "history.json"),
        name="training history",
    )
    _keys(history_payload, {"method_id", "seed", "updates", "records"}, name="training history")
    _exact_str(history_payload["method_id"], name="history method", expected=task.method_id)
    _exact_int(history_payload["seed"], name="history seed", expected=task.seed)
    _exact_int(history_payload["updates"], name="history updates", expected=400)
    records = history_payload["records"]
    if type(records) is not list or len(records) != 400:
        raise OpenAxesExternalModelV1Error("training history length changed")
    scores: list[float] = []
    for index, raw_record in enumerate(records, start=1):
        record = _mapping(raw_record, name=f"history record {index}")
        _keys(
            record,
            {
                "update",
                "objective",
                "mask_loss",
                "tail",
                "interaction_penalty",
                "training_outage_rmse_m",
                "checkpoint_score_m",
                "became_checkpoint",
            },
            name=f"history record {index}",
        )
        _exact_int(record["update"], name="history update", expected=index)
        for key in ("objective", "mask_loss", "tail", "interaction_penalty", "checkpoint_score_m"):
            _exact_float(record[key], name=f"history {key}")
        outages = record["training_outage_rmse_m"]
        if type(outages) is not list or len(outages) != 4 or any(
            type(value) is not float or not math.isfinite(value) or value < 0.0
            for value in outages
        ):
            raise OpenAxesExternalModelV1Error("training outage history changed")
        _exact_bool(
            record["became_checkpoint"], name="history checkpoint marker", expected=record["became_checkpoint"]
        )
        scores.append(record["checkpoint_score_m"])
    first_minimum = min(range(len(scores)), key=scores.__getitem__) + 1
    if first_minimum != checkpoint_update or scores[first_minimum - 1] != score:
        raise OpenAxesExternalModelV1Error("earliest strict checkpoint history changed")
    expected_markers = [
        index == 0 or scores[index] < min(scores[:index]) for index in range(len(scores))
    ]
    if [record["became_checkpoint"] for record in records] != expected_markers:
        raise OpenAxesExternalModelV1Error("checkpoint improvement markers changed")
    if verify_checkpoint_state:
        import torch

        try:
            state = torch.load(
                _task_result_dir(root, task) / "checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise OpenAxesExternalModelV1Error("cannot strict-load completed checkpoint") from exc
        if not isinstance(state, Mapping) or state_dict_sha256(state) != payload[
            "checkpoint_state_sha256"
        ]:
            raise OpenAxesExternalModelV1Error("completed checkpoint tensor hash changed")
    return dict(payload)


def _load_external_preparation_parquet(root: Path) -> Path:
    manifest = _mapping(
        _strict_json_load(root / MODEL_PREPARED_MANIFEST), name="model prepared manifest"
    )
    preparation = _mapping(manifest["preparation"], name="preparation snapshot")
    artifacts = _mapping(preparation["artifacts"], name="preparation artifacts")
    if artifacts.get("synthetic_test_only") is not False:
        raise OpenAxesExternalModelV1Error("production task cannot use synthetic preparation")
    return _verify_external_reference(artifacts["parquet"], name="preparation parquet")


def _load_fold_frame(
    parquet_path: Path,
    scenarios: Sequence[str],
    columns: Sequence[str],
) -> pd.DataFrame:
    if not scenarios or any(type(item) is not str for item in scenarios) or any(
        type(column) is not str for column in columns
    ):
        raise OpenAxesExternalModelV1Error("fold parquet filter is invalid")
    try:
        frame = pd.read_parquet(
            parquet_path,
            columns=list(columns),
            filters=[("scenario_id", "in", list(scenarios))],
        )
    except Exception as exc:
        raise OpenAxesExternalModelV1Error("cannot read fold-filtered preparation parquet") from exc
    if len(frame) < 1 or set(frame["scenario_id"]) != set(scenarios):
        raise OpenAxesExternalModelV1Error("fold-filtered preparation rows changed")
    return frame.reset_index(drop=True)


def _atomic_torch_save(path: Path, value: Any) -> None:
    import torch

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _serialize_training_result(root: Path, task: TaskSpec, result: TrainingResult) -> dict[str, Any]:
    directory = _task_result_dir(root, task)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    _atomic_torch_save(checkpoint, dict(result.state_dict))
    transform_receipt = result.transform.receipt()
    _atomic_json(directory / "transform.json", transform_receipt)
    _atomic_json(
        directory / "history.json",
        {
            "method_id": result.method_id,
            "seed": result.seed,
            "updates": result.updates,
            "records": list(result.history),
        },
    )
    receipt = {
        "method_id": result.method_id,
        "seed": result.seed,
        "updates": result.updates,
        "checkpoint_update": result.checkpoint_update,
        "checkpoint_score_m": result.checkpoint_score_m,
        "checkpoint_selection_scope": result.checkpoint_selection_scope,
        "outer_eval_count": result.outer_eval_count,
        "initial_state_sha256": result.initial_state_sha256,
        "checkpoint_state_sha256": result.checkpoint_state_sha256,
        "view_sha256": result.view_sha256,
    }
    _atomic_json(directory / "training_receipt.json", receipt)
    return {**receipt, "artifacts": _training_artifact_references(root, task)}


def _deserialize_training_result(root: Path, task: TaskSpec, payload: Mapping[str, Any]) -> TrainingResult:
    import torch

    _verify_training_completed(
        root,
        task,
        {
            "payload": payload,
        },
    )
    directory = _task_result_dir(root, task)
    transform_payload = _mapping(_strict_json_load(directory / "transform.json"), name="transform receipt")
    _keys(
        transform_payload,
        {
            "feature_mean",
            "feature_scale",
            "target_mean_m",
            "target_scale_m",
            "target_scale_isotropic",
            "train_rows",
            "train_uid_sha256",
            "outer_rows_used",
            "sha256",
        },
        name="transform receipt",
    )
    base = {key: transform_payload[key] for key in transform_payload if key != "sha256"}
    if _canonical_sha256(base) != transform_payload["sha256"]:
        raise OpenAxesExternalModelV1Error("transform receipt hash changed")
    _exact_bool(transform_payload["target_scale_isotropic"], name="isotropic scale", expected=True)
    _exact_int(transform_payload["outer_rows_used"], name="transform outer rows", expected=0)
    feature_mean = np.asarray(transform_payload["feature_mean"])
    feature_scale = np.asarray(transform_payload["feature_scale"])
    target_mean = np.asarray(transform_payload["target_mean_m"])
    if feature_mean.shape != (36,) or feature_scale.shape != (36,) or target_mean.shape != (3,):
        raise OpenAxesExternalModelV1Error("transform vector shapes changed")
    if any(type(item) is not float for item in transform_payload["feature_mean"] + transform_payload["feature_scale"] + transform_payload["target_mean_m"]):
        raise OpenAxesExternalModelV1Error("transform vectors are not exact floats")
    target_scale = _exact_float(transform_payload["target_scale_m"], name="target scale")
    transform = TrainOnlyTransform(
        feature_mean=np.asarray(feature_mean, dtype=np.float64),
        feature_scale=np.asarray(feature_scale, dtype=np.float64),
        target_mean_m=np.asarray(target_mean, dtype=np.float64),
        target_scale_m=target_scale,
        train_rows=_exact_int(transform_payload["train_rows"], name="transform train rows"),
        train_uid_sha256=_hex64(transform_payload["train_uid_sha256"], name="transform UID hash"),
    )
    transform.validate()
    try:
        state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    except Exception as exc:
        raise OpenAxesExternalModelV1Error("cannot strict-load checkpoint") from exc
    if not isinstance(state, Mapping) or state_dict_sha256(state) != payload["checkpoint_state_sha256"]:
        raise OpenAxesExternalModelV1Error("checkpoint tensor state hash changed")
    history_payload = _mapping(_strict_json_load(directory / "history.json"), name="history")
    if (
        history_payload.get("method_id") != task.method_id
        or history_payload.get("seed") != task.seed
        or history_payload.get("updates") != 400
        or type(history_payload.get("records")) is not list
        or len(history_payload["records"]) != 400
    ):
        raise OpenAxesExternalModelV1Error("training history changed")
    return TrainingResult(
        method_id=task.method_id,
        seed=task.seed,
        transform=transform,
        state_dict=state,
        initial_state_sha256=payload["initial_state_sha256"],
        checkpoint_state_sha256=payload["checkpoint_state_sha256"],
        view_sha256=payload["view_sha256"],
        checkpoint_update=payload["checkpoint_update"],
        checkpoint_score_m=payload["checkpoint_score_m"],
        updates=400,
        history=tuple(history_payload["records"]),
        training_group_order=tuple(sorted(task.train_scenarios)),
    )


def _write_outer_prediction(
    root: Path,
    task: TaskSpec,
    metadata: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, Any]:
    if (
        list(metadata.columns) != ["row_uid", "scenario_id", "sample_index", "aligned_time_us"]
        or prediction.shape != (len(metadata), 5, 3)
        or not np.isfinite(prediction).all()
    ):
        raise OpenAxesExternalModelV1Error("outer prediction payload is invalid")
    pieces: list[pd.DataFrame] = []
    for view_index, view_id in enumerate(OPENAXES_MODEL_VIEW_IDS):
        piece = metadata[["row_uid", "scenario_id", "sample_index", "aligned_time_us"]].copy()
        piece["view_id"] = view_id
        piece["prediction_x_m"] = prediction[:, view_index, 0]
        piece["prediction_y_m"] = prediction[:, view_index, 1]
        piece["prediction_z_m"] = prediction[:, view_index, 2]
        pieces.append(piece)
    frame = pd.concat(pieces, ignore_index=True)
    expected_columns = [
        "row_uid",
        "scenario_id",
        "sample_index",
        "aligned_time_us",
        "view_id",
        "prediction_x_m",
        "prediction_y_m",
        "prediction_z_m",
    ]
    if list(frame.columns) != expected_columns or frame.duplicated(["row_uid", "view_id"]).any():
        raise OpenAxesExternalModelV1Error("outer prediction long table changed")
    path = _task_result_dir(root, task) / "outer_predictions.parquet"
    write_parquet(path, frame)
    return {
        "rows": len(frame),
        "base_rows": len(metadata),
        "views": 5,
        "finite": True,
        "prediction": _reference(path, relative=path.relative_to(root).as_posix()),
    }


def _verify_outer_prediction_completed(
    root: Path, task: TaskSpec, phase: Mapping[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    payload = _mapping(phase["payload"], name="outer prediction payload")
    _keys(payload, {"rows", "base_rows", "views", "finite", "prediction"}, name="outer prediction payload")
    rows = _exact_int(payload["rows"], name="prediction rows")
    base_rows = _exact_int(payload["base_rows"], name="prediction base rows")
    _exact_int(payload["views"], name="prediction views", expected=5)
    _exact_bool(payload["finite"], name="prediction finite", expected=True)
    if rows != base_rows * 5 or base_rows < 1:
        raise OpenAxesExternalModelV1Error("prediction 5-view row accounting changed")
    path = _verify_reference(root, payload["prediction"], name="outer prediction artifact")
    if path != _task_result_dir(root, task) / "outer_predictions.parquet":
        raise OpenAxesExternalModelV1Error("outer prediction artifact path changed")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise OpenAxesExternalModelV1Error("cannot decode outer prediction artifact") from exc
    expected_columns = [
        "row_uid",
        "scenario_id",
        "sample_index",
        "aligned_time_us",
        "view_id",
        "prediction_x_m",
        "prediction_y_m",
        "prediction_z_m",
    ]
    if (
        list(frame.columns) != expected_columns
        or len(frame) != rows
        or frame.duplicated(["row_uid", "view_id"]).any()
        or set(frame["view_id"]) != set(OPENAXES_MODEL_VIEW_IDS)
        or set(frame["scenario_id"]) != set(task.test_scenarios)
        or not np.isfinite(frame[["prediction_x_m", "prediction_y_m", "prediction_z_m"]].to_numpy()).all()
    ):
        raise OpenAxesExternalModelV1Error("outer prediction semantic replay failed")
    return dict(payload), frame


def _write_outer_evaluation(
    root: Path,
    task: TaskSpec,
    prediction_frame: pd.DataFrame,
    truth_frame: pd.DataFrame,
) -> dict[str, Any]:
    if list(truth_frame.columns) != [
        "row_uid",
        "scenario_id",
        "sample_index",
        "aligned_time_us",
        "tcp_x_m",
        "tcp_y_m",
        "tcp_z_m",
    ]:
        raise OpenAxesExternalModelV1Error("outer truth loader exposed an invalid schema")
    truth = truth_frame.set_index("row_uid", drop=False)
    if truth.index.has_duplicates:
        raise OpenAxesExternalModelV1Error("outer truth UID is duplicated")
    if set(prediction_frame["row_uid"]) != set(truth.index):
        raise OpenAxesExternalModelV1Error("outer prediction/truth UID identity changed")
    predicted_identity = (
        prediction_frame[prediction_frame["view_id"] == "full"]
        .set_index("row_uid")[["scenario_id", "sample_index", "aligned_time_us"]]
        .sort_index()
    )
    truth_identity = truth[["scenario_id", "sample_index", "aligned_time_us"]].sort_index()
    try:
        pd.testing.assert_frame_equal(
            predicted_identity, truth_identity, check_exact=True, check_dtype=True
        )
    except AssertionError as exc:
        raise OpenAxesExternalModelV1Error(
            "outer prediction/truth metadata identity changed"
        ) from exc
    ordered_uids = prediction_frame.loc[
        prediction_frame["view_id"] == OPENAXES_MODEL_VIEW_IDS[0], "row_uid"
    ].tolist()
    if len(ordered_uids) != len(truth) or len(set(ordered_uids)) != len(truth):
        raise OpenAxesExternalModelV1Error("outer base prediction order changed")
    tensor = np.empty((len(ordered_uids), 5, 3), dtype=np.float64)
    for view_index, view_id in enumerate(OPENAXES_MODEL_VIEW_IDS):
        view = prediction_frame[prediction_frame["view_id"] == view_id].set_index("row_uid")
        if set(view.index) != set(ordered_uids):
            raise OpenAxesExternalModelV1Error("outer view UID identity changed")
        tensor[:, view_index, :] = view.loc[
            ordered_uids, ["prediction_x_m", "prediction_y_m", "prediction_z_m"]
        ].to_numpy(dtype=np.float64)
    truth_array = truth.loc[ordered_uids, ["tcp_x_m", "tcp_y_m", "tcp_z_m"]].to_numpy(dtype=np.float64)
    metrics = evaluate_outer_predictions_once(tensor, truth_array)
    evaluation_rows = prediction_frame.merge(
        truth_frame[["row_uid", "tcp_x_m", "tcp_y_m", "tcp_z_m"]],
        on="row_uid",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    path = _task_result_dir(root, task) / "outer_evaluation_rows.parquet"
    write_parquet(path, evaluation_rows)
    metric_path = _task_result_dir(root, task) / "outer_evaluation.json"
    metric_payload = {
        "task_id": task.task_id,
        "method_id": task.method_id,
        "fold_id": task.fold_id,
        "family": task.family,
        "role": task.role,
        "seed": task.seed,
        "checkpoint_selection_scope": "training_scenarios_only",
        "outer_eval_count": 1,
        "metrics": metrics,
    }
    _atomic_json(metric_path, metric_payload)
    return {
        "outer_eval_count": 1,
        "checkpoint_selection_scope": "training_scenarios_only",
        "evaluation_rows": _reference(path, relative=path.relative_to(root).as_posix()),
        "evaluation_metrics": _reference(metric_path, relative=metric_path.relative_to(root).as_posix()),
    }


def _verify_outer_evaluation_completed(
    root: Path, task: TaskSpec, phase: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _mapping(phase["payload"], name="outer evaluation completed payload")
    _keys(
        payload,
        {"outer_eval_count", "checkpoint_selection_scope", "evaluation_rows", "evaluation_metrics"},
        name="outer evaluation completed payload",
    )
    _exact_int(payload["outer_eval_count"], name="outer evaluation count", expected=1)
    _exact_str(
        payload["checkpoint_selection_scope"],
        name="outer checkpoint scope",
        expected="training_scenarios_only",
    )
    rows_path = _verify_reference(root, payload["evaluation_rows"], name="evaluation rows")
    metrics_path = _verify_reference(root, payload["evaluation_metrics"], name="evaluation metrics")
    if (
        rows_path != _task_result_dir(root, task) / "outer_evaluation_rows.parquet"
        or metrics_path != _task_result_dir(root, task) / "outer_evaluation.json"
    ):
        raise OpenAxesExternalModelV1Error(
            "outer evaluation canonical artifact path changed"
        )
    metrics = _mapping(_strict_json_load(metrics_path), name="outer metrics payload")
    _keys(
        metrics,
        {
            "task_id",
            "method_id",
            "fold_id",
            "family",
            "role",
            "seed",
            "checkpoint_selection_scope",
            "outer_eval_count",
            "metrics",
        },
        name="outer metrics payload",
    )
    for key, expected in (
        ("task_id", task.task_id),
        ("method_id", task.method_id),
        ("fold_id", task.fold_id),
        ("family", task.family),
        ("role", task.role),
    ):
        _exact_str(metrics[key], name=f"outer metrics {key}", expected=expected)
    _exact_int(metrics["seed"], name="outer metrics seed", expected=task.seed)
    _exact_int(metrics["outer_eval_count"], name="outer metrics count", expected=1)
    _exact_str(
        metrics["checkpoint_selection_scope"],
        name="outer metrics checkpoint scope",
        expected="training_scenarios_only",
    )
    metric_body = _mapping(metrics["metrics"], name="outer metric body")
    _keys(metric_body, {"outer_eval_count", "views", "finite"}, name="outer metric body")
    _exact_int(metric_body["outer_eval_count"], name="metric evaluator count", expected=1)
    _exact_bool(metric_body["finite"], name="outer metric finite", expected=True)
    views = metric_body["views"]
    if type(views) is not list or len(views) != 5:
        raise OpenAxesExternalModelV1Error("outer metric view set changed")
    for index, view in enumerate(views):
        item = _mapping(view, name=f"outer metric view {index}")
        _keys(item, {"view_id", "rows", "rmse3d_m", "p95_distance_m", "mean_distance_m"}, name=f"outer metric view {index}")
        _exact_str(item["view_id"], name="outer metric view id", expected=OPENAXES_MODEL_VIEW_IDS[index])
        _exact_int(item["rows"], name="outer metric view rows")
        for name in ("rmse3d_m", "p95_distance_m", "mean_distance_m"):
            value = _exact_float(item[name], name=f"outer metric {name}")
            if value < 0.0:
                raise OpenAxesExternalModelV1Error("outer metric is negative")
    try:
        evaluation_rows = pd.read_parquet(rows_path)
    except Exception as exc:
        raise OpenAxesExternalModelV1Error("cannot decode persisted outer evaluation rows") from exc
    required_numeric = [
        "prediction_x_m",
        "prediction_y_m",
        "prediction_z_m",
        "tcp_x_m",
        "tcp_y_m",
        "tcp_z_m",
    ]
    if (
        evaluation_rows.empty
        or evaluation_rows.duplicated(["row_uid", "view_id"]).any()
        or set(evaluation_rows["view_id"]) != set(OPENAXES_MODEL_VIEW_IDS)
        or set(evaluation_rows["scenario_id"]) != set(task.test_scenarios)
        or not np.isfinite(evaluation_rows[required_numeric].to_numpy()).all()
    ):
        raise OpenAxesExternalModelV1Error("persisted outer evaluation rows are invalid")
    # Recompute metrics from authenticated persisted truth/prediction without any source read.
    recomputed: list[dict[str, Any]] = []
    for view_id in OPENAXES_MODEL_VIEW_IDS:
        view = evaluation_rows[evaluation_rows["view_id"] == view_id]
        error = view[["prediction_x_m", "prediction_y_m", "prediction_z_m"]].to_numpy() - view[
            ["tcp_x_m", "tcp_y_m", "tcp_z_m"]
        ].to_numpy()
        distance = np.linalg.norm(error, axis=1)
        recomputed.append(
            {
                "view_id": view_id,
                "rows": len(view),
                "rmse3d_m": float(
                    np.sqrt(np.mean(np.square(distance), dtype=np.float64))
                ),
                "p95_distance_m": float(np.percentile(distance, 95.0)),
                "mean_distance_m": float(np.mean(distance, dtype=np.float64)),
            }
        )
    if _canonical(recomputed) != _canonical(views):
        raise OpenAxesExternalModelV1Error("outer metric semantic recomputation changed")
    return dict(payload)


def _write_bundle_and_result(
    root: Path,
    task: TaskSpec,
    *,
    attempt_id: str,
    execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    phases = _verify_phase_chain(root, task)
    if not phases or phases[-1]["phase"] != "outer_eval.completed":
        raise OpenAxesExternalModelV1Error("cannot bundle before completed outer evaluation")
    training = _verify_training_completed(
        root, task, phases[1], verify_checkpoint_state=True
    )
    evaluation = _verify_outer_evaluation_completed(root, task, phases[4])
    bundle_path = _task_result_dir(root, task) / "bundle.json"
    bundle = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "task": task.payload(),
        "training": training,
        "evaluation": evaluation,
        "phase_lineage_sha256": _canonical_sha256(list(phases)),
    }
    _atomic_json(bundle_path, bundle)
    bundle_phase = _write_phase(
        root,
        task,
        "bundle.completed",
        attempt_id=attempt_id,
        execution_binding=execution_binding,
        payload={
            "bundle": _reference(bundle_path, relative=bundle_path.relative_to(root).as_posix())
        },
    )
    result_path = _task_result_dir(root, task) / "result.json"
    result = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "task": task.payload(),
        "bundle": bundle_phase["payload"]["bundle"],
        "checkpoint_selection_scope": "training_scenarios_only",
        "outer_eval_count": 1,
        "scientific_fit_count": 1,
        "synthetic_test_only": False,
    }
    _atomic_json(result_path, result)
    _write_phase(
        root,
        task,
        "result.completed",
        attempt_id=attempt_id,
        execution_binding=execution_binding,
        payload={
            "result": _reference(result_path, relative=result_path.relative_to(root).as_posix())
        },
    )
    return result


def _verify_completed_result(root: Path, task: TaskSpec) -> dict[str, Any]:
    phases = _verify_phase_chain(root, task)
    if len(phases) != len(OPENAXES_MODEL_PHASES):
        raise OpenAxesExternalModelV1Error("task result phase chain is incomplete")
    _validate_completed_task_result_member_set(root, task)
    attempt = _verify_task_attempt_ledger(root, task)
    if attempt is None:
        raise OpenAxesExternalModelV1Error(
            "completed task lacks its persistent attempt ledger"
        )
    if (
        attempt["attempt_id"] != phases[0]["attempt_id"]
        or _canonical(attempt["binding"])
        != _canonical(phases[0]["execution_binding"])
    ):
        raise OpenAxesExternalModelV1Error(
            "completed task attempt/binding lineage changed"
        )
    _verify_training_completed(root, task, phases[1], verify_checkpoint_state=True)
    _verify_outer_prediction_completed(root, task, phases[2])
    _verify_outer_evaluation_completed(root, task, phases[4])
    bundle_payload = _mapping(phases[5]["payload"], name="bundle phase payload")
    _keys(bundle_payload, {"bundle"}, name="bundle phase payload")
    bundle_path = _verify_reference(
        root, bundle_payload["bundle"], name="bundle artifact"
    )
    if bundle_path != _task_result_dir(root, task) / "bundle.json":
        raise OpenAxesExternalModelV1Error("bundle artifact path changed")
    bundle = _mapping(_strict_json_load(bundle_path), name="task bundle")
    _keys(
        bundle,
        {
            "contract",
            "task",
            "training",
            "evaluation",
            "phase_lineage_sha256",
        },
        name="task bundle",
    )
    _exact_str(
        bundle["contract"],
        name="bundle contract",
        expected=OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
    )
    verify_task_payload(bundle["task"], task)
    if (
        _canonical(bundle["training"]) != _canonical(phases[1]["payload"])
        or _canonical(bundle["evaluation"]) != _canonical(phases[4]["payload"])
    ):
        raise OpenAxesExternalModelV1Error(
            "bundle training/evaluation evidence changed"
        )
    _exact_str(
        bundle["phase_lineage_sha256"],
        name="bundle phase lineage hash",
        expected=_canonical_sha256(list(phases[:5])),
    )
    result_payload = _mapping(phases[6]["payload"], name="result phase payload")
    _keys(result_payload, {"result"}, name="result phase payload")
    result_path = _verify_reference(root, result_payload["result"], name="result artifact")
    if result_path != _task_result_dir(root, task) / "result.json":
        raise OpenAxesExternalModelV1Error("result artifact path changed")
    result = _mapping(_strict_json_load(result_path), name="task result")
    _keys(
        result,
        {
            "contract",
            "task",
            "bundle",
            "checkpoint_selection_scope",
            "outer_eval_count",
            "scientific_fit_count",
            "synthetic_test_only",
        },
        name="task result",
    )
    _exact_str(result["contract"], name="result contract", expected=OPENAXES_EXTERNAL_MODEL_V1_CONTRACT)
    verify_task_payload(result["task"], task)
    _exact_str(
        result["checkpoint_selection_scope"], name="result checkpoint scope", expected="training_scenarios_only"
    )
    _exact_int(result["outer_eval_count"], name="result outer count", expected=1)
    _exact_int(result["scientific_fit_count"], name="result fit count", expected=1)
    _exact_bool(result["synthetic_test_only"], name="result synthetic marker", expected=False)
    if _canonical(result["bundle"]) != _canonical(bundle_payload["bundle"]):
        raise OpenAxesExternalModelV1Error("result bundle reference changed")
    return dict(result)


def _write_result_from_completed_bundle(
    root: Path,
    task: TaskSpec,
    *,
    attempt_id: str,
    execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    phases = _verify_phase_chain(root, task)
    if not phases or phases[-1]["phase"] != "bundle.completed":
        raise OpenAxesExternalModelV1Error("result-only recovery requires a completed bundle")
    bundle_payload = _mapping(phases[-1]["payload"], name="bundle phase payload")
    _keys(bundle_payload, {"bundle"}, name="bundle phase payload")
    bundle_reference = deepcopy(dict(bundle_payload["bundle"]))
    bundle_path = _verify_reference(root, bundle_reference, name="completed bundle")
    if bundle_path != _task_result_dir(root, task) / "bundle.json":
        raise OpenAxesExternalModelV1Error("completed bundle path changed")
    result_path = _task_result_dir(root, task) / "result.json"
    result = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "task": task.payload(),
        "bundle": bundle_reference,
        "checkpoint_selection_scope": "training_scenarios_only",
        "outer_eval_count": 1,
        "scientific_fit_count": 1,
        "synthetic_test_only": False,
    }
    _atomic_json(result_path, result)
    _write_phase(
        root,
        task,
        "result.completed",
        attempt_id=attempt_id,
        execution_binding=execution_binding,
        payload={"result": _reference(result_path, relative=result_path.relative_to(root).as_posix())},
    )
    return result


def _existing_attempt_id(root: Path, task: TaskSpec) -> str:
    phases = _verify_phase_chain(root, task)
    if not phases:
        raise OpenAxesExternalModelV1Error("task has no existing attempt")
    return _hex64(phases[0]["attempt_id"], name="existing attempt id")


def _production_task_execute(
    run_dir: str,
    task: TaskSpec,
    *,
    resume: bool,
    execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute/recover one scientific fit inside its fixed spawned GPU child."""

    root = Path(run_dir).resolve()
    if type(resume) is not bool:
        raise OpenAxesExternalModelV1Error("task execution transport is invalid")
    authenticated_binding = _verify_execution_binding(
        root, task, execution_binding
    )
    phases = _verify_phase_chain(root, task)
    if phases and not resume:
        raise OpenAxesExternalModelV1Error("existing task state requires explicit resume")
    state = classify_task_resume_state(root, task)
    if state == "complete":
        return _verify_completed_result(root, task)
    if phases:
        attempt_id = _existing_attempt_id(root, task)
    else:
        attempt_id = _task_attempt_semantic_id(task, authenticated_binding)
        _write_task_attempt_ledger(
            root,
            task,
            attempt_id=attempt_id,
            binding=authenticated_binding,
        )
        (_phase_path(root, task, "execution.started").parent).mkdir(parents=True, exist_ok=False)
        _task_result_dir(root, task).mkdir(parents=True, exist_ok=False)
        _write_phase(
            root,
            task,
            "execution.started",
            attempt_id=attempt_id,
            execution_binding=authenticated_binding,
            payload=authenticated_binding,
        )
        state = "fit_after_started"

    if state == "bundle_without_reevaluation":
        # A completed-evaluation recovery reads only authenticated persisted
        # phase artifacts.  It does not reopen preparation data or call an
        # evaluator/truth loader.
        completed_phases = _verify_phase_chain(root, task)
        _verify_outer_evaluation_completed(root, task, completed_phases[4])
        return _write_bundle_and_result(
            root,
            task,
            attempt_id=attempt_id,
            execution_binding=authenticated_binding,
        )
    if state == "write_result_only":
        return _write_result_from_completed_bundle(
            root,
            task,
            attempt_id=attempt_id,
            execution_binding=authenticated_binding,
        )

    manifest = _mapping(
        _strict_json_load(root / MODEL_PREPARED_MANIFEST), name="model prepared manifest"
    )
    preparation = _mapping(manifest["preparation"], name="preparation snapshot")
    schema = _mapping(preparation["schema"], name="preparation schema")
    feature_columns = schema["model_feature_columns"]
    target_columns = schema["target_columns"]
    metadata_columns = schema["metadata_columns_not_model_inputs"]
    if (
        type(feature_columns) is not list
        or len(feature_columns) != 36
        or type(target_columns) is not list
        or target_columns != ["tcp_x_m", "tcp_y_m", "tcp_z_m"]
        or type(metadata_columns) is not list
        or metadata_columns != ["row_uid", "scenario_id", "sample_index", "aligned_time_us"]
    ):
        raise OpenAxesExternalModelV1Error("prepared model schema snapshot changed")
    parquet = _load_external_preparation_parquet(root)

    if state == "fit_after_started":
        # Only training-scenario rows and columns are materialized before checkpoint lock.
        training_frame = _load_fold_frame(
            parquet,
            task.train_scenarios,
            [*metadata_columns, *feature_columns, *target_columns],
        )
        training = train_model_train_only(
            training_frame[feature_columns].to_numpy(dtype=np.float64),
            training_frame[target_columns].to_numpy(dtype=np.float64),
            training_frame["scenario_id"].tolist(),
            training_frame["row_uid"].tolist(),
            method_id=task.method_id,
            seed=task.seed,
            updates=400,
            device="cuda:0",
        )
        training_payload = _serialize_training_result(root, task, training)
        _write_phase(
            root,
            task,
            "training.completed",
            attempt_id=attempt_id,
            execution_binding=authenticated_binding,
            payload=training_payload,
        )
        state = "predict_without_refit"

    if state == "predict_without_refit":
        phases = _verify_phase_chain(root, task)
        training_payload = _verify_training_completed(root, task, phases[1])
        training = _deserialize_training_result(root, task, training_payload)
        # Feature-only outer loader: target columns are not requested.
        outer_features = _load_fold_frame(
            parquet,
            task.test_scenarios,
            [*metadata_columns, *feature_columns],
        )
        prediction = predict_outer_views_once(
            training,
            outer_features[feature_columns].to_numpy(dtype=np.float64),
            device="cuda:0",
        )
        prediction_payload = _write_outer_prediction(
            root, task, outer_features[metadata_columns], prediction
        )
        _write_phase(
            root,
            task,
            "outer_prediction.completed",
            attempt_id=attempt_id,
            execution_binding=authenticated_binding,
            payload=prediction_payload,
        )
        state = "evaluate_once"

    if state == "evaluate_once":
        phases = _verify_phase_chain(root, task)
        prediction_payload, prediction_frame = _verify_outer_prediction_completed(
            root, task, phases[2]
        )
        _write_phase(
            root,
            task,
            "outer_eval.started",
            attempt_id=attempt_id,
            execution_binding=authenticated_binding,
            payload={
                "prediction": deepcopy(dict(prediction_payload["prediction"])),
                "truth_read_count_before": 0,
                "evaluator_call_count_before": 0,
            },
        )
        # This is the sole task-level outer truth materialization/evaluator call.
        outer_truth = _load_fold_frame(
            parquet,
            task.test_scenarios,
            [*metadata_columns, *target_columns],
        )
        evaluation_payload = _write_outer_evaluation(
            root, task, prediction_frame, outer_truth
        )
        _write_phase(
            root,
            task,
            "outer_eval.completed",
            attempt_id=attempt_id,
            execution_binding=authenticated_binding,
            payload=evaluation_payload,
        )
        state = "bundle_without_reevaluation"

    if state == "bundle_without_reevaluation":
        # Completed recovery verifies persisted truth/prediction/metrics only; it
        # does not call the evaluator or preparation truth loader again.
        phases = _verify_phase_chain(root, task)
        _verify_outer_evaluation_completed(root, task, phases[4])
        return _write_bundle_and_result(
            root,
            task,
            attempt_id=attempt_id,
            execution_binding=authenticated_binding,
        )
    if state == "write_result_only":
        return _write_result_from_completed_bundle(
            root,
            task,
            attempt_id=attempt_id,
            execution_binding=authenticated_binding,
        )
    raise OpenAxesExternalModelV1Error(f"unhandled task resume state: {state}")


def quarantine_task_state_nondelete(
    run_dir: str | Path, task: TaskSpec, *, reason: str
) -> dict[str, Any]:
    """Move corrupt non-ambiguous state through an intent/move/audit chain."""

    root = Path(run_dir).resolve()
    if type(reason) is not str or not reason:
        raise OpenAxesExternalModelV1Error("quarantine reason is required")
    ledger = root / MODEL_LEDGERS_DIR / task.task_id
    result = _task_result_dir(root, task)
    if not ledger.exists() and not result.exists():
        raise OpenAxesExternalModelV1Error("there is no task state to quarantine")
    quarantine = root / MODEL_QUARANTINE_DIR
    quarantine.mkdir(exist_ok=True)
    persistent_roster = _verify_task_attempt_roster(root, task)
    persistent_attempt = _verify_task_attempt_ledger(root, task)
    execution_started_observed = (
        ledger.is_dir() and (ledger / "execution.started.json").exists()
    )
    if persistent_attempt is None and persistent_roster is None:
        recovered_attempt_id: str | None = None
        if ledger.is_dir() and not ledger.is_symlink():
            for phase in OPENAXES_MODEL_PHASES:
                phase_path = ledger / f"{phase}.json"
                if not phase_path.is_file() or phase_path.is_symlink():
                    continue
                try:
                    candidate = _mapping(
                        _strict_json_load(phase_path),
                        name="quarantine recovery phase",
                    )
                    if (
                        candidate.get("task_id") == task.task_id
                        and candidate.get("task_spec_sha256") == task.spec_sha256
                    ):
                        recovered_attempt_id = _hex64(
                            candidate.get("attempt_id"),
                            name="quarantine recovered attempt id",
                        )
                except OpenAxesExternalModelV1Error:
                    recovered_attempt_id = None
                break
        if recovered_attempt_id is None:
            recovered_attempt_id = hashlib.sha256(
                f"quarantine|{task.task_id}|{time.time_ns()}".encode("utf-8")
            ).hexdigest()
        persistent_roster = _write_task_attempt_roster(
            root,
            task,
            attempt_id=recovered_attempt_id,
            origin="quarantine_recovery",
            binding=None,
        )
        recovered = {
            "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
            "task_id": task.task_id,
            "task_spec_sha256": task.spec_sha256,
            "attempt_id": recovered_attempt_id,
            "origin": "quarantine_recovery",
            "binding": None,
            "attempt_started_or_state_observed": True,
            "survives_quarantine": True,
        }
        recovered["attempt_ledger_sha256"] = _canonical_sha256(recovered)
        _atomic_json(_attempt_ledger_path(root, task), recovered, exclusive=True)
        persistent_attempt = recovered
    event_id = hashlib.sha256(
        f"{task.task_id}|{time.time_ns()}|{reason}".encode("utf-8")
    ).hexdigest()
    event_dir = quarantine / f"event_{event_id}"
    event_dir.mkdir()
    intent = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "event_id": event_id,
        "task_id": task.task_id,
        "task_spec_sha256": task.spec_sha256,
        "attempt_id": persistent_roster["attempt_id"],
        "reason": reason,
        "sources": {
            "ledger_exists": ledger.exists(),
            "result_exists": result.exists(),
            "execution_started_file_observed": execution_started_observed,
            "monotonic_attempt_roster": _reference(
                _attempt_roster_path(root, task),
                relative=_attempt_roster_path(root, task)
                .relative_to(root)
                .as_posix(),
            ),
            "persistent_attempt_ledger": (
                None
                if persistent_attempt is None
                else _reference(
                    _attempt_ledger_path(root, task),
                    relative=_attempt_ledger_path(root, task)
                    .relative_to(root)
                    .as_posix(),
                )
            ),
        },
        "nondelete": True,
    }
    _atomic_json(event_dir / "intent.json", intent)
    moved: list[str] = []
    if ledger.exists():
        os.replace(ledger, event_dir / "ledger")
        moved.append("ledger")
    if result.exists():
        os.replace(result, event_dir / "result")
        moved.append("result")
    moved_receipts = {
        name: _validate_quarantine_moved_directory(
            event_dir,
            name=name,
            task=task,
            attempt_id=persistent_roster["attempt_id"],
        )
        for name in moved
    }
    audit = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "event_id": event_id,
        "task_id": task.task_id,
        "task_spec_sha256": task.spec_sha256,
        "attempt_id": persistent_roster["attempt_id"],
        "moved": moved,
        "moved_receipts": moved_receipts,
        "intent": _reference(event_dir / "intent.json", relative="intent.json"),
        "recoverable": True,
        "fresh_refit_permanently_forbidden": True,
    }
    _atomic_json(event_dir / "audit.json", audit)
    return audit


def _query_physical_gpu_rows() -> list[dict[str, str]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OpenAxesExternalModelV1Error("cannot query physical GPU identities") from exc
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            raise OpenAxesExternalModelV1Error("nvidia-smi GPU row is malformed")
        rows.append(
            {"index": parts[0], "uuid": parts[1], "pci_bus_id": parts[2], "name": parts[3]}
        )
    return rows


def _validate_child_visible_gpu(frozen_gpu: Mapping[str, Any]) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != frozen_gpu["uuid"]:
        raise OpenAxesExternalModelV1Error("child CUDA visibility differs from frozen UUID")
    frozen_uuid = _canonical_gpu_uuid(frozen_gpu["uuid"], name="frozen GPU UUID")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise OpenAxesExternalModelV1Error("child must see exactly one CUDA device")
    properties = torch.cuda.get_device_properties(0)
    if torch.cuda.get_device_name(0) != frozen_gpu["name"]:
        raise OpenAxesExternalModelV1Error("visible CUDA device name changed")
    property_uuid = getattr(properties, "uuid", None)
    if _canonical_gpu_uuid(str(property_uuid), name="visible CUDA UUID") != frozen_uuid:
        raise OpenAxesExternalModelV1Error("visible CUDA UUID changed")
    physical_rows = _query_physical_gpu_rows()
    matches = [
        row
        for row in physical_rows
        if _canonical_gpu_uuid(row["uuid"], name="nvidia-smi GPU UUID")
        == frozen_uuid
    ]
    if len(matches) != 1:
        raise OpenAxesExternalModelV1Error("frozen physical GPU UUID is not unique/live")
    match = matches[0]
    if (
        match["index"] != str(frozen_gpu["physical_index"])
        or match["pci_bus_id"].lower() != frozen_gpu["pci_bus_id"].lower()
        or match["name"] != frozen_gpu["name"]
    ):
        raise OpenAxesExternalModelV1Error("visible CUDA UUID/PCI/name mapping changed")


def _production_slot_bootstrap(
    run_dir: str,
    session: Mapping[str, Any],
    launch: Mapping[str, Any],
    slot: Mapping[str, Any],
    tasks: Sequence[TaskSpec],
    resume: bool,
    output_queue: Any,
    controller_ready_barrier: Any,
) -> None:
    """Spawn child entry: visibility is fixed before the first torch import."""

    try:
        slot_index = slot["slot_index"]
        frozen_gpu = slot["gpu"]
        if "torch" in sys.modules:
            raise OpenAxesExternalModelV1Error(
                "spawn child imported torch before CUDA visibility bootstrap"
            )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(frozen_gpu["uuid"])
        _validate_child_visible_gpu(frozen_gpu)
        root = Path(run_dir).resolve()
        ready = _write_worker_ready_receipt(root, session, launch, slot)
        output_queue.put(
            {
                "kind": "slot_ready",
                "slot_index": slot_index,
                "launch_id": launch["launch_id"],
                "ready_receipt_sha256": ready["ready_receipt_sha256"],
                "gpu": deepcopy(dict(frozen_gpu)),
            }
        )
        if not controller_ready_barrier.wait(timeout=120.0):
            raise OpenAxesExternalModelV1Error(
                "controller did not durably authenticate all worker identities"
            )
        binding = _execution_binding(root, session, launch, slot)
        for task in tasks:
            _production_task_execute(
                run_dir,
                task,
                resume=resume,
                execution_binding=binding,
            )
            output_queue.put(
                {
                    "kind": "task_completed",
                    "slot_index": slot_index,
                    "task_id": task.task_id,
                    "ordinal": task.ordinal,
                }
            )
        output_queue.put({"kind": "slot_completed", "slot_index": slot_index})
    except BaseException as exc:
        output_queue.put(
            {
                "kind": "slot_error",
                "slot_index": slot_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise


def run_openaxes_external_model_tasks(
    model_config: Mapping[str, Any],
    run_dir: str | Path,
    *,
    workers_per_gpu: int,
    resume: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Run all 336 production tasks in fixed spawned per-GPU-slot FIFOs."""

    if type(resume) is not bool:
        raise OpenAxesExternalModelV1Error("resume marker is not exact")
    if "torch" in sys.modules:
        raise OpenAxesExternalModelV1Error(
            "production controller imported torch before GPU-child spawn"
        )
    validation = validate_openaxes_external_model_prepared_run(
        model_config, run_dir, allow_synthetic_test_only=False
    )
    _exact_bool(validation["synthetic_test_only"], name="production run marker", expected=False)
    root = Path(run_dir).resolve()
    manifest = _mapping(
        _strict_json_load(root / MODEL_PREPARED_MANIFEST), name="model prepared manifest"
    )
    tasks = _load_task_index(root, manifest)
    session = _freeze_or_load_execution_session(
        root,
        manifest,
        tasks,
        workers_per_gpu=workers_per_gpu,
        resume=resume,
        pre_fit_validation=validation,
    )
    launch = _start_execution_launch(root, session, resume=resume)
    slots = tuple(deepcopy(slot) for slot in session["slots"])
    by_id = {task.task_id: task for task in tasks}
    context = multiprocessing.get_context("spawn")
    output_queue = context.Queue()
    controller_ready_barrier = context.Event()
    processes: list[Any] = []
    for slot in slots:
        slot_tasks = tuple(by_id[task_id] for task_id in slot["task_ids"])
        process = context.Process(
            target=_production_slot_bootstrap,
            args=(
                str(root),
                session,
                launch,
                slot,
                slot_tasks,
                resume,
                output_queue,
                controller_ready_barrier,
            ),
            name=f"openaxes-model-slot-{slot['slot_index']}",
        )
        process.start()
        processes.append(process)
    completed: dict[str, dict[str, Any]] = {}
    completed_slots: set[int] = set()
    ready_slots: set[int] = set()
    try:
        while (
            len(completed) < len(tasks)
            or len(completed_slots) < len(slots)
            or len(ready_slots) < len(slots)
        ):
            try:
                message = output_queue.get(timeout=2.0)
            except queue.Empty:
                crashed = [
                    (index, process.exitcode)
                    for index, process in enumerate(processes)
                    if process.exitcode not in (None, 0)
                ]
                if crashed:
                    raise OpenAxesExternalModelV1Error(
                        f"spawned GPU worker crashed: {crashed}"
                    )
                continue
            payload = _mapping(message, name="worker transport message")
            kind = payload.get("kind")
            slot_index = payload.get("slot_index")
            if type(slot_index) is not int or not 0 <= slot_index < len(slots):
                raise OpenAxesExternalModelV1Error("worker transport slot identity changed")
            if kind == "slot_ready":
                _keys(
                    payload,
                    {
                        "kind",
                        "slot_index",
                        "launch_id",
                        "ready_receipt_sha256",
                        "gpu",
                    },
                    name="slot-ready transport",
                )
                if slot_index in ready_slots:
                    raise OpenAxesExternalModelV1Error("worker slot became ready twice")
                _exact_str(
                    payload["launch_id"],
                    name="ready launch id",
                    expected=launch["launch_id"],
                )
                ready = _verify_worker_ready_receipt(
                    root, session, launch, slots[slot_index]
                )
                process_id = _exact_int(
                    processes[slot_index].pid,
                    name="controller transport process id",
                )
                _exact_int(
                    ready["child_process_id"],
                    name="ready transport process id",
                    expected=process_id,
                )
                _exact_str(
                    payload["ready_receipt_sha256"],
                    name="transport ready hash",
                    expected=ready["ready_receipt_sha256"],
                )
                if _canonical(payload["gpu"]) != _canonical(slots[slot_index]["gpu"]):
                    raise OpenAxesExternalModelV1Error(
                        "worker ready GPU identity changed in transport"
                    )
                ready_slots.add(slot_index)
                if len(ready_slots) == len(slots):
                    _write_controller_ready_observations(
                        root, session, launch, slots, processes
                    )
                    controller_ready_barrier.set()
            elif kind == "task_completed":
                _keys(payload, {"kind", "slot_index", "task_id", "ordinal"}, name="task transport")
                task_id = _exact_str(payload["task_id"], name="transport task id")
                ordinal = _exact_int(payload["ordinal"], name="transport ordinal")
                task = by_id.get(task_id)
                if task is None or task.ordinal != ordinal or task_id in completed:
                    raise OpenAxesExternalModelV1Error("worker returned an unknown/duplicate task")
                if task_id not in slots[slot_index]["task_ids"]:
                    raise OpenAxesExternalModelV1Error("worker returned another slot's task")
                completed[task_id] = dict(payload)
            elif kind == "slot_completed":
                _keys(payload, {"kind", "slot_index"}, name="slot transport")
                if slot_index in completed_slots:
                    raise OpenAxesExternalModelV1Error("worker slot completed twice")
                completed_slots.add(slot_index)
            elif kind == "slot_error":
                raise OpenAxesExternalModelV1Error(
                    f"GPU slot {slot_index} failed: {payload.get('error_type')}: {payload.get('error')}"
                )
            else:
                raise OpenAxesExternalModelV1Error("worker transport kind changed")
    except Exception:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10.0)
        raise
    for process in processes:
        process.join(timeout=30.0)
        if process.is_alive() or process.exitcode != 0:
            raise OpenAxesExternalModelV1Error("spawned GPU worker did not exit cleanly")
    ordered: list[dict[str, Any]] = []
    for task in tasks:
        _verify_completed_result(root, task)
        ordered.append(completed[task.task_id])
    launch_dir = (
        root
        / MODEL_EXECUTION_LAUNCHES_DIR
        / f"launch_{launch['launch_id']}"
    )
    controller_observations = _verify_controller_ready_observations(
        root, session, launch
    )
    controller_observations_path = _controller_ready_observations_path(
        root, launch["launch_id"]
    )
    launch_completed = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "execution_session_id": session["execution_session_id"],
        "launch_id": launch["launch_id"],
        "launch_started": _reference(
            _launch_started_path(root, launch["launch_id"]),
            relative=_launch_started_path(root, launch["launch_id"])
            .relative_to(root)
            .as_posix(),
        ),
        "launch_payload_sha256": launch["launch_payload_sha256"],
        "launch_ordinal": launch["launch_ordinal"],
        "resume": launch["resume"],
        "slot_count": len(slots),
        "task_count": len(tasks),
        "task_assignment_sha256": session["task_assignment_sha256"],
        "controller_source_sha256": session["controller_source_sha256"],
        "model_source_set_sha256": session["model_source_set_sha256"],
        "parent_public_validation_sha256": session[
            "parent_public_validation_sha256"
        ],
        "pre_fit_validation_receipt_sha256": session[
            "pre_fit_validation_receipt_sha256"
        ],
        "controller_ready_observations": _reference(
            controller_observations_path,
            relative=controller_observations_path.relative_to(root).as_posix(),
        ),
        "controller_observations_sha256": controller_observations[
            "controller_observations_sha256"
        ],
        "ready_receipts": [
            _reference(
                _worker_ready_path(root, launch["launch_id"], slot["slot_index"]),
                relative=_worker_ready_path(
                    root, launch["launch_id"], slot["slot_index"]
                ).relative_to(root).as_posix(),
            )
            for slot in slots
        ],
        "all_tasks_authenticated_complete": True,
    }
    launch_completed["launch_completed_sha256"] = _canonical_sha256(
        launch_completed
    )
    _atomic_json(
        launch_dir / "launch.completed.json", launch_completed, exclusive=True
    )
    return tuple(ordered)


def _process_probe_bootstrap(
    slot_index: int,
    gpu_uuid: str,
    items: Sequence[Mapping[str, Any]],
    output_queue: Any,
) -> None:
    """Private synthetic process probe; never accepted as scientific evidence."""

    try:
        torch_before = "torch" in sys.modules
        if torch_before:
            raise OpenAxesExternalModelV1Error("torch was loaded before probe bootstrap")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        for item in items:
            item_id = _exact_str(item.get("item_id"), name="probe item id")
            seed = _exact_int(item.get("seed"), name="probe seed")
            duration_ms = _exact_int(item.get("duration_ms"), name="probe duration")
            if not 0 <= duration_ms <= 100:
                raise OpenAxesExternalModelV1Error("probe duration is outside test bound")
            time.sleep(duration_ms / 1000.0)
            rng_value = int(np.random.default_rng(seed).integers(0, 2**31 - 1))
            initial_hash = None
            if item.get("initial_state_probe") is True:
                from .openaxes_external_model_v1_training import build_masked_mlp

                initial_hash = state_dict_sha256(
                    build_masked_mlp(seed=seed, device="cpu").state_dict()
                )
            output_queue.put(
                {
                    "kind": "probe",
                    "slot_index": slot_index,
                    "item_id": item_id,
                    "seed": seed,
                    "rng_value": rng_value,
                    "initial_state_sha256": initial_hash,
                    "torch_imported_before_cuda_visible_devices": torch_before,
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                    "synthetic_test_only": True,
                }
            )
    except BaseException as exc:
        output_queue.put(
            {"kind": "error", "slot_index": slot_index, "error": str(exc)}
        )
        raise


def run_spawn_process_probe(
    slot_items: Sequence[Sequence[Mapping[str, Any]]],
    *,
    _private_test_only_mode: bool,
) -> tuple[dict[str, Any], ...]:
    """Exercise spawn/FIFO/RNG/init isolation without GPUs or benchmark data."""

    if _private_test_only_mode is not True:
        raise OpenAxesExternalModelV1Error("process probe is private-test-only")
    if not 1 <= len(slot_items) <= 12:
        raise OpenAxesExternalModelV1Error("process probe slot count is invalid")
    context = multiprocessing.get_context("spawn")
    output_queue = context.Queue()
    processes: list[Any] = []
    expected_items = sum(len(items) for items in slot_items)
    for slot_index, items in enumerate(slot_items):
        process = context.Process(
            target=_process_probe_bootstrap,
            args=(slot_index, f"GPU-SYNTHETIC-{slot_index}", tuple(items), output_queue),
        )
        process.start()
        processes.append(process)
    results: list[dict[str, Any]] = []
    try:
        while len(results) < expected_items:
            try:
                payload = _mapping(output_queue.get(timeout=2.0), name="probe message")
            except queue.Empty:
                if any(process.exitcode not in (None, 0) for process in processes):
                    raise OpenAxesExternalModelV1Error("process probe child crashed")
                continue
            if payload.get("kind") == "error":
                raise OpenAxesExternalModelV1Error(f"process probe failed: {payload.get('error')}")
            if payload.get("kind") != "probe":
                raise OpenAxesExternalModelV1Error("process probe message kind changed")
            results.append(dict(payload))
    finally:
        for process in processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
    if any(process.exitcode != 0 for process in processes):
        raise OpenAxesExternalModelV1Error("process probe did not exit cleanly")
    return tuple(results)


def verify_paired_method_controls(root: Path, tasks: Sequence[TaskSpec]) -> dict[str, Any]:
    """Prove paired methods share init, scaler, views, budget, and update count."""

    groups: dict[tuple[str, int], list[tuple[TaskSpec, dict[str, Any]]]] = {}
    for task in tasks:
        phases = _verify_phase_chain(root, task)
        if len(phases) != len(OPENAXES_MODEL_PHASES):
            raise OpenAxesExternalModelV1Error("paired-control task is incomplete")
        payload = _verify_training_completed(root, task, phases[1])
        groups.setdefault((task.fold_id, task.seed), []).append((task, payload))
    if len(groups) != 14 * 3:
        raise OpenAxesExternalModelV1Error("paired-control fold/seed group count changed")
    audits: list[dict[str, Any]] = []
    for (fold_id, seed), entries in sorted(groups.items()):
        if {task.method_id for task, _payload in entries} != set(OPENAXES_MODEL_METHOD_IDS):
            raise OpenAxesExternalModelV1Error("paired-control method set changed")
        initial = {payload["initial_state_sha256"] for _task, payload in entries}
        views = {payload["view_sha256"] for _task, payload in entries}
        transforms = {
            payload["artifacts"]["transform"]["sha256"] for _task, payload in entries
        }
        updates = {payload["updates"] for _task, payload in entries}
        if len(initial) != 1 or len(views) != 1 or len(transforms) != 1 or updates != {400}:
            raise OpenAxesExternalModelV1Error("paired method controls diverged")
        audits.append(
            {
                "fold_id": fold_id,
                "seed": seed,
                "method_count": 8,
                "initial_state_sha256": next(iter(initial)),
                "five_view_sha256": next(iter(views)),
                "transform_sha256": next(iter(transforms)),
                "updates": 400,
            }
        )
    return {
        "fold_seed_groups": len(audits),
        "method_count_per_group": 8,
        "identical_initial_state": True,
        "identical_train_only_transform": True,
        "identical_exact_five_views": True,
        "identical_update_budget": True,
        "records": audits,
    }


def _ensemble_evaluation_rows(root: Path, tasks: Sequence[TaskSpec]) -> pd.DataFrame:
    if len(tasks) != 3 or {task.seed for task in tasks} != set(OPENAXES_MODEL_SEEDS):
        raise OpenAxesExternalModelV1Error("seed ensemble must contain the exact paired three seeds")
    frames: list[pd.DataFrame] = []
    for task in sorted(tasks, key=lambda item: item.seed):
        phases = _verify_phase_chain(root, task)
        evaluation = _verify_outer_evaluation_completed(root, task, phases[4])
        path = _verify_reference(root, evaluation["evaluation_rows"], name="ensemble evaluation rows")
        frame = pd.read_parquet(path).sort_values(["row_uid", "view_id"], kind="stable").reset_index(drop=True)
        frames.append(frame)
    identity_columns = [
        "row_uid",
        "scenario_id",
        "sample_index",
        "aligned_time_us",
        "view_id",
        "tcp_x_m",
        "tcp_y_m",
        "tcp_z_m",
    ]
    reference = frames[0][identity_columns]
    for frame in frames[1:]:
        try:
            pd.testing.assert_frame_equal(
                frame[identity_columns], reference, check_exact=True, check_dtype=True
            )
        except AssertionError as exc:
            raise OpenAxesExternalModelV1Error("seed ensemble outer identity/truth changed") from exc
    output = reference.copy()
    for axis in ("x", "y", "z"):
        values = np.stack(
            [frame[f"prediction_{axis}_m"].to_numpy(dtype=np.float64) for frame in frames],
            axis=0,
        )
        output[f"prediction_{axis}_m"] = np.mean(values, axis=0, dtype=np.float64)
    output["seed_ensemble_size"] = 3
    return output


def _context_metrics(frame: pd.DataFrame, *, include_full: bool) -> dict[str, Any]:
    views = ["full"] if include_full else list(OPENAXES_MODEL_VIEW_IDS[1:])
    selected = frame[frame["view_id"].isin(views)].copy()
    if selected.empty or set(selected["view_id"]) != set(views):
        raise OpenAxesExternalModelV1Error("metric context view set changed")
    context_rows: list[dict[str, Any]] = []
    for (scenario, view), group in selected.groupby(["scenario_id", "view_id"], sort=True):
        identity_columns = [
            "row_uid",
            "scenario_id",
            "sample_index",
            "aligned_time_us",
            "view_id",
            "tcp_x_m",
            "tcp_y_m",
            "tcp_z_m",
        ]
        if any(column not in group.columns for column in identity_columns):
            raise OpenAxesExternalModelV1Error("metric context identity columns changed")
        identity_rows = (
            group[identity_columns]
            .sort_values(identity_columns, kind="stable")
            .to_dict(orient="records")
        )
        error = group[["prediction_x_m", "prediction_y_m", "prediction_z_m"]].to_numpy() - group[
            ["tcp_x_m", "tcp_y_m", "tcp_z_m"]
        ].to_numpy()
        distance = np.linalg.norm(error, axis=1)
        context_rows.append(
            {
                "scenario_id": str(scenario),
                "view_id": str(view),
                "rows": len(group),
                "rmse3d_m": float(np.sqrt(np.mean(np.square(distance), dtype=np.float64))),
                "p95_distance_m": float(np.percentile(distance, 95.0)),
                "identity_sha256": _canonical_sha256(identity_rows),
            }
        )
    context = pd.DataFrame(context_rows)
    if context.duplicated(["scenario_id", "view_id"]).any():
        raise OpenAxesExternalModelV1Error("metric context identity is duplicated")
    A = float(context["rmse3d_m"].mean())
    WS = float(context.groupby("scenario_id", sort=True)["rmse3d_m"].mean().max())
    WM = float(context.groupby("view_id", sort=True)["rmse3d_m"].mean().max())
    robust = (A + WS + WM) / 3.0
    return {
        "context_count": len(context),
        "A_mean_context_rmse3d_m": A,
        "WS_worst_scenario_mean_rmse3d_m": WS,
        "WM_worst_mask_mean_rmse3d_m": WM,
        "robust_J_m": robust,
        "contexts": context_rows,
    }


def _build_loso_directional_gate(
    method_scores: Sequence[Mapping[str, Any]],
    fold_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen PORD direction test on exactly matched LOSO contexts."""

    scores = {record["method_id"]: record for record in method_scores}
    if set(scores) != set(OPENAXES_MODEL_METHOD_IDS):
        raise OpenAxesExternalModelV1Error("directional-gate method score set changed")
    context_hashes = {
        record["primary_context_identity_sha256"] for record in method_scores
    }
    if len(context_hashes) != 1:
        raise OpenAxesExternalModelV1Error(
            "directional-gate methods do not share exact context identity"
        )
    shared_context_hash = next(iter(context_hashes))
    objective = scores["paired_pord"]
    comparator_ids = ("masked_erm", "group_mean_pord", "paired_wrong_pair")
    robust_comparisons: list[dict[str, Any]] = []
    for comparator_id in comparator_ids:
        comparator = scores[comparator_id]
        objective_value = _exact_float(
            objective["robust_J_m"], name="directional objective J"
        )
        comparator_value = _exact_float(
            comparator["robust_J_m"], name="directional comparator J"
        )
        robust_comparisons.append(
            {
                "comparator_method_id": comparator_id,
                "paired_pord_robust_J_m": objective_value,
                "comparator_robust_J_m": comparator_value,
                "strict_win": objective_value < comparator_value,
                "equality_passes": False,
            }
        )

    primary = [
        record for record in fold_records if record["family"] == "scenario_loso"
    ]
    by_method_scenario: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in primary:
        scenarios = record["test_scenarios"]
        if type(scenarios) is not list or len(scenarios) != 1:
            raise OpenAxesExternalModelV1Error(
                "directional LOSO record must contain one held-out scenario"
            )
        key = (record["method_id"], scenarios[0])
        if key in by_method_scenario:
            raise OpenAxesExternalModelV1Error("directional LOSO identity duplicated")
        by_method_scenario[key] = record
    scenario_ids = sorted(
        scenario
        for method, scenario in by_method_scenario
        if method == "paired_pord"
    )
    if len(scenario_ids) != 8:
        raise OpenAxesExternalModelV1Error("directional LOSO scenario count changed")
    scenario_comparisons: list[dict[str, Any]] = []
    for scenario in scenario_ids:
        objective_record = by_method_scenario.get(("paired_pord", scenario))
        comparator_record = by_method_scenario.get(("masked_erm", scenario))
        if objective_record is None or comparator_record is None:
            raise OpenAxesExternalModelV1Error(
                "directional LOSO comparator scenario is missing"
            )
        objective_value = _exact_float(
            objective_record["outage"]["A_mean_context_rmse3d_m"],
            name="directional scenario PORD score",
        )
        comparator_value = _exact_float(
            comparator_record["outage"]["A_mean_context_rmse3d_m"],
            name="directional scenario ERM score",
        )
        scenario_comparisons.append(
            {
                "scenario_id": scenario,
                "paired_pord_mean_four_outage_context_rmse3d_m": objective_value,
                "masked_erm_mean_four_outage_context_rmse3d_m": comparator_value,
                "strict_win": objective_value < comparator_value,
                "equality_passes": False,
            }
        )
    wins = sum(record["strict_win"] for record in scenario_comparisons)
    gate = {
        "objective_method_id": "paired_pord",
        "strict_robust_J_comparator_method_ids": list(comparator_ids),
        "scenario_comparator_method_id": "masked_erm",
        "scenario_score": "mean_four_outage_context_rmse3d_m",
        "shared_primary_context_identity_sha256": shared_context_hash,
        "method_context_identity_hashes_equal": True,
        "robust_J_comparisons": robust_comparisons,
        "scenario_comparisons": scenario_comparisons,
        "strict_scenario_wins": wins,
        "required_strict_scenario_wins": 6,
        "scenario_count": 8,
        "equality_passes": False,
        "stress_used": False,
        "alignment_sensitivity_used": False,
        "passes": all(record["strict_win"] for record in robust_comparisons)
        and wins >= 6,
    }
    gate["gate_sha256"] = _canonical_sha256(gate)
    return gate


def aggregate_zero_shift_development(
    root: Path, tasks: Sequence[TaskSpec]
) -> tuple[dict[str, Any], dict[tuple[str, str], pd.DataFrame]]:
    """Aggregate raw three-seed predictions; only 8 LOSO folds select."""

    grouped: dict[tuple[str, str], list[TaskSpec]] = {}
    for task in tasks:
        grouped.setdefault((task.method_id, task.fold_id), []).append(task)
    if len(grouped) != 8 * 14:
        raise OpenAxesExternalModelV1Error("method/fold ensemble grid changed")
    ensembles: dict[tuple[str, str], pd.DataFrame] = {}
    fold_records: list[dict[str, Any]] = []
    for (method_id, fold_id), seed_tasks in sorted(grouped.items()):
        ensemble = _ensemble_evaluation_rows(root, seed_tasks)
        ensembles[(method_id, fold_id)] = ensemble
        task = seed_tasks[0]
        outage = _context_metrics(ensemble, include_full=False)
        full = _context_metrics(
            ensemble[ensemble["view_id"] == "full"], include_full=True
        )
        fold_records.append(
            {
                "method_id": method_id,
                "fold_id": fold_id,
                "family": task.family,
                "role": task.role,
                "test_scenarios": list(task.test_scenarios),
                "seed_ensemble_size": 3,
                "outage": outage,
                "full_diagnostic": full,
            }
        )

    method_scores: list[dict[str, Any]] = []
    for method_id in OPENAXES_MODEL_METHOD_IDS:
        primary = [
            record
            for record in fold_records
            if record["method_id"] == method_id and record["family"] == "scenario_loso"
        ]
        if len(primary) != 8 or sum(record["outage"]["context_count"] for record in primary) != 32:
            raise OpenAxesExternalModelV1Error("primary 8-fold/32-context selection set changed")
        contexts = [
            context
            for record in primary
            for context in record["outage"]["contexts"]
        ]
        context_identity = sorted(
            [
                {
                    "fold_id": record["fold_id"],
                    "scenario_id": context["scenario_id"],
                    "view_id": context["view_id"],
                    "rows": context["rows"],
                    "identity_sha256": context["identity_sha256"],
                }
                for record in primary
                for context in record["outage"]["contexts"]
            ],
            key=lambda item: (item["fold_id"], item["scenario_id"], item["view_id"]),
        )
        context_frame = pd.DataFrame(contexts)
        A = float(context_frame["rmse3d_m"].mean())
        WS = float(context_frame.groupby("scenario_id")["rmse3d_m"].mean().max())
        WM = float(context_frame.groupby("view_id")["rmse3d_m"].mean().max())
        J = (A + WS + WM) / 3.0
        full_values = [record["full_diagnostic"]["robust_J_m"] for record in primary]
        method_scores.append(
            {
                "method_id": method_id,
                "selection_family": "scenario_loso",
                "folds": 8,
                "outage_contexts": 32,
                "A_mean_context_rmse3d_m": A,
                "WS_worst_scenario_mean_rmse3d_m": WS,
                "WM_worst_mask_mean_rmse3d_m": WM,
                "robust_J_m": J,
                "mean_full_diagnostic_rmse3d_m": float(np.mean(full_values)),
                "primary_context_identity_sha256": _canonical_sha256(
                    context_identity
                ),
            }
        )
    order = {method: index for index, method in enumerate(OPENAXES_MODEL_METHOD_IDS)}
    selected = min(method_scores, key=lambda item: (item["robust_J_m"], order[item["method_id"]]))
    stress = [record for record in fold_records if record["role"] == "stress"]
    if len(stress) != 8 * 6:
        raise OpenAxesExternalModelV1Error("stress report grid changed")
    directional_gate = _build_loso_directional_gate(method_scores, fold_records)
    result = {
        "selection": {
            "selected_method_id": selected["method_id"],
            "criterion": "robust_J_m",
            "source_family": "scenario_loso",
            "primary_folds": 8,
            "primary_outage_contexts_per_method": 32,
            "tie_rule": "exact_score_then_ordered_method_id",
            "stress_used": False,
            "alignment_sensitivity_used": False,
        },
        "method_scores": method_scores,
        "directional_gate": directional_gate,
        "fold_records": fold_records,
        "stress_report_only": stress,
        "counts": {
            "method_fold_seed_ensembles": 112,
            "primary_seed_ensemble_outage_contexts": 256,
            "stress_seed_ensemble_outage_contexts": 192,
        },
    }
    return result, ensembles


def _validate_signed_shift_reconstruction(
    truth: pd.DataFrame, receipt: Mapping[str, Any], preparation_manifest: Mapping[str, Any]
) -> None:
    expected_columns = [
        "row_uid",
        "scenario_id",
        "base_aligned_time_us",
        "shift_ms",
        "shifted_target_time_us",
        "shifted_tcp_x_m",
        "shifted_tcp_y_m",
        "shifted_tcp_z_m",
    ]
    if (
        not isinstance(truth, pd.DataFrame)
        or list(truth.columns) != expected_columns
        or truth.empty
        or truth.duplicated(["row_uid", "shift_ms"]).any()
        or set(truth["shift_ms"]) != set(OPENAXES_MODEL_SHIFT_MS)
        or not np.isfinite(
            truth[["shifted_tcp_x_m", "shifted_tcp_y_m", "shifted_tcp_z_m"]].to_numpy()
        ).all()
    ):
        raise OpenAxesExternalModelV1Error("signed-shift truth table changed")
    value = _mapping(receipt, name="signed-shift receipt")
    _keys(
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
        name="signed-shift receipt",
    )
    _exact_str(
        value.get("contract"),
        name="signed-shift contract",
        expected="openaxes.external.signed_shift_truth.v1",
    )
    _hex64(value["config_sha256"], name="signed-shift config hash")
    _exact_bool(
        value["deterministic_receipt"], name="signed-shift deterministic receipt", expected=True
    )
    _exact_str(
        value.get("preparation_contract"),
        name="signed-shift preparation contract",
        expected=OPENAXES_EXTERNAL_PREPARATION_V1_CONTRACT,
    )
    scope = _mapping(value.get("scope"), name="signed-shift scope")
    for key in (
        "read_only_truth_reconstruction",
        "semantic_validator_required_before_model_use",
    ):
        _exact_bool(scope.get(key), name=f"signed-shift {key}", expected=True)
    for key in (
        "features_included",
        "model_fit_performed",
        "metrics_computed",
        "selection_performed",
        "claim_gate_decided",
        "artifact_written",
    ):
        _exact_bool(scope.get(key), name=f"signed-shift {key}", expected=False)
    output = _mapping(value.get("output"), name="signed-shift output receipt")
    _keys(
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
        name="signed-shift output receipt",
    )
    if output.get("closed_schema") != expected_columns or output.get("rows") != len(truth):
        raise OpenAxesExternalModelV1Error("signed-shift output receipt changed")
    rows_per_shift = _mapping(output.get("rows_per_shift"), name="rows per shift")
    for shift in OPENAXES_MODEL_SHIFT_MS:
        count = _exact_int(rows_per_shift.get(str(shift)), name=f"shift {shift} rows")
        if count != int((truth["shift_ms"] == shift).sum()):
            raise OpenAxesExternalModelV1Error("signed-shift row receipt disagrees")
    dropped_edges = _mapping(
        output.get("dropped_edge_rows_by_scenario_and_shift"),
        name="signed-shift dropped edges",
    )
    expected_scenarios = {
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    }
    expected_shift_keys = {str(shift) for shift in OPENAXES_MODEL_SHIFT_MS}
    if set(dropped_edges) != expected_scenarios:
        raise OpenAxesExternalModelV1Error(
            "signed-shift dropped-edge scenario set changed"
        )
    for scenario_id, raw_counts in dropped_edges.items():
        counts = _mapping(
            raw_counts, name=f"signed-shift dropped edges {scenario_id}"
        )
        if set(counts) != expected_shift_keys:
            raise OpenAxesExternalModelV1Error(
                "signed-shift dropped-edge shift set changed"
            )
        for shift, count in counts.items():
            _exact_int(
                count,
                name=f"signed-shift fixed-source dropped edge {scenario_id}/{shift}",
                expected=0,
            )
    _exact_str(
        output["row_order"],
        name="signed-shift row order",
        expected="scenario_then_signed_shift_then_base_time",
    )
    _exact_bool(
        output["target_columns_only_with_pairing_metadata"],
        name="signed-shift target-only scope",
        expected=True,
    )
    schema = (
        ("row_uid", "utf8"),
        ("scenario_id", "utf8"),
        ("base_aligned_time_us", "int64"),
        ("shift_ms", "int64"),
        ("shifted_target_time_us", "int64"),
        ("shifted_tcp_x_m", "float64"),
        ("shifted_tcp_y_m", "float64"),
        ("shifted_tcp_z_m", "float64"),
    )
    if _hex64(output["long_table_sha256"], name="signed-shift table hash") != _closed_frame_sha256(
        truth, schema
    ):
        raise OpenAxesExternalModelV1Error("signed-shift long-table hash changed")
    prepared = _mapping(value.get("prepared"), name="signed-shift prepared receipt")
    _exact_bool(prepared.get("base_exact_source_replay"), name="shift base replay", expected=True)
    stage = _mapping(prepared.get("stage"), name="signed-shift stage")
    _exact_bool(
        stage.get("downstream_model_results_blocked_by_uncommitted_contract"),
        name="shift preparation downstream block",
        expected=False,
    )
    if prepared.get("base_rows") != preparation_manifest["counts"]["prepared_active_rows"]:
        raise OpenAxesExternalModelV1Error("signed-shift base row identity count changed")


def _build_sensitivity_family_summaries(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(list(records))
    if len(frame) != 168:
        raise OpenAxesExternalModelV1Error(
            "signed-sensitivity family summary input count changed"
        )
    summaries: list[dict[str, Any]] = []
    for (method_id, family, shift_ms), group in frame.groupby(
        ["method_id", "family", "shift_ms"], sort=True
    ):
        summaries.append(
            {
                "method_id": str(method_id),
                "family": str(family),
                "shift_ms": int(shift_ms),
                "folds": len(group),
                "outage_contexts": int(group["outage_context_count"].sum()),
                "mean_rmse3d_m": float(group["rmse3d_m"].mean()),
                "mean_robust_J_m": float(group["robust_J_m"].mean()),
                "mean_delta_rmse3d_m_shift_minus_zero": float(
                    group["delta_rmse3d_m_shift_minus_zero"].mean()
                ),
                "mean_delta_robust_J_m_shift_minus_zero": float(
                    group["delta_robust_J_m_shift_minus_zero"].mean()
                ),
            }
        )
    if len(summaries) != 2 * 3 * 6:
        raise OpenAxesExternalModelV1Error(
            "signed-sensitivity family summary identity changed"
        )
    return summaries


def aggregate_alignment_sensitivity(
    ensembles: Mapping[tuple[str, str], pd.DataFrame],
    tasks: Sequence[TaskSpec],
    shifted_truth: pd.DataFrame,
    shifted_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Report comparator/objective signed shifts; never return a selector input."""

    fold_lookup = {task.fold_id: task for task in tasks}
    records: list[dict[str, Any]] = []
    for method_id in ("masked_erm", "paired_pord"):
        for fold_id in sorted(fold_lookup):
            task = fold_lookup[fold_id]
            ensemble = ensembles[(method_id, fold_id)]
            outage = ensemble[ensemble["view_id"] != "full"].copy()
            for shift_ms in OPENAXES_MODEL_SHIFT_MS:
                shifted = shifted_truth[shifted_truth["shift_ms"] == shift_ms]
                merged = outage.merge(
                    shifted,
                    on=["row_uid", "scenario_id"],
                    how="inner",
                    validate="many_to_one",
                    sort=False,
                )
                if merged.empty or merged["row_uid"].nunique() != shifted[
                    shifted["scenario_id"].isin(task.test_scenarios)
                ]["row_uid"].nunique():
                    raise OpenAxesExternalModelV1Error("signed-shift outer UID intersection changed")
                shifted_metric_frame = merged.copy()
                for axis in ("x", "y", "z"):
                    shifted_metric_frame[f"tcp_{axis}_m"] = shifted_metric_frame[
                        f"shifted_tcp_{axis}_m"
                    ]
                shifted_metrics = _context_metrics(shifted_metric_frame, include_full=False)
                supported_uids = sorted(set(merged["row_uid"]))
                if supported_uids != sorted(set(outage["row_uid"])):
                    raise OpenAxesExternalModelV1Error(
                        "signed-shift replay lacks exact persisted ensemble UID support"
                    )
                zero_frame = outage[outage["row_uid"].isin(supported_uids)]
                zero_metrics = _context_metrics(zero_frame, include_full=False)

                shifted_error = merged[["prediction_x_m", "prediction_y_m", "prediction_z_m"]].to_numpy() - merged[
                    ["shifted_tcp_x_m", "shifted_tcp_y_m", "shifted_tcp_z_m"]
                ].to_numpy()
                zero_error = zero_frame[["prediction_x_m", "prediction_y_m", "prediction_z_m"]].to_numpy() - zero_frame[
                    ["tcp_x_m", "tcp_y_m", "tcp_z_m"]
                ].to_numpy()
                shifted_rmse = float(np.sqrt(np.mean(np.sum(np.square(shifted_error), axis=1))))
                zero_rmse = float(np.sqrt(np.mean(np.sum(np.square(zero_error), axis=1))))
                records.append(
                    {
                        "method_id": method_id,
                        "fold_id": fold_id,
                        "family": task.family,
                        "shift_ms": shift_ms,
                        "base_uid_rows": len(supported_uids),
                        "base_uids": supported_uids,
                        "base_uid_sha256": _canonical_sha256(
                            supported_uids
                        ),
                        "outage_context_count": shifted_metrics["context_count"],
                        "rmse3d_m": shifted_rmse,
                        "zero_same_uid_rmse3d_m": zero_rmse,
                        "delta_rmse3d_m_shift_minus_zero": shifted_rmse - zero_rmse,
                        "robust_J_m": shifted_metrics["robust_J_m"],
                        "zero_same_uid_robust_J_m": zero_metrics["robust_J_m"],
                        "delta_robust_J_m_shift_minus_zero": shifted_metrics["robust_J_m"]
                        - zero_metrics["robust_J_m"],
                    }
                )
    expected = 2 * 14 * 6
    if len(records) != expected:
        raise OpenAxesExternalModelV1Error("signed-sensitivity record count changed")
    summaries = _build_sensitivity_family_summaries(records)
    result = {
        "selection_use_allowed": False,
        "claim_gate_use_allowed": False,
        "methods": ["masked_erm", "paired_pord"],
        "signed_shifts_ms": list(OPENAXES_MODEL_SHIFT_MS),
        "fold_records": records,
        "family_summaries": summaries,
        "fold_record_identity_sha256": _canonical_sha256(
            [
                {
                    "method_id": record["method_id"],
                    "fold_id": record["fold_id"],
                    "family": record["family"],
                    "shift_ms": record["shift_ms"],
                    "base_uid_rows": record["base_uid_rows"],
                    "base_uids": record["base_uids"],
                    "base_uid_sha256": record["base_uid_sha256"],
                    "outage_context_count": record["outage_context_count"],
                }
                for record in records
            ]
        ),
        "family_summary_sha256": _canonical_sha256(summaries),
        "reconstruction_receipt": deepcopy(dict(shifted_receipt)),
        "shifted_truth_persisted_or_redistributed": False,
    }
    result["sensitivity_semantic_sha256"] = _canonical_sha256(result)
    return result


def _result_reference_index(root: Path, tasks: Sequence[TaskSpec]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in tasks:
        _verify_completed_result(root, task)
        path = _task_result_dir(root, task) / "result.json"
        result[task.task_id] = _reference(path, relative=path.relative_to(root).as_posix())
    if len(result) != 336:
        raise OpenAxesExternalModelV1Error("result reference index is not 336")
    return result


def _full_signed_source_replay_path(root: Path) -> Path:
    return root / MODEL_FULL_SIGNED_SOURCE_REPLAY


def _signed_shift_support_identity(
    shifted_truth: pd.DataFrame,
) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for (scenario_id, shift_ms), group in shifted_truth.groupby(
        ["scenario_id", "shift_ms"], sort=True
    ):
        uids = sorted(group["row_uid"].tolist())
        if len(uids) != len(set(uids)):
            raise OpenAxesExternalModelV1Error(
                "signed-source replay support UID identity changed"
            )
        identities.append(
            {
                "scenario_id": str(scenario_id),
                "shift_ms": int(shift_ms),
                "base_uid_rows": len(uids),
                "base_uid_sha256": _canonical_sha256(uids),
            }
        )
    if len(identities) != 8 * len(OPENAXES_MODEL_SHIFT_MS):
        raise OpenAxesExternalModelV1Error(
            "signed-source replay scenario/shift support set changed"
        )
    return identities


def _expected_full_signed_source_replay_receipt(
    root: Path,
    prepared: Mapping[str, Any],
    development: Mapping[str, Any],
    shifted_truth: pd.DataFrame,
    reconstruction_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    preparation = _mapping(prepared["preparation"], name="full replay preparation")
    artifacts = _mapping(
        preparation["artifacts"], name="full replay preparation artifacts"
    )
    _exact_bool(
        artifacts.get("synthetic_test_only"),
        name="full replay synthetic marker",
        expected=False,
    )
    artifact_references = {
        key: deepcopy(dict(artifacts[key]))
        for key in ("source", "config", "manifest", "parquet", "complete")
    }
    for key, reference in artifact_references.items():
        _verify_external_reference(reference, name=f"full replay {key}")
    source_closure = _mapping(
        prepared["source_closure"], name="full replay source closure"
    )
    source_files = _mapping(
        source_closure["files"], name="full replay source files"
    )
    support = _signed_shift_support_identity(shifted_truth)
    sensitivity = _mapping(
        development["alignment_sensitivity_report_only"],
        name="full replay sensitivity",
    )
    parent_authority = _mapping(
        preparation["parent_git_authority"], name="full replay parent authority"
    )
    receipt = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "receipt_kind": "immutable_full_signed_source_semantic_replay",
        "config_sha256": OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
        "prepared_manifest": _reference(
            root / MODEL_PREPARED_MANIFEST, relative=MODEL_PREPARED_MANIFEST
        ),
        "development_manifest": _reference(
            root / MODEL_DEVELOPMENT_MANIFEST,
            relative=MODEL_DEVELOPMENT_MANIFEST,
        ),
        "preparation_artifacts": artifact_references,
        "parent_public_validation_sha256": _canonical_sha256(
            preparation["validation"]
        ),
        "parent_git_authority_sha256": _canonical_sha256(parent_authority),
        "reconstruction_receipt_sha256": _canonical_sha256(
            reconstruction_receipt
        ),
        "scenario_shift_support_identity": support,
        "scenario_shift_support_identity_sha256": _canonical_sha256(support),
        "signed_report_sha256": _canonical_sha256(sensitivity),
        "validator_source_sha256": source_files[
            "src/imu_vive_learning/openaxes_external_model_v1_experiment.py"
        ]["sha256"],
        "validator_source_set_sha256": source_closure["set_sha256"],
        "full_signed_source_replay": True,
        "semantic_replay": True,
    }
    receipt["receipt_payload_sha256"] = _canonical_sha256(receipt)
    return receipt


def _freeze_or_validate_full_signed_source_replay_receipt(
    root: Path, expected: Mapping[str, Any], *, allow_create: bool
) -> dict[str, Any]:
    if type(allow_create) is not bool:
        raise OpenAxesExternalModelV1Error(
            "full signed-source receipt creation marker is not exact"
        )
    path = _full_signed_source_replay_path(root)
    if not path.exists() and allow_create:
        _atomic_json(path, expected, exclusive=True)
    if path.is_symlink() or not path.is_file():
        raise OpenAxesExternalModelV1Error(
            "full signed-source replay receipt is missing"
        )
    observed = _mapping(
        _strict_json_load(path), name="full signed-source replay receipt"
    )
    if _canonical(observed) != _canonical(expected):
        raise OpenAxesExternalModelV1Error(
            "full signed-source replay receipt changed"
        )
    return deepcopy(dict(observed))


def _replay_and_validate_full_signed_source(
    root: Path,
    prepared: Mapping[str, Any],
    tasks: Sequence[TaskSpec],
    development: Mapping[str, Any],
    *,
    allow_receipt_create: bool,
) -> dict[str, Any]:
    preparation = _mapping(prepared["preparation"], name="preparation snapshot")
    artifacts = _mapping(preparation["artifacts"], name="preparation artifacts")
    source = _verify_external_reference(artifacts["source"], name="official source")
    config_path = _verify_external_reference(
        artifacts["config"], name="preparation config"
    )
    prep_manifest_path = _verify_external_reference(
        artifacts["manifest"], name="preparation manifest"
    )
    from .openaxes_external_v1 import reconstruct_openaxes_signed_shift_truth_v1
    from .openaxes_external_v1_config import load_openaxes_config

    prep_manifest = _mapping(
        _strict_json_load(prep_manifest_path), name="preparation manifest"
    )
    public_validation = _run_parent_public_validator_at_recorded_git_state(
        source_path=source,
        preparation_config_path=config_path,
        preparation_run_dir=prep_manifest_path.parent,
        preparation_manifest=prep_manifest,
    )
    if _canonical(public_validation) != _canonical(preparation["validation"]):
        raise OpenAxesExternalModelV1Error(
            "full signed-source replay parent validation changed"
        )
    truth, reconstruction_receipt = reconstruct_openaxes_signed_shift_truth_v1(
        source, load_openaxes_config(config_path), prep_manifest_path.parent
    )
    _validate_signed_shift_reconstruction(
        truth, reconstruction_receipt, prep_manifest
    )
    _zero_shift, zero_ensembles = aggregate_zero_shift_development(root, tasks)
    replayed_sensitivity = aggregate_alignment_sensitivity(
        zero_ensembles, tasks, truth, reconstruction_receipt
    )
    if _canonical(replayed_sensitivity) != _canonical(
        development["alignment_sensitivity_report_only"]
    ):
        raise OpenAxesExternalModelV1Error(
            "signed-shift source replay canonical aggregation changed"
        )
    expected = _expected_full_signed_source_replay_receipt(
        root,
        prepared,
        development,
        truth,
        reconstruction_receipt,
    )
    return _freeze_or_validate_full_signed_source_replay_receipt(
        root, expected, allow_create=allow_receipt_create
    )


def _write_model_complete_marker(
    root: Path,
    development: Mapping[str, Any],
    full_replay: Mapping[str, Any],
) -> dict[str, Any]:
    development_path = root / MODEL_DEVELOPMENT_MANIFEST
    marker = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "config_sha256": OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
        "prepared_manifest": _reference(
            root / MODEL_PREPARED_MANIFEST, relative=MODEL_PREPARED_MANIFEST
        ),
        "development_manifest": _reference(
            development_path, relative=MODEL_DEVELOPMENT_MANIFEST
        ),
        "scientific_fits": 336,
        "outer_evaluations": 336,
        "selected_method_id": development["zero_shift"]["selection"][
            "selected_method_id"
        ],
        "selection_sha256": _canonical_sha256(
            development["zero_shift"]["selection"]
        ),
        "zero_shift_sha256": _canonical_sha256(development["zero_shift"]),
        "directional_gate_passes": development["zero_shift"]["directional_gate"][
            "passes"
        ],
        "directional_gate_sha256": _canonical_sha256(
            development["zero_shift"]["directional_gate"]
        ),
        "alignment_sensitivity_sha256": _canonical_sha256(
            development["alignment_sensitivity_report_only"]
        ),
        "full_signed_source_replay": _reference(
            _full_signed_source_replay_path(root),
            relative=MODEL_FULL_SIGNED_SOURCE_REPLAY,
        ),
        "full_signed_source_replay_sha256": _canonical_sha256(full_replay),
        "full_signed_source_replay_performed": True,
        "production_validation_requires_signed_source_replay": True,
        "stress_or_sensitivity_used_for_selection": False,
        "challenge_access_performed": False,
        "immutable": True,
    }
    _atomic_json(root / MODEL_COMPLETE_MARKER, marker, exclusive=True)
    return marker


def finalize_openaxes_external_model_development(
    model_config: Mapping[str, Any], run_dir: str | Path
) -> dict[str, Any]:
    """Aggregate all authenticated fits and public shift replay; no challenge exists."""

    root = Path(run_dir).resolve()
    if (root / MODEL_COMPLETE_MARKER).exists():
        return validate_openaxes_external_model_development(
            model_config, root, replay_signed_source=True
        )
    prepared_validation = validate_openaxes_external_model_prepared_run(
        model_config, root, allow_synthetic_test_only=False
    )
    _exact_bool(
        prepared_validation["synthetic_test_only"],
        name="development synthetic marker",
        expected=False,
    )
    prepared_manifest = _mapping(
        _strict_json_load(root / MODEL_PREPARED_MANIFEST), name="prepared model manifest"
    )
    tasks = _load_task_index(root, prepared_manifest)
    if (root / MODEL_DEVELOPMENT_MANIFEST).exists():
        # Crash after durable development evidence: verify task semantics, then
        # require a fresh public signed-source replay before publishing COMPLETE.
        development = _mapping(
            _strict_json_load(root / MODEL_DEVELOPMENT_MANIFEST),
            name="development recovery manifest",
        )
        _verify_development_manifest_semantics(root, tasks, development)
        full_replay = _replay_and_validate_full_signed_source(
            root,
            prepared_manifest,
            tasks,
            development,
            allow_receipt_create=True,
        )
        _write_model_complete_marker(root, development, full_replay)
        return validate_openaxes_external_model_development(
            model_config, root, replay_signed_source=True
        )

    result_index = _result_reference_index(root, tasks)
    paired_controls = verify_paired_method_controls(root, tasks)
    zero_shift, ensembles = aggregate_zero_shift_development(root, tasks)

    preparation = _mapping(prepared_manifest["preparation"], name="preparation snapshot")
    artifacts = _mapping(preparation["artifacts"], name="preparation artifacts")
    source_path = _verify_external_reference(artifacts["source"], name="official source")
    prep_config_path = _verify_external_reference(artifacts["config"], name="preparation config")
    prep_manifest_path = _verify_external_reference(artifacts["manifest"], name="preparation manifest")
    prep_run_dir = prep_manifest_path.parent
    from .openaxes_external_v1 import (
        reconstruct_openaxes_signed_shift_truth_v1,
    )
    from .openaxes_external_v1_config import load_openaxes_config

    preparation_config = load_openaxes_config(prep_config_path)
    preparation_semantic = _mapping(
        _strict_json_load(prep_manifest_path), name="preparation manifest"
    )
    validation = _run_parent_public_validator_at_recorded_git_state(
        source_path=source_path,
        preparation_config_path=prep_config_path,
        preparation_run_dir=prep_run_dir,
        preparation_manifest=preparation_semantic,
    )
    if _canonical(validation) != _canonical(preparation["validation"]):
        raise OpenAxesExternalModelV1Error(
            "preparation validation changed before signed-shift replay"
        )
    _exact_bool(
        validation["downstream_model_results_blocked_by_uncommitted_contract"],
        name="preparation downstream model block",
        expected=False,
    )
    shifted_truth, shifted_receipt = reconstruct_openaxes_signed_shift_truth_v1(
        source_path, preparation_config, prep_run_dir
    )
    _validate_signed_shift_reconstruction(
        shifted_truth, shifted_receipt, preparation_semantic
    )
    sensitivity = aggregate_alignment_sensitivity(
        ensembles, tasks, shifted_truth, shifted_receipt
    )
    development = {
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "config_sha256": OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
        "stage": "development_complete_before_any_external_outcome_reinterpretation",
        "synthetic_test_only": False,
        "prepared_manifest": _reference(
            root / MODEL_PREPARED_MANIFEST, relative=MODEL_PREPARED_MANIFEST
        ),
        "scientific_fit_count": 336,
        "outer_evaluation_count": 336,
        "result_index": result_index,
        "paired_controls": paired_controls,
        "zero_shift": zero_shift,
        "alignment_sensitivity_report_only": sensitivity,
        "claim_boundary": {
            "secondary_objective_portability_development_benchmark": True,
            "independent_so101_confirmation": False,
            "actual_sensor_outage": False,
            "synthetic_one_imu_masks_on_measured_complete_streams": True,
            "label_assisted_offline_transductive_acausal_alignment": True,
            "first_causal_provable_or_regret_claim": False,
            "stress_or_sensitivity_selected_model": False,
            "challenge_access": False,
        },
    }
    _atomic_json(root / MODEL_DEVELOPMENT_MANIFEST, development, exclusive=True)
    expected_full_replay = _expected_full_signed_source_replay_receipt(
        root,
        prepared_manifest,
        development,
        shifted_truth,
        shifted_receipt,
    )
    full_replay = _freeze_or_validate_full_signed_source_replay_receipt(
        root, expected_full_replay, allow_create=True
    )
    _write_model_complete_marker(root, development, full_replay)
    return validate_openaxes_external_model_development(
        model_config, root, replay_signed_source=True
    )


def _verify_alignment_sensitivity_semantics(
    sensitivity_value: Any,
    tasks: Sequence[TaskSpec],
    ensembles: Mapping[tuple[str, str], pd.DataFrame],
) -> None:
    sensitivity = _mapping(sensitivity_value, name="alignment sensitivity")
    _keys(
        sensitivity,
        {
            "selection_use_allowed",
            "claim_gate_use_allowed",
            "methods",
            "signed_shifts_ms",
            "fold_records",
            "family_summaries",
            "fold_record_identity_sha256",
            "family_summary_sha256",
            "reconstruction_receipt",
            "shifted_truth_persisted_or_redistributed",
            "sensitivity_semantic_sha256",
        },
        name="alignment sensitivity",
    )
    for key in (
        "selection_use_allowed",
        "claim_gate_use_allowed",
        "shifted_truth_persisted_or_redistributed",
    ):
        _exact_bool(sensitivity[key], name=f"shift {key}", expected=False)
    if sensitivity["methods"] != ["masked_erm", "paired_pord"] or sensitivity[
        "signed_shifts_ms"
    ] != list(OPENAXES_MODEL_SHIFT_MS):
        raise OpenAxesExternalModelV1Error("shift report method/shift set changed")

    fold_lookup = {task.fold_id: task for task in tasks}
    if len(fold_lookup) != 14:
        raise OpenAxesExternalModelV1Error("shift replay fold identity changed")
    expected_identities = [
        (method_id, fold_id, shift_ms)
        for method_id in ("masked_erm", "paired_pord")
        for fold_id in sorted(fold_lookup)
        for shift_ms in OPENAXES_MODEL_SHIFT_MS
    ]
    records_raw = sensitivity["fold_records"]
    if type(records_raw) is not list or len(records_raw) != 168:
        raise OpenAxesExternalModelV1Error("shift report record count changed")
    records: list[dict[str, Any]] = []
    method_support: dict[tuple[str, int], tuple[int, tuple[str, ...], str]] = {}
    identity_receipts: list[dict[str, Any]] = []
    for index, (record_raw, expected_identity) in enumerate(
        zip(records_raw, expected_identities, strict=True)
    ):
        record = _mapping(record_raw, name=f"shift record {index}")
        _keys(
            record,
            {
                "method_id",
                "fold_id",
                "family",
                "shift_ms",
                "base_uid_rows",
                "base_uids",
                "base_uid_sha256",
                "outage_context_count",
                "rmse3d_m",
                "zero_same_uid_rmse3d_m",
                "delta_rmse3d_m_shift_minus_zero",
                "robust_J_m",
                "zero_same_uid_robust_J_m",
                "delta_robust_J_m_shift_minus_zero",
            },
            name=f"shift record {index}",
        )
        identity = (record["method_id"], record["fold_id"], record["shift_ms"])
        if identity != expected_identity:
            raise OpenAxesExternalModelV1Error(
                "shift record method/fold/shift order changed"
            )
        task = fold_lookup[record["fold_id"]]
        _exact_str(
            record["family"], name="shift record family", expected=task.family
        )
        rows = _exact_int(record["base_uid_rows"], name="shift base rows")
        if rows <= 0:
            raise OpenAxesExternalModelV1Error("shift base UID count is not positive")
        base_uids_raw = record["base_uids"]
        if (
            type(base_uids_raw) is not list
            or any(type(uid) is not str or not uid for uid in base_uids_raw)
            or base_uids_raw != sorted(base_uids_raw)
            or len(base_uids_raw) != len(set(base_uids_raw))
            or rows != len(base_uids_raw)
        ):
            raise OpenAxesExternalModelV1Error(
                "shift base UID support list changed"
            )
        base_uids = tuple(base_uids_raw)
        uid_hash = _hex64(record["base_uid_sha256"], name="shift base UID hash")
        if uid_hash != _canonical_sha256(list(base_uids)):
            raise OpenAxesExternalModelV1Error(
                "shift base UID support/identity changed"
            )
        contexts = _exact_int(
            record["outage_context_count"], name="shift outage context count"
        )
        if contexts != 4 * len(task.test_scenarios):
            raise OpenAxesExternalModelV1Error("shift outage context count changed")
        support_key = (record["fold_id"], record["shift_ms"])
        support = (rows, base_uids, uid_hash)
        if support_key in method_support and method_support[support_key] != support:
            raise OpenAxesExternalModelV1Error(
                "shift UID support differs between reported methods"
            )
        method_support[support_key] = support
        numeric = {
            key: _exact_float(record[key], name=f"shift record {key}")
            for key in (
                "rmse3d_m",
                "zero_same_uid_rmse3d_m",
                "delta_rmse3d_m_shift_minus_zero",
                "robust_J_m",
                "zero_same_uid_robust_J_m",
                "delta_robust_J_m_shift_minus_zero",
            )
        }
        if numeric["rmse3d_m"] < 0.0 or numeric["zero_same_uid_rmse3d_m"] < 0.0:
            raise OpenAxesExternalModelV1Error("shift RMSE is negative")
        if numeric["robust_J_m"] < 0.0 or numeric["zero_same_uid_robust_J_m"] < 0.0:
            raise OpenAxesExternalModelV1Error("shift robust J is negative")
        if numeric["delta_rmse3d_m_shift_minus_zero"] != (
            numeric["rmse3d_m"] - numeric["zero_same_uid_rmse3d_m"]
        ):
            raise OpenAxesExternalModelV1Error("shift RMSE delta arithmetic changed")
        if numeric["delta_robust_J_m_shift_minus_zero"] != (
            numeric["robust_J_m"] - numeric["zero_same_uid_robust_J_m"]
        ):
            raise OpenAxesExternalModelV1Error("shift robust-J delta arithmetic changed")

        ensemble = ensembles[(record["method_id"], record["fold_id"])]
        outage = ensemble[ensemble["view_id"] != "full"]
        ensemble_uids = tuple(sorted(set(outage["row_uid"])))
        if base_uids != ensemble_uids:
            raise OpenAxesExternalModelV1Error(
                "shift base UID support/identity changed"
            )
        zero_frame = outage[outage["row_uid"].isin(base_uids)]
        zero_metrics = _context_metrics(zero_frame, include_full=False)
        zero_error = zero_frame[
            ["prediction_x_m", "prediction_y_m", "prediction_z_m"]
        ].to_numpy() - zero_frame[["tcp_x_m", "tcp_y_m", "tcp_z_m"]].to_numpy()
        zero_rmse = float(
            np.sqrt(np.mean(np.sum(np.square(zero_error), axis=1)))
        )
        if (
            numeric["zero_same_uid_rmse3d_m"] != zero_rmse
            or numeric["zero_same_uid_robust_J_m"]
            != zero_metrics["robust_J_m"]
        ):
            raise OpenAxesExternalModelV1Error(
                "shift zero-support metric replay changed"
            )
        identity_receipts.append(
            {
                "method_id": record["method_id"],
                "fold_id": record["fold_id"],
                "family": record["family"],
                "shift_ms": record["shift_ms"],
                "base_uid_rows": rows,
                "base_uids": list(base_uids),
                "base_uid_sha256": uid_hash,
                "outage_context_count": contexts,
            }
        )
        records.append(deepcopy(dict(record)))
    if _hex64(
        sensitivity["fold_record_identity_sha256"],
        name="shift fold-record identity hash",
    ) != _canonical_sha256(identity_receipts):
        raise OpenAxesExternalModelV1Error("shift fold-record identity hash changed")
    summaries = _build_sensitivity_family_summaries(records)
    if _canonical(sensitivity["family_summaries"]) != _canonical(summaries):
        raise OpenAxesExternalModelV1Error(
            "shift family summary semantic replay changed"
        )
    if _hex64(
        sensitivity["family_summary_sha256"], name="shift family summary hash"
    ) != _canonical_sha256(summaries):
        raise OpenAxesExternalModelV1Error("shift family summary hash changed")
    receipt = _mapping(
        sensitivity["reconstruction_receipt"], name="shift reconstruction receipt"
    )
    _exact_str(
        receipt.get("contract"),
        name="shift reconstruction contract",
        expected="openaxes.external.signed_shift_truth.v1",
    )
    scope = _mapping(receipt.get("scope"), name="shift reconstruction scope")
    _exact_bool(
        scope.get("metrics_computed"),
        name="parent reconstruction metrics boundary",
        expected=False,
    )
    base = {
        key: sensitivity[key]
        for key in sensitivity
        if key != "sensitivity_semantic_sha256"
    }
    if _hex64(
        sensitivity["sensitivity_semantic_sha256"],
        name="shift semantic hash",
    ) != _canonical_sha256(base):
        raise OpenAxesExternalModelV1Error("shift semantic hash changed")


def _verify_development_manifest_semantics(
    root: Path, tasks: Sequence[TaskSpec], development: Mapping[str, Any]
) -> None:
    _keys(
        development,
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
        },
        name="development manifest",
    )
    _exact_str(development["contract"], name="development contract", expected=OPENAXES_EXTERNAL_MODEL_V1_CONTRACT)
    _exact_str(
        development["config_sha256"],
        name="development config hash",
        expected=OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
    )
    _exact_str(
        development["stage"],
        name="development stage",
        expected="development_complete_before_any_external_outcome_reinterpretation",
    )
    _exact_bool(development["synthetic_test_only"], name="development synthetic marker", expected=False)
    _exact_int(development["scientific_fit_count"], name="development fits", expected=336)
    _exact_int(development["outer_evaluation_count"], name="development outer evals", expected=336)
    development_prepared_path = _verify_reference(
        root,
        development["prepared_manifest"],
        name="development prepared reference",
    )
    if development_prepared_path != root / MODEL_PREPARED_MANIFEST:
        raise OpenAxesExternalModelV1Error(
            "development prepared-manifest path changed"
        )
    result_index = _mapping(development["result_index"], name="development result index")
    if set(result_index) != {task.task_id for task in tasks}:
        raise OpenAxesExternalModelV1Error("development result task set changed")
    for task in tasks:
        path = _verify_reference(root, result_index[task.task_id], name=f"result {task.task_id}")
        if path != _task_result_dir(root, task) / "result.json":
            raise OpenAxesExternalModelV1Error("development result path changed")
        _verify_completed_result(root, task)
    observed_controls = verify_paired_method_controls(root, tasks)
    if _canonical(observed_controls) != _canonical(development["paired_controls"]):
        raise OpenAxesExternalModelV1Error("paired-control semantic replay changed")
    observed_zero, ensembles = aggregate_zero_shift_development(root, tasks)
    if _canonical(observed_zero) != _canonical(development["zero_shift"]):
        raise OpenAxesExternalModelV1Error("zero-shift development semantic replay changed")
    _verify_alignment_sensitivity_semantics(
        development["alignment_sensitivity_report_only"], tasks, ensembles
    )
    claim = _mapping(development["claim_boundary"], name="development claim boundary")
    expected_claim = {
        "secondary_objective_portability_development_benchmark": True,
        "independent_so101_confirmation": False,
        "actual_sensor_outage": False,
        "synthetic_one_imu_masks_on_measured_complete_streams": True,
        "label_assisted_offline_transductive_acausal_alignment": True,
        "first_causal_provable_or_regret_claim": False,
        "stress_or_sensitivity_selected_model": False,
        "challenge_access": False,
    }
    if dict(claim) != expected_claim or any(type(value) is not bool for value in claim.values()):
        raise OpenAxesExternalModelV1Error("development claim boundary changed")


def _verify_complete_semantic_hashes(
    complete: Mapping[str, Any], development: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    zero_shift = _mapping(development.get("zero_shift"), name="complete zero shift")
    selected = _mapping(zero_shift.get("selection"), name="complete selection")
    gate = _mapping(zero_shift.get("directional_gate"), name="complete directional gate")
    sensitivity = _mapping(
        development.get("alignment_sensitivity_report_only"),
        name="complete alignment sensitivity",
    )
    _exact_str(
        complete.get("selected_method_id"),
        name="complete selected method",
        expected=selected["selected_method_id"],
    )
    _exact_str(
        complete.get("selection_sha256"),
        name="complete selection hash",
        expected=_canonical_sha256(selected),
    )
    _exact_str(
        complete.get("zero_shift_sha256"),
        name="complete zero-shift hash",
        expected=_canonical_sha256(zero_shift),
    )
    _exact_bool(
        complete.get("directional_gate_passes"),
        name="complete directional gate outcome",
        expected=gate["passes"],
    )
    _exact_str(
        complete.get("directional_gate_sha256"),
        name="complete directional gate hash",
        expected=_canonical_sha256(gate),
    )
    _exact_str(
        complete.get("alignment_sensitivity_sha256"),
        name="complete alignment sensitivity hash",
        expected=_canonical_sha256(sensitivity),
    )
    return selected, gate


def _validate_complete_execution_cardinality(
    root: Path, tasks: Sequence[TaskSpec]
) -> None:
    _validate_quarantine_evidence(root, tasks, require_empty=True)
    expected_task_ids = {task.task_id for task in tasks}
    attempt_names = {
        path.stem for path in (root / MODEL_ATTEMPT_LEDGER_DIR).iterdir()
    }
    roster_names = {
        path.stem for path in (root / MODEL_ATTEMPT_ROSTER_DIR).iterdir()
    }
    ledger_names = {path.name for path in (root / MODEL_LEDGERS_DIR).iterdir()}
    result_names = {path.name for path in (root / MODEL_RESULTS_DIR).iterdir()}
    if (
        len(tasks) != 336
        or attempt_names != expected_task_ids
        or roster_names != expected_task_ids
        or ledger_names != expected_task_ids
        or result_names != expected_task_ids
    ):
        raise OpenAxesExternalModelV1Error(
            "COMPLETE requires exactly 336 roster/attempt/ledger/result task identities"
        )
    for task in tasks:
        attempt = _verify_task_attempt_ledger(root, task)
        phases = _verify_phase_chain(root, task)
        if attempt is None or len(phases) != len(OPENAXES_MODEL_PHASES):
            raise OpenAxesExternalModelV1Error(
                "COMPLETE task attempt/phase cardinality changed"
            )
        if (
            attempt["attempt_id"] != phases[0]["attempt_id"]
            or _canonical(attempt["binding"])
            != _canonical(phases[0]["execution_binding"])
        ):
            raise OpenAxesExternalModelV1Error(
                "COMPLETE task binding lineage changed"
            )
    prepared = _mapping(
        _strict_json_load(root / MODEL_PREPARED_MANIFEST),
        name="COMPLETE prepared manifest",
    )
    session = _validate_execution_session(
        root,
        prepared,
        tasks,
        _strict_json_load(_execution_session_path(root)),
    )
    _validate_latest_execution_launch_terminal(root, session)


def validate_openaxes_external_model_development(
    model_config: Mapping[str, Any],
    run_dir: str | Path,
    *,
    replay_signed_source: bool = True,
) -> dict[str, Any]:
    """Validate completed evidence with mandatory canonical signed-source replay."""

    if type(replay_signed_source) is not bool:
        raise OpenAxesExternalModelV1Error("signed-source replay marker is not exact")
    if not replay_signed_source:
        raise OpenAxesExternalModelV1Error(
            "production completed validation requires signed-source replay"
        )
    root = Path(run_dir).resolve()
    validate_openaxes_external_model_prepared_run(
        model_config, root, allow_synthetic_test_only=False
    )
    if not (root / MODEL_DEVELOPMENT_MANIFEST).is_file() or not (
        root / MODEL_COMPLETE_MARKER
    ).is_file():
        raise OpenAxesExternalModelV1Error("completed development markers are missing")
    prepared = _mapping(
        _strict_json_load(root / MODEL_PREPARED_MANIFEST), name="prepared manifest"
    )
    tasks = _load_task_index(root, prepared)
    _validate_complete_execution_cardinality(root, tasks)
    development = _mapping(
        _strict_json_load(root / MODEL_DEVELOPMENT_MANIFEST), name="development manifest"
    )
    _verify_development_manifest_semantics(root, tasks, development)
    complete = _mapping(
        _strict_json_load(root / MODEL_COMPLETE_MARKER), name="development COMPLETE"
    )
    _keys(
        complete,
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
        },
        name="development COMPLETE",
    )
    _exact_str(complete["contract"], name="complete contract", expected=OPENAXES_EXTERNAL_MODEL_V1_CONTRACT)
    _exact_str(complete["config_sha256"], name="complete config", expected=OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256)
    complete_prepared_path = _verify_reference(
        root, complete["prepared_manifest"], name="complete prepared reference"
    )
    complete_development_path = _verify_reference(
        root,
        complete["development_manifest"],
        name="complete development reference",
    )
    if (
        complete_prepared_path != root / MODEL_PREPARED_MANIFEST
        or complete_development_path != root / MODEL_DEVELOPMENT_MANIFEST
    ):
        raise OpenAxesExternalModelV1Error("complete manifest path changed")
    _exact_int(complete["scientific_fits"], name="complete fits", expected=336)
    _exact_int(complete["outer_evaluations"], name="complete evals", expected=336)
    selected, gate = _verify_complete_semantic_hashes(complete, development)
    _exact_bool(
        complete["full_signed_source_replay_performed"],
        name="complete full signed-source replay",
        expected=True,
    )
    _exact_bool(
        complete["production_validation_requires_signed_source_replay"],
        name="complete signed-source replay requirement",
        expected=True,
    )
    for key in ("stress_or_sensitivity_used_for_selection", "challenge_access_performed"):
        _exact_bool(complete[key], name=f"complete {key}", expected=False)
    _exact_bool(complete["immutable"], name="complete immutable", expected=True)

    full_replay = _replay_and_validate_full_signed_source(
        root,
        prepared,
        tasks,
        development,
        allow_receipt_create=False,
    )
    full_replay_path = _verify_reference(
        root,
        complete["full_signed_source_replay"],
        name="complete full signed-source replay reference",
    )
    if full_replay_path != _full_signed_source_replay_path(root):
        raise OpenAxesExternalModelV1Error(
            "complete full signed-source replay path changed"
        )
    _exact_str(
        complete["full_signed_source_replay_sha256"],
        name="complete full signed-source replay hash",
        expected=_canonical_sha256(full_replay),
    )
    return {
        "status": "ok",
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "scientific_fits": 336,
        "outer_evaluations": 336,
        "selected_method_id": selected["selected_method_id"],
        "directional_gate_passes": gate["passes"],
        "primary_selection_folds": 8,
        "stress_report_only_folds": 6,
        "signed_shift_report_only_values": 6,
        "challenge_access_performed": False,
        "independent_so101_confirmation": False,
        "actual_sensor_outage": False,
        "semantic_replay": True,
        "signed_source_replay": True,
    }


def doctor_openaxes_external_model_v1(
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Perform a source/config closure check without data, CUDA, or metrics."""

    validate_openaxes_external_model_config(model_config)
    workspace = _workspace_from_model_config(model_config)
    sources = _model_source_receipts(workspace)
    if openaxes_external_model_config_sha256(model_config) != OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256:
        raise OpenAxesExternalModelV1Error("doctor observed a model config digest drift")
    from . import openaxes_external_v1 as preparation_module

    public_reconstructor = getattr(
        preparation_module, "reconstruct_openaxes_signed_shift_truth_v1", None
    )
    if not callable(public_reconstructor):
        raise OpenAxesExternalModelV1Error(
            "public signed-shift reconstruction API is unavailable"
        )
    experiment_source = (
        workspace
        / "src/imu_vive_learning/openaxes_external_model_v1_experiment.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(experiment_source)
    private_preparation_imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.endswith("openaxes_external_v1")
        for alias in node.names
        if alias.name.startswith("_")
    ]
    challenge_symbols = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "challenge" in node.name.lower()
    ]
    if private_preparation_imports or challenge_symbols:
        raise OpenAxesExternalModelV1Error(
            "model source crossed preparation-private/challenge boundary"
        )
    return {
        "status": "ok",
        "contract": OPENAXES_EXTERNAL_MODEL_V1_CONTRACT,
        "config_sha256": OPENAXES_EXTERNAL_MODEL_V1_CONFIG_SHA256,
        "source_files": len(sources),
        "source_set_sha256": _canonical_sha256(sources),
        "public_signed_shift_api": True,
        "private_preparation_helpers_imported": False,
        "challenge_route_present": False,
        "data_read": False,
        "cuda_used": False,
        "metrics_computed": False,
    }
