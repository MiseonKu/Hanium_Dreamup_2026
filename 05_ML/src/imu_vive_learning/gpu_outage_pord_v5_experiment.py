"""End-to-end, sealed-label orchestration for the PORD-v5 experiment.

PORD-v5 reuses the authenticated same-session rows frozen by completed v3 and
development-only v4.  Under the frozen single-controller assumption it is the
protocol owner of the one-pass challenge gate, but there is no machine-wide
mutex against a concurrently launched parent finalizer.  It cannot open the
gate until all preregistered development claims pass and an eligible novel
candidate is frozen.  The runner never collects new data and never refits a
final model: its final predictor is the 12-checkpoint fold/seed ensemble.

The public pure helpers in this module intentionally make the scientific grid,
v3 J score, claim gates, and selection policy independently testable on CPU.
The filesystem runner adds closed-world receipts, source/environment/GPU
bindings, non-deleting quarantine, and fail-closed label access.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sys
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
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
from .gpu_outage_gbdt_v4_experiment import (
    GBDT_V4_SOURCE_FILES,
    validate_gbdt_v4_development_run,
)
from .gpu_outage_experiment import (
    GPUOutageExperimentError,
    load_outage_checkpoint,
    outage_development_tasks,
    validate_outage_development_run,
)
from .gpu_outage_pord_v5_config import (
    PORD_V5_CANDIDATE_IDS,
    PORD_V5_CHALLENGE_SEGMENTS,
    PORD_V5_CONTRACT,
    PORD_V5_DEVELOPMENT_SEGMENTS,
    PORD_V5_FIT_COUNT,
    PORD_V5_FULL_CONTEXT_COUNT,
    PORD_V5_OUTAGE_CONTEXT_COUNT,
    PORD_V5_PRIMARY_OUTAGE_CONTEXT_COUNT,
    PORD_V5_PAIRED_SEEDS,
    PORD_V5_PRIMARY_BACKBONE_ID,
    PORD_V5_PRIMARY_CANDIDATE_IDS,
    PORD_V5_PRIMARY_METHOD_IDS,
    PORD_V5_TRANSFER_BACKBONE_ID,
    PORD_V5_TRANSFER_CANDIDATE_IDS,
    PORD_V5_TRANSFER_METHOD_IDS,
    PORD_V5_TRANSFER_OUTAGE_CONTEXT_COUNT,
    PORD_V5_ROW_EXCESS_SUMMARY_COUNT,
    load_pord_v5_config,
    pord_v5_candidate_rows,
    pord_v5_config_sha256,
    public_pord_v5_config,
    validate_pord_v5_config,
)
from .gpu_outage_pord_v5_training import (
    PORD_CHECKPOINT_CRITERION,
    PORD_CHECKPOINT_SELECTION_SCOPE,
    PORD_EPOCH_BUDGET,
    PORD_PRIMARY_METHOD_IDS as PORD_TRAINING_PRIMARY_METHOD_IDS,
    PORD_TRANSFER_METHOD_IDS as PORD_TRAINING_TRANSFER_METHOD_IDS,
    array_sha256,
    build_constant_availability_five_views,
    build_exact_five_views,
    fixed_within_segment_circular_derangement,
    build_pord_backbone,
    evaluate_pord_checkpoint_once,
    evaluate_pord_outer_predictions_once,
    full_sensor_numeric_gate,
    initial_pord_state_sha256,
    predict_pord_checkpoint_outer,
    state_dict_sha256,
    train_pord_fold_seed,
    verify_pord_outer_evaluation,
    verify_pord_result_hash,
)

if (
    tuple(PORD_TRAINING_PRIMARY_METHOD_IDS) != tuple(PORD_V5_PRIMARY_METHOD_IDS)
    or tuple(PORD_TRAINING_TRANSFER_METHOD_IDS)
    != tuple(PORD_V5_TRANSFER_METHOD_IDS)
):  # import-time source-closure guard; no Torch import occurs here
    raise RuntimeError("PORD-v5 config/training closed-world method grids differ")


PORD_V5_RUN_CONTRACT = "alphaf.endpoint_regression.gpu_outage_pord_run.v5"
PORD_V5_PARENT_FREEZE_CONTRACT = (
    "alphaf.endpoint_regression.gpu_outage_pord_parent_freeze.v5"
)
PORD_V5_TASK_EXECUTION_CONTRACT = (
    "alphaf.endpoint_regression.gpu_outage_pord_task_execution.v5"
)
PORD_V5_TASK_COMPLETION_CONTRACT = (
    "alphaf.endpoint_regression.gpu_outage_pord_task_completion.v5"
)
PORD_V5_TASK_RESULT_CONTRACT = (
    "alphaf.endpoint_regression.gpu_outage_pord_task_result.v5"
)
PORD_V5_CHALLENGE_FREEZE_CONTRACT = (
    "alphaf.endpoint_regression.gpu_outage_pord_challenge_freeze.v5"
)
PORD_V5_CHALLENGE_ACCESS_CONTRACT = (
    "alphaf.endpoint_regression.gpu_outage_pord_challenge_access.v5"
)
PORD_V5_PRODUCTION_MODE = "production_cuda_spawn"
PORD_V5_SYNTHETIC_TEST_MODE = "synthetic_test_only"
PORD_V5_TASK_PHASES = (
    "execution.started",
    "training.completed",
    "outer_prediction.completed",
    "outer_eval.started",
    "outer_eval.completed",
    "bundle.completed",
    "result",
)

PORD_V5_SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *GBDT_V4_SOURCE_FILES,
            "docs/GPU_ENDPOINT_REGRESSION_OUTAGE_V3_CONTRACT.md",
            "tests/test_gpu_outage_data.py",
            "tests/test_gpu_outage_experiment.py",
            "tests/test_gpu_outage_experiment_config.py",
            "tests/test_gpu_outage_training.py",
            "configs/endpoint_regression_gpu_outage_v5_pord.yaml",
            "docs/GPU_ENDPOINT_REGRESSION_OUTAGE_V5_PORD_CONTRACT.md",
            "src/imu_vive_learning/gpu_outage_pord_v5_config.py",
            "src/imu_vive_learning/gpu_outage_pord_v5_experiment.py",
            "src/imu_vive_learning/gpu_outage_pord_v5_training.py",
            "tests/test_gpu_outage_pord_v5_config.py",
            "tests/test_gpu_outage_pord_v5_experiment.py",
            "tests/test_gpu_outage_pord_v5_training.py",
            "requirements-gpu.lock",
            "requirements-eval.lock",
            "pyproject.toml",
        )
    )
)

_TRUE_COLUMNS = ("true_x_mm", "true_y_mm", "true_z_mm")
_PREDICTION_VIEW_NAMES = ("full", "drop0", "drop1", "drop2", "drop3")
_BARE_CUDA_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_PREDICTION_COLUMNS = tuple(
    f"{view}_pred_{axis}_mm"
    for view in _PREDICTION_VIEW_NAMES
    for axis in ("x", "y", "z")
)
_TASK_KEYS = frozenset(
    {
        "task_id",
        "candidate_id",
        "backbone_id",
        "method_id",
        "table_role",
        "held_out_segment",
        "paired_seed_index",
        "model_seed",
        "evaluation_views",
    }
)
_OUTPUT_KEYS = frozenset(
    {
        "preprocessing",
        "history",
        "checkpoint_state",
        "training",
        "outer_evaluation",
        "trial_uid",
        "segment_id",
        "target_id",
        "truth_mm",
        "predictions_mm",
    }
)
_RUNTIME_GPU_RECEIPT_KEYS = frozenset(
    {
        "execution_kind",
        "transport_bound",
        "logical_worker_id",
        "process_id",
        "physical_gpu_id",
        "uuid",
        "pci_bus_id",
        "name",
        "cuda_visible_devices",
        "visible_device_index",
    }
)
_OUTER_EVALUATION_HASH_KEYS = (
    "contract",
    "training_result_sha256",
    "selected_state_sha256",
    "backbone_id",
    "held_out_segment",
    "checkpoint_selection_scope",
    "checkpoint_fixed_before_outer_labels",
    "outer_eval_count",
    "base_outer_view_bundle_sha256",
    "outer_view_bundle_sha256",
    "availability_semantics",
    "outer_trial_uid_sha256",
    "outer_truth_raw_mm_sha256",
    "outer_prediction_raw_mm_sha256",
    "target_transform_sha256",
    "evaluated_view_names",
    "same_checkpoint_for_all_four_outages",
    "outer_full_rmse_3d_mm",
    "outer_outage_rmse_3d_mm",
)
_OUTER_EVALUATION_KEYS = frozenset(
    (*_OUTER_EVALUATION_HASH_KEYS, "outer_evaluation_sha256")
)


class GPUOutagePORDV5Error(RuntimeError):
    """Raised when v5 execution or evidence violates the frozen contract."""


def _execution_mode(*, private_test_only: bool) -> str:
    if type(private_test_only) is not bool:
        raise GPUOutagePORDV5Error("private test-only mode must be an exact bool")
    return PORD_V5_SYNTHETIC_TEST_MODE if private_test_only else PORD_V5_PRODUCTION_MODE


def _prepared_execution_mode(run_dir: str | Path) -> str:
    prepared = read_json_object(Path(run_dir) / "prepared_manifest.json")
    mode = prepared.get("execution_mode")
    if mode not in {PORD_V5_PRODUCTION_MODE, PORD_V5_SYNTHETIC_TEST_MODE}:
        raise GPUOutagePORDV5Error("prepared execution mode is missing or invalid")
    if prepared.get("synthetic_test_only") is not (
        mode == PORD_V5_SYNTHETIC_TEST_MODE
    ):
        raise GPUOutagePORDV5Error("prepared synthetic-test marker changed")
    return str(mode)


def _require_mode_authority(
    run_dir: str | Path,
    *,
    private_test_only: bool,
) -> str:
    requested = _execution_mode(private_test_only=private_test_only)
    actual = _prepared_execution_mode(run_dir)
    if requested != actual:
        raise GPUOutagePORDV5Error(
            "production and private synthetic-test execution modes cannot mix"
        )
    return actual


def _require_python_312() -> None:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise GPUOutagePORDV5Error("PORD-v5 requires CPython 3.12.x")


def _runner_workspace_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "configs" / "endpoint_regression_gpu_outage_v5_pord.yaml").is_file():
        raise GPUOutagePORDV5Error("cannot locate the PORD-v5 source workspace")
    return root


def _relative(path: str | Path, root: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError as exc:
        raise GPUOutagePORDV5Error("artifact path escapes run root") from exc


_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REFERENCE_EXTRA_SCHEMAS = frozenset(
    {
        frozenset(),
        frozenset({"rows"}),
        frozenset({"updates"}),
        frozenset({"rows", "views"}),
    }
)


def _exact_string(value: Any, *, name: str, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise GPUOutagePORDV5Error(f"{name} must be an exact string")
    return value


def _exact_sha256(value: Any, *, name: str) -> str:
    token = _exact_string(value, name=name)
    if _LOWER_SHA256.fullmatch(token) is None:
        raise GPUOutagePORDV5Error(
            f"{name} must be a 64-character lowercase hexadecimal SHA-256"
        )
    return token


def _exact_int(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise GPUOutagePORDV5Error(f"{name} must be an exact integer")
    if minimum is not None and value < minimum:
        raise GPUOutagePORDV5Error(f"{name} is below its closed-world minimum")
    if maximum is not None and value > maximum:
        raise GPUOutagePORDV5Error(f"{name} exceeds its closed-world maximum")
    return value


def _exact_finite_float(
    value: Any, *, name: str, nonnegative: bool = False
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise GPUOutagePORDV5Error(f"{name} must be an exact finite float")
    if nonnegative and value < 0.0:
        raise GPUOutagePORDV5Error(f"{name} must be non-negative")
    return value


def _safe_member(root: str | Path, relative: str) -> Path:
    token = PurePosixPath(_exact_string(relative, name="artifact path"))
    if token.is_absolute() or ".." in token.parts or not token.parts:
        raise GPUOutagePORDV5Error("unsafe relative artifact path")
    path = (Path(root).resolve() / Path(*token.parts)).resolve()
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise GPUOutagePORDV5Error("artifact path escapes run root") from exc
    return path


def _artifact_reference(
    path: str | Path, root: str | Path, **extra: Any
) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.is_file():
        raise GPUOutagePORDV5Error(f"artifact is missing: {artifact}")
    payload: dict[str, Any] = {
        "path": _relative(artifact, root),
        "bytes": int(artifact.stat().st_size),
        "sha256": sha256_file(artifact),
    }
    payload.update(extra)
    return payload


def _validate_reference_schema(
    reference: Mapping[str, Any],
) -> tuple[str, int, str]:
    if type(reference) is not dict:
        raise GPUOutagePORDV5Error("artifact reference must be an exact JSON object")
    required = {"path", "bytes", "sha256"}
    extras = frozenset(reference) - required
    if not required.issubset(reference) or extras not in _REFERENCE_EXTRA_SCHEMAS:
        raise GPUOutagePORDV5Error("artifact reference schema changed")
    relative = _exact_string(reference["path"], name="artifact reference path")
    byte_count = _exact_int(
        reference["bytes"], name="artifact reference bytes", minimum=0
    )
    digest = _exact_sha256(reference["sha256"], name="artifact reference SHA-256")
    for extra in extras:
        minimum = 1 if extra in {"rows", "views", "updates"} else 0
        _exact_int(
            reference[extra],
            name=f"artifact reference {extra}",
            minimum=minimum,
        )
    return relative, byte_count, digest


def _verify_reference(root: str | Path, reference: Mapping[str, Any]) -> Path:
    relative, byte_count, digest = _validate_reference_schema(reference)
    path = _safe_member(root, relative)
    if (
        not path.is_file()
        or byte_count != path.stat().st_size
        or digest != sha256_file(path)
    ):
        raise GPUOutagePORDV5Error(f"artifact reference changed: {path}")
    return path


def _verify_v3_parent_preprocessing_reference(
    root: str | Path, reference: Mapping[str, Any]
) -> Path:
    """Verify the exact base-only artifact schema used by v3 preprocessors."""

    if type(reference) is not dict or set(reference) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise GPUOutagePORDV5Error(
            "v3 parent preprocessing reference schema changed"
        )
    path = _verify_reference(root, reference)
    if path.suffix != ".json":
        raise GPUOutagePORDV5Error("v3 parent preprocessing artifact is not JSON")
    return path


def _verify_v3_parent_checkpoint_reference(
    root: str | Path, reference: Mapping[str, Any]
) -> Path:
    """Strictly adapt the enriched sealed-v3 checkpoint reference contract."""

    if type(reference) is not dict or set(reference) != {
        "path",
        "bytes",
        "sha256",
        "tensor_count",
    }:
        raise GPUOutagePORDV5Error("v3 parent checkpoint reference schema changed")
    relative = _exact_string(
        reference["path"], name="v3 parent checkpoint reference path"
    )
    byte_count = _exact_int(
        reference["bytes"],
        name="v3 parent checkpoint reference bytes",
        minimum=1,
    )
    digest = _exact_sha256(
        reference["sha256"], name="v3 parent checkpoint reference SHA-256"
    )
    tensor_count = _exact_int(
        reference["tensor_count"],
        name="v3 parent checkpoint tensor_count",
        minimum=1,
    )
    path = _safe_member(root, relative)
    if path.suffix != ".npz":
        raise GPUOutagePORDV5Error("v3 parent checkpoint artifact is not NPZ")
    if (
        not path.is_file()
        or byte_count != path.stat().st_size
        or digest != sha256_file(path)
    ):
        raise GPUOutagePORDV5Error(f"v3 parent checkpoint reference changed: {path}")
    try:
        state = load_outage_checkpoint(path)
    except (
        GPUOutageExperimentError,
        OSError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        raise GPUOutagePORDV5Error(
            "v3 parent checkpoint NPZ is invalid"
        ) from exc
    if (
        len(state) != tensor_count
        or any(type(name) is not str or not name for name in state)
        or any(
            value.dtype.kind not in "iufc" or not np.isfinite(value).all()
            for value in state.values()
        )
    ):
        raise GPUOutagePORDV5Error(
            "v3 parent checkpoint tensor names/count/values changed"
        )
    return path


def _verify_v3_parent_model_artifacts(
    root: str | Path, result: Mapping[str, Any]
) -> tuple[Path, Path]:
    """Bind both v3 model artifacts for development and challenge replay."""

    artifacts = result.get("artifacts")
    if type(artifacts) is not dict or not {
        "preprocessing",
        "checkpoint",
    }.issubset(artifacts):
        raise GPUOutagePORDV5Error("v3 parent model artifacts are incomplete")
    return (
        _verify_v3_parent_preprocessing_reference(
            root, artifacts["preprocessing"]
        ),
        _verify_v3_parent_checkpoint_reference(root, artifacts["checkpoint"]),
    )


def _copy_atomic(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    if not source_path.is_file():
        raise GPUOutagePORDV5Error(f"parent artifact is missing: {source_path}")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    shutil.copyfile(source_path, temporary)
    os.replace(temporary, output)
    return output


def _require_bidirectionally_disjoint_paths(*paths: Path) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise GPUOutagePORDV5Error("parent/output paths must be distinct")
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            left_contains_right = False
            right_contains_left = False
            try:
                right.relative_to(left)
                left_contains_right = True
            except ValueError:
                pass
            try:
                left.relative_to(right)
                right_contains_left = True
            except ValueError:
                pass
            if left_contains_right or right_contains_left:
                raise GPUOutagePORDV5Error(
                    "parent source and v5 output paths must be bidirectionally disjoint"
                )


def _exclusive_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Durably claim a one-time receipt with O_CREAT|O_EXCL."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GPUOutagePORDV5Error("exclusive evidence is not finite JSON") from exc
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise GPUOutagePORDV5Error(
            f"exclusive evidence path already exists: {output}"
        ) from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory_descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return output


def _save_checkpoint(path: str | Path, state: Mapping[str, Any]) -> Path:
    if not isinstance(state, Mapping) or not state:
        raise GPUOutagePORDV5Error("checkpoint state must be a non-empty mapping")
    arrays: dict[str, np.ndarray] = {}
    for name, value in state.items():
        if type(name) is not str or not name:
            raise GPUOutagePORDV5Error("checkpoint tensor names must be strings")
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
            raise GPUOutagePORDV5Error("checkpoint contains non-finite/non-numeric data")
        arrays[name] = np.array(array, copy=True)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)
    return output


def _read_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        state = {str(name): np.array(archive[name], copy=True) for name in archive.files}
    if not state or any(not np.isfinite(value).all() for value in state.values()):
        raise GPUOutagePORDV5Error("checkpoint is empty or non-finite")
    return state


_PARQUET_STRING_COLUMNS = frozenset({"trial_uid", "target_id", "candidate_id"})
_PARQUET_INT64_COLUMNS = frozenset(
    {
        "segment_id",
        "held_out_segment",
        "paired_seed_index",
        "forced_outage_index",
        "imu0_base_available",
        "imu1_link_root_available",
        "imu2_link_mid_available",
        "imu3_tip_available",
    }
)


def _expected_parquet_arrow_type(column: str) -> pa.DataType:
    if type(column) is not str or not column:
        raise GPUOutagePORDV5Error("Parquet column names must be exact strings")
    if column in _PARQUET_STRING_COLUMNS:
        return pa.string()
    if column in _PARQUET_INT64_COLUMNS:
        return pa.int64()
    return pa.float64()


def _expected_parquet_pandas_metadata(columns: Sequence[str]) -> dict[str, Any]:
    pandas_columns: list[dict[str, Any]] = []
    for column in columns:
        arrow_type = _expected_parquet_arrow_type(column)
        if arrow_type == pa.string():
            pandas_type, numpy_type = "unicode", "object"
        elif arrow_type == pa.int64():
            pandas_type = numpy_type = "int64"
        elif arrow_type == pa.float64():
            pandas_type = numpy_type = "float64"
        else:  # pragma: no cover - closed by _expected_parquet_arrow_type
            raise GPUOutagePORDV5Error("unsupported v5 Parquet Arrow type")
        pandas_columns.append(
            {
                "name": column,
                "field_name": column,
                "pandas_type": pandas_type,
                "numpy_type": numpy_type,
                "metadata": None,
            }
        )
    # atomic_write_parquet delegates to the locked pandas/pyarrow runtime.
    # Both versions are already environment-bound; retaining them here also
    # rejects a re-signed file written by a different metadata authority.
    return {
        "index_columns": [],
        "column_indexes": [],
        "columns": pandas_columns,
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": pd.__version__,
    }


def _read_exact_parquet(path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    expected = list(columns)
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    if schema.names != expected:
        raise GPUOutagePORDV5Error(f"Parquet schema/order changed: {path}")
    for field in schema:
        expected_type = _expected_parquet_arrow_type(field.name)
        if field.type != expected_type or field.nullable is not True:
            raise GPUOutagePORDV5Error(
                "Parquet Arrow dtype/nullability changed: "
                f"{path}:{field.name} expected {expected_type}/nullable, "
                f"got {field.type}/nullable={field.nullable}"
            )
        if field.metadata is not None:
            raise GPUOutagePORDV5Error(
                f"Parquet Arrow field metadata changed: {path}:{field.name}"
            )
    expected_pandas_metadata = _expected_parquet_pandas_metadata(expected)
    expected_pandas_bytes = json.dumps(expected_pandas_metadata).encode("utf-8")
    schema_metadata = schema.metadata
    if type(schema_metadata) is not dict or set(schema_metadata) != {b"pandas"}:
        raise GPUOutagePORDV5Error(f"Parquet schema metadata changed: {path}")
    pandas_bytes = schema_metadata[b"pandas"]
    try:
        parsed_pandas_metadata = json.loads(pandas_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GPUOutagePORDV5Error(
            f"Parquet pandas metadata is not canonical JSON: {path}"
        ) from exc
    if (
        type(parsed_pandas_metadata) is not dict
        or parsed_pandas_metadata != expected_pandas_metadata
        or pandas_bytes != expected_pandas_bytes
    ):
        raise GPUOutagePORDV5Error(
            f"Parquet canonical pandas metadata changed: {path}"
        )
    frame = parquet.read(columns=expected).to_pandas()
    if list(frame.columns) != expected:
        raise GPUOutagePORDV5Error(f"Parquet projection order changed: {path}")
    return frame


def pord_v5_source_payload(workspace_root: str | Path) -> dict[str, Any]:
    """Hash the complete v5 import, test, config, documentation, and lock closure."""

    root = Path(workspace_root).resolve()
    files: list[dict[str, Any]] = []
    for relative in PORD_V5_SOURCE_FILES:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise GPUOutagePORDV5Error(f"source path escapes workspace: {relative}") from exc
        if not path.is_file():
            raise GPUOutagePORDV5Error(f"PORD-v5 source file is missing: {relative}")
        files.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {"files": files, "fingerprint": stable_hash(files)}


def _validate_source_closure_payload(source: Any) -> dict[str, Any]:
    if (
        type(source) is not dict
        or set(source) != {"files", "fingerprint"}
        or type(source["files"]) is not list
    ):
        raise GPUOutagePORDV5Error("v5 source closure payload schema changed")
    seen: set[str] = set()
    for row in source["files"]:
        relative, _, _ = _validate_reference_schema(row)
        token = PurePosixPath(relative)
        if token.is_absolute() or ".." in token.parts or relative in seen:
            raise GPUOutagePORDV5Error("v5 source closure path grid changed")
        seen.add(relative)
    digest = _exact_sha256(source["fingerprint"], name="source fingerprint")
    if stable_hash(source["files"]) != digest:
        raise GPUOutagePORDV5Error("v5 source closure fingerprint changed")
    return source


def _canonical_cuda_uuid_selector(uuid: Any) -> str:
    """Map one frozen bare CUDA UUID to its exact NVIDIA selector token."""

    if (
        type(uuid) is not str
        or uuid.startswith("GPU-")
        or _BARE_CUDA_UUID_PATTERN.fullmatch(uuid) is None
    ):
        raise GPUOutagePORDV5Error(
            "CUDA identity UUID must be one canonical bare lowercase UUID"
        )
    return f"GPU-{uuid}"


def _cuda_device_identity(index: int) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or index < 0 or index >= torch.cuda.device_count():
        raise GPUOutagePORDV5Error(f"CUDA device is unavailable: {index}")
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    pci = getattr(properties, "pci_bus_id", None)
    if uuid is None:
        try:
            uuid = torch.cuda.get_device_uuid(index)
        except AttributeError as exc:
            raise GPUOutagePORDV5Error("CUDA UUID API is unavailable") from exc
    if pci is None:
        try:
            pci = torch.cuda.get_device_properties(index).pci_bus_id
        except AttributeError as exc:
            raise GPUOutagePORDV5Error("CUDA PCI identity is unavailable") from exc
    bare_uuid = str(uuid)
    _canonical_cuda_uuid_selector(bare_uuid)
    return {
        "physical_gpu_id": int(index),
        "uuid": bare_uuid,
        "pci_bus_id": str(pci),
        "name": str(properties.name),
    }


def pord_v5_environment_payload(
    gpu_ids: Sequence[int] = (0, 1, 2, 3),
) -> dict[str, Any]:
    """Bind CPython/packages and four physical GPU UUID+PCI identities."""

    _require_python_312()
    import torch

    if any(type(value) is not int for value in gpu_ids):
        raise GPUOutagePORDV5Error("GPU IDs must be exact integers without coercion")
    ids = tuple(gpu_ids)
    if len(ids) != 4 or len(set(ids)) != 4 or any(value < 0 for value in ids):
        raise GPUOutagePORDV5Error("PORD-v5 requires four distinct physical GPU IDs")
    devices = [_cuda_device_identity(index) for index in ids]
    if len({row["uuid"] for row in devices}) != 4 or len(
        {row["pci_bus_id"] for row in devices}
    ) != 4:
        raise GPUOutagePORDV5Error("GPU UUID/PCI identities must be unique")
    return {
        "python_implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_devices": devices,
    }


def _validate_gpu_devices(devices: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(devices, list) or len(devices) != 4:
        raise GPUOutagePORDV5Error("environment must bind exactly four GPUs")
    rows: list[dict[str, Any]] = []
    for raw in devices:
        if not isinstance(raw, Mapping) or set(raw) != {
            "physical_gpu_id",
            "uuid",
            "pci_bus_id",
            "name",
        }:
            raise GPUOutagePORDV5Error("GPU identity schema changed")
        if type(raw["physical_gpu_id"]) is not int or raw["physical_gpu_id"] < 0:
            raise GPUOutagePORDV5Error("GPU physical ID is invalid")
        if any(type(raw[key]) is not str or not raw[key] for key in ("uuid", "pci_bus_id", "name")):
            raise GPUOutagePORDV5Error("GPU UUID/PCI/name is invalid")
        _canonical_cuda_uuid_selector(raw["uuid"])
        rows.append(dict(raw))
    if (
        len({row["physical_gpu_id"] for row in rows}) != 4
        or len({row["uuid"] for row in rows}) != 4
        or len({row["pci_bus_id"] for row in rows}) != 4
    ):
        raise GPUOutagePORDV5Error("GPU identities are not one-to-one")
    return tuple(rows)


def _validate_live_production_environment(
    stored_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the current controller CUDA/runtime identity to equal prepare."""

    if stored_environment.get("execution_mode") != PORD_V5_PRODUCTION_MODE:
        raise GPUOutagePORDV5Error("live environment validation is production-only")
    frozen = _validate_gpu_devices(
        stored_environment.get("payload", {}).get("cuda_devices")
    )
    live = pord_v5_environment_payload(
        tuple(row["physical_gpu_id"] for row in frozen)
    )
    if live != stored_environment.get("payload"):
        raise GPUOutagePORDV5Error(
            "live production Python/package/CUDA UUID/PCI environment changed"
        )
    return live


def _process_runtime_gpu_receipt(
    worker: Mapping[str, Any], *, synthetic_test_only: bool
) -> dict[str, Any]:
    """Bind GPU identity to fixed process transport, never executor output."""

    expected_selector = _canonical_cuda_uuid_selector(worker.get("uuid"))
    selector = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selector != expected_selector:
        raise GPUOutagePORDV5Error(
            "logical worker did not inherit its canonical frozen-UUID CUDA selector"
        )
    if synthetic_test_only:
        kind = "synthetic_test_only_spawn_transport"
    else:
        visible = _cuda_device_identity(0)
        if (
            visible["uuid"] != worker.get("uuid")
            or visible["pci_bus_id"] != worker.get("pci_bus_id")
            or visible["name"] != worker.get("name")
        ):
            raise GPUOutagePORDV5Error(
                "visible cuda:0 UUID/PCI/name differs from frozen physical mapping"
            )
        kind = "production_cuda_spawn_transport"
    return {
        "execution_kind": kind,
        "transport_bound": True,
        "logical_worker_id": str(worker["logical_worker_id"]),
        "process_id": int(os.getpid()),
        "physical_gpu_id": int(worker["physical_gpu_id"]),
        "uuid": str(worker["uuid"]),
        "pci_bus_id": str(worker["pci_bus_id"]),
        "name": str(worker["name"]),
        "cuda_visible_devices": str(selector),
        "visible_device_index": 0,
    }


def pord_v5_development_tasks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return exactly 14 rows x four folds x three paired seeds = 168 fits."""

    validate_pord_v5_config(config)
    tasks: list[dict[str, Any]] = []
    for candidate in pord_v5_candidate_rows(config):
        for segment in PORD_V5_DEVELOPMENT_SEGMENTS:
            for seed_index, model_seed in enumerate(PORD_V5_PAIRED_SEEDS):
                tasks.append(
                    {
                        "task_id": (
                            f"{candidate['candidate_id']}__segment{segment}__seed{seed_index}"
                        ),
                        **candidate,
                        "held_out_segment": segment,
                        "paired_seed_index": seed_index,
                        "model_seed": model_seed,
                        "evaluation_views": ["full", "drop0", "drop1", "drop2", "drop3"],
                    }
                )
    if (
        len(tasks) != PORD_V5_FIT_COUNT
        or len({row["task_id"] for row in tasks}) != PORD_V5_FIT_COUNT
        or any(set(row) != _TASK_KEYS for row in tasks)
    ):
        raise GPUOutagePORDV5Error("PORD-v5 168-fit task grid changed")
    primary = sum(
        row["candidate_id"] in PORD_V5_PRIMARY_CANDIDATE_IDS for row in tasks
    )
    transfer = sum(
        row["candidate_id"] in PORD_V5_TRANSFER_CANDIDATE_IDS for row in tasks
    )
    if primary != 144 or transfer != 24:
        raise GPUOutagePORDV5Error("PORD-v5 primary/transfer fit allocation changed")
    return tasks


def _validate_task_row(task: Any) -> dict[str, Any]:
    if type(task) is not dict or set(task) != _TASK_KEYS:
        raise GPUOutagePORDV5Error("scientific task schema changed")
    for key in ("task_id", "candidate_id", "backbone_id", "method_id", "table_role"):
        _exact_string(task[key], name=f"task {key}")
    segment = _exact_int(task["held_out_segment"], name="task held_out_segment")
    seed_index = _exact_int(
        task["paired_seed_index"],
        name="task paired_seed_index",
        minimum=0,
        maximum=2,
    )
    model_seed = _exact_int(task["model_seed"], name="task model_seed")
    if (
        segment not in PORD_V5_DEVELOPMENT_SEGMENTS
        or model_seed != PORD_V5_PAIRED_SEEDS[seed_index]
        or type(task["evaluation_views"]) is not list
        or task["evaluation_views"] != ["full", "drop0", "drop1", "drop2", "drop3"]
        or task["candidate_id"] != f"{task['backbone_id']}__{task['method_id']}"
        or task["task_id"]
        != f"{task['candidate_id']}__segment{segment}__seed{seed_index}"
    ):
        raise GPUOutagePORDV5Error("scientific task identity changed")
    allowed = (
        task["table_role"] == "primary_controlled"
        and task["backbone_id"] == PORD_V5_PRIMARY_BACKBONE_ID
        and task["method_id"] in PORD_V5_PRIMARY_METHOD_IDS
    ) or (
        task["table_role"] == "chain_transfer"
        and task["backbone_id"] == PORD_V5_TRANSFER_BACKBONE_ID
        and task["method_id"] in PORD_V5_TRANSFER_METHOD_IDS
    )
    if not allowed:
        raise GPUOutagePORDV5Error("scientific task method/backbone role changed")
    return task


def assign_pord_v5_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    gpu_devices: Sequence[Mapping[str, Any]],
    workers_per_gpu: int,
) -> list[dict[str, Any]]:
    """Assign scientific tasks once; worker multiplicity changes only execution."""

    if type(workers_per_gpu) is not int or not 1 <= workers_per_gpu <= 3:
        raise GPUOutagePORDV5Error("workers_per_gpu must be an exact integer in [1, 3]")
    devices = _validate_gpu_devices(list(gpu_devices))
    task_rows = [dict(row) for row in tasks]
    if (
        len(task_rows) != PORD_V5_FIT_COUNT
        or any(_validate_task_row(row) is not row for row in task_rows)
        or len({row["task_id"] for row in task_rows}) != PORD_V5_FIT_COUNT
    ):
        raise GPUOutagePORDV5Error("assignment requires the exact 168 unique tasks")
    slots = [
        {
            "logical_worker_id": f"gpu{device['physical_gpu_id']}__worker{worker}",
            "worker_index_on_gpu": worker,
            **device,
        }
        for device in devices
        for worker in range(workers_per_gpu)
    ]
    assignments = [
        {"task": task, "worker": dict(slots[index % len(slots)])}
        for index, task in enumerate(task_rows)
    ]
    if [row["task"]["task_id"] for row in assignments] != [
        row["task_id"] for row in task_rows
    ]:
        raise AssertionError("task assignment duplicated or reordered scientific tasks")
    return assignments


def _spawn_slot_worker_main(
    *,
    run_dir: str,
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    worker: Mapping[str, Any],
    workers_per_gpu: int,
    task_executor: Callable[..., Mapping[str, Any]],
    resume: bool,
    synthetic_test_only: bool,
    input_queue: Any,
    output_queue: Any,
) -> None:
    """Persistent one-slot FIFO worker; CUDA visibility is inherited at spawn."""

    import traceback

    try:
        torch_preloaded = "torch" in sys.modules
        if not synthetic_test_only and torch_preloaded:
            raise GPUOutagePORDV5Error(
                "torch was imported before spawn-child CUDA bootstrap"
            )
        receipt = _process_runtime_gpu_receipt(
            worker, synthetic_test_only=synthetic_test_only
        )
        output_queue.put(
            {
                "kind": "ready",
                "logical_worker_id": worker["logical_worker_id"],
                "process_id": os.getpid(),
                "torch_preloaded_before_worker_bootstrap": torch_preloaded,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        )
        while True:
            message = input_queue.get()
            if message is None:
                break
            index = message["assignment_index"]
            task = message["task"]
            result = _execute_or_resume_task(
                run_dir=Path(run_dir),
                config=config,
                data=data,
                task=task,
                worker=worker,
                workers_per_gpu=workers_per_gpu,
                task_executor=task_executor,
                resume=resume,
                runtime_gpu_receipt=receipt,
            )
            output_queue.put(
                {
                    "kind": "result",
                    "logical_worker_id": worker["logical_worker_id"],
                    "assignment_index": index,
                    "task_id": task["task_id"],
                    "result": result,
                }
            )
        output_queue.put(
            {
                "kind": "stopped",
                "logical_worker_id": worker["logical_worker_id"],
                "process_id": os.getpid(),
            }
        )
    except BaseException as exc:  # child must report before its nonzero exit
        output_queue.put(
            {
                "kind": "fatal",
                "logical_worker_id": worker.get("logical_worker_id"),
                "process_id": os.getpid(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def _run_spawn_slot_assignments(
    *,
    run_dir: Path,
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    workers_per_gpu: int,
    task_executor: Callable[..., Mapping[str, Any]],
    resume: bool,
    synthetic_test_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run fixed per-slot FIFO queues under multiprocessing ``spawn`` only."""

    import multiprocessing as mp
    import queue

    context = mp.get_context("spawn")
    by_worker: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    worker_rows: dict[str, dict[str, Any]] = {}
    for index, assignment in enumerate(assignments):
        worker = dict(assignment["worker"])
        worker_id = str(worker["logical_worker_id"])
        worker_rows.setdefault(worker_id, worker)
        by_worker.setdefault(worker_id, []).append((index, dict(assignment["task"])))
    expected_slots = 4 * workers_per_gpu
    if len(by_worker) != expected_slots:
        raise GPUOutagePORDV5Error("spawn scheduler logical slot count changed")
    output_queue = context.Queue()
    processes: dict[str, Any] = {}
    input_queues: dict[str, Any] = {}
    previous_selector = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        for worker_id in sorted(by_worker):
            worker = worker_rows[worker_id]
            input_queue = context.Queue()
            os.environ["CUDA_VISIBLE_DEVICES"] = _canonical_cuda_uuid_selector(
                worker["uuid"]
            )
            process = context.Process(
                target=_spawn_slot_worker_main,
                kwargs={
                    "run_dir": str(run_dir),
                    "config": dict(config),
                    "data": dict(data),
                    "worker": worker,
                    "workers_per_gpu": workers_per_gpu,
                    "task_executor": task_executor,
                    "resume": resume,
                    "synthetic_test_only": synthetic_test_only,
                    "input_queue": input_queue,
                    "output_queue": output_queue,
                },
                name=f"pord-v5-{worker_id}",
            )
            process.start()
            processes[worker_id] = process
            input_queues[worker_id] = input_queue
    finally:
        if previous_selector is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_selector
    clean_exit = False
    try:
        for worker_id, queue_rows in by_worker.items():
            for index, task in queue_rows:
                input_queues[worker_id].put(
                    {"assignment_index": index, "task": task}
                )
            input_queues[worker_id].put(None)

        results: list[dict[str, Any] | None] = [None] * len(assignments)
        ready: dict[str, dict[str, Any]] = {}
        stopped: set[str] = set()
        fatal: list[dict[str, Any]] = []
        while len(stopped) < expected_slots:
            try:
                message = output_queue.get(timeout=1.0)
            except queue.Empty:
                crashed = [
                    worker_id
                    for worker_id, process in processes.items()
                    if process.exitcode not in {None, 0} and worker_id not in stopped
                ]
                if crashed:
                    raise GPUOutagePORDV5Error(
                        f"spawn worker crashed without complete transport: {crashed}"
                    )
                continue
            kind = message.get("kind")
            worker_id = str(message.get("logical_worker_id"))
            if worker_id not in processes:
                raise GPUOutagePORDV5Error("spawn message used an unknown worker")
            if kind == "ready":
                if worker_id in ready:
                    raise GPUOutagePORDV5Error("spawn worker emitted duplicate ready")
                ready[worker_id] = dict(message)
            elif kind == "result":
                index = message.get("assignment_index")
                if type(index) is not int or not 0 <= index < len(results):
                    raise GPUOutagePORDV5Error("spawn result index is invalid")
                if results[index] is not None:
                    raise GPUOutagePORDV5Error(
                        "spawn emitted duplicate scientific result"
                    )
                expected = assignments[index]
                if (
                    worker_id != expected["worker"]["logical_worker_id"]
                    or message.get("task_id") != expected["task"]["task_id"]
                ):
                    raise GPUOutagePORDV5Error(
                        "spawn result transport binding changed"
                    )
                results[index] = dict(message["result"])
            elif kind == "stopped":
                if worker_id in stopped:
                    raise GPUOutagePORDV5Error("spawn worker emitted duplicate stop")
                stopped.add(worker_id)
            elif kind == "fatal":
                fatal.append(dict(message))
            else:
                raise GPUOutagePORDV5Error("spawn worker emitted an unknown message")
            if fatal:
                raise GPUOutagePORDV5Error(
                    "spawn worker failed: "
                    + "; ".join(
                        f"{row['logical_worker_id']}: {row['error']}" for row in fatal
                    )
                )
        for process in processes.values():
            process.join(timeout=10.0)
        bad_exit = {
            worker_id: process.exitcode
            for worker_id, process in processes.items()
            if process.exitcode != 0
        }
        if (
            bad_exit
            or len(ready) != expected_slots
            or any(row is None for row in results)
        ):
            raise GPUOutagePORDV5Error(
                f"spawn scheduler incomplete (exit={bad_exit}, ready={len(ready)})"
            )
        scheduler = {
            "start_method": "spawn",
            "persistent_process_per_logical_gpu_slot": True,
            "logical_slot_count": expected_slots,
            "workers_per_gpu": workers_per_gpu,
            "max_processes_per_physical_gpu": workers_per_gpu,
            "fixed_slot_fifo_serial": True,
            "controller_restored_cuda_visible_devices": True,
            "result_order_reconstructed_by_assignment_index": True,
            "workers": [ready[key] for key in sorted(ready)],
        }
        clean_exit = True
        return [dict(row) for row in results if row is not None], scheduler
    finally:
        if not clean_exit:
            for process in processes.values():
                if process.is_alive():
                    process.terminate()
            for process in processes.values():
                process.join(timeout=10.0)
        for input_queue in input_queues.values():
            input_queue.close()
        output_queue.close()


def _finite_nonnegative(value: Any, *, name: str) -> float:
    return _exact_finite_float(value, name=name, nonnegative=True)


_SCORE_METRIC_KEYS = frozenset({"rmse_3d_mm", "p95_3d_error_mm"})
_REGRESSION_METRIC_KEYS = frozenset(
    {
        "row_count",
        "weight_mass",
        "rmse_3d_mm",
        "rmse_xyz_mm",
        "mean_3d_error_mm",
        "median_3d_error_mm",
        "p90_3d_error_mm",
        "p95_3d_error_mm",
        "bias_x_mm",
        "mae_x_mm",
        "rmse_x_mm",
        "r2_x",
        "bias_y_mm",
        "mae_y_mm",
        "rmse_y_mm",
        "r2_y",
        "bias_z_mm",
        "mae_z_mm",
        "rmse_z_mm",
        "r2_z",
    }
)
_SCORE_CONTEXT_CORE_KEYS = frozenset(
    {
        "candidate_id",
        "held_out_segment",
        "forced_outage_index",
        "metrics",
        "seed_rmse_population_std_mm",
    }
)
_SCORE_CONTEXT_PERSISTED_KEYS = _SCORE_CONTEXT_CORE_KEYS | frozenset(
    {
        "view",
        "seed_count",
        "seed_rmse_3d_mm",
        "prediction_file",
        "context_id",
    }
)
_SCORE_CONTEXT_BLEND_KEYS = _SCORE_CONTEXT_PERSISTED_KEYS | frozenset(
    {"blend_weights", "blend_space"}
)


def _validate_score_metrics(value: Any) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) not in {
        _SCORE_METRIC_KEYS,
        _REGRESSION_METRIC_KEYS,
    }:
        raise GPUOutagePORDV5Error("context metric schema changed")
    if "row_count" in value:
        _exact_int(value["row_count"], name="context metric row_count", minimum=1)
    for key, metric in value.items():
        if key == "row_count":
            continue
        _exact_finite_float(
            metric,
            name=f"context metric {key}",
            nonnegative=key not in {
                "bias_x_mm",
                "bias_y_mm",
                "bias_z_mm",
                "r2_x",
                "r2_y",
                "r2_z",
            },
        )
    return value


def score_pord_v5_candidates(
    context_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_ids: Sequence[str] = PORD_V5_CANDIDATE_IDS,
) -> dict[str, Any]:
    """Apply exact v3 A/WS/WM/J and its deterministic tie-break semantics."""

    if isinstance(candidate_ids, (str, bytes)):
        raise GPUOutagePORDV5Error("candidate IDs must be a sequence of strings")
    candidates = tuple(candidate_ids)
    if (
        not candidates
        or any(type(value) is not str or not value for value in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        raise GPUOutagePORDV5Error("candidate IDs must be non-empty and unique")
    expected = {
        (candidate, segment, outage)
        for candidate in candidates
        for segment in PORD_V5_DEVELOPMENT_SEGMENTS
        for outage in range(4)
    }
    by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    validated_metrics: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in context_rows:
        if type(row) is not dict or frozenset(row) not in {
            _SCORE_CONTEXT_CORE_KEYS,
            _SCORE_CONTEXT_PERSISTED_KEYS,
            _SCORE_CONTEXT_BLEND_KEYS,
        }:
            raise GPUOutagePORDV5Error("outage context row schema changed")
        candidate_id = _exact_string(row["candidate_id"], name="candidate_id")
        segment = _exact_int(
            row["held_out_segment"], name="held_out_segment"
        )
        outage = _exact_int(
            row["forced_outage_index"],
            name="forced_outage_index",
            minimum=0,
            maximum=3,
        )
        key = (candidate_id, segment, outage)
        if key in by_key or key not in expected:
            raise GPUOutagePORDV5Error("duplicate or unexpected outage context")
        metrics = _validate_score_metrics(row["metrics"])
        _finite_nonnegative(
            row["seed_rmse_population_std_mm"],
            name="context seed dispersion",
        )
        if frozenset(row) in {
            _SCORE_CONTEXT_PERSISTED_KEYS,
            _SCORE_CONTEXT_BLEND_KEYS,
        }:
            if (
                row["view"] != f"drop{outage}"
                or row["context_id"] != f"segment{segment}__drop{outage}"
                or row["seed_count"] != 3
                or type(row["seed_count"]) is not int
                or type(row["seed_rmse_3d_mm"]) is not list
                or len(row["seed_rmse_3d_mm"]) != 3
            ):
                raise GPUOutagePORDV5Error("persisted outage context binding changed")
            for seed_metric in row["seed_rmse_3d_mm"]:
                _finite_nonnegative(seed_metric, name="seed context RMSE")
            reference = row["prediction_file"]
            _validate_reference_schema(reference)
            if frozenset(row) == _SCORE_CONTEXT_BLEND_KEYS and (
                row["blend_weights"]
                != {
                    "frozen_v4_incumbent": 0.5,
                    "masked_mlp_64x64_relu__paired_pord": 0.5,
                }
                or any(
                    type(value) is not float
                    for value in row["blend_weights"].values()
                )
                or row["blend_space"] != "raw_vive_world_xyz_mm"
            ):
                raise GPUOutagePORDV5Error("blend outage context binding changed")
        by_key[key] = row
        validated_metrics[key] = metrics
    if set(by_key) != expected:
        raise GPUOutagePORDV5Error("missing outage context before scoring")

    trace: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = [
            _finite_nonnegative(
                validated_metrics[(candidate, segment, outage)]["rmse_3d_mm"],
                name="context RMSE",
            )
            for segment in PORD_V5_DEVELOPMENT_SEGMENTS
            for outage in range(4)
        ]
        p95 = [
            _finite_nonnegative(
                validated_metrics[(candidate, segment, outage)]["p95_3d_error_mm"],
                name="context P95",
            )
            for segment in PORD_V5_DEVELOPMENT_SEGMENTS
            for outage in range(4)
        ]
        std = [
            _finite_nonnegative(
                by_key[(candidate, segment, outage)][
                    "seed_rmse_population_std_mm"
                ],
                name="context seed dispersion",
            )
            for segment in PORD_V5_DEVELOPMENT_SEGMENTS
            for outage in range(4)
        ]
        segment_means = {
            str(segment): float(
                np.mean(
                    [
                        validated_metrics[(candidate, segment, outage)][
                            "rmse_3d_mm"
                        ]
                        for outage in range(4)
                    ]
                )
            )
            for segment in PORD_V5_DEVELOPMENT_SEGMENTS
        }
        sensor_means = {
            str(outage): float(
                np.mean(
                    [
                        validated_metrics[(candidate, segment, outage)][
                            "rmse_3d_mm"
                        ]
                        for segment in PORD_V5_DEVELOPMENT_SEGMENTS
                    ]
                )
            )
            for outage in range(4)
        }
        mean_rmse = float(np.mean(metrics))
        worst_segment = max(segment_means.values())
        worst_sensor = max(sensor_means.values())
        trace.append(
            {
                "candidate_id": candidate,
                "mean_rmse_3d_mm": mean_rmse,
                "segment_mean_rmse_3d_mm": segment_means,
                "sensor_mean_rmse_3d_mm": sensor_means,
                "worst_segment_mean_rmse_3d_mm": worst_segment,
                "worst_sensor_mean_rmse_3d_mm": worst_sensor,
                "outage_robust_j_mm": float(
                    (mean_rmse + worst_segment + worst_sensor) / 3.0
                ),
                "mean_p95_3d_mm": float(np.mean(p95)),
                "mean_seed_rmse_population_std_mm": float(np.mean(std)),
            }
        )
    tolerance = 1.0e-12
    winner = trace[0]
    tie_fields = (
        "outage_robust_j_mm",
        "mean_p95_3d_mm",
        "mean_seed_rmse_population_std_mm",
    )
    for challenger in trace[1:]:
        replace = False
        for metric in tie_fields:
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
        "exact_v3_scoring_semantics": True,
        "context_count_per_candidate": 16,
        "trace": trace,
        "selected_candidate_id": winner["candidate_id"],
        "selected_outage_robust_j_mm": winner["outage_robust_j_mm"],
        "challenge_used_for_selection": False,
    }


def row_positive_excess_summary(
    truth_mm: Any, full_prediction_mm: Any, outage_prediction_mm: Any
) -> dict[str, float]:
    """Summarize paired row 3D-error excess ``max(e_out-e_full, 0)``."""

    truth = np.asarray(truth_mm, dtype=np.float64)
    full = np.asarray(full_prediction_mm, dtype=np.float64)
    outage = np.asarray(outage_prediction_mm, dtype=np.float64)
    if (
        truth.ndim != 2
        or truth.shape[1] != 3
        or full.shape != truth.shape
        or outage.shape != truth.shape
        or len(truth) == 0
        or not np.isfinite(truth).all()
        or not np.isfinite(full).all()
        or not np.isfinite(outage).all()
    ):
        raise GPUOutagePORDV5Error("paired row excess arrays are invalid")
    full_error = np.linalg.norm(full - truth, axis=1)
    outage_error = np.linalg.norm(outage - truth, axis=1)
    positive = np.maximum(outage_error - full_error, 0.0)
    return {
        "mean_mm": float(np.mean(positive)),
        "p95_mm": float(np.percentile(positive, 95.0)),
        "max_mm": float(np.max(positive)),
        "positive_fraction": float(np.mean(positive > 0.0)),
    }


_SCORE_TRACE_KEYS = frozenset(
    {
        "candidate_id",
        "mean_rmse_3d_mm",
        "segment_mean_rmse_3d_mm",
        "sensor_mean_rmse_3d_mm",
        "worst_segment_mean_rmse_3d_mm",
        "worst_sensor_mean_rmse_3d_mm",
        "outage_robust_j_mm",
        "mean_p95_3d_mm",
        "mean_seed_rmse_population_std_mm",
    }
)
_SELECTION_TRACE_KEYS = frozenset(
    {
        "candidate_id",
        "outage_robust_j_mm",
        "mean_p95_3d_mm",
        "mean_seed_rmse_population_std_mm",
    }
)


def _validate_score_trace_row(row: Any) -> dict[str, Any]:
    if type(row) is not dict or frozenset(row) != _SCORE_TRACE_KEYS:
        raise GPUOutagePORDV5Error("score trace row schema changed")
    _exact_string(row["candidate_id"], name="score trace candidate_id")
    for key in _SCORE_TRACE_KEYS - {
        "candidate_id",
        "segment_mean_rmse_3d_mm",
        "sensor_mean_rmse_3d_mm",
    }:
        _finite_nonnegative(row[key], name=f"score trace {key}")
    for key, expected_keys in (
        (
            "segment_mean_rmse_3d_mm",
            {str(segment) for segment in PORD_V5_DEVELOPMENT_SEGMENTS},
        ),
        ("sensor_mean_rmse_3d_mm", {str(index) for index in range(4)}),
    ):
        values = row[key]
        if type(values) is not dict or set(values) != expected_keys:
            raise GPUOutagePORDV5Error(f"score trace {key} schema changed")
        for value in values.values():
            _finite_nonnegative(value, name=f"score trace {key}")
    return row


def _validate_selection_trace_row(row: Any) -> dict[str, Any]:
    if type(row) is not dict or frozenset(row) not in {
        _SELECTION_TRACE_KEYS,
        _SCORE_TRACE_KEYS,
    }:
        raise GPUOutagePORDV5Error("selection trace row schema changed")
    if frozenset(row) == _SCORE_TRACE_KEYS:
        return _validate_score_trace_row(row)
    _exact_string(row["candidate_id"], name="selection trace candidate_id")
    for key in _SELECTION_TRACE_KEYS - {"candidate_id"}:
        _finite_nonnegative(row[key], name=f"selection trace {key}")
    return row


def evaluate_pord_v5_claim_gates(
    *,
    score_trace: Sequence[Mapping[str, Any]],
    seed_score_traces: Mapping[int, Sequence[Mapping[str, Any]]],
    full_fold_rmse_by_candidate: Mapping[str, Sequence[float]],
    frozen_v3_masked_full_fold_rmse: Sequence[float],
) -> dict[str, Any]:
    """Evaluate every preregistered gate; no subset can authorize challenge use."""

    validated_trace = [_validate_score_trace_row(row) for row in score_trace]
    by_candidate = {row["candidate_id"]: row for row in validated_trace}
    if len(by_candidate) != len(validated_trace):
        raise GPUOutagePORDV5Error("claim-gate score trace has duplicate candidates")
    required = {
        f"{PORD_V5_PRIMARY_BACKBONE_ID}__{method}"
        for method in PORD_V5_PRIMARY_METHOD_IDS
    }
    if not required <= set(by_candidate):
        raise GPUOutagePORDV5Error("claim-gate score trace is incomplete")
    pord_id = f"{PORD_V5_PRIMARY_BACKBONE_ID}__paired_pord"
    erm_id = f"{PORD_V5_PRIMARY_BACKBONE_ID}__masked_erm"
    full_gradient_control_id = (
        f"{PORD_V5_PRIMARY_BACKBONE_ID}__"
        "full_gradient_only_mask_selected_control"
    )
    constant_availability_id = (
        f"{PORD_V5_PRIMARY_BACKBONE_ID}__constant_availability_erm"
    )
    pord = by_candidate[pord_id]
    erm = by_candidate[erm_id]
    strict_competitors = {
        method: f"{PORD_V5_PRIMARY_BACKBONE_ID}__{method}"
        for method in (
            "group_mean_pord",
            "paired_tail",
            "paired_interaction",
            "paired_no_stopgrad",
            "paired_wrong_pair",
            "groupdro",
            "vrex",
            "mofe",
        )
    }
    segment_flags = {
        str(segment): pord["segment_mean_rmse_3d_mm"][str(segment)]
        < erm["segment_mean_rmse_3d_mm"][str(segment)]
        for segment in PORD_V5_DEVELOPMENT_SEGMENTS
    }
    if set(seed_score_traces) != {0, 1, 2}:
        raise GPUOutagePORDV5Error("claim gates require all three paired-seed traces")
    seed_flags: dict[str, bool] = {}
    for seed_index in range(3):
        seed_rows = [
            _validate_score_trace_row(row)
            for row in seed_score_traces[seed_index]
        ]
        seed_by_candidate = {row["candidate_id"]: row for row in seed_rows}
        if len(seed_by_candidate) != len(seed_rows):
            raise GPUOutagePORDV5Error("paired-seed trace has duplicate candidates")
        if pord_id not in seed_by_candidate or erm_id not in seed_by_candidate:
            raise GPUOutagePORDV5Error("paired-seed claim trace is incomplete")
        seed_flags[str(seed_index)] = seed_by_candidate[pord_id][
            "outage_robust_j_mm"
        ] < seed_by_candidate[erm_id]["outage_robust_j_mm"]
    try:
        pord_full = full_fold_rmse_by_candidate[pord_id]
        erm_full = full_fold_rmse_by_candidate[erm_id]
    except KeyError as exc:
        raise GPUOutagePORDV5Error("full-sensor gate inputs are incomplete") from exc
    frozen_gate = full_sensor_numeric_gate(
        candidate_fold_rmse_3d_mm=pord_full,
        baseline_fold_rmse_3d_mm=frozen_v3_masked_full_fold_rmse,
        baseline_name="frozen_v3_same_backbone",
        absolute_numeric_tolerance_mm=1.0e-9,
    )
    same_v5_gate = full_sensor_numeric_gate(
        candidate_fold_rmse_3d_mm=pord_full,
        baseline_fold_rmse_3d_mm=erm_full,
        baseline_name="same_v5_backbone_masked_erm",
        absolute_numeric_tolerance_mm=0.0,
    )
    gates = {
        "mask_augmentation_masked_erm_strict_j_lower_than_full_gradient_control": erm[
            "outage_robust_j_mm"
        ]
        < by_candidate[full_gradient_control_id]["outage_robust_j_mm"],
        "strict_j_lower_than_masked_erm": pord["outage_robust_j_mm"]
        < erm["outage_robust_j_mm"],
        "strict_worst_sensor_mean_lower_than_masked_erm": pord[
            "worst_sensor_mean_rmse_3d_mm"
        ]
        < erm["worst_sensor_mean_rmse_3d_mm"],
        "strict_j_lower_than_group_mean_pord": pord["outage_robust_j_mm"]
        < by_candidate[strict_competitors["group_mean_pord"]]["outage_robust_j_mm"],
        "strict_j_lower_than_paired_tail": pord["outage_robust_j_mm"]
        < by_candidate[strict_competitors["paired_tail"]]["outage_robust_j_mm"],
        "strict_j_lower_than_paired_interaction": pord["outage_robust_j_mm"]
        < by_candidate[strict_competitors["paired_interaction"]][
            "outage_robust_j_mm"
        ],
        "strict_j_lower_than_paired_no_stopgrad": pord["outage_robust_j_mm"]
        < by_candidate[strict_competitors["paired_no_stopgrad"]][
            "outage_robust_j_mm"
        ],
        "strict_j_lower_than_wrong_pair": pord["outage_robust_j_mm"]
        < by_candidate[strict_competitors["paired_wrong_pair"]]["outage_robust_j_mm"],
        "strict_j_lower_than_groupdro": pord["outage_robust_j_mm"]
        < by_candidate[strict_competitors["groupdro"]]["outage_robust_j_mm"],
        "strict_j_lower_than_vrex": pord["outage_robust_j_mm"]
        < by_candidate[strict_competitors["vrex"]]["outage_robust_j_mm"],
        "strict_j_lower_than_mofe": pord["outage_robust_j_mm"]
        < by_candidate[strict_competitors["mofe"]]["outage_robust_j_mm"],
        "held_out_segment_mean_improvement_at_least_3_of_4": sum(
            segment_flags.values()
        )
        >= 3,
        "paired_seed_j_improvement_at_least_2_of_3": sum(seed_flags.values()) >= 2,
        "frozen_v3_same_backbone_full_gate": bool(frozen_gate["passed"]),
        "same_v5_masked_erm_full_gate": bool(same_v5_gate["passed"]),
    }
    return {
        "candidate_id": pord_id,
        "baseline_id": erm_id,
        "gates": gates,
        "all_claim_gates_passed": all(gates.values()),
        "held_out_segment_improvement": segment_flags,
        "paired_seed_j_improvement": seed_flags,
        "full_sensor_gates": {
            "frozen_v3_same_backbone": frozen_gate,
            "same_v5_backbone_masked_erm": same_v5_gate,
        },
        "constant_availability_interpretation_only": {
            "candidate_id": constant_availability_id,
            "outage_robust_j_mm": by_candidate[constant_availability_id][
                "outage_robust_j_mm"
            ],
            "masked_erm_outage_robust_j_mm": erm["outage_robust_j_mm"],
            "challenge_eligibility_gate": False,
        },
        "challenge_labels_used": False,
    }


def select_pord_v5_novel_candidate(
    *,
    claim_gate_record: Mapping[str, Any],
    incumbent_trace: Mapping[str, Any],
    pord_trace: Mapping[str, Any],
    blend_trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply claim gating, incumbent non-worsening, then v3-style tie breaks."""

    incumbent_id = "frozen_v4_incumbent"
    pord_id = f"{PORD_V5_PRIMARY_BACKBONE_ID}__paired_pord"
    blend_id = "fixed_half_v4_incumbent_half_pord"
    supplied = [incumbent_trace, pord_trace, blend_trace]
    expected_ids = (incumbent_id, pord_id, blend_id)
    validated = [_validate_selection_trace_row(row) for row in supplied]
    if tuple(row["candidate_id"] for row in validated) != expected_ids:
        raise GPUOutagePORDV5Error("novel-selection candidate identity/order changed")
    if claim_gate_record.get("all_claim_gates_passed") is not True:
        return {
            "stage": "claim_gates_failed_challenge_sealed",
            "claim_gates_passed": False,
            "incumbent_candidate_id": incumbent_id,
            "selected_candidate_id": None,
            "eligible_novel_candidates": [],
            "challenge_finalize_allowed": False,
            "challenge_used_for_selection": False,
        }
    incumbent_j = _finite_nonnegative(
        incumbent_trace.get("outage_robust_j_mm"), name="incumbent J"
    )
    incumbent_trace, pord_trace, blend_trace = validated
    novel = [pord_trace, blend_trace]
    eligible = [
        row
        for row in novel
        if _finite_nonnegative(row.get("outage_robust_j_mm"), name="novel J")
        <= incumbent_j
    ]
    if not eligible:
        return {
            "stage": "novelty_gate_failed_challenge_sealed",
            "claim_gates_passed": True,
            "incumbent_candidate_id": incumbent_id,
            "selected_candidate_id": None,
            "eligible_novel_candidates": [],
            "challenge_finalize_allowed": False,
            "next_version_allowed": "v6",
            "challenge_used_for_selection": False,
        }
    tolerance = 1.0e-12
    fields = (
        "outage_robust_j_mm",
        "mean_p95_3d_mm",
        "mean_seed_rmse_population_std_mm",
    )
    winner = eligible[0]
    for challenger in eligible[1:]:
        replace = False
        for field in fields:
            left = _finite_nonnegative(challenger.get(field), name=field)
            right = _finite_nonnegative(winner.get(field), name=field)
            if left < right - tolerance:
                replace = True
                break
            if left > right + tolerance:
                break
        if replace:
            winner = challenger
    return {
        "stage": "development_selected_challenge_labels_sealed",
        "claim_gates_passed": True,
        "incumbent_candidate_id": incumbent_id,
        "incumbent_outage_robust_j_mm": incumbent_j,
        "selected_candidate_id": winner["candidate_id"],
        "selected_outage_robust_j_mm": winner["outage_robust_j_mm"],
        "eligible_novel_candidates": [row["candidate_id"] for row in eligible],
        "challenge_finalize_allowed": True,
        "challenge_used_for_selection": False,
    }


def _snapshot_input_tables(v4_run: Path, v5_run: Path) -> dict[str, Any]:
    """Copy v4-authenticated replay and label-blind identity as fresh v5 snapshots."""

    from .gpu_outage_gbdt_v4_experiment import (
        _load_challenge_identity,
        _load_development_replay,
    )
    from .gpu_outage_data import AVAILABILITY_NAMES

    development = _load_development_replay(v4_run)
    challenge = _load_challenge_identity(v4_run)
    development_frame = pd.DataFrame(
        {
            "trial_uid": development.trial_uids.astype(str),
            "segment_id": development.segment_ids.astype(np.int64),
            "target_id": development.target_ids.astype(str),
        }
    )
    for index, name in enumerate(FUSION_COLUMNS):
        development_frame[name] = development.X[:, index].astype(np.float64)
    for index, name in enumerate(_TRUE_COLUMNS):
        development_frame[name] = development.y[:, index].astype(np.float64)
    development_path = atomic_write_parquet(
        v5_run / "inputs" / "development_replay.parquet", development_frame
    )
    development_meta = {
        "role": "v4_authenticated_development_replay_snapshot",
        "rows": int(len(development_frame)),
        "segments": list(PORD_V5_DEVELOPMENT_SEGMENTS),
        "segment_rows": [
            int(np.count_nonzero(development.segment_ids == segment))
            for segment in PORD_V5_DEVELOPMENT_SEGMENTS
        ],
        "feature_names": list(FUSION_COLUMNS),
        "label_names": list(_TRUE_COLUMNS),
        "sorted_trial_uid_sha256": hashlib.sha256(
            "\n".join(sorted(development.trial_uids.astype(str).tolist())).encode(
                "utf-8"
            )
        ).hexdigest(),
        "input_file": _artifact_reference(
            development_path, v5_run, rows=len(development_frame)
        ),
    }
    development_meta_path = atomic_write_json(
        v5_run / "inputs" / "development_replay.json", development_meta
    )

    challenge_frame = pd.DataFrame(
        {
            "trial_uid": challenge.trial_uids.astype(str),
            "segment_id": challenge.segment_ids.astype(np.int64),
            "target_id": challenge.target_ids.astype(str),
        }
    )
    for index, name in enumerate(FUSION_COLUMNS):
        challenge_frame[name] = challenge.X[:, index].astype(np.float64)
    for index, name in enumerate(AVAILABILITY_NAMES):
        challenge_frame[name] = challenge.availability[:, index].astype(np.int64)
    challenge_path = atomic_write_parquet(
        v5_run / "inputs" / "challenge_identity_inputs.parquet", challenge_frame
    )
    challenge_meta = {
        "role": "v4_label_blind_challenge_identity_snapshot",
        "rows": int(len(challenge_frame)),
        "segments": list(PORD_V5_CHALLENGE_SEGMENTS),
        "segment_rows": [
            int(np.count_nonzero(challenge.segment_ids == segment))
            for segment in PORD_V5_CHALLENGE_SEGMENTS
        ],
        "feature_names": list(FUSION_COLUMNS),
        "availability_names": list(AVAILABILITY_NAMES),
        "labels_projected": False,
        "recorded_imu1_projected": False,
        "ordered_trial_uid_sha256": challenge.ordered_trial_uid_sha256,
        "identity_file": _artifact_reference(
            challenge_path, v5_run, rows=len(challenge_frame)
        ),
    }
    challenge_meta_path = atomic_write_json(
        v5_run / "inputs" / "challenge_identity.json", challenge_meta
    )
    return {
        "development_metadata": _artifact_reference(development_meta_path, v5_run),
        "challenge_identity_metadata": _artifact_reference(challenge_meta_path, v5_run),
    }


def _snapshot_v4_incumbent_predictions(
    v4_run: Path, v5_run: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Freeze the v4 winner's 48 paired development predictions in raw mm."""

    from .gpu_outage_gbdt_v4_config import load_gbdt_v4_config
    from .gpu_outage_gbdt_v4_experiment import (
        _incumbent_seed_map,
        _load_valid_xgb_result,
        _xgb_seed_map,
        gbdt_v4_development_tasks,
    )

    config_path = v4_run / "config.resolved.json"
    if config_path.is_file():
        v4_config = read_json_object(config_path)
    else:  # pragma: no cover - only malformed parent layouts reach this branch
        v4_config = load_gbdt_v4_config(
            _runner_workspace_root()
            / "configs"
            / "endpoint_regression_gpu_outage_v4_gbdt.yaml"
        )
    selection = read_json_object(v4_run / "development" / "selection.json")
    selected_id = str(selection.get("selected_candidate_id", ""))
    allowed = {
        "v3_incumbent",
        "xgboost_mask_replicated",
        "equal_blend_incumbent_xgboost",
    }
    if selected_id not in allowed or selection.get("challenge_used_for_selection") is not False:
        raise GPUOutagePORDV5Error("frozen v4 winner identity is invalid")
    xgb_results = [
        _load_valid_xgb_result(v4_run, task)
        for task in gbdt_v4_development_tasks(v4_config)
    ]
    if any(result is None for result in xgb_results):
        raise GPUOutagePORDV5Error("v4 winner component results are incomplete")
    incumbent = _incumbent_seed_map(v4_run)
    xgb = _xgb_seed_map(v4_run, xgb_results)
    snapshots: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    identity_columns = ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS]
    for segment in PORD_V5_DEVELOPMENT_SEGMENTS:
        for outage in range(4):
            seed_frames: list[pd.DataFrame] = []
            for seed in range(3):
                left = incumbent[(segment, outage, seed)]
                right = xgb[(segment, outage, seed)]
                if not left[identity_columns].equals(right[identity_columns]):
                    raise GPUOutagePORDV5Error("v4 paired component rows are misaligned")
                if selected_id == "v3_incumbent":
                    frame = left.copy()
                elif selected_id == "xgboost_mask_replicated":
                    frame = right.copy()
                else:
                    frame = left[identity_columns].copy()
                    values = 0.5 * left.loc[:, ["pred_x_mm", "pred_y_mm", "pred_z_mm"]].to_numpy(
                        dtype=np.float64
                    )
                    values += 0.5 * right.loc[:, ["pred_x_mm", "pred_y_mm", "pred_z_mm"]].to_numpy(
                        dtype=np.float64
                    )
                    for axis, name in enumerate(("pred_x_mm", "pred_y_mm", "pred_z_mm")):
                        frame[name] = values[:, axis]
                path = atomic_write_parquet(
                    v5_run
                    / "parents"
                    / "v4_incumbent_seed_predictions"
                    / f"segment{segment}__drop{outage}__seed{seed}.parquet",
                    frame,
                )
                snapshots.append(
                    {
                        "held_out_segment": segment,
                        "forced_outage_index": outage,
                        "paired_seed_index": seed,
                        "prediction_file": _artifact_reference(path, v5_run, rows=len(frame)),
                    }
                )
                seed_frames.append(frame)
            truth = seed_frames[0].loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
            predictions = [
                frame.loc[:, ["pred_x_mm", "pred_y_mm", "pred_z_mm"]].to_numpy(
                    dtype=np.float64
                )
                for frame in seed_frames
            ]
            metrics = regression_metrics(truth, np.mean(np.stack(predictions), axis=0))
            seed_rmse = [
                float(regression_metrics(truth, prediction)["rmse_3d_mm"])
                for prediction in predictions
            ]
            contexts.append(
                {
                    "candidate_id": "frozen_v4_incumbent",
                    "held_out_segment": segment,
                    "forced_outage_index": outage,
                    "metrics": metrics,
                    "seed_rmse_population_std_mm": float(np.std(seed_rmse, ddof=0)),
                }
            )
    if len(snapshots) != 48 or len(contexts) != 16:
        raise GPUOutagePORDV5Error("v4 incumbent prediction snapshot grid changed")
    return snapshots, contexts, selected_id


def _snapshot_v3_full_baselines(
    v3_run: Path, v4_run: Path, v5_run: Path
) -> dict[str, Any]:
    """Replay v3 forced-IMU1 checkpoints with a full mask, foldwise 3-seed ensemble."""

    from .gpu_outage_gbdt_v4_experiment import (
        _load_development_replay,
        _load_v3_config,
        _preprocessor_from_metadata,
        _read_v3_results,
    )
    from .gpu_outage_training import predict_outage_state

    v3_config = _load_v3_config(v3_run / "config.resolved.json")
    results, _ = _read_v3_results(v3_run, v3_config)
    data = _load_development_replay(v4_run)
    by_task = {
        (
            str(result["task"]["candidate_id"]),
            int(result["task"]["held_out_segment"]),
            int(result["task"]["forced_outage_index"]),
            int(result["task"]["paired_seed_index"]),
        ): result
        for result in results
    }
    candidates = {
        "masked_mlp_64x64_relu": "masked_mlp",
        "chain_endpoint_only": "chain_endpoint_only",
    }
    output: dict[str, Any] = {}
    for baseline_name, v3_candidate_id in candidates.items():
        fold_rmse: list[float] = []
        snapshots: list[dict[str, Any]] = []
        for segment in PORD_V5_DEVELOPMENT_SEGMENTS:
            eval_rows = data.segment_ids == segment
            truth = data.y[eval_rows]
            seed_predictions: list[np.ndarray] = []
            for seed in range(3):
                try:
                    result = by_task[(v3_candidate_id, segment, 1, seed)]
                except KeyError as exc:
                    raise GPUOutagePORDV5Error(
                        "v3 same-backbone IMU1 checkpoint grid is incomplete"
                    ) from exc
                prep_path, checkpoint_path = _verify_v3_parent_model_artifacts(
                    v3_run, result
                )
                preprocessor = _preprocessor_from_metadata(read_json_object(prep_path))
                prediction = predict_outage_state(
                    state_dict=load_outage_checkpoint(checkpoint_path),
                    X_scaled=preprocessor.transform_features(data.X[eval_rows]).astype(
                        np.float32
                    ),
                    availability=np.ones((int(np.count_nonzero(eval_rows)), 4), dtype=np.float32),
                    preprocessor=preprocessor,
                    candidate=result["task"]["candidate"],
                    device="cpu",
                )
                frame = pd.DataFrame(
                    {
                        "trial_uid": data.trial_uids[eval_rows].astype(str),
                        "segment_id": data.segment_ids[eval_rows].astype(np.int64),
                        "target_id": data.target_ids[eval_rows].astype(str),
                        **{
                            name: truth[:, axis]
                            for axis, name in enumerate(_TRUE_COLUMNS)
                        },
                        **{
                            name: prediction[:, axis]
                            for axis, name in enumerate(
                                ("pred_x_mm", "pred_y_mm", "pred_z_mm")
                            )
                        },
                    }
                )
                path = atomic_write_parquet(
                    v5_run
                    / "parents"
                    / "v3_full_replay"
                    / baseline_name
                    / f"segment{segment}__seed{seed}.parquet",
                    frame,
                )
                snapshots.append(
                    {
                        "held_out_segment": segment,
                        "paired_seed_index": seed,
                        "source_forced_outage_index": 1,
                        "prediction_file": _artifact_reference(path, v5_run, rows=len(frame)),
                    }
                )
                seed_predictions.append(np.asarray(prediction, dtype=np.float64))
            ensemble = np.mean(np.stack(seed_predictions), axis=0)
            fold_rmse.append(float(regression_metrics(truth, ensemble)["rmse_3d_mm"]))
        output[baseline_name] = {
            "v3_candidate_id": v3_candidate_id,
            "source_forced_outage_index": 1,
            "fold_rmse_3d_mm": fold_rmse,
            "seed_prediction_count": len(snapshots),
            "seed_predictions": snapshots,
        }
    return output


def _default_parent_snapshotter(
    *, v3_run: Path, v4_run: Path, v5_run: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    del config
    inputs = _snapshot_input_tables(v4_run, v5_run)
    incumbent_predictions, incumbent_contexts, selected_id = (
        _snapshot_v4_incumbent_predictions(v4_run, v5_run)
    )
    full_baselines = _snapshot_v3_full_baselines(v3_run, v4_run, v5_run)
    return {
        "inputs": inputs,
        "v4_selected_candidate_id": selected_id,
        "v4_incumbent_seed_prediction_count": len(incumbent_predictions),
        "v4_incumbent_seed_predictions": incumbent_predictions,
        "v4_incumbent_contexts": incumbent_contexts,
        "v3_same_backbone_full_baselines": full_baselines,
        "component_recipe": {
            "v3_run_origin": str(v3_run),
            "v4_run_origin": str(v4_run),
            "v4_selected_candidate_id": selected_id,
            "challenge_prediction": "semantic_replay_of_frozen_v3_v4_components",
        },
    }


def _validate_parent_snapshot(
    snapshot: Mapping[str, Any],
    run_dir: Path,
    *,
    synthetic_test_only: bool,
) -> None:
    required = {
        "inputs",
        "v4_selected_candidate_id",
        "v4_incumbent_seed_prediction_count",
        "v4_incumbent_seed_predictions",
        "v4_incumbent_contexts",
        "v3_same_backbone_full_baselines",
        "component_recipe",
    }
    if set(snapshot) != required:
        raise GPUOutagePORDV5Error("parent snapshot schema changed")
    inputs = snapshot["inputs"]
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "development_metadata",
        "challenge_identity_metadata",
    }:
        raise GPUOutagePORDV5Error("parent input snapshot schema changed")
    for reference in inputs.values():
        _verify_reference(run_dir, reference)
    predictions = snapshot["v4_incumbent_seed_predictions"]
    contexts = snapshot["v4_incumbent_contexts"]
    if (
        not isinstance(predictions, list)
        or len(predictions) != 48
        or type(snapshot["v4_incumbent_seed_prediction_count"]) is not int
        or snapshot["v4_incumbent_seed_prediction_count"] != 48
        or not isinstance(contexts, list)
        or len(contexts) != 16
    ):
        raise GPUOutagePORDV5Error("v4 incumbent snapshot count changed")
    development = _load_development_snapshot(run_dir)
    seen: set[tuple[int, int, int]] = set()
    prediction_groups: dict[tuple[int, int], list[tuple[int, pd.DataFrame]]] = {}
    for row in predictions:
        if type(row) is not dict or set(row) != {
            "held_out_segment",
            "forced_outage_index",
            "paired_seed_index",
            "prediction_file",
        }:
            raise GPUOutagePORDV5Error("v4 incumbent seed snapshot schema changed")
        key = (
            _exact_int(row["held_out_segment"], name="parent held_out_segment"),
            _exact_int(
                row["forced_outage_index"],
                name="parent forced_outage_index",
                minimum=0,
                maximum=3,
            ),
            _exact_int(
                row["paired_seed_index"],
                name="parent paired_seed_index",
                minimum=0,
                maximum=2,
            ),
        )
        if key in seen:
            raise GPUOutagePORDV5Error("duplicate v4 incumbent seed snapshot")
        seen.add(key)
        frame = _read_exact_parquet(
            _verify_reference(run_dir, row["prediction_file"]),
            [
                "trial_uid",
                "segment_id",
                "target_id",
                *_TRUE_COLUMNS,
                "pred_x_mm",
                "pred_y_mm",
                "pred_z_mm",
            ],
        )
        eval_rows = np.asarray(development["segment_id"], dtype=np.int64) == key[0]
        numeric = frame.loc[
            :, [*_TRUE_COLUMNS, "pred_x_mm", "pred_y_mm", "pred_z_mm"]
        ].to_numpy(dtype=np.float64)
        if (
            frame["trial_uid"].astype(str).tolist()
            != np.asarray(development["trial_uid"])[eval_rows].astype(str).tolist()
            or frame["target_id"].astype(str).tolist()
            != np.asarray(development["target_id"])[eval_rows].astype(str).tolist()
            or not frame["segment_id"].eq(key[0]).all()
            or not np.array_equal(
                frame.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64),
                np.asarray(development["y"], dtype=np.float64)[eval_rows],
            )
            or not np.isfinite(numeric).all()
        ):
            raise GPUOutagePORDV5Error("v4 incumbent parent prediction changed")
        prediction_groups.setdefault((key[0], key[1]), []).append((key[2], frame))
    expected = {
        (segment, outage, seed)
        for segment in PORD_V5_DEVELOPMENT_SEGMENTS
        for outage in range(4)
        for seed in range(3)
    }
    if seen != expected:
        raise GPUOutagePORDV5Error("v4 incumbent seed snapshot grid is incomplete")
    context_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in contexts:
        if type(row) is not dict:
            raise GPUOutagePORDV5Error("v4 incumbent parent context changed type")
        key = (
            _exact_int(row.get("held_out_segment"), name="parent context segment"),
            _exact_int(
                row.get("forced_outage_index"),
                name="parent context outage",
                minimum=0,
                maximum=3,
            ),
        )
        if key in context_by_key:
            raise GPUOutagePORDV5Error("v4 incumbent parent contexts are duplicated")
        context_by_key[key] = row
    if len(context_by_key) != 16:
        raise GPUOutagePORDV5Error("v4 incumbent parent contexts are duplicated")
    for key, frames in prediction_groups.items():
        ordered = [frame for _, frame in sorted(frames, key=lambda pair: pair[0])]
        if [seed for seed, _ in sorted(frames, key=lambda pair: pair[0])] != [0, 1, 2]:
            raise GPUOutagePORDV5Error("v4 parent paired-seed grid changed")
        truth = ordered[0].loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
        seed_predictions = [
            frame.loc[:, ["pred_x_mm", "pred_y_mm", "pred_z_mm"]].to_numpy(
                dtype=np.float64
            )
            for frame in ordered
        ]
        replay_context = {
            "candidate_id": "frozen_v4_incumbent",
            "held_out_segment": key[0],
            "forced_outage_index": key[1],
            "metrics": regression_metrics(
                truth, np.mean(np.stack(seed_predictions), axis=0)
            ),
            "seed_rmse_population_std_mm": float(
                np.std(
                    [
                        regression_metrics(truth, prediction)["rmse_3d_mm"]
                        for prediction in seed_predictions
                    ],
                    ddof=0,
                )
            ),
        }
        _assert_json_close(
            context_by_key[key], replay_context, path=f"parent.context.{key}"
        )
    score_pord_v5_candidates(contexts, candidate_ids=("frozen_v4_incumbent",))
    baselines = snapshot["v3_same_backbone_full_baselines"]
    if not isinstance(baselines, Mapping) or set(baselines) != {
        "masked_mlp_64x64_relu",
        "chain_endpoint_only",
    }:
        raise GPUOutagePORDV5Error("v3 same-backbone baseline identities changed")
    expected_v3_candidates = {
        "masked_mlp_64x64_relu": "masked_mlp",
        "chain_endpoint_only": "chain_endpoint_only",
    }
    for baseline_name, baseline in baselines.items():
        values = np.asarray(baseline.get("fold_rmse_3d_mm"), dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all() or np.any(values < 0):
            raise GPUOutagePORDV5Error("v3 full baseline fold values are invalid")
        if baseline.get("source_forced_outage_index") != 1:
            raise GPUOutagePORDV5Error("v3 full baseline did not use IMU1 checkpoints")
        if synthetic_test_only:
            continue
        if (
            set(baseline)
            != {
                "v3_candidate_id",
                "source_forced_outage_index",
                "fold_rmse_3d_mm",
                "seed_prediction_count",
                "seed_predictions",
            }
            or baseline.get("v3_candidate_id")
            != expected_v3_candidates[baseline_name]
            or baseline.get("seed_prediction_count") != 12
            or not isinstance(baseline.get("seed_predictions"), list)
            or len(baseline["seed_predictions"]) != 12
        ):
            raise GPUOutagePORDV5Error("v3 full baseline provenance changed")
        fold_predictions: dict[int, list[tuple[int, pd.DataFrame]]] = {}
        baseline_seen: set[tuple[int, int]] = set()
        for row in baseline["seed_predictions"]:
            if type(row) is not dict or set(row) != {
                "held_out_segment",
                "paired_seed_index",
                "source_forced_outage_index",
                "prediction_file",
            }:
                raise GPUOutagePORDV5Error("v3 full baseline seed schema changed")
            key = (
                _exact_int(
                    row["held_out_segment"], name="v3 baseline held_out_segment"
                ),
                _exact_int(
                    row["paired_seed_index"],
                    name="v3 baseline paired_seed_index",
                    minimum=0,
                    maximum=2,
                ),
            )
            if (
                key in baseline_seen
                or row.get("source_forced_outage_index") != 1
            ):
                raise GPUOutagePORDV5Error("v3 full baseline seed grid changed")
            baseline_seen.add(key)
            frame = _read_exact_parquet(
                _verify_reference(run_dir, row["prediction_file"]),
                [
                    "trial_uid",
                    "segment_id",
                    "target_id",
                    *_TRUE_COLUMNS,
                    "pred_x_mm",
                    "pred_y_mm",
                    "pred_z_mm",
                ],
            )
            eval_rows = (
                np.asarray(development["segment_id"], dtype=np.int64) == key[0]
            )
            if (
                frame["trial_uid"].astype(str).tolist()
                != np.asarray(development["trial_uid"])[eval_rows]
                .astype(str)
                .tolist()
                or not np.array_equal(
                    frame.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64),
                    np.asarray(development["y"], dtype=np.float64)[eval_rows],
                )
            ):
                raise GPUOutagePORDV5Error("v3 full baseline identity changed")
            fold_predictions.setdefault(key[0], []).append((key[1], frame))
        expected_baseline_grid = {
            (segment, seed)
            for segment in PORD_V5_DEVELOPMENT_SEGMENTS
            for seed in range(3)
        }
        if baseline_seen != expected_baseline_grid:
            raise GPUOutagePORDV5Error("v3 full baseline grid is incomplete")
        replay_fold_rmse: list[float] = []
        for segment in PORD_V5_DEVELOPMENT_SEGMENTS:
            ordered = [
                frame
                for _, frame in sorted(
                    fold_predictions[segment], key=lambda pair: pair[0]
                )
            ]
            truth = ordered[0].loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
            ensemble = np.mean(
                np.stack(
                    [
                        frame.loc[
                            :, ["pred_x_mm", "pred_y_mm", "pred_z_mm"]
                        ].to_numpy(dtype=np.float64)
                        for frame in ordered
                    ]
                ),
                axis=0,
            )
            replay_fold_rmse.append(
                float(regression_metrics(truth, ensemble)["rmse_3d_mm"])
            )
        if not np.array_equal(values, np.asarray(replay_fold_rmse)):
            raise GPUOutagePORDV5Error("v3 full baseline metrics changed")
    recipe = snapshot["component_recipe"]
    if synthetic_test_only:
        if recipe != {"synthetic_test": True}:
            raise GPUOutagePORDV5Error("synthetic parent recipe changed")
    elif (
        not isinstance(recipe, Mapping)
        or set(recipe)
        != {
            "v3_run_origin",
            "v4_run_origin",
            "v4_selected_candidate_id",
            "challenge_prediction",
        }
        or recipe.get("v4_selected_candidate_id")
        != snapshot["v4_selected_candidate_id"]
        or recipe.get("challenge_prediction")
        != "semantic_replay_of_frozen_v3_v4_components"
    ):
        raise GPUOutagePORDV5Error("parent component recipe changed")


def prepare_pord_v5_run(
    config: Mapping[str, Any],
    *,
    v3_run_dir: str | Path,
    v4_run_dir: str | Path,
    run_dir: str | Path,
    v3_validator: Callable[[str | Path], Mapping[str, Any]] = validate_outage_development_run,
    v4_validator: Callable[[str | Path], Mapping[str, Any]] = validate_gbdt_v4_development_run,
    parent_snapshotter: Callable[..., Mapping[str, Any]] = _default_parent_snapshotter,
    source_payload: Mapping[str, Any] | None = None,
    environment_payload: Mapping[str, Any] | None = None,
    _private_test_only_mode: bool = False,
) -> dict[str, Any]:
    """Semantically validate and immutably freeze completed, still-sealed v3/v4."""

    _require_python_312()
    validate_pord_v5_config(config)
    mode = _execution_mode(private_test_only=_private_test_only_mode)
    custom_prepare_hooks = (
        v3_validator is not validate_outage_development_run
        or v4_validator is not validate_gbdt_v4_development_run
        or parent_snapshotter is not _default_parent_snapshotter
        or source_payload is not None
        or environment_payload is not None
    )
    if custom_prepare_hooks and mode != PORD_V5_SYNTHETIC_TEST_MODE:
        raise GPUOutagePORDV5Error(
            "custom validator/snapshot/source/environment hooks are private test-only"
        )
    v3_run = Path(v3_run_dir).resolve()
    v4_run = Path(v4_run_dir).resolve()
    v5_run = Path(run_dir).resolve()
    _require_bidirectionally_disjoint_paths(v3_run, v4_run, v5_run)
    for parent, name in ((v3_run, "v3"), (v4_run, "v4")):
        if (parent / "challenge" / "label_access_1.started.json").exists():
            raise GPUOutagePORDV5Error(f"{name} challenge label-access marker is present")
        if not (parent / "development_manifest.json").is_file():
            raise GPUOutagePORDV5Error(f"{name} development is not complete")
    v3_before = manifest_files(v3_run)
    v4_before = manifest_files(v4_run)
    v3_validation = dict(v3_validator(v3_run))
    v4_validation = dict(v4_validator(v4_run))
    if (
        v3_validation.get("status") != "ok"
        or v3_validation.get("stage") != "development_complete_challenge_labels_sealed"
        or v3_validation.get("challenge_labels_sealed") is not True
        or type(v3_validation.get("development_seed_fits")) is not int
        or v3_validation.get("development_seed_fits") != 240
        or type(v3_validation.get("scored_contexts")) is not int
        or v3_validation.get("scored_contexts") != 96
    ):
        raise GPUOutagePORDV5Error("v3 validator did not return the exact sealed stage")
    if (
        v4_validation.get("status") != "ok"
        or v4_validation.get("stage") != "development_complete_challenge_labels_sealed"
        or v4_validation.get("challenge_labels_sealed") is not True
        or type(v4_validation.get("xgb_unique_bundle_fits")) is not int
        or v4_validation.get("xgb_unique_bundle_fits") != 12
        or type(v4_validation.get("scored_contexts")) is not int
        or v4_validation.get("scored_contexts") != 48
        or type(v4_validation.get("challenge_label_access_passes")) is not int
        or v4_validation.get("challenge_label_access_passes") != 0
    ):
        raise GPUOutagePORDV5Error("v4 validator did not return development-only sealed stage")
    source = dict(source_payload or pord_v5_source_payload(_runner_workspace_root()))
    _validate_source_closure_payload(source)
    environment = dict(environment_payload or pord_v5_environment_payload())
    _validate_gpu_devices(environment.get("cuda_devices"))
    environment_snapshot = {
        "payload": environment,
        "fingerprint": stable_hash(environment),
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
    }
    identity = {
        "contract": PORD_V5_CONTRACT,
        "config_sha256": pord_v5_config_sha256(config),
        "v3_manifest_sha256": sha256_file(v3_run / "development_manifest.json"),
        "v4_manifest_sha256": sha256_file(v4_run / "development_manifest.json"),
        "training_source_fingerprint": source["fingerprint"],
        "environment_fingerprint": environment_snapshot["fingerprint"],
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
    }
    prepare_experiment_directory(v5_run, identity=identity, resume=False)
    atomic_write_json(v5_run / "config.resolved.json", public_pord_v5_config(config))
    source_path = atomic_write_json(v5_run / "provenance" / "source_snapshot.json", source)
    environment_path = atomic_write_json(
        v5_run / "provenance" / "environment_snapshot.json", environment_snapshot
    )
    v3_manifest_copy = _copy_atomic(
        v3_run / "development_manifest.json", v5_run / "parents" / "v3_development_manifest.json"
    )
    v4_manifest_copy = _copy_atomic(
        v4_run / "development_manifest.json", v5_run / "parents" / "v4_development_manifest.json"
    )
    v3_validation_path = atomic_write_json(
        v5_run / "parents" / "v3_validator_payload.json", v3_validation
    )
    v4_validation_path = atomic_write_json(
        v5_run / "parents" / "v4_validator_payload.json", v4_validation
    )
    snapshot = dict(
        parent_snapshotter(
            v3_run=v3_run, v4_run=v4_run, v5_run=v5_run, config=config
        )
    )
    _validate_parent_snapshot(
        snapshot,
        v5_run,
        synthetic_test_only=mode == PORD_V5_SYNTHETIC_TEST_MODE,
    )
    freeze = {
        "contract": PORD_V5_PARENT_FREEZE_CONTRACT,
        "status": "prepared_from_semantically_validated_completed_sealed_v3_v4",
        "config_sha256": pord_v5_config_sha256(config),
        "v3_run_origin": str(v3_run),
        "v4_run_origin": str(v4_run),
        "v3_manifest_sha256": identity["v3_manifest_sha256"],
        "v4_manifest_sha256": identity["v4_manifest_sha256"],
        "v3_validator_sha256": stable_hash(v3_validation),
        "v4_validator_sha256": stable_hash(v4_validation),
        "training_source_fingerprint": source["fingerprint"],
        "environment_fingerprint": environment_snapshot["fingerprint"],
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
        "parents_modified_by_prepare": False,
        "challenge_labels_sealed": True,
        "references": {
            "source_snapshot": _artifact_reference(source_path, v5_run),
            "environment_snapshot": _artifact_reference(environment_path, v5_run),
            "v3_manifest": _artifact_reference(v3_manifest_copy, v5_run),
            "v4_manifest": _artifact_reference(v4_manifest_copy, v5_run),
            "v3_validator": _artifact_reference(v3_validation_path, v5_run),
            "v4_validator": _artifact_reference(v4_validation_path, v5_run),
        },
        "snapshot": snapshot,
    }
    freeze_path = atomic_write_json(v5_run / "parent.freeze.json", freeze)
    prepared = {
        "contract": PORD_V5_RUN_CONTRACT,
        "stage": "prepared_v3_v4_frozen_challenge_labels_sealed",
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
        "config_sha256": pord_v5_config_sha256(config),
        "fit_count": PORD_V5_FIT_COUNT,
        "parent_freeze": _artifact_reference(freeze_path, v5_run),
        "challenge_labels_sealed": True,
        "files": [
            row
            for row in manifest_files(v5_run)
            if row["path"] != "prepared_manifest.json"
        ],
    }
    atomic_write_json(v5_run / "prepared_manifest.json", prepared)
    if manifest_files(v3_run) != v3_before or manifest_files(v4_run) != v4_before:
        raise GPUOutagePORDV5Error("a parent run changed while v5 prepare was reading it")
    validate_pord_v5_prepared_run(
        v5_run,
        verify_current_source=source_payload is None,
        allow_synthetic_test_only=_private_test_only_mode,
        verify_live_environment=not _private_test_only_mode,
    )
    return prepared


def validate_pord_v5_prepared_run(
    run_dir: str | Path,
    *,
    verify_current_source: bool = True,
    allow_synthetic_test_only: bool = False,
    verify_live_environment: bool = True,
) -> dict[str, Any]:
    """Validate frozen parent evidence without opening source archives or labels."""

    run = Path(run_dir).resolve()
    mode = _prepared_execution_mode(run)
    if mode == PORD_V5_SYNTHETIC_TEST_MODE and not allow_synthetic_test_only:
        raise GPUOutagePORDV5Error(
            "synthetic test-only run is rejected by the production validator"
        )
    config = read_json_object(run / "config.resolved.json")
    validate_pord_v5_config(config)
    config_sha = pord_v5_config_sha256(config)
    source = read_json_object(run / "provenance" / "source_snapshot.json")
    environment = read_json_object(run / "provenance" / "environment_snapshot.json")
    _validate_source_closure_payload(source)
    if verify_current_source:
        current = pord_v5_source_payload(_runner_workspace_root())
        if current != source:
            raise GPUOutagePORDV5Error("current v5 source differs from prepare snapshot")
    if (
        set(environment)
        != {"payload", "fingerprint", "execution_mode", "synthetic_test_only"}
        or type(environment.get("payload")) is not dict
        or stable_hash(environment.get("payload")) != environment.get("fingerprint")
        or environment.get("execution_mode") != mode
        or environment.get("synthetic_test_only") is not (
            mode == PORD_V5_SYNTHETIC_TEST_MODE
        )
    ):
        raise GPUOutagePORDV5Error("stored environment fingerprint changed")
    _exact_sha256(environment["fingerprint"], name="environment fingerprint")
    _validate_gpu_devices(environment.get("payload", {}).get("cuda_devices"))
    if mode == PORD_V5_PRODUCTION_MODE and verify_live_environment:
        _validate_live_production_environment(environment)
    freeze_path = run / "parent.freeze.json"
    freeze = read_json_object(freeze_path)
    expected_freeze_keys = {
        "contract",
        "status",
        "config_sha256",
        "v3_run_origin",
        "v4_run_origin",
        "v3_manifest_sha256",
        "v4_manifest_sha256",
        "v3_validator_sha256",
        "v4_validator_sha256",
        "training_source_fingerprint",
        "environment_fingerprint",
        "execution_mode",
        "synthetic_test_only",
        "parents_modified_by_prepare",
        "challenge_labels_sealed",
        "references",
        "snapshot",
    }
    if (
        set(freeze) != expected_freeze_keys
        or freeze.get("contract") != PORD_V5_PARENT_FREEZE_CONTRACT
        or freeze.get("status")
        != "prepared_from_semantically_validated_completed_sealed_v3_v4"
        or freeze.get("config_sha256") != config_sha
        or freeze.get("challenge_labels_sealed") is not True
        or freeze.get("parents_modified_by_prepare") is not False
        or freeze.get("training_source_fingerprint") != source["fingerprint"]
        or freeze.get("environment_fingerprint") != environment["fingerprint"]
        or freeze.get("execution_mode") != mode
        or freeze.get("synthetic_test_only") is not (
            mode == PORD_V5_SYNTHETIC_TEST_MODE
        )
    ):
        raise GPUOutagePORDV5Error("v5 parent freeze binding changed")
    for key in (
        "config_sha256",
        "v3_manifest_sha256",
        "v4_manifest_sha256",
        "v3_validator_sha256",
        "v4_validator_sha256",
        "training_source_fingerprint",
        "environment_fingerprint",
    ):
        _exact_sha256(freeze[key], name=f"parent freeze {key}")
    for key in ("v3_run_origin", "v4_run_origin"):
        _exact_string(freeze[key], name=f"parent freeze {key}")
    references = freeze.get("references")
    if type(references) is not dict or set(references) != {
        "source_snapshot",
        "environment_snapshot",
        "v3_manifest",
        "v4_manifest",
        "v3_validator",
        "v4_validator",
    }:
        raise GPUOutagePORDV5Error("parent freeze reference schema changed")
    for reference in references.values():
        _verify_reference(run, reference)
    _validate_parent_snapshot(
        freeze.get("snapshot", {}),
        run,
        synthetic_test_only=mode == PORD_V5_SYNTHETIC_TEST_MODE,
    )
    identity = read_json_object(run / "_state" / "run_identity.json")
    expected_identity = {
        "contract": PORD_V5_CONTRACT,
        "config_sha256": config_sha,
        "v3_manifest_sha256": freeze["v3_manifest_sha256"],
        "v4_manifest_sha256": freeze["v4_manifest_sha256"],
        "training_source_fingerprint": source["fingerprint"],
        "environment_fingerprint": environment["fingerprint"],
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
    }
    if identity != expected_identity:
        raise GPUOutagePORDV5Error("v5 run identity changed")
    for parent_key, manifest_key in (("v3_run_origin", "v3_manifest_sha256"), ("v4_run_origin", "v4_manifest_sha256")):
        parent = Path(_exact_string(freeze[parent_key], name=parent_key))
        if (parent / "challenge" / "label_access_1.started.json").exists():
            raise GPUOutagePORDV5Error("a frozen parent challenge marker is now present")
        if sha256_file(parent / "development_manifest.json") != freeze[manifest_key]:
            raise GPUOutagePORDV5Error("a frozen parent development manifest changed")
    prepared = read_json_object(run / "prepared_manifest.json")
    expected_files = [
        row
        for row in manifest_files(run)
        if row["path"] != "prepared_manifest.json"
        and not row["path"].startswith("development/")
        and not row["path"].startswith("challenge/")
        and not row["path"].startswith("recovery/")
        and not row["path"].startswith("runtime_auth/")
        and row["path"] != "development_manifest.json"
    ]
    if (
        set(prepared)
        != {
            "contract",
            "stage",
            "execution_mode",
            "synthetic_test_only",
            "config_sha256",
            "fit_count",
            "parent_freeze",
            "challenge_labels_sealed",
            "files",
        }
        or prepared.get("contract") != PORD_V5_RUN_CONTRACT
        or prepared.get("stage") != "prepared_v3_v4_frozen_challenge_labels_sealed"
        or prepared.get("config_sha256") != config_sha
        or type(prepared.get("fit_count")) is not int
        or prepared.get("fit_count") != PORD_V5_FIT_COUNT
        or prepared.get("parent_freeze") != _artifact_reference(freeze_path, run)
        or prepared.get("execution_mode") != mode
        or prepared.get("synthetic_test_only") is not (
            mode == PORD_V5_SYNTHETIC_TEST_MODE
        )
        or prepared.get("challenge_labels_sealed") is not True
        or type(prepared.get("files")) is not list
        or prepared.get("files") != expected_files
    ):
        raise GPUOutagePORDV5Error("v5 prepared manifest changed")
    return {
        "status": "ok",
        "stage": "prepared_v3_v4_frozen_challenge_labels_sealed",
        "config_sha256": config_sha,
        "fit_count": PORD_V5_FIT_COUNT,
        "development_rows": 6635,
        "challenge_rows_identity_only": 3295,
        "challenge_labels_sealed": True,
        "source_archives_opened_by_validator": False,
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
    }


def _load_development_snapshot(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    metadata = read_json_object(run / "inputs" / "development_replay.json")
    expected_meta_keys = {
        "role",
        "rows",
        "segments",
        "segment_rows",
        "feature_names",
        "label_names",
        "sorted_trial_uid_sha256",
        "input_file",
    }
    if (
        set(metadata) != expected_meta_keys
        or metadata.get("role") != "v4_authenticated_development_replay_snapshot"
        or metadata.get("rows") != 6635
        or tuple(metadata.get("segments", ())) != PORD_V5_DEVELOPMENT_SEGMENTS
        or tuple(metadata.get("segment_rows", ())) != (1672, 1597, 1707, 1659)
        or tuple(metadata.get("feature_names", ())) != FUSION_COLUMNS
        or tuple(metadata.get("label_names", ())) != _TRUE_COLUMNS
    ):
        raise GPUOutagePORDV5Error("development replay metadata changed")
    columns = ["trial_uid", "segment_id", "target_id", *FUSION_COLUMNS, *_TRUE_COLUMNS]
    frame = _read_exact_parquet(_verify_reference(run, metadata["input_file"]), columns)
    if len(frame) != 6635:
        raise GPUOutagePORDV5Error("development replay row count changed")
    segments = frame["segment_id"].to_numpy(dtype=np.int64)
    counts = tuple(
        int(np.count_nonzero(segments == segment))
        for segment in PORD_V5_DEVELOPMENT_SEGMENTS
    )
    uid_hash = hashlib.sha256(
        "\n".join(sorted(frame["trial_uid"].astype(str).tolist())).encode("utf-8")
    ).hexdigest()
    numeric = frame.loc[:, [*FUSION_COLUMNS, *_TRUE_COLUMNS]].to_numpy(
        dtype=np.float64
    )
    if (
        counts != (1672, 1597, 1707, 1659)
        or uid_hash != metadata["sorted_trial_uid_sha256"]
        or not np.isfinite(numeric).all()
        or frame["trial_uid"].astype(str).duplicated().any()
    ):
        raise GPUOutagePORDV5Error("development replay identity/value contract changed")
    return {
        "X": frame.loc[:, list(FUSION_COLUMNS)].to_numpy(dtype=np.float64),
        "y": frame.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64),
        "trial_uid": frame["trial_uid"].astype(str).to_numpy(),
        "segment_id": segments,
        "target_id": frame["target_id"].astype(str).to_numpy(),
    }


def _load_challenge_snapshot(run_dir: str | Path) -> dict[str, Any]:
    from .gpu_outage_data import AVAILABILITY_NAMES

    run = Path(run_dir)
    metadata = read_json_object(run / "inputs" / "challenge_identity.json")
    expected_meta_keys = {
        "role",
        "rows",
        "segments",
        "segment_rows",
        "feature_names",
        "availability_names",
        "labels_projected",
        "recorded_imu1_projected",
        "ordered_trial_uid_sha256",
        "identity_file",
    }
    if (
        set(metadata) != expected_meta_keys
        or metadata.get("role") != "v4_label_blind_challenge_identity_snapshot"
        or metadata.get("rows") != 3295
        or tuple(metadata.get("segments", ())) != PORD_V5_CHALLENGE_SEGMENTS
        or tuple(metadata.get("segment_rows", ())) != (1672, 1623)
        or tuple(metadata.get("feature_names", ())) != FUSION_COLUMNS
        or tuple(metadata.get("availability_names", ())) != tuple(AVAILABILITY_NAMES)
        or metadata.get("labels_projected") is not False
        or metadata.get("recorded_imu1_projected") is not False
    ):
        raise GPUOutagePORDV5Error("challenge identity metadata changed")
    columns = [
        "trial_uid",
        "segment_id",
        "target_id",
        *FUSION_COLUMNS,
        *AVAILABILITY_NAMES,
    ]
    frame = _read_exact_parquet(_verify_reference(run, metadata["identity_file"]), columns)
    segments = frame["segment_id"].to_numpy(dtype=np.int64)
    availability = frame.loc[:, list(AVAILABILITY_NAMES)].to_numpy(dtype=np.float64)
    expected_availability = np.tile(
        np.asarray((1.0, 0.0, 1.0, 1.0), dtype=np.float64), (3295, 1)
    )
    uid_hash = hashlib.sha256(
        "\n".join(frame["trial_uid"].astype(str).tolist()).encode("utf-8")
    ).hexdigest()
    if (
        len(frame) != 3295
        or tuple(
            int(np.count_nonzero(segments == segment))
            for segment in PORD_V5_CHALLENGE_SEGMENTS
        )
        != (1672, 1623)
        or uid_hash != metadata["ordered_trial_uid_sha256"]
        or not np.array_equal(availability, expected_availability)
        or not np.isfinite(
            frame.loc[:, list(FUSION_COLUMNS)].to_numpy(dtype=np.float64)
        ).all()
        or frame["trial_uid"].astype(str).duplicated().any()
    ):
        raise GPUOutagePORDV5Error("challenge identity/value contract changed")
    return {
        "X": frame.loc[:, list(FUSION_COLUMNS)].to_numpy(dtype=np.float64),
        "availability": availability,
        "trial_uid": frame["trial_uid"].astype(str).to_numpy(),
        "segment_id": segments,
        "target_id": frame["target_id"].astype(str).to_numpy(),
        "ordered_trial_uid_sha256": uid_hash,
    }


def _fit_v5_preprocessor(
    data: Mapping[str, Any], *, held_out_segment: int
) -> TrainOnlyPreprocessor:
    segments = np.asarray(data["segment_id"], dtype=np.int64)
    eval_rows = segments == held_out_segment
    train_rows = ~eval_rows
    if not eval_rows.any() or not train_rows.any():
        raise GPUOutagePORDV5Error("v5 fold has an empty train or holdout role")
    preprocessor = TrainOnlyPreprocessor(
        feature_names=FUSION_COLUMNS,
        scale_features=True,
        split_context=f"outage_v5_pord_segment{held_out_segment}",
    ).fit_real_train(
        np.asarray(data["X"])[train_rows],
        np.asarray(data["y"])[train_rows],
        trial_uids=np.asarray(data["trial_uid"])[train_rows].astype(str).tolist(),
        forbidden_uids=np.asarray(data["trial_uid"])[eval_rows].astype(str).tolist(),
    )
    return preprocessor


def _assemble_task_executor_output(
    *,
    task: Mapping[str, Any],
    preprocessor_metadata: Mapping[str, Any],
    core: Mapping[str, Any],
    outer: Mapping[str, Any],
    trial_uid: Sequence[Any],
    segment_id: Sequence[Any],
    target_id: Sequence[Any],
) -> dict[str, Any]:
    """Build the final task bundle from authenticated train/outer phases."""

    outer_receipt = dict(outer["receipt"])
    training = {
        "backbone_id": str(core["backbone_id"]),
        "method_id": str(core["method_id"]),
        "epoch_budget": int(core["epoch_budget"]),
        "optimizer_update_count": int(core["optimizer_update_count"]),
        "contract_epoch_budget_complete": bool(core["contract_epoch_budget_complete"]),
        "initial_state_sha256": str(core["initial_state_sha256"]),
        "selected_state_sha256": str(core["selected_state_sha256"]),
        "training_core_result_sha256": str(core["result_sha256"]),
        "training_bundle_sha256": str(core["training_bundle_sha256"]),
        "base_train_view_bundle_sha256": str(core["base_train_view_bundle_sha256"]),
        "train_view_bundle_sha256": str(core["train_view_bundle_sha256"]),
        "availability_semantics": str(core["availability_semantics"]),
        "gradient_view_semantics": str(core["gradient_view_semantics"]),
        "pre_registered_wrong_pair_map_sha256": str(
            core["pre_registered_wrong_pair_map_sha256"]
        ),
        "pairing_mode": str(core["pairing_mode"]),
        "full_reference_detached": bool(core["full_reference_detached"]),
        "base_outer_view_bundle_sha256": str(
            outer_receipt["base_outer_view_bundle_sha256"]
        ),
        "outer_view_bundle_sha256": str(outer_receipt["outer_view_bundle_sha256"]),
        "outer_evaluation_sha256": str(outer_receipt["outer_evaluation_sha256"]),
        "target_transform_sha256": str(core["target_transform_sha256"]),
        "target_mean_mm": list(core["target_mean_mm"]),
        "target_scale_mm": list(core["target_scale_mm"]),
        "best_epoch": int(core["best_epoch"]),
        "best_checkpoint_score": float(core["best_checkpoint_score"]),
        "checkpoint_selection_scope": str(core["checkpoint_selection_scope"]),
        "checkpoint_fixed_before_outer_labels": bool(
            outer_receipt["checkpoint_fixed_before_outer_labels"]
        ),
        "outer_eval_count": int(outer_receipt["outer_eval_count"]),
        "same_checkpoint_for_all_four_outages": bool(
            core["same_checkpoint_for_all_four_outages"]
        ),
        "checkpoint_criterion": str(core["checkpoint_criterion"]),
        "exact_five_views": True,
    }
    return {
        "preprocessing": dict(preprocessor_metadata),
        "history": deepcopy(core["history"]),
        "checkpoint_state": dict(core["state_dict"]),
        "training": training,
        "outer_evaluation": outer_receipt,
        "trial_uid": [str(value) for value in trial_uid],
        "segment_id": [int(value) for value in segment_id],
        "target_id": [str(value) for value in target_id],
        "truth_mm": np.asarray(outer["truth_raw_mm"], dtype=np.float64),
        "predictions_mm": np.asarray(outer["predictions_raw_mm"], dtype=np.float64),
    }


def _default_task_executor(
    *,
    task: Mapping[str, Any],
    data: Mapping[str, Any],
    worker: Mapping[str, Any],
    config: Mapping[str, Any],
    _phase_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute one real 400-update fold/seed fit and return an in-memory bundle."""

    del config
    if os.environ.get("CUDA_VISIBLE_DEVICES") != _canonical_cuda_uuid_selector(
        worker.get("uuid")
    ):
        raise GPUOutagePORDV5Error(
            "production task must run inside its fixed spawn CUDA visibility"
        )
    held_out = int(task["held_out_segment"])
    segments = np.asarray(data["segment_id"], dtype=np.int64)
    eval_rows = segments == held_out
    train_rows = ~eval_rows
    preprocessor = _fit_v5_preprocessor(data, held_out_segment=held_out)
    X_train = preprocessor.transform_features(np.asarray(data["X"])[train_rows]).astype(
        np.float32
    )
    y_train = preprocessor.transform_targets(np.asarray(data["y"])[train_rows]).astype(
        np.float32
    )
    target_mean = np.asarray(preprocessor.target_mean_, dtype=np.float64)
    target_scale = np.repeat(float(preprocessor.target_scale_), 3)
    core = train_pord_fold_seed(
        X_train_scaled=X_train,
        y_train_scaled=y_train,
        train_segment_ids=segments[train_rows],
        train_trial_uids=np.asarray(data["trial_uid"])[train_rows].astype(str).tolist(),
        held_out_segment=held_out,
        target_mean_mm=target_mean,
        target_scale_mm=target_scale,
        method_id=str(task["method_id"]),
        model_seed=int(task["model_seed"]),
        device="cuda:0",
        backbone_id=str(task["backbone_id"]),
        epoch_budget=PORD_EPOCH_BUDGET,
    )
    if _phase_callback is not None:
        _phase_callback(
            "training.completed",
            {"core": core, "preprocessing": preprocessor.metadata()},
        )
    # The outer feature/label projection is intentionally below the completed
    # trainer call.  The training API cannot accept outer arrays, so the
    # checkpoint hash is fixed before this executor accesses the held-out
    # label.  The worker nevertheless holds the full authenticated snapshot;
    # this is code-order enforcement, not process capability isolation.
    X_holdout = preprocessor.transform_features(
        np.asarray(data["X"])[eval_rows]
    ).astype(np.float32)
    outer_prediction = predict_pord_checkpoint_outer(
        training_result=core,
        X_outer_scaled=X_holdout,
        outer_segment_ids=segments[eval_rows],
        outer_trial_uids=np.asarray(data["trial_uid"])[eval_rows]
        .astype(str)
        .tolist(),
        held_out_segment=held_out,
        target_mean_mm=target_mean,
        target_scale_mm=target_scale,
        device="cuda:0",
    )
    if _phase_callback is not None:
        _phase_callback(
            "outer_prediction.completed",
            {
                "outer_prediction": outer_prediction,
                "trial_uid": np.asarray(data["trial_uid"])[eval_rows]
                .astype(str)
                .tolist(),
                "segment_id": segments[eval_rows].astype(int).tolist(),
                "target_id": np.asarray(data["target_id"])[eval_rows]
                .astype(str)
                .tolist(),
            },
        )
        _phase_callback(
            "outer_eval.started",
            {"outer_prediction_receipt": outer_prediction["receipt"]},
        )
    y_holdout_raw_mm = np.asarray(data["y"], dtype=np.float64)[eval_rows]
    outer = evaluate_pord_outer_predictions_once(
        training_result=core,
        outer_prediction=outer_prediction,
        y_outer_raw_mm=y_holdout_raw_mm,
    )
    if _phase_callback is not None:
        _phase_callback(
            "outer_eval.completed",
            {
                "outer_evaluation": outer["receipt"],
                "truth_raw_mm": outer["truth_raw_mm"],
                "predictions_raw_mm": outer["predictions_raw_mm"],
            },
        )
    return _assemble_task_executor_output(
        task=task,
        preprocessor_metadata=preprocessor.metadata(),
        core=core,
        outer=outer,
        trial_uid=np.asarray(data["trial_uid"])[eval_rows],
        segment_id=segments[eval_rows],
        target_id=np.asarray(data["target_id"])[eval_rows],
    )


def _validate_executor_output(
    output: Mapping[str, Any], task: Mapping[str, Any]
) -> dict[str, Any]:
    if type(output) is not dict or set(output) != _OUTPUT_KEYS:
        raise GPUOutagePORDV5Error("task executor output schema changed")
    if (
        type(output["trial_uid"]) is not list
        or type(output["target_id"]) is not list
        or type(output["segment_id"]) is not list
        or any(
            type(value) is not str or not value for value in output["trial_uid"]
        )
        or any(
            type(value) is not str or not value for value in output["target_id"]
        )
        or any(type(value) is not int for value in output["segment_id"])
    ):
        raise GPUOutagePORDV5Error("task prediction identity types changed")
    uids = tuple(output["trial_uid"])
    target_ids = tuple(output["target_id"])
    segments = np.asarray(output["segment_id"], dtype=np.int64)
    truth = np.asarray(output["truth_mm"], dtype=np.float64)
    predictions = np.asarray(output["predictions_mm"], dtype=np.float64)
    rows = len(uids)
    if (
        rows == 0
        or len(set(uids)) != rows
        or len(target_ids) != rows
        or segments.shape != (rows,)
        or not np.all(
            segments
            == _exact_int(task["held_out_segment"], name="task held_out_segment")
        )
        or truth.shape != (rows, 3)
        or predictions.shape != (5, rows, 3)
        or not np.isfinite(truth).all()
        or not np.isfinite(predictions).all()
    ):
        raise GPUOutagePORDV5Error("task prediction rows/views are invalid")
    prep = output["preprocessing"]
    if not isinstance(prep, Mapping):
        raise GPUOutagePORDV5Error("task preprocessing metadata is missing")
    try:
        if type(prep["feature_names"]) is not list or any(
            type(value) is not str for value in prep["feature_names"]
        ):
            raise TypeError("feature_names changed type")
        feature_names = tuple(prep["feature_names"])
        feature_mean = np.asarray(prep["feature_mean"], dtype=np.float64)
        feature_scale = np.asarray(prep["feature_scale"], dtype=np.float64)
        target_mean = np.asarray(prep["target_mean"], dtype=np.float64)
        target_scale_scalar = _exact_finite_float(
            prep["target_scale"], name="preprocessor target_scale"
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GPUOutagePORDV5Error("task preprocessor metadata is malformed") from exc
    if (
        feature_names != FUSION_COLUMNS
        or feature_mean.shape != (43,)
        or feature_scale.shape != (43,)
        or target_mean.shape != (3,)
        or not np.isfinite(feature_mean).all()
        or not np.isfinite(feature_scale).all()
        or not np.isfinite(target_mean).all()
        or np.any(feature_scale <= 0)
        or not math.isfinite(target_scale_scalar)
        or target_scale_scalar <= 0
    ):
        raise GPUOutagePORDV5Error("task train-only scaler is invalid")
    training = output["training"]
    expected_training_keys = {
        "backbone_id",
        "method_id",
        "epoch_budget",
        "optimizer_update_count",
        "contract_epoch_budget_complete",
        "initial_state_sha256",
        "selected_state_sha256",
        "training_core_result_sha256",
        "training_bundle_sha256",
        "base_train_view_bundle_sha256",
        "train_view_bundle_sha256",
        "availability_semantics",
        "gradient_view_semantics",
        "pre_registered_wrong_pair_map_sha256",
        "pairing_mode",
        "full_reference_detached",
        "base_outer_view_bundle_sha256",
        "outer_view_bundle_sha256",
        "outer_evaluation_sha256",
        "target_transform_sha256",
        "target_mean_mm",
        "target_scale_mm",
        "best_epoch",
        "best_checkpoint_score",
        "checkpoint_selection_scope",
        "checkpoint_fixed_before_outer_labels",
        "outer_eval_count",
        "same_checkpoint_for_all_four_outages",
        "checkpoint_criterion",
        "exact_five_views",
    }
    if not isinstance(training, Mapping) or set(training) != expected_training_keys:
        raise GPUOutagePORDV5Error("task training metadata schema changed")
    repeated_scale = np.asarray(training["target_scale_mm"], dtype=np.float64)
    method = _exact_string(task["method_id"], name="task method_id")
    expected_availability = (
        "constant_all_available_negative_control"
        if method == "constant_availability_erm"
        else "physical_mask_state"
    )
    expected_gradient = (
        "full_view_gradient_only_outage_selected_checkpoint"
        if method == "full_gradient_only_mask_selected_control"
        else "five_view_mask_weighted_gradient"
    )
    expected_pairing = {
        "group_mean_pord": "group_mean_difference_not_pair_specific",
        "paired_tail": "correct_uid_stop_gradient",
        "paired_interaction": "correct_uid_stop_gradient",
        "paired_pord": "correct_uid_stop_gradient",
        "paired_wrong_pair": "sorted_uid_within_segment_circular_derangement",
        "paired_no_stopgrad": "correct_uid_no_stop_gradient",
    }.get(method, "none")
    if (
        training["backbone_id"] != task["backbone_id"]
        or training["method_id"] != task["method_id"]
        or training["epoch_budget"] != 400
        or training["optimizer_update_count"] != 400
        or training["contract_epoch_budget_complete"] is not True
        or training["same_checkpoint_for_all_four_outages"] is not True
        or training["exact_five_views"] is not True
        or training["checkpoint_selection_scope"]
        != PORD_CHECKPOINT_SELECTION_SCOPE
        or training["checkpoint_fixed_before_outer_labels"] is not True
        or training["outer_eval_count"] != 1
        or training["checkpoint_criterion"] != PORD_CHECKPOINT_CRITERION
        or training["availability_semantics"] != expected_availability
        or training["gradient_view_semantics"] != expected_gradient
        or training["pairing_mode"] != expected_pairing
        or training["full_reference_detached"]
        is not (method != "paired_no_stopgrad")
        or type(training["epoch_budget"]) is not int
        or type(training["optimizer_update_count"]) is not int
        or type(training["outer_eval_count"]) is not int
        or _LOWER_SHA256.fullmatch(
            training["pre_registered_wrong_pair_map_sha256"]
            if type(training["pre_registered_wrong_pair_map_sha256"]) is str
            else ""
        )
        is None
        or (
            method == "constant_availability_erm"
            and training["train_view_bundle_sha256"]
            == training["base_train_view_bundle_sha256"]
        )
        or (
            method != "constant_availability_erm"
            and training["train_view_bundle_sha256"]
            != training["base_train_view_bundle_sha256"]
        )
        or repeated_scale.shape != (3,)
        or not np.array_equal(
            repeated_scale,
            np.repeat(target_scale_scalar, 3).astype(np.float64),
        )
        or not np.array_equal(
            np.asarray(training["target_mean_mm"], dtype=np.float64), target_mean
        )
        or type(training["best_epoch"]) is not int
        or not 1 <= training["best_epoch"] <= 400
        or type(training["best_checkpoint_score"]) is not float
        or not math.isfinite(training["best_checkpoint_score"])
    ):
        raise GPUOutagePORDV5Error("task training controls/scaler binding changed")
    outer_evaluation = output["outer_evaluation"]
    full_metric = regression_metrics(truth, predictions[0])
    outage_metrics = [
        regression_metrics(truth, predictions[index]) for index in range(1, 5)
    ]
    if (
        not isinstance(outer_evaluation, Mapping)
        or set(outer_evaluation) != _OUTER_EVALUATION_KEYS
        or outer_evaluation.get("contract")
        != "alphaf.endpoint_regression.gpu_outage_pord.v5.outer_evaluation"
        or outer_evaluation.get("training_result_sha256")
        != training["training_core_result_sha256"]
        or outer_evaluation.get("selected_state_sha256")
        != training["selected_state_sha256"]
        or outer_evaluation.get("backbone_id") != task["backbone_id"]
        or type(outer_evaluation.get("held_out_segment")) is not str
        or outer_evaluation.get("held_out_segment")
        != f"{task['held_out_segment']}"
        or outer_evaluation.get("checkpoint_selection_scope")
        != PORD_CHECKPOINT_SELECTION_SCOPE
        or outer_evaluation.get("checkpoint_fixed_before_outer_labels") is not True
        or outer_evaluation.get("outer_eval_count") != 1
        or outer_evaluation.get("outer_view_bundle_sha256")
        != training["outer_view_bundle_sha256"]
        or outer_evaluation.get("base_outer_view_bundle_sha256")
        != training["base_outer_view_bundle_sha256"]
        or outer_evaluation.get("availability_semantics")
        != training["availability_semantics"]
        or outer_evaluation.get("outer_trial_uid_sha256") != stable_hash(list(uids))
        or outer_evaluation.get("outer_truth_raw_mm_sha256") != array_sha256(truth)
        or outer_evaluation.get("outer_prediction_raw_mm_sha256")
        != array_sha256(predictions)
        or outer_evaluation.get("target_transform_sha256")
        != training["target_transform_sha256"]
        or tuple(outer_evaluation.get("evaluated_view_names", ()))
        != ("full", "drop_imu0", "drop_imu1", "drop_imu2", "drop_imu3")
        or outer_evaluation.get("same_checkpoint_for_all_four_outages") is not True
        or type(outer_evaluation.get("outer_full_rmse_3d_mm")) is not float
        or not math.isfinite(outer_evaluation["outer_full_rmse_3d_mm"])
        or type(outer_evaluation.get("outer_outage_rmse_3d_mm")) is not list
        or len(outer_evaluation["outer_outage_rmse_3d_mm"]) != 4
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in outer_evaluation["outer_outage_rmse_3d_mm"]
        )
        or not math.isclose(
            outer_evaluation["outer_full_rmse_3d_mm"],
            full_metric["rmse_3d_mm"],
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not np.array_equal(
            np.asarray(
                outer_evaluation["outer_outage_rmse_3d_mm"], dtype=np.float64
            ),
            np.asarray(
                [row["rmse_3d_mm"] for row in outage_metrics], dtype=np.float64
            ),
        )
        or outer_evaluation.get("outer_evaluation_sha256")
        != stable_hash(
            {key: outer_evaluation[key] for key in _OUTER_EVALUATION_HASH_KEYS}
        )
        or training["outer_evaluation_sha256"]
        != outer_evaluation.get("outer_evaluation_sha256")
    ):
        raise GPUOutagePORDV5Error("single post-checkpoint outer evaluation changed")
    history = output["history"]
    if (
        not isinstance(history, list)
        or len(history) != 400
        or [row.get("epoch") for row in history] != list(range(1, 401))
        or any(
            row.get("checkpoint_selection_scope")
            != PORD_CHECKPOINT_SELECTION_SCOPE
            or row.get("outer_eval_count") != 0
            or row.get("outer_labels_accessed") is not False
            for row in history
        )
    ):
        raise GPUOutagePORDV5Error("task history must bind all 400 ordered updates")
    checkpoint = output["checkpoint_state"]
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise GPUOutagePORDV5Error("task checkpoint state is missing")
    if training["selected_state_sha256"] != state_dict_sha256(checkpoint):
        raise GPUOutagePORDV5Error("task checkpoint state hash changed")
    return {
        "preprocessing": dict(prep),
        "history": deepcopy(history),
        "checkpoint_state": dict(checkpoint),
        "training": dict(training),
        "outer_evaluation": dict(outer_evaluation),
        "trial_uid": list(uids),
        "segment_id": segments.astype(np.int64),
        "target_id": list(target_ids),
        "truth_mm": truth,
        "predictions_mm": predictions,
    }


def _task_dir(run_dir: str | Path, task: Mapping[str, Any]) -> Path:
    return Path(run_dir) / "development" / "tasks" / str(task["task_id"])


def _write_phase_record(path: Path, payload: Mapping[str, Any]) -> Path:
    core = dict(payload)
    core["phase_sha256"] = stable_hash(core)
    return _exclusive_write_json(path, core)


def _phase_link(path: Path, run_dir: Path) -> dict[str, Any]:
    payload = read_json_object(path)
    return {
        "file": _artifact_reference(path, run_dir),
        "phase": payload["phase"],
        "phase_sha256": payload["phase_sha256"],
    }


def _production_task_phase_callback(
    *,
    run_dir: Path,
    task: Mapping[str, Any],
    execution_path: Path,
    runtime_gpu_receipt: Mapping[str, Any] | None = None,
) -> Callable[[str, Mapping[str, Any]], None]:
    """Persist each scientific phase before the next irreversible action."""

    task_dir = execution_path.parent

    def callback(phase: str, payload: Mapping[str, Any]) -> None:
        if phase == "training.completed":
            core = payload["core"]
            prep_path = atomic_write_json(
                task_dir / "preprocessing.json", payload["preprocessing"]
            )
            history_payload = {
                "contract": "alphaf.endpoint_regression.gpu_outage_pord_history.v5",
                "task_id": task["task_id"],
                "epoch_budget": 400,
                "update_count": 400,
                "records": list(core["history"]),
            }
            history_payload["history_sha256"] = stable_hash(history_payload["records"])
            history_path = atomic_write_json(task_dir / "history.json", history_payload)
            checkpoint_path = _save_checkpoint(
                task_dir / "checkpoint.npz", core["state_dict"]
            )
            core_receipt = {
                key: deepcopy(value)
                for key, value in core.items()
                if key not in {"state_dict", "history"}
            }
            record = {
                "contract": "alphaf.endpoint_regression.gpu_outage_pord_task_phase.v5",
                "task_id": task["task_id"],
                "attempt_index": 1,
                "phase": phase,
                "previous_phase": {
                    "file": _artifact_reference(execution_path, run_dir),
                    "phase": "execution.started",
                    "sha256": sha256_file(execution_path),
                },
                "training_result_sha256": core["result_sha256"],
                "training_bundle_sha256": core["training_bundle_sha256"],
                "selected_state_sha256": core["selected_state_sha256"],
                "initial_state_sha256": core["initial_state_sha256"],
                "checkpoint_selection_scope": core["checkpoint_selection_scope"],
                "outer_eval_count": 0,
                "training_core_receipt": core_receipt,
                "runtime_gpu_transport": (
                    dict(runtime_gpu_receipt)
                    if runtime_gpu_receipt is not None
                    else None
                ),
                "artifacts": {
                    "preprocessing": _artifact_reference(prep_path, run_dir),
                    "history": _artifact_reference(history_path, run_dir, updates=400),
                    "checkpoint": _artifact_reference(checkpoint_path, run_dir),
                },
            }
            _write_phase_record(task_dir / "training.completed.json", record)
            return
        training_phase = task_dir / "training.completed.json"
        if not training_phase.is_file():
            raise GPUOutagePORDV5Error("outer phase preceded training completion")
        if phase == "outer_prediction.completed":
            prediction = payload["outer_prediction"]
            values = np.asarray(prediction["predictions_raw_mm"], dtype=np.float64)
            frame = pd.DataFrame(
                {
                    "trial_uid": list(payload["trial_uid"]),
                    "segment_id": list(payload["segment_id"]),
                    "target_id": list(payload["target_id"]),
                }
            )
            for view_index, view in enumerate(_PREDICTION_VIEW_NAMES):
                for axis, axis_name in enumerate(("x", "y", "z")):
                    frame[f"{view}_pred_{axis_name}_mm"] = values[view_index, :, axis]
            prediction_path = atomic_write_parquet(
                task_dir / "outer_predictions.parquet", frame
            )
            receipt_path = atomic_write_json(
                task_dir / "outer_prediction_receipt.json", prediction["receipt"]
            )
            record = {
                "contract": "alphaf.endpoint_regression.gpu_outage_pord_task_phase.v5",
                "task_id": task["task_id"],
                "attempt_index": 1,
                "phase": phase,
                "previous_phase": _phase_link(training_phase, run_dir),
                "outer_labels_accessed": False,
                "runtime_gpu_transport": (
                    dict(runtime_gpu_receipt)
                    if runtime_gpu_receipt is not None
                    else None
                ),
                "outer_prediction_raw_mm_sha256": array_sha256(values),
                "artifacts": {
                    "prediction": _artifact_reference(
                        prediction_path, run_dir, rows=len(frame), views=5
                    ),
                    "receipt": _artifact_reference(receipt_path, run_dir),
                },
            }
            _write_phase_record(task_dir / "outer_prediction.completed.json", record)
            return
        prediction_phase = task_dir / "outer_prediction.completed.json"
        if not prediction_phase.is_file():
            raise GPUOutagePORDV5Error("outer evaluation preceded prediction completion")
        if phase == "outer_eval.started":
            record = {
                "contract": "alphaf.endpoint_regression.gpu_outage_pord_task_phase.v5",
                "task_id": task["task_id"],
                "attempt_index": 1,
                "phase": phase,
                "previous_phase": _phase_link(prediction_phase, run_dir),
                "maximum_outer_eval_count": 1,
                "outer_eval_count_before": 0,
                "runtime_gpu_transport": (
                    dict(runtime_gpu_receipt)
                    if runtime_gpu_receipt is not None
                    else None
                ),
            }
            _write_phase_record(task_dir / "outer_eval.started.json", record)
            return
        if phase != "outer_eval.completed":
            raise GPUOutagePORDV5Error(f"unknown durable task phase: {phase}")
        started_phase = task_dir / "outer_eval.started.json"
        if not started_phase.is_file():
            raise GPUOutagePORDV5Error("outer evaluation completion lacks start marker")
        outer_path = atomic_write_json(
            task_dir / "outer_evaluation.json", payload["outer_evaluation"]
        )
        prediction_frame = _read_exact_parquet(
            task_dir / "outer_predictions.parquet",
            ["trial_uid", "segment_id", "target_id", *_PREDICTION_COLUMNS],
        )
        truth = np.asarray(payload["truth_raw_mm"], dtype=np.float64)
        if truth.shape != (len(prediction_frame), 3):
            raise GPUOutagePORDV5Error("outer evaluation truth row count changed")
        for axis, name in enumerate(_TRUE_COLUMNS):
            prediction_frame[name] = truth[:, axis]
        prediction_frame = prediction_frame[
            ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS, *_PREDICTION_COLUMNS]
        ]
        evaluation_path = atomic_write_parquet(
            task_dir / "predictions.parquet", prediction_frame
        )
        record = {
            "contract": "alphaf.endpoint_regression.gpu_outage_pord_task_phase.v5",
            "task_id": task["task_id"],
            "attempt_index": 1,
            "phase": phase,
            "previous_phase": _phase_link(started_phase, run_dir),
            "outer_eval_count": 1,
            "runtime_gpu_transport": (
                dict(runtime_gpu_receipt)
                if runtime_gpu_receipt is not None
                else None
            ),
            "outer_evaluation_sha256": payload["outer_evaluation"][
                "outer_evaluation_sha256"
            ],
            "artifacts": {
                "outer_evaluation": _artifact_reference(outer_path, run_dir),
                "predictions": _artifact_reference(
                    evaluation_path, run_dir, rows=len(prediction_frame), views=5
                ),
            },
        }
        _write_phase_record(task_dir / "outer_eval.completed.json", record)

    return callback


def _materialize_synthetic_test_phase_chain(
    *,
    run_dir: Path,
    task: Mapping[str, Any],
    execution_path: Path,
    output: Mapping[str, Any],
) -> None:
    """Give private synthetic fixtures explicit, non-production phase evidence."""

    if _prepared_execution_mode(run_dir) != PORD_V5_SYNTHETIC_TEST_MODE:
        raise GPUOutagePORDV5Error("synthetic phase materialization is test-only")
    normalized = _validate_executor_output(output, task)
    training = normalized["training"]
    callback = _production_task_phase_callback(
        run_dir=run_dir,
        task=task,
        execution_path=execution_path,
        runtime_gpu_receipt=read_json_object(execution_path).get(
            "runtime_gpu_transport"
        ),
    )
    callback(
        "training.completed",
        {
            "preprocessing": normalized["preprocessing"],
            "core": {
                "history": normalized["history"],
                "state_dict": normalized["checkpoint_state"],
                "result_sha256": training["training_core_result_sha256"],
                "training_bundle_sha256": training["training_bundle_sha256"],
                "selected_state_sha256": training["selected_state_sha256"],
                "initial_state_sha256": training["initial_state_sha256"],
                "checkpoint_selection_scope": training[
                    "checkpoint_selection_scope"
                ],
            },
        },
    )
    prediction_receipt = {
        "contract": "synthetic_test_only_outer_prediction_receipt",
        "training_result_sha256": training["training_core_result_sha256"],
        "outer_view_bundle_sha256": training["outer_view_bundle_sha256"],
        "outer_trial_uid_sha256": stable_hash(normalized["trial_uid"]),
        "outer_prediction_raw_mm_sha256": array_sha256(
            normalized["predictions_mm"]
        ),
        "outer_labels_accessed": False,
    }
    prediction_receipt["outer_prediction_receipt_sha256"] = stable_hash(
        prediction_receipt
    )
    callback(
        "outer_prediction.completed",
        {
            "outer_prediction": {
                "receipt": prediction_receipt,
                "trial_uids": normalized["trial_uid"],
                "predictions_raw_mm": normalized["predictions_mm"],
            },
            "trial_uid": normalized["trial_uid"],
            "segment_id": normalized["segment_id"],
            "target_id": normalized["target_id"],
        },
    )
    callback(
        "outer_eval.started",
        {"outer_prediction_receipt": prediction_receipt},
    )
    callback(
        "outer_eval.completed",
        {
            "outer_evaluation": normalized["outer_evaluation"],
            "truth_raw_mm": normalized["truth_mm"],
            "predictions_raw_mm": normalized["predictions_mm"],
        },
    )


def _task_result_hash(payload: Mapping[str, Any]) -> str:
    core = {key: value for key, value in payload.items() if key != "result_sha256"}
    return stable_hash(core)


def _task_execution_payload(
    *,
    run_dir: Path,
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    worker: Mapping[str, Any],
    workers_per_gpu: int,
    runtime_gpu_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    source = read_json_object(run_dir / "provenance" / "source_snapshot.json")
    environment = read_json_object(run_dir / "provenance" / "environment_snapshot.json")
    frozen_devices = _validate_gpu_devices(environment["payload"]["cuda_devices"])
    matches = [
        row
        for row in frozen_devices
        if row["physical_gpu_id"] == worker.get("physical_gpu_id")
    ]
    if len(matches) != 1 or any(
        worker.get(key) != matches[0][key] for key in ("uuid", "pci_bus_id", "name")
    ):
        raise GPUOutagePORDV5Error("worker GPU UUID/PCI mapping is not frozen")
    if (
        runtime_gpu_receipt.get("cuda_visible_devices")
        != _canonical_cuda_uuid_selector(worker.get("uuid"))
    ):
        raise GPUOutagePORDV5Error("worker CUDA selector is not canonically frozen")
    mode = _prepared_execution_mode(run_dir)
    return {
        "contract": PORD_V5_TASK_EXECUTION_CONTRACT,
        "phase": "execution.started",
        "attempt_index": 1,
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
        "config_sha256": pord_v5_config_sha256(config),
        "task": dict(task),
        "worker": dict(worker),
        "workers_per_gpu": workers_per_gpu,
        "training_source_fingerprint": source["fingerprint"],
        "environment_fingerprint": environment["fingerprint"],
        "runtime_gpu_transport": dict(runtime_gpu_receipt),
        "scientific_fit_identity_sha256": stable_hash(task),
        # This is an honest capability boundary, not a claim of process-level
        # label isolation.  The trainer API receives train-only projections,
        # while the worker process holds the authenticated full development
        # snapshot and enforces non-use through code order plus durable phases.
        "worker_development_label_authority": (
            "full_snapshot_in_process_not_capability_isolated"
        ),
        "trainer_outer_array_surface": "absent",
        "outer_nonuse_enforcement": "code_order_and_durable_phase_receipts",
        "challenge_labels_accessed": False,
    }


def _materialize_task_output(
    *,
    run_dir: Path,
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    worker: Mapping[str, Any],
    execution_path: Path,
    output: Mapping[str, Any],
    data: Mapping[str, Any],
    runtime_gpu_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    task_dir = execution_path.parent
    normalized = _validate_executor_output(output, task)
    _validate_holdout_source_binding(normalized, data, task)
    if (
        not isinstance(runtime_gpu_receipt, Mapping)
        or set(runtime_gpu_receipt) != _RUNTIME_GPU_RECEIPT_KEYS
        or runtime_gpu_receipt.get("transport_bound") is not True
        or runtime_gpu_receipt.get("visible_device_index") != 0
        or runtime_gpu_receipt.get("logical_worker_id")
        != worker.get("logical_worker_id")
        or runtime_gpu_receipt.get("cuda_visible_devices")
        != _canonical_cuda_uuid_selector(worker.get("uuid"))
        or any(
            runtime_gpu_receipt.get(key) != worker.get(key)
            for key in ("physical_gpu_id", "uuid", "pci_bus_id", "name")
        )
    ):
        raise GPUOutagePORDV5Error(
            "runtime CUDA identity is not bound to fixed process transport"
        )
    # Every irreversible artifact was durably written by the phase callback.
    # Finalization authenticates and references those bytes; it never rewrites
    # them after a later phase has already hash-bound them.
    prep_path = task_dir / "preprocessing.json"
    outer_evaluation_path = task_dir / "outer_evaluation.json"
    history_payload = {
        "contract": "alphaf.endpoint_regression.gpu_outage_pord_history.v5",
        "task_id": task["task_id"],
        "epoch_budget": 400,
        "update_count": 400,
        "records": normalized["history"],
    }
    history_payload["history_sha256"] = stable_hash(history_payload["records"])
    history_path = task_dir / "history.json"
    checkpoint_path = task_dir / "checkpoint.npz"
    if not all(
        path.is_file()
        for path in (prep_path, outer_evaluation_path, history_path, checkpoint_path)
    ):
        raise GPUOutagePORDV5Error("durable task phase artifacts are incomplete")
    _assert_json_close(
        read_json_object(prep_path), normalized["preprocessing"], path="preprocessing"
    )
    _assert_json_close(
        read_json_object(outer_evaluation_path),
        normalized["outer_evaluation"],
        path="outer_evaluation",
    )
    _assert_json_close(
        read_json_object(history_path), history_payload, path="history"
    )
    if state_dict_sha256(_read_checkpoint(checkpoint_path)) != state_dict_sha256(
        normalized["checkpoint_state"]
    ):
        raise GPUOutagePORDV5Error("durable checkpoint bytes changed")
    checkpoint_metadata = {
        "contract": "alphaf.endpoint_regression.gpu_outage_pord_checkpoint.v5",
        "task_id": task["task_id"],
        "candidate_id": task["candidate_id"],
        "backbone_id": task["backbone_id"],
        "method_id": task["method_id"],
        "model_seed": task["model_seed"],
        "held_out_segment": task["held_out_segment"],
        "selected_state_sha256": normalized["training"]["selected_state_sha256"],
        "best_epoch": normalized["training"]["best_epoch"],
        "best_checkpoint_score": normalized["training"]["best_checkpoint_score"],
        "checkpoint_selection_scope": normalized["training"][
            "checkpoint_selection_scope"
        ],
        "checkpoint_fixed_before_outer_labels": normalized["training"][
            "checkpoint_fixed_before_outer_labels"
        ],
        "outer_eval_count": normalized["training"]["outer_eval_count"],
        "outer_evaluation_sha256": normalized["training"][
            "outer_evaluation_sha256"
        ],
        "same_checkpoint_for_all_four_outages": True,
        "checkpoint_file": _artifact_reference(checkpoint_path, run_dir),
    }
    checkpoint_metadata_path = atomic_write_json(
        task_dir / "checkpoint.json", checkpoint_metadata
    )
    prediction_frame = pd.DataFrame(
        {
            "trial_uid": normalized["trial_uid"],
            "segment_id": normalized["segment_id"],
            "target_id": normalized["target_id"],
        }
    )
    for axis, name in enumerate(_TRUE_COLUMNS):
        prediction_frame[name] = normalized["truth_mm"][:, axis]
    for view_index, view in enumerate(_PREDICTION_VIEW_NAMES):
        for axis, axis_name in enumerate(("x", "y", "z")):
            prediction_frame[f"{view}_pred_{axis_name}_mm"] = normalized[
                "predictions_mm"
            ][view_index, :, axis]
    prediction_path = task_dir / "predictions.parquet"
    persisted_prediction = _read_exact_parquet(
        prediction_path,
        ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS, *_PREDICTION_COLUMNS],
    )
    if (
        persisted_prediction[["trial_uid", "target_id"]].astype(str).to_dict("list")
        != prediction_frame[["trial_uid", "target_id"]].astype(str).to_dict("list")
        or not np.array_equal(
            persisted_prediction["segment_id"].to_numpy(dtype=np.int64),
            prediction_frame["segment_id"].to_numpy(dtype=np.int64),
        )
        or not np.array_equal(
            persisted_prediction.loc[:, [*_TRUE_COLUMNS, *_PREDICTION_COLUMNS]].to_numpy(
                dtype=np.float64
            ),
            prediction_frame.loc[:, [*_TRUE_COLUMNS, *_PREDICTION_COLUMNS]].to_numpy(
                dtype=np.float64
            ),
        )
    ):
        raise GPUOutagePORDV5Error("durable prediction bytes changed")
    metrics = {
        "full": regression_metrics(
            normalized["truth_mm"], normalized["predictions_mm"][0]
        ),
        "outages": [
            regression_metrics(
                normalized["truth_mm"], normalized["predictions_mm"][index]
            )
            for index in range(1, 5)
        ],
    }
    source = read_json_object(run_dir / "provenance" / "source_snapshot.json")
    environment = read_json_object(run_dir / "provenance" / "environment_snapshot.json")
    phase_paths = {
        "training_phase": task_dir / "training.completed.json",
        "outer_prediction_phase": task_dir / "outer_prediction.completed.json",
        "outer_prediction": task_dir / "outer_predictions.parquet",
        "outer_prediction_receipt": task_dir / "outer_prediction_receipt.json",
        "outer_eval_started": task_dir / "outer_eval.started.json",
        "outer_eval_phase": task_dir / "outer_eval.completed.json",
    }
    if not all(path.is_file() for path in phase_paths.values()):
        raise GPUOutagePORDV5Error("durable task phase chain is incomplete")
    mode = _prepared_execution_mode(run_dir)
    completion = {
        "contract": PORD_V5_TASK_COMPLETION_CONTRACT,
        "phase": "bundle.completed",
        "attempt_index": 1,
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
        "previous_phase": _phase_link(phase_paths["outer_eval_phase"], run_dir),
        "config_sha256": pord_v5_config_sha256(config),
        "task": dict(task),
        "worker": dict(worker),
        "training_source_fingerprint": source["fingerprint"],
        "environment_fingerprint": environment["fingerprint"],
        "runtime_gpu": dict(runtime_gpu_receipt),
        "training": normalized["training"],
        "metrics": metrics,
        "artifacts": {
            "execution": _artifact_reference(execution_path, run_dir),
            "preprocessing": _artifact_reference(prep_path, run_dir),
            "outer_evaluation": _artifact_reference(
                outer_evaluation_path, run_dir
            ),
            "history": _artifact_reference(history_path, run_dir, updates=400),
            "checkpoint": _artifact_reference(checkpoint_path, run_dir),
            "checkpoint_metadata": _artifact_reference(
                checkpoint_metadata_path, run_dir
            ),
            "predictions": _artifact_reference(
                prediction_path, run_dir, rows=len(prediction_frame), views=5
            ),
            "training_phase": _artifact_reference(
                phase_paths["training_phase"], run_dir
            ),
            "outer_prediction_phase": _artifact_reference(
                phase_paths["outer_prediction_phase"], run_dir
            ),
            "outer_prediction": _artifact_reference(
                phase_paths["outer_prediction"],
                run_dir,
                rows=len(prediction_frame),
                views=5,
            ),
            "outer_prediction_receipt": _artifact_reference(
                phase_paths["outer_prediction_receipt"], run_dir
            ),
            "outer_eval_started": _artifact_reference(
                phase_paths["outer_eval_started"], run_dir
            ),
            "outer_eval_phase": _artifact_reference(
                phase_paths["outer_eval_phase"], run_dir
            ),
        },
        "challenge_labels_accessed": False,
    }
    completion_path = _write_phase_record(
        task_dir / "bundle.completed.json", completion
    )
    completion = read_json_object(completion_path)
    result = {
        **{key: value for key, value in completion.items() if key != "phase_sha256"},
        "contract": PORD_V5_TASK_RESULT_CONTRACT,
        "phase": "result",
        "previous_phase": _phase_link(completion_path, run_dir),
        "completion": _artifact_reference(completion_path, run_dir),
    }
    result["result_sha256"] = _task_result_hash(result)
    atomic_write_json(task_dir / "result.json", result)
    _validate_task_result(run_dir, config, task, result)
    return result


def _validate_holdout_source_binding(
    normalized: Mapping[str, Any],
    data: Mapping[str, Any],
    task: Mapping[str, Any],
) -> None:
    eval_rows = np.asarray(data["segment_id"], dtype=np.int64) == int(
        task["held_out_segment"]
    )
    if (
        normalized["trial_uid"]
        != np.asarray(data["trial_uid"])[eval_rows].astype(str).tolist()
        or normalized["target_id"]
        != np.asarray(data["target_id"])[eval_rows].astype(str).tolist()
        or not np.array_equal(
            np.asarray(normalized["segment_id"], dtype=np.int64),
            np.asarray(data["segment_id"], dtype=np.int64)[eval_rows],
        )
        or not np.array_equal(
            np.asarray(normalized["truth_mm"], dtype=np.float64),
            np.asarray(data["y"], dtype=np.float64)[eval_rows],
        )
    ):
        raise GPUOutagePORDV5Error(
            "task holdout identity/truth is not exactly bound to development replay"
        )


def _validate_production_task_semantics(
    *,
    run_dir: Path,
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    development_data: Mapping[str, Any],
    persisted_prediction_views: np.ndarray,
) -> None:
    """Replay the production scaler/core/checkpoint and all five CUDA views."""

    artifacts = result["artifacts"]
    training_phase = _read_authenticated_phase(
        _verify_reference(run_dir, artifacts["training_phase"]),
        task_id=str(task["task_id"]),
        expected_phase="training.completed",
    )
    history_payload = read_json_object(
        _verify_reference(run_dir, artifacts["history"])
    )
    core_receipt = training_phase.get("training_core_receipt")
    if not isinstance(core_receipt, Mapping):
        raise GPUOutagePORDV5Error("production training core receipt is missing")
    checkpoint_state = _read_checkpoint(
        _verify_reference(run_dir, artifacts["checkpoint"])
    )
    if len(checkpoint_state) < 4:
        raise GPUOutagePORDV5Error("production checkpoint is structurally incomplete")
    core = {
        **dict(core_receipt),
        "history": history_payload.get("records"),
        "state_dict": checkpoint_state,
    }
    verify_pord_result_hash(core)
    required_history_fields = {
        "epoch",
        "optimizer_update_index",
        "objective_audit",
        "parameter_gradient_sha256",
        "parameter_gradient_l2",
        "parameter_gradient_tensor_count",
        "parameter_gradient_absent_names",
        "checkpoint_selection_scope",
        "outer_eval_count",
        "outer_labels_accessed",
        "selection_full_rmse_3d_mm",
        "selection_outage_rmse_3d_mm",
        "checkpoint_score",
        "checkpoint_improved",
        "best_epoch_so_far",
        "same_checkpoint_for_all_four_outages",
        "auxiliary_reconstruction_used",
    }
    if any(set(row) != required_history_fields for row in core["history"]):
        raise GPUOutagePORDV5Error("production history is a minimal/fake receipt")

    held_out = int(task["held_out_segment"])
    segments = np.asarray(development_data["segment_id"], dtype=np.int64)
    train_rows = segments != held_out
    eval_rows = ~train_rows
    persisted_prep = read_json_object(
        _verify_reference(run_dir, artifacts["preprocessing"])
    )
    replay_prep = _fit_v5_preprocessor(
        development_data, held_out_segment=held_out
    )
    _assert_json_close(
        persisted_prep, replay_prep.metadata(), path="production_preprocessor"
    )
    if np.all(
        np.asarray(persisted_prep["feature_scale"], dtype=np.float64) == 0.0
    ):
        raise GPUOutagePORDV5Error("production all-zero feature scaler is forbidden")
    X_train = replay_prep.transform_features(
        np.asarray(development_data["X"])[train_rows]
    ).astype(np.float32)
    y_train = replay_prep.transform_targets(
        np.asarray(development_data["y"])[train_rows]
    ).astype(np.float32)
    train_uids = (
        np.asarray(development_data["trial_uid"])[train_rows]
        .astype(str)
        .tolist()
    )
    base_train_views = build_exact_five_views(X_train, trial_uids=train_uids)
    train_views = (
        build_constant_availability_five_views(X_train, trial_uids=train_uids)
        if str(task["method_id"]) == "constant_availability_erm"
        else base_train_views
    )
    if (
        base_train_views.sha256 != core.get("base_train_view_bundle_sha256")
        or train_views.sha256 != core.get("train_view_bundle_sha256")
    ):
        raise GPUOutagePORDV5Error("production five-view training replay changed")
    target_mean = np.asarray(replay_prep.target_mean_, dtype=np.float64)
    target_scale = np.repeat(float(replay_prep.target_scale_), 3)
    inverse_truth = y_train.astype(np.float64) * target_scale + target_mean
    target_hash = stable_hash(
        {
            "target_mean_mm": target_mean.tolist(),
            "target_scale_mm": target_scale.tolist(),
            "train_inverse_target_sha256": array_sha256(inverse_truth),
        }
    )
    segment_labels = tuple(sorted(str(value) for value in set(segments[train_rows])))
    index_by_label = {label: index for index, label in enumerate(segment_labels)}
    segment_index = np.asarray(
        [index_by_label[str(value)] for value in segments[train_rows]], dtype=np.int64
    )
    wrong_pair_map = fixed_within_segment_circular_derangement(
        trial_uids=train_uids, segment_index=segment_index
    )
    bundle_hash = stable_hash(
        {
            "base_train_views_sha256": base_train_views.sha256,
            "train_targets_sha256": array_sha256(y_train),
            "train_segment_index_sha256": array_sha256(segment_index),
            "train_segment_labels": list(segment_labels),
            "held_out_segment": str(held_out),
            "target_transform_sha256": target_hash,
            "model_seed": int(task["model_seed"]),
            "epoch_budget": 400,
            "backbone": core["backbone"],
            "optimizer": core["optimizer"],
            "views": core["training_view_names"],
            "mask_probabilities": core["mask_probabilities"],
            "pre_registered_wrong_pair_map_sha256": wrong_pair_map.sha256,
            "checkpoint_selection_scope": PORD_CHECKPOINT_SELECTION_SCOPE,
            "outer_eval_count": 0,
        }
    )
    if (
        target_hash != core.get("target_transform_sha256")
        or wrong_pair_map.sha256
        != core.get("pre_registered_wrong_pair_map_sha256")
        or bundle_hash != core.get("training_bundle_sha256")
        or initial_pord_state_sha256(
            backbone_id=str(task["backbone_id"]),
            model_seed=int(task["model_seed"]),
        )
        != core.get("initial_state_sha256")
    ):
        raise GPUOutagePORDV5Error("production train/core semantic hash changed")

    worker = result["worker"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") == _canonical_cuda_uuid_selector(
        worker["uuid"]
    ):
        device_index = 0
    else:
        device_index = int(worker["physical_gpu_id"])
    live = _cuda_device_identity(device_index)
    if any(live[key] != worker[key] for key in ("uuid", "pci_bus_id", "name")):
        raise GPUOutagePORDV5Error("production validation GPU identity changed")
    X_outer = replay_prep.transform_features(
        np.asarray(development_data["X"])[eval_rows]
    ).astype(np.float32)
    replay = predict_pord_checkpoint_outer(
        training_result=core,
        X_outer_scaled=X_outer,
        outer_segment_ids=segments[eval_rows],
        outer_trial_uids=np.asarray(development_data["trial_uid"])[eval_rows]
        .astype(str)
        .tolist(),
        held_out_segment=held_out,
        target_mean_mm=target_mean,
        target_scale_mm=target_scale,
        device=f"cuda:{device_index}",
    )
    replay_views = np.asarray(replay["predictions_raw_mm"], dtype=np.float64)
    if not np.array_equal(replay_views, persisted_prediction_views):
        raise GPUOutagePORDV5Error(
            "production checkpoint five-view CUDA replay changed"
        )


def _assert_json_close(actual: Any, expected: Any, *, path: str = "root") -> None:
    if isinstance(expected, Mapping):
        if type(actual) is not dict or set(actual) != set(expected):
            raise GPUOutagePORDV5Error(f"JSON mapping changed at {path}")
        for key in expected:
            _assert_json_close(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if type(actual) is not list or len(actual) != len(expected):
            raise GPUOutagePORDV5Error(f"JSON list changed at {path}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_json_close(left, right, path=f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if type(actual) is not float or not math.isfinite(actual):
            raise GPUOutagePORDV5Error(f"JSON numeric type/value changed at {path}")
        value = actual
        if not math.isclose(value, expected, rel_tol=1.0e-12, abs_tol=1.0e-10):
            raise GPUOutagePORDV5Error(f"JSON numeric value changed at {path}")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise GPUOutagePORDV5Error(f"JSON value changed at {path}")


def _read_authenticated_phase(
    path: Path, *, task_id: str, expected_phase: str
) -> dict[str, Any]:
    payload = read_json_object(path)
    phase_hash = payload.get("phase_sha256")
    core = {key: value for key, value in payload.items() if key != "phase_sha256"}
    if (
        payload.get("contract")
        not in {
            "alphaf.endpoint_regression.gpu_outage_pord_task_phase.v5",
            PORD_V5_TASK_COMPLETION_CONTRACT,
        }
        or payload.get("task_id") != task_id
        or payload.get("attempt_index") != 1
        or payload.get("phase") != expected_phase
        or phase_hash != stable_hash(core)
    ):
        raise GPUOutagePORDV5Error(
            f"durable task phase changed: {task_id}/{expected_phase}"
        )
    return payload


def _validate_task_phase_chain(
    run_dir: Path,
    task: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    completion: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    """Authenticate the one-attempt, no-repeat fit/evaluation phase lineage."""

    task_id = str(task["task_id"])
    execution_path = _verify_reference(run_dir, artifacts["execution"])
    training_path = _verify_reference(run_dir, artifacts["training_phase"])
    prediction_path = _verify_reference(run_dir, artifacts["outer_prediction_phase"])
    started_path = _verify_reference(run_dir, artifacts["outer_eval_started"])
    evaluation_path = _verify_reference(run_dir, artifacts["outer_eval_phase"])
    training = _read_authenticated_phase(
        training_path, task_id=task_id, expected_phase="training.completed"
    )
    prediction = _read_authenticated_phase(
        prediction_path,
        task_id=task_id,
        expected_phase="outer_prediction.completed",
    )
    started = _read_authenticated_phase(
        started_path, task_id=task_id, expected_phase="outer_eval.started"
    )
    evaluation = _read_authenticated_phase(
        evaluation_path, task_id=task_id, expected_phase="outer_eval.completed"
    )
    if training.get("previous_phase") != {
        "file": _artifact_reference(execution_path, run_dir),
        "phase": "execution.started",
        "sha256": sha256_file(execution_path),
    }:
        raise GPUOutagePORDV5Error("training phase/execution linkage changed")
    if prediction.get("previous_phase") != _phase_link(training_path, run_dir):
        raise GPUOutagePORDV5Error("outer prediction/training linkage changed")
    if started.get("previous_phase") != _phase_link(prediction_path, run_dir):
        raise GPUOutagePORDV5Error("outer evaluation start linkage changed")
    if evaluation.get("previous_phase") != _phase_link(started_path, run_dir):
        raise GPUOutagePORDV5Error("outer evaluation completion linkage changed")
    if (
        training.get("outer_eval_count") != 0
        or training.get("checkpoint_selection_scope")
        != PORD_CHECKPOINT_SELECTION_SCOPE
        or prediction.get("outer_labels_accessed") is not False
        or started.get("outer_eval_count_before") != 0
        or started.get("maximum_outer_eval_count") != 1
        or evaluation.get("outer_eval_count") != 1
    ):
        raise GPUOutagePORDV5Error("fit/evaluation cardinality evidence changed")
    training_transport = training.get("runtime_gpu_transport")
    if training_transport != result.get("runtime_gpu"):
        raise GPUOutagePORDV5Error("training transport/execution binding changed")
    for phase_payload in (prediction, started, evaluation):
        transport = phase_payload.get("runtime_gpu_transport")
        if (
            not isinstance(transport, Mapping)
            or set(transport) != _RUNTIME_GPU_RECEIPT_KEYS
            or transport.get("transport_bound") is not True
            or transport.get("logical_worker_id")
            != result.get("worker", {}).get("logical_worker_id")
            or transport.get("cuda_visible_devices")
            != _canonical_cuda_uuid_selector(
                result.get("worker", {}).get("uuid")
            )
            or transport.get("visible_device_index") != 0
            or any(
                transport.get(key) != result.get("worker", {}).get(key)
                for key in ("physical_gpu_id", "uuid", "pci_bus_id", "name")
            )
        ):
            raise GPUOutagePORDV5Error("phase runtime GPU transport changed")
    phase_artifacts = {
        "preprocessing": training.get("artifacts", {}).get("preprocessing"),
        "history": training.get("artifacts", {}).get("history"),
        "checkpoint": training.get("artifacts", {}).get("checkpoint"),
        "outer_prediction": prediction.get("artifacts", {}).get("prediction"),
        "outer_prediction_receipt": prediction.get("artifacts", {}).get("receipt"),
        "outer_evaluation": evaluation.get("artifacts", {}).get("outer_evaluation"),
        "predictions": evaluation.get("artifacts", {}).get("predictions"),
    }
    for key, reference in phase_artifacts.items():
        if reference != artifacts.get(key):
            raise GPUOutagePORDV5Error(f"phase artifact binding changed: {key}")
    if (
        completion.get("phase") != "bundle.completed"
        or completion.get("previous_phase") != _phase_link(evaluation_path, run_dir)
        or result.get("phase") != "result"
        or result.get("previous_phase")
        != _phase_link(_verify_reference(run_dir, result["completion"]), run_dir)
    ):
        raise GPUOutagePORDV5Error("bundle/result phase linkage changed")


def _validate_task_result(
    run_dir: Path,
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    result: Mapping[str, Any] | None = None,
    development_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate every task/result/checkpoint/history/prediction binding."""

    task_dir = _task_dir(run_dir, task)
    if result is None:
        result = read_json_object(task_dir / "result.json")
    expected_result_keys = {
        "contract",
        "phase",
        "attempt_index",
        "execution_mode",
        "synthetic_test_only",
        "previous_phase",
        "config_sha256",
        "task",
        "worker",
        "training_source_fingerprint",
        "environment_fingerprint",
        "runtime_gpu",
        "training",
        "metrics",
        "artifacts",
        "challenge_labels_accessed",
        "completion",
        "result_sha256",
    }
    if (
        set(result) != expected_result_keys
        or result.get("contract") != PORD_V5_TASK_RESULT_CONTRACT
        or result.get("phase") != "result"
        or result.get("attempt_index") != 1
        or result.get("config_sha256") != pord_v5_config_sha256(config)
        or result.get("task") != dict(task)
        or result.get("challenge_labels_accessed") is not False
        or result.get("result_sha256") != _task_result_hash(result)
    ):
        raise GPUOutagePORDV5Error(f"task result binding changed: {task['task_id']}")
    source = read_json_object(run_dir / "provenance" / "source_snapshot.json")
    environment = read_json_object(run_dir / "provenance" / "environment_snapshot.json")
    if (
        result.get("training_source_fingerprint") != source.get("fingerprint")
        or result.get("environment_fingerprint") != environment.get("fingerprint")
    ):
        raise GPUOutagePORDV5Error("task source/environment binding changed")
    mode = _prepared_execution_mode(run_dir)
    runtime_gpu = result.get("runtime_gpu")
    completion_path = _verify_reference(run_dir, result["completion"])
    completion = read_json_object(completion_path)
    common_result = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "contract",
            "phase",
            "previous_phase",
            "completion",
            "result_sha256",
        }
    }
    common_completion = {
        key: value
        for key, value in completion.items()
        if key not in {"contract", "phase", "previous_phase", "phase_sha256"}
    }
    if common_completion != common_result:
        raise GPUOutagePORDV5Error("task completion/result evidence differs")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "execution",
        "preprocessing",
        "outer_evaluation",
        "history",
        "checkpoint",
        "checkpoint_metadata",
        "predictions",
        "training_phase",
        "outer_prediction_phase",
        "outer_prediction",
        "outer_prediction_receipt",
        "outer_eval_started",
        "outer_eval_phase",
    }:
        raise GPUOutagePORDV5Error("task artifact schema changed")
    paths = {key: _verify_reference(run_dir, value) for key, value in artifacts.items()}
    _validate_task_phase_chain(run_dir, task, artifacts, completion, result)
    execution = read_json_object(paths["execution"])
    if (
        set(execution)
        != {
            "contract",
            "phase",
            "attempt_index",
            "execution_mode",
            "synthetic_test_only",
            "config_sha256",
            "task",
            "worker",
            "workers_per_gpu",
            "training_source_fingerprint",
            "environment_fingerprint",
            "runtime_gpu_transport",
            "scientific_fit_identity_sha256",
            "worker_development_label_authority",
            "trainer_outer_array_surface",
            "outer_nonuse_enforcement",
            "challenge_labels_accessed",
        }
        or execution.get("contract") != PORD_V5_TASK_EXECUTION_CONTRACT
        or execution.get("phase") != "execution.started"
        or execution.get("attempt_index") != 1
        or execution.get("execution_mode") != mode
        or execution.get("synthetic_test_only") is not (
            mode == PORD_V5_SYNTHETIC_TEST_MODE
        )
        or execution.get("config_sha256") != pord_v5_config_sha256(config)
        or execution.get("task") != dict(task)
        or execution.get("worker") != result.get("worker")
        or type(execution.get("workers_per_gpu")) is not int
        or not 1 <= execution["workers_per_gpu"] <= 3
        or execution.get("training_source_fingerprint") != source["fingerprint"]
        or execution.get("environment_fingerprint") != environment["fingerprint"]
        or execution.get("runtime_gpu_transport") != runtime_gpu
        or execution.get("scientific_fit_identity_sha256") != stable_hash(task)
        or execution.get("worker_development_label_authority")
        != "full_snapshot_in_process_not_capability_isolated"
        or execution.get("trainer_outer_array_surface") != "absent"
        or execution.get("outer_nonuse_enforcement")
        != "code_order_and_durable_phase_receipts"
        or execution.get("challenge_labels_accessed") is not False
    ):
        raise GPUOutagePORDV5Error("task execution receipt changed")
    frozen_devices = _validate_gpu_devices(environment["payload"]["cuda_devices"])
    matching = [
        row
        for row in frozen_devices
        if row["physical_gpu_id"] == result["worker"].get("physical_gpu_id")
    ]
    if len(matching) != 1 or any(
        matching[0][key] != result["worker"].get(key)
        for key in ("uuid", "pci_bus_id", "name")
    ):
        raise GPUOutagePORDV5Error("task GPU UUID/PCI binding changed")
    expected_execution_kind = (
        "production_cuda_spawn_transport"
        if mode == PORD_V5_PRODUCTION_MODE
        else "synthetic_test_only_spawn_transport"
    )
    if (
        not isinstance(runtime_gpu, Mapping)
        or set(runtime_gpu) != _RUNTIME_GPU_RECEIPT_KEYS
        or runtime_gpu.get("execution_kind") != expected_execution_kind
        or runtime_gpu.get("transport_bound") is not True
        or runtime_gpu.get("logical_worker_id")
        != result["worker"].get("logical_worker_id")
        or type(runtime_gpu.get("process_id")) is not int
        or int(runtime_gpu["process_id"]) <= 0
        or runtime_gpu.get("cuda_visible_devices")
        != _canonical_cuda_uuid_selector(result["worker"].get("uuid"))
        or runtime_gpu.get("visible_device_index") != 0
        or any(
            runtime_gpu.get(key) != result["worker"].get(key)
            for key in ("physical_gpu_id", "uuid", "pci_bus_id", "name")
        )
    ):
        raise GPUOutagePORDV5Error("runtime CUDA UUID/PCI receipt changed")
    prep = read_json_object(paths["preprocessing"])
    history = read_json_object(paths["history"])
    if (
        history.get("contract")
        != "alphaf.endpoint_regression.gpu_outage_pord_history.v5"
        or history.get("task_id") != task["task_id"]
        or history.get("epoch_budget") != 400
        or history.get("update_count") != 400
        or not isinstance(history.get("records"), list)
        or len(history["records"]) != 400
        or [row.get("epoch") for row in history["records"]] != list(range(1, 401))
        or history.get("history_sha256") != stable_hash(history["records"])
    ):
        raise GPUOutagePORDV5Error("task history binding changed")
    checkpoint_state = _read_checkpoint(paths["checkpoint"])
    checkpoint_metadata = read_json_object(paths["checkpoint_metadata"])
    expected_checkpoint_keys = {
        "contract",
        "task_id",
        "candidate_id",
        "backbone_id",
        "method_id",
        "model_seed",
        "held_out_segment",
        "selected_state_sha256",
        "best_epoch",
        "best_checkpoint_score",
        "checkpoint_selection_scope",
        "checkpoint_fixed_before_outer_labels",
        "outer_eval_count",
        "outer_evaluation_sha256",
        "same_checkpoint_for_all_four_outages",
        "checkpoint_file",
    }
    training = result.get("training", {})
    if (
        set(checkpoint_metadata) != expected_checkpoint_keys
        or checkpoint_metadata.get("contract")
        != "alphaf.endpoint_regression.gpu_outage_pord_checkpoint.v5"
        or checkpoint_metadata.get("task_id") != task["task_id"]
        or checkpoint_metadata.get("candidate_id") != task["candidate_id"]
        or checkpoint_metadata.get("backbone_id") != task["backbone_id"]
        or checkpoint_metadata.get("method_id") != task["method_id"]
        or checkpoint_metadata.get("model_seed") != task["model_seed"]
        or checkpoint_metadata.get("held_out_segment") != task["held_out_segment"]
        or checkpoint_metadata.get("selected_state_sha256")
        != training.get("selected_state_sha256")
        or checkpoint_metadata.get("selected_state_sha256")
        != state_dict_sha256(checkpoint_state)
        or checkpoint_metadata.get("best_epoch") != training.get("best_epoch")
        or checkpoint_metadata.get("best_checkpoint_score")
        != training.get("best_checkpoint_score")
        or checkpoint_metadata.get("checkpoint_selection_scope")
        != PORD_CHECKPOINT_SELECTION_SCOPE
        or checkpoint_metadata.get("checkpoint_fixed_before_outer_labels") is not True
        or checkpoint_metadata.get("outer_eval_count") != 1
        or checkpoint_metadata.get("outer_evaluation_sha256")
        != training.get("outer_evaluation_sha256")
        or checkpoint_metadata.get("same_checkpoint_for_all_four_outages") is not True
        or checkpoint_metadata.get("checkpoint_file") != artifacts["checkpoint"]
    ):
        raise GPUOutagePORDV5Error("task checkpoint metadata/state binding changed")
    columns = [
        "trial_uid",
        "segment_id",
        "target_id",
        *_TRUE_COLUMNS,
        *_PREDICTION_COLUMNS,
    ]
    prediction = _read_exact_parquet(paths["predictions"], columns)
    prediction_uids, prediction_segments, prediction_target_ids = (
        _exact_prediction_identity(
            prediction,
            held_out_segment=_exact_int(
                task["held_out_segment"], name="task held_out_segment"
            ),
            name="task prediction artifact",
        )
    )
    numeric = prediction.loc[:, [*_TRUE_COLUMNS, *_PREDICTION_COLUMNS]].to_numpy(
        dtype=np.float64
    )
    if (
        len(prediction) != artifacts["predictions"].get("rows")
        or type(artifacts["predictions"].get("views")) is not int
        or artifacts["predictions"].get("views") != 5
        or not np.isfinite(numeric).all()
    ):
        raise GPUOutagePORDV5Error("task prediction artifact changed")
    truth = prediction.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
    views = np.stack(
        [
            prediction.loc[
                :,
                [f"{view}_pred_{axis}_mm" for axis in ("x", "y", "z")],
            ].to_numpy(dtype=np.float64)
            for view in _PREDICTION_VIEW_NAMES
        ],
        axis=0,
    )
    recomputed_metrics = {
        "full": regression_metrics(truth, views[0]),
        "outages": [regression_metrics(truth, views[index]) for index in range(1, 5)],
    }
    _assert_json_close(result.get("metrics"), recomputed_metrics, path="task.metrics")
    normalized = {
        "preprocessing": prep,
        "history": history["records"],
        "checkpoint_state": checkpoint_state,
        "training": training,
        "outer_evaluation": read_json_object(paths["outer_evaluation"]),
        "trial_uid": prediction_uids,
        "segment_id": prediction_segments.tolist(),
        "target_id": prediction_target_ids,
        "truth_mm": truth,
        "predictions_mm": views,
    }
    _validate_executor_output(normalized, task)
    if development_data is not None:
        _validate_holdout_source_binding(normalized, development_data, task)
        if mode == PORD_V5_PRODUCTION_MODE:
            _validate_production_task_semantics(
                run_dir=run_dir,
                task=task,
                result=result,
                development_data=development_data,
                persisted_prediction_views=views,
            )
    return dict(result)


def _quarantine_task_directory(run_dir: Path, task_dir: Path, *, reason: str) -> Path:
    recovery = run_dir / "recovery" / "development_tasks"
    recovery.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        destination = recovery / f"{task_dir.name}__attempt{attempt}"
        if not destination.exists():
            break
        attempt += 1
    intent_path = (
        run_dir
        / "recovery"
        / "quarantine_intents"
        / f"{task_dir.name}__attempt{attempt}.json"
    )
    intent = {
        "task_id": task_dir.name,
        "attempt": attempt,
        "source": _relative(task_dir, run_dir),
        "destination": _relative(destination, run_dir),
        "reason": reason,
        "operation": "nondelete_atomic_directory_move",
        "status": "intent_durable_before_move",
    }
    _exclusive_write_json(intent_path, intent)
    os.replace(task_dir, destination)
    quarantine_path = atomic_write_json(
        destination / "quarantine.json",
        {
            "task_id": task_dir.name,
            "reason": reason,
            "nondelete_quarantine": True,
            "attempt": attempt,
        },
    )
    audit = {
        **intent,
        "status": "move_completed_and_audited",
        "intent_file": _artifact_reference(intent_path, run_dir),
        "quarantine_file": _artifact_reference(quarantine_path, run_dir),
        "destination_manifest_sha256": stable_hash(manifest_files(destination)),
    }
    atomic_write_json(
        run_dir
        / "recovery"
        / "quarantine_audits"
        / f"{task_dir.name}__attempt{attempt}.json",
        audit,
    )
    return destination


def _reconcile_quarantine_intents(run_dir: Path, task_id: str) -> None:
    """Complete intent/move/audit crash windows without deleting evidence."""

    intent_root = run_dir / "recovery" / "quarantine_intents"
    if not intent_root.is_dir():
        return
    for intent_path in sorted(intent_root.glob(f"{task_id}__attempt*.json")):
        intent = read_json_object(intent_path)
        audit_path = (
            run_dir / "recovery" / "quarantine_audits" / intent_path.name
        )
        if audit_path.is_file():
            continue
        source = _safe_member(run_dir, str(intent["source"]))
        destination = _safe_member(run_dir, str(intent["destination"]))
        if source.exists() and destination.exists():
            raise GPUOutagePORDV5Error(
                "quarantine crash recovery found both source and destination"
            )
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        if not destination.is_dir():
            raise GPUOutagePORDV5Error(
                "quarantine intent has neither recoverable source nor destination"
            )
        quarantine_path = destination / "quarantine.json"
        if not quarantine_path.is_file():
            quarantine_path = atomic_write_json(
                quarantine_path,
                {
                    "task_id": intent["task_id"],
                    "reason": intent["reason"],
                    "nondelete_quarantine": True,
                    "attempt": intent["attempt"],
                    "recovered_from_crash_window": True,
                },
            )
        audit = {
            **intent,
            "status": "move_completed_and_audited",
            "intent_file": _artifact_reference(intent_path, run_dir),
            "quarantine_file": _artifact_reference(quarantine_path, run_dir),
            "destination_manifest_sha256": stable_hash(manifest_files(destination)),
        }
        atomic_write_json(audit_path, audit)


def _reconstruct_result_from_completion(
    run_dir: Path,
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    development_data: Mapping[str, Any],
) -> dict[str, Any]:
    task_dir = _task_dir(run_dir, task)
    completion_path = task_dir / "bundle.completed.json"
    completion = read_json_object(completion_path)
    if (
        completion.get("contract") != PORD_V5_TASK_COMPLETION_CONTRACT
        or completion.get("phase") != "bundle.completed"
        or completion.get("attempt_index") != 1
        or completion.get("phase_sha256")
        != stable_hash(
            {
                key: value
                for key, value in completion.items()
                if key != "phase_sha256"
            }
        )
    ):
        raise GPUOutagePORDV5Error("task completion contract changed")
    result = {
        **{key: value for key, value in completion.items() if key != "phase_sha256"},
        "contract": PORD_V5_TASK_RESULT_CONTRACT,
        "phase": "result",
        "previous_phase": _phase_link(completion_path, run_dir),
        "completion": _artifact_reference(completion_path, run_dir),
    }
    result["result_sha256"] = _task_result_hash(result)
    atomic_write_json(task_dir / "result.json", result)
    return _validate_task_result(
        run_dir, config, task, result, development_data=development_data
    )


def _worker_ledger_path(
    run_dir: Path, task: Mapping[str, Any], result: Mapping[str, Any]
) -> Path:
    worker_id = _exact_string(
        result["worker"]["logical_worker_id"], name="logical_worker_id"
    )
    return (
        run_dir
        / "runtime_auth"
        / "worker_ledgers"
        / worker_id
        / f"{task['task_id']}.json"
    )


def _worker_ledger_payload(
    run_dir: Path, task: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "attempt_index": 1,
        "terminal_phase": "result",
        "scientific_fit_identity_sha256": stable_hash(task),
        "worker": dict(result["worker"]),
        "result_sha256": result["result_sha256"],
        "result_file": _artifact_reference(
            _task_dir(run_dir, task) / "result.json", run_dir
        ),
        "challenge_labels_accessed": False,
    }


def _record_worker_ledger(
    run_dir: Path, task: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Create/reconcile a ledger only from an explicit execution/resume route."""

    path = _worker_ledger_path(run_dir, task, result)
    payload = _worker_ledger_payload(run_dir, task, result)
    if path.exists():
        if read_json_object(path) != payload:
            raise GPUOutagePORDV5Error("durable per-worker ledger changed")
    else:
        atomic_write_json(path, payload)
    return payload


def _verify_worker_ledger(
    run_dir: Path, task: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Pure verification: never create or repair runtime evidence."""

    path = _worker_ledger_path(run_dir, task, result)
    expected = _worker_ledger_payload(run_dir, task, result)
    if not path.is_file() or read_json_object(path) != expected:
        raise GPUOutagePORDV5Error("durable per-worker ledger changed")
    return expected


def _exact_prediction_identity(
    frame: pd.DataFrame, *, held_out_segment: int, name: str
) -> tuple[list[str], np.ndarray, list[str]]:
    trial_uids = frame["trial_uid"].tolist()
    target_ids = frame["target_id"].tolist()
    if any(type(value) is not str or not value for value in trial_uids):
        raise GPUOutagePORDV5Error(f"{name} trial_uid values changed type")
    if any(type(value) is not str or not value for value in target_ids):
        raise GPUOutagePORDV5Error(f"{name} target_id values changed type")
    segment_values = frame["segment_id"].to_numpy(copy=False)
    if (
        segment_values.dtype != np.dtype(np.int64)
        or segment_values.ndim != 1
        or not np.all(segment_values == held_out_segment)
        or len(set(trial_uids)) != len(trial_uids)
    ):
        raise GPUOutagePORDV5Error(f"{name} identity values changed")
    return trial_uids, segment_values, target_ids


def _recover_completed_outer_evaluation(
    *,
    run_dir: Path,
    task: Mapping[str, Any],
    core: Mapping[str, Any],
    training_path: Path,
    prediction_phase_path: Path,
    started_path: Path,
    completed_path: Path,
) -> dict[str, Any]:
    """Recover an authenticated evaluation without reading labels or evaluating."""

    task_id = _exact_string(task.get("task_id"), name="task_id")
    held_out = _exact_int(
        task.get("held_out_segment"), name="held_out_segment"
    )
    if not prediction_phase_path.is_file() or not started_path.is_file():
        raise GPUOutagePORDV5Error(
            "completed outer evaluation lacks its durable phase lineage"
        )
    prediction_phase = _read_authenticated_phase(
        prediction_phase_path,
        task_id=task_id,
        expected_phase="outer_prediction.completed",
    )
    if prediction_phase.get("previous_phase") != _phase_link(
        training_path, run_dir
    ):
        raise GPUOutagePORDV5Error("resumed prediction phase linkage changed")
    prediction_artifacts = prediction_phase.get("artifacts")
    if type(prediction_artifacts) is not dict or set(prediction_artifacts) != {
        "prediction",
        "receipt",
    }:
        raise GPUOutagePORDV5Error("resumed prediction artifact schema changed")
    prediction_frame = _read_exact_parquet(
        _verify_reference(run_dir, prediction_artifacts["prediction"]),
        ["trial_uid", "segment_id", "target_id", *_PREDICTION_COLUMNS],
    )
    prediction_uids, prediction_segments, prediction_target_ids = (
        _exact_prediction_identity(
            prediction_frame,
            held_out_segment=held_out,
            name="resumed outer prediction",
        )
    )
    prediction_values = np.stack(
        [
            prediction_frame.loc[
                :,
                [f"{view}_pred_{axis}_mm" for axis in ("x", "y", "z")],
            ].to_numpy(dtype=np.float64)
            for view in _PREDICTION_VIEW_NAMES
        ],
        axis=0,
    )
    if not np.isfinite(prediction_values).all():
        raise GPUOutagePORDV5Error("resumed outer predictions are non-finite")
    prediction_receipt = read_json_object(
        _verify_reference(run_dir, prediction_artifacts["receipt"])
    )
    if (
        prediction_phase.get("outer_labels_accessed") is not False
        or prediction_phase.get("outer_prediction_raw_mm_sha256")
        != array_sha256(prediction_values)
        or prediction_receipt.get("outer_labels_accessed") is not False
        or prediction_receipt.get("outer_trial_uid_sha256")
        != stable_hash(prediction_uids)
        or prediction_receipt.get("outer_prediction_raw_mm_sha256")
        != array_sha256(prediction_values)
    ):
        raise GPUOutagePORDV5Error("resumed prediction receipt binding changed")

    started_phase = _read_authenticated_phase(
        started_path, task_id=task_id, expected_phase="outer_eval.started"
    )
    if (
        started_phase.get("previous_phase")
        != _phase_link(prediction_phase_path, run_dir)
        or started_phase.get("maximum_outer_eval_count") != 1
        or started_phase.get("outer_eval_count_before") != 0
    ):
        raise GPUOutagePORDV5Error("resumed outer evaluation start changed")
    completed_phase = _read_authenticated_phase(
        completed_path,
        task_id=task_id,
        expected_phase="outer_eval.completed",
    )
    completed_artifacts = completed_phase.get("artifacts")
    if (
        completed_phase.get("previous_phase") != _phase_link(started_path, run_dir)
        or completed_phase.get("outer_eval_count") != 1
        or type(completed_artifacts) is not dict
        or set(completed_artifacts) != {"outer_evaluation", "predictions"}
    ):
        raise GPUOutagePORDV5Error("resumed outer completion linkage changed")
    receipt = read_json_object(
        _verify_reference(run_dir, completed_artifacts["outer_evaluation"])
    )
    evaluation_frame = _read_exact_parquet(
        _verify_reference(run_dir, completed_artifacts["predictions"]),
        [
            "trial_uid",
            "segment_id",
            "target_id",
            *_TRUE_COLUMNS,
            *_PREDICTION_COLUMNS,
        ],
    )
    evaluation_uids, evaluation_segments, evaluation_target_ids = (
        _exact_prediction_identity(
            evaluation_frame,
            held_out_segment=held_out,
            name="resumed outer evaluation",
        )
    )
    truth = evaluation_frame.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
    predictions = np.stack(
        [
            evaluation_frame.loc[
                :,
                [f"{view}_pred_{axis}_mm" for axis in ("x", "y", "z")],
            ].to_numpy(dtype=np.float64)
            for view in _PREDICTION_VIEW_NAMES
        ],
        axis=0,
    )
    if (
        evaluation_uids != prediction_uids
        or evaluation_target_ids != prediction_target_ids
        or not np.array_equal(evaluation_segments, prediction_segments)
        or not np.array_equal(predictions, prediction_values)
        or not np.isfinite(truth).all()
        or not np.isfinite(predictions).all()
        or completed_phase.get("outer_evaluation_sha256")
        != receipt.get("outer_evaluation_sha256")
    ):
        raise GPUOutagePORDV5Error("resumed outer evaluation artifacts changed")
    verify_pord_outer_evaluation(
        receipt,
        training_result=core,
        outer_trial_uids=evaluation_uids,
        truth_raw_mm=truth,
        predictions_raw_mm=predictions,
    )
    return {
        "outer": {
            "receipt": receipt,
            "truth_raw_mm": truth,
            "predictions_raw_mm": predictions,
        },
        "trial_uid": evaluation_uids,
        "segment_id": evaluation_segments,
        "target_id": evaluation_target_ids,
    }


def _resume_production_task_phases(
    *,
    run_dir: Path,
    task: Mapping[str, Any],
    data: Mapping[str, Any],
    execution_path: Path,
    runtime_gpu_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Resume after an authenticated train phase without repeating a fit/eval."""

    task_dir = execution_path.parent
    training_path = task_dir / "training.completed.json"
    prediction_phase_path = task_dir / "outer_prediction.completed.json"
    started_path = task_dir / "outer_eval.started.json"
    completed_path = task_dir / "outer_eval.completed.json"
    if not training_path.is_file():
        raise GPUOutagePORDV5Error(
            "execution.started without training completion is ambiguous; fail closed"
        )
    if started_path.is_file() and not completed_path.is_file():
        raise GPUOutagePORDV5Error(
            "outer_eval.started without authenticated completion; fail closed"
        )
    training_phase = _read_authenticated_phase(
        training_path,
        task_id=str(task["task_id"]),
        expected_phase="training.completed",
    )
    if training_phase.get("previous_phase") != {
        "file": _artifact_reference(execution_path, run_dir),
        "phase": "execution.started",
        "sha256": sha256_file(execution_path),
    }:
        raise GPUOutagePORDV5Error("resumed training/execution linkage changed")
    training_artifacts = training_phase.get("artifacts", {})
    if set(training_artifacts) != {"preprocessing", "history", "checkpoint"}:
        raise GPUOutagePORDV5Error("resumed training artifact schema changed")
    prep_path = _verify_reference(run_dir, training_artifacts["preprocessing"])
    history_path = _verify_reference(run_dir, training_artifacts["history"])
    checkpoint_path = _verify_reference(run_dir, training_artifacts["checkpoint"])
    history_payload = read_json_object(history_path)
    core_receipt = training_phase.get("training_core_receipt")
    if not isinstance(core_receipt, Mapping):
        raise GPUOutagePORDV5Error("resumed training core receipt is missing")
    core = {
        **dict(core_receipt),
        "history": history_payload.get("records"),
        "state_dict": _read_checkpoint(checkpoint_path),
    }
    verify_pord_result_hash(core)
    if (
        core.get("result_sha256") != training_phase.get("training_result_sha256")
        or core.get("selected_state_sha256")
        != training_phase.get("selected_state_sha256")
        or core.get("checkpoint_selection_scope")
        != PORD_CHECKPOINT_SELECTION_SCOPE
        or core.get("outer_eval_count") != 0
        or core.get("outer_labels_accessed") is not False
    ):
        raise GPUOutagePORDV5Error("resumed train-only core binding changed")

    preprocessor_metadata = read_json_object(prep_path)
    if completed_path.is_file():
        recovered = _recover_completed_outer_evaluation(
            run_dir=run_dir,
            task=task,
            core=core,
            training_path=training_path,
            prediction_phase_path=prediction_phase_path,
            started_path=started_path,
            completed_path=completed_path,
        )
        return _assemble_task_executor_output(
            task=task,
            preprocessor_metadata=preprocessor_metadata,
            core=core,
            outer=recovered["outer"],
            trial_uid=recovered["trial_uid"],
            segment_id=recovered["segment_id"],
            target_id=recovered["target_id"],
        )

    held_out = _exact_int(task["held_out_segment"], name="held_out_segment")
    segments = np.asarray(data["segment_id"], dtype=np.int64)
    eval_rows = segments == held_out
    # Resume authenticates and restores the already frozen train-only scaler.
    # Re-fitting it here would materialize the full label array before the
    # outer-evaluation O_EXCL marker.  Full semantic train-only scaler replay
    # remains mandatory in the completed production-result validator.
    replay_preprocessor = _preprocessor_from_metadata(preprocessor_metadata)
    target_mean = np.asarray(replay_preprocessor.target_mean_, dtype=np.float64)
    target_scale = np.repeat(float(replay_preprocessor.target_scale_), 3)
    X_holdout = replay_preprocessor.transform_features(
        np.asarray(data["X"])[eval_rows]
    ).astype(np.float32)
    callback = _production_task_phase_callback(
        run_dir=run_dir,
        task=task,
        execution_path=execution_path,
        runtime_gpu_receipt=runtime_gpu_receipt,
    )
    outer_uids = np.asarray(data["trial_uid"])[eval_rows].astype(str).tolist()
    target_ids = np.asarray(data["target_id"])[eval_rows].astype(str).tolist()
    if prediction_phase_path.is_file():
        prediction_phase = _read_authenticated_phase(
            prediction_phase_path,
            task_id=str(task["task_id"]),
            expected_phase="outer_prediction.completed",
        )
        if prediction_phase.get("previous_phase") != _phase_link(
            training_path, run_dir
        ):
            raise GPUOutagePORDV5Error("resumed prediction phase linkage changed")
        frame = _read_exact_parquet(
            _verify_reference(
                run_dir, prediction_phase["artifacts"]["prediction"]
            ),
            ["trial_uid", "segment_id", "target_id", *_PREDICTION_COLUMNS],
        )
        if (
            frame["trial_uid"].astype(str).tolist() != outer_uids
            or frame["target_id"].astype(str).tolist() != target_ids
            or not np.array_equal(
                frame["segment_id"].to_numpy(dtype=np.int64), segments[eval_rows]
            )
        ):
            raise GPUOutagePORDV5Error("resumed outer prediction identity changed")
        predictions = np.stack(
            [
                frame.loc[
                    :,
                    [f"{view}_pred_{axis}_mm" for axis in ("x", "y", "z")],
                ].to_numpy(dtype=np.float64)
                for view in _PREDICTION_VIEW_NAMES
            ],
            axis=0,
        )
        outer_prediction = {
            "receipt": read_json_object(
                _verify_reference(
                    run_dir, prediction_phase["artifacts"]["receipt"]
                )
            ),
            "trial_uids": outer_uids,
            "predictions_raw_mm": predictions,
        }
    else:
        if started_path.exists() or completed_path.exists():
            raise GPUOutagePORDV5Error("outer phase ordering changed during resume")
        outer_prediction = predict_pord_checkpoint_outer(
            training_result=core,
            X_outer_scaled=X_holdout,
            outer_segment_ids=segments[eval_rows],
            outer_trial_uids=outer_uids,
            held_out_segment=held_out,
            target_mean_mm=target_mean,
            target_scale_mm=target_scale,
            device="cuda:0",
        )
        callback(
            "outer_prediction.completed",
            {
                "outer_prediction": outer_prediction,
                "trial_uid": outer_uids,
                "segment_id": segments[eval_rows].astype(int).tolist(),
                "target_id": target_ids,
            },
        )

    callback(
        "outer_eval.started",
        {"outer_prediction_receipt": outer_prediction["receipt"]},
    )
    # The durable start marker is the first operation allowed to precede this
    # held-out-label materialization.  A crash from here until completion is
    # deliberately fail-closed on the next resume.
    truth = np.asarray(data["y"], dtype=np.float64)[eval_rows]
    # This is the one and only label-bearing outer evaluation for the fit.
    outer = evaluate_pord_outer_predictions_once(
        training_result=core,
        outer_prediction=outer_prediction,
        y_outer_raw_mm=truth,
    )
    callback(
        "outer_eval.completed",
        {
            "outer_evaluation": outer["receipt"],
            "truth_raw_mm": outer["truth_raw_mm"],
            "predictions_raw_mm": outer["predictions_raw_mm"],
        },
    )
    return _assemble_task_executor_output(
        task=task,
        preprocessor_metadata=preprocessor_metadata,
        core=core,
        outer=outer,
        trial_uid=outer_uids,
        segment_id=segments[eval_rows],
        target_id=target_ids,
    )


def _execute_or_resume_task(
    *,
    run_dir: Path,
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    task: Mapping[str, Any],
    worker: Mapping[str, Any],
    workers_per_gpu: int,
    task_executor: Callable[..., Mapping[str, Any]],
    resume: bool,
    runtime_gpu_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    task_dir = _task_dir(run_dir, task)
    _reconcile_quarantine_intents(run_dir, str(task["task_id"]))
    result_path = task_dir / "result.json"
    completion_path = task_dir / "bundle.completed.json"
    if task_dir.exists():
        if not resume:
            raise GPUOutagePORDV5Error(
                f"task directory already exists without --resume: {task['task_id']}"
            )
        if result_path.is_file():
            try:
                result = _validate_task_result(
                    run_dir, config, task, development_data=data
                )
            except Exception as exc:
                raise GPUOutagePORDV5Error(
                    "existing result is invalid; scientific refit is forbidden"
                ) from exc
            else:
                _record_worker_ledger(run_dir, task, result)
                return result
        elif completion_path.is_file():
            try:
                result = _reconstruct_result_from_completion(
                    run_dir, config, task, data
                )
            except Exception as exc:
                raise GPUOutagePORDV5Error(
                    "completed bundle is invalid; scientific refit is forbidden"
                ) from exc
            else:
                _record_worker_ledger(run_dir, task, result)
                return result
        else:
            if (
                task_executor is not _default_task_executor
                or _prepared_execution_mode(run_dir) != PORD_V5_PRODUCTION_MODE
            ):
                raise GPUOutagePORDV5Error(
                    "incomplete synthetic/custom task is ambiguous; fail closed"
                )
            execution_path = task_dir / "execution.started.json"
            execution = read_json_object(execution_path)
            if execution.get("runtime_gpu_transport") is None:
                raise GPUOutagePORDV5Error("task execution transport is missing")
            output = _resume_production_task_phases(
                run_dir=run_dir,
                task=task,
                data=data,
                execution_path=execution_path,
                runtime_gpu_receipt=runtime_gpu_receipt,
            )
            result = _materialize_task_output(
                run_dir=run_dir,
                config=config,
                task=task,
                worker=execution["worker"],
                execution_path=execution_path,
                output=output,
                data=data,
                runtime_gpu_receipt=execution["runtime_gpu_transport"],
            )
            _record_worker_ledger(run_dir, task, result)
            return result
    task_dir.mkdir(parents=True, exist_ok=False)
    execution = _task_execution_payload(
        run_dir=run_dir,
        config=config,
        task=task,
        worker=worker,
        workers_per_gpu=workers_per_gpu,
        runtime_gpu_receipt=runtime_gpu_receipt,
    )
    execution_path = _exclusive_write_json(task_dir / "execution.started.json", execution)
    if task_executor is _default_task_executor:
        phase_callback = _production_task_phase_callback(
            run_dir=run_dir,
            task=task,
            execution_path=execution_path,
            runtime_gpu_receipt=runtime_gpu_receipt,
        )
        output = task_executor(
            task=task,
            data=data,
            worker=worker,
            config=config,
            _phase_callback=phase_callback,
        )
    else:
        output = task_executor(task=task, data=data, worker=worker, config=config)
        _materialize_synthetic_test_phase_chain(
            run_dir=run_dir,
            task=task,
            execution_path=execution_path,
            output=output,
        )
    result = _materialize_task_output(
        run_dir=run_dir,
        config=config,
        task=task,
        worker=worker,
        execution_path=execution_path,
        output=output,
        data=data,
        runtime_gpu_receipt=runtime_gpu_receipt,
    )
    _record_worker_ledger(run_dir, task, result)
    return result


def verify_pord_v5_controlled_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify paired initialization/scaler/views/update/checkpoint controls."""

    if len(results) != PORD_V5_FIT_COUNT:
        raise GPUOutagePORDV5Error("controlled result table must contain 168 fits")
    by_key: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    task_ids: set[str] = set()
    for result in results:
        task = result["task"]
        task_id = str(task["task_id"])
        if task_id in task_ids:
            raise GPUOutagePORDV5Error("duplicate scientific task result")
        task_ids.add(task_id)
        key = (
            str(task["backbone_id"]),
            int(task["held_out_segment"]),
            int(task["paired_seed_index"]),
        )
        by_key.setdefault(key, []).append(result)
        training = result["training"]
        if (
            training["epoch_budget"] != 400
            or training["optimizer_update_count"] != 400
            or training["contract_epoch_budget_complete"] is not True
            or training["same_checkpoint_for_all_four_outages"] is not True
            or training["exact_five_views"] is not True
            or training["checkpoint_selection_scope"]
            != PORD_CHECKPOINT_SELECTION_SCOPE
            or training["checkpoint_fixed_before_outer_labels"] is not True
            or training["outer_eval_count"] != 1
        ):
            raise GPUOutagePORDV5Error("controlled update/view/checkpoint budget changed")
    for (backbone, _segment, _seed), rows in by_key.items():
        expected_methods = (
            set(PORD_V5_PRIMARY_METHOD_IDS)
            if backbone == PORD_V5_PRIMARY_BACKBONE_ID
            else set(PORD_V5_TRANSFER_METHOD_IDS)
        )
        if {str(row["task"]["method_id"]) for row in rows} != expected_methods:
            raise GPUOutagePORDV5Error("closed-world backbone method table changed")
        scalar_fields = (
            "initial_state_sha256",
            "training_bundle_sha256",
            "base_train_view_bundle_sha256",
            "base_outer_view_bundle_sha256",
            "target_transform_sha256",
        )
        for field in scalar_fields:
            if len({str(row["training"][field]) for row in rows}) != 1:
                raise GPUOutagePORDV5Error(
                    f"paired controlled methods differ in {field}"
                )
        if len({row["artifacts"]["preprocessing"]["sha256"] for row in rows}) != 1:
            raise GPUOutagePORDV5Error("paired controlled methods used different scalers")
    expected_groups = 2 * 4 * 3
    if len(by_key) != expected_groups:
        raise GPUOutagePORDV5Error("controlled backbone/fold/seed grid changed")
    return {
        "status": "ok",
        "fit_count": len(results),
        "backbone_fold_seed_groups": len(by_key),
        "same_init_scaler_views_updates_checkpoint_policy": True,
        "primary_method_count": 12,
        "transfer_method_count": 2,
    }


def _worker_aggregate_payload(
    results: Sequence[Mapping[str, Any]], workers_per_gpu: int
) -> dict[str, Any]:
    if type(workers_per_gpu) is not int or not 1 <= workers_per_gpu <= 3:
        raise GPUOutagePORDV5Error("workers_per_gpu changed type or range")
    rows = sorted(
        [
            {
                "task_id": result["task"]["task_id"],
                "logical_worker_id": result["worker"]["logical_worker_id"],
                "physical_gpu_id": result["worker"]["physical_gpu_id"],
                "gpu_uuid": result["worker"]["uuid"],
                "gpu_pci_bus_id": result["worker"]["pci_bus_id"],
                "result_sha256": result["result_sha256"],
                "attempt_index": result["attempt_index"],
                "terminal_phase": result["phase"],
                "scientific_fit_identity_sha256": stable_hash(result["task"]),
            }
            for result in results
        ],
        key=lambda row: row["task_id"],
    )
    return {
        "workers_per_gpu": workers_per_gpu,
        "scientific_task_count": len(rows),
        "scientific_task_ids_unique": len({row["task_id"] for row in rows}) == len(rows),
        "per_worker_ledgers_reconstructible": True,
        "tasks": rows,
    }


def _materialize_worker_aggregate(
    run_dir: Path, results: Sequence[Mapping[str, Any]], workers_per_gpu: int
) -> dict[str, Any]:
    payload = _worker_aggregate_payload(results, workers_per_gpu)
    atomic_write_json(
        run_dir / "runtime_auth" / "development_gpu_workers.json", payload
    )
    return payload


def _verify_worker_aggregate(
    run_dir: Path, results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Purely authenticate the aggregate ledger against task results."""

    path = run_dir / "runtime_auth" / "development_gpu_workers.json"
    if not path.is_file():
        raise GPUOutagePORDV5Error("aggregate worker ledger is missing")
    payload = read_json_object(path)
    workers_per_gpu = payload.get("workers_per_gpu")
    expected = _worker_aggregate_payload(results, workers_per_gpu)
    if payload != expected:
        raise GPUOutagePORDV5Error("aggregate worker ledger changed")
    return payload


def _all_tasks_sentinel_payload(
    run_dir: Path,
    results: Sequence[Mapping[str, Any]],
    worker_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        len(results) != PORD_V5_FIT_COUNT
        or len({row["task"]["task_id"] for row in results}) != PORD_V5_FIT_COUNT
    ):
        raise GPUOutagePORDV5Error("all-task sentinel requires 168 unique results")
    if (
        worker_ledger.get("scientific_task_count") != PORD_V5_FIT_COUNT
        or worker_ledger.get("scientific_task_ids_unique") is not True
    ):
        raise GPUOutagePORDV5Error("all-task sentinel requires the complete worker ledger")
    ledger_path = run_dir / "runtime_auth" / "development_gpu_workers.json"
    return {
        "status": "all_168_tasks_and_worker_ledgers_authenticated",
        "scientific_task_count": PORD_V5_FIT_COUNT,
        "result_tree_sha256": stable_hash(
            [
                {
                    "task_id": row["task"]["task_id"],
                    "result_sha256": row["result_sha256"],
                }
                for row in results
            ]
        ),
        "worker_ledger_file": _artifact_reference(ledger_path, run_dir),
        "sentinel_created_after_all_authenticated_completions": True,
    }


def _authenticate_all_tasks_sentinel(
    run_dir: Path,
    results: Sequence[Mapping[str, Any]],
    worker_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the sentinel only after all 168 result/ledger authentications."""

    payload = _all_tasks_sentinel_payload(run_dir, results, worker_ledger)
    path = run_dir / "development" / "all_tasks_authenticated.completed.json"
    if path.exists():
        if read_json_object(path) != payload:
            raise GPUOutagePORDV5Error("all-task authenticated sentinel changed")
    else:
        _exclusive_write_json(path, payload)
    return payload


def _verify_all_tasks_sentinel(
    run_dir: Path,
    results: Sequence[Mapping[str, Any]],
    worker_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Purely authenticate the completion sentinel; never materialize it."""

    path = run_dir / "development" / "all_tasks_authenticated.completed.json"
    expected = _all_tasks_sentinel_payload(run_dir, results, worker_ledger)
    if not path.is_file() or read_json_object(path) != expected:
        raise GPUOutagePORDV5Error("all-task authenticated sentinel changed")
    return expected


def _task_prediction_frame(run_dir: Path, result: Mapping[str, Any]) -> pd.DataFrame:
    return _read_exact_parquet(
        _verify_reference(run_dir, result["artifacts"]["predictions"]),
        [
            "trial_uid",
            "segment_id",
            "target_id",
            *_TRUE_COLUMNS,
            *_PREDICTION_COLUMNS,
        ],
    )


def _aggregate_prediction_context(
    *,
    run_dir: Path,
    candidate_id: str,
    segment: int,
    outage_index: int | None,
    seed_frames: Sequence[pd.DataFrame],
    output_path: Path,
    materialize: bool = True,
) -> dict[str, Any]:
    if len(seed_frames) != 3:
        raise GPUOutagePORDV5Error("context aggregation requires three paired seeds")
    identity_columns = ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS]
    base = seed_frames[0]
    if any(not frame[identity_columns].equals(base[identity_columns]) for frame in seed_frames[1:]):
        raise GPUOutagePORDV5Error("paired-seed context rows are misaligned")
    view = "full" if outage_index is None else f"drop{outage_index}"
    pred_columns = [f"{view}_pred_{axis}_mm" for axis in ("x", "y", "z")]
    truth = base.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
    predictions = [
        frame.loc[:, pred_columns].to_numpy(dtype=np.float64) for frame in seed_frames
    ]
    ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
    output = base[identity_columns].copy()
    for axis, name in enumerate(("pred_x_mm", "pred_y_mm", "pred_z_mm")):
        output[name] = ensemble[:, axis]
    if materialize:
        path = atomic_write_parquet(output_path, output)
    else:
        path = output_path
        persisted = _read_exact_parquet(path, output.columns.tolist())
        if not persisted.equals(output):
            raise GPUOutagePORDV5Error("aggregate prediction context changed")
    seed_rmse = [
        float(regression_metrics(truth, prediction)["rmse_3d_mm"])
        for prediction in predictions
    ]
    record = {
        "candidate_id": candidate_id,
        "held_out_segment": segment,
        "view": view,
        "seed_count": 3,
        "metrics": regression_metrics(truth, ensemble),
        "seed_rmse_3d_mm": seed_rmse,
        "seed_rmse_population_std_mm": float(np.std(seed_rmse, ddof=0)),
        "prediction_file": _artifact_reference(path, run_dir, rows=len(output)),
    }
    if outage_index is not None:
        record["forced_outage_index"] = outage_index
        record["context_id"] = f"segment{segment}__drop{outage_index}"
    else:
        record["context_id"] = f"segment{segment}__full"
    return record


def _single_seed_outage_context(
    *,
    candidate_id: str,
    segment: int,
    outage: int,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    truth = frame.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
    prediction = frame.loc[
        :, [f"drop{outage}_pred_{axis}_mm" for axis in ("x", "y", "z")]
    ].to_numpy(dtype=np.float64)
    return {
        "candidate_id": candidate_id,
        "held_out_segment": segment,
        "forced_outage_index": outage,
        "metrics": regression_metrics(truth, prediction),
        "seed_rmse_population_std_mm": 0.0,
    }


def _v4_incumbent_seed_map(
    run_dir: Path, snapshot: Mapping[str, Any]
) -> dict[tuple[int, int, int], pd.DataFrame]:
    output: dict[tuple[int, int, int], pd.DataFrame] = {}
    columns = [
        "trial_uid",
        "segment_id",
        "target_id",
        *_TRUE_COLUMNS,
        "pred_x_mm",
        "pred_y_mm",
        "pred_z_mm",
    ]
    for row in snapshot["v4_incumbent_seed_predictions"]:
        key = (
            _exact_int(row["held_out_segment"], name="v4 snapshot segment"),
            _exact_int(
                row["forced_outage_index"],
                name="v4 snapshot outage",
                minimum=0,
                maximum=3,
            ),
            _exact_int(
                row["paired_seed_index"],
                name="v4 snapshot seed",
                minimum=0,
                maximum=2,
            ),
        )
        output[key] = _read_exact_parquet(
            _verify_reference(run_dir, row["prediction_file"]), columns
        )
    if len(output) != 48:
        raise GPUOutagePORDV5Error("frozen v4 incumbent seed map changed")
    return output


def _aggregate_novel_selection_contexts(
    *,
    run_dir: Path,
    pord_frames: Mapping[tuple[int, int], pd.DataFrame],
    incumbent_seed_map: Mapping[tuple[int, int, int], pd.DataFrame],
    materialize: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    pord_id = f"{PORD_V5_PRIMARY_BACKBONE_ID}__paired_pord"
    blend_id = "fixed_half_v4_incumbent_half_pord"
    blend_contexts: list[dict[str, Any]] = []
    seed_contexts: dict[str, list[dict[str, Any]]] = {
        "frozen_v4_incumbent": [],
        pord_id: [],
        blend_id: [],
    }
    identity_columns = ["trial_uid", "segment_id", "target_id", *_TRUE_COLUMNS]
    for segment in PORD_V5_DEVELOPMENT_SEGMENTS:
        for outage in range(4):
            blended_seed_frames: list[pd.DataFrame] = []
            for seed in range(3):
                pord = pord_frames[(segment, seed)]
                incumbent = incumbent_seed_map[(segment, outage, seed)]
                if not pord[identity_columns].equals(incumbent[identity_columns]):
                    raise GPUOutagePORDV5Error(
                        "PORD/v4 incumbent paired rows are misaligned"
                    )
                pord_prediction = pord.loc[
                    :,
                    [f"drop{outage}_pred_{axis}_mm" for axis in ("x", "y", "z")],
                ].to_numpy(dtype=np.float64)
                incumbent_prediction = incumbent.loc[
                    :, ["pred_x_mm", "pred_y_mm", "pred_z_mm"]
                ].to_numpy(dtype=np.float64)
                frame = pord[identity_columns].copy()
                blend = 0.5 * pord_prediction + 0.5 * incumbent_prediction
                for axis, axis_name in enumerate(("x", "y", "z")):
                    frame[f"drop{outage}_pred_{axis_name}_mm"] = blend[:, axis]
                blended_seed_frames.append(frame)
                truth = frame.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
                seed_contexts["frozen_v4_incumbent"].append(
                    {
                        "candidate_id": "frozen_v4_incumbent",
                        "held_out_segment": segment,
                        "forced_outage_index": outage,
                        "paired_seed_index": seed,
                        "metrics": regression_metrics(truth, incumbent_prediction),
                    }
                )
                seed_contexts[pord_id].append(
                    {
                        "candidate_id": pord_id,
                        "held_out_segment": segment,
                        "forced_outage_index": outage,
                        "paired_seed_index": seed,
                        "metrics": regression_metrics(truth, pord_prediction),
                    }
                )
                seed_contexts[blend_id].append(
                    {
                        "candidate_id": blend_id,
                        "held_out_segment": segment,
                        "forced_outage_index": outage,
                        "paired_seed_index": seed,
                        "metrics": regression_metrics(truth, blend),
                    }
                )
            context = _aggregate_prediction_context(
                run_dir=run_dir,
                candidate_id=blend_id,
                segment=segment,
                outage_index=outage,
                seed_frames=blended_seed_frames,
                output_path=(
                    run_dir
                    / "development"
                    / "novel_contexts"
                    / blend_id
                    / f"segment{segment}__drop{outage}.parquet"
                ),
                materialize=materialize,
            )
            context["blend_weights"] = {
                "frozen_v4_incumbent": 0.5,
                pord_id: 0.5,
            }
            context["blend_space"] = "raw_vive_world_xyz_mm"
            blend_contexts.append(context)
    return blend_contexts, seed_contexts


def aggregate_pord_v5_development(
    config: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    run_dir: str | Path,
    *,
    materialize: bool = True,
) -> dict[str, Any]:
    """Materialize exact 224 outage and 56 full contexts, gates, and selection."""

    validate_pord_v5_config(config)
    run = Path(run_dir).resolve()
    verify_pord_v5_controlled_results(results)
    result_by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    frames: dict[tuple[str, int, int], pd.DataFrame] = {}
    for result in results:
        task = result["task"]
        key = (
            str(task["candidate_id"]),
            int(task["held_out_segment"]),
            int(task["paired_seed_index"]),
        )
        if key in result_by_key:
            raise GPUOutagePORDV5Error("duplicate candidate/fold/seed result")
        result_by_key[key] = result
        frames[key] = _task_prediction_frame(run, result)
    expected_keys = {
        (candidate, segment, seed)
        for candidate in PORD_V5_CANDIDATE_IDS
        for segment in PORD_V5_DEVELOPMENT_SEGMENTS
        for seed in range(3)
    }
    if set(result_by_key) != expected_keys:
        raise GPUOutagePORDV5Error("PORD-v5 candidate/fold/seed result grid changed")

    outage_contexts: list[dict[str, Any]] = []
    full_contexts: list[dict[str, Any]] = []
    seed_outage_contexts: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    full_fold_rmse: dict[str, list[float]] = {
        candidate: [] for candidate in PORD_V5_CANDIDATE_IDS
    }
    for candidate in PORD_V5_CANDIDATE_IDS:
        for segment in PORD_V5_DEVELOPMENT_SEGMENTS:
            seed_frames = [frames[(candidate, segment, seed)] for seed in range(3)]
            full = _aggregate_prediction_context(
                run_dir=run,
                candidate_id=candidate,
                segment=segment,
                outage_index=None,
                seed_frames=seed_frames,
                output_path=(
                    run
                    / "development"
                    / "full_contexts"
                    / candidate
                    / f"segment{segment}__full.parquet"
                ),
                materialize=materialize,
            )
            full_contexts.append(full)
            full_fold_rmse[candidate].append(float(full["metrics"]["rmse_3d_mm"]))
            for outage in range(4):
                outage_contexts.append(
                    _aggregate_prediction_context(
                        run_dir=run,
                        candidate_id=candidate,
                        segment=segment,
                        outage_index=outage,
                        seed_frames=seed_frames,
                        output_path=(
                            run
                            / "development"
                            / "outage_contexts"
                            / candidate
                            / f"segment{segment}__drop{outage}.parquet"
                        ),
                        materialize=materialize,
                    )
                )
                for seed, frame in enumerate(seed_frames):
                    seed_outage_contexts[seed].append(
                        _single_seed_outage_context(
                            candidate_id=candidate,
                            segment=segment,
                            outage=outage,
                            frame=frame,
                        )
                    )
    if (
        len(outage_contexts) != PORD_V5_OUTAGE_CONTEXT_COUNT
        or len(full_contexts) != PORD_V5_FULL_CONTEXT_COUNT
    ):
        raise GPUOutagePORDV5Error("PORD-v5 aggregate context counts changed")
    primary_outages = sum(
        row["candidate_id"] in PORD_V5_PRIMARY_CANDIDATE_IDS
        for row in outage_contexts
    )
    transfer_outages = sum(
        row["candidate_id"] in PORD_V5_TRANSFER_CANDIDATE_IDS
        for row in outage_contexts
    )
    if (
        primary_outages != PORD_V5_PRIMARY_OUTAGE_CONTEXT_COUNT
        or transfer_outages != PORD_V5_TRANSFER_OUTAGE_CONTEXT_COUNT
    ):
        raise GPUOutagePORDV5Error("primary/transfer outage context allocation changed")

    excess_frames: list[pd.DataFrame] = []
    excess_summaries: list[dict[str, Any]] = []
    for key, frame in frames.items():
        candidate, segment, seed = key
        truth = frame.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
        full_prediction = frame.loc[
            :, [f"full_pred_{axis}_mm" for axis in ("x", "y", "z")]
        ].to_numpy(dtype=np.float64)
        full_error = np.linalg.norm(full_prediction - truth, axis=1)
        for outage in range(4):
            outage_prediction = frame.loc[
                :,
                [f"drop{outage}_pred_{axis}_mm" for axis in ("x", "y", "z")],
            ].to_numpy(dtype=np.float64)
            outage_error = np.linalg.norm(outage_prediction - truth, axis=1)
            positive = np.maximum(outage_error - full_error, 0.0)
            summary = row_positive_excess_summary(
                truth, full_prediction, outage_prediction
            )
            excess_summaries.append(
                {
                    "candidate_id": candidate,
                    "held_out_segment": segment,
                    "paired_seed_index": seed,
                    "forced_outage_index": outage,
                    **summary,
                }
            )
            excess_frames.append(
                pd.DataFrame(
                    {
                        "candidate_id": candidate,
                        "held_out_segment": segment,
                        "paired_seed_index": seed,
                        "forced_outage_index": outage,
                        "trial_uid": frame["trial_uid"].astype(str),
                        "full_error_3d_mm": full_error,
                        "outage_error_3d_mm": outage_error,
                        "positive_excess_mm": positive,
                    }
                )
            )
    excess_output = pd.concat(excess_frames, ignore_index=True)
    excess_path = run / "development" / "row_positive_excess.parquet"
    if materialize:
        atomic_write_parquet(excess_path, excess_output)
    else:
        persisted_excess = _read_exact_parquet(
            excess_path, excess_output.columns.tolist()
        )
        if not persisted_excess.equals(excess_output):
            raise GPUOutagePORDV5Error("paired row excess artifact changed")
    excess_summary_payload = {
        "definition": "max(outage_row_3d_error_mm-full_row_3d_error_mm,0)",
        "summary_count": len(excess_summaries),
        "summaries": excess_summaries,
        "row_file": _artifact_reference(excess_path, run),
    }
    excess_summary_path = run / "development" / "row_positive_excess_summaries.json"
    if materialize:
        atomic_write_json(excess_summary_path, excess_summary_payload)
    else:
        _assert_json_close(
            read_json_object(excess_summary_path),
            excess_summary_payload,
            path="row_positive_excess_summaries",
        )
    if len(excess_summaries) != PORD_V5_ROW_EXCESS_SUMMARY_COUNT:
        raise GPUOutagePORDV5Error("paired row excess summary grid changed")

    score = score_pord_v5_candidates(outage_contexts)
    seed_scores = {
        seed: score_pord_v5_candidates(seed_outage_contexts[seed])
        for seed in range(3)
    }
    freeze = read_json_object(run / "parent.freeze.json")
    snapshot = freeze["snapshot"]
    frozen_masked = snapshot["v3_same_backbone_full_baselines"][
        "masked_mlp_64x64_relu"
    ]["fold_rmse_3d_mm"]
    gates = evaluate_pord_v5_claim_gates(
        score_trace=score["trace"],
        seed_score_traces={seed: seed_scores[seed]["trace"] for seed in range(3)},
        full_fold_rmse_by_candidate=full_fold_rmse,
        frozen_v3_masked_full_fold_rmse=frozen_masked,
    )
    pord_id = f"{PORD_V5_PRIMARY_BACKBONE_ID}__paired_pord"
    erm_id = f"{PORD_V5_PRIMARY_BACKBONE_ID}__masked_erm"
    chain_pord_id = f"{PORD_V5_TRANSFER_BACKBONE_ID}__paired_pord"
    chain_erm_id = f"{PORD_V5_TRANSFER_BACKBONE_ID}__masked_erm"
    full_diagnostics = {
        "primary_pord_vs_frozen_v3_same_backbone": full_sensor_numeric_gate(
            candidate_fold_rmse_3d_mm=full_fold_rmse[pord_id],
            baseline_fold_rmse_3d_mm=frozen_masked,
            baseline_name="frozen_v3_same_backbone",
            absolute_numeric_tolerance_mm=1.0e-9,
        ),
        "primary_pord_vs_same_v5_masked_erm": full_sensor_numeric_gate(
            candidate_fold_rmse_3d_mm=full_fold_rmse[pord_id],
            baseline_fold_rmse_3d_mm=full_fold_rmse[erm_id],
            baseline_name="same_v5_backbone_masked_erm",
            absolute_numeric_tolerance_mm=0.0,
        ),
        "transfer_pord_vs_frozen_v3_same_backbone": full_sensor_numeric_gate(
            candidate_fold_rmse_3d_mm=full_fold_rmse[chain_pord_id],
            baseline_fold_rmse_3d_mm=snapshot["v3_same_backbone_full_baselines"][
                "chain_endpoint_only"
            ]["fold_rmse_3d_mm"],
            baseline_name="frozen_v3_same_backbone",
            absolute_numeric_tolerance_mm=1.0e-9,
        ),
        "transfer_pord_vs_same_v5_masked_erm": full_sensor_numeric_gate(
            candidate_fold_rmse_3d_mm=full_fold_rmse[chain_pord_id],
            baseline_fold_rmse_3d_mm=full_fold_rmse[chain_erm_id],
            baseline_name="same_v5_backbone_masked_erm",
            absolute_numeric_tolerance_mm=0.0,
        ),
    }
    incumbent_seed_map = _v4_incumbent_seed_map(run, snapshot)
    pord_frames = {
        (segment, seed): frames[(pord_id, segment, seed)]
        for segment in PORD_V5_DEVELOPMENT_SEGMENTS
        for seed in range(3)
    }
    blend_contexts, novel_seed_contexts = _aggregate_novel_selection_contexts(
        run_dir=run,
        pord_frames=pord_frames,
        incumbent_seed_map=incumbent_seed_map,
        materialize=materialize,
    )
    incumbent_contexts = snapshot["v4_incumbent_contexts"]
    incumbent_score = score_pord_v5_candidates(
        incumbent_contexts, candidate_ids=("frozen_v4_incumbent",)
    )
    pord_contexts = [row for row in outage_contexts if row["candidate_id"] == pord_id]
    pord_score = score_pord_v5_candidates(pord_contexts, candidate_ids=(pord_id,))
    blend_score = score_pord_v5_candidates(
        blend_contexts,
        candidate_ids=("fixed_half_v4_incumbent_half_pord",),
    )
    selection = select_pord_v5_novel_candidate(
        claim_gate_record=gates,
        incumbent_trace=incumbent_score["trace"][0],
        pord_trace=pord_score["trace"][0],
        blend_trace=blend_score["trace"][0],
    )
    selection["development_trace"] = {
        "frozen_v4_incumbent": incumbent_score["trace"][0],
        pord_id: pord_score["trace"][0],
        "fixed_half_v4_incumbent_half_pord": blend_score["trace"][0],
    }
    selection["v4_frozen_winner_component_id"] = snapshot[
        "v4_selected_candidate_id"
    ]
    selection["challenge_values_can_trigger_reselection"] = False
    contexts_payload = {
            "config_sha256": pord_v5_config_sha256(config),
            "outage_context_count": len(outage_contexts),
            "primary_outage_context_count": primary_outages,
            "transfer_outage_context_count": transfer_outages,
            "full_context_count": len(full_contexts),
            "outage_contexts": outage_contexts,
            "full_contexts": full_contexts,
            "blend_contexts": blend_contexts,
        }
    scores_payload = {
            "all_fourteen_rows": score,
            "paired_seed_scores": {str(seed): seed_scores[seed] for seed in range(3)},
            "incumbent_score": incumbent_score,
            "pord_score": pord_score,
            "blend_score": blend_score,
            "full_fold_rmse_by_candidate": full_fold_rmse,
            "full_diagnostics": full_diagnostics,
        }
    honest_report_payload = {
            "tail_only_ablation": next(
                row
                for row in score["trace"]
                if row["candidate_id"]
                == f"{PORD_V5_PRIMARY_BACKBONE_ID}__paired_tail"
            ),
            "paired_interaction_ablation": next(
                row
                for row in score["trace"]
                if row["candidate_id"]
                == f"{PORD_V5_PRIMARY_BACKBONE_ID}__paired_interaction"
            ),
            "chain_transfer_rows": [
                row
                for row in score["trace"]
                if row["candidate_id"] in PORD_V5_TRANSFER_CANDIDATE_IDS
            ],
            "used_as_primary_claim_gate": False,
        }
    contexts_path = run / "development" / "context_metrics.json"
    score_path = run / "development" / "scores.json"
    gates_path = run / "development" / "claim_gates.json"
    honest_report_path = run / "development" / "separate_honest_report.json"
    evidence_payloads = (
        (contexts_path, contexts_payload, "context_metrics"),
        (score_path, scores_payload, "scores"),
        (gates_path, gates, "claim_gates"),
        (honest_report_path, honest_report_payload, "separate_honest_report"),
    )
    for path, payload, name in evidence_payloads:
        if materialize:
            atomic_write_json(path, payload)
        else:
            _assert_json_close(read_json_object(path), payload, path=name)
    selection["evidence"] = {
        "context_metrics": _artifact_reference(contexts_path, run),
        "scores": _artifact_reference(score_path, run),
        "claim_gates": _artifact_reference(gates_path, run),
        "row_positive_excess_summaries": _artifact_reference(
            excess_summary_path, run
        ),
        "separate_honest_report": _artifact_reference(honest_report_path, run),
    }
    controls_path = run / "development" / "controlled_pairing_audit.json"
    if controls_path.is_file():
        selection["evidence"]["controlled_pairing_audit"] = _artifact_reference(
            controls_path, run
        )
    selection_path = run / "development" / "selection.json"
    if materialize:
        atomic_write_json(selection_path, selection)
    else:
        _assert_json_close(
            read_json_object(selection_path), selection, path="selection"
        )
    return {
        "outage_contexts": outage_contexts,
        "full_contexts": full_contexts,
        "score": score,
        "seed_scores": seed_scores,
        "claim_gates": gates,
        "selection": selection,
        "selection_file": _artifact_reference(selection_path, run),
        "novel_seed_contexts": novel_seed_contexts,
    }


def _development_manifest_files(run_dir: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in manifest_files(run_dir)
        if row["path"] not in {"development_manifest.json", "manifest.json"}
        and not row["path"].startswith("challenge/")
    ]


def _write_pord_v5_development_manifest(
    run_dir: Path,
    config: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    selection = aggregate["selection"]
    payload = {
        "contract": PORD_V5_RUN_CONTRACT,
        "experiment_contract": PORD_V5_CONTRACT,
        "stage": str(selection["stage"]),
        "config_sha256": pord_v5_config_sha256(config),
        "counts": {
            "controlled_fit_count": PORD_V5_FIT_COUNT,
            "primary_fit_count": 144,
            "transfer_fit_count": 24,
            "primary_outage_contexts": PORD_V5_PRIMARY_OUTAGE_CONTEXT_COUNT,
            "transfer_outage_contexts": PORD_V5_TRANSFER_OUTAGE_CONTEXT_COUNT,
            "outage_contexts_total": PORD_V5_OUTAGE_CONTEXT_COUNT,
            "full_contexts_total": PORD_V5_FULL_CONTEXT_COUNT,
            "row_positive_excess_summaries": PORD_V5_ROW_EXCESS_SUMMARY_COUNT,
            "challenge_label_access_passes": 0,
        },
        "claim_gates_passed": bool(
            aggregate["claim_gates"]["all_claim_gates_passed"]
        ),
        "selection": {
            "selected_candidate_id": selection.get("selected_candidate_id"),
            "challenge_finalize_allowed": bool(
                selection["challenge_finalize_allowed"]
            ),
            "challenge_used_for_selection": False,
        },
        "challenge_labels_sealed": True,
        "files": _development_manifest_files(run_dir),
    }
    atomic_write_json(run_dir / "development_manifest.json", payload)
    return payload


def run_pord_v5_development(
    run_dir: str | Path,
    *,
    workers_per_gpu: int = 1,
    resume: bool = False,
    task_executor: Callable[..., Mapping[str, Any]] = _default_task_executor,
    parallel: bool = True,
    verify_current_source: bool = True,
    _private_test_only_mode: bool = False,
) -> dict[str, Any]:
    """Run/resume the exact 168 fits and stop with challenge labels sealed."""

    run = Path(run_dir).resolve()
    mode = _require_mode_authority(
        run, private_test_only=_private_test_only_mode
    )
    if task_executor is not _default_task_executor and mode != PORD_V5_SYNTHETIC_TEST_MODE:
        raise GPUOutagePORDV5Error("custom task executors are private test-only")
    validate_pord_v5_prepared_run(
        run,
        verify_current_source=verify_current_source,
        allow_synthetic_test_only=_private_test_only_mode,
        verify_live_environment=not _private_test_only_mode,
    )
    config = read_json_object(run / "config.resolved.json")
    validate_pord_v5_config(config)
    if (run / "challenge" / "label_access_1.started.json").exists():
        raise GPUOutagePORDV5Error("development cannot run after challenge access started")
    if (run / "development_manifest.json").is_file():
        if not resume:
            raise GPUOutagePORDV5Error("development is complete; use --resume")
        recovery_data = _load_development_snapshot(run)
        for recovery_task in pord_v5_development_tasks(config):
            recovery_dir = _task_dir(run, recovery_task)
            if not (recovery_dir / "result.json").is_file():
                if not (recovery_dir / "bundle.completed.json").is_file():
                    raise GPUOutagePORDV5Error(
                        "completed development manifest has a started-only task; fail closed"
                    )
                _reconstruct_result_from_completion(
                    run, config, recovery_task, recovery_data
                )
        recovery_tasks = pord_v5_development_tasks(config)
        recovered_results = [
            _validate_task_result(
                run, config, task, development_data=recovery_data
            )
            for task in recovery_tasks
        ]
        worker_counts = {
            _exact_int(
                read_json_object(
                    _verify_reference(run, result["artifacts"]["execution"])
                )["workers_per_gpu"],
                name="recovered workers_per_gpu",
                minimum=1,
                maximum=3,
            )
            for result in recovered_results
        }
        if len(worker_counts) != 1:
            raise GPUOutagePORDV5Error(
                "durable task receipts disagree on workers_per_gpu"
            )
        # Only this explicit --resume execution route may reconstruct missing
        # runtime-auth evidence.  Standalone validators below are pure.
        for task, result in zip(recovery_tasks, recovered_results, strict=True):
            _record_worker_ledger(run, task, result)
        aggregate_ledger_path = run / "runtime_auth" / "development_gpu_workers.json"
        if not aggregate_ledger_path.is_file():
            worker_ledger = _materialize_worker_aggregate(
                run, recovered_results, worker_counts.pop()
            )
        else:
            worker_ledger = _verify_worker_aggregate(run, recovered_results)
        sentinel_path = run / "development" / "all_tasks_authenticated.completed.json"
        if not sentinel_path.is_file():
            _authenticate_all_tasks_sentinel(
                run, recovered_results, worker_ledger
            )
        else:
            _verify_all_tasks_sentinel(run, recovered_results, worker_ledger)
        return validate_pord_v5_development_run(
            run,
            verify_current_source=verify_current_source,
            allow_synthetic_test_only=_private_test_only_mode,
            verify_live_environment=not _private_test_only_mode,
        )
    data = _load_development_snapshot(run)
    environment = read_json_object(run / "provenance" / "environment_snapshot.json")
    devices = _validate_gpu_devices(environment["payload"]["cuda_devices"])
    tasks = pord_v5_development_tasks(config)
    assignments = assign_pord_v5_tasks(
        tasks,
        gpu_devices=devices,
        workers_per_gpu=workers_per_gpu,
    )

    if parallel is not True:
        raise GPUOutagePORDV5Error(
            "fresh v5 development requires the fixed spawn slot scheduler"
        )
    results, scheduler = _run_spawn_slot_assignments(
        run_dir=run,
        config=config,
        data=data,
        assignments=assignments,
        workers_per_gpu=workers_per_gpu,
        task_executor=task_executor,
        resume=resume,
        synthetic_test_only=mode == PORD_V5_SYNTHETIC_TEST_MODE,
    )
    atomic_write_json(
        run / "runtime_auth" / "spawn_process_workers.json", scheduler
    )
    if [result["task"]["task_id"] for result in results] != [
        task["task_id"] for task in tasks
    ]:
        raise GPUOutagePORDV5Error("development result order/task grid changed")
    controls = verify_pord_v5_controlled_results(results)
    controls_path = atomic_write_json(
        run / "development" / "controlled_pairing_audit.json", controls
    )
    worker_ledger = _materialize_worker_aggregate(run, results, workers_per_gpu)
    _authenticate_all_tasks_sentinel(run, results, worker_ledger)
    aggregate = aggregate_pord_v5_development(config, results, run)
    aggregate["selection"]["evidence"]["controlled_pairing_audit"] = (
        _artifact_reference(controls_path, run)
    )
    atomic_write_json(
        run / "development" / "selection.json", aggregate["selection"]
    )
    manifest = _write_pord_v5_development_manifest(run, config, aggregate)
    validation = validate_pord_v5_development_run(
        run,
        verify_current_source=verify_current_source,
        allow_synthetic_test_only=_private_test_only_mode,
        verify_live_environment=not _private_test_only_mode,
    )
    return {
        "status": "ok",
        "manifest": manifest,
        "validation": validation,
        "worker_ledger": worker_ledger,
        "selection": aggregate["selection"],
    }


def _assert_task_directory_closed_world(run: Path, task: Mapping[str, Any]) -> None:
    directory = _task_dir(run, task)
    expected = {
        "execution.started.json",
        "training.completed.json",
        "preprocessing.json",
        "outer_prediction.completed.json",
        "outer_predictions.parquet",
        "outer_prediction_receipt.json",
        "outer_eval.started.json",
        "outer_eval.completed.json",
        "outer_evaluation.json",
        "history.json",
        "checkpoint.npz",
        "checkpoint.json",
        "predictions.parquet",
        "bundle.completed.json",
        "result.json",
    }
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != expected:
        raise GPUOutagePORDV5Error(
            f"task directory is not closed-world: {task['task_id']}"
        )


def validate_pord_v5_development_run(
    run_dir: str | Path,
    *,
    verify_current_source: bool = True,
    allow_completed_challenge: bool = False,
    allow_synthetic_test_only: bool = False,
    verify_live_environment: bool = True,
) -> dict[str, Any]:
    """Purely replay durable evidence without creating or repairing files."""

    run = Path(run_dir).resolve()
    mode = _prepared_execution_mode(run)
    validate_pord_v5_prepared_run(
        run,
        verify_current_source=verify_current_source,
        allow_synthetic_test_only=allow_synthetic_test_only,
        verify_live_environment=verify_live_environment,
    )
    config = read_json_object(run / "config.resolved.json")
    tasks = pord_v5_development_tasks(config)
    development_data = _load_development_snapshot(run)
    task_root = run / "development" / "tasks"
    if not task_root.is_dir() or {path.name for path in task_root.iterdir()} != {
        task["task_id"] for task in tasks
    }:
        raise GPUOutagePORDV5Error("development task directory grid changed")
    results: list[dict[str, Any]] = []
    for task in tasks:
        _assert_task_directory_closed_world(run, task)
        result = _validate_task_result(
            run, config, task, development_data=development_data
        )
        _verify_worker_ledger(run, task, result)
        results.append(result)
    controls = verify_pord_v5_controlled_results(results)
    if read_json_object(run / "development" / "controlled_pairing_audit.json") != controls:
        raise GPUOutagePORDV5Error("controlled pairing audit changed")
    worker = _verify_worker_aggregate(run, results)
    sentinel = _verify_all_tasks_sentinel(run, results, worker)
    if sentinel.get("sentinel_created_after_all_authenticated_completions") is not True:
        raise GPUOutagePORDV5Error("all-task completion sentinel changed")
    # Full semantic replay from the 168 task predictions: this re-derives all
    # 224 outage contexts, 56 full contexts, 672 row-excess summaries, paired
    # seed scores, gates, diagnostics and final development selection without
    # rewriting evidence.
    replayed_aggregate = aggregate_pord_v5_development(
        config, results, run, materialize=False
    )
    contexts = read_json_object(run / "development" / "context_metrics.json")
    if (
        contexts.get("config_sha256") != pord_v5_config_sha256(config)
        or contexts.get("outage_context_count") != PORD_V5_OUTAGE_CONTEXT_COUNT
        or contexts.get("primary_outage_context_count")
        != PORD_V5_PRIMARY_OUTAGE_CONTEXT_COUNT
        or contexts.get("transfer_outage_context_count")
        != PORD_V5_TRANSFER_OUTAGE_CONTEXT_COUNT
        or contexts.get("full_context_count") != PORD_V5_FULL_CONTEXT_COUNT
        or len(contexts.get("outage_contexts", ()))
        != PORD_V5_OUTAGE_CONTEXT_COUNT
        or len(contexts.get("full_contexts", ())) != PORD_V5_FULL_CONTEXT_COUNT
        or len(contexts.get("blend_contexts", ())) != 16
    ):
        raise GPUOutagePORDV5Error("durable v5 context count/grid changed")
    for row in [
        *contexts["outage_contexts"],
        *contexts["full_contexts"],
        *contexts["blend_contexts"],
    ]:
        _verify_reference(run, row["prediction_file"])
    recomputed_score = score_pord_v5_candidates(contexts["outage_contexts"])
    scores = read_json_object(run / "development" / "scores.json")
    _assert_json_close(scores["all_fourteen_rows"], recomputed_score, path="scores")
    for seed in range(3):
        if str(seed) not in scores.get("paired_seed_scores", {}):
            raise GPUOutagePORDV5Error("paired seed score evidence is missing")
    row_excess = read_json_object(
        run / "development" / "row_positive_excess_summaries.json"
    )
    if (
        row_excess.get("summary_count") != PORD_V5_ROW_EXCESS_SUMMARY_COUNT
        or len(row_excess.get("summaries", ()))
        != PORD_V5_ROW_EXCESS_SUMMARY_COUNT
    ):
        raise GPUOutagePORDV5Error("row positive-excess evidence changed")
    _verify_reference(run, row_excess["row_file"])
    gates = read_json_object(run / "development" / "claim_gates.json")
    if set(gates.get("gates", {})) != {
        "mask_augmentation_masked_erm_strict_j_lower_than_full_gradient_control",
        "strict_j_lower_than_masked_erm",
        "strict_worst_sensor_mean_lower_than_masked_erm",
        "strict_j_lower_than_group_mean_pord",
        "strict_j_lower_than_paired_tail",
        "strict_j_lower_than_paired_interaction",
        "strict_j_lower_than_paired_no_stopgrad",
        "strict_j_lower_than_wrong_pair",
        "strict_j_lower_than_groupdro",
        "strict_j_lower_than_vrex",
        "strict_j_lower_than_mofe",
        "held_out_segment_mean_improvement_at_least_3_of_4",
        "paired_seed_j_improvement_at_least_2_of_3",
        "frozen_v3_same_backbone_full_gate",
        "same_v5_masked_erm_full_gate",
    } or gates.get("all_claim_gates_passed") is not all(gates["gates"].values()):
        raise GPUOutagePORDV5Error("claim gate evidence changed")
    selection = read_json_object(run / "development" / "selection.json")
    allowed_stages = {
        "claim_gates_failed_challenge_sealed",
        "novelty_gate_failed_challenge_sealed",
        "development_selected_challenge_labels_sealed",
    }
    if (
        selection.get("stage") not in allowed_stages
        or selection.get("challenge_used_for_selection") is not False
        or selection.get("challenge_values_can_trigger_reselection") is not False
        or bool(selection.get("challenge_finalize_allowed"))
        != (selection.get("stage") == "development_selected_challenge_labels_sealed")
    ):
        raise GPUOutagePORDV5Error("development selection lifecycle changed")
    for reference in selection.get("evidence", {}).values():
        _verify_reference(run, reference)
    started = (run / "challenge" / "label_access_1.started.json").exists()
    completed = (run / "challenge" / "label_access_1.completed.json").exists()
    if started and (not allow_completed_challenge or not completed):
        raise GPUOutagePORDV5Error("challenge label access is not sealed")
    manifest = read_json_object(run / "development_manifest.json")
    expected_counts = {
        "controlled_fit_count": PORD_V5_FIT_COUNT,
        "primary_fit_count": 144,
        "transfer_fit_count": 24,
        "primary_outage_contexts": PORD_V5_PRIMARY_OUTAGE_CONTEXT_COUNT,
        "transfer_outage_contexts": PORD_V5_TRANSFER_OUTAGE_CONTEXT_COUNT,
        "outage_contexts_total": PORD_V5_OUTAGE_CONTEXT_COUNT,
        "full_contexts_total": PORD_V5_FULL_CONTEXT_COUNT,
        "row_positive_excess_summaries": PORD_V5_ROW_EXCESS_SUMMARY_COUNT,
        "challenge_label_access_passes": 0,
    }
    manifest_selection = manifest.get("selection")
    if (
        set(manifest)
        != {
            "contract",
            "experiment_contract",
            "stage",
            "config_sha256",
            "counts",
            "claim_gates_passed",
            "selection",
            "challenge_labels_sealed",
            "files",
        }
        or manifest.get("contract") != PORD_V5_RUN_CONTRACT
        or manifest.get("experiment_contract") != PORD_V5_CONTRACT
        or manifest.get("stage") != selection["stage"]
        or manifest.get("config_sha256") != pord_v5_config_sha256(config)
        or type(manifest.get("counts")) is not dict
        or set(manifest["counts"]) != set(expected_counts)
        or any(
            type(manifest["counts"].get(key)) is not int
            or manifest["counts"].get(key) != expected
            for key, expected in expected_counts.items()
        )
        or manifest.get("claim_gates_passed")
        is not bool(gates["all_claim_gates_passed"])
        or type(manifest_selection) is not dict
        or set(manifest_selection)
        != {
            "selected_candidate_id",
            "challenge_finalize_allowed",
            "challenge_used_for_selection",
        }
        or manifest_selection.get("selected_candidate_id")
        != selection.get("selected_candidate_id")
        or manifest_selection.get("challenge_finalize_allowed")
        is not bool(selection["challenge_finalize_allowed"])
        or manifest_selection.get("challenge_used_for_selection") is not False
        or manifest.get("challenge_labels_sealed") is not True
        or type(manifest.get("files")) is not list
        or manifest.get("files") != _development_manifest_files(run)
    ):
        raise GPUOutagePORDV5Error("development manifest changed")
    return {
        "status": "ok",
        "stage": selection["stage"],
        "config_sha256": pord_v5_config_sha256(config),
        "controlled_fit_count": PORD_V5_FIT_COUNT,
        "primary_outage_contexts": PORD_V5_PRIMARY_OUTAGE_CONTEXT_COUNT,
        "transfer_outage_contexts": PORD_V5_TRANSFER_OUTAGE_CONTEXT_COUNT,
        "outage_contexts_total": PORD_V5_OUTAGE_CONTEXT_COUNT,
        "full_contexts_total": PORD_V5_FULL_CONTEXT_COUNT,
        "claim_gates_passed": bool(gates["all_claim_gates_passed"]),
        "selected_candidate_id": selection.get("selected_candidate_id"),
        "challenge_finalize_allowed": bool(selection["challenge_finalize_allowed"]),
        "challenge_labels_sealed": True,
        "challenge_label_access_passes": 0,
        "source_archives_opened_by_validator": False,
        "semantic_replay_from_task_predictions": True,
        "semantic_replay_selected_candidate_id": replayed_aggregate[
            "selection"
        ].get("selected_candidate_id"),
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
    }


def _preprocessor_from_metadata(metadata: Mapping[str, Any]) -> TrainOnlyPreprocessor:
    try:
        preprocessor = TrainOnlyPreprocessor(
            feature_names=tuple(str(value) for value in metadata["feature_names"]),
            scale_features=True,
            split_context=str(metadata["leakage_audit"]["split_context"]),
        )
        preprocessor.feature_mean_ = np.asarray(
            metadata["feature_mean"], dtype=np.float64
        )
        preprocessor.feature_scale_ = np.asarray(
            metadata["feature_scale"], dtype=np.float64
        )
        preprocessor.target_mean_ = np.asarray(
            metadata["target_mean"], dtype=np.float64
        )
        preprocessor.target_scale_ = float(metadata["target_scale"])
        preprocessor.leakage_audit_ = dict(metadata["leakage_audit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GPUOutagePORDV5Error("frozen preprocessor metadata is malformed") from exc
    if tuple(preprocessor.feature_names) != FUSION_COLUMNS or not preprocessor.is_fitted:
        raise GPUOutagePORDV5Error("frozen preprocessor state/layout changed")
    return preprocessor


def _predict_pord_challenge_component(
    *, run_dir: Path, identity: Mapping[str, Any], device: str
) -> np.ndarray:
    import torch

    config = read_json_object(run_dir / "config.resolved.json")
    pord_id = f"{PORD_V5_PRIMARY_BACKBONE_ID}__paired_pord"
    predictions: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    for task in pord_v5_development_tasks(config):
        if task["candidate_id"] != pord_id:
            continue
        result = _validate_task_result(run_dir, config, task)
        prep = _preprocessor_from_metadata(
            read_json_object(_verify_reference(run_dir, result["artifacts"]["preprocessing"]))
        )
        X_scaled = prep.transform_features(np.asarray(identity["X"])).astype(np.float32)
        views = build_exact_five_views(
            X_scaled, trial_uids=np.asarray(identity["trial_uid"]).astype(str).tolist()
        )
        features = torch.as_tensor(views.features[2], dtype=torch.float32, device=device)
        model = build_pord_backbone(
            torch, backbone_id=PORD_V5_PRIMARY_BACKBONE_ID
        ).to(torch.device(device))
        state = _read_checkpoint(
            _verify_reference(run_dir, result["artifacts"]["checkpoint"])
        )
        try:
            model.load_state_dict(
                {
                    name: torch.as_tensor(value, device=torch.device(device))
                    for name, value in state.items()
                },
                strict=True,
            )
        except RuntimeError as exc:
            raise GPUOutagePORDV5Error("PORD challenge checkpoint is incompatible") from exc
        model.eval()
        with torch.inference_mode():
            scaled = model(features).detach().cpu().numpy().astype(np.float64)
        prediction = prep.inverse_targets(scaled)
        if prediction.shape != (3295, 3) or not np.isfinite(prediction).all():
            raise GPUOutagePORDV5Error("PORD challenge prediction is invalid")
        predictions.append(prediction)
        seen.add((int(task["held_out_segment"]), int(task["paired_seed_index"])))
        del model
    expected = {
        (segment, seed)
        for segment in PORD_V5_DEVELOPMENT_SEGMENTS
        for seed in range(3)
    }
    if seen != expected or len(predictions) != 12:
        raise GPUOutagePORDV5Error("PORD challenge ensemble is not 4-fold x 3-seed")
    return np.mean(np.stack(predictions, axis=0), axis=0)


def _predict_v3_incumbent_challenge(
    *, v3_run: Path, identity: Mapping[str, Any], device: str
) -> np.ndarray:
    from .gpu_outage_gbdt_v4_experiment import (
        _load_v3_config,
        _preprocessor_from_metadata as _v4_preprocessor_from_metadata,
        _read_v3_results,
        _selected_v3_components,
    )
    from .gpu_outage_training import predict_outage_state

    v3_config = _load_v3_config(v3_run / "config.resolved.json")
    selection = read_json_object(v3_run / "development" / "selection.json")
    components = _selected_v3_components(
        v3_config, str(selection["selected_candidate_id"])
    )
    results, _ = _read_v3_results(v3_run, v3_config)
    by_key = {
        (
            str(result["task"]["candidate_id"]),
            int(result["task"]["held_out_segment"]),
            int(result["task"]["forced_outage_index"]),
            int(result["task"]["paired_seed_index"]),
        ): result
        for result in results
    }
    predictions: list[np.ndarray] = []
    for segment in PORD_V5_DEVELOPMENT_SEGMENTS:
        for seed in range(3):
            paired = np.zeros((3295, 3), dtype=np.float64)
            for component, weight in components:
                result = by_key.get((component, segment, 1, seed))
                if result is None:
                    raise GPUOutagePORDV5Error("v3 incumbent challenge component is missing")
                prep_path, checkpoint_path = _verify_v3_parent_model_artifacts(
                    v3_run, result
                )
                prep = _v4_preprocessor_from_metadata(read_json_object(prep_path))
                prediction = predict_outage_state(
                    state_dict=load_outage_checkpoint(checkpoint_path),
                    X_scaled=prep.transform_features(np.asarray(identity["X"])).astype(
                        np.float32
                    ),
                    availability=np.asarray(identity["availability"], dtype=np.float32),
                    preprocessor=prep,
                    candidate=result["task"]["candidate"],
                    device=device,
                )
                paired += float(weight) * prediction
            predictions.append(paired)
    if len(predictions) != 12:
        raise GPUOutagePORDV5Error("v3 incumbent challenge ensemble changed")
    return np.mean(np.stack(predictions), axis=0)


def _predict_v4_xgb_challenge(
    *, v4_run: Path, identity: Mapping[str, Any], device: str
) -> np.ndarray:
    from .gpu_outage_gbdt_v4_experiment import (
        _load_valid_xgb_result,
        _preprocessor_from_metadata as _v4_preprocessor_from_metadata,
        _verify_bundle_checkpoint,
        gbdt_v4_development_tasks,
    )
    from .gpu_outage_gbdt_v4_training import (
        load_xgb_bundle,
        predict_xgb_bundle_forced_outage,
    )

    v4_config = read_json_object(v4_run / "config.resolved.json")
    predictions: list[np.ndarray] = []
    for task in gbdt_v4_development_tasks(v4_config):
        result = _load_valid_xgb_result(v4_run, task)
        if result is None:
            raise GPUOutagePORDV5Error("v4 XGB challenge component is missing")
        prep = _v4_preprocessor_from_metadata(
            read_json_object(
                _verify_reference(v4_run, result["artifacts"]["preprocessing"])
            )
        )
        bundle_dir = _verify_bundle_checkpoint(v4_run, result["artifacts"]["bundle"])
        bundle = load_xgb_bundle(
            bundle_dir,
            expected_held_out_segment=int(task["held_out_segment"]),
            expected_paired_seed=int(task["model_seed"]),
            expected_source_sorted_trial_uid_sha256=prep.leakage_audit_[
                "fit_trial_uid_sha256"
            ],
        )
        prediction_scaled = predict_xgb_bundle_forced_outage(
            bundle,
            prep.transform_features(np.asarray(identity["X"])).astype(np.float32),
            1,
            device=device,
        )
        prediction = prep.inverse_targets(prediction_scaled)
        if prediction.shape != (3295, 3) or not np.isfinite(prediction).all():
            raise GPUOutagePORDV5Error("v4 XGB challenge prediction is invalid")
        predictions.append(prediction)
    if len(predictions) != 12:
        raise GPUOutagePORDV5Error("v4 XGB challenge ensemble changed")
    return np.mean(np.stack(predictions), axis=0)


def _default_challenge_component_predictor(
    *,
    component_id: str,
    identity: Mapping[str, Any],
    run_dir: Path,
    config: Mapping[str, Any],
    device: str,
) -> np.ndarray:
    del config
    if component_id == f"{PORD_V5_PRIMARY_BACKBONE_ID}__paired_pord":
        return _predict_pord_challenge_component(
            run_dir=run_dir, identity=identity, device=device
        )
    if component_id != "frozen_v4_incumbent":
        raise GPUOutagePORDV5Error("unknown challenge component predictor")
    freeze = read_json_object(run_dir / "parent.freeze.json")
    v3_run = Path(freeze["v3_run_origin"])
    v4_run = Path(freeze["v4_run_origin"])
    selected = str(freeze["snapshot"]["v4_selected_candidate_id"])
    v3_prediction: np.ndarray | None = None
    xgb_prediction: np.ndarray | None = None
    if selected in {"v3_incumbent", "equal_blend_incumbent_xgboost"}:
        v3_prediction = _predict_v3_incumbent_challenge(
            v3_run=v3_run, identity=identity, device=device
        )
    if selected in {"xgboost_mask_replicated", "equal_blend_incumbent_xgboost"}:
        xgb_prediction = _predict_v4_xgb_challenge(
            v4_run=v4_run, identity=identity, device=device
        )
    if selected == "v3_incumbent" and v3_prediction is not None:
        return v3_prediction
    if selected == "xgboost_mask_replicated" and xgb_prediction is not None:
        return xgb_prediction
    if (
        selected == "equal_blend_incumbent_xgboost"
        and v3_prediction is not None
        and xgb_prediction is not None
    ):
        return 0.5 * v3_prediction + 0.5 * xgb_prediction
    raise GPUOutagePORDV5Error("frozen v4 incumbent component recipe is invalid")


def _feature_only_prediction_frame(
    identity: Mapping[str, Any], prediction: np.ndarray
) -> pd.DataFrame:
    values = np.asarray(prediction, dtype=np.float64)
    if values.shape != (3295, 3) or not np.isfinite(values).all():
        raise GPUOutagePORDV5Error("feature-only challenge prediction is invalid")
    return pd.DataFrame(
        {
            "trial_uid": np.asarray(identity["trial_uid"]).astype(str),
            "segment_id": np.asarray(identity["segment_id"], dtype=np.int64),
            "target_id": np.asarray(identity["target_id"]).astype(str),
            "pred_x_mm": values[:, 0],
            "pred_y_mm": values[:, 1],
            "pred_z_mm": values[:, 2],
        }
    )


def freeze_pord_v5_challenge_predictions(
    run_dir: str | Path,
    *,
    component_predictor: Callable[..., Any] = _default_challenge_component_predictor,
    device: str = "cpu",
    verify_current_source: bool = True,
    _private_test_only_mode: bool = False,
) -> dict[str, Any]:
    """Freeze selected-only feature predictions and replay them before label access."""

    run = Path(run_dir).resolve()
    mode = _require_mode_authority(
        run, private_test_only=_private_test_only_mode
    )
    if (
        component_predictor is not _default_challenge_component_predictor
        and mode != PORD_V5_SYNTHETIC_TEST_MODE
    ):
        raise GPUOutagePORDV5Error(
            "custom challenge predictors are private test-only"
        )
    if mode == PORD_V5_PRODUCTION_MODE and device != "cuda:0":
        raise GPUOutagePORDV5Error("production challenge replay requires cuda:0")
    existing = run / "challenge" / "freeze.before_label_access.json"
    if existing.is_file():
        validate_pord_v5_challenge_freeze(
            run,
            allow_synthetic_test_only=_private_test_only_mode,
            verify_live_environment=not _private_test_only_mode,
        )
        return read_json_object(existing)
    validation = validate_pord_v5_development_run(
        run,
        verify_current_source=verify_current_source,
        allow_synthetic_test_only=_private_test_only_mode,
        verify_live_environment=not _private_test_only_mode,
    )
    if validation["challenge_finalize_allowed"] is not True:
        raise GPUOutagePORDV5Error(
            "challenge prediction freeze is refused until every claim/novelty gate passes"
        )
    selection = read_json_object(run / "development" / "selection.json")
    selected = str(selection["selected_candidate_id"])
    pord_id = f"{PORD_V5_PRIMARY_BACKBONE_ID}__paired_pord"
    blend_id = "fixed_half_v4_incumbent_half_pord"
    if selected == pord_id:
        components = [pord_id]
    elif selected == blend_id:
        components = [pord_id, "frozen_v4_incumbent"]
    else:
        raise GPUOutagePORDV5Error("selection names a non-preregistered novel predictor")
    identity = _load_challenge_snapshot(run)
    config = read_json_object(run / "config.resolved.json")
    component_predictions: dict[str, np.ndarray] = {}
    component_records: list[dict[str, Any]] = []
    for component in components:
        prediction = np.asarray(
            component_predictor(
                component_id=component,
                identity=identity,
                run_dir=run,
                config=config,
                device=device,
            ),
            dtype=np.float64,
        )
        replay = np.asarray(
            component_predictor(
                component_id=component,
                identity=identity,
                run_dir=run,
                config=config,
                device=device,
            ),
            dtype=np.float64,
        )
        if not np.array_equal(prediction, replay):
            raise GPUOutagePORDV5Error(
                "challenge component prediction failed exact semantic replay"
            )
        component_predictions[component] = prediction
        path = atomic_write_parquet(
            run / "challenge" / "components" / f"{component}.parquet",
            _feature_only_prediction_frame(identity, prediction),
        )
        component_records.append(
            {
                "component_id": component,
                "prediction_sha256": array_sha256(prediction),
                "semantic_replay_exact": True,
                "prediction_file": _artifact_reference(path, run, rows=3295),
            }
        )
    if selected == pord_id:
        selected_prediction = component_predictions[pord_id]
        recipe = {
            "selected_candidate_id": selected,
            "kind": "mean_of_twelve_frozen_pord_fold_seed_checkpoints",
            "component_weights": {pord_id: 1.0},
            "outage_index": 1,
            "refit_performed": False,
        }
    else:
        selected_prediction = 0.5 * component_predictions[pord_id] + 0.5 * component_predictions[
            "frozen_v4_incumbent"
        ]
        recipe = {
            "selected_candidate_id": selected,
            "kind": "fixed_raw_mm_half_blend",
            "component_weights": {
                "frozen_v4_incumbent": 0.5,
                pord_id: 0.5,
            },
            "outage_index": 1,
            "refit_performed": False,
        }
    recipe["recipe_sha256"] = stable_hash(recipe)
    recipe_path = atomic_write_json(run / "challenge" / "prediction_recipe.json", recipe)
    selected_path = atomic_write_parquet(
        run / "challenge" / "selected_feature_only_prediction.parquet",
        _feature_only_prediction_frame(identity, selected_prediction),
    )
    selection_path = run / "development" / "selection.json"
    development_manifest_path = run / "development_manifest.json"
    identity_binding = {
        "ordered_trial_uid_sha256": identity["ordered_trial_uid_sha256"],
        "ordered_segment_id_sha256": array_sha256(
            np.asarray(identity["segment_id"], dtype=np.int64)
        ),
        "ordered_target_id_sha256": stable_hash(
            np.asarray(identity["target_id"]).astype(str).tolist()
        ),
        "feature_matrix_sha256": array_sha256(
            np.asarray(identity["X"], dtype=np.float64)
        ),
        "availability_sha256": array_sha256(
            np.asarray(identity["availability"], dtype=np.float64)
        ),
    }
    parent_freeze_path = run / "parent.freeze.json"
    identity_metadata_path = run / "inputs" / "challenge_identity.json"
    identity_metadata = read_json_object(identity_metadata_path)
    authority = _prepare_challenge_label_source_authority(
        run,
        preload={
            "parent_freeze": read_json_object(parent_freeze_path),
            "references": {
                "parent_freeze": _artifact_reference(parent_freeze_path, run),
                "challenge_identity_metadata": _artifact_reference(
                    identity_metadata_path, run
                ),
                "challenge_identity": dict(identity_metadata["identity_file"]),
            },
        },
        use_default_loader=mode == PORD_V5_PRODUCTION_MODE,
    )
    prelabel_payload = {
        "status": "ok",
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
        "config_sha256": pord_v5_config_sha256(config),
        "development_manifest": _artifact_reference(development_manifest_path, run),
        "selection": _artifact_reference(selection_path, run),
        "selected_candidate_id": selected,
        "selected_prediction_sha256": array_sha256(selected_prediction),
        "selected_prediction_file": _artifact_reference(
            selected_path, run, rows=3295
        ),
        "recipe_sha256": recipe["recipe_sha256"],
        "recipe_file": _artifact_reference(recipe_path, run),
        "component_weights": dict(recipe["component_weights"]),
        "identity_binding": identity_binding,
        "label_source_authority_file": authority["reference"],
        "frozen_v3_config_file": authority["frozen_v3_config_reference"],
        "semantic_replay_exact": True,
        "challenge_labels_accessed": False,
    }
    validation_path = atomic_write_json(
        run / "challenge" / "prelabel_semantic_validation.json",
        prelabel_payload,
    )
    freeze = {
        "contract": PORD_V5_CHALLENGE_FREEZE_CONTRACT,
        "status": "feature_only_prediction_recipe_frozen_labels_sealed",
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
        "config_sha256": pord_v5_config_sha256(config),
        "development_manifest": _artifact_reference(development_manifest_path, run),
        "selection_file": _artifact_reference(selection_path, run),
        "selected_candidate_id": selected,
        "selected_only_component_count": len(components),
        "selected_only_component_ids": components,
        "component_predictions": component_records,
        "selected_prediction_sha256": array_sha256(selected_prediction),
        "selected_prediction_file": _artifact_reference(selected_path, run, rows=3295),
        "recipe_sha256": recipe["recipe_sha256"],
        "recipe_file": _artifact_reference(recipe_path, run),
        "ordered_trial_uid_sha256": identity["ordered_trial_uid_sha256"],
        "identity_binding": identity_binding,
        "prelabel_validation_file": _artifact_reference(validation_path, run),
        "label_source_authority_file": authority["reference"],
        "frozen_v3_config_file": authority["frozen_v3_config_reference"],
        "semantic_replay_before_labels": True,
        "challenge_labels_accessed": False,
        "challenge_can_trigger_reselection": False,
    }
    # This O_EXCL marker is deliberately the final pre-label write.  Every
    # component, recipe, selected prediction, identity binding and semantic
    # validation artifact already exists and is hash-bound above.
    _exclusive_write_json(existing, freeze)
    return read_json_object(existing)


def validate_pord_v5_challenge_freeze(
    run_dir: str | Path,
    *,
    allow_synthetic_test_only: bool = False,
    verify_live_environment: bool = True,
) -> dict[str, Any]:
    """Authenticate selected-only, label-blind prediction/recipe evidence."""

    run = Path(run_dir).resolve()
    mode = _prepared_execution_mode(run)
    if mode == PORD_V5_SYNTHETIC_TEST_MODE and not allow_synthetic_test_only:
        raise GPUOutagePORDV5Error(
            "synthetic test-only freeze is rejected by the production validator"
        )
    if mode == PORD_V5_PRODUCTION_MODE and verify_live_environment:
        environment = read_json_object(
            run / "provenance" / "environment_snapshot.json"
        )
        _validate_live_production_environment(environment)
    selection = read_json_object(run / "development" / "selection.json")
    config = read_json_object(run / "config.resolved.json")
    freeze = read_json_object(run / "challenge" / "freeze.before_label_access.json")
    required = {
        "contract",
        "status",
        "execution_mode",
        "synthetic_test_only",
        "config_sha256",
        "development_manifest",
        "selection_file",
        "selected_candidate_id",
        "selected_only_component_count",
        "selected_only_component_ids",
        "component_predictions",
        "selected_prediction_sha256",
        "selected_prediction_file",
        "recipe_sha256",
        "recipe_file",
        "ordered_trial_uid_sha256",
        "identity_binding",
        "prelabel_validation_file",
        "label_source_authority_file",
        "frozen_v3_config_file",
        "semantic_replay_before_labels",
        "challenge_labels_accessed",
        "challenge_can_trigger_reselection",
    }
    if (
        set(freeze) != required
        or freeze.get("contract") != PORD_V5_CHALLENGE_FREEZE_CONTRACT
        or freeze.get("status")
        != "feature_only_prediction_recipe_frozen_labels_sealed"
        or freeze.get("execution_mode") != mode
        or freeze.get("synthetic_test_only") is not (
            mode == PORD_V5_SYNTHETIC_TEST_MODE
        )
        or freeze.get("config_sha256") != pord_v5_config_sha256(config)
        or freeze.get("selected_candidate_id") != selection.get("selected_candidate_id")
        or freeze.get("semantic_replay_before_labels") is not True
        or freeze.get("challenge_labels_accessed") is not False
        or freeze.get("challenge_can_trigger_reselection") is not False
    ):
        raise GPUOutagePORDV5Error("challenge pre-label freeze changed")
    authority = _validate_label_source_authority(
        run,
        read_json_object(
            _verify_reference(run, freeze["label_source_authority_file"])
        ),
    )
    if freeze["frozen_v3_config_file"] != authority["frozen_v3_config_file"]:
        raise GPUOutagePORDV5Error("challenge label-source authority changed")
    if (
        freeze.get("development_manifest")
        != _artifact_reference(run / "development_manifest.json", run)
        or freeze.get("selection_file")
        != _artifact_reference(run / "development" / "selection.json", run)
    ):
        raise GPUOutagePORDV5Error("challenge freeze development binding changed")
    pord_id = f"{PORD_V5_PRIMARY_BACKBONE_ID}__paired_pord"
    expected_components = (
        [pord_id]
        if selection["selected_candidate_id"] == pord_id
        else [pord_id, "frozen_v4_incumbent"]
    )
    if (
        freeze.get("selected_only_component_ids") != expected_components
        or freeze.get("selected_only_component_count") != len(expected_components)
        or len(freeze.get("component_predictions", ())) != len(expected_components)
    ):
        raise GPUOutagePORDV5Error("challenge selected-only component set changed")
    identity = _load_challenge_snapshot(run)
    expected_identity_binding = {
        "ordered_trial_uid_sha256": identity["ordered_trial_uid_sha256"],
        "ordered_segment_id_sha256": array_sha256(
            np.asarray(identity["segment_id"], dtype=np.int64)
        ),
        "ordered_target_id_sha256": stable_hash(
            np.asarray(identity["target_id"]).astype(str).tolist()
        ),
        "feature_matrix_sha256": array_sha256(
            np.asarray(identity["X"], dtype=np.float64)
        ),
        "availability_sha256": array_sha256(
            np.asarray(identity["availability"], dtype=np.float64)
        ),
    }
    if (
        freeze.get("ordered_trial_uid_sha256")
        != identity["ordered_trial_uid_sha256"]
        or freeze.get("identity_binding") != expected_identity_binding
    ):
        raise GPUOutagePORDV5Error("challenge identity/freeze binding changed")
    component_arrays: dict[str, np.ndarray] = {}
    for expected_component, row in zip(
        expected_components, freeze["component_predictions"], strict=True
    ):
        if (
            row.get("component_id") != expected_component
            or row.get("semantic_replay_exact") is not True
        ):
            raise GPUOutagePORDV5Error("challenge component evidence changed")
        frame = _read_exact_parquet(
            _verify_reference(run, row["prediction_file"]),
            [
                "trial_uid",
                "segment_id",
                "target_id",
                "pred_x_mm",
                "pred_y_mm",
                "pred_z_mm",
            ],
        )
        if (
            frame["trial_uid"].astype(str).tolist()
            != np.asarray(identity["trial_uid"]).astype(str).tolist()
        ):
            raise GPUOutagePORDV5Error("challenge component row identity changed")
        values = frame.loc[:, ["pred_x_mm", "pred_y_mm", "pred_z_mm"]].to_numpy(
            dtype=np.float64
        )
        if row.get("prediction_sha256") != array_sha256(values):
            raise GPUOutagePORDV5Error("challenge component prediction hash changed")
        component_arrays[expected_component] = values
    recipe = read_json_object(_verify_reference(run, freeze["recipe_file"]))
    recipe_core = {key: value for key, value in recipe.items() if key != "recipe_sha256"}
    if (
        recipe.get("recipe_sha256") != stable_hash(recipe_core)
        or freeze.get("recipe_sha256") != recipe["recipe_sha256"]
        or recipe.get("selected_candidate_id") != selection["selected_candidate_id"]
        or recipe.get("refit_performed") is not False
        or recipe.get("outage_index") != 1
    ):
        raise GPUOutagePORDV5Error("challenge prediction recipe changed")
    expected_weights = (
        {pord_id: 1.0}
        if selection["selected_candidate_id"] == pord_id
        else {"frozen_v4_incumbent": 0.5, pord_id: 0.5}
    )
    if recipe.get("component_weights") != expected_weights:
        raise GPUOutagePORDV5Error("challenge component weights changed")
    selected_frame = _read_exact_parquet(
        _verify_reference(run, freeze["selected_prediction_file"]),
        [
            "trial_uid",
            "segment_id",
            "target_id",
            "pred_x_mm",
            "pred_y_mm",
            "pred_z_mm",
        ],
    )
    selected_values = selected_frame.loc[
        :, ["pred_x_mm", "pred_y_mm", "pred_z_mm"]
    ].to_numpy(dtype=np.float64)
    if selection["selected_candidate_id"] == pord_id:
        replay = component_arrays[pord_id]
    else:
        replay = 0.5 * component_arrays[pord_id] + 0.5 * component_arrays[
            "frozen_v4_incumbent"
        ]
    if (
        not np.array_equal(selected_values, replay)
        or freeze.get("selected_prediction_sha256") != array_sha256(selected_values)
    ):
        raise GPUOutagePORDV5Error("selected challenge prediction semantic replay changed")
    prelabel = read_json_object(
        _verify_reference(run, freeze["prelabel_validation_file"])
    )
    if (
        set(prelabel)
        != {
            "status",
            "execution_mode",
            "synthetic_test_only",
            "config_sha256",
            "development_manifest",
            "selection",
            "selected_candidate_id",
            "selected_prediction_sha256",
            "selected_prediction_file",
            "recipe_sha256",
            "recipe_file",
            "component_weights",
            "identity_binding",
            "label_source_authority_file",
            "frozen_v3_config_file",
            "semantic_replay_exact",
            "challenge_labels_accessed",
        }
        or prelabel.get("status") != "ok"
        or prelabel.get("execution_mode") != mode
        or prelabel.get("synthetic_test_only") is not (
            mode == PORD_V5_SYNTHETIC_TEST_MODE
        )
        or prelabel.get("config_sha256") != freeze["config_sha256"]
        or prelabel.get("development_manifest") != freeze["development_manifest"]
        or prelabel.get("selection") != freeze["selection_file"]
        or prelabel.get("selected_candidate_id") != selection["selected_candidate_id"]
        or prelabel.get("selected_prediction_sha256")
        != freeze["selected_prediction_sha256"]
        or prelabel.get("recipe_sha256") != freeze["recipe_sha256"]
        or prelabel.get("selected_prediction_file")
        != freeze["selected_prediction_file"]
        or prelabel.get("recipe_file") != freeze["recipe_file"]
        or prelabel.get("component_weights") != expected_weights
        or prelabel.get("identity_binding") != expected_identity_binding
        or prelabel.get("label_source_authority_file")
        != freeze["label_source_authority_file"]
        or prelabel.get("frozen_v3_config_file")
        != freeze["frozen_v3_config_file"]
        or prelabel.get("semantic_replay_exact") is not True
        or prelabel.get("challenge_labels_accessed") is not False
    ):
        raise GPUOutagePORDV5Error("pre-label semantic validation changed")
    return {
        "status": "ok",
        "selected_candidate_id": selection["selected_candidate_id"],
        "selected_only_component_count": len(expected_components),
        "selected_prediction_sha256": freeze["selected_prediction_sha256"],
        "recipe_sha256": freeze["recipe_sha256"],
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
        "challenge_labels_sealed": not (
            run / "challenge" / "label_access_1.started.json"
        ).exists(),
    }


_LABEL_SOURCE_AUTHORITY_CONTRACT = (
    "alphaf.endpoint_regression.gpu_outage_pord_label_source_authority.v5"
)


def _validate_label_source_authority(
    run: Path, descriptor: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "contract",
        "execution_mode",
        "synthetic_test_only",
        "loader_kind",
        "parent_freeze_file",
        "challenge_identity_metadata_file",
        "challenge_identity_file",
        "v3_config_origin",
        "frozen_v3_config_file",
        "source_authority_sha256",
        "authority_sha256",
    }
    if type(descriptor) is not dict or set(descriptor) != required:
        raise GPUOutagePORDV5Error("label-source authority schema changed")
    core = {key: value for key, value in descriptor.items() if key != "authority_sha256"}
    if (
        descriptor.get("contract") != _LABEL_SOURCE_AUTHORITY_CONTRACT
        or descriptor.get("execution_mode") != _prepared_execution_mode(run)
        or descriptor.get("synthetic_test_only") is not (
            _prepared_execution_mode(run) == PORD_V5_SYNTHETIC_TEST_MODE
        )
        or descriptor.get("loader_kind")
        not in {"production_default_v3_projection", "synthetic_test_only_callback"}
        or descriptor.get("authority_sha256") != stable_hash(core)
    ):
        raise GPUOutagePORDV5Error("label-source authority binding changed")
    _exact_sha256(
        descriptor["source_authority_sha256"],
        name="label-source authority fingerprint",
    )
    for key in (
        "parent_freeze_file",
        "challenge_identity_metadata_file",
        "challenge_identity_file",
    ):
        _verify_reference(run, descriptor[key])
    if descriptor["loader_kind"] == "production_default_v3_projection":
        _exact_string(descriptor["v3_config_origin"], name="v3 config origin")
        if type(descriptor["frozen_v3_config_file"]) is not dict:
            raise GPUOutagePORDV5Error("frozen v3 config reference is missing")
        _verify_reference(run, descriptor["frozen_v3_config_file"])
    elif (
        descriptor["v3_config_origin"] is not None
        or descriptor["frozen_v3_config_file"] is not None
    ):
        raise GPUOutagePORDV5Error("synthetic label authority names a v3 source")
    return dict(descriptor)


def _prepare_challenge_label_source_authority(
    run: Path,
    *,
    preload: Mapping[str, Any],
    use_default_loader: bool,
) -> dict[str, Any]:
    """Freeze non-label source authority and import loader code before O_EXCL."""

    mode = _prepared_execution_mode(run)
    references = preload["references"]
    runtime: dict[str, Any] = {}
    if use_default_loader:
        if mode != PORD_V5_PRODUCTION_MODE:
            raise GPUOutagePORDV5Error(
                "default challenge label authority is production-only"
            )
        from .gpu_outage_data import ChallengeIdentityData, load_challenge_labels_once
        from .gpu_outage_gbdt_v4_experiment import _load_v3_config

        parent = preload["parent_freeze"]
        origin = Path(
            _exact_string(parent["v3_run_origin"], name="v3 run origin")
        ) / "config.resolved.json"
        local_config_path = _copy_atomic(
            origin,
            run / "challenge" / "label_source_authority" / "v3_config.resolved.json",
        )
        v3_config = _load_v3_config(local_config_path)
        frozen_config_reference: dict[str, Any] | None = _artifact_reference(
            local_config_path, run
        )
        source_authority_sha256 = stable_hash(v3_config["source"])
        runtime = {
            "challenge_identity_type": ChallengeIdentityData,
            "load_labels_once": load_challenge_labels_once,
            "v3_config": deepcopy(v3_config),
        }
        loader_kind = "production_default_v3_projection"
        config_origin: str | None = str(origin.resolve())
    else:
        if mode != PORD_V5_SYNTHETIC_TEST_MODE:
            raise GPUOutagePORDV5Error("custom label authority is test-only")
        frozen_config_reference = None
        source_authority_sha256 = stable_hash(
            {"loader_kind": "synthetic_test_only_callback"}
        )
        loader_kind = "synthetic_test_only_callback"
        config_origin = None
    descriptor = {
        "contract": _LABEL_SOURCE_AUTHORITY_CONTRACT,
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
        "loader_kind": loader_kind,
        "parent_freeze_file": references["parent_freeze"],
        "challenge_identity_metadata_file": references[
            "challenge_identity_metadata"
        ],
        "challenge_identity_file": references["challenge_identity"],
        "v3_config_origin": config_origin,
        "frozen_v3_config_file": frozen_config_reference,
        "source_authority_sha256": source_authority_sha256,
    }
    descriptor["authority_sha256"] = stable_hash(descriptor)
    descriptor_path = atomic_write_json(
        run / "challenge" / "label_source_authority.json", descriptor
    )
    _validate_label_source_authority(run, descriptor)
    return {
        "descriptor": descriptor,
        "reference": _artifact_reference(descriptor_path, run),
        "frozen_v3_config_reference": frozen_config_reference,
        "runtime": runtime,
    }


def _load_challenge_label_source_authority_runtime(
    run: Path, *, preload: Mapping[str, Any]
) -> dict[str, Any]:
    """Load and authenticate already-frozen authority before the access marker."""

    descriptor_path = _verify_reference(
        run, preload["references"]["label_source_authority"]
    )
    descriptor = _validate_label_source_authority(
        run, read_json_object(descriptor_path)
    )
    runtime: dict[str, Any] = {}
    if descriptor["loader_kind"] == "production_default_v3_projection":
        from .gpu_outage_data import ChallengeIdentityData, load_challenge_labels_once
        from .gpu_outage_gbdt_v4_experiment import _load_v3_config

        config_path = _verify_reference(run, descriptor["frozen_v3_config_file"])
        v3_config = _load_v3_config(config_path)
        if stable_hash(v3_config["source"]) != descriptor["source_authority_sha256"]:
            raise GPUOutagePORDV5Error("frozen v3 source authority changed")
        runtime = {
            "challenge_identity_type": ChallengeIdentityData,
            "load_labels_once": load_challenge_labels_once,
            "v3_config": deepcopy(v3_config),
        }
    return {
        "descriptor": descriptor,
        "reference": dict(preload["references"]["label_source_authority"]),
        "frozen_v3_config_reference": preload["references"].get(
            "frozen_v3_config"
        ),
        "runtime": runtime,
    }


def _default_challenge_label_loader(
    identity: Mapping[str, Any], authority: Mapping[str, Any]
) -> Any:
    """Read labels using only authority objects preloaded before O_EXCL."""

    if authority.get("descriptor", {}).get("loader_kind") != (
        "production_default_v3_projection"
    ):
        raise GPUOutagePORDV5Error("default label loader lacks frozen authority")
    runtime = authority.get("runtime")
    if type(runtime) is not dict or set(runtime) != {
        "challenge_identity_type",
        "load_labels_once",
        "v3_config",
    }:
        raise GPUOutagePORDV5Error("default label loader runtime authority changed")
    challenge_identity = runtime["challenge_identity_type"](
        feature_names=FUSION_COLUMNS,
        X=np.asarray(identity["X"], dtype=np.float64),
        availability=np.asarray(identity["availability"], dtype=np.float64),
        trial_uids=np.asarray(identity["trial_uid"]).astype(str),
        segment_ids=np.asarray(identity["segment_id"], dtype=np.int64),
        target_ids=np.asarray(identity["target_id"]).astype(str),
    )
    v3_config = runtime["v3_config"]
    return runtime["load_labels_once"](
        v3_config["source"], v3_config, challenge_identity
    )


def _normalise_challenge_labels(
    raw: Any, identity: Mapping[str, Any]
) -> np.ndarray:
    if isinstance(raw, Mapping):
        if set(raw) != {"trial_uid", "segment_id", "target_id", "y_mm"}:
            raise GPUOutagePORDV5Error("injected challenge label payload schema changed")
        trial_uid = np.asarray(raw["trial_uid"]).astype(str)
        segment_id = np.asarray(raw["segment_id"], dtype=np.int64)
        target_id = np.asarray(raw["target_id"]).astype(str)
        y = np.asarray(raw["y_mm"], dtype=np.float64)
    else:
        try:
            trial_uid = np.asarray(raw.trial_uids).astype(str)
            segment_id = np.asarray(raw.segment_ids, dtype=np.int64)
            target_id = np.asarray(raw.target_ids).astype(str)
            y = np.asarray(raw.y, dtype=np.float64)
        except AttributeError as exc:
            raise GPUOutagePORDV5Error("challenge label loader returned an invalid object") from exc
    if (
        y.shape != (3295, 3)
        or not np.isfinite(y).all()
        or not np.array_equal(trial_uid, np.asarray(identity["trial_uid"]).astype(str))
        or not np.array_equal(segment_id, np.asarray(identity["segment_id"], dtype=np.int64))
        or not np.array_equal(target_id, np.asarray(identity["target_id"]).astype(str))
    ):
        raise GPUOutagePORDV5Error(
            "challenge labels are non-finite, wrong-shaped, or identity-misaligned"
        )
    return y


def _preload_challenge_access_bundle(
    run: Path,
    *,
    allow_synthetic_test_only: bool,
    verify_live_environment: bool,
) -> dict[str, Any]:
    """Validate and load every non-label byte before the access marker."""

    validate_pord_v5_challenge_freeze(
        run,
        allow_synthetic_test_only=allow_synthetic_test_only,
        verify_live_environment=verify_live_environment,
    )
    freeze_path = run / "challenge" / "freeze.before_label_access.json"
    selection_path = run / "development" / "selection.json"
    development_manifest_path = run / "development_manifest.json"
    parent_freeze_path = run / "parent.freeze.json"
    identity_metadata_path = run / "inputs" / "challenge_identity.json"
    freeze = read_json_object(freeze_path)
    selection = read_json_object(selection_path)
    parent_freeze = read_json_object(parent_freeze_path)
    identity_metadata = read_json_object(identity_metadata_path)
    identity = _load_challenge_snapshot(run)
    prediction_path = _verify_reference(run, freeze["selected_prediction_file"])
    prediction_frame = _read_exact_parquet(
        prediction_path,
        [
            "trial_uid",
            "segment_id",
            "target_id",
            "pred_x_mm",
            "pred_y_mm",
            "pred_z_mm",
        ],
    )
    if (
        prediction_frame["trial_uid"].astype(str).tolist()
        != np.asarray(identity["trial_uid"]).astype(str).tolist()
        or prediction_frame["target_id"].astype(str).tolist()
        != np.asarray(identity["target_id"]).astype(str).tolist()
        or not np.array_equal(
            prediction_frame["segment_id"].to_numpy(dtype=np.int64),
            np.asarray(identity["segment_id"], dtype=np.int64),
        )
    ):
        raise GPUOutagePORDV5Error("frozen challenge prediction identity changed")
    prediction = prediction_frame.loc[
        :, ["pred_x_mm", "pred_y_mm", "pred_z_mm"]
    ].to_numpy(dtype=np.float64)
    if (
        prediction.shape != (3295, 3)
        or not np.isfinite(prediction).all()
        or array_sha256(prediction) != freeze["selected_prediction_sha256"]
    ):
        raise GPUOutagePORDV5Error("frozen challenge prediction values changed")
    references = {
        "development_manifest": _artifact_reference(development_manifest_path, run),
        "selection": _artifact_reference(selection_path, run),
        "freeze": _artifact_reference(freeze_path, run),
        "prelabel_validation": dict(freeze["prelabel_validation_file"]),
        "recipe": dict(freeze["recipe_file"]),
        "selected_prediction": dict(freeze["selected_prediction_file"]),
        "parent_freeze": _artifact_reference(parent_freeze_path, run),
        "challenge_identity_metadata": _artifact_reference(
            identity_metadata_path, run
        ),
        "challenge_identity": dict(identity_metadata["identity_file"]),
    }
    authority_path = run / "challenge" / "label_source_authority.json"
    if authority_path.is_file():
        authority = _validate_label_source_authority(
            run, read_json_object(authority_path)
        )
        references["label_source_authority"] = _artifact_reference(
            authority_path, run
        )
        if authority["frozen_v3_config_file"] is not None:
            references["frozen_v3_config"] = dict(
                authority["frozen_v3_config_file"]
            )
        if (
            freeze.get("label_source_authority_file")
            != references["label_source_authority"]
            or freeze.get("frozen_v3_config_file")
            != references.get("frozen_v3_config")
        ):
            raise GPUOutagePORDV5Error(
                "challenge freeze label-source authority reference changed"
            )
    for reference in references.values():
        _verify_reference(run, reference)
    return {
        "freeze": freeze,
        "selection": selection,
        "parent_freeze": parent_freeze,
        "identity": identity,
        "prediction": prediction.copy(),
        "references": references,
    }


def _final_manifest_payload(
    run: Path, *, selected_candidate_id: str
) -> dict[str, Any]:
    return {
        "contract": PORD_V5_RUN_CONTRACT,
        "experiment_contract": PORD_V5_CONTRACT,
        "stage": "challenge_complete_no_reselection",
        "selected_candidate_id": selected_candidate_id,
        "challenge_label_access_passes": 1,
        "primary_segment": 41,
        "secondary_segment": 40,
        "challenge_used_for_reselection": False,
        "files": [row for row in manifest_files(run) if row["path"] != "manifest.json"],
    }


def _assert_parent_challenge_markers_absent(
    parent_freeze: Mapping[str, Any],
) -> None:
    """Check parent markers under the declared single-controller assumption."""

    for key in ("v3_run_origin", "v4_run_origin"):
        origin = Path(
            _exact_string(parent_freeze.get(key), name=f"parent freeze {key}")
        )
        if (origin / "challenge" / "label_access_1.started.json").exists():
            raise GPUOutagePORDV5Error(
                "a prior v3/v4 label-access marker is present"
            )


def _recover_completed_challenge(
    run: Path,
    *,
    allow_synthetic_test_only: bool = False,
    verify_live_environment: bool = True,
    repair_root_manifest: bool = True,
) -> dict[str, Any]:
    """Recover final evidence without any source or label-loader reread."""

    completed_path = run / "challenge" / "label_access_1.completed.json"
    completed = read_json_object(completed_path)
    required = {
        "contract",
        "access_index",
        "status",
        "execution_mode",
        "synthetic_test_only",
        "selected_candidate_id",
        "started_file",
        "started_sha256",
        "development_manifest",
        "selection_file",
        "report_file",
        "evaluation_file",
        "freeze_file",
        "prelabel_validation_file",
        "recipe_file",
        "selected_prediction_file",
        "parent_freeze_file",
        "challenge_identity_metadata_file",
        "challenge_identity_file",
        "label_source_authority_file",
        "frozen_v3_config_file",
        "prelabel_reference_chain_sha256",
        "identity_binding",
        "selected_prediction_sha256",
        "challenge_truth_sha256",
        "challenge_label_access_passes",
        "challenge_used_for_reselection",
        "thresholding_after_labels",
        "source_reread_for_recovery",
        "cross_parent_finalizer_exclusion",
    }
    if (
        set(completed) != required
        or completed.get("contract") != PORD_V5_CHALLENGE_ACCESS_CONTRACT
        or type(completed.get("access_index")) is not int
        or completed.get("access_index") != 1
        or completed.get("status") != "completed_exactly_once"
        or type(completed.get("challenge_label_access_passes")) is not int
        or completed.get("challenge_label_access_passes") != 1
        or completed.get("challenge_used_for_reselection") is not False
        or completed.get("thresholding_after_labels") is not False
        or completed.get("source_reread_for_recovery") is not False
        or completed.get("cross_parent_finalizer_exclusion")
        != "single_controller_assumption_not_machine_mutex"
    ):
        raise GPUOutagePORDV5Error("completed challenge access evidence changed")
    mode = _prepared_execution_mode(run)
    if (
        mode == PORD_V5_SYNTHETIC_TEST_MODE
        and not allow_synthetic_test_only
    ):
        raise GPUOutagePORDV5Error(
            "synthetic completed challenge is rejected by production recovery"
        )
    if (
        completed.get("execution_mode") != mode
        or completed.get("synthetic_test_only") is not (
            mode == PORD_V5_SYNTHETIC_TEST_MODE
        )
    ):
        raise GPUOutagePORDV5Error("completed challenge execution mode changed")
    preload = _preload_challenge_access_bundle(
        run,
        allow_synthetic_test_only=allow_synthetic_test_only,
        verify_live_environment=verify_live_environment,
    )
    _assert_parent_challenge_markers_absent(preload["parent_freeze"])
    started_path = _verify_reference(run, completed["started_file"])
    started = read_json_object(started_path)
    expected_started_keys = {
        "contract",
        "access_index",
        "status",
        "execution_mode",
        "synthetic_test_only",
        "config_sha256",
        "selected_candidate_id",
        "development_manifest",
        "selection_file",
        "freeze_file",
        "prelabel_validation_file",
        "recipe_file",
        "selected_prediction_file",
        "parent_freeze_file",
        "challenge_identity_metadata_file",
        "challenge_identity_file",
        "label_source_authority_file",
        "frozen_v3_config_file",
        "prelabel_reference_chain_sha256",
        "selected_prediction_sha256",
        "identity_binding",
        "ordered_trial_uid_sha256",
        "maximum_label_access_passes",
        "marker_creation",
        "challenge_used_for_selection",
        "cross_parent_finalizer_exclusion",
        "started_sha256",
    }
    started_hash = started.get("started_sha256")
    if (
        set(started) != expected_started_keys
        or type(started.get("access_index")) is not int
        or type(started.get("maximum_label_access_passes")) is not int
        or started.get("maximum_label_access_passes") != 1
        or started_hash
        != stable_hash(
            {key: value for key, value in started.items() if key != "started_sha256"}
        )
        or completed.get("started_sha256") != started_hash
        or started.get("status") != "started_fail_closed"
        or started.get("access_index") != 1
        or started.get("selected_candidate_id")
        != completed.get("selected_candidate_id")
        or started.get("prelabel_reference_chain_sha256")
        != completed.get("prelabel_reference_chain_sha256")
        or started.get("cross_parent_finalizer_exclusion")
        != "single_controller_assumption_not_machine_mutex"
    ):
        raise GPUOutagePORDV5Error("started/completed challenge linkage changed")
    for key in (
        "config_sha256",
        "prelabel_reference_chain_sha256",
        "selected_prediction_sha256",
        "ordered_trial_uid_sha256",
        "started_sha256",
        "challenge_truth_sha256",
    ):
        source = started if key in started else completed
        _exact_sha256(source[key], name=f"challenge access {key}")
    for key in (
        "development_manifest",
        "selection_file",
        "report_file",
        "evaluation_file",
        "freeze_file",
        "prelabel_validation_file",
        "recipe_file",
        "selected_prediction_file",
        "parent_freeze_file",
        "challenge_identity_metadata_file",
        "challenge_identity_file",
        "label_source_authority_file",
    ):
        _verify_reference(run, completed[key])
    if completed["frozen_v3_config_file"] is not None:
        _verify_reference(run, completed["frozen_v3_config_file"])
    expected_reference_bindings = {
        "development_manifest": preload["references"]["development_manifest"],
        "selection_file": preload["references"]["selection"],
        "freeze_file": preload["references"]["freeze"],
        "prelabel_validation_file": preload["references"]["prelabel_validation"],
        "recipe_file": preload["references"]["recipe"],
        "selected_prediction_file": preload["references"]["selected_prediction"],
        "parent_freeze_file": preload["references"]["parent_freeze"],
        "challenge_identity_metadata_file": preload["references"][
            "challenge_identity_metadata"
        ],
        "challenge_identity_file": preload["references"]["challenge_identity"],
        "label_source_authority_file": preload["references"][
            "label_source_authority"
        ],
        "frozen_v3_config_file": preload["references"].get("frozen_v3_config"),
    }
    if (
        any(
            completed.get(key) != value
            for key, value in expected_reference_bindings.items()
        )
        or completed.get("prelabel_reference_chain_sha256")
        != stable_hash(preload["references"])
        or started.get("prelabel_reference_chain_sha256")
        != stable_hash(preload["references"])
    ):
        raise GPUOutagePORDV5Error("completed challenge freeze-chain changed")
    evaluation = _read_exact_parquet(
        _verify_reference(run, completed["evaluation_file"]),
        [
            "trial_uid",
            "segment_id",
            "target_id",
            *_TRUE_COLUMNS,
            "pred_x_mm",
            "pred_y_mm",
            "pred_z_mm",
        ],
    )
    identity = preload["identity"]
    prediction = evaluation.loc[
        :, ["pred_x_mm", "pred_y_mm", "pred_z_mm"]
    ].to_numpy(dtype=np.float64)
    truth = evaluation.loc[:, list(_TRUE_COLUMNS)].to_numpy(dtype=np.float64)
    if (
        len(evaluation) != 3295
        or evaluation["trial_uid"].astype(str).tolist()
        != np.asarray(identity["trial_uid"]).astype(str).tolist()
        or evaluation["target_id"].astype(str).tolist()
        != np.asarray(identity["target_id"]).astype(str).tolist()
        or not np.array_equal(
            evaluation["segment_id"].to_numpy(dtype=np.int64),
            np.asarray(identity["segment_id"], dtype=np.int64),
        )
        or not np.isfinite(truth).all()
        or not np.isfinite(prediction).all()
        or not np.array_equal(prediction, preload["prediction"])
        or array_sha256(prediction) != completed["selected_prediction_sha256"]
        or array_sha256(truth) != completed["challenge_truth_sha256"]
        or completed.get("identity_binding")
        != preload["freeze"]["identity_binding"]
    ):
        raise GPUOutagePORDV5Error("completed challenge evaluation changed")
    report = read_json_object(_verify_reference(run, completed["report_file"]))
    segments = np.asarray(identity["segment_id"], dtype=np.int64)
    expected_report = {
        "selected_candidate_id": completed["selected_candidate_id"],
        "overall_metrics": regression_metrics(truth, prediction),
        "primary_segment": 41,
        "primary_segment_metrics": regression_metrics(
            truth[segments == 41], prediction[segments == 41]
        ),
        "secondary_segment": 40,
        "secondary_segment_metrics": regression_metrics(
            truth[segments == 40], prediction[segments == 40]
        ),
        "same_collection_session": True,
        "independent_session_confirmation": False,
        "challenge_used_for_reselection": False,
        "thresholding_after_labels": False,
        "acceptance_gate": "none_diagnostic_only_after_frozen_development_selection",
    }
    try:
        _assert_json_close(report, expected_report, path="challenge.report")
    except GPUOutagePORDV5Error as exc:
        raise GPUOutagePORDV5Error("completed challenge report changed") from exc
    manifest_path = run / "manifest.json"
    expected_manifest = _final_manifest_payload(
        run, selected_candidate_id=str(completed["selected_candidate_id"])
    )
    if not manifest_path.is_file():
        if not repair_root_manifest:
            raise GPUOutagePORDV5Error("final v5 manifest is missing")
        atomic_write_json(manifest_path, expected_manifest)
    elif read_json_object(manifest_path) != expected_manifest:
        if not repair_root_manifest:
            raise GPUOutagePORDV5Error("final v5 manifest changed")
        atomic_write_json(manifest_path, expected_manifest)
    if read_json_object(manifest_path) != _final_manifest_payload(
        run, selected_candidate_id=str(completed["selected_candidate_id"])
    ):
        raise GPUOutagePORDV5Error("completed challenge report changed")
    return {
        "status": "ok",
        "stage": "challenge_complete_no_reselection",
        "selected_candidate_id": completed["selected_candidate_id"],
        "challenge_label_access_passes": 1,
        "recovered_without_source_reread": True,
        "report": report,
        "completed": completed,
    }


def finalize_pord_v5_challenge(
    run_dir: str | Path,
    *,
    resume: bool,
    finalize_challenge: bool,
    label_loader: Callable[[Mapping[str, Any], Path], Any] = _default_challenge_label_loader,
    verify_current_source: bool = True,
    _private_test_only_mode: bool = False,
) -> dict[str, Any]:
    """Open labels exactly once after an O_EXCL marker; never tune or reselect."""

    run = Path(run_dir).resolve()
    if not resume or not finalize_challenge:
        raise GPUOutagePORDV5Error(
            "challenge finalization requires both --resume and --finalize-challenge"
        )
    completed_path = run / "challenge" / "label_access_1.completed.json"
    started_path = run / "challenge" / "label_access_1.started.json"
    if started_path.exists() and not completed_path.is_file():
        raise GPUOutagePORDV5Error(
            "challenge access previously started without durable completion; fail closed"
        )
    mode = _require_mode_authority(
        run, private_test_only=_private_test_only_mode
    )
    if (
        label_loader is not _default_challenge_label_loader
        and mode != PORD_V5_SYNTHETIC_TEST_MODE
    ):
        raise GPUOutagePORDV5Error("custom challenge label loaders are private test-only")
    if completed_path.is_file():
        return _recover_completed_challenge(
            run,
            allow_synthetic_test_only=_private_test_only_mode,
            verify_live_environment=not _private_test_only_mode,
        )
    development = validate_pord_v5_development_run(
        run,
        verify_current_source=verify_current_source,
        allow_synthetic_test_only=_private_test_only_mode,
        verify_live_environment=not _private_test_only_mode,
    )
    if development["challenge_finalize_allowed"] is not True:
        raise GPUOutagePORDV5Error(
            "challenge finalize is refused until all claim gates and novelty gate pass"
        )
    preload = _preload_challenge_access_bundle(
        run,
        allow_synthetic_test_only=_private_test_only_mode,
        verify_live_environment=not _private_test_only_mode,
    )
    authority = _load_challenge_label_source_authority_runtime(
        run, preload=preload
    )
    if (
        (label_loader is _default_challenge_label_loader)
        is not (
            authority["descriptor"]["loader_kind"]
            == "production_default_v3_projection"
        )
    ):
        raise GPUOutagePORDV5Error("challenge label-loader authority kind changed")
    references = dict(preload["references"])
    parent = preload["parent_freeze"]
    _assert_parent_challenge_markers_absent(parent)
    selection = preload["selection"]
    identity = preload["identity"]
    started = {
        "contract": PORD_V5_CHALLENGE_ACCESS_CONTRACT,
        "access_index": 1,
        "status": "started_fail_closed",
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
        "config_sha256": preload["freeze"]["config_sha256"],
        "selected_candidate_id": selection["selected_candidate_id"],
        "development_manifest": references["development_manifest"],
        "selection_file": references["selection"],
        "freeze_file": references["freeze"],
        "prelabel_validation_file": references["prelabel_validation"],
        "recipe_file": references["recipe"],
        "selected_prediction_file": references["selected_prediction"],
        "parent_freeze_file": references["parent_freeze"],
        "challenge_identity_metadata_file": references[
            "challenge_identity_metadata"
        ],
        "challenge_identity_file": references["challenge_identity"],
        "label_source_authority_file": references["label_source_authority"],
        "frozen_v3_config_file": references.get("frozen_v3_config"),
        "prelabel_reference_chain_sha256": stable_hash(references),
        "selected_prediction_sha256": preload["freeze"][
            "selected_prediction_sha256"
        ],
        "identity_binding": preload["freeze"]["identity_binding"],
        "ordered_trial_uid_sha256": identity["ordered_trial_uid_sha256"],
        "maximum_label_access_passes": 1,
        "marker_creation": "O_CREAT|O_EXCL",
        "challenge_used_for_selection": False,
        "cross_parent_finalizer_exclusion": (
            "single_controller_assumption_not_machine_mutex"
        ),
    }
    started["started_sha256"] = stable_hash(started)
    _exclusive_write_json(started_path, started)
    started_file_sha256 = sha256_file(started_path)
    raw_labels = (
        _default_challenge_label_loader(identity, authority)
        if label_loader is _default_challenge_label_loader
        else label_loader(identity, run)
    )
    truth = _normalise_challenge_labels(raw_labels, identity)
    # Detect post-marker disk mutation, but never reload prediction values for
    # computation.  Metrics below use only the preloaded immutable array.
    for reference in references.values():
        _verify_reference(run, reference)
    _assert_parent_challenge_markers_absent(parent)
    if (
        sha256_file(started_path) != started_file_sha256
        or read_json_object(started_path) != started
    ):
        raise GPUOutagePORDV5Error("challenge access marker changed during label load")
    prediction = np.asarray(preload["prediction"], dtype=np.float64)
    evaluation = pd.DataFrame(
        {
            "trial_uid": np.asarray(identity["trial_uid"]).astype(str),
            "segment_id": np.asarray(identity["segment_id"], dtype=np.int64),
            "target_id": np.asarray(identity["target_id"]).astype(str),
        }
    )
    for axis, name in enumerate(_TRUE_COLUMNS):
        evaluation[name] = truth[:, axis]
    for axis, name in enumerate(("pred_x_mm", "pred_y_mm", "pred_z_mm")):
        evaluation[name] = prediction[:, axis]
    evaluation_path = atomic_write_parquet(
        run / "challenge" / "evaluation_once.parquet", evaluation
    )
    segments = np.asarray(identity["segment_id"], dtype=np.int64)
    primary_rows = segments == 41
    secondary_rows = segments == 40
    report = {
        "selected_candidate_id": selection["selected_candidate_id"],
        "overall_metrics": regression_metrics(truth, prediction),
        "primary_segment": 41,
        "primary_segment_metrics": regression_metrics(
            truth[primary_rows], prediction[primary_rows]
        ),
        "secondary_segment": 40,
        "secondary_segment_metrics": regression_metrics(
            truth[secondary_rows], prediction[secondary_rows]
        ),
        "same_collection_session": True,
        "independent_session_confirmation": False,
        "challenge_used_for_reselection": False,
        "thresholding_after_labels": False,
        "acceptance_gate": "none_diagnostic_only_after_frozen_development_selection",
    }
    report_path = atomic_write_json(run / "challenge" / "report.json", report)
    completed = {
        "contract": PORD_V5_CHALLENGE_ACCESS_CONTRACT,
        "access_index": 1,
        "status": "completed_exactly_once",
        "execution_mode": mode,
        "synthetic_test_only": mode == PORD_V5_SYNTHETIC_TEST_MODE,
        "selected_candidate_id": selection["selected_candidate_id"],
        "started_file": _artifact_reference(started_path, run),
        "started_sha256": started["started_sha256"],
        "development_manifest": references["development_manifest"],
        "selection_file": references["selection"],
        "report_file": _artifact_reference(report_path, run),
        "evaluation_file": _artifact_reference(evaluation_path, run, rows=3295),
        "freeze_file": references["freeze"],
        "prelabel_validation_file": references["prelabel_validation"],
        "recipe_file": references["recipe"],
        "selected_prediction_file": references["selected_prediction"],
        "parent_freeze_file": references["parent_freeze"],
        "challenge_identity_metadata_file": references[
            "challenge_identity_metadata"
        ],
        "challenge_identity_file": references["challenge_identity"],
        "label_source_authority_file": references["label_source_authority"],
        "frozen_v3_config_file": references.get("frozen_v3_config"),
        "prelabel_reference_chain_sha256": stable_hash(references),
        "identity_binding": preload["freeze"]["identity_binding"],
        "selected_prediction_sha256": preload["freeze"][
            "selected_prediction_sha256"
        ],
        "challenge_truth_sha256": array_sha256(truth),
        "challenge_label_access_passes": 1,
        "challenge_used_for_reselection": False,
        "thresholding_after_labels": False,
        "source_reread_for_recovery": False,
        "cross_parent_finalizer_exclusion": (
            "single_controller_assumption_not_machine_mutex"
        ),
    }
    _assert_parent_challenge_markers_absent(parent)
    _exclusive_write_json(completed_path, completed)
    manifest = _final_manifest_payload(
        run, selected_candidate_id=str(selection["selected_candidate_id"])
    )
    atomic_write_json(run / "manifest.json", manifest)
    recovered = _recover_completed_challenge(
        run,
        allow_synthetic_test_only=_private_test_only_mode,
        verify_live_environment=not _private_test_only_mode,
    )
    recovered["recovered_without_source_reread"] = False
    return recovered


def validate_pord_v5_final_run(
    run_dir: str | Path,
    *,
    verify_current_source: bool = True,
    allow_synthetic_test_only: bool = False,
    verify_live_environment: bool = True,
) -> dict[str, Any]:
    """Validate final evidence without reopening the challenge label source."""

    run = Path(run_dir).resolve()
    validate_pord_v5_development_run(
        run,
        verify_current_source=verify_current_source,
        allow_completed_challenge=True,
        allow_synthetic_test_only=allow_synthetic_test_only,
        verify_live_environment=verify_live_environment,
    )
    recovered = _recover_completed_challenge(
        run,
        allow_synthetic_test_only=allow_synthetic_test_only,
        verify_live_environment=verify_live_environment,
        repair_root_manifest=False,
    )
    manifest = read_json_object(run / "manifest.json")
    if (
        set(manifest)
        != {
            "contract",
            "experiment_contract",
            "stage",
            "selected_candidate_id",
            "challenge_label_access_passes",
            "primary_segment",
            "secondary_segment",
            "challenge_used_for_reselection",
            "files",
        }
        or manifest.get("contract") != PORD_V5_RUN_CONTRACT
        or manifest.get("experiment_contract") != PORD_V5_CONTRACT
        or manifest.get("stage") != "challenge_complete_no_reselection"
        or manifest.get("selected_candidate_id") != recovered["selected_candidate_id"]
        or type(manifest.get("challenge_label_access_passes")) is not int
        or manifest.get("challenge_label_access_passes") != 1
        or type(manifest.get("primary_segment")) is not int
        or manifest.get("primary_segment") != 41
        or type(manifest.get("secondary_segment")) is not int
        or manifest.get("secondary_segment") != 40
        or manifest.get("challenge_used_for_reselection") is not False
        or type(manifest.get("files")) is not list
        or manifest.get("files")
        != [row for row in manifest_files(run) if row["path"] != "manifest.json"]
    ):
        raise GPUOutagePORDV5Error("final v5 manifest changed")
    return {
        "status": "ok",
        "stage": "challenge_complete_no_reselection",
        "selected_candidate_id": recovered["selected_candidate_id"],
        "challenge_label_access_passes": 1,
        "primary_segment": 41,
        "secondary_segment": 40,
        "challenge_used_for_reselection": False,
        "source_archives_opened_by_validator": False,
    }


def run_pord_v5_experiment(
    config: Mapping[str, Any],
    *,
    v3_run_dir: str | Path,
    v4_run_dir: str | Path,
    run_dir: str | Path,
    workers_per_gpu: int = 1,
    resume: bool = False,
    finalize_challenge: bool = False,
    challenge_device: str = "cuda:0",
) -> dict[str, Any]:
    """Prepare/run/freeze, or finalize v5 under the single-controller rule."""

    run = Path(run_dir).resolve()
    if not (run / "prepared_manifest.json").is_file():
        if resume:
            raise GPUOutagePORDV5Error("cannot resume an unprepared v5 run")
        prepare_pord_v5_run(
            config,
            v3_run_dir=v3_run_dir,
            v4_run_dir=v4_run_dir,
            run_dir=run,
        )
    if finalize_challenge:
        if not resume:
            raise GPUOutagePORDV5Error(
                "--finalize-challenge is valid only with --resume"
            )
        if not (run / "challenge" / "freeze.before_label_access.json").is_file():
            raise GPUOutagePORDV5Error(
                "pre-label prediction/recipe freeze must complete in an earlier invocation"
            )
        return finalize_pord_v5_challenge(
            run, resume=True, finalize_challenge=True
        )
    development = run_pord_v5_development(
        run,
        workers_per_gpu=workers_per_gpu,
        resume=resume,
    )
    selection = read_json_object(run / "development" / "selection.json")
    freeze: dict[str, Any] | None = None
    if selection.get("challenge_finalize_allowed") is True:
        freeze = freeze_pord_v5_challenge_predictions(
            run, device=challenge_device
        )
    return {
        "status": "ok",
        "stage": selection["stage"],
        "development": development,
        "challenge_freeze": freeze,
        "challenge_labels_sealed": True,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sealed-label PORD-v5 outage experiment"
    )
    parser.add_argument(
        "--config",
        default="configs/endpoint_regression_gpu_outage_v5_pord.yaml",
    )
    parser.add_argument("--v3-run")
    parser.add_argument("--v4-run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers-per-gpu", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-challenge", action="store_true")
    parser.add_argument("--challenge-device", default="cuda:0")
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument("--validate-only", action="store_true")
    validation.add_argument("--validate-final-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.validate_only:
        result = validate_pord_v5_development_run(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.validate_final_only:
        result = validate_pord_v5_final_run(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not args.v3_run or not args.v4_run:
        raise GPUOutagePORDV5Error(
            "--v3-run and --v4-run are required unless validating"
        )
    config = load_pord_v5_config(args.config)
    result = run_pord_v5_experiment(
        config,
        v3_run_dir=args.v3_run,
        v4_run_dir=args.v4_run,
        run_dir=args.output,
        workers_per_gpu=args.workers_per_gpu,
        resume=args.resume,
        finalize_challenge=args.finalize_challenge,
        challenge_device=args.challenge_device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "GPUOutagePORDV5Error",
    "PORD_V5_CHALLENGE_ACCESS_CONTRACT",
    "PORD_V5_CHALLENGE_FREEZE_CONTRACT",
    "PORD_V5_PARENT_FREEZE_CONTRACT",
    "PORD_V5_RUN_CONTRACT",
    "PORD_V5_SOURCE_FILES",
    "PORD_V5_TASK_COMPLETION_CONTRACT",
    "PORD_V5_TASK_EXECUTION_CONTRACT",
    "PORD_V5_TASK_RESULT_CONTRACT",
    "aggregate_pord_v5_development",
    "assign_pord_v5_tasks",
    "evaluate_pord_v5_claim_gates",
    "finalize_pord_v5_challenge",
    "freeze_pord_v5_challenge_predictions",
    "main",
    "pord_v5_development_tasks",
    "pord_v5_environment_payload",
    "pord_v5_source_payload",
    "prepare_pord_v5_run",
    "row_positive_excess_summary",
    "run_pord_v5_development",
    "run_pord_v5_experiment",
    "score_pord_v5_candidates",
    "select_pord_v5_novel_candidate",
    "validate_pord_v5_challenge_freeze",
    "validate_pord_v5_development_run",
    "validate_pord_v5_final_run",
    "validate_pord_v5_prepared_run",
    "verify_pord_v5_controlled_results",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
