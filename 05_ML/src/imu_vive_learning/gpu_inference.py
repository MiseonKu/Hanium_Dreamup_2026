"""Safe offline inference for the versioned GPU endpoint MLP ensemble.

Unlike the legacy joblib package, the GPU model is stored as JSON metadata and
plain NumPy tensors.  This module never executes serialized Python objects: it
validates the model contract, referenced SHA-256 digests, preprocessing state,
and exact tensor schema before reconstructing the frozen PyTorch network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from numpy.lib import format as numpy_format
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from .experiment_io import atomic_write_parquet, stable_hash
from .features import FUSION_COLUMNS, LABEL_COLUMNS


GPU_MODEL_CONTRACT = "alphaf.endpoint_regression.gpu_model.v1"
ROBUST_GPU_MODEL_CONTRACT = "alphaf.endpoint_regression.gpu_model.robust_v2"
ROBUST_GPU_EXPERIMENT_CONTRACT = "alphaf.endpoint_regression.gpu_experiment.robust_v2"
ROBUST_MODEL_TRAINING_CONTRACT = "alphaf.endpoint_regression.gpu_training.robust_v2"
PREDICTION_COLUMNS = (
    "pred_vive_world_x_mm",
    "pred_vive_world_y_mm",
    "pred_vive_world_z_mm",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG_ID_PATTERN = re.compile(r"[0-9a-f]{16}\Z")
_MAX_JSON_BYTES = 1_000_000
_MAX_NPZ_BYTES = 2_000_000
_MAX_NPZ_UNCOMPRESSED_BYTES = 2_000_000
_ALLOWED_HIDDEN_LAYERS = {(64, 64), (128, 64)}
_ALLOWED_WEIGHT_DECAYS = {0.001, 0.0001}
_ROBUST_CANDIDATES = (
    {"candidate_id": "balanced_erm", "objective": "balanced_erm"},
    {"candidate_id": "group_dro_eta_0_01", "objective": "group_dro", "eta": 0.01},
    {"candidate_id": "group_dro_eta_0_1", "objective": "group_dro", "eta": 0.1},
    {
        "candidate_id": "v_rex_beta_10",
        "objective": "v_rex",
        "beta": 10.0,
        "warmup_epochs": 40,
    },
    {
        "candidate_id": "v_rex_beta_100",
        "objective": "v_rex",
        "beta": 100.0,
        "warmup_epochs": 40,
    },
    {
        "candidate_id": "balanced_erm_sam_rho_0_05",
        "objective": "balanced_erm",
        "sam": True,
        "rho": 0.05,
    },
)


class GPUInferenceError(ValueError):
    """Raised when a GPU model or inference table fails closed validation."""


def _reject_json_constant(value: str) -> None:
    raise GPUInferenceError(f"JSON contains a non-finite constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GPUInferenceError(f"JSON contains a duplicate key: {key!r}")
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise GPUInferenceError(f"missing GPU model JSON: {path}") from exc
    if not path.is_file() or size <= 0 or size > _MAX_JSON_BYTES:
        raise GPUInferenceError(f"GPU model JSON has an invalid size: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GPUInferenceError(f"cannot read GPU model JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise GPUInferenceError(f"GPU model JSON root must be an object: {path}")
    return payload


def _enforce_robust_promotion_gate(
    root: Path, *, allow_unpromoted_diagnostic: bool
) -> dict[str, Any] | None:
    """Reject an unpromoted robust-development checkpoint by default."""

    config_path = root / "config.resolved.json"
    recipe_path = root / "final" / "training_recipe.json"
    robust_config = False
    if config_path.is_file():
        resolved = _read_json_object(config_path)
        contract = resolved.get("contract")
        robust_config = (
            isinstance(contract, dict)
            and contract.get("name") == ROBUST_GPU_EXPERIMENT_CONTRACT
        )
    recipe_contract = None
    if recipe_path.is_file():
        recipe_contract = _read_json_object(recipe_path).get("contract")
    if not robust_config:
        return None
    if recipe_contract != ROBUST_MODEL_TRAINING_CONTRACT:
        raise GPUInferenceError("robust GPU model provenance is incomplete")

    run_manifest = _read_json_object(root / "manifest.json")
    if run_manifest.get("contract") != ROBUST_GPU_EXPERIMENT_CONTRACT:
        raise GPUInferenceError("robust GPU run manifest contract changed")
    recipe_reference = run_manifest.get("training_recipe")
    if not isinstance(recipe_reference, dict):
        raise GPUInferenceError("robust GPU training recipe reference is missing")
    _require_exact_keys(
        recipe_reference,
        {"path", "sha256"},
        name="robust GPU training recipe reference",
    )
    authenticated_recipe_path = _safe_member_path(
        root,
        recipe_reference.get("path"),
        expected="final/training_recipe.json",
    )
    expected_digest = _recorded_sha256(
        recipe_reference.get("sha256"), name="robust training recipe SHA-256"
    )
    if _sha256_file(authenticated_recipe_path) != expected_digest:
        raise GPUInferenceError("robust training recipe SHA-256 mismatch")
    recipe = _read_json_object(authenticated_recipe_path)
    eligible = recipe.get("promotion_eligible")
    action = recipe.get("promotion_action")
    if type(eligible) is not bool or action not in {"promote_robust_v2", "retain_v1"}:
        raise GPUInferenceError("robust GPU promotion evidence is invalid")
    if eligible is not (action == "promote_robust_v2"):
        raise GPUInferenceError("robust GPU promotion evidence is inconsistent")
    if allow_unpromoted_diagnostic:
        return None
    if not eligible:
        raise GPUInferenceError(
            "robust candidate did not beat the frozen v1 gate; retain_v1 forbids inference"
        )
    if run_manifest.get("status") != "complete":
        raise GPUInferenceError("robust GPU run is not durably validated and complete")
    validation_reference = run_manifest.get("validation")
    if not isinstance(validation_reference, dict):
        raise GPUInferenceError("robust GPU semantic validation reference is missing")
    _require_exact_keys(
        validation_reference,
        {"path", "sha256"},
        name="robust GPU semantic validation reference",
    )
    validation_path = _safe_member_path(
        root, validation_reference.get("path"), expected="validation.json"
    )
    validation_digest = _recorded_sha256(
        validation_reference.get("sha256"), name="robust validation SHA-256"
    )
    if _sha256_file(validation_path) != validation_digest:
        raise GPUInferenceError("robust semantic validation SHA-256 mismatch")
    validation = _read_json_object(validation_path)
    if (
        validation.get("status") != "ok"
        or validation.get("contract") != ROBUST_GPU_EXPERIMENT_CONTRACT
        or validation.get("promotion_action") != "promote_robust_v2"
        or validation.get("config_sha256") != recipe.get("robust_config_sha256")
        or validation.get("source_fingerprint") != recipe.get("source_fingerprint")
    ):
        raise GPUInferenceError("robust semantic validation evidence is inconsistent")
    return validation


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GPUInferenceError(f"cannot hash GPU model member: {path}") from exc
    return digest.hexdigest()


def _recorded_sha256(value: Any, *, name: str) -> str:
    digest = str(value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise GPUInferenceError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _safe_member_path(root: Path, recorded: Any, *, expected: str) -> Path:
    if not isinstance(recorded, str) or recorded != expected or "\\" in recorded:
        raise GPUInferenceError(
            f"GPU model member path changed; expected {expected!r}, received {recorded!r}"
        )
    pure = PurePosixPath(recorded)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise GPUInferenceError(f"unsafe GPU model member path: {recorded!r}")
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise GPUInferenceError(
            f"GPU model member is missing or escapes the run directory: {recorded!r}"
        ) from exc
    if not resolved.is_file():
        raise GPUInferenceError(f"GPU model member is not a file: {recorded!r}")
    return resolved


def _finite_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GPUInferenceError(f"{name} must be a numeric vector") from exc
    if result.shape != (length,) or not np.isfinite(result).all():
        raise GPUInferenceError(f"{name} must contain {length} finite values")
    result.setflags(write=False)
    return result


def _positive_finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise GPUInferenceError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GPUInferenceError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise GPUInferenceError(f"{name} must be a positive finite number")
    return result


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], *, name: str
) -> None:
    actual = set(payload)
    if actual != expected:
        raise GPUInferenceError(
            f"{name} field set changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _validate_model_manifest(
    manifest: Mapping[str, Any], *, allow_unpromoted_diagnostic: bool
) -> tuple[int, int]:
    contract = manifest.get("contract")
    robust = contract == ROBUST_GPU_MODEL_CONTRACT
    extra_fields = (
        {
            "candidate_id",
            "candidate",
            "promotion_eligible",
            "promotion_action",
            "training_recipe",
        }
        if robust
        else set()
    )
    _require_exact_keys(
        manifest,
        {
            "contract",
            "model_family",
            "ensemble_kind",
            "ensemble_seed_count",
            "feature_view",
            "feature_names",
            "label_columns",
            "label_frame",
            "label_unit",
            "selected_config_id",
            "selected_params",
            "selection_source",
            "selection_score_mm",
            "independent_test_score_available",
            "preprocessing",
            "checkpoints",
        }
        | extra_fields,
        name="GPU model manifest",
    )
    checks = {
        "contract": ROBUST_GPU_MODEL_CONTRACT if robust else GPU_MODEL_CONTRACT,
        "model_family": "torch_mlp",
        "ensemble_kind": "mean_raw_vive_world_xyz",
        "ensemble_seed_count": 3,
        "feature_view": "fusion_v1",
        "label_frame": "vive_world",
        "label_unit": "mm",
        "selection_source": "20_segment_inner_validation_contexts",
        "independent_test_score_available": False,
    }
    for key, expected in checks.items():
        actual = manifest.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise GPUInferenceError(
                f"GPU model manifest {key!r} changed: {actual!r}"
            )
    if robust:
        candidate = manifest.get("candidate")
        candidate_id = manifest.get("candidate_id")
        if (
            not isinstance(candidate, dict)
            or candidate not in _ROBUST_CANDIDATES
            or type(candidate_id) is not str
            or candidate.get("candidate_id") != candidate_id
        ):
            raise GPUInferenceError("robust GPU model candidate evidence is invalid")
        eligible = manifest.get("promotion_eligible")
        action = manifest.get("promotion_action")
        if type(eligible) is not bool or action not in {
            "promote_robust_v2",
            "retain_v1",
        }:
            raise GPUInferenceError("robust GPU model promotion evidence is invalid")
        if eligible is not (action == "promote_robust_v2"):
            raise GPUInferenceError("robust GPU model promotion evidence is inconsistent")
        recipe_reference = manifest.get("training_recipe")
        if not isinstance(recipe_reference, dict):
            raise GPUInferenceError("robust GPU model training recipe reference is missing")
        _require_exact_keys(
            recipe_reference,
            {"path", "sha256"},
            name="robust GPU model training recipe reference",
        )
        if not eligible and not allow_unpromoted_diagnostic:
            raise GPUInferenceError(
                "robust candidate did not beat the frozen v1 gate; retain_v1 forbids inference"
            )
    raw_feature_names = manifest.get("feature_names")
    if (
        not isinstance(raw_feature_names, list)
        or any(type(value) is not str for value in raw_feature_names)
        or tuple(raw_feature_names) != FUSION_COLUMNS
    ):
        raise GPUInferenceError("GPU model feature allow-list/order changed")
    raw_label_columns = manifest.get("label_columns")
    if (
        not isinstance(raw_label_columns, list)
        or any(type(value) is not str for value in raw_label_columns)
        or tuple(raw_label_columns) != LABEL_COLUMNS
    ):
        raise GPUInferenceError("GPU model label columns changed")

    params = manifest.get("selected_params")
    if not isinstance(params, dict) or set(params) != {
        "hidden_layer_sizes",
        "weight_decay",
    }:
        raise GPUInferenceError("GPU model selected-parameter schema changed")
    raw_hidden = params.get("hidden_layer_sizes")
    if (
        not isinstance(raw_hidden, list)
        or any(type(value) is not int for value in raw_hidden)
    ):
        raise GPUInferenceError("GPU model hidden-layer schema changed")
    hidden = tuple(raw_hidden)
    if hidden not in _ALLOWED_HIDDEN_LAYERS:
        raise GPUInferenceError(f"GPU model hidden layers are not frozen: {hidden!r}")
    if type(params.get("weight_decay")) is not float:
        raise GPUInferenceError("GPU model weight decay must be a JSON float")
    decay = _positive_finite_float(params.get("weight_decay"), name="weight_decay")
    if decay not in _ALLOWED_WEIGHT_DECAYS:
        raise GPUInferenceError("GPU model weight decay is not in the frozen grid")
    config_id = manifest.get("selected_config_id")
    if not isinstance(config_id, str) or _CONFIG_ID_PATTERN.fullmatch(config_id) is None:
        raise GPUInferenceError("GPU model selected_config_id is malformed")
    expected_config_id = stable_hash(
        {"model": "torch_mlp", "params": dict(params)}
    )[:16]
    if config_id != expected_config_id:
        raise GPUInferenceError("GPU model selected_config_id disagrees with parameters")
    score = manifest.get("selection_score_mm")
    if type(score) is not float:
        raise GPUInferenceError("GPU model selection score must be finite and non-negative")
    try:
        score_value = float(score)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GPUInferenceError(
            "GPU model selection score must be finite and non-negative"
        ) from exc
    if not math.isfinite(score_value) or score_value < 0.0:
        raise GPUInferenceError("GPU model selection score must be finite and non-negative")
    return hidden


@dataclass(frozen=True)
class _PreprocessingState:
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: float

    def transform_features(self, matrix: np.ndarray) -> np.ndarray:
        scaled = (matrix - self.feature_mean) / self.feature_scale
        result = np.asarray(scaled, dtype=np.float32)
        if result.shape != matrix.shape or not np.isfinite(result).all():
            raise GPUInferenceError("scaled GPU inference features are non-finite")
        return result

    def inverse_targets(self, scaled: np.ndarray) -> np.ndarray:
        result = (
            np.asarray(scaled, dtype=np.float64) * self.target_scale
            + self.target_mean
        )
        if result.ndim != 2 or result.shape[1:] != (3,) or not np.isfinite(result).all():
            raise GPUInferenceError("GPU model produced non-finite raw XYZ predictions")
        return result


def _validate_preprocessing(payload: Mapping[str, Any]) -> _PreprocessingState:
    _require_exact_keys(
        payload,
        {
            "feature_names",
            "feature_mean",
            "feature_scale",
            "target_mean",
            "target_scale",
            "leakage_audit",
        },
        name="GPU preprocessing",
    )
    raw_names = payload.get("feature_names")
    if (
        not isinstance(raw_names, list)
        or any(type(value) is not str for value in raw_names)
        or tuple(raw_names) != FUSION_COLUMNS
    ):
        raise GPUInferenceError("GPU preprocessing feature allow-list/order changed")
    names = tuple(raw_names)
    count = len(FUSION_COLUMNS)
    feature_mean = _finite_vector(
        payload.get("feature_mean"), name="feature_mean", length=count
    )
    feature_scale = _finite_vector(
        payload.get("feature_scale"), name="feature_scale", length=count
    )
    if np.any(feature_scale <= 0.0):
        raise GPUInferenceError("GPU preprocessing feature scales must be positive")
    target_mean = _finite_vector(
        payload.get("target_mean"), name="target_mean", length=3
    )
    if type(payload.get("target_scale")) is not float:
        raise GPUInferenceError("GPU preprocessing target_scale must be a JSON float")
    target_scale = _positive_finite_float(payload.get("target_scale"), name="target_scale")
    audit = payload.get("leakage_audit")
    if not isinstance(audit, dict):
        raise GPUInferenceError("GPU preprocessing leakage audit is missing")
    expected_audit = {
        "fit_scope": "real_train_only",
        "split_context": "all_eligible_real",
        "fit_row_count": 7702,
        "forbidden_uid_count": 0,
        "forbidden_uid_overlap_count": 0,
        "feature_scaling_enabled": True,
        "target_centered_per_axis": True,
        "target_scale_kind": "single_isotropic_xyz_scale",
    }
    _require_exact_keys(
        audit,
        set(expected_audit) | {"fit_trial_uid_sha256"},
        name="GPU preprocessing leakage audit",
    )
    for key, expected in expected_audit.items():
        actual = audit.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise GPUInferenceError(f"GPU preprocessing leakage audit changed at {key!r}")
    _recorded_sha256(
        audit.get("fit_trial_uid_sha256"), name="fit_trial_uid_sha256"
    )
    return _PreprocessingState(
        feature_names=names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
    )


def _build_torch_model(torch: Any, hidden: Sequence[int]) -> Any:
    layers: list[Any] = []
    previous = len(FUSION_COLUMNS)
    for width in hidden:
        layers.append(torch.nn.Linear(previous, int(width)))
        layers.append(torch.nn.ReLU())
        previous = int(width)
    layers.append(torch.nn.Linear(previous, 3))
    return torch.nn.Sequential(*layers)


def _validate_npz_container(
    path: Path, expected_tensors: Mapping[str, Any]
) -> None:
    expected_names = tuple(expected_tensors)
    try:
        if path.stat().st_size <= 0 or path.stat().st_size > _MAX_NPZ_BYTES:
            raise GPUInferenceError(f"GPU checkpoint has an invalid size: {path}")
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            expected_members = [f"{name}.npy" for name in expected_names]
            if names != expected_members or len(names) != len(set(names)):
                raise GPUInferenceError("GPU checkpoint NPZ member schema/order changed")
            if any(
                PurePosixPath(name).is_absolute()
                or len(PurePosixPath(name).parts) != 1
                or "\\" in name
                for name in names
            ):
                raise GPUInferenceError(
                    "GPU checkpoint contains an unsafe NPZ member path"
                )
            if sum(info.file_size for info in infos) > _MAX_NPZ_UNCOMPRESSED_BYTES:
                raise GPUInferenceError("GPU checkpoint uncompressed payload is too large")

            for info, tensor_name in zip(infos, expected_names, strict=True):
                expected_tensor = expected_tensors[tensor_name]
                with archive.open(info) as stream:
                    version = numpy_format.read_magic(stream)
                    if version != (1, 0):
                        raise GPUInferenceError(
                            "GPU checkpoint NPY format version changed"
                        )
                    shape, fortran_order, dtype = (
                        numpy_format.read_array_header_1_0(
                            stream, max_header_size=10_000
                        )
                    )
                    expected_shape = tuple(expected_tensor.shape)
                    expected_nbytes = math.prod(expected_shape) * np.dtype(
                        np.float32
                    ).itemsize
                    remaining_bytes = info.file_size - stream.tell()
                    if (
                        tuple(shape) != expected_shape
                        or fortran_order is not False
                        or dtype != np.dtype(np.float32)
                        or remaining_bytes != expected_nbytes
                    ):
                        raise GPUInferenceError(
                            f"GPU checkpoint tensor {tensor_name!r} header changed"
                        )
                    data = stream.read(expected_nbytes + 1)
                    if len(data) != expected_nbytes:
                        raise GPUInferenceError(
                            f"GPU checkpoint tensor {tensor_name!r} payload size changed"
                        )
    except GPUInferenceError:
        raise
    except (OSError, BadZipFile, EOFError, ValueError) as exc:
        raise GPUInferenceError(f"GPU checkpoint is not a valid NPZ file: {path}") from exc


def _load_state_dict(
    path: Path,
    *,
    torch: Any,
    model: Any,
    tensor_count: Any,
) -> dict[str, Any]:
    expected_state = model.state_dict()
    expected_names = tuple(expected_state)
    if type(tensor_count) is not int or tensor_count != len(expected_names):
        raise GPUInferenceError("GPU checkpoint tensor count changed")
    _validate_npz_container(path, expected_state)
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=False) as payload:
            if tuple(payload.files) != expected_names:
                raise GPUInferenceError("GPU checkpoint tensor key/order changed")
            for name, expected_tensor in expected_state.items():
                value = payload[name]
                if value.dtype != np.dtype(np.float32):
                    raise GPUInferenceError(
                        f"GPU checkpoint tensor {name!r} is not float32"
                    )
                if tuple(value.shape) != tuple(expected_tensor.shape):
                    raise GPUInferenceError(
                        f"GPU checkpoint tensor {name!r} has the wrong shape"
                    )
                if not np.isfinite(value).all():
                    raise GPUInferenceError(
                        f"GPU checkpoint tensor {name!r} is non-finite"
                    )
                arrays[name] = np.array(value, dtype=np.float32, copy=True, order="C")
    except GPUInferenceError:
        raise
    except (OSError, ValueError, TypeError, BadZipFile) as exc:
        raise GPUInferenceError(
            f"cannot safely read GPU checkpoint with allow_pickle=False: {path}"
        ) from exc
    return {name: torch.from_numpy(value) for name, value in arrays.items()}


def _resolve_device(torch: Any, requested: str | Any) -> Any:
    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise GPUInferenceError(f"invalid PyTorch inference device: {requested!r}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise GPUInferenceError("GPU endpoint inference supports only CPU or CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise GPUInferenceError("CUDA inference was requested but CUDA is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise GPUInferenceError(f"CUDA inference device index is unavailable: {index}")
    return device


@dataclass
class GPUEndpointEnsemble:
    """Validated three-seed PyTorch ensemble for static Vive-world XYZ output."""

    models: tuple[Any, ...]
    preprocessing: _PreprocessingState
    manifest: dict[str, Any]
    device: Any
    model_fingerprint: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.preprocessing.feature_names

    def validate_inference_frame(
        self, frame: Any
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not isinstance(frame, pd.DataFrame):
            raise GPUInferenceError("GPU inference input must be a pandas DataFrame")
        raw_columns = list(frame.columns)
        if any(not isinstance(column, str) for column in raw_columns):
            raise GPUInferenceError("GPU inference column names must be strings")
        if len(raw_columns) != len(set(raw_columns)):
            raise GPUInferenceError("GPU inference columns must be unique")
        expected = list(self.feature_names)
        row_ids: np.ndarray | None = None
        if raw_columns == ["row_id", *expected]:
            row_series = frame["row_id"]
            try:
                duplicate_ids = bool(row_series.duplicated().any())
            except TypeError as exc:
                raise GPUInferenceError("GPU inference row_id values must be scalar") from exc
            if row_series.isna().any() or duplicate_ids:
                raise GPUInferenceError(
                    "GPU inference row_id values must be non-missing and unique"
                )
            row_ids = row_series.to_numpy(copy=True)
        elif raw_columns != expected:
            raise GPUInferenceError(
                "GPU inference feature schema/order mismatch; expected the exact "
                "fusion_v1 allow-list, optionally preceded by row_id"
            )
        if len(frame) == 0:
            raise GPUInferenceError("GPU inference feature table must be non-empty")
        invalid_types = [
            name
            for name in expected
            if not is_numeric_dtype(frame[name].dtype)
            or is_bool_dtype(frame[name].dtype)
            or is_complex_dtype(frame[name].dtype)
        ]
        if invalid_types:
            raise GPUInferenceError(
                f"GPU inference features must be real numeric columns: {invalid_types}"
            )
        try:
            matrix = frame.loc[:, expected].to_numpy(dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GPUInferenceError("GPU inference features are not numeric") from exc
        if matrix.shape != (len(frame), len(expected)) or not np.isfinite(matrix).all():
            raise GPUInferenceError("GPU inference features must be finite")
        return matrix, row_ids

    def predict(self, matrix: Any, *, batch_size: int = 4096) -> np.ndarray:
        try:
            values = np.asarray(matrix, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GPUInferenceError("GPU inference feature matrix is not numeric") from exc
        if (
            values.ndim != 2
            or values.shape[1:] != (len(self.feature_names),)
            or len(values) == 0
            or not np.isfinite(values).all()
        ):
            raise GPUInferenceError(
                "GPU inference feature matrix must be non-empty, finite, and have 43 columns"
            )
        if type(batch_size) is not int or batch_size <= 0:
            raise GPUInferenceError("GPU inference batch_size must be a positive integer")
        scaled = self.preprocessing.transform_features(values)
        try:
            import torch
        except ImportError as exc:
            raise GPUInferenceError("PyTorch is required for GPU model inference") from exc
        raw_members: list[np.ndarray] = []
        with torch.inference_mode():
            for model in self.models:
                chunks: list[np.ndarray] = []
                for start in range(0, len(scaled), batch_size):
                    batch = torch.from_numpy(scaled[start : start + batch_size]).to(
                        self.device
                    )
                    output = model(batch)
                    if tuple(output.shape) != (len(batch), 3) or not bool(
                        torch.isfinite(output).all().item()
                    ):
                        raise GPUInferenceError(
                            "GPU model produced an invalid scaled XYZ prediction"
                        )
                    chunks.append(output.detach().cpu().numpy().astype(np.float64))
                raw_members.append(
                    self.preprocessing.inverse_targets(np.concatenate(chunks, axis=0))
                )
        result = np.mean(np.stack(raw_members, axis=0), axis=0, dtype=np.float64)
        if result.shape != (len(values), 3) or not np.isfinite(result).all():
            raise GPUInferenceError("GPU ensemble produced invalid raw XYZ predictions")
        return result

    def predict_frame(
        self, frame: Any, *, batch_size: int = 4096
    ) -> tuple[np.ndarray, np.ndarray | None]:
        matrix, row_ids = self.validate_inference_frame(frame)
        return self.predict(matrix, batch_size=batch_size), row_ids

    def prediction_frame(self, frame: Any, *, batch_size: int = 4096) -> pd.DataFrame:
        prediction, row_ids = self.predict_frame(frame, batch_size=batch_size)
        result = pd.DataFrame(
            {
                PREDICTION_COLUMNS[index]: prediction[:, index]
                for index in range(3)
            }
        )
        if row_ids is not None:
            result.insert(0, "row_id", row_ids)
        result["model_contract_fingerprint"] = self.model_fingerprint
        result["output_frame"] = "vive_world"
        result["output_unit"] = "mm"
        return result


def load_gpu_endpoint_model(
    run_dir: str | Path,
    *,
    device: str | Any = "cpu",
    allow_unpromoted_diagnostic: bool = False,
) -> GPUEndpointEnsemble:
    """Load a final GPU model directory without executing serialized code."""

    source = Path(run_dir)
    if source.name == "model_manifest.json":
        manifest_path = source.resolve()
        root = manifest_path.parent.parent.resolve()
    else:
        root = source.resolve()
        manifest_path = root / "final" / "model_manifest.json"
    robust_validation = _enforce_robust_promotion_gate(
        root, allow_unpromoted_diagnostic=allow_unpromoted_diagnostic
    )
    manifest = _read_json_object(manifest_path)
    hidden = _validate_model_manifest(
        manifest, allow_unpromoted_diagnostic=allow_unpromoted_diagnostic
    )
    if manifest["contract"] == ROBUST_GPU_MODEL_CONTRACT:
        if not allow_unpromoted_diagnostic and robust_validation is None:
            raise GPUInferenceError(
                "robust model default inference requires its completed validated run directory"
            )
        recipe_reference = manifest["training_recipe"]
        robust_recipe_path = _safe_member_path(
            root,
            recipe_reference.get("path"),
            expected="final/training_recipe.json",
        )
        robust_recipe_hash = _recorded_sha256(
            recipe_reference.get("sha256"), name="robust model training recipe SHA-256"
        )
        if _sha256_file(robust_recipe_path) != robust_recipe_hash:
            raise GPUInferenceError("robust model training recipe SHA-256 mismatch")
        robust_recipe = _read_json_object(robust_recipe_path)
        if (
            robust_recipe.get("contract") != ROBUST_MODEL_TRAINING_CONTRACT
            or robust_recipe.get("candidate_id") != manifest["candidate_id"]
            or robust_recipe.get("candidate") != manifest["candidate"]
            or robust_recipe.get("promotion_eligible")
            is not manifest["promotion_eligible"]
            or robust_recipe.get("promotion_action") != manifest["promotion_action"]
        ):
            raise GPUInferenceError("robust model/training recipe evidence differs")
    try:
        import torch
    except ImportError as exc:
        raise GPUInferenceError("PyTorch is required for GPU model inference") from exc
    resolved_device = _resolve_device(torch, device)

    preprocessing_reference = manifest.get("preprocessing")
    if not isinstance(preprocessing_reference, dict):
        raise GPUInferenceError("GPU model preprocessing reference is missing")
    _require_exact_keys(
        preprocessing_reference,
        {"path", "sha256"},
        name="GPU preprocessing reference",
    )
    preprocessing_path = _safe_member_path(
        root,
        preprocessing_reference.get("path"),
        expected="final/preprocessing.json",
    )
    preprocessing_hash = _recorded_sha256(
        preprocessing_reference.get("sha256"), name="preprocessing SHA-256"
    )
    if _sha256_file(preprocessing_path) != preprocessing_hash:
        raise GPUInferenceError("GPU preprocessing SHA-256 mismatch")
    preprocessing = _validate_preprocessing(_read_json_object(preprocessing_path))
    if tuple(manifest["feature_names"]) != preprocessing.feature_names:
        raise GPUInferenceError("GPU manifest/preprocessing feature contracts differ")

    checkpoint_entries = manifest.get("checkpoints")
    if not isinstance(checkpoint_entries, list) or len(checkpoint_entries) != 3:
        raise GPUInferenceError("GPU model must contain exactly three checkpoints")
    seed_indices = [
        entry.get("seed_index") if isinstance(entry, dict) else None
        for entry in checkpoint_entries
    ]
    if any(type(value) is not int for value in seed_indices) or seed_indices != [
        0,
        1,
        2,
    ]:
        raise GPUInferenceError("GPU checkpoint seed order changed")

    models: list[Any] = []
    checkpoint_fingerprints: list[dict[str, Any]] = []
    for seed_index, entry in enumerate(checkpoint_entries):
        assert isinstance(entry, dict)
        _require_exact_keys(
            entry,
            {
                "seed_index",
                "model_seed",
                "trained_epochs",
                "train_rmse_3d_mm",
                "path",
                "sha256",
                "tensor_count",
            },
            name=f"GPU checkpoint {seed_index} manifest entry",
        )
        model_seed = entry.get("model_seed")
        trained_epochs = entry.get("trained_epochs")
        if type(model_seed) is not int or model_seed < 0:
            raise GPUInferenceError("GPU checkpoint model seed is invalid")
        if type(trained_epochs) is not int or not 1 <= trained_epochs <= 400:
            raise GPUInferenceError("GPU checkpoint trained epoch count is invalid")
        train_rmse = entry.get("train_rmse_3d_mm")
        if type(train_rmse) is not float:
            raise GPUInferenceError("GPU checkpoint train RMSE is invalid")
        try:
            train_rmse_value = float(train_rmse)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GPUInferenceError("GPU checkpoint train RMSE is invalid") from exc
        if not math.isfinite(train_rmse_value) or train_rmse_value < 0.0:
            raise GPUInferenceError("GPU checkpoint train RMSE is invalid")
        checkpoint_path = _safe_member_path(
            root,
            entry.get("path"),
            expected=f"final/checkpoints/torch_mlp_seed{seed_index}.npz",
        )
        checkpoint_hash = _recorded_sha256(
            entry.get("sha256"), name=f"checkpoint {seed_index} SHA-256"
        )
        if _sha256_file(checkpoint_path) != checkpoint_hash:
            raise GPUInferenceError(f"GPU checkpoint {seed_index} SHA-256 mismatch")
        model = _build_torch_model(torch, hidden)
        state_dict = _load_state_dict(
            checkpoint_path,
            torch=torch,
            model=model,
            tensor_count=entry.get("tensor_count"),
        )
        try:
            model.load_state_dict(state_dict, strict=True)
            model.requires_grad_(False)
            model.eval()
            model.to(resolved_device)
        except (RuntimeError, ValueError) as exc:
            raise GPUInferenceError(
                f"cannot reconstruct GPU checkpoint {seed_index} on {resolved_device}"
            ) from exc
        models.append(model)
        checkpoint_fingerprints.append(
            {"seed_index": seed_index, "sha256": checkpoint_hash}
        )

    fingerprint_evidence = {
        "contract": manifest["contract"],
        "selected_config_id": manifest["selected_config_id"],
        "feature_names": list(FUSION_COLUMNS),
        "preprocessing_sha256": preprocessing_hash,
        "checkpoints": checkpoint_fingerprints,
    }
    if manifest["contract"] == ROBUST_GPU_MODEL_CONTRACT:
        fingerprint_evidence.update(
            {
                "candidate_id": manifest["candidate_id"],
                "candidate": manifest["candidate"],
                "promotion_eligible": manifest["promotion_eligible"],
                "promotion_action": manifest["promotion_action"],
                "training_recipe_sha256": manifest["training_recipe"]["sha256"],
            }
        )
    fingerprint = stable_hash(fingerprint_evidence)
    if robust_validation is not None and robust_validation.get(
        "model_fingerprint"
    ) != fingerprint:
        raise GPUInferenceError("robust semantic validation/model fingerprint mismatch")
    return GPUEndpointEnsemble(
        models=tuple(models),
        preprocessing=preprocessing,
        manifest=dict(manifest),
        device=resolved_device,
        model_fingerprint=fingerprint,
    )


def predict_gpu_endpoint(
    run_dir: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    device: str | Any = "cpu",
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Validate a final model, predict an ordered Parquet table, and write it."""

    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    try:
        frame = pd.read_parquet(source)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise GPUInferenceError(f"cannot read GPU inference Parquet: {source}") from exc
    ensemble = load_gpu_endpoint_model(run_dir, device=device)
    result = ensemble.prediction_frame(frame, batch_size=batch_size)
    atomic_write_parquet(output, result)
    return {
        "status": "ok",
        "contract": ensemble.manifest["contract"],
        "rows": int(len(result)),
        "output": str(output),
        "model_contract_fingerprint": ensemble.model_fingerprint,
        "frame": "vive_world",
        "unit": "mm",
    }


__all__ = [
    "GPUEndpointEnsemble",
    "GPUInferenceError",
    "GPU_MODEL_CONTRACT",
    "ROBUST_GPU_MODEL_CONTRACT",
    "PREDICTION_COLUMNS",
    "load_gpu_endpoint_model",
    "predict_gpu_endpoint",
]
