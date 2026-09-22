"""Frozen scalar-XGBoost training primitives for the outage-v4 follow-up.

This module is intentionally separate from the prospective outage-v3 neural
experiment.  It consumes already standardized, complete ``fusion_v1`` rows,
materializes all five predeclared availability states for every training UID,
and fits three independent scalar XGBoost boosters.  A forced evaluation
outage is never a fit argument: it is an evaluation-only feature transform.

The four IMUs form a supervised static endpoint-regression/fusion proxy.  Vive
coordinates may be supplied only as supervised targets by an outer runner;
they are never part of the 47 model inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
import json
import math
import numbers
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import numpy as np

from .config import canonical_json_bytes
from .features import FUSION_COLUMNS
from .gpu_outage_training import MASK_FULL_STATE, apply_sensor_masks
from .gpu_v3_backbones import (
    IMU_SENSOR_INDEX_MAP,
    V3_IMU_COUNT,
    V3_INPUT_FEATURE_COUNT,
    V3_SENSOR_FEATURE_COUNT,
)


GBDT_V4_BUNDLE_CONTRACT = "alphaf.endpoint_regression.gbdt_outage_v4.bundle"
GBDT_V4_SCHEMA_VERSION = 1
GBDT_V4_XGBOOST_VERSION = "3.3.0"
GBDT_V4_EXPECTED_LIBXGBOOST_SHA256 = (
    "528a3f97676e67977c29483868a3be5b4dd9da2a8766bdb7145f24adb06764a2"
)
GBDT_V4_DEVELOPMENT_SEGMENTS = (36, 37, 38, 39)
GBDT_V4_PAIRED_SEEDS = (20_260_824, 20_260_825, 20_260_826)
GBDT_V4_AXIS_NAMES = ("x", "y", "z")
AVAILABILITY_NAMES = (
    "imu0_base_available",
    "imu1_link_root_available",
    "imu2_link_mid_available",
    "imu3_tip_available",
)
GBDT_V4_MASK_STATES = (MASK_FULL_STATE, 0, 1, 2, 3)
GBDT_V4_MASK_PROBABILITIES = (0.25, 0.1875, 0.1875, 0.1875, 0.1875)
GBDT_V4_FEATURE_NAMES = FUSION_COLUMNS + AVAILABILITY_NAMES
GBDT_V4_NUM_BOOST_ROUND = 1_000
GBDT_V4_AXIS_SEED_RECIPE = (
    "uint32_big_endian_first4_sha256(canonical_json([paired_seed,axis_index]))"
)

# These are the complete frozen model parameters.  ``device``, the
# SHA256-derived axis ``seed``, and the train-only weighted-mean ``base_score``
# are added for each scalar fit.  No evaluation set or early stopping is used.
GBDT_V4_FIXED_XGB_PARAMS: Mapping[str, Any] = MappingProxyType({
    "booster": "gbtree",
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eta": 0.03,
    "max_depth": 6,
    "min_child_weight": 5.0,
    "subsample": 1.0,
    "sampling_method": "uniform",
    "colsample_bytree": 0.8,
    "colsample_bylevel": 1.0,
    "colsample_bynode": 1.0,
    "reg_lambda": 5.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "max_bin": 256,
    "grow_policy": "depthwise",
    "num_parallel_tree": 1,
    "seed_per_iteration": False,
    "nthread": 1,
})

_AXIS_FILENAMES = ("axis_x.ubj", "axis_y.ubj", "axis_z.ubj")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CUDA_DEVICE_RE = re.compile(r"^cuda:[0-9]+$")


class GBDTV4TrainingError(RuntimeError):
    """Raised when a v4 data, fit, model, or replay invariant is violated."""


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise GBDTV4TrainingError(f"{name} must be a boolean")
    return bool(value)


def _strict_int(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
    allowed: Sequence[int] | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, numbers.Integral
    ):
        raise GBDTV4TrainingError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise GBDTV4TrainingError(f"{name} must be at least {minimum}")
    if allowed is not None and result not in tuple(int(item) for item in allowed):
        raise GBDTV4TrainingError(f"{name} is outside the frozen values")
    return result


def _strict_string(value: Any, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise GBDTV4TrainingError(f"{name} must be a string")
    return value


def _strict_held_out_segment(value: Any, *, name: str) -> int:
    return _strict_int(
        value,
        name=name,
        allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
    )


def fixed_xgb_parameter_sha256() -> str:
    """Return the digest of the complete non-data-dependent XGBoost recipe."""

    payload = {
        "axis_seed_recipe": GBDT_V4_AXIS_SEED_RECIPE,
        "feature_names": list(GBDT_V4_FEATURE_NAMES),
        "fixed_params": dict(GBDT_V4_FIXED_XGB_PARAMS),
        "mask_probabilities": list(GBDT_V4_MASK_PROBABILITIES),
        "mask_states": list(GBDT_V4_MASK_STATES),
        "num_boost_round": GBDT_V4_NUM_BOOST_ROUND,
        "expected_libxgboost_sha256": GBDT_V4_EXPECTED_LIBXGBOOST_SHA256,
        "xgboost_version": GBDT_V4_XGBOOST_VERSION,
    }
    return sha256(canonical_json_bytes(payload)).hexdigest()


def derive_xgb_axis_seed(paired_seed: int, axis: int | str) -> int:
    """Derive the frozen uint32 seed from only ``(paired_seed, axis)``.

    Fold identity and forced-outage identity are deliberately absent.  Paired
    seeds therefore stay paired across held-out segments and outage contexts.
    """

    resolved_seed = _strict_int(
        paired_seed,
        name="paired_seed",
        allowed=GBDT_V4_PAIRED_SEEDS,
    )
    if isinstance(axis, str):
        if axis not in GBDT_V4_AXIS_NAMES:
            raise GBDTV4TrainingError("axis must be x, y, z, or index 0..2")
        axis_index = GBDT_V4_AXIS_NAMES.index(axis)
    elif isinstance(axis, (bool, np.bool_)) or not isinstance(
        axis, numbers.Integral
    ):
        raise GBDTV4TrainingError("axis must be x, y, z, or index 0..2")
    else:
        axis_index = int(axis)
        if axis_index not in range(3):
            raise GBDTV4TrainingError("axis must be x, y, z, or index 0..2")
    digest = sha256(canonical_json_bytes([resolved_seed, axis_index])).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _require_sha256(value: Any, *, name: str) -> str:
    text = _strict_string(value, name=name)
    if not _SHA256_RE.fullmatch(text):
        raise GBDTV4TrainingError(f"{name} must be lowercase SHA-256 hex")
    return text


def _array_sha256(values: Any) -> str:
    """Hash one numeric array with explicit shape/dtype/byte-order framing."""

    array = np.asarray(values)
    if array.dtype.hasobject:
        raise GBDTV4TrainingError("object arrays cannot enter numeric audit hashes")
    little_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(little_dtype, copy=False))
    header = canonical_json_bytes(
        {"dtype": little_dtype.str, "shape": list(canonical.shape)}
    )
    return sha256(header + b"\x00" + canonical.tobytes(order="C")).hexdigest()


def _sorted_uid_sha256(trial_uids: Sequence[str]) -> str:
    values = tuple(trial_uids)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise GBDTV4TrainingError("trial_uids must be non-empty strings")
    if len(values) != len(set(values)):
        raise GBDTV4TrainingError("trial_uids must be unique before replication")
    return sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _ordered_uid_sha256(trial_uids: Sequence[str]) -> str:
    """Hash ordered UID strings using canonical compact JSON framing."""

    values = tuple(trial_uids)
    _sorted_uid_sha256(values)
    return sha256(canonical_json_bytes(list(values))).hexdigest()


def _complete_matrix(values: Any, *, name: str, columns: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) == 0 or matrix.shape[1] != columns:
        raise GBDTV4TrainingError(
            f"{name} must be a non-empty matrix with {columns} columns"
        )
    if not np.isfinite(matrix).all():
        raise GBDTV4TrainingError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


def _source_uids(values: Sequence[Any], *, rows: int) -> tuple[str, ...]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != rows:
        raise GBDTV4TrainingError("trial_uids must contain one value per source row")
    normalized: list[str] = []
    for value in array.tolist():
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, str) or not value:
            raise GBDTV4TrainingError("trial_uids must be non-empty strings")
        normalized.append(value)
    _sorted_uid_sha256(normalized)
    return tuple(normalized)


def _source_segments(values: Sequence[Any], *, rows: int) -> tuple[int, ...]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != rows:
        raise GBDTV4TrainingError("segment_ids must contain one value per source row")
    normalized: list[int] = []
    for value in array.tolist():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Integral
        ):
            raise GBDTV4TrainingError("segment_ids must contain integer IDs")
        segment = int(value)
        if segment not in GBDT_V4_DEVELOPMENT_SEGMENTS:
            raise GBDTV4TrainingError(
                "segment_ids must stay within development segments 36--39"
            )
        normalized.append(segment)
    if not normalized:
        raise GBDTV4TrainingError("at least one source segment is required")
    return tuple(normalized)


def _segment_counts(segment_ids: Sequence[int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (segment, int(sum(value == segment for value in segment_ids)))
        for segment in sorted(set(segment_ids))
    )


@dataclass(frozen=True, slots=True, repr=False)
class GBDTReplicatedTrainingData:
    """The deterministic ``5 * N`` weighted training table for one bundle."""

    feature_names: tuple[str, ...]
    features: np.ndarray = field(repr=False)
    targets: np.ndarray = field(repr=False)
    sample_weight: np.ndarray = field(repr=False)
    source_row_indices: np.ndarray = field(repr=False)
    mask_states: np.ndarray = field(repr=False)
    source_trial_uids: tuple[str, ...]
    source_segment_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        names = tuple(
            _strict_string(value, name="feature_names item")
            for value in self.feature_names
        )
        if names != GBDT_V4_FEATURE_NAMES:
            raise GBDTV4TrainingError("GBDT feature order must be exact 47D v4 order")
        features = np.array(self.features, dtype=np.float32, copy=True, order="C")
        targets = np.array(self.targets, dtype=np.float32, copy=True, order="C")
        weights = np.array(self.sample_weight, dtype=np.float64, copy=True, order="C")
        raw_indices = np.asarray(self.source_row_indices)
        raw_states = np.asarray(self.mask_states)
        if not np.issubdtype(raw_indices.dtype, np.integer) or np.issubdtype(
            raw_indices.dtype, np.bool_
        ):
            raise GBDTV4TrainingError("source_row_indices must be integers")
        if not np.issubdtype(raw_states.dtype, np.integer) or np.issubdtype(
            raw_states.dtype, np.bool_
        ):
            raise GBDTV4TrainingError("mask_states must be integers")
        indices = np.array(self.source_row_indices, dtype=np.int64, copy=True)
        states = np.array(self.mask_states, dtype=np.int8, copy=True)
        source_uids = tuple(
            _strict_string(value, name="source_trial_uids item")
            for value in self.source_trial_uids
        )
        source_segments = tuple(
            _strict_int(
                value,
                name="source_segment_ids item",
                allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
            )
            for value in self.source_segment_ids
        )

        source_rows = len(source_uids)
        replicated_rows = source_rows * len(GBDT_V4_MASK_STATES)
        if source_rows == 0 or len(source_segments) != source_rows:
            raise GBDTV4TrainingError("source UID/segment metadata is misaligned")
        _sorted_uid_sha256(source_uids)
        if any(value not in GBDT_V4_DEVELOPMENT_SEGMENTS for value in source_segments):
            raise GBDTV4TrainingError("source segment metadata left segments 36--39")
        if features.shape != (replicated_rows, V3_INPUT_FEATURE_COUNT):
            raise GBDTV4TrainingError("replicated features must have shape [5*N, 47]")
        if targets.shape != (replicated_rows, 3):
            raise GBDTV4TrainingError("replicated targets must have shape [5*N, 3]")
        if weights.shape != (replicated_rows,) or indices.shape != (
            replicated_rows,
        ) or states.shape != (replicated_rows,):
            raise GBDTV4TrainingError("replicated row metadata must have length 5*N")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise GBDTV4TrainingError("replicated features/targets must be finite")
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise GBDTV4TrainingError("replicated sample weights must be positive")

        expected_indices = np.repeat(np.arange(source_rows, dtype=np.int64), 5)
        expected_states = np.tile(
            np.asarray(GBDT_V4_MASK_STATES, dtype=np.int8), source_rows
        )
        if not np.array_equal(indices, expected_indices):
            raise GBDTV4TrainingError("each source row must own exactly five replicas")
        if not np.array_equal(states, expected_states):
            raise GBDTV4TrainingError("replica mask order must be full, IMU0..IMU3")
        expected_targets = targets[::5]
        if not np.array_equal(targets, np.repeat(expected_targets, 5, axis=0)):
            raise GBDTV4TrainingError("all five replicas must retain one source target")

        masks = features[:, V3_SENSOR_FEATURE_COUNT:]
        expected_masks = np.ones((replicated_rows, V3_IMU_COUNT), dtype=np.float32)
        missing_rows = states >= 0
        expected_masks[
            np.flatnonzero(missing_rows), states[missing_rows].astype(np.int64)
        ] = 0.0
        if not np.array_equal(masks, expected_masks):
            raise GBDTV4TrainingError("availability flags disagree with mask states")
        complete_source_rows = features[::5, :V3_SENSOR_FEATURE_COUNT]
        expected_features = apply_sensor_masks(
            complete_source_rows[indices], states
        )
        if not np.array_equal(features, expected_features):
            raise GBDTV4TrainingError(
                "every replica must equal apply_sensor_masks of its complete row"
            )
        for sensor_index, sensor_columns in enumerate(IMU_SENSOR_INDEX_MAP):
            rows = states == sensor_index
            columns = np.asarray(sensor_columns, dtype=np.int64)
            if rows.any() and not np.equal(
                features[np.ix_(rows, columns)], 0.0
            ).all():
                raise GBDTV4TrainingError(
                    "unavailable standardized sensor slots must be exactly zero"
                )

        counts = dict(_segment_counts(source_segments))
        segment_count = len(counts)
        probabilities = np.asarray(GBDT_V4_MASK_PROBABILITIES, dtype=np.float64)
        expected_weights = np.empty(replicated_rows, dtype=np.float64)
        for source_index, segment in enumerate(source_segments):
            balance = source_rows / (segment_count * counts[segment])
            expected_weights[source_index * 5 : (source_index + 1) * 5] = (
                balance * probabilities
            )
        if not np.array_equal(weights, expected_weights):
            raise GBDTV4TrainingError(
                "weights must equal N/(K*n_segment) times the five q values"
            )
        if not math.isclose(
            float(weights.sum()), float(source_rows), rel_tol=0.0, abs_tol=1e-12
        ):
            raise GBDTV4TrainingError("total replicated objective mass must equal N")
        expected_segment_mass = source_rows / segment_count
        for segment in counts:
            replicated_segment = np.repeat(
                np.asarray(source_segments) == segment, 5
            )
            if not math.isclose(
                float(weights[replicated_segment].sum()),
                expected_segment_mass,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise GBDTV4TrainingError("objective mass must be equal by segment")

        for array in (features, targets, weights, indices, states):
            array.setflags(write=False)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "sample_weight", weights)
        object.__setattr__(self, "source_row_indices", indices)
        object.__setattr__(self, "mask_states", states)
        object.__setattr__(self, "source_trial_uids", source_uids)
        object.__setattr__(self, "source_segment_ids", source_segments)

    @property
    def source_rows(self) -> int:
        return len(self.source_trial_uids)

    @property
    def replicated_rows(self) -> int:
        return len(self.features)

    @property
    def sorted_trial_uid_sha256(self) -> str:
        return _sorted_uid_sha256(self.source_trial_uids)

    @property
    def ordered_trial_uid_sha256(self) -> str:
        return _ordered_uid_sha256(self.source_trial_uids)

    @property
    def row_segment_sha256(self) -> str:
        segments = np.asarray(self.source_segment_ids, dtype=np.int64)
        return _array_sha256(segments)

    @property
    def segment_counts(self) -> tuple[tuple[int, int], ...]:
        return _segment_counts(self.source_segment_ids)

    @property
    def data_sha256(self) -> str:
        payload = {
            "feature_sha256": _array_sha256(self.features),
            "mask_state_sha256": _array_sha256(self.mask_states),
            "ordered_trial_uid_sha256": self.ordered_trial_uid_sha256,
            "row_segment_int64_sha256": self.row_segment_sha256,
            "sample_weight_sha256": _array_sha256(self.sample_weight),
            "sorted_trial_uid_sha256": self.sorted_trial_uid_sha256,
            "source_row_index_sha256": _array_sha256(self.source_row_indices),
            "target_sha256": _array_sha256(self.targets),
        }
        return sha256(canonical_json_bytes(payload)).hexdigest()

    @property
    def weighted_base_scores(self) -> tuple[float, float, float]:
        result = np.average(self.targets.astype(np.float64), axis=0, weights=self.sample_weight)
        if result.shape != (3,) or not np.isfinite(result).all():
            raise GBDTV4TrainingError("train-only weighted base scores are invalid")
        return tuple(float(value) for value in result)

    def audit(self) -> dict[str, Any]:
        return {
            "data_sha256": self.data_sha256,
            "feature_count": V3_INPUT_FEATURE_COUNT,
            "mask_probabilities": list(GBDT_V4_MASK_PROBABILITIES),
            "mask_states": list(GBDT_V4_MASK_STATES),
            "replicas_per_uid": 5,
            "replicated_rows": self.replicated_rows,
            "segment_counts": [
                {"segment_id": segment, "rows": rows}
                for segment, rows in self.segment_counts
            ],
            "ordered_trial_uid_sha256": self.ordered_trial_uid_sha256,
            "row_segment_int64_sha256": self.row_segment_sha256,
            "sorted_trial_uid_sha256": self.sorted_trial_uid_sha256,
            "source_rows": self.source_rows,
            "total_sample_weight": float(self.sample_weight.sum()),
            "weighted_base_scores": list(self.weighted_base_scores),
        }

    def __repr__(self) -> str:
        return (
            "GBDTReplicatedTrainingData("
            f"source_rows={self.source_rows}, replicated_rows={self.replicated_rows}, "
            f"segments={tuple(segment for segment, _ in self.segment_counts)})"
        )


def build_mask_replicated_training_data(
    complete_scaled_features: Any,
    targets: Any,
    trial_uids: Sequence[Any],
    segment_ids: Sequence[Any],
    *,
    feature_names: Sequence[str] = FUSION_COLUMNS,
) -> GBDTReplicatedTrainingData:
    """Materialize exactly five weighted mask states for every source UID."""

    normalized_feature_names = tuple(
        _strict_string(value, name="feature_names item") for value in feature_names
    )
    if normalized_feature_names != FUSION_COLUMNS:
        raise GBDTV4TrainingError(
            "feature_names must be the exact ordered 43-column fusion_v1 allow-list"
        )
    features = _complete_matrix(
        complete_scaled_features,
        name="complete_scaled_features",
        columns=V3_SENSOR_FEATURE_COUNT,
    )
    labels = _complete_matrix(targets, name="targets", columns=3)
    if len(features) != len(labels):
        raise GBDTV4TrainingError("feature and target source row counts differ")
    uids = _source_uids(trial_uids, rows=len(features))
    segments = _source_segments(segment_ids, rows=len(features))

    source_indices = np.repeat(np.arange(len(features), dtype=np.int64), 5)
    states = np.tile(np.asarray(GBDT_V4_MASK_STATES, dtype=np.int8), len(features))
    replicated_features = apply_sensor_masks(features[source_indices], states)
    replicated_targets = labels[source_indices]

    counts = dict(_segment_counts(segments))
    segment_count = len(counts)
    q = np.asarray(GBDT_V4_MASK_PROBABILITIES, dtype=np.float64)
    weights = np.empty(len(states), dtype=np.float64)
    for source_index, segment in enumerate(segments):
        segment_balance = len(features) / (segment_count * counts[segment])
        weights[source_index * 5 : (source_index + 1) * 5] = segment_balance * q

    return GBDTReplicatedTrainingData(
        feature_names=GBDT_V4_FEATURE_NAMES,
        features=replicated_features,
        targets=replicated_targets,
        sample_weight=weights,
        source_row_indices=source_indices,
        mask_states=states,
        source_trial_uids=uids,
        source_segment_ids=segments,
    )


def _model_matrix(values: Any, *, name: str = "model_features") -> np.ndarray:
    features = _complete_matrix(values, name=name, columns=V3_INPUT_FEATURE_COUNT)
    availability = features[:, V3_SENSOR_FEATURE_COUNT:]
    if not np.isin(availability, (0.0, 1.0)).all():
        raise GBDTV4TrainingError("availability flags must be exact zero/one values")
    unavailable_count = np.sum(availability == 0.0, axis=1)
    if np.any(unavailable_count > 1):
        raise GBDTV4TrainingError(
            "v4 supports only complete or one-IMU-outage model rows"
        )
    for sensor_index, sensor_columns in enumerate(IMU_SENSOR_INDEX_MAP):
        rows = availability[:, sensor_index] == 0.0
        if rows.any() and not np.equal(
            features[np.ix_(rows, np.asarray(sensor_columns, dtype=np.int64))], 0.0
        ).all():
            raise GBDTV4TrainingError(
                "unavailable standardized sensor slots must be exactly zero"
            )
    return features


def transform_gbdt_outage_features(
    scaled_features: Any, availability: Any
) -> np.ndarray:
    """Append availability after validating zero-imputed unavailable slots."""

    features = _complete_matrix(
        scaled_features, name="scaled_features", columns=V3_SENSOR_FEATURE_COUNT
    )
    raw_masks = np.asarray(availability)
    if not (
        np.issubdtype(raw_masks.dtype, np.number)
        or np.issubdtype(raw_masks.dtype, np.bool_)
    ):
        raise GBDTV4TrainingError("availability must be numeric or boolean")
    masks = np.asarray(availability, dtype=np.float32)
    if masks.shape != (len(features), V3_IMU_COUNT) or not np.isfinite(masks).all():
        raise GBDTV4TrainingError("availability must have finite shape [rows, 4]")
    result = np.concatenate((features, masks), axis=1).astype(np.float32, copy=False)
    return _model_matrix(result, name="outage_model_features")


def force_gbdt_outage_features(
    complete_scaled_features: Any, forced_outage: int
) -> np.ndarray:
    """Build one evaluation-only forced-outage 47D matrix.

    This function is intentionally separate from the fit API, which always
    trains one bundle across all five mask replicas and has no outage argument.
    """

    outage_index = _strict_int(
        forced_outage,
        name="forced_outage",
        allowed=tuple(range(V3_IMU_COUNT)),
    )
    features = _complete_matrix(
        complete_scaled_features,
        name="complete_scaled_features",
        columns=V3_SENSOR_FEATURE_COUNT,
    )
    states = np.full(len(features), outage_index, dtype=np.int8)
    return _model_matrix(apply_sensor_masks(features, states), name="forced_outage")


@dataclass(frozen=True, slots=True, order=True)
class GBDTBundleSpec:
    """One fold/paired-seed unit; outage state is intentionally absent."""

    held_out_segment: int
    paired_seed: int

    def __post_init__(self) -> None:
        held_out = _strict_held_out_segment(
            self.held_out_segment, name="held_out_segment"
        )
        paired_seed = _strict_int(
            self.paired_seed,
            name="paired_seed",
            allowed=GBDT_V4_PAIRED_SEEDS,
        )
        object.__setattr__(self, "held_out_segment", held_out)
        object.__setattr__(self, "paired_seed", paired_seed)

    @property
    def scope(self) -> str:
        return "development_segment_loso"

    @property
    def bundle_id(self) -> str:
        return f"heldout_{self.held_out_segment}__paired_seed_{self.paired_seed}"


def development_bundle_specs() -> tuple[GBDTBundleSpec, ...]:
    """Return the exact 12 fold-by-seed fits, with no outage multiplier."""

    return tuple(
        GBDTBundleSpec(held_out_segment=segment, paired_seed=seed)
        for segment in GBDT_V4_DEVELOPMENT_SEGMENTS
        for seed in GBDT_V4_PAIRED_SEEDS
    )


def development_evaluation_contexts() -> tuple[tuple[GBDTBundleSpec, int], ...]:
    """Return 48 evaluation contexts backed by only the 12 fitted bundles."""

    return tuple(
        (spec, forced_outage)
        for spec in development_bundle_specs()
        for forced_outage in range(V3_IMU_COUNT)
    )


def _validate_scope_segments(
    segment_counts: Sequence[tuple[int, int]], held_out_segment: int
) -> None:
    normalized = tuple(
        (
            _strict_int(
                segment,
                name="source segment count ID",
                allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
            ),
            _strict_int(rows, name="source segment row count", minimum=1),
        )
        for segment, rows in segment_counts
    )
    observed = dict(normalized)
    if len(observed) != len(normalized) or any(
        rows <= 0 for rows in observed.values()
    ):
        raise GBDTV4TrainingError("source segment counts must be unique and positive")
    expected = set(GBDT_V4_DEVELOPMENT_SEGMENTS)
    expected.remove(
        _strict_int(
            held_out_segment,
            name="held_out_segment",
            allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
        )
    )
    if set(observed) != expected:
        raise GBDTV4TrainingError(
            "training segments do not match the held-out development scope"
        )


def _canonical_build_value(value: Any, *, name: str) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        result = float(value)
        if not math.isfinite(result):
            raise GBDTV4TrainingError(f"XGBoost build_info {name} is non-finite")
        return result
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        keys = tuple(value.keys())
        if any(not isinstance(key, str) or not key for key in keys):
            raise GBDTV4TrainingError("XGBoost build_info keys must be strings")
        for key in sorted(keys):
            normalized[key] = _canonical_build_value(
                value[key], name=f"{name}.{key}"
            )
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _canonical_build_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise GBDTV4TrainingError(
        f"XGBoost build_info {name} has an unsupported value type"
    )


@lru_cache(maxsize=4)
def _resolved_file_sha256(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns  # Cache-key provenance; content is read from the path.
    digest = sha256()
    with Path(path_text).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class XGBoostRuntimeProvenance:
    """Canonical XGBoost build identity and the exact resolved shared library."""

    version: str
    canonical_build_info_json: str
    canonical_build_info_sha256: str
    libxgboost_resolved_path: str
    libxgboost_sha256: str

    def __post_init__(self) -> None:
        version = _strict_string(self.version, name="runtime XGBoost version")
        if version != GBDT_V4_XGBOOST_VERSION:
            raise GBDTV4TrainingError("runtime provenance XGBoost version changed")
        build_json = _strict_string(
            self.canonical_build_info_json,
            name="canonical XGBoost build_info JSON",
        )
        try:
            build_info = json.loads(build_json)
        except json.JSONDecodeError as exc:
            raise GBDTV4TrainingError(
                "canonical XGBoost build_info JSON is invalid"
            ) from exc
        if not isinstance(build_info, Mapping) or (
            canonical_json_bytes(build_info).decode("utf-8") != build_json
        ):
            raise GBDTV4TrainingError("XGBoost build_info JSON is not canonical")
        build_digest = _require_sha256(
            self.canonical_build_info_sha256,
            name="canonical_build_info_sha256",
        )
        if sha256(build_json.encode("utf-8")).hexdigest() != build_digest:
            raise GBDTV4TrainingError("XGBoost build_info digest changed")
        library_path = _strict_string(
            self.libxgboost_resolved_path,
            name="libxgboost_resolved_path",
        )
        if not Path(library_path).is_absolute() or Path(library_path).name != "libxgboost.so":
            raise GBDTV4TrainingError(
                "runtime provenance must name an absolute libxgboost.so path"
            )
        library_digest = _require_sha256(
            self.libxgboost_sha256, name="libxgboost_sha256"
        )
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "canonical_build_info_json", build_json)
        object.__setattr__(self, "canonical_build_info_sha256", build_digest)
        object.__setattr__(self, "libxgboost_resolved_path", library_path)
        object.__setattr__(self, "libxgboost_sha256", library_digest)

    @property
    def canonical_build_info(self) -> dict[str, Any]:
        payload = json.loads(self.canonical_build_info_json)
        if not isinstance(payload, dict):  # Guarded by __post_init__.
            raise GBDTV4TrainingError("canonical build_info is not an object")
        return payload

    def metadata(self) -> dict[str, Any]:
        return {
            "canonical_build_info": self.canonical_build_info,
            "canonical_build_info_sha256": self.canonical_build_info_sha256,
            "libxgboost_resolved_path": self.libxgboost_resolved_path,
            "libxgboost_sha256": self.libxgboost_sha256,
            "xgboost_version": self.version,
        }


def _xgboost_runtime_provenance(xgboost: Any) -> XGBoostRuntimeProvenance:
    raw_info = xgboost.build_info()
    if not isinstance(raw_info, Mapping):
        raise GBDTV4TrainingError("XGBoost build_info must be a mapping")
    raw_library = _strict_string(
        raw_info.get("libxgboost"), name="XGBoost build_info.libxgboost"
    )
    try:
        library_path = Path(raw_library).resolve(strict=True)
    except OSError as exc:
        raise GBDTV4TrainingError("resolved libxgboost path is missing") from exc
    if not library_path.is_file() or library_path.name != "libxgboost.so":
        raise GBDTV4TrainingError("resolved XGBoost library is not libxgboost.so")
    canonical_raw = dict(raw_info)
    canonical_raw["libxgboost"] = library_path.name
    canonical_info = _canonical_build_value(canonical_raw, name="root")
    if not isinstance(canonical_info, dict):
        raise GBDTV4TrainingError("canonical XGBoost build_info must be an object")
    build_json = canonical_json_bytes(canonical_info).decode("utf-8")
    stat = library_path.stat()
    library_digest = _resolved_file_sha256(
        str(library_path), int(stat.st_size), int(stat.st_mtime_ns)
    )
    return XGBoostRuntimeProvenance(
        version=_strict_string(xgboost.__version__, name="xgboost.__version__"),
        canonical_build_info_json=build_json,
        canonical_build_info_sha256=sha256(build_json.encode("utf-8")).hexdigest(),
        libxgboost_resolved_path=str(library_path),
        libxgboost_sha256=library_digest,
    )


@dataclass(frozen=True, slots=True, repr=False)
class XGBBundle:
    """Three immutable scalar boosters plus authenticated training metadata."""

    spec: GBDTBundleSpec
    axis_model_bytes: tuple[bytes, bytes, bytes] = field(repr=False)
    axis_base_scores: tuple[float, float, float]
    axis_seeds: tuple[int, int, int]
    source_row_count: int
    source_segment_counts: tuple[tuple[int, int], ...]
    source_sorted_trial_uid_sha256: str
    source_ordered_trial_uid_sha256: str
    source_row_segment_sha256: str
    training_data_sha256: str
    training_prediction_sha256: str
    training_device: str
    runtime_provenance: XGBoostRuntimeProvenance
    xgboost_version: str = GBDT_V4_XGBOOST_VERSION
    feature_names: tuple[str, ...] = GBDT_V4_FEATURE_NAMES
    num_boost_round: int = GBDT_V4_NUM_BOOST_ROUND

    def __post_init__(self) -> None:
        if not isinstance(self.spec, GBDTBundleSpec):
            raise GBDTV4TrainingError("spec must be a GBDTBundleSpec")
        if any(
            not isinstance(value, (bytes, bytearray))
            for value in self.axis_model_bytes
        ):
            raise GBDTV4TrainingError("axis models must be UBJ byte payloads")
        models = tuple(bytes(value) for value in self.axis_model_bytes)
        if len(models) != 3 or any(not value for value in models):
            raise GBDTV4TrainingError("bundle must contain three non-empty boosters")
        if len(self.axis_base_scores) != 3 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Real)
            for value in self.axis_base_scores
        ):
            raise GBDTV4TrainingError("axis base scores must be three real values")
        base_scores = tuple(float(value) for value in self.axis_base_scores)
        if not all(math.isfinite(v) for v in base_scores):
            raise GBDTV4TrainingError("axis base scores must be three finite values")
        seeds = tuple(
            _strict_int(value, name="axis seed", minimum=0)
            for value in self.axis_seeds
        )
        expected_seeds = tuple(
            derive_xgb_axis_seed(self.spec.paired_seed, axis) for axis in range(3)
        )
        if seeds != expected_seeds:
            raise GBDTV4TrainingError("axis seeds do not match the SHA-256 recipe")
        source_row_count = _strict_int(
            self.source_row_count, name="source_row_count", minimum=1
        )
        normalized_counts: list[tuple[int, int]] = []
        for record in self.source_segment_counts:
            if (
                not isinstance(record, Sequence)
                or isinstance(record, (str, bytes))
                or len(record) != 2
            ):
                raise GBDTV4TrainingError(
                    "source_segment_counts records must be integer pairs"
                )
            normalized_counts.append(
                (
                    _strict_int(
                        record[0],
                        name="source segment count ID",
                        allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
                    ),
                    _strict_int(
                        record[1], name="source segment row count", minimum=1
                    ),
                )
            )
        counts = tuple(sorted(normalized_counts))
        _validate_scope_segments(counts, self.spec.held_out_segment)
        if sum(rows for _, rows in counts) != source_row_count:
            raise GBDTV4TrainingError("source segment counts disagree with row count")
        feature_names = tuple(
            _strict_string(value, name="bundle feature_names item")
            for value in self.feature_names
        )
        if feature_names != GBDT_V4_FEATURE_NAMES:
            raise GBDTV4TrainingError("bundle feature order changed")
        num_boost_round = _strict_int(
            self.num_boost_round, name="num_boost_round", minimum=1
        )
        if num_boost_round != GBDT_V4_NUM_BOOST_ROUND:
            raise GBDTV4TrainingError("bundle boosting-round count changed")
        xgboost_version = _strict_string(
            self.xgboost_version, name="bundle XGBoost version"
        )
        if xgboost_version != GBDT_V4_XGBOOST_VERSION:
            raise GBDTV4TrainingError("bundle XGBoost version changed")
        device = _strict_string(self.training_device, name="training_device")
        if device != "cpu" and not _CUDA_DEVICE_RE.fullmatch(device):
            raise GBDTV4TrainingError("training_device must be cuda[:index] or test CPU")
        for name, value in (
            ("source_sorted_trial_uid_sha256", self.source_sorted_trial_uid_sha256),
            ("source_ordered_trial_uid_sha256", self.source_ordered_trial_uid_sha256),
            ("source_row_segment_sha256", self.source_row_segment_sha256),
            ("training_data_sha256", self.training_data_sha256),
            ("training_prediction_sha256", self.training_prediction_sha256),
        ):
            _require_sha256(value, name=name)
        if not isinstance(self.runtime_provenance, XGBoostRuntimeProvenance):
            raise GBDTV4TrainingError(
                "runtime_provenance must be XGBoostRuntimeProvenance"
            )
        if self.runtime_provenance.version != xgboost_version:
            raise GBDTV4TrainingError("bundle/runtime XGBoost versions disagree")
        object.__setattr__(self, "axis_model_bytes", models)
        object.__setattr__(self, "axis_base_scores", base_scores)
        object.__setattr__(self, "axis_seeds", seeds)
        object.__setattr__(self, "source_row_count", source_row_count)
        object.__setattr__(self, "source_segment_counts", counts)
        object.__setattr__(self, "training_device", device)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "num_boost_round", num_boost_round)
        object.__setattr__(self, "xgboost_version", xgboost_version)

    @property
    def bundle_id(self) -> str:
        return self.spec.bundle_id

    @property
    def axis_model_sha256(self) -> tuple[str, str, str]:
        return tuple(sha256(value).hexdigest() for value in self.axis_model_bytes)  # type: ignore[return-value]

    def _core_metadata(self) -> dict[str, Any]:
        return {
            "axis_models": [
                {
                    "axis_index": axis,
                    "axis_name": GBDT_V4_AXIS_NAMES[axis],
                    "base_score": self.axis_base_scores[axis],
                    "bytes": len(self.axis_model_bytes[axis]),
                    "filename": _AXIS_FILENAMES[axis],
                    "seed": self.axis_seeds[axis],
                    "sha256": self.axis_model_sha256[axis],
                }
                for axis in range(3)
            ],
            "axis_seed_recipe": GBDT_V4_AXIS_SEED_RECIPE,
            "bundle_id": self.bundle_id,
            "contract": GBDT_V4_BUNDLE_CONTRACT,
            "feature_names": list(self.feature_names),
            "fixed_parameter_sha256": fixed_xgb_parameter_sha256(),
            "fixed_xgb_params": dict(GBDT_V4_FIXED_XGB_PARAMS),
            "num_boost_round": self.num_boost_round,
            "replication": {
                "mask_probabilities": list(GBDT_V4_MASK_PROBABILITIES),
                "mask_states": list(GBDT_V4_MASK_STATES),
                "replicas_per_uid": 5,
                "weight_formula": "N/(K*n_segment)*q_mask",
            },
            "schema_version": GBDT_V4_SCHEMA_VERSION,
            "scope": {
                "held_out_segment": self.spec.held_out_segment,
                "kind": self.spec.scope,
                "paired_seed": self.spec.paired_seed,
            },
            "source": {
                "ordered_trial_uid_sha256": self.source_ordered_trial_uid_sha256,
                "row_count": self.source_row_count,
                "row_segment_int64_sha256": self.source_row_segment_sha256,
                "segment_counts": [
                    {"segment_id": segment, "rows": rows}
                    for segment, rows in self.source_segment_counts
                ],
                "sorted_trial_uid_sha256": self.source_sorted_trial_uid_sha256,
            },
            "training_data_sha256": self.training_data_sha256,
            "training_device": self.training_device,
            "training_prediction_sha256": self.training_prediction_sha256,
            "runtime": self.runtime_provenance.metadata(),
            "xgboost_version": self.xgboost_version,
        }

    @property
    def bundle_sha256(self) -> str:
        return sha256(canonical_json_bytes(self._core_metadata())).hexdigest()

    def metadata(self) -> dict[str, Any]:
        payload = self._core_metadata()
        payload["bundle_sha256"] = self.bundle_sha256
        return payload

    def __repr__(self) -> str:
        return (
            f"XGBBundle(bundle_id={self.bundle_id!r}, source_rows={self.source_row_count}, "
            f"rounds={self.num_boost_round}, device={self.training_device!r})"
        )


def _require_xgboost(*, require_cuda_build: bool) -> Any:
    require_cuda = _strict_bool(
        require_cuda_build, name="require_cuda_build"
    )
    try:
        import xgboost
    except ImportError as exc:  # pragma: no cover - install diagnostic
        raise GBDTV4TrainingError("v4 requires xgboost==3.3.0") from exc
    version = _strict_string(xgboost.__version__, name="xgboost.__version__")
    if version != GBDT_V4_XGBOOST_VERSION:
        raise GBDTV4TrainingError(
            "v4 requires xgboost==3.3.0; "
            f"received {xgboost.__version__!r}"
        )
    runtime = _xgboost_runtime_provenance(xgboost)
    if runtime.libxgboost_sha256 != GBDT_V4_EXPECTED_LIBXGBOOST_SHA256:
        raise GBDTV4TrainingError(
            "resolved libxgboost.so SHA-256 differs from the frozen build"
        )
    use_cuda = runtime.canonical_build_info.get("USE_CUDA")
    if not isinstance(use_cuda, bool):
        raise GBDTV4TrainingError("XGBoost USE_CUDA build flag is not boolean")
    if require_cuda and not use_cuda:
        raise GBDTV4TrainingError("v4 production fitting requires a CUDA XGBoost build")
    return xgboost


def validate_xgboost_runtime_provenance(
    provenance: XGBoostRuntimeProvenance,
) -> dict[str, Any]:
    """Require the current imported XGBoost runtime to match stored provenance."""

    if not isinstance(provenance, XGBoostRuntimeProvenance):
        raise GBDTV4TrainingError(
            "runtime provenance must be XGBoostRuntimeProvenance"
        )
    xgboost = _require_xgboost(require_cuda_build=False)
    current = _xgboost_runtime_provenance(xgboost)
    if (
        current.version != provenance.version
        or current.canonical_build_info_json
        != provenance.canonical_build_info_json
        or current.canonical_build_info_sha256
        != provenance.canonical_build_info_sha256
        or current.libxgboost_sha256 != provenance.libxgboost_sha256
    ):
        raise GBDTV4TrainingError(
            "current XGBoost build/library differs from bundle provenance"
        )
    return {
        "canonical_build_info_sha256": current.canonical_build_info_sha256,
        "current_libxgboost_resolved_path": current.libxgboost_resolved_path,
        "libxgboost_sha256": current.libxgboost_sha256,
        "stored_libxgboost_resolved_path": provenance.libxgboost_resolved_path,
        "xgboost_version": current.version,
    }


def _resolve_fit_device(device: str, *, allow_cpu_for_testing: bool) -> str:
    resolved = _strict_string(device, name="device")
    allow_cpu = _strict_bool(
        allow_cpu_for_testing, name="allow_cpu_for_testing"
    )
    if _CUDA_DEVICE_RE.fullmatch(resolved):
        return resolved
    if resolved == "cpu" and allow_cpu:
        return resolved
    raise GBDTV4TrainingError(
        "v4 fitting requires device='cuda[:index]'; CPU is test-only"
    )


def _booster_from_bytes(xgboost: Any, model_bytes: bytes) -> Any:
    booster = xgboost.Booster()
    try:
        booster.load_model(bytearray(model_bytes))
    except Exception as exc:
        raise GBDTV4TrainingError("XGBoost UBJ booster could not be decoded") from exc
    return booster


def _predict_boosters(
    xgboost: Any,
    model_bytes: Sequence[bytes],
    features: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    matrix = xgboost.DMatrix(
        features,
        feature_names=list(GBDT_V4_FEATURE_NAMES),
        nthread=1,
    )
    predictions: list[np.ndarray] = []
    for raw in model_bytes:
        booster = _booster_from_bytes(xgboost, bytes(raw))
        booster.set_param({"device": device, "nthread": 1})
        predicted = np.asarray(
            booster.predict(matrix, validate_features=True, strict_shape=True),
            dtype=np.float64,
        ).reshape(len(features), -1)
        if predicted.shape != (len(features), 1) or not np.isfinite(predicted).all():
            raise GBDTV4TrainingError("scalar booster returned invalid predictions")
        predictions.append(predicted[:, 0])
    result = np.column_stack(predictions).astype(np.float64, copy=False)
    if result.shape != (len(features), 3) or not np.isfinite(result).all():
        raise GBDTV4TrainingError("XYZ bundle returned invalid predictions")
    return result


def _validate_live_booster_fit_config(
    booster: Any,
    *,
    axis_seed: int,
    base_score: float,
    device: str,
) -> None:
    """Audit train-time parameters before UBJ drops optimizer configuration.

    XGBoost's inference-model UBJ intentionally retains trees, objective,
    features, and base score but resets several train-only parameters when it
    is reloaded.  We therefore inspect those parameters on the live fitted
    booster and bind the frozen recipe into checkpoint metadata before saving.
    """

    try:
        config = json.loads(booster.save_config())
        learner = config["learner"]
        generic = learner["generic_param"]
        gradient = learner["gradient_booster"]
        model_param = learner["learner_model_param"]
        train_param = learner["learner_train_param"]
        tree = gradient["tree_train_param"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise GBDTV4TrainingError("live XGBoost fit config is incomplete") from exc
    if gradient.get("name") != "gbtree" or train_param.get("booster") != "gbtree":
        raise GBDTV4TrainingError("live booster family changed")
    if train_param.get("objective") != "reg:squarederror" or learner.get(
        "objective", {}
    ).get("name") != "reg:squarederror":
        raise GBDTV4TrainingError("live objective changed")
    if gradient.get("gbtree_train_param", {}).get("tree_method") != "hist":
        raise GBDTV4TrainingError("live tree method changed")
    if str(model_param.get("num_feature")) != str(V3_INPUT_FEATURE_COUNT) or str(
        model_param.get("num_target")
    ) != "1":
        raise GBDTV4TrainingError("live booster must be scalar 47D")
    if int(generic.get("seed", -1)) != int(axis_seed):
        raise GBDTV4TrainingError("live booster axis seed changed")
    if str(generic.get("seed_per_iteration")) != "0" or str(
        generic.get("nthread")
    ) != "1":
        raise GBDTV4TrainingError("live seed/thread policy changed")
    observed_device = str(generic.get("device"))
    if device == "cpu":
        if observed_device != "cpu":
            raise GBDTV4TrainingError("live test CPU device changed")
    elif observed_device != device:
        raise GBDTV4TrainingError(
            "live CUDA device ordinal differs from the requested ordinal"
        )
    if not math.isclose(
        _serialized_base_score(model_param.get("base_score")),
        float(base_score),
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise GBDTV4TrainingError("live train-only base score changed")

    expected_text = {
        "grow_policy": "depthwise",
        "max_depth": "6",
        "max_bin": "256",
        "sampling_method": "uniform",
    }
    for key, expected in expected_text.items():
        if str(tree.get(key)) != expected:
            raise GBDTV4TrainingError(f"live {key} changed")
    expected_float = {
        "alpha": 0.0,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "colsample_bytree": 0.8,
        "eta": 0.03,
        "gamma": 0.0,
        "lambda": 5.0,
        "min_child_weight": 5.0,
        "subsample": 1.0,
    }
    for key, expected in expected_float.items():
        if not math.isclose(
            _config_float(tree.get(key), name=key),
            expected,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise GBDTV4TrainingError(f"live {key} changed")
    if str(gradient.get("gbtree_model_param", {}).get("num_parallel_tree")) != "1":
        raise GBDTV4TrainingError("live num_parallel_tree changed")


def fit_xgb_bundle(
    complete_scaled_features: Any,
    targets: Any,
    trial_uids: Sequence[Any],
    segment_ids: Sequence[Any],
    *,
    held_out_segment: int,
    paired_seed: int,
    device: str = "cuda:0",
    feature_names: Sequence[str] = FUSION_COLUMNS,
    allow_cpu_for_testing: bool = False,
) -> XGBBundle:
    """Fit one fold/seed bundle for exactly 1,000 rounds per XYZ axis.

    There is intentionally no ``forced_outage`` parameter and no validation
    target/evaluation-set parameter.  Selection contexts are scored later by
    applying the four evaluation-only outage transforms to the same bundle.
    """

    spec = GBDTBundleSpec(
        held_out_segment=held_out_segment, paired_seed=paired_seed
    )
    resolved_device = _resolve_fit_device(
        device, allow_cpu_for_testing=allow_cpu_for_testing
    )
    training = build_mask_replicated_training_data(
        complete_scaled_features,
        targets,
        trial_uids,
        segment_ids,
        feature_names=feature_names,
    )
    _validate_scope_segments(training.segment_counts, spec.held_out_segment)
    xgboost = _require_xgboost(require_cuda_build=resolved_device != "cpu")
    runtime_provenance = _xgboost_runtime_provenance(xgboost)

    axis_seeds = tuple(
        derive_xgb_axis_seed(spec.paired_seed, axis) for axis in range(3)
    )
    base_scores = training.weighted_base_scores
    model_bytes: list[bytes] = []
    original_predictions: list[np.ndarray] = []
    for axis in range(3):
        matrix = xgboost.DMatrix(
            training.features,
            label=training.targets[:, axis],
            weight=training.sample_weight,
            feature_names=list(GBDT_V4_FEATURE_NAMES),
            nthread=1,
        )
        params = dict(GBDT_V4_FIXED_XGB_PARAMS)
        params.update(
            {
                "base_score": base_scores[axis],
                "device": resolved_device,
                "seed": axis_seeds[axis],
            }
        )
        try:
            booster = xgboost.train(
                params,
                matrix,
                num_boost_round=GBDT_V4_NUM_BOOST_ROUND,
                evals=(),
                verbose_eval=False,
            )
            _validate_live_booster_fit_config(
                booster,
                axis_seed=axis_seeds[axis],
                base_score=base_scores[axis],
                device=resolved_device,
            )
            raw = bytes(booster.save_raw(raw_format="ubj"))
            predicted = np.asarray(
                booster.predict(matrix, validate_features=True, strict_shape=True),
                dtype=np.float64,
            ).reshape(len(training.features), -1)
        except Exception as exc:
            raise GBDTV4TrainingError(
                f"XGBoost axis {GBDT_V4_AXIS_NAMES[axis]} fit failed"
            ) from exc
        if predicted.shape != (len(training.features), 1) or not np.isfinite(
            predicted
        ).all():
            raise GBDTV4TrainingError("trained scalar booster prediction is invalid")
        model_bytes.append(raw)
        original_predictions.append(predicted[:, 0])

    fitted_prediction = np.column_stack(original_predictions).astype(
        np.float64, copy=False
    )
    serialized_replay = _predict_boosters(
        xgboost,
        model_bytes,
        training.features,
        device=resolved_device,
    )
    if not np.array_equal(fitted_prediction, serialized_replay):
        raise GBDTV4TrainingError(
            "in-memory and UBJ-reloaded training predictions differ"
        )
    bundle = XGBBundle(
        spec=spec,
        axis_model_bytes=(model_bytes[0], model_bytes[1], model_bytes[2]),
        axis_base_scores=base_scores,
        axis_seeds=axis_seeds,
        source_row_count=training.source_rows,
        source_segment_counts=training.segment_counts,
        source_sorted_trial_uid_sha256=training.sorted_trial_uid_sha256,
        source_ordered_trial_uid_sha256=training.ordered_trial_uid_sha256,
        source_row_segment_sha256=training.row_segment_sha256,
        training_data_sha256=training.data_sha256,
        training_prediction_sha256=_array_sha256(serialized_replay),
        training_device=resolved_device,
        runtime_provenance=runtime_provenance,
    )
    validate_xgb_bundle_models(bundle)
    return bundle


def _config_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GBDTV4TrainingError(f"serialized booster {name} is not numeric") from exc
    if not math.isfinite(result):
        raise GBDTV4TrainingError(f"serialized booster {name} is non-finite")
    return result


def _serialized_base_score(value: Any) -> float:
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return _config_float(text, name="base_score")


def validate_xgb_bundle_models(bundle: XGBBundle) -> dict[str, Any]:
    """Decode all boosters and verify rounds, scalar shape, params, and seeds."""

    if not isinstance(bundle, XGBBundle):
        raise GBDTV4TrainingError("bundle must be an XGBBundle")
    runtime_audit = validate_xgboost_runtime_provenance(bundle.runtime_provenance)
    xgboost = _require_xgboost(require_cuda_build=False)
    for axis, raw in enumerate(bundle.axis_model_bytes):
        booster = _booster_from_bytes(xgboost, raw)
        if int(booster.num_boosted_rounds()) != GBDT_V4_NUM_BOOST_ROUND:
            raise GBDTV4TrainingError("serialized booster round count is not 1000")
        if tuple(booster.feature_names or ()) != GBDT_V4_FEATURE_NAMES:
            raise GBDTV4TrainingError("serialized booster feature order changed")
        attributes = booster.attributes()
        if "best_iteration" in attributes or "best_score" in attributes:
            raise GBDTV4TrainingError("early-stopping attributes are forbidden")
        try:
            config = json.loads(booster.save_config())
            learner = config["learner"]
            gradient = learner["gradient_booster"]
            model_param = learner["learner_model_param"]
            train_param = learner["learner_train_param"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise GBDTV4TrainingError("serialized XGBoost config is incomplete") from exc
        if gradient.get("name") != "gbtree" or train_param.get("booster") != "gbtree":
            raise GBDTV4TrainingError("serialized booster family changed")
        if train_param.get("objective") != "reg:squarederror" or learner.get(
            "objective", {}
        ).get("name") != "reg:squarederror":
            raise GBDTV4TrainingError("serialized objective changed")
        if str(model_param.get("num_feature")) != str(V3_INPUT_FEATURE_COUNT) or str(
            model_param.get("num_target")
        ) != "1":
            raise GBDTV4TrainingError("serialized booster must be scalar 47D")
        if not math.isclose(
            _serialized_base_score(model_param.get("base_score")),
            bundle.axis_base_scores[axis],
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise GBDTV4TrainingError("serialized train-only base score changed")

        if str(gradient.get("gbtree_model_param", {}).get("num_parallel_tree")) != "1":
            raise GBDTV4TrainingError("serialized num_parallel_tree changed")
    return {
        "axis_model_sha256": list(bundle.axis_model_sha256),
        "axis_seeds": list(bundle.axis_seeds),
        "bundle_id": bundle.bundle_id,
        "bundle_sha256": bundle.bundle_sha256,
        "early_stopping_attributes": 0,
        "feature_count": V3_INPUT_FEATURE_COUNT,
        "num_boost_round": GBDT_V4_NUM_BOOST_ROUND,
        "runtime": runtime_audit,
        "scalar_booster_count": 3,
    }


def predict_xgb_bundle_matrix(
    bundle: XGBBundle,
    model_features: Any,
    *,
    device: str | None = None,
) -> np.ndarray:
    """Predict scaled XYZ from an already validated 47D outage matrix."""

    if not isinstance(bundle, XGBBundle):
        raise GBDTV4TrainingError("bundle must be an XGBBundle")
    features = _model_matrix(model_features)
    resolved_device = (
        bundle.training_device
        if device is None
        else _strict_string(device, name="prediction device")
    )
    if resolved_device != "cpu" and not _CUDA_DEVICE_RE.fullmatch(resolved_device):
        raise GBDTV4TrainingError("prediction device must be cpu or cuda[:index]")
    xgboost = _require_xgboost(require_cuda_build=resolved_device != "cpu")
    return _predict_boosters(
        xgboost,
        bundle.axis_model_bytes,
        features,
        device=resolved_device,
    )


def predict_xgb_bundle(
    bundle: XGBBundle,
    scaled_features: Any,
    availability: Any,
    *,
    device: str | None = None,
) -> np.ndarray:
    """Predict scaled XYZ for natural/full availability supplied by a caller."""

    model_features = transform_gbdt_outage_features(scaled_features, availability)
    return predict_xgb_bundle_matrix(bundle, model_features, device=device)


def predict_xgb_bundle_forced_outage(
    bundle: XGBBundle,
    complete_scaled_features: Any,
    forced_outage: int,
    *,
    device: str | None = None,
) -> np.ndarray:
    """Predict one evaluation-only forced-outage context."""

    model_features = force_gbdt_outage_features(
        complete_scaled_features, forced_outage
    )
    return predict_xgb_bundle_matrix(bundle, model_features, device=device)


def validate_xgb_bundle_replay(
    bundle: XGBBundle,
    model_features: Any,
    expected_prediction: Any,
    *,
    expected_prediction_sha256: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Require exact prediction replay from the immutable UBJ boosters."""

    features = _model_matrix(model_features)
    expected = _complete_matrix(
        expected_prediction, name="expected_prediction", columns=3
    ).astype(np.float64)
    if len(expected) != len(features):
        raise GBDTV4TrainingError("expected prediction rows do not align")
    expected_hash = _array_sha256(expected)
    if expected_prediction_sha256 is not None and expected_hash != _require_sha256(
        expected_prediction_sha256, name="expected_prediction_sha256"
    ):
        raise GBDTV4TrainingError("expected prediction hash disagrees with its rows")
    replay = predict_xgb_bundle_matrix(bundle, features, device=device)
    replay_hash = _array_sha256(replay)
    if replay_hash != expected_hash or not np.array_equal(replay, expected):
        raise GBDTV4TrainingError("serialized bundle prediction replay differs")
    return {
        "bundle_sha256": bundle.bundle_sha256,
        "prediction_sha256": replay_hash,
        "replay_exact": True,
        "rows": len(replay),
    }


def validate_xgb_training_replay(
    bundle: XGBBundle,
    complete_scaled_features: Any,
    targets: Any,
    trial_uids: Sequence[Any],
    segment_ids: Sequence[Any],
    *,
    device: str | None = None,
    feature_names: Sequence[str] = FUSION_COLUMNS,
) -> dict[str, Any]:
    """Rebuild the 5*N table and replay the bundle's frozen training digest."""

    training = build_mask_replicated_training_data(
        complete_scaled_features,
        targets,
        trial_uids,
        segment_ids,
        feature_names=feature_names,
    )
    _validate_scope_segments(training.segment_counts, bundle.spec.held_out_segment)
    if training.source_rows != bundle.source_row_count:
        raise GBDTV4TrainingError("training replay source row count changed")
    if training.segment_counts != bundle.source_segment_counts:
        raise GBDTV4TrainingError("training replay segment counts changed")
    if training.sorted_trial_uid_sha256 != bundle.source_sorted_trial_uid_sha256:
        raise GBDTV4TrainingError("training replay UID identity changed")
    if training.ordered_trial_uid_sha256 != bundle.source_ordered_trial_uid_sha256:
        raise GBDTV4TrainingError("training replay ordered UID identity changed")
    if training.row_segment_sha256 != bundle.source_row_segment_sha256:
        raise GBDTV4TrainingError("training replay row-level segment identity changed")
    if training.data_sha256 != bundle.training_data_sha256:
        raise GBDTV4TrainingError("training replay data digest changed")
    replay = predict_xgb_bundle_matrix(bundle, training.features, device=device)
    replay_hash = _array_sha256(replay)
    if replay_hash != bundle.training_prediction_sha256:
        raise GBDTV4TrainingError("training prediction replay digest changed")
    return {
        "bundle_sha256": bundle.bundle_sha256,
        "data_sha256": training.data_sha256,
        "prediction_sha256": replay_hash,
        "ordered_trial_uid_sha256": training.ordered_trial_uid_sha256,
        "replay_exact": True,
        "replicated_rows": training.replicated_rows,
        "row_segment_int64_sha256": training.row_segment_sha256,
        "source_rows": training.source_rows,
    }


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def save_xgb_bundle(bundle: XGBBundle, directory: str | Path) -> dict[str, Any]:
    """Write one immutable four-file checkpoint directory without overwrite."""

    validation = validate_xgb_bundle_models(bundle)
    root = Path(directory)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise GBDTV4TrainingError(
            f"bundle checkpoint directory already exists: {root}"
        ) from exc
    for filename, payload in zip(_AXIS_FILENAMES, bundle.axis_model_bytes, strict=True):
        _exclusive_write(root / filename, payload)
    metadata_bytes = canonical_json_bytes(bundle.metadata()) + b"\n"
    _exclusive_write(root / "metadata.json", metadata_bytes)
    return {
        **validation,
        "checkpoint_directory": str(root),
        "metadata_sha256": sha256(metadata_bytes).hexdigest(),
    }


_UNSET = object()


def _metadata_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise GBDTV4TrainingError(f"{name} must be a JSON object")
    return value


def _metadata_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise GBDTV4TrainingError(f"{name} must be a JSON array")
    return value


def _metadata_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise GBDTV4TrainingError(f"{name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise GBDTV4TrainingError(f"{name} must be finite")
    return result


def load_xgb_bundle(
    directory: str | Path,
    *,
    expected_held_out_segment: int | object = _UNSET,
    expected_paired_seed: int | None = None,
    expected_source_sorted_trial_uid_sha256: str | None = None,
    expected_source_ordered_trial_uid_sha256: str | None = None,
    expected_source_row_segment_sha256: str | None = None,
    validate_models: bool = True,
) -> XGBBundle:
    """Load a hash-bound UBJ checkpoint and fail on missing or extra members."""

    should_validate_models = _strict_bool(validate_models, name="validate_models")
    resolved_expected_held_out: int | object = _UNSET
    if expected_held_out_segment is not _UNSET:
        resolved_expected_held_out = _strict_held_out_segment(
            expected_held_out_segment, name="expected_held_out_segment"
        )
    resolved_expected_seed = None
    if expected_paired_seed is not None:
        resolved_expected_seed = _strict_int(
            expected_paired_seed,
            name="expected_paired_seed",
            allowed=GBDT_V4_PAIRED_SEEDS,
        )
    expected_sorted_uid = (
        None
        if expected_source_sorted_trial_uid_sha256 is None
        else _require_sha256(
            expected_source_sorted_trial_uid_sha256,
            name="expected_source_sorted_trial_uid_sha256",
        )
    )
    expected_ordered_uid = (
        None
        if expected_source_ordered_trial_uid_sha256 is None
        else _require_sha256(
            expected_source_ordered_trial_uid_sha256,
            name="expected_source_ordered_trial_uid_sha256",
        )
    )
    expected_row_segments = (
        None
        if expected_source_row_segment_sha256 is None
        else _require_sha256(
            expected_source_row_segment_sha256,
            name="expected_source_row_segment_sha256",
        )
    )

    root = Path(directory)
    expected_members = {"metadata.json", *_AXIS_FILENAMES}
    if not root.is_dir():
        raise GBDTV4TrainingError(f"bundle checkpoint directory is missing: {root}")
    observed_members = {path.name for path in root.iterdir() if path.is_file()}
    if observed_members != expected_members or any(path.is_dir() for path in root.iterdir()):
        raise GBDTV4TrainingError("bundle checkpoint members differ from the contract")
    try:
        raw_metadata = (root / "metadata.json").read_bytes()
        metadata = json.loads(raw_metadata.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GBDTV4TrainingError("bundle metadata is unreadable") from exc
    if not isinstance(metadata, Mapping):
        raise GBDTV4TrainingError("bundle metadata must be a JSON object")
    try:
        canonical_metadata = canonical_json_bytes(metadata) + b"\n"
    except (TypeError, ValueError) as exc:
        raise GBDTV4TrainingError("bundle metadata contains non-canonical values") from exc
    if raw_metadata != canonical_metadata:
        raise GBDTV4TrainingError("bundle metadata bytes are not canonical JSON")
    contract = _strict_string(metadata.get("contract"), name="metadata contract")
    schema_version = _strict_int(
        metadata.get("schema_version"), name="metadata schema_version", minimum=1
    )
    if contract != GBDT_V4_BUNDLE_CONTRACT or schema_version != GBDT_V4_SCHEMA_VERSION:
        raise GBDTV4TrainingError("bundle metadata contract/schema changed")
    fixed_params = _metadata_mapping(
        metadata.get("fixed_xgb_params"), name="fixed_xgb_params"
    )
    if (
        _require_sha256(
            metadata.get("fixed_parameter_sha256"),
            name="fixed_parameter_sha256",
        )
        != fixed_xgb_parameter_sha256()
        or canonical_json_bytes(fixed_params)
        != canonical_json_bytes(dict(GBDT_V4_FIXED_XGB_PARAMS))
    ):
        raise GBDTV4TrainingError("bundle fixed XGBoost recipe changed")
    if _strict_string(
        metadata.get("axis_seed_recipe"), name="axis_seed_recipe"
    ) != GBDT_V4_AXIS_SEED_RECIPE:
        raise GBDTV4TrainingError("bundle axis seed recipe changed")

    scope = _metadata_mapping(metadata.get("scope"), name="scope")
    source = _metadata_mapping(metadata.get("source"), name="source")
    runtime = _metadata_mapping(metadata.get("runtime"), name="runtime")
    axes = _metadata_sequence(metadata.get("axis_models"), name="axis_models")
    if len(axes) != 3:
        raise GBDTV4TrainingError("bundle metadata scope/source/axis records are invalid")
    held_out = _strict_held_out_segment(
        scope.get("held_out_segment"), name="scope.held_out_segment"
    )
    spec = GBDTBundleSpec(
        held_out_segment=held_out,
        paired_seed=_strict_int(
            scope.get("paired_seed"),
            name="scope.paired_seed",
            allowed=GBDT_V4_PAIRED_SEEDS,
        ),
    )
    raw_models: list[bytes] = []
    base_scores: list[float] = []
    seeds: list[int] = []
    for axis, record in enumerate(axes):
        record = _metadata_mapping(record, name=f"axis_models[{axis}]")
        if (
            _strict_int(
                record.get("axis_index"), name="axis model index", minimum=0
            )
            != axis
            or _strict_string(record.get("axis_name"), name="axis model name")
            != GBDT_V4_AXIS_NAMES[axis]
            or _strict_string(record.get("filename"), name="axis model filename")
            != _AXIS_FILENAMES[axis]
        ):
            raise GBDTV4TrainingError("axis model identity/order changed")
        try:
            raw = (root / _AXIS_FILENAMES[axis]).read_bytes()
        except OSError as exc:
            raise GBDTV4TrainingError("axis model file is unreadable") from exc
        expected_bytes = _strict_int(
            record.get("bytes"), name="axis model bytes", minimum=1
        )
        if len(raw) != expected_bytes or sha256(raw).hexdigest() != _require_sha256(
            record.get("sha256"), name="axis model sha256"
        ):
            raise GBDTV4TrainingError("axis model size/hash changed")
        raw_models.append(raw)
        base_scores.append(
            _metadata_real(record.get("base_score"), name="axis base_score")
        )
        seeds.append(
            _strict_int(record.get("seed"), name="axis seed", minimum=0)
        )
    segment_rows = _metadata_sequence(
        source.get("segment_counts"), name="source.segment_counts"
    )
    counts: list[tuple[int, int]] = []
    for row in segment_rows:
        row = _metadata_mapping(row, name="source segment count row")
        counts.append(
            (
                _strict_int(
                    row.get("segment_id"),
                    name="source segment ID",
                    allowed=GBDT_V4_DEVELOPMENT_SEGMENTS,
                ),
                _strict_int(
                    row.get("rows"), name="source segment rows", minimum=1
                ),
            )
        )
    canonical_build_info = _metadata_mapping(
        runtime.get("canonical_build_info"), name="runtime.canonical_build_info"
    )
    normalized_build_info = _canonical_build_value(
        canonical_build_info, name="runtime.canonical_build_info"
    )
    if canonical_json_bytes(canonical_build_info) != canonical_json_bytes(
        normalized_build_info
    ):
        raise GBDTV4TrainingError("runtime canonical build_info value types changed")
    runtime_provenance = XGBoostRuntimeProvenance(
        version=_strict_string(
            runtime.get("xgboost_version"), name="runtime.xgboost_version"
        ),
        canonical_build_info_json=canonical_json_bytes(canonical_build_info).decode(
            "utf-8"
        ),
        canonical_build_info_sha256=_require_sha256(
            runtime.get("canonical_build_info_sha256"),
            name="runtime.canonical_build_info_sha256",
        ),
        libxgboost_resolved_path=_strict_string(
            runtime.get("libxgboost_resolved_path"),
            name="runtime.libxgboost_resolved_path",
        ),
        libxgboost_sha256=_require_sha256(
            runtime.get("libxgboost_sha256"),
            name="runtime.libxgboost_sha256",
        ),
    )
    feature_rows = _metadata_sequence(
        metadata.get("feature_names"), name="feature_names"
    )
    bundle = XGBBundle(
        spec=spec,
        axis_model_bytes=(raw_models[0], raw_models[1], raw_models[2]),
        axis_base_scores=(base_scores[0], base_scores[1], base_scores[2]),
        axis_seeds=(seeds[0], seeds[1], seeds[2]),
        source_row_count=_strict_int(
            source.get("row_count"), name="source.row_count", minimum=1
        ),
        source_segment_counts=tuple(counts),
        source_sorted_trial_uid_sha256=_require_sha256(
            source.get("sorted_trial_uid_sha256"),
            name="source.sorted_trial_uid_sha256",
        ),
        source_ordered_trial_uid_sha256=_require_sha256(
            source.get("ordered_trial_uid_sha256"),
            name="source.ordered_trial_uid_sha256",
        ),
        source_row_segment_sha256=_require_sha256(
            source.get("row_segment_int64_sha256"),
            name="source.row_segment_int64_sha256",
        ),
        training_data_sha256=_require_sha256(
            metadata.get("training_data_sha256"), name="training_data_sha256"
        ),
        training_prediction_sha256=_require_sha256(
            metadata.get("training_prediction_sha256"),
            name="training_prediction_sha256",
        ),
        training_device=_strict_string(
            metadata.get("training_device"), name="training_device"
        ),
        runtime_provenance=runtime_provenance,
        xgboost_version=_strict_string(
            metadata.get("xgboost_version"), name="xgboost_version"
        ),
        feature_names=tuple(
            _strict_string(value, name="feature_names item")
            for value in feature_rows
        ),
        num_boost_round=_strict_int(
            metadata.get("num_boost_round"), name="num_boost_round", minimum=1
        ),
    )
    if canonical_json_bytes(metadata) != canonical_json_bytes(bundle.metadata()):
        raise GBDTV4TrainingError("bundle metadata has missing, extra, or changed fields")
    if (
        resolved_expected_held_out is not _UNSET
        and bundle.spec.held_out_segment != resolved_expected_held_out
    ):
        raise GBDTV4TrainingError("bundle held-out segment differs from expectation")
    if resolved_expected_seed is not None and bundle.spec.paired_seed != resolved_expected_seed:
        raise GBDTV4TrainingError("bundle paired seed differs from expectation")
    if expected_sorted_uid is not None and (
        bundle.source_sorted_trial_uid_sha256 != expected_sorted_uid
    ):
        raise GBDTV4TrainingError("bundle source UID identity differs from expectation")
    if expected_ordered_uid is not None and (
        bundle.source_ordered_trial_uid_sha256 != expected_ordered_uid
    ):
        raise GBDTV4TrainingError(
            "bundle ordered source UID identity differs from expectation"
        )
    if expected_row_segments is not None and (
        bundle.source_row_segment_sha256 != expected_row_segments
    ):
        raise GBDTV4TrainingError(
            "bundle row-level segment identity differs from expectation"
        )
    validate_xgboost_runtime_provenance(bundle.runtime_provenance)
    if should_validate_models:
        validate_xgb_bundle_models(bundle)
    return bundle


def validate_development_bundle_coverage(
    bundles: Sequence[XGBBundle | GBDTBundleSpec], *, require_cuda: bool = True
) -> dict[str, Any]:
    """Require exactly 12 unique fold/seed bundles and no outage-fit dimension."""

    require_cuda_fits = _strict_bool(require_cuda, name="require_cuda")
    if not isinstance(bundles, Sequence) or isinstance(bundles, (str, bytes)):
        raise GBDTV4TrainingError("bundles must be a sequence")
    resolved = tuple(
        value.spec if isinstance(value, XGBBundle) else value for value in bundles
    )
    if any(not isinstance(value, GBDTBundleSpec) for value in resolved):
        raise GBDTV4TrainingError("bundle coverage accepts only specs or XGB bundles")
    expected = development_bundle_specs()
    if len(resolved) != 12 or len(set(resolved)) != 12 or set(resolved) != set(expected):
        raise GBDTV4TrainingError(
            "development coverage must be exactly 4 folds by 3 paired seeds"
        )
    concrete = tuple(value for value in bundles if isinstance(value, XGBBundle))
    if concrete and len(concrete) != len(bundles):
        raise GBDTV4TrainingError("coverage cannot mix specs and fitted bundles")
    if require_cuda_fits and any(
        bundle.training_device != "cuda:0" for bundle in concrete
    ):
        raise GBDTV4TrainingError(
            "development bundle coverage requires isolated cuda:0 fits"
        )
    return {
        "bundle_count": 12,
        "fold_count": 4,
        "forced_outage_fit_multiplier": 1,
        "paired_seed_count": 3,
        "scalar_booster_count": 36,
    }


__all__ = [
    "GBDTBundleSpec",
    "GBDTReplicatedTrainingData",
    "GBDTV4TrainingError",
    "GBDT_V4_AXIS_NAMES",
    "GBDT_V4_BUNDLE_CONTRACT",
    "GBDT_V4_DEVELOPMENT_SEGMENTS",
    "GBDT_V4_EXPECTED_LIBXGBOOST_SHA256",
    "GBDT_V4_FEATURE_NAMES",
    "GBDT_V4_FIXED_XGB_PARAMS",
    "GBDT_V4_MASK_PROBABILITIES",
    "GBDT_V4_MASK_STATES",
    "GBDT_V4_NUM_BOOST_ROUND",
    "GBDT_V4_PAIRED_SEEDS",
    "GBDT_V4_XGBOOST_VERSION",
    "XGBBundle",
    "XGBoostRuntimeProvenance",
    "build_mask_replicated_training_data",
    "derive_xgb_axis_seed",
    "development_bundle_specs",
    "development_evaluation_contexts",
    "fit_xgb_bundle",
    "fixed_xgb_parameter_sha256",
    "force_gbdt_outage_features",
    "load_xgb_bundle",
    "predict_xgb_bundle",
    "predict_xgb_bundle_forced_outage",
    "predict_xgb_bundle_matrix",
    "save_xgb_bundle",
    "transform_gbdt_outage_features",
    "validate_development_bundle_coverage",
    "validate_xgb_bundle_models",
    "validate_xgb_bundle_replay",
    "validate_xgb_training_replay",
    "validate_xgboost_runtime_provenance",
]
