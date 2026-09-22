"""Fail-closed data boundary for the prospective GPU-v3 outage study.

Development data comes only from authenticated ``alphaf.ml_preparation.v1``
rows in segments 36--39. Challenge identity/input preparation uses a fixed
38-column projection of the authenticated trial source which contains neither
Vive coordinate labels nor any IMU1 value. Challenge labels are decoded by a
separate, final-only API after model selection and prediction have been frozen
by the runner.

Vive remains a label only. Planned coordinates, target identity, segment
identity, timestamps, and quality fields never enter ``X``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import source as source_module
from .config import canonical_json_bytes
from .experiment_release import PINNED_RELEASE_SHA256, verify_and_load_release
from .features import FUSION_COLUMNS, LABEL_COLUMNS
from .gpu_outage_experiment_config import (
    PINNED_OUTAGE_GPU_CONFIG_SHA256,
    validate_outage_gpu_config,
)


class GPUOutageDataError(ValueError):
    """Raised before unverified or role-ineligible data reaches GPU-v3."""


OUTAGE_V3_CONTRACT = "alphaf.endpoint_regression.gpu_experiment.outage_v3"
PINNED_OUTAGE_V3_CONFIG_SHA256 = PINNED_OUTAGE_GPU_CONFIG_SHA256
PINNED_CHALLENGE_DATA_CONFIG_SHA256 = PINNED_OUTAGE_V3_CONFIG_SHA256

DEVELOPMENT_SEGMENTS = (36, 37, 38, 39)
DEVELOPMENT_SEGMENT_ROWS = {36: 1_672, 37: 1_597, 38: 1_707, 39: 1_659}
DEVELOPMENT_ROWS = 6_635
DEVELOPMENT_SORTED_TRIAL_UID_SHA256 = (
    "f8ef6f2ea62aa4e3591e8ea10fb195ade5fc0804342a5925a420039b2fd33ad0"
)
CHALLENGE_SEGMENTS = (40, 41)
CHALLENGE_SEGMENT_ROWS = {40: 1_672, 41: 1_623}
CHALLENGE_ROWS = 3_295
CHALLENGE_ORDERED_TRIAL_UID_SHA256 = (
    "fa24e77ba5ce5f333bc4bfc5d7b75dd567d098651493ec89cda8dfbe9f12c0a1"
)

AVAILABILITY_NAMES = (
    "imu0_base_available",
    "imu1_link_root_available",
    "imu2_link_mid_available",
    "imu3_tip_available",
)
AVAILABLE_CHALLENGE_IMU_INDICES = (0, 2, 3)
UNAVAILABLE_CHALLENGE_IMU_INDEX = 1

SOURCE_ARCHIVE_SHA256 = (
    "ceadf164a15a3ce302f8b2ffdde450df0f6d3a91c3b0c258a28acb5de9b92d64"
)
SOURCE_MEMBER = "derived/parquet/real_motion_demo/trials.parquet"
SOURCE_MEMBER_SHA256 = (
    "1b0a821a1822f237adadb1c978ea73da02b0e4bb7f099ce085a63931e9792b54"
)
SOURCE_ROWS = 10_055
SOURCE_SESSION_ID = "real_motion_demo"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMU_PREFIXES = {
    0: "imu0_base",
    1: "imu1_link_root",
    2: "imu2_link_mid",
    3: "imu3_tip",
}
_ENCODER_REFERENCE_TICKS = (2055.0, 2093.0, 1970.0)
_ENCODER_ACTIVE_RANGES = (
    (1160.0, 3083.0),
    (1202.0, 3186.0),
    (1937.0, 3968.0),
)
_FUSION_ALLOWLIST_SHA256 = (
    "1756b0aafb8ba4c9a740683d0a0d12236d8ee6e0065cfbe9fc6fb9d656345b6f"
)
_MODEL_INPUT_ALLOWLIST_SHA256 = (
    "ebd4c9b1087daeb3e6badcaad3ddfc1f3d0ef48951fa8d2c8599e16defa3b2a0"
)
_FORBIDDEN_FEATURE_TOKENS = (
    "vive",
    "tracker",
    "planned",
    "target_id",
    "segment",
    "timestamp",
    "qc",
    "provenance",
)


def _available_imu_columns(prefix: str) -> tuple[str, ...]:
    return (
        *(f"{prefix}_euler_{axis}_rad" for axis in ("x", "y", "z")),
        *(f"{prefix}_accel_{axis}_mps2" for axis in ("x", "y", "z")),
        f"{prefix}_gyro_rms_rad_s",
        f"{prefix}_n",
    )


# Exactly 38 columns. Labels and all IMU1 values are deliberately absent.
IDENTITY_PROJECTION_COLUMNS = (
    "segment_id",
    "block",
    "set_id",
    "target_id",
    "repeat_index",
    "t_start_ns",
    "duration_s",
    "vive_n",
    "vive_tracking_valid_ratio",
    "encoder_n",
    "keep",
    "encoder_tick_j0",
    "encoder_tick_j1",
    "encoder_tick_j2",
    *_available_imu_columns("imu0_base"),
    *_available_imu_columns("imu2_link_mid"),
    *_available_imu_columns("imu3_tip"),
)
assert len(IDENTITY_PROJECTION_COLUMNS) == 38

LABEL_PROJECTION_COLUMNS = (
    "segment_id",
    "t_start_ns",
    "vive_pos_x_m",
    "vive_pos_y_m",
    "vive_pos_z_m",
)

_LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"


def outage_config_sha256(config: Mapping[str, Any]) -> str:
    """Return canonical SHA-256 for the material resolved outage-v3 config."""

    public = {
        key: value for key, value in config.items() if not str(key).startswith("_")
    }
    return sha256(canonical_json_bytes(public)).hexdigest()


challenge_data_config_sha256 = outage_config_sha256


def ordered_trial_uid_sha256(trial_uids: Sequence[str]) -> str:
    """Hash the compact canonical JSON ordered UID string array."""

    return sha256(canonical_json_bytes([str(value) for value in trial_uids])).hexdigest()


def sorted_trial_uid_sha256(trial_uids: Sequence[str]) -> str:
    """Hash unique identifiers using the preprocessing audit recipe."""

    values = [str(value) for value in trial_uids]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise GPUOutageDataError("trial_uids must be unique non-empty strings")
    return sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    text = str(value)
    if not _SHA256_RE.fullmatch(text):
        raise GPUOutageDataError(f"{name} must be 64 lowercase SHA-256 hex digits")
    return text


def _mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise GPUOutageDataError(f"outage-v3 config is missing mapping: {name}")
    return value


def _numeric_pair(value: Any, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise GPUOutageDataError(f"{name} must contain exactly two numeric values")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result) or result[0] > result[1]:
        raise GPUOutageDataError(f"{name} must be a finite ordered interval")
    return result


def _count_mapping(value: Any, name: str) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise GPUOutageDataError(f"{name} must be a mapping")
    result = {int(key): int(count) for key, count in value.items()}
    if any(count < 0 for count in result.values()):
        raise GPUOutageDataError(f"{name} cannot contain negative counts")
    return result


def validate_outage_data_config(
    config: Mapping[str, Any],
    *,
    expected_config_sha256: str = PINNED_OUTAGE_V3_CONFIG_SHA256,
    enforce_pinned_counts: bool = True,
) -> None:
    """Validate config hash plus all data-role and label-seal semantics.

    ``enforce_pinned_counts=False`` exists only so tiny unit fixtures can
    exercise selection. Archive-loading public APIs never disable it.
    """

    expected_digest = _require_sha256(
        expected_config_sha256, "expected_config_sha256"
    )
    actual_digest = outage_config_sha256(config)
    if actual_digest != expected_digest:
        raise GPUOutageDataError(
            "outage-v3 config SHA-256 mismatch: "
            f"expected {expected_digest}, received {actual_digest}"
        )

    if enforce_pinned_counts:
        # Keep the data boundary subordinate to the complete experiment
        # validator rather than silently accepting a hash-valid but
        # semantically incomplete resolved config.
        validate_outage_gpu_config(dict(config))

    contract = _mapping(config, "contract")
    source = _mapping(config, "source")
    roles = _mapping(config, "row_roles")
    development = _mapping(roles, "development")
    challenge_role = _mapping(roles, "challenge")
    eligibility = _mapping(config, "label_blind_eligibility")
    task = _mapping(config, "task")
    masking = _mapping(config, "masking")
    challenge = _mapping(config, "challenge")

    if contract.get("name") != OUTAGE_V3_CONTRACT:
        raise GPUOutageDataError("outage-v3 contract name changed")
    if contract.get("frozen_before_challenge_label_access") is not True:
        raise GPUOutageDataError("challenge label freeze was weakened")
    if contract.get("confirmatory_claim_allowed") is not False:
        raise GPUOutageDataError("outage-v3 cannot make a confirmatory claim")

    expected_source = {
        "member": SOURCE_MEMBER,
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
        "member_sha256": SOURCE_MEMBER_SHA256,
        "session_id": SOURCE_SESSION_ID,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise GPUOutageDataError(f"outage-v3 source {key} changed")
    source_rows = int(source.get("expected_rows", -1))
    if source_rows <= 0 or (enforce_pinned_counts and source_rows != SOURCE_ROWS):
        raise GPUOutageDataError("outage-v3 source row count changed")
    if source.get("reject_git_lfs_pointer") is not True:
        raise GPUOutageDataError("outage-v3 source pointer rejection was weakened")
    if source.get("immutable_read_only") is not True:
        raise GPUOutageDataError("outage-v3 source read-only boundary was weakened")

    if tuple(int(value) for value in development.get("segments", ())) != DEVELOPMENT_SEGMENTS:
        raise GPUOutageDataError("development segments changed")
    development_rows = int(development.get("expected_rows", -1))
    if development_rows <= 0 or (
        enforce_pinned_counts and development_rows != DEVELOPMENT_ROWS
    ):
        raise GPUOutageDataError("development expected rows changed")
    if development.get("require_all_four_imu_summaries") is not True:
        raise GPUOutageDataError("development complete-IMU requirement changed")
    development_counts = _count_mapping(
        development.get("expected_segment_rows"),
        "row_roles.development.expected_segment_rows",
    )
    if sum(development_counts.values()) != development_rows:
        raise GPUOutageDataError("development total disagrees with segment counts")
    if enforce_pinned_counts and development_counts != DEVELOPMENT_SEGMENT_ROWS:
        raise GPUOutageDataError("development segment counts changed")
    development_uid_hash = _require_sha256(
        development.get("sorted_trial_uid_sha256"),
        "row_roles.development.sorted_trial_uid_sha256",
    )
    if development.get("sorted_trial_uid_hash_algorithm") != (
        "sha256_of_newline_joined_lexicographically_sorted_strings"
    ):
        raise GPUOutageDataError("development UID hash recipe changed")
    if (
        enforce_pinned_counts
        and development_uid_hash != DEVELOPMENT_SORTED_TRIAL_UID_SHA256
    ):
        raise GPUOutageDataError("development UID identity changed")

    if tuple(int(value) for value in challenge_role.get("segments", ())) != CHALLENGE_SEGMENTS:
        raise GPUOutageDataError("challenge segments changed")
    challenge_rows = int(challenge_role.get("expected_rows", -1))
    challenge_counts = _count_mapping(
        challenge_role.get("expected_segment_rows"),
        "row_roles.challenge.expected_segment_rows",
    )
    if set(challenge_counts) != set(CHALLENGE_SEGMENTS):
        raise GPUOutageDataError("challenge count mapping must cover segments 40 and 41")
    if sum(challenge_counts.values()) != challenge_rows:
        raise GPUOutageDataError("challenge total disagrees with expected segment counts")
    if enforce_pinned_counts and (
        challenge_rows != CHALLENGE_ROWS or challenge_counts != CHALLENGE_SEGMENT_ROWS
    ):
        raise GPUOutageDataError("challenge expected rows changed")
    challenge_uid_hash = _require_sha256(
        challenge_role.get("ordered_trial_uid_sha256"),
        "row_roles.challenge.ordered_trial_uid_sha256",
    )
    if enforce_pinned_counts and challenge_uid_hash != CHALLENGE_ORDERED_TRIAL_UID_SHA256:
        raise GPUOutageDataError("challenge ordered trial UID hash changed")
    if challenge_role.get("identity_projection_excludes_vive_coordinate_labels") is not True:
        raise GPUOutageDataError("challenge identity projection may not include labels")
    if challenge_role.get("identity_projection_excludes_all_imu1_values") is not True:
        raise GPUOutageDataError("challenge identity projection may not include IMU1 values")
    if challenge_role.get("exempt_sensor_from_completeness_and_sample_count") != "imu1_link_root":
        raise GPUOutageDataError("challenge exempt sensor changed")
    if challenge_role.get("recorded_imu1_values_must_not_be_projected_or_used") is not True:
        raise GPUOutageDataError("recorded challenge IMU1 values may not be used")
    if challenge_role.get("labels_sealed_until_final_model_and_prediction_freeze") is not True:
        raise GPUOutageDataError("challenge label seal was weakened")

    if eligibility.get("require_keep_true") is not True:
        raise GPUOutageDataError("challenge keep=true requirement changed")
    if _numeric_pair(eligibility.get("duration_s"), "duration_s") != (2.8, 3.2):
        raise GPUOutageDataError("challenge duration interval changed")
    if float(eligibility.get("require_vive_tracking_valid_ratio", -1.0)) != 1.0:
        raise GPUOutageDataError("challenge Vive tracking-ratio rule changed")
    if int(eligibility.get("min_vive_samples", -1)) != 170:
        raise GPUOutageDataError("challenge Vive sample threshold changed")
    if int(eligibility.get("required_encoder_samples", -1)) != 5:
        raise GPUOutageDataError("challenge encoder sample count changed")
    if int(eligibility.get("min_available_imu_samples", -1)) != 28:
        raise GPUOutageDataError("challenge available-IMU sample threshold changed")
    raw_ranges = eligibility.get("encoder_active_ranges")
    if not isinstance(raw_ranges, Sequence) or len(raw_ranges) != 3:
        raise GPUOutageDataError("encoder_active_ranges must define three joints")
    ranges = tuple(
        _numeric_pair(bounds, f"encoder_active_ranges[{index}]")
        for index, bounds in enumerate(raw_ranges)
    )
    if ranges != _ENCODER_ACTIVE_RANGES:
        raise GPUOutageDataError("challenge encoder active ranges changed")
    if set(eligibility.get("challenge_exempts_only", ())) != {
        "imu1_link_root_feature_finiteness",
        "imu1_link_root_sample_count",
    }:
        raise GPUOutageDataError("challenge exemption set changed")
    if eligibility.get("label_coordinate_values_must_not_affect_selection") is not True:
        raise GPUOutageDataError("challenge selection must remain label blind")
    if eligibility.get("nonfinite_label_after_unseal_action") != (
        "fail_entire_challenge_without_row_filtering"
    ):
        raise GPUOutageDataError("challenge non-finite label action changed")

    if task.get("sensor_feature_view") != "fusion_v1":
        raise GPUOutageDataError("outage-v3 sensor feature view changed")
    if int(task.get("sensor_feature_count", -1)) != len(FUSION_COLUMNS):
        raise GPUOutageDataError("outage-v3 sensor feature count changed")
    if task.get("sensor_feature_allowlist_sha256") != _FUSION_ALLOWLIST_SHA256:
        raise GPUOutageDataError("outage-v3 fusion_v1 allow-list hash changed")
    if tuple(task.get("availability_features", ())) != AVAILABILITY_NAMES:
        raise GPUOutageDataError("outage-v3 availability order changed")
    if int(task.get("model_input_feature_count", -1)) != 47:
        raise GPUOutageDataError("outage-v3 model input feature count changed")
    if task.get("model_input_allowlist_sha256") != _MODEL_INPUT_ALLOWLIST_SHA256:
        raise GPUOutageDataError("outage-v3 model input allow-list hash changed")
    if tuple(task.get("source_label_columns", ())) != (
        "vive_pos_x_m",
        "vive_pos_y_m",
        "vive_pos_z_m",
    ):
        raise GPUOutageDataError("challenge source label order changed")
    if tuple(task.get("labels", ())) != LABEL_COLUMNS:
        raise GPUOutageDataError("challenge output label order changed")
    if tuple(float(value) for value in masking.get("challenge_availability", ())) != (
        1.0,
        0.0,
        1.0,
        1.0,
    ):
        raise GPUOutageDataError("challenge availability mask changed")

    forbidden = {str(value) for value in task.get("inference_forbidden_tokens", ())}
    if forbidden != set(_FORBIDDEN_FEATURE_TOKENS):
        raise GPUOutageDataError("outage-v3 inference-forbidden token set changed")
    leaked = [
        name
        for name in FUSION_COLUMNS
        if any(token in name.lower() for token in _FORBIDDEN_FEATURE_TOKENS)
    ]
    if leaked:
        raise GPUOutageDataError(f"fusion_v1 feature allow-list leaks context: {leaked}")

    if challenge.get("selected_model_and_preprocessing_frozen_before_label_access") is not True:
        raise GPUOutageDataError("challenge final-selection freeze was weakened")
    if challenge.get("label_blind_predictions_hashed_before_label_access") is not True:
        raise GPUOutageDataError("challenge prediction hash boundary was weakened")
    if int(challenge.get("maximum_source_label_access_passes", -1)) != 1:
        raise GPUOutageDataError("challenge must allow exactly one label access pass")
    if challenge.get("evaluate_selected_frozen_model_only") is not True:
        raise GPUOutageDataError("challenge may evaluate only the frozen selected model")
    if challenge.get("tuning_reselection_or_threshold_changes_after_access") != "forbidden":
        raise GPUOutageDataError("challenge post-access reselection must remain forbidden")


validate_challenge_data_config = validate_outage_data_config


def _readonly_array(value: Any, *, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _imu_feature_indices(sensor_index: int) -> np.ndarray:
    prefix = _IMU_PREFIXES[int(sensor_index)]
    return np.asarray(
        [
            index
            for index, name in enumerate(FUSION_COLUMNS)
            if name.startswith(f"{prefix}_")
        ],
        dtype=np.int64,
    )


def _validate_input_arrays(
    *,
    feature_names: Sequence[str],
    X: Any,
    availability: Any,
    trial_uids: Any,
    segment_ids: Any,
    target_ids: Any,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = tuple(str(value) for value in feature_names)
    if names != FUSION_COLUMNS or len(names) != 43:
        raise GPUOutageDataError("feature_names must be ordered 43-column fusion_v1")
    features = _readonly_array(X, dtype=np.float64)
    masks = _readonly_array(availability, dtype=np.float64)
    uids = _readonly_array(trial_uids, dtype=str)
    segments = _readonly_array(segment_ids, dtype=np.int64)
    targets = _readonly_array(target_ids, dtype=str)
    rows = int(features.shape[0]) if features.ndim == 2 else -1
    if features.ndim != 2 or features.shape[1] != 43:
        raise GPUOutageDataError("X must have shape [rows, 43]")
    if masks.shape != (rows, 4):
        raise GPUOutageDataError("availability must have shape [rows, 4]")
    if uids.shape != (rows,) or segments.shape != (rows,) or targets.shape != (rows,):
        raise GPUOutageDataError("row metadata arrays must each have shape [rows]")
    if not np.isfinite(masks).all() or not np.isin(masks, (0.0, 1.0)).all():
        raise GPUOutageDataError("availability must contain only finite zero/one values")
    if not np.isfinite(features[:, :3]).all():
        raise GPUOutageDataError("encoder inputs contain non-finite values")
    for sensor_index in range(4):
        indices = _imu_feature_indices(sensor_index)
        values = features[:, indices]
        available_rows = masks[:, sensor_index] == 1.0
        unavailable_rows = ~available_rows
        if available_rows.any() and not np.isfinite(values[available_rows]).all():
            raise GPUOutageDataError(
                f"available IMU{sensor_index} inputs contain non-finite values"
            )
        if unavailable_rows.any() and not np.isnan(values[unavailable_rows]).all():
            raise GPUOutageDataError(
                f"unavailable IMU{sensor_index} inputs must be NaN sentinels before scaling"
            )
    if rows != len(set(uids.tolist())):
        raise GPUOutageDataError("trial_uids must be unique")
    if any(not re.fullmatch(r"trial_[0-9a-f]{64}", uid) for uid in uids):
        raise GPUOutageDataError("trial_uids do not follow the frozen identity recipe")
    if any(not value for value in targets.tolist()):
        raise GPUOutageDataError("target_ids metadata cannot be empty")
    return names, features, masks, uids, segments, targets


@dataclass(frozen=True, slots=True, repr=False)
class ChallengeIdentityData:
    """Label-free challenge identities and raw sensor inputs."""

    feature_names: tuple[str, ...]
    X: np.ndarray = field(repr=False)
    availability: np.ndarray = field(repr=False)
    trial_uids: np.ndarray = field(repr=False)
    segment_ids: np.ndarray = field(repr=False)
    target_ids: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        names, X, availability, uids, segments, targets = _validate_input_arrays(
            feature_names=self.feature_names,
            X=self.X,
            availability=self.availability,
            trial_uids=self.trial_uids,
            segment_ids=self.segment_ids,
            target_ids=self.target_ids,
        )
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "trial_uids", uids)
        object.__setattr__(self, "segment_ids", segments)
        object.__setattr__(self, "target_ids", targets)

    @property
    def rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def ordered_trial_uid_sha256(self) -> str:
        return ordered_trial_uid_sha256(self.trial_uids.tolist())

    def __repr__(self) -> str:
        segments = tuple(int(value) for value in np.unique(self.segment_ids))
        return (
            f"ChallengeIdentityData(rows={self.rows}, features=43, availability=4, "
            f"segments={segments})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class DevelopmentData:
    """Authenticated development arrays from train-eligible segments 36--39."""

    feature_names: tuple[str, ...]
    X: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    availability: np.ndarray = field(repr=False)
    trial_uids: np.ndarray = field(repr=False)
    segment_ids: np.ndarray = field(repr=False)
    target_ids: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        names, X, availability, uids, segments, targets = _validate_input_arrays(
            feature_names=self.feature_names,
            X=self.X,
            availability=self.availability,
            trial_uids=self.trial_uids,
            segment_ids=self.segment_ids,
            target_ids=self.target_ids,
        )
        labels = _readonly_array(self.y, dtype=np.float64)
        if labels.shape != (len(X), 3) or not np.isfinite(labels).all():
            raise GPUOutageDataError("development y must be finite with shape [rows, 3]")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", labels)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "trial_uids", uids)
        object.__setattr__(self, "segment_ids", segments)
        object.__setattr__(self, "target_ids", targets)

    @property
    def rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def sorted_trial_uid_sha256(self) -> str:
        return sorted_trial_uid_sha256(self.trial_uids.tolist())

    def __repr__(self) -> str:
        segments = tuple(int(value) for value in np.unique(self.segment_ids))
        return (
            f"DevelopmentData(rows={self.rows}, features=43, labels=3, "
            f"availability=4, segments={segments})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChallengeData:
    """Final-only challenge inputs plus exactly aligned finite XYZ labels."""

    feature_names: tuple[str, ...]
    X: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    availability: np.ndarray = field(repr=False)
    trial_uids: np.ndarray = field(repr=False)
    segment_ids: np.ndarray = field(repr=False)
    target_ids: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        names, X, availability, uids, segments, targets = _validate_input_arrays(
            feature_names=self.feature_names,
            X=self.X,
            availability=self.availability,
            trial_uids=self.trial_uids,
            segment_ids=self.segment_ids,
            target_ids=self.target_ids,
        )
        labels = _readonly_array(self.y, dtype=np.float64)
        # No challenge label-value logic or metric is permitted here: only the
        # predeclared all-or-nothing finite XYZ shape gate.
        if labels.shape != (len(X), 3) or not np.isfinite(labels).all():
            raise GPUOutageDataError(
                "challenge y must be wholly finite with shape [rows, 3]"
            )
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", labels)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "trial_uids", uids)
        object.__setattr__(self, "segment_ids", segments)
        object.__setattr__(self, "target_ids", targets)

    @property
    def rows(self) -> int:
        return int(self.X.shape[0])

    def __repr__(self) -> str:
        segments = tuple(int(value) for value in np.unique(self.segment_ids))
        return (
            f"ChallengeData(rows={self.rows}, features=43, labels=3, "
            f"availability=4, segments={segments})"
        )


def _segment_counts(frame: pd.DataFrame) -> dict[int, int]:
    return {
        int(key): int(value)
        for key, value in frame["segment_id"].value_counts().sort_index().items()
    }


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, role: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise GPUOutageDataError(f"{role} frame is missing required columns: {missing}")


def prepare_development_data(
    eligible_real: pd.DataFrame,
    *,
    expected_segment_rows: Mapping[int, int] = DEVELOPMENT_SEGMENT_ROWS,
    expected_trial_uid_sha256: str | None = DEVELOPMENT_SORTED_TRIAL_UID_SHA256,
) -> DevelopmentData:
    """Project verified train-eligible real rows into immutable dev arrays."""

    required = [
        "train_eligible",
        "trial_uid",
        "segment_id",
        "target_id",
        *FUSION_COLUMNS,
        *LABEL_COLUMNS,
    ]
    _require_columns(eligible_real, required, role="development")
    if not eligible_real["train_eligible"].eq(True).all():
        raise GPUOutageDataError("development input contains a non-train-eligible row")
    selected = eligible_real.loc[
        eligible_real["segment_id"].isin(DEVELOPMENT_SEGMENTS)
    ].copy()
    selected.sort_values("trial_uid", kind="mergesort", inplace=True)
    selected.reset_index(drop=True, inplace=True)
    expected = {int(key): int(value) for key, value in expected_segment_rows.items()}
    actual = _segment_counts(selected)
    if actual != expected:
        raise GPUOutageDataError(
            f"development segment counts changed: expected {expected}, received {actual}"
        )
    rows = len(selected)
    development = DevelopmentData(
        feature_names=FUSION_COLUMNS,
        X=selected.loc[:, list(FUSION_COLUMNS)].to_numpy(dtype=np.float64),
        y=selected.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64),
        availability=np.ones((rows, 4), dtype=np.float64),
        trial_uids=selected["trial_uid"].astype(str).to_numpy(),
        segment_ids=selected["segment_id"].to_numpy(dtype=np.int64),
        target_ids=selected["target_id"].astype(str).to_numpy(),
    )
    if (
        expected_trial_uid_sha256 is not None
        and development.sorted_trial_uid_sha256
        != _require_sha256(
            expected_trial_uid_sha256, "expected_trial_uid_sha256"
        )
    ):
        raise GPUOutageDataError("development sorted trial UID identity changed")
    return development


def load_development_data(
    prep_path: str | Path,
    expected_hash: str = PINNED_RELEASE_SHA256,
) -> DevelopmentData:
    """Authenticate the preparation ZIP and load development segments only."""

    release = verify_and_load_release(prep_path, expected_sha256=expected_hash)
    if tuple(release.feature_sets.get("fusion_v1", ())) != FUSION_COLUMNS:
        raise GPUOutageDataError("authenticated preparation fusion_v1 order changed")
    return prepare_development_data(release.eligible_real)


def _trial_uid(
    *, source_member_sha256: str, session_id: str, segment_id: int, t_start_ns: int
) -> str:
    identity = f"{source_member_sha256}|{session_id}|{segment_id}|{t_start_ns}"
    return f"trial_{sha256(identity.encode('utf-8')).hexdigest()}"


def _challenge_feature_matrix(selected: pd.DataFrame) -> np.ndarray:
    matrix = np.full((len(selected), len(FUSION_COLUMNS)), np.nan, dtype=np.float64)
    column_index = {name: index for index, name in enumerate(FUSION_COLUMNS)}
    for joint, reference in enumerate(_ENCODER_REFERENCE_TICKS):
        name = f"encoder_delta_rad_j{joint}"
        matrix[:, column_index[name]] = (
            selected[f"encoder_tick_j{joint}"].to_numpy(dtype=np.float64) - reference
        ) * (2.0 * math.pi / 4096.0)
    for sensor_index in AVAILABLE_CHALLENGE_IMU_INDICES:
        prefix = _IMU_PREFIXES[sensor_index]
        for axis in ("x", "y", "z"):
            source_name = f"{prefix}_euler_{axis}_rad"
            values = selected[source_name].to_numpy(dtype=np.float64)
            matrix[:, column_index[f"{source_name}_sin"]] = np.sin(values)
            matrix[:, column_index[f"{source_name}_cos"]] = np.cos(values)
        for axis in ("x", "y", "z"):
            name = f"{prefix}_accel_{axis}_mps2"
            matrix[:, column_index[name]] = selected[name].to_numpy(dtype=np.float64)
        gyro_name = f"{prefix}_gyro_rms_rad_s"
        matrix[:, column_index[gyro_name]] = selected[gyro_name].to_numpy(
            dtype=np.float64
        )
    return matrix


def prepare_challenge_identity(
    source: pd.DataFrame,
    *,
    source_member_sha256: str,
    challenge_config: Mapping[str, Any],
    expected_config_sha256: str,
    enforce_pinned_counts: bool = True,
) -> ChallengeIdentityData:
    """Build challenge IDs/inputs without projecting labels or IMU1 values."""

    validate_outage_data_config(
        challenge_config,
        expected_config_sha256=expected_config_sha256,
        enforce_pinned_counts=enforce_pinned_counts,
    )
    source_config = _mapping(challenge_config, "source")
    declared_member_hash = _require_sha256(
        source_config.get("member_sha256"), "source.member_sha256"
    )
    if _require_sha256(source_member_sha256, "source_member_sha256") != declared_member_hash:
        raise GPUOutageDataError("challenge source member identity changed")
    expected_source_rows = int(source_config["expected_rows"])
    if len(source) != expected_source_rows:
        raise GPUOutageDataError(
            "challenge source row count changed: "
            f"expected {expected_source_rows}, received {len(source)}"
        )
    _require_columns(
        source, IDENTITY_PROJECTION_COLUMNS, role="challenge identity source"
    )
    projection = source.loc[:, list(IDENTITY_PROJECTION_COLUMNS)].copy()
    if any(name.startswith("vive_pos_") for name in projection.columns):
        raise GPUOutageDataError("challenge identity projection unexpectedly includes labels")
    if any(name.startswith("imu1_link_root_") for name in projection.columns):
        raise GPUOutageDataError("challenge identity projection unexpectedly includes IMU1")

    eligibility = _mapping(challenge_config, "label_blind_eligibility")
    duration_min, duration_max = _numeric_pair(
        eligibility["duration_s"], "duration_s"
    )
    mask = projection["segment_id"].isin(CHALLENGE_SEGMENTS)
    mask &= projection["keep"].eq(True)
    mask &= projection["duration_s"].between(
        duration_min, duration_max, inclusive="both"
    )
    mask &= projection["vive_tracking_valid_ratio"].eq(
        float(eligibility["require_vive_tracking_valid_ratio"])
    )
    mask &= projection["vive_n"].ge(int(eligibility["min_vive_samples"]))
    mask &= projection["encoder_n"].eq(
        int(eligibility["required_encoder_samples"])
    )
    for joint, bounds in enumerate(eligibility["encoder_active_ranges"]):
        low, high = _numeric_pair(bounds, f"encoder_active_ranges[{joint}]")
        mask &= projection[f"encoder_tick_j{joint}"].between(
            low, high, inclusive="both"
        )
    for sensor_index in AVAILABLE_CHALLENGE_IMU_INDICES:
        prefix = _IMU_PREFIXES[sensor_index]
        mask &= projection[f"{prefix}_n"].ge(
            int(eligibility["min_available_imu_samples"])
        )

    finite_columns = [
        "t_start_ns",
        *[f"encoder_tick_j{joint}" for joint in range(3)],
    ]
    for sensor_index in AVAILABLE_CHALLENGE_IMU_INDICES:
        finite_columns.extend(_available_imu_columns(_IMU_PREFIXES[sensor_index])[:-1])
    finite_values = projection.loc[:, finite_columns].to_numpy(dtype=np.float64)
    mask &= pd.Series(np.isfinite(finite_values).all(axis=1), index=projection.index)

    selected = projection.loc[mask].copy()
    selected.sort_values(
        ["segment_id", "t_start_ns", "target_id"], kind="mergesort", inplace=True
    )
    selected.reset_index(drop=True, inplace=True)
    role = _mapping(_mapping(challenge_config, "row_roles"), "challenge")
    expected_counts = _count_mapping(
        role["expected_segment_rows"], "row_roles.challenge.expected_segment_rows"
    )
    actual_counts = _segment_counts(selected)
    if actual_counts != expected_counts:
        raise GPUOutageDataError(
            f"challenge segment counts changed: expected {expected_counts}, "
            f"received {actual_counts}"
        )
    if len(selected) != int(role["expected_rows"]):
        raise GPUOutageDataError("challenge total row count changed")
    if selected["target_id"].isna().any() or selected["target_id"].astype(str).eq("").any():
        raise GPUOutageDataError("selected challenge target metadata is missing")

    session_id = str(_mapping(challenge_config, "source")["session_id"])
    trial_uids = np.asarray(
        [
            _trial_uid(
                source_member_sha256=declared_member_hash,
                session_id=session_id,
                segment_id=int(segment),
                t_start_ns=int(start_ns),
            )
            for segment, start_ns in zip(
                selected["segment_id"], selected["t_start_ns"], strict=True
            )
        ],
        dtype=str,
    )
    actual_uid_hash = ordered_trial_uid_sha256(trial_uids.tolist())
    expected_uid_hash = str(role["ordered_trial_uid_sha256"])
    if actual_uid_hash != expected_uid_hash:
        raise GPUOutageDataError(
            "challenge ordered trial UID SHA-256 changed: "
            f"expected {expected_uid_hash}, received {actual_uid_hash}"
        )

    rows = len(selected)
    availability = np.tile(
        np.asarray((1.0, 0.0, 1.0, 1.0), dtype=np.float64), (rows, 1)
    )
    return ChallengeIdentityData(
        feature_names=FUSION_COLUMNS,
        X=_challenge_feature_matrix(selected),
        availability=availability,
        trial_uids=trial_uids,
        segment_ids=selected["segment_id"].to_numpy(dtype=np.int64),
        target_ids=selected["target_id"].astype(str).to_numpy(),
    )


def prepare_challenge_labels(
    source: pd.DataFrame,
    identity: ChallengeIdentityData,
    *,
    source_member_sha256: str,
    challenge_config: Mapping[str, Any],
    expected_config_sha256: str,
    enforce_pinned_counts: bool = True,
) -> ChallengeData:
    """Decode one aligned challenge XYZ matrix after the runner opens the seal."""

    validate_outage_data_config(
        challenge_config,
        expected_config_sha256=expected_config_sha256,
        enforce_pinned_counts=enforce_pinned_counts,
    )
    source_config = _mapping(challenge_config, "source")
    member_hash = _require_sha256(source_member_sha256, "source_member_sha256")
    if member_hash != source_config["member_sha256"]:
        raise GPUOutageDataError("challenge label source member identity changed")
    if len(source) != int(source_config["expected_rows"]):
        raise GPUOutageDataError("challenge label source row count changed")
    if identity.ordered_trial_uid_sha256 != _mapping(
        _mapping(challenge_config, "row_roles"), "challenge"
    )["ordered_trial_uid_sha256"]:
        raise GPUOutageDataError("challenge identity does not match the frozen UID list")

    label_source_columns = tuple(
        str(value) for value in _mapping(challenge_config, "task")["source_label_columns"]
    )
    required = ("segment_id", "t_start_ns", *label_source_columns)
    _require_columns(source, required, role="challenge label source")
    projection = source.loc[
        source["segment_id"].isin(CHALLENGE_SEGMENTS), list(required)
    ].copy()
    session_id = str(source_config["session_id"])
    projection["trial_uid"] = [
        _trial_uid(
            source_member_sha256=member_hash,
            session_id=session_id,
            segment_id=int(segment),
            t_start_ns=int(start_ns),
        )
        for segment, start_ns in zip(
            projection["segment_id"], projection["t_start_ns"], strict=True
        )
    ]
    if projection["trial_uid"].duplicated().any():
        raise GPUOutageDataError("challenge label source contains duplicate trial identities")
    indexed = projection.set_index("trial_uid", verify_integrity=True)
    missing = sorted(set(identity.trial_uids.tolist()).difference(indexed.index))
    if missing:
        raise GPUOutageDataError(
            f"challenge label source is missing {len(missing)} frozen trial identities"
        )
    labels_m = indexed.loc[
        identity.trial_uids.tolist(), list(label_source_columns)
    ].to_numpy(dtype=np.float64)
    if labels_m.shape != (identity.rows, 3) or not np.isfinite(labels_m).all():
        raise GPUOutageDataError(
            "entire challenge failed: unsealed labels are not wholly finite XYZ"
        )
    labels_mm = labels_m * 1000.0
    return ChallengeData(
        feature_names=identity.feature_names,
        X=identity.X,
        y=labels_mm,
        availability=identity.availability,
        trial_uids=identity.trial_uids,
        segment_ids=identity.segment_ids,
        target_ids=identity.target_ids,
    )


def _source_archive_path(
    source_config: Mapping[str, Any], challenge_config: Mapping[str, Any]
) -> Path:
    try:
        configured = Path(str(source_config["archive"]))
    except KeyError as exc:
        raise GPUOutageDataError("source config is missing archive") from exc
    if configured.is_absolute():
        return configured.resolve()
    base = Path(str(challenge_config.get("_workspace_root", ".")))
    return (base / configured).resolve()


def _validate_source_argument(
    source_config: Mapping[str, Any], challenge_config: Mapping[str, Any]
) -> None:
    declared = _mapping(challenge_config, "source")
    required = {
        "archive",
        "member",
        "archive_sha256",
        "member_sha256",
        "expected_rows",
    }
    missing = sorted(required.difference(source_config))
    if missing:
        raise GPUOutageDataError(f"source config is missing required fields: {missing}")
    for key in ("member", "archive_sha256", "member_sha256"):
        if str(source_config[key]) != str(declared[key]):
            raise GPUOutageDataError(f"source config {key} differs from outage-v3 config")
    if int(source_config["expected_rows"]) != int(declared["expected_rows"]):
        raise GPUOutageDataError("source config expected_rows differs from outage-v3 config")


def _load_authenticated_projection(
    archive_path: str | Path,
    *,
    member: str,
    expected_archive_sha256: str,
    expected_member_sha256: str,
    expected_rows: int,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Authenticate the full source but deserialize only an approved projection."""

    path = Path(archive_path).resolve()
    if not path.is_file():
        raise GPUOutageDataError(f"source archive does not exist: {path}")
    with path.open("rb") as stream:
        if stream.read(128).startswith(_LFS_POINTER_HEADER):
            raise GPUOutageDataError(
                f"source archive is an unmaterialized Git LFS pointer: {path}"
            )
    declared_archive_hash = _require_sha256(
        expected_archive_sha256, "expected_archive_sha256"
    )
    actual_archive_hash = source_module.sha256_file(path)
    if actual_archive_hash != declared_archive_hash:
        raise GPUOutageDataError(
            "source archive SHA-256 mismatch: "
            f"expected {declared_archive_hash}, received {actual_archive_hash}"
        )
    try:
        with ZipFile(path) as archive:
            matches = [entry for entry in archive.infolist() if entry.filename == member]
            if len(matches) != 1:
                raise GPUOutageDataError(
                    f"source archive must contain exactly one {member!r}; "
                    f"found {len(matches)}"
                )
            raw_member = archive.read(matches[0])
    except BadZipFile as exc:
        raise GPUOutageDataError(f"source archive is not a valid ZIP: {path}") from exc
    if raw_member.startswith(_LFS_POINTER_HEADER):
        raise GPUOutageDataError(f"source member {member!r} is a Git LFS pointer")
    declared_member_hash = _require_sha256(
        expected_member_sha256, "expected_member_sha256"
    )
    actual_member_hash = sha256(raw_member).hexdigest()
    if actual_member_hash != declared_member_hash:
        raise GPUOutageDataError(
            "source member SHA-256 mismatch: "
            f"expected {declared_member_hash}, received {actual_member_hash}"
        )

    requested = tuple(str(value) for value in columns)
    if not requested or len(requested) != len(set(requested)):
        raise GPUOutageDataError("source projection columns must be non-empty and unique")
    expected_names = set(source_module.EXPECTED_TRIALS_SCHEMA.names)
    unexpected = sorted(set(requested).difference(expected_names))
    if unexpected:
        raise GPUOutageDataError(f"source projection contains unknown columns: {unexpected}")
    try:
        parquet_file = pq.ParquetFile(pa.BufferReader(raw_member))
        actual_schema = parquet_file.schema_arrow.remove_metadata()
        expected_schema = source_module.EXPECTED_TRIALS_SCHEMA.remove_metadata()
        if not actual_schema.equals(expected_schema, check_metadata=False):
            raise GPUOutageDataError("trials.parquet full Arrow schema mismatch")
        actual_rows = int(parquet_file.metadata.num_rows)
        if actual_rows != int(expected_rows):
            raise GPUOutageDataError(
                "source row-count mismatch: "
                f"expected {int(expected_rows)}, received {actual_rows}"
            )
        # The second buffer is intentional: schema/row-count inspection above
        # never decodes arrays; this call materializes only approved columns.
        table = pq.read_table(pa.BufferReader(raw_member), columns=list(requested))
    except (pa.ArrowException, OSError) as exc:
        raise GPUOutageDataError(f"source member {member!r} is not readable Parquet") from exc
    if table.num_rows != int(expected_rows) or tuple(table.column_names) != requested:
        raise GPUOutageDataError("source projection shape/order changed")
    frame = table.to_pandas()
    frame.attrs.update(
        {
            "source_archive_path": str(path),
            "source_archive_sha256": actual_archive_hash,
            "source_member": member,
            "source_member_sha256": actual_member_hash,
            "source_projection_columns": requested,
        }
    )
    return frame


def _load_configured_projection(
    source_config: Mapping[str, Any],
    challenge_config: Mapping[str, Any],
    *,
    columns: Sequence[str],
) -> pd.DataFrame:
    _validate_source_argument(source_config, challenge_config)
    declared = _mapping(challenge_config, "source")
    return _load_authenticated_projection(
        _source_archive_path(source_config, challenge_config),
        member=str(declared["member"]),
        expected_archive_sha256=str(declared["archive_sha256"]),
        expected_member_sha256=str(declared["member_sha256"]),
        expected_rows=int(declared["expected_rows"]),
        columns=columns,
    )


def load_challenge_identity(
    source_config: Mapping[str, Any],
    challenge_config: Mapping[str, Any],
    *,
    expected_config_sha256: str = PINNED_OUTAGE_V3_CONFIG_SHA256,
) -> ChallengeIdentityData:
    """Authenticate source and freeze challenge inputs without label projection."""

    validate_outage_data_config(
        challenge_config, expected_config_sha256=expected_config_sha256
    )
    frame = _load_configured_projection(
        source_config,
        challenge_config,
        columns=IDENTITY_PROJECTION_COLUMNS,
    )
    return prepare_challenge_identity(
        frame,
        source_member_sha256=str(source_config["member_sha256"]),
        challenge_config=challenge_config,
        expected_config_sha256=expected_config_sha256,
    )


def load_challenge_labels_once(
    source_config: Mapping[str, Any],
    challenge_config: Mapping[str, Any],
    identity: ChallengeIdentityData,
    *,
    expected_config_sha256: str = PINNED_OUTAGE_V3_CONFIG_SHA256,
) -> ChallengeData:
    """Final-only label access; the runner must durably enforce the one-pass gate."""

    validate_outage_data_config(
        challenge_config, expected_config_sha256=expected_config_sha256
    )
    frame = _load_configured_projection(
        source_config,
        challenge_config,
        columns=LABEL_PROJECTION_COLUMNS,
    )
    return prepare_challenge_labels(
        frame,
        identity,
        source_member_sha256=str(source_config["member_sha256"]),
        challenge_config=challenge_config,
        expected_config_sha256=expected_config_sha256,
    )


__all__ = [
    "AVAILABILITY_NAMES",
    "AVAILABLE_CHALLENGE_IMU_INDICES",
    "CHALLENGE_ORDERED_TRIAL_UID_SHA256",
    "CHALLENGE_ROWS",
    "CHALLENGE_SEGMENTS",
    "CHALLENGE_SEGMENT_ROWS",
    "ChallengeData",
    "ChallengeIdentityData",
    "DEVELOPMENT_ROWS",
    "DEVELOPMENT_SEGMENTS",
    "DEVELOPMENT_SEGMENT_ROWS",
    "DEVELOPMENT_SORTED_TRIAL_UID_SHA256",
    "DevelopmentData",
    "GPUOutageDataError",
    "IDENTITY_PROJECTION_COLUMNS",
    "LABEL_PROJECTION_COLUMNS",
    "OUTAGE_V3_CONTRACT",
    "PINNED_CHALLENGE_DATA_CONFIG_SHA256",
    "PINNED_OUTAGE_V3_CONFIG_SHA256",
    "SOURCE_ARCHIVE_SHA256",
    "SOURCE_MEMBER",
    "SOURCE_MEMBER_SHA256",
    "SOURCE_ROWS",
    "SOURCE_SESSION_ID",
    "UNAVAILABLE_CHALLENGE_IMU_INDEX",
    "challenge_data_config_sha256",
    "load_challenge_identity",
    "load_challenge_labels_once",
    "load_development_data",
    "ordered_trial_uid_sha256",
    "outage_config_sha256",
    "prepare_challenge_identity",
    "prepare_challenge_labels",
    "prepare_development_data",
    "sorted_trial_uid_sha256",
    "validate_challenge_data_config",
    "validate_outage_data_config",
]
