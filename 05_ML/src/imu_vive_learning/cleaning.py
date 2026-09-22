"""Label-blind cleaning for the static four-IMU endpoint dataset."""

from __future__ import annotations

from hashlib import sha256
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class CleaningContractError(ValueError):
    """Raised when cleaning no longer reproduces the versioned data contract."""


IMU_PREFIXES = (
    "imu0_base",
    "imu1_link_root",
    "imu2_link_mid",
    "imu3_tip",
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise CleaningContractError(f"source frame is missing required columns: {missing}")


def _complete_finite_mask(frame: pd.DataFrame) -> pd.Series:
    complete = frame.notna().all(axis=1)
    numeric_columns = list(frame.select_dtypes(include=[np.number]).columns)
    if numeric_columns:
        finite = np.isfinite(frame[numeric_columns].to_numpy(dtype=np.float64)).all(axis=1)
        complete &= pd.Series(finite, index=frame.index)
    return complete


def _trial_uid(
    *, source_member_sha256: str, session_id: str, segment_id: int, t_start_ns: int
) -> str:
    identity = f"{source_member_sha256}|{session_id}|{segment_id}|{t_start_ns}"
    return f"trial_{sha256(identity.encode('utf-8')).hexdigest()}"


def _joined_reasons(reason_masks: Sequence[tuple[str, pd.Series]]) -> pd.Series:
    if not reason_masks:
        raise CleaningContractError("at least one reason mask is required")
    index = reason_masks[0][1].index
    reasons = np.full(len(index), "", dtype=object)
    for name, mask in reason_masks:
        values = mask.to_numpy(dtype=bool)
        reasons[values] = [
            name if not existing else f"{existing};{name}" for existing in reasons[values]
        ]
    return pd.Series(reasons, index=index, dtype="string")


def _expected_counts(
    frame: pd.DataFrame, expected: Mapping[Any, Any], *, label: str
) -> None:
    actual = frame["segment_id"].value_counts().sort_index().to_dict()
    normalized_expected = {int(key): int(value) for key, value in expected.items()}
    if actual != normalized_expected:
        raise CleaningContractError(
            f"{label} segment counts changed: expected {normalized_expected}, received {actual}"
        )


def clean_trials(
    source: pd.DataFrame,
    cleaning_config: Mapping[str, Any],
    *,
    session_id: str,
    source_member_sha256: str,
) -> pd.DataFrame:
    """Build canonical real rows and label eligible rows without label filtering."""

    required_config = {
        "canonical_segments",
        "expected_canonical_rows",
        "expected_train_eligible_rows",
        "expected_segment_rows",
        "expected_train_segment_rows",
        "duration_s",
        "min_vive_samples",
        "required_encoder_samples",
        "min_imu_samples",
        "encoder_reference_ticks",
        "encoder_active_ranges",
        "warning_thresholds",
    }
    missing_config = sorted(required_config.difference(cleaning_config))
    if missing_config:
        raise CleaningContractError(f"cleaning config is missing required fields: {missing_config}")

    required_columns = {
        "segment_id",
        "set_id",
        "target_id",
        "repeat_index",
        "t_start_ns",
        "duration_s",
        "vive_n",
        "vive_tracking_valid_ratio",
        "vive_pos_x_m",
        "vive_pos_y_m",
        "vive_pos_z_m",
        "vive_pos_x_std_m",
        "vive_pos_y_std_m",
        "vive_pos_z_std_m",
        "encoder_n",
        "keep",
        "planned_p_x_mm",
        "planned_p_y_mm",
        "planned_p_z_mm",
    }
    for joint in range(3):
        required_columns.update(
            {f"encoder_tick_j{joint}", f"encoder_tick_j{joint}_std"}
        )
    for prefix in IMU_PREFIXES:
        required_columns.update(
            {
                f"{prefix}_euler_{axis}_rad" for axis in ("x", "y", "z")
            }
        )
        required_columns.update(
            {
                f"{prefix}_accel_{axis}_mps2" for axis in ("x", "y", "z")
            }
        )
        required_columns.update(
            {
                f"{prefix}_mag_{axis}_raw" for axis in ("x", "y", "z")
            }
        )
        required_columns.update({f"{prefix}_gyro_rms_rad_s", f"{prefix}_n"})
    _require_columns(source, required_columns)

    canonical_segments = [int(value) for value in cleaning_config["canonical_segments"]]
    structural_mask = (
        source["segment_id"].isin(canonical_segments)
        & source["keep"].eq(True)
        & _complete_finite_mask(source.loc[:, sorted(required_columns)])
    )
    canonical = source.loc[structural_mask].copy()
    canonical.sort_values(
        ["segment_id", "t_start_ns", "target_id"], kind="mergesort", inplace=True
    )
    canonical.reset_index(drop=True, inplace=True)

    expected_canonical = int(cleaning_config["expected_canonical_rows"])
    if len(canonical) != expected_canonical:
        raise CleaningContractError(
            f"canonical row count changed: expected {expected_canonical}, received {len(canonical)}"
        )
    _expected_counts(
        canonical, cleaning_config["expected_segment_rows"], label="canonical"
    )

    canonical["session_id"] = str(session_id)
    canonical["source_member_sha256"] = str(source_member_sha256).lower()
    canonical["trial_uid"] = [
        _trial_uid(
            source_member_sha256=str(source_member_sha256).lower(),
            session_id=str(session_id),
            segment_id=int(segment),
            t_start_ns=int(start_ns),
        )
        for segment, start_ns in zip(
            canonical["segment_id"], canonical["t_start_ns"], strict=True
        )
    ]
    if canonical["trial_uid"].duplicated().any():
        raise CleaningContractError("trial_uid collision detected")

    for axis in ("x", "y", "z"):
        canonical[f"vive_pos_raw_{axis}_mm"] = canonical[f"vive_pos_{axis}_m"] * 1000.0

    references = [float(value) for value in cleaning_config["encoder_reference_ticks"]]
    active_ranges = cleaning_config["encoder_active_ranges"]
    if len(references) != 3 or len(active_ranges) != 3:
        raise CleaningContractError(
            "encoder_reference_ticks and encoder_active_ranges must each define three joints"
        )
    for joint, reference in enumerate(references):
        canonical[f"encoder_delta_rad_j{joint}"] = (
            canonical[f"encoder_tick_j{joint}"] - reference
        ) * (2.0 * math.pi / 4096.0)

    duration_min, duration_max = [float(value) for value in cleaning_config["duration_s"]]
    exclusion_masks: list[tuple[str, pd.Series]] = [
        (
            "vive_tracking_invalid",
            ~canonical["vive_tracking_valid_ratio"].eq(1.0),
        ),
        (
            "duration_out_of_range",
            ~canonical["duration_s"].between(duration_min, duration_max, inclusive="both"),
        ),
        (
            "vive_samples_below_min",
            canonical["vive_n"].lt(int(cleaning_config["min_vive_samples"])),
        ),
        (
            "encoder_sample_count_mismatch",
            ~canonical["encoder_n"].eq(int(cleaning_config["required_encoder_samples"])),
        ),
    ]
    min_imu_samples = int(cleaning_config["min_imu_samples"])
    for prefix in IMU_PREFIXES:
        exclusion_masks.append(
            (f"{prefix}_samples_below_min", canonical[f"{prefix}_n"].lt(min_imu_samples))
        )
    for joint, bounds in enumerate(active_ranges):
        low, high = [float(value) for value in bounds]
        exclusion_masks.append(
            (
                f"encoder_tick_j{joint}_outside_active_range",
                ~canonical[f"encoder_tick_j{joint}"].between(low, high, inclusive="both"),
            )
        )

    canonical["train_exclusion_reasons"] = _joined_reasons(exclusion_masks)
    canonical["train_eligible"] = canonical["train_exclusion_reasons"].eq("")

    thresholds = cleaning_config["warning_thresholds"]
    gyro_threshold = float(thresholds["gyro_rms_rad_s"])
    vive_std_threshold = float(thresholds["vive_std_m"])
    encoder_std_threshold = float(thresholds["encoder_std_ticks"])
    warning_masks: list[tuple[str, pd.Series]] = []
    for prefix in IMU_PREFIXES:
        column = f"{prefix}_gyro_rms_rad_s"
        warning_column = f"warning_{prefix}_gyro_rms_high"
        canonical[warning_column] = canonical[column].gt(gyro_threshold)
        warning_masks.append((f"{prefix}_gyro_rms_high", canonical[warning_column]))
    canonical["warning_vive_std_high"] = canonical[
        [f"vive_pos_{axis}_std_m" for axis in ("x", "y", "z")]
    ].gt(vive_std_threshold).any(axis=1)
    warning_masks.append(("vive_std_high", canonical["warning_vive_std_high"]))
    canonical["warning_encoder_std_high"] = canonical[
        [f"encoder_tick_j{joint}_std" for joint in range(3)]
    ].gt(encoder_std_threshold).any(axis=1)
    warning_masks.append(("encoder_std_high", canonical["warning_encoder_std_high"]))
    canonical["qc_warning_flags"] = _joined_reasons(warning_masks)
    canonical["strict_qc_eligible"] = canonical["train_eligible"] & canonical[
        "qc_warning_flags"
    ].eq("")
    canonical["is_synthetic"] = False
    canonical["sample_weight"] = 1.0

    eligible = canonical.loc[canonical["train_eligible"]]
    expected_eligible = int(cleaning_config["expected_train_eligible_rows"])
    if len(eligible) != expected_eligible:
        raise CleaningContractError(
            f"train-eligible row count changed: expected {expected_eligible}, received {len(eligible)}"
        )
    _expected_counts(
        eligible,
        cleaning_config["expected_train_segment_rows"],
        label="train-eligible",
    )

    canonical.attrs.update(source.attrs)
    canonical.attrs.update(
        {
            "session_id": str(session_id),
            "source_member_sha256": str(source_member_sha256).lower(),
            "canonical_rows": len(canonical),
            "train_eligible_rows": len(eligible),
        }
    )
    return canonical
