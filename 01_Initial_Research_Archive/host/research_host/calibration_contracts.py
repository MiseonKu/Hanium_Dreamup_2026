from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_CALIBRATION_TYPES = (
    "imu_bias",
    "imu_alignment",
    "baseframe",
    "dh_params",
    "latency",
)

TYPE_REQUIRED_KEYS = {
    "imu_bias": {"calibration_type", "version", "date", "imus"},
    "imu_alignment": {"calibration_type", "version", "date", "alignments"},
    "baseframe": {
        "calibration_type",
        "version",
        "date",
        "R_world_to_base",
        "p_world_origin_in_base_mm",
    },
    "dh_params": {
        "calibration_type",
        "version",
        "date",
        "robot_model",
        "effective_fk_lengths_mm",
        "encoder_zero_offsets_rad",
    },
    "latency": {"calibration_type", "version", "date", "latency_s", "offset_s"},
}


@dataclass(frozen=True, slots=True)
class CalibrationValidationResult:
    calibration_type: str
    path: str
    issues: list[str]

    @property
    def ok(self) -> bool:
        return not self.issues


def validate_calibration_file(path: str | Path) -> CalibrationValidationResult:
    calibration_path = Path(path)
    issues: list[str] = []
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CalibrationValidationResult(
            calibration_type="unknown",
            path=str(calibration_path),
            issues=[f"invalid JSON: {exc}"],
        )

    calibration_type = str(payload.get("calibration_type", "unknown"))
    if calibration_type not in TYPE_REQUIRED_KEYS:
        issues.append(f"unknown calibration_type: {calibration_type}")
    else:
        missing = sorted(TYPE_REQUIRED_KEYS[calibration_type] - set(payload))
        if missing:
            issues.append("missing keys: " + ", ".join(missing))
        issues.extend(_type_specific_issues(calibration_type, payload))

    if payload.get("date") == "YYYY-MM-DD":
        issues.append("date still contains template placeholder")
    return CalibrationValidationResult(
        calibration_type=calibration_type,
        path=str(calibration_path),
        issues=issues,
    )


def validate_calibration_dir(path: str | Path) -> list[CalibrationValidationResult]:
    calibration_dir = Path(path)
    results = [
        validate_calibration_file(candidate)
        for candidate in sorted(calibration_dir.glob("*.json"))
    ]
    present_types = {result.calibration_type for result in results if result.calibration_type != "unknown"}
    for calibration_type in REQUIRED_CALIBRATION_TYPES:
        if calibration_type not in present_types:
            results.append(
                CalibrationValidationResult(
                    calibration_type=calibration_type,
                    path=str(calibration_dir / f"{calibration_type}.json"),
                    issues=["missing calibration file"],
                )
            )
    return results


def load_template_required_keys(template_dir: str | Path) -> dict[str, set[str]]:
    keys: dict[str, set[str]] = {}
    for template_path in sorted(Path(template_dir).glob("*.json")):
        payload = json.loads(template_path.read_text(encoding="utf-8"))
        calibration_type = str(payload.get("calibration_type", template_path.stem))
        keys[calibration_type] = set(payload)
    return keys


def _type_specific_issues(
    calibration_type: str,
    payload: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    if calibration_type in {"baseframe", "imu_alignment"}:
        matrices = []
        if calibration_type == "baseframe":
            matrices.append(("R_world_to_base", payload.get("R_world_to_base")))
        else:
            alignments = payload.get("alignments", {})
            if isinstance(alignments, Mapping):
                matrices.extend(
                    (f"alignments.{imu_id}.R_imu_to_link", value.get("R_imu_to_link"))
                    for imu_id, value in alignments.items()
                    if isinstance(value, Mapping)
                )
        for name, matrix in matrices:
            if not _is_3x3(matrix):
                issues.append(f"{name} must be a 3x3 matrix")

    if calibration_type == "dh_params":
        for key in ("effective_fk_lengths_mm", "encoder_zero_offsets_rad"):
            if not _is_length(payload.get(key), 3):
                issues.append(f"{key} must contain exactly 3 values")

    if calibration_type == "latency":
        for key in ("latency_s", "offset_s"):
            if not isinstance(payload.get(key), Mapping):
                issues.append(f"{key} must be an object")
    return issues


def _is_3x3(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in value)
    )


def _is_length(value: object, length: int) -> bool:
    return isinstance(value, list) and len(value) == length
