"""Leakage-safe preprocessing and recipe weighting for endpoint regression."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np


_ALWAYS_FORBIDDEN_FEATURE_PREFIXES = (
    "vive_",
    "pred_vive_",
    "label_",
    "target_id",
    "segment_id",
    "timestamp",
    "qc_",
    "provenance_",
    "parent_",
    "trial_uid",
)


def _matrix(values: Any, *, name: str, columns: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(
            f"{name} must have {columns} columns; received {array.shape[1]}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_feature_allowlist(
    feature_names: Sequence[str], *, allow_planned_diagnostic: bool = False
) -> tuple[str, ...]:
    """Validate a model input allow-list and return its immutable form.

    Raw Vive values and row/split/provenance identifiers are forbidden for all
    models.  Planned coordinates may only be enabled explicitly for the locked
    shortcut diagnostic; they are never accepted accidentally by a primary
    model.
    """

    names = tuple(str(name) for name in feature_names)
    if not names or any(not name for name in names):
        raise ValueError("feature allow-list must contain non-empty names")
    if len(names) != len(set(names)):
        raise ValueError("feature allow-list contains duplicate names")
    forbidden: list[str] = []
    for name in names:
        lowered = name.lower()
        if lowered.startswith("planned_"):
            if not allow_planned_diagnostic:
                forbidden.append(name)
            continue
        if lowered.startswith(_ALWAYS_FORBIDDEN_FEATURE_PREFIXES):
            forbidden.append(name)
    if forbidden:
        raise ValueError(f"feature allow-list contains leakage fields: {forbidden}")
    return names


def _uid_digest(trial_uids: Iterable[Any]) -> tuple[int, str]:
    values = [str(value) for value in trial_uids]
    if any(not value for value in values):
        raise ValueError("trial_uids must be non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError("trial_uids must be unique for preprocessor fitting")
    payload = "\n".join(sorted(values)).encode("utf-8")
    return len(values), hashlib.sha256(payload).hexdigest()


@dataclass
class TrainOnlyPreprocessor:
    """Feature scaling and isotropic target scaling fit on real train only.

    The target is centred per XYZ axis but divided by one shared scalar.  That
    affine transformation therefore cannot silently change relative axis
    weights.  Feature standardization can be disabled for tree estimators.
    """

    feature_names: tuple[str, ...]
    scale_features: bool = True
    split_context: str = ""
    allow_planned_diagnostic: bool = False
    feature_mean_: np.ndarray | None = field(default=None, init=False, repr=False)
    feature_scale_: np.ndarray | None = field(default=None, init=False, repr=False)
    target_mean_: np.ndarray | None = field(default=None, init=False, repr=False)
    target_scale_: float | None = field(default=None, init=False)
    leakage_audit_: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.feature_names = validate_feature_allowlist(
            self.feature_names,
            allow_planned_diagnostic=self.allow_planned_diagnostic,
        )

    @property
    def is_fitted(self) -> bool:
        return self.feature_mean_ is not None

    def fit_real_train(
        self,
        X_real_train: Any,
        y_real_train: Any,
        *,
        trial_uids: Sequence[Any],
        forbidden_uids: Sequence[Any] = (),
    ) -> "TrainOnlyPreprocessor":
        """Fit exclusively on measured rows assigned the train role."""

        X = _matrix(
            X_real_train, name="X_real_train", columns=len(self.feature_names)
        )
        y = _matrix(y_real_train, name="y_real_train", columns=3)
        if len(X) != len(y):
            raise ValueError("X_real_train and y_real_train row counts differ")
        uid_count, uid_hash = _uid_digest(trial_uids)
        if uid_count != len(X):
            raise ValueError("trial_uids must contain one identifier per real train row")
        forbidden = {str(value) for value in forbidden_uids}
        overlap = forbidden.intersection(str(value) for value in trial_uids)
        if overlap:
            preview = sorted(overlap)[:5]
            raise ValueError(
                "preprocessor fit rows overlap forbidden validation/test IDs: "
                f"{preview}"
            )

        observed_feature_mean = np.mean(X, axis=0, dtype=np.float64)
        raw_feature_scale = np.std(X, axis=0, ddof=0, dtype=np.float64)
        if self.scale_features:
            self.feature_mean_ = observed_feature_mean
            self.feature_scale_ = np.where(raw_feature_scale > 0.0, raw_feature_scale, 1.0)
        else:
            self.feature_mean_ = np.zeros(X.shape[1], dtype=np.float64)
            self.feature_scale_ = np.ones(X.shape[1], dtype=np.float64)
        self.target_mean_ = np.mean(y, axis=0, dtype=np.float64)
        target_scale = float(np.sqrt(np.mean(np.square(y - self.target_mean_))))
        self.target_scale_ = target_scale if target_scale > 0.0 else 1.0
        self.leakage_audit_ = {
            "fit_scope": "real_train_only",
            "split_context": self.split_context,
            "fit_row_count": int(len(X)),
            "fit_trial_uid_sha256": uid_hash,
            "forbidden_uid_count": int(len(forbidden)),
            "forbidden_uid_overlap_count": 0,
            "feature_scaling_enabled": bool(self.scale_features),
            "target_centered_per_axis": True,
            "target_scale_kind": "single_isotropic_xyz_scale",
        }
        return self

    def _check_fitted(self) -> None:
        if (
            self.feature_mean_ is None
            or self.feature_scale_ is None
            or self.target_mean_ is None
            or self.target_scale_ is None
            or self.leakage_audit_ is None
        ):
            raise RuntimeError("preprocessor has not been fit")

    def transform_features(self, X: Any) -> np.ndarray:
        self._check_fitted()
        matrix = _matrix(X, name="X", columns=len(self.feature_names))
        assert self.feature_mean_ is not None
        assert self.feature_scale_ is not None
        return (matrix - self.feature_mean_) / self.feature_scale_

    def transform_targets(self, y: Any) -> np.ndarray:
        self._check_fitted()
        matrix = _matrix(y, name="y", columns=3)
        assert self.target_mean_ is not None
        assert self.target_scale_ is not None
        return (matrix - self.target_mean_) / self.target_scale_

    def inverse_targets(self, y_scaled: Any) -> np.ndarray:
        self._check_fitted()
        matrix = _matrix(y_scaled, name="y_scaled", columns=3)
        assert self.target_mean_ is not None
        assert self.target_scale_ is not None
        result = matrix * self.target_scale_ + self.target_mean_
        if not np.isfinite(result).all():
            raise ValueError("inverse-transformed targets contain non-finite values")
        return result

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe fitted state and leakage-audit evidence."""

        self._check_fitted()
        assert self.feature_mean_ is not None
        assert self.feature_scale_ is not None
        assert self.target_mean_ is not None
        assert self.target_scale_ is not None
        assert self.leakage_audit_ is not None
        return {
            "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean_.tolist(),
            "feature_scale": self.feature_scale_.tolist(),
            "target_mean": self.target_mean_.tolist(),
            "target_scale": float(self.target_scale_),
            "leakage_audit": dict(self.leakage_audit_),
        }


def recipe_sample_weights(
    real_row_count: int,
    additional_row_count: int = 0,
    *,
    recipe: str,
) -> np.ndarray:
    """Return the locked objective-mass weights for one training recipe."""

    real_count = int(real_row_count)
    additional_count = int(additional_row_count)
    if real_count <= 0 or additional_count < 0:
        raise ValueError("row counts must be positive for real and non-negative for added")
    if recipe == "real_only":
        if additional_count != 0:
            raise ValueError("real_only cannot contain additional rows")
        return np.ones(real_count, dtype=np.float64)
    if recipe not in {"replay", "local_interp_v2", "cmixup_offline_v1"}:
        raise KeyError(f"unknown training recipe {recipe!r}")
    if additional_count != real_count:
        raise ValueError(
            f"{recipe} requires exactly 1:1 real/additional rows; "
            f"received {real_count}:{additional_count}"
        )
    return np.concatenate(
        [
            np.full(real_count, 0.5, dtype=np.float64),
            np.full(additional_count, 0.5, dtype=np.float64),
        ]
    )


def weight_mass_audit(
    weights: Any, *, real_row_count: int, recipe: str
) -> dict[str, float | int | str]:
    """Fail unless the recipe keeps the real-only total objective mass."""

    vector = np.asarray(weights, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all() or np.any(vector < 0.0):
        raise ValueError("weights must be a finite non-negative vector")
    expected_mass = float(real_row_count)
    actual_mass = float(vector.sum())
    if not np.isclose(actual_mass, expected_mass, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"objective mass changed for {recipe}: {actual_mass} != {expected_mass}"
        )
    return {
        "recipe": recipe,
        "row_count": int(len(vector)),
        "objective_mass": actual_mass,
        "expected_real_only_mass": expected_mass,
    }
