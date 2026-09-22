"""Training-context recipes for endpoint-regression experiment v1.

All functions in this module require an already resolved split context.  Inner
augmentation is regenerated exclusively from real inner-train rows.  Only an
outer-refit context may read the immutable, materialized synthetic members
from ``alphaf.ml_preparation.v1``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .augmentation import (
    CMIXUP_METHOD,
    LOCAL_METHOD,
    cmixup_offline_v1,
    derived_seed,
    local_interp_v2,
)
from .features import (
    ACCEL_COLUMNS,
    ENCODER_COLUMNS,
    EULER_SOURCE_COLUMNS,
    GYRO_COLUMNS,
    LABEL_COLUMNS,
    MAG_COLUMNS,
    PLANNED_COLUMNS,
)
from .experiment_release import ReleaseContractError, ReleaseData
from .experiment_splits import context_assignments


BASE_SEED = 20_260_818
REAL_ONLY_RECIPE = "real_only"
REPLAY_RECIPE = "replay"
SUPPORTED_RECIPES = (REAL_ONLY_RECIPE, REPLAY_RECIPE, LOCAL_METHOD, CMIXUP_METHOD)
_LINEAR_COLUMNS = ENCODER_COLUMNS + ACCEL_COLUMNS + MAG_COLUMNS + GYRO_COLUMNS
_TRIG_MAP = {
    name: (f"{name}_sin", f"{name}_cos") for name in EULER_SOURCE_COLUMNS
}


class ExperimentAugmentationError(ValueError):
    """Raised when a recipe would violate its training context."""


@dataclass(frozen=True)
class AugmentationSettings:
    planned_max_distance_mm: float = 100.0
    joint_max_distance_deg: float = 20.0
    local_neighbor_count: int = 5
    local_lambdas: tuple[float, ...] = (0.25, 0.5, 0.75)
    cmixup_alpha: float = 2.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AugmentationSettings":
        return cls(
            planned_max_distance_mm=float(payload["planned_max_distance_mm"]),
            joint_max_distance_deg=float(payload["joint_max_distance_deg"]),
            local_neighbor_count=int(payload["local_neighbor_count"]),
            local_lambdas=tuple(float(value) for value in payload["local_lambdas"]),
            cmixup_alpha=float(payload["cmixup_alpha"]),
        ).validated()

    def validated(self) -> "AugmentationSettings":
        if not math.isfinite(self.planned_max_distance_mm) or self.planned_max_distance_mm < 0:
            raise ExperimentAugmentationError(
                "planned_max_distance_mm must be finite and non-negative"
            )
        if not math.isfinite(self.joint_max_distance_deg) or self.joint_max_distance_deg < 0:
            raise ExperimentAugmentationError(
                "joint_max_distance_deg must be finite and non-negative"
            )
        if self.local_neighbor_count < 1:
            raise ExperimentAugmentationError("local_neighbor_count must be positive")
        lambdas = np.asarray(self.local_lambdas, dtype=np.float64)
        if (
            lambdas.ndim != 1
            or len(lambdas) == 0
            or not np.isfinite(lambdas).all()
            or np.any(lambdas <= 0.0)
            or np.any(lambdas >= 1.0)
        ):
            raise ExperimentAugmentationError(
                "local_lambdas must be finite and strictly between zero and one"
            )
        if not math.isfinite(self.cmixup_alpha) or self.cmixup_alpha <= 0:
            raise ExperimentAugmentationError("cmixup_alpha must be finite and positive")
        return self


@dataclass(frozen=True)
class TrainingRecipe:
    """One leakage-audited model-fit frame and its two mass components."""

    name: str
    scheme: str
    outer_fold: int
    inner_fold: int
    real: pd.DataFrame
    additional: pd.DataFrame
    combined: pd.DataFrame
    context_seed: int

    @property
    def real_rows(self) -> int:
        return len(self.real)

    @property
    def additional_rows(self) -> int:
        return len(self.additional)


def context_seed(
    *,
    scheme: str,
    outer_fold: int,
    inner_fold: int,
    recipe: str,
    base_seed: int = BASE_SEED,
) -> int:
    """Derive a deterministic uint63 seed from the frozen base and context hash."""

    if int(base_seed) != BASE_SEED:
        raise ExperimentAugmentationError(
            f"endpoint-regression v1 requires base seed {BASE_SEED}"
        )
    if recipe not in SUPPORTED_RECIPES:
        raise ExperimentAugmentationError(f"unknown training recipe: {recipe!r}")
    return derived_seed(
        int(base_seed),
        "alphaf.endpoint_regression.experiment.v1",
        str(scheme),
        int(outer_fold),
        int(inner_fold),
        str(recipe),
    )


def build_training_recipe(
    release: ReleaseData,
    experiment_assignments: pd.DataFrame,
    *,
    scheme: str,
    outer_fold: int,
    inner_fold: int,
    recipe: str,
    experiment_config_sha256: str,
    settings: AugmentationSettings | None = None,
) -> TrainingRecipe:
    """Build one weighted recipe without accessing validation/test observations."""

    if recipe not in SUPPORTED_RECIPES:
        raise ExperimentAugmentationError(f"unknown training recipe: {recipe!r}")
    _validate_digest(experiment_config_sha256, "experiment_config_sha256")
    context = context_assignments(
        experiment_assignments,
        scheme=scheme,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    real = _select_real_train(release.canonical, context)
    seed = context_seed(
        scheme=scheme,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        recipe=recipe,
        base_seed=release.base_seed,
    )
    if recipe == REAL_ONLY_RECIPE:
        real_fit = _mark_real(real, sample_weight=1.0)
        additional = real_fit.iloc[0:0].copy()
    elif recipe == REPLAY_RECIPE:
        real_fit = _mark_real(real, sample_weight=0.5)
        additional = _replay_rows(real, seed=seed)
    else:
        real_fit = _mark_real(real, sample_weight=0.5)
        if int(inner_fold) == -1:
            additional = load_outer_recipe(
                release,
                context,
                scheme=scheme,
                outer_fold=outer_fold,
                method=recipe,
                experiment_config_sha256=experiment_config_sha256,
            )
        else:
            augmentation_settings = (
                settings
                if settings is not None
                else AugmentationSettings.from_mapping(
                    release.preparation_config["augmentation"]
                )
            ).validated()
            additional = materialize_inner_synthetic(
                real,
                scheme=scheme,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                method=recipe,
                source_hash=release.source_member_sha256,
                experiment_config_sha256=experiment_config_sha256,
                settings=augmentation_settings,
                base_seed=release.base_seed,
            )
    combined = _combine(real_fit, additional, recipe=recipe)
    result = TrainingRecipe(
        name=recipe,
        scheme=str(scheme),
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        real=real_fit,
        additional=additional,
        combined=combined,
        context_seed=seed,
    )
    audit_training_recipe(result, context=context)
    return result


def materialize_inner_synthetic(
    train_real: pd.DataFrame,
    *,
    scheme: str,
    outer_fold: int,
    inner_fold: int,
    method: str,
    source_hash: str,
    experiment_config_sha256: str,
    settings: AugmentationSettings = AugmentationSettings(),
    base_seed: int = BASE_SEED,
) -> pd.DataFrame:
    """Regenerate 1:1 synthetic data for one declared inner-train context."""

    if int(inner_fold) == -1:
        raise ExperimentAugmentationError(
            "inner synthetic regeneration requires a non-refit inner fold"
        )
    if method not in {LOCAL_METHOD, CMIXUP_METHOD}:
        raise ExperimentAugmentationError(f"unsupported augmentation method: {method!r}")
    settings = settings.validated()
    _validate_digest(source_hash, "source_hash")
    _validate_digest(experiment_config_sha256, "experiment_config_sha256")
    seed = context_seed(
        scheme=scheme,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        recipe=method,
        base_seed=base_seed,
    )
    shared: dict[str, Any] = {
        "planned_columns": PLANNED_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "linear_columns": _LINEAR_COLUMNS,
        "euler_columns": EULER_SOURCE_COLUMNS,
        "euler_trig_columns": _TRIG_MAP,
        "split_scheme": str(scheme),
        "fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "base_seed": seed,
        "source_hash": source_hash,
        "config_hash": experiment_config_sha256,
        "planned_max_distance_mm": settings.planned_max_distance_mm,
        "sample_weight": 0.5,
    }
    if method == LOCAL_METHOD:
        result = local_interp_v2(
            train_real,
            neighbor_count=settings.local_neighbor_count,
            lambdas=settings.local_lambdas,
            **shared,
        )
    else:
        result = cmixup_offline_v1(
            train_real,
            joint_rad_columns=ENCODER_COLUMNS,
            joint_max_distance_deg=settings.joint_max_distance_deg,
            alpha=settings.cmixup_alpha,
            **shared,
        )
    result = result.copy()
    result["experiment_config_sha256"] = experiment_config_sha256
    result["training_origin"] = method
    result["training_row_id"] = result["synthetic_id"].astype(str)
    _audit_synthetic_parents(result, set(train_real["trial_uid"].astype(str)))
    if len(result) != len(train_real):
        raise ExperimentAugmentationError(
            f"inner augmentation is not 1:1: real={len(train_real)}, synthetic={len(result)}"
        )
    return result


def load_outer_recipe(
    release: ReleaseData,
    context: pd.DataFrame,
    *,
    scheme: str,
    outer_fold: int,
    method: str,
    experiment_config_sha256: str,
) -> pd.DataFrame:
    """Load a preparation member only for an outer-refit training context."""

    if context.empty or set(context["inner_fold"].astype(int)) != {-1}:
        raise ExperimentAugmentationError(
            "materialized preparation synthetic may be used for outer refit only"
        )
    _validate_digest(experiment_config_sha256, "experiment_config_sha256")
    try:
        synthetic = release.load_outer_synthetic(scheme, outer_fold, method)
    except ReleaseContractError as exc:
        raise ExperimentAugmentationError(str(exc)) from exc
    train_ids = set(context.loc[context["role"].eq("train"), "trial_uid"].astype(str))
    synthetic = synthetic.copy()
    synthetic["sample_weight"] = 0.5
    synthetic["experiment_config_sha256"] = experiment_config_sha256
    synthetic["training_origin"] = method
    synthetic["training_row_id"] = synthetic["synthetic_id"].astype(str)
    _audit_synthetic_parents(synthetic, train_ids)
    if len(synthetic) != len(train_ids):
        raise ExperimentAugmentationError(
            f"outer materialized augmentation is not 1:1: real={len(train_ids)}, "
            f"synthetic={len(synthetic)}"
        )
    return synthetic


def generate_final_full_synthetic(
    release: ReleaseData,
    *,
    method: str,
    experiment_config_sha256: str,
    settings: AugmentationSettings | None = None,
) -> pd.DataFrame:
    """Generate the selected final recipe from all 7,702 eligible real rows."""

    if method not in {LOCAL_METHOD, CMIXUP_METHOD}:
        raise ExperimentAugmentationError(
            "final full-data synthetic method must be local_interp_v2 or cmixup_offline_v1"
        )
    _validate_digest(experiment_config_sha256, "experiment_config_sha256")
    real = release.eligible_real
    augmentation_settings = (
        settings
        if settings is not None
        else AugmentationSettings.from_mapping(release.preparation_config["augmentation"])
    ).validated()
    seed = context_seed(
        scheme="final_full",
        outer_fold=0,
        inner_fold=0,
        recipe=method,
        base_seed=release.base_seed,
    )
    shared: dict[str, Any] = {
        "planned_columns": PLANNED_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "linear_columns": _LINEAR_COLUMNS,
        "euler_columns": EULER_SOURCE_COLUMNS,
        "euler_trig_columns": _TRIG_MAP,
        "split_scheme": "final_full",
        "fold": 0,
        "inner_fold": -1,
        "base_seed": seed,
        "source_hash": release.source_member_sha256,
        "config_hash": experiment_config_sha256,
        "planned_max_distance_mm": augmentation_settings.planned_max_distance_mm,
        "sample_weight": 0.5,
    }
    if method == LOCAL_METHOD:
        synthetic = local_interp_v2(
            real,
            neighbor_count=augmentation_settings.local_neighbor_count,
            lambdas=augmentation_settings.local_lambdas,
            **shared,
        )
    else:
        synthetic = cmixup_offline_v1(
            real,
            joint_rad_columns=ENCODER_COLUMNS,
            joint_max_distance_deg=augmentation_settings.joint_max_distance_deg,
            alpha=augmentation_settings.cmixup_alpha,
            **shared,
        )
    synthetic = synthetic.copy()
    synthetic["experiment_config_sha256"] = experiment_config_sha256
    synthetic["training_origin"] = method
    synthetic["training_row_id"] = synthetic["synthetic_id"].astype(str)
    _audit_synthetic_parents(synthetic, set(real["trial_uid"].astype(str)))
    if len(synthetic) != len(real):
        raise ExperimentAugmentationError("final full-data augmentation is not 1:1")
    return synthetic


def audit_training_recipe(recipe: TrainingRecipe, *, context: pd.DataFrame) -> None:
    """Assert role separation, parent scope, and frozen objective mass."""

    train_ids = set(context.loc[context["role"].eq("train"), "trial_uid"].astype(str))
    held_ids = set(
        context.loc[context["role"].isin(["validation", "test"]), "trial_uid"].astype(str)
    )
    real_ids = set(recipe.real["trial_uid"].astype(str))
    if real_ids != train_ids or real_ids & held_ids:
        raise ExperimentAugmentationError("recipe real rows differ from declared train IDs")
    if recipe.name == REAL_ONLY_RECIPE:
        if len(recipe.additional) != 0 or not np.allclose(recipe.real["sample_weight"], 1.0):
            raise ExperimentAugmentationError("real-only recipe mass changed")
    else:
        if len(recipe.additional) != len(recipe.real):
            raise ExperimentAugmentationError("additional recipe rows must be 1:1 with real train")
        if not np.allclose(recipe.real["sample_weight"], 0.5) or not np.allclose(
            recipe.additional["sample_weight"], 0.5
        ):
            raise ExperimentAugmentationError("replay/augmentation weights must each equal 0.5")
        if recipe.name in {LOCAL_METHOD, CMIXUP_METHOD}:
            _audit_synthetic_parents(recipe.additional, train_ids)
            parent_ids = set(recipe.additional["parent_a_trial_uid"].astype(str)) | set(
                recipe.additional["parent_b_trial_uid"].astype(str)
            )
            if parent_ids & held_ids:
                raise ExperimentAugmentationError("held-out ID appears in synthetic provenance")
    objective_mass = float(recipe.combined["sample_weight"].sum())
    if not math.isclose(objective_mass, float(len(recipe.real)), rel_tol=0.0, abs_tol=1e-9):
        raise ExperimentAugmentationError(
            f"recipe objective mass changed: expected {len(recipe.real)}, received {objective_mass}"
        )


def _select_real_train(canonical: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    train_ids = set(context.loc[context["role"].eq("train"), "trial_uid"].astype(str))
    if not train_ids:
        raise ExperimentAugmentationError("split context has no real training rows")
    eligible = canonical.loc[canonical["train_eligible"].eq(True)].copy()
    real = eligible.loc[eligible["trial_uid"].astype(str).isin(train_ids)].copy()
    real.sort_values("trial_uid", kind="mergesort", inplace=True)
    real.reset_index(drop=True, inplace=True)
    if len(real) != len(train_ids) or set(real["trial_uid"].astype(str)) != train_ids:
        raise ExperimentAugmentationError("split train IDs are absent or duplicated in canonical real")
    if "is_synthetic" in real and real["is_synthetic"].astype(bool).any():
        raise ExperimentAugmentationError("canonical train selector received synthetic rows")
    return real


def _mark_real(real: pd.DataFrame, *, sample_weight: float) -> pd.DataFrame:
    result = real.copy()
    result["sample_weight"] = float(sample_weight)
    result["training_origin"] = "real"
    result["training_row_id"] = result["trial_uid"].astype(str)
    return result


def _replay_rows(real: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    result = real.copy()
    result["sample_weight"] = 0.5
    result["training_origin"] = REPLAY_RECIPE
    result["replay_parent_trial_uid"] = result["trial_uid"].astype(str)
    result["replay_seed"] = int(seed)
    result["training_row_id"] = "replay:" + result["trial_uid"].astype(str)
    return result


def _combine(real: pd.DataFrame, additional: pd.DataFrame, *, recipe: str) -> pd.DataFrame:
    if recipe == REAL_ONLY_RECIPE:
        return real.reset_index(drop=True).copy()
    return pd.concat([real, additional], ignore_index=True, sort=False)


def _audit_synthetic_parents(synthetic: pd.DataFrame, train_ids: set[str]) -> None:
    required = {
        "synthetic_id",
        "parent_a_trial_uid",
        "parent_b_trial_uid",
        "sample_weight",
    }
    missing = sorted(required - set(synthetic.columns))
    if missing:
        raise ExperimentAugmentationError(f"synthetic provenance columns missing: {missing}")
    if synthetic.empty or synthetic["synthetic_id"].isna().any() or synthetic["synthetic_id"].duplicated().any():
        raise ExperimentAugmentationError("synthetic IDs must be non-null, unique, and non-empty")
    parents_a = set(synthetic["parent_a_trial_uid"].astype(str))
    parents_b = set(synthetic["parent_b_trial_uid"].astype(str))
    if not parents_a.issubset(train_ids) or not parents_b.issubset(train_ids):
        raise ExperimentAugmentationError("synthetic parent is outside the real train context")
    if (synthetic["parent_a_trial_uid"].astype(str) == synthetic["parent_b_trial_uid"].astype(str)).any():
        raise ExperimentAugmentationError("synthetic parent self-pair detected")
    weights = synthetic["sample_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ExperimentAugmentationError("synthetic weights must be finite and positive")


def _validate_digest(value: str, description: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentAugmentationError(
            f"{description} must be 64 lowercase hexadecimal digits"
        )


__all__ = [
    "AugmentationSettings",
    "BASE_SEED",
    "REAL_ONLY_RECIPE",
    "REPLAY_RECIPE",
    "SUPPORTED_RECIPES",
    "ExperimentAugmentationError",
    "TrainingRecipe",
    "audit_training_recipe",
    "build_training_recipe",
    "context_seed",
    "generate_final_full_synthetic",
    "load_outer_recipe",
    "materialize_inner_synthetic",
]
