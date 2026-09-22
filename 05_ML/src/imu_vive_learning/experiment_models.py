"""Locked model families and serializable endpoint-regression wrappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .experiment_preprocessing import (
    TrainOnlyPreprocessor,
    validate_feature_allowlist,
    weight_mass_audit,
)

try:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
except ImportError as exc:  # pragma: no cover - exercised by install diagnostics
    raise ImportError(
        "endpoint evaluation requires the project 'eval' dependencies; "
        "install with python -m pip install -e '.[eval]'"
    ) from exc


BASE_SEED = 20_260_818
MODEL_TIE_ORDER = ("dummy_mean", "ridge", "extra_trees", "xgboost", "mlp")
RECIPE_TIE_ORDER = (
    "real_only",
    "local_interp_v2",
    "cmixup_offline_v1",
    "replay",
)
SCALABLE_MODELS = {"ridge", "mlp"}


def _normalized_json(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(item[key]) for key in sorted(item)}
        if isinstance(item, (list, tuple)):
            return [normalize(element) for element in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(
        normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def derive_context_seed(base_seed: int = BASE_SEED, *context: Any) -> int:
    """Derive a stable uint32 seed without Python's process-randomized hash."""

    payload = _normalized_json([int(base_seed), *context]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def paired_seed_sequence(
    split_context: str, *, base_seed: int = BASE_SEED, count: int = 3
) -> tuple[int, ...]:
    """Return the locked paired model seeds for a split context."""

    if count <= 0:
        raise ValueError("seed count must be positive")
    return tuple(
        derive_context_seed(base_seed, "paired_model_seed", split_context, index)
        for index in range(count)
    )


def mlp_epochs_for_recipe(recipe: str, *, real_only_epochs: int = 400) -> int:
    """Keep the fixed real-equivalent row-visit budget across 1:1 recipes."""

    if real_only_epochs <= 0 or real_only_epochs % 2:
        raise ValueError("real_only_epochs must be a positive even integer")
    if recipe == "real_only":
        return int(real_only_epochs)
    if recipe in {"replay", "local_interp_v2", "cmixup_offline_v1"}:
        return int(real_only_epochs // 2)
    raise KeyError(f"unknown training recipe {recipe!r}")


def model_parameter_grid(model_name: str) -> list[dict[str, Any]]:
    """Return the pre-registered grid in deterministic simple-first order."""

    if model_name in {"dummy", "dummy_mean"}:
        return [{}]
    if model_name == "ridge":
        return [{"alpha": alpha} for alpha in (100.0, 10.0, 1.0, 0.1)]
    if model_name == "extra_trees":
        return [
            {"max_features": max_features, "min_samples_leaf": min_leaf}
            for max_features, min_leaf in itertools.product((0.5, 1.0), (4, 1))
        ]
    if model_name == "xgboost":
        return [
            {
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "reg_lambda": reg_lambda,
            }
            for max_depth, min_child_weight, reg_lambda in itertools.product(
                (4, 8), (5.0, 1.0), (10.0, 1.0)
            )
        ]
    if model_name == "mlp":
        return [
            {"hidden_layer_sizes": hidden, "alpha": alpha}
            for hidden, alpha in itertools.product(
                ((64, 64), (128, 64)), (1e-3, 1e-4)
            )
        ]
    raise KeyError(f"unknown model family {model_name!r}")


def configuration_id(model_name: str, params: Mapping[str, Any]) -> str:
    """Return a content-derived ID for selection traces and checkpoints."""

    payload = _normalized_json({"model": model_name, "params": params})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def select_best_configuration(
    model_name: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    score_key: str = "macro_group_rmse_3d_mm",
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Select the lowest inner-macro score with a frozen simple-first tie.

    A candidate may provide ``score_key`` directly or ``inner_scores``.  The
    latter is averaged with equal inner-group weight.  Parameters must belong
    to the locked grid so selection cannot silently expand after seeing data.
    """

    if not candidates:
        raise ValueError("at least one selection candidate is required")
    grid = model_parameter_grid(model_name)
    grid_ids = [configuration_id(model_name, params) for params in grid]
    priority = {config_id: index for index, config_id in enumerate(grid_ids)}
    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        if "params" not in candidate:
            raise ValueError("selection candidate is missing params")
        params = dict(candidate["params"])
        config_id = configuration_id(model_name, params)
        if config_id not in priority:
            raise ValueError(
                f"candidate parameters are outside the locked {model_name} grid: {params}"
            )
        if score_key in candidate:
            score = float(candidate[score_key])
        elif "inner_scores" in candidate:
            scores = np.asarray(candidate["inner_scores"], dtype=np.float64)
            if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
                raise ValueError("inner_scores must be a non-empty finite vector")
            score = float(np.mean(scores))
        else:
            raise ValueError(f"selection candidate lacks {score_key!r} or inner_scores")
        if not np.isfinite(score):
            raise ValueError("selection scores must be finite")
        normalized.append(
            {
                **dict(candidate),
                "params": params,
                "config_id": config_id,
                score_key: score,
                "tie_priority": priority[config_id],
            }
        )
    best_score = min(float(row[score_key]) for row in normalized)
    tied = [
        row
        for row in normalized
        if float(row[score_key]) <= best_score + float(tie_tolerance)
    ]
    selected = min(tied, key=lambda row: int(row["tie_priority"]))
    return {
        **selected,
        "selection_model": model_name,
        "selection_metric": score_key,
        "tied_candidate_count": len(tied),
    }


def augmentation_gate(
    *,
    reference_macro_rmse_3d_mm: float,
    candidate_macro_rmse_3d_mm: float,
    reference_macro_p95_3d_error_mm: float,
    candidate_macro_p95_3d_error_mm: float,
) -> dict[str, float | bool]:
    """Apply the pre-registered engineering gate for augmentation."""

    reference_rmse = float(reference_macro_rmse_3d_mm)
    candidate_rmse = float(candidate_macro_rmse_3d_mm)
    reference_p95 = float(reference_macro_p95_3d_error_mm)
    candidate_p95 = float(candidate_macro_p95_3d_error_mm)
    values = np.asarray(
        [reference_rmse, candidate_rmse, reference_p95, candidate_p95],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("gate metrics must be finite and non-negative")
    required_reduction = max(0.10 * reference_rmse, 3.0)
    actual_reduction = reference_rmse - candidate_rmse
    p95_limit = 1.05 * reference_p95
    rmse_pass = actual_reduction >= required_reduction
    p95_pass = candidate_p95 <= p95_limit
    return {
        "required_rmse_reduction_mm": required_reduction,
        "actual_rmse_reduction_mm": actual_reduction,
        "p95_limit_mm": p95_limit,
        "rmse_pass": bool(rmse_pass),
        "p95_pass": bool(p95_pass),
        "passed": bool(rmse_pass and p95_pass),
    }


def select_final_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select among deployment-safe primary candidates.

    Every row must include ``recipe``,
    ``model_name``, ``macro_rmse_3d_mm`` and ``augmentation_gate_passed``.
    Encoder, four-IMU, and fusion real-only candidates remain eligible;
    augmentation is fusion-only and must pass its gate. Diagnostic
    planned/magnetometer/replay/Dummy candidates are rejected here.
    """

    allowed_recipes = {"real_only", "local_interp_v2", "cmixup_offline_v1"}
    primary_views = {
        "encoder",
        "four_imu",
        "fusion",
        "encoder_only_v1",
        "four_imu_endpoint_v1",
        "fusion_v1",
    }
    fusion_views = {"fusion", "fusion_v1"}
    rows = [dict(candidate) for candidate in candidates]
    if not rows:
        raise ValueError("at least one final candidate is required")
    eligible: list[dict[str, Any]] = []
    real_rows: list[dict[str, Any]] = []
    for row in rows:
        recipe = str(row.get("recipe", ""))
        if recipe not in allowed_recipes:
            continue
        feature_view = str(row.get("feature_view", ""))
        model_name = str(row.get("model_name", ""))
        if feature_view not in primary_views or model_name == "dummy_mean":
            continue
        score = float(row["macro_rmse_3d_mm"])
        if not np.isfinite(score):
            raise ValueError("final selection score must be finite")
        if recipe == "real_only":
            real_rows.append(row)
            eligible.append(row)
        elif feature_view in fusion_views and bool(
            row.get("augmentation_gate_passed", False)
        ):
            eligible.append(row)
    if not real_rows:
        raise ValueError("final selection requires at least one primary real-only fallback")
    if not any(row["recipe"] != "real_only" for row in eligible):
        eligible = real_rows
    model_priority = {name: index for index, name in enumerate(MODEL_TIE_ORDER)}
    recipe_priority = {name: index for index, name in enumerate(RECIPE_TIE_ORDER)}
    feature_priority = {
        "encoder": 0,
        "encoder_only_v1": 0,
        "four_imu": 1,
        "four_imu_endpoint_v1": 1,
        "fusion": 2,
        "fusion_v1": 2,
    }
    selected = min(
        eligible,
        key=lambda row: (
            float(row["macro_rmse_3d_mm"]),
            model_priority.get(str(row["model_name"]), len(model_priority)),
            recipe_priority.get(str(row["recipe"]), len(recipe_priority)),
            feature_priority.get(str(row["feature_view"]), len(feature_priority)),
        ),
    )
    return {
        **selected,
        "selection_rule": "lowest_segment_inner_macro_rmse_with_gated_augmentation",
        "augmentation_selected": selected["recipe"] != "real_only",
    }


class ConstantXYZRegressor:
    """Measured-real train mean baseline that ignores every input feature."""

    def __init__(self, mean_xyz: Any):
        mean = np.asarray(mean_xyz, dtype=np.float64)
        if mean.shape != (3,) or not np.isfinite(mean).all():
            raise ValueError("mean_xyz must be a finite XYZ vector")
        self.mean_xyz_ = mean

    def predict(self, X: Any) -> np.ndarray:
        row_count = len(X)
        return np.repeat(self.mean_xyz_[None, :], row_count, axis=0)


class IndependentXYZXGBRegressor:
    """Three independent deterministic XGBoost regressors, one per axis."""

    def __init__(self, *, params: Mapping[str, Any], seed: int):
        self.params = dict(params)
        self.seed = int(seed)
        self.estimators_: list[Any] = []

    @staticmethod
    def _xgb_class() -> Any:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "model 'xgboost' requires the eval extra with xgboost installed"
            ) from exc
        return XGBRegressor

    def fit(
        self, X: Any, y: Any, sample_weight: Any | None = None
    ) -> "IndependentXYZXGBRegressor":
        matrix = np.asarray(X, dtype=np.float64)
        targets = np.asarray(y, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 3 or len(matrix) != len(targets):
            raise ValueError("X/y shapes are invalid for independent XYZ fitting")
        XGBRegressor = self._xgb_class()
        self.estimators_ = []
        for axis in range(3):
            estimator = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=int(self.params["max_depth"]),
                min_child_weight=float(self.params["min_child_weight"]),
                reg_lambda=float(self.params["reg_lambda"]),
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=1,
                random_state=derive_context_seed(self.seed, "xgb_axis", axis),
                tree_method="hist",
                verbosity=0,
            )
            estimator.fit(matrix, targets[:, axis], sample_weight=sample_weight)
            self.estimators_.append(estimator)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if len(self.estimators_) != 3:
            raise RuntimeError("independent XYZ XGBoost model has not been fit")
        matrix = np.asarray(X, dtype=np.float64)
        return np.column_stack([model.predict(matrix) for model in self.estimators_])


def _build_estimator(
    model_name: str,
    params: Mapping[str, Any],
    *,
    seed: int,
    recipe: str,
) -> Any:
    if model_name == "ridge":
        return Ridge(alpha=float(params["alpha"]))
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=400,
            max_features=float(params["max_features"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=int(seed),
            n_jobs=1,
            bootstrap=False,
        )
    if model_name == "xgboost":
        return IndependentXYZXGBRegressor(params=params, seed=seed)
    if model_name == "mlp":
        epochs = mlp_epochs_for_recipe(recipe)
        return MLPRegressor(
            hidden_layer_sizes=tuple(int(value) for value in params["hidden_layer_sizes"]),
            activation="relu",
            solver="adam",
            alpha=float(params["alpha"]),
            batch_size=128,
            learning_rate="constant",
            learning_rate_init=1e-3,
            max_iter=epochs,
            shuffle=True,
            random_state=int(seed),
            tol=0.0,
            early_stopping=False,
            n_iter_no_change=epochs + 1,
        )
    raise KeyError(f"unknown trainable model family {model_name!r}")


@dataclass
class SerializableEndpointRegressor:
    """Fitted estimator plus its exact feature and preprocessing contract."""

    model_name: str
    recipe: str
    feature_names: tuple[str, ...]
    estimator: Any
    preprocessor: TrainOnlyPreprocessor | None
    seed: int
    params: dict[str, Any] = field(default_factory=dict)
    fit_metadata: dict[str, Any] = field(default_factory=dict)

    def _prediction_matrix(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"model expects a matrix with {len(self.feature_names)} columns"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("inference features contain non-finite values")
        if self.preprocessor is None:
            prediction = self.estimator.predict(matrix)
        else:
            prediction_scaled = self.estimator.predict(
                self.preprocessor.transform_features(matrix)
            )
            prediction = self.preprocessor.inverse_targets(prediction_scaled)
        result = np.asarray(prediction, dtype=np.float64)
        if result.shape != (len(matrix), 3) or not np.isfinite(result).all():
            raise ValueError("model produced an invalid finite XYZ prediction matrix")
        return result

    def predict(self, X: Any) -> np.ndarray:
        return self._prediction_matrix(X)

    def validate_inference_frame(self, frame: Any) -> tuple[np.ndarray, np.ndarray | None]:
        """Require optional ``row_id`` followed by the exact feature allow-list."""

        if not hasattr(frame, "columns"):
            raise TypeError("inference input must be a pandas-like table with columns")
        columns = [str(column) for column in frame.columns]
        expected = list(self.feature_names)
        row_ids: np.ndarray | None = None
        if columns == ["row_id", *expected]:
            row_ids = np.asarray(frame["row_id"], dtype=object)
            if len(set(row_ids.tolist())) != len(row_ids):
                raise ValueError("row_id values must be unique")
        elif columns != expected:
            raise ValueError(
                "inference schema/order mismatch; expected "
                f"{expected} or {['row_id', *expected]}, received {columns}"
            )
        matrix = frame.loc[:, expected].to_numpy(dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) == 0 or not np.isfinite(matrix).all():
            raise ValueError("inference feature table must be non-empty and finite")
        return matrix, row_ids

    def predict_frame(self, frame: Any) -> tuple[np.ndarray, np.ndarray | None]:
        matrix, row_ids = self.validate_inference_frame(frame)
        return self.predict(matrix), row_ids

    def contract_fingerprint(self) -> str:
        payload = {
            "model_name": self.model_name,
            "recipe": self.recipe,
            "feature_names": self.feature_names,
            "seed": self.seed,
            "params": self.params,
            "preprocessing": (
                None if self.preprocessor is None else self.preprocessor.metadata()
            ),
            "fit_metadata": self.fit_metadata,
        }
        return hashlib.sha256(_normalized_json(payload).encode("utf-8")).hexdigest()


def fit_endpoint_regressor(
    model_name: str,
    params: Mapping[str, Any],
    X_train: Any,
    y_train: Any,
    *,
    feature_names: Sequence[str],
    X_real_train: Any,
    y_real_train: Any,
    real_trial_uids: Sequence[Any],
    forbidden_uids: Sequence[Any] = (),
    sample_weight: Any | None = None,
    recipe: str = "real_only",
    split_context: str,
    seed: int = BASE_SEED,
    allow_planned_diagnostic: bool = False,
) -> SerializableEndpointRegressor:
    """Fit one locked estimator while fitting transforms on real train only."""

    names = tuple(str(name) for name in feature_names)
    train_y = np.asarray(y_train, dtype=np.float64)
    real_y = np.asarray(y_real_train, dtype=np.float64)
    if train_y.ndim != 2 or train_y.shape[1] != 3 or not np.isfinite(train_y).all():
        raise ValueError("y_train must be a finite XYZ matrix")
    if real_y.ndim != 2 or real_y.shape[1] != 3 or not np.isfinite(real_y).all():
        raise ValueError("y_real_train must be a finite XYZ matrix")
    if len(real_y) != len(real_trial_uids):
        raise ValueError("real_trial_uids length differs from y_real_train")
    weights = (
        np.ones(len(train_y), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if weights.shape != (len(train_y),) or not np.isfinite(weights).all():
        raise ValueError("sample_weight must be finite with one value per train row")
    weight_audit = weight_mass_audit(
        weights, real_row_count=len(real_y), recipe=recipe
    )
    # ``seed`` is already derived from the split context and seed index by the
    # runner.  Reuse it verbatim so recipe, feature-view, and hyperparameter
    # comparisons are genuinely paired instead of confounding the candidate
    # effect with a different random initialization.
    model_seed = int(seed)

    if model_name in {"dummy", "dummy_mean"}:
        if names:
            raise ValueError("dummy baseline must not declare input features")
        X = np.asarray(X_train, dtype=np.float64)
        if X.shape != (len(train_y), 0):
            raise ValueError("dummy X_train must have shape (n_rows, 0)")
        estimator = ConstantXYZRegressor(np.mean(real_y, axis=0))
        return SerializableEndpointRegressor(
            model_name="dummy_mean",
            recipe="real_only",
            feature_names=(),
            estimator=estimator,
            preprocessor=None,
            seed=model_seed,
            params={},
            fit_metadata={
                "fit_scope": "real_train_only",
                "split_context": split_context,
                "fit_row_count": int(len(real_y)),
                "objective_weight_audit": weight_audit,
            },
        )

    names = validate_feature_allowlist(
        names, allow_planned_diagnostic=allow_planned_diagnostic
    )
    X = np.asarray(X_train, dtype=np.float64)
    real_X = np.asarray(X_real_train, dtype=np.float64)
    if X.shape != (len(train_y), len(names)) or not np.isfinite(X).all():
        raise ValueError("X_train shape/finiteness does not match y/features")
    if real_X.shape != (len(real_y), len(names)) or not np.isfinite(real_X).all():
        raise ValueError("X_real_train shape/finiteness does not match y/features")

    preprocessor = TrainOnlyPreprocessor(
        feature_names=names,
        scale_features=model_name in SCALABLE_MODELS,
        split_context=split_context,
        allow_planned_diagnostic=allow_planned_diagnostic,
    ).fit_real_train(
        real_X,
        real_y,
        trial_uids=real_trial_uids,
        forbidden_uids=forbidden_uids,
    )
    estimator = _build_estimator(
        model_name, params, seed=model_seed, recipe=recipe
    )
    transformed_X = preprocessor.transform_features(X)
    transformed_y = preprocessor.transform_targets(train_y)
    estimator.fit(transformed_X, transformed_y, sample_weight=weights)
    return SerializableEndpointRegressor(
        model_name=model_name,
        recipe=recipe,
        feature_names=names,
        estimator=estimator,
        preprocessor=preprocessor,
        seed=model_seed,
        params=dict(params),
        fit_metadata={
            "split_context": split_context,
            "fit_row_count": int(len(X)),
            "real_fit_row_count": int(len(real_X)),
            "objective_weight_audit": weight_audit,
            "estimator_thread_count": 1,
        },
    )


def save_endpoint_regressor(
    model: SerializableEndpointRegressor, path: str | Path
) -> dict[str, str | int]:
    """Serialize a trusted local model and return its file/hash evidence."""

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("joblib is required to serialize the final model") from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination, compress=0, protocol=5)
    payload = destination.read_bytes()
    return {
        "path": destination.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "contract_fingerprint": model.contract_fingerprint(),
    }


def load_endpoint_regressor(path: str | Path) -> SerializableEndpointRegressor:
    """Load a trusted package member and enforce the wrapper type."""

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("joblib is required to load the final model") from exc
    model = joblib.load(Path(path))
    if not isinstance(model, SerializableEndpointRegressor):
        raise TypeError("serialized object is not a SerializableEndpointRegressor")
    return model
