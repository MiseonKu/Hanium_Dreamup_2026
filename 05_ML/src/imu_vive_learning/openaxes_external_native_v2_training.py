"""Outcome-isolated model and training mathematics for OpenAXES native V2.

PyTorch and SciPy are imported only inside fit/prediction call sites.  Merely
importing this module is therefore CUDA-free, which lets a persistent slot
controller bind ``CUDA_VISIBLE_DEVICES`` before it starts a fresh fit child.
The trainer accepts only one fold's training partition; outer features and
labels have separate types and never appear in its signature.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Mapping, Sequence

import numpy as np

from . import openaxes_external_native_v2_config


NATIVE_CANDIDATE_IDS = (
    "dense_6979__scenario_balanced_masked_erm",
    "sensor_experts_6979__scenario_balanced_masked_erm",
    "sensor_experts_6979__scenario_balanced_cagrad",
)
NATIVE_SENSOR_GROUPS = ("humerus", "radius", "carpus", "digitus")
NATIVE_VIEW_IDS = (
    "full",
    "outage_humerus",
    "outage_radius",
    "outage_carpus",
    "outage_digitus",
)
NATIVE_OUTAGE_VIEW_IDS = NATIVE_VIEW_IDS[1:]
NATIVE_MASK_LAW = (0.25, 0.1875, 0.1875, 0.1875, 0.1875)
CAGRAD_COEFFICIENT = 0.4
CAGRAD_EPSILON = 1.0e-12
EXACT_UPDATES = 400


class OpenAxesExternalNativeV2TrainingError(RuntimeError):
    """Raised when a train-only, prediction-only, or reduction invariant fails."""


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
        raise OpenAxesExternalNativeV2TrainingError(
            "training receipt is not finite canonical JSON"
        ) from exc


def _finite_matrix(
    value: Any,
    *,
    columns: int,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[1] != columns
        or (not allow_empty and array.shape[0] == 0)
        or array.dtype.kind not in "fc"
        or not bool(np.isfinite(array).all())
    ):
        raise OpenAxesExternalNativeV2TrainingError(
            f"{name} must be a finite N x {columns} floating matrix"
        )
    return np.asarray(array, dtype="float64")


def _text_vector(value: Sequence[str], *, rows: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype="object")
    if array.shape != (rows,) or any(
        type(item) is not str or not item for item in array.tolist()
    ):
        raise OpenAxesExternalNativeV2TrainingError(
            f"{name} must contain exactly {rows} nonempty strings"
        )
    return array


def fixed_sum_float64(values: Any, *, name: str = "values") -> float:
    """Adjacent-pair binary-tree sum with odd-tail carry."""

    array = np.asarray(values, dtype="float64")
    if array.size == 0 or not bool(np.isfinite(array).all()):
        raise OpenAxesExternalNativeV2TrainingError(
            f"{name} must be a nonempty finite float64 vector"
        )
    level = [float(item) for item in array.reshape(-1)]
    while len(level) > 1:
        following: list[float] = []
        pair_limit = len(level) - len(level) % 2
        for index in range(0, pair_limit, 2):
            following.append(float(level[index] + level[index + 1]))
        if pair_limit != len(level):
            following.append(level[-1])
        level = following
    result = level[0]
    if not bool(np.isfinite(result)):
        raise OpenAxesExternalNativeV2TrainingError(f"{name} reduction is non-finite")
    return result


def fixed_mean_float64(values: Any, *, name: str = "values") -> float:
    array = np.asarray(values, dtype="float64").reshape(-1)
    if array.size == 0:
        raise OpenAxesExternalNativeV2TrainingError(f"{name} mean is empty")
    return float(fixed_sum_float64(array, name=name) / int(array.size))


def fixed_dot_float64(left: Any, right: Any, *, name: str = "dot") -> float:
    a = np.asarray(left, dtype="float64").reshape(-1)
    b = np.asarray(right, dtype="float64").reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise OpenAxesExternalNativeV2TrainingError(f"{name} operands changed")
    product = np.multiply(a, b, dtype="float64")
    return fixed_sum_float64(product, name=name)


def fixed_norm_float64(value: Any, *, name: str = "norm") -> float:
    squared = fixed_dot_float64(value, value, name=name)
    if squared < 0.0:
        raise OpenAxesExternalNativeV2TrainingError(f"{name} squared norm is negative")
    return float(np.sqrt(squared))


def fixed_rmse3d_m(prediction: Any, truth: Any, *, name: str = "rmse3d") -> float:
    predicted = _finite_matrix(prediction, columns=3, name=f"{name} prediction")
    observed = _finite_matrix(truth, columns=3, name=f"{name} truth")
    if predicted.shape != observed.shape:
        raise OpenAxesExternalNativeV2TrainingError(f"{name} row shape changed")
    squared_rows: list[float] = []
    difference = predicted - observed
    for row in difference:
        squared_rows.append(fixed_dot_float64(row, row, name=f"{name} row"))
    return float(np.sqrt(fixed_mean_float64(squared_rows, name=name)))


def _identity_sha256(values: Sequence[Any]) -> str:
    return hashlib.sha256(_canonical(list(values))).hexdigest()


@dataclass(frozen=True)
class TrainOnlyTransform:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean_m: np.ndarray
    target_scale_m: float
    train_rows: int
    ordered_row_uid_sha256: str
    ordered_scenario_id_sha256: str
    ordered_uid_scenario_pair_sha256: str

    def validate(self) -> None:
        if (
            self.feature_mean.shape != (36,)
            or self.feature_scale.shape != (36,)
            or self.target_mean_m.shape != (3,)
            or not bool(np.isfinite(self.feature_mean).all())
            or not bool(np.isfinite(self.feature_scale).all())
            or not bool(np.isfinite(self.target_mean_m).all())
            or bool((self.feature_scale <= 0.0).any())
            or type(self.target_scale_m) is not float
            or not bool(np.isfinite(self.target_scale_m))
            or self.target_scale_m <= 0.0
            or type(self.train_rows) is not int
            or self.train_rows <= 0
        ):
            raise OpenAxesExternalNativeV2TrainingError(
                "train-only transform is invalid"
            )
        for digest in (
            self.ordered_row_uid_sha256,
            self.ordered_scenario_id_sha256,
            self.ordered_uid_scenario_pair_sha256,
        ):
            if type(digest) is not str or len(digest) != 64:
                raise OpenAxesExternalNativeV2TrainingError(
                    "train-only ordered identity is invalid"
                )

    def receipt(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "target_mean_m": self.target_mean_m.tolist(),
            "target_scale_m": self.target_scale_m,
            "target_scale_isotropic": True,
            "train_rows": self.train_rows,
            "ordered_row_uid_sha256": self.ordered_row_uid_sha256,
            "ordered_scenario_id_sha256": self.ordered_scenario_id_sha256,
            "ordered_uid_scenario_pair_sha256": self.ordered_uid_scenario_pair_sha256,
            "outer_rows_used": 0,
        }
        return {**payload, "sha256": hashlib.sha256(_canonical(payload)).hexdigest()}


@dataclass(frozen=True)
class FeatureOnlyBatch:
    """Outer input type that structurally cannot carry a target column."""

    continuous_features: np.ndarray
    row_uids: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    sample_indices: tuple[int, ...]
    aligned_time_us: tuple[int, ...]

    def validate(self) -> None:
        matrix = _finite_matrix(
            self.continuous_features,
            columns=36,
            name="feature-only batch",
        )
        rows = len(matrix)
        _text_vector(self.row_uids, rows=rows, name="feature-only row_uids")
        _text_vector(self.scenario_ids, rows=rows, name="feature-only scenarios")
        if (
            len(self.sample_indices) != rows
            or len(self.aligned_time_us) != rows
            or any(type(item) is not int for item in self.sample_indices)
            or any(type(item) is not int for item in self.aligned_time_us)
        ):
            raise OpenAxesExternalNativeV2TrainingError(
                "feature-only metadata shape changed"
            )


@dataclass
class TrainingResult:
    candidate_id: str
    seed: int
    transform: TrainOnlyTransform
    state_dict: Mapping[str, Any]
    initial_state_sha256: str
    checkpoint_state_sha256: str
    optimizer_initial_state_sha256: str
    checkpoint_update: int
    checkpoint_score_m: float
    checkpoint_outage_components_m: tuple[float, float, float, float]
    updates: int
    history: tuple[dict[str, Any], ...]
    training_scenario_order: tuple[str, ...]
    training_model_forward_calls: int
    checkpoint_model_forward_calls: int
    parameter_gradient_traversals: int
    cagrad_solver_calls: int


def fit_train_only_transform(
    X_train: Any,
    y_train_m: Any,
    train_uids: Sequence[str],
    training_scenarios: Sequence[str],
) -> TrainOnlyTransform:
    X = _finite_matrix(X_train, columns=36, name="X_train")
    y = _finite_matrix(y_train_m, columns=3, name="y_train_m")
    if len(X) != len(y):
        raise OpenAxesExternalNativeV2TrainingError("training X/y row mismatch")
    uids = _text_vector(train_uids, rows=len(X), name="train_uids")
    scenarios = _text_vector(
        training_scenarios, rows=len(X), name="training_scenarios"
    )
    if len(set(uids.tolist())) != len(uids):
        raise OpenAxesExternalNativeV2TrainingError("training UIDs are not unique")

    feature_mean_values: list[float] = []
    feature_scale_values: list[float] = []
    for column in range(36):
        values = X[:, column]
        mean = fixed_mean_float64(values, name=f"feature[{column}] mean")
        centered = values - mean
        variance = fixed_mean_float64(
            np.multiply(centered, centered, dtype="float64"),
            name=f"feature[{column}] variance",
        )
        scale = float(np.sqrt(variance))
        if not bool(np.isfinite(scale)) or scale <= 0.0:
            raise OpenAxesExternalNativeV2TrainingError(
                "training feature has zero or invalid population scale"
            )
        feature_mean_values.append(mean)
        feature_scale_values.append(scale)

    target_mean_values = [
        fixed_mean_float64(y[:, axis], name=f"target[{axis}] mean")
        for axis in range(3)
    ]
    target_mean = np.asarray(target_mean_values, dtype="float64")
    centered_target = y - target_mean
    flattened_squares = np.multiply(
        centered_target.reshape(-1),
        centered_target.reshape(-1),
        dtype="float64",
    )
    target_scale = float(
        np.sqrt(fixed_mean_float64(flattened_squares, name="target isotropic scale"))
    )
    if not bool(np.isfinite(target_scale)) or target_scale <= 0.0:
        raise OpenAxesExternalNativeV2TrainingError(
            "training target isotropic population scale is invalid"
        )
    uid_list = uids.tolist()
    scenario_list = scenarios.tolist()
    transform = TrainOnlyTransform(
        feature_mean=np.asarray(feature_mean_values, dtype="float64"),
        feature_scale=np.asarray(feature_scale_values, dtype="float64"),
        target_mean_m=target_mean,
        target_scale_m=target_scale,
        train_rows=len(X),
        ordered_row_uid_sha256=_identity_sha256(uid_list),
        ordered_scenario_id_sha256=_identity_sha256(scenario_list),
        ordered_uid_scenario_pair_sha256=_identity_sha256(
            [[uid, scenario] for uid, scenario in zip(uid_list, scenario_list, strict=True)]
        ),
    )
    transform.validate()
    return transform


def transform_features(value: Any, transform: TrainOnlyTransform) -> np.ndarray:
    transform.validate()
    X = _finite_matrix(value, columns=36, name="continuous features")
    result = (X - transform.feature_mean) / transform.feature_scale
    if not bool(np.isfinite(result).all()):
        raise OpenAxesExternalNativeV2TrainingError("standardized features are non-finite")
    return result


def transform_targets(value: Any, transform: TrainOnlyTransform) -> np.ndarray:
    transform.validate()
    y = _finite_matrix(value, columns=3, name="targets")
    result = (y - transform.target_mean_m) / transform.target_scale_m
    if not bool(np.isfinite(result).all()):
        raise OpenAxesExternalNativeV2TrainingError("standardized targets are non-finite")
    return result


def inverse_targets(value: Any, transform: TrainOnlyTransform) -> np.ndarray:
    transform.validate()
    array = np.asarray(value)
    if (
        array.ndim < 2
        or array.shape[-1] != 3
        or array.dtype.kind not in "fc"
        or not bool(np.isfinite(array).all())
    ):
        raise OpenAxesExternalNativeV2TrainingError(
            "scaled prediction must be finite with trailing width three"
        )
    return np.asarray(array, dtype="float64") * transform.target_scale_m + transform.target_mean_m


def build_exact_five_views(X_scaled: Any) -> np.ndarray:
    """Return view-major ``5 x N x 40`` full/outage inputs."""

    X = _finite_matrix(X_scaled, columns=36, name="X_scaled")
    views = np.empty((5, len(X), 40), dtype="float64")
    for view_index in range(5):
        views[view_index, :, :36] = X
        views[view_index, :, 36:] = 1.0
        if view_index > 0:
            sensor_index = view_index - 1
            start = sensor_index * 9
            views[view_index, :, start : start + 9] = 0.0
            views[view_index, :, 36 + sensor_index] = 0.0
    if not bool(np.isfinite(views).all()):
        raise OpenAxesExternalNativeV2TrainingError("five-view tensor is non-finite")
    return views


def validate_exact_five_views(X_scaled: Any, views: Any) -> None:
    expected = build_exact_five_views(X_scaled)
    observed = np.asarray(views)
    if (
        observed.shape != expected.shape
        or observed.dtype.kind not in "fc"
        or not bool(np.array_equal(observed, expected))
    ):
        raise OpenAxesExternalNativeV2TrainingError("exact five views changed")


def compact_scenario_codes(
    training_scenarios: Sequence[str],
    declared_scenario_order: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    scenarios = _text_vector(
        training_scenarios,
        rows=len(training_scenarios),
        name="training scenarios",
    )
    order = tuple(declared_scenario_order)
    if (
        not order
        or any(type(item) is not str or not item for item in order)
        or len(set(order)) != len(order)
        or set(scenarios.tolist()) != set(order)
    ):
        raise OpenAxesExternalNativeV2TrainingError(
            "declared training scenario order changed"
        )
    lookup = {scenario: index for index, scenario in enumerate(order)}
    return np.asarray([lookup[item] for item in scenarios.tolist()], dtype="int64"), order


def build_dense_6979(*, seed: int, device: str = "cpu") -> Any:
    if type(seed) is not int or type(device) is not str or not device:
        raise OpenAxesExternalNativeV2TrainingError("dense seed/device is invalid")
    import torch

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(40, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
        )
    if sum(parameter.numel() for parameter in model.parameters()) != 6979:
        raise OpenAxesExternalNativeV2TrainingError("dense parameter count changed")
    return model.to(device)


def build_sensor_experts_6979(*, seed: int, device: str = "cpu") -> Any:
    if type(seed) is not int or type(device) is not str or not device:
        raise OpenAxesExternalNativeV2TrainingError("expert seed/device is invalid")
    import torch

    class SensorExperts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(9, 20),
                        torch.nn.ReLU(),
                        torch.nn.Linear(20, 24),
                        torch.nn.ReLU(),
                    )
                    for _sensor in NATIVE_SENSOR_GROUPS
                ]
            )
            self.fusion = torch.nn.Linear(100, 40)
            self.output = torch.nn.Linear(40, 3)

        def forward(self, inputs: Any) -> Any:
            if inputs.ndim != 2 or inputs.shape[1] != 40:
                raise OpenAxesExternalNativeV2TrainingError(
                    "sensor-expert input must be N x 40"
                )
            availability = inputs[:, 36:40]
            pieces: list[Any] = []
            for sensor_index, expert in enumerate(self.experts):
                start = sensor_index * 9
                encoded = expert(inputs[:, start : start + 9])
                pieces.append(encoded * availability[:, sensor_index : sensor_index + 1])
            pieces.append(availability)
            fused = torch.cat(pieces, dim=1)
            return self.output(torch.relu(self.fusion(fused)))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SensorExperts()
    if sum(parameter.numel() for parameter in model.parameters()) != 6979:
        raise OpenAxesExternalNativeV2TrainingError("expert parameter count changed")
    return model.to(device)


def build_candidate_model(candidate_id: str, *, seed: int, device: str = "cpu") -> Any:
    if candidate_id == NATIVE_CANDIDATE_IDS[0]:
        return build_dense_6979(seed=seed, device=device)
    if candidate_id in NATIVE_CANDIDATE_IDS[1:]:
        return build_sensor_experts_6979(seed=seed, device=device)
    raise OpenAxesExternalNativeV2TrainingError("candidate is outside the exact grid")


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    names = sorted(state_dict)
    for name in names:
        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            raise OpenAxesExternalNativeV2TrainingError(
                "state dictionary contains a non-tensor"
            )
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def optimizer_initial_state_sha256(optimizer: Any) -> str:
    state = optimizer.state_dict()
    if state.get("state") != {}:
        raise OpenAxesExternalNativeV2TrainingError("optimizer initial state is not empty")
    return hashlib.sha256(_canonical(state)).hexdigest()


def scenario_balanced_view_risks(
    row_losses: Any,
    scenario_codes: Any,
    scenario_count: int,
) -> Any:
    import torch

    if (
        not torch.is_tensor(row_losses)
        or row_losses.ndim != 2
        or row_losses.shape[0] < 1
        or row_losses.shape[1] != 5
        or not bool(torch.isfinite(row_losses).all().item())
        or not torch.is_tensor(scenario_codes)
        or scenario_codes.ndim != 1
        or scenario_codes.shape[0] != row_losses.shape[0]
        or scenario_codes.dtype != torch.int64
        or type(scenario_count) is not int
        or scenario_count < 2
    ):
        raise OpenAxesExternalNativeV2TrainingError(
            "scenario-balanced risk inputs changed"
        )
    rows: list[Any] = []
    for scenario_index in range(scenario_count):
        mask = scenario_codes == scenario_index
        if int(mask.sum().item()) < 1:
            raise OpenAxesExternalNativeV2TrainingError(
                "scenario-balanced risk has an empty scenario"
            )
        rows.append(row_losses[mask].mean(dim=0))
    return torch.stack(rows, dim=0).mean(dim=0)


def direct_mask_law_objective(view_risks: Any) -> Any:
    import torch

    if (
        not torch.is_tensor(view_risks)
        or view_risks.shape != (5,)
        or not bool(torch.isfinite(view_risks).all().item())
    ):
        raise OpenAxesExternalNativeV2TrainingError("five view risks changed")
    law = view_risks.new_tensor(NATIVE_MASK_LAW)
    return torch.sum(law * view_risks)


def solve_cagrad_outage_weights(gradient_rows: Any) -> dict[str, Any]:
    """Perform the one frozen float64 SLSQP solve for four outage gradients."""

    from scipy.optimize import minimize

    gradients = np.asarray(gradient_rows, dtype="float64")
    if (
        gradients.ndim != 2
        or gradients.shape[0] != 4
        or gradients.shape[1] < 1
        or not bool(np.isfinite(gradients).all())
    ):
        raise OpenAxesExternalNativeV2TrainingError(
            "CAGrad gradients must be finite 4 x P"
        )
    gram = np.empty((4, 4), dtype="float64")
    for left in range(4):
        for right in range(4):
            gram[left, right] = fixed_dot_float64(
                gradients[left], gradients[right], name="CAGrad Gram"
            )
    gram = np.asarray((gram + gram.T) * 0.5, dtype="float64")
    uniform = np.asarray([0.25, 0.25, 0.25, 0.25], dtype="float64")
    mean_gradient = np.asarray(
        [
            fixed_mean_float64(gradients[:, column], name="mean outage gradient")
            for column in range(gradients.shape[1])
        ],
        dtype="float64",
    )
    norm_mean = fixed_norm_float64(mean_gradient, name="mean outage gradient")

    def weighted_gradient(weights: Any) -> np.ndarray:
        values = np.asarray(weights, dtype="float64")
        return np.asarray(
            [
                fixed_dot_float64(
                    values,
                    gradients[:, column],
                    name="weighted outage gradient",
                )
                for column in range(gradients.shape[1])
            ],
            dtype="float64",
        )

    def objective(weights: Any) -> float:
        weighted = weighted_gradient(weights)
        return float(
            fixed_dot_float64(weighted, mean_gradient, name="CAGrad objective dot")
            + CAGRAD_COEFFICIENT
            * norm_mean
            * fixed_norm_float64(weighted, name="weighted outage gradient")
        )

    def jacobian(weights: Any) -> np.ndarray:
        values = np.asarray(weights, dtype="float64")
        gram_u = np.asarray(
            [fixed_dot_float64(gram[row], uniform, name="CAGrad G u") for row in range(4)],
            dtype="float64",
        )
        gram_w = np.asarray(
            [fixed_dot_float64(gram[row], values, name="CAGrad G w") for row in range(4)],
            dtype="float64",
        )
        weighted_norm = fixed_norm_float64(
            weighted_gradient(values), name="CAGrad weighted norm"
        )
        if weighted_norm == 0.0:
            return gram_u
        return gram_u + CAGRAD_COEFFICIENT * norm_mean * gram_w / weighted_norm

    def equality(weights: Any) -> float:
        return float(fixed_sum_float64(weights, name="CAGrad equality") - 1.0)

    def equality_jacobian(_weights: Any) -> np.ndarray:
        return np.asarray([1.0, 1.0, 1.0, 1.0], dtype="float64")

    result = minimize(
        objective,
        uniform,
        method="SLSQP",
        jac=jacobian,
        bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        constraints={"type": "eq", "fun": equality, "jac": equality_jacobian},
        options={"maxiter": 100, "ftol": 1.0e-12, "disp": False},
    )
    weights = np.asarray(result["x"], dtype="float64")
    objective_value = float(result["fun"])
    residual = abs(equality(weights))
    lower_violation = max(0.0, -min(float(item) for item in weights))
    upper_violation = max(0.0, max(float(item) for item in weights) - 1.0)
    if (
        result["success"] is not True
        or weights.shape != (4,)
        or not bool(np.isfinite(weights).all())
        or not bool(np.isfinite(objective_value))
        or not bool(np.isfinite(jacobian(weights)).all())
        or residual > 1.0e-10
        or max(lower_violation, upper_violation) > 1.0e-10
    ):
        raise OpenAxesExternalNativeV2TrainingError(
            "frozen CAGrad SLSQP solve failed closed"
        )
    weighted = weighted_gradient(weights)
    weighted_norm = fixed_norm_float64(weighted, name="CAGrad weighted norm")
    direction = mean_gradient + (
        CAGRAD_COEFFICIENT
        * norm_mean
        * weighted
        / (weighted_norm + CAGRAD_EPSILON)
    )
    if not bool(np.isfinite(direction).all()):
        raise OpenAxesExternalNativeV2TrainingError("CAGrad direction is non-finite")
    return {
        "weights": weights,
        "gram": gram,
        "mean_gradient": mean_gradient,
        "weighted_gradient": weighted,
        "outage_direction": direction,
        "objective": objective_value,
        "equality_residual": residual,
        "bound_violation": max(lower_violation, upper_violation),
        "mean_gradient_norm": norm_mean,
        "weighted_gradient_norm": weighted_norm,
        "outage_direction_norm": fixed_norm_float64(
            direction, name="CAGrad outage direction"
        ),
        "iterations": int(result["nit"]),
    }


def assemble_cagrad_full_direction(full_gradient: Any, outage_rows: Any) -> dict[str, Any]:
    full = np.asarray(full_gradient, dtype="float64").reshape(-1)
    outages = np.asarray(outage_rows, dtype="float64")
    if outages.ndim != 2 or outages.shape != (4, len(full)):
        raise OpenAxesExternalNativeV2TrainingError("CAGrad full/outage shapes changed")
    solved = solve_cagrad_outage_weights(outages)
    direction = 0.25 * full + 0.75 * solved["outage_direction"]
    if not bool(np.isfinite(direction).all()):
        raise OpenAxesExternalNativeV2TrainingError("assembled CAGrad direction is non-finite")
    return {**solved, "final_direction": np.asarray(direction, dtype="float64")}


def training_checkpoint_score_m(
    predictions_scaled: Any,
    targets_scaled: Any,
    scenario_codes: Any,
    scenario_count: int,
    target_scale_m: float,
) -> tuple[float, tuple[float, float, float, float], tuple[float, ...]]:
    predictions = np.asarray(predictions_scaled, dtype="float64")
    targets = _finite_matrix(targets_scaled, columns=3, name="checkpoint targets")
    codes = np.asarray(scenario_codes, dtype="int64")
    if (
        predictions.shape != (5, len(targets), 3)
        or not bool(np.isfinite(predictions).all())
        or codes.shape != (len(targets),)
        or type(scenario_count) is not int
        or scenario_count < 2
        or type(target_scale_m) is not float
        or not bool(np.isfinite(target_scale_m))
        or target_scale_m <= 0.0
    ):
        raise OpenAxesExternalNativeV2TrainingError("checkpoint inputs changed")
    view_components: list[float] = []
    for view_index in range(5):
        values: list[float] = []
        for scenario_index in range(scenario_count):
            mask = codes == scenario_index
            if int(mask.sum()) < 1:
                raise OpenAxesExternalNativeV2TrainingError(
                    "checkpoint scenario is empty"
                )
            rmse = target_scale_m * fixed_rmse3d_m(
                predictions[view_index, mask, :],
                targets[mask],
                name="training scenario view",
            )
            values.append(float(rmse))
        view_components.append(
            fixed_mean_float64(values, name="checkpoint scenario mean")
        )
    outages = tuple(view_components[1:])
    score = 0.5 * (
        fixed_mean_float64(outages, name="checkpoint outage mean") + max(outages)
    )
    if not bool(np.isfinite(score)):
        raise OpenAxesExternalNativeV2TrainingError("checkpoint score is non-finite")
    return float(score), outages, tuple(view_components)


def _flat_gradients(gradient_tuples: Sequence[Sequence[Any]]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for gradients in gradient_tuples:
        pieces: list[np.ndarray] = []
        for gradient in gradients:
            if gradient is None:
                raise OpenAxesExternalNativeV2TrainingError("CAGrad task gradient is missing")
            piece = gradient.detach().cpu().contiguous().numpy().reshape(-1)
            if piece.dtype.name != "float32" or not bool(np.isfinite(piece).all()):
                raise OpenAxesExternalNativeV2TrainingError(
                    "CAGrad task gradient is not finite float32"
                )
            pieces.append(piece)
        rows.append(np.concatenate(pieces).astype("float64", copy=False))
    result = np.stack(rows, axis=0)
    if result.shape[0] != 5:
        raise OpenAxesExternalNativeV2TrainingError("CAGrad requires exact five gradients")
    return result


def train_candidate_train_only(
    X_train: Any,
    y_train_m: Any,
    training_scenarios: Sequence[str],
    train_uids: Sequence[str],
    declared_scenario_order: Sequence[str],
    *,
    candidate_id: str,
    seed: int,
    updates: int = EXACT_UPDATES,
    device: str = "cuda:0",
    expected_dense_initial_state_sha256: str | None = None,
    expected_expert_initial_state_sha256: str | None = None,
) -> TrainingResult:
    """Fit exactly one candidate without accepting an outer partition."""

    if candidate_id not in NATIVE_CANDIDATE_IDS:
        raise OpenAxesExternalNativeV2TrainingError("candidate is outside exact three")
    if type(seed) is not int or type(updates) is not int or updates != EXACT_UPDATES:
        raise OpenAxesExternalNativeV2TrainingError("fit must use the exact seed/400 updates")
    if type(device) is not str or not device:
        raise OpenAxesExternalNativeV2TrainingError("training device is invalid")
    X = _finite_matrix(X_train, columns=36, name="X_train")
    y = _finite_matrix(y_train_m, columns=3, name="y_train_m")
    scenarios = _text_vector(
        training_scenarios, rows=len(X), name="training scenarios"
    )
    uids = _text_vector(train_uids, rows=len(X), name="training UIDs")
    if len(y) != len(X):
        raise OpenAxesExternalNativeV2TrainingError("training X/y row mismatch")
    codes_numpy, scenario_order = compact_scenario_codes(
        scenarios.tolist(), declared_scenario_order
    )
    transform = fit_train_only_transform(X, y, uids.tolist(), scenarios.tolist())
    X_scaled = transform_features(X, transform)
    y_scaled = transform_targets(y, transform)
    views = build_exact_five_views(X_scaled)
    validate_exact_five_views(X_scaled, views)

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    model = build_candidate_model(candidate_id, seed=seed, device=device)
    initial_hash = state_dict_sha256(model.state_dict())
    if (
        candidate_id == NATIVE_CANDIDATE_IDS[0]
        and expected_dense_initial_state_sha256 is not None
        and initial_hash != expected_dense_initial_state_sha256
    ):
        raise OpenAxesExternalNativeV2TrainingError(
            "dense initial state differs from receipt-bound V1 masked_erm"
        )
    if (
        candidate_id in NATIVE_CANDIDATE_IDS[1:]
        and expected_expert_initial_state_sha256 is not None
        and initial_hash != expected_expert_initial_state_sha256
    ):
        raise OpenAxesExternalNativeV2TrainingError(
            "paired expert initial state differs"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.001,
        amsgrad=False,
        maximize=False,
        foreach=False,
        fused=False,
        capturable=False,
        differentiable=False,
    )
    optimizer_hash = optimizer_initial_state_sha256(optimizer)
    tensor_views = torch.as_tensor(
        views.astype("float32", copy=False), device=device
    )
    tensor_targets = torch.as_tensor(
        y_scaled.astype("float32", copy=False), device=device
    )
    tensor_codes = torch.as_tensor(codes_numpy, device=device)
    parameters = tuple(model.parameters())
    best_score = float("inf")
    best_update = 0
    best_state: dict[str, Any] | None = None
    best_components: tuple[float, float, float, float] | None = None
    history: list[dict[str, Any]] = []
    gradient_traversals = 0
    solver_calls = 0

    cudnn_scope = torch.backends.cudnn.flags(
        enabled=True,
        benchmark=False,
        deterministic=True,
        allow_tf32=False,
    )
    with cudnn_scope:
        for update in range(1, updates + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tensor_views.reshape(-1, 40)).reshape(5, len(X), 3)
            row_losses = torch.mean(
                torch.square(prediction - tensor_targets[None, :, :]), dim=2
            ).transpose(0, 1)
            view_risks = scenario_balanced_view_risks(
                row_losses, tensor_codes, len(scenario_order)
            )
            cagrad_receipt: dict[str, Any] | None = None
            if candidate_id != NATIVE_CANDIDATE_IDS[2]:
                objective = direct_mask_law_objective(view_risks)
                objective.backward()
                gradient_traversals += 1
            else:
                gradient_sets: list[tuple[Any, ...]] = []
                for view_index in range(5):
                    gradients = torch.autograd.grad(
                        view_risks[view_index],
                        parameters,
                        retain_graph=view_index < 4,
                        create_graph=False,
                        allow_unused=False,
                    )
                    gradient_sets.append(tuple(gradients))
                gradient_traversals += 5
                flattened = _flat_gradients(gradient_sets)
                cagrad_receipt = assemble_cagrad_full_direction(
                    flattened[0], flattened[1:]
                )
                solver_calls += 1
                offset = 0
                final = cagrad_receipt["final_direction"]
                for parameter in parameters:
                    count = int(parameter.numel())
                    piece = final[offset : offset + count]
                    parameter.grad = torch.as_tensor(
                        piece.astype("float32", copy=False).reshape(tuple(parameter.shape)),
                        device=device,
                    )
                    offset += count
                if offset != len(final):
                    raise OpenAxesExternalNativeV2TrainingError(
                        "assembled gradient parameter partition changed"
                    )
                objective = direct_mask_law_objective(view_risks).detach()
            optimizer.step()

            model.train(False)
            with torch.no_grad():
                checkpoint_prediction = (
                    model(tensor_views.reshape(-1, 40))
                    .reshape(5, len(X), 3)
                    .detach()
                    .cpu()
                    .numpy()
                )
            score, outage_components, view_components = training_checkpoint_score_m(
                checkpoint_prediction,
                y_scaled,
                codes_numpy,
                len(scenario_order),
                transform.target_scale_m,
            )
            became_checkpoint = score < best_score
            if became_checkpoint:
                best_score = score
                best_update = update
                best_components = outage_components
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            row = {
                "update": update,
                "objective": float(objective.detach().cpu()),
                "training_full_component_m": float(view_components[0]),
                "training_outage_components_m": list(outage_components),
                "checkpoint_score_m": score,
                "became_checkpoint": became_checkpoint,
            }
            if cagrad_receipt is not None:
                row["cagrad"] = {
                    "weights": cagrad_receipt["weights"].tolist(),
                    "objective": cagrad_receipt["objective"],
                    "equality_residual": cagrad_receipt["equality_residual"],
                    "bound_violation": cagrad_receipt["bound_violation"],
                    "mean_gradient_norm": cagrad_receipt["mean_gradient_norm"],
                    "weighted_gradient_norm": cagrad_receipt["weighted_gradient_norm"],
                    "outage_direction_norm": cagrad_receipt["outage_direction_norm"],
                    "iterations": cagrad_receipt["iterations"],
                }
            history.append(row)
    if best_state is None or best_components is None or best_update < 1:
        raise OpenAxesExternalNativeV2TrainingError("training produced no checkpoint")
    expected_traversals = updates * (5 if candidate_id == NATIVE_CANDIDATE_IDS[2] else 1)
    expected_solves = updates if candidate_id == NATIVE_CANDIDATE_IDS[2] else 0
    if gradient_traversals != expected_traversals or solver_calls != expected_solves:
        raise OpenAxesExternalNativeV2TrainingError("fit accounting changed")
    return TrainingResult(
        candidate_id=candidate_id,
        seed=seed,
        transform=transform,
        state_dict=best_state,
        initial_state_sha256=initial_hash,
        checkpoint_state_sha256=state_dict_sha256(best_state),
        optimizer_initial_state_sha256=optimizer_hash,
        checkpoint_update=best_update,
        checkpoint_score_m=best_score,
        checkpoint_outage_components_m=best_components,
        updates=updates,
        history=tuple(history),
        training_scenario_order=scenario_order,
        training_model_forward_calls=updates,
        checkpoint_model_forward_calls=updates,
        parameter_gradient_traversals=gradient_traversals,
        cagrad_solver_calls=solver_calls,
    )


def predict_feature_only_batch_once(
    *,
    candidate_id: str,
    seed: int,
    state_dict: Mapping[str, Any],
    transform: TrainOnlyTransform,
    feature_batch: FeatureOnlyBatch,
    device: str = "cuda:0",
) -> np.ndarray:
    """Run one locked checkpoint on all five views of a feature-only batch."""

    feature_batch.validate()
    transform.validate()
    scaled = transform_features(feature_batch.continuous_features, transform)
    views = build_exact_five_views(scaled)
    import torch

    model = build_candidate_model(candidate_id, seed=seed, device=device)
    model.load_state_dict(state_dict, strict=True)
    model.train(False)
    with torch.no_grad():
        tensor = torch.as_tensor(views.astype("float32", copy=False), device=device)
        scaled_prediction = (
            model(tensor.reshape(-1, 40))
            .reshape(5, len(feature_batch.row_uids), 3)
            .detach()
            .cpu()
            .numpy()
        )
    prediction = inverse_targets(scaled_prediction, transform)
    if prediction.shape != (5, len(feature_batch.row_uids), 3):
        raise OpenAxesExternalNativeV2TrainingError("outer prediction shape changed")
    return prediction
