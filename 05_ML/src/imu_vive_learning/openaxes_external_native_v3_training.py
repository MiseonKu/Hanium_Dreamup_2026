"""Deterministic train-only core for OpenAXES native V3 all-mask DMoME.

The public trainer accepts one authenticated training fold and never accepts
outer labels.  Torch and SciPy stay lazy imports so a fit child can bind its
GPU environment before importing either runtime.  The implementation freezes
the 15-view lattice, 28 adjacent More-vs-Fewer edges, three parameter-matched
architectures, the 100+300 two-stage path, and four-task CAGrad arithmetic.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np

from . import openaxes_external_native_v2_training as v2


OpenAxesExternalNativeV3TrainingError = v2.OpenAxesExternalNativeV2TrainingError
TrainOnlyTransform = v2.TrainOnlyTransform
FeatureOnlyBatch = v2.FeatureOnlyBatch
fixed_sum_float64 = v2.fixed_sum_float64
fixed_mean_float64 = v2.fixed_mean_float64
fixed_dot_float64 = v2.fixed_dot_float64
fixed_norm_float64 = v2.fixed_norm_float64
fixed_rmse3d_m = v2.fixed_rmse3d_m
fit_train_only_transform = v2.fit_train_only_transform
transform_features = v2.transform_features
transform_targets = v2.transform_targets
inverse_targets = v2.inverse_targets
compact_scenario_codes = v2.compact_scenario_codes
state_dict_sha256 = v2.state_dict_sha256
optimizer_initial_state_sha256 = v2.optimizer_initial_state_sha256


OPENAXES_NATIVE_V3_CANDIDATE_IDS = (
    "dense_6979__allmask_cardinality_balanced_erm",
    "static_sensor_experts_6979__allmask_cardinality_balanced_erm",
    "dmome_6979__one_stage_allmask_cardinality_balanced_erm",
    "dmome_6979__two_stage_allmask_cardinality_balanced_erm",
    "dmome_6979__two_stage_allmask_cardinality_balanced_mofe",
    "dmome_6979__two_stage_allmask_cardinality_balanced_mofe_cagrad",
)
_SEEDS = (1701, 2603, 3907)
_SENSORS = ("humerus", "radius", "carpus", "digitus")
_VIEW_IDS = (
    "full",
    "drop_humerus",
    "drop_radius",
    "drop_carpus",
    "drop_digitus",
    "only_humerus_radius",
    "only_humerus_carpus",
    "only_humerus_digitus",
    "only_radius_carpus",
    "only_radius_digitus",
    "only_carpus_digitus",
    "only_humerus",
    "only_radius",
    "only_carpus",
    "only_digitus",
)
_MASKS = (
    (1, 1, 1, 1),
    (0, 1, 1, 1),
    (1, 0, 1, 1),
    (1, 1, 0, 1),
    (1, 1, 1, 0),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (0, 1, 0, 1),
    (0, 0, 1, 1),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
)
_CARDINALITIES = tuple(sum(mask) for mask in _MASKS)
_CARDINALITY_ORDER = (1, 2, 3, 4)
_MASK_WEIGHT_BY_CARDINALITY = {1: 1.0 / 16.0, 2: 1.0 / 24.0, 3: 1.0 / 16.0, 4: 1.0 / 4.0}
EXACT_UPDATES = 400
STAGE_ONE_UPDATES = 100
MOFE_COEFFICIENT = 0.1
CAGRAD_COEFFICIENT = 0.4
CAGRAD_EPSILON = 1.0e-12
_EXPERIMENT_TRAIN_ONLY_FIELDS = (
    "X_train",
    "y_train_m",
    "training_scenarios",
    "train_uids",
    "declared_scenario_order",
    "candidate_id",
    "seed",
)


def _fail(message: str) -> None:
    raise OpenAxesExternalNativeV3TrainingError(message)


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
        raise OpenAxesExternalNativeV3TrainingError(
            "V3 training authority is not finite canonical JSON"
        ) from exc


def _canonical_bytes(value: Any) -> bytes:
    return _canonical(value) + b"\n"


def _reference(path: str, raw: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def candidate_schedule(candidate_id: str) -> dict[str, Any]:
    if candidate_id not in OPENAXES_NATIVE_V3_CANDIDATE_IDS:
        _fail("candidate is outside the exact V3 grid")
    ordinal = OPENAXES_NATIVE_V3_CANDIDATE_IDS.index(candidate_id)
    architecture = "dense" if ordinal == 0 else "static_sensor_experts" if ordinal == 1 else "dmome_output_space"
    stage_one = STAGE_ONE_UPDATES if ordinal >= 3 else 0
    return {
        "candidate_id": candidate_id,
        "candidate_ordinal": ordinal,
        "architecture": architecture,
        "independent_expert_updates": stage_one,
        "cooperative_updates": EXACT_UPDATES - stage_one,
        "mofe": ordinal >= 4,
        "cagrad": ordinal == 5,
    }


def adjacent_mask_edges() -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for rich_index, rich in enumerate(_MASKS):
        rich_cardinality = _CARDINALITIES[rich_index]
        for poor_index, poor in enumerate(_MASKS):
            if (
                _CARDINALITIES[poor_index] == rich_cardinality - 1
                and all(poor[item] <= rich[item] for item in range(4))
                and sum(rich[item] - poor[item] for item in range(4)) == 1
            ):
                edges.append((rich_index, poor_index))
    if len(edges) != 28:
        _fail("adjacent availability lattice changed")
    return tuple(edges)


_ADJACENT_EDGES = adjacent_mask_edges()


def build_exact_fifteen_views(X_scaled: Any) -> np.ndarray:
    X = v2._finite_matrix(X_scaled, columns=36, name="V3 X_scaled")
    result = np.empty((15, len(X), 40), dtype="float64")
    for view_index, mask in enumerate(_MASKS):
        result[view_index, :, :36] = X
        result[view_index, :, 36:] = np.asarray(mask, dtype="float64")
        for sensor_index, available in enumerate(mask):
            if not available:
                start = sensor_index * 9
                result[view_index, :, start : start + 9] = 0.0
    if not bool(np.isfinite(result).all()):
        _fail("15-view tensor is non-finite")
    return result


def validate_exact_fifteen_views(X_scaled: Any, views: Any) -> None:
    expected = build_exact_fifteen_views(X_scaled)
    observed = np.asarray(views)
    if (
        observed.shape != expected.shape
        or observed.dtype.kind not in "fc"
        or not bool(np.array_equal(observed, expected))
    ):
        _fail("exact 15-view lattice changed")


def build_dense_6979(*, seed: int, device: str = "cpu") -> Any:
    return v2.build_dense_6979(seed=seed, device=device)


def build_static_sensor_experts_6979(*, seed: int, device: str = "cpu") -> Any:
    return v2.build_sensor_experts_6979(seed=seed, device=device)


def build_dmome_6979(*, seed: int, device: str = "cpu") -> Any:
    if type(seed) is not int or type(device) is not str or not device:
        _fail("DMoME seed/device is invalid")
    import torch

    class DMoME(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(9, 27),
                        torch.nn.ReLU(),
                        torch.nn.Linear(27, 42),
                        torch.nn.ReLU(),
                        torch.nn.Linear(42, 3),
                    )
                    for _sensor in _SENSORS
                ]
            )
            self.router = torch.nn.Sequential(
                torch.nn.Linear(40, 15),
                torch.nn.ReLU(),
                torch.nn.Linear(15, 4),
            )

        def expert_outputs(self, inputs: Any) -> Any:
            if inputs.ndim != 2 or inputs.shape[1] != 40:
                _fail("DMoME input must be N x 40")
            pieces = [
                expert(inputs[:, index * 9 : (index + 1) * 9])
                for index, expert in enumerate(self.experts)
            ]
            return torch.stack(pieces, dim=1)

        def router_weights(self, inputs: Any) -> Any:
            if inputs.ndim != 2 or inputs.shape[1] != 40:
                _fail("DMoME router input must be N x 40")
            availability = inputs[:, 36:40]
            if bool(torch.any(torch.sum(availability, dim=1) < 1).item()):
                _fail("DMoME received the forbidden empty mask")
            logits = self.router(inputs)
            masked_logits = logits.masked_fill(availability == 0, float("-inf"))
            weights = torch.softmax(masked_logits, dim=1)
            if not bool(torch.isfinite(weights).all().item()):
                _fail("DMoME router weights are non-finite")
            return weights

        def forward(self, inputs: Any) -> Any:
            outputs = self.expert_outputs(inputs)
            weights = self.router_weights(inputs)
            return torch.sum(outputs * weights[:, :, None], dim=1)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = DMoME()
    if sum(parameter.numel() for parameter in model.parameters()) != 6979:
        _fail("DMoME parameter count changed")
    return model.to(device)


def build_candidate_model(candidate_id: str, *, seed: int, device: str = "cpu") -> Any:
    schedule = candidate_schedule(candidate_id)
    if schedule["architecture"] == "dense":
        return build_dense_6979(seed=seed, device=device)
    if schedule["architecture"] == "static_sensor_experts":
        return build_static_sensor_experts_6979(seed=seed, device=device)
    return build_dmome_6979(seed=seed, device=device)


def _torch_fixed_mean(values: Sequence[Any], *, name: str) -> Any:
    if not values:
        _fail(f"{name} is empty")
    level = list(values)
    while len(level) > 1:
        following: list[Any] = []
        pair_limit = len(level) - len(level) % 2
        for index in range(0, pair_limit, 2):
            following.append(level[index] + level[index + 1])
        if pair_limit != len(level):
            following.append(level[-1])
        level = following
    return level[0] / len(values)


def row_task_losses(predictions: Any, targets: Any) -> Any:
    import torch

    if (
        not torch.is_tensor(predictions)
        or predictions.ndim != 3
        or predictions.shape[0] != 15
        or predictions.shape[2] != 3
        or not torch.is_tensor(targets)
        or tuple(targets.shape) != (predictions.shape[1], 3)
    ):
        _fail("V3 row-loss inputs changed")
    difference = predictions.to(torch.float64) - targets[None, :, :].to(torch.float64)
    squared = torch.square(difference)
    return (squared[:, :, 0] + squared[:, :, 1] + squared[:, :, 2]) / 3.0


def _scenario_mean(values: Any, scenario_codes: Any, scenario_count: int) -> Any:
    rows: list[Any] = []
    for scenario_index in range(scenario_count):
        selected = values[scenario_codes == scenario_index]
        if selected.numel() < 1:
            _fail("training scenario is empty")
        rows.append(_torch_fixed_mean(tuple(selected.unbind()), name="scenario rows"))
    return _torch_fixed_mean(rows, name="scenario risks")


def cardinality_balanced_risks(
    losses_view_row: Any,
    scenario_codes: Any,
    scenario_count: int,
) -> Any:
    import torch

    if (
        not torch.is_tensor(losses_view_row)
        or losses_view_row.ndim != 2
        or losses_view_row.shape[0] != 15
        or losses_view_row.shape[1] < 1
        or not bool(torch.isfinite(losses_view_row).all().item())
        or not torch.is_tensor(scenario_codes)
        or scenario_codes.dtype != torch.int64
        or tuple(scenario_codes.shape) != (losses_view_row.shape[1],)
        or type(scenario_count) is not int
        or scenario_count < 2
    ):
        _fail("cardinality-risk inputs changed")
    risks: list[Any] = []
    for cardinality in _CARDINALITY_ORDER:
        mask_risks = [
            _scenario_mean(losses_view_row[index], scenario_codes, scenario_count)
            for index, observed in enumerate(_CARDINALITIES)
            if observed == cardinality
        ]
        risks.append(_torch_fixed_mean(mask_risks, name="mask risks"))
    return torch.stack(risks)


def direct_cardinality_objective(cardinality_risks: Any) -> Any:
    import torch

    if (
        not torch.is_tensor(cardinality_risks)
        or tuple(cardinality_risks.shape) != (4,)
        or not bool(torch.isfinite(cardinality_risks).all().item())
    ):
        _fail("cardinality objective inputs changed")
    return _torch_fixed_mean(tuple(cardinality_risks.unbind()), name="cardinality risks")


def more_vs_fewer_loss(
    losses_view_row: Any,
    scenario_codes: Any,
    scenario_count: int,
) -> Any:
    import torch

    edge_risks: list[Any] = []
    for rich, poor in _ADJACENT_EDGES:
        atoms = torch.relu(losses_view_row[rich] - losses_view_row[poor])
        edge_risks.append(_scenario_mean(atoms, scenario_codes, scenario_count))
    if len(edge_risks) != 28:
        _fail("MoFe edge count changed")
    return _torch_fixed_mean(edge_risks, name="MoFe edges")


def solve_cagrad_weights(gradient_rows: Any) -> dict[str, Any]:
    gradients = np.asarray(gradient_rows, dtype="float64")
    if (
        gradients.ndim != 2
        or gradients.shape[0] != 4
        or gradients.shape[1] < 1
        or not bool(np.isfinite(gradients).all())
    ):
        _fail("CAGrad requires a finite 4 x P gradient matrix")
    gram = np.empty((4, 4), dtype="float64")
    for left in range(4):
        for right in range(4):
            gram[left, right] = fixed_dot_float64(
                gradients[left], gradients[right], name="CAGrad Gram"
            )
    mean_gradient = np.asarray(
        [fixed_mean_float64(gradients[:, index], name="CAGrad mean gradient") for index in range(gradients.shape[1])],
        dtype="float64",
    )
    mean_norm = fixed_norm_float64(mean_gradient, name="CAGrad mean norm")

    def objective(weights: Any) -> float:
        vector = np.asarray(weights, dtype="float64")
        weighted = np.asarray(
            [fixed_dot_float64(vector, gradients[:, index], name="CAGrad weighted gradient") for index in range(gradients.shape[1])],
            dtype="float64",
        )
        return float(
            fixed_dot_float64(weighted, mean_gradient, name="CAGrad dual dot")
            + CAGRAD_COEFFICIENT
            * mean_norm
            * fixed_norm_float64(weighted, name="CAGrad weighted norm")
        )

    from scipy.optimize import minimize

    solved = minimize(
        objective,
        np.full(4, 0.25, dtype="float64"),
        method="SLSQP",
        bounds=((0.0, 1.0),) * 4,
        constraints=({"type": "eq", "fun": lambda value: float(np.sum(value) - 1.0)},),
        jac=None,
        tol=None,
        callback=None,
        options={"maxiter": 100, "ftol": 1.0e-12, "disp": False},
    )
    weights = np.asarray(solved.x, dtype="float64")
    equality_residual = abs(fixed_sum_float64(weights, name="CAGrad weights") - 1.0)
    bound_violation = max(float(np.max(-weights)), float(np.max(weights - 1.0)), 0.0)
    if (
        solved.success is not True
        or weights.shape != (4,)
        or not bool(np.isfinite(weights).all())
        or not bool(np.isfinite(float(solved.fun)))
        or equality_residual > 1.0e-10
        or bound_violation > 1.0e-10
    ):
        _fail("CAGrad SLSQP solve failed closed")
    weighted_gradient = np.asarray(
        [fixed_dot_float64(weights, gradients[:, index], name="CAGrad weighted gradient") for index in range(gradients.shape[1])],
        dtype="float64",
    )
    weighted_norm = fixed_norm_float64(weighted_gradient, name="CAGrad weighted norm")
    direction = mean_gradient + (
        CAGRAD_COEFFICIENT
        * mean_norm
        * weighted_gradient
        / (weighted_norm + CAGRAD_EPSILON)
    )
    if not bool(np.isfinite(direction).all()):
        _fail("CAGrad direction is non-finite")
    return {
        "weights": weights,
        "direction": direction,
        "objective": float(solved.fun),
        "iterations": int(solved.nit),
        "equality_residual": equality_residual,
        "bound_violation": bound_violation,
        "mean_gradient_norm": mean_norm,
        "weighted_gradient_norm": weighted_norm,
    }


def _flat_gradient_rows(gradient_sets: Sequence[Sequence[Any]]) -> np.ndarray:
    rows: list[np.ndarray] = []
    if len(gradient_sets) != 4:
        _fail("CAGrad task count changed")
    for gradients in gradient_sets:
        pieces: list[np.ndarray] = []
        for gradient in gradients:
            if gradient is None:
                _fail("CAGrad parameter gradient is missing")
            piece = gradient.detach().cpu().contiguous().numpy().reshape(-1)
            if piece.dtype.name != "float32" or not bool(np.isfinite(piece).all()):
                _fail("CAGrad parameter gradient changed dtype or finiteness")
            pieces.append(piece)
        rows.append(np.concatenate(pieces).astype("float64", copy=False))
    return np.stack(rows, axis=0)


def optimizer_state_sha256(optimizer: Any) -> str:
    import torch

    digest = hashlib.sha256()

    def visit(value: Any) -> None:
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(_canonical(list(array.shape)))
            digest.update(array.tobytes(order="C"))
        elif isinstance(value, Mapping):
            digest.update(b"map")
            for key in sorted(value, key=lambda item: str(item)):
                digest.update(str(key).encode("utf-8"))
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"seq")
            for item in value:
                visit(item)
        elif value is None or type(value) in {bool, int, float, str}:
            digest.update(_canonical(value))
        else:
            _fail("optimizer state contains an unsupported value")

    visit(optimizer.state_dict())
    return digest.hexdigest()


def training_checkpoint_score_m(
    predictions_scaled: Any,
    targets_scaled: Any,
    scenario_codes: Any,
    scenario_count: int,
    target_scale_m: float,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    predictions = np.asarray(predictions_scaled, dtype="float64")
    targets = v2._finite_matrix(targets_scaled, columns=3, name="V3 checkpoint targets")
    codes = np.asarray(scenario_codes, dtype="int64")
    if (
        predictions.shape != (15, len(targets), 3)
        or not bool(np.isfinite(predictions).all())
        or codes.shape != (len(targets),)
        or type(scenario_count) is not int
        or scenario_count < 2
        or type(target_scale_m) is not float
        or not bool(np.isfinite(target_scale_m))
        or target_scale_m <= 0.0
    ):
        _fail("V3 checkpoint inputs changed")
    components: list[float] = []
    for view_index in range(15):
        scenario_values: list[float] = []
        for scenario_index in range(scenario_count):
            selected = codes == scenario_index
            if int(selected.sum()) < 1:
                _fail("V3 checkpoint scenario is empty")
            scenario_values.append(
                target_scale_m
                * fixed_rmse3d_m(
                    predictions[view_index, selected],
                    targets[selected],
                    name="V3 checkpoint view",
                )
            )
        components.append(
            fixed_mean_float64(scenario_values, name="V3 checkpoint scenarios")
        )
    missing = tuple(components[1:])
    score = 0.5 * (fixed_mean_float64(missing, name="V3 missing-mask mean") + max(missing))
    if not bool(np.isfinite(score)):
        _fail("V3 checkpoint score is non-finite")
    return float(score), missing, tuple(components)


@dataclass
class TrainingResult:
    candidate_id: str
    seed: int
    transform: TrainOnlyTransform
    state_dict: Mapping[str, Any]
    initial_state_sha256: str
    checkpoint_state_sha256: str
    optimizer_initial_state_sha256: str
    stage_one_model_state_sha256: str | None
    stage_one_optimizer_state_sha256: str | None
    checkpoint_update: int
    checkpoint_score_m: float
    checkpoint_missing_components_m: tuple[float, ...]
    updates: int
    history: tuple[dict[str, Any], ...]
    training_scenario_order: tuple[str, ...]
    training_model_forward_calls: int
    checkpoint_model_forward_calls: int
    parameter_gradient_traversals: int
    cagrad_solver_calls: int


def _set_router_trainable(model: Any, value: bool) -> None:
    if not hasattr(model, "router"):
        _fail("two-stage candidate does not expose a router")
    for parameter in model.router.parameters():
        parameter.requires_grad_(value)


def _stage_one_objective(model: Any, inputs: Any, targets: Any, codes: Any, scenario_count: int) -> Any:
    import torch

    outputs = model.expert_outputs(inputs)
    expert_risks: list[Any] = []
    for sensor_index in range(4):
        difference = outputs[:, sensor_index, :].to(torch.float64) - targets.to(torch.float64)
        losses = (torch.square(difference[:, 0]) + torch.square(difference[:, 1]) + torch.square(difference[:, 2])) / 3.0
        expert_risks.append(_scenario_mean(losses, codes, scenario_count))
    return _torch_fixed_mean(expert_risks, name="stage-one expert risks")


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
    expected_dmome_initial_state_sha256: str | None = None,
    expected_dmome_optimizer_initial_state_sha256: str | None = None,
) -> TrainingResult:
    schedule = candidate_schedule(candidate_id)
    if type(seed) is not int or seed not in _SEEDS:
        _fail("fit seed is outside the exact V3 seeds")
    if type(updates) is not int or updates != EXACT_UPDATES:
        _fail("V3 fit must use exactly 400 updates")
    if schedule["independent_expert_updates"] not in {0, STAGE_ONE_UPDATES}:
        _fail("V3 stage schedule changed")
    X = v2._finite_matrix(X_train, columns=36, name="V3 X_train")
    y = v2._finite_matrix(y_train_m, columns=3, name="V3 y_train_m")
    scenarios = v2._text_vector(training_scenarios, rows=len(X), name="V3 training scenarios")
    uids = v2._text_vector(train_uids, rows=len(X), name="V3 training UIDs")
    if len(y) != len(X):
        _fail("V3 training X/y row mismatch")
    codes_numpy, scenario_order = compact_scenario_codes(
        scenarios.tolist(), declared_scenario_order
    )
    transform = fit_train_only_transform(X, y, uids.tolist(), scenarios.tolist())
    X_scaled = transform_features(X, transform)
    y_scaled = transform_targets(y, transform)
    views = build_exact_fifteen_views(X_scaled)
    validate_exact_fifteen_views(X_scaled, views)

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
    if schedule["architecture"] == "dmome_output_space" and expected_dmome_initial_state_sha256 is not None and initial_hash != expected_dmome_initial_state_sha256:
        _fail("paired DMoME initial state changed")
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
    if schedule["architecture"] == "dmome_output_space" and expected_dmome_optimizer_initial_state_sha256 is not None and optimizer_hash != expected_dmome_optimizer_initial_state_sha256:
        _fail("paired DMoME optimizer initial state changed")
    tensor_views = torch.as_tensor(views.astype("float32", copy=False), device=device)
    tensor_inputs = tensor_views[0]
    tensor_targets = torch.as_tensor(y_scaled.astype("float32", copy=False), device=device)
    tensor_codes = torch.as_tensor(codes_numpy, dtype=torch.int64, device=device)
    parameters = tuple(parameter for _name, parameter in model.named_parameters())
    stage_one_updates = int(schedule["independent_expert_updates"])
    if stage_one_updates:
        _set_router_trainable(model, False)
    best_score = float("inf")
    best_update = 0
    best_state: dict[str, Any] | None = None
    best_missing: tuple[float, ...] | None = None
    stage_one_model_hash: str | None = None
    stage_one_optimizer_hash: str | None = None
    history: list[dict[str, Any]] = []
    gradient_traversals = 0
    solver_calls = 0

    cudnn_scope = torch.backends.cudnn.flags(
        enabled=True, benchmark=False, deterministic=True, allow_tf32=False
    )
    with cudnn_scope:
        for update in range(1, updates + 1):
            if stage_one_updates and update == stage_one_updates + 1:
                _set_router_trainable(model, True)
            model.train(True)
            optimizer.zero_grad(set_to_none=True)
            cagrad_receipt: dict[str, Any] | None = None
            if update <= stage_one_updates:
                objective = _stage_one_objective(
                    model, tensor_inputs, tensor_targets, tensor_codes, len(scenario_order)
                )
                objective.backward()
                gradient_traversals += 1
            else:
                prediction = model(tensor_views.reshape(-1, 40)).reshape(15, len(X), 3)
                losses = row_task_losses(prediction, tensor_targets)
                cardinality_risks = cardinality_balanced_risks(
                    losses, tensor_codes, len(scenario_order)
                )
                mofe = (
                    more_vs_fewer_loss(losses, tensor_codes, len(scenario_order))
                    if schedule["mofe"]
                    else None
                )
                if not schedule["cagrad"]:
                    objective = direct_cardinality_objective(cardinality_risks)
                    if mofe is not None:
                        objective = objective + MOFE_COEFFICIENT * mofe
                    objective.backward()
                    gradient_traversals += 1
                else:
                    tasks = [
                        cardinality_risks[index] + MOFE_COEFFICIENT * mofe
                        for index in range(4)
                    ]
                    gradient_sets: list[tuple[Any, ...]] = []
                    for index, task in enumerate(tasks):
                        gradient_sets.append(
                            tuple(
                                torch.autograd.grad(
                                    task,
                                    parameters,
                                    retain_graph=index < 3,
                                    create_graph=False,
                                    allow_unused=False,
                                )
                            )
                        )
                    gradient_traversals += 4
                    cagrad_receipt = solve_cagrad_weights(
                        _flat_gradient_rows(gradient_sets)
                    )
                    solver_calls += 1
                    direction = cagrad_receipt["direction"].astype("float32", copy=False)
                    offset = 0
                    for parameter in parameters:
                        count = int(parameter.numel())
                        parameter.grad = torch.as_tensor(
                            direction[offset : offset + count].reshape(tuple(parameter.shape)),
                            dtype=torch.float32,
                            device=device,
                        )
                        offset += count
                    if offset != len(direction):
                        _fail("CAGrad parameter partition changed")
                    objective = direct_cardinality_objective(cardinality_risks).detach() + MOFE_COEFFICIENT * mofe.detach()
            optimizer.step()
            if stage_one_updates and update == stage_one_updates:
                stage_one_model_hash = state_dict_sha256(model.state_dict())
                stage_one_optimizer_hash = optimizer_state_sha256(optimizer)

            model.train(False)
            with torch.no_grad():
                checkpoint_prediction = (
                    model(tensor_views.reshape(-1, 40))
                    .reshape(15, len(X), 3)
                    .detach()
                    .cpu()
                    .numpy()
                )
            score, missing_components, view_components = training_checkpoint_score_m(
                checkpoint_prediction,
                y_scaled,
                codes_numpy,
                len(scenario_order),
                transform.target_scale_m,
            )
            eligible = update > stage_one_updates
            became_checkpoint = eligible and score < best_score
            if became_checkpoint:
                best_score = score
                best_update = update
                best_missing = missing_components
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            row = {
                "update": update,
                "stage": "independent_experts" if update <= stage_one_updates else "cooperative",
                "objective": float(objective.detach().cpu()),
                "checkpoint_eligible": eligible,
                "checkpoint_score_m": score,
                "training_full_component_m": float(view_components[0]),
                "training_missing_components_m": list(missing_components),
                "became_checkpoint": became_checkpoint,
            }
            if cagrad_receipt is not None:
                row["cagrad"] = {
                    "weights": cagrad_receipt["weights"].tolist(),
                    "objective": cagrad_receipt["objective"],
                    "iterations": cagrad_receipt["iterations"],
                    "equality_residual": cagrad_receipt["equality_residual"],
                    "bound_violation": cagrad_receipt["bound_violation"],
                }
            history.append(row)
    if best_state is None or best_missing is None or best_update <= stage_one_updates:
        _fail("V3 training produced no eligible checkpoint")
    model.load_state_dict(best_state, strict=True)
    expected_traversals = (
        stage_one_updates + (updates - stage_one_updates) * 4
        if schedule["cagrad"]
        else updates
    )
    expected_solves = updates - stage_one_updates if schedule["cagrad"] else 0
    if gradient_traversals != expected_traversals or solver_calls != expected_solves:
        _fail("V3 fit action accounting changed")
    return TrainingResult(
        candidate_id=candidate_id,
        seed=seed,
        transform=transform,
        state_dict=best_state,
        initial_state_sha256=initial_hash,
        checkpoint_state_sha256=state_dict_sha256(best_state),
        optimizer_initial_state_sha256=optimizer_hash,
        stage_one_model_state_sha256=stage_one_model_hash,
        stage_one_optimizer_state_sha256=stage_one_optimizer_hash,
        checkpoint_update=best_update,
        checkpoint_score_m=best_score,
        checkpoint_missing_components_m=best_missing,
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
    feature_batch.validate()
    transform.validate()
    scaled = transform_features(feature_batch.continuous_features, transform)
    views = build_exact_fifteen_views(scaled)
    import torch

    model = build_candidate_model(candidate_id, seed=seed, device=device)
    model.load_state_dict(state_dict, strict=True)
    model.train(False)
    with torch.no_grad():
        tensor = torch.as_tensor(views.astype("float32", copy=False), device=device)
        prediction_scaled = (
            model(tensor.reshape(-1, 40))
            .reshape(15, len(feature_batch.row_uids), 3)
            .detach()
            .cpu()
            .numpy()
        )
    prediction = inverse_targets(prediction_scaled, transform)
    if prediction.shape != (15, len(feature_batch.row_uids), 3):
        _fail("V3 outer prediction shape changed")
    return prediction


def _state_dict_payload(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    payload: dict[str, Any] = {}
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            _fail("checkpoint state contains a non-tensor")
        array = tensor.detach().cpu().contiguous().numpy()
        payload[name] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        }
    return payload


def load_checkpoint_state(raw: bytes) -> dict[str, Any]:
    import torch

    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OpenAxesExternalNativeV3TrainingError("checkpoint JSON is malformed") from exc
    if raw != _canonical_bytes(value) or not isinstance(value, Mapping) or set(value) != {"contract", "candidate_id", "seed", "state_dict", "state_dict_sha256"}:
        _fail("checkpoint envelope changed")
    state: dict[str, Any] = {}
    for name, row in value["state_dict"].items():
        if not isinstance(row, Mapping) or set(row) != {"dtype", "shape", "data_base64"}:
            _fail("checkpoint tensor row changed")
        try:
            array = np.frombuffer(base64.b64decode(row["data_base64"], validate=True), dtype=np.dtype(row["dtype"])).copy().reshape(tuple(row["shape"]))
        except (ValueError, TypeError) as exc:
            raise OpenAxesExternalNativeV3TrainingError("checkpoint tensor payload changed") from exc
        state[name] = torch.from_numpy(array)
    if state_dict_sha256(state) != value["state_dict_sha256"]:
        _fail("checkpoint state hash changed")
    return state


def _write_exact(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_bytes() != raw:
            _fail("existing deterministic artifact differs")
        return
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail("deterministic artifact short write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_fit_task(
    task_payload: Mapping[str, Any],
    run_dir: str | os.PathLike[str] | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run one exact fit and return a JSON-serializable controller result."""

    if not isinstance(task_payload, Mapping):
        _fail("fit task payload must be a mapping")
    required = {
        "X_train",
        "y_train_m",
        "training_scenarios",
        "train_uids",
        "declared_scenario_order",
        "candidate_id",
        "seed",
    }
    if not required.issubset(task_payload) or set(task_payload) - (
        required
        | {
            "expected_dmome_initial_state_sha256",
            "expected_dmome_optimizer_initial_state_sha256",
        }
    ):
        _fail("fit task payload keys changed")
    result = train_candidate_train_only(
        task_payload["X_train"],
        task_payload["y_train_m"],
        task_payload["training_scenarios"],
        task_payload["train_uids"],
        task_payload["declared_scenario_order"],
        candidate_id=task_payload["candidate_id"],
        seed=task_payload["seed"],
        device=device,
        expected_dmome_initial_state_sha256=task_payload.get(
            "expected_dmome_initial_state_sha256"
        ),
        expected_dmome_optimizer_initial_state_sha256=task_payload.get(
            "expected_dmome_optimizer_initial_state_sha256"
        ),
    )
    checkpoint = {
        "contract": "openaxes.external.native_sensor_experts.v3.checkpoint.v1",
        "candidate_id": result.candidate_id,
        "seed": result.seed,
        "state_dict": _state_dict_payload(result.state_dict),
        "state_dict_sha256": result.checkpoint_state_sha256,
    }
    checkpoint_raw = _canonical_bytes(checkpoint)
    checkpoint_reference: dict[str, Any] | None = None
    if run_dir is not None:
        relative = f"checkpoints/{result.candidate_id}__seed_{result.seed}.json"
        _write_exact(Path(run_dir) / relative, checkpoint_raw)
        checkpoint_reference = _reference(relative, checkpoint_raw)
    return {
        "candidate_id": result.candidate_id,
        "seed": result.seed,
        "updates": result.updates,
        "initial_state_sha256": result.initial_state_sha256,
        "optimizer_initial_state_sha256": result.optimizer_initial_state_sha256,
        "stage_one_model_state_sha256": result.stage_one_model_state_sha256,
        "stage_one_optimizer_state_sha256": result.stage_one_optimizer_state_sha256,
        "checkpoint_update": result.checkpoint_update,
        "checkpoint_score_m": result.checkpoint_score_m,
        "checkpoint_state_sha256": result.checkpoint_state_sha256,
        "checkpoint_reference": checkpoint_reference,
        "transform": result.transform.receipt(),
        "training_scenario_order": list(result.training_scenario_order),
        "training_model_forward_calls": result.training_model_forward_calls,
        "checkpoint_model_forward_calls": result.checkpoint_model_forward_calls,
        "parameter_gradient_traversals": result.parameter_gradient_traversals,
        "cagrad_solver_calls": result.cagrad_solver_calls,
    }


def run_experiment_fit(
    task: Mapping[str, Any],
    *,
    prepared_root: str,
    output_dir: str,
    device: str,
) -> Mapping[str, Any]:
    """Stable fresh-child API used by the V3 experiment controller.

    The controller retains responsibility for authenticating and projecting
    the prepared fold into the train-only arrays carried in ``task``.  This
    function deliberately does not reopen outer labels or infer a partition
    from an unauthenticated path.  ``prepared_root`` is retained in the API so
    the controller can bind one uniform child invocation contract.
    """

    if (
        not isinstance(task, Mapping)
        or any(field not in task for field in _EXPERIMENT_TRAIN_ONLY_FIELDS)
        or
        type(prepared_root) is not str
        or not prepared_root
        or type(output_dir) is not str
        or not output_dir
        or type(device) is not str
        or not device
    ):
        _fail("experiment fit paths/device are invalid")
    # The controller may carry slot/GPU/fold provenance alongside the arrays.
    # Project only the authenticated train-only interface: this keeps those
    # controller fields out of training authority and, critically, prevents
    # this module from opening the prepared or outer-evaluation data itself.
    train_only_task = {
        field: task[field] for field in _EXPERIMENT_TRAIN_ONLY_FIELDS
    }
    result = dict(
        run_fit_task(train_only_task, run_dir=output_dir, device=device)
    )
    result["prepared_root_bound"] = prepared_root
    result["outer_view_metrics"] = None
    result["outer_evaluation_deferred_to_experiment"] = True
    return result


__all__ = [
    "OPENAXES_NATIVE_V3_CANDIDATE_IDS",
    "_SEEDS",
    "_MASKS",
    "_VIEW_IDS",
    "TrainingResult",
    "FeatureOnlyBatch",
    "TrainOnlyTransform",
    "candidate_schedule",
    "adjacent_mask_edges",
    "build_exact_fifteen_views",
    "validate_exact_fifteen_views",
    "build_dense_6979",
    "build_static_sensor_experts_6979",
    "build_dmome_6979",
    "build_candidate_model",
    "cardinality_balanced_risks",
    "direct_cardinality_objective",
    "more_vs_fewer_loss",
    "solve_cagrad_weights",
    "training_checkpoint_score_m",
    "train_candidate_train_only",
    "predict_feature_only_batch_once",
    "load_checkpoint_state",
    "run_fit_task",
    "run_experiment_fit",
]
