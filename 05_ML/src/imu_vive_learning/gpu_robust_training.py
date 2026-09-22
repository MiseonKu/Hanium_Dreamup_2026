"""Segment-robust PyTorch training for the static endpoint proxy.

This module deliberately keeps ``segment_id`` beside, rather than inside, the
feature matrix.  Segment labels are used only to construct balanced training
batches and group-aware objectives.  Validation and test segment labels are
never consulted by the trainer.

The emitted state dictionaries use the same 43->64->64->3 ReLU architecture
as :func:`imu_vive_learning.gpu_experiment._build_torch_model`, so checkpoints
remain consumable by the existing safe GPU inference loader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import time
from typing import Any

import numpy as np

from .experiment_metrics import regression_metrics
from .experiment_models import derive_context_seed


ROBUST_OBJECTIVES = ("balanced_erm", "group_dro", "v_rex")
ROBUST_HIDDEN_LAYERS = (64, 64)
ROBUST_ENSEMBLE_SEED_COUNT = 3
ROBUST_MAX_EPOCHS = 400
GROUP_DRO_WEIGHT_FLOOR = 1e-12


class RobustTrainingError(RuntimeError):
    """Raised when segment-robust training violates its local contract."""


def _finite_vector(values: Any, *, name: str, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _normalized_segments(segment_ids: Sequence[Any]) -> np.ndarray:
    values = np.asarray(segment_ids, dtype=object)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("segment_ids must be a non-empty one-dimensional sequence")
    normalized: list[str] = []
    for value in values.tolist():
        if isinstance(value, np.generic):
            value = value.item()
        if value is None:
            raise ValueError("segment_ids contains a missing value")
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise ValueError("segment_ids contains a non-finite value")
        text = str(value)
        if not text:
            raise ValueError("segment_ids contains an empty value")
        normalized.append(text)
    return np.asarray(normalized, dtype=object)


def group_mass(
    segment_ids: Sequence[Any], row_weights: Sequence[float] | None = None
) -> dict[str, float]:
    """Return normalized objective mass by segment as a pure CPU helper.

    With no explicit weights this is empirical row mass.  Calling it on one
    batch from :func:`segment_balanced_batches` yields exactly ``1 / G`` for
    every training group and is used as an invariant in the trainer.
    """

    segments = _normalized_segments(segment_ids)
    if row_weights is None:
        weights = np.ones(len(segments), dtype=np.float64)
    else:
        weights = _finite_vector(row_weights, name="row_weights", nonnegative=True)
        if weights.shape != segments.shape:
            raise ValueError("row_weights must have one value per segment_id")
    total = float(weights.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("row_weights must have positive total mass")
    return {
        group: float(weights[segments == group].sum(dtype=np.float64) / total)
        for group in sorted(set(segments.tolist()))
    }


def segment_balanced_batches(
    segment_ids: Sequence[Any], *, batch_size: int, seed: int
) -> tuple[np.ndarray, ...]:
    """Build one deterministic, largest-group-length balanced epoch.

    Each batch receives ``batch_size // G`` rows from each of the ``G``
    training segments.  The epoch has enough batches to visit every row in the
    largest group.  Smaller groups (and a non-divisible largest group tail)
    are deterministically cycled; segment IDs are never appended to features.
    """

    segments = _normalized_segments(segment_ids)
    groups = tuple(sorted(set(segments.tolist())))
    requested_batch_size = int(batch_size)
    if requested_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows_per_group = requested_batch_size // len(groups)
    if rows_per_group <= 0:
        raise ValueError("batch_size must be at least the number of segments")

    generator = np.random.default_rng(int(seed))
    shuffled: dict[str, np.ndarray] = {}
    largest_count = 0
    for group in groups:
        indices = np.flatnonzero(segments == group).astype(np.int64, copy=False)
        shuffled[group] = generator.permutation(indices)
        largest_count = max(largest_count, len(indices))
    batch_count = int(math.ceil(largest_count / rows_per_group))
    required_per_group = batch_count * rows_per_group

    cycled: dict[str, np.ndarray] = {}
    for group in groups:
        order = shuffled[group]
        repetitions = int(math.ceil(required_per_group / len(order)))
        cycled[group] = np.tile(order, repetitions)[:required_per_group]

    batches: list[np.ndarray] = []
    expected_mass = 1.0 / len(groups)
    for batch_index in range(batch_count):
        start = batch_index * rows_per_group
        stop = start + rows_per_group
        batch = np.concatenate(
            [cycled[group][start:stop] for group in groups]
        ).astype(np.int64, copy=False)
        batch = batch[generator.permutation(len(batch))]
        masses = group_mass(segments[batch])
        if any(not math.isclose(value, expected_mass, abs_tol=1e-12) for value in masses.values()):
            raise AssertionError("internal segment-balanced batch mass changed")
        batches.append(batch)
    return tuple(batches)


def update_group_dro_weights(
    weights: Sequence[float], group_losses: Sequence[float], *, eta: float
) -> np.ndarray:
    """Apply a numerically stable exponentiated-gradient GroupDRO update."""

    q = _finite_vector(weights, name="weights", nonnegative=True)
    losses = _finite_vector(group_losses, name="group_losses", nonnegative=True)
    if q.shape != losses.shape:
        raise ValueError("weights and group_losses must have identical shapes")
    if np.any(q <= 0.0):
        raise ValueError("GroupDRO weights must be strictly positive")
    step = float(eta)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("eta must be finite and positive")
    log_weights = np.log(q / q.sum(dtype=np.float64)) + step * losses
    log_weights -= float(np.max(log_weights))
    updated = np.exp(log_weights)
    # Long runs can otherwise underflow non-worst groups to exactly zero,
    # making the next exponentiated-gradient update undefined.  This floor is
    # purely a float64 numerical guard; weights are renormalized immediately.
    updated = np.maximum(updated, GROUP_DRO_WEIGHT_FLOOR)
    updated /= updated.sum(dtype=np.float64)
    if not np.isfinite(updated).all():
        raise ValueError("GroupDRO update produced non-finite weights")
    return updated


def v_rex_loss(group_losses: Sequence[float], *, beta: float) -> float:
    """Return mean group loss plus population variance penalty on CPU."""

    losses = _finite_vector(group_losses, name="group_losses", nonnegative=True)
    penalty = float(beta)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("beta must be finite and non-negative")
    center = float(np.mean(losses, dtype=np.float64))
    variance = float(np.mean(np.square(losses - center), dtype=np.float64))
    result = center + penalty * variance
    if not math.isfinite(result):
        raise ValueError("V-REx loss is non-finite")
    return result


def post_warmup_value(target: float, *, epoch: int, warmup_epochs: int) -> float:
    """Return zero through the ERM warm-up, then the fixed coefficient."""

    value = float(target)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("warmup target must be finite and non-negative")
    current_epoch = int(epoch)
    duration = int(warmup_epochs)
    if current_epoch <= 0 or duration < 0:
        raise ValueError("epoch must be positive and warmup_epochs non-negative")
    return 0.0 if current_epoch <= duration else value


def _matrix(
    values: Any, *, name: str, columns: int | None = None, rows: int | None = None
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or len(array) == 0:
        raise RobustTrainingError(f"{name} must be a non-empty matrix")
    if columns is not None and array.shape[1] != columns:
        raise RobustTrainingError(
            f"{name} must have {columns} columns; received {array.shape[1]}"
        )
    if rows is not None and len(array) != rows:
        raise RobustTrainingError(
            f"{name} must have {rows} rows; received {len(array)}"
        )
    if not np.isfinite(array).all():
        raise RobustTrainingError(f"{name} contains non-finite values")
    return array


def _training_segment_ids(arrays: Mapping[str, Any], row_count: int) -> np.ndarray:
    if "train_segment_ids" in arrays:
        source = arrays["train_segment_ids"]
    elif "train" in arrays and hasattr(arrays["train"], "columns"):
        train = arrays["train"]
        if "segment_id" not in train.columns:
            raise RobustTrainingError("training table lacks segment_id")
        source = train["segment_id"].to_numpy()
    else:
        raise RobustTrainingError(
            "segment-aware training requires training-only segment IDs"
        )
    try:
        segments = _normalized_segments(source)
    except ValueError as exc:
        raise RobustTrainingError(str(exc)) from exc
    if len(segments) != row_count:
        raise RobustTrainingError(
            "training segment ID count differs from the training feature rows"
        )
    if len(set(segments.tolist())) < 2:
        raise RobustTrainingError("robust training requires at least two segments")
    return segments


def _settings(task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    model_config = config["models"]["torch_mlp"]
    params: dict[str, Any] = {}
    candidate = task.get("candidate")
    if isinstance(candidate, Mapping):
        params.update(candidate)
    task_params = task.get("params")
    if isinstance(task_params, Mapping):
        params.update(task_params)
    for key in ("candidate_id", "objective", "eta", "beta", "warmup_epochs", "sam", "rho"):
        if key in task:
            params[key] = task[key]

    raw_objective = str(params.get("objective", "balanced_erm"))
    sam_enabled = bool(params.get("sam", False))
    if raw_objective == "sam":
        raw_objective = "balanced_erm"
        sam_enabled = True
    if raw_objective not in ROBUST_OBJECTIVES:
        raise RobustTrainingError(f"unknown robust objective {raw_objective!r}")
    if sam_enabled and raw_objective != "balanced_erm":
        raise RobustTrainingError("SAM is supported only with balanced_erm")

    hidden = tuple(
        int(value)
        for value in params.get(
            "hidden_layer_sizes", model_config.get("hidden_layer_sizes", ROBUST_HIDDEN_LAYERS)
        )
    )
    if hidden != ROBUST_HIDDEN_LAYERS:
        raise RobustTrainingError("robust checkpoint architecture must remain 64-64 ReLU")
    eta = params.get("eta")
    beta = params.get("beta")
    warmup_epochs = params.get("warmup_epochs", 0)
    rho = params.get("rho")
    if raw_objective == "group_dro":
        try:
            update_group_dro_weights([0.5, 0.5], [0.0, 0.0], eta=float(eta))
        except (TypeError, ValueError) as exc:
            raise RobustTrainingError(f"invalid GroupDRO eta: {eta!r}") from exc
    if raw_objective == "v_rex":
        try:
            post_warmup_value(
                float(beta), epoch=1, warmup_epochs=int(warmup_epochs)
            )
        except (TypeError, ValueError) as exc:
            raise RobustTrainingError("invalid V-REx beta or warmup_epochs") from exc
    if sam_enabled:
        try:
            rho_value = float(rho)
        except (TypeError, ValueError) as exc:
            raise RobustTrainingError(f"invalid SAM rho: {rho!r}") from exc
        if not math.isfinite(rho_value) or rho_value <= 0.0:
            raise RobustTrainingError("SAM rho must be finite and positive")

    seed_count = int(model_config.get("ensemble_seed_count", ROBUST_ENSEMBLE_SEED_COUNT))
    if seed_count != ROBUST_ENSEMBLE_SEED_COUNT:
        raise RobustTrainingError("robust training requires exactly three ensemble seeds")
    epoch_budget = int(model_config.get("epoch_budget", ROBUST_MAX_EPOCHS))
    if epoch_budget <= 0 or epoch_budget > ROBUST_MAX_EPOCHS:
        raise RobustTrainingError("robust epoch budget must lie in [1, 400]")
    batch_size = int(model_config["batch_size"])
    learning_rate = float(model_config["learning_rate"])
    weight_decay = float(params.get("weight_decay", model_config["weight_decay"]))
    if batch_size <= 0:
        raise RobustTrainingError("robust batch_size must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise RobustTrainingError("robust learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise RobustTrainingError("robust weight_decay must be finite and non-negative")

    return {
        "candidate_id": str(params.get("candidate_id", raw_objective)),
        "objective": raw_objective,
        "eta": None if eta is None else float(eta),
        "beta": None if beta is None else float(beta),
        "warmup_epochs": int(warmup_epochs),
        "sam": sam_enabled,
        "rho": None if rho is None else float(rho),
        "hidden": hidden,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epoch_budget": epoch_budget,
        "seed_count": seed_count,
    }


def _torch_group_losses(
    prediction: Any,
    truth: Any,
    batch_segments: np.ndarray,
    groups: tuple[str, ...],
    *,
    torch: Any,
) -> Any:
    losses: list[Any] = []
    for group in groups:
        mask = torch.as_tensor(batch_segments == group, device=prediction.device)
        if not bool(torch.any(mask).item()):
            raise RobustTrainingError(f"balanced batch omitted training segment {group!r}")
        losses.append(torch.mean(torch.square(prediction[mask] - truth[mask])))
    stacked = torch.stack(losses)
    if not bool(torch.isfinite(stacked).all().item()):
        raise RobustTrainingError("non-finite per-segment PyTorch loss")
    return stacked


def _torch_objective(
    group_losses: Any,
    *,
    settings: Mapping[str, Any],
    dro_weights: np.ndarray,
    epoch: int,
    torch: Any,
) -> tuple[Any, np.ndarray, float]:
    objective = str(settings["objective"])
    effective_beta = 0.0
    next_weights = dro_weights
    if objective == "balanced_erm":
        loss = torch.mean(group_losses)
    elif objective == "group_dro":
        next_weights = update_group_dro_weights(
            dro_weights,
            group_losses.detach().cpu().numpy(),
            eta=float(settings["eta"]),
        )
        q = torch.as_tensor(next_weights, dtype=group_losses.dtype, device=group_losses.device)
        loss = torch.sum(q * group_losses)
    elif objective == "v_rex":
        effective_beta = post_warmup_value(
            float(settings["beta"]),
            epoch=epoch,
            warmup_epochs=int(settings["warmup_epochs"]),
        )
        center = torch.mean(group_losses)
        loss = center + effective_beta * torch.mean(torch.square(group_losses - center))
    else:  # pragma: no cover - guarded by _settings
        raise AssertionError(objective)
    if not bool(torch.isfinite(loss).item()):
        raise RobustTrainingError("non-finite robust PyTorch objective")
    return loss, next_weights, effective_beta


def _raw_prediction(
    model: Any,
    features: Any,
    *,
    torch: Any,
    preprocessor: Any,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        scaled = model(features).detach().cpu().numpy().astype(np.float64)
    try:
        result = preprocessor.inverse_targets(scaled)
    except (RuntimeError, ValueError) as exc:
        raise RobustTrainingError("failed to inverse-transform robust predictions") from exc
    if not np.isfinite(result).all():
        raise RobustTrainingError("robust PyTorch prediction contains non-finite values")
    return result


def _sam_step(
    *,
    model: Any,
    optimizer: Any,
    first_loss: Any,
    second_loss_factory: Any,
    rho: float,
    torch: Any,
) -> float:
    """Perform one non-adaptive SAM update and return the second loss."""

    optimizer.zero_grad(set_to_none=True)
    first_loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise RobustTrainingError("SAM found no gradients")
    norm = torch.sqrt(
        torch.sum(torch.stack([torch.sum(torch.square(gradient)) for gradient in gradients]))
    )
    if not bool(torch.isfinite(norm).item()):
        raise RobustTrainingError("SAM gradient norm is non-finite")
    if float(norm.detach().item()) == 0.0:
        optimizer.step()
        return float(first_loss.detach().item())

    perturbations: list[tuple[Any, Any]] = []
    scale = float(rho) / float(norm.detach().item())
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            perturbation = parameter.grad * scale
            parameter.add_(perturbation)
            perturbations.append((parameter, perturbation))
    optimizer.zero_grad(set_to_none=True)
    second_loss = second_loss_factory()
    if not bool(torch.isfinite(second_loss).item()):
        raise RobustTrainingError("SAM perturbed objective is non-finite")
    second_loss.backward()
    with torch.no_grad():
        for parameter, perturbation in perturbations:
            parameter.sub_(perturbation)
    optimizer.step()
    return float(second_loss.detach().item())


def train_robust_torch(
    arrays: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    config: Mapping[str, Any],
    return_state: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Train a three-seed segment-robust 64-64 endpoint ensemble.

    ``inner`` and ``smoke`` tasks checkpoint the epoch with the best measured
    validation RMSE.  ``outer`` and ``final`` tasks execute their fixed epoch
    budgets without consulting evaluation metrics for checkpoint selection.
    The result mirrors the v1 GPU trainer and additionally exposes
    ``group_history``.
    """

    import torch

    settings = _settings(task, config)
    X_train_array = _matrix(arrays["X_train"], name="X_train")
    y_train_array = _matrix(
        arrays["y_train"], name="y_train", columns=3, rows=len(X_train_array)
    )
    segment_ids = _training_segment_ids(arrays, len(X_train_array))
    groups = tuple(sorted(set(segment_ids.tolist())))
    if settings["batch_size"] < len(groups):
        raise RobustTrainingError("batch_size must be at least the number of training segments")

    preprocessor = arrays.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "inverse_targets"):
        raise RobustTrainingError("robust training requires a fitted TrainOnlyPreprocessor")
    feature_names = tuple(getattr(preprocessor, "feature_names", ()))
    if any(str(name).lower() == "segment_id" for name in feature_names):
        raise RobustTrainingError("segment_id must never be a model feature")
    if feature_names and len(feature_names) != X_train_array.shape[1]:
        raise RobustTrainingError("preprocessor feature count differs from X_train")

    select_checkpoint = str(task["phase"]) in {"inner", "smoke"}
    X_eval_array: np.ndarray | None = None
    y_eval_raw: np.ndarray | None = None
    if "X_eval" in arrays:
        X_eval_array = _matrix(
            arrays["X_eval"], name="X_eval", columns=X_train_array.shape[1]
        )
        y_eval_raw = np.asarray(arrays.get("y_eval_raw"), dtype=np.float64)
        if y_eval_raw.shape != (len(X_eval_array), 3) or not np.isfinite(y_eval_raw).all():
            raise RobustTrainingError("y_eval_raw must be finite and aligned XYZ")
    if select_checkpoint and X_eval_array is None:
        raise RobustTrainingError("inner checkpoint selection lacks validation data")

    try:
        train_truth_raw = preprocessor.inverse_targets(y_train_array.astype(np.float64))
    except (RuntimeError, ValueError) as exc:
        raise RobustTrainingError("failed to inverse-transform robust training targets") from exc
    if not np.isfinite(train_truth_raw).all():
        raise RobustTrainingError("inverse-transformed training targets are non-finite")

    resolved_device = torch.device(device or "cuda:0")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RobustTrainingError("CUDA robust training requested without an available GPU")
    X_train = torch.as_tensor(X_train_array, device=resolved_device)
    y_train = torch.as_tensor(y_train_array, device=resolved_device)
    X_eval = (
        None if X_eval_array is None else torch.as_tensor(X_eval_array, device=resolved_device)
    )

    seed_indices = [
        int(value)
        for value in task.get("seed_indices", range(int(settings["seed_count"])))
    ]
    if (
        not seed_indices
        or len(seed_indices) != len(set(seed_indices))
        or any(index not in range(ROBUST_ENSEMBLE_SEED_COUNT) for index in seed_indices)
    ):
        raise RobustTrainingError("seed_indices must be a unique non-empty subset of 0, 1, 2")

    history_interval = int(config.get("monitoring", {}).get("history_interval_epochs", 25))
    if history_interval <= 0:
        raise RobustTrainingError("history interval must be positive")
    default_epochs = int(task.get("epoch_budget_override", settings["epoch_budget"]))
    epoch_map = task.get("epochs_per_seed", {})
    if not isinstance(epoch_map, Mapping):
        raise RobustTrainingError("epochs_per_seed must be a mapping")

    predictions: list[np.ndarray] = []
    seed_summaries: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    group_history: list[dict[str, Any]] = []
    states: dict[int, dict[str, np.ndarray]] = {}
    split_context = str(arrays.get("split_context", ""))
    task_id = str(task.get("task_id", "robust_torch"))
    empirical_mass = group_mass(segment_ids)

    # Lazy import avoids a module cycle when the experiment runner dispatches
    # into this trainer.  This is the canonical architecture constructor.
    from .gpu_experiment import _build_torch_model

    for seed_index in seed_indices:
        epochs = int(epoch_map.get(str(seed_index), epoch_map.get(seed_index, default_epochs)))
        if epochs <= 0 or epochs > int(settings["epoch_budget"]) or epochs > ROBUST_MAX_EPOCHS:
            raise RobustTrainingError(
                f"invalid robust epoch budget for seed {seed_index}: {epochs}"
            )
        model_seed = derive_context_seed(
            int(config["contract"]["base_seed"]),
            # Pair initializations across candidates within each split context.
            # Candidate-specific seeds would confound objective comparisons with
            # initialization luck and would make the balanced-ERM control less
            # directly comparable to the v1 training path.
            "gpu_model_seed",
            split_context,
            seed_index,
        )
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = _build_torch_model(
            torch, X_train.shape[1], ROBUST_HIDDEN_LAYERS
        ).to(resolved_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
        )
        dro_weights = np.full(len(groups), 1.0 / len(groups), dtype=np.float64)
        best_score = math.inf
        best_epoch = 0
        best_state: dict[str, Any] | None = None
        final_loss = math.nan
        final_effective_beta = 0.0
        started = time.perf_counter()

        for epoch in range(1, epochs + 1):
            model.train()
            batches = segment_balanced_batches(
                segment_ids,
                batch_size=int(settings["batch_size"]),
                seed=derive_context_seed(model_seed, "segment_balanced_epoch", epoch),
            )
            group_loss_mass = np.zeros(len(groups), dtype=np.float64)
            loss_mass = 0.0
            batch_mass = 0
            row_visits = 0
            for batch_cpu in batches:
                indices = torch.as_tensor(batch_cpu, dtype=torch.long, device=resolved_device)
                batch_segments = segment_ids[batch_cpu]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(X_train[indices])
                losses = _torch_group_losses(
                    prediction,
                    y_train[indices],
                    batch_segments,
                    groups,
                    torch=torch,
                )
                objective_loss, next_dro_weights, effective_beta = _torch_objective(
                    losses,
                    settings=settings,
                    dro_weights=dro_weights,
                    epoch=epoch,
                    torch=torch,
                )
                dro_weights = next_dro_weights
                first_group_losses = losses.detach().cpu().numpy().astype(np.float64)

                if bool(settings["sam"]):
                    def perturbed_loss() -> Any:
                        perturbed_prediction = model(X_train[indices])
                        perturbed_groups = _torch_group_losses(
                            perturbed_prediction,
                            y_train[indices],
                            batch_segments,
                            groups,
                            torch=torch,
                        )
                        return torch.mean(perturbed_groups)

                    batch_loss = _sam_step(
                        model=model,
                        optimizer=optimizer,
                        first_loss=objective_loss,
                        second_loss_factory=perturbed_loss,
                        rho=float(settings["rho"]),
                        torch=torch,
                    )
                else:
                    objective_loss.backward()
                    optimizer.step()
                    batch_loss = float(objective_loss.detach().item())
                if not math.isfinite(batch_loss):
                    raise RobustTrainingError("non-finite robust batch loss")
                group_loss_mass += first_group_losses
                loss_mass += batch_loss
                batch_mass += 1
                row_visits += len(batch_cpu)
                final_effective_beta = effective_beta
            if batch_mass <= 0:
                raise RobustTrainingError("segment-balanced epoch produced no batches")
            final_loss = loss_mass / batch_mass
            epoch_group_losses = group_loss_mass / batch_mass
            if not math.isfinite(final_loss) or not np.isfinite(epoch_group_losses).all():
                raise RobustTrainingError("non-finite robust epoch summary")

            validation_score: float | None = None
            if select_checkpoint:
                assert X_eval is not None and y_eval_raw is not None
                validation_prediction = _raw_prediction(
                    model, X_eval, torch=torch, preprocessor=preprocessor
                )
                validation_score = float(
                    regression_metrics(y_eval_raw, validation_prediction)["rmse_3d_mm"]
                )
                if not math.isfinite(validation_score):
                    raise RobustTrainingError("validation RMSE is non-finite")
                if validation_score < best_score:
                    best_score = validation_score
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }

            if epoch == 1 or epoch % history_interval == 0 or epoch == epochs:
                train_prediction = _raw_prediction(
                    model, X_train, torch=torch, preprocessor=preprocessor
                )
                train_score = float(
                    regression_metrics(train_truth_raw, train_prediction)["rmse_3d_mm"]
                )
                history_row: dict[str, Any] = {
                    "task_id": task_id,
                    "candidate_id": settings["candidate_id"],
                    "objective": settings["objective"],
                    "sam": bool(settings["sam"]),
                    "seed_index": seed_index,
                    "model_seed": model_seed,
                    "epoch": epoch,
                    "scaled_objective_loss": float(final_loss),
                    "train_rmse_3d_mm": train_score,
                    "segment_count": len(groups),
                    "balanced_row_visits": int(row_visits),
                    "v_rex_beta": float(final_effective_beta),
                }
                if select_checkpoint:
                    assert validation_score is not None
                    history_row.update(
                        {
                            "validation_rmse_3d_mm": validation_score,
                            "train_validation_gap_mm": validation_score - train_score,
                            "best_validation_rmse_3d_mm": best_score,
                            "best_epoch_so_far": best_epoch,
                        }
                    )
                history_rows.append(history_row)
                for group_index, group in enumerate(groups):
                    group_history.append(
                        {
                            "task_id": task_id,
                            "candidate_id": settings["candidate_id"],
                            "objective": settings["objective"],
                            "sam": bool(settings["sam"]),
                            "seed_index": seed_index,
                            "model_seed": model_seed,
                            "epoch": epoch,
                            "segment_id": group,
                            "scaled_group_mse_loss": float(epoch_group_losses[group_index]),
                            "empirical_train_mass": float(empirical_mass[group]),
                            "balanced_batch_mass": float(1.0 / len(groups)),
                            "group_dro_q": (
                                float(dro_weights[group_index])
                                if settings["objective"] == "group_dro"
                                else None
                            ),
                            "v_rex_beta": float(final_effective_beta),
                        }
                    )

        if select_checkpoint:
            if best_state is None or best_epoch <= 0:
                raise RobustTrainingError("robust checkpoint selection found no finite state")
            model.load_state_dict(best_state)

        evaluation_prediction: np.ndarray | None = None
        if X_eval is not None:
            assert y_eval_raw is not None
            evaluation_prediction = _raw_prediction(
                model, X_eval, torch=torch, preprocessor=preprocessor
            )
            predictions.append(evaluation_prediction)
        selected_train_prediction = _raw_prediction(
            model, X_train, torch=torch, preprocessor=preprocessor
        )
        summary: dict[str, Any] = {
            "seed_index": seed_index,
            "model_seed": model_seed,
            "candidate_id": settings["candidate_id"],
            "objective": settings["objective"],
            "sam": bool(settings["sam"]),
            "trained_epochs": epochs,
            "selected_epoch": best_epoch if select_checkpoint else epochs,
            "best_validation_rmse_3d_mm": best_score if select_checkpoint else None,
            "final_scaled_mse_loss": float(final_loss),
            "selected_train_rmse_3d_mm": float(
                regression_metrics(train_truth_raw, selected_train_prediction)["rmse_3d_mm"]
            ),
            "fit_seconds": float(time.perf_counter() - started),
        }
        if evaluation_prediction is not None:
            evaluation_metrics = regression_metrics(y_eval_raw, evaluation_prediction)
            summary["evaluation_rmse_3d_mm"] = float(evaluation_metrics["rmse_3d_mm"])
            summary["evaluation_p95_3d_error_mm"] = float(
                evaluation_metrics["p95_3d_error_mm"]
            )
        if not all(
            math.isfinite(float(value))
            for key, value in summary.items()
            if isinstance(value, (float, np.floating)) and key != "best_validation_rmse_3d_mm"
        ):
            raise RobustTrainingError("robust seed summary contains non-finite values")
        seed_summaries.append(summary)

        if return_state:
            state = {
                name: value.detach().cpu().numpy()
                for name, value in model.state_dict().items()
            }
            if any(not np.isfinite(value).all() for value in state.values()):
                raise RobustTrainingError("robust checkpoint contains non-finite tensors")
            states[seed_index] = state
        del model, optimizer
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble = None
    metrics = None
    if predictions:
        ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
        if not np.isfinite(ensemble).all():
            raise RobustTrainingError("robust ensemble prediction is non-finite")
        assert y_eval_raw is not None
        metrics = regression_metrics(y_eval_raw, ensemble)
    return {
        "ensemble_prediction": ensemble,
        "metrics": metrics,
        "seed_summaries": seed_summaries,
        "history": history_rows,
        "states": states,
        "group_history": group_history,
    }


# Keep a private-name entrypoint analogous to gpu_experiment._train_torch for
# runner integration while retaining a descriptive public function for tests.
_train_robust_torch = train_robust_torch


__all__ = [
    "GROUP_DRO_WEIGHT_FLOOR",
    "ROBUST_ENSEMBLE_SEED_COUNT",
    "ROBUST_HIDDEN_LAYERS",
    "ROBUST_MAX_EPOCHS",
    "ROBUST_OBJECTIVES",
    "RobustTrainingError",
    "_train_robust_torch",
    "group_mass",
    "post_warmup_value",
    "segment_balanced_batches",
    "train_robust_torch",
    "update_group_dro_weights",
    "v_rex_loss",
]
