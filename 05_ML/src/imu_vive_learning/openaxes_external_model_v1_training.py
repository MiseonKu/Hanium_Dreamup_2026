"""Train-only math and model core for the OpenAXES external benchmark.

PyTorch is imported lazily.  A controller can therefore import configuration,
task, and recovery code before a spawned CUDA worker fixes its visible device.
The trainer intentionally has no outer-row argument; outer prediction and the
single outer evaluation are separate operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .openaxes_external_model_v1_config import (
    OPENAXES_MODEL_METHOD_IDS,
    OPENAXES_MODEL_SENSOR_GROUPS,
    OPENAXES_MODEL_VIEW_IDS,
)


MASK_LAW = (0.25, 0.1875, 0.1875, 0.1875, 0.1875)
GROUPDRO_ETA = 0.01
VREX_BETA = 1.0
MOFE_LAMBDA = 0.1


class OpenAxesExternalModelV1TrainingError(RuntimeError):
    """Raised when a training-only mathematical invariant is violated."""


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_float_matrix(
    value: Any, *, columns: int, name: str, allow_empty: bool = False
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[1] != columns
        or (not allow_empty and array.shape[0] == 0)
        or array.dtype.kind not in "fc"
        or not np.isfinite(array).all()
    ):
        raise OpenAxesExternalModelV1TrainingError(
            f"{name} must be a finite N x {columns} floating matrix"
        )
    return np.asarray(array, dtype=np.float64)


def _require_exact_text_vector(value: Sequence[str], *, rows: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.shape != (rows,) or any(type(item) is not str or not item for item in array):
        raise OpenAxesExternalModelV1TrainingError(
            f"{name} must contain exactly {rows} nonempty strings"
        )
    return array


@dataclass(frozen=True)
class TrainOnlyTransform:
    """One fold's immutable train-only feature and target transform."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean_m: np.ndarray
    target_scale_m: float
    train_rows: int
    train_uid_sha256: str

    def validate(self) -> None:
        if (
            self.feature_mean.shape != (36,)
            or self.feature_scale.shape != (36,)
            or self.target_mean_m.shape != (3,)
            or not np.isfinite(self.feature_mean).all()
            or not np.isfinite(self.feature_scale).all()
            or not np.isfinite(self.target_mean_m).all()
            or np.any(self.feature_scale <= 0.0)
            or type(self.target_scale_m) is not float
            or not math.isfinite(self.target_scale_m)
            or self.target_scale_m <= 0.0
            or type(self.train_rows) is not int
            or self.train_rows <= 0
            or type(self.train_uid_sha256) is not str
            or len(self.train_uid_sha256) != 64
        ):
            raise OpenAxesExternalModelV1TrainingError(
                "train-only transform receipt is invalid"
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
            "train_uid_sha256": self.train_uid_sha256,
            "outer_rows_used": 0,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def fit_train_only_transform(
    X_train: Any,
    y_train_m: Any,
    train_uids: Sequence[str],
) -> TrainOnlyTransform:
    """Fit population statistics using only one outer fold's training rows."""

    X = _require_float_matrix(X_train, columns=36, name="X_train")
    y = _require_float_matrix(y_train_m, columns=3, name="y_train_m")
    if len(X) != len(y):
        raise OpenAxesExternalModelV1TrainingError("training X/y row mismatch")
    uids = _require_exact_text_vector(train_uids, rows=len(X), name="train_uids")
    if len(set(uids.tolist())) != len(uids):
        raise OpenAxesExternalModelV1TrainingError("training row_uid values are not unique")
    feature_mean = np.mean(X, axis=0, dtype=np.float64)
    feature_scale = np.std(X, axis=0, ddof=0, dtype=np.float64)
    if not np.isfinite(feature_scale).all() or np.any(feature_scale <= 0.0):
        raise OpenAxesExternalModelV1TrainingError(
            "training feature has zero or invalid population scale"
        )
    target_mean = np.mean(y, axis=0, dtype=np.float64)
    centered = y - target_mean
    target_scale = float(np.sqrt(np.mean(np.square(centered), dtype=np.float64)))
    if not math.isfinite(target_scale) or target_scale <= 0.0:
        raise OpenAxesExternalModelV1TrainingError(
            "training target isotropic population scale is invalid"
        )
    uid_bytes = "\n".join(uids.tolist()).encode("utf-8")
    result = TrainOnlyTransform(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean_m=target_mean,
        target_scale_m=target_scale,
        train_rows=len(X),
        train_uid_sha256=hashlib.sha256(uid_bytes).hexdigest(),
    )
    result.validate()
    return result


def transform_features(value: Any, transform: TrainOnlyTransform) -> np.ndarray:
    transform.validate()
    X = _require_float_matrix(value, columns=36, name="features")
    result = (X - transform.feature_mean) / transform.feature_scale
    if not np.isfinite(result).all():
        raise OpenAxesExternalModelV1TrainingError("standardized features are non-finite")
    return result


def transform_targets(value: Any, transform: TrainOnlyTransform) -> np.ndarray:
    transform.validate()
    y = _require_float_matrix(value, columns=3, name="targets")
    result = (y - transform.target_mean_m) / transform.target_scale_m
    if not np.isfinite(result).all():
        raise OpenAxesExternalModelV1TrainingError("standardized targets are non-finite")
    return result


def inverse_targets(value: Any, transform: TrainOnlyTransform) -> np.ndarray:
    transform.validate()
    y = np.asarray(value)
    if y.ndim < 2 or y.shape[-1] != 3 or y.dtype.kind not in "fc" or not np.isfinite(y).all():
        raise OpenAxesExternalModelV1TrainingError(
            "scaled prediction must be finite with trailing width three"
        )
    return np.asarray(y, dtype=np.float64) * transform.target_scale_m + transform.target_mean_m


def build_exact_five_views(X_scaled: Any) -> np.ndarray:
    """Build full plus four one-sensor masks after standardization."""

    X = _require_float_matrix(X_scaled, columns=36, name="X_scaled")
    views = np.empty((len(X), 5, 40), dtype=np.float64)
    for view_index in range(5):
        views[:, view_index, :36] = X
        views[:, view_index, 36:] = 1.0
        if view_index:
            sensor_index = view_index - 1
            start = sensor_index * 9
            views[:, view_index, start : start + 9] = 0.0
            views[:, view_index, 36 + sensor_index] = 0.0
    if not np.isfinite(views).all():
        raise OpenAxesExternalModelV1TrainingError("five-view construction is non-finite")
    return views


def validate_exact_five_views(X_scaled: Any, views: Any) -> None:
    expected = build_exact_five_views(X_scaled)
    observed = np.asarray(views)
    if (
        observed.shape != expected.shape
        or observed.dtype.kind not in "fc"
        or not np.array_equal(observed, expected)
    ):
        raise OpenAxesExternalModelV1TrainingError(
            "full plus four outage views changed"
        )


def within_training_group_derangement(
    train_uids: Sequence[str], training_groups: Sequence[str]
) -> np.ndarray:
    """Return the frozen sorted-UID circular wrong-pair mapping.

    Each scenario group's full-loss marginal is preserved exactly while every
    row is paired with a different UID.  A group with fewer than two rows is
    not silently relaxed.
    """

    rows = len(train_uids)
    uids = _require_exact_text_vector(train_uids, rows=rows, name="train_uids")
    groups = _require_exact_text_vector(training_groups, rows=rows, name="training_groups")
    if len(set(uids.tolist())) != rows:
        raise OpenAxesExternalModelV1TrainingError("derangement UID values are not unique")
    result = np.empty(rows, dtype=np.int64)
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group)
        if len(indices) < 2:
            raise OpenAxesExternalModelV1TrainingError(
                f"training group cannot be deranged: {group}"
            )
        ordered = indices[np.argsort(uids[indices], kind="stable")]
        result[ordered] = np.roll(ordered, -1)
    if np.any(result == np.arange(rows)):
        raise OpenAxesExternalModelV1TrainingError("wrong-pair derangement has a fixed point")
    for group in set(groups.tolist()):
        indices = np.flatnonzero(groups == group)
        if set(result[indices].tolist()) != set(indices.tolist()):
            raise OpenAxesExternalModelV1TrainingError(
                "wrong-pair derangement changed a group marginal"
            )
    return result


def compact_group_codes(training_groups: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    groups = _require_exact_text_vector(
        training_groups, rows=len(training_groups), name="training_groups"
    )
    names = tuple(sorted(set(groups.tolist())))
    if len(names) < 2:
        raise OpenAxesExternalModelV1TrainingError(
            "at least two blocked training groups are required"
        )
    lookup = {name: index for index, name in enumerate(names)}
    return np.asarray([lookup[item] for item in groups], dtype=np.int64), names


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise OpenAxesExternalModelV1TrainingError("PyTorch is required for model training") from exc
    return torch


def build_masked_mlp(*, seed: int, device: str = "cpu") -> Any:
    """Create the exact 40-64-64-3 ReLU backbone from a paired CPU init."""

    if type(seed) is not int or type(device) is not str or not device:
        raise OpenAxesExternalModelV1TrainingError("model seed/device is invalid")
    torch = _torch_module()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(40, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
        )
    return model.to(device)


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    torch = _torch_module()
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            raise OpenAxesExternalModelV1TrainingError("state dict contains a non-tensor")
        value = tensor.detach().cpu().contiguous()
        array = value.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def symmetric_top_quarter(values: Any) -> Any:
    """Fixed-atom top-quarter mean with a symmetric cutoff-tie gradient."""

    torch = _torch_module()
    if not torch.is_tensor(values) or values.ndim != 1 or values.numel() < 1:
        raise OpenAxesExternalModelV1TrainingError("tail atoms must be a nonempty tensor")
    if not bool(torch.isfinite(values).all().item()):
        raise OpenAxesExternalModelV1TrainingError("tail atoms are non-finite")
    atom_count = int(values.numel())
    k = int(math.ceil(atom_count / 4.0))
    sorted_detached, _ = torch.sort(values.detach(), descending=True)
    cutoff = sorted_detached[k - 1]
    strict = values.detach() > cutoff
    ties = values.detach() == cutoff
    strict_count = int(strict.sum().item())
    tie_count = int(ties.sum().item())
    remaining = k - strict_count
    if not 0 <= remaining <= tie_count or tie_count < 1:
        raise OpenAxesExternalModelV1TrainingError("tail cutoff accounting failed")
    tie_fraction = float(remaining) / float(tie_count)
    return (values[strict].sum() + tie_fraction * values[ties].sum()) / float(k)


def two_way_center(values: Any) -> Any:
    torch = _torch_module()
    if not torch.is_tensor(values) or values.ndim != 2 or min(values.shape) < 1:
        raise OpenAxesExternalModelV1TrainingError("interaction atoms must be a matrix")
    return (
        values
        - values.mean(dim=1, keepdim=True)
        - values.mean(dim=0, keepdim=True)
        + values.mean()
    )


def _group_risks(row_losses: Any, group_codes: Any, group_count: int) -> Any:
    torch = _torch_module()
    rows: list[Any] = []
    for group_index in range(group_count):
        mask = group_codes == group_index
        if int(mask.sum().item()) < 1:
            raise OpenAxesExternalModelV1TrainingError("empty training group atom")
        rows.append(row_losses[mask].mean(dim=0))
    return torch.stack(rows, dim=0)


def compute_method_objective(
    row_losses: Any,
    group_codes: Any,
    method_id: str,
    *,
    wrong_pair_index: Any | None = None,
    groupdro_weights: Any | None = None,
) -> dict[str, Any]:
    """Pure backbone-independent objective over N x five row losses."""

    torch = _torch_module()
    if method_id not in OPENAXES_MODEL_METHOD_IDS:
        raise OpenAxesExternalModelV1TrainingError(f"unknown method: {method_id}")
    if (
        not torch.is_tensor(row_losses)
        or row_losses.ndim != 2
        or row_losses.shape[1] != 5
        or row_losses.shape[0] < 1
        or not bool(torch.isfinite(row_losses).all().item())
    ):
        raise OpenAxesExternalModelV1TrainingError("row losses must be finite N x five")
    if (
        not torch.is_tensor(group_codes)
        or group_codes.ndim != 1
        or group_codes.shape[0] != row_losses.shape[0]
        or group_codes.dtype != torch.long
    ):
        raise OpenAxesExternalModelV1TrainingError("group codes must be an N-vector of long")
    unique = torch.unique(group_codes.detach().cpu())
    group_count = int(unique.numel())
    if group_count < 2 or not torch.equal(unique, torch.arange(group_count, dtype=torch.long)):
        raise OpenAxesExternalModelV1TrainingError("group codes must be compact from zero")
    risks = _group_risks(row_losses, group_codes, group_count)
    law = row_losses.new_tensor(MASK_LAW)
    mask_loss = torch.sum(law * row_losses.mean(dim=0))
    zero = row_losses.sum() * 0.0
    output: dict[str, Any] = {
        "mask_loss": mask_loss,
        "tail": zero,
        "interaction_penalty": zero,
        "group_risks": risks,
        "positive_atoms": row_losses.new_empty((0,)),
    }

    if method_id == "full_gradient_only_mask_selected_control":
        objective = row_losses[:, 0].mean()
    elif method_id == "masked_erm":
        objective = mask_loss
    elif method_id in {"group_mean_pord", "paired_pord", "paired_wrong_pair"}:
        atom_rows: list[Any] = []
        for group_index in range(group_count):
            mask = group_codes == group_index
            missing = row_losses[mask, 1:]
            if method_id == "group_mean_pord":
                gap = missing.mean(dim=0) - row_losses[mask, 0].mean().detach()
                positive = torch.relu(gap)
            else:
                if method_id == "paired_wrong_pair":
                    if (
                        wrong_pair_index is None
                        or not torch.is_tensor(wrong_pair_index)
                        or wrong_pair_index.dtype != torch.long
                        or wrong_pair_index.shape != group_codes.shape
                    ):
                        raise OpenAxesExternalModelV1TrainingError(
                            "wrong-pair method requires an exact long mapping"
                        )
                    source = wrong_pair_index[mask]
                    paired_full = row_losses[source, 0].detach()
                else:
                    paired_full = row_losses[mask, 0].detach()
                positive = torch.relu(missing - paired_full[:, None]).mean(dim=0)
            atom_rows.append(positive)
        atoms = torch.stack(atom_rows, dim=0)
        tail = symmetric_top_quarter(atoms.reshape(-1))
        interaction = two_way_center(atoms)
        interaction_penalty = torch.abs(interaction).mean()
        objective = mask_loss + tail + interaction_penalty
        output.update(
            {
                "tail": tail,
                "interaction_penalty": interaction_penalty,
                "interaction": interaction,
                "positive_atoms": atoms,
            }
        )
    elif method_id == "groupdro":
        atom_count = group_count * 5
        if groupdro_weights is None:
            weights = law.repeat(group_count) / float(group_count)
        else:
            weights = groupdro_weights
        if (
            not torch.is_tensor(weights)
            or weights.shape != (atom_count,)
            or not bool(torch.isfinite(weights).all().item())
            or bool((weights <= 0.0).any().item())
            or not math.isclose(float(weights.sum().detach().cpu()), 1.0, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise OpenAxesExternalModelV1TrainingError("GroupDRO weights are invalid")
        objective = torch.sum(weights * risks.reshape(-1))
        output["groupdro_weights"] = weights
    elif method_id == "vrex":
        weights = law.repeat(group_count) / float(group_count)
        flat = risks.reshape(-1)
        weighted_mean = torch.sum(weights * flat)
        variance = torch.sum(weights * torch.square(flat - weighted_mean))
        objective = weighted_mean + VREX_BETA * variance
        output["vrex_variance"] = variance
    elif method_id == "mofe":
        hinge = torch.relu(row_losses[:, 0, None] - row_losses[:, 1:]).mean()
        objective = mask_loss + MOFE_LAMBDA * hinge
        output["mofe_hinge"] = hinge
    else:  # exhaustive guard
        raise AssertionError(method_id)
    if not bool(torch.isfinite(objective).item()):
        raise OpenAxesExternalModelV1TrainingError("objective is non-finite")
    output["loss"] = objective
    return output


def update_groupdro_weights(group_risks: Any, weights: Any) -> Any:
    torch = _torch_module()
    if not torch.is_tensor(group_risks) or group_risks.ndim != 2 or group_risks.shape[1] != 5:
        raise OpenAxesExternalModelV1TrainingError("GroupDRO risk matrix is invalid")
    flat = group_risks.detach().reshape(-1)
    if not torch.is_tensor(weights) or weights.shape != flat.shape:
        raise OpenAxesExternalModelV1TrainingError("GroupDRO weight shape changed")
    logits = torch.log(weights) + GROUPDRO_ETA * flat
    result = torch.softmax(logits, dim=0)
    if not bool(torch.isfinite(result).all().item()):
        raise OpenAxesExternalModelV1TrainingError("GroupDRO update is non-finite")
    return result


def training_checkpoint_score_m(
    predictions_scaled: Any, targets_scaled: Any, target_scale_m: float
) -> tuple[float, tuple[float, float, float, float]]:
    predictions = np.asarray(predictions_scaled, dtype=np.float64)
    targets = _require_float_matrix(targets_scaled, columns=3, name="targets_scaled")
    if (
        predictions.shape != (len(targets), 5, 3)
        or not np.isfinite(predictions).all()
        or type(target_scale_m) is not float
        or not math.isfinite(target_scale_m)
        or target_scale_m <= 0.0
    ):
        raise OpenAxesExternalModelV1TrainingError("checkpoint inputs are invalid")
    rmse = tuple(
        float(
            target_scale_m
            * np.sqrt(
                np.mean(
                    np.sum(np.square(predictions[:, view, :] - targets), axis=1),
                    dtype=np.float64,
                )
            )
        )
        for view in range(1, 5)
    )
    score = 0.5 * (float(np.mean(rmse)) + max(rmse))
    return score, rmse


@dataclass
class TrainingResult:
    method_id: str
    seed: int
    transform: TrainOnlyTransform
    state_dict: Mapping[str, Any]
    initial_state_sha256: str
    checkpoint_state_sha256: str
    view_sha256: str
    checkpoint_update: int
    checkpoint_score_m: float
    updates: int
    history: tuple[dict[str, Any], ...]
    training_group_order: tuple[str, ...]
    checkpoint_selection_scope: str = "training_scenarios_only"
    outer_eval_count: int = 0


def train_model_train_only(
    X_train: Any,
    y_train_m: Any,
    training_groups: Sequence[str],
    train_uids: Sequence[str],
    *,
    method_id: str,
    seed: int,
    updates: int = 400,
    device: str = "cuda:0",
) -> TrainingResult:
    """Fit one method without accepting or inspecting any outer partition."""

    if method_id not in OPENAXES_MODEL_METHOD_IDS:
        raise OpenAxesExternalModelV1TrainingError("method is outside the closed grid")
    if type(seed) is not int or type(updates) is not int or updates < 1:
        raise OpenAxesExternalModelV1TrainingError("seed/update budget is invalid")
    X = _require_float_matrix(X_train, columns=36, name="X_train")
    y = _require_float_matrix(y_train_m, columns=3, name="y_train_m")
    if len(X) != len(y):
        raise OpenAxesExternalModelV1TrainingError("training X/y row mismatch")
    groups_text = _require_exact_text_vector(
        training_groups, rows=len(X), name="training_groups"
    )
    uids = _require_exact_text_vector(train_uids, rows=len(X), name="train_uids")
    transform = fit_train_only_transform(X, y, uids.tolist())
    X_scaled = transform_features(X, transform)
    y_scaled = transform_targets(y, transform)
    views = build_exact_five_views(X_scaled)
    validate_exact_five_views(X_scaled, views)
    group_codes_numpy, group_names = compact_group_codes(groups_text.tolist())
    wrong_numpy = within_training_group_derangement(uids.tolist(), groups_text.tolist())

    torch = _torch_module()
    model = build_masked_mlp(seed=seed, device=device)
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.001
    )
    tensor_views = torch.as_tensor(views, dtype=torch.float32, device=device)
    tensor_targets = torch.as_tensor(y_scaled, dtype=torch.float32, device=device)
    group_codes = torch.as_tensor(group_codes_numpy, dtype=torch.long, device=device)
    wrong_pair = torch.as_tensor(wrong_numpy, dtype=torch.long, device=device)
    law = tensor_views.new_tensor(MASK_LAW)
    groupdro_weights = law.repeat(len(group_names)) / float(len(group_names))
    best_score = math.inf
    best_update = 0
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for update in range(1, updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(tensor_views.reshape(-1, 40)).reshape(len(X), 5, 3)
        row_losses = torch.sum(torch.square(prediction - tensor_targets[:, None, :]), dim=2)
        audit = compute_method_objective(
            row_losses,
            group_codes,
            method_id,
            wrong_pair_index=wrong_pair,
            groupdro_weights=groupdro_weights if method_id == "groupdro" else None,
        )
        audit["loss"].backward()
        optimizer.step()
        if method_id == "groupdro":
            groupdro_weights = update_groupdro_weights(
                audit["group_risks"], groupdro_weights
            )

        model.eval()
        with torch.no_grad():
            checkpoint_prediction = (
                model(tensor_views.reshape(-1, 40))
                .reshape(len(X), 5, 3)
                .detach()
                .cpu()
                .numpy()
            )
        checkpoint_score, outage_rmse = training_checkpoint_score_m(
            checkpoint_prediction, y_scaled, transform.target_scale_m
        )
        if checkpoint_score < best_score:
            best_score = checkpoint_score
            best_update = update
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        history.append(
            {
                "update": update,
                "objective": float(audit["loss"].detach().cpu()),
                "mask_loss": float(audit["mask_loss"].detach().cpu()),
                "tail": float(audit["tail"].detach().cpu()),
                "interaction_penalty": float(
                    audit["interaction_penalty"].detach().cpu()
                ),
                "training_outage_rmse_m": list(outage_rmse),
                "checkpoint_score_m": checkpoint_score,
                "became_checkpoint": best_update == update,
            }
        )
    if best_state is None or best_update < 1 or not math.isfinite(best_score):
        raise OpenAxesExternalModelV1TrainingError("training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    checkpoint_hash = state_dict_sha256(best_state)
    return TrainingResult(
        method_id=method_id,
        seed=seed,
        transform=transform,
        state_dict=best_state,
        initial_state_sha256=initial_hash,
        checkpoint_state_sha256=checkpoint_hash,
        view_sha256=_array_sha256(views),
        checkpoint_update=best_update,
        checkpoint_score_m=best_score,
        updates=updates,
        history=tuple(history),
        training_group_order=group_names,
    )


def predict_outer_views_once(
    training: TrainingResult,
    X_outer: Any,
    *,
    device: str = "cuda:0",
) -> np.ndarray:
    """Replay one frozen checkpoint on all five outer feature-only views."""

    if training.outer_eval_count != 0:
        raise OpenAxesExternalModelV1TrainingError(
            "training result already carries an outer evaluation"
        )
    X = _require_float_matrix(X_outer, columns=36, name="X_outer")
    scaled = transform_features(X, training.transform)
    views = build_exact_five_views(scaled)
    torch = _torch_module()
    model = build_masked_mlp(seed=training.seed, device=device)
    model.load_state_dict(training.state_dict, strict=True)
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(views, dtype=torch.float32, device=device)
        scaled_prediction = (
            model(tensor.reshape(-1, 40))
            .reshape(len(X), 5, 3)
            .detach()
            .cpu()
            .numpy()
        )
    prediction = inverse_targets(scaled_prediction, training.transform)
    if prediction.shape != (len(X), 5, 3) or not np.isfinite(prediction).all():
        raise OpenAxesExternalModelV1TrainingError("outer predictions are invalid")
    return prediction


def evaluate_outer_predictions_once(
    prediction_m: Any,
    truth_m: Any,
) -> dict[str, Any]:
    """Evaluate one already-created outer prediction table exactly once."""

    prediction = np.asarray(prediction_m, dtype=np.float64)
    truth = _require_float_matrix(truth_m, columns=3, name="outer truth")
    if prediction.shape != (len(truth), 5, 3) or not np.isfinite(prediction).all():
        raise OpenAxesExternalModelV1TrainingError("outer prediction/truth mismatch")
    per_view: list[dict[str, Any]] = []
    for view_index, view_id in enumerate(OPENAXES_MODEL_VIEW_IDS):
        distance = np.linalg.norm(prediction[:, view_index, :] - truth, axis=1)
        rmse = float(np.sqrt(np.mean(np.square(distance), dtype=np.float64)))
        per_view.append(
            {
                "view_id": view_id,
                "rows": len(truth),
                "rmse3d_m": rmse,
                "p95_distance_m": float(np.percentile(distance, 95.0)),
                "mean_distance_m": float(np.mean(distance, dtype=np.float64)),
            }
        )
    return {
        "outer_eval_count": 1,
        "views": per_view,
        "finite": True,
    }
