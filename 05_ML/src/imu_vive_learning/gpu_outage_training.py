"""Deterministic one-context/one-seed training for outage-v3.

The frozen GPU-v1 and robust-v2 trainers are intentionally not imported or
modified here.  This module consumes already standardized, complete 43-column
``fusion_v1`` arrays, applies the prospective UID-keyed sensor-mask schedule,
and appends four raw availability bits before fitting one candidate and one
paired seed.  Evaluation always forces one specified IMU outage.

Vive labels are used only for the supervised endpoint target and the
held-out raw-millimetre checkpoint metric.  Segment identifiers and trial
identifiers control balanced sampling and masking respectively; neither is
ever appended to model features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import numbers
import os
from typing import Any

import numpy as np

from .experiment_metrics import regression_metrics
from .experiment_models import derive_context_seed
from .features import FUSION_COLUMNS
from .gpu_robust_training import segment_balanced_batches
from .gpu_v3_backbones import (
    IMU_SENSOR_INDEX_MAP,
    V3_IMU_COUNT,
    V3_IMU_FEATURE_COUNT,
    V3_SENSOR_FEATURE_COUNT,
    build_chain_sensor_message_passing_backbone,
    build_masked_mlp_backbone,
    build_residual_mlp_backbone,
    build_tabm_batchensemble_backbone,
)


OUTAGE_CANDIDATE_IDS = (
    "masked_mlp",
    "residual_mlp",
    "tabm_batchensemble",
    "chain_endpoint_only",
    "chain_masked_reconstruction",
)
MASK_FULL_STATE = -1
MASK_STATE_PROBABILITIES = (0.25, 0.1875, 0.1875, 0.1875, 0.1875)
MAX_OUTAGE_EPOCHS = 400

_EXPECTED_BACKBONES = {
    "masked_mlp": "plain_mlp",
    "residual_mlp": "gated_residual_mlp",
    "tabm_batchensemble": "tabm_style_batchensemble_mlp",
    "chain_endpoint_only": "encoder_conditioned_four_sensor_chain_message_passing",
    "chain_masked_reconstruction": (
        "encoder_conditioned_four_sensor_chain_message_passing"
    ),
}


class OutageTrainingError(RuntimeError):
    """Raised when one outage-v3 seed fit violates its local contract."""


def _matrix(
    values: Any,
    *,
    name: str,
    columns: int,
    rows: int | None = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 2 or len(array) == 0:
        raise OutageTrainingError(f"{name} must be a non-empty matrix")
    if int(array.shape[1]) != int(columns):
        raise OutageTrainingError(
            f"{name} must have {columns} columns; received {array.shape[1]}"
        )
    if rows is not None and len(array) != int(rows):
        raise OutageTrainingError(
            f"{name} must have {rows} rows; received {len(array)}"
        )
    if not np.isfinite(array).all():
        raise OutageTrainingError(f"{name} contains non-finite values")
    return array


def _trial_uids(values: Sequence[Any], *, rows: int) -> tuple[str, ...]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != rows:
        raise OutageTrainingError("trial_uids must contain one value per training row")
    normalized: list[str] = []
    for value in array.tolist():
        if isinstance(value, np.generic):
            value = value.item()
        if value is None:
            raise OutageTrainingError("trial_uids contains a missing value")
        text = str(value)
        if not text:
            raise OutageTrainingError("trial_uids contains an empty value")
        normalized.append(text)
    if len(normalized) != len(set(normalized)):
        raise OutageTrainingError("trial_uids must be unique within a train context")
    return tuple(normalized)


def _segment_ids(values: Sequence[Any], *, rows: int) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != rows:
        raise OutageTrainingError("segment_ids must contain one value per training row")
    normalized: list[str] = []
    for value in array.tolist():
        if isinstance(value, np.generic):
            value = value.item()
        if value is None:
            raise OutageTrainingError("segment_ids contains a missing value")
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise OutageTrainingError("segment_ids contains a non-finite value")
        text = str(value)
        if not text:
            raise OutageTrainingError("segment_ids contains an empty value")
        normalized.append(text)
    return np.asarray(normalized, dtype=object)


def _forced_outage_index(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise OutageTrainingError("forced_outage must be an integer sensor index 0..3")
    index = int(value)
    if index not in range(V3_IMU_COUNT):
        raise OutageTrainingError("forced_outage must be an integer sensor index 0..3")
    return index


def _canonical_mask_context(
    *,
    base_seed: int,
    held_out_segment: Any,
    forced_outage: int,
    paired_seed_index: int,
) -> bytes:
    """Serialize every non-row, non-epoch mask-key field stably."""

    payload = [
        int(base_seed),
        str(held_out_segment),
        int(forced_outage),
        int(paired_seed_index),
    ]
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _uid_mask_keys(
    trial_uids: Sequence[str],
    *,
    base_seed: int,
    held_out_segment: Any,
    forced_outage: int,
    paired_seed_index: int,
) -> np.ndarray:
    context = _canonical_mask_context(
        base_seed=base_seed,
        held_out_segment=held_out_segment,
        forced_outage=forced_outage,
        paired_seed_index=paired_seed_index,
    )
    keys = np.empty(len(trial_uids), dtype=np.uint64)
    for row_index, trial_uid in enumerate(trial_uids):
        digest = hashlib.sha256(
            context + b"\x00" + str(trial_uid).encode("utf-8")
        ).digest()
        keys[row_index] = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return keys


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized uint64 mixer used after the stable SHA-256 row key."""

    with np.errstate(over="ignore"):
        mixed = values.astype(np.uint64, copy=True) + np.uint64(
            0x9E3779B97F4A7C15
        )
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        return mixed ^ (mixed >> np.uint64(31))


def _mask_states_from_uid_keys(row_keys: np.ndarray, *, epoch: int) -> np.ndarray:
    """Apply the epoch portion of the key to pre-hashed UID/context keys."""

    if isinstance(epoch, (bool, np.bool_)) or not isinstance(epoch, numbers.Integral):
        raise ValueError("epoch must be a positive integer")
    if int(epoch) <= 0:
        raise ValueError("epoch must be a positive integer")
    epoch_digest = hashlib.sha256(
        json.dumps(["epoch", int(epoch)], separators=(",", ":")).encode("utf-8")
    ).digest()
    epoch_key = np.uint64(
        int.from_bytes(epoch_digest[:8], byteorder="big", signed=False)
    )
    buckets = (_splitmix64(row_keys ^ epoch_key) >> np.uint64(60)).astype(
        np.int8, copy=False
    )
    states = np.empty(len(buckets), dtype=np.int8)
    states[buckets < 4] = MASK_FULL_STATE
    states[(buckets >= 4) & (buckets < 7)] = 0
    states[(buckets >= 7) & (buckets < 10)] = 1
    states[(buckets >= 10) & (buckets < 13)] = 2
    states[buckets >= 13] = 3
    return states


def deterministic_mask_states(
    trial_uids: Sequence[Any],
    *,
    base_seed: int,
    held_out_segment: Any,
    forced_outage: int,
    paired_seed_index: int,
    epoch: int,
) -> np.ndarray:
    """Return UID-keyed categorical states ``-1, 0, 1, 2, 3``.

    Four of sixteen equally likely high-nibble buckets mean fully available;
    each consecutive set of three buckets drops one physical IMU.  Therefore
    the exact categorical probabilities are 0.25 and four times 0.1875.  The
    candidate identity is deliberately absent from this API and key.
    """

    if isinstance(epoch, (bool, np.bool_)) or not isinstance(epoch, numbers.Integral):
        raise ValueError("epoch must be a positive integer")
    if int(epoch) <= 0:
        raise ValueError("epoch must be a positive integer")
    outage_index = _forced_outage_index(forced_outage)
    if isinstance(paired_seed_index, (bool, np.bool_)) or not isinstance(
        paired_seed_index, numbers.Integral
    ):
        raise ValueError("paired_seed_index must be a non-negative integer")
    seed_index = int(paired_seed_index)
    if seed_index < 0:
        raise ValueError("paired_seed_index must be a non-negative integer")
    normalized = tuple(str(value) for value in trial_uids)
    if not normalized or any(not value for value in normalized):
        raise ValueError("trial_uids must be non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError("trial_uids must be unique")
    row_keys = _uid_mask_keys(
        normalized,
        base_seed=int(base_seed),
        held_out_segment=held_out_segment,
        forced_outage=outage_index,
        paired_seed_index=seed_index,
    )
    return _mask_states_from_uid_keys(row_keys, epoch=int(epoch))


def apply_sensor_masks(
    complete_scaled_features: Any, missing_sensor_states: Sequence[int]
) -> np.ndarray:
    """Zero physical sensor features after scaling and append availability.

    ``missing_sensor_states`` contains ``-1`` for complete rows or the missing
    physical sensor index 0..3.  The literal ``IMU_SENSOR_INDEX_MAP`` performs
    the signal-family-to-sensor permutation; a blind reshape is never used.
    """

    features = _matrix(
        complete_scaled_features,
        name="complete_scaled_features",
        columns=V3_SENSOR_FEATURE_COUNT,
    )
    states = np.asarray(missing_sensor_states)
    if states.shape != (len(features),):
        raise OutageTrainingError(
            "missing_sensor_states must contain one value per feature row"
        )
    if not np.issubdtype(states.dtype, np.integer):
        raise OutageTrainingError("missing_sensor_states must contain integers")
    states = states.astype(np.int8, copy=False)
    if not np.isin(states, (MASK_FULL_STATE, 0, 1, 2, 3)).all():
        raise OutageTrainingError("mask state must be -1 or a sensor index 0..3")

    masked = features.astype(np.float32, copy=True)
    availability = np.ones((len(features), V3_IMU_COUNT), dtype=np.float32)
    for sensor_index, feature_indices in enumerate(IMU_SENSOR_INDEX_MAP):
        rows = np.flatnonzero(states == sensor_index)
        if len(rows) == 0:
            continue
        masked[np.ix_(rows, np.asarray(feature_indices, dtype=np.int64))] = 0.0
        availability[rows, sensor_index] = 0.0
    result = np.concatenate((masked, availability), axis=1)
    if not np.isfinite(result).all():
        raise OutageTrainingError("masked features contain non-finite values")
    return result


def balanced_member_mse(
    torch: Any,
    member_predictions: Any,
    truth: Any,
    segment_ids: Sequence[Any],
) -> Any:
    """Average supervised MSE equally over segments and ensemble members."""

    if int(member_predictions.ndim) == 2:
        member_predictions = member_predictions.unsqueeze(1)
    if int(member_predictions.ndim) != 3 or int(member_predictions.shape[-1]) != 3:
        raise OutageTrainingError(
            "member_predictions must have shape [rows, members, 3]"
        )
    if tuple(truth.shape) != (int(member_predictions.shape[0]), 3):
        raise OutageTrainingError("truth must have shape [rows, 3]")
    segments = np.asarray(segment_ids, dtype=object)
    if segments.shape != (int(member_predictions.shape[0]),):
        raise OutageTrainingError("segment_ids must align with member predictions")
    groups = tuple(sorted(set(str(value) for value in segments.tolist())))
    if not groups:
        raise OutageTrainingError("at least one training segment is required")
    losses: list[Any] = []
    normalized = np.asarray([str(value) for value in segments.tolist()], dtype=object)
    for group in groups:
        group_mask = torch.as_tensor(
            normalized == group,
            dtype=torch.bool,
            device=member_predictions.device,
        )
        if not bool(torch.any(group_mask).item()):
            raise OutageTrainingError("balanced endpoint loss omitted a segment")
        # First average rows and XYZ separately for each member.  The final
        # mean supervises every TabM member rather than only the member mean.
        losses.append(
            torch.square(member_predictions[group_mask] - truth[group_mask].unsqueeze(1))
            .mean(dim=(0, 2))
            .mean()
        )
    loss = torch.stack(losses).mean()
    if not bool(torch.isfinite(loss).item()):
        raise OutageTrainingError("endpoint loss is non-finite")
    return loss


def deliberate_missing_reconstruction_mse(
    torch: Any,
    reconstruction: Any,
    complete_standardized_sensor_targets: Any,
    missing_sensor_states: Any,
) -> Any:
    """Return 10-D MSE only for deliberately hidden training nodes.

    Complete training rows contribute zero.  Evaluation/challenge code never
    calls this function because a genuinely absent sensor has no target.
    """

    expected_shape = tuple(complete_standardized_sensor_targets.shape)
    if tuple(reconstruction.shape) != expected_shape or expected_shape[-2:] != (
        V3_IMU_COUNT,
        V3_IMU_FEATURE_COUNT,
    ):
        raise OutageTrainingError(
            "reconstruction and target must both have shape [rows, 4, 10]"
        )
    if tuple(missing_sensor_states.shape) != (expected_shape[0],):
        raise OutageTrainingError("missing_sensor_states must have shape [rows]")
    deliberate = missing_sensor_states >= 0
    if not bool(torch.any(deliberate).item()):
        return reconstruction.sum() * 0.0
    row_indices = torch.nonzero(deliberate, as_tuple=False).squeeze(1)
    sensor_indices = missing_sensor_states[row_indices].to(dtype=torch.long)
    predicted = reconstruction[row_indices, sensor_indices, :]
    target = complete_standardized_sensor_targets[row_indices, sensor_indices, :]
    loss = torch.square(predicted - target).mean()
    if not bool(torch.isfinite(loss).item()):
        raise OutageTrainingError("masked reconstruction loss is non-finite")
    return loss


def _candidate_settings(candidate: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id", ""))
    if candidate_id not in OUTAGE_CANDIDATE_IDS:
        raise OutageTrainingError(
            f"unknown trainable outage candidate {candidate_id!r}"
        )
    if str(candidate.get("kind", "trainable")) != "trainable":
        raise OutageTrainingError("outage seed trainer accepts trainable candidates only")
    expected_backbone = _EXPECTED_BACKBONES[candidate_id]
    if str(candidate.get("backbone", "")) != expected_backbone:
        raise OutageTrainingError(
            f"candidate {candidate_id!r} must use backbone {expected_backbone!r}"
        )

    reconstruction_weight = float(
        candidate.get("masked_reconstruction_loss_weight", 0.0)
    )
    if not math.isfinite(reconstruction_weight) or reconstruction_weight < 0.0:
        raise OutageTrainingError(
            "masked reconstruction loss weight must be finite and non-negative"
        )
    if candidate_id == "chain_masked_reconstruction":
        if reconstruction_weight <= 0.0:
            raise OutageTrainingError(
                "chain_masked_reconstruction requires a positive loss weight"
            )
    elif reconstruction_weight != 0.0:
        raise OutageTrainingError(
            "only chain_masked_reconstruction may use reconstruction loss"
        )
    return {
        "candidate_id": candidate_id,
        "reconstruction_weight": reconstruction_weight,
    }


def _training_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise OutageTrainingError("config.training must be a mapping")
    if str(training.get("optimizer", "")) != "AdamW":
        raise OutageTrainingError("outage-v3 requires AdamW")
    if str(training.get("dtype", "")) != "float32":
        raise OutageTrainingError("outage-v3 requires deterministic float32 training")
    if not bool(training.get("deterministic_algorithms", False)):
        raise OutageTrainingError("outage-v3 requires deterministic_algorithms=true")
    learning_rate = float(training.get("learning_rate", math.nan))
    weight_decay = float(training.get("weight_decay", math.nan))
    batch_size = int(training.get("batch_size", 0))
    epoch_budget = int(training.get("epoch_budget", 0))
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise OutageTrainingError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise OutageTrainingError("weight_decay must be finite and non-negative")
    if batch_size <= 0:
        raise OutageTrainingError("batch_size must be positive")
    if epoch_budget <= 0 or epoch_budget > MAX_OUTAGE_EPOCHS:
        raise OutageTrainingError("epoch_budget must lie in [1, 400]")
    paired_seed_values = training.get("paired_seed_values")
    if (
        not isinstance(paired_seed_values, Sequence)
        or isinstance(paired_seed_values, (str, bytes))
        or len(paired_seed_values) != 3
    ):
        raise OutageTrainingError("training must define exactly three paired seeds")
    paired_seeds = tuple(int(value) for value in paired_seed_values)
    if len(set(paired_seeds)) != 3:
        raise OutageTrainingError("paired model seeds must be distinct")
    if int(training.get("ensemble_seed_count", -1)) != 3:
        raise OutageTrainingError("outage-v3 requires three ensemble seeds")
    contract = config.get("contract")
    if not isinstance(contract, Mapping) or "base_seed" not in contract:
        raise OutageTrainingError("config.contract.base_seed is required")
    return {
        "base_seed": int(contract["base_seed"]),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epoch_budget": epoch_budget,
        "paired_seed_values": paired_seeds,
    }


def _validate_paired_model_seed(
    *, training: Mapping[str, Any], paired_seed_index: int, model_seed: int
) -> None:
    seeds = tuple(int(value) for value in training["paired_seed_values"])
    if paired_seed_index not in range(len(seeds)):
        raise OutageTrainingError("paired_seed_index must be one of 0, 1, 2")
    if isinstance(model_seed, (bool, np.bool_)) or not isinstance(
        model_seed, numbers.Integral
    ):
        raise OutageTrainingError("model_seed must be an integer")
    if int(model_seed) != seeds[paired_seed_index]:
        raise OutageTrainingError("model_seed does not match its paired seed index")


def _sorted_uid_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(str(value) for value in values)).encode()).hexdigest()


def _validate_preprocessor_fit_binding(
    preprocessor: Any,
    trial_uids: Sequence[str],
    *,
    require_no_forbidden_uids: bool,
) -> None:
    audit = getattr(preprocessor, "leakage_audit_", None)
    if not isinstance(audit, Mapping):
        raise OutageTrainingError("preprocessor leakage audit is missing")
    expected_hash = _sorted_uid_sha256(trial_uids)
    if (
        audit.get("fit_scope") != "real_train_only"
        or int(audit.get("fit_row_count", -1)) != len(trial_uids)
        or str(audit.get("fit_trial_uid_sha256", "")) != expected_hash
        or int(audit.get("forbidden_uid_overlap_count", -1)) != 0
    ):
        raise OutageTrainingError("preprocessor is not bound to the supplied train rows")
    if require_no_forbidden_uids and int(audit.get("forbidden_uid_count", -1)) != 0:
        raise OutageTrainingError("final preprocessor must have no forbidden UID set")


def _build_candidate(torch: Any, candidate: Mapping[str, Any]) -> Any:
    candidate_id = str(candidate["candidate_id"])
    if candidate_id == "masked_mlp":
        return build_masked_mlp_backbone(
            torch,
            hidden_layer_sizes=tuple(candidate.get("hidden_layer_sizes", (64, 64))),
        )
    if candidate_id == "residual_mlp":
        return build_residual_mlp_backbone(
            torch,
            width=int(candidate.get("width", 192)),
            depth=int(candidate.get("depth", 4)),
            expansion=int(candidate.get("expansion", 2)),
            dropout=float(candidate.get("dropout", 0.0)),
        )
    if candidate_id == "tabm_batchensemble":
        return build_tabm_batchensemble_backbone(
            torch,
            width=int(candidate.get("width", 128)),
            depth=int(candidate.get("depth", 3)),
            expansion=int(candidate.get("expansion", 2)),
            ensemble_size=int(candidate.get("ensemble_size", 8)),
            dropout=float(candidate.get("dropout", 0.0)),
            rank1_init_std=float(candidate.get("rank1_init_std", 0.1)),
        )
    if candidate_id in {"chain_endpoint_only", "chain_masked_reconstruction"}:
        return build_chain_sensor_message_passing_backbone(
            torch,
            width=int(candidate.get("width", 96)),
            depth=int(candidate.get("depth", 4)),
            expansion=int(candidate.get("expansion", 2)),
            dropout=float(candidate.get("dropout", 0.0)),
        )
    raise AssertionError(candidate_id)  # guarded by _candidate_settings


def _complete_sensor_targets(torch: Any, complete_features: Any) -> Any:
    return torch.stack(
        [complete_features[:, list(indices)] for indices in IMU_SENSOR_INDEX_MAP],
        dim=1,
    )


def _predict_raw(
    model: Any,
    features: Any,
    *,
    torch: Any,
    preprocessor: Any,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        members = model.forward_members(features)
        if (
            int(members.ndim) != 3
            or int(members.shape[0]) != int(features.shape[0])
            or int(members.shape[2]) != 3
        ):
            raise OutageTrainingError("candidate emitted an invalid member prediction")
        if not bool(torch.isfinite(members).all().item()):
            raise OutageTrainingError("candidate emitted non-finite member predictions")
        scaled = members.mean(dim=1).detach().cpu().numpy().astype(np.float64)
    try:
        raw = np.asarray(preprocessor.inverse_targets(scaled), dtype=np.float64)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise OutageTrainingError("failed to inverse-transform endpoint prediction") from exc
    if raw.shape != (len(scaled), 3) or not np.isfinite(raw).all():
        raise OutageTrainingError("inverse-transformed prediction is invalid")
    return raw


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    if not state or any(not bool(value.isfinite().all().item()) for value in state.values()):
        raise OutageTrainingError("candidate state_dict contains non-finite tensors")
    return state


def _configure_determinism(torch: Any, *, model_seed: int) -> None:
    torch.manual_seed(int(model_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(model_seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_outage_seed(
    *,
    X_train: Any,
    y_train: Any,
    segment_ids: Sequence[Any],
    trial_uids: Sequence[Any],
    X_eval: Any,
    y_eval_raw: Any,
    preprocessor: Any,
    held_out_segment: Any,
    forced_outage: int,
    paired_seed_index: int,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
    model_seed: int,
    device: str,
) -> dict[str, Any]:
    """Fit one outage-v3 candidate/context/paired seed deterministically.

    Inputs are complete, already standardized 43-column arrays.  The returned
    checkpoint is selected by minimum forced-outage held-out raw RMSE across
    the fixed epoch budget.  Evaluation labels affect only that checkpoint
    choice and its metrics; they never enter an optimizer loss.
    """

    # cuBLAS reads this before creating a CUDA handle.  ``setdefault`` respects
    # an equivalent caller-level deterministic workspace choice.
    runtime = config.get("runtime", {})
    workspace = str(runtime.get("cublas_workspace_config", ":4096:8"))
    if workspace not in {":4096:8", ":16:8"}:
        raise OutageTrainingError("unsupported deterministic cuBLAS workspace config")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", workspace)

    import torch

    candidate_settings = _candidate_settings(candidate)
    training = _training_settings(config)
    outage_index = _forced_outage_index(forced_outage)
    if isinstance(paired_seed_index, (bool, np.bool_)) or not isinstance(
        paired_seed_index, numbers.Integral
    ):
        raise OutageTrainingError("paired_seed_index must be a non-negative integer")
    seed_index = int(paired_seed_index)
    _validate_paired_model_seed(
        training=training,
        paired_seed_index=seed_index,
        model_seed=model_seed,
    )

    X_train_array = _matrix(
        X_train, name="X_train", columns=V3_SENSOR_FEATURE_COUNT
    )
    y_train_array = _matrix(
        y_train, name="y_train", columns=3, rows=len(X_train_array)
    )
    train_segments = _segment_ids(segment_ids, rows=len(X_train_array))
    train_uids = _trial_uids(trial_uids, rows=len(X_train_array))
    mask_row_keys = _uid_mask_keys(
        train_uids,
        base_seed=int(training["base_seed"]),
        held_out_segment=held_out_segment,
        forced_outage=outage_index,
        paired_seed_index=seed_index,
    )
    X_eval_array = _matrix(X_eval, name="X_eval", columns=V3_SENSOR_FEATURE_COUNT)
    y_eval_array = _matrix(
        y_eval_raw,
        name="y_eval_raw",
        columns=3,
        rows=len(X_eval_array),
        dtype=np.float64,
    )
    if preprocessor is None or not hasattr(preprocessor, "inverse_targets"):
        raise OutageTrainingError("a fitted TrainOnlyPreprocessor is required")
    if not bool(getattr(preprocessor, "is_fitted", False)):
        raise OutageTrainingError("TrainOnlyPreprocessor must be fitted")
    feature_names = tuple(getattr(preprocessor, "feature_names", ()))
    if feature_names != FUSION_COLUMNS:
        raise OutageTrainingError(
            "preprocessor must describe the exact ordered fusion_v1 features"
        )
    _validate_preprocessor_fit_binding(
        preprocessor,
        train_uids,
        require_no_forbidden_uids=False,
    )
    lowered_names = tuple(str(name).lower() for name in feature_names)
    if any("segment" in name or "trial_uid" in name for name in lowered_names):
        raise OutageTrainingError("identifiers must never be model features")
    try:
        train_truth_raw = np.asarray(
            preprocessor.inverse_targets(y_train_array.astype(np.float64)),
            dtype=np.float64,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise OutageTrainingError("failed to inverse-transform training targets") from exc
    if train_truth_raw.shape != (len(X_train_array), 3) or not np.isfinite(
        train_truth_raw
    ).all():
        raise OutageTrainingError("inverse-transformed training targets are invalid")

    resolved_device = torch.device(str(device))
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise OutageTrainingError("CUDA outage training requested without an available GPU")
    _configure_determinism(torch, model_seed=int(model_seed))
    model = _build_candidate(torch, candidate).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    complete_train = torch.as_tensor(
        X_train_array, dtype=torch.float32, device=resolved_device
    )
    train_targets = torch.as_tensor(
        y_train_array, dtype=torch.float32, device=resolved_device
    )
    reconstruction_targets = _complete_sensor_targets(torch, complete_train)
    forced_eval_states = np.full(len(X_eval_array), outage_index, dtype=np.int8)
    masked_eval_array = apply_sensor_masks(X_eval_array, forced_eval_states)
    masked_eval = torch.as_tensor(
        masked_eval_array, dtype=torch.float32, device=resolved_device
    )

    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    groups = tuple(sorted(set(train_segments.tolist())))
    if int(training["batch_size"]) < len(groups):
        raise OutageTrainingError("batch_size must be at least the segment count")

    for epoch in range(1, int(training["epoch_budget"]) + 1):
        states = _mask_states_from_uid_keys(mask_row_keys, epoch=epoch)
        masked_train_array = apply_sensor_masks(X_train_array, states)
        masked_train = torch.as_tensor(
            masked_train_array, dtype=torch.float32, device=resolved_device
        )
        states_tensor = torch.as_tensor(
            states, dtype=torch.int64, device=resolved_device
        )
        batches = segment_balanced_batches(
            train_segments,
            batch_size=int(training["batch_size"]),
            seed=derive_context_seed(
                int(training["base_seed"]),
                "outage_v3_segment_balanced_epoch",
                str(held_out_segment),
                outage_index,
                seed_index,
                epoch,
            ),
        )
        objective_mass = 0.0
        endpoint_mass = 0.0
        reconstruction_mass = 0.0
        batch_count = 0
        model.train()
        for batch_cpu in batches:
            indices = torch.as_tensor(
                batch_cpu, dtype=torch.long, device=resolved_device
            )
            optimizer.zero_grad(set_to_none=True)
            batch_features = masked_train[indices]
            if candidate_settings["candidate_id"] == "chain_masked_reconstruction":
                endpoint_prediction, reconstruction = model.forward_with_aux(
                    batch_features
                )
                member_predictions = endpoint_prediction.unsqueeze(1)
            else:
                member_predictions = model.forward_members(batch_features)
                reconstruction = None
            endpoint_loss = balanced_member_mse(
                torch,
                member_predictions,
                train_targets[indices],
                train_segments[batch_cpu],
            )
            if reconstruction is None:
                reconstruction_loss = endpoint_loss.detach() * 0.0
            else:
                reconstruction_loss = deliberate_missing_reconstruction_mse(
                    torch,
                    reconstruction,
                    reconstruction_targets[indices],
                    states_tensor[indices],
                )
            objective = endpoint_loss + float(
                candidate_settings["reconstruction_weight"]
            ) * reconstruction_loss
            if not bool(torch.isfinite(objective).item()):
                raise OutageTrainingError("training objective is non-finite")
            objective.backward()
            try:
                # An infinite maximum leaves finite gradients unchanged while
                # PyTorch's fused norm check rejects NaN/Inf with one device
                # synchronization instead of one synchronization per tensor.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=math.inf,
                    error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                raise OutageTrainingError("candidate gradient is non-finite") from exc
            optimizer.step()
            objective_mass += float(objective.detach().item())
            endpoint_mass += float(endpoint_loss.detach().item())
            reconstruction_mass += float(reconstruction_loss.detach().item())
            batch_count += 1
        if batch_count <= 0:
            raise OutageTrainingError("segment-balanced epoch produced no batches")

        validation_prediction = _predict_raw(
            model, masked_eval, torch=torch, preprocessor=preprocessor
        )
        validation_metrics = regression_metrics(y_eval_array, validation_prediction)
        validation_score = float(validation_metrics["rmse_3d_mm"])
        if not math.isfinite(validation_score):
            raise OutageTrainingError("held-out validation RMSE is non-finite")
        checkpoint_improved = validation_score < best_score
        if checkpoint_improved:
            best_score = validation_score
            best_epoch = epoch
            best_state = _cpu_state_dict(model)

        counts = {
            state: int(np.count_nonzero(states == state))
            for state in (MASK_FULL_STATE, 0, 1, 2, 3)
        }
        history.append(
            {
                "epoch": epoch,
                "scaled_objective_loss": objective_mass / batch_count,
                "scaled_endpoint_member_mse": endpoint_mass / batch_count,
                "scaled_masked_reconstruction_mse": (
                    reconstruction_mass / batch_count
                ),
                "validation_rmse_3d_mm": validation_score,
                "validation_p95_3d_error_mm": float(
                    validation_metrics["p95_3d_error_mm"]
                ),
                "checkpoint_improved": bool(checkpoint_improved),
                "best_epoch_so_far": best_epoch,
                "best_validation_rmse_3d_mm": best_score,
                "mask_schedule_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
                "mask_full_row_count": counts[MASK_FULL_STATE],
                "mask_drop_imu0_row_count": counts[0],
                "mask_drop_imu1_row_count": counts[1],
                "mask_drop_imu2_row_count": counts[2],
                "mask_drop_imu3_row_count": counts[3],
            }
        )

    if best_state is None or best_epoch <= 0 or not math.isfinite(best_score):
        raise OutageTrainingError("checkpoint selection found no finite state")
    model.load_state_dict(best_state, strict=True)
    selected_state = _cpu_state_dict(model)
    eval_prediction_raw = _predict_raw(
        model, masked_eval, torch=torch, preprocessor=preprocessor
    )
    eval_metrics = regression_metrics(y_eval_array, eval_prediction_raw)
    if not math.isclose(
        float(eval_metrics["rmse_3d_mm"]), best_score, rel_tol=0.0, abs_tol=1e-9
    ):
        raise OutageTrainingError("restored checkpoint does not reproduce best RMSE")

    complete_train_states = np.full(
        len(X_train_array), MASK_FULL_STATE, dtype=np.int8
    )
    complete_train_input = torch.as_tensor(
        apply_sensor_masks(X_train_array, complete_train_states),
        dtype=torch.float32,
        device=resolved_device,
    )
    train_prediction_raw = _predict_raw(
        model, complete_train_input, torch=torch, preprocessor=preprocessor
    )
    train_metrics = regression_metrics(train_truth_raw, train_prediction_raw)
    if not all(
        math.isfinite(float(value))
        for metrics in (eval_metrics, train_metrics)
        for value in metrics.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    ):
        raise OutageTrainingError("final outage metrics contain non-finite values")

    result = {
        "candidate_id": candidate_settings["candidate_id"],
        "model_seed": int(model_seed),
        "paired_seed_index": seed_index,
        "held_out_segment": str(held_out_segment),
        "forced_outage": outage_index,
        "trained_epochs": int(training["epoch_budget"]),
        "best_epoch": best_epoch,
        "best_validation_rmse_3d_mm": best_score,
        "history": history,
        "state_dict": selected_state,
        "eval_prediction_raw_mm": eval_prediction_raw,
        "eval_metrics": eval_metrics,
        "train_prediction_raw_mm": train_prediction_raw,
        "train_metrics": train_metrics,
    }
    del model, optimizer
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def transform_outage_features(
    *, preprocessor: Any, X_raw: Any, availability: Any
) -> np.ndarray:
    """Apply frozen development statistics without reading absent raw values.

    Available sensor values and the three encoders must be finite.  Every raw
    slot for an unavailable sensor must be a NaN sentinel; those slots are set
    directly to standardized zero, so no recorded-but-forbidden value can
    affect challenge inference.
    """

    raw = np.asarray(X_raw, dtype=np.float64)
    if raw.ndim != 2 or len(raw) == 0 or raw.shape[1] != V3_SENSOR_FEATURE_COUNT:
        raise OutageTrainingError("X_raw must be a non-empty 43-column matrix")
    masks = _matrix(
        availability,
        name="availability",
        columns=V3_IMU_COUNT,
        rows=len(raw),
        dtype=np.float64,
    )
    if not np.isin(masks, (0.0, 1.0)).all():
        raise OutageTrainingError("availability must contain only zero or one")
    if preprocessor is None or not bool(getattr(preprocessor, "is_fitted", False)):
        raise OutageTrainingError("a fitted TrainOnlyPreprocessor is required")
    if tuple(getattr(preprocessor, "feature_names", ())) != FUSION_COLUMNS:
        raise OutageTrainingError(
            "preprocessor must describe the exact ordered fusion_v1 features"
        )
    means = np.asarray(getattr(preprocessor, "feature_mean_", None), dtype=np.float64)
    scales = np.asarray(getattr(preprocessor, "feature_scale_", None), dtype=np.float64)
    if means.shape != (V3_SENSOR_FEATURE_COUNT,) or scales.shape != (
        V3_SENSOR_FEATURE_COUNT,
    ):
        raise OutageTrainingError("preprocessor feature statistics must have length 43")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(
        scales <= 0.0
    ):
        raise OutageTrainingError("preprocessor feature statistics are invalid")
    if not np.isfinite(raw[:, :3]).all():
        raise OutageTrainingError("raw encoder features contain non-finite values")

    allowed_finite = np.zeros(raw.shape, dtype=bool)
    allowed_finite[:, :3] = True
    for sensor_index, feature_indices in enumerate(IMU_SENSOR_INDEX_MAP):
        columns = np.asarray(feature_indices, dtype=np.int64)
        available_rows = masks[:, sensor_index] == 1.0
        unavailable_rows = ~available_rows
        if available_rows.any() and not np.isfinite(
            raw[np.ix_(available_rows, columns)]
        ).all():
            raise OutageTrainingError(
                f"available IMU{sensor_index} raw features contain non-finite values"
            )
        if unavailable_rows.any() and not np.isnan(
            raw[np.ix_(unavailable_rows, columns)]
        ).all():
            raise OutageTrainingError(
                f"unavailable IMU{sensor_index} raw features must be NaN sentinels"
            )
        allowed_finite[np.ix_(available_rows, columns)] = True
    if not np.isfinite(raw[allowed_finite]).all():
        raise OutageTrainingError("available raw features contain non-finite values")

    scaled = np.zeros(raw.shape, dtype=np.float64)
    scaled[allowed_finite] = ((raw - means) / scales)[allowed_finite]
    if not np.isfinite(scaled).all():
        raise OutageTrainingError("standardized outage features are non-finite")
    return scaled.astype(np.float32)


def _explicit_availability_input(
    scaled_features: Any, availability: Any
) -> np.ndarray:
    """Append a validated availability mask to standardized sensor features."""

    features = _matrix(
        scaled_features, name="scaled_features", columns=V3_SENSOR_FEATURE_COUNT
    )
    masks = _matrix(
        availability,
        name="availability",
        columns=V3_IMU_COUNT,
        rows=len(features),
    )
    if not np.isin(masks, (0.0, 1.0)).all():
        raise OutageTrainingError("availability must contain only zero or one")
    for sensor_index, feature_indices in enumerate(IMU_SENSOR_INDEX_MAP):
        unavailable_rows = masks[:, sensor_index] == 0.0
        if unavailable_rows.any() and not np.equal(
            features[np.ix_(unavailable_rows, np.asarray(feature_indices))], 0.0
        ).all():
            raise OutageTrainingError(
                "standardized unavailable-sensor features must be exactly zero"
            )
    return np.concatenate(
        (features.astype(np.float32, copy=False), masks.astype(np.float32, copy=False)),
        axis=1,
    )


def fit_outage_seed_fixed_epochs(
    *,
    X_train: Any,
    y_train: Any,
    segment_ids: Sequence[Any],
    trial_uids: Sequence[Any],
    preprocessor: Any,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
    model_seed: int,
    paired_seed_index: int,
    epochs: int,
    mask_held_out_segment: Any,
    mask_forced_outage: int,
    device: str,
) -> dict[str, Any]:
    """Refit one selected seed on all development rows for exactly ``epochs``.

    This final-refit path has no evaluation input and performs no checkpoint
    selection.  Its returned state is the exact final epoch state.  The two
    mask-context fields are explicit so the runner must consume the frozen
    final-refit key fields instead of silently inventing a random schedule.
    """

    runtime = config.get("runtime", {})
    workspace = str(runtime.get("cublas_workspace_config", ":4096:8"))
    if workspace not in {":4096:8", ":16:8"}:
        raise OutageTrainingError("unsupported deterministic cuBLAS workspace config")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", workspace)

    import torch

    candidate_settings = _candidate_settings(candidate)
    training = _training_settings(config)
    outage_index = _forced_outage_index(mask_forced_outage)
    if str(mask_held_out_segment) != "final_refit_all_development":
        raise OutageTrainingError("final-refit mask held-out token changed")
    final_refit = config.get("final_refit")
    if not isinstance(final_refit, Mapping):
        raise OutageTrainingError("config.final_refit must be a mapping")
    if final_refit.get("mask_draw_held_out_segment_token") != str(
        mask_held_out_segment
    ):
        raise OutageTrainingError("final-refit mask context differs from config")
    if final_refit.get("mask_draw_forced_outage_sensor") != "imu1_link_root":
        raise OutageTrainingError("final-refit mask outage field differs from config")
    if outage_index != 1:
        raise OutageTrainingError("final-refit frozen mask outage index must be IMU1")
    if final_refit.get("mask_draw_reuses_training_key_recipe") is not True:
        raise OutageTrainingError("final-refit mask key recipe is not enabled")
    if final_refit.get("exact_rounded_median_epoch_final_state") is not True:
        raise OutageTrainingError("final-refit exact final-epoch rule is not enabled")

    if isinstance(epochs, (bool, np.bool_)) or not isinstance(
        epochs, numbers.Integral
    ):
        raise OutageTrainingError("epochs must be an integer")
    resolved_epochs = int(epochs)
    if resolved_epochs <= 0 or resolved_epochs > int(training["epoch_budget"]):
        raise OutageTrainingError("epochs must lie within the frozen epoch budget")
    if isinstance(paired_seed_index, (bool, np.bool_)) or not isinstance(
        paired_seed_index, numbers.Integral
    ):
        raise OutageTrainingError("paired_seed_index must be a non-negative integer")
    seed_index = int(paired_seed_index)
    _validate_paired_model_seed(
        training=training,
        paired_seed_index=seed_index,
        model_seed=model_seed,
    )

    X_train_array = _matrix(
        X_train, name="X_train", columns=V3_SENSOR_FEATURE_COUNT
    )
    y_train_array = _matrix(
        y_train, name="y_train", columns=3, rows=len(X_train_array)
    )
    train_segments = _segment_ids(segment_ids, rows=len(X_train_array))
    train_uids = _trial_uids(trial_uids, rows=len(X_train_array))
    row_roles = config.get("row_roles")
    if not isinstance(row_roles, Mapping) or not isinstance(
        row_roles.get("development"), Mapping
    ):
        raise OutageTrainingError("config.row_roles.development is required")
    development_role = row_roles["development"]
    expected_segments = tuple(
        str(int(value)) for value in development_role.get("segments", ())
    )
    actual_segments = tuple(sorted(set(train_segments.tolist())))
    if actual_segments != tuple(sorted(expected_segments)):
        raise OutageTrainingError("final refit does not contain the exact development segments")
    if len(X_train_array) != int(development_role.get("expected_rows", -1)):
        raise OutageTrainingError("final refit development row count changed")
    raw_expected_counts = development_role.get("expected_segment_rows")
    if not isinstance(raw_expected_counts, Mapping):
        raise OutageTrainingError("final refit expected segment counts are missing")
    expected_counts = {
        str(int(segment)): int(count)
        for segment, count in raw_expected_counts.items()
    }
    actual_counts = {
        segment: int(np.count_nonzero(train_segments == segment))
        for segment in actual_segments
    }
    if actual_counts != expected_counts:
        raise OutageTrainingError("final refit development segment counts changed")
    expected_uid_hash = str(development_role.get("sorted_trial_uid_sha256", ""))
    if _sorted_uid_sha256(train_uids) != expected_uid_hash:
        raise OutageTrainingError("final refit development UID identity changed")
    mask_row_keys = _uid_mask_keys(
        train_uids,
        base_seed=int(training["base_seed"]),
        held_out_segment=mask_held_out_segment,
        forced_outage=outage_index,
        paired_seed_index=seed_index,
    )
    if preprocessor is None or not hasattr(preprocessor, "inverse_targets"):
        raise OutageTrainingError("a fitted TrainOnlyPreprocessor is required")
    if not bool(getattr(preprocessor, "is_fitted", False)):
        raise OutageTrainingError("TrainOnlyPreprocessor must be fitted")
    if tuple(getattr(preprocessor, "feature_names", ())) != FUSION_COLUMNS:
        raise OutageTrainingError(
            "preprocessor must describe the exact ordered fusion_v1 features"
        )
    _validate_preprocessor_fit_binding(
        preprocessor,
        train_uids,
        require_no_forbidden_uids=True,
    )
    try:
        train_truth_raw = np.asarray(
            preprocessor.inverse_targets(y_train_array.astype(np.float64)),
            dtype=np.float64,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise OutageTrainingError("failed to inverse-transform training targets") from exc

    resolved_device = torch.device(str(device))
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise OutageTrainingError("CUDA outage training requested without an available GPU")
    _configure_determinism(torch, model_seed=int(model_seed))
    model = _build_candidate(torch, candidate).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    complete_train = torch.as_tensor(
        X_train_array, dtype=torch.float32, device=resolved_device
    )
    train_targets = torch.as_tensor(
        y_train_array, dtype=torch.float32, device=resolved_device
    )
    reconstruction_targets = _complete_sensor_targets(torch, complete_train)
    groups = tuple(sorted(set(train_segments.tolist())))
    if int(training["batch_size"]) < len(groups):
        raise OutageTrainingError("batch_size must be at least the segment count")

    history: list[dict[str, Any]] = []
    for epoch in range(1, resolved_epochs + 1):
        states = _mask_states_from_uid_keys(mask_row_keys, epoch=epoch)
        masked_train = torch.as_tensor(
            apply_sensor_masks(X_train_array, states),
            dtype=torch.float32,
            device=resolved_device,
        )
        states_tensor = torch.as_tensor(
            states, dtype=torch.int64, device=resolved_device
        )
        batches = segment_balanced_batches(
            train_segments,
            batch_size=int(training["batch_size"]),
            seed=derive_context_seed(
                int(training["base_seed"]),
                "outage_v3_segment_balanced_epoch",
                str(mask_held_out_segment),
                outage_index,
                seed_index,
                epoch,
            ),
        )
        objective_mass = 0.0
        endpoint_mass = 0.0
        reconstruction_mass = 0.0
        batch_count = 0
        model.train()
        for batch_cpu in batches:
            indices = torch.as_tensor(
                batch_cpu, dtype=torch.long, device=resolved_device
            )
            optimizer.zero_grad(set_to_none=True)
            batch_features = masked_train[indices]
            if candidate_settings["candidate_id"] == "chain_masked_reconstruction":
                endpoint_prediction, reconstruction = model.forward_with_aux(
                    batch_features
                )
                member_predictions = endpoint_prediction.unsqueeze(1)
            else:
                member_predictions = model.forward_members(batch_features)
                reconstruction = None
            endpoint_loss = balanced_member_mse(
                torch,
                member_predictions,
                train_targets[indices],
                train_segments[batch_cpu],
            )
            if reconstruction is None:
                reconstruction_loss = endpoint_loss.detach() * 0.0
            else:
                reconstruction_loss = deliberate_missing_reconstruction_mse(
                    torch,
                    reconstruction,
                    reconstruction_targets[indices],
                    states_tensor[indices],
                )
            objective = endpoint_loss + float(
                candidate_settings["reconstruction_weight"]
            ) * reconstruction_loss
            if not bool(torch.isfinite(objective).item()):
                raise OutageTrainingError("training objective is non-finite")
            objective.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=math.inf,
                    error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                raise OutageTrainingError("candidate gradient is non-finite") from exc
            optimizer.step()
            objective_mass += float(objective.detach().item())
            endpoint_mass += float(endpoint_loss.detach().item())
            reconstruction_mass += float(reconstruction_loss.detach().item())
            batch_count += 1
        if batch_count <= 0:
            raise OutageTrainingError("segment-balanced epoch produced no batches")
        counts = {
            state: int(np.count_nonzero(states == state))
            for state in (MASK_FULL_STATE, 0, 1, 2, 3)
        }
        history.append(
            {
                "epoch": epoch,
                "scaled_objective_loss": objective_mass / batch_count,
                "scaled_endpoint_member_mse": endpoint_mass / batch_count,
                "scaled_masked_reconstruction_mse": (
                    reconstruction_mass / batch_count
                ),
                "mask_schedule_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
                "mask_full_row_count": counts[MASK_FULL_STATE],
                "mask_drop_imu0_row_count": counts[0],
                "mask_drop_imu1_row_count": counts[1],
                "mask_drop_imu2_row_count": counts[2],
                "mask_drop_imu3_row_count": counts[3],
            }
        )

    selected_state = _cpu_state_dict(model)
    complete_states = np.full(len(X_train_array), MASK_FULL_STATE, dtype=np.int8)
    complete_input = torch.as_tensor(
        apply_sensor_masks(X_train_array, complete_states),
        dtype=torch.float32,
        device=resolved_device,
    )
    train_prediction_raw = _predict_raw(
        model, complete_input, torch=torch, preprocessor=preprocessor
    )
    train_metrics = regression_metrics(train_truth_raw, train_prediction_raw)
    if not all(
        math.isfinite(float(value))
        for value in train_metrics.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    ):
        raise OutageTrainingError("final-refit train metrics contain non-finite values")
    result = {
        "candidate_id": candidate_settings["candidate_id"],
        "fit_kind": "all_development_exact_fixed_epochs",
        "model_seed": int(model_seed),
        "paired_seed_index": seed_index,
        "mask_held_out_segment": str(mask_held_out_segment),
        "mask_forced_outage": outage_index,
        "trained_epochs": resolved_epochs,
        "history": history,
        "state_dict": selected_state,
        "train_prediction_raw_mm": train_prediction_raw,
        "train_metrics": train_metrics,
    }
    del model, optimizer
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def predict_outage_state(
    *,
    state_dict: Mapping[str, Any],
    X_scaled: Any,
    availability: Any,
    preprocessor: Any,
    candidate: Mapping[str, Any],
    device: str,
) -> np.ndarray:
    """Run raw-mm inference from a frozen state without any label argument."""

    import torch

    _candidate_settings(candidate)
    if preprocessor is None or not hasattr(preprocessor, "inverse_targets"):
        raise OutageTrainingError("a fitted TrainOnlyPreprocessor is required")
    if not bool(getattr(preprocessor, "is_fitted", False)):
        raise OutageTrainingError("TrainOnlyPreprocessor must be fitted")
    if tuple(getattr(preprocessor, "feature_names", ())) != FUSION_COLUMNS:
        raise OutageTrainingError(
            "preprocessor must describe the exact ordered fusion_v1 features"
        )
    explicit_input = _explicit_availability_input(X_scaled, availability)
    resolved_device = torch.device(str(device))
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise OutageTrainingError("CUDA outage inference requested without an available GPU")
    model = _build_candidate(torch, candidate).to(resolved_device)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise OutageTrainingError("state_dict must be a non-empty mapping")
    try:
        resolved_state: dict[str, Any] = {}
        for name, value in state_dict.items():
            if hasattr(value, "detach"):
                tensor = value.detach().to(resolved_device)
            else:
                array = np.asarray(value)
                if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
                    raise ValueError("checkpoint array is not finite numeric data")
                tensor = torch.as_tensor(array, device=resolved_device)
            resolved_state[str(name)] = tensor
        model.load_state_dict(
            resolved_state,
            strict=True,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise OutageTrainingError("state_dict is incompatible with candidate") from exc
    features = torch.as_tensor(
        explicit_input, dtype=torch.float32, device=resolved_device
    )
    prediction = _predict_raw(
        model, features, torch=torch, preprocessor=preprocessor
    )
    del model
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction


_train_outage_seed = train_outage_seed


__all__ = [
    "MASK_FULL_STATE",
    "MASK_STATE_PROBABILITIES",
    "MAX_OUTAGE_EPOCHS",
    "OUTAGE_CANDIDATE_IDS",
    "OutageTrainingError",
    "_train_outage_seed",
    "apply_sensor_masks",
    "balanced_member_mse",
    "deliberate_missing_reconstruction_mse",
    "deterministic_mask_states",
    "fit_outage_seed_fixed_epochs",
    "predict_outage_state",
    "train_outage_seed",
    "transform_outage_features",
]
