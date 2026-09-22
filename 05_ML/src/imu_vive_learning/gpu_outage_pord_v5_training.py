"""Controlled outage-risk v5 loss and fold/seed training core.

This module is prospective and deliberately separate from the frozen outage-v3
and GBDT-v4 runners.  It consumes *already train-fold standardized* 43-column
``fusion_v1`` features and standardized XYZ targets.  Every training UID is
expanded, after the split, into exactly five views: fully available and one
view for each single-IMU outage.  Segment identifiers are used only to form
empirical group means; they are never appended to the 47-column model input.

The primary twelve-method controlled table shares the frozen v3
``masked_mlp_64x64_relu`` architecture, initialization, five views, AdamW
settings, and one full-fold optimizer update per epoch.  The frozen v3
``chain_endpoint_only`` architecture is limited to the two transfer rows
masked ERM and paired PORD.  This allocation was fixed before v5 metrics, after frozen v3
development evidence showed the masked backbone was the stronger direction.
Within a backbone, the intended differences are pre-registered objective or
availability controls.  The old group-mean objective is retained only as
``group_mean_pord`` because

``mean(L_missing - L_full) = mean(L_missing) - mean(L_full)``

does not depend on same-UID covariance and its full-risk term cancels from a
two-way interaction.  The primary paired objective instead uses

``A[s,m] = mean_i ReLU(L_missing[i,m] - stop_gradient(L_full[i]))``

before the fixed top-three and two-way interaction penalties.  This ordering
is joint-distribution dependent.  It is an empirical paired outage-risk /
degradation objective, never oracle-relative minimax regret.  It defines no
Wasserstein ambiguity set, oracle comparator, modality-expert router, or
regret theorem.

The fixed wrong-pair control circularly deranges full losses within segment in
sorted UID order.  The no-stop-gradient control has an identical forward value
and differs only in its full-row gradient.  The constant-availability control
still zeroes the requested IMU block but fixes all availability bits to one.

Vive remains a supervised label/evaluation signal only.  This code performs
no collection, hardware I/O, inference promotion, or challenge-label access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import numbers
import os
from typing import Any

import numpy as np

from .gpu_outage_training import MASK_FULL_STATE, apply_sensor_masks
from .gpu_v3_backbones import (
    IMU_SENSOR_INDEX_MAP,
    V3_IMU_COUNT,
    V3_INPUT_FEATURE_COUNT,
    V3_SENSOR_FEATURE_COUNT,
    build_chain_sensor_message_passing_backbone,
    build_masked_mlp_backbone,
)


PORD_METHOD_IDS = (
    "full_gradient_only_mask_selected_control",
    "masked_erm",
    "constant_availability_erm",
    "group_mean_pord",
    "paired_tail",
    "paired_interaction",
    "paired_pord",
    "paired_wrong_pair",
    "paired_no_stopgrad",
    "groupdro",
    "vrex",
    "mofe",
)
PORD_VIEW_NAMES = (
    "full",
    "drop_imu0",
    "drop_imu1",
    "drop_imu2",
    "drop_imu3",
)
PORD_MASK_PROBABILITIES = (0.25, 0.1875, 0.1875, 0.1875, 0.1875)
PORD_FULL_GROUP_WEIGHT = 1.0 / 12.0
PORD_MISSING_GROUP_WEIGHT = 1.0 / 16.0
PORD_TOP_K = 3
PORD_GROUPDRO_ETA = 0.01
PORD_VREX_BETA = 1.0
PORD_MOFE_LAMBDA = 0.1
PORD_LEARNING_RATE = 1.0e-3
PORD_WEIGHT_DECAY = 1.0e-3
PORD_EPOCH_BUDGET = 400
PORD_MASKED_MLP_BACKBONE_ID = "masked_mlp_64x64_relu"
PORD_PRIMARY_BACKBONE_ID = PORD_MASKED_MLP_BACKBONE_ID
PORD_CHAIN_BACKBONE_ID = "chain_endpoint_only"
PORD_CHAIN_TRANSFER_BACKBONE_ID = PORD_CHAIN_BACKBONE_ID
PORD_BACKBONE_IDS = (
    PORD_PRIMARY_BACKBONE_ID,
    PORD_CHAIN_TRANSFER_BACKBONE_ID,
)
PORD_PRIMARY_BACKBONE_SETTINGS = {
    "backbone_id": PORD_PRIMARY_BACKBONE_ID,
    "backbone": "plain_mlp",
    "hidden_layer_sizes": [64, 64],
    "activation": "ReLU",
    "masked_reconstruction_loss_weight": 0.0,
}
PORD_CHAIN_TRANSFER_BACKBONE_SETTINGS = {
    "backbone_id": PORD_CHAIN_TRANSFER_BACKBONE_ID,
    "backbone": "encoder_conditioned_four_sensor_chain_message_passing",
    "width": 96,
    "depth": 4,
    "expansion": 2,
    "activation": "SiLU",
    "normalization": "LayerNorm",
    "dropout": 0.0,
    "masked_reconstruction_loss_weight": 0.0,
}
PORD_BACKBONE_SETTINGS = PORD_PRIMARY_BACKBONE_SETTINGS
PORD_BACKBONE_SETTINGS_BY_ID = {
    PORD_PRIMARY_BACKBONE_ID: PORD_PRIMARY_BACKBONE_SETTINGS,
    PORD_CHAIN_TRANSFER_BACKBONE_ID: PORD_CHAIN_TRANSFER_BACKBONE_SETTINGS,
}
PORD_PRIMARY_METHOD_IDS = PORD_METHOD_IDS
PORD_TRANSFER_METHOD_IDS = ("masked_erm", "paired_pord")
PORD_OPTIMIZER_SETTINGS = {
    "optimizer": "AdamW",
    "learning_rate": PORD_LEARNING_RATE,
    "weight_decay": PORD_WEIGHT_DECAY,
    "epoch_budget": PORD_EPOCH_BUDGET,
    "updates_per_epoch": 1,
}
PORD_CHECKPOINT_SELECTION_SCOPE = "training_segments_only"
PORD_CHECKPOINT_CRITERION = (
    "0.5*(mean4_training_outage_rmse_3d_mm+"
    "max4_training_outage_rmse_3d_mm)"
)


class PORDV5TrainingError(RuntimeError):
    """Raised when the PORD-v5 mathematical or training contract is violated."""


@dataclass(frozen=True)
class FiveViewBundle:
    """Exact post-split full plus four single-outage feature views."""

    features: np.ndarray
    view_names: tuple[str, ...]
    availability: np.ndarray
    trial_uids: tuple[str, ...]
    availability_semantics: str
    sha256: str


@dataclass(frozen=True)
class PairingMap:
    """A frozen row-to-full-reference mapping for paired objectives."""

    indices: np.ndarray
    mode: str
    trial_uids: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class GroupRiskTensors:
    """Empirical segment-balanced risks and their per-row parents."""

    full: Any
    missing: Any
    row_full: Any
    row_missing: Any
    segment_index: Any
    segment_labels: tuple[str, ...]


@dataclass(frozen=True)
class ObjectiveTensors:
    """One controlled objective and all tensors required for an exact audit."""

    method_id: str
    objective: Any
    mask_risk: Any
    tail_penalty: Any
    interaction_penalty: Any
    vrex_penalty: Any
    mofe_penalty: Any
    paired_excess: Any
    positive_excess: Any
    interaction: Any
    tail_weights: Any
    tail_cutoff: float
    next_groupdro_weights: Any | None
    groupdro_weights_before: Any | None
    full_risks: Any
    missing_risks: Any
    mofe_hinge: Any | None
    row_full_losses: Any | None
    row_missing_losses: Any | None
    row_pair_excess: Any | None
    row_full_reference: Any | None
    pairing_mode: str
    full_reference_detached: bool
    pairing_map_sha256: str | None


def _canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PORDV5TrainingError("record is not finite canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def array_sha256(values: Any) -> str:
    """Hash an exact numeric array with dtype and shape binding."""

    array = np.asarray(values)
    if array.dtype.hasobject:
        raise PORDV5TrainingError("object arrays cannot be bound as numeric data")
    if not np.issubdtype(array.dtype, np.number):
        raise PORDV5TrainingError("array hash requires numeric data")
    if not np.isfinite(array).all():
        raise PORDV5TrainingError("array hash rejects non-finite data")
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(b"\x00")
    hasher.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
    hasher.update(b"\x00")
    hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash an ordered, finite tensor state without pickle serialization."""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise PORDV5TrainingError("state_dict must be a non-empty mapping")
    hasher = hashlib.sha256()
    for name in sorted(str(key) for key in state_dict):
        if name not in state_dict:
            raise PORDV5TrainingError("state_dict keys must be strings")
        value = state_dict[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
            raise PORDV5TrainingError("state_dict contains non-numeric data")
        if not np.isfinite(array).all():
            raise PORDV5TrainingError("state_dict contains non-finite data")
        contiguous = np.ascontiguousarray(array)
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(str(contiguous.dtype).encode("ascii"))
        hasher.update(b"\x00")
        hasher.update(
            json.dumps(list(contiguous.shape), separators=(",", ":")).encode()
        )
        hasher.update(b"\x00")
        hasher.update(contiguous.tobytes(order="C"))
        hasher.update(b"\xff")
    return hasher.hexdigest()


def _matrix(
    values: Any,
    *,
    name: str,
    columns: int,
    rows: int | None = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise PORDV5TrainingError(f"{name} must be numeric") from exc
    if array.ndim != 2 or len(array) == 0 or int(array.shape[1]) != int(columns):
        raise PORDV5TrainingError(
            f"{name} must be a non-empty matrix with {columns} columns"
        )
    if rows is not None and len(array) != int(rows):
        raise PORDV5TrainingError(f"{name} row count changed")
    if not np.isfinite(array).all():
        raise PORDV5TrainingError(f"{name} contains non-finite values")
    return array


def _uids(values: Sequence[Any], *, rows: int, name: str) -> tuple[str, ...]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != rows:
        raise PORDV5TrainingError(f"{name} must contain one UID per row")
    normalized: list[str] = []
    for value in array.tolist():
        if isinstance(value, np.generic):
            value = value.item()
        if value is None:
            raise PORDV5TrainingError(f"{name} contains a missing UID")
        text = str(value)
        if not text:
            raise PORDV5TrainingError(f"{name} contains an empty UID")
        normalized.append(text)
    if len(normalized) != len(set(normalized)):
        raise PORDV5TrainingError(f"{name} must be unique")
    return tuple(normalized)


def _segments(
    values: Sequence[Any], *, rows: int
) -> tuple[np.ndarray, tuple[str, str, str]]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != rows:
        raise PORDV5TrainingError("segment_ids must contain one value per train row")
    normalized: list[str] = []
    for value in array.tolist():
        if isinstance(value, np.generic):
            value = value.item()
        if value is None:
            raise PORDV5TrainingError("segment_ids contains a missing value")
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise PORDV5TrainingError("segment_ids contains a non-finite value")
        text = str(value)
        if not text:
            raise PORDV5TrainingError("segment_ids contains an empty value")
        normalized.append(text)
    labels = tuple(sorted(set(normalized)))
    if len(labels) != 3:
        raise PORDV5TrainingError("a PORD train fold must contain exactly three segments")
    index_by_label = {label: index for index, label in enumerate(labels)}
    indices = np.asarray([index_by_label[value] for value in normalized], dtype=np.int64)
    if any(int(np.count_nonzero(indices == index)) == 0 for index in range(3)):
        raise PORDV5TrainingError("every train segment must contain at least one row")
    return indices, labels  # type: ignore[return-value]


def _target_transform(
    target_mean_mm: Any, target_scale_mm: Any
) -> tuple[np.ndarray, np.ndarray]:
    try:
        mean = np.asarray(target_mean_mm, dtype=np.float64)
        scale = np.asarray(target_scale_mm, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PORDV5TrainingError("target transform must be numeric") from exc
    if mean.shape != (3,) or scale.shape != (3,):
        raise PORDV5TrainingError("target_mean_mm and target_scale_mm must have shape (3,)")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise PORDV5TrainingError("target transform contains non-finite values")
    if np.any(scale <= 0.0):
        raise PORDV5TrainingError("target_scale_mm must be strictly positive")
    if not np.array_equal(scale, np.repeat(scale[0], 3)):
        raise PORDV5TrainingError(
            "target_scale_mm must repeat the exact v3 isotropic scalar"
        )
    return mean, scale


def inverse_standardized_targets(
    values: Any, *, target_mean_mm: Any, target_scale_mm: Any
) -> np.ndarray:
    """Apply v3's train-fold mean and repeated isotropic scale in float64 mm."""

    array = _matrix(values, name="standardized targets", columns=3, dtype=np.float64)
    mean, scale = _target_transform(target_mean_mm, target_scale_mm)
    raw = array * scale.reshape(1, 3) + mean.reshape(1, 3)
    if not np.isfinite(raw).all():
        raise PORDV5TrainingError("inverse-transformed targets are non-finite")
    return raw


def _build_five_views(
    X_scaled: Any,
    *,
    trial_uids: Sequence[Any],
    availability_semantics: str,
) -> FiveViewBundle:
    """Build a pre-registered five-view availability control."""

    features = _matrix(
        X_scaled, name="X_scaled", columns=V3_SENSOR_FEATURE_COUNT
    )
    uids = _uids(trial_uids, rows=len(features), name="trial_uids")
    states = (
        np.full(len(features), MASK_FULL_STATE, dtype=np.int8),
        *(np.full(len(features), sensor, dtype=np.int8) for sensor in range(4)),
    )
    views = np.stack(
        [apply_sensor_masks(features, state) for state in states], axis=0
    ).astype(np.float32, copy=False)
    if views.shape != (5, len(features), V3_INPUT_FEATURE_COUNT):
        raise AssertionError("five-view expansion changed shape")
    physical_availability = np.asarray(
        [
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ],
        dtype=np.float32,
    )
    if availability_semantics == "physical_mask_state":
        expected_availability = physical_availability
    elif availability_semantics == "constant_all_available_negative_control":
        views[:, :, V3_SENSOR_FEATURE_COUNT:] = 1.0
        expected_availability = np.ones((5, 4), dtype=np.float32)
    else:  # pragma: no cover - private caller is closed-world
        raise AssertionError(availability_semantics)
    availability = np.stack(
        [views[index, 0, V3_SENSOR_FEATURE_COUNT:] for index in range(5)], axis=0
    )
    if not np.array_equal(availability, expected_availability):
        raise AssertionError("single-outage availability layout changed")
    for sensor, columns in enumerate(IMU_SENSOR_INDEX_MAP):
        if not np.equal(views[sensor + 1, :, list(columns)], 0.0).all():
            raise AssertionError("outage view did not zero the physical IMU")
    uid_hash = hashlib.sha256("\n".join(uids).encode("utf-8")).hexdigest()
    digest = _canonical_json_sha256(
        {
            "features_sha256": array_sha256(views),
            "trial_uid_order_sha256": uid_hash,
            "view_names": list(PORD_VIEW_NAMES),
            "mask_probabilities": list(PORD_MASK_PROBABILITIES),
            "availability_semantics": availability_semantics,
        }
    )
    return FiveViewBundle(
        features=views,
        view_names=PORD_VIEW_NAMES,
        availability=availability,
        trial_uids=uids,
        availability_semantics=availability_semantics,
        sha256=digest,
    )


def build_exact_five_views(
    X_scaled: Any, *, trial_uids: Sequence[Any]
) -> FiveViewBundle:
    """Build physical full plus four single-outage views per supplied UID."""

    return _build_five_views(
        X_scaled,
        trial_uids=trial_uids,
        availability_semantics="physical_mask_state",
    )


def build_constant_availability_five_views(
    X_scaled: Any, *, trial_uids: Sequence[Any]
) -> FiveViewBundle:
    """Zero physical outage blocks but hold every availability bit at one.

    This is a negative control for the explicit availability indicator.  It is
    not a wrong cyclic relabel: train, checkpoint, outer prediction, and outer
    evaluation all use the same constant-bit rule.
    """

    return _build_five_views(
        X_scaled,
        trial_uids=trial_uids,
        availability_semantics="constant_all_available_negative_control",
    )


def _views_for_method(
    X_scaled: Any, *, trial_uids: Sequence[Any], method_id: str
) -> FiveViewBundle:
    if str(method_id) == "constant_availability_erm":
        return build_constant_availability_five_views(
            X_scaled, trial_uids=trial_uids
        )
    return build_exact_five_views(X_scaled, trial_uids=trial_uids)


def fixed_within_segment_circular_derangement(
    *,
    trial_uids: Sequence[Any],
    segment_index: Sequence[Any],
) -> PairingMap:
    """Derange full-loss references by one sorted-UID step within segment.

    Every segment must have at least two rows.  The returned indices form an
    exact within-segment permutation and contain no fixed point, so the full
    loss marginal is preserved while correct row pairing is destroyed.
    """

    uids = _uids(trial_uids, rows=len(trial_uids), name="trial_uids")
    raw_segments = np.asarray(segment_index, dtype=object)
    if raw_segments.ndim != 1 or len(raw_segments) != len(uids):
        raise PORDV5TrainingError("segment_index must contain one value per UID")
    segments: list[int] = []
    for value in raw_segments.tolist():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise PORDV5TrainingError("segment_index must use an integer dtype")
        index = int(value)
        if index not in (0, 1, 2):
            raise PORDV5TrainingError("segment_index must contain only 0, 1, 2")
        segments.append(index)
    indices = np.empty(len(uids), dtype=np.int64)
    segment_array = np.asarray(segments, dtype=np.int64)
    for group in range(3):
        positions = np.flatnonzero(segment_array == group).tolist()
        if len(positions) < 2:
            raise PORDV5TrainingError(
                "wrong-pair derangement requires at least two rows per segment"
            )
        ordered = sorted(positions, key=lambda position: uids[position])
        for offset, position in enumerate(ordered):
            indices[position] = ordered[(offset + 1) % len(ordered)]
    if (
        np.array_equal(indices, np.arange(len(indices), dtype=np.int64))
        or np.any(indices == np.arange(len(indices), dtype=np.int64))
        or sorted(indices.tolist()) != list(range(len(indices)))
        or not np.array_equal(segment_array[indices], segment_array)
    ):
        raise AssertionError("within-segment circular derangement changed")
    digest = _canonical_json_sha256(
        {
            "mode": "sorted_uid_within_segment_circular_shift_plus_one",
            "trial_uids": list(uids),
            "segment_index": segment_array.tolist(),
            "full_reference_indices": indices.tolist(),
        }
    )
    return PairingMap(
        indices=indices,
        mode="sorted_uid_within_segment_circular_shift_plus_one",
        trial_uids=uids,
        sha256=digest,
    )


def build_exact_pord_backbone(torch: Any) -> Any:
    """Build the exact frozen 64x64 ReLU primary PORD backbone."""

    model = build_masked_mlp_backbone(torch, hidden_layer_sizes=(64, 64))
    if (
        getattr(model, "backbone_name", None) != "masked_mlp"
        or int(getattr(model, "input_feature_count", -1)) != 47
        or int(getattr(model, "output_count", -1)) != 3
    ):
        raise AssertionError("masked-MLP primary backbone contract changed")
    return model


def build_exact_chain_transfer_backbone(torch: Any) -> Any:
    """Build the exact v3 chain endpoint transfer backbone, without aux loss."""

    model = build_chain_sensor_message_passing_backbone(
        torch, width=96, depth=4, expansion=2, dropout=0.0
    )
    if (
        getattr(model, "backbone_name", None) != "chain_sensor_message_passing"
        or int(getattr(model, "input_feature_count", -1)) != 47
        or int(getattr(model, "output_count", -1)) != 3
    ):
        raise AssertionError("chain endpoint backbone contract changed")
    return model


def build_pord_backbone(torch: Any, *, backbone_id: str) -> Any:
    """Build one pre-registered controlled or transfer backbone.

    The full twelve-method table is defined only for the frozen ``64x64`` ReLU
    masked MLP.  The chain transfer control is intentionally limited by the
    trainer to the two pre-registered rows ``masked_erm`` and ``paired_pord``.
    """

    resolved = str(backbone_id)
    if resolved == PORD_PRIMARY_BACKBONE_ID:
        return build_exact_pord_backbone(torch)
    if resolved == PORD_CHAIN_TRANSFER_BACKBONE_ID:
        return build_exact_chain_transfer_backbone(torch)
    raise PORDV5TrainingError(f"unknown PORD backbone: {resolved!r}")


def _torch_segment_index(
    torch: Any, values: Any, *, rows: int, device: Any
) -> Any:
    """Resolve an exact integer 0..2 segment index without lossy casting."""

    if isinstance(values, torch.Tensor):
        if (
            values.dtype == torch.bool
            or values.dtype.is_floating_point
            or values.dtype.is_complex
        ):
            raise PORDV5TrainingError("segment_index must use an integer dtype")
        indices = values.to(device=device, dtype=torch.long)
    else:
        array = np.asarray(values, dtype=object)
        if array.ndim != 1 or len(array) != rows:
            raise PORDV5TrainingError("segment_index must contain one value per row")
        normalized: list[int] = []
        for value in array.tolist():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise PORDV5TrainingError("segment_index must use an integer dtype")
            normalized.append(int(value))
        indices = torch.as_tensor(normalized, dtype=torch.long, device=device)
    if indices.ndim != 1 or int(indices.shape[0]) != rows:
        raise PORDV5TrainingError("segment_index must contain one value per row")
    if not bool(torch.logical_and(indices >= 0, indices < 3).all().item()):
        raise PORDV5TrainingError("segment_index must contain only 0, 1, 2")
    return indices


def compute_empirical_group_risks(
    predictions: Any,
    targets: Any,
    segment_index: Any,
    *,
    segment_labels: Sequence[str] = ("0", "1", "2"),
) -> GroupRiskTensors:
    """Mean XYZ MSE within each segment and view, never a minibatch surrogate."""

    import torch

    if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise PORDV5TrainingError("predictions and targets must be torch tensors")
    if predictions.ndim != 3 or int(predictions.shape[0]) != 5 or int(
        predictions.shape[2]
    ) != 3:
        raise PORDV5TrainingError("predictions must have shape [5, rows, 3]")
    if targets.ndim != 2 or tuple(targets.shape) != (
        int(predictions.shape[1]),
        3,
    ):
        raise PORDV5TrainingError("targets must have shape [rows, 3]")
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise PORDV5TrainingError("predictions and targets must be floating point")
    if predictions.dtype != targets.dtype or predictions.device != targets.device:
        raise PORDV5TrainingError("predictions and targets must share dtype and device")
    if not bool(torch.isfinite(predictions).all().item()) or not bool(
        torch.isfinite(targets).all().item()
    ):
        raise PORDV5TrainingError("predictions or targets are non-finite")
    indices = _torch_segment_index(
        torch,
        segment_index,
        rows=int(predictions.shape[1]),
        device=predictions.device,
    )
    labels = tuple(str(value) for value in segment_labels)
    if len(labels) != 3 or len(set(labels)) != 3:
        raise PORDV5TrainingError("segment_labels must contain three unique values")

    row_losses = torch.square(predictions - targets.unsqueeze(0)).mean(dim=-1)
    full_groups: list[Any] = []
    missing_groups: list[Any] = []
    for group_index in range(3):
        selected = indices == group_index
        if int(selected.sum().item()) <= 0:
            raise PORDV5TrainingError("every segment group must be non-empty")
        full_groups.append(row_losses[0, selected].mean())
        missing_groups.append(row_losses[1:, selected].mean(dim=1))
    full = torch.stack(full_groups)
    missing = torch.stack(missing_groups, dim=0)
    if tuple(full.shape) != (3,) or tuple(missing.shape) != (3, 4):
        raise AssertionError("empirical group-risk shape changed")
    return GroupRiskTensors(
        full=full,
        missing=missing,
        row_full=row_losses[0],
        row_missing=row_losses[1:].transpose(0, 1),
        segment_index=indices,
        segment_labels=labels,
    )


def two_way_center(values: Any) -> Any:
    """Two-way center a fixed 3-segment by 4-outage matrix."""

    import torch

    if not isinstance(values, torch.Tensor) or tuple(values.shape) != (3, 4):
        raise PORDV5TrainingError("two-way centering requires a [3, 4] tensor")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all().item()):
        raise PORDV5TrainingError("interaction input must be finite floating point")
    return (
        values
        - values.mean(dim=1, keepdim=True)
        - values.mean(dim=0, keepdim=True)
        + values.mean()
    )


def fixed_symmetric_positive_topk(
    values: Any, *, k: int = PORD_TOP_K
) -> tuple[Any, Any, float]:
    """Return fixed-denominator positive top-k with symmetric cutoff ties.

    Selection and cutoff weights are computed from detached values.  Values
    above the cutoff receive ``1/k``; cutoff ties share the remaining top-k
    mass.  If fewer than ``k`` values are positive, each positive value still
    receives ``1/k`` and the unused slots contribute zero.  PyTorch's ReLU
    therefore supplies the required zero subgradient at an exact zero.
    """

    import torch

    if not isinstance(values, torch.Tensor) or tuple(values.shape) != (3, 4):
        raise PORDV5TrainingError("fixed PORD top-k input must have shape [3, 4]")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all().item()):
        raise PORDV5TrainingError("top-k input must be finite floating point")
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or int(k) <= 0:
        raise PORDV5TrainingError("top-k k must be a positive integer")
    resolved_k = int(k)
    if resolved_k != PORD_TOP_K:
        raise PORDV5TrainingError("PORD uses exactly top-3 over twelve cells")
    positive = torch.relu(values)
    detached = positive.detach().reshape(-1)
    positive_count = int((detached > 0).sum().item())
    flat_weights = torch.zeros_like(detached)
    cutoff = 0.0
    if positive_count > 0:
        if positive_count < resolved_k:
            flat_weights[detached > 0] = 1.0 / resolved_k
        else:
            sorted_values = torch.sort(detached, descending=True).values
            cutoff_tensor = sorted_values[resolved_k - 1]
            cutoff = float(cutoff_tensor.item())
            above = detached > cutoff_tensor
            tied = detached == cutoff_tensor
            above_count = int(above.sum().item())
            tied_count = int(tied.sum().item())
            remaining = resolved_k - above_count
            if remaining <= 0 or tied_count <= 0:
                raise AssertionError("invalid symmetric top-k cutoff accounting")
            flat_weights[above] = 1.0 / resolved_k
            flat_weights[tied] = remaining / (resolved_k * tied_count)
    weights = flat_weights.reshape(values.shape)
    penalty = (weights * positive).sum()
    return penalty, weights, cutoff


def _validate_risk_tensors(full_risks: Any, missing_risks: Any) -> None:
    import torch

    if not isinstance(full_risks, torch.Tensor) or not isinstance(
        missing_risks, torch.Tensor
    ):
        raise PORDV5TrainingError("group risks must be torch tensors")
    if tuple(full_risks.shape) != (3,) or tuple(missing_risks.shape) != (3, 4):
        raise PORDV5TrainingError("full/missing risks must have shape [3] and [3, 4]")
    if not full_risks.is_floating_point() or not missing_risks.is_floating_point():
        raise PORDV5TrainingError("group risks must be floating point")
    if full_risks.dtype != missing_risks.dtype or full_risks.device != missing_risks.device:
        raise PORDV5TrainingError("full and missing risks must share dtype and device")
    if not bool(torch.isfinite(full_risks).all().item()) or not bool(
        torch.isfinite(missing_risks).all().item()
    ):
        raise PORDV5TrainingError("group risks contain non-finite values")
    if bool((full_risks < 0).any().item()) or bool(
        (missing_risks < 0).any().item()
    ):
        raise PORDV5TrainingError("MSE group risks must be non-negative")


def _groupdro_prior(torch: Any, *, like: Any) -> Any:
    return torch.as_tensor(
        [PORD_FULL_GROUP_WEIGHT] * 3 + [PORD_MISSING_GROUP_WEIGHT] * 12,
        dtype=like.dtype,
        device=like.device,
    )


def _validate_row_losses(
    *,
    torch: Any,
    row_full_losses: Any,
    row_missing_losses: Any,
    full_risks: Any,
    segment_index: Any,
) -> Any:
    if not isinstance(row_full_losses, torch.Tensor) or not isinstance(
        row_missing_losses, torch.Tensor
    ):
        raise PORDV5TrainingError("row objectives require torch row losses")
    if row_full_losses.ndim != 1 or tuple(row_missing_losses.shape) != (
        int(row_full_losses.shape[0]),
        4,
    ):
        raise PORDV5TrainingError("row losses must have shape [N] and [N, 4]")
    if not row_full_losses.is_floating_point() or not row_missing_losses.is_floating_point():
        raise PORDV5TrainingError("row losses must be floating point")
    if (
        row_full_losses.dtype != full_risks.dtype
        or row_missing_losses.dtype != full_risks.dtype
        or row_full_losses.device != full_risks.device
        or row_missing_losses.device != full_risks.device
    ):
        raise PORDV5TrainingError("row and group losses must share dtype and device")
    if not bool(torch.isfinite(row_full_losses).all().item()) or not bool(
        torch.isfinite(row_missing_losses).all().item()
    ):
        raise PORDV5TrainingError("row losses contain non-finite values")
    if bool((row_full_losses < 0).any().item()) or bool(
        (row_missing_losses < 0).any().item()
    ):
        raise PORDV5TrainingError("MSE row losses must be non-negative")
    return _torch_segment_index(
        torch,
        segment_index,
        rows=int(row_full_losses.shape[0]),
        device=row_full_losses.device,
    )


def compute_row_paired_atoms(
    *,
    row_full_losses: Any,
    row_missing_losses: Any,
    segment_index: Any,
    full_risks: Any,
    full_reference_index: Any | None = None,
    detach_full_reference: bool = True,
) -> tuple[Any, Any, Any]:
    """Return ``A[s,m]=mean_i ReLU(Lmiss-Lfull_ref)`` and row evidence.

    This pure function is independent of the backbone.  An omitted reference
    index means the true same-row pair.  A supplied index must be an exact
    within-segment permutation; a wrong-pair caller additionally enforces no
    fixed points when constructing the frozen map.
    """

    import torch

    indices = _validate_row_losses(
        torch=torch,
        row_full_losses=row_full_losses,
        row_missing_losses=row_missing_losses,
        full_risks=full_risks,
        segment_index=segment_index,
    )
    rows = int(row_full_losses.shape[0])
    if full_reference_index is None:
        reference_index = torch.arange(rows, dtype=torch.long, device=indices.device)
    else:
        reference_index = _torch_segment_index_reference(
            torch,
            full_reference_index,
            rows=rows,
            device=indices.device,
        )
        if not bool(torch.equal(torch.sort(reference_index).values, torch.arange(
            rows, dtype=torch.long, device=indices.device
        ))):
            raise PORDV5TrainingError("full reference index must be a permutation")
        if not bool(torch.equal(indices[reference_index], indices)):
            raise PORDV5TrainingError("full reference permutation crossed segments")
    full_reference = row_full_losses[reference_index]
    if detach_full_reference:
        full_reference = full_reference.detach()
    row_excess = row_missing_losses - full_reference.unsqueeze(1)
    positive_rows = torch.relu(row_excess)
    atoms: list[Any] = []
    for group_index in range(3):
        selected = indices == group_index
        if int(selected.sum().item()) <= 0:
            raise PORDV5TrainingError("paired atoms require every segment group")
        atoms.append(positive_rows[selected].mean(dim=0))
    result = torch.stack(atoms)
    if tuple(result.shape) != (3, 4) or not bool(torch.isfinite(result).all().item()):
        raise PORDV5TrainingError("paired atoms are non-finite or wrong-shaped")
    return result, row_excess, full_reference


def _torch_segment_index_reference(
    torch: Any, values: Any, *, rows: int, device: Any
) -> Any:
    """Resolve arbitrary integer row references without numeric coercion."""

    if isinstance(values, torch.Tensor):
        if values.dtype == torch.bool or values.dtype.is_floating_point or values.dtype.is_complex:
            raise PORDV5TrainingError("full reference index must use an integer dtype")
        result = values.to(device=device, dtype=torch.long)
    else:
        array = np.asarray(values, dtype=object)
        if array.ndim != 1 or len(array) != rows:
            raise PORDV5TrainingError("full reference index must contain one row index")
        normalized: list[int] = []
        for value in array.tolist():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise PORDV5TrainingError("full reference index must use integers")
            normalized.append(int(value))
        result = torch.as_tensor(normalized, dtype=torch.long, device=device)
    if result.ndim != 1 or int(result.shape[0]) != rows:
        raise PORDV5TrainingError("full reference index must contain one row index")
    if not bool(torch.logical_and(result >= 0, result < rows).all().item()):
        raise PORDV5TrainingError("full reference index is out of bounds")
    return result


def compute_controlled_objective(
    *,
    method_id: str,
    full_risks: Any,
    missing_risks: Any,
    row_full_losses: Any | None = None,
    row_missing_losses: Any | None = None,
    segment_index: Any | None = None,
    full_reference_index: Any | None = None,
    pairing_map_sha256: str | None = None,
    groupdro_weights: Any | None = None,
) -> ObjectiveTensors:
    """Pure twelve-method objective over exact full-fold empirical losses."""

    import torch

    method = str(method_id)
    if method not in PORD_METHOD_IDS:
        raise PORDV5TrainingError(f"unknown PORD method: {method!r}")
    _validate_risk_tensors(full_risks, missing_risks)

    # Direct coefficients on segment means: 3*(1/12)=.25 and
    # 12*(1/16)=.75. Unequal segment row counts do not alter nominal mass.
    mask_risk = (
        PORD_FULL_GROUP_WEIGHT * full_risks.sum()
        + PORD_MISSING_GROUP_WEIGHT * missing_risks.sum()
    )
    group_mean_excess = missing_risks - full_risks.detach().unsqueeze(1)
    atom_values = torch.relu(group_mean_excess)
    row_pair_excess = None
    row_full_reference = None
    pairing_mode = "none"
    full_reference_detached = True
    resolved_pairing_hash = None
    paired_methods = {
        "paired_tail",
        "paired_interaction",
        "paired_pord",
        "paired_wrong_pair",
        "paired_no_stopgrad",
    }
    if method in paired_methods:
        detach = method != "paired_no_stopgrad"
        if method == "paired_wrong_pair":
            if full_reference_index is None:
                raise PORDV5TrainingError("wrong-pair objective requires a frozen map")
            if not isinstance(pairing_map_sha256, str) or len(pairing_map_sha256) != 64:
                raise PORDV5TrainingError("wrong-pair objective requires its map hash")
            pairing_mode = "sorted_uid_within_segment_circular_derangement"
            resolved_pairing_hash = pairing_map_sha256
        else:
            if full_reference_index is not None or pairing_map_sha256 is not None:
                raise PORDV5TrainingError("correct pairing rejects a wrong-pair map")
            pairing_mode = (
                "correct_uid_no_stop_gradient"
                if not detach
                else "correct_uid_stop_gradient"
            )
        atom_values, row_pair_excess, row_full_reference = compute_row_paired_atoms(
            row_full_losses=row_full_losses,
            row_missing_losses=row_missing_losses,
            segment_index=segment_index,
            full_risks=full_risks,
            full_reference_index=full_reference_index,
            detach_full_reference=detach,
        )
        full_reference_detached = detach
    elif method == "group_mean_pord":
        pairing_mode = "group_mean_difference_not_pair_specific"

    tail_penalty, tail_weights, tail_cutoff = fixed_symmetric_positive_topk(
        atom_values, k=PORD_TOP_K
    )
    interaction = two_way_center(
        group_mean_excess if method == "group_mean_pord" else atom_values
    )
    interaction_penalty = torch.abs(interaction).mean()

    risk_vector = torch.cat((full_risks, missing_risks.reshape(-1)))
    prior = _groupdro_prior(torch, like=full_risks)
    weighted_mean = (prior * risk_vector).sum()
    vrex_penalty = (prior * torch.square(risk_vector - weighted_mean)).sum()

    mofe_penalty = mask_risk.detach() * 0.0
    mofe_hinge = None
    if method == "mofe":
        indices = _validate_row_losses(
            torch=torch,
            row_full_losses=row_full_losses,
            row_missing_losses=row_missing_losses,
            full_risks=full_risks,
            segment_index=segment_index,
        )
        mofe_hinge = torch.relu(row_full_losses.unsqueeze(1) - row_missing_losses)
        group_hinges = [mofe_hinge[indices == group].mean(dim=0) for group in range(3)]
        mofe_penalty = torch.stack(group_hinges).mean()

    before = None
    next_weights = None
    if method == "groupdro":
        if groupdro_weights is None:
            before = prior
        else:
            before = torch.as_tensor(
                groupdro_weights, dtype=full_risks.dtype, device=full_risks.device
            )
            if before.shape != (15,) or not bool(torch.isfinite(before).all().item()):
                raise PORDV5TrainingError("GroupDRO weights must be finite shape [15]")
            if bool((before <= 0).any().item()) or not math.isclose(
                float(before.sum().item()), 1.0, rel_tol=0.0, abs_tol=1e-6
            ):
                raise PORDV5TrainingError("GroupDRO weights must be positive and sum to one")
        logits = torch.log(before) + PORD_GROUPDRO_ETA * risk_vector.detach()
        next_weights = torch.softmax(logits, dim=0)

    if method == "full_gradient_only_mask_selected_control":
        objective = full_risks.mean()
    elif method in {"masked_erm", "constant_availability_erm"}:
        objective = mask_risk
    elif method == "group_mean_pord":
        objective = mask_risk + tail_penalty + interaction_penalty
    elif method == "paired_tail":
        objective = mask_risk + tail_penalty
    elif method == "paired_interaction":
        objective = mask_risk + interaction_penalty
    elif method in {"paired_pord", "paired_wrong_pair", "paired_no_stopgrad"}:
        objective = mask_risk + tail_penalty + interaction_penalty
    elif method == "groupdro":
        assert next_weights is not None
        objective = (next_weights * risk_vector).sum()
    elif method == "vrex":
        objective = mask_risk + PORD_VREX_BETA * vrex_penalty
    elif method == "mofe":
        objective = mask_risk + PORD_MOFE_LAMBDA * mofe_penalty
    else:  # pragma: no cover - closed-world guard above
        raise AssertionError(method)
    if objective.ndim != 0 or not bool(torch.isfinite(objective).item()):
        raise PORDV5TrainingError("controlled objective is non-finite or non-scalar")
    return ObjectiveTensors(
        method_id=method,
        objective=objective,
        mask_risk=mask_risk,
        tail_penalty=tail_penalty,
        interaction_penalty=interaction_penalty,
        vrex_penalty=vrex_penalty,
        mofe_penalty=mofe_penalty,
        paired_excess=(group_mean_excess if method == "group_mean_pord" else atom_values),
        positive_excess=atom_values,
        interaction=interaction,
        tail_weights=tail_weights,
        tail_cutoff=tail_cutoff,
        next_groupdro_weights=next_weights,
        groupdro_weights_before=before,
        full_risks=full_risks,
        missing_risks=missing_risks,
        mofe_hinge=mofe_hinge,
        row_full_losses=row_full_losses,
        row_missing_losses=row_missing_losses,
        row_pair_excess=row_pair_excess,
        row_full_reference=row_full_reference,
        pairing_mode=pairing_mode,
        full_reference_detached=full_reference_detached,
        pairing_map_sha256=resolved_pairing_hash,
    )


def direct_group_risk_gradients(objective: ObjectiveTensors) -> tuple[Any, Any]:
    """Return direct objective gradients for the 3 + 12 group-risk nodes."""

    import torch

    if not objective.full_risks.requires_grad or not objective.missing_risks.requires_grad:
        raise PORDV5TrainingError("group risks must require gradients for an audit")
    gradients = torch.autograd.grad(
        objective.objective,
        (objective.full_risks, objective.missing_risks),
        retain_graph=True,
        allow_unused=True,
    )
    resolved = tuple(
        torch.zeros_like(parent) if value is None else value
        for value, parent in zip(
            gradients, (objective.full_risks, objective.missing_risks), strict=True
        )
    )
    if any(not bool(torch.isfinite(value).all().item()) for value in resolved):
        raise PORDV5TrainingError("group-risk gradient audit is non-finite")
    return resolved  # type: ignore[return-value]


def direct_row_loss_gradients(objective: ObjectiveTensors) -> tuple[Any, Any] | None:
    """Return row-loss gradients, exposing stop-gradient control differences."""

    import torch

    if objective.row_full_losses is None or objective.row_missing_losses is None:
        return None
    gradients = torch.autograd.grad(
        objective.objective,
        (objective.row_full_losses, objective.row_missing_losses),
        retain_graph=True,
        allow_unused=True,
    )
    resolved = tuple(
        torch.zeros_like(parent) if value is None else value
        for value, parent in zip(
            gradients,
            (objective.row_full_losses, objective.row_missing_losses),
            strict=True,
        )
    )
    if any(not bool(torch.isfinite(value).all().item()) for value in resolved):
        raise PORDV5TrainingError("row-loss gradient audit is non-finite")
    return resolved  # type: ignore[return-value]


def _tensor_values(value: Any) -> Any:
    if value is None:
        return None
    array = value.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(array).all():
        raise PORDV5TrainingError("audit tensor contains non-finite values")
    return array.tolist()


def objective_audit_record(
    objective: ObjectiveTensors,
    *,
    full_gradient: Any,
    missing_gradient: Any,
    row_gradients: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Materialize exact risks, atoms, interaction, tails, and gradients."""

    import torch

    interaction_delta = None
    interaction_tolerance = None
    if objective.method_id == "group_mean_pord":
        missing_centered = two_way_center(objective.missing_risks)
        interaction_delta = float(
            (objective.interaction - missing_centered).detach().abs().max().item()
        )
        interaction_scale = max(
            1.0,
            float(objective.missing_risks.detach().abs().max().item()),
            float(objective.paired_excess.detach().abs().max().item()),
        )
        interaction_tolerance = (
            64.0
            * float(torch.finfo(objective.missing_risks.dtype).eps)
            * interaction_scale
        )
        if interaction_delta > interaction_tolerance:
            raise PORDV5TrainingError(
                "group-mean interaction no longer equals missing-risk nonadditivity"
            )
    weights = objective.tail_weights.detach().cpu().numpy()
    active = [
        {
            "segment_index": int(segment),
            "sensor_index": int(sensor),
            "weight": float(weights[segment, sensor]),
        }
        for segment, sensor in zip(*np.nonzero(weights > 0.0), strict=True)
    ]
    record = {
        "method_id": objective.method_id,
        "objective": float(objective.objective.detach().item()),
        "mask_risk": float(objective.mask_risk.detach().item()),
        "tail_penalty": float(objective.tail_penalty.detach().item()),
        "interaction_penalty": float(objective.interaction_penalty.detach().item()),
        "vrex_population_weighted_variance": float(objective.vrex_penalty.detach().item()),
        "mofe_segment_balanced_per_row_hinge": float(objective.mofe_penalty.detach().item()),
        "full_group_risks": _tensor_values(objective.full_risks),
        "missing_group_risks": _tensor_values(objective.missing_risks),
        "objective_atom_values": _tensor_values(objective.positive_excess),
        "row_pair_excess": _tensor_values(objective.row_pair_excess),
        "row_full_reference": _tensor_values(objective.row_full_reference),
        "pairing_mode": objective.pairing_mode,
        "full_reference_detached": objective.full_reference_detached,
        "pairing_map_sha256": objective.pairing_map_sha256,
        "interaction_two_way_centered_atoms": _tensor_values(objective.interaction),
        "group_mean_interaction_missing_nonadditivity_max_abs_delta": interaction_delta,
        "group_mean_interaction_numeric_tolerance": interaction_tolerance,
        "tail_fixed_k": PORD_TOP_K,
        "tail_cutoff": float(objective.tail_cutoff),
        "tail_symmetric_weights": _tensor_values(objective.tail_weights),
        "tail_active_cells": active,
        "direct_full_group_gradient": _tensor_values(full_gradient),
        "direct_missing_group_gradient": _tensor_values(missing_gradient),
        "direct_full_row_loss_gradient": (
            None if row_gradients is None else _tensor_values(row_gradients[0])
        ),
        "direct_missing_row_loss_gradient": (
            None if row_gradients is None else _tensor_values(row_gradients[1])
        ),
        "groupdro_weights_before": _tensor_values(objective.groupdro_weights_before),
        "groupdro_weights_after": _tensor_values(objective.next_groupdro_weights),
        "mofe_active_hinge_sha256": (
            None
            if objective.mofe_hinge is None
            else array_sha256(
                (objective.mofe_hinge.detach().cpu().numpy() > 0).astype(np.uint8)
            )
        ),
        "relu_and_abs_zero_subgradient": 0.0,
    }
    record["audit_sha256"] = _canonical_json_sha256(record)
    return record


def outage_checkpoint_score(outage_rmse_3d_mm: Sequence[Any]) -> float:
    """Return ``.5 * (mean four outage RMSE + worst outage RMSE)``."""

    try:
        values = np.asarray(outage_rmse_3d_mm, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PORDV5TrainingError("outage RMSE values must be numeric") from exc
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values < 0):
        raise PORDV5TrainingError("checkpoint scoring requires four finite RMSE values")
    return float(0.5 * (values.mean() + values.max()))


def earliest_checkpoint_epoch(history: Sequence[Mapping[str, Any]]) -> int:
    """Replay the strict-less earliest-minimum checkpoint rule."""

    if not history:
        raise PORDV5TrainingError("checkpoint history is empty")
    best_epoch = 0
    best_score = math.inf
    for expected_epoch, row in enumerate(history, start=1):
        if not isinstance(row, Mapping):
            raise PORDV5TrainingError("checkpoint history row is not a mapping")
        epoch_value = row.get("epoch")
        if isinstance(epoch_value, (bool, np.bool_)) or not isinstance(
            epoch_value, numbers.Integral
        ):
            raise PORDV5TrainingError("checkpoint history epoch is not an integer")
        if int(epoch_value) != expected_epoch:
            raise PORDV5TrainingError("checkpoint history epochs are not contiguous")
        try:
            score = float(row["checkpoint_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PORDV5TrainingError("checkpoint history score is missing") from exc
        if not math.isfinite(score):
            raise PORDV5TrainingError("checkpoint history score is non-finite")
        improved = score < best_score
        if "checkpoint_improved" in row and bool(row["checkpoint_improved"]) is not improved:
            raise PORDV5TrainingError("checkpoint improved flag changed")
        if improved:
            best_score = score
            best_epoch = expected_epoch
        if "best_epoch_so_far" in row and int(row["best_epoch_so_far"]) != best_epoch:
            raise PORDV5TrainingError("checkpoint best-epoch trace changed")
    return best_epoch


def _rmse_3d_mm(truth: np.ndarray, prediction: np.ndarray) -> float:
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 3:
        raise PORDV5TrainingError("RMSE arrays must share shape [rows, 3]")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise PORDV5TrainingError("RMSE arrays contain non-finite values")
    return float(np.sqrt(np.mean(np.square(truth - prediction).sum(axis=1))))


def _configure_determinism(torch: Any, *, model_seed: int) -> None:
    torch.manual_seed(int(model_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(model_seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    state = {
        str(name): value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    state_dict_sha256(state)
    return state


def initial_pord_state_sha256(*, backbone_id: str, model_seed: int) -> str:
    """Recreate the paired pre-optimizer initialization on CPU."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if isinstance(model_seed, (bool, np.bool_)) or not isinstance(
        model_seed, numbers.Integral
    ):
        raise PORDV5TrainingError("model_seed must be an integer")
    _configure_determinism(torch, model_seed=int(model_seed))
    model = build_pord_backbone(torch, backbone_id=str(backbone_id))
    value = state_dict_sha256(_cpu_state_dict(model))
    del model
    return value


def _parameter_gradient_audit(model: Any) -> tuple[str, float, int, list[str]]:
    hasher = hashlib.sha256()
    squared_norm = 0.0
    tensor_count = 0
    absent_names: list[str] = []
    for name, parameter in sorted(model.named_parameters(), key=lambda pair: pair[0]):
        gradient = parameter.grad
        if gradient is None:
            # The reused endpoint-only backbone owns an unused reconstruction
            # head.  No auxiliary loss is allowed, so those gradients remain
            # absent and are explicitly bound into this audit.
            hasher.update(str(name).encode("utf-8") + b"\x00NONE\xff")
            absent_names.append(str(name))
            continue
        detached = gradient.detach()
        if not bool(detached.isfinite().all().item()):
            raise PORDV5TrainingError("model parameter gradient is non-finite")
        array = detached.cpu().numpy()
        hasher.update(str(name).encode("utf-8") + b"\x00")
        hasher.update(array_sha256(array).encode("ascii") + b"\xff")
        squared_norm += float(detached.double().square().sum().item())
        tensor_count += 1
    if tensor_count <= 0:
        raise PORDV5TrainingError("training produced no parameter gradients")
    unexpected_absent = [
        name for name in absent_names if not name.startswith("reconstruction_head.")
    ]
    if unexpected_absent:
        raise PORDV5TrainingError(
            f"unexpected model parameters lacked gradients: {unexpected_absent}"
        )
    return hasher.hexdigest(), math.sqrt(squared_norm), tensor_count, absent_names


def _predict_views(model: Any, feature_views: Any, *, torch: Any) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        flat = feature_views.reshape(-1, V3_INPUT_FEATURE_COUNT)
        predictions = model(flat).reshape(
            int(feature_views.shape[0]), int(feature_views.shape[1]), 3
        )
    if not bool(torch.isfinite(predictions).all().item()):
        raise PORDV5TrainingError("model emitted non-finite predictions")
    return predictions.detach().cpu().numpy().astype(np.float64)


def _result_hash_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "method_id",
        "backbone_id",
        "held_out_segment",
        "model_seed",
        "epoch_budget",
        "optimizer_update_count",
        "training_bundle_sha256",
        "base_train_view_bundle_sha256",
        "train_view_bundle_sha256",
        "availability_semantics",
        "gradient_view_semantics",
        "pre_registered_wrong_pair_map_sha256",
        "pairing_mode",
        "full_reference_detached",
        "initial_state_sha256",
        "selected_state_sha256",
        "best_epoch",
        "best_checkpoint_score",
        "best_selection_outage_rmse_3d_mm",
        "best_selection_full_rmse_3d_mm",
        "history_sha256",
        "target_transform_sha256",
        "target_mean_mm",
        "target_scale_mm",
        "checkpoint_selection_scope",
        "outer_eval_count",
        "outer_labels_accessed",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise PORDV5TrainingError(f"result hash fields are missing: {missing}")
    return {key: result[key] for key in required}


def verify_pord_result_hash(result: Mapping[str, Any]) -> None:
    """Fail closed if any bound training-result summary field changed."""

    if not isinstance(result, Mapping) or "result_sha256" not in result:
        raise PORDV5TrainingError("PORD result hash is missing")
    if "state_dict" not in result or state_dict_sha256(result["state_dict"]) != str(
        result.get("selected_state_sha256", "")
    ):
        raise PORDV5TrainingError("PORD selected state hash changed")
    try:
        history = result["history"]
    except KeyError as exc:
        raise PORDV5TrainingError("PORD result artifacts are missing") from exc
    if _canonical_json_sha256(history) != str(result.get("history_sha256", "")):
        raise PORDV5TrainingError("PORD history hash changed")
    if (
        result.get("checkpoint_selection_scope")
        != PORD_CHECKPOINT_SELECTION_SCOPE
        or result.get("outer_eval_count") != 0
        or result.get("outer_labels_accessed") is not False
    ):
        raise PORDV5TrainingError(
            "PORD checkpoint selection must remain train-only before outer evaluation"
        )
    best_epoch = earliest_checkpoint_epoch(history)
    if best_epoch != int(result.get("best_epoch", -1)):
        raise PORDV5TrainingError("PORD best epoch changed")
    selected_history = history[best_epoch - 1]
    try:
        outage_values = np.asarray(
            result["best_selection_outage_rmse_3d_mm"], dtype=np.float64
        )
        selected_outage_values = np.asarray(
            selected_history["selection_outage_rmse_3d_mm"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PORDV5TrainingError("PORD selected metrics are malformed") from exc
    if outage_values.shape != (4,) or not np.array_equal(
        outage_values, selected_outage_values
    ):
        raise PORDV5TrainingError("PORD selected outage metrics changed")
    if not math.isclose(
        float(result.get("best_checkpoint_score", math.nan)),
        outage_checkpoint_score(outage_values),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise PORDV5TrainingError("PORD checkpoint score changed")
    if not math.isclose(
        float(result.get("best_selection_full_rmse_3d_mm", math.nan)),
        float(selected_history.get("selection_full_rmse_3d_mm", math.nan)),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise PORDV5TrainingError("PORD selected full-sensor metric changed")
    _target_transform(result.get("target_mean_mm"), result.get("target_scale_mm"))
    method = str(result.get("method_id"))
    if method not in PORD_METHOD_IDS:
        raise PORDV5TrainingError("PORD result method changed")
    expected_availability = (
        "constant_all_available_negative_control"
        if method == "constant_availability_erm"
        else "physical_mask_state"
    )
    expected_gradient = (
        "full_view_gradient_only_outage_selected_checkpoint"
        if method == "full_gradient_only_mask_selected_control"
        else "five_view_mask_weighted_gradient"
    )
    expected_pairing = {
        "group_mean_pord": "group_mean_difference_not_pair_specific",
        "paired_tail": "correct_uid_stop_gradient",
        "paired_interaction": "correct_uid_stop_gradient",
        "paired_pord": "correct_uid_stop_gradient",
        "paired_wrong_pair": "sorted_uid_within_segment_circular_derangement",
        "paired_no_stopgrad": "correct_uid_no_stop_gradient",
    }.get(method, "none")
    if (
        result.get("availability_semantics") != expected_availability
        or result.get("gradient_view_semantics") != expected_gradient
        or result.get("pairing_mode") != expected_pairing
        or result.get("full_reference_detached")
        is not (method != "paired_no_stopgrad")
        or not isinstance(result.get("pre_registered_wrong_pair_map_sha256"), str)
        or len(str(result.get("pre_registered_wrong_pair_map_sha256"))) != 64
        or (
            method == "constant_availability_erm"
            and result.get("train_view_bundle_sha256")
            == result.get("base_train_view_bundle_sha256")
        )
        or (
            method != "constant_availability_erm"
            and result.get("train_view_bundle_sha256")
            != result.get("base_train_view_bundle_sha256")
        )
        or any(
            row.get("objective_audit", {}).get("pairing_mode") != expected_pairing
            for row in history
        )
        or any(
            row.get("objective_audit", {}).get("full_reference_detached")
            is not (method != "paired_no_stopgrad")
            for row in history
        )
        or any(
            row.get("objective_audit", {}).get("pairing_map_sha256")
            != (
                result.get("pre_registered_wrong_pair_map_sha256")
                if method == "paired_wrong_pair"
                else None
            )
            for row in history
        )
    ):
        raise PORDV5TrainingError("PORD method-specific result semantics changed")
    expected = _canonical_json_sha256(_result_hash_payload(result))
    if str(result["result_sha256"]) != expected:
        raise PORDV5TrainingError("PORD result hash changed")


def train_pord_fold_seed(
    *,
    X_train_scaled: Any,
    y_train_scaled: Any,
    train_segment_ids: Sequence[Any],
    train_trial_uids: Sequence[Any],
    held_out_segment: Any,
    target_mean_mm: Any,
    target_scale_mm: Any,
    method_id: str,
    model_seed: int,
    device: str,
    backbone_id: str = PORD_PRIMARY_BACKBONE_ID,
    epoch_budget: int = PORD_EPOCH_BUDGET,
) -> dict[str, Any]:
    """Fit one method/fold/seed and select from training segments only.

    Outer-fold features, labels, and UIDs are deliberately absent from this
    API.  The checkpoint is fixed using the same four-outage criterion on the
    three training segments, after which :func:`evaluate_pord_checkpoint_once`
    is the only API that may evaluate the held-out segment.

    ``epoch_budget=400`` is the controlled experiment.  A smaller positive
    value is accepted only to make deterministic CPU smoke/property tests
    practical; the returned ``contract_epoch_budget_complete`` flag remains
    false and such a result must never be promoted as a v5 experiment result.
    """

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    method = str(method_id)
    if method not in PORD_METHOD_IDS:
        raise PORDV5TrainingError(f"unknown PORD method: {method!r}")
    resolved_backbone_id = str(backbone_id)
    if resolved_backbone_id not in PORD_BACKBONE_IDS:
        raise PORDV5TrainingError(f"unknown PORD backbone: {resolved_backbone_id!r}")
    if (
        resolved_backbone_id == PORD_CHAIN_TRANSFER_BACKBONE_ID
        and method not in PORD_TRANSFER_METHOD_IDS
    ):
        raise PORDV5TrainingError(
            "chain transfer is pre-registered only for masked ERM and paired PORD"
        )
    if isinstance(model_seed, (bool, np.bool_)) or not isinstance(
        model_seed, numbers.Integral
    ):
        raise PORDV5TrainingError("model_seed must be an integer")
    if int(model_seed) < 0:
        raise PORDV5TrainingError("model_seed must be non-negative")
    if isinstance(epoch_budget, (bool, np.bool_)) or not isinstance(
        epoch_budget, numbers.Integral
    ):
        raise PORDV5TrainingError("epoch_budget must be an integer")
    epochs = int(epoch_budget)
    if epochs <= 0 or epochs > PORD_EPOCH_BUDGET:
        raise PORDV5TrainingError("epoch_budget must lie in [1, 400]")

    X_train = _matrix(
        X_train_scaled, name="X_train_scaled", columns=V3_SENSOR_FEATURE_COUNT
    )
    y_train = _matrix(
        y_train_scaled, name="y_train_scaled", columns=3, rows=len(X_train)
    )
    train_uids = _uids(
        train_trial_uids, rows=len(X_train), name="train_trial_uids"
    )
    segment_indices, segment_labels = _segments(
        train_segment_ids, rows=len(X_train)
    )
    if held_out_segment is None or (
        isinstance(held_out_segment, (float, np.floating))
        and not math.isfinite(float(held_out_segment))
    ):
        raise PORDV5TrainingError("held_out_segment is missing or non-finite")
    holdout_token = str(held_out_segment)
    if not holdout_token or holdout_token in set(segment_labels):
        raise PORDV5TrainingError(
            "held_out_segment must be non-empty and absent from train segments"
        )
    target_mean, target_scale = _target_transform(
        target_mean_mm, target_scale_mm
    )
    train_truth_mm = inverse_standardized_targets(
        y_train, target_mean_mm=target_mean, target_scale_mm=target_scale
    )

    base_train_views = build_exact_five_views(X_train, trial_uids=train_uids)
    train_views = _views_for_method(
        X_train, trial_uids=train_uids, method_id=method
    )
    wrong_pair_map = fixed_within_segment_circular_derangement(
        trial_uids=train_uids,
        segment_index=segment_indices,
    )
    resolved_device = torch.device(str(device))
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise PORDV5TrainingError("CUDA training requested without an available GPU")
    _configure_determinism(torch, model_seed=int(model_seed))
    model = build_pord_backbone(
        torch, backbone_id=resolved_backbone_id
    ).to(resolved_device)
    initial_state = _cpu_state_dict(model)
    initial_state_hash = state_dict_sha256(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PORD_LEARNING_RATE, weight_decay=PORD_WEIGHT_DECAY
    )

    train_features = torch.as_tensor(
        train_views.features, dtype=torch.float32, device=resolved_device
    )
    train_targets = torch.as_tensor(
        y_train, dtype=torch.float32, device=resolved_device
    )
    train_groups = torch.as_tensor(
        segment_indices, dtype=torch.long, device=resolved_device
    )
    groupdro_weights = None

    target_transform_hash = _canonical_json_sha256(
        {
            "target_mean_mm": target_mean.tolist(),
            "target_scale_mm": target_scale.tolist(),
            "train_inverse_target_sha256": array_sha256(train_truth_mm),
        }
    )
    training_bundle_hash = _canonical_json_sha256(
        {
            "base_train_views_sha256": base_train_views.sha256,
            "train_targets_sha256": array_sha256(y_train),
            "train_segment_index_sha256": array_sha256(segment_indices),
            "train_segment_labels": list(segment_labels),
            "held_out_segment": holdout_token,
            "target_transform_sha256": target_transform_hash,
            "model_seed": int(model_seed),
            "epoch_budget": epochs,
            "backbone": PORD_BACKBONE_SETTINGS_BY_ID[resolved_backbone_id],
            "optimizer": PORD_OPTIMIZER_SETTINGS,
            "views": list(PORD_VIEW_NAMES),
            "mask_probabilities": list(PORD_MASK_PROBABILITIES),
            "pre_registered_wrong_pair_map_sha256": wrong_pair_map.sha256,
            "checkpoint_selection_scope": PORD_CHECKPOINT_SELECTION_SCOPE,
            "outer_eval_count": 0,
        }
    )

    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    best_predictions_mm: np.ndarray | None = None
    best_outage_rmse: list[float] | None = None
    best_full_rmse = math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(
            train_features.reshape(-1, V3_INPUT_FEATURE_COUNT)
        ).reshape(5, len(X_train), 3)
        risks = compute_empirical_group_risks(
            predictions,
            train_targets,
            train_groups,
            segment_labels=segment_labels,
        )
        controlled = compute_controlled_objective(
            method_id=method,
            full_risks=risks.full,
            missing_risks=risks.missing,
            row_full_losses=risks.row_full,
            row_missing_losses=risks.row_missing,
            segment_index=risks.segment_index,
            full_reference_index=(
                wrong_pair_map.indices if method == "paired_wrong_pair" else None
            ),
            pairing_map_sha256=(
                wrong_pair_map.sha256 if method == "paired_wrong_pair" else None
            ),
            groupdro_weights=groupdro_weights,
        )
        full_gradient, missing_gradient = direct_group_risk_gradients(controlled)
        row_gradients = direct_row_loss_gradients(controlled)
        audit = objective_audit_record(
            controlled,
            full_gradient=full_gradient,
            missing_gradient=missing_gradient,
            row_gradients=row_gradients,
        )
        controlled.objective.backward()
        (
            gradient_hash,
            gradient_l2,
            gradient_tensor_count,
            gradient_absent_names,
        ) = _parameter_gradient_audit(model)
        optimizer.step()
        if controlled.next_groupdro_weights is not None:
            groupdro_weights = controlled.next_groupdro_weights.detach()

        prediction_scaled = _predict_views(model, train_features, torch=torch)
        prediction_mm = (
            prediction_scaled * target_scale.reshape(1, 1, 3)
            + target_mean.reshape(1, 1, 3)
        )
        if not np.isfinite(prediction_mm).all():
            raise PORDV5TrainingError("inverse train prediction is non-finite")
        full_rmse = _rmse_3d_mm(train_truth_mm, prediction_mm[0])
        outage_rmse = [
            _rmse_3d_mm(train_truth_mm, prediction_mm[index])
            for index in range(1, 5)
        ]
        checkpoint_score = outage_checkpoint_score(outage_rmse)
        improved = checkpoint_score < best_score
        if improved:
            best_score = checkpoint_score
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            best_predictions_mm = prediction_mm.copy()
            best_outage_rmse = list(outage_rmse)
            best_full_rmse = full_rmse
        history.append(
            {
                "epoch": epoch,
                "optimizer_update_index": epoch,
                "objective_audit": audit,
                "parameter_gradient_sha256": gradient_hash,
                "parameter_gradient_l2": gradient_l2,
                "parameter_gradient_tensor_count": gradient_tensor_count,
                "parameter_gradient_absent_names": gradient_absent_names,
                "checkpoint_selection_scope": PORD_CHECKPOINT_SELECTION_SCOPE,
                "outer_eval_count": 0,
                "outer_labels_accessed": False,
                "selection_full_rmse_3d_mm": full_rmse,
                "selection_outage_rmse_3d_mm": outage_rmse,
                "checkpoint_score": checkpoint_score,
                "checkpoint_improved": bool(improved),
                "best_epoch_so_far": best_epoch,
                "same_checkpoint_for_all_four_outages": True,
                "auxiliary_reconstruction_used": False,
            }
        )

    if best_state is None or best_predictions_mm is None or best_outage_rmse is None:
        raise PORDV5TrainingError("training found no finite checkpoint")
    if earliest_checkpoint_epoch(history) != best_epoch:
        raise AssertionError("checkpoint history does not replay to the selected epoch")
    model.load_state_dict(best_state, strict=True)
    replay_scaled = _predict_views(model, train_features, torch=torch)
    replay_mm = (
        replay_scaled * target_scale.reshape(1, 1, 3)
        + target_mean.reshape(1, 1, 3)
    )
    if not np.array_equal(replay_mm, best_predictions_mm):
        raise PORDV5TrainingError("selected checkpoint does not replay exact predictions")
    selected_state_hash = state_dict_sha256(best_state)
    history_hash = _canonical_json_sha256(history)
    result: dict[str, Any] = {
        "contract": "alphaf.endpoint_regression.gpu_outage_pord.v5.training_core",
        "method_id": method,
        "backbone_id": resolved_backbone_id,
        "held_out_segment": holdout_token,
        "train_segment_labels": list(segment_labels),
        "model_seed": int(model_seed),
        "epoch_budget": epochs,
        "contract_epoch_budget_complete": epochs == PORD_EPOCH_BUDGET,
        "optimizer_update_count": epochs,
        "optimizer": dict(PORD_OPTIMIZER_SETTINGS),
        "backbone": dict(PORD_BACKBONE_SETTINGS_BY_ID[resolved_backbone_id]),
        "training_view_names": list(PORD_VIEW_NAMES),
        "mask_probabilities": list(PORD_MASK_PROBABILITIES),
        "base_train_view_bundle_sha256": base_train_views.sha256,
        "train_view_bundle_sha256": train_views.sha256,
        "availability_semantics": train_views.availability_semantics,
        "gradient_view_semantics": (
            "full_view_gradient_only_outage_selected_checkpoint"
            if method == "full_gradient_only_mask_selected_control"
            else "five_view_mask_weighted_gradient"
        ),
        "pre_registered_wrong_pair_map_sha256": wrong_pair_map.sha256,
        "pairing_mode": history[0]["objective_audit"]["pairing_mode"],
        "full_reference_detached": history[0]["objective_audit"][
            "full_reference_detached"
        ],
        "training_bundle_sha256": training_bundle_hash,
        "target_mean_mm": target_mean.tolist(),
        "target_scale_mm": target_scale.tolist(),
        "target_transform_sha256": target_transform_hash,
        "initial_state_sha256": initial_state_hash,
        "selected_state_sha256": selected_state_hash,
        "best_epoch": best_epoch,
        "best_checkpoint_score": best_score,
        "best_selection_outage_rmse_3d_mm": best_outage_rmse,
        "best_selection_full_rmse_3d_mm": best_full_rmse,
        "checkpoint_selection_scope": PORD_CHECKPOINT_SELECTION_SCOPE,
        "checkpoint_criterion": PORD_CHECKPOINT_CRITERION,
        "outer_eval_count": 0,
        "outer_labels_accessed": False,
        "checkpoint_evaluated_view_names": list(PORD_VIEW_NAMES),
        "same_checkpoint_for_all_four_outages": True,
        "state_dict": best_state,
        "history": history,
        "history_sha256": history_hash,
    }
    result["result_sha256"] = _canonical_json_sha256(_result_hash_payload(result))
    verify_pord_result_hash(result)
    del model, optimizer
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _outer_evaluation_hash_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "contract",
        "training_result_sha256",
        "selected_state_sha256",
        "backbone_id",
        "held_out_segment",
        "checkpoint_selection_scope",
        "checkpoint_fixed_before_outer_labels",
        "outer_eval_count",
        "base_outer_view_bundle_sha256",
        "outer_view_bundle_sha256",
        "availability_semantics",
        "outer_trial_uid_sha256",
        "outer_truth_raw_mm_sha256",
        "outer_prediction_raw_mm_sha256",
        "target_transform_sha256",
        "evaluated_view_names",
        "same_checkpoint_for_all_four_outages",
        "outer_full_rmse_3d_mm",
        "outer_outage_rmse_3d_mm",
    )
    missing = [key for key in required if key not in receipt]
    if missing:
        raise PORDV5TrainingError(
            f"outer evaluation receipt fields are missing: {missing}"
        )
    return {key: receipt[key] for key in required}


def verify_pord_outer_evaluation(
    receipt: Mapping[str, Any],
    *,
    training_result: Mapping[str, Any],
    outer_trial_uids: Sequence[Any],
    truth_raw_mm: Any,
    predictions_raw_mm: Any,
) -> None:
    """Authenticate the single post-checkpoint held-out evaluation receipt."""

    verify_pord_result_hash(training_result)
    if type(receipt) is not dict:
        raise PORDV5TrainingError("outer evaluation receipt must be an exact object")
    expected_keys = {
        *_outer_evaluation_hash_payload(receipt),
        "outer_evaluation_sha256",
    }
    sha_fields = (
        "training_result_sha256",
        "selected_state_sha256",
        "base_outer_view_bundle_sha256",
        "outer_view_bundle_sha256",
        "outer_trial_uid_sha256",
        "outer_truth_raw_mm_sha256",
        "outer_prediction_raw_mm_sha256",
        "target_transform_sha256",
        "outer_evaluation_sha256",
    )
    outage_metrics = receipt.get("outer_outage_rmse_3d_mm")
    if (
        type(receipt) is not dict
        or set(receipt) != expected_keys
        or any(
            type(receipt.get(key)) is not str
            or len(receipt[key]) != 64
            or any(character not in "0123456789abcdef" for character in receipt[key])
            for key in sha_fields
        )
        or type(receipt.get("held_out_segment")) is not str
        or type(receipt.get("outer_eval_count")) is not int
        or type(receipt.get("evaluated_view_names")) is not list
        or any(
            type(value) is not str for value in receipt["evaluated_view_names"]
        )
        or type(receipt.get("outer_full_rmse_3d_mm")) is not float
        or not math.isfinite(receipt["outer_full_rmse_3d_mm"])
        or receipt["outer_full_rmse_3d_mm"] < 0.0
        or type(outage_metrics) is not list
        or len(outage_metrics) != 4
        or any(
            type(value) is not float or not math.isfinite(value) or value < 0.0
            for value in outage_metrics
        )
    ):
        raise PORDV5TrainingError("outer evaluation receipt schema changed")
    truth = _matrix(
        truth_raw_mm, name="outer truth raw mm", columns=3, dtype=np.float64
    )
    predictions = np.asarray(predictions_raw_mm, dtype=np.float64)
    if (
        predictions.shape != (5, len(truth), 3)
        or not np.isfinite(predictions).all()
    ):
        raise PORDV5TrainingError("outer predictions must have finite shape [5,N,3]")
    uids = _uids(outer_trial_uids, rows=len(truth), name="outer_trial_uids")
    outage_rmse = [
        _rmse_3d_mm(truth, predictions[index]) for index in range(1, 5)
    ]
    if (
        receipt.get("contract")
        != "alphaf.endpoint_regression.gpu_outage_pord.v5.outer_evaluation"
        or receipt.get("training_result_sha256")
        != training_result.get("result_sha256")
        or receipt.get("selected_state_sha256")
        != training_result.get("selected_state_sha256")
        or receipt.get("backbone_id") != training_result.get("backbone_id")
        or receipt.get("held_out_segment")
        != training_result.get("held_out_segment")
        or receipt.get("checkpoint_selection_scope")
        != PORD_CHECKPOINT_SELECTION_SCOPE
        or receipt.get("checkpoint_fixed_before_outer_labels") is not True
        or receipt.get("outer_eval_count") != 1
        or receipt.get("base_outer_view_bundle_sha256") is None
        or receipt.get("outer_view_bundle_sha256") is None
        or receipt.get("availability_semantics")
        != training_result.get("availability_semantics")
        or receipt.get("outer_trial_uid_sha256")
        != _canonical_json_sha256(list(uids))
        or receipt.get("outer_truth_raw_mm_sha256") != array_sha256(truth)
        or receipt.get("outer_prediction_raw_mm_sha256")
        != array_sha256(predictions)
        or receipt.get("target_transform_sha256")
        != training_result.get("target_transform_sha256")
        or tuple(receipt.get("evaluated_view_names", ())) != PORD_VIEW_NAMES
        or receipt.get("same_checkpoint_for_all_four_outages") is not True
        or not math.isclose(
            receipt["outer_full_rmse_3d_mm"],
            _rmse_3d_mm(truth, predictions[0]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not np.array_equal(
            np.asarray(outage_metrics, dtype=np.float64),
            np.asarray(outage_rmse, dtype=np.float64),
        )
    ):
        raise PORDV5TrainingError("outer evaluation receipt binding changed")
    expected_hash = _canonical_json_sha256(_outer_evaluation_hash_payload(receipt))
    if receipt.get("outer_evaluation_sha256") != expected_hash:
        raise PORDV5TrainingError("outer evaluation receipt hash changed")


def predict_pord_checkpoint_outer(
    *,
    training_result: Mapping[str, Any],
    X_outer_scaled: Any,
    outer_segment_ids: Sequence[Any],
    outer_trial_uids: Sequence[Any],
    held_out_segment: Any,
    target_mean_mm: Any,
    target_scale_mm: Any,
    device: str,
) -> dict[str, Any]:
    """Generate feature-only outer predictions after checkpoint selection."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    verify_pord_result_hash(training_result)
    holdout_token = str(held_out_segment)
    if holdout_token != str(training_result.get("held_out_segment", "")):
        raise PORDV5TrainingError("outer segment differs from the training receipt")
    X_outer = _matrix(
        X_outer_scaled, name="X_outer_scaled", columns=V3_SENSOR_FEATURE_COUNT
    )
    outer_uids = _uids(
        outer_trial_uids, rows=len(X_outer), name="outer_trial_uids"
    )
    outer_segments = np.asarray(outer_segment_ids, dtype=object)
    if (
        outer_segments.ndim != 1
        or len(outer_segments) != len(X_outer)
        or any(str(value) != holdout_token for value in outer_segments.tolist())
    ):
        raise PORDV5TrainingError("outer rows do not belong to the held-out segment")
    target_mean, target_scale = _target_transform(
        target_mean_mm, target_scale_mm
    )
    if (
        not np.array_equal(
            target_mean,
            np.asarray(training_result.get("target_mean_mm"), dtype=np.float64),
        )
        or not np.array_equal(
            target_scale,
            np.asarray(training_result.get("target_scale_mm"), dtype=np.float64),
        )
    ):
        raise PORDV5TrainingError("outer target transform differs from training")
    base_outer_views = build_exact_five_views(X_outer, trial_uids=outer_uids)
    outer_views = (
        build_constant_availability_five_views(X_outer, trial_uids=outer_uids)
        if str(training_result["method_id"]) == "constant_availability_erm"
        else base_outer_views
    )
    if outer_views.availability_semantics != training_result.get(
        "availability_semantics"
    ):
        raise PORDV5TrainingError("outer availability semantics differ from training")
    resolved_device = torch.device(str(device))
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise PORDV5TrainingError("CUDA evaluation requested without an available GPU")
    _configure_determinism(torch, model_seed=int(training_result["model_seed"]))
    model = build_pord_backbone(
        torch, backbone_id=str(training_result["backbone_id"])
    ).to(resolved_device)
    model.load_state_dict(training_result["state_dict"], strict=True)
    outer_features = torch.as_tensor(
        outer_views.features, dtype=torch.float32, device=resolved_device
    )
    predictions_scaled = _predict_views(model, outer_features, torch=torch)
    predictions_raw_mm = (
        predictions_scaled * target_scale.reshape(1, 1, 3)
        + target_mean.reshape(1, 1, 3)
    )
    if not np.isfinite(predictions_raw_mm).all():
        raise PORDV5TrainingError("inverse outer prediction is non-finite")
    prediction_receipt = {
        "contract": "alphaf.endpoint_regression.gpu_outage_pord.v5.outer_prediction",
        "training_result_sha256": str(training_result["result_sha256"]),
        "selected_state_sha256": str(training_result["selected_state_sha256"]),
        "backbone_id": str(training_result["backbone_id"]),
        "held_out_segment": holdout_token,
        "checkpoint_selection_scope": PORD_CHECKPOINT_SELECTION_SCOPE,
        "checkpoint_fixed_before_outer_prediction": True,
        "base_outer_view_bundle_sha256": base_outer_views.sha256,
        "outer_view_bundle_sha256": outer_views.sha256,
        "availability_semantics": outer_views.availability_semantics,
        "outer_trial_uid_sha256": _canonical_json_sha256(list(outer_uids)),
        "outer_prediction_raw_mm_sha256": array_sha256(predictions_raw_mm),
        "target_transform_sha256": str(training_result["target_transform_sha256"]),
        "predicted_view_names": list(PORD_VIEW_NAMES),
        "same_checkpoint_for_all_four_outages": True,
        "outer_labels_accessed": False,
    }
    prediction_receipt["outer_prediction_receipt_sha256"] = _canonical_json_sha256(
        prediction_receipt
    )
    del model
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "receipt": prediction_receipt,
        "trial_uids": list(outer_uids),
        "predictions_raw_mm": predictions_raw_mm,
    }


def evaluate_pord_outer_predictions_once(
    *,
    training_result: Mapping[str, Any],
    outer_prediction: Mapping[str, Any],
    y_outer_raw_mm: Any,
) -> dict[str, Any]:
    """Evaluate one frozen feature-only outer prediction without model calls."""

    verify_pord_result_hash(training_result)
    prediction_receipt = outer_prediction.get("receipt")
    predictions_raw_mm = np.asarray(
        outer_prediction.get("predictions_raw_mm"), dtype=np.float64
    )
    outer_uids = _uids(
        outer_prediction.get("trial_uids", ()),
        rows=int(predictions_raw_mm.shape[1]) if predictions_raw_mm.ndim == 3 else 0,
        name="outer_trial_uids",
    )
    if (
        not isinstance(prediction_receipt, Mapping)
        or predictions_raw_mm.shape != (5, len(outer_uids), 3)
        or not np.isfinite(predictions_raw_mm).all()
        or prediction_receipt.get("outer_labels_accessed") is not False
        or prediction_receipt.get("training_result_sha256")
        != training_result.get("result_sha256")
        or prediction_receipt.get("availability_semantics")
        != training_result.get("availability_semantics")
        or not isinstance(
            prediction_receipt.get("base_outer_view_bundle_sha256"), str
        )
        or prediction_receipt.get("outer_trial_uid_sha256")
        != _canonical_json_sha256(list(outer_uids))
        or prediction_receipt.get("outer_prediction_raw_mm_sha256")
        != array_sha256(predictions_raw_mm)
        or prediction_receipt.get("outer_prediction_receipt_sha256")
        != _canonical_json_sha256(
            {
                key: value
                for key, value in prediction_receipt.items()
                if key != "outer_prediction_receipt_sha256"
            }
        )
    ):
        raise PORDV5TrainingError("outer feature-only prediction receipt changed")
    truth_raw_mm = _matrix(
        y_outer_raw_mm,
        name="y_outer_raw_mm",
        columns=3,
        rows=len(outer_uids),
        dtype=np.float64,
    )
    receipt: dict[str, Any] = {
        "contract": "alphaf.endpoint_regression.gpu_outage_pord.v5.outer_evaluation",
        "training_result_sha256": str(training_result["result_sha256"]),
        "selected_state_sha256": str(training_result["selected_state_sha256"]),
        "backbone_id": str(training_result["backbone_id"]),
        "held_out_segment": str(training_result["held_out_segment"]),
        "checkpoint_selection_scope": PORD_CHECKPOINT_SELECTION_SCOPE,
        "checkpoint_fixed_before_outer_labels": True,
        "outer_eval_count": 1,
        "base_outer_view_bundle_sha256": str(
            prediction_receipt["base_outer_view_bundle_sha256"]
        ),
        "outer_view_bundle_sha256": str(
            prediction_receipt["outer_view_bundle_sha256"]
        ),
        "availability_semantics": str(
            prediction_receipt["availability_semantics"]
        ),
        "outer_trial_uid_sha256": _canonical_json_sha256(list(outer_uids)),
        "outer_truth_raw_mm_sha256": array_sha256(truth_raw_mm),
        "outer_prediction_raw_mm_sha256": array_sha256(predictions_raw_mm),
        "target_transform_sha256": str(training_result["target_transform_sha256"]),
        "evaluated_view_names": list(PORD_VIEW_NAMES),
        "same_checkpoint_for_all_four_outages": True,
        "outer_full_rmse_3d_mm": _rmse_3d_mm(
            truth_raw_mm, predictions_raw_mm[0]
        ),
        "outer_outage_rmse_3d_mm": [
            _rmse_3d_mm(truth_raw_mm, predictions_raw_mm[index])
            for index in range(1, 5)
        ],
    }
    receipt["outer_evaluation_sha256"] = _canonical_json_sha256(
        _outer_evaluation_hash_payload(receipt)
    )
    verify_pord_outer_evaluation(
        receipt,
        training_result=training_result,
        outer_trial_uids=outer_uids,
        truth_raw_mm=truth_raw_mm,
        predictions_raw_mm=predictions_raw_mm,
    )
    return {
        "receipt": receipt,
        "truth_raw_mm": truth_raw_mm,
        "predictions_raw_mm": predictions_raw_mm,
    }


def evaluate_pord_checkpoint_once(
    *,
    training_result: Mapping[str, Any],
    X_outer_scaled: Any,
    y_outer_raw_mm: Any,
    outer_segment_ids: Sequence[Any],
    outer_trial_uids: Sequence[Any],
    held_out_segment: Any,
    target_mean_mm: Any,
    target_scale_mm: Any,
    device: str,
) -> dict[str, Any]:
    """Compatibility wrapper for one feature-only prediction and one evaluation."""

    prediction = predict_pord_checkpoint_outer(
        training_result=training_result,
        X_outer_scaled=X_outer_scaled,
        outer_segment_ids=outer_segment_ids,
        outer_trial_uids=outer_trial_uids,
        held_out_segment=held_out_segment,
        target_mean_mm=target_mean_mm,
        target_scale_mm=target_scale_mm,
        device=device,
    )
    return evaluate_pord_outer_predictions_once(
        training_result=training_result,
        outer_prediction=prediction,
        y_outer_raw_mm=y_outer_raw_mm,
    )


def verify_paired_method_controls(
    results: Sequence[Mapping[str, Any]],
    *,
    require_all_methods: bool = True,
    require_complete_epoch_budget: bool = True,
) -> dict[str, Any]:
    """Verify identical data/views/init/update/checkpoint policy across methods."""

    if not results:
        raise PORDV5TrainingError("paired method results are empty")
    for result in results:
        verify_pord_result_hash(result)
    methods = tuple(str(result.get("method_id")) for result in results)
    if len(methods) != len(set(methods)):
        raise PORDV5TrainingError("paired method results contain a duplicate method")
    if require_all_methods and set(methods) != set(PORD_METHOD_IDS):
        raise PORDV5TrainingError("paired controls require all twelve methods")
    if require_all_methods and any(
        str(result.get("backbone_id")) != PORD_PRIMARY_BACKBONE_ID
        for result in results
    ):
        raise PORDV5TrainingError(
            "the twelve-method table requires the masked-MLP primary backbone"
        )
    if require_complete_epoch_budget and any(
        result.get("contract_epoch_budget_complete") is not True
        or int(result.get("epoch_budget", -1)) != PORD_EPOCH_BUDGET
        or int(result.get("optimizer_update_count", -1)) != PORD_EPOCH_BUDGET
        for result in results
    ):
        raise PORDV5TrainingError(
            "paired production controls require exactly 400 optimizer updates"
        )
    fields = (
        "backbone_id",
        "training_bundle_sha256",
        "initial_state_sha256",
        "base_train_view_bundle_sha256",
        "pre_registered_wrong_pair_map_sha256",
        "epoch_budget",
        "optimizer_update_count",
        "held_out_segment",
        "model_seed",
        "target_transform_sha256",
        "checkpoint_selection_scope",
        "checkpoint_criterion",
        "outer_eval_count",
        "outer_labels_accessed",
        "same_checkpoint_for_all_four_outages",
        "checkpoint_evaluated_view_names",
    )
    mismatched = [
        field
        for field in fields
        if len({_canonical_json_sha256(result[field]) for result in results}) != 1
    ]
    if mismatched:
        raise PORDV5TrainingError(f"paired method controls changed: {mismatched}")
    for result in results:
        method = str(result["method_id"])
        expected_availability = (
            "constant_all_available_negative_control"
            if method == "constant_availability_erm"
            else "physical_mask_state"
        )
        expected_gradient = (
            "full_view_gradient_only_outage_selected_checkpoint"
            if method == "full_gradient_only_mask_selected_control"
            else "five_view_mask_weighted_gradient"
        )
        expected_pairing = {
            "group_mean_pord": "group_mean_difference_not_pair_specific",
            "paired_tail": "correct_uid_stop_gradient",
            "paired_interaction": "correct_uid_stop_gradient",
            "paired_pord": "correct_uid_stop_gradient",
            "paired_wrong_pair": "sorted_uid_within_segment_circular_derangement",
            "paired_no_stopgrad": "correct_uid_no_stop_gradient",
        }.get(method, "none")
        if (
            result.get("availability_semantics") != expected_availability
            or result.get("gradient_view_semantics") != expected_gradient
            or result.get("pairing_mode") != expected_pairing
            or result.get("full_reference_detached")
            is not (method != "paired_no_stopgrad")
        ):
            raise PORDV5TrainingError(
                f"pre-registered method semantics changed for {method!r}"
            )
    return {
        "method_ids": sorted(methods),
        "paired_field_names": list(fields),
        "training_bundle_sha256": str(results[0]["training_bundle_sha256"]),
        "initial_state_sha256": str(results[0]["initial_state_sha256"]),
        "verification_sha256": _canonical_json_sha256(
            {field: results[0][field] for field in fields}
        ),
    }


def verify_transfer_method_controls(
    results: Sequence[Mapping[str, Any]],
    *,
    require_complete_epoch_budget: bool = True,
) -> dict[str, Any]:
    """Verify the closed-world two-row chain transfer table exactly."""

    methods = tuple(str(result.get("method_id")) for result in results)
    if len(results) != len(PORD_TRANSFER_METHOD_IDS) or set(methods) != set(
        PORD_TRANSFER_METHOD_IDS
    ):
        raise PORDV5TrainingError(
            "chain transfer requires exactly masked-ERM and paired-PORD rows"
        )
    if any(
        str(result.get("backbone_id")) != PORD_CHAIN_TRANSFER_BACKBONE_ID
        for result in results
    ):
        raise PORDV5TrainingError(
            "transfer rows must use the chain_endpoint_only backbone"
        )
    audit = verify_paired_method_controls(
        results,
        require_all_methods=False,
        require_complete_epoch_budget=require_complete_epoch_budget,
    )
    audit["table_kind"] = "chain_transfer_masked_erm_paired_pord_only"
    audit["verification_sha256"] = _canonical_json_sha256(
        {key: value for key, value in audit.items() if key != "verification_sha256"}
    )
    return audit


def half_up_median_epoch(values: Sequence[Any]) -> int:
    """Median integer epoch with exact .5 ties rounded away from zero."""

    epochs: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Integral
        ):
            raise PORDV5TrainingError("best epochs must be integers")
        epoch = int(value)
        if epoch <= 0 or epoch > PORD_EPOCH_BUDGET:
            raise PORDV5TrainingError("best epoch lies outside [1, 400]")
        epochs.append(epoch)
    if not epochs:
        raise PORDV5TrainingError("best epoch list is empty")
    ordered = sorted(epochs)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median = Fraction(ordered[middle], 1)
    else:
        median = Fraction(ordered[middle - 1] + ordered[middle], 2)
    return int(median.numerator // median.denominator + (median % 1 >= Fraction(1, 2)))


def final_epochs_by_method_seed(
    records: Sequence[Mapping[str, Any]], *, expected_fold_count: int = 4
) -> list[dict[str, Any]]:
    """Choose one half-up median final epoch for each method/paired seed."""

    if isinstance(expected_fold_count, bool) or not isinstance(
        expected_fold_count, numbers.Integral
    ):
        raise PORDV5TrainingError("expected_fold_count must be a positive integer")
    if int(expected_fold_count) <= 0:
        raise PORDV5TrainingError("expected_fold_count must be positive")
    grouped: dict[tuple[str, int], dict[str, int]] = {}
    for record in records:
        try:
            method = str(record["method_id"])
            raw_model_seed = record["model_seed"]
            raw_fold = record["held_out_segment"]
            epoch = record["best_epoch"]
        except (KeyError, TypeError, ValueError) as exc:
            raise PORDV5TrainingError("final-epoch record is malformed") from exc
        if isinstance(raw_model_seed, (bool, np.bool_)) or not isinstance(
            raw_model_seed, numbers.Integral
        ):
            raise PORDV5TrainingError("model_seed must be an integer")
        model_seed = int(raw_model_seed)
        if raw_fold is None or (
            isinstance(raw_fold, (float, np.floating))
            and not math.isfinite(float(raw_fold))
        ):
            raise PORDV5TrainingError("held-out fold is missing or non-finite")
        fold = str(raw_fold)
        if method not in PORD_METHOD_IDS or not fold:
            raise PORDV5TrainingError("final-epoch method/fold is invalid")
        if isinstance(epoch, (bool, np.bool_)) or not isinstance(
            epoch, numbers.Integral
        ):
            raise PORDV5TrainingError("best epochs must be integers")
        key = (method, model_seed)
        if fold in grouped.setdefault(key, {}):
            raise PORDV5TrainingError("duplicate method/seed/fold best epoch")
        grouped[key][fold] = int(epoch)
    if not grouped:
        raise PORDV5TrainingError("final-epoch records are empty")
    output: list[dict[str, Any]] = []
    expected_folds: tuple[str, ...] | None = None
    for (method, model_seed), folds in sorted(grouped.items()):
        if len(folds) != int(expected_fold_count):
            raise PORDV5TrainingError("method/seed does not have every required fold")
        fold_labels = tuple(sorted(folds))
        if expected_folds is None:
            expected_folds = fold_labels
        elif fold_labels != expected_folds:
            raise PORDV5TrainingError("method/seed fold identities do not match")
        epoch = half_up_median_epoch(list(folds.values()))
        row = {
            "method_id": method,
            "model_seed": model_seed,
            "fold_best_epochs": {key: folds[key] for key in sorted(folds)},
            "final_epoch": epoch,
            "rounding": "median_half_up",
        }
        row["selection_sha256"] = _canonical_json_sha256(row)
        output.append(row)
    return output


def full_sensor_diagnostic_score(fold_rmse_3d_mm: Sequence[Any]) -> float:
    """Return ``F=.5*(mean held-out full RMSE + max held-out full RMSE)``."""

    try:
        values = np.asarray(fold_rmse_3d_mm, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PORDV5TrainingError("full-sensor RMSE values must be numeric") from exc
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values < 0):
        raise PORDV5TrainingError(
            "full-sensor scoring requires four finite non-negative fold RMSE values"
        )
    return float(0.5 * (values.mean() + values.max()))


def full_sensor_numeric_gate(
    *,
    candidate_fold_rmse_3d_mm: Sequence[Any],
    baseline_fold_rmse_3d_mm: Sequence[Any],
    baseline_name: str,
    absolute_numeric_tolerance_mm: float = 1.0e-9,
) -> dict[str, Any]:
    """Compare candidate ``F`` to any explicitly named numeric baseline.

    Orchestration uses ``baseline_name='frozen_v3_same_backbone'`` with the
    ``1e-9`` tolerance for the frozen-v3 diagnostic, and may call the same
    helper with ``baseline_name='same_v5_backbone_erm'`` and zero tolerance.
    Neither call is a statistical non-inferiority test.
    """

    candidate = full_sensor_diagnostic_score(candidate_fold_rmse_3d_mm)
    baseline = full_sensor_diagnostic_score(baseline_fold_rmse_3d_mm)
    if not isinstance(baseline_name, str) or not baseline_name.strip():
        raise PORDV5TrainingError("full-sensor gate baseline_name is empty")
    name = baseline_name.strip()
    try:
        tolerance = float(absolute_numeric_tolerance_mm)
    except (TypeError, ValueError) as exc:
        raise PORDV5TrainingError("full-sensor gate tolerance must be numeric") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise PORDV5TrainingError(
            "full-sensor gate tolerance must be finite and non-negative"
        )
    passed = candidate <= baseline + tolerance
    record = {
        "candidate_F": candidate,
        "baseline_name": name,
        "baseline_F": baseline,
        "absolute_numeric_tolerance_mm": tolerance,
        "passed": bool(passed),
        "interpretation": "numeric_only_not_statistical_noninferiority",
    }
    record["gate_sha256"] = _canonical_json_sha256(record)
    return record


__all__ = [
    "FiveViewBundle",
    "GroupRiskTensors",
    "ObjectiveTensors",
    "PairingMap",
    "PORDV5TrainingError",
    "PORD_BACKBONE_SETTINGS",
    "PORD_BACKBONE_IDS",
    "PORD_BACKBONE_SETTINGS_BY_ID",
    "PORD_CHAIN_BACKBONE_ID",
    "PORD_CHAIN_TRANSFER_BACKBONE_ID",
    "PORD_CHAIN_TRANSFER_BACKBONE_SETTINGS",
    "PORD_CHECKPOINT_CRITERION",
    "PORD_CHECKPOINT_SELECTION_SCOPE",
    "PORD_EPOCH_BUDGET",
    "PORD_FULL_GROUP_WEIGHT",
    "PORD_GROUPDRO_ETA",
    "PORD_LEARNING_RATE",
    "PORD_MASK_PROBABILITIES",
    "PORD_METHOD_IDS",
    "PORD_MASKED_MLP_BACKBONE_ID",
    "PORD_MISSING_GROUP_WEIGHT",
    "PORD_MOFE_LAMBDA",
    "PORD_OPTIMIZER_SETTINGS",
    "PORD_PRIMARY_BACKBONE_ID",
    "PORD_PRIMARY_BACKBONE_SETTINGS",
    "PORD_PRIMARY_METHOD_IDS",
    "PORD_TOP_K",
    "PORD_VIEW_NAMES",
    "PORD_VREX_BETA",
    "PORD_TRANSFER_METHOD_IDS",
    "array_sha256",
    "build_constant_availability_five_views",
    "build_exact_five_views",
    "build_exact_pord_backbone",
    "build_exact_chain_transfer_backbone",
    "build_pord_backbone",
    "compute_controlled_objective",
    "compute_empirical_group_risks",
    "compute_row_paired_atoms",
    "direct_group_risk_gradients",
    "direct_row_loss_gradients",
    "earliest_checkpoint_epoch",
    "evaluate_pord_checkpoint_once",
    "evaluate_pord_outer_predictions_once",
    "final_epochs_by_method_seed",
    "fixed_symmetric_positive_topk",
    "fixed_within_segment_circular_derangement",
    "full_sensor_diagnostic_score",
    "full_sensor_numeric_gate",
    "half_up_median_epoch",
    "inverse_standardized_targets",
    "initial_pord_state_sha256",
    "objective_audit_record",
    "outage_checkpoint_score",
    "predict_pord_checkpoint_outer",
    "state_dict_sha256",
    "train_pord_fold_seed",
    "two_way_center",
    "verify_paired_method_controls",
    "verify_pord_outer_evaluation",
    "verify_pord_result_hash",
    "verify_transfer_method_controls",
]
