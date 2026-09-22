"""Metrics and paired uncertainty summaries for endpoint-regression experiments.

The primary error in the repository research contract is the Euclidean error
per trial, aggregated as ``sqrt(mean(dx**2 + dy**2 + dz**2))``.  This module
keeps that quantity deliberately separate from the older axis-element RMSE.
All paired comparisons keep predictions aligned to the same real trials.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


AXIS_NAMES = ("x", "y", "z")


def _xyz(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n_rows, 3); received {array.shape}")
    if len(array) == 0:
        raise ValueError(f"{name} must contain at least one row")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _paired_xyz(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    truth = _xyz(y_true, name="y_true")
    prediction = _xyz(y_pred, name="y_pred")
    if truth.shape != prediction.shape:
        raise ValueError(
            f"y_true and y_pred shapes differ: {truth.shape} != {prediction.shape}"
        )
    return truth, prediction


def _weights(sample_weight: Any | None, row_count: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(row_count, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (row_count,):
        raise ValueError(
            f"sample_weight must have shape ({row_count},); received {weights.shape}"
        )
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("sample_weight must be finite and non-negative")
    if float(weights.sum()) <= 0.0:
        raise ValueError("sample_weight must have positive total mass")
    return weights


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights, dtype=np.float64) / weights.sum())


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, dtype=np.float64)
    threshold = quantile * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _weighted_quantile_from_order(
    sorted_values: np.ndarray,
    order: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Weighted quantile when the value order is fixed across resamples."""

    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, dtype=np.float64)
    threshold = quantile * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def three_d_errors_mm(y_true: Any, y_pred: Any) -> np.ndarray:
    """Return one Euclidean XYZ error in millimetres per aligned row."""

    truth, prediction = _paired_xyz(y_true, y_pred)
    return np.linalg.norm(prediction - truth, axis=1)


def rmse_3d_mm(
    y_true: Any, y_pred: Any, sample_weight: Any | None = None
) -> float:
    """Compute ``sqrt(weighted_mean(dx² + dy² + dz²))``."""

    truth, prediction = _paired_xyz(y_true, y_pred)
    weights = _weights(sample_weight, len(truth))
    squared_distance = np.sum(np.square(prediction - truth), axis=1)
    return float(np.sqrt(_weighted_mean(squared_distance, weights)))


def rmse_xyz_mm(
    y_true: Any, y_pred: Any, sample_weight: Any | None = None
) -> float:
    """Compute axis-element RMSE, equal to ``RMSE_3D / sqrt(3)``."""

    truth, prediction = _paired_xyz(y_true, y_pred)
    weights = _weights(sample_weight, len(truth))
    row_axis_mse = np.mean(np.square(prediction - truth), axis=1)
    return float(np.sqrt(_weighted_mean(row_axis_mse, weights)))


def _weighted_r2(
    truth: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> float:
    center = _weighted_mean(truth, weights)
    residual = float(np.sum(weights * np.square(truth - prediction)))
    total = float(np.sum(weights * np.square(truth - center)))
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return float(1.0 - residual / total)


def regression_metrics(
    y_true: Any, y_pred: Any, sample_weight: Any | None = None
) -> dict[str, float | int]:
    """Return pooled 3D and per-axis endpoint-regression metrics.

    Percentiles are ordinary empirical percentiles for unweighted evaluation.
    When weights are supplied, deterministic weighted order statistics are
    used instead.
    """

    truth, prediction = _paired_xyz(y_true, y_pred)
    weights = _weights(sample_weight, len(truth))
    residual = prediction - truth
    errors = np.linalg.norm(residual, axis=1)
    quantile = (
        (lambda q: float(np.quantile(errors, q)))
        if sample_weight is None
        else (lambda q: _weighted_quantile(errors, weights, q))
    )
    result: dict[str, float | int] = {
        "row_count": int(len(truth)),
        "weight_mass": float(weights.sum()),
        "rmse_3d_mm": rmse_3d_mm(truth, prediction, weights),
        "rmse_xyz_mm": rmse_xyz_mm(truth, prediction, weights),
        "mean_3d_error_mm": _weighted_mean(errors, weights),
        "median_3d_error_mm": quantile(0.50),
        "p90_3d_error_mm": quantile(0.90),
        "p95_3d_error_mm": quantile(0.95),
    }
    for axis_index, axis_name in enumerate(AXIS_NAMES):
        axis_residual = residual[:, axis_index]
        result[f"bias_{axis_name}_mm"] = _weighted_mean(axis_residual, weights)
        result[f"mae_{axis_name}_mm"] = _weighted_mean(
            np.abs(axis_residual), weights
        )
        result[f"rmse_{axis_name}_mm"] = float(
            np.sqrt(_weighted_mean(np.square(axis_residual), weights))
        )
        result[f"r2_{axis_name}"] = _weighted_r2(
            truth[:, axis_index], prediction[:, axis_index], weights
        )
    return result


def macro_group_metrics(
    y_true: Any,
    y_pred: Any,
    groups: Sequence[Any],
) -> dict[str, Any]:
    """Give every outer group equal weight, independent of its row count."""

    truth, prediction = _paired_xyz(y_true, y_pred)
    group_array = np.asarray(groups)
    if group_array.shape != (len(truth),):
        raise ValueError("groups must contain one value per prediction row")
    unique_groups = sorted(np.unique(group_array).tolist(), key=str)
    if not unique_groups:
        raise ValueError("at least one group is required")
    per_group: list[dict[str, Any]] = []
    for group in unique_groups:
        mask = group_array == group
        metrics = regression_metrics(truth[mask], prediction[mask])
        per_group.append({"group": group, **metrics})
    numeric_keys = [
        key
        for key, value in per_group[0].items()
        if key not in {"group", "row_count", "weight_mass"}
        and isinstance(value, (float, int))
    ]
    macro = {
        key: float(np.mean([float(row[key]) for row in per_group]))
        for key in numeric_keys
    }
    return {
        "group_count": len(per_group),
        "macro": macro,
        "per_group": per_group,
    }


def cell_balanced_metrics(
    y_true: Any,
    y_pred: Any,
    segment_ids: Sequence[Any],
    target_ids: Sequence[Any],
) -> dict[str, Any]:
    """Summarize errors after assigning equal weight to segment×target cells.

    ``macro_cell_rmse_3d_mm`` is the arithmetic mean of cell RMSE values.  The
    companion ``cell_balanced_rmse_3d_mm`` first gives every cell equal mass
    and then takes the square root, which retains the definition of RMSE.
    Both are reported to make the aggregation convention explicit.
    """

    truth, prediction = _paired_xyz(y_true, y_pred)
    segments = np.asarray(segment_ids)
    targets = np.asarray(target_ids)
    if segments.shape != (len(truth),) or targets.shape != (len(truth),):
        raise ValueError("segment_ids and target_ids must contain one value per row")

    cells: list[dict[str, Any]] = []
    cell_mse: list[float] = []
    keys = sorted(set(zip(segments.tolist(), targets.tolist())), key=lambda x: str(x))
    squared_distance = np.sum(np.square(prediction - truth), axis=1)
    errors = np.sqrt(squared_distance)
    for segment, target in keys:
        mask = (segments == segment) & (targets == target)
        mse = float(np.mean(squared_distance[mask]))
        cell_mse.append(mse)
        cells.append(
            {
                "segment_id": segment,
                "target_id": target,
                "row_count": int(mask.sum()),
                "rmse_3d_mm": float(np.sqrt(mse)),
                "mean_3d_error_mm": float(np.mean(errors[mask])),
                "median_3d_error_mm": float(np.median(errors[mask])),
            }
        )
    return {
        "cell_count": len(cells),
        "cell_balanced_rmse_3d_mm": float(np.sqrt(np.mean(cell_mse))),
        "macro_cell_rmse_3d_mm": float(
            np.mean([row["rmse_3d_mm"] for row in cells])
        ),
        "macro_cell_mean_3d_error_mm": float(
            np.mean([row["mean_3d_error_mm"] for row in cells])
        ),
        "cells": cells,
    }


def paired_group_effects(
    y_true: Any,
    reference_pred: Any,
    candidate_pred: Any,
    groups: Sequence[Any],
) -> list[dict[str, Any]]:
    """Expose candidate-minus-reference effects for every outer group."""

    truth, reference = _paired_xyz(y_true, reference_pred)
    _, candidate = _paired_xyz(y_true, candidate_pred)
    group_array = np.asarray(groups)
    if group_array.shape != (len(truth),):
        raise ValueError("groups must contain one value per prediction row")
    effects: list[dict[str, Any]] = []
    for group in sorted(np.unique(group_array).tolist(), key=str):
        mask = group_array == group
        reference_metrics = regression_metrics(truth[mask], reference[mask])
        candidate_metrics = regression_metrics(truth[mask], candidate[mask])
        effects.append(
            {
                "group": group,
                "row_count": int(mask.sum()),
                "reference_rmse_3d_mm": reference_metrics["rmse_3d_mm"],
                "candidate_rmse_3d_mm": candidate_metrics["rmse_3d_mm"],
                "delta_rmse_3d_mm": float(
                    candidate_metrics["rmse_3d_mm"]
                    - reference_metrics["rmse_3d_mm"]
                ),
                "reference_p95_3d_error_mm": reference_metrics["p95_3d_error_mm"],
                "candidate_p95_3d_error_mm": candidate_metrics["p95_3d_error_mm"],
            }
        )
    return effects


def paired_multiway_bootstrap(
    y_true: Any,
    reference_pred: Any,
    candidate_pred: Any,
    segment_ids: Sequence[Any],
    target_ids: Sequence[Any],
    *,
    n_resamples: int = 10_000,
    seed: int = 20_260_818,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Deterministic crossed-factor (segment×target) paired bootstrap.

    Each replicate independently resamples the observed segment levels and
    target levels.  A trial receives the product of its two factor-level
    multiplicities, the standard pigeonhole/multiway bootstrap weighting.
    The reported delta is always candidate minus reference, so negative RMSE
    values favour the candidate.  No p-value is produced.
    """

    truth, reference = _paired_xyz(y_true, reference_pred)
    _, candidate = _paired_xyz(y_true, candidate_pred)
    segments = np.asarray(segment_ids)
    targets = np.asarray(target_ids)
    if segments.shape != (len(truth),) or targets.shape != (len(truth),):
        raise ValueError("segment_ids and target_ids must contain one value per row")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    _, segment_inverse = np.unique(segments, return_inverse=True)
    _, target_inverse = np.unique(targets, return_inverse=True)
    segment_count = int(segment_inverse.max()) + 1
    target_count = int(target_inverse.max()) + 1
    reference_sq = np.sum(np.square(reference - truth), axis=1)
    candidate_sq = np.sum(np.square(candidate - truth), axis=1)
    reference_error = np.sqrt(reference_sq)
    candidate_error = np.sqrt(candidate_sq)
    reference_order = np.argsort(reference_error, kind="mergesort")
    candidate_order = np.argsort(candidate_error, kind="mergesort")
    reference_error_sorted = reference_error[reference_order]
    candidate_error_sorted = candidate_error[candidate_order]

    rng = np.random.default_rng(int(seed))
    rmse_deltas = np.empty(n_resamples, dtype=np.float64)
    p95_deltas = np.empty(n_resamples, dtype=np.float64)
    accepted = 0
    rejected_empty = 0
    maximum_attempts = max(100, n_resamples * 100)
    attempts = 0
    while accepted < n_resamples:
        attempts += 1
        if attempts > maximum_attempts:
            raise RuntimeError("too many empty crossed-factor bootstrap replicates")
        segment_draw = rng.integers(0, segment_count, size=segment_count)
        target_draw = rng.integers(0, target_count, size=target_count)
        segment_weight = np.bincount(segment_draw, minlength=segment_count)
        target_weight = np.bincount(target_draw, minlength=target_count)
        weights = (
            segment_weight[segment_inverse] * target_weight[target_inverse]
        ).astype(np.float64)
        if float(weights.sum()) == 0.0:
            rejected_empty += 1
            continue
        reference_rmse = np.sqrt(_weighted_mean(reference_sq, weights))
        candidate_rmse = np.sqrt(_weighted_mean(candidate_sq, weights))
        rmse_deltas[accepted] = candidate_rmse - reference_rmse
        p95_deltas[accepted] = _weighted_quantile_from_order(
            candidate_error_sorted, candidate_order, weights, 0.95
        ) - _weighted_quantile_from_order(
            reference_error_sorted, reference_order, weights, 0.95
        )
        accepted += 1

    alpha = (1.0 - confidence) / 2.0
    reference_metrics = regression_metrics(truth, reference)
    candidate_metrics = regression_metrics(truth, candidate)
    observed_rmse_delta = float(
        candidate_metrics["rmse_3d_mm"] - reference_metrics["rmse_3d_mm"]
    )
    observed_p95_delta = float(
        candidate_metrics["p95_3d_error_mm"]
        - reference_metrics["p95_3d_error_mm"]
    )
    return {
        "method": "crossed_segment_target_multiway_bootstrap",
        "delta_direction": "candidate_minus_reference",
        "seed": int(seed),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
        "rejected_empty_replicates": int(rejected_empty),
        "rmse_3d_mm": {
            "observed_delta": observed_rmse_delta,
            "ci_low": float(np.quantile(rmse_deltas, alpha)),
            "ci_high": float(np.quantile(rmse_deltas, 1.0 - alpha)),
            "bootstrap_mean_delta": float(np.mean(rmse_deltas)),
        },
        "p95_3d_error_mm": {
            "observed_delta": observed_p95_delta,
            "ci_low": float(np.quantile(p95_deltas, alpha)),
            "ci_high": float(np.quantile(p95_deltas, 1.0 - alpha)),
            "bootstrap_mean_delta": float(np.mean(p95_deltas)),
        },
    }
