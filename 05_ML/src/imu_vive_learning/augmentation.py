"""Leakage-safe, deterministic augmentation for trial-level regression data.

The functions in this module deliberately accept an already selected *real
training* frame.  They never inspect a validation/test frame, and they return
only interpolated numeric values plus explicit provenance.  Callers therefore
cannot accidentally inherit a real timestamp, target id, quality flag, or
sample count from an anchor row.

``local_interp_v2`` chooses one of the five nearest admissible neighbours in
the same collection group.  ``cmixup_offline_v1`` implements the label-local
pairing component of C-Mixup for deterministic, offline materialisation.  It
is described as C-Mixup-inspired rather than as the paper's complete training
algorithm.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import math
from typing import Any

import numpy as np
import pandas as pd


LOCAL_METHOD = "local_interp_v2"
CMIXUP_METHOD = "cmixup_offline_v1"
_PROVENANCE_COLUMNS = (
    "synthetic_id",
    "is_synthetic",
    "augmentation_method",
    "split_scheme",
    "outer_fold",
    "inner_fold",
    "split_context",
    "anchor_trial_uid",
    "anchor_replica_index",
    "replacement_for_missing_anchor",
    "parent_a_trial_uid",
    "parent_b_trial_uid",
    "mix_lambda",
    "planned_parent_distance_mm",
    "label_parent_distance_mm",
    "joint_parent_distance_deg",
    "cmixup_sigma_mm",
    "derived_seed",
    "source_hash",
    "config_hash",
    "sample_weight",
)


class AugmentationError(ValueError):
    """Raised when a requested fold cannot be augmented without relaxation."""


def local_interp_v2(
    train_real: pd.DataFrame,
    *,
    id_column: str = "trial_uid",
    group_columns: Sequence[str] = ("segment_id", "set_id", "repeat_index"),
    planned_columns: Sequence[str],
    label_columns: Sequence[str],
    linear_columns: Sequence[str] = (),
    euler_columns: Sequence[str] = (),
    euler_trig_columns: Mapping[str, tuple[str, str]] | None = None,
    split_scheme: str,
    fold: str | int,
    inner_fold: int = -1,
    base_seed: int,
    source_hash: str,
    config_hash: str,
    planned_max_distance_mm: float = 100.0,
    neighbor_count: int = 5,
    lambdas: Sequence[float] = (0.25, 0.5, 0.75),
    sample_weight: float = 1.0,
    synthetic_flag_column: str = "is_synthetic",
) -> pd.DataFrame:
    """Create exactly one local interpolation per real training row in total.

    Parent B is selected, deterministically per anchor, from at most
    ``neighbor_count`` nearest rows that share ``group_columns`` and are no
    farther than ``planned_max_distance_mm`` in planned XYZ.  If an anchor has
    no candidate, an admissible anchor is resampled and marked in provenance;
    no grouping or distance constraint is relaxed.
    """

    spec = _prepare_spec(
        train_real,
        id_column=id_column,
        planned_columns=planned_columns,
        label_columns=label_columns,
        linear_columns=linear_columns,
        euler_columns=euler_columns,
        euler_trig_columns=euler_trig_columns,
        required_extra=group_columns,
        synthetic_flag_column=synthetic_flag_column,
    )
    _validate_context(
        split_scheme=split_scheme,
        inner_fold=inner_fold,
        source_hash=source_hash,
        config_hash=config_hash,
        planned_max_distance_mm=planned_max_distance_mm,
        sample_weight=sample_weight,
    )
    if not isinstance(neighbor_count, int) or neighbor_count < 1:
        raise AugmentationError("neighbor_count must be a positive integer")
    lambda_values = np.asarray(tuple(lambdas), dtype=np.float64)
    if (
        lambda_values.ndim != 1
        or len(lambda_values) == 0
        or not np.isfinite(lambda_values).all()
        or np.any(lambda_values <= 0.0)
        or np.any(lambda_values >= 1.0)
    ):
        raise AugmentationError("lambdas must contain finite values strictly between 0 and 1")

    frame = spec["frame"]
    ids = spec["ids"]
    planned = spec["planned"]
    labels = spec["labels"]
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row_index, key in enumerate(
        frame.loc[:, list(group_columns)].itertuples(index=False, name=None)
    ):
        groups[tuple(key)].append(row_index)

    candidate_map: dict[int, list[tuple[int, float]]] = {}
    for anchor_index in spec["anchor_order"]:
        group_key = tuple(frame.iloc[anchor_index][list(group_columns)].tolist())
        pool = np.asarray(groups[group_key], dtype=np.int64)
        pool = pool[pool != anchor_index]
        if len(pool):
            distances = np.linalg.norm(planned[pool] - planned[anchor_index], axis=1)
            admissible = distances <= planned_max_distance_mm + 1e-12
            pool = pool[admissible]
            distances = distances[admissible]
        else:
            distances = np.empty(0, dtype=np.float64)
        if len(pool) == 0:
            continue

        # Distance first and uid second gives a stable nearest-neighbour set
        # even if the caller changes DataFrame row order.
        candidate_map[int(anchor_index)] = sorted(
            zip(pool.tolist(), distances.tolist(), strict=True),
            key=lambda item: (item[1], str(ids[item[0]])),
        )[:neighbor_count]

    if not candidate_map:
        raise AugmentationError(
            f"{LOCAL_METHOD}: no admissible real training pairs exist "
            f"(same {tuple(group_columns)!r}, planned distance <= "
            f"{planned_max_distance_mm:g} mm); constraints were not relaxed"
        )
    schedule = _one_to_one_schedule(
        spec["anchor_order"],
        tuple(candidate_map),
        base_seed=base_seed,
        method=LOCAL_METHOD,
        split_scheme=split_scheme,
        fold=fold,
        inner_fold=inner_fold,
    )
    rows: list[dict[str, Any]] = []
    for anchor_index, replica_index, is_replacement in schedule:
        ranked = candidate_map[anchor_index]
        rng_seed = derived_seed(
            base_seed,
            LOCAL_METHOD,
            split_scheme,
            fold,
            inner_fold,
            str(ids[anchor_index]),
            f"replica={replica_index}",
        )
        rng = np.random.default_rng(rng_seed)
        selected_index, planned_distance = ranked[int(rng.integers(len(ranked)))]
        mix_lambda = float(lambda_values[int(rng.integers(len(lambda_values)))])
        label_distance = float(
            np.linalg.norm(labels[selected_index] - labels[anchor_index])
        )
        row = _interpolate_row(
            spec,
            anchor_index=anchor_index,
            partner_index=selected_index,
            mix_lambda=mix_lambda,
        )
        row.update(
            _provenance(
                method=LOCAL_METHOD,
                split_scheme=split_scheme,
                fold=fold,
                inner_fold=inner_fold,
                anchor_id=ids[anchor_index],
                anchor_replica_index=replica_index,
                replacement_for_missing_anchor=is_replacement,
                partner_id=ids[selected_index],
                mix_lambda=mix_lambda,
                planned_distance=planned_distance,
                label_distance=label_distance,
                joint_distance=math.nan,
                sigma=math.nan,
                derived_seed=rng_seed,
                source_hash=source_hash,
                config_hash=config_hash,
                sample_weight=sample_weight,
            )
        )
        rows.append(row)

    result = _finish(rows, spec)
    result.attrs["missing_anchor_count"] = len(spec["frame"]) - len(candidate_map)
    result.attrs["replacement_count"] = sum(item[2] for item in schedule)
    return result


def cmixup_offline_v1(
    train_real: pd.DataFrame,
    *,
    id_column: str = "trial_uid",
    planned_columns: Sequence[str],
    label_columns: Sequence[str],
    joint_rad_columns: Sequence[str],
    linear_columns: Sequence[str] = (),
    euler_columns: Sequence[str] = (),
    euler_trig_columns: Mapping[str, tuple[str, str]] | None = None,
    split_scheme: str,
    fold: str | int,
    inner_fold: int = -1,
    base_seed: int,
    source_hash: str,
    config_hash: str,
    planned_max_distance_mm: float = 100.0,
    joint_max_distance_deg: float = 20.0,
    alpha: float = 2.0,
    sigma_epsilon_mm: float = 1e-9,
    sample_weight: float = 1.0,
    synthetic_flag_column: str = "is_synthetic",
) -> pd.DataFrame:
    """Materialise a 1:1 label-local C-Mixup-inspired training artifact.

    Candidate parents are selected exclusively from ``train_real``.  The
    fold-local bandwidth is the median, over anchors, of the nearest raw-Vive
    label distance among candidates satisfying both physical constraints.
    Parent probability is proportional to ``exp(-d_y**2 / (2*sigma**2))`` and
    lambda is sampled from ``Beta(alpha, alpha)`` using an anchor-derived seed.
    Missing-anchor coverage is filled only from admissible anchors and is
    marked explicitly in provenance.
    """

    spec = _prepare_spec(
        train_real,
        id_column=id_column,
        planned_columns=planned_columns,
        label_columns=label_columns,
        linear_columns=linear_columns,
        euler_columns=euler_columns,
        euler_trig_columns=euler_trig_columns,
        required_extra=joint_rad_columns,
        synthetic_flag_column=synthetic_flag_column,
    )
    _validate_context(
        split_scheme=split_scheme,
        inner_fold=inner_fold,
        source_hash=source_hash,
        config_hash=config_hash,
        planned_max_distance_mm=planned_max_distance_mm,
        sample_weight=sample_weight,
    )
    if len(tuple(joint_rad_columns)) == 0:
        raise AugmentationError("joint_rad_columns must not be empty")
    if not math.isfinite(joint_max_distance_deg) or joint_max_distance_deg < 0:
        raise AugmentationError("joint_max_distance_deg must be finite and non-negative")
    if not math.isfinite(alpha) or alpha <= 0:
        raise AugmentationError("alpha must be finite and positive")
    if not math.isfinite(sigma_epsilon_mm) or sigma_epsilon_mm <= 0:
        raise AugmentationError("sigma_epsilon_mm must be finite and positive")

    frame = spec["frame"]
    ids = spec["ids"]
    planned = spec["planned"]
    labels = spec["labels"]
    joints = _numeric_matrix(frame, joint_rad_columns, "joint_rad_columns")
    spatial_index = _PlannedGrid(planned, planned_max_distance_mm)

    nearest_label_distances: list[float] = []
    candidate_map: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for anchor_index in spec["anchor_order"]:
        candidate_data = _cmixup_candidates(
            anchor_index,
            planned=planned,
            labels=labels,
            joints=joints,
            ids=ids,
            grid=spatial_index,
            planned_max_distance_mm=planned_max_distance_mm,
            joint_max_distance_deg=joint_max_distance_deg,
        )
        candidates, _, label_distances, _ = candidate_data
        if len(candidates) == 0:
            continue
        candidate_map[int(anchor_index)] = candidate_data
        nearest_label_distances.append(float(np.min(label_distances)))

    if not candidate_map:
        raise AugmentationError(
            f"{CMIXUP_METHOD}: no admissible real training pairs exist "
            f"(planned distance <= {planned_max_distance_mm:g} mm, max joint delta <= "
            f"{joint_max_distance_deg:g} deg); constraints were not relaxed"
        )

    sigma = max(float(np.median(nearest_label_distances)), sigma_epsilon_mm)
    schedule = _one_to_one_schedule(
        spec["anchor_order"],
        tuple(candidate_map),
        base_seed=base_seed,
        method=CMIXUP_METHOD,
        split_scheme=split_scheme,
        fold=fold,
        inner_fold=inner_fold,
    )
    rows: list[dict[str, Any]] = []
    for anchor_index, replica_index, is_replacement in schedule:
        candidates, planned_distances, label_distances, joint_distances = candidate_map[
            anchor_index
        ]
        rng_seed = derived_seed(
            base_seed,
            CMIXUP_METHOD,
            split_scheme,
            fold,
            inner_fold,
            str(ids[anchor_index]),
            f"replica={replica_index}",
        )
        rng = np.random.default_rng(rng_seed)
        log_weights = -np.square(label_distances / sigma) / 2.0
        log_weights -= float(np.max(log_weights))
        weights = np.exp(log_weights)
        probabilities = weights / float(np.sum(weights))
        selected_offset = int(rng.choice(len(candidates), p=probabilities))
        selected_index = int(candidates[selected_offset])
        mix_lambda = float(rng.beta(alpha, alpha))
        row = _interpolate_row(
            spec,
            anchor_index=anchor_index,
            partner_index=selected_index,
            mix_lambda=mix_lambda,
        )
        row.update(
            _provenance(
                method=CMIXUP_METHOD,
                split_scheme=split_scheme,
                fold=fold,
                inner_fold=inner_fold,
                anchor_id=ids[anchor_index],
                anchor_replica_index=replica_index,
                replacement_for_missing_anchor=is_replacement,
                partner_id=ids[selected_index],
                mix_lambda=mix_lambda,
                planned_distance=float(planned_distances[selected_offset]),
                label_distance=float(label_distances[selected_offset]),
                joint_distance=float(joint_distances[selected_offset]),
                sigma=sigma,
                derived_seed=rng_seed,
                source_hash=source_hash,
                config_hash=config_hash,
                sample_weight=sample_weight,
            )
        )
        rows.append(row)

    result = _finish(rows, spec)
    result.attrs["cmixup_sigma_mm"] = sigma
    result.attrs["missing_anchor_count"] = len(spec["frame"]) - len(candidate_map)
    result.attrs["replacement_count"] = sum(item[2] for item in schedule)
    return result


def _prepare_spec(
    train_real: pd.DataFrame,
    *,
    id_column: str,
    planned_columns: Sequence[str],
    label_columns: Sequence[str],
    linear_columns: Sequence[str],
    euler_columns: Sequence[str],
    euler_trig_columns: Mapping[str, tuple[str, str]] | None,
    required_extra: Sequence[str],
    synthetic_flag_column: str,
) -> dict[str, Any]:
    if not isinstance(train_real, pd.DataFrame):
        raise TypeError("train_real must be a pandas DataFrame")
    if train_real.empty:
        raise AugmentationError("train_real must not be empty")
    planned_columns = _unique_names(planned_columns, "planned_columns")
    label_columns = _unique_names(label_columns, "label_columns")
    linear_columns = _unique_names(linear_columns, "linear_columns")
    euler_columns = _unique_names(euler_columns, "euler_columns")
    required_extra = _unique_names(required_extra, "required_extra")
    if not planned_columns or not label_columns:
        raise AugmentationError("planned_columns and label_columns must not be empty")
    requested_linear = set(linear_columns) | set(planned_columns) | set(label_columns)
    if set(euler_columns) & requested_linear:
        overlap = sorted(set(euler_columns) & requested_linear)
        raise AugmentationError(f"Euler columns cannot also be linear: {overlap}")

    required = [
        id_column,
        *planned_columns,
        *label_columns,
        *linear_columns,
        *euler_columns,
        *required_extra,
    ]
    missing = sorted(set(required) - set(train_real.columns))
    if missing:
        raise AugmentationError(f"train_real is missing required columns: {missing}")
    if train_real[id_column].isna().any() or train_real[id_column].duplicated().any():
        raise AugmentationError(f"{id_column!r} must be non-null and unique")
    if train_real.loc[:, list(required_extra)].isna().any(axis=None):
        raise AugmentationError("group/joint constraint columns must be non-null")
    if synthetic_flag_column in train_real.columns:
        synthetic = train_real[synthetic_flag_column]
        if synthetic.isna().any() or synthetic.astype(bool).any():
            raise AugmentationError("train_real must contain real rows only")

    # Constraints and labels are always materialised linearly.  De-duplication
    # preserves the caller's order and avoids conflicting output assignments.
    # Canonicalise row order once. Grid buckets then have deterministic member
    # order, avoiding an expensive Python UID sort for every C-Mixup anchor.
    frame = train_real.copy()
    frame["__augmentation_uid_sort_key"] = frame[id_column].astype(str)
    frame.sort_values("__augmentation_uid_sort_key", kind="mergesort", inplace=True)
    frame.drop(columns="__augmentation_uid_sort_key", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    all_linear = _deduplicate((*linear_columns, *planned_columns, *label_columns))
    numeric_columns = _deduplicate((*all_linear, *euler_columns))
    _numeric_matrix(frame, numeric_columns, "interpolation columns")
    linear_matrix = _numeric_matrix(frame, all_linear, "linear interpolation columns")
    euler_matrix = _numeric_matrix(frame, euler_columns, "Euler interpolation columns")
    planned = _numeric_matrix(frame, planned_columns, "planned_columns")
    labels = _numeric_matrix(frame, label_columns, "label_columns")

    if euler_trig_columns is None:
        trig_map = {name: (f"{name}_sin", f"{name}_cos") for name in euler_columns}
    else:
        unknown = sorted(set(euler_trig_columns) - set(euler_columns))
        missing_angles = sorted(set(euler_columns) - set(euler_trig_columns))
        if unknown or missing_angles:
            raise AugmentationError(
                "euler_trig_columns must map every Euler column exactly; "
                f"unknown={unknown}, missing={missing_angles}"
            )
        trig_map = dict(euler_trig_columns)
    trig_names: list[str] = []
    for angle, pair in trig_map.items():
        if len(pair) != 2 or not all(isinstance(name, str) and name for name in pair):
            raise AugmentationError(f"invalid sin/cos output names for {angle!r}: {pair!r}")
        trig_names.extend(pair)
    if len(trig_names) != len(set(trig_names)):
        raise AugmentationError("Euler sin/cos output names must be unique")
    output_numeric = [*all_linear, *euler_columns, *trig_names]
    if len(output_numeric) != len(set(output_numeric)):
        collisions = sorted(
            name for name in set(output_numeric) if output_numeric.count(name) > 1
        )
        raise AugmentationError(
            f"interpolation and Euler sin/cos output names must be disjoint: {collisions}"
        )
    collision = sorted(set(output_numeric) & set(_PROVENANCE_COLUMNS))
    if collision:
        raise AugmentationError(f"interpolation columns collide with provenance: {collision}")

    ids = frame[id_column].to_numpy(copy=True)
    anchor_order = np.arange(len(frame), dtype=np.int64)
    return {
        "frame": frame,
        "ids": ids,
        "planned": planned,
        "labels": labels,
        "linear_columns": all_linear,
        "linear_matrix": linear_matrix,
        "euler_columns": euler_columns,
        "euler_matrix": euler_matrix,
        "trig_map": trig_map,
        "output_numeric": output_numeric,
        "anchor_order": anchor_order,
    }


def _interpolate_row(
    spec: Mapping[str, Any],
    *,
    anchor_index: int,
    partner_index: int,
    mix_lambda: float,
) -> dict[str, float]:
    # Lambda weights parent A (the anchor), matching conventional mixup.
    linear = (
        mix_lambda * spec["linear_matrix"][anchor_index]
        + (1.0 - mix_lambda) * spec["linear_matrix"][partner_index]
    )
    result = {
        name: float(value)
        for name, value in zip(spec["linear_columns"], linear, strict=True)
    }
    for offset, name in enumerate(spec["euler_columns"]):
        a = float(spec["euler_matrix"][anchor_index, offset])
        b = float(spec["euler_matrix"][partner_index, offset])
        angle = circular_interpolate_rad(a, b, mix_lambda)
        result[name] = angle
        sin_name, cos_name = spec["trig_map"][name]
        result[sin_name] = math.sin(angle)
        result[cos_name] = math.cos(angle)
    return result


def circular_interpolate_rad(parent_a: float, parent_b: float, mix_lambda: float) -> float:
    """Interpolate radians along the shortest arc and normalise to [-pi, pi)."""

    if not all(math.isfinite(value) for value in (parent_a, parent_b, mix_lambda)):
        raise AugmentationError("circular interpolation inputs must be finite")
    if not 0.0 <= mix_lambda <= 1.0:
        raise AugmentationError("mix_lambda must be in [0, 1]")
    delta = (parent_b - parent_a + math.pi) % (2.0 * math.pi) - math.pi
    angle = parent_a + (1.0 - mix_lambda) * delta
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _cmixup_candidates(
    anchor_index: int,
    *,
    planned: np.ndarray,
    labels: np.ndarray,
    joints: np.ndarray,
    ids: np.ndarray,
    grid: "_PlannedGrid",
    planned_max_distance_mm: float,
    joint_max_distance_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pool = grid.query(anchor_index)
    pool = pool[pool != anchor_index]
    if len(pool) == 0:
        empty = np.empty(0, dtype=np.float64)
        return pool, empty, empty, empty
    planned_distances = np.linalg.norm(planned[pool] - planned[anchor_index], axis=1)
    joint_distances = np.rad2deg(
        np.max(np.abs(joints[pool] - joints[anchor_index]), axis=1)
    )
    mask = (planned_distances <= planned_max_distance_mm + 1e-12) & (
        joint_distances <= joint_max_distance_deg + 1e-12
    )
    pool = pool[mask]
    planned_distances = planned_distances[mask]
    joint_distances = joint_distances[mask]
    if len(pool) == 0:
        empty = np.empty(0, dtype=np.float64)
        return pool, empty, empty, empty
    label_distances = np.linalg.norm(labels[pool] - labels[anchor_index], axis=1)
    return pool, planned_distances, label_distances, joint_distances


class _PlannedGrid:
    """Small dependency-free spatial index for the planned-distance gate."""

    def __init__(self, points: np.ndarray, radius: float) -> None:
        self._points = points
        self._cell_size = radius if radius > 0 else 1.0
        self._radius_cells = 1 if radius > 0 else 0
        self._buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index, point in enumerate(points):
            self._buckets[self._key(point)].append(index)

    def _key(self, point: np.ndarray) -> tuple[int, ...]:
        return tuple(np.floor(point / self._cell_size).astype(np.int64).tolist())

    def query(self, index: int) -> np.ndarray:
        base = self._key(self._points[index])
        offsets = np.ndindex(*([2 * self._radius_cells + 1] * len(base)))
        candidates: list[int] = []
        for raw_offset in offsets:
            offset = tuple(value - self._radius_cells for value in raw_offset)
            key = tuple(base[axis] + offset[axis] for axis in range(len(base)))
            candidates.extend(self._buckets.get(key, ()))
        return np.asarray(candidates, dtype=np.int64)


def _provenance(
    *,
    method: str,
    split_scheme: str,
    fold: str | int,
    inner_fold: int,
    anchor_id: Any,
    anchor_replica_index: int,
    replacement_for_missing_anchor: bool,
    partner_id: Any,
    mix_lambda: float,
    planned_distance: float,
    label_distance: float,
    joint_distance: float,
    sigma: float,
    derived_seed: int,
    source_hash: str,
    config_hash: str,
    sample_weight: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "method": method,
            "split_scheme": split_scheme,
            "fold": str(fold),
            "inner_fold": int(inner_fold),
            "anchor": str(anchor_id),
            "anchor_replica_index": int(anchor_replica_index),
            "partner": str(partner_id),
            "lambda": format(mix_lambda, ".17g"),
            "seed": derived_seed,
            "source_hash": source_hash,
            "config_hash": config_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    synthetic_id = f"syn:{method}:{sha256(payload.encode('utf-8')).hexdigest()}"
    return {
        "synthetic_id": synthetic_id,
        "is_synthetic": True,
        "augmentation_method": method,
        "split_scheme": split_scheme,
        "outer_fold": fold,
        "inner_fold": int(inner_fold),
        "split_context": (
            f"{split_scheme}:outer={fold}:inner="
            f"{'refit' if int(inner_fold) == -1 else int(inner_fold)}"
        ),
        "anchor_trial_uid": anchor_id,
        "anchor_replica_index": int(anchor_replica_index),
        "replacement_for_missing_anchor": bool(replacement_for_missing_anchor),
        "parent_a_trial_uid": anchor_id,
        "parent_b_trial_uid": partner_id,
        "mix_lambda": mix_lambda,
        "planned_parent_distance_mm": planned_distance,
        "label_parent_distance_mm": label_distance,
        "joint_parent_distance_deg": joint_distance,
        "cmixup_sigma_mm": sigma,
        "derived_seed": derived_seed,
        "source_hash": source_hash,
        "config_hash": config_hash,
        "sample_weight": sample_weight,
    }


def _finish(rows: list[dict[str, Any]], spec: Mapping[str, Any]) -> pd.DataFrame:
    result = pd.DataFrame.from_records(rows)
    columns = [*spec["output_numeric"], *_PROVENANCE_COLUMNS]
    result = result.loc[:, columns]
    numeric = result.loc[:, spec["output_numeric"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise AssertionError("augmentation produced a non-finite interpolated value")
    if len(result) != len(spec["frame"]):
        raise AssertionError("augmentation must produce exactly one row per anchor")
    if result["synthetic_id"].duplicated().any():
        raise AssertionError("augmentation produced duplicate synthetic ids")
    return result.reset_index(drop=True)


def _one_to_one_schedule(
    anchor_order: np.ndarray,
    admissible_anchors: Sequence[int],
    *,
    base_seed: int,
    method: str,
    split_scheme: str,
    fold: str | int,
    inner_fold: int,
) -> list[tuple[int, int, bool]]:
    """Keep admissible anchors once, then deterministically fill missing coverage.

    The physical/group constraints are never relaxed.  If an original anchor
    has no legal partner, another admissible anchor is sampled with replacement
    so the fold remains exactly 1:1.  Provenance marks every such replacement.
    """

    admissible = set(int(value) for value in admissible_anchors)
    primary = [int(value) for value in anchor_order if int(value) in admissible]
    if not primary:
        return []
    schedule: list[tuple[int, int, bool]] = [(value, 0, False) for value in primary]
    missing = len(anchor_order) - len(primary)
    if missing <= 0:
        return schedule
    seed = derived_seed(
        base_seed,
        method,
        split_scheme,
        fold,
        inner_fold,
        "coverage_replacement",
    )
    rng = np.random.default_rng(seed)
    replica_counts = {value: 0 for value in primary}
    for _ in range(missing):
        anchor = primary[int(rng.integers(len(primary)))]
        replica_counts[anchor] += 1
        schedule.append((anchor, replica_counts[anchor], True))
    return schedule


def _numeric_matrix(
    frame: pd.DataFrame, columns: Sequence[str], description: str
) -> np.ndarray:
    if len(tuple(columns)) == 0:
        return np.empty((len(frame), 0), dtype=np.float64)
    try:
        matrix = frame.loc[:, list(columns)].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise AugmentationError(f"{description} must be numeric") from exc
    if not np.isfinite(matrix).all():
        raise AugmentationError(f"{description} must contain only finite values")
    return matrix


def _unique_names(columns: Sequence[str], description: str) -> tuple[str, ...]:
    names = tuple(columns)
    if any(not isinstance(name, str) or not name for name in names):
        raise AugmentationError(f"{description} must contain non-empty strings")
    if len(names) != len(set(names)):
        raise AugmentationError(f"{description} must not contain duplicates")
    return names


def _deduplicate(columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(columns))


def derived_seed(base_seed: int, *parts: Any) -> int:
    payload = json.dumps(
        [int(base_seed), *(str(part) for part in parts)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _validate_context(
    *,
    split_scheme: str,
    inner_fold: int,
    source_hash: str,
    config_hash: str,
    planned_max_distance_mm: float,
    sample_weight: float,
) -> None:
    if not split_scheme:
        raise AugmentationError("split_scheme must not be empty")
    if not isinstance(inner_fold, int) or inner_fold < -1:
        raise AugmentationError("inner_fold must be -1 or a non-negative integer")
    if not source_hash or not config_hash:
        raise AugmentationError("source_hash and config_hash are required provenance")
    if not math.isfinite(planned_max_distance_mm) or planned_max_distance_mm < 0:
        raise AugmentationError("planned_max_distance_mm must be finite and non-negative")
    if not math.isfinite(sample_weight) or sample_weight <= 0:
        raise AugmentationError("sample_weight must be finite and positive")


def _no_candidate_error(*, method: str, anchor_id: Any, constraints: str) -> AugmentationError:
    return AugmentationError(
        f"{method}: anchor {anchor_id!r} has no admissible real training parent "
        f"({constraints}); constraints were not relaxed"
    )


__all__ = [
    "AugmentationError",
    "CMIXUP_METHOD",
    "LOCAL_METHOD",
    "circular_interpolate_rad",
    "cmixup_offline_v1",
    "derived_seed",
    "local_interp_v2",
]
