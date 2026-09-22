"""Exact primary analysis for OpenAxes native-V3 persisted predictions.

The analyzer never refits a model.  It authenticates the experiment task and
result JSON, the referenced ``outer_predictions.npy`` files, and the frozen
prepared table before computing the preregistered seed-ensemble LOSO metrics.
Only the exact 42-context final scope, exact 0--23 deadline-scoped primary
support, and exact 0--11 screen scope are accepted.  The 0--23 scope computes
all eight LOSO folds but is never represented as completion of the full
42-context grid; the 0--11 scope is always labelled direction-only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import openaxes_external_native_v3_experiment as experiment
from .openaxes_external_native_v3_config import MASKS


ANALYSIS_CONTRACT = "openaxes.external.native_sensor_experts.v3.primary_analysis.v1"
FULL_CONTEXTS = tuple(range(42))
DEADLINE_PRIMARY_CONTEXTS = tuple(range(24))
SCREEN_CONTEXTS = tuple(range(12))
FULL_PRIMARY_FOLDS = tuple(range(8))
SCREEN_PRIMARY_FOLDS = tuple(range(4))
MISSING_VIEW_INDICES = tuple(range(1, 15))
FULL_TOLERANCE_M = 1.0e-12
ADJACENT_TOLERANCE = 1.0e-12


class OpenAxesExternalNativeV3AnalysisError(RuntimeError):
    """Raised when persisted inputs are incomplete or violate the fixed grid."""


def _fail(message: str) -> None:
    raise OpenAxesExternalNativeV3AnalysisError(message)


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
        raise OpenAxesExternalNativeV3AnalysisError(
            "analysis report is not finite canonical JSON"
        ) from exc


def _payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["payload_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _file_reference(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"input is not a regular no-symlink file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    try:
        relative = path.relative_to(relative_to).as_posix()
    except ValueError as exc:
        raise OpenAxesExternalNativeV3AnalysisError(
            f"input escapes its authority root: {path}"
        ) from exc
    return {"path": relative, "bytes": size, "sha256": digest.hexdigest()}


def fixed_sum_float64(values: Any, *, name: str = "values") -> float:
    """Adjacent-pair binary-tree sum with odd-tail carry."""

    array = np.asarray(values, dtype="float64").reshape(-1)
    if array.size == 0 or not bool(np.isfinite(array).all()):
        _fail(f"{name} must be nonempty finite float64 data")
    level = [float(item) for item in array]
    while len(level) > 1:
        following: list[float] = []
        pair_limit = len(level) - len(level) % 2
        for index in range(0, pair_limit, 2):
            following.append(float(level[index] + level[index + 1]))
        if pair_limit != len(level):
            following.append(level[-1])
        level = following
    if not math.isfinite(level[0]):
        _fail(f"{name} reduction is non-finite")
    return level[0]


def fixed_mean_float64(values: Any, *, name: str = "values") -> float:
    array = np.asarray(values, dtype="float64").reshape(-1)
    if array.size == 0:
        _fail(f"{name} mean is empty")
    return float(fixed_sum_float64(array, name=name) / int(array.size))


def _fixed_mean_arrays(values: Sequence[np.ndarray], *, name: str) -> np.ndarray:
    if not values:
        _fail(f"{name} array mean is empty")
    arrays = [np.asarray(value, dtype="float64") for value in values]
    shape = arrays[0].shape
    if any(value.shape != shape or not bool(np.isfinite(value).all()) for value in arrays):
        _fail(f"{name} arrays have unequal shape or non-finite values")
    level = [value.copy() for value in arrays]
    while len(level) > 1:
        following: list[np.ndarray] = []
        pair_limit = len(level) - len(level) % 2
        for index in range(0, pair_limit, 2):
            following.append(np.add(level[index], level[index + 1], dtype="float64"))
        if pair_limit != len(level):
            following.append(level[-1])
        level = following
    result = np.divide(level[0], np.float64(len(arrays)), dtype="float64")
    if not bool(np.isfinite(result).all()):
        _fail(f"{name} array mean is non-finite")
    return result


def _row_errors_m(prediction: np.ndarray, truth: np.ndarray, *, name: str) -> np.ndarray:
    predicted = np.asarray(prediction, dtype="float64")
    observed = np.asarray(truth, dtype="float64")
    if (
        predicted.ndim != 2
        or predicted.shape[1:] != (3,)
        or predicted.shape != observed.shape
        or predicted.shape[0] < 1
        or not bool(np.isfinite(predicted).all())
        or not bool(np.isfinite(observed).all())
    ):
        _fail(f"{name} prediction/truth shape changed")
    difference = predicted - observed
    values = []
    for ordinal, row in enumerate(difference):
        squared = fixed_sum_float64(np.multiply(row, row, dtype="float64"), name=f"{name} row {ordinal}")
        if squared < 0.0:
            _fail(f"{name} squared error is negative")
        values.append(float(np.sqrt(squared)))
    return np.asarray(values, dtype="float64")


def fixed_rmse3d_m(prediction: np.ndarray, truth: np.ndarray, *, name: str) -> float:
    predicted = np.asarray(prediction, dtype="float64")
    observed = np.asarray(truth, dtype="float64")
    if (
        predicted.ndim != 2
        or predicted.shape[1:] != (3,)
        or predicted.shape != observed.shape
        or predicted.shape[0] < 1
        or not bool(np.isfinite(predicted).all())
        or not bool(np.isfinite(observed).all())
    ):
        _fail(f"{name} prediction/truth shape changed")
    squared_rows = [
        fixed_sum_float64(
            np.multiply(row, row, dtype="float64"), name=f"{name} row {ordinal}"
        )
        for ordinal, row in enumerate(predicted - observed)
    ]
    return float(
        np.sqrt(fixed_mean_float64(squared_rows, name=f"{name} squared rows"))
    )


def adjacent_mask_edges() -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    cardinalities = tuple(sum(mask) for mask in MASKS)
    for rich_index, rich in enumerate(MASKS):
        for poor_index, poor in enumerate(MASKS):
            if (
                cardinalities[poor_index] == cardinalities[rich_index] - 1
                and all(poor[sensor] <= rich[sensor] for sensor in range(4))
                and sum(rich[sensor] - poor[sensor] for sensor in range(4)) == 1
            ):
                edges.append((rich_index, poor_index))
    if len(edges) != 28:
        _fail("adjacent mask lattice is not the frozen 28-edge order")
    return tuple(edges)


ADJACENT_EDGES = adjacent_mask_edges()


@dataclass(frozen=True)
class PrimaryFoldPredictions:
    """One LOSO fold with predictions in frozen seed and view order."""

    fold_ordinal: int
    fold_id: str
    scenario_id: str
    row_uids: tuple[str, ...]
    truth_m: np.ndarray
    predictions_by_seed: tuple[np.ndarray, ...]


def _metrics_from_cells(cells: np.ndarray, *, name: str) -> dict[str, Any]:
    values = np.asarray(cells, dtype="float64")
    if (
        values.ndim != 2
        or values.shape[1] != len(experiment.VIEW_IDS)
        or values.shape[0] < 1
        or not bool(np.isfinite(values).all())
    ):
        _fail(f"{name} metric cells changed")
    scenario_missing_means = [
        fixed_mean_float64(values[fold, 1:], name=f"{name} scenario {fold} missing")
        for fold in range(values.shape[0])
    ]
    mask_means = [
        fixed_mean_float64(values[:, view], name=f"{name} mask {view}")
        for view in MISSING_VIEW_INDICES
    ]
    amean = fixed_mean_float64(
        values[:, 1:].reshape(-1), name=f"{name} Amean 8x14 cells"
    )
    ws = max(scenario_missing_means)
    wm = max(mask_means)
    j_value = fixed_sum_float64((amean, wm, ws), name=f"{name} J") / 3.0
    full_values = values[:, 0]
    full_mean = fixed_mean_float64(full_values, name=f"{name} full mean")
    full_score = 0.5 * full_mean + 0.5 * float(max(full_values))
    cardinality_means: dict[str, float] = {}
    for cardinality in (1, 2, 3):
        view_indices = [
            index for index, mask in enumerate(MASKS)
            if index != 0 and sum(mask) == cardinality
        ]
        scenario_values = [
            fixed_mean_float64(
                [values[fold, index] for index in view_indices],
                name=f"{name} cardinality {cardinality} scenario {fold}",
            )
            for fold in range(values.shape[0])
        ]
        cardinality_means[str(cardinality)] = fixed_mean_float64(
            scenario_values, name=f"{name} cardinality {cardinality}"
        )
    return {
        "Amean_m": amean,
        "WM_m": wm,
        "WS_m": ws,
        "J_m": j_value,
        "F_m": full_score,
        "full_mean_rmse3d_m": full_mean,
        "scenario_fold_scores_m": scenario_missing_means,
        "mask_mean_rmse3d_m": mask_means,
        "cardinality_mean_rmse3d_m": cardinality_means,
    }


def aggregate_primary_predictions(
    candidate_id: str,
    folds: Sequence[PrimaryFoldPredictions],
    *,
    partial: bool,
    analysis_label: str | None = None,
    final_gate_eligible: bool | None = None,
) -> dict[str, Any]:
    """Compute exact primary metrics from authenticated in-memory fold arrays."""

    expected_folds = SCREEN_PRIMARY_FOLDS if partial else FULL_PRIMARY_FOLDS
    if candidate_id not in experiment.CANDIDATE_IDS:
        _fail("candidate is outside the frozen roster")
    ordered = tuple(sorted(folds, key=lambda item: item.fold_ordinal))
    if tuple(item.fold_ordinal for item in ordered) != expected_folds:
        _fail("primary fold roster is incomplete or out of scope")

    ensemble_cells = np.empty((len(ordered), 15), dtype="float64")
    per_seed_cells = [np.empty((len(ordered), 15), dtype="float64") for _ in experiment.SEEDS]
    ensemble_row_errors: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_position, fold in enumerate(ordered):
        truth = np.asarray(fold.truth_m, dtype="float64")
        predictions = tuple(np.asarray(value, dtype="float64") for value in fold.predictions_by_seed)
        if (
            len(predictions) != len(experiment.SEEDS)
            or truth.ndim != 2
            or truth.shape[1:] != (3,)
            or truth.shape[0] < 1
            or len(fold.row_uids) != truth.shape[0]
            or len(set(fold.row_uids)) != len(fold.row_uids)
            or any(value.shape != (15, truth.shape[0], 3) for value in predictions)
        ):
            _fail(f"fold arrays changed: {fold.fold_id}")
        ensemble = _fixed_mean_arrays(predictions, name=f"{candidate_id} {fold.fold_id} seed ensemble")
        row_errors = np.empty((15, truth.shape[0]), dtype="float64")
        for view_index, view_id in enumerate(experiment.VIEW_IDS):
            row_errors[view_index] = _row_errors_m(
                ensemble[view_index], truth, name=f"{fold.fold_id} {view_id} ensemble"
            )
            ensemble_cells[fold_position, view_index] = fixed_rmse3d_m(
                ensemble[view_index],
                truth,
                name=f"{fold.fold_id} {view_id} ensemble",
            )
            for seed_index, seed in enumerate(experiment.SEEDS):
                per_seed_cells[seed_index][fold_position, view_index] = fixed_rmse3d_m(
                    predictions[seed_index][view_index],
                    truth,
                    name=f"{fold.fold_id} {view_id} seed {seed}",
                )
        ensemble_row_errors.append(row_errors)
        fold_rows.append(
            {
                "fold_ordinal": fold.fold_ordinal,
                "fold_id": fold.fold_id,
                "scenario_id": fold.scenario_id,
                "row_count": truth.shape[0],
                "row_uid_sha256": hashlib.sha256(_canonical(list(fold.row_uids))).hexdigest(),
            }
        )

    metrics = _metrics_from_cells(ensemble_cells, name=f"{candidate_id} ensemble")
    metrics["equal_scenario_mean_rmse3d_m_by_view"] = [
        {
            "view_id": view_id,
            "rmse3d_m": fixed_mean_float64(
                ensemble_cells[:, index], name=f"{candidate_id} view {view_id}"
            ),
        }
        for index, view_id in enumerate(experiment.VIEW_IDS)
    ]
    metrics["per_seed_J_m"] = [
        {
            "seed": seed,
            "J_m": _metrics_from_cells(per_seed_cells[index], name=f"{candidate_id} seed {seed}")["J_m"],
        }
        for index, seed in enumerate(experiment.SEEDS)
    ]

    cr_edges: list[float] = []
    pe_edges: list[float] = []
    for edge_ordinal, (rich, poor) in enumerate(ADJACENT_EDGES):
        cr_scenarios: list[float] = []
        pe_scenarios: list[float] = []
        for fold_position, errors in enumerate(ensemble_row_errors):
            difference = np.subtract(errors[rich], errors[poor], dtype="float64")
            cr_scenarios.append(
                fixed_mean_float64(
                    np.asarray(difference > 0.0, dtype="float64"),
                    name=f"edge {edge_ordinal} scenario {fold_position} CR",
                )
            )
            pe_scenarios.append(
                fixed_mean_float64(
                    np.maximum(difference, np.float64(0.0)),
                    name=f"edge {edge_ordinal} scenario {fold_position} PE",
                )
            )
        cr_edges.append(fixed_mean_float64(cr_scenarios, name=f"edge {edge_ordinal} CR"))
        pe_edges.append(fixed_mean_float64(pe_scenarios, name=f"edge {edge_ordinal} PE"))
    metrics["CR_adjacent"] = fixed_mean_float64(cr_edges, name="CR adjacent edges")
    metrics["PE_adjacent_m"] = fixed_mean_float64(pe_edges, name="PE adjacent edges")
    metrics["adjacent_edge_count"] = len(ADJACENT_EDGES)

    default_label = "screen_partial_direction_only" if partial else "full_preregistered_primary"
    resolved_label = analysis_label or default_label
    resolved_eligibility = (not partial) if final_gate_eligible is None else final_gate_eligible
    if partial and resolved_eligibility:
        _fail("partial screen analysis cannot be final-gate eligible")
    return {
        "candidate_id": candidate_id,
        "analysis_label": resolved_label,
        "final_gate_eligible": resolved_eligibility,
        "primary_fold_count": len(ordered),
        "primary_fold_rows": fold_rows,
        "metrics": metrics,
    }


_PAIR_SPECS = (
    ("static_vs_dense", experiment.CANDIDATE_IDS[1], experiment.CANDIDATE_IDS[0], "common"),
    ("dynamic_vs_static", experiment.CANDIDATE_IDS[2], experiment.CANDIDATE_IDS[1], "common"),
    ("two_stage_vs_one_stage", experiment.CANDIDATE_IDS[3], experiment.CANDIDATE_IDS[2], "common"),
    ("mofe_vs_two_stage_erm", experiment.CANDIDATE_IDS[4], experiment.CANDIDATE_IDS[3], "mofe"),
    ("cagrad_vs_mofe", experiment.CANDIDATE_IDS[5], experiment.CANDIDATE_IDS[4], "cagrad"),
    ("terminal_vs_dense_primary", experiment.CANDIDATE_IDS[5], experiment.CANDIDATE_IDS[0], "primary"),
    ("terminal_vs_static_primary", experiment.CANDIDATE_IDS[5], experiment.CANDIDATE_IDS[1], "primary"),
)


def compare_candidate_summaries(
    challenger: Mapping[str, Any],
    comparator: Mapping[str, Any],
    *,
    pair_id: str,
    gate_family: str,
    partial: bool,
    final_claim_eligible: bool | None = None,
) -> dict[str, Any]:
    left = challenger.get("metrics")
    right = comparator.get("metrics")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        _fail("candidate comparison lacks metric mappings")
    left_folds = list(left.get("scenario_fold_scores_m", []))
    right_folds = list(right.get("scenario_fold_scores_m", []))
    left_seeds = [row.get("J_m") for row in left.get("per_seed_J_m", [])]
    right_seeds = [row.get("J_m") for row in right.get("per_seed_J_m", [])]
    if (
        len(left_folds) != len(right_folds)
        or len(left_seeds) != 3
        or len(right_seeds) != 3
    ):
        _fail("paired fold or seed metric support changed")
    fold_wins = sum(float(a) < float(b) for a, b in zip(left_folds, right_folds))
    seed_wins = sum(float(a) < float(b) for a, b in zip(left_seeds, right_seeds))
    conditions: dict[str, bool] = {
        "J_strictly_lower": float(left["J_m"]) < float(right["J_m"]),
        "WM_strictly_lower": float(left["WM_m"]) < float(right["WM_m"]),
        "full_F_not_higher_beyond_tolerance": float(left["F_m"]) <= float(right["F_m"]) + FULL_TOLERANCE_M,
        "scenario_fold_score_wins_at_least_6_of_8": fold_wins >= 6 if not partial else False,
        "seed_level_J_wins_at_least_2_of_3": seed_wins >= 2,
    }
    if gate_family in {"mofe", "primary"}:
        conditions["CR_adjacent_strictly_lower"] = float(left["CR_adjacent"]) < float(right["CR_adjacent"])
        conditions["PE_adjacent_m_strictly_lower"] = float(left["PE_adjacent_m"]) < float(right["PE_adjacent_m"])
    elif gate_family == "cagrad":
        conditions["CR_adjacent_not_higher_beyond_tolerance"] = (
            float(left["CR_adjacent"]) <= float(right["CR_adjacent"]) + ADJACENT_TOLERANCE
        )
        conditions["PE_adjacent_m_not_higher_beyond_tolerance"] = (
            float(left["PE_adjacent_m"]) <= float(right["PE_adjacent_m"]) + FULL_TOLERANCE_M
        )
    elif gate_family != "common":
        _fail("unknown frozen gate family")
    resolved_claim_eligibility = (not partial) if final_claim_eligible is None else final_claim_eligible
    if partial and resolved_claim_eligibility:
        _fail("partial comparison cannot authorize a final claim")
    gate_passed = None if partial else all(conditions.values())
    return {
        "pair_id": pair_id,
        "gate_family": gate_family,
        "challenger": challenger.get("candidate_id"),
        "comparator": comparator.get("candidate_id"),
        "direction_only": partial,
        "fold_score_strict_wins": fold_wins,
        "fold_score_support": len(left_folds),
        "seed_J_strict_wins": seed_wins,
        "seed_J_support": len(left_seeds),
        "conditions": conditions,
        "gate_passed": gate_passed,
        "final_claim_eligible": resolved_claim_eligibility,
        "claim_authorized": bool(gate_passed) and resolved_claim_eligibility,
    }


def _load_prepared(prepared_root: Path) -> tuple[Any, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise OpenAxesExternalNativeV3AnalysisError("pyarrow is required for V3 analysis") from exc
    table_path = prepared_root / "prepared.parquet"
    reference = _file_reference(table_path, relative_to=prepared_root)
    columns = [*experiment.METADATA_COLUMNS, *experiment.TARGET_COLUMNS]
    try:
        frame = pq.read_table(table_path, columns=columns, use_threads=False).to_pandas()
    except (OSError, ValueError) as exc:
        raise OpenAxesExternalNativeV3AnalysisError("cannot read prepared.parquet") from exc
    if (
        list(frame.columns) != columns
        or len(frame) < 1
        or frame["row_uid"].duplicated().any()
        or any(type(value) is not str or not value for value in frame["row_uid"].tolist())
        or any(type(value) is not str or not value for value in frame["scenario_id"].tolist())
    ):
        _fail("prepared row identity changed")
    truth = frame[list(experiment.TARGET_COLUMNS)].to_numpy(dtype="float64", copy=True)
    if not bool(np.isfinite(truth).all()):
        _fail("prepared targets are non-finite")
    return frame, reference


def _outer_frame(frame: Any, task: experiment.TaskSpec) -> Any:
    outer = frame[frame["scenario_id"].isin(task.test_scenarios)].copy()
    observed_order = tuple(dict.fromkeys(outer["scenario_id"].tolist()))
    if len(outer) < 1 or observed_order != task.test_scenarios:
        _fail(f"outer scenario order/support changed: {task.task_id}")
    return outer


def _load_prediction(
    run_root: Path,
    task: experiment.TaskSpec,
    result: Mapping[str, Any],
    *,
    truth: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    training = result.get("training_result")
    if not isinstance(training, Mapping):
        _fail(f"training result is malformed: {task.task_id}")
    stored = training.get("outer_predictions_reference")
    if not isinstance(stored, Mapping) or set(stored) != {"path", "bytes", "sha256"} or stored.get("path") != "outer_predictions.npy":
        _fail(f"prediction reference changed: {task.task_id}")
    result_dir = run_root / "results" / task.task_id
    prediction_path = result_dir / "outer_predictions.npy"
    observed = _file_reference(prediction_path, relative_to=result_dir)
    if observed != dict(stored):
        _fail(f"prediction bytes changed: {task.task_id}")
    try:
        prediction = np.load(prediction_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise OpenAxesExternalNativeV3AnalysisError(
            f"cannot load prediction: {task.task_id}"
        ) from exc
    rows = len(truth)
    if (
        prediction.dtype != np.dtype("float64")
        or prediction.shape != (15, rows, 3)
        or not bool(np.isfinite(prediction).all())
    ):
        _fail(f"prediction dtype/shape changed: {task.task_id}")
    metrics = training.get("view_metrics")
    if (
        type(metrics) is not list
        or len(metrics) != 15
        or [row.get("view_id") for row in metrics if isinstance(row, Mapping)] != list(experiment.VIEW_IDS)
        or any(not isinstance(row, Mapping) or row.get("rows") != rows for row in metrics)
    ):
        _fail(f"stored view metric support changed: {task.task_id}")
    if training.get("outer_view_metrics") != metrics:
        _fail(f"outer/view metric aliases changed: {task.task_id}")
    for index, row in enumerate(metrics):
        observed_rmse = row.get("rmse3d_m")
        if type(observed_rmse) not in {float, int} or isinstance(observed_rmse, bool):
            _fail(f"stored RMSE is not numeric: {task.task_id}")
        recomputed = fixed_rmse3d_m(
            prediction[index], truth, name=f"{task.task_id} stored {experiment.VIEW_IDS[index]}"
        )
        if not math.isfinite(float(observed_rmse)) or float(observed_rmse) != recomputed:
            _fail(f"stored RMSE does not match persisted prediction: {task.task_id}")
    if any(
        training.get(field) != 1
        for field in (
            "outer_feature_loads",
            "outer_label_reads",
            "outer_evaluator_calls",
            "outer_model_forward_calls",
        )
    ):
        _fail(f"outer accounting changed: {task.task_id}")
    result_path = result_dir / "result.json"
    result_reference = _file_reference(result_path, relative_to=run_root)
    return prediction, observed, result_reference


def _scope(contexts: Sequence[int]) -> tuple[str, tuple[int, ...]]:
    value = tuple(contexts)
    if value == FULL_CONTEXTS:
        return "full_preregistered_primary", FULL_PRIMARY_FOLDS
    if value == DEADLINE_PRIMARY_CONTEXTS:
        return "deadline_scoped_primary_8_not_full_42", FULL_PRIMARY_FOLDS
    if value == SCREEN_CONTEXTS:
        return "screen_partial_direction_only", SCREEN_PRIMARY_FOLDS
    _fail("analysis contexts must be exactly 0-41, deadline primary 0-23, or screen 0-11")


def analyze_run(
    run_dir: Path | str,
    prepared_root: Path | str,
    *,
    candidates: Sequence[str],
    contexts: Sequence[int],
) -> dict[str, Any]:
    run_root = Path(run_dir)
    prepared = Path(prepared_root)
    scope_label, primary_folds = _scope(contexts)
    screen_partial = scope_label == "screen_partial_direction_only"
    full_grid_complete = scope_label == "full_preregistered_primary"
    deadline_primary = scope_label == "deadline_scoped_primary_8_not_full_42"
    candidate_ids = tuple(candidates)
    expected_candidate_order = tuple(
        candidate for candidate in experiment.CANDIDATE_IDS if candidate in candidate_ids
    )
    if (
        not candidate_ids
        or len(set(candidate_ids)) != len(candidate_ids)
        or any(candidate not in experiment.CANDIDATE_IDS for candidate in candidate_ids)
        or candidate_ids != expected_candidate_order
    ):
        _fail("analysis candidate selector changed")
    try:
        tasks = experiment.load_task_grid(run_root)
    except (OSError, ValueError, experiment.OpenAxesExternalNativeV3ExperimentError) as exc:
        raise OpenAxesExternalNativeV3AnalysisError("run task authority is incomplete or invalid") from exc
    frame, prepared_reference = _load_prepared(prepared)
    expected_scenarios = set(tasks[0].train_scenarios) | set(tasks[0].test_scenarios)
    if len(expected_scenarios) != 8 or set(frame["scenario_id"].tolist()) != expected_scenarios:
        _fail("prepared scenario roster is not the frozen eight-scenario support")
    task_index_reference = _file_reference(run_root / "task_index.json", relative_to=run_root)
    task_map = {(task.candidate_id, task.context_ordinal): task for task in tasks}
    summaries: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        fold_inputs: dict[int, dict[str, Any]] = {}
        for context in contexts:
            task = task_map.get((candidate_id, context))
            if task is None:
                _fail(f"selected task is absent: {candidate_id} context {context}")
            try:
                result = experiment.validate_task_result(run_root, task)
            except (OSError, ValueError, experiment.OpenAxesExternalNativeV3ExperimentError) as exc:
                raise OpenAxesExternalNativeV3AnalysisError(
                    f"selected task is incomplete or invalid: {task.task_id}"
                ) from exc
            outer = _outer_frame(frame, task)
            outer_truth = outer[list(experiment.TARGET_COLUMNS)].to_numpy(
                dtype="float64", copy=True
            )
            prediction, prediction_reference, result_reference = _load_prediction(
                run_root, task, result, truth=outer_truth
            )
            source_rows.append(
                {
                    "candidate_id": candidate_id,
                    "context_ordinal": context,
                    "task_id": task.task_id,
                    "result_reference": result_reference,
                    "outer_predictions_reference": {
                        **prediction_reference,
                        "path": f"results/{task.task_id}/{prediction_reference['path']}",
                    },
                }
            )
            if task.fold_ordinal not in primary_folds:
                continue
            if task.fold_family != "scenario_loso" or len(task.test_scenarios) != 1:
                _fail(f"primary task is not one-scenario LOSO: {task.task_id}")
            slot = fold_inputs.setdefault(
                task.fold_ordinal,
                {
                    "fold_id": task.fold_id,
                    "scenario_id": task.test_scenarios[0],
                    "row_uids": tuple(outer["row_uid"].tolist()),
                    "truth": outer_truth,
                    "predictions": {},
                },
            )
            if (
                slot["fold_id"] != task.fold_id
                or slot["scenario_id"] != task.test_scenarios[0]
                or slot["row_uids"] != tuple(outer["row_uid"].tolist())
                or not np.array_equal(
                    slot["truth"],
                    outer[list(experiment.TARGET_COLUMNS)].to_numpy(dtype="float64", copy=True),
                )
                or task.seed_ordinal in slot["predictions"]
            ):
                _fail(f"seed pairing support changed: {task.fold_id}")
            slot["predictions"][task.seed_ordinal] = prediction
        fold_rows: list[PrimaryFoldPredictions] = []
        for fold_ordinal in primary_folds:
            slot = fold_inputs.get(fold_ordinal)
            if slot is None or tuple(sorted(slot["predictions"])) != tuple(range(len(experiment.SEEDS))):
                _fail(f"primary seed ensemble is incomplete: fold {fold_ordinal}")
            fold_rows.append(
                PrimaryFoldPredictions(
                    fold_ordinal=fold_ordinal,
                    fold_id=slot["fold_id"],
                    scenario_id=slot["scenario_id"],
                    row_uids=slot["row_uids"],
                    truth_m=slot["truth"],
                    predictions_by_seed=tuple(
                        slot["predictions"][index] for index in range(len(experiment.SEEDS))
                    ),
                )
            )
        summaries.append(
            aggregate_primary_predictions(
                candidate_id,
                fold_rows,
                partial=screen_partial,
                analysis_label=scope_label,
                final_gate_eligible=full_grid_complete,
            )
        )

    by_candidate = {row["candidate_id"]: row for row in summaries}
    comparisons: list[dict[str, Any]] = []
    deferred: list[str] = []
    for pair_id, challenger, comparator, family in _PAIR_SPECS:
        if challenger in by_candidate and comparator in by_candidate:
            comparisons.append(
                compare_candidate_summaries(
                    by_candidate[challenger],
                    by_candidate[comparator],
                    pair_id=pair_id,
                    gate_family=family,
                    partial=screen_partial,
                    final_claim_eligible=full_grid_complete,
                )
            )
        else:
            deferred.append(pair_id)
    return _payload(
        {
            "contract": ANALYSIS_CONTRACT,
            "schema_version": 1,
            "run_id": experiment.RUN_ID,
            "analysis_label": scope_label,
            "final_gate_eligible": full_grid_complete,
            "full_42_context_grid_complete": full_grid_complete,
            "deadline_scoped_primary_8": deadline_primary,
            "primary_8_fold_metric_support_complete": len(primary_folds) == 8,
            "candidate_order": list(candidate_ids),
            "context_ordinals": list(contexts),
            "primary_fold_ordinals": list(primary_folds),
            "seed_order": list(experiment.SEEDS),
            "view_order": list(experiment.VIEW_IDS),
            "mask_order": [list(mask) for mask in MASKS],
            "input_authority": {
                "task_index_reference": task_index_reference,
                "prepared_parquet_reference": prepared_reference,
                "ordered_task_inputs": source_rows,
            },
            "candidate_summaries": summaries,
            "frozen_pair_comparisons": comparisons,
            "deferred_pair_ids": deferred,
            "candidate_selected_by_analyzer": False,
        }
    )


def write_canonical_report(path: Path | str, report: Mapping[str, Any]) -> None:
    output = Path(path)
    expected = dict(report)
    supplied_hash = expected.pop("payload_sha256", None)
    if supplied_hash != hashlib.sha256(_canonical(expected)).hexdigest():
        _fail("analysis report payload hash changed before persistence")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = _canonical(report) + b"\n"
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise OpenAxesExternalNativeV3AnalysisError(
            f"analysis output already exists: {output}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            output.unlink()
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--run-dir", required=True)
    analyze.add_argument("--prepared-root", required=True)
    analyze.add_argument("--candidates", required=True)
    analyze.add_argument("--contexts", required=True)
    analyze.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "analyze":
        _fail("unknown analysis command")
    report = analyze_run(
        args.run_dir,
        args.prepared_root,
        candidates=experiment.parse_candidate_selector(args.candidates),
        contexts=experiment.parse_context_selector(args.contexts),
    )
    write_canonical_report(args.output, report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OpenAxesExternalNativeV3AnalysisError, experiment.OpenAxesExternalNativeV3ExperimentError) as exc:
        raise SystemExit(f"error: {exc}") from exc


__all__ = [
    "ANALYSIS_CONTRACT",
    "ADJACENT_EDGES",
    "OpenAxesExternalNativeV3AnalysisError",
    "PrimaryFoldPredictions",
    "adjacent_mask_edges",
    "aggregate_primary_predictions",
    "analyze_run",
    "compare_candidate_summaries",
    "fixed_mean_float64",
    "fixed_rmse3d_m",
    "fixed_sum_float64",
    "main",
    "write_canonical_report",
]
