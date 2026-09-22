from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any


DEFAULT_OBJECTIVE_WEIGHTS = {
    "position_rmse_mm": 1.0,
    "shape_rmse_mm": 0.5,
    "settling_time_s": 1.0,
    "transient_rms_mm": 0.2,
    "residual_vibration_rms_mm": 0.25,
    "dropped_sample_ratio": 100.0,
}

DEFAULT_CONSTRAINT_MAX = {
    "dropped_sample_ratio": 0.01,
}

CONSTRAINT_PENALTY = 1_000_000.0


@dataclass(frozen=True, slots=True)
class ParameterCandidate:
    candidate_id: str
    params: dict[str, float | int | str | bool]


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    candidate_id: str
    objective_score: float
    metrics: dict[str, float]
    params: dict[str, Any]
    constraint_violations: list[str]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "objective_score": self.objective_score,
            "metrics": self.metrics,
            "params": self.params,
            "constraint_violations": self.constraint_violations,
        }


def grid_candidates(
    param_grid: Mapping[str, Iterable[float | int | str | bool]],
    *,
    prefix: str = "cand",
) -> list[ParameterCandidate]:
    keys = list(param_grid)
    values = [list(param_grid[key]) for key in keys]
    if not keys or any(not column for column in values):
        raise ValueError("param_grid must contain at least one non-empty parameter")

    candidates: list[ParameterCandidate] = []
    for index, combination in enumerate(product(*values), start=1):
        candidates.append(
            ParameterCandidate(
                candidate_id=f"{prefix}_{index:04d}",
                params=dict(zip(keys, combination)),
            )
        )
    return candidates


def score_candidate(
    metrics: Mapping[str, object],
    *,
    weights: Mapping[str, float] | None = None,
    constraint_max: Mapping[str, float] | None = None,
) -> tuple[float, list[str]]:
    active_weights = dict(DEFAULT_OBJECTIVE_WEIGHTS if weights is None else weights)
    active_constraints = dict(
        DEFAULT_CONSTRAINT_MAX if constraint_max is None else constraint_max
    )
    parsed_metrics = _parse_metric_map(metrics)
    score = 0.0
    for metric_name, weight in active_weights.items():
        score += weight * parsed_metrics.get(metric_name, 0.0)

    violations = constraint_violations(
        parsed_metrics,
        constraint_max=active_constraints,
    )
    score += CONSTRAINT_PENALTY * len(violations)
    return score, violations


def constraint_violations(
    metrics: Mapping[str, float],
    *,
    constraint_max: Mapping[str, float] | None = None,
) -> list[str]:
    active_constraints = dict(
        DEFAULT_CONSTRAINT_MAX if constraint_max is None else constraint_max
    )
    violations: list[str] = []
    for metric_name, max_value in active_constraints.items():
        observed = metrics.get(metric_name)
        if observed is None:
            violations.append(f"missing constraint metric: {metric_name}")
        elif observed > max_value:
            violations.append(f"{metric_name}={observed:.6g} > {max_value:.6g}")
    return violations


def select_best_candidate(
    rows: Iterable[Mapping[str, object]],
    *,
    weights: Mapping[str, float] | None = None,
    constraint_max: Mapping[str, float] | None = None,
    allowed_split: str = "validation",
) -> OptimizationResult:
    results = [
        evaluate_candidate_row(
            row,
            weights=weights,
            constraint_max=constraint_max,
            allowed_split=allowed_split,
        )
        for row in rows
    ]
    if not results:
        raise ValueError("at least one candidate metric row is required")
    return min(results, key=lambda result: (result.objective_score, result.candidate_id))


def evaluate_candidate_row(
    row: Mapping[str, object],
    *,
    weights: Mapping[str, float] | None = None,
    constraint_max: Mapping[str, float] | None = None,
    allowed_split: str = "validation",
) -> OptimizationResult:
    split_id = str(row.get("split_id", allowed_split))
    guard_against_split_leakage(split_id, allowed_split=allowed_split)
    candidate_id = str(row.get("candidate_id", "")).strip()
    if not candidate_id:
        raise ValueError("candidate metric row is missing candidate_id")

    metrics = _parse_metric_map(row)
    params = _extract_params(row)
    score, violations = score_candidate(
        metrics,
        weights=weights,
        constraint_max=constraint_max,
    )
    return OptimizationResult(
        candidate_id=candidate_id,
        objective_score=score,
        metrics=metrics,
        params=params,
        constraint_violations=violations,
    )


def guard_against_split_leakage(
    split_id: str,
    *,
    allowed_split: str = "validation",
) -> None:
    if split_id != allowed_split:
        raise ValueError(
            f"offline optimization must use split_id={allowed_split!r}, got {split_id!r}"
        )


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    weights: dict[str, float]
    constraint_max: dict[str, float]
    allowed_split: str = "validation"


def load_objective_config(path: str | Path) -> ObjectiveConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = {
        str(key): float(value)
        for key, value in payload.get("weights", DEFAULT_OBJECTIVE_WEIGHTS).items()
    }
    constraints = {
        str(key): float(value)
        for key, value in payload.get("constraint_max", DEFAULT_CONSTRAINT_MAX).items()
    }
    allowed_split = str(payload.get("allowed_split", "validation"))
    return ObjectiveConfig(
        weights=weights,
        constraint_max=constraints,
        allowed_split=allowed_split,
    )


def _parse_metric_map(row: Mapping[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in row.items():
        key_str = str(key)
        if key_str.startswith("param_"):
            continue
        if key_str in {"candidate_id", "split_id", "notes"}:
            continue
        parsed = _try_float(value)
        if parsed is not None:
            metrics[key_str] = parsed
    return metrics


def _extract_params(row: Mapping[str, object]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in row.items():
        key_str = str(key)
        if key_str.startswith("param_"):
            params[key_str.removeprefix("param_")] = value
    return params


def _try_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
