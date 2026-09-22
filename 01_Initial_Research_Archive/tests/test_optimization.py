from __future__ import annotations

import pytest

from research_host.optimization import (
    CONSTRAINT_PENALTY,
    grid_candidates,
    load_objective_config,
    score_candidate,
    select_best_candidate,
)


def test_grid_candidates_are_deterministic() -> None:
    candidates = grid_candidates({"alpha": [0.1, 0.2], "mode": ["raw", "kalman"]})

    assert [candidate.candidate_id for candidate in candidates] == [
        "cand_0001",
        "cand_0002",
        "cand_0003",
        "cand_0004",
    ]
    assert candidates[2].params == {"alpha": 0.2, "mode": "raw"}


def test_score_candidate_penalizes_constraint_violation() -> None:
    score, violations = score_candidate(
        {"position_rmse_mm": 3.0, "dropped_sample_ratio": 0.2},
        weights={"position_rmse_mm": 1.0},
        constraint_max={"dropped_sample_ratio": 0.01},
    )

    assert score >= CONSTRAINT_PENALTY
    assert violations == ["dropped_sample_ratio=0.2 > 0.01"]


def test_select_best_candidate_uses_validation_split_only() -> None:
    rows = [
        {
            "candidate_id": "cand_b",
            "split_id": "validation",
            "position_rmse_mm": "4.0",
            "dropped_sample_ratio": "0.0",
            "param_damping": "0.02",
        },
        {
            "candidate_id": "cand_a",
            "split_id": "validation",
            "position_rmse_mm": "3.0",
            "dropped_sample_ratio": "0.0",
            "param_damping": "0.04",
        },
    ]

    best = select_best_candidate(
        rows,
        weights={"position_rmse_mm": 1.0},
        constraint_max={"dropped_sample_ratio": 0.01},
    )

    assert best.candidate_id == "cand_a"
    assert best.params == {"damping": "0.04"}


def test_select_best_candidate_rejects_test_split() -> None:
    rows = [
        {
            "candidate_id": "cand_a",
            "split_id": "test",
            "position_rmse_mm": "1.0",
        }
    ]

    with pytest.raises(ValueError, match="validation"):
        select_best_candidate(rows)


def test_load_objective_config_reads_allowed_split(tmp_path) -> None:
    path = tmp_path / "objective.json"
    path.write_text(
        """
{
  "allowed_split": "pilot_validation",
  "weights": {"position_rmse_mm": 2.0},
  "constraint_max": {"dropped_sample_ratio": 0.05}
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    objective = load_objective_config(path)

    assert objective.allowed_split == "pilot_validation"
    assert objective.weights == {"position_rmse_mm": 2.0}
    assert objective.constraint_max == {"dropped_sample_ratio": 0.05}
