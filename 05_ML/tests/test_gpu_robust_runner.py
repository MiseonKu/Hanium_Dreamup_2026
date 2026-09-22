from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from imu_vive_learning.gpu_robust_experiment import (
    robust_candidate_config_id,
    robust_final_tasks,
    robust_inner_tasks,
    robust_selection_and_outer_tasks,
)
from imu_vive_learning.gpu_robust_experiment_config import (
    HISTORICAL_V1_ROBUST_J_MM,
    robust_candidate_grid,
)


WORKSPACE = Path(__file__).parents[1]
CONFIG_PATH = WORKSPACE / "configs" / "endpoint_regression_gpu_robust_v2.yaml"


def _config() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _default_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Give every candidate complete, deterministic validation evidence."""

    return {
        str(candidate["candidate_id"]): {
            "rmse": 30.0 + priority,
            "p95": 40.0 + priority,
            "seed_offsets": (-0.25, 0.0, 0.25),
        }
        for priority, candidate in enumerate(robust_candidate_grid(config))
    }


def _fake_inner_results(
    config: dict[str, Any], specs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build only the result fields consumed by pure robust selection."""

    results: list[dict[str, Any]] = []
    for task in robust_inner_tasks(config):
        candidate_id = str(task["candidate_id"])
        spec = specs[candidate_id]
        inner_fold = int(task["inner_fold"])
        rmse = float(spec.get("rmse_by_segment", {}).get(inner_fold, spec["rmse"]))
        p95 = float(spec.get("p95_by_segment", {}).get(inner_fold, spec["p95"]))
        offsets = tuple(float(value) for value in spec["seed_offsets"])
        assert len(offsets) == 3
        results.append(
            {
                "task": deepcopy(task),
                "metrics": {
                    "rmse_3d_mm": rmse,
                    "p95_3d_error_mm": p95,
                },
                "seed_summaries": [
                    {
                        "seed_index": seed_index,
                        "evaluation_rmse_3d_mm": rmse + offsets[seed_index],
                        "selected_epoch": 101 + 10 * seed_index,
                    }
                    for seed_index in range(3)
                ],
            }
        )
    return results


def test_robust_inner_and_smoke_task_matrix_is_exact() -> None:
    config = _config()
    candidates = robust_candidate_grid(config)
    candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}

    full = robust_inner_tasks(config)
    assert len(full) == 120
    assert len({str(task["task_id"]) for task in full}) == 120
    assert Counter(str(task["candidate_id"]) for task in full) == {
        candidate_id: 20 for candidate_id in candidate_ids
    }
    assert {
        (int(task["outer_fold"]), int(task["inner_fold"])) for task in full
    } == {
        (outer_fold, inner_fold)
        for outer_fold in (36, 37, 38, 39, 40)
        for inner_fold in (36, 37, 38, 39, 40)
        if inner_fold != outer_fold
    }
    assert all(task["phase"] == "inner" for task in full)
    assert all(task["scheme"] == "segment_loso" for task in full)
    assert all(task["model"] == "torch_mlp" for task in full)

    smoke = robust_inner_tasks(config, smoke=True)
    assert len(smoke) == 6
    assert {str(task["candidate_id"]) for task in smoke} == candidate_ids
    assert {(task["outer_fold"], task["inner_fold"]) for task in smoke} == {
        (36, 37)
    }
    assert all(task["phase"] == "smoke" for task in smoke)
    assert all(task["epoch_budget_override"] == 50 for task in smoke)
    assert all(task["smoke_contract_eligible"] is False for task in smoke)


def test_robust_j_selection_outer_and_final_tasks_preserve_candidate_identity() -> None:
    config = _config()
    candidates = robust_candidate_grid(config)
    selected_candidate = candidates[2]
    selected_id = str(selected_candidate["candidate_id"])
    specs = _default_specs(config)
    specs[selected_id] = {
        "rmse": 14.0,
        "rmse_by_segment": {36: 10.0, 37: 12.0, 38: 14.0, 39: 16.0, 40: 18.0},
        "p95": 22.0,
        "seed_offsets": (-0.5, 0.0, 0.5),
    }
    inner_results = _fake_inner_results(config, specs)

    selection, outer_tasks = robust_selection_and_outer_tasks(config, inner_results)
    trace = next(
        row for row in selection["trace"] if row["candidate_id"] == selected_id
    )

    # Equal context count gives mean([10, 12, 14, 16, 18]) == 14;
    # robust J is 0.5 * (mean 14 + worst-segment mean 18) == 16.
    assert trace["context_count"] == 20
    assert trace["mean_rmse_3d_mm"] == pytest.approx(14.0)
    assert trace["worst_validation_segment"] == {
        "segment_id": 40,
        "context_count": 4,
        "mean_rmse_3d_mm": 18.0,
    }
    assert trace["robust_j_mm"] == pytest.approx(16.0)
    assert selection["selected_candidate_id"] == selected_id
    assert selection["selected_candidate"] == selected_candidate
    assert selection["selected_robust_j_mm"] == pytest.approx(16.0)
    assert selection["outer_diagnostic_used_for_selection"] is False
    assert selection["promotion_eligible"] is True
    assert selection["promotion_action"] == "promote_robust_v2"
    assert selection["final_refit_epochs_per_seed"] == {
        "0": 101,
        "1": 111,
        "2": 121,
    }

    expected_candidate_config_id = robust_candidate_config_id(
        selected_candidate,
        {"hidden_layer_sizes": [64, 64], "weight_decay": 0.001},
    )
    assert len(outer_tasks) == 5
    assert [int(task["outer_fold"]) for task in outer_tasks] == [36, 37, 38, 39, 40]
    assert all(task["phase"] == "outer" for task in outer_tasks)
    assert all(task["inner_fold"] == -1 for task in outer_tasks)
    assert all(task["candidate_id"] == selected_id for task in outer_tasks)
    assert all(task["candidate"] == selected_candidate for task in outer_tasks)
    assert all(task["training_recipe"] == selected_candidate for task in outer_tasks)
    assert all(
        task["candidate_config_id"] == expected_candidate_config_id
        for task in outer_tasks
    )
    assert all(
        task["epochs_per_seed"] == {"0": 101, "1": 111, "2": 121}
        for task in outer_tasks
    )

    final_tasks = robust_final_tasks(config, selection)
    assert len(final_tasks) == 3
    assert [task["seed_indices"] for task in final_tasks] == [[0], [1], [2]]
    assert [task["epochs_per_seed"] for task in final_tasks] == [
        {"0": 101},
        {"1": 111},
        {"2": 121},
    ]
    assert all(task["phase"] == "final" for task in final_tasks)
    assert all((task["outer_fold"], task["inner_fold"]) == (-1, -1) for task in final_tasks)
    assert all(task["candidate_id"] == selected_id for task in final_tasks)
    assert all(task["candidate"] == selected_candidate for task in final_tasks)
    assert all(task["training_recipe"] == selected_candidate for task in final_tasks)
    assert all(
        task["candidate_config_id"] == expected_candidate_config_id
        for task in final_tasks
    )
    assert len({str(task["task_id"]) for task in final_tasks}) == 3


@pytest.mark.parametrize(
    ("tie_level", "expected_index"),
    [
        ("p95", 1),
        ("seed_dispersion", 1),
        ("listed_order", 0),
    ],
)
def test_robust_selection_tie_break_order_is_exact(
    tie_level: str, expected_index: int
) -> None:
    config = _config()
    candidates = robust_candidate_grid(config)
    first_id = str(candidates[0]["candidate_id"])
    second_id = str(candidates[1]["candidate_id"])
    specs = _default_specs(config)

    specs[first_id] = {
        "rmse": 10.0,
        "p95": 11.0,
        "seed_offsets": (-0.5, 0.0, 0.5),
    }
    specs[second_id] = deepcopy(specs[first_id])
    if tie_level == "p95":
        specs[first_id]["p95"] = 12.0
    elif tie_level == "seed_dispersion":
        specs[first_id]["seed_offsets"] = (-1.0, 0.0, 1.0)

    selection, _ = robust_selection_and_outer_tasks(
        config, _fake_inner_results(config, specs)
    )
    assert selection["selected_candidate_id"] == candidates[expected_index]["candidate_id"]


def test_robust_promotion_is_strict_and_fallback_retains_v1() -> None:
    config = _config()
    first_id = str(robust_candidate_grid(config)[0]["candidate_id"])

    fallback_specs = _default_specs(config)
    fallback_specs[first_id] = {
        "rmse": HISTORICAL_V1_ROBUST_J_MM + 0.01,
        "p95": 20.0,
        "seed_offsets": (-0.1, 0.0, 0.1),
    }
    fallback, _ = robust_selection_and_outer_tasks(
        config, _fake_inner_results(config, fallback_specs)
    )
    assert fallback["selected_robust_j_mm"] > HISTORICAL_V1_ROBUST_J_MM
    assert fallback["promotion_eligible"] is False
    assert fallback["promotion_action"] == "retain_v1"

    promotion_specs = _default_specs(config)
    promotion_specs[first_id] = {
        "rmse": HISTORICAL_V1_ROBUST_J_MM - 0.01,
        "p95": 20.0,
        "seed_offsets": (-0.1, 0.0, 0.1),
    }
    promotion, _ = robust_selection_and_outer_tasks(
        config, _fake_inner_results(config, promotion_specs)
    )
    assert promotion["selected_robust_j_mm"] < HISTORICAL_V1_ROBUST_J_MM
    assert promotion["promotion_eligible"] is True
    assert promotion["promotion_action"] == "promote_robust_v2"
