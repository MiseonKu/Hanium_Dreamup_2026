from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import pandas as pd
import yaml

from imu_vive_learning.cli import build_parser
from imu_vive_learning.gpu_experiment import (
    _WorkerData,
    _all_real_arrays,
    _final_tasks,
    _inner_tasks,
    _round_median,
    _selection_and_outer_tasks,
    _training_direction_warnings,
    gpu_configuration_id,
)
from imu_vive_learning.gpu_experiment_config import (
    PINNED_GPU_CONFIG_SHA256,
    gpu_config_sha256,
    gpu_model_grid,
    validate_gpu_config,
)
from imu_vive_learning.features import FUSION_COLUMNS, LABEL_COLUMNS


WORKSPACE = Path(__file__).parents[1]
CONFIG_PATH = WORKSPACE / "configs" / "endpoint_regression_gpu_v1.yaml"


def _config() -> dict:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_gpu_contract_hash_task_and_claim_boundaries_are_frozen() -> None:
    config = _config()
    validate_gpu_config(config)
    assert gpu_config_sha256(config) == PINNED_GPU_CONFIG_SHA256
    assert config["contract"]["pilot_observed_before_freeze"] is True
    assert config["contract"]["independent_test_available"] is False
    assert config["task"]["input_view"] == "fusion_v1"
    assert config["task"]["expected_feature_count"] == len(FUSION_COLUMNS) == 43
    assert tuple(config["task"]["labels"]) == LABEL_COLUMNS
    assert config["task"]["recipe"] == "real_only"
    assert config["selection"]["final_model_family"] == "torch_mlp"
    assert config["selection"]["outer_scores_not_used_for_final_config"] is True
    assert all(
        config["claim_boundaries"][key]
        for key in (
            "not_robot_base_or_tcp_calibrated",
            "not_dynamic_state_estimation",
            "not_shapefk",
            "not_control_authority",
        )
    )


def test_gpu_contract_rejects_any_material_drift() -> None:
    changed = deepcopy(_config())
    changed["models"]["torch_mlp"]["epoch_budget"] = 401
    with pytest.raises(ValueError, match="budget|pinned"):
        validate_gpu_config(changed)


def test_gpu_grids_and_task_counts_are_exact() -> None:
    config = _config()
    torch_grid = gpu_model_grid(config, "torch_mlp")
    xgb_grid = gpu_model_grid(config, "xgboost_cuda")
    assert len(torch_grid) == 4
    assert torch_grid[0] == {
        "hidden_layer_sizes": [64, 64],
        "weight_decay": 0.001,
    }
    assert len(xgb_grid) == 8
    assert xgb_grid[0] == {
        "max_depth": 4,
        "min_child_weight": 5.0,
        "reg_lambda": 10.0,
    }
    tasks = _inner_tasks(config)
    assert len(tasks) == 480
    assert {task["phase"] for task in tasks} == {"inner"}
    assert {task["scheme"] for task in tasks} == {"segment_loso", "target_group"}
    assert all(task["inner_fold"] != task["outer_fold"] for task in tasks)
    assert len({task["task_id"] for task in tasks}) == len(tasks)

    smoke = _inner_tasks(config, smoke=True)
    assert len(smoke) == 12
    assert all(task["smoke_contract_eligible"] is False for task in smoke)
    assert all(task["epoch_budget_override"] == 20 for task in smoke)


def _fake_inner_results(config: dict) -> list[dict]:
    priorities = {
        model: {
            gpu_configuration_id(model, params): index
            for index, params in enumerate(gpu_model_grid(config, model))
        }
        for model in ("torch_mlp", "xgboost_cuda")
    }
    results: list[dict] = []
    for task in _inner_tasks(config):
        priority = priorities[task["model"]][task["config_id"]]
        score = 10.0 + priority + 0.001 * int(task["inner_fold"])
        seed_summaries = (
            [
                {
                    "seed_index": seed_index,
                    "selected_epoch": 100 + seed_index * 10 + int(task["inner_fold"]) % 4,
                }
                for seed_index in range(3)
            ]
            if task["model"] == "torch_mlp"
            else [{"seed_index": 0, "selected_epoch": 500}]
        )
        results.append(
            {
                "task": task,
                "metrics": {"rmse_3d_mm": score},
                "seed_summaries": seed_summaries,
            }
        )
    return results


def test_outer_tasks_exist_only_after_inner_selection_and_final_family_is_fixed() -> None:
    config = _config()
    selection, outer_tasks = _selection_and_outer_tasks(
        config, _fake_inner_results(config)
    )
    assert selection["outer_test_used"] is False
    assert len(selection["outer_selections"]) == 20
    assert len(outer_tasks) == 20
    assert {task["phase"] for task in outer_tasks} == {"outer"}
    assert {task["inner_fold"] for task in outer_tasks} == {-1}
    assert all(
        item["selected_config_id"]
        == gpu_configuration_id(item["model"], gpu_model_grid(config, item["model"])[0])
        for item in selection["outer_selections"]
    )

    final_tasks = _final_tasks(config, selection)
    assert len(final_tasks) == 3
    assert {task["model"] for task in final_tasks} == {"torch_mlp"}
    assert [task["seed_indices"] for task in final_tasks] == [[0], [1], [2]]
    assert all(task["outer_fold"] == -1 for task in final_tasks)


def test_refit_epoch_rounding_and_gpu_cli_shapes_are_stable() -> None:
    assert _round_median([100, 101, 102, 103]) == 102
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-gpu-experiment",
            "--release",
            "release.zip",
            "--config",
            "config.yaml",
            "--output",
            "run",
            "--gpus",
            "0",
            "1",
            "2",
            "3",
        ]
    )
    assert args.gpus == [0, 1, 2, 3]


def test_all_real_refit_accepts_trial_uid_as_index_and_column() -> None:
    rows = []
    for index, trial_uid in enumerate(("a", "b", "c")):
        row = {name: float(index + offset) for offset, name in enumerate(FUSION_COLUMNS)}
        row.update(
            {
                "trial_uid": trial_uid,
                LABEL_COLUMNS[0]: float(index),
                LABEL_COLUMNS[1]: float(index + 10),
                LABEL_COLUMNS[2]: float(index + 20),
            }
        )
        rows.append(row)
    eligible = pd.DataFrame(rows).set_index("trial_uid", drop=False)
    data = _WorkerData(  # type: ignore[arg-type]
        release=None,
        assignments=pd.DataFrame(),
        eligible=eligible,
        feature_names=tuple(FUSION_COLUMNS),
    )
    arrays = _all_real_arrays(data)
    assert arrays["X_train"].shape == (3, 43)
    assert arrays["y_train"].shape == (3, 3)
    assert arrays["preprocessor"].metadata()["leakage_audit"]["fit_row_count"] == 3


def test_training_direction_warnings_apply_frozen_thresholds(tmp_path: Path) -> None:
    history_path = tmp_path / "inner" / "history" / "task.parquet"
    history_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "seed_index": 0,
                "epoch": 400,
                "validation_rmse_3d_mm": 12.0,
                "best_validation_rmse_3d_mm": 10.0,
                "best_epoch_so_far": 300,
            },
            {
                "seed_index": 1,
                "epoch": 400,
                "validation_rmse_3d_mm": 10.5,
                "best_validation_rmse_3d_mm": 10.0,
                "best_epoch_so_far": 390,
            },
        ]
    ).to_parquet(history_path, index=False)
    result = {
        "phase": "inner",
        "history_file": {"path": "inner/history/task.parquet"},
    }
    warnings = _training_direction_warnings(
        result, run_dir=tmp_path, config=_config()
    )
    assert [warning["kind"] for warning in warnings] == [
        "validation_regression",
        "stale_validation_best",
    ]
    assert all(warning["seed_index"] == 0 for warning in warnings)
