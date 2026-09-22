from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from imu_vive_learning.config import canonical_json_bytes
from imu_vive_learning.cli import build_parser, main
from imu_vive_learning.features import FUSION_COLUMNS, LABEL_COLUMNS
from imu_vive_learning.gpu_experiment_config import PINNED_GPU_CONFIG_SHA256
from imu_vive_learning.gpu_robust_experiment_config import (
    FUSION_V1_ALLOWLIST_SHA256,
    HISTORICAL_V1_ROBUST_J_MM,
    PINNED_ROBUST_GPU_CONFIG_SHA256,
    load_robust_gpu_config,
    resolve_robust_gpu_release_path,
    robust_candidate_grid,
    robust_gpu_config_sha256,
    validate_robust_gpu_config,
)


WORKSPACE = Path(__file__).parents[1]
CONFIG_PATH = WORKSPACE / "configs" / "endpoint_regression_gpu_robust_v2.yaml"


def _config() -> dict:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_robust_v2_hash_lineage_inputs_and_disclosure_are_frozen() -> None:
    config = _config()
    validate_robust_gpu_config(config)

    assert robust_gpu_config_sha256(config) == PINNED_ROBUST_GPU_CONFIG_SHA256
    assert config["contract"] == {
        "name": "alphaf.endpoint_regression.gpu_experiment.robust_v2",
        "parent_contract": "alphaf.endpoint_regression.gpu_experiment.v1",
        "base_seed": 20260818,
        "research_status": "post_v1_robustness_development",
        "label_frame": "vive_world",
        "label_unit": "mm",
        "post_v1_outer_observed": True,
        "independent_test_available": False,
    }
    assert config["lineage"]["base_v1_config_sha256"] == PINNED_GPU_CONFIG_SHA256
    assert (
        config["lineage"]["preparation_release_sha256"]
        == config["release"]["archive_sha256"]
        == "94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930"
    )
    assert config["task"]["input_view"] == "fusion_v1"
    assert config["task"]["expected_feature_count"] == len(FUSION_COLUMNS) == 43
    assert tuple(config["task"]["labels"]) == LABEL_COLUMNS
    assert config["task"]["recipe"] == "real_only"
    assert hashlib.sha256(
        canonical_json_bytes(list(FUSION_COLUMNS))
    ).hexdigest() == FUSION_V1_ALLOWLIST_SHA256
    assert config["task"]["feature_allowlist_sha256"] == FUSION_V1_ALLOWLIST_SHA256


def test_robust_v2_candidates_and_common_budget_are_exact() -> None:
    config = _config()
    candidates = robust_candidate_grid(config)

    assert candidates == [
        {"candidate_id": "balanced_erm", "objective": "balanced_erm"},
        {
            "candidate_id": "group_dro_eta_0_01",
            "objective": "group_dro",
            "eta": 0.01,
        },
        {
            "candidate_id": "group_dro_eta_0_1",
            "objective": "group_dro",
            "eta": 0.1,
        },
        {
            "candidate_id": "v_rex_beta_10",
            "objective": "v_rex",
            "beta": 10.0,
            "warmup_epochs": 40,
        },
        {
            "candidate_id": "v_rex_beta_100",
            "objective": "v_rex",
            "beta": 100.0,
            "warmup_epochs": 40,
        },
        {
            "candidate_id": "balanced_erm_sam_rho_0_05",
            "objective": "balanced_erm",
            "sam": True,
            "rho": 0.05,
        },
    ]
    assert config["models"]["torch_mlp"] == {
        "role": "sole_candidate_family",
        "output_strategy": "shared_xyz",
        "hidden_layer_sizes": [64, 64],
        "weight_decay": 0.001,
        "activation": "relu",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "batch_size": 128,
        "epoch_budget": 400,
        "ensemble_seed_count": 3,
        "dtype": "float32",
        "deterministic_algorithms": True,
        "checkpoint_rule": "minimum_inner_validation_rmse_3d_mm",
        "refit_epoch_rule": "rounded_median_inner_best_epoch_per_seed",
    }
    assert config["training_protocol"] == {
        "paired_candidate_model_seeds": True,
        "balanced_batch_rule": "floor_max_batch_rows_divided_by_training_group_count",
        "group_dro_update": "exponentiated_gradient_per_batch",
        "group_dro_weight_floor": 1.0e-12,
        "v_rex_schedule": "zero_beta_through_warmup_epochs_then_full_beta",
        "sam_variant": "nonadaptive_l2_two_pass",
        "inner_checkpoint_evaluation_cadence": "every_epoch",
        "unpromoted_candidate_default_inference": "forbidden",
    }
    groups = config["splits"]["primary"]["groups"]
    context_count = len(groups) * (len(groups) - 1)
    assert groups == [36, 37, 38, 39, 40]
    assert context_count == config["selection"]["context_count_per_candidate"] == 20
    assert context_count == config["splits"]["expected_inner_contexts_per_candidate"]
    assert len(candidates) * context_count == 120


def test_robust_v2_selection_promotion_and_confirmation_boundaries_are_exact() -> None:
    config = _config()
    selection = config["selection"]

    mean_rmse = 14.0
    worst_segment_mean_rmse = 18.0
    robust_j = (mean_rmse + worst_segment_mean_rmse) / 2.0
    assert robust_j == 16.0
    assert selection["score_name"] == "robust_j_mm"
    assert selection["tie_break_order"] == [
        "mean_p95_3d_mm",
        "seed_dispersion_rmse_3d_mm",
        "listed_order",
    ]
    assert selection["historical_v1_j_mm"] == HISTORICAL_V1_ROBUST_J_MM == 16.3801
    assert selection["numeric_equality_abs_tolerance_mm"] == 1.0e-12
    assert robust_j < selection["historical_v1_j_mm"]
    assert selection["promotion_rule"] == "strict_candidate_j_less_than_historical_v1_j"
    assert selection["no_improvement_action"] == "retain_v1"
    assert selection["diagnostic_scores_can_trigger_reselection"] is False

    assert config["evaluation"] == {
        "outer_designation": "reused_holdout_development_diagnostic",
        "outer_scores_are_independent_test": False,
        "selection_must_be_frozen_before_access": True,
        "reselection_after_outer_diagnostic": "forbidden",
    }
    assert config["confirmatory"]["occurs_after_candidate_and_model_freeze"] is True
    assert config["confirmatory"]["data_requirement"] == "independent_restart_sessions"
    assert config["confirmatory"]["minimum_session_count"] >= 10
    assert config["confirmatory"]["no_tuning_or_reselection_on_confirmatory_data"] is True


@pytest.mark.parametrize(
    ("section", "key", "changed_value"),
    [
        ("contract", "post_v1_outer_observed", False),
        ("lineage", "base_v1_config_sha256", "0" * 64),
        ("task", "recipe", "augmented"),
        ("selection", "historical_v1_j_mm", 16.4),
        ("evaluation", "outer_designation", "independent_test"),
        ("confirmatory", "minimum_session_count", 9),
    ],
)
def test_robust_v2_rejects_material_drift(
    section: str,
    key: str,
    changed_value: object,
) -> None:
    changed = deepcopy(_config())
    changed[section][key] = changed_value
    with pytest.raises(ValueError):
        validate_robust_gpu_config(changed)


def test_robust_v2_rejects_candidate_value_or_tie_order_drift() -> None:
    changed_candidate = deepcopy(_config())
    changed_candidate["candidates"][1]["eta"] = 0.02
    with pytest.raises(ValueError, match="candidate"):
        validate_robust_gpu_config(changed_candidate)

    changed_order = deepcopy(_config())
    changed_order["selection"]["tie_break_order"].reverse()
    with pytest.raises(ValueError, match="tie-break"):
        validate_robust_gpu_config(changed_order)


def test_robust_v2_loader_allows_only_non_material_runtime_paths(tmp_path: Path) -> None:
    release_override = tmp_path / "release.zip"
    loaded = load_robust_gpu_config(CONFIG_PATH, release_override=release_override)

    assert robust_gpu_config_sha256(loaded) == PINNED_ROBUST_GPU_CONFIG_SHA256
    assert loaded["_config_path"] == str(CONFIG_PATH.resolve())
    assert resolve_robust_gpu_release_path(loaded) == release_override.resolve()


@pytest.mark.parametrize(
    "command",
    ["smoke-gpu-robust-experiment", "run-gpu-robust-experiment"],
)
def test_robust_v2_gpu_run_cli_shapes_are_stable(command: str) -> None:
    args = build_parser().parse_args(
        [
            command,
            "--release",
            "release.zip",
            "--config",
            "robust.yaml",
            "--output",
            "robust-run",
            "--gpus",
            "3",
            "2",
            "1",
            "0",
            "--resume",
        ]
    )
    assert args.command == command
    assert args.release == Path("release.zip")
    assert args.config == Path("robust.yaml")
    assert args.output == Path("robust-run")
    assert args.gpus == [3, 2, 1, 0]
    assert args.resume is True


def test_robust_v2_gpu_validate_cli_shape_is_stable() -> None:
    args = build_parser().parse_args(
        ["validate-gpu-robust-experiment", "--run-dir", "robust-run"]
    )
    assert args.command == "validate-gpu-robust-experiment"
    assert args.run_dir == Path("robust-run")


@pytest.mark.parametrize(
    ("command", "runner_name"),
    [
        ("smoke-gpu-robust-experiment", "run_robust_gpu_smoke"),
        ("run-gpu-robust-experiment", "run_robust_gpu_experiment"),
    ],
)
def test_robust_v2_gpu_run_cli_dispatches_without_eager_gpu_work(
    command: str,
    runner_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import imu_vive_learning.gpu_robust_experiment as robust_experiment
    import imu_vive_learning.gpu_robust_experiment_config as robust_config

    calls: dict[str, object] = {}
    loaded_config = {"loaded": True}

    def fake_load(path: Path, *, release_override: Path) -> dict:
        calls["load"] = (path, release_override)
        return loaded_config

    def fake_runner(
        config: dict,
        output: Path,
        *,
        gpu_ids: list[int],
        resume: bool,
    ) -> dict:
        calls["run"] = (config, output, gpu_ids, resume)
        return {"status": "ok", "kind": command}

    monkeypatch.setattr(robust_config, "load_robust_gpu_config", fake_load)
    monkeypatch.setattr(robust_experiment, runner_name, fake_runner)
    status = main(
        [
            command,
            "--release",
            "release.zip",
            "--config",
            "robust.yaml",
            "--output",
            "robust-run",
            "--gpus",
            "0",
            "1",
            "2",
            "3",
            "--resume",
        ]
    )

    assert status == 0
    assert calls == {
        "load": (Path("robust.yaml"), Path("release.zip")),
        "run": (loaded_config, Path("robust-run"), [0, 1, 2, 3], True),
    }
    assert json.loads(capsys.readouterr().out) == {"status": "ok", "kind": command}


def test_robust_v2_gpu_validate_cli_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import imu_vive_learning.gpu_robust_experiment as robust_experiment

    calls: list[Path] = []

    def fake_validate(run_dir: Path) -> dict:
        calls.append(run_dir)
        return {"status": "ok", "gate_count": 16}

    monkeypatch.setattr(
        robust_experiment, "validate_robust_gpu_run", fake_validate
    )
    status = main(
        ["validate-gpu-robust-experiment", "--run-dir", "robust-run"]
    )

    assert status == 0
    assert calls == [Path("robust-run")]
    assert json.loads(capsys.readouterr().out) == {
        "status": "ok",
        "gate_count": 16,
    }
