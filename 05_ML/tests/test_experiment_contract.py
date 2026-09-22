from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import tomllib

import pytest
import yaml

from imu_vive_learning.experiment_config import (
    PINNED_EXPERIMENT_CONFIG_SHA256,
    experiment_config_sha256,
    validate_experiment_config,
)
from imu_vive_learning.features import feature_views


WORKSPACE = Path(__file__).parents[1]
CONFIG_PATH = WORKSPACE / "configs" / "endpoint_regression_v1.yaml"


def _config() -> dict:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_experiment_identity_and_parent_release_are_frozen() -> None:
    config = _config()
    validate_experiment_config(config)
    assert experiment_config_sha256(config) == PINNED_EXPERIMENT_CONFIG_SHA256
    assert config["contract"] == {
        "name": "alphaf.endpoint_regression.experiment.v1",
        "base_seed": 20260818,
        "research_status": "exploratory_v1",
        "parent_contract": "alphaf.ml_preparation.v1",
        "label_frame": "vive_world",
        "label_unit": "mm",
        "independent_test_available": False,
    }
    release = config["release"]
    assert release["archive_sha256"] == (
        "94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930"
    )
    assert release["expected_contract"] == "alphaf.ml_preparation.v1"
    assert (
        release["expected_canonical_rows"],
        release["expected_train_eligible_rows"],
        release["expected_strict_qc_rows"],
    ) == (7707, 7702, 7670)
    assert release["require_manifest_checksums"] is True
    assert release["reject_lfs_pointer"] is True


def test_any_scientifically_material_config_drift_is_rejected() -> None:
    changed = deepcopy(_config())
    changed["models"]["xgboost"]["learning_rate"] = 0.06
    with pytest.raises(ValueError, match="pinned canonical SHA-256"):
        validate_experiment_config(changed)


def test_nested_group_contexts_imply_exact_assignment_count() -> None:
    splits = _config()["splits"]
    schemes = splits["schemes"]
    assert set(schemes) == {"segment_loso", "target_group"}
    assert schemes["segment_loso"]["role"] == "primary"
    assert schemes["segment_loso"]["outer_groups"] == [36, 37, 38, 39, 40]
    assert schemes["target_group"]["role"] == "secondary"
    assert schemes["target_group"]["outer_groups"] == [0, 1, 2, 3, 4]
    assert schemes["target_group"]["group_rule"] == "int(target_id[1:]) % 5"

    contexts = 0
    for scheme in schemes.values():
        groups = scheme["outer_groups"]
        assert scheme["outer_contexts"] == len(groups) == 5
        assert scheme["inner_contexts"] == len(groups) * (len(groups) - 1) == 20
        assert scheme["inner_policy"] == "each_remaining_group_once"
        contexts += scheme["outer_contexts"] + scheme["inner_contexts"]
    assert contexts == splits["expected_contexts"] == 50
    assert contexts * 7702 == splits["expected_assignment_rows"] == 385100
    assert splits["validation_and_test_real_only"] is True


def test_primary_views_and_matrix_keep_shortcuts_out_of_deployment() -> None:
    config = _config()
    features = config["features"]
    views = feature_views()
    assert features["primary_views"] == {
        "encoder": "encoder_only_v1",
        "four_imu": "four_imu_endpoint_v1",
        "fusion": "fusion_v1",
    }
    for view_name in features["primary_views"].values():
        columns = views[view_name]
        assert not any(
            token in column.lower()
            for column in columns
            for token in ("vive", "tracker", "planned", "_mag_")
        )

    matrix = config["experiment_matrix"]
    assert matrix["baseline"] == {
        "models": ["dummy_mean"],
        "feature_views": ["none"],
        "recipes": ["real_only"],
    }
    assert matrix["sensor_comparison"] == {
        "models": ["ridge", "extra_trees", "xgboost", "mlp"],
        "feature_views": ["encoder", "four_imu", "fusion"],
        "recipes": ["real_only"],
    }
    assert matrix["augmentation_comparison"]["models"] == [
        "ridge",
        "extra_trees",
        "xgboost",
        "mlp",
    ]
    assert matrix["augmentation_comparison"]["feature_views"] == ["fusion"]
    assert matrix["augmentation_comparison"]["recipes"] == [
        "real_only",
        "replay",
        "local_interp_v2",
        "cmixup_offline_v1",
    ]
    assert matrix["shortcut_diagnostics"]["models"] == ["ridge", "extra_trees"]
    assert matrix["shortcut_diagnostics"]["feature_views"] == [
        "planned_only",
        "fusion_planned",
        "fusion_mag",
    ]
    excluded = matrix["deployment_exclusions"]
    assert excluded == {
        "models": ["dummy_mean"],
        "recipes": ["replay"],
        "feature_views": ["planned_only", "fusion_planned", "fusion_mag"],
    }
    assert "replay" not in config["selection"]["recipe_priority"]


def test_model_grids_and_cpu_thread_contract_are_exact() -> None:
    models = _config()["models"]
    assert models["ridge"]["alpha"] == [0.1, 1.0, 10.0, 100.0]
    assert models["extra_trees"] == {
        "n_estimators": 400,
        "max_features": [0.5, 1.0],
        "min_samples_leaf": [1, 4],
        "n_jobs": 1,
    }
    assert models["xgboost"] == {
        "output_strategy": "independent_xyz",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": [4, 8],
        "min_child_weight": [1.0, 5.0],
        "reg_lambda": [1.0, 10.0],
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "n_jobs": 1,
    }
    assert models["mlp"]["hidden_layer_sizes"] == [[64, 64], [128, 64]]
    assert models["mlp"]["alpha"] == [0.0001, 0.001]
    assert models["mlp"]["ensemble_seed_count"] == 3
    assert models["mlp"]["early_stopping"] is False

    runtime = _config()["runtime"]
    assert runtime["default_jobs"] == 4
    assert runtime["estimator_threads"] == 1
    assert runtime["deterministic_context_hash_seeds"] is True


def test_preprocessing_weights_budget_metric_gate_and_bootstrap_are_frozen() -> None:
    config = _config()
    preprocessing = config["preprocessing"]
    assert preprocessing["fit_scope"] == "real_train_only"
    assert preprocessing["feature_scaling_models"] == ["ridge", "mlp"]
    assert preprocessing["target_center"] == "train_xyz_mean"
    assert preprocessing["target_scale"] == "one_isotropic_scale"
    assert preprocessing["preserve_axis_weighting"] is True

    recipes = config["recipes"]
    assert recipes["real_only"]["real_weight"] == 1.0
    assert recipes["real_only"]["epoch_budget"] == 400
    for name in ("replay", "local_interp_v2", "cmixup_offline_v1"):
        assert recipes[name]["real_weight"] == 0.5
        assert recipes[name]["added_weight"] == 0.5
        assert recipes[name]["epoch_budget"] == 200
    assert recipes["replay"]["source"] == "duplicate_real_train_once"
    assert recipes["outer_refit_materialized_from_release"] is True

    metrics = config["metrics"]
    assert metrics["primary"] == "rmse_3d_mm"
    assert metrics["legacy_axis_pooled_name"] == "rmse_xyz_mm"
    assert metrics["verify_rmse_relation"] is True
    residual = [[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]
    squared = [value * value for row in residual for value in row]
    rmse_xyz = math.sqrt(sum(squared) / len(squared))
    rmse_3d = math.sqrt(sum(squared) / len(residual))
    assert math.isclose(rmse_3d, math.sqrt(3.0) * rmse_xyz)

    gate = config["augmentation_gate"]
    assert gate["comparison_feature_view"] == "fusion"
    assert gate["baseline_recipe"] == "real_only"
    assert gate["minimum_relative_rmse_improvement"] == 0.10
    assert gate["minimum_absolute_rmse_improvement_mm"] == 3.0
    assert gate["maximum_relative_p95_degradation"] == 0.05
    assert gate["require_both_improvement_thresholds_via_max"] is True

    uncertainty = config["uncertainty"]
    assert uncertainty == {
        "method": "deterministic_multiway_bootstrap",
        "replicates": 10000,
        "cluster_factors": ["segment_id", "target_id"],
        "confidence_level": 0.95,
        "report_p_values": False,
    }


def test_final_model_and_inference_claim_boundaries_are_machine_readable() -> None:
    config = _config()
    final_model = config["final_model"]
    assert final_model["selection_scheme"] == "segment_loso"
    assert final_model["selection_contexts"] == 20
    assert final_model["allow_only_gate_passing_augmentation"] is True
    assert final_model["fallback_recipe"] == "real_only"
    assert final_model["train_on_all_eligible_real"] is True
    assert final_model["independent_test_score_available"] is False

    assert config["outputs"]["prediction_columns"] == [
        "pred_vive_world_x_mm",
        "pred_vive_world_y_mm",
        "pred_vive_world_z_mm",
    ]
    assert config["outputs"]["inference_optional_columns"] == ["row_id"]


def test_eval_extra_and_exact_lock_use_available_cpu_versions() -> None:
    pyproject = tomllib.loads((WORKSPACE / "pyproject.toml").read_text(encoding="utf-8"))
    evaluation = pyproject["project"]["optional-dependencies"]["eval"]
    assert "scikit-learn>=1.9,<1.10" in evaluation
    assert "xgboost>=3.3,<3.4" in evaluation
    assert "scipy>=1.17,<1.18" in evaluation
    assert "matplotlib>=3.11,<3.12" in evaluation
    assert "joblib>=1.5,<1.6" in evaluation
    assert "threadpoolctl>=3.6,<3.7" in evaluation

    lock_lines = {
        line.strip()
        for line in (WORKSPACE / "requirements-eval.lock")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert {
        "scikit-learn==1.9.0",
        "xgboost==3.3.0",
        "scipy==1.17.1",
        "matplotlib==3.11.1",
        "joblib==1.5.3",
        "threadpoolctl==3.6.0",
    } <= lock_lines
    assert not any(line.startswith("xgboost==3.2") for line in lock_lines)
