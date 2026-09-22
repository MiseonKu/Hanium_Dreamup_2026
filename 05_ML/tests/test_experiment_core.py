from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from imu_vive_learning.experiment_metrics import (
    cell_balanced_metrics,
    macro_group_metrics,
    paired_group_effects,
    paired_multiway_bootstrap,
    regression_metrics,
    rmse_3d_mm,
    rmse_xyz_mm,
)
from imu_vive_learning.experiment_models import (
    augmentation_gate,
    derive_context_seed,
    fit_endpoint_regressor,
    mlp_epochs_for_recipe,
    model_parameter_grid,
    paired_seed_sequence,
    select_best_configuration,
    select_final_candidate,
)
from imu_vive_learning.experiment_preprocessing import (
    TrainOnlyPreprocessor,
    recipe_sample_weights,
    validate_feature_allowlist,
    weight_mass_audit,
)


def test_rmse_3d_definition_and_axis_element_relation() -> None:
    truth = np.zeros((2, 3))
    prediction = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]])

    expected_3d = math.sqrt((25.0 + 144.0) / 2.0)
    assert rmse_3d_mm(truth, prediction) == pytest.approx(expected_3d)
    assert rmse_3d_mm(truth, prediction) == pytest.approx(
        math.sqrt(3.0) * rmse_xyz_mm(truth, prediction)
    )

    metrics = regression_metrics(truth, prediction)
    assert metrics["rmse_3d_mm"] == pytest.approx(expected_3d)
    assert metrics["mean_3d_error_mm"] == pytest.approx(8.5)
    assert metrics["median_3d_error_mm"] == pytest.approx(8.5)
    assert metrics["bias_x_mm"] == pytest.approx(1.5)
    assert metrics["mae_y_mm"] == pytest.approx(2.0)
    assert metrics["rmse_z_mm"] == pytest.approx(math.sqrt(72.0))


def test_macro_and_cell_balancing_do_not_weight_large_groups_more() -> None:
    truth = np.zeros((4, 3))
    prediction = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 0.0, 0.0]]
    )
    macro = macro_group_metrics(truth, prediction, ["large", "large", "large", "small"])
    assert macro["macro"]["rmse_3d_mm"] == pytest.approx(5.0)

    cells = cell_balanced_metrics(
        truth,
        prediction,
        segment_ids=[36, 36, 36, 37],
        target_ids=["T01", "T01", "T01", "T01"],
    )
    assert cells["cell_count"] == 2
    assert cells["macro_cell_rmse_3d_mm"] == pytest.approx(5.0)
    assert cells["cell_balanced_rmse_3d_mm"] == pytest.approx(math.sqrt(41.0))


def test_paired_multiway_bootstrap_is_deterministic_and_reports_direction() -> None:
    truth = np.zeros((8, 3))
    reference = np.column_stack([np.full(8, 2.0), np.zeros((8, 2))])
    candidate = np.column_stack([np.full(8, 1.0), np.zeros((8, 2))])
    segments = [36, 36, 36, 36, 37, 37, 37, 37]
    targets = ["T01", "T01", "T02", "T02"] * 2

    first = paired_multiway_bootstrap(
        truth,
        reference,
        candidate,
        segments,
        targets,
        n_resamples=100,
        seed=123,
    )
    second = paired_multiway_bootstrap(
        truth,
        reference,
        candidate,
        segments,
        targets,
        n_resamples=100,
        seed=123,
    )
    assert first == second
    assert first["delta_direction"] == "candidate_minus_reference"
    assert first["rmse_3d_mm"]["observed_delta"] == pytest.approx(-1.0)
    assert first["rmse_3d_mm"]["ci_low"] == pytest.approx(-1.0)
    assert first["rmse_3d_mm"]["ci_high"] == pytest.approx(-1.0)
    effects = paired_group_effects(truth, reference, candidate, segments)
    assert [row["delta_rmse_3d_mm"] for row in effects] == pytest.approx([-1.0, -1.0])


def test_train_only_preprocessor_uses_shared_target_scale_and_audits_ids() -> None:
    X = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    y = np.array([[0.0, 100.0, 1000.0], [2.0, 104.0, 1006.0], [4.0, 108.0, 1012.0]])
    preprocessor = TrainOnlyPreprocessor(
        feature_names=("encoder_delta_rad_j0", "imu_accel_x"),
        split_context="segment_loso:outer=36:inner=37",
    ).fit_real_train(
        X,
        y,
        trial_uids=["a", "b", "c"],
        forbidden_uids=["test"],
    )

    transformed = preprocessor.transform_targets(y)
    assert np.mean(transformed, axis=0) == pytest.approx([0.0, 0.0, 0.0])
    assert np.sqrt(np.mean(np.square(transformed))) == pytest.approx(1.0)
    assert preprocessor.inverse_targets(transformed) == pytest.approx(y)
    metadata = preprocessor.metadata()
    assert metadata["leakage_audit"]["fit_scope"] == "real_train_only"
    assert metadata["leakage_audit"]["forbidden_uid_overlap_count"] == 0

    with pytest.raises(ValueError, match="overlap forbidden"):
        TrainOnlyPreprocessor(feature_names=("safe",)).fit_real_train(
            np.ones((2, 1)),
            np.ones((2, 3)),
            trial_uids=["train", "test"],
            forbidden_uids=["test"],
        )


def test_tree_preprocessor_does_not_transform_features() -> None:
    X = np.array([[1.0], [2.0], [8.0]])
    y = np.arange(9.0).reshape(3, 3)
    preprocessor = TrainOnlyPreprocessor(
        feature_names=("safe",), scale_features=False
    ).fit_real_train(X, y, trial_uids=["a", "b", "c"])
    assert preprocessor.transform_features(X) == pytest.approx(X)


def test_allowlist_rejects_labels_context_and_unapproved_planned_columns() -> None:
    assert validate_feature_allowlist(["encoder_delta_rad_j0"]) == (
        "encoder_delta_rad_j0",
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_feature_allowlist(["vive_pos_raw_x_mm"])
    with pytest.raises(ValueError, match="leakage"):
        validate_feature_allowlist(["planned_p_x_mm"])
    assert validate_feature_allowlist(
        ["planned_p_x_mm"], allow_planned_diagnostic=True
    ) == ("planned_p_x_mm",)


def test_locked_recipe_weights_preserve_objective_mass() -> None:
    real = recipe_sample_weights(4, recipe="real_only")
    replay = recipe_sample_weights(4, 4, recipe="replay")
    assert real.tolist() == [1.0] * 4
    assert replay.tolist() == [0.5] * 8
    assert weight_mass_audit(replay, real_row_count=4, recipe="replay")[
        "objective_mass"
    ] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="exactly 1:1"):
        recipe_sample_weights(4, 3, recipe="local_interp_v2")


def test_seed_grids_epochs_and_tie_break_are_frozen() -> None:
    assert derive_context_seed(20260818, "fold", 36) == derive_context_seed(
        20260818, "fold", 36
    )
    assert len(set(paired_seed_sequence("outer36"))) == 3
    assert len(model_parameter_grid("ridge")) == 4
    assert len(model_parameter_grid("extra_trees")) == 4
    assert len(model_parameter_grid("xgboost")) == 8
    assert len(model_parameter_grid("mlp")) == 4
    assert mlp_epochs_for_recipe("real_only") == 400
    assert mlp_epochs_for_recipe("cmixup_offline_v1") == 200

    ridge_grid = model_parameter_grid("ridge")
    selected = select_best_configuration(
        "ridge",
        [
            {"params": ridge_grid[2], "inner_scores": [10.0, 10.0]},
            {"params": ridge_grid[0], "inner_scores": [10.0, 10.0]},
        ],
    )
    assert selected["params"] == ridge_grid[0]
    assert selected["tied_candidate_count"] == 2


def test_gate_and_final_selection_fall_back_or_accept_only_gated_augmentation() -> None:
    failed = augmentation_gate(
        reference_macro_rmse_3d_mm=20.0,
        candidate_macro_rmse_3d_mm=17.1,
        reference_macro_p95_3d_error_mm=40.0,
        candidate_macro_p95_3d_error_mm=40.0,
    )
    passed = augmentation_gate(
        reference_macro_rmse_3d_mm=40.0,
        candidate_macro_rmse_3d_mm=35.0,
        reference_macro_p95_3d_error_mm=60.0,
        candidate_macro_p95_3d_error_mm=62.0,
    )
    assert failed["passed"] is False
    assert passed["passed"] is True

    base = {
        "model_name": "ridge",
        "feature_view": "fusion_v1",
        "recipe": "real_only",
        "macro_rmse_3d_mm": 40.0,
    }
    rejected_aug = {
        **base,
        "recipe": "local_interp_v2",
        "macro_rmse_3d_mm": 30.0,
        "augmentation_gate_passed": False,
    }
    assert select_final_candidate([base, rejected_aug])["recipe"] == "real_only"
    accepted_aug = {
        **rejected_aug,
        "augmentation_gate_passed": True,
    }
    assert select_final_candidate([base, accepted_aug])["recipe"] == "local_interp_v2"
    better_encoder_real = {
        **base,
        "feature_view": "encoder_only_v1",
        "macro_rmse_3d_mm": 20.0,
    }
    selected = select_final_candidate([base, accepted_aug, better_encoder_real])
    assert selected["feature_view"] == "encoder_only_v1"
    assert selected["recipe"] == "real_only"


def test_ridge_replay_is_equivalent_and_inference_schema_is_exact() -> None:
    real_X = np.arange(18.0).reshape(6, 3)
    real_y = np.column_stack(
        [real_X[:, 0] * 2.0, real_X[:, 1] * -1.0, real_X[:, 2] + 4.0]
    )
    names = ("f0", "f1", "f2")
    uids = [f"trial-{index}" for index in range(len(real_X))]
    real_model = fit_endpoint_regressor(
        "ridge",
        {"alpha": 0.1},
        real_X,
        real_y,
        feature_names=names,
        X_real_train=real_X,
        y_real_train=real_y,
        real_trial_uids=uids,
        sample_weight=recipe_sample_weights(6, recipe="real_only"),
        recipe="real_only",
        split_context="fixture",
        seed=1234,
    )
    replay_model = fit_endpoint_regressor(
        "ridge",
        {"alpha": 0.1},
        np.concatenate([real_X, real_X]),
        np.concatenate([real_y, real_y]),
        feature_names=names,
        X_real_train=real_X,
        y_real_train=real_y,
        real_trial_uids=uids,
        sample_weight=recipe_sample_weights(6, 6, recipe="replay"),
        recipe="replay",
        split_context="fixture",
        seed=1234,
    )
    assert replay_model.seed == real_model.seed

    alternate_alpha = fit_endpoint_regressor(
        "ridge",
        {"alpha": 10.0},
        real_X,
        real_y,
        feature_names=names,
        X_real_train=real_X,
        y_real_train=real_y,
        real_trial_uids=uids,
        sample_weight=np.ones(len(real_y)),
        recipe="real_only",
        split_context="segment_loso:outer=36:inner=37",
        seed=1234,
    )
    assert alternate_alpha.seed == real_model.seed == 1234
    assert replay_model.predict(real_X) == pytest.approx(real_model.predict(real_X))

    frame = pd.DataFrame(real_X, columns=names)
    predictions, row_ids = real_model.predict_frame(frame)
    assert predictions.shape == (6, 3)
    assert row_ids is None
    with pytest.raises(ValueError, match="schema/order mismatch"):
        real_model.predict_frame(frame.loc[:, ["f1", "f0", "f2"]])
    with pytest.raises(ValueError, match="non-empty and finite"):
        bad = frame.copy()
        bad.loc[0, "f0"] = np.nan
        real_model.predict_frame(bad)
