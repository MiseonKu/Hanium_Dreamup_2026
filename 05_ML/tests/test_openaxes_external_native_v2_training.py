from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from imu_vive_learning.openaxes_external_native_v2_training import (
    FeatureOnlyBatch,
    OpenAxesExternalNativeV2TrainingError,
    assemble_cagrad_full_direction,
    build_dense_6979,
    build_exact_five_views,
    build_sensor_experts_6979,
    direct_mask_law_objective,
    fit_train_only_transform,
    fixed_dot_float64,
    fixed_mean_float64,
    fixed_sum_float64,
    predict_feature_only_batch_once,
    scenario_balanced_view_risks,
    solve_cagrad_outage_weights,
    state_dict_sha256,
    train_candidate_train_only,
    training_checkpoint_score_m,
    transform_features,
    validate_exact_five_views,
)


def _synthetic_training() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rows = 8
    base = np.arange(rows, dtype=np.float64)[:, None]
    offsets = np.arange(36, dtype=np.float64)[None, :] / 10.0
    features = base + offsets
    targets = np.column_stack(
        (base[:, 0] * 0.01, base[:, 0] * -0.02, base[:, 0] * 0.03)
    )
    scenarios = ["circle-100"] * 4 + ["triangle-100"] * 4
    uids = [f"synthetic-{index:02d}" for index in range(rows)]
    return features, targets, scenarios, uids


def test_exact_five_view_masking_and_train_only_transform() -> None:
    features, targets, scenarios, uids = _synthetic_training()
    transform = fit_train_only_transform(features, targets, uids, scenarios)
    assert transform.train_rows == 8
    assert transform.receipt()["outer_rows_used"] == 0
    scaled = transform_features(features, transform)
    views = build_exact_five_views(scaled)
    validate_exact_five_views(scaled, views)
    assert views.shape == (5, 8, 40)
    assert np.array_equal(views[0, :, 36:], np.ones((8, 4)))
    for sensor_index in range(4):
        outage = views[sensor_index + 1]
        assert np.array_equal(
            outage[:, sensor_index * 9 : (sensor_index + 1) * 9],
            np.zeros((8, 9)),
        )
        assert np.array_equal(
            outage[:, 36:],
            np.tile(
                np.asarray(
                    [0.0 if item == sensor_index else 1.0 for item in range(4)]
                ),
                (8, 1),
            ),
        )


def test_exact_6979_parameter_models_and_paired_expert_state() -> None:
    dense = build_dense_6979(seed=1701, device="cpu")
    expert_b = build_sensor_experts_6979(seed=2603, device="cpu")
    expert_c = build_sensor_experts_6979(seed=2603, device="cpu")
    assert sum(parameter.numel() for parameter in dense.parameters()) == 6979
    assert sum(parameter.numel() for parameter in expert_b.parameters()) == 6979
    assert state_dict_sha256(expert_b.state_dict()) == state_dict_sha256(
        expert_c.state_dict()
    )
    sample = torch.zeros((3, 40), dtype=torch.float32)
    sample[:, 36:] = 1.0
    assert dense(sample).shape == (3, 3)
    assert expert_b(sample).shape == (3, 3)


def test_fixed_reductions_scenario_balance_and_checkpoint() -> None:
    values = np.asarray([1.0e16, 1.0, -1.0e16, 3.0, 5.0], dtype=np.float64)
    assert fixed_sum_float64(values) == ((1.0e16 + 1.0) + (-1.0e16 + 3.0)) + 5.0
    assert fixed_mean_float64([2.0, 4.0, 6.0]) == 4.0
    assert fixed_dot_float64([1.0, 2.0], [3.0, 4.0]) == 11.0
    losses = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    codes = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    risks = scenario_balanced_view_risks(losses, codes, 2)
    expected = torch.stack((losses[:2].mean(0), losses[2:].mean(0))).mean(0)
    assert torch.equal(risks, expected)
    objective = direct_mask_law_objective(risks)
    assert torch.isfinite(objective)
    prediction = np.zeros((5, 4, 3), dtype=np.float64)
    truth = np.ones((4, 3), dtype=np.float64)
    score, outages, components = training_checkpoint_score_m(
        prediction,
        truth,
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        2,
        0.5,
    )
    assert score == outages[0]
    assert outages == tuple(components[1:])
    assert score == pytest.approx(np.sqrt(3.0) * 0.5)


def test_cagrad_one_start_solution_and_fail_closed_inputs() -> None:
    gradients = np.eye(4, dtype=np.float64)
    solved = solve_cagrad_outage_weights(gradients)
    assert solved["weights"].shape == (4,)
    assert solved["equality_residual"] <= 1.0e-10
    assert solved["bound_violation"] <= 1.0e-10
    assert np.isfinite(solved["outage_direction"]).all()
    assembled = assemble_cagrad_full_direction(
        np.asarray([1.0, 2.0, 3.0, 4.0]), gradients
    )
    assert assembled["final_direction"].shape == (4,)
    with pytest.raises(OpenAxesExternalNativeV2TrainingError):
        solve_cagrad_outage_weights(np.full((4, 3), np.nan))
    with pytest.raises(OpenAxesExternalNativeV2TrainingError):
        assemble_cagrad_full_direction(np.ones(3), np.ones((4, 4)))


def test_feature_only_type_and_training_signatures_exclude_outer_authority() -> None:
    features, _targets, scenarios, uids = _synthetic_training()
    batch = FeatureOnlyBatch(
        continuous_features=features,
        row_uids=tuple(uids),
        scenario_ids=tuple(scenarios),
        sample_indices=tuple(range(8)),
        aligned_time_us=tuple(index * 10000 for index in range(8)),
    )
    batch.validate()
    training_parameters = set(inspect.signature(train_candidate_train_only).parameters)
    prediction_parameters = set(
        inspect.signature(predict_feature_only_batch_once).parameters
    )
    assert not {
        "outer_features",
        "outer_labels",
        "outer_targets",
        "test_features",
        "test_targets",
    }.intersection(training_parameters)
    assert not {"truth", "labels", "targets"}.intersection(prediction_parameters)
    with pytest.raises(TypeError):
        train_candidate_train_only(outer_labels=np.zeros((1, 3)))
