from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from imu_vive_learning.gpu_outage_pord_v5_training import (
    PORD_BACKBONE_IDS,
    PORD_BACKBONE_SETTINGS_BY_ID,
    PORD_CHAIN_BACKBONE_ID,
    PORD_CHAIN_TRANSFER_BACKBONE_ID,
    PORD_CHECKPOINT_CRITERION,
    PORD_CHECKPOINT_SELECTION_SCOPE,
    PORD_EPOCH_BUDGET,
    PORD_FULL_GROUP_WEIGHT,
    PORD_GROUPDRO_ETA,
    PORD_LEARNING_RATE,
    PORD_MASKED_MLP_BACKBONE_ID,
    PORD_MASK_PROBABILITIES,
    PORD_METHOD_IDS,
    PORD_MISSING_GROUP_WEIGHT,
    PORD_MOFE_LAMBDA,
    PORD_PRIMARY_BACKBONE_ID,
    PORD_TOP_K,
    PORD_TRANSFER_METHOD_IDS,
    PORD_VREX_BETA,
    PORD_WEIGHT_DECAY,
    PORDV5TrainingError,
    build_constant_availability_five_views,
    build_exact_five_views,
    build_pord_backbone,
    compute_controlled_objective,
    compute_empirical_group_risks,
    compute_row_paired_atoms,
    direct_group_risk_gradients,
    direct_row_loss_gradients,
    earliest_checkpoint_epoch,
    evaluate_pord_checkpoint_once,
    final_epochs_by_method_seed,
    fixed_symmetric_positive_topk,
    fixed_within_segment_circular_derangement,
    full_sensor_numeric_gate,
    half_up_median_epoch,
    inverse_standardized_targets,
    objective_audit_record,
    outage_checkpoint_score,
    state_dict_sha256,
    train_pord_fold_seed,
    two_way_center,
    verify_paired_method_controls,
    verify_pord_outer_evaluation,
    verify_pord_result_hash,
    verify_transfer_method_controls,
)
from imu_vive_learning.gpu_v3_backbones import IMU_SENSOR_INDEX_MAP


torch = pytest.importorskip("torch")


def _loss_predictions(losses: np.ndarray) -> torch.Tensor:
    """Encode desired per-row XYZ MSE in the first prediction coordinate."""

    values = np.zeros((*losses.shape, 3), dtype=np.float64)
    values[..., 0] = np.sqrt(3.0 * losses)
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


def _risks(
    full: list[float] | np.ndarray,
    missing: list[list[float]] | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(full, dtype=torch.float64, requires_grad=True),
        torch.tensor(missing, dtype=torch.float64, requires_grad=True),
    )


def _tiny_problem(seed: int = 941) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(9, 43)).astype(np.float32)
    weights = np.asarray(
        [[0.7, -0.2, 0.1], [-0.3, 0.5, 0.2], [0.2, 0.1, -0.6]],
        dtype=np.float32,
    )
    y_train = X_train[:, :3] @ weights
    return {
        "X_train_scaled": X_train,
        "y_train_scaled": y_train,
        "train_segment_ids": np.asarray([36, 36, 36, 37, 37, 37, 38, 38, 38]),
        "train_trial_uids": [f"train-{index}" for index in range(9)],
        "held_out_segment": 39,
        "target_mean_mm": np.asarray([310.0, -80.0, 720.0]),
        "target_scale_mm": np.asarray([25.0, 25.0, 25.0]),
        "model_seed": 811,
        "device": "cpu",
        "epoch_budget": 1,
    }


def _tiny_outer_problem(seed: int = 942) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    X_outer = rng.normal(size=(4, 43)).astype(np.float32)
    weights = np.asarray(
        [[0.7, -0.2, 0.1], [-0.3, 0.5, 0.2], [0.2, 0.1, -0.6]],
        dtype=np.float32,
    )
    y_outer_scaled = X_outer[:, :3] @ weights
    return {
        "X_outer_scaled": X_outer,
        "y_outer_raw_mm": (
            y_outer_scaled.astype(np.float64) * 25.0
            + np.asarray([310.0, -80.0, 720.0])
        ),
        "outer_segment_ids": [39, 39, 39, 39],
        "outer_trial_uids": [f"outer-{index}" for index in range(4)],
        "held_out_segment": 39,
        "target_mean_mm": np.asarray([310.0, -80.0, 720.0]),
        "target_scale_mm": np.asarray([25.0, 25.0, 25.0]),
        "device": "cpu",
    }


def test_exact_post_split_five_views_and_mask_law() -> None:
    features = np.tile(np.arange(1, 44, dtype=np.float32), (3, 1))
    uids = ["u0", "u1", "u2"]
    bundle = build_exact_five_views(features, trial_uids=uids)
    assert bundle.features.shape == (5, 3, 47)
    assert bundle.trial_uids == tuple(uids)
    assert np.array_equal(bundle.features[0, :, :43], features)
    assert np.array_equal(
        bundle.availability,
        np.asarray(
            [
                [1, 1, 1, 1],
                [0, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0],
            ],
            dtype=np.float32,
        ),
    )
    for sensor, columns in enumerate(IMU_SENSOR_INDEX_MAP):
        assert np.array_equal(
            bundle.features[sensor + 1][:, list(columns)],
            np.zeros((3, 10), dtype=np.float32),
        )
        other = [index for index in range(43) if index not in set(columns)]
        assert np.array_equal(bundle.features[sensor + 1][:, other], features[:, other])
    constant = build_constant_availability_five_views(features, trial_uids=uids)
    assert constant.availability_semantics == "constant_all_available_negative_control"
    assert np.array_equal(constant.availability, np.ones((5, 4), dtype=np.float32))
    assert constant.sha256 != bundle.sha256
    assert np.array_equal(constant.features[:, :, :43], bundle.features[:, :, :43])
    assert PORD_MASK_PROBABILITIES == (0.25, 0.1875, 0.1875, 0.1875, 0.1875)
    assert sum(PORD_MASK_PROBABILITIES) == 1.0
    assert 3 * PORD_FULL_GROUP_WEIGHT == 0.25
    assert 12 * PORD_MISSING_GROUP_WEIGHT == 0.75


def test_full_fold_group_means_ignore_unequal_sizes_and_uniform_duplication() -> None:
    # Segment sizes are 3, 1, and 2.  Risks must mean within segment before
    # giving every segment the same nominal objective mass.
    segments = np.asarray([0, 0, 0, 1, 2, 2])
    row_losses = np.asarray(
        [
            [1, 3, 5, 7, 9, 11],
            [2, 4, 6, 8, 10, 12],
            [3, 5, 7, 9, 11, 13],
            [4, 6, 8, 10, 12, 14],
            [5, 7, 9, 11, 13, 15],
        ],
        dtype=np.float64,
    )
    predictions = _loss_predictions(row_losses)
    targets = torch.zeros((6, 3), dtype=torch.float64)
    grouped = compute_empirical_group_risks(predictions, targets, segments)
    assert torch.allclose(
        grouped.full, torch.tensor([3.0, 7.0, 10.0], dtype=torch.float64)
    )
    expected_missing = torch.tensor(
        [[4, 5, 6, 7], [8, 9, 10, 11], [11, 12, 13, 14]],
        dtype=torch.float64,
    )
    assert torch.allclose(grouped.missing, expected_missing)

    # Uniformly duplicate every row of the first segment.  Raw row weighting
    # would change; empirical group means and every group-risk-only objective
    # remain identical.
    duplicate = np.asarray([0, 1, 2])
    expanded_losses = np.concatenate((row_losses, row_losses[:, duplicate]), axis=1)
    expanded_segments = np.concatenate((segments, segments[duplicate]))
    expanded = compute_empirical_group_risks(
        _loss_predictions(expanded_losses),
        torch.zeros((9, 3), dtype=torch.float64),
        expanded_segments,
    )
    assert torch.equal(grouped.full, expanded.full)
    assert torch.equal(grouped.missing, expanded.missing)
    first = compute_controlled_objective(
        method_id="group_mean_pord", full_risks=grouped.full, missing_risks=grouped.missing
    )
    second = compute_controlled_objective(
        method_id="group_mean_pord", full_risks=expanded.full, missing_risks=expanded.missing
    )
    assert torch.equal(first.objective, second.objective)
    paired_first = compute_controlled_objective(
        method_id="paired_pord",
        full_risks=grouped.full,
        missing_risks=grouped.missing,
        row_full_losses=grouped.row_full,
        row_missing_losses=grouped.row_missing,
        segment_index=grouped.segment_index,
    )
    paired_second = compute_controlled_objective(
        method_id="paired_pord",
        full_risks=expanded.full,
        missing_risks=expanded.missing,
        row_full_losses=expanded.row_full,
        row_missing_losses=expanded.row_missing,
        segment_index=expanded.segment_index,
    )
    assert torch.equal(paired_first.positive_excess, paired_second.positive_excess)
    assert torch.equal(paired_first.objective, paired_second.objective)


def test_group_mean_ablation_and_primary_relu_before_mean_are_distinct() -> None:
    # Segment 0, outage 0 has row excesses -1 and +1. The legacy group-mean
    # ablation is zero; the primary row-ReLU-before-mean atom is exactly .5.
    segments = np.asarray([0, 0, 1, 2])
    row_losses = np.ones((5, 4), dtype=np.float64)
    row_losses[1, :2] = [0.0, 2.0]
    grouped = compute_empirical_group_risks(
        _loss_predictions(row_losses),
        torch.zeros((4, 3), dtype=torch.float64),
        segments,
    )
    group_mean = compute_controlled_objective(
        method_id="group_mean_pord",
        full_risks=grouped.full,
        missing_risks=grouped.missing,
    )
    paired = compute_controlled_objective(
        method_id="paired_pord",
        full_risks=grouped.full,
        missing_risks=grouped.missing,
        row_full_losses=grouped.row_full,
        row_missing_losses=grouped.row_missing,
        segment_index=grouped.segment_index,
    )
    assert group_mean.positive_excess[0, 0].item() == 0.0
    assert paired.positive_excess[0, 0].item() == pytest.approx(0.5)
    assert paired.row_pair_excess[:2, 0].tolist() == pytest.approx([-1.0, 1.0])

    full, missing = _risks(
        [1.0, 2.0, 3.0],
        [[9, 8, 7, 6], [2, 2, 2, 2], [3, 3, 3, 3]],
    )
    pord = compute_controlled_objective(
        method_id="group_mean_pord", full_risks=full, missing_risks=missing
    )
    full_gradient, _ = direct_group_risk_gradients(pord)
    # Tail and interaction see stop_gradient(R_full); only the direct mask
    # coefficient reaches full risk.
    assert torch.equal(
        full_gradient,
        torch.full((3,), PORD_FULL_GROUP_WEIGHT, dtype=torch.float64),
    )


def test_wrong_pair_preserves_marginals_and_no_stopgrad_changes_only_gradient() -> None:
    uids = ["b0", "a0", "b1", "a1", "b2", "a2"]
    segments = torch.tensor([0, 0, 1, 1, 2, 2])
    pairing = fixed_within_segment_circular_derangement(
        trial_uids=uids, segment_index=segments.tolist()
    )
    assert sorted(pairing.indices.tolist()) == list(range(6))
    assert np.all(pairing.indices != np.arange(6))
    assert np.array_equal(
        np.asarray(segments)[pairing.indices], np.asarray(segments)
    )
    replay = fixed_within_segment_circular_derangement(
        trial_uids=uids, segment_index=segments.tolist()
    )
    assert pairing.sha256 == replay.sha256
    assert np.array_equal(pairing.indices, replay.indices)

    row_full = torch.tensor(
        [1, 9, 2, 8, 3, 7], dtype=torch.float64, requires_grad=True
    )
    row_missing = torch.tensor(
        [[2, 1, 1, 1], [8, 9, 9, 9], [3, 2, 2, 2],
         [7, 8, 8, 8], [4, 3, 3, 3], [6, 7, 7, 7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    full, missing = _risks([5, 5, 5], np.full((3, 4), 5.0))
    common = dict(
        full_risks=full,
        missing_risks=missing,
        row_full_losses=row_full,
        row_missing_losses=row_missing,
        segment_index=segments,
    )
    correct = compute_controlled_objective(method_id="paired_pord", **common)
    no_stop = compute_controlled_objective(method_id="paired_no_stopgrad", **common)
    wrong = compute_controlled_objective(
        method_id="paired_wrong_pair",
        full_reference_index=pairing.indices,
        pairing_map_sha256=pairing.sha256,
        **common,
    )
    assert torch.equal(correct.objective, no_stop.objective)
    assert not torch.equal(correct.positive_excess, wrong.positive_excess)
    correct_row_gradient = direct_row_loss_gradients(correct)
    no_stop_row_gradient = direct_row_loss_gradients(no_stop)
    assert correct_row_gradient is not None and no_stop_row_gradient is not None
    assert not torch.equal(correct_row_gradient[0], no_stop_row_gradient[0])
    assert correct.pairing_mode == "correct_uid_stop_gradient"
    assert wrong.pairing_map_sha256 == pairing.sha256

    atoms, row_excess, reference = compute_row_paired_atoms(
        row_full_losses=row_full,
        row_missing_losses=row_missing,
        segment_index=segments,
        full_risks=full,
    )
    assert torch.equal(atoms, correct.positive_excess)
    assert torch.equal(row_excess, correct.row_pair_excess)
    assert not reference.requires_grad


def test_fixed_top3_values_symmetric_cutoff_ties_and_permutation_gradients() -> None:
    values = torch.tensor(
        [[5.0, 4.0, 3.0, 3.0], [3.0, 0.0, -1.0, -2.0], [0, 0, 0, 0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    penalty, weights, cutoff = fixed_symmetric_positive_topk(values)
    assert penalty.item() == pytest.approx((5.0 + 4.0 + 3.0) / 3.0)
    assert cutoff == 3.0
    assert weights[0, 0].item() == pytest.approx(1 / 3)
    assert weights[0, 1].item() == pytest.approx(1 / 3)
    # Three cutoff ties share one remaining slot symmetrically.
    assert torch.equal(
        weights[values.detach() == 3.0],
        torch.full((3,), 1 / 9, dtype=torch.float64),
    )
    penalty.backward()
    assert torch.equal(values.grad, weights)

    permutation = torch.tensor([4, 2, 9, 0, 8, 7, 3, 6, 1, 11, 10, 5])
    permuted_leaf = values.detach().reshape(-1)[permutation].clone().requires_grad_()
    permuted_penalty, permuted_weights, _ = fixed_symmetric_positive_topk(
        permuted_leaf.reshape(3, 4)
    )
    permuted_penalty.backward()
    restored = torch.empty_like(permuted_leaf.grad)
    restored[permutation] = permuted_leaf.grad
    assert torch.equal(restored.reshape(3, 4), weights)
    assert torch.equal(permuted_leaf.grad.reshape(3, 4), permuted_weights)


def test_top3_relu_and_interaction_abs_have_zero_subgradients_at_zero() -> None:
    zeros = torch.zeros((3, 4), dtype=torch.float64, requires_grad=True)
    penalty, weights, cutoff = fixed_symmetric_positive_topk(zeros)
    penalty.backward()
    assert penalty.item() == 0.0
    assert cutoff == 0.0
    assert torch.equal(weights, torch.zeros_like(weights))
    assert torch.equal(zeros.grad, torch.zeros_like(zeros))

    full, missing = _risks([1, 1, 1], np.ones((3, 4)))
    interaction_only = compute_controlled_objective(
        method_id="group_mean_pord", full_risks=full, missing_risks=missing
    )
    full_gradient, missing_gradient = direct_group_risk_gradients(interaction_only)
    assert torch.equal(
        full_gradient,
        torch.full((3,), PORD_FULL_GROUP_WEIGHT, dtype=torch.float64),
    )
    assert torch.equal(
        missing_gradient,
        torch.full((3, 4), PORD_MISSING_GROUP_WEIGHT, dtype=torch.float64),
    )


def test_nominal_erm_coefficients_and_interaction_algebra_and_spike() -> None:
    full, missing = _risks(
        [2, 5, 9],
        [[1, 4, 8, 2], [3, 7, 6, 5], [9, 2, 4, 8]],
    )
    erm = compute_controlled_objective(
        method_id="masked_erm", full_risks=full, missing_risks=missing
    )
    full_gradient, missing_gradient = direct_group_risk_gradients(erm)
    assert torch.equal(
        full_gradient,
        torch.full((3,), 1 / 12, dtype=torch.float64),
    )
    assert torch.equal(
        missing_gradient,
        torch.full((3, 4), 1 / 16, dtype=torch.float64),
    )

    interaction_from_excess = two_way_center(
        missing - full.detach().unsqueeze(1)
    )
    interaction_from_missing = two_way_center(missing)
    assert torch.allclose(interaction_from_excess, interaction_from_missing, atol=1e-15)
    assert torch.allclose(
        interaction_from_excess.sum(dim=0),
        torch.zeros(4, dtype=torch.float64),
        atol=1e-15,
    )
    assert torch.allclose(
        interaction_from_excess.sum(dim=1),
        torch.zeros(3, dtype=torch.float64),
        atol=1e-15,
    )

    c = 12.0
    spike = torch.zeros((3, 4), dtype=torch.float64)
    spike[1, 2] = c
    assert torch.abs(two_way_center(spike)).mean().item() == pytest.approx(abs(c) / 6)


def test_all_method_constants_directions_and_groupdro_prior() -> None:
    assert PORD_METHOD_IDS == (
        "full_gradient_only_mask_selected_control",
        "masked_erm",
        "constant_availability_erm",
        "group_mean_pord",
        "paired_tail",
        "paired_interaction",
        "paired_pord",
        "paired_wrong_pair",
        "paired_no_stopgrad",
        "groupdro",
        "vrex",
        "mofe",
    )
    assert PORD_TOP_K == 3
    assert PORD_GROUPDRO_ETA == 0.01
    assert PORD_VREX_BETA == 1.0
    assert PORD_MOFE_LAMBDA == 0.1
    assert PORD_LEARNING_RATE == 1e-3
    assert PORD_WEIGHT_DECAY == 1e-3
    assert PORD_EPOCH_BUDGET == 400

    full, missing = _risks(
        [1, 2, 3],
        [[4, 1, 2, 7], [2, 8, 2, 3], [9, 3, 4, 3]],
    )
    row_full = torch.tensor([3.0, 1.0, 2.0, 4.0, 1.5, 2.5], dtype=torch.float64)
    row_missing = torch.tensor(
        [
            [1, 4, 3, 3], [2, 1, 2, 1],
            [1, 1, 1, 1], [3, 5, 2, 6],
            [4, 1, 4, 1], [1, 3, 1, 3],
        ],
        dtype=torch.float64,
    )
    segments = torch.tensor([0, 0, 1, 1, 2, 2])
    wrong = fixed_within_segment_circular_derangement(
        trial_uids=[f"u{i}" for i in range(6)], segment_index=segments.tolist()
    )
    objectives = {}
    for method in PORD_METHOD_IDS:
        objectives[method] = compute_controlled_objective(
            method_id=method,
            full_risks=full,
            missing_risks=missing,
            row_full_losses=row_full,
            row_missing_losses=row_missing,
            segment_index=segments,
            full_reference_index=(wrong.indices if method == "paired_wrong_pair" else None),
            pairing_map_sha256=(wrong.sha256 if method == "paired_wrong_pair" else None),
        )
    erm_value = objectives["masked_erm"].objective.item()
    assert objectives["paired_tail"].objective.item() > erm_value
    assert objectives["paired_interaction"].objective.item() > erm_value
    assert objectives["paired_pord"].objective.item() == pytest.approx(
        erm_value
        + objectives["paired_pord"].tail_penalty.item()
        + objectives["paired_pord"].interaction_penalty.item()
    )
    assert objectives["paired_pord"].objective.item() == pytest.approx(
        objectives["paired_no_stopgrad"].objective.item()
    )
    assert objectives["full_gradient_only_mask_selected_control"].objective.item() == pytest.approx(
        full.mean().item()
    )
    full_only_gradients = direct_group_risk_gradients(
        objectives["full_gradient_only_mask_selected_control"]
    )
    assert torch.equal(
        full_only_gradients[0], torch.full((3,), 1 / 3, dtype=torch.float64)
    )
    assert torch.equal(full_only_gradients[1], torch.zeros((3, 4), dtype=torch.float64))
    assert objectives["vrex"].objective.item() == pytest.approx(
        erm_value + objectives["vrex"].vrex_penalty.item()
    )
    assert objectives["mofe"].objective.item() == pytest.approx(
        erm_value + 0.1 * objectives["mofe"].mofe_penalty.item()
    )
    groupdro = objectives["groupdro"]
    assert groupdro.groupdro_weights_before is not None
    assert torch.allclose(
        groupdro.groupdro_weights_before[:3],
        torch.full((3,), 1 / 12, dtype=torch.float64),
    )
    assert torch.allclose(
        groupdro.groupdro_weights_before[3:],
        torch.full((12,), 1 / 16, dtype=torch.float64),
    )
    assert groupdro.next_groupdro_weights[torch.argmax(torch.cat((full, missing.flatten())))] > min(
        groupdro.next_groupdro_weights
    )


def test_mofe_is_per_row_hinge_and_segment_duplication_invariant() -> None:
    full, missing = _risks([1, 1, 1], np.ones((3, 4)))
    row_full = torch.tensor([3.0, 1.0, 2.0, 4.0], dtype=torch.float64)
    row_missing = torch.tensor(
        [[1, 4, 3, 2], [2, 0, 1, 1], [1, 3, 2, 4], [2, 5, 4, 1]],
        dtype=torch.float64,
    )
    segments = torch.tensor([0, 0, 1, 2])
    first = compute_controlled_objective(
        method_id="mofe",
        full_risks=full,
        missing_risks=missing,
        row_full_losses=row_full,
        row_missing_losses=row_missing,
        segment_index=segments,
    )
    manual = torch.stack(
        [
            torch.relu(
                row_full[segments == group].unsqueeze(1)
                - row_missing[segments == group]
            ).mean(0)
            for group in range(3)
        ]
    ).mean()
    assert torch.equal(first.mofe_penalty, manual)
    # Uniform duplication of both segment-0 rows cannot change its group mean.
    duplicated = compute_controlled_objective(
        method_id="mofe",
        full_risks=full,
        missing_risks=missing,
        row_full_losses=torch.cat((row_full, row_full[:2])),
        row_missing_losses=torch.cat((row_missing, row_missing[:2])),
        segment_index=torch.cat((segments, segments[:2])),
    )
    assert torch.equal(first.mofe_penalty, duplicated.mofe_penalty)


def test_objective_is_backbone_independent_and_factories_are_exact() -> None:
    assert PORD_BACKBONE_IDS == (
        PORD_MASKED_MLP_BACKBONE_ID,
        PORD_CHAIN_TRANSFER_BACKBONE_ID,
    )
    assert PORD_PRIMARY_BACKBONE_ID == PORD_MASKED_MLP_BACKBONE_ID
    assert PORD_TRANSFER_METHOD_IDS == ("masked_erm", "paired_pord")
    chain = build_pord_backbone(torch, backbone_id=PORD_CHAIN_BACKBONE_ID)
    masked = build_pord_backbone(
        torch, backbone_id=PORD_MASKED_MLP_BACKBONE_ID
    )
    assert len(chain.rounds) == 4
    assert chain.imu_stem.out_features == 96
    assert chain.rounds[0].forward_message[0].out_features == 192
    assert chain.readout[-1].out_features == 3
    assert masked.layers[0].in_features == 47
    assert masked.layers[0].out_features == 64
    assert masked.layers[2].out_features == 64
    assert masked.layers[4].out_features == 3
    assert PORD_BACKBONE_SETTINGS_BY_ID[PORD_CHAIN_BACKBONE_ID]["dropout"] == 0.0

    full, missing = _risks([1, 2, 3], np.arange(12).reshape(3, 4) + 2)
    # The pure objective receives no model/backbone argument; its exact result
    # is transferable without changing any PORD mathematics.
    row_full = torch.tensor([1, 2, 3], dtype=torch.float64)
    row_missing = torch.tensor([[3, 2, 4, 1], [4, 5, 1, 3], [5, 1, 2, 4]], dtype=torch.float64)
    kwargs = dict(
        method_id="paired_pord", full_risks=full, missing_risks=missing,
        row_full_losses=row_full, row_missing_losses=row_missing,
        segment_index=torch.tensor([0, 1, 2]),
    )
    first = compute_controlled_objective(**kwargs)
    second = compute_controlled_objective(**kwargs)
    assert torch.equal(first.objective, second.objective)


def test_gradient_audit_contains_exact_risks_tail_interaction_and_gradients() -> None:
    full, missing = _risks(
        [1, 2, 3],
        [[7, 6, 5, 1], [2, 3, 4, 5], [9, 2, 4, 6]],
    )
    objective = compute_controlled_objective(
        method_id="group_mean_pord", full_risks=full, missing_risks=missing
    )
    gradients = direct_group_risk_gradients(objective)
    audit = objective_audit_record(
        objective, full_gradient=gradients[0], missing_gradient=gradients[1]
    )
    assert audit["method_id"] == "group_mean_pord"
    assert audit["tail_fixed_k"] == 3
    assert audit["group_mean_interaction_missing_nonadditivity_max_abs_delta"] < 1e-12
    assert audit["direct_full_group_gradient"] == pytest.approx([1 / 12] * 3)
    assert sum(cell["weight"] for cell in audit["tail_active_cells"]) == pytest.approx(1.0)
    assert len(audit["audit_sha256"]) == 64


def test_chain_transfer_pair_has_same_init_views_updates_and_checkpoint_policy() -> None:
    problem = _tiny_problem()
    erm = train_pord_fold_seed(
        **problem,
        method_id="masked_erm",
        backbone_id=PORD_CHAIN_TRANSFER_BACKBONE_ID,
    )
    pord = train_pord_fold_seed(
        **problem,
        method_id="paired_pord",
        backbone_id=PORD_CHAIN_TRANSFER_BACKBONE_ID,
    )
    paired = verify_transfer_method_controls(
        [erm, pord],
        require_complete_epoch_budget=False,
    )
    assert paired["table_kind"] == "chain_transfer_masked_erm_paired_pord_only"
    assert erm["initial_state_sha256"] == pord["initial_state_sha256"]
    assert erm["training_bundle_sha256"] == pord["training_bundle_sha256"]
    assert erm["optimizer_update_count"] == pord["optimizer_update_count"] == 1
    assert erm["best_epoch"] == pord["best_epoch"] == 1
    assert erm["checkpoint_evaluated_view_names"] == [
        "full",
        "drop_imu0",
        "drop_imu1",
        "drop_imu2",
        "drop_imu3",
    ]
    assert len(erm["best_selection_outage_rmse_3d_mm"]) == 4
    assert erm["best_checkpoint_score"] == pytest.approx(
        outage_checkpoint_score(erm["best_selection_outage_rmse_3d_mm"])
    )
    assert erm["checkpoint_selection_scope"] == PORD_CHECKPOINT_SELECTION_SCOPE
    assert erm["checkpoint_criterion"] == PORD_CHECKPOINT_CRITERION
    assert erm["outer_eval_count"] == 0
    assert erm["outer_labels_accessed"] is False
    assert paired["initial_state_sha256"] == erm["initial_state_sha256"]
    assert not erm["contract_epoch_budget_complete"]
    for result in (erm, pord):
        absent = result["history"][0]["parameter_gradient_absent_names"]
        assert absent
        assert all(name.startswith("reconstruction_head.") for name in absent)
        assert not result["history"][0]["auxiliary_reconstruction_used"]
    verify_pord_result_hash(erm)
    verify_pord_result_hash(pord)

    tampered = dict(erm)
    tampered["best_epoch"] = 2
    with pytest.raises(PORDV5TrainingError, match="changed"):
        verify_pord_result_hash(tampered)

    with pytest.raises(PORDV5TrainingError, match="exactly masked-ERM"):
        verify_transfer_method_controls(
            [erm], require_complete_epoch_budget=False
        )
    primary_pord = train_pord_fold_seed(
        **problem,
        method_id="paired_pord",
        backbone_id=PORD_PRIMARY_BACKBONE_ID,
    )
    with pytest.raises(PORDV5TrainingError, match="chain_endpoint_only"):
        verify_transfer_method_controls(
            [erm, primary_pord], require_complete_epoch_budget=False
        )


def test_primary_masked_table_pairs_all_twelve_controlled_methods_exactly() -> None:
    results = [
        train_pord_fold_seed(
            **_tiny_problem(seed=117),
            method_id=method_id,
            backbone_id=PORD_PRIMARY_BACKBONE_ID,
        )
        for method_id in PORD_METHOD_IDS
    ]
    paired = verify_paired_method_controls(
        results,
        require_all_methods=True,
        require_complete_epoch_budget=False,
    )
    assert paired["method_ids"] == sorted(PORD_METHOD_IDS)
    assert len({result["initial_state_sha256"] for result in results}) == 1
    assert len({result["training_bundle_sha256"] for result in results}) == 1
    for method_id, result in zip(PORD_METHOD_IDS, results, strict=True):
        assert result["method_id"] == method_id
        assert result["backbone_id"] == PORD_PRIMARY_BACKBONE_ID
        assert result["optimizer_update_count"] == 1
        assert len(result["best_selection_outage_rmse_3d_mm"]) == 4
        assert result["same_checkpoint_for_all_four_outages"]
        audit = result["history"][0]
        assert not audit["auxiliary_reconstruction_used"]
        assert audit["parameter_gradient_absent_names"] == []
        if method_id == "constant_availability_erm":
            assert result["availability_semantics"] == "constant_all_available_negative_control"
            assert result["train_view_bundle_sha256"] != result["base_train_view_bundle_sha256"]
        else:
            assert result["availability_semantics"] == "physical_mask_state"
            assert result["train_view_bundle_sha256"] == result["base_train_view_bundle_sha256"]
        verify_pord_result_hash(result)
    constant_result = next(
        row for row in results if row["method_id"] == "constant_availability_erm"
    )
    constant_outer = evaluate_pord_checkpoint_once(
        training_result=constant_result, **_tiny_outer_problem()
    )
    assert (
        constant_outer["receipt"]["availability_semantics"]
        == "constant_all_available_negative_control"
    )
    assert (
        constant_outer["receipt"]["outer_view_bundle_sha256"]
        != constant_outer["receipt"]["base_outer_view_bundle_sha256"]
    )


def test_outer_arrays_are_absent_from_trainer_and_evaluated_once_after_checkpoint() -> None:
    core = train_pord_fold_seed(
        **_tiny_problem(),
        method_id="masked_erm",
        backbone_id=PORD_PRIMARY_BACKBONE_ID,
    )
    assert core["checkpoint_selection_scope"] == "training_segments_only"
    assert core["outer_eval_count"] == 0
    assert all(
        row["checkpoint_selection_scope"] == "training_segments_only"
        and row["outer_eval_count"] == 0
        and row["outer_labels_accessed"] is False
        for row in core["history"]
    )

    evaluated = evaluate_pord_checkpoint_once(
        training_result=core,
        **_tiny_outer_problem(),
    )
    receipt = evaluated["receipt"]
    assert receipt["checkpoint_fixed_before_outer_labels"] is True
    assert receipt["outer_eval_count"] == 1
    assert receipt["checkpoint_selection_scope"] == "training_segments_only"
    verify_pord_outer_evaluation(
        receipt,
        training_result=core,
        outer_trial_uids=_tiny_outer_problem()["outer_trial_uids"],
        truth_raw_mm=evaluated["truth_raw_mm"],
        predictions_raw_mm=evaluated["predictions_raw_mm"],
    )

    for invalid_count in (0, 2):
        attacked = dict(receipt)
        attacked["outer_eval_count"] = invalid_count
        with pytest.raises(PORDV5TrainingError, match="binding changed"):
            verify_pord_outer_evaluation(
                attacked,
                training_result=core,
                outer_trial_uids=_tiny_outer_problem()["outer_trial_uids"],
                truth_raw_mm=evaluated["truth_raw_mm"],
                predictions_raw_mm=evaluated["predictions_raw_mm"],
            )

    old_api = _tiny_problem()
    old_api["X_holdout_scaled"] = np.zeros((1, 43), dtype=np.float32)
    with pytest.raises(TypeError, match="X_holdout_scaled"):
        train_pord_fold_seed(
            **old_api,
            method_id="masked_erm",
            backbone_id=PORD_PRIMARY_BACKBONE_ID,
        )


def test_initial_state_pairing_is_exact_for_chain_and_transfer_backbones() -> None:
    hashes: dict[str, str] = {}
    for backbone_id in PORD_BACKBONE_IDS:
        torch.manual_seed(1729)
        first = build_pord_backbone(torch, backbone_id=backbone_id)
        torch.manual_seed(1729)
        second = build_pord_backbone(torch, backbone_id=backbone_id)
        first_hash = state_dict_sha256(first.state_dict())
        second_hash = state_dict_sha256(second.state_dict())
        assert first_hash == second_hash
        hashes[backbone_id] = first_hash
    assert hashes[PORD_CHAIN_BACKBONE_ID] != hashes[PORD_MASKED_MLP_BACKBONE_ID]


def test_checkpoint_earliest_rule_final_epoch_half_up_and_full_sensor_gate() -> None:
    history = [
        {"epoch": 1, "checkpoint_score": 4.0},
        {"epoch": 2, "checkpoint_score": 3.0},
        {"epoch": 3, "checkpoint_score": 3.0},
        {"epoch": 4, "checkpoint_score": 3.5},
    ]
    assert earliest_checkpoint_epoch(history) == 2
    assert outage_checkpoint_score([1, 2, 3, 4]) == pytest.approx(3.25)
    assert half_up_median_epoch([10, 11, 20, 21]) == 16
    assert half_up_median_epoch([10, 11, 20, 22]) == 16
    records = [
        {
            "method_id": "paired_pord",
            "model_seed": 7,
            "held_out_segment": fold,
            "best_epoch": epoch,
        }
        for fold, epoch in zip((36, 37, 38, 39), (10, 11, 20, 22), strict=True)
    ]
    selected = final_epochs_by_method_seed(records)
    assert selected[0]["final_epoch"] == 16
    assert selected[0]["rounding"] == "median_half_up"

    passing = full_sensor_numeric_gate(
        candidate_fold_rmse_3d_mm=[10, 10, 10, 10],
        baseline_fold_rmse_3d_mm=[10, 10, 10, 10],
        baseline_name="frozen_v3_same_backbone",
    )
    failing = full_sensor_numeric_gate(
        candidate_fold_rmse_3d_mm=[10, 10, 10, 10 + 3e-9],
        baseline_fold_rmse_3d_mm=[10, 10, 10, 10],
        baseline_name="same_v5_backbone_erm",
        absolute_numeric_tolerance_mm=0.0,
    )
    assert passing["passed"]
    assert not failing["passed"]
    assert passing["baseline_name"] == "frozen_v3_same_backbone"
    assert failing["absolute_numeric_tolerance_mm"] == 0.0
    assert passing["interpretation"] == "numeric_only_not_statistical_noninferiority"


def test_target_transform_split_and_fail_closed_inputs() -> None:
    scaled = np.asarray([[0, 1, -1], [2, -2, 0.5]], dtype=np.float32)
    raw = inverse_standardized_targets(
        scaled,
        target_mean_mm=[100, 200, 300],
        target_scale_mm=[10, 10, 10],
    )
    assert np.array_equal(
        raw, np.asarray([[100, 210, 290], [120, 180, 305]], dtype=np.float64)
    )
    with pytest.raises(PORDV5TrainingError, match="non-finite"):
        build_exact_five_views(
            np.full((2, 43), np.nan, dtype=np.float32), trial_uids=["a", "b"]
        )
    with pytest.raises(PORDV5TrainingError, match="unique"):
        build_exact_five_views(
            np.zeros((2, 43), dtype=np.float32), trial_uids=["same", "same"]
        )
    full, missing = _risks([1, 2, 3], np.ones((3, 4)))
    bad_missing = missing.detach().clone()
    bad_missing[0, 0] = float("nan")
    with pytest.raises(PORDV5TrainingError, match="non-finite"):
        compute_controlled_objective(
            method_id="group_mean_pord", full_risks=full, missing_risks=bad_missing
        )

    missing_segment = _tiny_problem()
    missing_segment["train_segment_ids"] = [36] * 8 + [37]
    with pytest.raises(PORDV5TrainingError, match="exactly three"):
        train_pord_fold_seed(
            **missing_segment,
            method_id="masked_erm",
            backbone_id=PORD_PRIMARY_BACKBONE_ID,
        )
    bad_transform = _tiny_problem()
    bad_transform["target_scale_mm"] = [1.0, 0.0, 1.0]
    with pytest.raises(PORDV5TrainingError, match="strictly positive"):
        train_pord_fold_seed(
            **bad_transform,
            method_id="masked_erm",
            backbone_id=PORD_PRIMARY_BACKBONE_ID,
        )
    anisotropic_transform = _tiny_problem()
    anisotropic_transform["target_scale_mm"] = [1.0, 2.0, 1.0]
    with pytest.raises(PORDV5TrainingError, match="isotropic"):
        train_pord_fold_seed(
            **anisotropic_transform,
            method_id="masked_erm",
            backbone_id=PORD_PRIMARY_BACKBONE_ID,
        )
    with pytest.raises(PORDV5TrainingError, match="pre-registered only"):
        train_pord_fold_seed(
            **_tiny_problem(),
            method_id="vrex",
            backbone_id=PORD_CHAIN_TRANSFER_BACKBONE_ID,
        )
