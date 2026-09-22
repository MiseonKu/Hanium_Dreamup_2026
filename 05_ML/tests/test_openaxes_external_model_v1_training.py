from __future__ import annotations

import inspect

import numpy as np
import pytest

from imu_vive_learning.openaxes_external_model_v1_training import (
    MASK_LAW,
    OpenAxesExternalModelV1TrainingError,
    build_exact_five_views,
    build_masked_mlp,
    compact_group_codes,
    compute_method_objective,
    evaluate_outer_predictions_once,
    fit_train_only_transform,
    inverse_targets,
    predict_outer_views_once,
    state_dict_sha256,
    symmetric_top_quarter,
    train_model_train_only,
    training_checkpoint_score_m,
    transform_features,
    transform_targets,
    two_way_center,
    update_groupdro_weights,
    validate_exact_five_views,
    within_training_group_derangement,
)


def _torch():
    return pytest.importorskip("torch")


def _arrays(rows: int = 12) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(1024)
    X = rng.normal(size=(rows, 36)).astype(np.float64)
    y = rng.normal(size=(rows, 3)).astype(np.float64) * 0.02 + np.array([0.1, 0.2, 0.3])
    groups = ["circle-100" if index < rows // 2 else "triangle-100" for index in range(rows)]
    uids = [f"uid-{index:03d}" for index in range(rows)]
    return X, y, groups, uids


def test_train_only_population_transform_is_isotropic_and_exactly_invertible() -> None:
    X, y, _groups, uids = _arrays()
    transform = fit_train_only_transform(X, y, uids)
    assert transform.feature_mean == pytest.approx(np.mean(X, axis=0))
    assert transform.feature_scale == pytest.approx(np.std(X, axis=0, ddof=0))
    assert transform.target_mean_m == pytest.approx(np.mean(y, axis=0))
    expected_scale = np.sqrt(np.mean(np.square(y - np.mean(y, axis=0))))
    assert transform.target_scale_m == pytest.approx(expected_scale)
    assert np.array_equal(inverse_targets(transform_targets(y, transform), transform), y)
    assert transform.receipt()["outer_rows_used"] == 0
    assert transform.receipt()["target_scale_isotropic"] is True


def test_transform_rejects_zero_variance_nonfinite_and_duplicate_uid() -> None:
    X, y, _groups, uids = _arrays()
    X[:, 4] = 1.0
    with pytest.raises(OpenAxesExternalModelV1TrainingError, match="zero"):
        fit_train_only_transform(X, y, uids)
    X, y, _groups, uids = _arrays()
    X[0, 0] = np.nan
    with pytest.raises(OpenAxesExternalModelV1TrainingError, match="finite"):
        fit_train_only_transform(X, y, uids)
    X, y, _groups, uids = _arrays()
    uids[-1] = uids[0]
    with pytest.raises(OpenAxesExternalModelV1TrainingError, match="unique"):
        fit_train_only_transform(X, y, uids)


def test_exact_40d_full_plus_four_mask_views() -> None:
    X = np.arange(72, dtype=np.float64).reshape(2, 36) / 10.0
    views = build_exact_five_views(X)
    assert views.shape == (2, 5, 40)
    assert np.array_equal(views[:, 0, :36], X)
    assert np.array_equal(views[:, 0, 36:], np.ones((2, 4)))
    for outage in range(4):
        view = outage + 1
        assert np.array_equal(views[:, view, outage * 9 : (outage + 1) * 9], np.zeros((2, 9)))
        assert np.array_equal(views[:, view, 36 + outage], np.zeros(2))
        keep = [index for index in range(4) if index != outage]
        assert np.array_equal(views[:, view, 36 + np.asarray(keep)], np.ones((2, 3)))
    validate_exact_five_views(X, views)
    attacked = views.copy()
    attacked[0, 2, 36] = 0.0
    with pytest.raises(OpenAxesExternalModelV1TrainingError, match="changed"):
        validate_exact_five_views(X, attacked)


def test_within_group_wrong_pair_is_fixed_point_free_and_marginal_preserving() -> None:
    uids = ["b", "a", "d", "c", "f", "e"]
    groups = ["g0", "g0", "g0", "g1", "g1", "g1"]
    mapping = within_training_group_derangement(uids, groups)
    assert np.all(mapping != np.arange(6))
    for group in ("g0", "g1"):
        indices = {index for index, value in enumerate(groups) if value == group}
        assert set(mapping[list(indices)]) == indices
    assert np.array_equal(mapping, within_training_group_derangement(uids, groups))
    with pytest.raises(OpenAxesExternalModelV1TrainingError, match="cannot be deranged"):
        within_training_group_derangement(["a", "b", "c"], ["g0", "g0", "g1"])


def test_symmetric_top_quarter_has_fixed_value_and_tie_gradient() -> None:
    torch = _torch()
    values = torch.tensor([5.0, 4.0, 4.0, 4.0, 1.0, 0.0, -1.0, -2.0], requires_grad=True)
    tail = symmetric_top_quarter(values)
    assert float(tail.detach()) == pytest.approx(4.5)
    tail.backward()
    # k=2: strict top gets 1/2; the remaining half is shared across 3 cutoff ties.
    assert values.grad.tolist() == pytest.approx([0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0, 0, 0, 0])
    permutation = torch.tensor([3, 0, 7, 2, 1, 6, 5, 4])
    permuted = values.detach()[permutation].clone().requires_grad_(True)
    symmetric_top_quarter(permuted).backward()
    restored = torch.empty_like(permuted.grad)
    restored[permutation] = permuted.grad
    assert restored.tolist() == pytest.approx(values.grad.tolist())


def test_two_way_interaction_is_double_centered_and_spike_algebra() -> None:
    torch = _torch()
    atoms = torch.zeros((3, 4), dtype=torch.float64)
    atoms[0, 0] = 6.0
    interaction = two_way_center(atoms)
    assert interaction.sum(dim=0).tolist() == pytest.approx([0.0] * 4)
    assert interaction.sum(dim=1).tolist() == pytest.approx([0.0] * 3)
    # For a single c spike in G x M, mean absolute interaction is
    # 4*c*(G-1)*(M-1)/(G*M)^2 = 1.0 here.
    assert float(torch.abs(interaction).mean()) == pytest.approx(1.0)


def _loss_tensor():
    torch = _torch()
    # Two groups, two rows each. Missing marginals are fixed; full pairing differs.
    return torch.tensor(
        [
            [0.0, 4.0, 1.0, 1.0, 1.0],
            [4.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 3.0, 1.0, 1.0, 1.0],
            [3.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )


def test_group_mean_and_row_level_pairing_apply_relu_in_different_order() -> None:
    torch = _torch()
    losses = _loss_tensor()
    groups = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    paired = compute_method_objective(losses, groups, "paired_pord")
    group_mean = compute_method_objective(losses, groups, "group_mean_pord")
    # First outage group means cancel, but row-level positive gaps remain.
    assert group_mean["positive_atoms"][:, 0].tolist() == pytest.approx([0.0, 0.0])
    assert paired["positive_atoms"][:, 0].tolist() == pytest.approx([2.0, 1.5])
    assert float(paired["loss"].detach()) > float(group_mean["loss"].detach())


def test_paired_wrong_pair_breaks_true_uid_pairing_while_preserving_marginals() -> None:
    torch = _torch()
    losses = _loss_tensor()
    groups = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    wrong = torch.tensor([1, 0, 3, 2], dtype=torch.long)
    true = compute_method_objective(losses, groups, "paired_pord")
    attacked = compute_method_objective(
        losses, groups, "paired_wrong_pair", wrong_pair_index=wrong
    )
    np.testing.assert_allclose(
        true["group_risks"].detach().numpy(),
        attacked["group_risks"].detach().numpy(),
        rtol=0.0,
        atol=0.0,
    )
    assert true["positive_atoms"][:, 0].tolist() != pytest.approx(
        attacked["positive_atoms"][:, 0].tolist()
    )


def test_paired_full_loss_is_detached_beyond_direct_mask_coefficient() -> None:
    torch = _torch()
    losses = torch.tensor(
        [
            [1.0, 5.0, 6.0, 7.0, 8.0],
            [1.5, 5.5, 6.5, 7.5, 8.5],
            [2.0, 6.0, 7.0, 8.0, 9.0],
            [2.5, 6.5, 7.5, 8.5, 9.5],
        ],
        requires_grad=True,
    )
    groups = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    result = compute_method_objective(losses, groups, "paired_pord")
    result["loss"].backward()
    assert losses.grad[:, 0].tolist() == pytest.approx([MASK_LAW[0] / 4.0] * 4)


def test_full_gradient_control_and_masked_erm_have_frozen_gradient_directions() -> None:
    torch = _torch()
    groups = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    full = torch.ones((4, 5), requires_grad=True)
    compute_method_objective(full, groups, "full_gradient_only_mask_selected_control")["loss"].backward()
    assert full.grad[:, 0].tolist() == pytest.approx([0.25] * 4)
    np.testing.assert_allclose(full.grad[:, 1:].numpy(), np.zeros((4, 4)), rtol=0.0, atol=0.0)
    erm = torch.ones((4, 5), requires_grad=True)
    compute_method_objective(erm, groups, "masked_erm")["loss"].backward()
    assert erm.grad[0].tolist() == pytest.approx([weight / 4.0 for weight in MASK_LAW])


def test_groupdro_vrex_and_mofe_constants_and_directions() -> None:
    torch = _torch()
    groups = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    losses = torch.tensor(
        [[4.0, 1.0, 1.0, 1.0, 1.0]] * 2 + [[1.0, 4.0, 4.0, 4.0, 4.0]] * 2
    )
    dro = compute_method_objective(losses, groups, "groupdro")
    updated = update_groupdro_weights(dro["group_risks"], dro["groupdro_weights"])
    assert float(updated[5:].sum()) > float(dro["groupdro_weights"][5:].sum())
    vrex = compute_method_objective(losses, groups, "vrex")
    assert float(vrex["vrex_variance"]) > 0.0
    mofe = compute_method_objective(losses, groups, "mofe")
    assert float(mofe["mofe_hinge"]) == pytest.approx(1.5)


def test_objective_is_backbone_independent_and_pair_initial_state_is_exact() -> None:
    torch = _torch()
    losses = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 10.0
    groups = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    first = compute_method_objective(losses, groups, "masked_erm")["loss"]
    # Objective accepts losses only and never inspects a model/backbone.
    assert "model" not in inspect.signature(compute_method_objective).parameters
    assert float(first) == pytest.approx(sum(MASK_LAW[i] * float(losses[:, i].mean()) for i in range(5)))
    assert state_dict_sha256(build_masked_mlp(seed=1701).state_dict()) == state_dict_sha256(
        build_masked_mlp(seed=1701).state_dict()
    )


def test_checkpoint_score_uses_only_four_training_outages_and_earliest_rule_input() -> None:
    predictions = np.zeros((2, 5, 3), dtype=np.float64)
    targets = np.zeros((2, 3), dtype=np.float64)
    predictions[:, 0, :] = 999.0
    for view in range(1, 5):
        predictions[:, view, 0] = float(view)
    score, rmse = training_checkpoint_score_m(predictions, targets, 2.0)
    assert rmse == pytest.approx([2.0, 4.0, 6.0, 8.0])
    assert score == pytest.approx(0.5 * (5.0 + 8.0))


def test_trainer_signature_cannot_receive_outer_data_and_cpu_smoke_is_separate() -> None:
    parameters = set(inspect.signature(train_model_train_only).parameters)
    assert not any("outer" in name or "holdout" in name or "test" in name for name in parameters)
    X, y, groups, uids = _arrays(8)
    result = train_model_train_only(
        X,
        y,
        groups,
        uids,
        method_id="masked_erm",
        seed=1701,
        updates=2,
        device="cpu",
    )
    assert result.updates == 2
    assert len(result.history) == 2
    assert result.checkpoint_selection_scope == "training_scenarios_only"
    assert result.outer_eval_count == 0
    prediction = predict_outer_views_once(result, X[:2], device="cpu")
    metrics = evaluate_outer_predictions_once(prediction, y[:2])
    assert prediction.shape == (2, 5, 3)
    assert metrics["outer_eval_count"] == 1
    assert len(metrics["views"]) == 5


def test_compact_groups_and_objective_fail_closed_on_nonfinite_or_missing_control() -> None:
    torch = _torch()
    codes, names = compact_group_codes(["b", "a", "b", "a"])
    assert names == ("a", "b")
    assert codes.tolist() == [1, 0, 1, 0]
    losses = torch.ones((4, 5))
    groups = torch.tensor(codes, dtype=torch.long)
    with pytest.raises(OpenAxesExternalModelV1TrainingError, match="mapping"):
        compute_method_objective(losses, groups, "paired_wrong_pair")
    losses[0, 0] = float("nan")
    with pytest.raises(OpenAxesExternalModelV1TrainingError, match="finite"):
        compute_method_objective(losses, groups, "masked_erm")
