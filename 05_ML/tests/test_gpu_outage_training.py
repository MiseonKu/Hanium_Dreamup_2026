from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

from imu_vive_learning.experiment_metrics import regression_metrics
from imu_vive_learning.experiment_preprocessing import TrainOnlyPreprocessor
from imu_vive_learning.features import FUSION_COLUMNS
from imu_vive_learning.gpu_outage_training import (
    MASK_FULL_STATE,
    OutageTrainingError,
    apply_sensor_masks,
    balanced_member_mse,
    deliberate_missing_reconstruction_mse,
    deterministic_mask_states,
    fit_outage_seed_fixed_epochs,
    predict_outage_state,
    train_outage_seed,
    transform_outage_features,
)
from imu_vive_learning.gpu_v3_backbones import IMU_SENSOR_INDEX_MAP


torch = pytest.importorskip("torch")


def _config(
    *,
    epochs: int = 3,
    batch_size: int = 24,
    paired_seed_values: tuple[int, int, int] = (913, 914, 915),
) -> dict:
    return {
        "contract": {"base_seed": 20260824},
        "training": {
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "batch_size": batch_size,
            "epoch_budget": epochs,
            "ensemble_seed_count": 3,
            "paired_seed_values": list(paired_seed_values),
            "dtype": "float32",
            "deterministic_algorithms": True,
        },
        "final_refit": {
            "mask_draw_held_out_segment_token": "final_refit_all_development",
            "mask_draw_forced_outage_sensor": "imu1_link_root",
            "mask_draw_reuses_training_key_recipe": True,
            "exact_rounded_median_epoch_final_state": True,
        },
        "runtime": {"cublas_workspace_config": ":4096:8"},
    }


def _candidate(candidate_id: str) -> dict:
    candidates = {
        "masked_mlp": {
            "candidate_id": "masked_mlp",
            "kind": "trainable",
            "backbone": "plain_mlp",
            "hidden_layer_sizes": [20, 16],
        },
        "residual_mlp": {
            "candidate_id": "residual_mlp",
            "kind": "trainable",
            "backbone": "gated_residual_mlp",
            "width": 16,
            "depth": 1,
            "expansion": 2,
            "dropout": 0.0,
        },
        "tabm_batchensemble": {
            "candidate_id": "tabm_batchensemble",
            "kind": "trainable",
            "backbone": "tabm_style_batchensemble_mlp",
            "width": 12,
            "depth": 1,
            "expansion": 2,
            "ensemble_size": 3,
            "rank1_init_std": 0.1,
            "dropout": 0.0,
        },
        "chain_endpoint_only": {
            "candidate_id": "chain_endpoint_only",
            "kind": "trainable",
            "backbone": "encoder_conditioned_four_sensor_chain_message_passing",
            "width": 10,
            "depth": 1,
            "expansion": 2,
            "dropout": 0.0,
            "masked_reconstruction_loss_weight": 0.0,
        },
        "chain_masked_reconstruction": {
            "candidate_id": "chain_masked_reconstruction",
            "kind": "trainable",
            "backbone": "encoder_conditioned_four_sensor_chain_message_passing",
            "width": 10,
            "depth": 1,
            "expansion": 2,
            "dropout": 0.0,
            "masked_reconstruction_loss_weight": 0.1,
        },
    }
    return deepcopy(candidates[candidate_id])


def _problem(
    *, train_rows: int = 48, eval_rows: int = 18, seed: int = 812
) -> dict:
    generator = np.random.default_rng(seed)
    X_train_raw = generator.normal(size=(train_rows, 43))
    X_eval_raw = generator.normal(size=(eval_rows, 43))
    coefficients = np.asarray(
        [[8.0, -3.0, 2.0], [-4.0, 7.0, 1.0], [2.0, 1.5, -6.0]],
        dtype=np.float64,
    )
    center = np.asarray([300.0, -120.0, 840.0], dtype=np.float64)
    y_train_raw = X_train_raw[:, :3] @ coefficients + center
    y_eval_raw = X_eval_raw[:, :3] @ coefficients + center
    train_uids = np.asarray([f"train-{index:05d}" for index in range(train_rows)])
    eval_uids = np.asarray([f"eval-{index:05d}" for index in range(eval_rows)])
    preprocessor = TrainOnlyPreprocessor(
        feature_names=FUSION_COLUMNS,
        split_context="outage-unit-test",
    ).fit_real_train(
        X_train_raw,
        y_train_raw,
        trial_uids=train_uids,
        forbidden_uids=eval_uids,
    )
    segment_ids = np.asarray(
        [36 + index % 3 for index in range(train_rows)], dtype=np.int64
    )
    return {
        "X_train": preprocessor.transform_features(X_train_raw).astype(np.float32),
        "y_train": preprocessor.transform_targets(y_train_raw).astype(np.float32),
        "segment_ids": segment_ids,
        "trial_uids": train_uids,
        "X_eval": preprocessor.transform_features(X_eval_raw).astype(np.float32),
        "y_eval_raw": y_eval_raw,
        "preprocessor": preprocessor,
    }


def _train(
    candidate_id: str = "masked_mlp",
    *,
    problem: dict | None = None,
    epochs: int = 3,
    model_seed: int = 913,
) -> dict:
    arrays = _problem() if problem is None else problem
    return train_outage_seed(
        **arrays,
        held_out_segment=39,
        forced_outage=1,
        paired_seed_index=0,
        candidate=_candidate(candidate_id),
        config=_config(
            epochs=epochs,
            paired_seed_values=(model_seed, model_seed + 1, model_seed + 2),
        ),
        model_seed=model_seed,
        device="cpu",
    )


def test_uid_keyed_mask_probability_permutation_and_candidate_pairing() -> None:
    uids = np.asarray([f"uid-{index:06d}" for index in range(32768)])
    settings = {
        "base_seed": 20260824,
        "held_out_segment": 38,
        "forced_outage": 2,
        "paired_seed_index": 1,
        "epoch": 17,
    }
    first = deterministic_mask_states(uids, **settings)
    repeated_for_another_candidate = deterministic_mask_states(uids, **settings)
    assert np.array_equal(first, repeated_for_another_candidate)

    # UID keys, rather than array positions, make the schedule permutation
    # equivariant.  Candidate ID cannot affect it because it is not accepted.
    permutation = np.random.default_rng(71).permutation(len(uids))
    permuted = deterministic_mask_states(uids[permutation], **settings)
    restored = np.empty_like(permuted)
    restored[permutation] = permuted
    assert np.array_equal(first, restored)

    observed = np.asarray(
        [np.mean(first == state) for state in (MASK_FULL_STATE, 0, 1, 2, 3)]
    )
    assert np.allclose(
        observed,
        np.asarray([0.25, 0.1875, 0.1875, 0.1875, 0.1875]),
        atol=0.008,
        rtol=0.0,
    )
    next_epoch = deterministic_mask_states(uids, **{**settings, "epoch": 18})
    assert not np.array_equal(first, next_epoch)


def test_sensor_mask_zeroes_only_explicit_signal_family_indices() -> None:
    complete = np.tile(np.arange(1, 44, dtype=np.float32), (5, 1))
    states = np.asarray([MASK_FULL_STATE, 0, 1, 2, 3], dtype=np.int8)
    masked = apply_sensor_masks(complete, states)
    assert masked.shape == (5, 47)
    assert np.array_equal(masked[0, :43], complete[0])
    assert np.array_equal(masked[0, 43:], np.ones(4, dtype=np.float32))
    for row, sensor_index in enumerate(range(4), start=1):
        sensor_columns = np.asarray(IMU_SENSOR_INDEX_MAP[sensor_index])
        other_columns = np.asarray(
            [index for index in range(43) if index not in set(sensor_columns)]
        )
        assert np.array_equal(
            masked[row, sensor_columns], np.zeros(10, dtype=np.float32)
        )
        assert np.array_equal(masked[row, other_columns], complete[row, other_columns])
        expected_availability = np.ones(4, dtype=np.float32)
        expected_availability[sensor_index] = 0.0
        assert np.array_equal(masked[row, 43:], expected_availability)


def test_one_seed_training_is_exactly_deterministic_and_restores_best_checkpoint() -> None:
    problem = _problem()
    first = _train(problem=problem, epochs=4, model_seed=1234)
    second = _train(problem=problem, epochs=4, model_seed=1234)

    assert first["best_epoch"] == second["best_epoch"]
    assert first["history"] == second["history"]
    assert np.array_equal(
        first["eval_prediction_raw_mm"], second["eval_prediction_raw_mm"]
    )
    assert np.array_equal(
        first["train_prediction_raw_mm"], second["train_prediction_raw_mm"]
    )
    assert tuple(first["state_dict"]) == tuple(second["state_dict"])
    assert all(
        value.device.type == "cpu" for value in first["state_dict"].values()
    )
    assert all(
        torch.equal(first["state_dict"][key], second["state_dict"][key])
        for key in first["state_dict"]
    )

    history_scores = [row["validation_rmse_3d_mm"] for row in first["history"]]
    assert first["best_epoch"] == int(np.argmin(history_scores)) + 1
    restored_metrics = regression_metrics(
        problem["y_eval_raw"], first["eval_prediction_raw_mm"]
    )
    assert restored_metrics["rmse_3d_mm"] == pytest.approx(min(history_scores))


def test_eval_labels_never_enter_optimizer_loss_or_one_epoch_state() -> None:
    original = _problem()
    shifted = dict(original)
    shifted["y_eval_raw"] = original["y_eval_raw"] + np.asarray(
        [500.0, -250.0, 100.0]
    )
    first = _train(problem=original, epochs=1, model_seed=57)
    second = _train(problem=shifted, epochs=1, model_seed=57)

    assert first["history"][0]["scaled_objective_loss"] == second["history"][0][
        "scaled_objective_loss"
    ]
    assert np.array_equal(
        first["train_prediction_raw_mm"], second["train_prediction_raw_mm"]
    )
    assert np.array_equal(
        first["eval_prediction_raw_mm"], second["eval_prediction_raw_mm"]
    )
    assert all(
        torch.equal(first["state_dict"][key], second["state_dict"][key])
        for key in first["state_dict"]
    )
    assert first["eval_metrics"]["rmse_3d_mm"] != second["eval_metrics"][
        "rmse_3d_mm"
    ]


def test_seed_fit_rejects_model_seed_that_does_not_match_paired_index() -> None:
    problem = _problem()
    with pytest.raises(OutageTrainingError, match="paired seed index"):
        train_outage_seed(
            **problem,
            held_out_segment=39,
            forced_outage=1,
            paired_seed_index=0,
            candidate=_candidate("masked_mlp"),
            config=_config(paired_seed_values=(101, 102, 103)),
            model_seed=999,
            device="cpu",
        )


def test_fixed_epoch_refit_and_label_free_state_prediction_are_deterministic() -> None:
    problem = _problem()
    final_X_raw = (
        problem["X_train"] * problem["preprocessor"].feature_scale_
        + problem["preprocessor"].feature_mean_
    )
    final_y_raw = problem["preprocessor"].inverse_targets(problem["y_train"])
    final_preprocessor = TrainOnlyPreprocessor(
        feature_names=FUSION_COLUMNS,
        split_context="outage-unit-final",
    ).fit_real_train(
        final_X_raw,
        final_y_raw,
        trial_uids=problem["trial_uids"],
    )
    uid_hash = sha256(
        "\n".join(sorted(problem["trial_uids"].tolist())).encode("utf-8")
    ).hexdigest()
    final_config = _config(
        epochs=3,
        paired_seed_values=(733, 734, 735),
    )
    final_config["row_roles"] = {
        "development": {
            "segments": [36, 37, 38],
            "expected_rows": len(problem["X_train"]),
            "expected_segment_rows": {36: 16, 37: 16, 38: 16},
            "sorted_trial_uid_sha256": uid_hash,
        }
    }
    kwargs = {
        "X_train": final_preprocessor.transform_features(final_X_raw),
        "y_train": final_preprocessor.transform_targets(final_y_raw),
        "segment_ids": problem["segment_ids"],
        "trial_uids": problem["trial_uids"],
        "preprocessor": final_preprocessor,
        "candidate": _candidate("masked_mlp"),
        "config": final_config,
        "model_seed": 733,
        "paired_seed_index": 0,
        "epochs": 2,
        "mask_held_out_segment": "final_refit_all_development",
        "mask_forced_outage": 1,
        "device": "cpu",
    }
    first = fit_outage_seed_fixed_epochs(**kwargs)
    second = fit_outage_seed_fixed_epochs(**kwargs)
    assert first["fit_kind"] == "all_development_exact_fixed_epochs"
    assert first["trained_epochs"] == 2
    assert first["history"] == second["history"]
    assert all(
        torch.equal(first["state_dict"][key], second["state_dict"][key])
        for key in first["state_dict"]
    )
    truncated = dict(kwargs)
    truncated.update(
        {
            "X_train": kwargs["X_train"][:-1],
            "y_train": kwargs["y_train"][:-1],
            "segment_ids": kwargs["segment_ids"][:-1],
            "trial_uids": kwargs["trial_uids"][:-1],
        }
    )
    with pytest.raises(OutageTrainingError, match="row count"):
        fit_outage_seed_fixed_epochs(**truncated)

    availability = np.tile(
        np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
        (len(problem["X_eval"]), 1),
    )
    raw = (
        problem["X_eval"] * problem["preprocessor"].feature_scale_
        + problem["preprocessor"].feature_mean_
    )
    raw[:, list(IMU_SENSOR_INDEX_MAP[1])] = np.nan
    scaled = transform_outage_features(
        preprocessor=final_preprocessor,
        X_raw=raw,
        availability=availability,
    )
    assert np.array_equal(
        scaled[:, list(IMU_SENSOR_INDEX_MAP[1])],
        np.zeros((len(scaled), 10), dtype=np.float32),
    )
    available_columns = [
        index
        for index in range(43)
        if index not in set(IMU_SENSOR_INDEX_MAP[1])
    ]
    assert np.allclose(
        scaled[:, available_columns],
        (
            raw[:, available_columns]
            - final_preprocessor.feature_mean_[available_columns]
        )
        / final_preprocessor.feature_scale_[available_columns],
        rtol=1e-6,
        atol=1e-6,
    )
    prediction = predict_outage_state(
        state_dict=first["state_dict"],
        X_scaled=scaled,
        availability=availability,
        preprocessor=final_preprocessor,
        candidate=_candidate("masked_mlp"),
        device="cpu",
    )
    repeated = predict_outage_state(
        state_dict=first["state_dict"],
        X_scaled=scaled,
        availability=availability,
        preprocessor=final_preprocessor,
        candidate=_candidate("masked_mlp"),
        device="cpu",
    )
    assert np.array_equal(prediction, repeated)
    assert prediction.shape == (len(scaled), 3)
    replayed_from_npz_arrays = predict_outage_state(
        state_dict={
            key: value.detach().cpu().numpy()
            for key, value in first["state_dict"].items()
        },
        X_scaled=scaled,
        availability=availability,
        preprocessor=final_preprocessor,
        candidate=_candidate("masked_mlp"),
        device="cpu",
    )
    assert np.array_equal(prediction, replayed_from_npz_arrays)
    with pytest.raises(OutageTrainingError, match="exactly zero"):
        predict_outage_state(
            state_dict=first["state_dict"],
            X_scaled=problem["X_eval"],
            availability=availability,
            preprocessor=final_preprocessor,
            candidate=_candidate("masked_mlp"),
            device="cpu",
        )


def test_masked_mlp_converges_on_encoder_only_synthetic_endpoint() -> None:
    problem = _problem(train_rows=96, eval_rows=30, seed=19)
    result = _train(problem=problem, epochs=40, model_seed=811)
    truth = problem["preprocessor"].inverse_targets(problem["y_train"])
    mean_prediction = np.repeat(np.mean(truth, axis=0, keepdims=True), len(truth), axis=0)
    baseline = regression_metrics(truth, mean_prediction)["rmse_3d_mm"]
    assert result["train_metrics"]["rmse_3d_mm"] < 0.8 * baseline
    assert np.isfinite(result["eval_prediction_raw_mm"]).all()


def test_tabm_loss_supervises_members_before_prediction_averaging() -> None:
    truth = torch.zeros(3, 3)
    members = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
            [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
            [[4.0, 4.0, 4.0], [0.0, 0.0, 0.0]],
        ]
    )
    segments = np.asarray(["a", "a", "b"])
    loss = balanced_member_mse(torch, members, truth, segments)
    group_a_member_mse = torch.square(members[:2]).mean(dim=(0, 2))
    group_b_member_mse = torch.square(members[2:]).mean(dim=(0, 2))
    expected = torch.stack((group_a_member_mse, group_b_member_mse)).mean()
    mean_prediction_loss = balanced_member_mse(
        torch, members.mean(dim=1), truth, segments
    )
    assert torch.equal(loss, expected)
    assert not torch.equal(loss, mean_prediction_loss)

    trained = _train("tabm_batchensemble", epochs=1, model_seed=29)
    assert any("member_input_scale" in key for key in trained["state_dict"])
    assert np.isfinite(trained["eval_prediction_raw_mm"]).all()


def test_chain_reconstruction_branch_uses_only_deliberately_missing_train_nodes() -> None:
    result = _train("chain_masked_reconstruction", epochs=1, model_seed=43)
    history = result["history"][0]
    assert history["scaled_masked_reconstruction_mse"] > 0.0
    assert history["scaled_objective_loss"] == pytest.approx(
        history["scaled_endpoint_member_mse"]
        + 0.1 * history["scaled_masked_reconstruction_mse"]
    )

    reconstruction = torch.randn(4, 4, 10, requires_grad=True)
    complete_targets = torch.randn(4, 4, 10)
    no_deliberate_outage = torch.full((4,), MASK_FULL_STATE, dtype=torch.int64)
    zero = deliberate_missing_reconstruction_mse(
        torch, reconstruction, complete_targets, no_deliberate_outage
    )
    assert zero.item() == 0.0
    zero.backward()
    assert torch.equal(reconstruction.grad, torch.zeros_like(reconstruction))


def test_all_trainable_candidate_branches_return_finite_raw_predictions() -> None:
    for candidate_id in (
        "masked_mlp",
        "residual_mlp",
        "chain_endpoint_only",
    ):
        result = _train(candidate_id, epochs=1, model_seed=101)
        assert result["candidate_id"] == candidate_id
        assert result["forced_outage"] == 1
        assert np.isfinite(result["eval_prediction_raw_mm"]).all()
        assert np.isfinite(result["train_prediction_raw_mm"]).all()
