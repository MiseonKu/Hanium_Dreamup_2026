from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from imu_vive_learning.experiment_preprocessing import TrainOnlyPreprocessor
from imu_vive_learning.gpu_robust_training import (
    ROBUST_HIDDEN_LAYERS,
    RobustTrainingError,
    group_mass,
    post_warmup_value,
    segment_balanced_batches,
    train_robust_torch,
    update_group_dro_weights,
    v_rex_loss,
)


torch = pytest.importorskip("torch")


def _arrays(*, evaluation: bool = True) -> dict:
    generator = np.random.default_rng(20260824)
    feature_names = tuple(f"f_{index:02d}" for index in range(43))
    X_raw = generator.normal(size=(9, 43))
    y_raw = np.column_stack(
        [
            100.0 + 2.0 * X_raw[:, 0],
            -50.0 + 3.0 * X_raw[:, 1],
            20.0 - X_raw[:, 2],
        ]
    )
    segments = np.asarray([36, 36, 36, 36, 37, 37, 37, 38, 38])
    preprocessor = TrainOnlyPreprocessor(
        feature_names, split_context="test_robust_train"
    ).fit_real_train(
        X_raw,
        y_raw,
        trial_uids=[f"train-{index}" for index in range(len(X_raw))],
        forbidden_uids=["validation-only"],
    )
    arrays = {
        "split_context": "segment_loso_outer36_inner37",
        "X_train": preprocessor.transform_features(X_raw),
        "y_train": preprocessor.transform_targets(y_raw),
        "train_segment_ids": segments,
        "preprocessor": preprocessor,
    }
    if evaluation:
        X_eval_raw = generator.normal(size=(3, 43))
        y_eval_raw = np.column_stack(
            [
                100.0 + 2.0 * X_eval_raw[:, 0],
                -50.0 + 3.0 * X_eval_raw[:, 1],
                20.0 - X_eval_raw[:, 2],
            ]
        )
        arrays.update(
            {
                "X_eval": preprocessor.transform_features(X_eval_raw),
                "y_eval_raw": y_eval_raw,
                # This must never enter batching or group-objective history.
                "evaluation_segment_ids": np.asarray([99, 99, 99]),
            }
        )
    return arrays


def _config(*, epochs: int = 2) -> dict:
    return {
        "contract": {"base_seed": 20260824},
        "models": {
            "torch_mlp": {
                "hidden_layer_sizes": [64, 64],
                "weight_decay": 1e-3,
                "batch_size": 6,
                "learning_rate": 1e-3,
                "epoch_budget": epochs,
                "ensemble_seed_count": 3,
            }
        },
        "monitoring": {"history_interval_epochs": 1},
    }


def _task(candidate: dict, *, phase: str = "inner", epochs: int = 2) -> dict:
    return {
        "task_id": f"test__{candidate['candidate_id']}__{phase}",
        "phase": phase,
        "candidate": candidate,
        "epoch_budget_override": epochs,
    }


def test_group_mass_reports_empirical_and_weighted_objective_mass() -> None:
    assert group_mass([36, 36, 37, 38]) == {"36": 0.5, "37": 0.25, "38": 0.25}
    weighted = group_mass([36, 36, 37, 38], [0.5, 0.5, 1.0, 2.0])
    assert weighted == {"36": 0.25, "37": 0.25, "38": 0.5}


def test_group_mass_rejects_nonfinite_or_zero_weight_mass() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        group_mass([36, 37], [1.0, np.nan])
    with pytest.raises(ValueError, match="positive total mass"):
        group_mass([36, 37], [0.0, 0.0])


def test_balanced_batches_use_largest_group_epoch_and_cycle_small_groups() -> None:
    segments = np.asarray([36, 36, 36, 36, 36, 37, 37, 38])
    batches = segment_balanced_batches(segments, batch_size=7, seed=123)
    # floor(7 / 3) == 2 rows/group and ceil(5 / 2) == 3 batches.
    assert len(batches) == 3
    assert all(len(batch) == 6 for batch in batches)
    assert all(
        group_mass(segments[batch])
        == {"36": 1 / 3, "37": 1 / 3, "38": 1 / 3}
        for batch in batches
    )
    visited_largest = np.concatenate([batch[segments[batch] == 36] for batch in batches])
    assert set(np.flatnonzero(segments == 36)).issubset(set(visited_largest.tolist()))
    assert sum(segments[np.concatenate(batches)] == 38) == 6


def test_balanced_batches_are_seed_deterministic_and_validate_capacity() -> None:
    segments = [36, 36, 36, 37, 37, 38, 38]
    first = segment_balanced_batches(segments, batch_size=6, seed=7)
    second = segment_balanced_batches(segments, batch_size=6, seed=7)
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    with pytest.raises(ValueError, match="at least the number"):
        segment_balanced_batches(segments, batch_size=2, seed=7)


def test_group_dro_update_increases_mass_on_the_worst_group_and_is_stable() -> None:
    updated = update_group_dro_weights(
        [1 / 3, 1 / 3, 1 / 3], [1.0, 4.0, 2.0], eta=0.1
    )
    assert updated[1] > updated[2] > updated[0]
    assert float(updated.sum()) == pytest.approx(1.0)
    huge = update_group_dro_weights([0.5, 0.5], [1e6, 1e6 + 1], eta=100.0)
    assert np.isfinite(huge).all()
    assert float(huge.sum()) == pytest.approx(1.0)


def test_group_dro_update_rejects_invalid_eta_and_weights() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        update_group_dro_weights([1.0, 0.0], [1.0, 2.0], eta=0.1)
    with pytest.raises(ValueError, match="eta"):
        update_group_dro_weights([0.5, 0.5], [1.0, 2.0], eta=0.0)


@pytest.mark.parametrize("eta", [0.01, 0.1])
def test_group_dro_weights_remain_strictly_interior_over_full_run_updates(
    eta: float,
) -> None:
    weights = np.full(3, 1.0 / 3.0)
    for _ in range(20_000):
        weights = update_group_dro_weights(weights, [1.0, 0.0, 0.0], eta=eta)
    assert np.all(weights > 0.0)
    assert np.all(weights < 1.0)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_vrex_loss_and_erm_warmup_are_exact() -> None:
    # mean([1, 3]) + beta * population_var([1, 3]) == 2 + beta.
    assert v_rex_loss([1.0, 3.0], beta=10.0) == pytest.approx(12.0)
    assert post_warmup_value(100.0, epoch=1, warmup_epochs=10) == 0.0
    assert post_warmup_value(100.0, epoch=10, warmup_epochs=10) == 0.0
    assert post_warmup_value(100.0, epoch=11, warmup_epochs=10) == 100.0
    assert post_warmup_value(100.0, epoch=20, warmup_epochs=10) == 100.0


@pytest.mark.parametrize(
    "candidate",
    [
        {"candidate_id": "balanced_erm", "objective": "balanced_erm"},
        {"candidate_id": "group_dro", "objective": "group_dro", "eta": 0.1},
        {
            "candidate_id": "v_rex",
            "objective": "v_rex",
            "beta": 10.0,
            "warmup_epochs": 2,
        },
        {
            "candidate_id": "sam",
            "objective": "balanced_erm",
            "sam": True,
            "rho": 0.05,
        },
    ],
)
def test_all_robust_objectives_train_finitely_on_cpu(candidate: dict) -> None:
    result = train_robust_torch(
        _arrays(),
        task=_task(candidate, epochs=1),
        config=_config(epochs=1),
        device="cpu",
    )
    assert result["ensemble_prediction"].shape == (3, 3)
    assert np.isfinite(result["ensemble_prediction"]).all()
    assert np.isfinite(float(result["metrics"]["rmse_3d_mm"]))
    assert len(result["seed_summaries"]) == 3
    assert {row["segment_id"] for row in result["group_history"]} == {"36", "37", "38"}
    assert 99 not in {row["segment_id"] for row in result["group_history"]}


def test_candidate_comparison_uses_paired_model_seeds() -> None:
    balanced = train_robust_torch(
        _arrays(),
        task=_task({"candidate_id": "balanced_erm", "objective": "balanced_erm"}, epochs=1),
        config=_config(epochs=1),
        device="cpu",
    )
    group_dro = train_robust_torch(
        _arrays(),
        task=_task(
            {"candidate_id": "group_dro", "objective": "group_dro", "eta": 0.1},
            epochs=1,
        ),
        config=_config(epochs=1),
        device="cpu",
    )
    assert [row["model_seed"] for row in balanced["seed_summaries"]] == [
        row["model_seed"] for row in group_dro["seed_summaries"]
    ]


def test_inner_checkpoint_and_state_match_existing_64_64_inference_shape() -> None:
    result = train_robust_torch(
        _arrays(),
        task=_task({"candidate_id": "balanced_erm", "objective": "balanced_erm"}),
        config=_config(),
        return_state=True,
        device="cpu",
    )
    assert len(result["states"]) == 3
    assert all(summary["selected_epoch"] in {1, 2} for summary in result["seed_summaries"])
    assert all(
        summary["best_validation_rmse_3d_mm"] is not None
        for summary in result["seed_summaries"]
    )

    from imu_vive_learning.gpu_experiment import _build_torch_model

    expected = _build_torch_model(torch, 43, ROBUST_HIDDEN_LAYERS).state_dict()
    for state in result["states"].values():
        assert tuple(state) == tuple(expected)
        assert {name: value.shape for name, value in state.items()} == {
            name: tuple(value.shape) for name, value in expected.items()
        }
        assert state["0.weight"].shape == (64, 43)  # no segment-ID feature
        assert all(np.isfinite(value).all() for value in state.values())


def test_outer_uses_fixed_epochs_without_validation_checkpoint_selection() -> None:
    task = _task(
        {"candidate_id": "balanced_erm", "objective": "balanced_erm"},
        phase="outer",
        epochs=2,
    )
    task["epochs_per_seed"] = {"0": 1, "1": 2, "2": 1}
    result = train_robust_torch(
        _arrays(), task=task, config=_config(), device="cpu"
    )
    assert [row["selected_epoch"] for row in result["seed_summaries"]] == [1, 2, 1]
    assert all(row["best_validation_rmse_3d_mm"] is None for row in result["seed_summaries"])
    assert all("validation_rmse_3d_mm" not in row for row in result["history"])


def test_training_rejects_nonfinite_features_and_more_than_400_epochs() -> None:
    arrays = _arrays()
    arrays["X_train"][0, 0] = np.nan
    with pytest.raises(RobustTrainingError, match="non-finite"):
        train_robust_torch(
            arrays,
            task=_task({"candidate_id": "balanced_erm", "objective": "balanced_erm"}),
            config=_config(),
            device="cpu",
        )

    config = _config()
    config["models"]["torch_mlp"]["epoch_budget"] = 401
    with pytest.raises(RobustTrainingError, match=r"\[1, 400\]"):
        train_robust_torch(
            _arrays(),
            task=_task({"candidate_id": "balanced_erm", "objective": "balanced_erm"}),
            config=config,
            device="cpu",
        )


def test_training_rejects_segment_feature_architecture_or_wrong_seed_count() -> None:
    arrays = _arrays()
    arrays["preprocessor"].feature_names = tuple(
        [*arrays["preprocessor"].feature_names[:-1], "segment_id"]
    )
    with pytest.raises(RobustTrainingError, match="never be a model feature"):
        train_robust_torch(
            arrays,
            task=_task({"candidate_id": "balanced_erm", "objective": "balanced_erm"}),
            config=_config(),
            device="cpu",
        )

    config = deepcopy(_config())
    config["models"]["torch_mlp"]["ensemble_seed_count"] = 2
    with pytest.raises(RobustTrainingError, match="exactly three"):
        train_robust_torch(
            _arrays(),
            task=_task({"candidate_id": "balanced_erm", "objective": "balanced_erm"}),
            config=config,
            device="cpu",
        )
