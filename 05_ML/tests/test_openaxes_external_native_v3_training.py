from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from imu_vive_learning import openaxes_external_native_v3_training as training


def _synthetic_training() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rows = 8
    base = np.arange(rows, dtype=np.float64)[:, None]
    features = base + np.arange(36, dtype=np.float64)[None, :] / 10.0
    targets = np.column_stack(
        (base[:, 0] * 0.01, base[:, 0] * -0.02, base[:, 0] * 0.03)
    )
    scenarios = ["circle-100"] * 4 + ["triangle-100"] * 4
    uids = [f"synthetic-{index:02d}" for index in range(rows)]
    return features, targets, scenarios, uids


def test_exact_fifteen_mask_lattice_and_twenty_eight_edges() -> None:
    assert len(training._VIEW_IDS) == len(training._MASKS) == 15
    assert training._MASKS[0] == (1, 1, 1, 1)
    assert [sum(mask) for mask in training._MASKS] == [4] + [3] * 4 + [2] * 6 + [1] * 4
    edges = training.adjacent_mask_edges()
    assert len(edges) == len(set(edges)) == 28
    transitions = [(sum(training._MASKS[rich]), sum(training._MASKS[poor])) for rich, poor in edges]
    assert transitions.count((4, 3)) == 4
    assert transitions.count((3, 2)) == 12
    assert transitions.count((2, 1)) == 12
    for rich, poor in edges:
        assert all(left >= right for left, right in zip(training._MASKS[rich], training._MASKS[poor], strict=True))


def test_exact_fifteen_views_zero_missing_channels_and_preserve_bits() -> None:
    features, _targets, _scenarios, _uids = _synthetic_training()
    views = training.build_exact_fifteen_views(features)
    training.validate_exact_fifteen_views(features, views)
    assert views.shape == (15, 8, 40)
    for view_index, mask in enumerate(training._MASKS):
        assert np.array_equal(views[view_index, :, 36:], np.tile(mask, (8, 1)))
        for sensor_index, available in enumerate(mask):
            block = views[view_index, :, sensor_index * 9 : (sensor_index + 1) * 9]
            assert np.array_equal(block, features[:, sensor_index * 9 : (sensor_index + 1) * 9]) if available else np.array_equal(block, np.zeros_like(block))


def test_all_six_candidates_are_parameter_matched_and_scheduled() -> None:
    assert len(training.OPENAXES_NATIVE_V3_CANDIDATE_IDS) == 6
    hashes: list[str] = []
    for candidate_id in training.OPENAXES_NATIVE_V3_CANDIDATE_IDS:
        model = training.build_candidate_model(candidate_id, seed=1701, device="cpu")
        assert sum(parameter.numel() for parameter in model.parameters()) == 6979
        schedule = training.candidate_schedule(candidate_id)
        assert schedule["independent_expert_updates"] + schedule["cooperative_updates"] == 400
        if schedule["architecture"] == "dmome_output_space":
            hashes.append(training.state_dict_sha256(model.state_dict()))
    assert len(set(hashes)) == 1
    assert training.candidate_schedule(training.OPENAXES_NATIVE_V3_CANDIDATE_IDS[-1]) == {
        "candidate_id": training.OPENAXES_NATIVE_V3_CANDIDATE_IDS[-1],
        "candidate_ordinal": 5,
        "architecture": "dmome_output_space",
        "independent_expert_updates": 100,
        "cooperative_updates": 300,
        "mofe": True,
        "cagrad": True,
    }


def test_dmome_router_is_zero_on_missing_and_simplex_on_available() -> None:
    model = training.build_dmome_6979(seed=2603, device="cpu")
    inputs = torch.zeros((15, 40), dtype=torch.float32)
    inputs[:, 36:] = torch.tensor(training._MASKS, dtype=torch.float32)
    weights = model.router_weights(inputs)
    prediction = model(inputs)
    assert prediction.shape == (15, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(15))
    assert torch.equal(weights[inputs[:, 36:] == 0], torch.zeros_like(weights[inputs[:, 36:] == 0]))
    assert bool((weights >= 0).all())


def test_cardinality_balancing_and_mofe_use_frozen_reductions() -> None:
    codes = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    losses = torch.stack(
        [torch.full((4,), float(sum(mask)), dtype=torch.float64) for mask in training._MASKS]
    )
    risks = training.cardinality_balanced_risks(losses, codes, 2)
    assert torch.equal(risks, torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
    assert training.direct_cardinality_objective(risks).item() == 2.5
    assert training.more_vs_fewer_loss(losses, codes, 2).item() == 1.0


def test_cagrad_exact_identity_solution_and_fail_closed_shape() -> None:
    solved = training.solve_cagrad_weights(np.eye(4, dtype=np.float64))
    assert np.allclose(solved["weights"], np.full(4, 0.25), atol=1.0e-12)
    assert np.allclose(solved["direction"], np.full(4, 0.35), atol=1.0e-12)
    assert solved["equality_residual"] <= 1.0e-10
    assert solved["bound_violation"] <= 1.0e-10
    with pytest.raises(training.OpenAxesExternalNativeV3TrainingError):
        training.solve_cagrad_weights(np.ones((3, 4)))
    with pytest.raises(training.OpenAxesExternalNativeV3TrainingError):
        training.solve_cagrad_weights(np.full((4, 3), np.nan))


def test_fifteen_view_checkpoint_excludes_full_view() -> None:
    prediction = np.zeros((15, 4, 3), dtype=np.float64)
    truth = np.ones((4, 3), dtype=np.float64)
    score, missing, components = training.training_checkpoint_score_m(
        prediction,
        truth,
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        2,
        0.5,
    )
    assert len(missing) == 14
    assert missing == tuple(components[1:])
    assert score == pytest.approx(np.sqrt(3.0) * 0.5)


def test_checkpoint_encoding_is_canonical_and_round_trips() -> None:
    model = training.build_dmome_6979(seed=3907, device="cpu")
    checkpoint = {
        "contract": "openaxes.external.native_sensor_experts.v3.checkpoint.v1",
        "candidate_id": training.OPENAXES_NATIVE_V3_CANDIDATE_IDS[-1],
        "seed": 3907,
        "state_dict": training._state_dict_payload(model.state_dict()),
        "state_dict_sha256": training.state_dict_sha256(model.state_dict()),
    }
    raw = training._canonical_bytes(checkpoint)
    restored = training.load_checkpoint_state(raw)
    assert training.state_dict_sha256(restored) == training.state_dict_sha256(model.state_dict())
    with pytest.raises(training.OpenAxesExternalNativeV3TrainingError):
        training.load_checkpoint_state(raw.rstrip())


def test_train_and_experiment_signatures_exclude_outer_labels() -> None:
    parameters = set(inspect.signature(training.train_candidate_train_only).parameters)
    assert not {"outer_labels", "outer_targets", "test_targets", "test_labels"}.intersection(parameters)
    experiment = inspect.signature(training.run_experiment_fit)
    assert list(experiment.parameters) == ["task", "prepared_root", "output_dir", "device"]
    assert experiment.parameters["prepared_root"].kind is inspect.Parameter.KEYWORD_ONLY


def test_experiment_wrapper_projects_exact_train_only_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, targets, scenarios, uids = _synthetic_training()
    expected = {
        "X_train": features,
        "y_train_m": targets,
        "training_scenarios": scenarios,
        "train_uids": uids,
        "declared_scenario_order": ("circle-100", "triangle-100"),
        "candidate_id": training.OPENAXES_NATIVE_V3_CANDIDATE_IDS[0],
        "seed": 1701,
    }
    observed: dict[str, object] = {}

    def fake_run_fit_task(
        task_payload: object,
        run_dir: object = None,
        device: object = "cuda:0",
    ) -> dict[str, object]:
        observed["task"] = task_payload
        observed["run_dir"] = run_dir
        observed["device"] = device
        return {"checkpoint_reference": {"path": "checkpoint.json"}}

    monkeypatch.setattr(training, "run_fit_task", fake_run_fit_task)
    controller_task = {
        **expected,
        "fold_id": "fold-0",
        "slot_id": "slot-3",
        "physical_gpu": 2,
        "test_scenarios": ("square-100",),
        "attempt_uuid_v4": "00000000-0000-4000-8000-000000000000",
    }
    result = training.run_experiment_fit(
        controller_task,
        prepared_root="/authenticated/prepared",
        output_dir="/private/output",
        device="cuda:0",
    )
    assert list(observed["task"]) == list(expected)
    projected = observed["task"]
    assert isinstance(projected, dict)
    assert set(projected) == set(expected)
    assert observed["run_dir"] == "/private/output"
    assert observed["device"] == "cuda:0"
    assert result["outer_evaluation_deferred_to_experiment"] is True
    assert result["prepared_root_bound"] == "/authenticated/prepared"


def test_experiment_wrapper_requires_all_seven_train_only_fields() -> None:
    with pytest.raises(training.OpenAxesExternalNativeV3TrainingError):
        training.run_experiment_fit(
            {"candidate_id": training.OPENAXES_NATIVE_V3_CANDIDATE_IDS[0]},
            prepared_root="/authenticated/prepared",
            output_dir="/private/output",
            device="cpu",
        )


def test_short_two_stage_cagrad_path_is_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    features, targets, scenarios, uids = _synthetic_training()
    monkeypatch.setattr(training, "EXACT_UPDATES", 2)
    monkeypatch.setattr(training, "STAGE_ONE_UPDATES", 1)
    result = training.train_candidate_train_only(
        features,
        targets,
        scenarios,
        uids,
        ("circle-100", "triangle-100"),
        candidate_id=training.OPENAXES_NATIVE_V3_CANDIDATE_IDS[-1],
        seed=1701,
        updates=2,
        device="cpu",
    )
    assert result.updates == 2
    assert result.checkpoint_update == 2
    assert result.stage_one_model_state_sha256 is not None
    assert result.stage_one_optimizer_state_sha256 is not None
    assert result.training_model_forward_calls == 2
    assert result.checkpoint_model_forward_calls == 2
    assert result.parameter_gradient_traversals == 5
    assert result.cagrad_solver_calls == 1
    assert len(result.checkpoint_missing_components_m) == 14
