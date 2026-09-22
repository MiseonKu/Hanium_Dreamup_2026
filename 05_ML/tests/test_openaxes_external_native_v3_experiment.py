from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import pytest

from imu_vive_learning import openaxes_external_native_v3_experiment as experiment


def _manifest() -> dict[str, object]:
    scenarios = [
        f"{shape}-{speed}"
        for shape in ("circle", "triangle")
        for speed in (100, 150, 200, 250)
    ]
    folds: list[dict[str, object]] = []
    for scenario in scenarios:
        folds.append(
            {
                "fold_id": f"scenario_loso::{scenario}",
                "family": "scenario_loso",
                "train_scenarios": [item for item in scenarios if item != scenario],
                "test_scenarios": [scenario],
            }
        )
    for speed in (100, 150, 200, 250):
        test = [f"circle-{speed}", f"triangle-{speed}"]
        folds.append(
            {
                "fold_id": f"speed_pair_out::{speed}",
                "family": "speed_pair_out",
                "train_scenarios": [item for item in scenarios if item not in test],
                "test_scenarios": test,
            }
        )
    for shape in ("circle", "triangle"):
        test = [f"{shape}-{speed}" for speed in (100, 150, 200, 250)]
        folds.append(
            {
                "fold_id": f"shape_out::{shape}",
                "family": "shape_out",
                "train_scenarios": [item for item in scenarios if item not in test],
                "test_scenarios": test,
            }
        )
    return {"contract": "openaxes.external.preparation.v1", "folds": {"items": folds}}


def _view_metrics(value: float = 0.01) -> list[dict[str, object]]:
    return [
        {"view_id": view_id, "rows": 10, "rmse3d_m": value + index / 1000}
        for index, view_id in enumerate(experiment.VIEW_IDS)
    ]


def test_config_grid_and_four_gpu_slot_accounting() -> None:
    experiment.validate_config_api()
    tasks = experiment.build_native_v3_task_grid(_manifest())
    assert len(tasks) == 252
    assert [task.task_id for task in tasks] == [f"task_{index:03d}" for index in range(252)]
    for task in tasks:
        assert task.context_ordinal == task.fold_ordinal * 3 + task.seed_ordinal
        assert task.task_ordinal == task.context_ordinal * 6 + task.candidate_ordinal
        assert task.logical_slot == task.context_ordinal % 12
        assert task.physical_gpu == task.logical_slot % 4
        assert task.local_slot == task.logical_slot // 4
    for context in range(42):
        sextet = tasks[context * 6 : context * 6 + 6]
        assert [task.candidate_id for task in sextet] == list(experiment.CANDIDATE_IDS)
        assert len({task.logical_slot for task in sextet}) == 1
    slots = experiment.build_slot_assignments(tasks)
    assert tuple(len(row["ordered_task_ids"]) for row in slots) == experiment.FITS_PER_SLOT
    assert tuple(
        sum(len(slots[slot]["ordered_task_ids"]) for slot in range(gpu, 12, 4))
        for gpu in range(4)
    ) == (66, 66, 60, 60)
    waves = experiment.build_launch_waves(tasks)
    assert len(waves) == 24
    assert sum(map(len, waves)) == 252
    assert all(
        sum(task.physical_gpu == gpu for task in wave) <= 3
        for wave in waves
        for gpu in range(4)
    )


def test_prepare_accepts_preissued_parent_and_round_trips(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "parents" / "native_v2").mkdir(parents=True)
    (root / "parents" / "native_v2" / "receipt.json").write_text("authority")
    tasks = experiment.prepare_run(root, _manifest())
    assert len(tasks) == 252
    assert experiment.load_task_grid(root) == tasks
    assert (root / "parents" / "native_v2" / "receipt.json").read_text() == "authority"
    status = experiment.status_projection(root, tasks)
    assert status["state_counts"]["pending"] == 252
    assert status["expected_fits_by_gpu"] == [66, 66, 60, 60]
    changed = experiment._read_json(root / "tasks" / "task_000.json", name="task")
    changed["physical_gpu"] = 3
    (root / "tasks" / "task_000.json").write_bytes(experiment._canonical_file(changed))
    with pytest.raises(experiment.OpenAxesExternalNativeV3ExperimentError):
        experiment.load_task_grid(root)


def test_subset_selectors_are_operational() -> None:
    assert experiment.parse_context_selector("0,3-5,41") == (0, 3, 4, 5, 41)
    assert experiment.parse_candidate_selector("all") == experiment.CANDIDATE_IDS
    selected = f"{experiment.CANDIDATE_IDS[0]},{experiment.CANDIDATE_IDS[-1]}"
    assert experiment.parse_candidate_selector(selected) == (
        experiment.CANDIDATE_IDS[0],
        experiment.CANDIDATE_IDS[-1],
    )
    with pytest.raises(experiment.OpenAxesExternalNativeV3ExperimentError):
        experiment.parse_context_selector("42")
    with pytest.raises(experiment.OpenAxesExternalNativeV3ExperimentError):
        experiment.parse_candidate_selector("unknown")


def test_six_candidate_screen_uses_six_fresh_children_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    tasks = experiment.prepare_run(root, _manifest())
    observed_commands: list[list[str]] = []
    observed_environments: list[dict[str, str]] = []

    def fit_runner(task: experiment.TaskSpec, **_kwargs: object) -> dict[str, object]:
        return {"view_metrics": _view_metrics(), "outer_view_metrics": _view_metrics()}

    class Process:
        pid = os.getpid()

        def wait(self) -> int:
            return 0

    def popen(command: list[str], **kwargs: object) -> Process:
        observed_commands.append(command)
        observed_environments.append(dict(kwargs["env"]))
        task_id = command[command.index("--task-id") + 1]
        gpu_index = int(command[command.index("--gpu-index") + 1])
        selector = command[command.index("--gpu-selector") + 1]
        attempt_id = command[command.index("--attempt-id") + 1]
        old = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = selector
        try:
            experiment.run_internal_fit(
                root,
                task_id,
                prepared_root="prepared",
                gpu_index=gpu_index,
                gpu_selector=selector,
                attempt_id=attempt_id,
                fit_runner=fit_runner,
            )
        finally:
            if old is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old
        return Process()

    result = experiment.run_scheduled_fits(
        root,
        prepared_root="prepared",
        python_executable="/venv/python",
        candidate_subset=experiment.CANDIDATE_IDS,
        context_subset=(0,),
        popen_factory=popen,
    )
    assert result["completed_task_count"] == 6
    assert result["required_task_count"] == 6
    assert len(observed_commands) == 6
    for environment in observed_environments:
        assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert environment["PYTHONSAFEPATH"] == "1"
        assert environment["LC_ALL"] == "C.UTF-8"
        assert environment["TZ"] == "UTC"
        assert environment["NUMEXPR_NUM_THREADS"] == "1"
        assert environment["BLIS_NUM_THREADS"] == "1"
        assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
        assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert {command[command.index("--gpu-index") + 1] for command in observed_commands} == {"0"}
    assert all(experiment.task_state(root, task.task_id) == "completed" for task in tasks[:6])
    experiment.run_scheduled_fits(
        root,
        prepared_root="prepared",
        candidate_subset=experiment.CANDIDATE_IDS,
        context_subset=(0,),
        popen_factory=popen,
    )
    assert len(observed_commands) == 6


def test_partial_two_candidate_evaluation_marks_deferred_grid() -> None:
    results: list[dict[str, object]] = []
    for context in range(42):
        for candidate in (experiment.CANDIDATE_IDS[0], experiment.CANDIDATE_IDS[-1]):
            results.append(
                {
                    "task_id": f"{candidate}:{context}",
                    "candidate_id": candidate,
                    "training_result": {"view_metrics": _view_metrics(0.01 + context / 10000)},
                }
            )
    evaluation = experiment.evaluate_result_payloads(results, require_full_grid=False)
    assert evaluation["task_count"] == 84
    assert evaluation["candidate_order"] == [
        experiment.CANDIDATE_IDS[0], experiment.CANDIDATE_IDS[-1]
    ]
    assert evaluation["full_preregistered_grid_complete"] is False
    assert evaluation["deferred_preregistered_ablation_fit_count"] == 168
    assert all(
        row["views"][0]["fit_count"] == 42
        for row in evaluation["candidate_summaries"]
    )


def test_interrupted_attempt_is_not_retried(tmp_path: Path) -> None:
    root = tmp_path / "run"
    tasks = experiment.prepare_run(root, _manifest())
    attempt = experiment._attempt_payload(tasks[0], "0")
    experiment._write_json_exclusive(root / "attempts" / "task_000.json", attempt)
    with pytest.raises(
        experiment.OpenAxesExternalNativeV3ExperimentError,
        match="attempted task",
    ):
        experiment.run_scheduled_fits(
            root,
            prepared_root="prepared",
            candidate_subset=(experiment.CANDIDATE_IDS[0],),
            context_subset=(0,),
            popen_factory=lambda *_args, **_kwargs: None,
        )
