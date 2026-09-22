from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imu_vive_learning import openaxes_external_native_v3_analysis as analysis
from imu_vive_learning import openaxes_external_native_v3_experiment as experiment
from imu_vive_learning.openaxes_external_native_v3_config import MASKS


def _folds(
    *,
    count: int,
    errors_by_cardinality: dict[int, float],
) -> tuple[analysis.PrimaryFoldPredictions, ...]:
    rows = []
    for fold in range(count):
        truth = np.zeros((1, 3), dtype="float64")
        prediction = np.zeros((15, 1, 3), dtype="float64")
        for view, mask in enumerate(MASKS):
            prediction[view, 0, 0] = errors_by_cardinality[sum(mask)]
        rows.append(
            analysis.PrimaryFoldPredictions(
                fold_ordinal=fold,
                fold_id=f"loso-{fold}",
                scenario_id=f"scenario-{fold}",
                row_uids=(f"uid-{fold}",),
                truth_m=truth,
                predictions_by_seed=(prediction.copy(), prediction.copy(), prediction.copy()),
            )
        )
    return tuple(rows)


def _summary(candidate: str, errors: dict[int, float]) -> dict:
    return analysis.aggregate_primary_predictions(
        candidate,
        _folds(count=8, errors_by_cardinality=errors),
        partial=False,
    )


def test_exact_primary_math_and_adjacent_metrics() -> None:
    summary = _summary(
        experiment.CANDIDATE_IDS[0],
        {4: 1.0, 3: 2.0, 2: 3.0, 1: 4.0},
    )
    metrics = summary["metrics"]
    assert summary["analysis_label"] == "full_preregistered_primary"
    assert summary["final_gate_eligible"] is True
    assert metrics["Amean_m"] == pytest.approx(3.0)
    assert metrics["WS_m"] == pytest.approx(3.0)
    assert metrics["WM_m"] == pytest.approx(4.0)
    assert metrics["J_m"] == pytest.approx(10.0 / 3.0)
    assert metrics["F_m"] == pytest.approx(1.0)
    assert metrics["cardinality_mean_rmse3d_m"] == {
        "1": pytest.approx(4.0),
        "2": pytest.approx(3.0),
        "3": pytest.approx(2.0),
    }
    assert metrics["scenario_fold_scores_m"] == pytest.approx([3.0] * 8)
    assert [row["J_m"] for row in metrics["per_seed_J_m"]] == pytest.approx(
        [10.0 / 3.0] * 3
    )
    assert [row["rmse3d_m"] for row in metrics["equal_scenario_mean_rmse3d_m_by_view"]] == pytest.approx(
        [1.0] + [2.0] * 4 + [3.0] * 6 + [4.0] * 4
    )
    assert len(analysis.ADJACENT_EDGES) == 28
    assert metrics["CR_adjacent"] == 0.0
    assert metrics["PE_adjacent_m"] == 0.0


def test_screen_is_partial_and_cannot_pass_final_gate() -> None:
    baseline = analysis.aggregate_primary_predictions(
        experiment.CANDIDATE_IDS[0],
        _folds(count=4, errors_by_cardinality={4: 2.0, 3: 3.0, 2: 4.0, 1: 5.0}),
        partial=True,
    )
    challenger = analysis.aggregate_primary_predictions(
        experiment.CANDIDATE_IDS[1],
        _folds(count=4, errors_by_cardinality={4: 1.0, 3: 2.0, 2: 3.0, 1: 4.0}),
        partial=True,
    )
    comparison = analysis.compare_candidate_summaries(
        challenger,
        baseline,
        pair_id="static_vs_dense",
        gate_family="common",
        partial=True,
    )
    assert challenger["analysis_label"] == "screen_partial_direction_only"
    assert challenger["final_gate_eligible"] is False
    assert comparison["fold_score_strict_wins"] == 4
    assert comparison["seed_J_strict_wins"] == 3
    assert comparison["direction_only"] is True
    assert comparison["gate_passed"] is None
    assert comparison["conditions"]["scenario_fold_score_wins_at_least_6_of_8"] is False


def test_full_common_and_monotonic_gate_rules() -> None:
    comparator = _summary(
        experiment.CANDIDATE_IDS[0],
        {4: 2.0, 3: 3.0, 2: 4.0, 1: 5.0},
    )
    challenger = _summary(
        experiment.CANDIDATE_IDS[1],
        {4: 1.0, 3: 2.0, 2: 3.0, 1: 4.0},
    )
    common = analysis.compare_candidate_summaries(
        challenger,
        comparator,
        pair_id="static_vs_dense",
        gate_family="common",
        partial=False,
    )
    assert common["fold_score_strict_wins"] == 8
    assert common["seed_J_strict_wins"] == 3
    assert common["gate_passed"] is True

    mofe = analysis.compare_candidate_summaries(
        challenger,
        comparator,
        pair_id="mofe_vs_two_stage_erm",
        gate_family="mofe",
        partial=False,
    )
    assert mofe["conditions"]["CR_adjacent_strictly_lower"] is False
    assert mofe["conditions"]["PE_adjacent_m_strictly_lower"] is False
    assert mofe["gate_passed"] is False

    cagrad = analysis.compare_candidate_summaries(
        challenger,
        comparator,
        pair_id="cagrad_vs_mofe",
        gate_family="cagrad",
        partial=False,
    )
    assert cagrad["conditions"]["CR_adjacent_not_higher_beyond_tolerance"] is True
    assert cagrad["conditions"]["PE_adjacent_m_not_higher_beyond_tolerance"] is True
    assert cagrad["gate_passed"] is True


def test_fixed_tree_and_canonical_exclusive_report(tmp_path: Path) -> None:
    assert analysis.fixed_sum_float64([1.0e16, -1.0e16, 1.0]) == 1.0
    report = analysis._payload({"contract": "synthetic", "finite": 1.25})
    output = tmp_path / "report.json"
    analysis.write_canonical_report(output, report)
    raw = output.read_bytes()
    assert raw.endswith(b"\n")
    assert json.loads(raw) == report
    assert raw == json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with pytest.raises(analysis.OpenAxesExternalNativeV3AnalysisError, match="already exists"):
        analysis.write_canonical_report(output, report)


def test_only_three_exact_context_scopes_are_accepted() -> None:
    assert analysis._scope(tuple(range(42))) == (
        "full_preregistered_primary",
        tuple(range(8)),
    )
    assert analysis._scope(tuple(range(24))) == (
        "deadline_scoped_primary_8_not_full_42",
        tuple(range(8)),
    )
    assert analysis._scope(tuple(range(12))) == (
        "screen_partial_direction_only",
        tuple(range(4)),
    )
    with pytest.raises(
        analysis.OpenAxesExternalNativeV3AnalysisError,
        match="exactly 0-41, deadline primary 0-23, or screen 0-11",
    ):
        analysis._scope(tuple(range(23)))


def test_empty_or_incomplete_run_rejected_before_science(tmp_path: Path) -> None:
    run_root = tmp_path / "empty-run"
    prepared_root = tmp_path / "empty-prepared"
    run_root.mkdir()
    prepared_root.mkdir()
    with pytest.raises(
        analysis.OpenAxesExternalNativeV3AnalysisError,
        match="run task authority is incomplete or invalid",
    ):
        analysis.analyze_run(
            run_root,
            prepared_root,
            candidates=(experiment.CANDIDATE_IDS[0],),
            contexts=tuple(range(12)),
        )


def test_screen_file_loader_uses_prepared_truth_and_prediction_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    run_root = tmp_path / "run"
    prepared_root = tmp_path / "prepared"
    run_root.mkdir()
    prepared_root.mkdir()
    (run_root / "task_index.json").write_text("{}\n", encoding="utf-8")
    scenarios = [f"scenario-{index}" for index in range(8)]
    table = pa.table(
        {
            "row_uid": [f"uid-{index}" for index in range(8)],
            "scenario_id": scenarios,
            "sample_index": list(range(8)),
            "aligned_time_us": [index * 1000 for index in range(8)],
            "tcp_x_m": [0.0] * 8,
            "tcp_y_m": [0.0] * 8,
            "tcp_z_m": [0.0] * 8,
        }
    )
    pq.write_table(table, prepared_root / "prepared.parquet")

    dense = experiment.CANDIDATE_IDS[0]
    terminal = experiment.CANDIDATE_IDS[5]
    tasks = []
    results = {}
    candidate_errors = {
        dense: [1.0] + [2.0] * 4 + [3.0] * 6 + [4.0] * 4,
        terminal: [0.5] + [1.5] * 4 + [2.5] * 6 + [3.5] * 4,
    }
    for context in range(24):
        fold = context // 3
        seed_ordinal = context % 3
        for candidate in (dense, terminal):
            candidate_ordinal = experiment.CANDIDATE_IDS.index(candidate)
            task_ordinal = context * 6 + candidate_ordinal
            task = experiment.TaskSpec(
                task_id=f"task_{task_ordinal:03d}",
                task_ordinal=task_ordinal,
                context_ordinal=context,
                fold_id=f"loso-{fold}",
                fold_ordinal=fold,
                fold_family="scenario_loso",
                train_scenarios=tuple(value for value in scenarios if value != scenarios[fold]),
                test_scenarios=(scenarios[fold],),
                seed=experiment.SEEDS[seed_ordinal],
                seed_ordinal=seed_ordinal,
                candidate_id=candidate,
                candidate_ordinal=candidate_ordinal,
                logical_slot=context % 12,
                physical_gpu=(context % 12) % 4,
                local_slot=(context % 12) // 4,
                task_spec_sha256="0" * 64,
            )
            tasks.append(task)
            result_dir = run_root / "results" / task.task_id
            result_dir.mkdir(parents=True)
            errors = candidate_errors[candidate]
            prediction = np.zeros((15, 1, 3), dtype="float64")
            prediction[:, 0, 0] = errors
            prediction_path = result_dir / "outer_predictions.npy"
            with prediction_path.open("wb") as handle:
                np.save(handle, prediction, allow_pickle=False)
            prediction_reference = analysis._file_reference(
                prediction_path, relative_to=result_dir
            )
            (result_dir / "result.json").write_text("{}\n", encoding="utf-8")
            view_metrics = [
                {"view_id": view_id, "rows": 1, "rmse3d_m": errors[index]}
                for index, view_id in enumerate(experiment.VIEW_IDS)
            ]
            results[task.task_id] = {
                "training_result": {
                    "outer_predictions_reference": prediction_reference,
                    "view_metrics": view_metrics,
                    "outer_view_metrics": view_metrics,
                    "outer_feature_loads": 1,
                    "outer_label_reads": 1,
                    "outer_evaluator_calls": 1,
                    "outer_model_forward_calls": 1,
                }
            }

    monkeypatch.setattr(experiment, "load_task_grid", lambda _root: tuple(tasks))
    monkeypatch.setattr(
        experiment,
        "validate_task_result",
        lambda _root, task: results[task.task_id],
    )
    report = analysis.analyze_run(
        run_root,
        prepared_root,
        candidates=(dense,),
        contexts=tuple(range(12)),
    )
    assert report["analysis_label"] == "screen_partial_direction_only"
    assert report["final_gate_eligible"] is False
    assert len(report["input_authority"]["ordered_task_inputs"]) == 12
    metrics = report["candidate_summaries"][0]["metrics"]
    assert metrics["Amean_m"] == pytest.approx(3.0)
    assert metrics["J_m"] == pytest.approx(10.0 / 3.0)
    assert metrics["CR_adjacent"] == 0.0

    deadline_report = analysis.analyze_run(
        run_root,
        prepared_root,
        candidates=(dense, terminal),
        contexts=tuple(range(24)),
    )
    assert deadline_report["analysis_label"] == "deadline_scoped_primary_8_not_full_42"
    assert deadline_report["final_gate_eligible"] is False
    assert deadline_report["full_42_context_grid_complete"] is False
    assert deadline_report["deadline_scoped_primary_8"] is True
    assert deadline_report["primary_8_fold_metric_support_complete"] is True
    assert len(deadline_report["input_authority"]["ordered_task_inputs"]) == 48
    assert [row["primary_fold_count"] for row in deadline_report["candidate_summaries"]] == [8, 8]
    assert len(deadline_report["frozen_pair_comparisons"]) == 1
    comparison = deadline_report["frozen_pair_comparisons"][0]
    assert comparison["pair_id"] == "terminal_vs_dense_primary"
    assert isinstance(comparison["gate_passed"], bool)
    assert comparison["final_claim_eligible"] is False
    assert comparison["claim_authorized"] is False
