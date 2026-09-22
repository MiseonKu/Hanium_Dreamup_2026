from __future__ import annotations

import csv

from research_host.analysis import (
    build_analysis_summary,
    compute_position_rmse,
    compute_residual_vibration_rms,
    compute_settling_time,
)
from research_host.logging import PROCESSED_FIELDNAMES, TrialLogger, read_json
from research_host.types import TrialMetadata


def _processed_row(loop_index: int, error_mm: float) -> dict[str, object]:
    row = {field: "" for field in PROCESSED_FIELDNAMES}
    row.update(
        {
            "loop_index": loop_index,
            "t_common_s": f"{loop_index * 0.1:.3f}",
            "P_hat_imu_base_mm_x": f"{error_mm:.3f}",
            "P_hat_imu_base_mm_y": "0.0",
            "P_hat_imu_base_mm_z": "0.0",
            "P_actual_base_mm_x": "0.0",
            "P_actual_base_mm_y": "0.0",
            "P_actual_base_mm_z": "0.0",
            "quality_flags": "ok",
        }
    )
    return row


def test_trial_logger_writes_metadata_and_processed_csv(tmp_path) -> None:
    metadata = TrialMetadata(
        trial_id="0001",
        condition_id="C2_imu_array_only",
        robot_model="long",
        target_id="T001",
        repeat_id=1,
        date="2026-05-01",
    )
    logger = TrialLogger(tmp_path, metadata)
    logger.write_metadata_json({"fixture": True})
    logger.write_processed_rows([_processed_row(0, 3.0)])

    assert read_json(logger.metadata_path())["trial_id"] == "0001"
    with logger.processed_csv_path().open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert set(PROCESSED_FIELDNAMES).issubset(set(reader.fieldnames or []))
        assert len(list(reader)) == 1


def test_analysis_metrics_known_values() -> None:
    rows = [_processed_row(0, 3.0), _processed_row(1, 4.0)]
    assert abs(compute_position_rmse(rows, estimate_prefix="P_hat_imu_base_mm") - 3.5355339059) < 1e-6
    summary = build_analysis_summary(
        rows,
        trial_id="0001",
        condition_id="C2",
        robot_model="long",
        target_id="T001",
        settling_threshold_mm=4.0,
    )
    assert summary["trial_id"] == "0001"
    assert summary["settling_time_s"] == "0.000000"


def test_settling_time_and_residual_vibration() -> None:
    assert compute_settling_time([0.0, 0.1, 0.2], [9.0, 2.0, 1.0], threshold_mm=2.0) == 0.1
    assert compute_settling_time([0.0, 0.1, 0.2], [9.0, 2.0, 9.0], threshold_mm=2.0) is None
    assert abs(compute_residual_vibration_rms([1.0, 2.0, 3.0], start_index=0) - 0.8164965809) < 1e-6
