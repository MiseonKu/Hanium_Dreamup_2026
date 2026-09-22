from __future__ import annotations

from research_host.quality import summarize_quality
from research_host.reports import generate_condition_report


def test_generate_condition_report_groups_by_condition() -> None:
    report = generate_condition_report(
        [
            {
                "condition_id": "C1",
                "position_rmse_mm": "2.0",
                "residual_vibration_rms_mm": "0.5",
                "settling_time_s": "0.1",
                "dropped_sample_ratio": "0.0",
            },
            {
                "condition_id": "C1",
                "position_rmse_mm": "4.0",
                "residual_vibration_rms_mm": "1.5",
                "settling_time_s": "0.3",
                "dropped_sample_ratio": "0.1",
            },
        ],
        quality_summary=summarize_quality([{"t_common_s": "0.0", "quality_flags": "ok"}]),
    )

    assert "Descriptive summary only" in report
    assert "| C1 | 2 | 3.000000 | 1.000000 | 0.200000 | 0.050000 |" in report
    assert "| ok | 1 |" in report
