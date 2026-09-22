from __future__ import annotations

from research_host.quality import quality_issues, summarize_quality


def test_quality_summary_counts_flags_and_gap() -> None:
    rows = [
        {"t_common_s": "0.00", "quality_flags": "ok"},
        {"t_common_s": "0.02", "quality_flags": "dropped_sample_flag|tracker_occlusion_flag"},
        {"t_common_s": "0.09", "quality_flags": "ok"},
    ]

    summary = summarize_quality(rows)

    assert summary.row_count == 3
    assert summary.flag_counts["dropped_sample_flag"] == 1
    assert summary.tracker_occlusion_count == 1
    assert abs(summary.max_timestamp_gap_s - 0.07) < 1e-9


def test_quality_issues_enforce_thresholds() -> None:
    summary = summarize_quality(
        [
            {"t_common_s": "0.00", "quality_flags": "ok"},
            {"t_common_s": "0.10", "quality_flags": "dropped_sample_flag|tracker_occlusion_flag"},
        ]
    )

    issues = quality_issues(
        summary,
        {
            "max_dropped_sample_ratio": 0.1,
            "max_timestamp_gap_s": 0.05,
            "tracker_occlusion_allowed": False,
        },
    )

    assert len(issues) == 3
    assert any("dropped_sample_ratio" in issue for issue in issues)
    assert any("max_timestamp_gap_s" in issue for issue in issues)
    assert any("tracker_occlusion_flag" in issue for issue in issues)
