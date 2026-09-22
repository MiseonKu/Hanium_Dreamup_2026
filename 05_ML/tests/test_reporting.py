from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from imu_vive_learning.reporting import (
    build_analysis,
    leakage_check_summary,
    write_analysis,
)


def _canonical_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_uid": ["s1-a", "s1-b", "s2-a", "s2-b"],
            "segment_id": [1, 1, 2, 2],
            "target_id": ["T000"] * 4,
            "train_eligible": [True] * 4,
            "strict_qc_eligible": [True, False, True, True],
            "train_exclusion_reasons": [""] * 4,
            "qc_warning_flags": [""] * 4,
            "vive_pos_raw_x_mm": [-1.0, 1.0, 99.0, 101.0],
            "vive_pos_raw_y_mm": [0.0] * 4,
            "vive_pos_raw_z_mm": [0.0] * 4,
        }
    )


def _assignments_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scheme": ["segment_loso"],
            "outer_fold": [0],
            "inner_fold": [-1],
            "trial_uid": ["s1-a"],
            "role": ["test"],
        }
    )


def test_repeatability_and_between_segment_drift_are_separate() -> None:
    canonical = _canonical_fixture()
    analysis, _, segment_target, target = build_analysis(
        canonical.copy(), canonical, _assignments_fixture(), []
    )

    assert len(segment_target) == 2
    np.testing.assert_allclose(
        segment_target["within_segment_target_repeatability_std_3d_mm"],
        [np.sqrt(2.0), np.sqrt(2.0)],
    )
    np.testing.assert_allclose(
        target["between_segment_centroid_drift_std_3d_mm"],
        [np.sqrt(5_000.0)],
    )
    assert target.loc[0, "rows"] == 4
    assert target.loc[0, "segments"] == 2
    assert target.loc[0, "vive_x_centroid_mean_mm"] == 50.0
    np.testing.assert_allclose(
        analysis["median_within_segment_target_repeatability_std_3d_mm"],
        np.sqrt(2.0),
    )
    np.testing.assert_allclose(
        analysis["median_between_segment_centroid_drift_std_3d_mm"],
        np.sqrt(5_000.0),
    )
    assert "median_target_repeatability_std_3d_mm" not in analysis
    assert analysis["strict_qc_sensitivity"]["full_eligible"]["rows"] == 4
    assert analysis["strict_qc_sensitivity"]["strict_qc"]["rows"] == 3
    assert all(value == 0 for value in analysis["leakage_checks"].values())


def test_write_analysis_emits_segment_target_summary_and_report_metrics(
    tmp_path: Path,
) -> None:
    canonical = _canonical_fixture()
    write_analysis(tmp_path, canonical.copy(), canonical, _assignments_fixture(), [])

    segment_target = pd.read_csv(tmp_path / "segment_target_summary.csv")
    assert list(segment_target[["segment_id", "target_id"]].itertuples(index=False, name=None)) == [
        (1, "T000"),
        (2, "T000"),
    ]

    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert "median_within_segment_target_repeatability_std_3d_mm" in audit
    assert "median_between_segment_centroid_drift_std_3d_mm" in audit
    assert audit["strict_qc_sensitivity"]["strict_qc"]["rows"] == 3
    assert all(value == 0 for value in audit["leakage_checks"].values())

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "within-(segment, target) repeatability" in report
    assert "between-segment target-centroid drift" in report
    assert "Strict-QC sensitivity rows: 3 of 4" in report
    assert "All leakage-check counts must remain zero" in report


def test_leakage_summary_detects_parent_outside_declared_train() -> None:
    assignments = pd.DataFrame(
        {
            "scheme": ["segment_loso", "segment_loso"],
            "outer_fold": [36, 36],
            "inner_fold": [-1, -1],
            "trial_uid": ["train", "held-out"],
            "role": ["train", "test"],
        }
    )
    provenance = pd.DataFrame(
        {
            "synthetic_id": ["syn-1"],
            "split_scheme": ["segment_loso"],
            "outer_fold": [36],
            "inner_fold": [-1],
            "parent_a_trial_uid": ["train"],
            "parent_b_trial_uid": ["held-out"],
        }
    )

    checks = leakage_check_summary(assignments, provenance)

    assert checks["synthetic_parent_outside_outer_train"] == 1
    assert checks["synthetic_self_pairs"] == 0
