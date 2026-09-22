from __future__ import annotations

from research_host.sync import (
    compute_jitter_s,
    compute_offset_by_correlation,
    correct_sample_time,
    dropped_sample_flags,
    estimate_sample_time,
    resample_scalar_series,
)


def test_timestamp_correction() -> None:
    assert estimate_sample_time(1.0, 1.2) == 1.1
    assert abs(correct_sample_time(1.0, 1.2, latency_s=0.01, offset_s=0.02) - 1.07) < 1e-9


def test_jitter_and_gap_flags() -> None:
    times = [0.0, 0.01, 0.021, 0.08]
    jitters = compute_jitter_s(times, 0.01)
    assert abs(jitters[2] - 0.001) < 1e-9
    assert dropped_sample_flags(times, max_gap_s=0.03) == [False, False, False, True]


def test_resample_scalar_series_linear() -> None:
    values = resample_scalar_series([0.0, 1.0], [0.0, 10.0], [0.0, 0.5, 1.0])
    assert values == [0.0, 5.0, 10.0]


def test_offset_by_correlation() -> None:
    reference = [0.0, 1.0, 0.0, 0.0]
    shifted = [0.0, 0.0, 1.0, 0.0]
    assert compute_offset_by_correlation(
        reference,
        shifted,
        sample_period_s=0.01,
        max_lag_samples=2,
    ) == 0.01
