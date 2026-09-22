"""PC<->ESP32 clock-sync estimator (master spec section 14).

PC-master monotonic clock is `core.schemas.monotonic_ns()` -- imported, not
reimplemented (DECISIONS.md #6, monotonic-clock-only policy).

This module only accumulates round-trip `SyncSample`s and computes/reports
a clock offset+drift estimate (`t_pc = a * t_esp + b`, simple least-squares
fit) plus RTT/residual statistics. Per spec section 14 it NEVER applies any
correction to raw data records -- `report()` returns a plain dict, nothing
here mutates a `SyncSample` or reaches into any sensor/encoder record.
Applying the reported offset/drift to raw timestamps is explicitly a
downstream/analysis-time decision, not something this acquisition-time
module does.

Does not talk to real ESP32 hardware -- samples are handed to
`ClockSyncEstimator.add_sample()` by whatever owns the serial exchange (a
mock, in tests / later real code).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SyncSample:
    seq: int
    t_pc_send_ns: int
    t_esp_receive_us: int
    t_esp_send_us: int
    t_pc_receive_ns: int

    @property
    def rtt_ns(self) -> int:
        return self.t_pc_receive_ns - self.t_pc_send_ns

    @property
    def esp_mid_us(self) -> float:
        """Midpoint of the ESP-side receive/send timestamps -- the ESP
        clock reading assumed to correspond to `pc_mid_ns` under a
        symmetric-latency assumption (standard NTP-style estimate)."""
        return (self.t_esp_receive_us + self.t_esp_send_us) / 2.0

    @property
    def pc_mid_ns(self) -> float:
        """Midpoint of the PC-side send/receive timestamps."""
        return (self.t_pc_send_ns + self.t_pc_receive_ns) / 2.0


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, stdlib-only (matches numpy's
    default `interpolation='linear'` behavior)."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    f = int(math.floor(k))
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


class ClockSyncEstimator:
    """Accumulates `SyncSample`s and fits `t_pc = a * t_esp + b` by
    ordinary least squares (manual sum-based OLS, no numpy)."""

    def __init__(self) -> None:
        self._samples: list[SyncSample] = []

    def add_sample(self, sample: SyncSample) -> None:
        self._samples.append(sample)

    def sample_count(self) -> int:
        return len(self._samples)

    def _fit(self) -> Optional[tuple[float, float]]:
        n = len(self._samples)
        if n < 2:
            return None
        xs = [s.esp_mid_us for s in self._samples]
        ys = [s.pc_mid_ns for s in self._samples]
        mean_x = statistics.fmean(xs)
        mean_y = statistics.fmean(ys)
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        sxx = sum((x - mean_x) ** 2 for x in xs)
        if sxx == 0:
            return None  # all esp timestamps identical -- slope undefined
        a = sxy / sxx
        b = mean_y - a * mean_x
        return a, b

    def report(self) -> dict:
        """Returns clock_offset/clock_drift/clock_sync_rmse/
        clock_sync_p95/rtt_mean_ns/rtt_p95_ns/sample_count. Offset/drift/
        rmse/p95 are `None` when fewer than 2 samples (or degenerate
        identical-x samples) make the fit undefined -- never fabricated."""
        n = len(self._samples)
        rtts = [float(s.rtt_ns) for s in self._samples]
        fit = self._fit()

        if fit is None:
            return {
                "clock_offset": None,
                "clock_drift": None,
                "clock_sync_rmse": None,
                "clock_sync_p95": None,
                "rtt_mean_ns": statistics.fmean(rtts) if rtts else None,
                "rtt_p95_ns": _percentile(rtts, 95) if rtts else None,
                "sample_count": n,
            }

        a, b = fit
        residuals = [s.pc_mid_ns - (a * s.esp_mid_us + b) for s in self._samples]
        rmse = math.sqrt(statistics.fmean(r * r for r in residuals))
        p95 = _percentile([abs(r) for r in residuals], 95)
        return {
            "clock_offset": b,
            "clock_drift": a,
            "clock_sync_rmse": rmse,
            "clock_sync_p95": p95,
            "rtt_mean_ns": statistics.fmean(rtts),
            "rtt_p95_ns": _percentile(rtts, 95),
            "sample_count": n,
        }


def _self_check() -> None:
    true_a, true_b = 2.0, 1000.0
    rtt = 2000
    est = ClockSyncEstimator()
    for seq, x in enumerate(range(0, 1000, 100)):
        target_mid = true_a * x + true_b
        t_pc_send = round(target_mid) - rtt // 2
        est.add_sample(
            SyncSample(
                seq=seq,
                t_pc_send_ns=t_pc_send,
                t_esp_receive_us=x,
                t_esp_send_us=x,
                t_pc_receive_ns=t_pc_send + rtt,
            )
        )
    report = est.report()
    assert report["sample_count"] == 10
    assert abs(report["clock_drift"] - true_a) < 1e-6
    assert abs(report["clock_offset"] - true_b) < 1e-6
    assert report["clock_sync_rmse"] < 1e-6
    assert report["rtt_mean_ns"] == rtt
    print("clock_sync.py self-check OK")


if __name__ == "__main__":
    _self_check()
