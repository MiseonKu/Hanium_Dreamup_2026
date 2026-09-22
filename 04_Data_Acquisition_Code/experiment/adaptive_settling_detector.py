"""Adaptive tip-IMU settling detector (master spec section 19).

Replaces a fixed 3-second post-motion wait with an online stability check on
the tip IMU (`imu3_tip`). No existing project policy defines this detector's
method or thresholds (DECISIONS.md's Korean-policy-doc audit found only a
fixed 0.5-1.0s post-motion dwell documented as a *different* experimental-
procedure default -- not an online detector). The closest precedent is
`07_WT901C/WT901C_imu0/live_4imu_visualizer.py`'s `update_imu3_stop_gate`
(gyro-magnitude-below-threshold-for-N-seconds, self-described as
provisional there) -- used here as design inspiration for the "debounce
timer" shape only; its exact numbers (3.0 dps / 0.5s) are NOT reused as
validated values.

ALL numeric thresholds in `SettlingConfig` are reasonable-but-UNVALIDATED
defaults, picked for this task's synthetic tests only. They must go through
a preregistration-style freeze against real tip-IMU noise-floor data before
being trusted for a real experiment run.

Metric formulas (deliberately simple, stdlib-only -- no numpy anywhere in
this repo's own requirements.txt, so none is added here either):

- tip_gyro_rms: per-sample 3-axis gyro magnitude ||gyro||_2, RMS'd across
  the window. A single scalar for "how much rotation energy is present",
  independent of which axis happens to be moving.
- dynamic_accel_rms: subtract the window's per-axis mean from each accel
  sample (a simple "subtract-the-window-mean" high-pass -- removes the
  gravity/DC component without building a real filter), then RMS the
  residual magnitude across the window.
- orientation_spread_rad: std-dev of each Euler channel (roll/pitch/yaw)
  across the window, combined via L2 norm into one scalar. Chosen over
  max-pairwise-angular-difference because it's O(n) instead of O(n^2) and
  avoids quaternion double-cover bookkeeping -- WT901C already emits Euler
  angles directly. Known limitation: does not unwrap angles crossing the
  +-180 deg boundary; not an issue for a near-static settling window.
- jerk_rms: finite-difference of consecutive accel samples divided by their
  time delta, magnitude RMS'd across the window -- the simplest correct
  approach per the task spec (no double-derivative-of-position estimate).
- drift_slope: ordinary-least-squares slope of the yaw (Euler z) channel
  vs. time across the window, same manual-OLS approach as
  `core/clock_sync.py` (one arbitrary but documented channel, per spec
  "linear-fit slope of one orientation channel").
- freshness: now - last_sample_t vs. `tip_expected_period_s`, flagged stale
  past `freshness_stale_multiplier` periods.

Frozen-sensor guard (master spec 19.3, the most important correctness
property here): a tip IMU that LOOKS stable because it's disconnected/
frozen (not because the arm is genuinely still) must never be declared
stable. `collectors/collect_imu_rs232.py` doesn't exist yet at the time
this module was built (concurrent workstream), so there is no
`quality_flags: ["stale_suspect"]` to consume -- this module does its own
lightweight check instead: if the last `frozen_min_repeats` samples' raw
payload (raw_hex, falling back to the parsed physical tuple when raw_hex
isn't available) are byte-for-byte identical, flag `not_frozen=False`. A
real sensor's ADC noise floor means bit-identical repeats essentially never
happen by chance.
"""
from __future__ import annotations

import math
import os
import statistics
import sys
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Sequence

# Needed for the `__main__` self-check at the bottom to run standalone.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import schemas  # noqa: E402
from core.bounded_queue import OVERFLOW, BoundedQueue  # noqa: E402


@dataclass
class SettlingConfig:
    """Every threshold here is config-driven per master spec ("Threshold는
    config로 분리한다") -- nothing in `AdaptiveSettlingDetector` hardcodes a
    number outside this dataclass. Defaults are reasonable-but-UNVALIDATED
    (see module docstring)."""

    tip_output_rate_hz: float = 100.0

    # rolling window used to compute the metrics below
    window_s: float = 0.3

    # tip-IMU primary metric thresholds (master spec section 19.2)
    tip_gyro_rms_threshold_rad_s: float = 0.05          # ~2.9 deg/s
    dynamic_accel_rms_threshold_mps2: float = 0.25
    orientation_spread_threshold_rad: float = 0.0175    # ~1 deg
    # Deliberately generous: a naive finite-difference derivative amplifies
    # a sensor's per-sample noise floor by 1/dt (e.g. a ~0.01g white-noise
    # floor differentiated at 100 Hz already RMS's to ~14 m/s^3 with zero
    # real motion) -- this is a real property of raw finite-differencing,
    # not just a mock artifact, so the threshold has to sit above the
    # noise-amplified floor rather than the true "signal" jerk. Needs
    # recalibration against a real WT901C noise spectrum (this mock's
    # settled-mode accel noise is likely pessimistic vs. real hardware).
    jerk_rms_threshold_mps3: float = 25.0
    drift_slope_threshold_rad_s: float = 0.0175          # ~1 deg/s trend

    # sensor freshness (time since last sample vs. expected period)
    tip_expected_period_s: float = 0.01                  # 1 / 100 Hz
    freshness_stale_multiplier: float = 5.0

    # much looser guards for imu0/imu1/imu2 (base + link, not the tip)
    base_gyro_threshold_rad_s: float = 0.35               # ~20 deg/s
    link_gyro_threshold_rad_s: float = 0.35

    # frozen-sensor guard
    frozen_min_repeats: int = 5

    # two duration checks, matching the state machine's two distinct states
    candidate_duration_s: float = 0.3
    confirmed_duration_s: float = 0.6

    # MEASUREMENT_WINDOW length -- master spec: 0.8-1.0s
    measurement_window_s: float = 0.9


@dataclass(frozen=True)
class TipSample:
    """One parsed tip-IMU (`imu3_tip`) sample. Deliberately shaped like the
    `data` sub-dict `core.schemas.imu_record` produces (plus `raw_hex` from
    the base record) -- this is the natural contract between
    `collectors/collect_imu_rs232.py` and this detector, decoupled from any
    wire-format parsing (that parser's job, not this detector's)."""

    t_pc_rx_ns: int
    gyro_rads: tuple
    accel_mps2: tuple
    euler_rad: Optional[tuple] = None
    raw_hex: Optional[str] = None
    source_seq: Optional[int] = None
    checksum_ok: bool = True
    stale_suspect: bool = False  # collector-flagged frozen/stale, see from_imu_record

    @staticmethod
    def from_imu_record(record: dict) -> "TipSample":
        """`collectors/collect_imu_rs232.py` runs its own sustained-byte-
        repeat check (per-frame-type run length -> `quality_flags:
        ["stale_suspect"]`) directly on the wire bytes, upstream of this
        detector. Consumed here directly per master spec 19.3, in addition
        to (not instead of) this module's own window-based repeat check in
        `_frozen_suspect` -- the two operate on different data (raw
        per-frame-type bytes over the whole session vs. reconstructed
        physical samples over the detector's rolling window), so keeping
        both costs nothing and closes more of the failure-mode gap."""
        data = record["data"]
        return TipSample(
            t_pc_rx_ns=record["t_pc_rx_ns"],
            gyro_rads=tuple(data["gyro_rads"]),
            accel_mps2=tuple(data["accel_mps2"]),
            euler_rad=tuple(data["euler_rad"]) if data.get("euler_rad") is not None else None,
            raw_hex=record.get("raw_hex"),
            source_seq=record.get("source_seq"),
            checksum_ok=bool(record.get("checksum_ok", True)),
            stale_suspect="stale_suspect" in (record.get("quality_flags") or []),
        )


@dataclass
class MetricsResult:
    ready: bool  # False until the window holds enough history to trust the numbers
    tip_gyro_rms: Optional[float] = None
    dynamic_accel_rms: Optional[float] = None
    orientation_spread_rad: Optional[float] = None
    jerk_rms: Optional[float] = None
    drift_slope: Optional[float] = None
    freshness_ok: bool = True
    frozen_suspect: bool = False
    within_threshold: bool = False


@dataclass
class SettlingEvaluation:
    metrics: MetricsResult
    guard_flags: dict
    overall_pass: bool
    stable_candidate_duration_s: float
    stability_margin: float
    is_candidate: bool
    is_confirmed: bool


def _magnitude(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


class AdaptiveSettlingDetector:
    def __init__(self, config: SettlingConfig):
        self.config = config
        self._window: Deque[TipSample] = deque()
        self._good_since_ns: Optional[int] = None
        self._good_duration_s: float = 0.0

    def reset(self) -> None:
        """Clear rolling window + debounce timer -- call at the start of
        every new target/trial (`ExperimentStateMachine.start_new_target`)."""
        self._window.clear()
        self._good_since_ns = None
        self._good_duration_s = 0.0

    def add_tip_sample(self, sample: TipSample) -> None:
        self._window.append(sample)
        cutoff = sample.t_pc_rx_ns - int(self.config.window_s * 1e9)
        while self._window and self._window[0].t_pc_rx_ns < cutoff:
            self._window.popleft()

    # -- metrics -------------------------------------------------------
    def _compute_metrics(self, now_ns: int) -> MetricsResult:
        window = list(self._window)
        min_samples = max(4, int(self.config.window_s * self.config.tip_output_rate_hz * 0.5))
        if len(window) < min_samples:
            return MetricsResult(ready=False, within_threshold=False, freshness_ok=False)

        gyro_rms = math.sqrt(statistics.fmean(_magnitude(s.gyro_rads) ** 2 for s in window))

        mean_ax = statistics.fmean(s.accel_mps2[0] for s in window)
        mean_ay = statistics.fmean(s.accel_mps2[1] for s in window)
        mean_az = statistics.fmean(s.accel_mps2[2] for s in window)
        dyn_accel_rms = math.sqrt(
            statistics.fmean(
                (s.accel_mps2[0] - mean_ax) ** 2
                + (s.accel_mps2[1] - mean_ay) ** 2
                + (s.accel_mps2[2] - mean_az) ** 2
                for s in window
            )
        )

        if all(s.euler_rad is not None for s in window):
            roll = [s.euler_rad[0] for s in window]
            pitch = [s.euler_rad[1] for s in window]
            yaw = [s.euler_rad[2] for s in window]
            std_roll = statistics.pstdev(roll)
            std_pitch = statistics.pstdev(pitch)
            std_yaw = statistics.pstdev(yaw)
            orientation_spread = math.sqrt(std_roll ** 2 + std_pitch ** 2 + std_yaw ** 2)
            drift_slope = self._linear_fit_slope(
                [s.t_pc_rx_ns for s in window], yaw
            )
        else:
            orientation_spread = None
            drift_slope = None

        jerk_samples = []
        for prev, cur in zip(window, window[1:]):
            dt = (cur.t_pc_rx_ns - prev.t_pc_rx_ns) / 1e9
            if dt <= 0:
                continue
            jerk_vec = tuple((c - p) / dt for c, p in zip(cur.accel_mps2, prev.accel_mps2))
            jerk_samples.append(_magnitude(jerk_vec))
        jerk_rms = math.sqrt(statistics.fmean(j * j for j in jerk_samples)) if jerk_samples else None

        last = window[-1]
        gap_s = (now_ns - last.t_pc_rx_ns) / 1e9
        freshness_ok = gap_s <= self.config.freshness_stale_multiplier * self.config.tip_expected_period_s

        frozen_suspect = self._frozen_suspect()

        checks = [
            gyro_rms <= self.config.tip_gyro_rms_threshold_rad_s,
            dyn_accel_rms <= self.config.dynamic_accel_rms_threshold_mps2,
            orientation_spread is None or orientation_spread <= self.config.orientation_spread_threshold_rad,
            jerk_rms is None or jerk_rms <= self.config.jerk_rms_threshold_mps3,
            drift_slope is None or abs(drift_slope) <= self.config.drift_slope_threshold_rad_s,
        ]
        within_threshold = all(checks) and freshness_ok and not frozen_suspect

        return MetricsResult(
            ready=True,
            tip_gyro_rms=gyro_rms,
            dynamic_accel_rms=dyn_accel_rms,
            orientation_spread_rad=orientation_spread,
            jerk_rms=jerk_rms,
            drift_slope=drift_slope,
            freshness_ok=freshness_ok,
            frozen_suspect=frozen_suspect,
            within_threshold=within_threshold,
        )

    def _frozen_suspect(self) -> bool:
        if self._window and self._window[-1].stale_suspect:
            # collect_imu_rs232.py already ran its own sustained-byte-repeat
            # check upstream -- trust it directly, no need to wait for our
            # own window to fill.
            return True
        n = self.config.frozen_min_repeats
        if len(self._window) < n:
            return False
        recent = list(self._window)[-n:]
        if all(s.raw_hex is not None for s in recent):
            keys = [s.raw_hex for s in recent]
        else:
            keys = [(s.gyro_rads, s.accel_mps2, s.euler_rad) for s in recent]
        return len(set(keys)) == 1

    @staticmethod
    def _linear_fit_slope(t_ns: Sequence[int], y: Sequence[float]) -> Optional[float]:
        """Manual-OLS slope of y vs. time-in-seconds (same sum-based
        approach as `core/clock_sync.py`, stdlib only)."""
        t_s = [(t - t_ns[0]) / 1e9 for t in t_ns]
        mean_t = statistics.fmean(t_s)
        mean_y = statistics.fmean(y)
        sxy = sum((t - mean_t) * (v - mean_y) for t, v in zip(t_s, y))
        sxx = sum((t - mean_t) ** 2 for t in t_s)
        if sxx == 0:
            return None
        return sxy / sxx

    def _metric_margin(self, m: MetricsResult) -> float:
        """min over the 5 threshold checks of (threshold - value) / threshold
        -- positive means comfortably inside every threshold, <=0 means at
        least one metric is violating its threshold. Deliberately ignores
        guard state; `evaluate()` clamps it separately when a guard fails,
        so a frozen sensor (near-zero metrics -> huge metric_margin) still
        reports a non-positive `stability_margin` overall."""
        if not m.ready:
            return -1.0
        pairs = [
            (m.tip_gyro_rms, self.config.tip_gyro_rms_threshold_rad_s),
            (m.dynamic_accel_rms, self.config.dynamic_accel_rms_threshold_mps2),
            (m.orientation_spread_rad, self.config.orientation_spread_threshold_rad),
            (m.jerk_rms, self.config.jerk_rms_threshold_mps3),
        ]
        margins = [(t - v) / t for v, t in pairs if v is not None and t > 0]
        if m.drift_slope is not None and self.config.drift_slope_threshold_rad_s > 0:
            t = self.config.drift_slope_threshold_rad_s
            margins.append((t - abs(m.drift_slope)) / t)
        return min(margins) if margins else -1.0

    # -- guards + top-level evaluate ------------------------------------
    def evaluate(
        self,
        now_ns: int,
        *,
        encoder_settled: bool,
        base_gyro_rad_s: Optional[float] = None,
        link_root_gyro_rad_s: Optional[float] = None,
        link_mid_gyro_rad_s: Optional[float] = None,
        is_clock_sync_valid: Callable[[], bool] = lambda: True,
        queues: Sequence[BoundedQueue] = (),
        has_motor_fault: Callable[[], bool] = lambda: False,
    ) -> SettlingEvaluation:
        """Run the tip-IMU metrics plus every guard (checklist, per-guard
        pass/fail -- not one boolean) and advance the candidate/confirmed
        debounce timer.

        Guards:
          encoder_settled       -- from `EncoderArrivalGate` (caller-owned)
          base_imu_stable       -- imu0_base, much looser than tip
          link_imu_no_vibration -- imu1/imu2, much looser than tip
          tip_freshness_ok      -- from the metrics computation above
          clock_sync_valid      -- injectable, defaults True (no real ESP32
                                    clock sync in this task)
          queue_ok              -- any monitored `BoundedQueue` in OVERFLOW
                                    blocks stable declaration
          motor_fault_ok        -- injectable, defaults no-fault
          not_frozen            -- master spec 19.3, see module docstring
        """
        metrics = self._compute_metrics(now_ns)

        queue_overflow = any(q.depth_state() == OVERFLOW for q in queues)

        guard_flags = {
            "encoder_settled": bool(encoder_settled),
            "base_imu_stable": base_gyro_rad_s is None
            or base_gyro_rad_s <= self.config.base_gyro_threshold_rad_s,
            "link_imu_no_vibration": (
                link_root_gyro_rad_s is None
                or link_root_gyro_rad_s <= self.config.link_gyro_threshold_rad_s
            )
            and (
                link_mid_gyro_rad_s is None
                or link_mid_gyro_rad_s <= self.config.link_gyro_threshold_rad_s
            ),
            "tip_freshness_ok": metrics.freshness_ok,
            "clock_sync_valid": bool(is_clock_sync_valid()),
            "queue_ok": not queue_overflow,
            "motor_fault_ok": not bool(has_motor_fault()),
            "not_frozen": not metrics.frozen_suspect,
        }
        overall_pass = metrics.within_threshold and all(guard_flags.values())

        if overall_pass:
            if self._good_since_ns is None:
                self._good_since_ns = now_ns
            self._good_duration_s = (now_ns - self._good_since_ns) / 1e9
        else:
            self._good_since_ns = None
            self._good_duration_s = 0.0

        metric_margin = self._metric_margin(metrics)
        stability_margin = metric_margin if all(guard_flags.values()) else min(metric_margin, 0.0)

        return SettlingEvaluation(
            metrics=metrics,
            guard_flags=guard_flags,
            overall_pass=overall_pass,
            stable_candidate_duration_s=self._good_duration_s,
            stability_margin=stability_margin,
            is_candidate=self._good_duration_s >= self.config.candidate_duration_s,
            is_confirmed=self._good_duration_s >= self.config.confirmed_duration_s,
        )


def stability_record_from_evaluation(base: dict, evaluation: SettlingEvaluation, settling_state: str) -> dict:
    """Build a `core.schemas.stability_record` from one evaluation. `base`
    must already have `schema=SCHEMA_STABILITY, stream=STREAM_STABILITY`."""
    m = evaluation.metrics
    return schemas.stability_record(
        base,
        tip_gyro_rms=m.tip_gyro_rms,
        dynamic_accel_rms=m.dynamic_accel_rms,
        orientation_spread_rad=m.orientation_spread_rad,
        jerk_rms=m.jerk_rms,
        drift_slope=m.drift_slope,
        stable_candidate_duration_s=evaluation.stable_candidate_duration_s,
        stability_margin=evaluation.stability_margin,
        settling_state=settling_state,
        guard_flags=evaluation.guard_flags,
    )


def _self_check() -> None:
    config = SettlingConfig(window_s=0.2, candidate_duration_s=0.1, confirmed_duration_s=0.2)
    detector = AdaptiveSettlingDetector(config)
    now = 0
    period_ns = int(1e9 / config.tip_output_rate_hz)

    # feed a perfectly still synthetic sample stream -- should ready up and
    # eventually confirm.
    confirmed = False
    for i in range(60):
        now += period_ns
        sample = TipSample(
            t_pc_rx_ns=now,
            gyro_rads=(0.001 * ((i % 3) - 1), 0.0, 0.0),
            accel_mps2=(0.0, 0.0, 9.81),
            euler_rad=(0.0, 0.0, 0.0),
            raw_hex=f"{i:08x}",  # changes every sample -- not frozen
        )
        detector.add_tip_sample(sample)
        ev = detector.evaluate(now, encoder_settled=True)
        if ev.is_confirmed:
            confirmed = True
            break
    assert confirmed, "still synthetic stream never confirmed stable"
    assert ev.stability_margin >= 0.0

    # a guard failure must reset the duration timer to 0 and clamp margin.
    now += period_ns
    ev2 = detector.evaluate(now, encoder_settled=False)
    assert ev2.stable_candidate_duration_s == 0.0
    assert ev2.stability_margin <= 0.0

    print("adaptive_settling_detector.py self-check OK")


if __name__ == "__main__":
    _self_check()
