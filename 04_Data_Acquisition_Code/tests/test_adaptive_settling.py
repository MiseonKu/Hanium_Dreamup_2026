"""Tests for the adaptive tip-IMU settling detector (experiment/).

Plain stdlib unittest, no fixtures/frameworks -- same style as
`tests/test_mocks.py` / `tests/test_queue_overflow.py`.

Run with:
    python -m pytest tests -q
from the `04_Data_Acquisition_Code/` directory.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)
sys.path.insert(0, os.path.join(WORKSPACE, "mocks"))

from core.bounded_queue import BoundedQueue  # noqa: E402
from mock_imu import MockWT901CImu  # noqa: E402
from fault_injector import with_frozen_values  # noqa: E402

from experiment.adaptive_settling_detector import (  # noqa: E402
    AdaptiveSettlingDetector,
    SettlingConfig,
    TipSample,
)
from experiment.experiment_state_machine import (  # noqa: E402
    EncoderArrivalGate,
    EncoderGateConfig,
    ExperimentStateMachine,
)
from experiment.safety_gate import (  # noqa: E402
    assert_hardware_motion_allowed,
    hardware_motion_allowed,
)

PERIOD_NS = int(1e9 / 100.0)  # matches MockWT901CImu's default 100 Hz


def _decode_wt901_frame(frame: bytes):
    """Inverse of `mocks/mock_imu.py`'s documented physical->raw scaling,
    used only to turn the mock's byte stream back into physical values for
    these tests (real parsing is `collectors/collect_imu_rs232.py`'s job,
    a separate concurrent workstream -- not duplicated here)."""
    ftype = frame[1]
    vals = struct.unpack("<4h", frame[2:10])
    if ftype == 0x51:  # accel: g -> m/s^2
        return "accel", [v / 32768.0 * 16.0 * 9.80665 for v in vals[:3]]
    if ftype == 0x52:  # gyro: dps -> rad/s
        return "gyro", [math.radians(v / 32768.0 * 2000.0) for v in vals[:3]]
    if ftype == 0x53:  # angle: deg -> rad
        return "angle", [math.radians(v / 32768.0 * 180.0) for v in vals[:3]]
    if ftype == 0x59:  # quaternion, unused by these tests
        return "quat", [v / 32768.0 for v in vals]
    raise ValueError(f"unknown WT901C frame type {ftype:#x}")


def _sample_from_mock(mock, t_ns: int, seq: int) -> TipSample:
    """Pull one full 4-frame sample (accel/gyro/angle/quat) from a
    `MockWT901CImu` (or a `fault_injector`-wrapped one) and build a
    `TipSample`, including the concatenated raw hex so the frozen-sensor
    guard has something to compare."""
    frames = [mock.next_packet_bytes() for _ in range(4)]
    raw_hex = "".join(f.hex() for f in frames)
    data = dict(_decode_wt901_frame(f) for f in frames)
    return TipSample(
        t_pc_rx_ns=t_ns,
        gyro_rads=tuple(data["gyro"]),
        accel_mps2=tuple(data["accel"]),
        euler_rad=tuple(data["angle"]),
        raw_hex=raw_hex,
        source_seq=seq,
        checksum_ok=True,
    )


def _fast_config(**overrides) -> SettlingConfig:
    """Shrunk durations so tests run in a small, deterministic number of
    steps -- the numeric thresholds themselves stay at the module's
    reasonable-but-unvalidated defaults."""
    base = dict(window_s=0.2, candidate_duration_s=0.1, confirmed_duration_s=0.2, measurement_window_s=0.15)
    base.update(overrides)
    return SettlingConfig(**base)


# This rig is 3-axis by design: shoulder_pan, shoulder_lift, wrist_flex
# (aliases D2/D3/D5, servo ids 1/2/4) -- see
# docs/THREE_AXIS_EXPERIMENT_SCOPE.md. 5-element joint vectors are the
# other (never-flashed) firmware's shape, not this one's.
N_JOINTS = 3


def _drive_to_encoder_arrival(sm: ExperimentStateMachine, now: int, ticks: int = 10) -> int:
    for _ in range(ticks):
        now += PERIOD_NS
        sm.update_encoder(now, [0.0] * N_JOINTS, [0.0] * N_JOINTS, [0.0] * N_JOINTS)
    return now


class TestEncoderArrivalGateInputGuards(unittest.TestCase):
    """`zip` truncates to the shortest sequence and `all([])` is True, so a
    mismatched or empty joint vector would silently declare "arrived"
    without ever checking the missing joints."""

    def _gate(self):
        return EncoderArrivalGate(EncoderGateConfig(dwell_s=0.0))

    def test_mismatched_vector_lengths_raise_instead_of_truncating(self):
        gate = self._gate()
        with self.assertRaises(ValueError):
            # 3 encoder readings vs a stale 5-element command vector: the
            # last two commanded joints would never have been compared.
            gate.update(0, [0.0] * 3, [0.0] * 5, [0.0] * 3)
        with self.assertRaises(ValueError):
            gate.update(0, [0.0] * 3, [0.0] * 3, [0.0] * 5)

    def test_empty_vectors_never_declare_arrival(self):
        gate = self._gate()
        with self.assertRaises(ValueError):
            gate.update(0, [], [], [])
        self.assertFalse(gate.arrived)

    def test_unmapped_axis_null_is_not_arrival(self):
        """A firmware `logical_deg=NA` axis parses to None. Unknown
        position must never satisfy the tolerance check."""
        gate = self._gate()
        self.assertFalse(gate.update(0, [0.0, None, 0.0], [0.0] * 3, [0.0] * 3))
        self.assertFalse(gate.arrived)
        # velocity still unknown (first encoder sample) is also not arrival
        self.assertFalse(gate.update(0, [0.0] * 3, [0.0] * 3, [None, 0.0, 0.0]))
        self.assertFalse(gate.arrived)
        # ...and a fully-known, in-tolerance vector still arrives normally
        self.assertTrue(gate.update(0, [0.0] * 3, [0.0] * 3, [0.0] * 3))


class TestMovingThenSettles(unittest.TestCase):
    def test_reaches_confirmed_then_measurement_accepted(self):
        config = _fast_config()
        detector = AdaptiveSettlingDetector(config)
        gate = EncoderArrivalGate(EncoderGateConfig(dwell_s=0.05))
        sm = ExperimentStateMachine(detector, gate)

        now = 0
        sm.start_new_target(now)
        now = _drive_to_encoder_arrival(sm, now)
        self.assertEqual(sm.current_state, "ADAPTIVE_SETTLING_MONITOR")

        imu = MockWT901CImu(seed=1, moving=True, motion_amplitude_dps=90.0)
        seq = 0
        for _ in range(30):
            now += PERIOD_NS
            seq += 1
            sm.update_tip_imu(now, _sample_from_mock(imu, now, seq))
        # still moving -- must not have escalated past monitoring/candidate
        self.assertIn(sm.current_state, ("ADAPTIVE_SETTLING_MONITOR", "STABLE_CANDIDATE"))

        imu.set_moving(False)
        reached_confirmed = False
        reached_accepted = False
        for _ in range(500):
            now += PERIOD_NS
            seq += 1
            sm.update_tip_imu(now, _sample_from_mock(imu, now, seq))
            if sm.current_state == "TIP_IMU_STABLE_CONFIRMED":
                reached_confirmed = True
            if sm.current_state == "MEASUREMENT_ACCEPTED":
                reached_accepted = True
                break

        self.assertTrue(reached_confirmed, "never reached TIP_IMU_STABLE_CONFIRMED after settling")
        self.assertTrue(reached_accepted, f"never reached MEASUREMENT_ACCEPTED, ended at {sm.current_state}")
        self.assertIsNotNone(sm.settle_time_from_encoder_s)
        self.assertGreaterEqual(sm.settle_time_from_encoder_s, 0.0)


class TestFrozenSensorNeverStable(unittest.TestCase):
    def test_frozen_tip_imu_is_never_declared_stable(self):
        """The single most important test in this file (master spec 19.3):
        a tip IMU that looks stable only because it's frozen/disconnected
        must never be declared stable, even though its (frozen) noise
        floor is objectively lower than a genuinely-settled live sensor."""
        config = _fast_config()
        detector = AdaptiveSettlingDetector(config)
        gate = EncoderArrivalGate(EncoderGateConfig(dwell_s=0.05))
        sm = ExperimentStateMachine(detector, gate)

        now = 0
        sm.start_new_target(now)
        now = _drive_to_encoder_arrival(sm, now)
        self.assertEqual(sm.current_state, "ADAPTIVE_SETTLING_MONITOR")

        raw_imu = MockWT901CImu(seed=1, moving=False)  # already near-zero noise
        frozen = with_frozen_values(raw_imu, after_n=8)  # freezes after 2 full samples

        seq = 0
        reached_stable = False
        for _ in range(500):
            now += PERIOD_NS
            seq += 1
            sm.update_tip_imu(now, _sample_from_mock(frozen, now, seq))
            if sm.current_state in ("TIP_IMU_STABLE_CONFIRMED", "MEASUREMENT_WINDOW", "MEASUREMENT_ACCEPTED"):
                reached_stable = True
                break

        self.assertFalse(reached_stable, f"frozen sensor was falsely declared stable, state={sm.current_state}")
        self.assertIsNotNone(sm.last_evaluation)
        self.assertFalse(sm.last_evaluation.guard_flags["not_frozen"], "not_frozen guard should have tripped")


class TestMeasurementWindowBreaksMidway(unittest.TestCase):
    def test_motion_burst_during_measurement_window_discards_and_returns_to_monitoring(self):
        config = _fast_config(measurement_window_s=0.3)  # give the burst room to land inside the window
        detector = AdaptiveSettlingDetector(config)
        gate = EncoderArrivalGate(EncoderGateConfig(dwell_s=0.05))
        sm = ExperimentStateMachine(detector, gate)

        now = 0
        sm.start_new_target(now)
        now = _drive_to_encoder_arrival(sm, now)

        imu = MockWT901CImu(seed=3, moving=False)
        seq = 0
        entered_window = False
        for _ in range(300):
            now += PERIOD_NS
            seq += 1
            sm.update_tip_imu(now, _sample_from_mock(imu, now, seq))
            if sm.current_state == "MEASUREMENT_WINDOW":
                entered_window = True
                break
        self.assertTrue(entered_window, "never entered MEASUREMENT_WINDOW to begin with")

        # inject a motion burst partway through the window -- must discard,
        # not crash, not falsely accept.
        imu.set_moving(True)
        for _ in range(5):
            now += PERIOD_NS
            seq += 1
            state = sm.update_tip_imu(now, _sample_from_mock(imu, now, seq))

        self.assertEqual(state, "ADAPTIVE_SETTLING_MONITOR", "stability break mid-window must return to monitoring")
        self.assertNotEqual(state, "MEASUREMENT_ACCEPTED")


class TestDurationAndMarginResetOnGuardFailure(unittest.TestCase):
    """Master spec section 23: 'adaptive settling reset/confirmation' --
    `stable_candidate_duration_s`/`stability_margin` must climb while
    genuinely stable and reset the instant any guard fails."""

    def test_duration_climbs_then_resets_on_guard_failure(self):
        config = _fast_config()
        detector = AdaptiveSettlingDetector(config)
        imu = MockWT901CImu(seed=2, moving=False)

        now = 0
        seq = 0
        durations = []
        evaluation = None
        for _ in range(40):
            now += PERIOD_NS
            seq += 1
            detector.add_tip_sample(_sample_from_mock(imu, now, seq))
            evaluation = detector.evaluate(now, encoder_settled=True)
            durations.append(evaluation.stable_candidate_duration_s)

        # once metrics become "ready" the duration must be monotonically
        # non-decreasing while continuously stable.
        ready_durations = [d for d in durations if d > 0.0] or durations
        self.assertTrue(
            all(b >= a - 1e-9 for a, b in zip(ready_durations, ready_durations[1:])),
            f"duration did not climb monotonically while stable: {ready_durations}",
        )
        self.assertGreater(evaluation.stable_candidate_duration_s, 0.0)
        self.assertGreaterEqual(evaluation.stability_margin, 0.0)

        # force a guard failure (encoder not settled) -- duration and margin
        # must reset even though the tip IMU data itself hasn't changed.
        now += PERIOD_NS
        ev_fail = detector.evaluate(now, encoder_settled=False)
        self.assertEqual(ev_fail.stable_candidate_duration_s, 0.0)
        self.assertLessEqual(ev_fail.stability_margin, 0.0)

        # recovering lets it start climbing again from zero.
        now += PERIOD_NS
        ev_recover = detector.evaluate(now, encoder_settled=True)
        self.assertGreaterEqual(ev_recover.stable_candidate_duration_s, 0.0)
        self.assertLessEqual(ev_recover.stable_candidate_duration_s, PERIOD_NS / 1e9 + 1e-6)

    def test_queue_overflow_guard_blocks_via_bounded_queue(self):
        """The queue-overflow guard checks a real `BoundedQueue.depth_state()`,
        not just a boolean flag."""
        config = _fast_config()
        detector = AdaptiveSettlingDetector(config)
        imu = MockWT901CImu(seed=4, moving=False)

        overflowed_queue = BoundedQueue(expected_rate_hz=10, buffer_seconds=1.0)  # maxsize=10
        for i in range(10):
            overflowed_queue.put_nowait_tracked(i, seq=i)  # fill to OVERFLOW

        now = 0
        seq = 0
        for _ in range(40):
            now += PERIOD_NS
            seq += 1
            detector.add_tip_sample(_sample_from_mock(imu, now, seq))
            ev = detector.evaluate(now, encoder_settled=True, queues=[overflowed_queue])

        self.assertFalse(ev.guard_flags["queue_ok"])
        self.assertEqual(ev.stable_candidate_duration_s, 0.0)


class TestSafetyGate(unittest.TestCase):
    def test_blocks_unless_both_flags_set(self):
        self.assertFalse(hardware_motion_allowed(False, False))
        self.assertFalse(hardware_motion_allowed(True, False))
        self.assertFalse(hardware_motion_allowed(False, True))
        self.assertTrue(hardware_motion_allowed(True, True))

    def test_assert_variant_raises_unless_both_set(self):
        with self.assertRaises(PermissionError):
            assert_hardware_motion_allowed(True, False)
        with self.assertRaises(PermissionError):
            assert_hardware_motion_allowed(False, True)
        with self.assertRaises(PermissionError):
            assert_hardware_motion_allowed(False, False)
        assert_hardware_motion_allowed(True, True)  # must not raise


if __name__ == "__main__":
    unittest.main()
