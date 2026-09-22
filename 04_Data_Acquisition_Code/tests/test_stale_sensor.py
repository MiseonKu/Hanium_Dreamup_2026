"""Tests for `collectors/collect_imu_rs232.py`'s stale/frozen-sensor
detection (`quality_flags: ["stale_suspect"]`) -- the input signal for the
adaptive settling detector's "frozen sensor" guard (master spec 19.3),
built by someone else; this collector only needs to flag it.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)
sys.path.insert(0, os.path.join(WORKSPACE, "mocks"))

from collectors.collect_imu_rs232 import IMURS232Collector  # noqa: E402
from mock_imu import MockWT901CImu  # noqa: E402
from fault_injector import with_frozen_values  # noqa: E402


class StaleSensorTests(unittest.TestCase):
    def _run_collector(self, byte_source, stale_run_length, n_frames):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        collector = IMURS232Collector(
            sensor_id="imu3_tip",
            session_id="session_stale_test",
            segment_id=0,
            output_dir=tmp.name,
            byte_source=byte_source,
            stale_run_length=stale_run_length,
            queue_expected_rate_hz=400.0,
        )
        collector.start()
        try:
            deadline = time.time() + 8.0
            while collector.frames_emitted < n_frames and time.time() < deadline:
                time.sleep(0.01)
        finally:
            collector.stop()
        self.assertGreaterEqual(collector.frames_emitted, n_frames, collector.stats())
        with open(collector.writer.final_path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]

    def test_frozen_sensor_sets_stale_suspect_after_threshold(self):
        imu = MockWT901CImu(seed=11, moving=True)
        frozen = with_frozen_values(imu, after_n=20)
        records = self._run_collector(frozen.next_packet_bytes, stale_run_length=6, n_frames=150)

        flagged = [r for r in records if "stale_suspect" in r["quality_flags"]]
        self.assertTrue(flagged, "expected stale_suspect to fire once frozen for long enough")

        # None of the early (pre-freeze, still-moving) records should be flagged.
        early = records[:15]
        self.assertFalse(
            any("stale_suspect" in r["quality_flags"] for r in early),
            "must not flag stale before the sensor actually freezes",
        )

    def test_normal_moving_stream_never_flags_stale(self):
        imu = MockWT901CImu(seed=12, moving=True)
        records = self._run_collector(imu.next_packet_bytes, stale_run_length=6, n_frames=150)

        flagged = [r for r in records if "stale_suspect" in r["quality_flags"]]
        self.assertEqual(flagged, [], "a normal moving stream must never be flagged stale")

    def test_settled_but_not_frozen_stream_does_not_false_positive(self):
        # "settled" (near-zero noise, not literally frozen) must not trip
        # the exact-byte-repeat freeze heuristic either -- the mock's noise
        # model perturbs every sample slightly, so consecutive same-type
        # raw bytes should essentially never be bit-identical.
        imu = MockWT901CImu(seed=13, moving=False)
        records = self._run_collector(imu.next_packet_bytes, stale_run_length=6, n_frames=150)

        flagged = [r for r in records if "stale_suspect" in r["quality_flags"]]
        self.assertEqual(flagged, [], "settled-but-still-noisy data must not be mistaken for frozen")


if __name__ == "__main__":
    unittest.main()
