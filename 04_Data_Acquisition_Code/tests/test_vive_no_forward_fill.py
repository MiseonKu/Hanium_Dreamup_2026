"""Proves `collectors/collect_vive_tracker.py` never forward-fills a lost
Vive pose (DECISIONS.md #6, master spec 1.3/17.2): every sample is written,
including tracking-lost ones, with null pose fields -- never the last-good
pose copied forward, and never skipped.

Run with:
    python -m pytest tests -q
from the `04_Data_Acquisition_Code/` directory.
"""
from __future__ import annotations

import os
import sys
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

from collectors.collect_vive_tracker import ViveTrackerCollector  # noqa: E402
from mocks.mock_vive import MockViveTracker  # noqa: E402


class ViveNoForwardFillTests(unittest.TestCase):
    def test_valid_lost_valid_sequence_never_copies_last_good_pose(self):
        vt = MockViveTracker(seed=1)
        collector = ViveTrackerCollector(vt, session_id="session_test", segment_id=0)

        valid1 = collector.read_record()
        self.assertTrue(valid1["valid"])
        self.assertIsNotNone(valid1["data"]["position_m"])
        last_good_position = valid1["data"]["position_m"]
        last_good_quat = valid1["data"]["quaternion_xyzw"]

        vt.set_tracking_lost(True)
        lost_records = [collector.read_record() for _ in range(3)]

        for i, rec in enumerate(lost_records):
            self.assertFalse(rec["valid"], f"lost record {i} must have valid=False")
            self.assertEqual(rec["parse_status"], "tracking_lost")
            self.assertFalse(rec["data"]["tracking_valid"])
            # the hard rule under test: null, never the previous good pose.
            self.assertIsNone(rec["data"]["position_m"], f"lost record {i} position_m must be null")
            self.assertIsNone(rec["data"]["quaternion_xyzw"], f"lost record {i} quaternion_xyzw must be null")
            self.assertIsNone(rec["data"]["pose_matrix_3x4"], f"lost record {i} pose_matrix_3x4 must be null")
            self.assertNotEqual(rec["data"]["position_m"], last_good_position)
            self.assertNotEqual(rec["data"]["quaternion_xyzw"], last_good_quat)

        vt.set_tracking_lost(False)
        valid2 = collector.read_record()
        self.assertTrue(valid2["valid"])
        self.assertEqual(valid2["parse_status"], "ok")
        self.assertIsNotNone(valid2["data"]["position_m"])
        # recovered pose is a fresh sample, not required to equal the old one
        # (mock's motion model keeps advancing) -- what matters is it's real.
        self.assertEqual(len(valid2["data"]["position_m"]), 3)

    def test_lost_records_are_written_not_skipped(self):
        vt = MockViveTracker(seed=2)
        vt.set_tracking_lost(True)
        collector = ViveTrackerCollector(vt, session_id="session_test")

        records = [collector.read_record() for _ in range(5)]

        # host_seq must be a clean, unbroken 1..5 -- no record was dropped
        # just because tracking was invalid the whole time.
        self.assertEqual([r["host_seq"] for r in records], [1, 2, 3, 4, 5])
        for rec in records:
            self.assertEqual(rec["schema"], "alphaf.vive.raw.v1")
            self.assertEqual(rec["stream"], "vive_tracker")
            self.assertFalse(rec["data"]["tracking_valid"])

    def test_disconnected_device_also_produces_null_not_stale_record(self):
        vt = MockViveTracker(seed=3)
        collector = ViveTrackerCollector(vt, session_id="session_test")
        collector.read_record()  # one good sample first

        vt.set_device_connected(False)
        rec = collector.read_record()
        self.assertFalse(rec["valid"])
        self.assertEqual(rec["parse_status"], "device_disconnected")
        self.assertIsNone(rec["data"]["position_m"])

    def test_source_exception_is_retried_not_crashed_or_faked(self):
        calls = {"n": 0}
        errors_seen = []

        class _FlakyThenGood:
            def next_sample(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("openvr hiccup")
                return {
                    "tracker_serial": "LHR-X",
                    "device_index": 1,
                    "tracking_result": "TrackingResult_Running_OK",
                    "tracking_valid": True,
                    "device_connected": True,
                    "tracking_origin": "TrackingUniverseStanding",
                    "pose_matrix_3x4": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                    "position_m": [0.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "source_timestamp": 1.0,
                }

        collector = ViveTrackerCollector(
            _FlakyThenGood(),
            session_id="session_test",
            retry_wait_s=0.001,
            on_error=lambda exc: errors_seen.append(exc),
        )
        rec = collector.read_record()

        self.assertEqual(len(errors_seen), 1)
        self.assertIsInstance(errors_seen[0], RuntimeError)
        self.assertTrue(rec["valid"])
        self.assertEqual(rec["host_seq"], 1)  # the failed attempt didn't consume a host_seq


if __name__ == "__main__":
    unittest.main()
