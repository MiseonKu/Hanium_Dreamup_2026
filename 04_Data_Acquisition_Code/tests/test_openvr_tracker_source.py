"""Tests `collectors/collect_vive_tracker.py`'s `OpenVRTrackerSource` --
the real SteamVR/OpenVR-backed `VivePoseSource` (Task 2). Uses a fake stub
`openvr` module injected via `sys.modules`, never the real package, so
this suite never touches an actual SteamVR runtime and passes with or
without `openvr` installed.

Run with:
    python -m pytest tests -q
from the `04_Data_Acquisition_Code/` directory.
"""
from __future__ import annotations

import os
import sys
import types
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)


def _make_fake_openvr(devices, init_error=None):
    """`devices`: list of (device_class, serial_or_None, pose_or_None).
    `pose_or_None` is (position_m, bPoseIsValid, bDeviceIsConnected,
    eTrackingResult) or None to mean "not GenericTracker, irrelevant"."""
    fake = types.ModuleType("openvr")
    fake.VRApplication_Other = 1
    fake.TrackedDeviceClass_GenericTracker = 3
    fake.k_unMaxTrackedDeviceCount = len(devices)
    fake.Prop_SerialNumber_String = 1001
    fake.TrackingUniverseStanding = 0
    fake.TrackingResult_Uninitialized = 1
    fake.TrackingResult_Calibrating_InProgress = 100
    fake.TrackingResult_Calibrating_OutOfRange = 101
    fake.TrackingResult_Running_OK = 200
    fake.TrackingResult_Running_OutOfRange = 201

    class _Pose:
        def __init__(self, position_m, valid, connected, result):
            self.bPoseIsValid = valid
            self.bDeviceIsConnected = connected
            self.eTrackingResult = result
            if position_m is not None:
                x, y, z = position_m
                self.mDeviceToAbsoluteTracking = [
                    [1.0, 0.0, 0.0, x],
                    [0.0, 1.0, 0.0, y],
                    [0.0, 0.0, 1.0, z],
                ]
            else:
                self.mDeviceToAbsoluteTracking = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]

    class _FakeVRSystem:
        def getTrackedDeviceClass(self, i):
            return devices[i][0]

        def getStringTrackedDeviceProperty(self, i, prop):
            return devices[i][1]

        def getDeviceToAbsoluteTrackingPose(self, universe, seconds, count):
            poses = []
            for _cls, _serial, pose in devices:
                if pose is None:
                    poses.append(_Pose(None, False, False, fake.TrackingResult_Uninitialized))
                else:
                    position_m, valid, connected, result = pose
                    poses.append(_Pose(position_m, valid, connected, result))
            return poses

    def _init(app_type):
        if init_error is not None:
            raise init_error
        return _FakeVRSystem()

    fake.init = _init
    fake.shutdown = lambda: None
    return fake


class OpenVRTrackerSourceTests(unittest.TestCase):
    def setUp(self):
        self._saved_openvr = sys.modules.get("openvr")

    def tearDown(self):
        if self._saved_openvr is not None:
            sys.modules["openvr"] = self._saved_openvr
        else:
            sys.modules.pop("openvr", None)

    def test_selects_configured_tracker_serial_among_multiple(self):
        GENERIC = 3
        devices = [
            (GENERIC, "LHR-AAAA", ([1.0, 2.0, 3.0], True, True, 200)),
            (GENERIC, "LHR-BBBB", ([4.0, 5.0, 6.0], True, True, 200)),
        ]
        sys.modules["openvr"] = _make_fake_openvr(devices)

        from collectors.collect_vive_tracker import OpenVRTrackerSource

        warnings = []
        source = OpenVRTrackerSource(tracker_serial="LHR-BBBB", on_warning=warnings.append)
        sample = source.next_sample()

        self.assertEqual(sample["tracker_serial"], "LHR-BBBB")
        self.assertEqual(sample["position_m"], [4.0, 5.0, 6.0])
        self.assertTrue(sample["tracking_valid"])
        self.assertEqual(sample["quaternion_xyzw"], [0.0, 0.0, 0.0, 1.0])  # identity rotation
        self.assertEqual(warnings, [], "an exact configured-serial match must not warn")

    def test_falls_back_to_first_found_and_warns_when_no_serial_configured(self):
        GENERIC = 3
        devices = [(GENERIC, "LHR-ONLY", ([0.1, 0.2, 0.3], True, True, 200))]
        sys.modules["openvr"] = _make_fake_openvr(devices)

        from collectors.collect_vive_tracker import OpenVRTrackerSource

        warnings = []
        source = OpenVRTrackerSource(tracker_serial=None, on_warning=warnings.append)
        sample = source.next_sample()

        self.assertEqual(sample["tracker_serial"], "LHR-ONLY")
        self.assertTrue(sample["tracking_valid"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("no tracker_serial configured", warnings[0])

    def test_falls_back_when_configured_serial_not_found(self):
        GENERIC = 3
        devices = [(GENERIC, "LHR-REAL", ([0.0, 0.0, 0.0], True, True, 200))]
        sys.modules["openvr"] = _make_fake_openvr(devices)

        from collectors.collect_vive_tracker import OpenVRTrackerSource

        warnings = []
        source = OpenVRTrackerSource(tracker_serial="LHR-MISSING", on_warning=warnings.append)
        sample = source.next_sample()

        self.assertEqual(sample["tracker_serial"], "LHR-REAL")  # fell back, not a hard failure
        self.assertEqual(len(warnings), 1)
        self.assertIn("not found among", warnings[0])

    def test_no_generic_tracker_present_returns_not_connected_never_raises(self):
        NOT_GENERIC = 0  # e.g. an HMD or controller
        devices = [(NOT_GENERIC, None, None)]
        sys.modules["openvr"] = _make_fake_openvr(devices)

        from collectors.collect_vive_tracker import OpenVRTrackerSource

        warnings = []
        source = OpenVRTrackerSource(on_warning=warnings.append)
        sample = source.next_sample()

        self.assertFalse(sample["tracking_valid"])
        self.assertFalse(sample["device_connected"])
        self.assertIsNone(sample["position_m"])
        self.assertIsNone(sample["quaternion_xyzw"])
        self.assertEqual(len(warnings), 1)

    def test_routine_tracking_loss_returns_invalid_sample_not_an_exception(self):
        GENERIC = 3
        devices = [(GENERIC, "LHR-X", (None, False, True, 201))]  # bPoseIsValid=False -- tracking lost, still connected
        sys.modules["openvr"] = _make_fake_openvr(devices)

        from collectors.collect_vive_tracker import OpenVRTrackerSource

        source = OpenVRTrackerSource(tracker_serial="LHR-X")
        sample = source.next_sample()  # must not raise

        self.assertFalse(sample["tracking_valid"])
        self.assertIsNone(sample["position_m"])
        self.assertIsNone(sample["quaternion_xyzw"])
        self.assertIsNone(sample["pose_matrix_3x4"])
        self.assertEqual(sample["tracking_result"], "TrackingResult_Running_OutOfRange")

    def test_openvr_init_failure_raises_a_genuine_connection_error(self):
        sys.modules["openvr"] = _make_fake_openvr([], init_error=RuntimeError("SteamVR not running"))

        from collectors.collect_vive_tracker import OpenVRTrackerSource

        source = OpenVRTrackerSource()
        with self.assertRaises(RuntimeError):
            source.next_sample()

    def test_works_through_vivetrackercollector_read_record_contract(self):
        """Same keys/shape mocks/mock_vive.py's MockViveTracker produces --
        ViveTrackerCollector.read_record() must not care which one it's
        talking to."""
        GENERIC = 3
        devices = [(GENERIC, "LHR-Y", ([1.0, 1.0, 1.0], True, True, 200))]
        sys.modules["openvr"] = _make_fake_openvr(devices)

        from collectors.collect_vive_tracker import OpenVRTrackerSource, ViveTrackerCollector

        source = OpenVRTrackerSource(tracker_serial="LHR-Y")
        collector = ViveTrackerCollector(source, session_id="session_test", segment_id=0)
        record = collector.read_record()

        self.assertTrue(record["valid"])
        self.assertEqual(record["stream"], "vive_tracker")
        self.assertEqual(record["data"]["position_m"], [1.0, 1.0, 1.0])
        self.assertEqual(record["host_seq"], 1)


if __name__ == "__main__":
    unittest.main()
