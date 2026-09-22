"""Mock Vive Tracker source.

No real device protocol exists to match here -- OpenVR/Vive isn't
reverse-engineered anywhere in this repo. Instead this mock's
`next_sample()` returns a dict with exactly the keys
`core.schemas.vive_record`'s `data` block expects (master spec section 12):
`tracker_serial, device_index, tracking_result, tracking_valid,
device_connected, tracking_origin, pose_matrix_3x4, position_m,
quaternion_xyzw, source_timestamp`.

Policy enforced here (DECISIONS.md #6, master spec 1.3/17.2, and
`core.schemas.vive_record`'s own docstring): "never forward-fill invalid
tracking". When `tracking_lost` is set, this mock returns
`tracking_valid=False` and `position_m`/`quaternion_xyzw`/`pose_matrix_3x4`
all `None` -- it does NOT cache and replay the last-good pose.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional


class MockViveTracker:
    def __init__(
        self,
        tracker_serial: str = "LHR-MOCK0001",
        device_index: int = 1,
        sample_period_s: float = 1.0 / 90.0,  # Vive base stations run ~90 Hz
        motion_radius_m: float = 0.05,
        motion_freq_hz: float = 0.2,
        yaw_rate_dps: float = 5.0,
        seed: Optional[int] = None,
    ):
        self.tracker_serial = tracker_serial
        self.device_index = device_index
        self.sample_period_s = sample_period_s
        self.motion_radius_m = motion_radius_m
        self.motion_freq_hz = motion_freq_hz
        self.yaw_rate_dps = yaw_rate_dps
        self.tracking_origin = "TrackingUniverseStanding"

        self._rng = random.Random(seed)
        self.tracking_lost = False
        self.device_connected = True
        self.t_s = 0.0

    def set_tracking_lost(self, lost: bool) -> None:
        self.tracking_lost = lost

    def set_device_connected(self, connected: bool) -> None:
        self.device_connected = connected

    def next_sample(self) -> Dict[str, Any]:
        self.t_s += self.sample_period_s
        source_timestamp = self.t_s

        if self.tracking_lost or not self.device_connected:
            return {
                "tracker_serial": self.tracker_serial,
                "device_index": self.device_index,
                "tracking_result": (
                    "TrackingResult_Uninitialized"
                    if not self.device_connected
                    else "TrackingResult_Running_OutOfRange"
                ),
                "tracking_valid": False,
                "device_connected": self.device_connected,
                "tracking_origin": self.tracking_origin,
                "pose_matrix_3x4": None,
                "position_m": None,
                "quaternion_xyzw": None,
                "source_timestamp": source_timestamp,
            }

        w = 2.0 * math.pi * self.motion_freq_hz
        x = self.motion_radius_m * math.sin(w * self.t_s)
        y = 0.75 + self.motion_radius_m * math.cos(w * self.t_s)
        z = 0.30 + 0.02 * math.sin(w * self.t_s * 0.5)
        position_m = [x, y, z]

        yaw_rad = math.radians(self.yaw_rate_dps * self.t_s)
        half = yaw_rad / 2.0
        quaternion_xyzw = [0.0, 0.0, math.sin(half), math.cos(half)]  # yaw-only rotation

        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        pose_matrix_3x4 = [
            [cos_y, -sin_y, 0.0, x],
            [sin_y, cos_y, 0.0, y],
            [0.0, 0.0, 1.0, z],
        ]

        return {
            "tracker_serial": self.tracker_serial,
            "device_index": self.device_index,
            "tracking_result": "TrackingResult_Running_OK",
            "tracking_valid": True,
            "device_connected": True,
            "tracking_origin": self.tracking_origin,
            "pose_matrix_3x4": pose_matrix_3x4,
            "position_m": position_m,
            "quaternion_xyzw": quaternion_xyzw,
            "source_timestamp": source_timestamp,
        }


def _self_check() -> None:
    vt = MockViveTracker(seed=1)
    good = vt.next_sample()
    assert good["tracking_valid"] is True
    assert good["position_m"] is not None and len(good["position_m"]) == 3
    assert good["quaternion_xyzw"] is not None and len(good["quaternion_xyzw"]) == 4
    assert good["pose_matrix_3x4"] is not None and len(good["pose_matrix_3x4"]) == 3

    vt.set_tracking_lost(True)
    lost = vt.next_sample()
    assert lost["tracking_valid"] is False
    assert lost["position_m"] is None
    assert lost["quaternion_xyzw"] is None
    assert lost["pose_matrix_3x4"] is None

    vt.set_tracking_lost(False)
    recovered = vt.next_sample()
    assert recovered["tracking_valid"] is True
    assert recovered["position_m"] is not None, "must not stay stuck on the lost state"

    print("mock_vive.py self-check OK")


if __name__ == "__main__":
    _self_check()
