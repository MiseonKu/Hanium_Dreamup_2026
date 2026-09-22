"""PC-side collector for Vive Tracker pose samples.

`VivePoseSource` below is a structural (duck-typed) interface:
`mocks/mock_vive.py`'s `MockViveTracker` (test/mock use) and
`OpenVRTrackerSource` below (real SteamVR/OpenVR use) both satisfy it --
`next_sample() -> dict`, same keys `core.schemas.vive_record` consumes --
without `ViveTrackerCollector`'s own logic needing to know which one it's
talking to.

Hard rule (DECISIONS.md #6, master spec 1.3/17.2, `core.schemas.vive_record`
docstring): every sample is written, including tracking-lost ones. A lost
sample becomes a record with `valid=False`/`data.tracking_valid=False` and
null pose fields -- never skipped, and never forward-filled from the last
good pose. This collector never caches a previous pose, so there is nothing
to fall back to even by accident.
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Callable, Optional, Protocol

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schemas import (  # noqa: E402
    SCHEMA_VIVE,
    STREAM_VIVE,
    base_record,
    monotonic_ns,
    vive_record,
)


class VivePoseSource(Protocol):
    """Anything with `next_sample() -> dict` in the shape
    `mocks/mock_vive.py:MockViveTracker.next_sample()` returns works here --
    no subclassing required, real OpenVR wrapper included."""

    def next_sample(self) -> dict[str, Any]: ...


def _parse_status(sample: dict[str, Any]) -> str:
    if not sample.get("device_connected", True):
        return "device_disconnected"
    if not sample.get("tracking_valid"):
        return "tracking_lost"
    return "ok"


class ViveTrackerCollector:
    def __init__(
        self,
        source: VivePoseSource,
        session_id: str,
        segment_id: int = 0,
        retry_wait_s: float = 0.5,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ):
        self.source = source
        self.session_id = session_id
        self.segment_id = segment_id
        self.retry_wait_s = retry_wait_s
        self.on_error = on_error
        self.host_seq = 0

    def read_record(self) -> dict[str, Any]:
        """Return exactly one `vive_record` per call. A tracking-lost
        sample is still returned as a record (never skipped, never
        forward-filled -- see module docstring). If `source.next_sample()`
        itself raises, that's logged (via `on_error` if given) and retried
        after `retry_wait_s` -- this method never crashes the process and
        never fabricates a record from a failed call."""
        while True:
            try:
                sample = self.source.next_sample()
                break
            except Exception as exc:  # source hiccup -- never crash the process
                if self.on_error is not None:
                    self.on_error(exc)
                time.sleep(self.retry_wait_s)

        t_pc_rx_ns = monotonic_ns()
        self.host_seq += 1
        valid = bool(sample.get("tracking_valid"))

        base = base_record(
            schema=SCHEMA_VIVE,
            session_id=self.session_id,
            segment_id=self.segment_id,
            stream=STREAM_VIVE,
            sensor_id=sample.get("tracker_serial"),
            host_seq=self.host_seq,
            t_pc_rx_ns=t_pc_rx_ns,
            valid=valid,
            parse_status=_parse_status(sample),
        )
        return vive_record(
            base,
            tracker_serial=sample.get("tracker_serial"),
            device_index=sample.get("device_index"),
            tracking_result=sample.get("tracking_result"),
            tracking_valid=sample.get("tracking_valid"),
            device_connected=sample.get("device_connected"),
            tracking_origin=sample.get("tracking_origin"),
            pose_matrix_3x4=sample.get("pose_matrix_3x4"),
            position_m=sample.get("position_m"),
            quaternion_xyzw=sample.get("quaternion_xyzw"),
            source_timestamp=sample.get("source_timestamp"),
        )


def _matrix_to_quat_xyzw(m: list) -> list[float]:
    """3x4 row-major pose matrix -> `[x, y, z, w]` quaternion. Adapted
    verbatim (same algorithm, same variable shape) from
    `experiment_scripts/collect_3d_sweep_50_v2.py:matrix_to_quat_xyzw` --
    that copy was exercised against a real Vive Tracker in the archived
    real-hardware run this module is closing the gap for, so the math
    itself is not rederived here, only relocated."""
    r = [[m[i][j] for j in range(3)] for i in range(3)]
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (r[2][1] - r[1][2]) / s
        y = (r[0][2] - r[2][0]) / s
        z = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        w = (r[2][1] - r[1][2]) / s
        x = 0.25 * s
        y = (r[0][1] + r[1][0]) / s
        z = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        w = (r[0][2] - r[2][0]) / s
        x = (r[0][1] + r[1][0]) / s
        y = 0.25 * s
        z = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        w = (r[1][0] - r[0][1]) / s
        x = (r[0][2] + r[2][0]) / s
        y = (r[1][2] + r[2][1]) / s
        z = 0.25 * s
    return [x, y, z, w]


def _not_connected_sample(tracker_serial: Optional[str]) -> dict[str, Any]:
    return {
        "tracker_serial": tracker_serial,
        "device_index": None,
        "tracking_result": "TrackingResult_Uninitialized",
        "tracking_valid": False,
        "device_connected": False,
        "tracking_origin": "TrackingUniverseStanding",
        "pose_matrix_3x4": None,
        "position_m": None,
        "quaternion_xyzw": None,
        "source_timestamp": time.time(),
    }


class OpenVRTrackerSource:
    """Real SteamVR/OpenVR-backed `VivePoseSource`.

    Pattern (device scan, pose poll, tracking-result checks) is carried
    over from `experiment_scripts/collect_3d_sweep_50_v2.py`'s
    `read_vive_raw()` / the module-level tracker-scan loop at that script's
    top -- both already exercised against real hardware in the archived
    run. What changed here vs. that script:

    - Tracker SELECTION: that script picked the first
      `TrackedDeviceClass_GenericTracker` found, unconditionally. Here,
      `tracker_serial` (if given) is matched against every GenericTracker's
      serial number and only that one is accepted -- multiple trackers on
      one bench no longer risk silently reading the wrong one. If
      `tracker_serial` is omitted, or set but not found among the trackers
      actually present, this falls back to first-found and reports the
      fallback via `on_warning` (never a hard failure -- still usable for
      a quick single-tracker bring-up with no config at all).
    - `next_sample()` never raises for routine tracking loss/no-tracker-
      found (same hard rule as `mocks/mock_vive.py` / this module's
      docstring: null fields, `tracking_valid=False`, never forward-filled)
      -- it only raises out of `openvr.init()` failing or a mid-session
      pose-poll call itself raising (both indicate the OpenVR/SteamVR
      connection itself is gone, not just this one sample), which
      `ViveTrackerCollector.read_record()` already retries with backoff.
    - `import openvr` is local to the methods that need it (not a module-
      top import), matching `collect_imu_rs232.py`'s `_open_real_serial`
      pattern -- this file stays importable without the `openvr` package
      installed (e.g. in CI).

    `source_timestamp`: OpenVR's `getDeviceToAbsoluteTrackingPose` does not
    return a per-pose device-side timestamp for this call shape (that's a
    predicted pose for `time.time()` + 0s, not a hardware capture time) --
    `time.time()` (wall clock, at the moment this Python call returns) is
    used instead and is NOT a device timestamp; `t_pc_rx_ns` (monotonic,
    stamped by `ViveTrackerCollector.read_record()`) remains the
    authoritative sample time per `core/schemas.py`'s timestamp policy.
    """

    def __init__(
        self,
        tracker_serial: Optional[str] = None,
        on_warning: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.tracker_serial = tracker_serial
        self.on_warning = on_warning or (lambda msg: None)
        self._vrsys = None
        self._device_index: Optional[int] = None
        self._resolved_serial: Optional[str] = None
        self._tracking_result_names: dict[int, str] = {}

    def _ensure_vrsys(self, openvr_mod) -> None:
        if self._vrsys is not None:
            return
        # Genuine connection-level failure (SteamVR not running, etc.) --
        # let it propagate; ViveTrackerCollector.read_record() retries
        # with backoff and reports via on_error.
        self._vrsys = openvr_mod.init(openvr_mod.VRApplication_Other)
        self._tracking_result_names = {
            getattr(openvr_mod, "TrackingResult_Uninitialized", 1): "TrackingResult_Uninitialized",
            getattr(openvr_mod, "TrackingResult_Calibrating_InProgress", 100): "TrackingResult_Calibrating_InProgress",
            getattr(openvr_mod, "TrackingResult_Calibrating_OutOfRange", 101): "TrackingResult_Calibrating_OutOfRange",
            getattr(openvr_mod, "TrackingResult_Running_OK", 200): "TrackingResult_Running_OK",
            getattr(openvr_mod, "TrackingResult_Running_OutOfRange", 201): "TrackingResult_Running_OutOfRange",
        }

    def _select_tracker(self, openvr_mod) -> Optional[int]:
        candidates = []
        for i in range(openvr_mod.k_unMaxTrackedDeviceCount):
            if self._vrsys.getTrackedDeviceClass(i) == openvr_mod.TrackedDeviceClass_GenericTracker:
                serial = self._vrsys.getStringTrackedDeviceProperty(i, openvr_mod.Prop_SerialNumber_String)
                candidates.append((i, serial))

        if self.tracker_serial:
            for i, serial in candidates:
                if serial == self.tracker_serial:
                    self._resolved_serial = serial
                    return i
            self.on_warning(
                f"configured tracker_serial={self.tracker_serial!r} not found among "
                f"{len(candidates)} GenericTracker device(s); falling back to first-found"
            )

        if candidates:
            i, serial = candidates[0]
            if not self.tracker_serial:
                self.on_warning(
                    f"no tracker_serial configured -- selected first-found GenericTracker "
                    f"(index={i}, serial={serial!r}); set tracker_serial to pin this explicitly"
                )
            self._resolved_serial = serial
            return i

        self.on_warning("no GenericTracker device found on this OpenVR system")
        return None

    def next_sample(self) -> dict[str, Any]:
        import openvr  # local import -- keep this file importable without openvr installed

        self._ensure_vrsys(openvr)  # may raise -- genuine connection failure, see class docstring
        if self._device_index is None:
            self._device_index = self._select_tracker(openvr)
            if self._device_index is None:
                return _not_connected_sample(self.tracker_serial)

        try:
            poses = self._vrsys.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )
            p = poses[self._device_index]
        except Exception:
            # Mid-session pose-poll failure -- treat as a dead connection
            # (not a routine tracking-loss sample): force re-init + re-scan
            # on the next call, then re-raise so read_record()'s existing
            # retry/backoff handles it, same as an openvr.init() failure.
            self._vrsys = None
            self._device_index = None
            raise

        source_timestamp = time.time()
        if not (p.bPoseIsValid and p.bDeviceIsConnected):
            return {
                "tracker_serial": self._resolved_serial or self.tracker_serial,
                "device_index": self._device_index,
                "tracking_result": self._tracking_result_names.get(p.eTrackingResult, f"unknown_{p.eTrackingResult}"),
                "tracking_valid": False,
                "device_connected": bool(p.bDeviceIsConnected),
                "tracking_origin": "TrackingUniverseStanding",
                "pose_matrix_3x4": None,
                "position_m": None,
                "quaternion_xyzw": None,
                "source_timestamp": source_timestamp,
            }

        m = p.mDeviceToAbsoluteTracking
        matrix = [[m[r][c] for c in range(4)] for r in range(3)]
        position_m = [matrix[0][3], matrix[1][3], matrix[2][3]]
        quaternion_xyzw = _matrix_to_quat_xyzw(matrix)

        return {
            "tracker_serial": self._resolved_serial or self.tracker_serial,
            "device_index": self._device_index,
            "tracking_result": self._tracking_result_names.get(p.eTrackingResult, f"unknown_{p.eTrackingResult}"),
            "tracking_valid": True,
            "device_connected": True,
            "tracking_origin": "TrackingUniverseStanding",
            "pose_matrix_3x4": matrix,
            "position_m": position_m,
            "quaternion_xyzw": quaternion_xyzw,
            "source_timestamp": source_timestamp,
        }


def _self_check() -> None:
    # Identity rotation must round-trip to the identity quaternion [0,0,0,1].
    identity_matrix = [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.3]]
    qx, qy, qz, qw = _matrix_to_quat_xyzw(identity_matrix)
    assert abs(qx) < 1e-9 and abs(qy) < 1e-9 and abs(qz) < 1e-9 and abs(qw - 1.0) < 1e-9

    not_connected = _not_connected_sample("LHR-TEST0001")
    assert not_connected["tracking_valid"] is False
    assert not_connected["position_m"] is None

    print("collect_vive_tracker.py self-check OK")


if __name__ == "__main__":
    _self_check()
