"""Mock WT901C IMU byte-stream source.

Emits byte-for-byte-valid WT901C wire packets, so the real parser (built
against the same protocol) round-trips them. Protocol matched from
`07_WT901C/WT901C_imu0/collect_4imu.py`'s `extract_frames`/`decode_frame`/
`s16` (commit `3a0c757e913454ade5007cbacbb52bae48f321a0`, see
`docs/07_WT901C_CODE_AUDIT.md` and `DECISIONS.md` #4):

- Frame: 11 bytes = header(0x55) + type(1) + payload(8) + checksum(1).
- Checksum: `sum(frame[:10]) & 0xFF == frame[10]` (8-bit sum mod 256).
- Payload = 4 little-endian signed int16 (`struct.unpack("<h", ...)`).
- Frame types used here: 0x51 acceleration, 0x52 angular_velocity,
  0x53 angle, 0x59 quaternion.
- Scaling (raw int16 -> physical), inverted here to go physical -> raw:
    accel:  raw = g    / 16.0   * 32768   (full scale +-16g)
    gyro:   raw = dps  / 2000.0 * 32768   (full scale +-2000 dps)
    angle:  raw = deg  / 180.0  * 32768   (full scale +-180 deg)
    quat:   raw = q            * 32768    (no multiplier, w,x,y,z in payload order)
  accel/gyro/angle frames carry temperature in the 4th int16 slot
  (raw = temp_c * 100); the quaternion frame uses all 4 slots for w,x,y,z
  and carries no temperature.
"""
from __future__ import annotations

import math
import random
import struct
from typing import Optional

HEADER = 0x55
FRAME_ACCEL = 0x51
FRAME_GYRO = 0x52
FRAME_ANGLE = 0x53
FRAME_QUATERNION = 0x59

_ACCEL_FULL_SCALE_G = 16.0
_GYRO_FULL_SCALE_DPS = 2000.0
_ANGLE_FULL_SCALE_DEG = 180.0


def _to_i16(value: float) -> bytes:
    """Physical value scaled to a raw int16 count, clamped to int16 range
    (mirrors what a saturating ADC would do) and packed little-endian."""
    raw = max(-32768, min(32767, int(round(value))))
    return struct.pack("<h", raw)


def _build_frame(frame_type: int, payload: bytes) -> bytes:
    assert len(payload) == 8, "WT901C payload must be exactly 8 bytes"
    body = bytes((HEADER, frame_type)) + payload
    checksum = sum(body) & 0xFF
    return body + bytes((checksum,))


def _accel_frame(ax_g: float, ay_g: float, az_g: float, temp_c: float) -> bytes:
    payload = b"".join(
        _to_i16(v / _ACCEL_FULL_SCALE_G * 32768.0) for v in (ax_g, ay_g, az_g)
    ) + _to_i16(temp_c * 100.0)
    return _build_frame(FRAME_ACCEL, payload)


def _gyro_frame(gx_dps: float, gy_dps: float, gz_dps: float, temp_c: float) -> bytes:
    payload = b"".join(
        _to_i16(v / _GYRO_FULL_SCALE_DPS * 32768.0) for v in (gx_dps, gy_dps, gz_dps)
    ) + _to_i16(temp_c * 100.0)
    return _build_frame(FRAME_GYRO, payload)


def _angle_frame(roll_deg: float, pitch_deg: float, yaw_deg: float, temp_c: float) -> bytes:
    payload = b"".join(
        _to_i16(v / _ANGLE_FULL_SCALE_DEG * 32768.0) for v in (roll_deg, pitch_deg, yaw_deg)
    ) + _to_i16(temp_c * 100.0)
    return _build_frame(FRAME_ANGLE, payload)


def _quaternion_frame(qw: float, qx: float, qy: float, qz: float) -> bytes:
    payload = b"".join(_to_i16(v * 32768.0) for v in (qw, qx, qy, qz))
    return _build_frame(FRAME_QUATERNION, payload)


class MockWT901CImu:
    """Simulates one WT901C sensor's serial output.

    Motion model: either "settled" (near-zero gyro/accel-gravity/angle
    noise) or "moving" (sinusoidal gyro/angle burst of configurable
    amplitude/frequency, with accel following consistently) so an external
    settling detector has something realistic to detect. Toggle with
    `set_moving()`.

    Frame emission choice: `next_packet_bytes()` returns ONE 11-byte frame
    per call, cycling accel -> gyro -> angle -> quaternion -> accel ... .
    This mirrors the real device, which streams frames one at a time over
    the serial line rather than batching them; a full "sample" (all 4
    frame types) is 4 calls. `next_sample_bytes()` is a convenience that
    returns one full 4-frame sample (44 bytes) in one call.

    `sequence` is a plain externally-reachable counter (not encoded into
    the wire bytes -- the real WT901C protocol has no in-frame sequence
    field, host_seq is assigned by the collector). Exposed directly so a
    fault injector can read/duplicate/rewind it without reaching into a
    closure.
    """

    _FRAME_ORDER = (FRAME_ACCEL, FRAME_GYRO, FRAME_ANGLE, FRAME_QUATERNION)

    def __init__(
        self,
        sensor_id: str = "imu0",
        output_rate_hz: float = 100.0,
        seed: Optional[int] = None,
        moving: bool = False,
        motion_amplitude_dps: float = 90.0,
        motion_amplitude_deg: float = 30.0,
        motion_freq_hz: float = 0.5,
        settle_noise_dps: float = 0.3,
        settle_noise_deg: float = 0.1,
    ):
        assert output_rate_hz in (100.0, 200.0, 100, 200), "spec allows 100 or 200 Hz"
        self.sensor_id = sensor_id
        self.output_rate_hz = float(output_rate_hz)
        self.moving = moving
        self.motion_amplitude_dps = motion_amplitude_dps
        self.motion_amplitude_deg = motion_amplitude_deg
        self.motion_freq_hz = motion_freq_hz
        self.settle_noise_dps = settle_noise_dps
        self.settle_noise_deg = settle_noise_deg

        self._rng = random.Random(seed)
        self.sequence = 0  # externally settable/readable, see docstring
        self.t_s = 0.0
        self._frame_idx = 0

    def set_moving(self, moving: bool) -> None:
        self.moving = moving

    def _physical_state(self):
        """Returns (accel_g[3], gyro_dps[3], angle_deg[3], quat_wxyz[4])."""
        if self.moving:
            w = 2.0 * math.pi * self.motion_freq_hz
            gyro = [
                self.motion_amplitude_dps * math.sin(w * self.t_s),
                self.motion_amplitude_dps * math.sin(w * self.t_s + 1.0),
                self.motion_amplitude_dps * math.sin(w * self.t_s + 2.0),
            ]
            angle = [
                self.motion_amplitude_deg * math.sin(w * self.t_s),
                self.motion_amplitude_deg * math.sin(w * self.t_s + 1.0),
                self.motion_amplitude_deg * math.sin(w * self.t_s + 2.0),
            ]
            accel = [
                0.05 * math.sin(w * self.t_s),
                0.05 * math.cos(w * self.t_s),
                1.0 + 0.05 * math.sin(w * self.t_s * 2.0),
            ]
        else:
            noise = lambda amp: self._rng.uniform(-amp, amp)
            gyro = [noise(self.settle_noise_dps) for _ in range(3)]
            angle = [noise(self.settle_noise_deg) for _ in range(3)]
            accel = [noise(0.01), noise(0.01), 1.0 + noise(0.01)]

        half = math.radians(angle[2]) / 2.0  # trivial yaw-only quaternion, good enough for a mock
        quat = (math.cos(half), 0.0, 0.0, math.sin(half))
        return accel, gyro, angle, quat

    def next_packet_bytes(self) -> bytes:
        """Return one checksummed 11-byte WT901C frame, rotating frame type."""
        accel, gyro, angle, quat = self._physical_state()
        temp_c = 25.0
        frame_type = self._FRAME_ORDER[self._frame_idx]
        if frame_type == FRAME_ACCEL:
            pkt = _accel_frame(accel[0], accel[1], accel[2], temp_c)
        elif frame_type == FRAME_GYRO:
            pkt = _gyro_frame(gyro[0], gyro[1], gyro[2], temp_c)
        elif frame_type == FRAME_ANGLE:
            pkt = _angle_frame(angle[0], angle[1], angle[2], temp_c)
        else:
            pkt = _quaternion_frame(*quat)

        self._frame_idx += 1
        if self._frame_idx >= len(self._FRAME_ORDER):
            self._frame_idx = 0
            self.t_s += 1.0 / self.output_rate_hz  # advance sim time once per full sample
        self.sequence += 1
        return pkt

    def next_sample_bytes(self) -> bytes:
        """Convenience: one full sample = accel+gyro+angle+quaternion frames
        concatenated (44 bytes), i.e. 4 calls to `next_packet_bytes()`."""
        return b"".join(self.next_packet_bytes() for _ in self._FRAME_ORDER)


def _self_check() -> None:
    imu = MockWT901CImu(sensor_id="imu_test", output_rate_hz=100.0, seed=1)
    for _ in range(8):
        pkt = imu.next_packet_bytes()
        assert len(pkt) == 11
        assert pkt[0] == HEADER
        assert (sum(pkt[:10]) & 0xFF) == pkt[10], "checksum must validate"
    assert imu.sequence == 8

    imu.set_moving(True)
    sample = imu.next_sample_bytes()
    assert len(sample) == 44
    print("mock_imu.py self-check OK")


if __name__ == "__main__":
    _self_check()
