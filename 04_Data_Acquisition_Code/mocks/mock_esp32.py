"""Mock ESP32 SO-101 RTOS motor driver telemetry source (`STREAM` protocol).

Line format matched from `02_SO-101_rtos_motor_driver/main/README.md`
("Compact Status Stream" section) and `02_SO-101_rtos_motor_driver/README.md`
(`STREAM ON [period_ms]`), per `DECISIONS.md` #3/#7 (the RTOS firmware in
`02_SO-101_rtos_motor_driver/main`, not `03_SO-101_control`, is the active
target this package's collector talks to). Real example line from the
README:

    STREAM t_ms=12345 seq=1 fresh=1 armed=1 cal=0 fault=0 fault_axis=-1
    mode=PID enabled=0x1F pos=2225,2100,1988,1900,700
    upd=12344,12344,12344,12344,12344 deg=80.00,35.00,77.00,155.00,50.00
    target=80.00,35.00,77.00,155.00,50.00

Fields, fixed joint order `shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, gripper` (05's axis order/IDs 1-5):
  t_ms          ESP32 stream timestamp from boot (ms)
  seq           stream sample number
  fresh         1 when feedback was refreshed this tick
  armed/cal/fault  state flags (0/1)
  fault_axis    -1 when no fault, else joint index
  mode          outer controller mode (PD|PID)
  enabled       bitmask over the 5 joints, hex (0x1F = all 5 enabled)
  pos           raw ST3215 encoder ticks, one per joint
  upd           per-axis ESP32 feedback update timestamps (ms)
  deg           logical joint angles (deg), or -999.00 if limits not ready
  target        current logical targets (deg)

This module implements only the `STREAM` line (the compact per-tick
telemetry this package's collector consumes), not the separate multi-line
`ENCODER` verbose format -- see 05's README for both.
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "gripper")
NUM_JOINTS = len(JOINT_NAMES)

# ST3215 is a 12-bit-position bus servo (0..4095 raw ticks over its
# configured travel); used only to make `pos=` look like plausible raw
# ticks, not asserted anywhere as a calibrated value.
_TICKS_PER_DEG = 4096.0 / 360.0
_RAW_ZERO_OFFSET = 2048


class MockESP32Encoder:
    """Simulates the `STREAM ON <period_ms>` ASCII telemetry line.

    `set_target()` sets commanded degrees per joint; each `next_line()`
    call moves the simulated encoder reading toward that target at
    `move_rate_deg_s`, so `deg=` visibly converges on `target=` over
    successive calls -- useful for testing an encoder-coarse-arrival gate.
    """

    def __init__(
        self,
        period_ms: int = 100,
        move_rate_deg_s: float = 60.0,
        seed: Optional[int] = None,
    ):
        self.period_ms = period_ms
        self.move_rate_deg_s = move_rate_deg_s
        self._rng = random.Random(seed)

        self.t_ms = 0
        self.seq = 0
        self.armed = True
        self.cal = False
        self.fault = False
        self.fault_axis = -1
        self.mode = "PID"
        self.enabled_mask = 0x1F

        self.q_cmd_deg: List[float] = [0.0] * NUM_JOINTS
        self.q_encoder_deg: List[float] = [0.0] * NUM_JOINTS

    def set_target(self, target_deg: Sequence[float]) -> None:
        assert len(target_deg) == NUM_JOINTS
        self.q_cmd_deg = list(target_deg)

    def next_line(self) -> str:
        dt_s = self.period_ms / 1000.0
        step = self.move_rate_deg_s * dt_s
        for i in range(NUM_JOINTS):
            delta = self.q_cmd_deg[i] - self.q_encoder_deg[i]
            if abs(delta) <= step:
                self.q_encoder_deg[i] = self.q_cmd_deg[i]
            else:
                self.q_encoder_deg[i] += step if delta > 0 else -step

        self.t_ms += self.period_ms
        self.seq += 1

        pos = [
            int(round(_RAW_ZERO_OFFSET + self.q_encoder_deg[i] * _TICKS_PER_DEG))
            for i in range(NUM_JOINTS)
        ]
        upd = [self.t_ms] * NUM_JOINTS  # feedback refreshed this same tick

        pos_str = ",".join(str(p) for p in pos)
        upd_str = ",".join(str(u) for u in upd)
        deg_str = ",".join(f"{d:.2f}" for d in self.q_encoder_deg)
        target_str = ",".join(f"{d:.2f}" for d in self.q_cmd_deg)

        return (
            f"STREAM t_ms={self.t_ms} seq={self.seq} fresh=1 "
            f"armed={int(self.armed)} cal={int(self.cal)} fault={int(self.fault)} "
            f"fault_axis={self.fault_axis} mode={self.mode} "
            f"enabled=0x{self.enabled_mask:02X} pos={pos_str} upd={upd_str} "
            f"deg={deg_str} target={target_str}"
        )


def _self_check() -> None:
    import re

    enc = MockESP32Encoder(period_ms=50, move_rate_deg_s=100.0, seed=1)
    enc.set_target([80.0, 35.0, 77.0, 155.0, 50.0])

    pattern = re.compile(
        r"^STREAM t_ms=(\d+) seq=(\d+) fresh=(\d) armed=(\d) cal=(\d) fault=(\d) "
        r"fault_axis=(-?\d+) mode=(\S+) enabled=(0x[0-9A-Fa-f]+) "
        r"pos=([\-\d,]+) upd=([\d,]+) deg=([\-\d.,]+) target=([\-\d.,]+)$"
    )

    lines = [enc.next_line() for _ in range(40)]
    for line in lines:
        m = pattern.match(line)
        assert m, f"line did not match expected STREAM format: {line!r}"
        assert len(m.group(10).split(",")) == NUM_JOINTS
        assert len(m.group(12).split(",")) == NUM_JOINTS

    last_deg = [float(v) for v in lines[-1].split("deg=")[1].split(" ")[0].split(",")]
    for got, want in zip(last_deg, enc.q_cmd_deg):
        assert abs(got - want) < 1e-6, "encoder should have converged onto target by now"

    print("mock_esp32.py self-check OK")


if __name__ == "__main__":
    _self_check()
