# ST3215 daisy-chain thermal behaviour and command sequencing

Written 2026-08-05 after repeated real thermal-fault trips during 50-point
sweeps. Covers *why* simultaneous multi-axis commands overheat this
particular bus, and what the collection scripts now do about it.

## Hardware in use

| Item | Value | Confidence |
|---|---|---|
| Servo | Waveshare ST3215 serial bus servo, 30 kg·cm, 360° magnetic encoder | Confirmed by the project owner |
| Count on this rig | 3 (shoulder_pan id=1, shoulder_lift id=2, wrist_flex id=4) | Confirmed — see `docs/THREE_AXIS_EXPERIMENT_SCOPE.md` |
| Bus topology | Single 3-wire **DATA / VCC / GND** daisy chain off the Waveshare ESP32 Servo Driver board; the board's DC-jack voltage *is* the servo rail | Confirmed by `03_SO-101_control/manuals/SO-101_Waveshare_ST3215_Final_Codex_Manual.md` §1.2 |
| Servo UART | 1 Mbps, ESP32 `HardwareSerial(1)` RX=18/TX=19 | Confirmed, manual §1.3 + `robot_config.h` |
| Nominal supply | 12 V class (ST3215's rated band is roughly 6–12.6 V) | **UNVERIFIED locally** — no ST3215 datasheet copy exists in this repo; figure is from Waveshare's published product spec, not a file you can diff against |
| Stall current | ~2.7 A per servo at 12 V (published spec) | **UNVERIFIED locally**, same reason. Treat as an order-of-magnitude figure for the reasoning below, not a calibrated number. |

The two "UNVERIFIED locally" rows are flagged the same way
`docs/07_WT901C_CODE_AUDIT.md` flags its `UNVERIFIED_HARDWARE_SETTING`
items: the conclusion below does not depend on the exact current figure
(only on "three simultaneous accelerations draw meaningfully more than
one"), but do not quote these numbers as repo-verified facts.

## The two distinct heat sources

**1. Simultaneous acceleration inrush (the one that was actually
happening).** A bus servo draws its peak current at motion start, when
commanded position error is largest. Three servos on one shared rail
starting at the same instant stack three peaks into one, and the resulting
rail sag also degrades the other servos' torque, which makes them draw
*longer*. This is exactly the failure the hardware manual already warns
about (§10.2: "처음부터 다축 동시 이동을 하면 ... 순간 전류 폭증" — start
with multi-axis simultaneous motion and you get an instantaneous current
surge; §4.2 warns off the broadcast example for the same reason). The
manual's own prescription is sequential per-axis homing before any
synchronized motion (§10.1 step 9, §18's "First multi-servo homing should
be sequential, not simultaneous").

**2. Sustained holding torque against gravity (the slower one).** After
motion completes, `shoulder_lift` in particular holds the whole arm's
moment. Holding current never drops to zero for a pose off the mechanical
neutral. This is a duty-cycle problem, not a peak problem, and is why the
sweep keeps a mandatory rest between targets rather than moving
continuously.

## Why the firmware made #1 unavoidable from the host side

`03_SO-101_control/main/src/main.cpp`'s legacy `D2nnn/D3nnn/D5nnn` command
path is a **deliberate batcher**:

- `queueLegacyAngleCommand()` (main.cpp:3159) marks the axis pending and
  opens a batch window instead of moving immediately.
- If every axis becomes pending, it calls `flushLegacyBatch("all_axes")`
  (main.cpp:3199) **immediately** — no window wait at all.
- `flushLegacyBatch()` (main.cpp:3099) issues `RegWritePosEx` per axis and
  then one `regWriteAction()` (main.cpp:3140) — the ST-bus "everyone start
  now" trigger.
- Otherwise `serviceLegacyBatch()` flushes on a `kLegacyBatchWindowMs = 40`
  ms timer (`robot_config.h:39`, main.cpp:3154).

So the old sweep loop's three back-to-back writes:

```python
ser.write(b"D2100\r\n"); ser.write(b"D3070\r\n"); ser.write(b"D5120\r\n")
```

hit the `all_axes` path and started all three servos on the same
`RegWriteAction`. That is the intended firmware behaviour for coordinated
motion — it is just the wrong mode for a long unattended sweep on a shared
rail.

## The fix (host-side, no reflash)

`experiment_scripts/collect_3d_sweep_50_v2.py` now dispatches through
`send_targets_sequential()`:

- one axis command at a time, in `D2 → D3 → D5` order;
- `AXIS_COMMAND_GAP_S = 0.6` s between axes — comfortably past the
  firmware's 40 ms batch window, so each axis flushes as its own
  single-axis batch and only one servo accelerates at a time;
- the gap is spent inside `poll_for()`, so encoder + Vive logging continues
  during it (no data hole, and the gap doubles as settle time);
- a fault latched mid-dispatch aborts the remaining axes for that target
  and goes straight to `recover_full()`, instead of piling two more
  commands onto an already-faulted bus.

Plus what was already there and is kept:

- `MANDATORY_REST_S = 2.5` s of idle polling after every target (addresses
  heat source #2 above);
- conservative per-joint speed/acc in `robot_config.h` (`120/100/90` speed,
  `10` acc) — inside the manual §17 recommended 80–150 speed / 5–15 acc
  band.

Firmware was deliberately **not** changed. Making `flushLegacyBatch()`
itself sequential would break coordinated motion for every other consumer
of the legacy protocol, and would need a reflash; spacing the host writes
achieves the same physical result with a 20-line diff and zero firmware
risk. If a future run genuinely needs both coordinated motion *and* thermal
headroom, the right move is a bigger power supply / current-limited rail,
not a firmware change.

## What this fix does NOT do

- It does not lower steady-state holding current. A pose that strains
  `shoulder_lift` still strains it; the rest interval is the only mitigation
  in software.
- It does not use the servo's own temperature reading as a control input.
  The onboard temperature sensor on the affected servo is **confirmed
  broken** by the project owner — readings from it are not trustworthy, and
  all software temperature-trip/cooldown logic was deliberately removed from
  the sweep script for that reason (see the comment block in
  `collect_3d_sweep_50_v2.py`). The real thermal safety net is an external
  sensor + hardware power cutoff on a separate machine.
- Fault-latch handling stays in place regardless, because the firmware may
  still latch a fault off that bad reading (`kFaultTempThresholdC`, raised
  at runtime to the firmware's `kMaxFaultTemperatureC = 120` ceiling via
  `experiment_scripts/raise_fault_temp.py`). `recover_full()` must remain
  `FAULTRESET → INIT → R`; `FAULTRESET` alone clears the flag without
  restoring torque, which previously produced 34 silently-frozen "successful"
  targets in one run.

## Remaining hardware verification

- Obtain the actual ST3215 datasheet and replace the two UNVERIFIED rows
  above with cited figures (stall/no-load current, rated duty cycle, the
  servo's own EEPROM over-temperature and over-current protection register
  defaults).
- Measure real rail current during a sweep with the sequential dispatch in
  place, to confirm the peak actually dropped rather than assuming it.
- Confirm whether `AXIS_COMMAND_GAP_S = 0.6` s is more than needed; it was
  chosen for headroom over the 40 ms batch window, not measured against
  actual settle time. Shortening it speeds up a 50-point run
  (3 × 0.6 s × 50 ≈ 90 s of the total), so it is worth measuring once the
  external thermal monitor is trusted.
