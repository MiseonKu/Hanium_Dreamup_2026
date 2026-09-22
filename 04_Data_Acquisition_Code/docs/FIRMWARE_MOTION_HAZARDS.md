# Firmware motion hazards (`03_SO-101_control/main`)

Found 2026-08-05 while bringing up a real run. These are properties of the
firmware on the board, not of this package. Documented because each one
caused, or nearly caused, real unintended arm motion.

---

## 1. `INIT` enables torque BEFORE writing any goal position

**This is the dangerous one.** `handleInit` does:

```
main.cpp:3278   setTorqueForAxis(index, true, force)    <- torque ON, all axes
main.cpp:3288   delay(kTorqueCommandDelayMs)
   ...
main.cpp:3309   g_bus.writePos(config.servo_id, home_raw, ...)   <- goal written
```

An ST3215 drives to whatever is in its goal-position register the instant
torque is enabled. Between line 3278 and line 3309 that register still holds
a **stale** goal — whatever was last commanded, possibly from a previous
session, possibly from before the arm was moved by hand.

So `INIT` makes the arm lunge to an arbitrary old target first, and only then
proceeds to home. Observed consequence: shoulder_lift was driven roughly
150° across its travel to the far mechanical end (raw 3304, past its
calibrated max of 3296), while wrist_flex stopped part-way (raw 2417, between
its start 2002 and home 2953). Every subsequent legacy `D3` command was then
silently ignored — see hazard 3.

### Safe replacement

`MOVE_RAW` (main.cpp:1847) writes the goal register via `WritePosEx` and
**never touches the torque-enable register**. So write goal = current
position while torque is still off, and the following `TORQUE ON` becomes a
pure hold:

```
CAL_AUTO LOAD                       # port open resets the board; reload mapping
STATE                               # read current raw_pos per axis
MOVE_RAW <axis> <its current raw>   # for each axis; no motion, torque is off
TORQUE ON ALL                       # holds exactly where it already is
```

Verified twice on hardware: before/after `raw_pos` identical (2082/2047/2002),
and on a second capture only 3 ticks of drift on shoulder_lift.

Implemented as `safe_engage()` in
`experiment_scripts/collect_3d_sweep_50_v2.py`, and used for fault recovery
too — a latched fault is precisely when the stale goal is most likely to be
far from where the arm actually is, i.e. the worst possible moment to call
`INIT`.

---

## 2. Opening the serial port resets the board, but NOT the servos' torque

Creating a `serial.Serial` on the ESP32 port toggles DTR/RTS and reboots it.
That clears the firmware's own bookkeeping — `g_is_initialized`,
`g_joint_states[].torque_on`, the loaded calibration mapping — but the
servos are on a separate bus and keep whatever torque state they were left
in.

Result: `STATE` can report `torque=false` while the arm is genuinely still
held. Do not trust that flag as "the arm is loose"; send an explicit
`TORQUE OFF ALL` if you need it released. (Encountered exactly this: `STATE`
showed `torque=false`, the arm was in fact still locked.)

Calibration itself survives — it lives in servo EEPROM — but must be
reloaded into the firmware's runtime with `CAL_AUTO LOAD` after every
reconnect, or `logical_deg` reads `NA` and motion commands are refused.

---

## 3. Legacy `D<axis><deg>` commands require `INIT`, `MOVE_RAW` does not

`queueLegacyAngleCommand` bails out unless `g_is_initialized`
(main.cpp:3167), which only `INIT` ever sets. `MOVE_RAW` requires just
`ensureAxisMappingReady`.

This is why a run can look alive — `STATE` keeps returning valid, plausible
telemetry — while every motion command is being dropped on the floor. Seven
consecutive `D3` commands produced exactly zero movement this way, with no
fault flag raised.

**Consequence for this package:** since `INIT` is unsafe (hazard 1) and
avoiding it means `g_is_initialized` stays false, all motion goes through
`MOVE_RAW`. That is better anyway — `MOVE_RAW` takes raw ticks, skipping the
firmware's distorted `logical_deg` label scale (see
`docs/FK_CONVENTION_CORRECTION.md`).

---

## 4. Legacy `D` commands batch into one simultaneous start

`flushLegacyBatch` (main.cpp:3099) collects every axis queued within
`kLegacyBatchWindowMs` (40 ms) and fires one `RegWriteAction()`
(main.cpp:3140), and flushes immediately once *all* axes are pending
(main.cpp:3199). Three back-to-back `D2`/`D3`/`D5` writes therefore start all
three servos at the same instant, stacking their inrush on one shared power
rail.

See `docs/ST3215_THERMAL_AND_SEQUENCING.md`. Moot once everything uses
`MOVE_RAW`, which is one axis per command by construction — sequencing
becomes a property of the protocol rather than a discipline the host has to
maintain.

---

## 5. `FAULTRESET` clears the flag without restoring control

Long-standing, already known: `FAULTRESET` alone leaves torque off. An early
run continued through 34 targets after a `FAULTRESET`-only recovery, frozen
in place, with byte-identical positions despite different commands — and
neither the fault flag nor the record counts caught it.

Recovery must re-engage torque. Use `safe_engage()`, not `INIT`.

---

## Summary: what to send, and what never to send

| goal | use | never |
|---|---|---|
| hold current pose | `MOVE_RAW`(current) → `TORQUE ON ALL` | `TORQUE ON` alone, `INIT` |
| move an axis | `MOVE_RAW <axis> <raw>` | `D<axis><deg>` (needs INIT) |
| recover from fault | `FAULTRESET` → `safe_engage()` | `FAULTRESET` alone, `INIT` |
| release | `TORQUE OFF ALL` | trusting `STATE`'s `torque=false` |
| after reconnect | `CAL_AUTO LOAD` | assuming mapping survived |
