# Hardware Connection Checklist

**Amendment (2026-08-05): the original blanket "this package has never been
connected to a real WT901C, ESP32, or Vive Tracker" is stale and has been
removed.** Bring-up has happened: 4x WT901C on COM16-19 @ 9600 baud, an
ESP32 running `03_SO-101_control/main` on COM20 @ 115200, and a Vive
Tracker via SteamVR, producing `sessions/real_motion_demo/` segments
000-012. See `docs/STATUS.md`.

What this document is now: the **repeatable bring-up procedure** to run each
time the rig is set up again, with per-step status reflecting what the first
bring-up actually established. `CAN_DO_NOW` = pure software/paperwork check.
`VERIFIED_ONCE` = done successfully on real hardware at least once, still do
it again. `NOT_YET_VERIFIED` = never established on hardware.

For the failure modes that actually ruined real captures (dead tracker
battery, `mapping_ready=false`, dead IMU port), run
`docs/PRE_RUN_CHECKLIST.md` immediately before each run — that list is
evidence-based and this one is not a substitute for it.

Do this checklist in order. Do not skip ahead to plugging in a sensor before
finishing the driver/software checks -- a device that enumerates
incorrectly is much easier to diagnose before other sensors are also in the
mix.

## 1. Software prerequisites -- `CAN_DO_NOW`

- [ ] `pip install -r requirements.txt` from this package's root succeeds.
- [ ] `python -m pytest tests -q` passes -- all of them, whatever the count
      is when you run it (77 when this document was written, 150 on
      2026-08-05; the suite is actively growing). Confirms the software side
      is healthy before touching hardware at all.
- [ ] `python tools/inventory_ports.py` runs without error (it will simply
      list whatever's currently plugged in, sensors or not).

## 2. Driver setup -- `VERIFIED_ONCE`

- [ ] Follow `docs/WINDOWS_DRIVER_SETUP.md` for each sensor's USB-serial
      bridge chip.
- [ ] Confirm each device shows cleanly in Windows Device Manager (no
      yellow warning icon) before proceeding.

## 3. Sensor map configuration -- `CAN_DO_NOW` (values) / `NOT_YET_VERIFIED` (real sessions so far opened ports directly, not via sensor_map)

- [ ] Copy `config/sensor_map.example.yaml` to a real `sensor_map.yaml`
      (not committed -- treat it as machine-local config).
- [ ] Plug in one sensor at a time. After each one, run
      `python tools/inventory_ports.py` and copy that sensor's
      `VID`/`PID`/`SERIAL_NUMBER`/`DESCRIPTION` into the corresponding
      `sensor_map.yaml` entry, replacing every `REPLACE_ME_*` placeholder.
- [ ] If two sensors of the same model report identical VID/PID and no
      distinguishing `serial_number`, note that in the config (the
      `fallback_com_port` field exists for exactly this case) -- do not
      silently rely on plug-in order.

## 4. Physical connection order -- `NOT_YET_VERIFIED`

A specific power-on sequence has still not been established: real sessions
have been run, but nobody varied the order to find out what matters. Treat
these as reasonable precautions, not verified requirements:

- [ ] Connect USB-serial sensors (4x IMU, ESP32) one at a time, confirming
      each enumerates before plugging in the next -- makes it obvious which
      physical device maps to which COM port if something goes wrong.
- [ ] Servo bus power. **Amended 2026-08-05:** the old text said this
      package "never sends `ARM`, `TORQUE`, or any motion command" and
      deferred servo sequencing to `02_SO-101_rtos_motor_driver`. Both are
      wrong now. `collectors/` is still read-only, but real motion runs are
      driven from `experiment_scripts/` against `03_SO-101_control/main`
      (the firmware actually flashed here — `DECISIONS.md` #3/#7). If the
      servo bus will be powered, read `docs/FIRMWARE_MOTION_HAZARDS.md`
      first: opening the serial port reboots the ESP32 but does **not**
      release servo torque, so `STATE` can report `torque=false` while the
      arm is genuinely held.
- [ ] `CAL_AUTO LOAD` after every reconnect. Calibration lives in servo
      EEPROM and survives, but the firmware's runtime copy does not —
      without it `logical_deg` reads `NA` and motion commands are refused
      (`docs/FIRMWARE_MOTION_HAZARDS.md` #2).
- [ ] Set up the Vive base stations / lighthouse tracking volume per
      standard SteamVR/OpenVR guidance (out of scope for this package to
      document). **Amended:** a real OpenVR-backed source *does* exist now —
      `collectors/collect_vive_tracker.py`'s `OpenVRTrackerSource`, covered
      by `tests/test_openvr_tracker_source.py` against a stub `openvr`
      module. The old "no real OpenVR integration exists here yet" note was
      stale. Confirm the tracker is awake and reporting `bPoseIsValid=1`
      immediately before a run — a flat tracker battery silently voided a
      whole 12-minute capture (segment 012, 0% valid). See
      `docs/PRE_RUN_CHECKLIST.md` A1.

## 5. Per-sensor smoke check -- `VERIFIED_ONCE`

For each IMU and the ESP32, before running a full session:

- [ ] Confirm the port opens at the expected baud. **Amended 2026-08-05:**
      9600 is no longer an unverified guess — the real WT901C units on this
      bench run at 9600 and produced 95k+ frames with 100% checksum-OK and
      zero parse errors (`docs/IMU_DATA_COVERAGE.md`). ESP32: 115200, per
      `03_SO-101_control/main/include/robot_config.h:35` (`kUsbBaud`) —
      the old citation to `02_SO-101_rtos_motor_driver/README.md` pointed at
      firmware that is not flashed on this rig, though the value matches.
      Still confirm per unit; a differently-configured WT901C is possible.
- [ ] Confirm at least one valid, checksum-passing frame/line is received
      within a few seconds (a real collector run, or a short manual serial
      read).
- [ ] Confirm `checksum_ok`/`valid` on the resulting records are mostly
      `True` -- a high early failure rate suggests a baud mismatch, wrong
      full-scale-range assumption (IMU;
      `docs/07_WT901C_CODE_AUDIT.md`'s `UNVERIFIED_HARDWARE_SETTING` items),
      or a wiring problem, not something to push past.

## 6. Full mock-vs-real comparison -- `NOT_YET_VERIFIED`

- [ ] Run `tools/run_mock_session.py` first (see
      `docs/EXPERIMENT_RUNBOOK.md`) so you know what a healthy manifest and
      raw file set look like.
- [ ] Only after that, attempt a real-hardware `passive` session (see
      `docs/EXPERIMENT_RUNBOOK.md` section 6). **Amended 2026-08-05:** real
      port resolution is now wired — pass
      `SessionController(sensor_map_path=...)`. The old
      `"no real port resolution configured (mode != mock)"` blocker is
      gone. A `passive` session against real sensors still has not been run
      end-to-end; all real captures so far came from `experiment_scripts/`.

## Before declaring hardware bring-up "done"

Every item in this checklist is necessary but not sufficient. See
`docs/HARDWARE_TEST_PLAN.md` for the fuller validation plan (WT901C
full-scale-range confirmation, axis-convention checks, real-hardware
sampling-rate audit, etc.) that should happen before this system is trusted
for a real data-collection run.
