# Hardware Test Plan

A plan for what real-hardware validation would need to happen before this
system is trusted on real WT901C/ESP32/Vive hardware. Pulled directly from
the specific `UNVERIFIED_HARDWARE_SETTING` items in
`docs/07_WT901C_CODE_AUDIT.md` and from `UNVALIDATED`/reference-only
comments in this package's own collector and detector code -- not invented
from scratch.

**Amendment (2026-08-05).** The original header said "Every item below is
`NOT_TESTED` -- nothing in this list has been run against real hardware."
That blanket claim is now false: real captures exist
(`sessions/real_motion_demo/` segments 000-041 as of 2026-08-10 — acquisition
for this experiment is complete, `docs/STATUS.md`), and several items below
are resolved by them. Each item now carries its own status. The ones that remain
open are genuinely open — notably the full-scale-range constants, which
were never read back from the device's own range registers and cannot be
settled by looking at captured data.

## WT901C IMU protocol assumptions

All items below trace to `collectors/collect_imu_rs232.py`'s
`IMUScaleConfig` and `decode_frame`, carried over from
`07_WT901C/WT901C_imu0/collect_4imu.py` and flagged
`UNVERIFIED_HARDWARE_SETTING` in `docs/07_WT901C_CODE_AUDIT.md`:

- [ ] **`NOT_TESTED` — Full-scale accelerometer range.** Code assumes ±16g
      (`IMUScaleConfig.accel_full_scale_g = 16.0`). Confirm against the real
      unit's configured range register -- a differently-configured sensor
      silently produces wrong `accel_mps2` values (still self-consistent,
      just scaled wrong).
- [ ] **`NOT_TESTED` — Full-scale gyroscope range.** Code assumes ±2000 dps
      (`gyro_full_scale_dps = 2000.0`). Same confirmation needed.
- [ ] **`NOT_TESTED` — Full-scale angle range.** Code assumes ±180°
      (`angle_full_scale_deg = 180.0`). Same confirmation needed.
- [x] **`RESOLVED 2026-08-05` — Byte-order / frame-layout convention.**
      `s16()`'s little-endian, low-byte-first-per-axis assumption and the
      `0x55` header/frame-type overload are now confirmed against real
      captures: 95,845 acceleration / 95,848 angular_velocity / 95,856
      angle / 95,854 magnetic_field frames decoded across 4 physically
      separate sensors with 100% checksum-OK, zero parse errors, every
      frame 11 bytes (`docs/IMU_DATA_COVERAGE.md`). The same data corrected
      a real bug the layout assumption had hidden: the angle packet's word4
      was being decoded as `temperature_c` (a constant, implausible
      195.82°C) and is now stored as `angle_packet_word4_raw`.
      Still unobserved from this device, so still unverified: `0x50` time,
      `0x55` port_status, and `0x59` quaternion packets — the output mask
      apparently never emits them.
- [x] **`RESOLVED 2026-08-05` — Baud rate.** The real units on this bench
      run at 9600, the collector's default. Re-confirm per unit if the
      hardware set changes.

## WT901C orientation-representation convention -- `NOT_TESTED`

- [ ] **Quaternion axis convention.** Note the device does not currently
      emit quaternion (`0x59`) packets at all, so this cannot be checked
      from captured data until the output mask is changed.
      `decode_frame`'s quaternion reorder
      (wire `(w,x,y,z)` -> schema `quaternion_xyzw = (x,y,z,w)`) is a
      structural reorder only, not a frame/axis convention check. Whether
      the WT901C's own reported quaternion/Euler axes match this package's
      assumed robot-base frame convention (`docs/PRE_IMU_FORWARD_HEMISPHERE_
      PROTOCOL.md`'s `+X` forward / `+Y` lateral / `+Z` up) has never been
      checked against a real sensor mounted on the real arm. Flagged
      `UNVERIFIED_HARDWARE_SETTING` in `docs/07_WT901C_CODE_AUDIT.md` for
      the closely-related quaternion/Euler helper functions in
      `07_WT901C/WT901C_imu0/live_4imu_visualizer.py` (display-only there,
      but the underlying convention question is the same).

## ESP32 protocol

**Amendment (2026-08-05): this whole section was aimed at the wrong
firmware.** The board on this rig runs `03_SO-101_control/main` and speaks
the host-polled `STATE` protocol, not `02_SO-101_rtos_motor_driver`'s pushed
`STREAM` protocol (`DECISIONS.md` #3/#7). The `STATE` path is now
`RESOLVED`: `collectors/collect_esp32_state_poll.py` was replayed against
every real encoder record — 310/310 (seg006), 542/542 (seg007), 759/759
(seg012) fully parsed, all 3 axes, all 25 firmware fields
(`docs/ENCODER_COMMAND_COVERAGE.md`). The `STREAM` items below remain
`NOT_TESTED` and will stay that way unless that firmware is ever flashed
here.

Two `STATE`-path items that are open and are NOT covered by that replay:

- [ ] **`NOT_TESTED` — reconnect timing against a real dropped USB link**
      on the state-poll collector.
- [ ] **ST3215 raw register → physical units.** `raw_voltage`/`raw_temp`/
      `raw_current`/`raw_load` are stored as counts, unscaled; no datasheet
      in the repo. **BLOCKED**, not merely untested.

- [ ] **`NOT_TESTED` (STREAM) — End-to-end against real
      `02_SO-101_rtos_motor_driver/main`
      firmware.** `collectors/collect_esp32_encoder.py`'s parser and
      `Esp32EncoderCollector` have only ever been run against
      `mocks/mock_esp32.py`'s synthetic `STREAM` lines. The line format was
      transcribed carefully from `02_SO-101_rtos_motor_driver/README.md`
      and `02_SO-101_rtos_motor_driver/main/README.md` (see
      `firmware/esp32_so101_raw_bridge/serial_protocol.md`), but transcribed
      correctly is not the same claim as verified against a real board.
- [ ] **`-999.00` sentinel handling on a real uncalibrated unit.** The
      `joint_limits_not_ready` quality-flag path
      (`q_encoder_rad[i] = None` when `deg == -999.0`) is exercised in unit
      tests with synthetic input, not against a real ESP32 that actually
      has `mapping_ready=false` for one or more joints.
- [ ] **Reconnect timing against a real dropped USB link** (vs. the fault
      injector's simulated `ConnectionError`) -- real USB-serial disconnect/
      reconnect timing and error types can differ from what
      `mocks/fault_injector.py:with_disconnect` simulates.

## Vive Tracker -- `PARTIAL`

**Amendment (2026-08-05):** the original claim ("no real OpenVR/SteamVR
integration exists at all", `VivePoseSource` satisfied only by
`mocks/mock_vive.py`) is stale. `OpenVRTrackerSource` exists in
`collectors/collect_vive_tracker.py` and is covered by
`tests/test_openvr_tracker_source.py` against a stub `openvr` module, and
real Vive data was captured (segments 006/007: 100% valid). Remaining:

- [ ] **`OpenVRTrackerSource` specifically has never run against real
      SteamVR.** The equivalent call pattern did, inside
      `experiment_scripts/collect_3d_sweep_50_v2.py`, but this class did not.
      `docs/STATUS.md`.
- [x] **Reconnect / sleep recovery** — mitigated, not solved (2026-08-12).
      Segment 012 recorded 759 correctly-invalid records because the tracker
      was asleep; the no-forward-fill policy held. Nothing recovers from it
      *in-run* still, but the sweep script now aborts before touching a motor
      if the tracker is absent, and `probe_vive.py` runs first every time.
      Segments 036-041 came in at ~99.6% `tracking_valid`.
      `docs/PRE_RUN_CHECKLIST.md` A1.
- [x] **A run with IMU + encoder + Vive all valid and time-aligned** —
      achieved 2026-08-10, segments 036-041. `docs/STATUS.md`.

## Adaptive tip-IMU settling detector -- module defaults still `UNVALIDATED`

**Amendment (2026-08-12):** the *thresholds actually used* to collect the
data were measured, not defaulted — `SETTLING_CONFIG` in
`experiment_scripts/collect_3d_sweep_50_v2.py` was derived from real
`imu3_tip` data (segment 014, 9,350 samples): 0.3 s-window gyro RMS separates
still (0.0005–0.0045 rad/s) from moving (0.15–0.32 rad/s) by ~34x, and 0.02
rad/s sits in the empty middle. That exercise also proved two module defaults
*unusable* here — `tip_output_rate_hz=100.0` and `tip_expected_period_s=0.01`
against a real ~10.07 Hz per-packet-type rate, which would leave `ready` and
`freshness_ok` permanently False.

The list below therefore describes the **in-module defaults**, which are
unchanged and must not be used as-is on this rig:

- [ ] `tip_gyro_rms_threshold_rad_s` (0.05, ~2.9 deg/s)
- [ ] `dynamic_accel_rms_threshold_mps2` (0.25)
- [ ] `orientation_spread_threshold_rad` (0.0175, ~1 deg)
- [ ] `jerk_rms_threshold_mps3` (25.0) -- the module's own docstring flags
      this one as needing "recalibration against a real WT901C noise
      spectrum," since the mock's settled-mode accel noise is "likely
      pessimistic vs. real hardware," making a naive finite-difference
      jerk metric especially sensitive to the true noise floor.
- [ ] `drift_slope_threshold_rad_s` (0.0175, ~1 deg/s trend)
- [ ] `base_gyro_threshold_rad_s` / `link_gyro_threshold_rad_s` (0.35 each)
- [ ] `candidate_duration_s` / `confirmed_duration_s` / `measurement_window_s`
      debounce timings (0.3s / 0.6s / 0.9s)

None of these have been checked against real tip-IMU noise-floor data. See
`docs/TASKS.md` for the recommended preregistration-style freeze process
before trusting them on a real experiment run.

## Sampling rate -- `BLOCKED ON DEVICE CONFIG`

**Amendment (2026-08-05):** the premise ("neither real hardware nor a real
motion reference existed") is stale, but the comparison still cannot be run.
The real units emit ~10.1 Hz per packet type (~40.3 Hz combined), using ~46%
of the 9600-baud byte budget — the bottleneck is the sensor's own internal
output-rate register, not the link (`docs/SAMPLING_AND_UART_BUDGET.md`).

- [ ] **Raise the WT901C output-rate register** (and baud with it) before a
      100Hz-vs-200Hz comparison is even meaningful. Nothing in this repo
      writes that register today.
- [ ] **Then** run the 100Hz vs 200Hz pilot comparison
      (`docs/SAMPLING_AND_ALIASING.md`).

## USB device identification -- `NOT_TESTED`

- [ ] **Real VID/PID/serial values for every sensor.** Every value in
      `config/sensor_map.example.yaml` is still a placeholder
      (`REPLACE_ME_*`) — real sessions so far opened known COM ports
      directly rather than going through `match_sensor_map()`.
      `core/port_discovery.py:match_sensor_map()`'s matching logic is
      unit-tested with synthetic port dicts (`port_discovery.py`'s own
      `_self_check()`), not against real enumerated devices.

## Servo calibration -- not on this list, but no longer for the old reason

**Amendment (2026-08-05).** The original note said servo calibration was out
of scope because "this package sends no motion commands and never will",
that the values were `-1` placeholders, and that calibration belonged to
`02_SO-101_rtos_motor_driver`'s `CAL_AUTO RUN`. All three are wrong now:

- The servos on this rig **are** calibrated; the real values live in servo
  EEPROM and are in `config/joint_limits.reference.yaml`. The `-1`s were
  only the compile-time defaults in `robot_config.h`.
- Real motion runs are driven from `experiment_scripts/` (not from
  `collectors/`, which remain read-only).
- The firmware here is `03_SO-101_control/main`, not
  `02_SO-101_rtos_motor_driver`.

What *is* still untested on hardware, and belongs on a plan somewhere:

- [ ] **Sequential axis dispatch actually lowering peak current.** Logic
      matches the firmware's documented batching behaviour but **peak
      current was never measured** (`docs/ST3215_THERMAL_AND_SEQUENCING.md`).
- [ ] **`AXIS_COMMAND_GAP_S = 0.6s` being the right gap.** Chosen for margin
      over the 40 ms legacy batch window, not measured.
- [ ] Onboard servo temperature sensor is **KNOWN BROKEN** (owner-confirmed);
      all software temperature-trip logic was deliberately removed. An
      external sensor plus hardware power cutoff is the only thermal safety
      net. Not a test item — a standing condition.
