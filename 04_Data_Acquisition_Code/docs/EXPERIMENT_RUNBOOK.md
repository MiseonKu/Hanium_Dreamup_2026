# Experiment Runbook

How to actually run this package. Every command below was executed against
this repository while writing this document (see the end of each section
for what was observed); none is a guess at what "should" work.

**Amendment (2026-08-05).** The original blanket header said
`HARDWARE STATUS: NOT_TESTED` for anything involving real sensors. That is
now wrong: real hardware has been connected and real data collected (4x
WT901C on COM16-19 @ 9600, ESP32 running `03_SO-101_control/main` on COM20,
Vive Tracker via SteamVR) — see `sessions/real_motion_demo/` and
`docs/STATUS.md`.

**Amendment (2026-08-12).** Acquisition is **complete**: segments 000-041,
the fused sweeps being 036-041, prepared into a dataset on 2026-08-10. The
real run is driven by `experiment_scripts/` (that folder's README is the
runbook for it); the post-run pipeline is three commands:

```powershell
python tools/prepare_dataset.py     # recover partials -> Parquet -> gzip raw in place
python tools/build_trials.py        # one row per settled trial  (10,055 -> 9,951 keep)
python tools/build_timeseries.py    # uniform 50 Hz grid per trial (6,456,460 rows)
```

Note that `prepare_dataset.py`'s gzip phase leaves only `*.jsonl.gz` behind,
and `tools/inspect_jsonl.py` / `validate_raw_session.py` / `audit_sampling.py`
below still open plain `.jsonl` only — decompress a copy before pointing them
at an archived session (`docs/TASKS.md` #14).

What is still true, and is the narrower claim this document now makes:

- The `mock`/`soak` walkthroughs below are what was exercised **through
  `SessionController`**.
- Real motion has only ever been driven by `experiment_scripts/`, never
  through `SessionController`'s gated `experiment` mode (section 6).
- Before sending anything to the arm, read
  `docs/FIRMWARE_MOTION_HAZARDS.md`. `INIT` enables torque before writing a
  goal position and will lunge the arm to a stale target; `MOVE_RAW` is the
  safe primitive.

## 0. Install

From this package's root (`04_Data_Acquisition_Code/`):

```powershell
pip install -r requirements.txt
```

Then confirm the test suite passes before doing anything else:

```powershell
python -m pytest tests -q
```

Observed: `150 passed` on a re-run 2026-08-05 (was `77 passed` when this
document was first written, `131` earlier the same day). The suite is
actively growing — re-run it rather than trusting the number here.

## 1. Mock-mode walkthrough (main smoke test)

`tools/run_mock_session.py` runs `experiment/session_controller.py`'s
`SessionController` in `mock` mode: every collector is backed by
`mocks/*.py` instead of real hardware, for a fixed duration, then a clean
shutdown.

```powershell
python tools/run_mock_session.py --duration-s 5
```

Observed output (session id/timestamps will differ on your run):

```text
session: session_20260804_001
dir:     <package_root>\sessions\session_20260804_001
collector_status: {'events': 'started', 'health': 'started', 'imu0_base': 'started', 'imu1_link_root': 'started', 'imu2_link_mid': 'started', 'imu3_tip': 'started', 'encoder': 'started', 'vive_tracker': 'started'}
{
  "session_id": "session_20260804_001",
  ...
  "session_complete": true,
  "interrupted": false,
  ...
}
```

Exit code `0` means `manifest.json["session_complete"] is True`. A
directory `sessions/session_YYYYMMDD_NNN/` is created (override the parent
with `--sessions-root <path>`), containing:

```text
manifest.json
raw/
  imu0_base_segment000.jsonl
  imu1_link_root_segment000.jsonl
  imu2_link_mid_segment000.jsonl
  imu3_tip_segment000.jsonl
  encoder_segment000.jsonl
  vive_tracker_segment000.jsonl
  events_segment000.jsonl
  health_segment000.jsonl
indexes/
reports/
```

(`indexes/`/`reports/` are created empty by `SessionController.start()`;
nothing in this task's scope writes into them yet.)

### Inspecting the output

Summarize any one stream file (record count, timestamp span, valid/
checksum pass rate, quality-flag histogram):

```powershell
python tools/inspect_jsonl.py sessions\session_20260804_001\raw\imu3_tip_segment000.jsonl
```

Observed (your numbers will differ):

```text
records:            1916  (parse errors: 0)
valid pass rate:    100.0%
checksum_ok rate:   100.0%
quality_flags histogram:
  (none)
```

Compute configured-vs-measured sampling rate, jitter, max gap, and packet
loss for that same file (see `docs/SAMPLING_AND_ALIASING.md` for what these
mean and why the IMU's measured rate is ~4x its `configured_output_rate_hz`):

```powershell
python tools/audit_sampling.py sessions\session_20260804_001\raw\imu3_tip_segment000.jsonl
```

Validate the whole session directory (manifest present/parses, every
enabled stream has a raw file, every `.jsonl` file parses line-by-line):

```powershell
python tools/validate_raw_session.py sessions\session_20260804_001
```

Observed: `PASS` for every check, ending in a final `PASS`.

## 2. Soak mode (extended-duration stability run)

Same mock-backed `SessionController`, `execution_mode="soak"` in the
manifest, defaulting to a much longer duration (master-spec section 22:
30+ minutes, no motor commands ever sent regardless of duration).

```powershell
python tools/run_soak_test.py --duration-s 1800
```

For a quick check that the command itself works (not a real soak run), a
short duration was used here instead and observed to complete cleanly:

```powershell
python tools/run_soak_test.py --duration-s 5
```

Same manifest/raw-file shape as mock mode above, just tagged
`"execution_mode": "soak"`.

## 3. Port inventory (no sensors required to run, only to be useful)

```powershell
python tools/inventory_ports.py
```

Prints every currently-enumerated serial device (DEVICE/DESCRIPTION/VID/
PID/SERIAL_NUMBER/MANUFACTURER/LOCATION). With no relevant sensors plugged
in, this just lists whatever else is on the machine (Bluetooth virtual
ports, unrelated USB-serial adapters, etc.) -- that's expected, not an
error. See `docs/WINDOWS_DRIVER_SETUP.md` and
`docs/HARDWARE_CONNECTION_CHECKLIST.md` for using this against real
sensors.

## 4. Front-120 target plan generation

Generates a target plan by calling into
`02_FK_IK`'s existing generator with
`--mode front_120` forced (see `docs/TARGET_WORKSPACE_FRONT_120.md`):

```powershell
python tools/generate_front_120_target_plan.py --stage P2_TWO_LINK_RIGID_TARGET_VIVE_LABEL --count 20 --seed 202
```

Observed:

```text
Targets: 20
Candidates: 800
Rejected: 6
Output: ...\outputs\pre_imu_front_120_targets.csv
Metadata: ...\outputs\pre_imu_front_120_targets_metadata.json
Pre-IMU target generation OK
```

Then audit the result against the acceptance criteria (zero azimuth
violations, zero duplicates, requested count == generated count):

```powershell
python tools/audit_target_plan.py --csv outputs\pre_imu_front_120_targets.csv --metadata outputs\pre_imu_front_120_targets_metadata.json --expected-count 20
```

Observed: every line `OK`/`0`, final line `PASS`.

## 5. Recovering an interrupted session

If a session ends uncleanly (crash, kill, power loss), its most recent
segment file is left as `<stream>_segment{NNN}.jsonl.partial` instead of
being renamed to `.jsonl`. Recover it with:

```powershell
python tools/recover_partial_jsonl.py <path-to-file>.jsonl.partial
```

This never modifies the original `.partial` file -- it writes
`<name>.recovered.jsonl` (every successfully-parsed record) and
`<name>.recovery_report.json` (what was dropped, and any `host_seq` gaps)
alongside it. Verified against a synthetic truncated `.partial` file
(4 lines, the last one deliberately cut mid-write, plus a deliberate
`host_seq` gap):

```text
scanned:            4 lines
valid records:      3
dropped fragments:  1
  - from line 4 (1 line(s)), truncated=False
host_seq gaps:      1
  - 1 -> 5 (missing 3)
recovered file:     ...\imu3_tip_segment000.recovered.jsonl
recovery report:    ...\imu3_tip_segment000.recovery_report.json
```

See `docs/FAILURE_HANDLING.md` for the rest of the failure-handling picture.

## 6. Passive / experiment mode -- NOT exercised in this task

`experiment/session_controller.py` defines two further modes,
`passive` and `experiment`, in its `MODES` tuple. **Neither was run as part
of this task, and neither is safe to run casually:**

- **`passive`**: runs the real collectors against real sensors (not mocks)
  while recording, with no motion commands. **Amended 2026-08-05:** real
  port resolution is now wired — pass
  `SessionController(sensor_map_path=<your sensor_map.yaml>)` and
  `_start_imu`/`_start_encoder`/`_start_vive` resolve ports through
  `core.port_discovery.match_sensor_map()`
  (`experiment/session_controller.py:337`, `:654`, `:683`, `:779`, `:824`).
  Without a `sensor_map_path` each collector still fails closed, now with
  `"no sensor_map configured"` / `"sensor_map has no matching port for
  ..."`, rather than guessing a COM port. The earlier claim that this mode
  "is not actually usable yet" and the `"no real port resolution configured
  (mode != mock)"` message are both stale. It has still not been run
  end-to-end against hardware.
- **`experiment`**: the only mode that would ever be allowed to progress
  through target dispatch. It requires **both** `hardware_motion_enabled`
  (config) **and** `--enable-hardware-motion` (CLI) to be true
  simultaneously (`experiment/safety_gate.py`'s double-arm gate) -- as
  implemented today, `SessionController._check_experiment_gate()` reads the
  config-side flag from the `SO101_ALLOW_EXPERIMENT_MODE=1` environment
  variable. Without both, `SessionController.start()` logs
  `"experiment_mode_refused"` and raises `RuntimeError` before starting
  anything.

**Do not attempt to exercise either mode against real hardware without:**
completing `docs/HARDWARE_CONNECTION_CHECKLIST.md`, `docs/PRE_RUN_CHECKLIST.md`
and `docs/HARDWARE_TEST_PLAN.md` first, reading
`docs/FIRMWARE_MOTION_HAZARDS.md` (non-negotiable — this firmware has motion
hazards that have already caused unintended travel), explicit operator
approval for any hardware motion, and a clear workspace. This package was
designed so that even a misconfigured `experiment` attempt fails closed
(refuses to start) rather than moving anything -- but that design has itself
never been exercised end-to-end against real hardware, so don't treat
"fails closed in this task's tests" as equivalent to "verified safe on a
real arm."

Note the safety-model cross-reference used to point at
`02_SO-101_rtos_motor_driver/README.md`. That is the wrong firmware: this
rig runs `03_SO-101_control/main` (`DECISIONS.md` #3/#7,
`docs/ENCODER_COMMAND_COVERAGE.md`). Its hazards are documented in
`docs/FIRMWARE_MOTION_HAZARDS.md`.
