# Tasks -- Open Work

Short remaining-work list for whoever picks this package up next. Pulled
from `grep -rn "TODO\|FIXME\|UNVERIFIED_HARDWARE_SETTING\|not implemented\|
ponytail:\|UNVALIDATED\|documented simplification\|see report"` across this
package's own code and docs, plus direct reading of the modules involved.
Ordered roughly by "blocks everything else" first.

## 1. Real hardware bring-up -- DONE

~~Nothing in this package has touched real hardware. Before anything below
about "verification" can happen, someone needs a real WT901C, a real ESP32
running `02_SO-101_rtos_motor_driver/main`, and (separately) a Vive base
station + tracker.~~

**Amendment (2026-08-05):** done, and the firmware named above was wrong.
Bring-up happened with 4x WT901C (COM16-19 @ 9600), an ESP32 running
**`03_SO-101_control/main`** (COM20 @ 115200 — `02_SO-101_rtos_motor_driver`
was never flashed here, `DECISIONS.md` #3/#7), and a Vive Tracker via
SteamVR. Real data: `sessions/real_motion_demo/` segments 000-041 (000-012 as
of that date).

~~What replaced this item: the remaining gap is **one run with IMU + encoder
+ Vive all valid and time-aligned** — 006/007 have all three but are short
pilots, 012 has good encoder/motion but 0% Vive.~~

**Amendment (2026-08-12): closed.** Segments 036–041 (2026-08-05 →
2026-08-10) are fully fused: all three streams valid and time-aligned,
~99.6% Vive valid, 9,951 usable trials over 50 targets. Acquisition for this
experiment is **complete**; the dataset was prepared, collapsed and
resampled on 2026-08-10 (`docs/STATUS.md` → "Collected dataset"). Nothing
below blocks data collection any more — the open items are analysis-side or
would only matter to a *future* run.

## 2. WT901C protocol verification against a real unit

Full, itemized checklist: `docs/HARDWARE_TEST_PLAN.md`. Highlights, all
`UNVERIFIED_HARDWARE_SETTING` per `docs/07_WT901C_CODE_AUDIT.md`:

- Full-scale accel/gyro/angle range constants
  (`collectors/collect_imu_rs232.py:IMUScaleConfig`, currently ±16g/
  ±2000dps/±180°).
- Byte-order/frame-layout convention (`s16`, `0x55` header/port-status
  overload).
- Default baud (`IMURS232Collector` defaults `baud=9600`).
- Quaternion/Euler axis convention vs. the assumed robot-base frame.

**Amendment (2026-08-05):** two of these four are resolved by the real
captures — baud 9600 is confirmed correct for these units, and the
byte-order/frame-layout convention is confirmed across 95k+ real frames at
100% checksum-OK (which also caught a real bug: the angle packet's word4 was
being mislabeled `temperature_c`). The **full-scale-range constants remain
genuinely unverified** — they were never read back from the device's range
registers, and captured data cannot settle them. The quaternion convention
also remains unverifiable for now: the device never emits `0x59` packets
with its current output mask. `docs/IMU_DATA_COVERAGE.md`.

## 3. Real Vive/OpenVR integration behind `VivePoseSource` -- DONE (mostly)

~~`VivePoseSource` is satisfied today only by
`mocks/mock_vive.py:MockViveTracker`. A real implementation needs the actual
SteamVR/OpenVR runtime and hardware -- out of scope for this task.~~

**Amendment (2026-08-05):** `OpenVRTrackerSource` now exists in
`collectors/collect_vive_tracker.py`, and the swap-in-via-`source=` design
held — `ViveTrackerCollector` was not touched. Coverage is
`tests/test_openvr_tracker_source.py` against a stub `openvr` module; **this
class specifically has never run against real SteamVR** — every real capture
went through `experiment_scripts/collect_3d_sweep_50_v2.py`'s own inline
`openvr` calls instead. That is the remaining gap: the call pattern is
proven on hardware, this wrapper class is not.

## 4. Preregistration-style freeze of the adaptive settling thresholds
   -- DONE for the run that collected the data

Every threshold in `experiment/adaptive_settling_detector.py:SettlingConfig`
is self-documented as `UNVALIDATED`, "picked for this task's synthetic
tests only." Before any real experiment run relies on
`TIP_IMU_STABLE_CONFIRMED`/`MEASUREMENT_WINDOW` transitions, these need a
preregistration-style freeze against real tip-IMU noise-floor data --
i.e. capture real still-vs-moving tip-IMU data first, pick thresholds from
that data, and lock them in before using them to gate real measurements
(not tune them against the same data you're about to collect with them).

**Amendment (2026-08-12):** done, but *outside* the module. The real sweeps
used `AdaptiveSettlingDetector` with `SETTLING_CONFIG` defined in
`experiment_scripts/collect_3d_sweep_50_v2.py`, and every number in it was
measured from `imu3_tip` in segment 014 (9,350 real samples, 232 s) before
the collection runs — the 0.3 s-window gyro RMS separates still (0.0005–0.0045
rad/s) from moving (0.15–0.32 rad/s) by ~34x, and the 0.02 rad/s threshold
sits in the empty middle. That freeze also caught two module defaults that
are *unusable* on this hardware: `tip_output_rate_hz=100.0` and
`tip_expected_period_s=0.01`, against a real ~10.07 Hz per-packet-type rate —
`min_samples` could never be reached and `freshness_ok` would be False on
every evaluation.

Still open: `SettlingConfig`'s in-module defaults are unchanged and still
`UNVALIDATED`, so anything using them without overriding (i.e. any future
`SessionController`-driven run) inherits the two broken values above. Copy
the measured config out of the sweep script, or make the defaults match real
WT901C rates. `jerk_rms_threshold_mps3` in particular is flagged in the
module's own docstring as likely to need adjustment (the mock's settled-mode
accel noise is "likely pessimistic vs. real hardware"); the sweep script
loosened it to 40.0 because accel and jerk do not separate still from moving
anywhere near as cleanly as gyro does on this rig.

## 5. Config loader -- PARTIAL

**Amendment (2026-08-05):** the first bullet is done. The blanket claim "no
`.py` file in this package imports `yaml` or reads them" is stale —
`SessionController._load_sensor_map()`
(`experiment/session_controller.py:337`) reads a `sensor_map:`-shaped YAML
and feeds it to `match_sensor_map()` for real-mode port resolution. The
other two bullets are still open; nothing reads `experiment.yaml` or
`sampling.yaml`.

- ~~reads `sensor_map.yaml` and feeds it to `core.port_discovery.
  match_sensor_map()`,~~ **done**
- reads `experiment.yaml`'s `hardware_motion_enabled` as the config-side
  input to `experiment.safety_gate.hardware_motion_allowed()` (today that
  value comes from the `SO101_ALLOW_EXPERIMENT_MODE=1` environment variable
  instead -- see `experiment/session_controller.py:_check_experiment_gate`),
- reads `sampling.yaml` into the various collector constructor kwargs
  (`queue_expected_rate_hz`, `flush_interval_s`, etc.) instead of their
  current hardcoded defaults.

## 6. Real port resolution for `passive`/`experiment` modes -- DONE

~~`_start_imu`/`_start_encoder`/`_start_vive` all fail closed with `"no real
port resolution configured (mode != mock)"` for any mode other than
`mock`/`soak`.~~

**Amendment (2026-08-05):** wired. Pass
`SessionController(sensor_map_path=...)`; ports resolve through
`match_sensor_map()` (`session_controller.py:654`), the encoder goes through
`collect_esp32_state_poll.py` (the rig's actual firmware), and the Vive
tracker serial comes from the optional `vive_tracker:` entry. Without a
sensor map each collector still fails closed, now with `"no sensor_map
configured"` / `"sensor_map has no matching port for ..."`.

Still open: nobody has run a real `passive` session end-to-end. All real
captures so far came from `experiment_scripts/`.

## 7. Multi-segment rotation at the orchestrator level

`SessionController.segment_id` is hardcoded to `0` for the whole session
(`experiment/session_controller.py`, comment: "single segment per session
in this task's scope"). `core/jsonl_writer.py`'s `start_new_segment()`
already supports rotating to a new segment cleanly, but nothing in
`SessionController` calls it -- e.g. on a size/time-based rollover, or to
recover mid-session after a partial collector failure without restarting
the whole session.

## 8. Dedicated safety-gate test file

`experiment/safety_gate.py`'s `hardware_motion_allowed`/
`assert_hardware_motion_allowed` are covered only by that module's own
`_self_check()` and indirectly by `session_controller.py`'s inline
fallback-AND logic. No `tests/test_safety_gate.py` exists in `tests/` --
worth adding given this is the one function standing between "mock only"
and "could move a real arm."

## 9. `workspace_mode` naming collision -- RESOLVED

`core/schemas.py`'s `base_record(workspace_mode=...)` field (the
azimuth-sector target mode -- `pilot`/`main`/`front_120`) and
`experiment/session_controller.py`'s manifest field (the session execution
mode -- `mock`/`soak`/`passive`/`experiment`/`inventory`) shared a name.
Fixed: `session_controller.py`'s manifest now writes `execution_mode`
instead, with a code comment pointing at this distinction; the
`SessionController` object's own `.mode` attribute is unchanged (only the
manifest key was renamed). `tests/test_session_controller_smoke.py` and
`config/experiment.example.yaml` were updated to match. No component yet
*writes* the azimuth-sector `workspace_mode` into a raw record (that
requires wiring a target plan into the session controller, still open --
see task 5-adjacent work), so this was a pure naming fix, not a behavior
change.

## 10. `psutil` decision for health metrics

`core/health_supervisor.py`'s `cpu_percent`/`memory_percent` are always
`None` today -- `psutil` is not a dependency anywhere in this repo (checked
via `grep -rl psutil` from the capstone-code root, zero hits), and
`os.getloadavg()` is POSIX-only. If real CPU/memory health signals become
necessary, add `psutil` as a real dependency (and to `requirements.txt`)
rather than working around its absence.

## 11. Threads vs. processes

`experiment/session_controller.py` uses `threading.Thread` for every
collector, a deliberate choice given no real hardware and no observed GIL
contention (see that module's own docstring). Revisit only if a real-
hardware run shows actual contention -- the collector interfaces already
communicate via queues/files, not shared Python objects, so a switch to
`multiprocessing` shouldn't require touching collector internals.

## 12. If a full 5-DOF pre-IMU target generator is ever needed

The `front_120` addition (`docs/TARGET_WORKSPACE_FRONT_120.md`) extends the
existing 3-DOF (`q1`/`q2`/`q3`) pre-IMU pipeline in
`02_FK_IK/` as-is. `wrist_flex`/
`gripper` are outside that pipeline's scope entirely. A full 5-joint
version would be new, larger scope -- not a config addition.

Amendment (2026-08-05): confirmed not needed -- 3 axes is the intentional,
final experiment scope (`docs/THREE_AXIS_EXPERIMENT_SCOPE.md`), not a
placeholder. Downgraded from "future task" to "will not be done."

## 13. Minor: stale `HANDOFF.md` reference -- RESOLVED

~~`DECISIONS.md` section 2 states a documentation gap was "Flagged in
`HANDOFF.md`" -- but no `HANDOFF.md` file exists anywhere in the
repository.~~

**Amendment (2026-08-05):** it exists now —
`04_Data_Acquisition_Code/HANDOFF.md`. The `DECISIONS.md`
cross-reference resolves. Whether the AGENTS.md routing-table gap it
describes was ever actually closed is a separate question, outside this
package.

## 14. Session-reading tools don't open `.jsonl.gz` (new 2026-08-12)

`tools/prepare_dataset.py`'s gzip phase compresses raw segments in place, so
after a dataset prep the only files on disk are `*.jsonl.gz`. Three tools
still `open()` plain text and now fail on the archived sessions:
`tools/inspect_jsonl.py`, `tools/validate_raw_session.py`,
`tools/audit_sampling.py`. `tests/test_demo_replay.py` hit the same wall (10
failures) and was fixed by choosing the opener from the suffix — the same
two-line pattern applies to the tools. Until then, decompress a copy.

## 15. Analysis-side follow-ups on the collected dataset (new 2026-08-12)

Acquisition is done, so the live work is downstream. Nothing here is a bug:

- **`tracker_offset_mm` is `[0,0]`, unmeasured.** The FK-vs-Vive residual
  therefore mixes flexible-link deflection with a constant tracker offset.
  Separable — the offset is pose-independent, the deflection is not — but not
  yet separated. `experiment_scripts/README.md`.
- **104 of 10,055 trials are dropped** by the `keep` filter (short window,
  partial tracking loss, or no encoder attached). Segment 034 is the worst
  (3/9 kept, mean Vive valid ratio 0.44). Worth confirming the drops are
  faults and not a windowing artifact before publishing counts.
- **Segments 023–035 are pilots** (2–30 trials each) with an earlier plan
  version; check plan provenance before pooling them with 036–041.
- **`.recovered.jsonl.gz` files are not automatically preferred** over their
  `.partial` originals by anything downstream — `prepare_dataset.py` handles
  this in its recover phase, but ad-hoc analysis reading `sessions/` directly
  should be explicit about which it loads. `docs/RAW_DATA_POLICY.md`.
