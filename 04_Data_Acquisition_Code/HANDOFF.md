# HANDOFF.md

Updated **2026-08-18**. Earlier versions of this file said "hardware
`NOT_TESTED`, everywhere" and "no commit or push was run" — both were true
when written and are both false now. See the git history of this file rather
than trusting any single revision.

## What this is

The raw-data-acquisition system for the SO-101 flexible-link IMU/Vive
capstone experiment, plus the scripts that ran the experiment and the
pipeline that turned its output into an analysable dataset.

Rig: 4x WT901C IMU over RS232-to-USB, an ESP32 running
`03_SO-101_control/main` (3 axes: shoulder_pan / shoulder_lift / wrist_flex),
and a Vive Tracker via SteamVR mounted where wrist_roll/gripper would be.
3 axes is the intentional, final scope — `docs/THREE_AXIS_EXPERIMENT_SCOPE.md`.

Read in this order: `README.md` (structure, quick start),
`experiment_scripts/README.md` (how the real run works, and the hazards),
`docs/STATUS.md` (what is verified, against what), `DECISIONS.md` (why),
`docs/TASKS.md` (what is left).

## State: acquisition is COMPLETE

`sessions/real_motion_demo/` holds segments 000–041. The final fused sweeps
are **036–041** (2026-08-05 → 2026-08-10): the three acquisition systems ran
in the same segments and ~99.6% of Vive samples are `tracking_valid`. This is
an acquisition-level statement; the trial audit below documents missing
`imu1_link_root` summaries in segments 040/041.

Dataset, built 2026-08-10, all of it git-ignored and rebuildable:

| Layer | Numbers |
|---|---|
| Raw JSONL (`sessions/`) | 363 files, 49.0 GB → 3.3 GB gzipped in place |
| Parquet working copy | IMU 24,880,024 rows / Vive 9,372,862 / encoder 339,965 |
| `trials.parquet` | 10,055 settled trials, **9,951 usable** (`keep`), 50 targets × ~200 repeats |
| `timeseries.parquet` | 6,456,460 rows on a uniform 50 Hz per-trial grid |

Schema for the derived layers: `docs/DATA_SCHEMA.md` → "Derived dataset".
Logs of the actual runs: `prepare_dataset.log`, `build_timeseries.log`.

## Downstream learning contract (2026-08-18)

Workspace `09_SO101_IMU_VIVE_LEARNING` now owns the audited static-trial
preparation contract `alphaf.ml_preparation.v1`. It reads only the pinned
`trials.parquet` member from the tracked 06 `derived_dataset_archive.zip`; it
does not read ignored sessions or mutate this acquisition package.

The audit narrows the learning population beyond the acquisition-level
`keep` flag:

- source: 10,055 trial rows;
- canonical: 7,707 structurally complete `keep=true` rows from segments
  036-040;
- train eligible: 7,702 rows after label-blind sample-count/duration/range
  requirements;
- segment 040 contributes 600 kept rows with missing `imu1_link_root`, and
  five additional complete rows have fewer than 28 samples;
- segment 041 has no `imu1_link_root` trial summary and contributes no v1 row.

The previous sentence in this handoff that calls all final 036-041 sweeps
fully valid should be read as an acquisition-level status, not as proof that
every trial contains all four static IMU summaries.

`timeseries.parquet` is excluded from v1. Its current uniform-grid nearest
alignment can reuse future/repeated IMU samples and does not expose a complete
source-age/missingness/drop-provenance contract. A later time-series release
needs a new causal or explicitly bounded alignment contract.

Vive remains the raw `vive_world` XYZ-mm label/evaluation signal only. The
four-WT901C feature view is an endpoint-regression/fusion proxy, not the formal
three-IMU-per-link ShapeFK `IMU-only` condition. See the 09
`docs/DATA_CONTRACT.md`, `docs/COLLECTED_DATA_AUDIT.md`, and
`docs/METHODS_AND_REFERENCES.md`.

Workspace 09 also owns the separate exploratory
`alphaf.endpoint_regression.experiment.v1`. It consumes the immutable
`ml_preparation_v1.zip` whose SHA-256 is
`94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930`;
it does not rewrite that preparation release or read this workspace's ignored
sessions. Segment-LOSO is its primary nested-group evaluation and unseen-target
groups are secondary. Scaling, target transforms, model selection, and
augmentation parent search stay inside each real-train context.

The validated experiment package is published separately through Git LFS as
`06_simulation_and_data_collect_preparation/imu_added_experiment/endpoint_regression_experiment_v1.zip`.
Its OOF evidence and final full-data model remain exploratory within this
collected session. Predictions are static three-second-summary raw
`vive_world` XYZ mm, not independently validated robot-base/TCP positions,
formal ShapeFK, `IMU-only`, online aggregation, or control output.

```powershell
cd "09_SO101_IMU_VIVE_LEARNING"
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements-eval.lock
python -m pip install -e . --no-deps
python -m imu_vive_learning.cli run-experiment `
  --release ../06_simulation_and_data_collect_preparation/imu_added_experiment/ml_preparation_v1.zip `
  --config configs/endpoint_regression_v1.yaml `
  --output artifacts/endpoint_regression_v1 --jobs 4 --resume
python -m imu_vive_learning.cli validate-experiment --run-dir artifacts/endpoint_regression_v1
python -m imu_vive_learning.cli report-experiment --run-dir artifacts/endpoint_regression_v1
python -m imu_vive_learning.cli package-experiment `
  --run-dir artifacts/endpoint_regression_v1 `
  --output artifacts/endpoint_regression_experiment_v1.zip
python -m imu_vive_learning.cli predict-endpoint `
  --model-package artifacts/endpoint_regression_experiment_v1.zip `
  --input feature_rows.parquet --output predictions.parquet
```

These commands are offline. They do not invoke the Tracker SDK, serial ports,
torque, calibration, or motor motion. See the 09
`docs/ENDPOINT_REGRESSION_CONTRACT.md` and
`docs/ENDPOINT_REGRESSION_METHODS.md` before interpreting the report.

## What the real run actually did differently from this package's defaults

Anyone changing the collection path needs these four, all of them found the
hard way:

1. **Motion goes through `MOVE_RAW`, never `INIT` + `D<axis><deg>`.** The
   firmware's `INIT` enables torque *before* writing a target, so the servo
   jumps to a stale one — shoulder_lift once crossed 150° into its mechanical
   limit. `safe_engage()` uses `CAL_AUTO LOAD` → `MOVE_RAW(current)` →
   `TORQUE ON`. `docs/FIRMWARE_MOTION_HAZARDS.md`.
2. **Angles are raw ticks, never `logical_deg`.** `logical_deg` is a label
   scale, distorted per axis (1.173 / 1.300 / 1.101 physical degrees per
   label degree). Using it as an angle put a 30% scale error into
   shoulder_lift; switching to raw ticks cut the rigid-FK residual from
   **77.4 mm to 48.7 mm**. `docs/FK_CONVENTION_CORRECTION.md`. 1 physical
   degree = 11.378 ticks (ST3215, 4096 counts/360°).
3. **Settling thresholds were measured, not defaulted.** The sweep script's
   `SETTLING_CONFIG` comes from real `imu3_tip` data (segment 014, 9,350
   samples). `SettlingConfig`'s *in-module* defaults are still `UNVALIDATED`
   and two of them are unusable here (`tip_output_rate_hz=100.0`,
   `tip_expected_period_s=0.01` vs. a real ~10.07 Hz per packet type) —
   `docs/TASKS.md` #4.
4. **Probe the tracker before touching a motor.** Segment 012 ran all 50
   points with the tracker asleep: 0% valid Vive, an hour lost. The sweep
   script now aborts pre-motion if the tracker is absent, and
   `probe_vive.py` runs first, every time. `docs/PRE_RUN_CHECKLIST.md`.

## Data contract

7 record schemas in `core/schemas.py` (`alphaf.{imu,encoder,command,vive,
event,health,stability_metrics}.raw.v1`), documented in
`docs/DATA_SCHEMA.md`. Unchanged since introduction, except that the angle
packet's word4 was renamed `angle_packet_word4_raw` — it was mislabeled
`temperature_c`, and 95,856 real records across 4 physically separate sensors
plus WitMotion's protocol doc prove it is a version number
(`docs/IMU_DATA_COVERAGE.md`).

`experiment_state` on the encoder and Vive streams (`SETTLING` /
`MEASUREMENT_WINDOW` / `SETTLE_TIMEOUT` / `REST`) is what makes the dataset
selectable downstream — both derived builders key off it.

The `02_FK_IK` target-plan
CSV/metadata contract is unchanged; `front_120` reuses the same columns as
`pilot`/`main`.

## Test suite

150 tests. Last full re-run 2026-08-12: **140 passed, 10 failed** — all 10 in
`tests/test_demo_replay.py`, which opened `sessions/real_motion_demo/*.jsonl`
after `prepare_dataset.py` had gzipped those files in place. The opener is now
suffix-aware (`.jsonl` or `.jsonl.gz`); re-run to confirm 150. No collector or
`core/` module was implicated.

Same failure mode is still open in `tools/inspect_jsonl.py`,
`tools/validate_raw_session.py` and `tools/audit_sampling.py` —
`docs/TASKS.md` #14.

## Known limitations

- **WT901C full-scale ranges are assumed**, never read back from the device's
  range registers, so every physical unit derived from raw counts inherits
  that assumption. The quaternion convention stays unverifiable while the
  units' output mask omits `0x59` packets.
- **`tracker_offset_mm` is `[0,0]`, unmeasured**, so the FK-vs-Vive residual
  mixes flexible-link deflection with a constant tracker offset. Separable —
  the offset is pose-independent — but not yet separated. Do **not** try to
  shrink that residual by tuning link lengths or frames: the residual is the
  signal this experiment measures.
- **`SessionController` never drove a real run.** Real port resolution is
  wired (`sensor_map_path=`), but every real capture came from
  `experiment_scripts/`. `OpenVRTrackerSource` likewise is covered only by a
  stub `openvr` module.
- **Servo temperature protection is deliberately absent in software.** The
  shoulder_lift onboard sensor is broken and `raw_temp` is register counts
  with no datasheet to convert. The real safety net is an external sensor plus
  hardware power cutoff. Sequential axis dispatch is the thermal mitigation in
  code, and its peak current was never measured
  (`docs/ST3215_THERMAL_AND_SEQUENCING.md`).
- **Threads, not processes**, and one segment per session at the orchestrator
  level (`JsonlWriter.start_new_segment()` exists but nothing calls it) — both
  deliberate, `docs/TASKS.md` #7 / #11.
- **Segments 023–035 are short pilots** (2–30 trials) on earlier plan
  versions; check plan provenance before pooling them with 036–041.

## Git

This package was committed and pushed to
`Inha-EEE-Capstone-26/capstone-code` on 2026-08-12 as
`04_Data_Acquisition_Code/` (102 files, code and docs only).
`sessions/`, `derived/`, `outputs/`, `*.zip`, `*.jsonl*` and logs are
git-ignored — no collected data is in the repository, per
`docs/RAW_DATA_POLICY.md` and the root `AGENTS.md`. The three modified files
in `02_FK_IK/` and
`03_SO-101_control/` were left uncommitted.

## Next exact action

Acquisition is done. For static learning data, build and validate workspace
09's pinned preparation contract; do not recreate one-off filtering or use the
legacy pre-split augmentation archives. For the frozen endpoint experiment,
run the 09 `run-experiment`, `validate-experiment`, `report-experiment`, and
`package-experiment` commands against the immutable preparation ZIP. Use
`predict-endpoint` only for offline static-summary inference with the packaged
allow-list. Model fitting, scaling, hyperparameter selection, and augmentation
must remain inside each published real-train context.

Tracker-to-robot-base/TCP calibration remains a separate future measurement.
Do not subtract the unmeasured `[0,0]` offset or fit Kabsch on all labels. If a
new collection run is explicitly requested, start from
`docs/PRE_RUN_CHECKLIST.md` and the measured sweep `SETTLING_CONFIG`; none of
the 09 offline commands move hardware.
