# DECISIONS.md

Decisions made before/during implementation, with the audit evidence behind
each one. All four survey/audit agents (07_WT901C code audit, target-
generator + active-firmware audit, ML reference repo schema audit, Korean
policy-doc extraction) are cited by file where relevant; full detail lives in
`docs/07_WT901C_CODE_AUDIT.md` and this file.

## 1. Package location and number

`08_` was unused at task start (`01_control` .. `07_WT901C` were the only
numbered packages). Created `04_Data_Acquisition_Code/` per the
master spec's default recommendation — no numbering conflict to resolve.

## 2. Target-generation authority: `02_FK_IK/`, not `06_.../`

The master spec's own priority list (section 2.3) pointed at
`06_simulation_and_data_collect_preparation/`. Investigation found that
folder's `p1_one_link_experiment/` and `p2_two_link_experiment/` are reduced-
DOF (1-link, 2-link) teaching/validation forks — confirmed duplicated module
names and a `pilot_azimuth_deg: [-30, 30]` value copied verbatim from the
real source (`p1_forward_workspace.yaml:12`).

The actual pre-IMU forward-hemisphere target generator with real IK, joint
limits, and an azimuth-sector mode dictionary lives in
`02_FK_IK/`
(`docs/PRE_IMU_FORWARD_HEMISPHERE_PROTOCOL.md`,
`workspace_points/pre_imu_targets.py`). This is the system extended for
`front_120`, per master-spec section 21.2 ("실제 repository convention이
다르면 기존 구현을 따른다").

**DOF mismatch, deliberately deferred to existing code:** the master spec
describes the arm as "5-DOF." The pre-IMU pipeline in
`02_FK_IK/` models only 3 joints
(`q1`=base yaw, `q2`=shoulder pitch, `q3`=elbow pitch —
`config/robot_config.yaml:3` `dof: 3`; README.md:1,9 states "SO-101 3DoF");
wrist_flex/gripper are outside its scope. Per master-spec section 21.2's own
override clause, the existing repository convention wins over the spec's
"5-DOF" assumption — `front_120` is added to this 3-DOF pipeline as-is, not
rewritten to cover 5 joints. This is a pre-existing repo characteristic, not
something introduced by this task.

**Amendment (2026-08-05):** confirmed by the project owner that 3 axes
(shoulder_pan/shoulder_lift/wrist_flex) is the intentional, final
experiment scope, not a temporary shortfall — wrist_roll/gripper carry a
Vive Tracker mount instead of a motor for this experiment. See
`docs/THREE_AXIS_EXPERIMENT_SCOPE.md`. The reasoning above (pipeline
predates this task, override-clause) is still accurate for *why the code
already matched*; it's no longer the reason 3 axes is *correct*.

**Exact mechanism (additive, verified in `workspace_points/pre_imu_targets.py`):**
```
v = p - center                                  # center = [0,0,0] m
forward_dot = v · [1,0,0] > 0                    # full ±90° hemisphere gate
azimuth_deg = degrees(atan2(v_y, v_x))
lo, hi = region.azimuth_limit_deg[mode]          # mode-specific narrower gate
accept if lo <= azimuth_deg <= hi
```
`config/pre_imu_forward_hemisphere_workspace.json` already has
`"azimuth_limit_deg": {"pilot": [-30.0, 30.0], "main": [-45.0, 45.0]}`.
**Decision: add a third key `"front_120": [-60.0, 60.0]`** — a pure config
addition, no formula change, since `is_inside_forward_trimmed_hemisphere`
and `classify_stage2_bins` already read `azimuth_limit_deg[mode]`
generically. `scripts/generate_pre_imu_targets.py`'s
`--mode choices=["pilot","main"]` argparse list must also gain
`"front_120"` — additive, 1-line list extension, not a logic rewrite.

Full pipeline order to preserve (do not reorder or add IK/collision checks
that don't exist upstream — the pre-IMU path is intentionally IK-free,
targets are direct rigid-FK positions):
`load config → sample joint offsets → rigid FK → hemisphere+azimuth gate →
bin classify → bin-balanced sampling → write CSV + metadata (SHA-256
digest)`. The separate full-workspace (non-pre-IMU) 360° pipeline in
`workspace_points/candidate_generator.py` is untouched, per the existing
docs' own statement that it's a "separate config, module, CLI, and data
contract."

**Documentation gap noted, not fixed by this task:** neither
`02_FK_IK/` nor
`02_SO-101_rtos_motor_driver/` nor `07_WT901C/` appear in root `AGENTS.md`'s
workspace-routing table or `configs/agent_context.json`, and none has a
local `AGENTS.md`. Flagged in `HANDOFF.md`; out of scope to fix here (would
be an unrelated docs change to someone else's package).

## 3. Active firmware: `02_SO-101_rtos_motor_driver/main`, not `03_SO-101_control/main`

Root `README.md` (dated 2026-05-03) claims `03_SO-101_control/main` is the
active hardware target. Git history shows `02_SO-101_rtos_motor_driver` was
created 2026-05-14 (`69aa903`) and promoted internally to be its own `main`
on 2026-05-15 (`66941ad`, replacing a placeholder `main2`), with
`03_SO-101_control/main`'s last real commit dated 2026-04-30 — the README
predates and was never updated after the RTOS work.

Decisive evidence: `02_SO-101_rtos_motor_driver/main/src/main.cpp` already
implements a `STREAM ON <period_ms>` / `ENCODER ON [period_ms]` ASCII
telemetry protocol (`02_SO-101_rtos_motor_driver/README.md:67-141`,
`02_SO-101_rtos_motor_driver/main/README.md:206-284`) that emits `t_ms, seq,
state flags, raw encoder positions, feedback timestamps, logical angles,
targets` per line — `03_SO-101_control/main/src/main.cpp` has zero matches
for `STREAM`/`ENCODER` and no raw-telemetry command at all. Both target
identical hardware (same axis names/IDs/ranges, same UART pins/baud —
`02_SO-101_rtos_motor_driver/include/rtos_motor_config.h:80-86` vs.
`03_SO-101_control/main/include/robot_config.h:73-79`).

**Decision:** `collectors/collect_esp32_encoder.py` and
`firmware/esp32_so101_raw_bridge/` wrap **05's `STREAM`/`ENCODER` ASCII
protocol**, not 03. This is read-only telemetry consumption (`STREAM
ON`/`ENCODER ON` commands, no `TORQUE`/`INIT`/`MOVE_RAW`/calibration
commands are ever sent) — consistent with master-spec section 1.5 (no real
motor commands) and root `AGENTS.md`'s hardware-safety rules. Existing
motor control / FK-IK inside `02_SO-101_rtos_motor_driver` is not modified;
the bridge is a new additive serial-client module that only reads the
already-implemented stream.

## 4. WT901C reuse decisions (full audit: `docs/07_WT901C_CODE_AUDIT.md`)

Blanket status: `HARDWARE_NOT_TESTED` for everything under `07_WT901C/` — no
real-hardware validation evidence exists anywhere in that folder.

| Source (07_WT901C, commit `3a0c757e9`) | Verdict | What the new package does |
|---|---|---|
| `WT901C_imu0/collect_4imu.py`: `extract_frames`, `s16`, `decode_frame`, `FRAME_NAMES`, `detect_baud` | REUSABLE_AFTER_FIX | Reimplemented in `collectors/collect_imu_rs232.py` with the same header (`0x55`)/checksum (8-bit sum mod 256)/resync/partial-frame logic, plus new unit tests (`tests/test_wt901_parser.py`) covering valid frame, corrupted-checksum resync, split-across-reads partial frame. Full-scale-range constants (±16g/±2000dps/±180°) carried over as **config-adjustable defaults**, each tagged `UNVERIFIED_HARDWARE_SETTING` in `docs/07_WT901C_CODE_AUDIT.md` and in the new package's sensor config. |
| `WT901C_imu0/live_4imu_visualizer.py`: `SensorReader` (reconnect/epoch design) | REUSABLE_AFTER_FIX | Reconnect-with-backoff + `connection_epoch` stamping design carried into `collectors/collect_imu_rs232.py`'s reader thread, decoupled from Tkinter, with the unlocked-`self.serial` race fixed (single owner + explicit lock). |
| `robust_io.py`: `AsyncJsonArrayWriter` (flush/fsync/drain-on-close design) | REUSABLE (design), not literally reused | Master spec requires JSONL (append-only lines), not a single growing JSON array, so this class isn't imported as-is — but its flush interval, queue-drain-before-close, and `fsync`-before-close pattern are carried into `core/jsonl_writer.py`. |
| `robust_io.py`: `WindowsCollectionGuard` | REUSABLE_AFTER_FIX | Reimplemented as-is (trivial, low-risk) inside `core/health_supervisor.py` or a small `core` helper, with a test added. |
| `imu_pose_alignment.py` (interpolate/unwrap/windowed-average) | REFERENCE_ONLY for this task | Out of scope — this task does not compute initial-pose alignment or any processed output (master-spec section 1.1/6.2). Cited for future post-processing work only. |
| Per-mode trigger logic (IMU3 stop-gate / encoder Δ=0 / fixed 3s delay) | REFERENCE_ONLY / DO_NOT_USE as tuned values | The IMU3 stop-gate concept (gyro-magnitude-below-threshold-for-N-seconds) is the closest existing precedent to the new adaptive settling detector, but its thresholds are self-described as provisional and are NOT reused verbatim — `experiment/adaptive_settling_detector.py` implements the fuller multi-metric spec (section 19) with its own config-declared thresholds. |
| `WT901C_encoder0/encoder_source.py` (`JsonFileEncoderSource`) | DO_NOT_USE | Explicit stub per its own docstring. Superseded entirely by the `02_SO-101_rtos_motor_driver` STREAM/ENCODER bridge (decision 3). |
| `tests/test_robust_io.py` (concurrent-writer test pattern) | REUSABLE (pattern) | Same multi-thread-producer test shape reused for `tests/test_jsonl_writer.py`. |

Every reused function is reimplemented (not imported across package
boundaries) with a comment citing `07_WT901C/<path>` and commit
`3a0c757e913454ade5007cbacbb52bae48f321a0`, per master-spec section 4.

## 5. ML reference repo (`Inha-EEE-Capstone-26/ML`, read-only) — informational only

No IMU precedent exists anywhere in the ML repo (raw schema, processed
schema, and model I/O are built entirely from `q_cmd`/`q_encoder`/target
pose/single-Vive-Tracker-position — grepped for imu/quaternion/gyro,
zero real hits). The multi-IMU raw schema this task builds is genuinely new
design space, not something to reverse-engineer from that repo. Two
conventions are carried over for consistency where they don't conflict with
the master spec's own explicit schema:

- Timestamp format: ISO-8601 UTC (`utc_iso` field, master-spec section 9)
  follows the same style as the ML repo's `timestamp_utc`/`utc_now_iso()`
  (`+00:00`→`Z` suffix convention).
- Vive raw pose: master-spec section 12 already requires both a raw 3x4
  pose matrix and `quaternion_xyzw` — consistent with the ML repo's
  preference for keeping raw rotation representations (`r00..r22`) rather
  than only a derived quaternion.

**Not adopted:** the ML repo's session-folder naming (`test-<YYMMDD>-<N>`),
its `clean_dataset_*` processed-layer schema, and its
vertical-home-relative/90°-axis-mapped label frame. Master-spec section 8
explicitly defines `session_YYYYMMDD_NNN` raw session folders and this task
performs no coordinate transforms on raw data (section 1.1) — building a
bridge between this raw acquisition format and the ML repo's processed
format is future post-processing work, out of scope here.

## 6. Korean policy docs (`필수 참고 사항.md`, `실험절차.md`) — confirmations, no conflicts

- Raw-per-sensor-stream storage (never merged into one training CSV) is
  existing project policy (실험절차.md:445-470), matching master-spec
  section 8 exactly.
- Vive-never-as-control-feedback and no-forward-fill-on-tracking-loss are
  both existing, repeatedly-stated project policy (필수 참고
  사항.md:643-654, 실험절차.md:91-106), matching master-spec sections 1.3
  and 17.2 exactly.
- No adaptive settling-detection method or threshold exists in either
  document — only a fixed 0.5-1.0s post-motion dwell is documented
  (실험절차.md:229, 386-391) as a *different* experimental-procedure
  default, not an online detector. The adaptive tip-IMU settling detector
  (master-spec section 19) is genuinely new work, not a replacement for a
  documented method.
- No front-120°/180°-sector policy exists in either document (confirmed via
  targeted grep) — nothing to reconcile.
- Repo-wide naming convention found: position/orientation/timestamp
  variables must carry an explicit frame suffix (필수 참고
  사항.md:1592-1620 — bare names like `P_hat`/`R_i` are called out as
  disallowed in code review). Applied where it doesn't conflict with the
  master spec's own explicit field names (e.g., manifest/doc prose refers
  to `p_target_..._base` frame explicitly); the master spec's literal
  schema field names (section 9-12) are used as specified since they are
  an explicit task instruction.
- Monotonic-clock-only timestamp policy (`time.perf_counter_ns()`, never
  `datetime.now()`) confirmed as existing project policy (실험절차.md:481-
  490) — matches master-spec section 14's PC-master clock design exactly,
  applied throughout `core/clock_sync.py` and every collector.

## 7. `firmware/esp32_so101_raw_bridge/` scope reduction: no new firmware source

The master spec's recommended structure (section 5) lists new ESP32 firmware
source files (`encoder_streamer.*`, `clock_sync.*`,
`experiment_state_machine.*`, `watchdog.*`, `safety.*`). Per decision 3,
`02_SO-101_rtos_motor_driver/main` **already implements** the needed raw
telemetry stream (`STREAM ON <period_ms>` / `ENCODER ON [period_ms]`,
already-promoted active firmware). Writing new, parallel ESP32 firmware that
duplicates this would:

- violate master-spec section 1.2 ("기존 로봇 모델 보존" — don't modify or
  fork the existing control firmware),
- risk diverging from the firmware that's actually flashed to the board,
  since this task never flashes anything to verify the new code matches
  (section 1.5), and
- duplicate work with no benefit — the wire contract already exists and is
  documented.

**Decision:** `firmware/esp32_so101_raw_bridge/` contains only
`README.md` and `serial_protocol.md` — a documented wire-protocol contract
(citing `02_SO-101_rtos_motor_driver/README.md` and
`02_SO-101_rtos_motor_driver/main/README.md` as the source of truth) that
`collectors/collect_esp32_encoder.py` implements against as a PC-side
serial client. No `.cpp`/`.h` firmware files are added, and
`02_SO-101_rtos_motor_driver/` is not touched. If a future task needs
ESP32-side changes (e.g. a dedicated watchdog or clock-sync response
frame not already in the `STREAM`/`ENCODER` protocol), that is new,
explicitly-scoped firmware work — not part of this raw-acquisition task.
