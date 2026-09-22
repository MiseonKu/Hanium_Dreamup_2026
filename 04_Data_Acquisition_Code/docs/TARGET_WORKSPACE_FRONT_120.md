# The `front_120` Target-Workspace Addition

User-facing explanation of the one change this task made to
`02_FK_IK`: a new azimuth-sector mode
called `front_120`, alongside the existing `pilot` and `main` modes. Full
audit trail: `DECISIONS.md` section 2. This document is the plain-language
version of that decision.

## Where this lives (and where it doesn't)

The actual target-generation code -- sampling, rigid FK, hemisphere/azimuth
gating, bin-balanced selection, CSV + metadata output -- lives entirely in
`02_FK_IK/`, specifically
`workspace_points/pre_imu_targets.py` and
`config/pre_imu_forward_hemisphere_workspace.json`. This package
(`04_Data_Acquisition_Code`) adds no generation logic of its own --
`tools/generate_front_120_target_plan.py` is a thin CLI wrapper that imports
the 02 package's existing `generate_pre_imu_targets.main()` and always calls
it with `--mode front_120` forced, writing output under this package's own
`outputs/` directory so the two packages' artifacts don't mix.

This was a deliberate choice, not a default: the master spec's own priority
list initially pointed at
`06_simulation_and_data_collect_preparation/`, but that folder's
`p1_one_link_experiment/`/`p2_two_link_experiment/` turned out to be
reduced-DOF teaching/validation forks (confirmed via duplicated module names
and a `pilot_azimuth_deg` value copied verbatim from the real source). The
02 package is the one with real IK-adjacent FK, joint limits, and the
azimuth-sector mode dictionary this feature extends. See `DECISIONS.md`
section 2 for the full evidence trail.

## The exact mechanism

Verified directly in `workspace_points/pre_imu_targets.py`:

```text
v = p - center                                  # center = [0,0,0] m
forward_dot = v . [1,0,0] > 0                    # full +-90 deg hemisphere gate
azimuth_deg = degrees(atan2(v_y, v_x))
lo, hi = region.azimuth_limit_deg[mode]          # mode-specific narrower gate
accept if lo <= azimuth_deg <= hi
```

`p` is a candidate target position (meters, robot-base frame); `center` is
fixed at `[0, 0, 0]`. Every candidate must first pass the full
±90° forward-hemisphere gate (`forward_dot > 0`), then the narrower,
mode-specific azimuth sector on top of that.

## What actually changed

**One config value, no formula change.**
`config/pre_imu_forward_hemisphere_workspace.json`'s `workspace.
azimuth_limit_deg` dictionary already held:

```json
{"pilot": [-30.0, 30.0], "main": [-45.0, 45.0]}
```

This task added a third key:

```json
{"pilot": [-30.0, 30.0], "main": [-45.0, 45.0], "front_120": [-60.0, 60.0]}
```

`[-60.0, 60.0]` is a full sector width of 120° (hence the name) -- wider
than `main`'s 90°, still well inside the hard ±90° hemisphere gate above.

That single dictionary entry was sufficient because
`is_inside_forward_trimmed_hemisphere` and `classify_stage2_bins` (both in
`workspace_points/pre_imu_targets.py`) already read `azimuth_limit_deg[mode]`
generically -- neither function has `pilot`/`main` hardcoded anywhere. The
only other change was additive: `scripts/generate_pre_imu_targets.py`'s
`--mode` argparse choices list grew from `["pilot", "main"]` to
`["pilot", "main", "front_120"]` (one line).

No IK/collision checks were added, and none should be -- the pre-IMU path is
intentionally IK-free. The full pipeline order (`load config -> sample joint
offsets -> rigid FK -> hemisphere+azimuth gate -> bin classify -> bin-
balanced sampling -> write CSV + metadata`) was preserved exactly as-is.

## How to generate a `front_120` plan

```powershell
python tools/generate_front_120_target_plan.py --stage P2_TWO_LINK_RIGID_TARGET_VIVE_LABEL --count 20 --seed 202
```

Writes `outputs/pre_imu_front_120_targets.csv` +
`outputs/pre_imu_front_120_targets_metadata.json` by default. Then verify
it against the acceptance criteria (zero azimuth violations, zero
duplicates, requested count == generated count):

```powershell
python tools/audit_target_plan.py --csv outputs\pre_imu_front_120_targets.csv --metadata outputs\pre_imu_front_120_targets_metadata.json --expected-count 20
```

Both commands were run against this repository while writing this document
-- see `docs/EXPERIMENT_RUNBOOK.md` section 4 for the observed output. Also
see `config/target_workspace_front_120.example.yaml`, a thin YAML mirror of
the real JSON config for reference.

## The DOF caveat (read this before assuming this covers the whole arm)

The master spec describes the SO-101 arm as "5-DOF." **The pre-IMU pipeline
this task extended models only 3 joints:**

- `q1` -- base yaw
- `q2` -- shoulder pitch
- `q3` -- elbow pitch

`config/robot_config.yaml`'s own `dof: 3` and that package's `README.md`
("SO-101 3DoF") both confirm this is a pre-existing characteristic of the
pipeline, not something introduced by adding `front_120`.

~~`wrist_flex` and `gripper` (the real robot's 4th and 5th joints, per
`config/joint_limits.reference.yaml`) are entirely outside this pipeline's
scope.~~ **Struck 2026-08-05:** that sentence was wrong twice over.
`wrist_flex` **is** the pipeline's `q3` on this rig (see the amendment at
the end of this document), and `config/joint_limits.reference.yaml` no
longer lists 5 joints — the rig is 3-axis by design
(`docs/THREE_AXIS_EXPERIMENT_SCOPE.md`).

Per the master spec's own override clause (section 21.2: "실제 repository
convention이 다르면 기존 구현을 따른다" -- "if the actual repository
convention differs, follow the existing implementation"), `front_120` was
added to this 3-DOF pipeline exactly as it already existed, rather than
being rewritten to cover all 5 joints as part of this task. If a full
5-DOF version of this target generator is ever needed, that is separate,
larger scope than a config-key addition -- see `docs/TASKS.md`.

**Amendment (2026-08-05):** confirmed by the project owner that 3 axes is
the intentional, final experiment scope (see
`docs/THREE_AXIS_EXPERIMENT_SCOPE.md`), not an under-coverage of a 5-DOF
arm -- wrist_roll/gripper carry a Vive Tracker mount, not a motor, for
this experiment. One naming note for real-hardware runs: this pipeline's
generic `q1/q2/q3` ("base yaw/shoulder pitch/elbow pitch") are reused
directly as this rig's `shoulder_pan/shoulder_lift/wrist_flex` -- e.g.
`experiment_scripts/generate_3d_points_50_v3.py` maps
`q -> logical_deg = HOME_DEG + degrees(q - q_init)` onto exactly those 3
physical joints. "Elbow pitch" in this doc's original wording is that same
`q3`/wrist_flex axis, not a separate joint.

**Second amendment (2026-08-05): do not copy the v3 mapping above.**
`logical_deg` is a firmware *label*, not a physical angle — each axis's
calibrated-active raw span is mapped linearly onto an arbitrary label
interval, so treating `logical_deg` as degrees carried a ~17% scale error on
shoulder_lift alone. Commanding in raw ticks instead
(`(raw - raw_ref) / 11.378` physical degrees) cut the rigid-registration
residual against real Vive data from 77.4 mm to 48.7 mm on segment 006. The
`q_init` in that expression was also wrong: it described an L-shaped arm,
while the real reference pose is straight. Both are corrected in
`generate_3d_points_v5.py`. Full detail and evidence:
`docs/FK_CONVENTION_CORRECTION.md`. The v3 line above is left in place as
the record of what was actually run for the v3 target plan.
