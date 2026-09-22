# Three-Axis Experiment Scope (intentional, not a data gap)

**Status: confirmed by the project owner, 2026-08-05. Supersedes any earlier
"5-DOF" framing in this package's docs (`DECISIONS.md`, `HANDOFF.md`,
`docs/TASKS.md`, `docs/TARGET_WORKSPACE_FRONT_120.md`, `STATUS_BEFORE.md`) --
those files are left as historical decision/snapshot records and are not
rewritten, but every one of them now links here.**

## The scope

```
physical platform:            SO-101-based structure
experiment actuation scope:   3 active axes
  shoulder_pan
  shoulder_lift
  wrist_flex
flexible-link modeling scope: shoulder_pan -> shoulder_lift -> wrist_flex
end-effector reference:       Vive Tracker
```

`wrist_roll` and `gripper` -- the two joint positions a full 5-DOF SO-101
would actuate -- carry a Vive Tracker mount instead of a servo in this rig.
There is no flexible link across those two joints for this experiment (the
tracker mount is rigid), so they are outside the flexible-link
pose/bend/end-effector-position estimation problem this package's data
supports. This was previously read (by this package's own docs, and by
`03_SO-101_control/main/include/robot_config.h`'s pre-fix comment) as a
temporary hardware shortfall -- "only 3 of 5 motors are wired up, restore to
5 once the rest are connected." That reading was wrong. There is no 5-motor
upgrade planned; 3 axes is the correct, final scope for this experiment.

## Why this matters for every layer of the stack

| Layer | What "3-axis" means there |
|---|---|
| Firmware (`03_SO-101_control/main`) | `kJointCount = 3`, `kJointConfigs[]` has exactly 3 rows (shoulder_pan id=1, shoulder_lift id=2, wrist_flex id=4), legacy aliases `D2/D3/D5` only |
| FK/IK (`02_FK_IK`) | `q1/q2/q3` already models base-yaw/shoulder/elbow-equivalent joints -- this pipeline was *already* 3-DOF before this task, for unrelated historical reasons (see `DECISIONS.md` #2); it happens to now match the physical rig exactly, which is a coincidence worth noting, not a fix that was made to match it |
| Target plan (`front_120`, `target_plan_3d_random_n50_v3.json`) | Every target is a `(q1, q2, q3)` triple / `logical_deg` triple over exactly these 3 axes, forward hemisphere, azimuth trimmed to +-60 deg (120 deg total) |
| `core/schemas.py` `encoder_record` | `q_cmd_rad`, `q_encoder_rad`, etc. are 3-element lists in every real record produced by this rig, ordered `[shoulder_pan, shoulder_lift, wrist_flex]` |
| End-effector ground truth | Vive Tracker pose, not FK of a 5th/6th joint -- FK predicts a position from the 3 actuated joints, Vive independently measures where the tracker (proxy for end-effector / wrist_roll+gripper assembly) actually ended up. The two are expected to *disagree* -- that disagreement is exactly the training signal a later model uses, not a bug to reconcile. |

## What is NOT this

- Not a "5-DOF data collection with 2 axes silently dropped" -- `elbow_flex`
  (id=3) and `gripper` (id=5) were never part of the flexible-link
  actuation chain for this experiment; there is no missing encoder/command
  stream for them to backfill.
- Not a placeholder pending more motors -- see the corrected comment in
  `robot_config.h` (`kJointCount = 3`).
- Not a change to the target-generation math -- `front_120`'s +-60 deg
  azimuth trim and the 3-axis FK/IK model were already correct for this
  scope; nothing in that pipeline needed to change because of this
  clarification.

## Cross-references

- `DECISIONS.md` #2 ("DOF mismatch, deliberately deferred to existing
  code") -- historical: written when the 3-DOF FK/IK pipeline's scope
  looked like a code-reuse tradeoff against a nominally-5-DOF arm. Still
  factually accurate about what the pipeline models; the framing of *why*
  3 axes is correct is superseded by this document.
- `docs/TARGET_WORKSPACE_FRONT_120.md` "DOF caveat" section -- same
  supersession.
- `HANDOFF.md` "Target generator is 3-DOF, not 5-DOF" -- same.
