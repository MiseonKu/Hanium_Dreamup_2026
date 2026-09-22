# FK convention corrections (2026-08-05)

The rigid FK model is the **reference line** for this experiment: command a
target assuming a rigid arm, then measure how far the real flexible link
lands from that prediction (Vive) and how much it bent getting there (IMUs).
The gap between prediction and reality is the data.

That only works if the reference line describes the actual mechanism. Two
errors meant it did not. Both are corrected here, and one thing that looks
like a third error is deliberately left alone.

**Discipline followed throughout: only discrete convention errors were
fixed. No continuous parameter (link length, offset, scale) was tuned to
reduce the Vive residual.** Fitting those would eat the deflection signal
this experiment exists to measure.

---

## Error 1 — `logical_deg` is a label, not an angle

The firmware maps each axis's calibrated-active raw span linearly onto an
arbitrary label interval (`logical_min_deg`..`logical_max_deg` in
`03_SO-101_control/main/include/robot_config.h`):

```
logical = logical_min + (raw - raw_active_min)/(raw_active_max - raw_active_min)
                        * (logical_max - logical_min)
```

Verified exactly against a live `STATE` read on all three axes
(pan 2055 → 74.5, lift 2093 → 71.9, wrist 1970 → 2.9).

The label interval and the real travel do **not** match. ST3215 is a 360°
encoder with 4096 counts, so 1 physical degree = 11.378 ticks:

| axis | active raw span | physical travel | logical label span | physical° per logical° |
|---|---|---|---|---|
| shoulder_pan | 1160–3083 (1923) | 169.0° | 160° | **1.056** |
| shoulder_lift | 1202–3186 (1984) | 174.4° | 149° | **1.170** |
| wrist_flex | 1937–3968 (2031) | 178.5° | 180° | **0.992** |

The span that matters is the **active** one (`raw_active_min..raw_active_max`,
i.e. `config.raw_min`/`raw_max` in the firmware), not the calibration span
(`raw_cal_*`). Confirmed at `03_SO-101_control/main/src/main.cpp:1096`, where
`rawToLogical` derives `raw_start`/`raw_end` from `config.raw_min`/`raw_max`.
These are also the numbers that reproduce the firmware's own output — the
reference pose 2055/2093/1970 → 74.5 / 71.9 / 2.9, exactly as `STATE`
reports it.

> **Correction, 2026-08-05.** The first version of this table used the
> *calibration* span and reported 1.173 / 1.300 / 1.101, making the headline
> "30% scale error on shoulder_lift". Those ratios were ~11% high; the real
> figure is ~17%. Found during the docs audit and independently recomputed.
> Nothing else in this document is affected: the fix (command in raw ticks,
> `/11.378`) never used these ratios, and the Vive residuals below were
> measured from raw ticks directly.

v3/v4 of the target generator fed those labels straight into the FK as if
they were angles, so shoulder_lift carried a ~17% scale error. Note that no
axis has a ratio of 1.0 — the label scale is wrong on all three, just by
different amounts, and wrist_flex is wrong in the opposite direction.

**Fix:** sample and command in raw ticks; convert to physical degrees with
`(raw - raw_ref) / 11.378`. Logical degrees are used only where the firmware
demands them.

**Evidence** (real Vive data, segments 006/007, rigid registration residual —
proper rotation + translation only, no reflection):

| | seg006 | seg007 |
|---|---|---|
| using `logical_deg` | 77.4 mm | 144.3 mm |
| using raw-tick physical angles | **48.7 mm** | **120.9 mm** |

---

## Error 2 — the model thought home was an L-shape

`robot_config.yaml` has `q_init_rad: [0, π/2, -π/2]`. Running that through
`fk_rigid` gives `[215, 0, 233]` — a bent, L-shaped arm. The physical arm at
its reference pose is **straight** (confirmed visually by the project owner).
A 90° elbow error, described as the reference for "how much did it bend".

For scale: `robot_config.yaml`'s own `flex_model.max_delta_rad` caps flex at
0.35 rad (20°). A 90° discrepancy cannot be deflection — it is a different
mechanism.

**Fix:** the wrist anchor is the value where the links are collinear, i.e.
`q3 = 0` at the reference pose.

**Evidence:** pairwise-distance correlation against real Vive positions
rises **0.79 → 0.96**. A grid search over every sign/offset combination put
`o3 = 0` in every single top-ranked variant. A proper rigid registration
(reflection excluded) further showed the lift and wrist signs must **match**
each other — mixed signs cost 155 mm vs 106 mm — confirming both are `+1`
and that `direction=normal` in the firmware config is right.

---

## Deliberately NOT calibrated — the pan and lift anchors

These two are **unobservable from tip position**. The shoulder sits at the
model's origin, so a constant offset in `q1` or `q2` rotates the entire arm
about that origin — a global rigid rotation, which the registration absorbs
completely. In the grid search, `o2 = +90°` and `o2 = -90°` scored
*identically to four decimal places* (0.9644).

So they are **declared**, not fitted: the measured reference pose is simply
defined as `q2 = π/2, q3 = 0`. Do not "improve" them against Vive residual.

They do still matter for two things that Vive cannot settle:

- **Which points get selected.** With the inherited anchor, the
  forward-hemisphere filter drove every planned pose into the bottom ~18% of
  shoulder_lift's travel (v3 commanded lift only within 10.2–45.2° of a
  149-label-degree range) — the "arm never comes up" symptom. Resolved not by
  fitting, but by the project owner stating the workspace side directly.
- **Gravity direction**, hence sag and floor clearance. Measurable from the
  IMUs, not from tip position. Not yet done.

---

## Measured facts this rests on

**Direction convention** (project owner, all three axes mounted alike):
**increasing raw = "left" = where the experiment's targets are.**
Independently consistent with gravity: releasing torque made shoulder_lift
fall 1593 → 1096, so decreasing raw is downward.

**Reference pose** (`config/reference_pose.yaml`, captured under torque):

| axis | raw | logical |
|---|---|---|
| shoulder_pan | 2055 | 74.5° |
| shoulder_lift | 2093 | 71.9° |
| wrist_flex | 1970 | 2.9° |

An earlier candidate reference (wrist raw 2954) was discarded: the arm had
collapsed at some point before that reading, so the pose was not the
intended one. Confirmed by the project owner — this is not a link slip and
no recalibration is needed.

**Reference pose total range context:** shoulder_lift's full travel is
193.7°, and the reference sits with roughly 90° of usable travel on the
"left" side, which is where `generate_3d_points_v5.py` samples.

---

## What the residual now contains

After both corrections, ~48.7 mm RMS remains on the good pilot segment.
That is the signal, and it is **not** all bending:

- flexible-link deflection under gravity — the target quantity
- the Vive tracker's fixed offset from the true link tip
  (`robot_config.yaml`'s `tracker_offset_mm: [0.0, 0.0]` is **unmeasured**)

These separate cleanly: the tracker offset is constant in the tip frame,
while deflection varies with pose (gravity moment). Estimating the constant
once across many poses leaves pure deflection. That is a separation, not a
fit that erases signal.

---

## Status

| item | state |
|---|---|
| raw-tick sampling (`generate_3d_points_v5.py`) | done |
| wrist anchor `q3 = 0` at reference | done, Vive-validated |
| pan/lift anchors | declared, deliberately unfitted |
| workspace side (left / increasing raw) | owner-confirmed |
| gravity direction from IMUs | **not yet measured** |
| tracker offset | **not yet measured** — needed before calling residual "bend" |
