"""v4 target generator. Two corrections over v3, both established from real
data rather than assumed -- see docs/FK_CONVENTION_CORRECTION.md.

1. `q_init3 = 0`, not `-pi/2`. v3 treated the robot's home pose as an L-shape
   in the FK model, but the physical arm at home (pan 80 / lift 35 / wrist 90)
   is STRAIGHT -- the two links are collinear. Validated against real Vive
   positions from segments 006/007: pairwise-distance correlation rises from
   0.79 to 0.96, and a proper rigid registration (rotation+translation, no
   reflection) confirms lift/wrist signs stay +1/+1.

2. Sampling happens in PHYSICAL joint degrees, not in the abstract
   `joint_limits_rad` from robot_config.yaml. Those limits are symmetric about
   zero (q2: +-1.8 rad) and do not describe this arm: combined with v3's
   offset they capped shoulder_lift at 48 deg when the joint physically
   reaches 154 deg. v3 therefore only ever commanded lift in 10.2-45.2 deg --
   the whole plan lived in the bottom ~18% of the joint's travel, which is
   why the arm never came up. Sampling in degrees and converting to q
   afterwards makes that class of bug structurally impossible.

Output is a plan file in the same shape v3 produced, so
`collect_3d_sweep_50_v2.py` reads it unchanged.
"""
import json
import math
import os
import sys

import numpy as np

raise SystemExit(
    "SUPERSEDED -- do not run. Use generate_3d_points_v5.py.\n"
    "\n"
    "v4 fixed the wrist anchor (straight arm at the reference pose) but still\n"
    "sampled in the firmware's `logical_deg`, which is a label scale and not a\n"
    "physical angle -- 1.17 / 1.30 / 1.10 physical degrees per logical degree\n"
    "on pan / lift / wrist. v5 samples in raw encoder ticks, which removed the\n"
    "remaining scale error (rigid-model residual against real Vive data:\n"
    "77.4mm -> 48.7mm on segment 006).\n"
    "\n"
    "Kept for the anchor-correction reasoning in its docstring; the full\n"
    "record is in docs/FK_CONVENTION_CORRECTION.md.\n"
)

REPO = r"C:\Users\hsw10\Documents\한이음 실험\capstone-code"
sys.path.insert(0, os.path.join(REPO, "02_FK_IK"))

from fkik.geometry import load_config  # noqa: E402
from fkik.rigid_fk import fk_rigid  # noqa: E402

CONFIG_PATH = os.path.join(
    REPO, "02_FK_IK", "config", "robot_config.yaml"
)
OUT_PATH = os.path.join(
    REPO, "04_Data_Acquisition_Code", "sessions", "real_motion_demo",
    "target_plan_3d_v4.json",
)

# Physical joint angle -> FK model joint angle. These three anchors are the
# ONLY thing that defines the mapping, and they are deliberately absolute
# rather than expressed relative to whatever pose we currently call "home":
#
#     q1 = radians(pan   - PAN_ANCHOR)
#     q2 = radians(lift  + LIFT_ANCHOR)
#     q3 = radians(wrist - WRIST_ANCHOR)
#
# WRIST_ANCHOR=90 is the load-bearing one -- it is what v3 got wrong (it
# effectively used 180, i.e. an L-shaped arm at home) and it is the only
# anchor real data can pin down: pairwise-distance correlation against Vive
# rises 0.79 -> 0.96 with this value, and a proper rigid registration
# confirms the lift/wrist signs are both +1.
#
# PAN_ANCHOR and LIFT_ANCHOR are NOT observable from tip position: the
# shoulder sits at the model's origin, so a constant offset in either one
# just rotates the whole arm about that origin -- a global rigid rotation
# that any registration absorbs (o2=+90 and o2=-90 scored identically to 4
# decimal places). They are carried over from the original robot_config.yaml
# convention and left alone. Do NOT "recalibrate" them against Vive residual;
# that residual is the flexible-link deflection this experiment measures.
PAN_ANCHOR = 80.0
LIFT_ANCHOR = 55.0    # q2 = radians(lift + 55) == pi/2 + radians(lift - 35)
WRIST_ANCHOR = 90.0

# The pose each run starts from and returns to. Measured, not assumed --
# config/reference_pose.yaml, captured 2026-08-05 with the arm held under
# torque. Changing this does NOT change the mapping above, by design.
REFERENCE_POSE_DEG = (76.7, 68.5, 5.8)

# Physical commandable range per axis (logical_min_deg..logical_max_deg from
# robot_config.h) with a safety margin held back from each hard stop.
# shoulder_pan is additionally trimmed to home +-60 deg for the front-120
# workspace.
PAN_RANGE = (20.0, 140.0)     # hard limits 0..160
LIFT_RANGE = (12.0, 148.0)    # hard limits 5..154
WRIST_RANGE = (10.0, 170.0)   # hard limits 0..180

MIN_RADIAL_MM = 150.0   # keep away from the base/singular region
# These are RIGID-model tip positions. The real flexible link sags BELOW its
# rigid prediction (robot_config.yaml caps flex at max_delta_rad=0.35 rad over
# a 215mm link, so tens of mm), and the base is clamped at a desk edge -- so
# the floor limit is held at base level rather than below it, leaving the sag
# itself as clearance instead of spending it.
MIN_TIP_Z_MM = 0.0
# Forward hemisphere, as in v3 (`p[0] > 0`). Two reasons, not one: it is the
# documented "front 120 deg" workspace, AND without it the sampler happily
# picks lift > 90 deg poses that fold the arm backwards over the desk the
# base is clamped to (v4 hit x = -425mm before this was restored).
MIN_TIP_X_MM = 50.0
N_CANDIDATES = 6000
N_FINAL = 15
SEED = 2026

config = load_config(CONFIG_PATH)


def logical_to_q(pan_deg, lift_deg, wrist_deg):
    """Physical joint degrees -> FK model joint angles. See the anchor
    comments above; this is the validated mapping and is independent of
    whatever pose is currently used as the reference."""
    return np.array([
        math.radians(pan_deg - PAN_ANCHOR),
        math.radians(lift_deg + LIFT_ANCHOR),
        math.radians(wrist_deg - WRIST_ANCHOR),
    ])


def main():
    rng = np.random.default_rng(SEED)
    candidates = []
    for _ in range(N_CANDIDATES * 6):
        if len(candidates) >= N_CANDIDATES:
            break
        pan = rng.uniform(*PAN_RANGE)
        lift = rng.uniform(*LIFT_RANGE)
        wrist = rng.uniform(*WRIST_RANGE)
        q = logical_to_q(pan, lift, wrist)
        p = fk_rigid(q, config)
        if float(np.linalg.norm(p)) < MIN_RADIAL_MM:
            continue
        if p[2] < MIN_TIP_Z_MM:
            continue
        if p[0] < MIN_TIP_X_MM:
            continue
        candidates.append({"q": q, "p": p, "logical_deg": [pan, lift, wrist]})

    print(f"candidates: {len(candidates)}")

    # Greedy farthest-point sampling -- every prefix of the result is itself a
    # well-spread subset, so truncating the plan later stays valid.
    pts = np.array([c["p"] for c in candidates])
    chosen = [int(rng.integers(0, len(pts)))]
    min_d = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    while len(chosen) < N_FINAL:
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(pts - pts[nxt], axis=1))

    points = []
    for i, idx in enumerate(chosen):
        c = candidates[idx]
        points.append({
            "target_id": f"T{i:03d}",
            "planned_order_index": i,
            "q_rad": c["q"].tolist(),
            "p_xyz_mm": c["p"].tolist(),
            "logical_deg": c["logical_deg"],
            "workspace_mode": "front_120_3d_v4_fps",
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(points, f, indent=2)

    import statistics as st
    print(f"\nselected {len(points)} targets -> {os.path.basename(OUT_PATH)}\n")
    for i, name in enumerate(("shoulder_pan", "shoulder_lift", "wrist_flex")):
        v = [p["logical_deg"][i] for p in points]
        print(f"  {name:15s} min={min(v):6.1f} max={max(v):6.1f} median={st.median(v):6.1f}")
    for ax, name in enumerate("xyz"):
        v = [p["p_xyz_mm"][ax] for p in points]
        print(f"  tip {name}(mm){'':7s} min={min(v):6.0f} max={max(v):6.0f}")
    lift = [p["logical_deg"][1] for p in points]
    print(f"\n  shoulder_lift above home(35 deg): {sum(1 for v in lift if v > 35)}/{len(lift)}")


if __name__ == "__main__":
    main()
