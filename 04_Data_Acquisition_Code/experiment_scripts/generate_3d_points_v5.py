"""v5 target generator -- samples in RAW ENCODER TICKS, not logical degrees.

Why the rewrite (each point established from real data, see
docs/FK_CONVENTION_CORRECTION.md):

1. `logical_deg` is NOT a physical angle. The firmware linearly maps each
   axis's calibrated-active raw span onto an arbitrary label interval
   (`logical_min_deg`..`logical_max_deg`), and the two spans do not match:
   pan 1.173, lift 1.300, wrist 1.101 physical degrees per logical degree.
   v3/v4 fed those labels to the FK as if they were angles, so shoulder_lift
   carried a 30% scale error. Verified against real Vive data: switching to
   tick-derived physical angles drops the rigid-model registration residual
   from 77.4mm to 48.7mm on segment 006 (and 144.3 -> 120.9 on 007).

2. The reference pose and the workspace side are measured//user-confirmed,
   not inferred: raw INCREASING is the "left" direction on all three axes
   (motors were mounted with matching direction), and the experiment's
   targets lie on that side of the straight ("1자") pose.

3. q1/q2 anchors are deliberately NOT calibrated. They are unobservable from
   tip position (the shoulder sits at the model origin, so a constant offset
   there is a global rotation that rigid registration absorbs) and trying to
   fit them against Vive residual would eat the flexible-link deflection this
   experiment exists to measure. The reference pose is simply *declared* to be
   q2 = pi/2, q3 = 0.
"""
import json
import math
import os
import statistics as st
import sys

import numpy as np

REPO = r"C:\Users\hsw10\Documents\한이음 실험\capstone-code"
sys.path.insert(0, os.path.join(REPO, "02_FK_IK"))

from fkik.geometry import load_config  # noqa: E402
from fkik.rigid_fk import fk_rigid  # noqa: E402

CONFIG_PATH = os.path.join(
    REPO, "02_FK_IK", "config", "robot_config.yaml"
)
OUT_PATH = os.path.join(
    REPO, "04_Data_Acquisition_Code", "sessions", "real_motion_demo",
    "target_plan_v5.json",
)

TICKS_PER_DEG = 4096.0 / 360.0

# Measured 2026-08-05 (config/reference_pose.yaml): the arm held straight
# under torque. raw_active/logical come from a live STATE read; the
# raw->logical formula below was checked against all three axes' reported
# logical_deg and reproduces them exactly.
AXES = {
    "shoulder_pan":  {"alias": "D2", "ref": 2055, "raw_active": (1160, 3083), "logical": (0.0, 160.0)},
    "shoulder_lift": {"alias": "D3", "ref": 2093, "raw_active": (1202, 3186), "logical": (5.0, 154.0)},
    "wrist_flex":    {"alias": "D5", "ref": 1970, "raw_active": (1937, 3968), "logical": (0.0, 180.0)},
}
ORDER = ("shoulder_pan", "shoulder_lift", "wrist_flex")

EDGE_MARGIN_TICKS = 40          # ~3.5 deg held back from each active limit
PAN_HALF_SPAN_DEG = 60.0        # the documented front-120 fan, about the reference
LIFT_UP_SPAN_DEG = 90.0         # "left" side only: reference -> up
WRIST_BEND_SPAN_DEG = 120.0     # elbow opens from straight; short of folding back on itself

MIN_RADIAL_MM = 150.0

# --- workspace shape, filtered in TRUE GRAVITY coordinates ----------------
# Two corrections, both needed:
#
# 1. Joint-space limits do NOT map to a Cartesian sector. Sampling
#    shoulder_pan within +-60 deg of the reference sounds like a 120 deg fan,
#    but when the arm folds over (high lift + bent wrist) the tip swings
#    behind the base and its azimuth flips ~180 deg. Measured on the previous
#    30-point plan: joint pan was +-60 deg, yet the tip's azimuth spanned
#    351 deg -- nearly the whole circle -- with 11 of 30 points below base
#    level. So the sector must be enforced on the tip POSITION.
#
# 2. But not in the model's own frame. The model's +z is not vertical: the
#    q1/q2 anchors are declared, not measured (docs/FK_CONVENTION_CORRECTION.md),
#    so model axes have no fixed relation to gravity. Filtering on model
#    elevation rejected 100% of candidates -- in that arbitrary frame,
#    "forward and up" simply isn't where the arm goes.
#
# MODEL_UP is the real vertical expressed in model coordinates, recovered by
# rigidly registering FK predictions against real Vive positions (segment 015,
# 975 paired samples; SteamVR's standing universe is gravity-aligned, +Y up).
# It comes out 24 deg off the model's +z. In this frame 46% of the joint
# sampling window satisfies the sector below, versus 0% in model coordinates.
#
# Registration residual was 219mm RMS -- large, because segment 015 was mostly
# captured mid-motion so flexible-link deflection is baked in. Fine for
# shaping a workspace at the tens-of-cm scale; do NOT reuse this transform as
# a precision calibration.
MODEL_UP = np.array([0.402, 0.015, 0.915])
MODEL_UP = MODEL_UP / np.linalg.norm(MODEL_UP)

AZIMUTH_HALF_SPAN_DEG = 60.0    # +-60 => 120 deg fan, about the reference heading
MAX_ELEVATION_DEG = 90.0

# Height floor, expressed in mm above the robot base rather than as an
# elevation angle: at a fixed angle the actual clearance shrinks as the arm
# pulls in, so an angle floor does not mean a constant height floor.
#
# 70mm above base level, chosen by the project owner with the measurement
# below in hand -- it is a deliberate, informed trade, not an oversight.
#
# Measured from the 30 settled poses of segment 023 (the MEASUREMENT_WINDOW
# samples, rigidly registered against real Vive positions): the height error
# between rigid prediction and reality is
#     mean -0.0mm, sd 52.3mm, range -98.1..+121.3mm, 95th pct +94.1mm
# so 70mm does NOT guarantee every point stays above base level; a few can
# land slightly under. Accepted because nothing sits directly below the base
# to collide with.
#
# Note what that error is and is not. A pure gravity sag would be one-sided
# (always downward); this is symmetric about zero, so it is not sag -- it is
# unmodelled geometry (link lengths, and `tracker_offset_mm` in
# robot_config.yaml which is still [0,0], unmeasured). Measuring the tracker
# offset would shrink the residual and let a smaller margin actually
# guarantee the floor. Until then this margin is a judgement call, not a
# bound.
MIN_HEIGHT_MM = 70.0

N_CANDIDATES = 8000
N_FINAL = 30
SEED = 2026


def raw_to_logical(axis, raw):
    lo, hi = AXES[axis]["raw_active"]
    llo, lhi = AXES[axis]["logical"]
    return llo + (raw - lo) / (hi - lo) * (lhi - llo)


# Horizontal basis perpendicular to true vertical, with azimuth measured from
# the reference pose's own heading -- so "0 deg" is the direction the arm
# points when straight, not an arbitrary model axis.
_E1 = np.cross(MODEL_UP, np.array([0.0, 1.0, 0.0]))
if np.linalg.norm(_E1) < 1e-6:
    _E1 = np.cross(MODEL_UP, np.array([1.0, 0.0, 0.0]))
_E1 = _E1 / np.linalg.norm(_E1)
_E2 = np.cross(MODEL_UP, _E1)


def _azimuth_deg(p, _ref=[None]):
    if _ref[0] is None:
        rp = fk_rigid(np.array([0.0, math.pi / 2, 0.0]), load_config(CONFIG_PATH))
        _ref[0] = math.degrees(math.atan2(float(np.dot(rp, _E2)), float(np.dot(rp, _E1))))
    a = math.degrees(math.atan2(float(np.dot(p, _E2)), float(np.dot(p, _E1)))) - _ref[0]
    return (a + 180.0) % 360.0 - 180.0


def sample_range(axis, lo_deg, hi_deg):
    """Tick window `lo_deg`..`hi_deg` relative to the reference, clipped to
    the active range with a margin so no target sits on a hard stop."""
    ref = AXES[axis]["ref"]
    amin, amax = AXES[axis]["raw_active"]
    lo = max(ref + lo_deg * TICKS_PER_DEG, amin + EDGE_MARGIN_TICKS)
    hi = min(ref + hi_deg * TICKS_PER_DEG, amax - EDGE_MARGIN_TICKS)
    return lo, hi


def main():
    config = load_config(CONFIG_PATH)
    rng = np.random.default_rng(SEED)

    windows = {
        "shoulder_pan": sample_range("shoulder_pan", -PAN_HALF_SPAN_DEG, PAN_HALF_SPAN_DEG),
        "shoulder_lift": sample_range("shoulder_lift", 0.0, LIFT_UP_SPAN_DEG),
        "wrist_flex": sample_range("wrist_flex", 0.0, WRIST_BEND_SPAN_DEG),
    }
    print("sampling windows (raw ticks -> deg from reference -> logical):")
    for a in ORDER:
        lo, hi = windows[a]
        ref = AXES[a]["ref"]
        print(f"  {a:15s} raw {lo:6.0f}..{hi:6.0f}   "
              f"{(lo-ref)/TICKS_PER_DEG:+6.1f}..{(hi-ref)/TICKS_PER_DEG:+6.1f} deg   "
              f"logical {raw_to_logical(a, lo):5.1f}..{raw_to_logical(a, hi):5.1f}")

    cands = []
    for _ in range(N_CANDIDATES * 4):
        if len(cands) >= N_CANDIDATES:
            break
        raws = {a: rng.uniform(*windows[a]) for a in ORDER}
        # Reference pose is declared to be q2 = pi/2, q3 = 0 (see docstring).
        q = np.array([
            (raws["shoulder_pan"] - AXES["shoulder_pan"]["ref"]) / TICKS_PER_DEG * math.pi / 180.0,
            math.pi / 2
            + (raws["shoulder_lift"] - AXES["shoulder_lift"]["ref"]) / TICKS_PER_DEG * math.pi / 180.0,
            (raws["wrist_flex"] - AXES["wrist_flex"]["ref"]) / TICKS_PER_DEG * math.pi / 180.0,
        ])
        p = fk_rigid(q, config)
        r = float(np.linalg.norm(p))
        if r < MIN_RADIAL_MM:
            continue
        height = float(np.dot(p, MODEL_UP))          # true height above the base
        elevation = math.degrees(math.asin(height / r))
        azimuth = _azimuth_deg(p)                    # about true vertical, 0 = reference heading
        if abs(azimuth) > AZIMUTH_HALF_SPAN_DEG:
            continue
        if height < MIN_HEIGHT_MM or elevation > MAX_ELEVATION_DEG:
            continue
        cands.append({"raws": raws, "q": q, "p": p,
                      "az": azimuth, "el": elevation, "h": height})

    print(f"\ncandidates: {len(cands)}")

    pts = np.array([c["p"] for c in cands])
    chosen = [int(rng.integers(0, len(pts)))]
    min_d = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    while len(chosen) < N_FINAL:
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(pts - pts[nxt], axis=1))

    points = []
    for i, idx in enumerate(chosen):
        c = cands[idx]
        logical = [raw_to_logical(a, c["raws"][a]) for a in ORDER]
        points.append({
            "target_id": f"T{i:03d}",
            "planned_order_index": i,
            "raw_ticks": [round(c["raws"][a]) for a in ORDER],
            "deg_from_reference": [
                round((c["raws"][a] - AXES[a]["ref"]) / TICKS_PER_DEG, 2) for a in ORDER
            ],
            "logical_deg": [round(v, 1) for v in logical],
            "q_rad": c["q"].tolist(),
            "p_xyz_mm": c["p"].tolist(),
            "workspace_mode": "left_of_straight_v5_rawtick_fps",
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(points, f, indent=2)

    print(f"selected {len(points)} -> {os.path.basename(OUT_PATH)}\n")
    for i, a in enumerate(ORDER):
        v = [p["logical_deg"][i] for p in points]
        d = [p["deg_from_reference"][i] for p in points]
        print(f"  {a:15s} logical {min(v):6.1f}..{max(v):6.1f}   "
              f"from-ref {min(d):+6.1f}..{max(d):+6.1f} deg  (median {st.median(d):+.1f})")
    for ax, nm in enumerate("xyz"):
        v = [p["p_xyz_mm"][ax] for p in points]
        print(f"  rigid tip {nm} (mm) {min(v):7.0f}..{max(v):7.0f}")
    az = [cands[i]["az"] for i in chosen]
    el = [cands[i]["el"] for i in chosen]
    rr = [float(np.linalg.norm(cands[i]["p"])) for i in chosen]
    print(f"\n  azimuth   {min(az):+7.1f}..{max(az):+7.1f}  (span {max(az)-min(az):.0f} deg, limit +-{AZIMUTH_HALF_SPAN_DEG:.0f})")
    print(f"  elevation {min(el):+7.1f}..{max(el):+7.1f}  (span {max(el)-min(el):.0f} deg, limit height>={MIN_HEIGHT_MM:.0f}mm, el<={MAX_ELEVATION_DEG:.0f})")
    print(f"  radius    {min(rr):7.0f}..{max(rr):7.0f} mm")
    print(f"  height above base: {min(cands[i][chr(104)] for i in chosen):.0f}..{max(cands[i][chr(104)] for i in chosen):.0f} mm  (floor {MIN_HEIGHT_MM:.0f})")


if __name__ == "__main__":
    main()
