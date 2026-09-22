"""v6 target generator: 200 points with FLAT coverage of the workspace.

Follows the approach already used in
`06_simulation_and_data_collect_preparation/p1_one_link_experiment/sampling/`
(`uniform_bin_sampler.allocate_bin_counts`, `binning.add_bin_labels`,
`balanced_shuffle._interleaved_target_order`), adapted to this rig's raw-tick
joint space.

The problem it solves: sampling uniformly in JOINT space does not give uniform
coverage in Cartesian space. A 2-link chain crowds its reachable points toward
certain radii and elevations, so a naive draw piles attempts into a few regions
and thins out elsewhere -- roughly Gaussian in radius rather than flat. Farthest-
point sampling alone spreads the CHOSEN points nicely but still inherits the
pool's density, so repeat attempts stay unbalanced.

Fix, in the order the 06 package does it:
  1. build a large joint-space candidate pool, keep only what lands in the
     workspace (azimuth sector, height floor, radius range);
  2. label every candidate with a (radius, azimuth, elevation) bin;
  3. allocate a near-equal number of targets to every occupied bin;
  4. farthest-point sample within each bin, so points inside a bin spread out
     too rather than clumping at its centre.

Result: every region of the workspace gets the same number of attempts, and
radius/distance variety is explicit rather than incidental.

Geometry note: the workspace filter runs in TRUE GRAVITY coordinates
(`MODEL_UP`), not the model's own axes -- see generate_3d_points_v5.py for why.
That vector came from registering FK against Vive, so it is an approximation
used only to SHAPE the sampling region; it never touches recorded data.
"""
import json
import math
import os
import sys
from collections import defaultdict

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
    "target_plan_v6.json",
)

TICKS_PER_DEG = 4096.0 / 360.0
AXES = {
    "shoulder_pan":  {"ref": 2055, "raw_active": (1160, 3083), "logical": (0.0, 160.0)},
    "shoulder_lift": {"ref": 2093, "raw_active": (1202, 3186), "logical": (5.0, 154.0)},
    "wrist_flex":    {"ref": 1970, "raw_active": (1937, 3968), "logical": (0.0, 180.0)},
}
ORDER = ("shoulder_pan", "shoulder_lift", "wrist_flex")
EDGE_MARGIN_TICKS = 40
PAN_HALF_SPAN_DEG = 60.0
LIFT_UP_SPAN_DEG = 90.0
WRIST_BEND_SPAN_DEG = 120.0

MODEL_UP = np.array([0.402, 0.015, 0.915])
MODEL_UP = MODEL_UP / np.linalg.norm(MODEL_UP)
AZIMUTH_HALF_SPAN_DEG = 60.0
MIN_HEIGHT_MM = 70.0
MAX_ELEVATION_DEG = 90.0
MIN_RADIAL_MM = 150.0

# Bin grid. Radius gets the most divisions because "varied radius and distance"
# is the explicit requirement; elevation the fewest because the reachable
# elevation band is narrower than the azimuth fan.
N_BINS_RADIUS = 5
N_BINS_AZIMUTH = 4
N_BINS_ELEVATION = 2

N_CANDIDATES = 60000
N_FINAL = 50
SEED = 2026

_E1 = np.cross(MODEL_UP, np.array([0.0, 1.0, 0.0]))
_E1 = _E1 / np.linalg.norm(_E1)
_E2 = np.cross(MODEL_UP, _E1)


def raw_to_logical(axis, raw):
    lo, hi = AXES[axis]["raw_active"]
    llo, lhi = AXES[axis]["logical"]
    return llo + (raw - lo) / (hi - lo) * (lhi - llo)


def sample_range(axis, lo_deg, hi_deg):
    ref = AXES[axis]["ref"]
    amin, amax = AXES[axis]["raw_active"]
    return (max(ref + lo_deg * TICKS_PER_DEG, amin + EDGE_MARGIN_TICKS),
            min(ref + hi_deg * TICKS_PER_DEG, amax - EDGE_MARGIN_TICKS))


def farthest_point_indices(points, k, rng):
    """Greedy max-min-distance selection -- the same idea as the 06 package's
    `farthest_point.farthest_point_sample`, kept local so this script has no
    cross-package import."""
    if k >= len(points):
        return list(range(len(points)))
    chosen = [int(rng.integers(0, len(points)))]
    dmin = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(points - points[nxt], axis=1))
    return chosen


def allocate_bin_counts(bin_sizes, n_targets):
    """Near-equal allocation across occupied bins, capped by what each bin
    actually holds; leftovers spread round-robin. Mirrors
    `uniform_bin_sampler.allocate_bin_counts`."""
    order = sorted(bin_sizes, key=lambda b: bin_sizes[b])
    base = n_targets // len(order)
    alloc = {b: min(base, bin_sizes[b]) for b in order}
    remaining = n_targets - sum(alloc.values())
    while remaining > 0:
        progressed = False
        for b in order:
            if remaining <= 0:
                break
            if alloc[b] < bin_sizes[b]:
                alloc[b] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return {b: n for b, n in alloc.items() if n > 0}


# Azimuth is measured from the reference pose's own heading, so "0 deg" is
# where the arm points when straight rather than an arbitrary model axis.
_refp = fk_rigid(np.array([0.0, math.pi / 2, 0.0]), load_config(CONFIG_PATH))
_REF_AZ = math.degrees(math.atan2(float(np.dot(_refp, _E2)), float(np.dot(_refp, _E1))))


def main():
    config = load_config(CONFIG_PATH)
    rng = np.random.default_rng(SEED)
    win = {
        "shoulder_pan": sample_range("shoulder_pan", -PAN_HALF_SPAN_DEG, PAN_HALF_SPAN_DEG),
        "shoulder_lift": sample_range("shoulder_lift", 0.0, LIFT_UP_SPAN_DEG),
        "wrist_flex": sample_range("wrist_flex", 0.0, WRIST_BEND_SPAN_DEG),
    }

    cands = []
    for _ in range(N_CANDIDATES * 6):
        if len(cands) >= N_CANDIDATES:
            break
        raws = {a: rng.uniform(*win[a]) for a in ORDER}
        # Same joint convention as generate_3d_points_v5.py, deliberately
        # unchanged: it is the model the previous runs used, and it is NOT
        # calibrated. The FK here only shapes WHICH points get visited; the
        # recorded data never depends on it.
        q = np.array([
            (raws["shoulder_pan"] - AXES["shoulder_pan"]["ref"]) / TICKS_PER_DEG * math.pi / 180.0,
            math.pi / 2 + (raws["shoulder_lift"] - AXES["shoulder_lift"]["ref"]) / TICKS_PER_DEG * math.pi / 180.0,
            (raws["wrist_flex"] - AXES["wrist_flex"]["ref"]) / TICKS_PER_DEG * math.pi / 180.0,
        ])
        p = fk_rigid(q, config)
        r = float(np.linalg.norm(p))
        if r < MIN_RADIAL_MM:
            continue
        h = float(np.dot(p, MODEL_UP))
        if h < MIN_HEIGHT_MM:
            continue
        el = math.degrees(math.asin(h / r))
        if el > MAX_ELEVATION_DEG:
            continue
        az = math.degrees(math.atan2(float(np.dot(p, _E2)), float(np.dot(p, _E1)))) - _REF_AZ
        az = (az + 180.0) % 360.0 - 180.0
        if abs(az) > AZIMUTH_HALF_SPAN_DEG:
            continue
        cands.append({"raws": raws, "q": q, "p": p, "r": r, "az": az, "el": el, "h": h})

    print(f"candidates in workspace: {len(cands)}")
    if len(cands) < N_FINAL:
        raise SystemExit(f"only {len(cands)} candidates, need {N_FINAL}")

    R = np.array([c["r"] for c in cands])
    A = np.array([c["az"] for c in cands])
    E = np.array([c["el"] for c in cands])
    # Quantile edges, not equal-width: equal-width bins over a non-uniform pool
    # leave some bins nearly empty, and the allocator then cannot give them a
    # fair share. Quantiles make every bin start with comparable stock.
    r_edges = np.quantile(R, np.linspace(0, 1, N_BINS_RADIUS + 1))
    a_edges = np.quantile(A, np.linspace(0, 1, N_BINS_AZIMUTH + 1))
    e_edges = np.quantile(E, np.linspace(0, 1, N_BINS_ELEVATION + 1))

    def which(v, edges):
        return int(min(max(np.searchsorted(edges, v, side="right") - 1, 0), len(edges) - 2))

    bins = defaultdict(list)
    for i, c in enumerate(cands):
        key = f"r{which(c['r'], r_edges)}|a{which(c['az'], a_edges)}|e{which(c['el'], e_edges)}"
        c["bin_id"] = key
        bins[key].append(i)

    sizes = {b: len(v) for b, v in bins.items()}
    alloc = allocate_bin_counts(sizes, N_FINAL)
    print(f"occupied bins: {len(sizes)} of {N_BINS_RADIUS*N_BINS_AZIMUTH*N_BINS_ELEVATION}"
          f"   -> allocated to {len(alloc)} bins, {sum(alloc.values())} points")

    picked = []
    for b, n in alloc.items():
        idxs = bins[b]
        pts = np.array([cands[i]["p"] for i in idxs])
        for j in farthest_point_indices(pts, n, rng):
            picked.append(idxs[j])

    points = []
    for i, idx in enumerate(picked):
        c = cands[idx]
        points.append({
            "target_id": f"T{i:03d}",
            "planned_order_index": i,
            "bin_id": c["bin_id"],
            "raw_ticks": [round(c["raws"][a]) for a in ORDER],
            "deg_from_reference": [round((c["raws"][a] - AXES[a]["ref"]) / TICKS_PER_DEG, 2) for a in ORDER],
            "logical_deg": [round(raw_to_logical(a, c["raws"][a]), 1) for a in ORDER],
            "q_rad": c["q"].tolist(),
            "p_xyz_mm": c["p"].tolist(),
            "radius_mm": round(c["r"], 1),
            "azimuth_deg": round(c["az"], 2),
            "elevation_deg": round(c["el"], 2),
            "height_mm": round(c["h"], 1),
            "workspace_mode": "binbalanced_v6",
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(points, f, indent=2)

    print(f"\nselected {len(points)} -> {os.path.basename(OUT_PATH)}\n")
    for i, a in enumerate(ORDER):
        v = [p["deg_from_reference"][i] for p in points]
        print(f"  {a:15s} from-ref {min(v):+7.1f}..{max(v):+7.1f} deg")
    for nm in ("radius_mm", "azimuth_deg", "elevation_deg", "height_mm"):
        v = [p[nm] for p in points]
        print(f"  {nm:15s} {min(v):8.1f}..{max(v):8.1f}")

    # Flatness check: how many points landed in each radius shell. A Gaussian
    # pile-up would show a heavy middle; this should be near-uniform.
    print("\n  points per radius bin (flatness check):")
    per = defaultdict(int)
    for p in points:
        per[p["bin_id"].split("|")[0]] += 1
    for k in sorted(per):
        lo, hi = r_edges[int(k[1:])], r_edges[int(k[1:]) + 1]
        print(f"    {k}  {lo:6.0f}-{hi:6.0f} mm : {per[k]:3d} points  {'#' * per[k]}")


if __name__ == "__main__":
    main()
