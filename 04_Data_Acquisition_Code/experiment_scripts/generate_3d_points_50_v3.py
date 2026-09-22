"""v3: generate a much larger candidate pool across the full forward
hemisphere, then greedily farthest-point-sample 50 of them for even
spatial coverage (v2's independent-uniform-then-take-first-50 produced an
uneven, banded distribution -- joint-space-uniform != Cartesian-uniform
for a 2-link revolute chain).
"""
import json
import random
import sys

raise SystemExit(
    "SUPERSEDED -- do not run. Use generate_3d_points_v5.py.\n"
    "\n"
    "This generator has two errors that were only found later, against real\n"
    "hardware data (docs/FK_CONVENTION_CORRECTION.md):\n"
    "  1. It feeds the firmware's `logical_deg` to the FK as if it were a\n"
    "     physical angle. It is not -- the firmware stretches each axis's\n"
    "     calibrated span over an arbitrary label interval, giving a 30%\n"
    "     scale error on shoulder_lift.\n"
    "  2. It inherits q_init_rad=[0, pi/2, -pi/2], which describes an\n"
    "     L-shaped arm at the reference pose. The real arm is straight there.\n"
    "\n"
    "Kept, not deleted: this produced target_plan_3d_random_n50_v3.json,\n"
    "which drove real segment 012, so it is part of that data's provenance.\n"
)

sys.path.insert(0, r"C:\Users\hsw10\Documents\한이음 실험\capstone-code\02_FK_IK")

import numpy as np
from fkik.geometry import load_config, within_joint_limits
from fkik.rigid_fk import fk_rigid

config = load_config(
    r"C:\Users\hsw10\Documents\한이음 실험\capstone-code\02_FK_IK\config\robot_config.yaml"
)
HOME_DEG = [80.0, 35.0, 90.0]
Q_INIT = np.array(config.q_init_rad)

Q1_HALF_RANGE_RAD = np.radians(60.0)
MIN_JOINT_MARGIN_RAD = 0.05
MIN_RADIAL_MM = 120.0
N_CANDIDATES = 4000
N_FINAL = 50

random.seed(2026)
rng = np.random.default_rng(2026)

q2_lo, q2_hi = config.joint_limits_rad[1]
q3_lo, q3_hi = config.joint_limits_rad[2]

candidates = []
tries = 0
while len(candidates) < N_CANDIDATES and tries < N_CANDIDATES * 40:
    tries += 1
    q1 = rng.uniform(-Q1_HALF_RANGE_RAD, Q1_HALF_RANGE_RAD)
    q2 = rng.uniform(q2_lo, q2_hi)
    q3 = rng.uniform(q3_lo, q3_hi)
    q = np.array([q1, q2, q3])
    if not within_joint_limits(q, config, margin=MIN_JOINT_MARGIN_RAD):
        continue
    p = fk_rigid(q, config)
    if p[0] <= 0:
        continue
    if float(np.linalg.norm(p)) < MIN_RADIAL_MM:
        continue
    dq_deg = np.degrees(q - Q_INIT)
    logical_deg = [HOME_DEG[i] + float(dq_deg[i]) for i in range(3)]
    if not (0 <= logical_deg[0] <= 160):
        continue
    if not (10 <= logical_deg[1] <= 150):
        continue
    if not (0 <= logical_deg[2] <= 175):
        continue
    candidates.append({"q": q, "p": p, "logical_deg": logical_deg})

print(f"generated {len(candidates)} candidates in {tries} tries")

# Farthest-point sampling for even spatial coverage.
pts = np.array([c["p"] for c in candidates])
selected_idx = [int(rng.integers(0, len(pts)))]
min_dist = np.linalg.norm(pts - pts[selected_idx[0]], axis=1)
while len(selected_idx) < N_FINAL:
    next_idx = int(np.argmax(min_dist))
    selected_idx.append(next_idx)
    dist_to_new = np.linalg.norm(pts - pts[next_idx], axis=1)
    min_dist = np.minimum(min_dist, dist_to_new)

points = []
for i, idx in enumerate(selected_idx):
    c = candidates[idx]
    points.append({
        "target_id": f"T{i:03d}",
        "planned_order_index": i,
        "q_rad": c["q"].tolist(),
        "p_xyz_mm": c["p"].tolist(),
        "logical_deg": c["logical_deg"],
        "workspace_mode": "front_120_3d_random_sample_v3_fps",
    })

xs = [p["p_xyz_mm"][0] for p in points]
ys = [p["p_xyz_mm"][1] for p in points]
zs = [p["p_xyz_mm"][2] for p in points]
print(f"x:[{min(xs):.0f},{max(xs):.0f}] y:[{min(ys):.0f},{max(ys):.0f}] z:[{min(zs):.0f},{max(zs):.0f}]")

with open(
    r"C:\Users\hsw10\Documents\한이음 실험\capstone-code\04_Data_Acquisition_Code\sessions\real_motion_demo\target_plan_3d_random_n50_v3.json",
    "w", encoding="utf-8",
) as f:
    json.dump(points, f, indent=2)
print("saved.")
