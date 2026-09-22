"""Plot a target plan's rigid-FK tip positions, to eyeball coverage before
committing real motion to it.

    python plot_points.py [plan.json] [out.png]

Defaults to the current plan (`target_plan_v5.json`). This previously had the
v3 plan's path hardcoded, so it silently kept plotting a stale file after the
plan was regenerated.

The positions plotted are the RIGID model's predictions. The real flexible
link lands somewhere else -- that difference is the experiment's signal, not
an error in this plot.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SESSIONS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sessions", "real_motion_demo",
)
DEFAULT_PLAN = os.path.join(SESSIONS, "target_plan_v5.json")

plan_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PLAN
out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(plan_path)[0] + "_coverage.png"

if not os.path.isfile(plan_path):
    raise SystemExit(f"no such plan file: {plan_path}")

with open(plan_path, encoding="utf-8") as f:
    plan = json.load(f)

missing = [p.get("target_id") for p in plan if "p_xyz_mm" not in p]
if missing:
    raise SystemExit(f"plan entries lack p_xyz_mm (e.g. {missing[:3]}) -- wrong file?")

xs = [p["p_xyz_mm"][0] for p in plan]
ys = [p["p_xyz_mm"][1] for p in plan]
zs = [p["p_xyz_mm"][2] for p in plan]

fig = plt.figure(figsize=(10, 8))
fig.suptitle(f"{os.path.basename(plan_path)}  ({len(plan)} targets, rigid FK)")

ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax1.scatter(xs, ys, zs, c="gray", s=15)
ax1.set_xlabel("x mm"); ax1.set_ylabel("y mm"); ax1.set_zlabel("z mm")
ax1.set_title("3D point cloud")

for idx, (a, b, la, lb, title) in enumerate(
    [(xs, ys, "x mm", "y mm", "XY"), (xs, zs, "x mm", "z mm", "XZ"), (ys, zs, "y mm", "z mm", "YZ")],
    start=2,
):
    ax = fig.add_subplot(2, 2, idx)
    ax.scatter(a, b, c="gray", s=15)
    ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_title(f"{title} projection")
    ax.axis("equal")

plt.tight_layout()
plt.savefig(out_path, dpi=120)
print(f"saved {out_path}")
print(f"x:[{min(xs):.0f},{max(xs):.0f}] y:[{min(ys):.0f},{max(ys):.0f}] z:[{min(zs):.0f},{max(zs):.0f}]")
