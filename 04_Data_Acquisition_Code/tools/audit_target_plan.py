"""Audit a generated pre-IMU target plan against acceptance criteria. Two
plan shapes are supported:

1. CSV+metadata, from `generate_front_120_target_plan.py` (the package's
   own generator, `--mode front_120` only) -- `audit()` / `--csv --metadata`.
2. A flat JSON list of `{target_id, planned_order_index, q_rad, p_xyz_mm,
   logical_deg, workspace_mode}` records, e.g.
   `sessions/real_motion_demo/target_plan_3d_random_n50_v3.json` -- produced
   by the ad hoc `experiment_scripts/generate_3d_points_50_v3.py` (4000-
   candidate pool + greedy farthest-point sampling for even Cartesian
   coverage, not this package's own generator). See
   `docs/TARGET_PLAN_AUDIT_V3.md` for why this plan shape diverges from (1)
   and isn't just run through `audit()`. `audit_json_plan()` / `--plan-json`.

CSV-mode acceptance criteria (`audit()`):
- azimuth violation count == 0 (every target's `azimuth_deg` inside the
  requested mode's `azimuth_limit_deg` bounds - bounds are re-read from the
  config the plan's own metadata points at, not hardcoded here)
- duplicate count == 0 (target_id and position both unique)
- requested count == generated count
- IK/joint-limit violation counts == 0, *if applicable*. Per DECISIONS.md
  #2, the pre-IMU pipeline (`workspace_points/pre_imu_targets.py`) is
  intentionally IK-free - direct rigid-FK targets, no IK solve, no
  post-hoc joint-limit check exists upstream. This script reports that as
  not-applicable rather than inventing a check the pipeline doesn't do.

JSON-mode acceptance criteria (`audit_json_plan()`): actual count vs.
expected, duplicate target_id/position counts, per-axis `logical_deg`
range, azimuth min/max derived from `p_xyz_mm` (not an assumed formula),
and a joint-limit re-check via `fkik.geometry.within_joint_limits` using
the exact same call signature (`margin=0.05`) the generator itself used --
this plan *is* IK-adjacent (it samples in joint space and rigid-FKs to
Cartesian), unlike the CSV pipeline above.

Prints a pass/fail summary. Exit code 0 on pass, 1 on fail.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
PKG08_ROOT = TOOLS_DIR.parent
REPO_ROOT = PKG08_ROOT.parent
PKG02_ROOT = REPO_ROOT / "02_FK_IK"
DEFAULT_ROBOT_CONFIG = PKG02_ROOT / "config" / "robot_config.yaml"
# Same margin `experiment_scripts/generate_3d_points_50_v3.py` filtered
# candidates with (MIN_JOINT_MARGIN_RAD) -- re-checking with a different
# margin would not actually verify what the generator did.
V3_JOINT_MARGIN_RAD = 0.05


def _load_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def audit(csv_path: Path, metadata_path: Path, expected_count: int | None) -> tuple[bool, list[str]]:
    metadata = _load_metadata(metadata_path)
    rows = _load_rows(csv_path)
    lines: list[str] = []
    ok = True

    mode = metadata["mode"]
    config = json.loads(Path(metadata["config_path"]).read_text(encoding="utf-8"))
    lo, hi = config["workspace"]["azimuth_limit_deg"][mode]
    azimuth_violations = [
        row["target_id"] for row in rows if not (lo <= float(row["azimuth_deg"]) <= hi)
    ]
    lines.append(f"azimuth violations ({mode} sector [{lo}, {hi}] deg): {len(azimuth_violations)}")
    if azimuth_violations:
        ok = False

    ids = [row["target_id"] for row in rows]
    positions = [
        (row["p_rigid_target_m_x"], row["p_rigid_target_m_y"], row["p_rigid_target_m_z"])
        for row in rows
    ]
    duplicate_ids = len(ids) - len(set(ids))
    duplicate_positions = len(positions) - len(set(positions))
    lines.append(f"duplicate target_id count: {duplicate_ids}")
    lines.append(f"duplicate position count: {duplicate_positions}")
    if duplicate_ids or duplicate_positions:
        ok = False

    generated_count = len(rows)
    requested_count = expected_count if expected_count is not None else metadata.get("target_count")
    validation_issues = metadata.get("validation_issues", [])
    count_ok = generated_count == requested_count and not validation_issues
    lines.append(
        f"requested count: {requested_count}, generated count: {generated_count} "
        f"({'OK' if count_ok else 'MISMATCH'})"
    )
    if validation_issues:
        lines.append(f"generator-reported validation issues: {validation_issues}")
    if not count_ok:
        ok = False

    lines.append(
        "IK/joint-limit violations: N/A - pre-IMU pipeline is intentionally IK-free "
        "(direct rigid-FK targets, no IK solve, no joint-limit post-check upstream; "
        "see DECISIONS.md #2)"
    )

    return ok, lines


def _azimuth_deg(p_xyz_mm: list[float]) -> float:
    """Same formula as `docs/TARGET_WORKSPACE_FRONT_120.md`'s documented
    mechanism: azimuth about the base-frame Z axis, center=[0,0,0]."""
    x, y = p_xyz_mm[0], p_xyz_mm[1]
    return math.degrees(math.atan2(y, x))


def _text_histogram(values: list[float], bins: int = 12) -> list[str]:
    """Numeric, matplotlib-free coverage summary -- `requirements.txt` does
    not list matplotlib, so this must be runnable without it."""
    lo, hi = min(values), max(values)
    width = (hi - lo) / bins if hi > lo else 1.0
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / width)) if hi > lo else 0
        counts[idx] += 1
    max_count = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bin_lo = lo + i * width
        bin_hi = lo + (i + 1) * width
        bar = "#" * round(30 * c / max_count)
        lines.append(f"  [{bin_lo:7.2f}, {bin_hi:7.2f}) deg: {c:3d} {bar}")
    return lines


def audit_json_plan(
    plan_path: Path,
    robot_config_path: Path = DEFAULT_ROBOT_CONFIG,
    expected_count: int = 50,
    joint_margin_rad: float = V3_JOINT_MARGIN_RAD,
) -> tuple[bool, list[str], dict]:
    """Audit a flat-JSON `target_id/planned_order_index/q_rad/p_xyz_mm/
    logical_deg/workspace_mode` plan. Returns (ok, human-readable lines,
    machine-readable summary dict)."""
    points = json.loads(plan_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    ok = True

    actual_count = len(points)
    count_ok = actual_count == expected_count
    lines.append(f"actual count: {actual_count} (expected {expected_count}) {'OK' if count_ok else 'MISMATCH'}")
    ok = ok and count_ok

    ids = [p["target_id"] for p in points]
    positions = [tuple(p["p_xyz_mm"]) for p in points]
    duplicate_ids = len(ids) - len(set(ids))
    duplicate_positions = len(positions) - len(set(positions))
    lines.append(f"duplicate target_id count: {duplicate_ids}")
    lines.append(f"duplicate position count: {duplicate_positions}")
    ok = ok and duplicate_ids == 0 and duplicate_positions == 0

    axis_ranges = []
    n_axes = len(points[0]["logical_deg"]) if points else 0
    for axis in range(n_axes):
        vals = [p["logical_deg"][axis] for p in points]
        axis_ranges.append((min(vals), max(vals)))
        lines.append(f"logical_deg axis[{axis}] range: [{min(vals):.2f}, {max(vals):.2f}] deg")

    azimuths = [_azimuth_deg(p["p_xyz_mm"]) for p in points]
    az_lo, az_hi = min(azimuths), max(azimuths)
    lines.append(f"azimuth (derived from p_xyz_mm, not assumed) range: [{az_lo:.2f}, {az_hi:.2f}] deg")

    sys.path.insert(0, str(PKG02_ROOT))
    from fkik.geometry import load_config, within_joint_limits  # noqa: E402  (02 package, same env)

    config = load_config(robot_config_path)
    violations = [p["target_id"] for p in points if not within_joint_limits(p["q_rad"], config, margin=joint_margin_rad)]
    lines.append(
        f"joint-limit violations (fkik.geometry.within_joint_limits, margin={joint_margin_rad} rad, "
        f"same call the generator used): {len(violations)}"
    )
    if violations:
        ok = False
        lines.append(f"  violating ids: {violations}")

    xs = [p["p_xyz_mm"][0] for p in points]
    ys = [p["p_xyz_mm"][1] for p in points]
    zs = [p["p_xyz_mm"][2] for p in points]
    lines.append(
        f"workspace coverage (p_xyz_mm): x=[{min(xs):.1f},{max(xs):.1f}] "
        f"y=[{min(ys):.1f},{max(ys):.1f}] z=[{min(zs):.1f},{max(zs):.1f}] mm"
    )
    lines.append(f"azimuth histogram ({len(azimuths)} points, 12 bins):")
    lines.extend(_text_histogram(azimuths, bins=12))

    summary = {
        "plan_path": str(plan_path),
        "actual_count": actual_count,
        "expected_count": expected_count,
        "duplicate_target_id_count": duplicate_ids,
        "duplicate_position_count": duplicate_positions,
        "logical_deg_axis_ranges": axis_ranges,
        "azimuth_deg_range": [az_lo, az_hi],
        "joint_limit_margin_rad": joint_margin_rad,
        "joint_limit_violation_count": len(violations),
        "joint_limit_violation_ids": violations,
        "workspace_coverage_mm": {
            "x": [min(xs), max(xs)],
            "y": [min(ys), max(ys)],
            "z": [min(zs), max(zs)],
        },
        "ok": ok,
    }
    return ok, lines, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a generated pre-IMU target plan (CSV+metadata, or a flat-JSON plan) against acceptance criteria."
    )
    parser.add_argument("--csv", type=Path, default=None, help="CSV-mode: path to the generated targets CSV.")
    parser.add_argument("--metadata", type=Path, default=None, help="CSV-mode: path to the generator's metadata JSON.")
    parser.add_argument("--plan-json", type=Path, default=None,
                         help="JSON-mode: path to a flat-JSON target plan (e.g. target_plan_3d_random_n50_v3.json).")
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG,
                         help="JSON-mode: robot_config.yaml to re-check joint limits against.")
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None,
                         help="JSON-mode: also write the machine-readable summary dict to this path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.plan_json is not None:
        expected = args.expected_count if args.expected_count is not None else 50
        ok, lines, summary = audit_json_plan(args.plan_json, args.robot_config, expected)
        print(f"Target plan audit (JSON mode): {args.plan_json}")
        for line in lines:
            print(f"- {line}")
        print("PASS" if ok else "FAIL")
        if args.json_out is not None:
            args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"wrote {args.json_out}")
        return 0 if ok else 1

    if args.csv is None or args.metadata is None:
        print("error: either --plan-json, or both --csv and --metadata, are required", file=sys.stderr)
        return 2

    ok, lines = audit(args.csv, args.metadata, args.expected_count)
    print(f"Target plan audit: {args.csv}")
    for line in lines:
        print(f"- {line}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
