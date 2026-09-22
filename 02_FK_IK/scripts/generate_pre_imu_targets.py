from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workspace_points.pre_imu_targets import (
    generate_pre_imu_candidate_targets,
    load_pre_imu_workspace_config,
    select_bin_balanced_targets,
    write_pre_imu_targets_csv,
    write_pre_imu_targets_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate pre-IMU forward-hemisphere rigid target points."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--mode", choices=["pilot", "main"], default="pilot")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_pre_imu_workspace_config(args.config)
    if args.stage not in config.stages:
        valid = ", ".join(sorted(config.stages))
        raise SystemExit(f"unknown stage {args.stage!r}; valid stages: {valid}")

    stage = config.stages[args.stage]
    count = args.count if args.count is not None else stage.default_target_count
    seed = args.seed if args.seed is not None else int(config.defaults.get("seed", 101))
    candidates, stats = generate_pre_imu_candidate_targets(
        config,
        args.stage,
        mode=args.mode,
        count=count,
        seed=seed,
    )
    prefix = config.defaults.get("target_id_prefix", {}).get(args.stage, "PREIMU")
    targets = select_bin_balanced_targets(candidates, count, seed=seed, target_id_prefix=prefix)
    issues: list[str] = []
    if len(targets) < count:
        issues.append(f"requested {count} targets but only selected {len(targets)}")
    if not targets:
        issues.append("no targets selected")

    write_pre_imu_targets_csv(args.output, targets)
    write_pre_imu_targets_metadata(
        args.metadata_output,
        config_path=args.config,
        config=config,
        stage_name=args.stage,
        mode=args.mode,
        seed=seed,
        targets=targets,
        candidate_count=stats["candidate_count"],
        rejected_count=stats["rejected_count"],
        validation_issues=issues,
    )

    print(f"Targets: {len(targets)}")
    print(f"Candidates: {stats['candidate_count']}")
    print(f"Rejected: {stats['rejected_count']}")
    print(f"Output: {args.output}")
    print(f"Metadata: {args.metadata_output}")
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("Pre-IMU target generation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
