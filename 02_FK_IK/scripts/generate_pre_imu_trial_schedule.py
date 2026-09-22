from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workspace_points.pre_imu_targets import (
    generate_pre_imu_trial_schedule_rows,
    read_pre_imu_targets_csv,
    write_pre_imu_trial_schedule_csv,
    write_pre_imu_trial_schedule_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a balanced pre-IMU schedule from a target CSV."
    )
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    target_rows = read_pre_imu_targets_csv(args.targets)
    schedule_rows = generate_pre_imu_trial_schedule_rows(
        target_rows,
        repeats=args.repeats,
        seed=args.seed,
    )
    write_pre_imu_trial_schedule_csv(args.output, schedule_rows)
    write_pre_imu_trial_schedule_metadata(
        args.metadata_output,
        target_csv_path=args.targets,
        repeats=args.repeats,
        seed=args.seed,
        schedule_rows=schedule_rows,
    )
    print(f"Schedule rows: {len(schedule_rows)}")
    print(f"Output: {args.output}")
    print(f"Metadata: {args.metadata_output}")
    print("Pre-IMU schedule OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
