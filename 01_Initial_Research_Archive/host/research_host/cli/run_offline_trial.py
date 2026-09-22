from __future__ import annotations

import argparse
from pathlib import Path

from ..analysis import build_analysis_summary
from ..logging import read_csv_dicts, read_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offline trial smoke pass.")
    parser.add_argument("processed_csv", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--estimate-prefix", default="P_hat_imu_base_mm")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_csv_dicts(args.processed_csv)
    metadata = read_json(args.metadata)
    summary = build_analysis_summary(
        rows,
        trial_id=str(metadata["trial_id"]),
        condition_id=str(metadata["condition_id"]),
        robot_model=str(metadata["robot_model"]),
        target_id=str(metadata["target_id"]),
        estimate_prefix=args.estimate_prefix,
    )
    print("Offline trial summary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
