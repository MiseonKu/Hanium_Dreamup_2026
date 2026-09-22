from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..analysis import build_analysis_summary
from ..logging import ANALYSIS_FIELDNAMES, read_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze one processed trial CSV.")
    parser.add_argument("processed_csv", type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--estimate-prefix", default="P_hat_imu_base_mm")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with args.processed_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    metadata = read_json(args.metadata) if args.metadata else {}
    summary = build_analysis_summary(
        rows,
        trial_id=str(metadata.get("trial_id", "unknown")),
        condition_id=str(metadata.get("condition_id", "unknown")),
        robot_model=str(metadata.get("robot_model", "unknown")),
        target_id=str(metadata.get("target_id", "unknown")),
        estimate_prefix=args.estimate_prefix,
    )

    output = args.output or args.processed_csv.with_name(
        args.processed_csv.stem.replace("_processed", "_analysis") + ".csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANALYSIS_FIELDNAMES)
        writer.writeheader()
        writer.writerow(summary)
    print(f"Wrote analysis: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
