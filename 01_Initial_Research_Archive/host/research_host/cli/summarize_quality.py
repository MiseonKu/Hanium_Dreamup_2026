from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..logging import read_csv_dicts
from ..quality import load_quality_thresholds, quality_issues, summarize_quality


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize processed CSV quality flags.")
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, default=Path("configs/quality_thresholds.json"))
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_csv_dicts(args.processed)
    summary = summarize_quality(rows)
    thresholds = load_quality_thresholds(args.thresholds)
    issues = quality_issues(summary, thresholds)
    payload = {"summary": summary.to_jsonable(), "issues": issues}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Quality summary")
        print(f"rows: {summary.row_count}")
        print(f"dropped_sample_ratio: {summary.dropped_sample_ratio:.6f}")
        print(f"max_timestamp_gap_s: {summary.max_timestamp_gap_s:.6f}")
        print("flags:", ", ".join(f"{k}={v}" for k, v in summary.flag_counts.items()))
        if issues:
            print("Quality issues:")
            for issue in issues:
                print(f"- {issue}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
