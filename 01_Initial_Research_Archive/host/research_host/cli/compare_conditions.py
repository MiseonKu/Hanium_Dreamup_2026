from __future__ import annotations

import argparse
from pathlib import Path

from ..reports import (
    generate_condition_report,
    read_analysis_rows,
    summarize_processed_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a descriptive condition comparison report.")
    parser.add_argument("--analysis", type=Path, nargs="+", required=True)
    parser.add_argument("--processed", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    quality_summary = summarize_processed_files(args.processed) if args.processed else None
    report = generate_condition_report(
        read_analysis_rows(args.analysis),
        quality_summary=quality_summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
