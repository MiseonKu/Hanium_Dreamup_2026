"""CLI wrapper around `core.recovery.recover_partial_jsonl`.

Usage:
    python tools/recover_partial_jsonl.py <path-to-partial-file>
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.recovery import recover_partial_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recover a crashed/interrupted *.jsonl.partial file into "
        "a *.recovered.jsonl + *.recovery_report.json pair. The original "
        ".partial file is never modified."
    )
    parser.add_argument("partial_path", help="path to the *.jsonl.partial file")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.partial_path):
        print(f"error: not a file: {args.partial_path}", file=sys.stderr)
        return 1

    result = recover_partial_jsonl(args.partial_path)

    print(f"scanned:            {result.total_lines} lines")
    print(f"valid records:      {result.valid_record_count}")
    print(f"dropped fragments:  {len(result.dropped_lines)}")
    for frag in result.dropped_lines:
        print(
            f"  - from line {frag['starting_line_number']} "
            f"({frag['line_count']} line(s)), truncated={frag['truncated']}"
        )
    print(f"host_seq gaps:      {len(result.seq_gaps)}")
    for before, after in result.seq_gaps:
        print(f"  - {before} -> {after} (missing {after - before - 1})")
    print(f"recovered file:     {result.recovered_path}")
    print(f"recovery report:    {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
