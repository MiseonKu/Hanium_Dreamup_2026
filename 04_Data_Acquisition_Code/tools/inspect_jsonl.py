"""CLI: python tools/inspect_jsonl.py <path-to-jsonl>

Prints a summary of a raw JSONL stream file: record count, first/last
timestamp, `valid`/`checksum_ok` pass rate, `quality_flags` histogram.
Read-only -- never modifies the file. Useful for eyeballing collector
output without opening it in an editor.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


def summarize(path: str) -> dict:
    count = 0
    parse_errors = 0
    first_utc = last_utc = None
    first_t_pc_rx_ns = last_t_pc_rx_ns = None
    valid_true = 0
    checksum_ok_true = checksum_ok_total = 0
    flags: Counter = Counter()

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            utc = rec.get("utc_iso")
            if utc:
                if first_utc is None:
                    first_utc = utc
                last_utc = utc

            t_ns = rec.get("t_pc_rx_ns")
            if isinstance(t_ns, int):
                if first_t_pc_rx_ns is None:
                    first_t_pc_rx_ns = t_ns
                last_t_pc_rx_ns = t_ns

            if rec.get("valid"):
                valid_true += 1

            checksum_ok = rec.get("checksum_ok")
            if checksum_ok is not None:
                checksum_ok_total += 1
                if checksum_ok:
                    checksum_ok_true += 1

            for flag in rec.get("quality_flags") or []:
                flags[flag] += 1

    duration_s = None
    if first_t_pc_rx_ns is not None and last_t_pc_rx_ns is not None:
        duration_s = (last_t_pc_rx_ns - first_t_pc_rx_ns) / 1e9

    return {
        "record_count": count,
        "parse_errors": parse_errors,
        "first_utc_iso": first_utc,
        "last_utc_iso": last_utc,
        "first_t_pc_rx_ns": first_t_pc_rx_ns,
        "last_t_pc_rx_ns": last_t_pc_rx_ns,
        "duration_s": duration_s,
        "valid_pass_rate": (valid_true / count) if count else None,
        "checksum_ok_pass_rate": (checksum_ok_true / checksum_ok_total) if checksum_ok_total else None,
        "quality_flags_histogram": dict(flags),
    }


def _fmt_rate(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.path):
        print(f"error: not a file: {args.path}", file=sys.stderr)
        return 1

    s = summarize(args.path)
    print(f"file:               {args.path}")
    print(f"records:            {s['record_count']}  (parse errors: {s['parse_errors']})")
    print(f"first utc_iso:      {s['first_utc_iso']}")
    print(f"last utc_iso:       {s['last_utc_iso']}")
    print(f"duration_s (t_pc):  {s['duration_s']}")
    print(f"valid pass rate:    {_fmt_rate(s['valid_pass_rate'])}")
    print(f"checksum_ok rate:   {_fmt_rate(s['checksum_ok_pass_rate'])} "
          f"{'' if s['checksum_ok_pass_rate'] is not None else '(field not present in any record)'}")
    print("quality_flags histogram:")
    if s["quality_flags_histogram"]:
        for flag, n in sorted(s["quality_flags_histogram"].items(), key=lambda kv: -kv[1]):
            print(f"  {flag}: {n}")
    else:
        print("  (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
