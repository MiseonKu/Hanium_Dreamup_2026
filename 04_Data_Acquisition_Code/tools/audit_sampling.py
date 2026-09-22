"""CLI: python tools/audit_sampling.py <path-to-jsonl> [--configured-rate-hz HZ] [--uart-baud BAUD]

Computes configured vs. measured sample rate, median period, P95 jitter,
max gap, and packet-loss ratio from a stream's `t_pc_rx_ns` sequence
(master spec section 18). Pure stdlib (`statistics` module) -- the core
computation lives in `compute_sampling_stats`, importable independently of
the CLI (see `tests/test_sampling_audit.py`).

`--uart-baud` additionally computes, for an IMU file, the measured rate of
each `packet_type` and a UART byte-budget report (bytes/sec required vs.
bytes/sec available at that baud, 8N1 framing) -- see
`docs/SAMPLING_AND_UART_BUDGET.md` for the worked numbers against the real
demo IMU captures (9600 baud).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.clock_sync import _percentile  # reuse the existing stdlib-only percentile helper

FRAME_SIZE_BYTES = 11  # WT901C fixed frame: 1 header + 1 type + 8 payload + 1 checksum
UART_BITS_PER_BYTE_8N1 = 10  # 1 start + 8 data + 1 stop, no parity


def uart_byte_capacity_per_sec(baud_rate: int, bits_per_byte: int = UART_BITS_PER_BYTE_8N1) -> float:
    """Raw byte throughput a UART link can carry at `baud_rate`, 8N1 framing."""
    return baud_rate / bits_per_byte


def compute_packet_type_rates(path: str) -> dict:
    """Per-`packet_type` record count and measured rate (count / the file's
    `t_pc_rx_ns` span), plus the combined (all packet types) rate. Pure
    function over a decoded IMU JSONL file."""
    counts: Counter = Counter()
    tmin = tmax = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pt = rec.get("packet_type")
            if pt is not None:
                counts[pt] += 1
            t = rec.get("t_pc_rx_ns")
            if isinstance(t, int):
                tmin = t if tmin is None else min(tmin, t)
                tmax = t if tmax is None else max(tmax, t)

    span_s = (tmax - tmin) / 1e9 if tmin is not None and tmax is not None and tmax > tmin else None
    combined_count = sum(counts.values())
    return {
        "span_s": span_s,
        "packet_type_counts": dict(counts),
        "packet_type_rate_hz": {pt: (n / span_s if span_s else None) for pt, n in counts.items()},
        "combined_record_count": combined_count,
        "combined_rate_hz": (combined_count / span_s) if span_s else None,
    }


def uart_budget_report(baud_rate: int, packet_type_rates: dict, frame_size_bytes: int = FRAME_SIZE_BYTES) -> dict:
    """Compare the measured combined byte rate (from `compute_packet_type_rates`)
    against the raw UART byte budget at `baud_rate` (8N1). Pure arithmetic,
    no I/O."""
    capacity_bytes_per_sec = uart_byte_capacity_per_sec(baud_rate)
    combined_rate_hz = packet_type_rates.get("combined_rate_hz")
    required_bytes_per_sec = (combined_rate_hz * frame_size_bytes) if combined_rate_hz else None
    n_types = len(packet_type_rates.get("packet_type_counts") or {}) or 1
    max_combined_rate_hz_at_capacity = capacity_bytes_per_sec / frame_size_bytes
    return {
        "baud_rate": baud_rate,
        "frame_size_bytes": frame_size_bytes,
        "capacity_bytes_per_sec": capacity_bytes_per_sec,
        "required_bytes_per_sec": required_bytes_per_sec,
        "utilization_ratio": (required_bytes_per_sec / capacity_bytes_per_sec) if required_bytes_per_sec else None,
        "n_packet_types": n_types,
        "max_combined_rate_hz_at_capacity": max_combined_rate_hz_at_capacity,
        "max_per_packet_type_rate_hz_at_capacity": max_combined_rate_hz_at_capacity / n_types,
    }


def compute_sampling_stats(t_pc_rx_ns: list[int], configured_rate_hz: Optional[float] = None) -> dict:
    """Pure function over a sequence of PC-receive timestamps (ns,
    monotonic). Sorts defensively -- `out_of_order` records exist by
    design (`core/schemas.py`), so a caller may hand this the raw
    host_seq-ordered sequence as-is.

    Returns sample_count, span_s, configured_rate_hz (echoed back),
    measured_rate_hz, median_period_ms, jitter_p95_ms (P95 of
    |period - median_period|), max_gap_ms + max_gap_index (index into the
    sorted sequence where the largest gap ends), and packet_loss_ratio
    (only computed when `configured_rate_hz` is given: how many fewer
    samples arrived than a gap-free stream at that rate would have
    produced over the observed span).
    """
    ts = sorted(t_pc_rx_ns)
    n = len(ts)
    if n < 2:
        return {
            "sample_count": n,
            "span_s": 0.0 if n else None,
            "configured_rate_hz": configured_rate_hz,
            "measured_rate_hz": None,
            "median_period_ms": None,
            "jitter_p95_ms": None,
            "max_gap_ms": None,
            "max_gap_index": None,
            "packet_loss_ratio": None,
        }

    periods_ns = [b - a for a, b in zip(ts, ts[1:])]
    span_ns = ts[-1] - ts[0]
    span_s = span_ns / 1e9
    measured_rate_hz = (n - 1) / span_s if span_s > 0 else None

    median_period_ns = statistics.median(periods_ns)
    jitter_ms = [abs(p - median_period_ns) / 1e6 for p in periods_ns]
    jitter_p95_ms = _percentile(jitter_ms, 95)

    max_gap_ns = max(periods_ns)
    max_gap_index = periods_ns.index(max_gap_ns) + 1

    packet_loss_ratio = None
    if configured_rate_hz and configured_rate_hz > 0 and span_s > 0:
        expected_n = span_s * configured_rate_hz + 1
        packet_loss_ratio = max(0.0, (expected_n - n) / expected_n)

    return {
        "sample_count": n,
        "span_s": span_s,
        "configured_rate_hz": configured_rate_hz,
        "measured_rate_hz": measured_rate_hz,
        "median_period_ms": median_period_ns / 1e6,
        "jitter_p95_ms": jitter_p95_ms,
        "max_gap_ms": max_gap_ns / 1e6,
        "max_gap_index": max_gap_index,
        "packet_loss_ratio": packet_loss_ratio,
    }


def _load_t_pc_rx_ns(path: str) -> list[int]:
    values = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("t_pc_rx_ns")
            if isinstance(t, int):
                values.append(t)
    return values


def _auto_configured_rate_hz(path: str) -> Optional[float]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rate = (rec.get("sensor_config") or {}).get("configured_output_rate_hz")
            if rate:
                return float(rate)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--configured-rate-hz", type=float, default=None)
    parser.add_argument("--uart-baud", type=int, default=None,
                         help="Also report per-packet_type rates and a UART byte-budget "
                              "at this baud (8N1). Meaningful for imuN_* files only.")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.path):
        print(f"error: not a file: {args.path}", file=sys.stderr)
        return 1

    configured = args.configured_rate_hz
    if configured is None:
        configured = _auto_configured_rate_hz(args.path)

    ts = _load_t_pc_rx_ns(args.path)
    stats = compute_sampling_stats(ts, configured_rate_hz=configured)

    print(f"file:               {args.path}")
    print(f"sample_count:       {stats['sample_count']}")
    print(f"span_s:             {stats['span_s']}")
    print(f"configured_rate_hz: {stats['configured_rate_hz']}")
    print(f"measured_rate_hz:   {stats['measured_rate_hz']}")
    print(f"median_period_ms:   {stats['median_period_ms']}")
    print(f"jitter_p95_ms:      {stats['jitter_p95_ms']}")
    print(f"max_gap_ms:         {stats['max_gap_ms']} (at sample index {stats['max_gap_index']})")
    print(f"packet_loss_ratio:  {stats['packet_loss_ratio']}")

    if args.uart_baud is not None:
        rates = compute_packet_type_rates(args.path)
        budget = uart_budget_report(args.uart_baud, rates)
        print(f"\npacket_type_counts:   {rates['packet_type_counts']}")
        print(f"packet_type_rate_hz:  {rates['packet_type_rate_hz']}")
        print(f"combined_rate_hz:     {rates['combined_rate_hz']}")
        print(f"uart_baud:                          {budget['baud_rate']}")
        print(f"uart_capacity_bytes_per_sec:         {budget['capacity_bytes_per_sec']}")
        print(f"required_bytes_per_sec:              {budget['required_bytes_per_sec']}")
        print(f"utilization_ratio:                   {budget['utilization_ratio']}")
        print(f"max_combined_rate_hz_at_capacity:    {budget['max_combined_rate_hz_at_capacity']}")
        print(f"max_per_packet_type_rate_hz_at_capacity: {budget['max_per_packet_type_rate_hz_at_capacity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
