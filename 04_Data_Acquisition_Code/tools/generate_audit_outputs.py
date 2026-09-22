"""CLI: python tools/generate_audit_outputs.py

One-shot generator that re-runs Tasks 2/3's audits against the real
`sessions/real_motion_demo/` files (read-only) and writes the
machine-readable outputs under `outputs/`:

- `sampling_audit.json`  -- per (segment, IMU sensor) packet_type rates +
  UART byte-budget report at 9600 baud (`tools/audit_sampling.py`).
- `packet_integrity.json` -- per (segment, IMU sensor) record count,
  frame_length distribution, checksum_ok pass rate, parse-error count.
- `target_plan_audit.json` -- `tools/audit_target_plan.py --plan-json`
  against `target_plan_3d_random_n50_v3.json`.

Human-readable write-ups of the same numbers live in
`docs/SAMPLING_AND_UART_BUDGET.md` and `docs/TARGET_PLAN_AUDIT_V3.md`.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
PKG08_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(PKG08_ROOT))

from tools.audit_sampling import compute_packet_type_rates, uart_budget_report
from tools.audit_target_plan import audit_json_plan

REAL_DIR = PKG08_ROOT / "sessions" / "real_motion_demo"
OUTPUTS_DIR = PKG08_ROOT / "outputs"

IMU_SENSOR_IDS = ["imu0_base", "imu1_link_root", "imu2_link_mid", "imu3_tip"]
SEGMENTS = ["006", "007", "012"]
UART_BAUD = 9600  # experiment_scripts/run_real_imu_capture2.py, experiment_scripts/README.md


def _imu_file_integrity(path: Path) -> dict:
    n = parse_errors = 0
    frame_length_counts: Counter = Counter()
    checksum_ok_true = checksum_ok_total = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            frame_length_counts[rec.get("frame_length")] += 1
            ok = rec.get("checksum_ok")
            if ok is not None:
                checksum_ok_total += 1
                if ok:
                    checksum_ok_true += 1
    return {
        "record_count": n,
        "parse_errors": parse_errors,
        "frame_length_counts": {str(k): v for k, v in frame_length_counts.items()},
        "checksum_ok_pass_rate": (checksum_ok_true / checksum_ok_total) if checksum_ok_total else None,
    }


def build_sampling_and_integrity() -> tuple[dict, dict]:
    sampling: dict = {"uart_baud": UART_BAUD, "frame_size_bytes": 11, "segments": {}}
    integrity: dict = {"segments": {}}

    for segment in SEGMENTS:
        sampling["segments"][segment] = {}
        integrity["segments"][segment] = {}
        for sensor_id in IMU_SENSOR_IDS:
            path = REAL_DIR / f"{sensor_id}_segment{segment}.jsonl"
            rates = compute_packet_type_rates(str(path))
            budget = uart_budget_report(UART_BAUD, rates)
            sampling["segments"][segment][sensor_id] = {"rates": rates, "uart_budget": budget}
            integrity["segments"][segment][sensor_id] = _imu_file_integrity(path)

    return sampling, integrity


def main() -> int:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    sampling, integrity = build_sampling_and_integrity()
    (OUTPUTS_DIR / "sampling_audit.json").write_text(json.dumps(sampling, indent=2), encoding="utf-8")
    (OUTPUTS_DIR / "packet_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    plan_path = REAL_DIR / "target_plan_3d_random_n50_v3.json"
    ok, _lines, summary = audit_json_plan(plan_path)
    (OUTPUTS_DIR / "target_plan_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote {OUTPUTS_DIR / 'sampling_audit.json'}")
    print(f"wrote {OUTPUTS_DIR / 'packet_integrity.json'}")
    print(f"wrote {OUTPUTS_DIR / 'target_plan_audit.json'} (plan {'PASS' if ok else 'FAIL'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
