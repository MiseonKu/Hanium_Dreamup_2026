"""Replay tests against the real, already-collected demo session at
`sessions/real_motion_demo/` (read-only -- never writes into that
directory). These are regression guards for a later parser/pipeline change,
plus the invalid-segment/overlap-detector requested for segment 012:

- Segments 006/007 are known-good pilot data: 4 IMUs + encoder + Vive
  genuinely overlap in time and Vive tracking is valid.
- Segment 012 is a real, successful 50-point encoder sweep, but the Vive
  tracker was offline the whole segment (0% `tracking_valid`) AND the IMU
  capture window does not overlap the encoder/Vive window at all (the IMU
  collectors were started ~18 minutes earlier with a fixed 900s duration --
  see `experiment_scripts/run_real_imu_capture2.py` -- that ran out before
  the actual motion sweep happened, per an interrupted/restarted run in
  between). Segment 012 must be positively detected as NOT a usable fused
  dataset, not silently pass a per-file parse check.

See `docs/TARGET_PLAN_AUDIT_V3.md` for why `target_plan_3d_random_n50_v3.json`
specifically (not the same-named v1/v2 plans, which share the same T000-T049
id scheme) is the plan that drove segment 012 -- confirmed here by exact
match of the plan's `logical_deg` for T000 against the real recorded
`logical_deg` in `encoder_segment012.jsonl`'s settled samples for T000.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

REAL_DIR = os.path.join(WORKSPACE, "sessions", "real_motion_demo")

IMU_SENSOR_IDS = ["imu0_base", "imu1_link_root", "imu2_link_mid", "imu3_tip"]
ALL_STREAM_PREFIXES = IMU_SENSOR_IDS + ["encoder", "vive_tracker"]
SEGMENTS = ["006", "007", "012"]

# core/schemas.py:base_record's fields that must be present on every record
# regardless of stream, per the task's minimum list.
REQUIRED_BASE_FIELDS = [
    "schema", "session_id", "segment_id", "stream",
    "host_seq", "t_pc_rx_ns", "valid", "parse_status",
]

V3_PLAN_PATH = os.path.join(REAL_DIR, "target_plan_3d_random_n50_v3.json")


def _segment_path(prefix: str, segment: str) -> str:
    """Plain path if it exists, else the gzipped one -- `tools/prepare_dataset.py`
    compresses raw segments in place, so archived sessions only have `.jsonl.gz`."""
    path = os.path.join(REAL_DIR, f"{prefix}_segment{segment}.jsonl")
    return path if os.path.isfile(path) else path + ".gz"


def _iter_records(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            yield lineno, json.loads(line)  # let a JSONDecodeError fail the test loudly


def _time_range(path: str) -> tuple[int, int]:
    tmin = tmax = None
    for _, rec in _iter_records(path):
        t = rec.get("t_pc_rx_ns")
        if not isinstance(t, int):
            continue
        tmin = t if tmin is None else min(tmin, t)
        tmax = t if tmax is None else max(tmax, t)
    assert tmin is not None, f"no t_pc_rx_ns values found in {path}"
    return tmin, tmax


def _segment_streams_overlap(segment: str) -> bool:
    """True iff every stream file for `segment` shares a common
    [max(mins), min(maxs)] non-empty intersection window."""
    ranges = [_time_range(_segment_path(p, segment)) for p in ALL_STREAM_PREFIXES]
    lo = max(r[0] for r in ranges)
    hi = min(r[1] for r in ranges)
    return lo <= hi


def _vive_tracking_valid_ratio(segment: str) -> float:
    n = valid = 0
    for _, rec in _iter_records(_segment_path("vive_tracker", segment)):
        n += 1
        if rec["data"].get("tracking_valid"):
            valid += 1
    assert n > 0, f"no vive_tracker records in segment {segment}"
    return valid / n


class ParseIntegrityReplayTests(unittest.TestCase):
    """Cheap regression guard: every real demo file still parses and still
    carries the base-record contract. Should trivially pass -- it's not new
    discovery, it's a tripwire for a future schema/parser regression."""

    def test_every_line_parses_and_has_required_base_fields(self):
        checked_files = 0
        checked_records = 0
        for segment in SEGMENTS:
            for prefix in ALL_STREAM_PREFIXES:
                path = _segment_path(prefix, segment)
                self.assertTrue(os.path.isfile(path), f"missing real demo file: {path}")
                checked_files += 1
                for lineno, rec in _iter_records(path):
                    checked_records += 1
                    for field in REQUIRED_BASE_FIELDS:
                        self.assertIn(field, rec, f"{path}:{lineno} missing field {field!r}")
                        self.assertIsNotNone(rec[field], f"{path}:{lineno} field {field!r} is null")
        # Sanity floor so a future refactor that points this test at an
        # empty fixture directory fails loudly instead of vacuously passing.
        self.assertEqual(checked_files, len(SEGMENTS) * len(ALL_STREAM_PREFIXES))
        self.assertGreater(checked_records, 1000)


class CrossStreamOverlapAuditTests(unittest.TestCase):
    """The invalid-segment/overlap-detector from section 14: must tell a
    genuinely fused segment apart from one where the streams don't actually
    share a time window, not just confirm every file parses."""

    def test_segment_006_streams_genuinely_overlap(self):
        self.assertTrue(_segment_streams_overlap("006"))

    def test_segment_007_streams_genuinely_overlap(self):
        self.assertTrue(_segment_streams_overlap("007"))

    def test_segment_012_streams_do_not_overlap(self):
        # Known-bad case: IMU capture window ends ~352s before encoder/Vive
        # start (IMU started ~18 minutes earlier with a fixed 900s duration
        # that expired before the real sweep began). This must be detected,
        # not silently pass.
        self.assertFalse(_segment_streams_overlap("012"))

        imu_ranges = [_time_range(_segment_path(p, "012")) for p in IMU_SENSOR_IDS]
        encoder_lo, _ = _time_range(_segment_path("encoder", "012"))
        imu_hi = min(r[1] for r in imu_ranges)
        gap_s = (encoder_lo - imu_hi) / 1e9
        self.assertGreater(gap_s, 60.0, "expected a large (>1min) IMU-to-encoder gap in segment 012")


class InvalidViveSegmentDetectionTests(unittest.TestCase):
    """Must distinguish good Vive data from bad, not just confirm the file
    parses."""

    def test_segment_006_vive_mostly_valid(self):
        self.assertGreater(_vive_tracking_valid_ratio("006"), 0.5)

    def test_segment_007_vive_mostly_valid(self):
        self.assertGreater(_vive_tracking_valid_ratio("007"), 0.5)

    def test_segment_012_vive_is_effectively_all_invalid(self):
        # Measured ratio is exactly 0.0 (tracker was offline the whole
        # segment); 0.05 leaves headroom for a stray valid sample without
        # weakening the "this segment is unusable" signal.
        self.assertLess(_vive_tracking_valid_ratio("012"), 0.05)


class TrialTargetLinkageTests(unittest.TestCase):
    """Missing-trial-linkage detector: every non-null target_id actually
    recorded in segment 012 must be one of the ids the plan that drove it
    (`target_plan_3d_random_n50_v3.json`) actually generated."""

    @classmethod
    def setUpClass(cls):
        with open(V3_PLAN_PATH, encoding="utf-8") as fh:
            plan = json.load(fh)
        cls.plan_ids = {p["target_id"] for p in plan}
        assert len(plan) == 50, f"expected 50 targets in {V3_PLAN_PATH}, got {len(plan)}"

    def test_encoder_segment012_target_ids_all_in_plan(self):
        seen = {rec["target_id"] for _, rec in _iter_records(_segment_path("encoder", "012"))
                if rec.get("target_id") is not None}
        self.assertTrue(seen, "expected at least one non-null target_id in encoder_segment012.jsonl")
        self.assertTrue(seen.issubset(self.plan_ids), f"target_ids not in plan: {seen - self.plan_ids}")

    def test_vive_segment012_target_ids_all_in_plan(self):
        seen = {rec["target_id"] for _, rec in _iter_records(_segment_path("vive_tracker", "012"))
                if rec.get("target_id") is not None}
        self.assertTrue(seen, "expected at least one non-null target_id in vive_tracker_segment012.jsonl")
        self.assertTrue(seen.issubset(self.plan_ids), f"target_ids not in plan: {seen - self.plan_ids}")

    def test_encoder_segment012_covers_every_plan_target(self):
        # The real sweep drove all 50 planned targets -- a missing id here
        # would mean a trial silently never ran.
        seen = {rec["target_id"] for _, rec in _iter_records(_segment_path("encoder", "012"))
                if rec.get("target_id") is not None}
        self.assertEqual(seen, self.plan_ids)


if __name__ == "__main__":
    unittest.main()
