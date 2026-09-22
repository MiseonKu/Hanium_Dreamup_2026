"""Tests for `tools/validate_raw_session.py`.

The load-bearing property here (master spec section 16, `core/jsonl_writer.py`):
a `.partial` file is an INTERRUPTED segment. Only a clean session close
renames it to the final `.jsonl`. So a session whose stream produced only a
`.partial` must not validate the same as one that closed cleanly -- treating
a `.partial` as complete is exactly the silent-data-loss failure the
partial/final split exists to prevent.

Run with:
    python -m pytest tests -q
from the `04_Data_Acquisition_Code/` directory.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

from tools.validate_raw_session import validate  # noqa: E402


def _make_session(tmp: str, *, streams: dict, session_complete: bool) -> str:
    """`streams` maps stream_id -> filename suffix ('.jsonl' or
    '.jsonl.partial'), or None for "enabled but no file at all"."""
    session_dir = os.path.join(tmp, "session_20260101_001")
    raw_dir = os.path.join(session_dir, "raw")
    os.makedirs(raw_dir)
    for stream_id, suffix in streams.items():
        if suffix is None:
            continue
        path = os.path.join(raw_dir, f"{stream_id}_segment000{suffix}")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"host_seq": 1, "stream": stream_id}) + "\n")
    manifest = {
        "session_id": "session_20260101_001",
        "enabled_streams": sorted(streams),
        "session_complete": session_complete,
        "interrupted": not session_complete,
    }
    with open(os.path.join(session_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return session_dir


class ValidateRawSessionTests(unittest.TestCase):
    def test_clean_session_with_final_files_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = _make_session(
                tmp, streams={"events": ".jsonl", "health": ".jsonl"}, session_complete=True
            )
            ok, lines = validate(session_dir)
            self.assertTrue(ok, "\n".join(lines))

    def test_partial_only_stream_does_not_count_as_a_completed_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = _make_session(
                tmp,
                streams={"events": ".jsonl", "health": ".jsonl.partial"},
                session_complete=False,
            )
            ok, lines = validate(session_dir)
            report = "\n".join(lines)
            self.assertFalse(ok, report)
            self.assertIn("health", report)
            self.assertIn(".partial", report)
            # and the clean stream is still reported as passing
            self.assertIn("PASS stream 'events'", report)

    def test_missing_stream_file_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = _make_session(
                tmp, streams={"events": ".jsonl", "health": None}, session_complete=True
            )
            ok, lines = validate(session_dir)
            self.assertFalse(ok, "\n".join(lines))

    def test_corrupt_jsonl_line_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = _make_session(tmp, streams={"events": ".jsonl"}, session_complete=True)
            with open(
                os.path.join(session_dir, "raw", "events_segment000.jsonl"), "a", encoding="utf-8"
            ) as fh:
                fh.write("{not json\n")
            ok, lines = validate(session_dir)
            self.assertFalse(ok, "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
