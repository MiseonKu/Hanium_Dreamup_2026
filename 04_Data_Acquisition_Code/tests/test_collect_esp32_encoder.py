"""Parse-correctness tests for `collectors/collect_esp32_encoder.py` against
`mocks/mock_esp32.py`'s `STREAM` lines.

Run with:
    python -m pytest tests -q
from the `04_Data_Acquisition_Code/` directory.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

from collectors.collect_esp32_encoder import (  # noqa: E402
    JOINT_NAMES,
    Esp32EncoderCollector,
    parse_stream_line,
)
from mocks.mock_esp32 import MockESP32Encoder  # noqa: E402


class ParseStreamLineTests(unittest.TestCase):
    def test_five_joint_values_land_in_correct_order(self):
        enc = MockESP32Encoder(period_ms=50, seed=1)
        target_deg = [10.0, 20.0, 30.0, 40.0, 50.0]
        enc.set_target(target_deg)
        line = enc.next_line()

        parsed = parse_stream_line(line)
        self.assertIsNotNone(parsed)
        self.assertEqual(len(JOINT_NAMES), 5)
        self.assertEqual(len(parsed["pos"]), 5)
        self.assertEqual(len(parsed["deg"]), 5)
        self.assertEqual(len(parsed["target"]), 5)
        # target= is commanded degrees directly -- exact match, positional.
        for i, joint in enumerate(JOINT_NAMES):
            self.assertAlmostEqual(parsed["target"][i], target_deg[i], places=6, msg=joint)

    def test_unparsable_line_returns_none(self):
        self.assertIsNone(parse_stream_line("not a stream line at all"))
        self.assertIsNone(parse_stream_line(""))

    def test_t_ms_and_seq_and_flags_extracted(self):
        enc = MockESP32Encoder(period_ms=50, seed=1)
        enc.set_target([0, 0, 0, 0, 0])
        line = enc.next_line()
        parsed = parse_stream_line(line)
        self.assertEqual(parsed["t_ms"], 50)
        self.assertEqual(parsed["seq"], 1)
        self.assertTrue(parsed["armed"])
        self.assertFalse(parsed["cal"])
        self.assertFalse(parsed["fault"])
        self.assertEqual(parsed["fault_axis"], -1)
        self.assertEqual(parsed["enabled_mask"], 0x1F)


class Esp32EncoderCollectorRecordTests(unittest.TestCase):
    def test_read_record_maps_5_positions_to_5_named_joints_in_order(self):
        enc = MockESP32Encoder(period_ms=100, move_rate_deg_s=1000.0, seed=1)
        target_deg = [80.0, 35.0, 77.0, 155.0, 50.0]
        enc.set_target(target_deg)
        # let the simulated encoder fully converge onto target before reading
        for _ in range(10):
            enc.next_line()
        line = enc.next_line()
        parsed = parse_stream_line(line)

        # feed that exact line through a collector wired to a one-shot
        # source, so read_record() parses precisely this captured line
        rec = Esp32EncoderCollector(_OneShotSource(line), session_id="session_test").read_record()

        self.assertTrue(rec["valid"])
        self.assertEqual(rec["parse_status"], "ok")
        self.assertEqual(rec["schema"], "alphaf.encoder.raw.v1")
        self.assertEqual(rec["stream"], "encoder")
        self.assertEqual(rec["raw_line"], line)
        self.assertEqual(rec["t_esp_us"], parsed["t_ms"] * 1000)
        self.assertEqual(rec["source_seq"], parsed["seq"])

        q_encoder_rad = rec["data"]["q_encoder_rad"]
        q_cmd_rad = rec["data"]["q_cmd_rad"]
        self.assertEqual(len(q_encoder_rad), 5)
        self.assertEqual(len(q_cmd_rad), 5)
        for i, joint in enumerate(JOINT_NAMES):
            self.assertAlmostEqual(q_cmd_rad[i], math.radians(target_deg[i]), places=3, msg=joint)
            self.assertAlmostEqual(q_encoder_rad[i], math.radians(target_deg[i]), places=1, msg=joint)

        self.assertEqual(rec["data"]["encoder_raw_tick"], parsed["pos"])

    def test_limits_not_ready_sentinel_is_null_in_BOTH_deg_and_target(self):
        """-999.00 is 05's "limits not ready" sentinel. It must become None
        in q_cmd_rad exactly as it already does in q_encoder_rad --
        radians(-999) = -17.4 rad would look like a real commanded pose
        (core/schemas.py: missing field = null, never a number)."""
        line = (
            "STREAM t_ms=1 seq=1 fresh=1 armed=1 cal=0 fault=0 fault_axis=-1 "
            "mode=PID enabled=0x1F pos=1,1,1,1,1 upd=1,1,1,1,1 "
            "deg=-999.00,10.00,10.00,10.00,10.00 "
            "target=-999.00,20.00,20.00,20.00,20.00"
        )
        rec = Esp32EncoderCollector(_OneShotSource(line), session_id="s").read_record()

        self.assertIsNone(rec["data"]["q_encoder_rad"][0])
        self.assertIsNone(rec["data"]["q_cmd_rad"][0], "target= sentinel must be null, not radians(-999)")
        self.assertAlmostEqual(rec["data"]["q_cmd_rad"][1], math.radians(20.0))
        self.assertIn("joint_limits_not_ready", rec["quality_flags"])

    def test_target_only_sentinel_still_flags_limits_not_ready(self):
        """A sentinel on target= alone (deg= fine) must still be flagged --
        previously only deg= was inspected."""
        line = (
            "STREAM t_ms=1 seq=1 fresh=1 armed=1 cal=0 fault=0 fault_axis=-1 "
            "mode=PID enabled=0x1F pos=1,1,1,1,1 upd=1,1,1,1,1 "
            "deg=10.00,10.00,10.00,10.00,10.00 "
            "target=-999.00,20.00,20.00,20.00,20.00"
        )
        rec = Esp32EncoderCollector(_OneShotSource(line), session_id="s").read_record()
        self.assertIsNone(rec["data"]["q_cmd_rad"][0])
        self.assertIn("joint_limits_not_ready", rec["quality_flags"])

    def test_unparsable_line_still_produces_a_record_not_a_skip(self):
        collector = Esp32EncoderCollector(_OneShotSource("garbage line"), session_id="s")
        rec = collector.read_record()
        self.assertFalse(rec["valid"])
        self.assertEqual(rec["parse_status"], "parse_error")
        self.assertEqual(rec["raw_line"], "garbage line")
        self.assertIsNone(rec["data"]["q_encoder_rad"])

    def test_start_never_sends_a_motion_or_calibration_command(self):
        sent = []

        class _RecordingConn:
            def next_line(self):
                return "STREAM t_ms=1 seq=1 fresh=1 armed=1 cal=0 fault=0 fault_axis=-1 mode=PID enabled=0x1F pos=1,1,1,1,1 upd=1,1,1,1,1 deg=0.00,0.00,0.00,0.00,0.00 target=0.00,0.00,0.00,0.00,0.00"

            def write(self, data):
                sent.append(data)

        collector = Esp32EncoderCollector(_RecordingConn(), session_id="s", period_ms=100)
        collector.start()
        self.assertEqual(sent, [b"STREAM ON 100\n"])
        forbidden = (b"TORQUE", b"INIT", b"MOVE_RAW", b"CAL_AUTO", b"ARM", b"SET ", b"FRAME")
        for chunk in sent:
            for word in forbidden:
                self.assertNotIn(word, chunk)


class _OneShotSource:
    """Minimal `next_line`-style source that returns `line` exactly once."""

    def __init__(self, line: str):
        self._line = line
        self._used = False

    def next_line(self) -> str:
        if self._used:
            raise StopIteration("only one line configured")
        self._used = True
        return self._line


if __name__ == "__main__":
    unittest.main()
