"""Parse-correctness tests for `collectors/collect_esp32_state_poll.py`,
the collector for the firmware actually on the board
(`03_SO-101_control/main`, `STATE` polling).

Sample lines are transcribed from that firmware's own `printJointDetailLine`
(main.cpp:1400) / `printFaultSummaryIfLatched` (main.cpp:1378) format
strings, and cross-checked against the real `raw_line` values recorded in
`sessions/real_motion_demo/encoder_segment*.jsonl`.

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

from collectors.collect_esp32_state_poll import (  # noqa: E402
    JOINT_NAMES,
    Esp32StatePollCollector,
    parse_state_text,
)

PAN_LINE = (
    "axis=shoulder_pan alias=D2 id=1 ping=true torque=true raw_pos=2225 "
    "raw_speed=0 raw_load=12 raw_voltage=118 raw_temp=34 raw_current=3 "
    "logical_deg=80.0 mapping_ready=true raw_cal_min=1054 raw_cal_max=3189 "
    "raw_active_min=1054 raw_active_max=3189 speed=120 acc=10 "
    "range_percent=100 direction=normal last_update_ms=12344 fault=false "
    "read_failures=0 overload_samples=0"
)
LIFT_LINE = (
    "axis=shoulder_lift alias=D3 id=2 ping=true torque=true raw_pos=2100 "
    "raw_speed=-5 raw_load=880 raw_voltage=117 raw_temp=41 raw_current=90 "
    "logical_deg=35.5 mapping_ready=true raw_cal_min=1092 raw_cal_max=3296 "
    "raw_active_min=1092 raw_active_max=3296 speed=100 acc=10 "
    "range_percent=100 direction=normal last_update_ms=12344 fault=false "
    "read_failures=0 overload_samples=0"
)
WRIST_LINE = (
    "axis=wrist_flex alias=D5 id=4 ping=true torque=true raw_pos=2900 "
    "raw_speed=0 raw_load=40 raw_voltage=118 raw_temp=33 raw_current=5 "
    "logical_deg=90.0 mapping_ready=true raw_cal_min=1825 raw_cal_max=4080 "
    "raw_active_min=1825 raw_active_max=4080 speed=90 acc=10 "
    "range_percent=100 direction=normal last_update_ms=12344 fault=false "
    "read_failures=0 overload_samples=0"
)
FULL_STATE = "\n".join([PAN_LINE, LIFT_LINE, WRIST_LINE]) + "\n"


class _FakeConn:
    """Minimal `serial.Serial` stand-in: records writes, replays a fixed
    reply (or a queue of replies) on read."""

    def __init__(self, *replies: bytes):
        self.replies = list(replies) or [b""]
        self.written: list = []

    def write(self, data):
        self.written.append(data)

    def read(self, n=1):
        return self.replies[0] if len(self.replies) == 1 else self.replies.pop(0)


class ParseStateTextTests(unittest.TestCase):
    def test_all_three_axes_parsed(self):
        parsed = parse_state_text(FULL_STATE)
        self.assertEqual(set(parsed["axes"]), set(JOINT_NAMES))
        self.assertIsNone(parsed["fault"])

    def test_numeric_and_boolean_fields_typed(self):
        pan = parse_state_text(FULL_STATE)["axes"]["shoulder_pan"]
        self.assertEqual(pan["raw_pos"], 2225)
        self.assertEqual(pan["id"], 1)
        self.assertIs(pan["torque"], True)
        self.assertIs(pan["fault"], False)
        self.assertEqual(pan["logical_deg"], 80.0)
        self.assertEqual(pan["direction"], "normal")

    def test_negative_raw_speed_parsed(self):
        lift = parse_state_text(FULL_STATE)["axes"]["shoulder_lift"]
        self.assertEqual(lift["raw_speed"], -5)

    def test_logical_deg_na_becomes_none_not_zero(self):
        """Firmware emits the literal "NA" when raw->logical mapping isn't
        ready (main.cpp:1366). Parsing that to 0.0 would look like a real
        commanded-to-zero pose."""
        text = PAN_LINE.replace("logical_deg=80.0", "logical_deg=NA")
        pan = parse_state_text(text)["axes"]["shoulder_pan"]
        self.assertIsNone(pan["logical_deg"])
        self.assertNotEqual(pan["logical_deg"], 0.0)

    def test_fault_summary_line_parsed(self):
        text = FULL_STATE + "fault_latched=true axis=shoulder_lift reason=overload latched_ms=98765\n"
        fault = parse_state_text(text)["fault"]
        self.assertIsNotNone(fault)
        self.assertIs(fault["latched"], True)
        self.assertEqual(fault["axis"], "shoulder_lift")
        self.assertEqual(fault["reason"], "overload")
        self.assertEqual(fault["latched_ms"], 98765)

    def test_unknown_extra_field_kept_not_dropped(self):
        """A firmware that starts reporting a new field must show up in the
        data, and must not knock out the rest of the line."""
        text = PAN_LINE + " newfield=7"
        pan = parse_state_text(text)["axes"]["shoulder_pan"]
        self.assertEqual(pan["newfield"], "7")
        self.assertEqual(pan["raw_pos"], 2225)

    def test_noise_lines_ignored(self):
        text = "[LEGACY] queued axis=shoulder_pan logical=100\n" + FULL_STATE + "garbage\n"
        self.assertEqual(set(parse_state_text(text)["axes"]), set(JOINT_NAMES))


class PollOnceTests(unittest.TestCase):
    def test_record_shape_and_ordering(self):
        c = Esp32StatePollCollector(_FakeConn(FULL_STATE.encode()), "sess", 3)
        rec = c.poll_once()
        self.assertTrue(rec["valid"])
        self.assertEqual(rec["parse_status"], "ok")
        self.assertEqual(rec["segment_id"], 3)
        self.assertEqual(rec["data"]["encoder_raw_tick"], [2225, 2100, 2900])
        self.assertAlmostEqual(rec["data"]["q_encoder_rad"][0], math.radians(80.0))
        self.assertEqual(rec["data"]["motor_status"], ["enabled"] * 3)
        self.assertEqual(rec["data"]["fault"], ["ok"] * 3)

    def test_raw_logical_deg_preserved_alongside_the_legacy_radian_field(self):
        """`logical_deg` is a firmware LABEL, not a physical angle (the
        firmware maps each axis's raw span onto an arbitrary label
        interval; physical deg per logical deg is 1.173/1.300/1.101 for
        pan/lift/wrist). `q_encoder_rad` = radians(logical_deg) is kept
        only for compatibility with already-collected sessions, so the
        untransformed firmware numbers must also be recorded -- otherwise
        downstream code has no way back to a physical angle."""
        c = Esp32StatePollCollector(_FakeConn(FULL_STATE.encode()), "sess", 0)
        rec = c.poll_once()
        self.assertEqual(rec["data"]["logical_deg"], [80.0, 35.5, 90.0])
        # raw encoder ticks stay the authoritative physical quantity
        self.assertEqual(rec["data"]["encoder_raw_tick"], [2225, 2100, 2900])

    def test_commanded_logical_deg_recorded_untransformed(self):
        conn = _FakeConn(
            b"[LEGACY] queued axis=shoulder_pan logical=100\n",
            FULL_STATE.encode(),
        )
        c = Esp32StatePollCollector(conn, "sess", 0)
        c.send_axis_command(0, 100)
        rec = c.poll_once()
        self.assertEqual(rec["data"]["q_cmd_logical_deg"], [100, None, None])

    def test_na_logical_deg_stays_null_in_the_raw_field_too(self):
        text = FULL_STATE.replace("logical_deg=80.0", "logical_deg=NA")
        c = Esp32StatePollCollector(_FakeConn(text.encode()), "sess", 0)
        rec = c.poll_once()
        self.assertIsNone(rec["data"]["logical_deg"][0])
        self.assertIsNone(rec["data"]["q_encoder_rad"][0])
        self.assertIn("mapping_not_ready", rec["quality_flags"])

    def test_target_context_threaded_through(self):
        """Every stream must be joinable on target_id."""
        c = Esp32StatePollCollector(_FakeConn(FULL_STATE.encode()), "sess", 0)
        rec = c.poll_once(target_id="T007", planned_order_index=7, workspace_mode="front_120")
        self.assertEqual(rec["target_id"], "T007")
        self.assertEqual(rec["planned_order_index"], 7)
        self.assertEqual(rec["workspace_mode"], "front_120")

    def test_missing_axis_is_none_not_zero(self):
        c = Esp32StatePollCollector(_FakeConn((PAN_LINE + "\n").encode()), "sess", 0)
        rec = c.poll_once()
        self.assertEqual(rec["data"]["encoder_raw_tick"], [2225, None, None])
        self.assertEqual(rec["data"]["motor_status"][1], "missing")

    def test_unparsable_response_still_produces_a_flagged_record(self):
        c = Esp32StatePollCollector(_FakeConn(b"total garbage\n"), "sess", 0)
        rec = c.poll_once()
        self.assertFalse(rec["valid"])
        self.assertEqual(rec["parse_status"], "parse_error")
        self.assertEqual(rec["raw_line"], "total garbage")

    def test_fault_latched_flagged_in_quality_flags(self):
        text = FULL_STATE + "fault_latched=true axis=shoulder_lift reason=overload latched_ms=1\n"
        c = Esp32StatePollCollector(_FakeConn(text.encode()), "sess", 0)
        rec = c.poll_once()
        self.assertIn("fault_latched", rec["quality_flags"])
        self.assertEqual(rec["data"]["esp32_state"], "fault")
        self.assertEqual(rec["data"]["fault_detail"]["reason"], "overload")

    def test_raw_telemetry_preserved_whole(self):
        """Fields with no dedicated schema slot must survive, not be dropped."""
        c = Esp32StatePollCollector(_FakeConn(FULL_STATE.encode()), "sess", 0)
        rec = c.poll_once()
        pan = rec["data"]["raw_telemetry"][0]
        self.assertEqual(pan["raw_voltage"], 118)
        self.assertEqual(pan["raw_current"], 3)
        self.assertEqual(pan["last_update_ms"], 12344)
        self.assertEqual(pan["read_failures"], 0)
        self.assertEqual(pan["overload_samples"], 0)
        self.assertEqual(pan["raw_cal_min"], 1054)

    def test_first_velocity_is_none_then_computed(self):
        """Unknown and not-moving must stay distinguishable."""
        c = Esp32StatePollCollector(_FakeConn(FULL_STATE.encode()), "sess", 0)
        first = c.poll_once()
        self.assertTrue(all(v is None for v in first["data"]["estimated_velocity_rad_s"]))
        second = c.poll_once()
        self.assertIsNotNone(second["data"]["estimated_velocity_rad_s"][0])

    def test_host_seq_increments(self):
        c = Esp32StatePollCollector(_FakeConn(FULL_STATE.encode()), "sess", 0)
        self.assertEqual(c.poll_once()["host_seq"], 1)
        self.assertEqual(c.poll_once()["host_seq"], 2)


class SendAxisCommandTests(unittest.TestCase):
    def test_wire_format_and_ack(self):
        conn = _FakeConn(b"[LEGACY] queued axis=shoulder_pan logical=100\n")
        c = Esp32StatePollCollector(conn, "sess", 0)
        rec = c.send_axis_command(0, 100.4)
        self.assertEqual(conn.written, [b"D2100\r\n"])
        self.assertTrue(rec["data"]["payload"]["command_ack"])
        self.assertEqual(rec["parse_status"], "ok")
        self.assertEqual(rec["command_type"], "legacy_axis_move")

    def test_one_axis_per_call(self):
        """Batching axes into one RegWriteAction is exactly what overheats
        the shared rail -- see docs/ST3215_THERMAL_AND_SEQUENCING.md."""
        conn = _FakeConn(b"[LEGACY] queued axis=wrist_flex logical=90\n")
        Esp32StatePollCollector(conn, "sess", 0).send_axis_command(2, 90)
        self.assertEqual(len(conn.written), 1)
        self.assertEqual(conn.written[0], b"D5090\r\n")

    def test_missing_ack_is_flagged_not_assumed(self):
        conn = _FakeConn(b"[LEGACY] axis=shoulder_pan id=1 is offline; skipped\n")
        rec = Esp32StatePollCollector(conn, "sess", 0).send_axis_command(0, 100)
        self.assertFalse(rec["data"]["payload"]["command_ack"])
        self.assertEqual(rec["parse_status"], "no_ack")
        self.assertIn("command_not_acked", rec["quality_flags"])

    def test_ack_for_a_different_axis_does_not_count(self):
        conn = _FakeConn(b"[LEGACY] queued axis=wrist_flex logical=90\n")
        rec = Esp32StatePollCollector(conn, "sess", 0).send_axis_command(0, 100)
        self.assertFalse(rec["data"]["payload"]["command_ack"])

    def test_command_timestamps_bracket_the_write(self):
        conn = _FakeConn(b"[LEGACY] queued axis=shoulder_pan logical=100\n")
        rec = Esp32StatePollCollector(conn, "sess", 0).send_axis_command(0, 100)
        p = rec["data"]["payload"]
        self.assertLessEqual(p["t_command_pc_ns"], p["t_ack_pc_ns"])
        self.assertGreaterEqual(p["ack_latency_ms"], 0.0)

    def test_command_is_recorded_on_the_next_encoder_record(self):
        """Commanded vs. reached must be joinable within one record."""
        conn = _FakeConn(
            b"[LEGACY] queued axis=shoulder_pan logical=100\n",
            FULL_STATE.encode(),
        )
        c = Esp32StatePollCollector(conn, "sess", 0)
        c.send_axis_command(0, 100)
        rec = c.poll_once()
        self.assertAlmostEqual(rec["data"]["q_cmd_rad"][0], math.radians(100))
        self.assertIsNone(rec["data"]["q_cmd_rad"][1], "uncommanded axis stays None")


class SendAxisRawTests(unittest.TestCase):
    """`MOVE_RAW` is the primitive real runs use -- it needs no INIT and so
    cannot trigger INIT's torque-before-goal lunge
    (docs/FIRMWARE_MOTION_HAZARDS.md)."""

    ECHO = b"[MOVE_RAW] axis=shoulder_lift id=2 raw=2500 speed=100 acc=10\n"

    def test_wire_format_and_ack(self):
        conn = _FakeConn(self.ECHO)
        rec = Esp32StatePollCollector(conn, "sess", 0).send_axis_raw(1, 2500)
        self.assertEqual(conn.written, [b"MOVE_RAW shoulder_lift 2500\r\n"])
        self.assertTrue(rec["data"]["payload"]["command_ack"])
        self.assertEqual(rec["command_type"], "move_raw")
        self.assertEqual(rec["data"]["payload"]["raw_tick"], 2500)

    def test_no_legacy_d_command_is_emitted(self):
        """A `D<axis><deg>` write here would silently do nothing without
        INIT, and INIT is the unsafe path we avoid."""
        conn = _FakeConn(self.ECHO)
        Esp32StatePollCollector(conn, "sess", 0).send_axis_raw(1, 2500)
        for w in conn.written:
            self.assertNotRegex(w.decode(), r"^D\d")

    def test_one_axis_per_call(self):
        conn = _FakeConn(self.ECHO)
        Esp32StatePollCollector(conn, "sess", 0).send_axis_raw(1, 2500)
        self.assertEqual(len(conn.written), 1)

    def test_float_tick_is_rounded(self):
        conn = _FakeConn(self.ECHO)
        Esp32StatePollCollector(conn, "sess", 0).send_axis_raw(1, 2499.7)
        self.assertEqual(conn.written, [b"MOVE_RAW shoulder_lift 2500\r\n"])

    def test_write_failure_echo_is_not_counted_as_ack(self):
        """main.cpp:1848 prints "[MOVE_RAW] write failed axis=..." on refusal.
        Matching that as success would report a move that never happened."""
        conn = _FakeConn(b"[MOVE_RAW] write failed axis=shoulder_lift id=2 raw=2500\n")
        rec = Esp32StatePollCollector(conn, "sess", 0).send_axis_raw(1, 2500)
        self.assertFalse(rec["data"]["payload"]["command_ack"])
        self.assertEqual(rec["parse_status"], "no_ack")
        self.assertIn("command_not_acked", rec["quality_flags"])

    def test_echo_for_a_different_axis_does_not_count(self):
        conn = _FakeConn(b"[MOVE_RAW] axis=wrist_flex id=4 raw=2000 speed=90 acc=10\n")
        rec = Esp32StatePollCollector(conn, "sess", 0).send_axis_raw(1, 2500)
        self.assertFalse(rec["data"]["payload"]["command_ack"])

    def test_timestamps_bracket_the_write(self):
        conn = _FakeConn(self.ECHO)
        p = Esp32StatePollCollector(conn, "sess", 0).send_axis_raw(1, 2500)["data"]["payload"]
        self.assertLessEqual(p["t_command_pc_ns"], p["t_ack_pc_ns"])
        self.assertGreaterEqual(p["ack_latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
