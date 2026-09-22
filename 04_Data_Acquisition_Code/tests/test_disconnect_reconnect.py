"""Disconnect/reconnect behavior for the serial-backed collectors.

Covers `collectors/collect_esp32_encoder.py` against
`mocks/fault_injector.py:with_disconnect` wrapping `mocks/mock_esp32.py`,
and `collectors/collect_imu_rs232.py` (thread-based reader/writer, see that
module's docstring) against the same fault injector wrapping
`mocks/mock_imu.py`. Add further sensors' disconnect/reconnect coverage
here too rather than creating a parallel file.

Run with:
    python -m pytest tests -q
from the `04_Data_Acquisition_Code/` directory.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

from collectors.collect_esp32_encoder import Esp32EncoderCollector  # noqa: E402
from collectors.collect_imu_rs232 import IMURS232Collector  # noqa: E402
from core.event_bus import EventBus  # noqa: E402
from mocks.fault_injector import with_disconnect  # noqa: E402
from mocks.mock_esp32 import MockESP32Encoder  # noqa: E402
from mocks.mock_imu import MockWT901CImu  # noqa: E402


class Esp32EncoderDisconnectReconnectTests(unittest.TestCase):
    def test_read_record_survives_a_simulated_link_drop_and_resumes(self):
        # after_n=3: first 3 next_line() calls succeed, next 2 raise
        # ConnectionError, then it resumes normally -- see fault_injector.py.
        enc = with_disconnect(MockESP32Encoder(period_ms=50, seed=1), after_n=3, duration_calls=2)
        collector = Esp32EncoderCollector(
            enc, session_id="session_test", segment_id=0, retry_backoff_s=0.001
        )

        records = [collector.read_record() for _ in range(7)]

        # every call returns a real record -- the drop never raises out of
        # read_record(), and no sample is silently skipped.
        self.assertEqual(len(records), 7)
        for rec in records:
            self.assertTrue(rec["valid"])
            self.assertEqual(rec["parse_status"], "ok")

        # host_seq is a clean 1..7 count -- the queue position across the
        # drop is exactly what a producer would see (one record per call).
        self.assertEqual([r["host_seq"] for r in records], list(range(1, 8)))

        # the collector actually noticed and recovered from the drop.
        self.assertGreaterEqual(collector.reconnect_count, 1)

    def test_reconnect_fn_is_used_to_reopen_a_real_connection(self):
        # Simulates the real-hardware path: `conn` itself becomes unusable
        # (raises forever) and a fresh one must be swapped in via
        # `reconnect_fn`, mirroring re-opening a dropped serial.Serial port.
        class _DeadThenAlive:
            def __init__(self):
                self.calls = 0

            def next_line(self):
                self.calls += 1
                raise ConnectionError("port unplugged")

        good_enc = MockESP32Encoder(period_ms=50, seed=2)
        good_enc.set_target([1, 2, 3, 4, 5])

        state = {"reconnected": False}

        def _reconnect():
            state["reconnected"] = True
            return good_enc

        collector = Esp32EncoderCollector(
            _DeadThenAlive(),
            session_id="session_test",
            reconnect_fn=_reconnect,
            retry_backoff_s=0.001,
            max_backoff_s=0.01,
        )
        rec = collector.read_record()

        self.assertTrue(state["reconnected"])
        self.assertTrue(rec["valid"])
        self.assertEqual(collector.reconnect_count, 1)


class ImuDisconnectReconnectTests(unittest.TestCase):
    def test_detects_disconnect_and_reconnects_with_incrementing_epoch(self):
        imu = MockWT901CImu(seed=21, moving=True)
        faulty = with_disconnect(imu, after_n=15, duration_calls=10)

        bus = EventBus()
        disconnect_events = []
        connect_events = []
        bus.subscribe("sensor_disconnected", lambda et, p: disconnect_events.append(p))
        bus.subscribe("sensor_connected", lambda et, p: connect_events.append(p))

        with tempfile.TemporaryDirectory() as tmp:
            collector = IMURS232Collector(
                sensor_id="imu3_tip",
                session_id="session_disconnect_test",
                segment_id=0,
                output_dir=tmp,
                byte_source=faulty.next_packet_bytes,
                reconnect_wait_s=0.2,
                queue_expected_rate_hz=400.0,
                event_bus=bus,
            )
            collector.start()
            try:
                deadline = time.time() + 10.0
                while collector.connection_epoch < 2 and time.time() < deadline:
                    time.sleep(0.05)
                self.assertGreaterEqual(collector.connection_epoch, 2, collector.stats())
                # `duration_calls=10` counts individual next_packet_bytes()
                # calls, not wall-clock time or connection attempts -- each
                # reconnect attempt during the still-down window consumes
                # exactly one of those 10 raising calls and then fails again
                # almost immediately, so the epoch counter typically climbs
                # well past 2 (a burst of short-lived, zero-frame epochs)
                # before the link "recovers" and a connection finally
                # survives long enough to produce frames again. Give it a
                # few seconds to run that churn all the way out.
                settle_deadline = time.time() + 4.0
                while collector.frames_emitted < 60 and time.time() < settle_deadline:
                    time.sleep(0.05)
            finally:
                collector.stop()

            self.assertGreaterEqual(collector.disconnect_count, 1)
            self.assertTrue(disconnect_events, "a disconnect must be published as an event, not silently swallowed")
            self.assertTrue(connect_events)
            self.assertEqual(connect_events[0]["connection_epoch"], 1)
            self.assertGreaterEqual(connect_events[-1]["connection_epoch"], 2)

            with open(collector.writer.final_path, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh]

            epochs_seen = {rec["connection_epoch"] for rec in records}
            self.assertIn(1, epochs_seen)
            # Not necessarily epoch 2 specifically -- see the comment above
            # on why several short-lived epochs can produce zero frames.
            # What matters: at least one post-reconnect epoch produced
            # records, proving recovery actually resumed real data flow.
            self.assertTrue(
                any(epoch > 1 for epoch in epochs_seen),
                f"must have records from after the reconnect too, got epochs {sorted(epochs_seen)}",
            )

            # host_seq must stay strictly increasing across the reconnect
            # boundary: some gap is expected/fine (bytes lost mid-disconnect
            # never become frames), but nothing may repeat or rewind, which
            # would corrupt downstream sequence tracking.
            host_seqs = [rec["host_seq"] for rec in records]
            self.assertEqual(host_seqs, sorted(set(host_seqs)), "host_seq must never repeat or go backwards")
            self.assertEqual(len(host_seqs), len(set(host_seqs)), "host_seq must never duplicate across a reconnect")

    def test_no_reconnect_needed_on_a_clean_stream(self):
        imu = MockWT901CImu(seed=22, moving=False)
        with tempfile.TemporaryDirectory() as tmp:
            collector = IMURS232Collector(
                sensor_id="imu0_base",
                session_id="session_clean_test",
                segment_id=0,
                output_dir=tmp,
                byte_source=imu.next_packet_bytes,
                reconnect_wait_s=0.2,
                queue_expected_rate_hz=400.0,
            )
            collector.start()
            try:
                deadline = time.time() + 5.0
                while collector.frames_emitted < 40 and time.time() < deadline:
                    time.sleep(0.02)
            finally:
                collector.stop()
            self.assertEqual(collector.connection_epoch, 1)
            self.assertEqual(collector.disconnect_count, 0)


if __name__ == "__main__":
    unittest.main()
