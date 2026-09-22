"""End-to-end smoke test: run a very short mock-mode session and confirm a
session_*/ directory + manifest.json + at least the events/health streams
land on disk (master spec section 7/8/16 wired together).

Skips (does not fail the suite) if `experiment.session_controller` itself
genuinely isn't importable -- but tries first. Individual sensor streams
(imu*/encoder/vive_tracker) are NOT asserted here: those collector modules
are concurrent work this test doesn't want to be coupled to landing in any
particular order; the manifest's `collector_status` records what actually
started for manual/other-tool inspection.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

try:
    from experiment.session_controller import SessionController

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only hit if a hard dependency is missing
    SessionController = None
    _IMPORT_ERROR = exc

# Never actually let a test reach real hardware: openvr.init() is patched
# out (raises immediately) in every real-mode test below, same rule as the
# rest of this task ("do NOT open any real serial port or touch physical
# hardware") -- SessionController's real-mode Vive path always attempts
# construction regardless of what a given test is focused on, since it's
# the same shared _start_all_collectors code path.
_FAKE_OPENVR_INIT_ERROR = RuntimeError("openvr.init unavailable in test environment (patched out intentionally)")


@unittest.skipIf(SessionController is None, f"experiment.session_controller not importable yet: {_IMPORT_ERROR}")
class SessionControllerSmokeTest(unittest.TestCase):
    def test_short_mock_session_produces_manifest_and_core_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = SessionController("mock", sessions_root=tmp)
            controller.start()
            controller.run_for(2.0)

            self.assertTrue(os.path.isdir(controller.session_dir))
            manifest_path = os.path.join(controller.session_dir, "manifest.json")
            self.assertTrue(os.path.isfile(manifest_path))

            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)

            for field in (
                "session_id",
                "start_utc",
                "execution_mode",
                "git_commit_hash",
                "data_schema_version",
                "session_complete",
                "interrupted",
            ):
                self.assertIn(field, manifest)

            self.assertEqual(manifest["execution_mode"], "mock")
            self.assertTrue(manifest["session_complete"])
            self.assertFalse(manifest["interrupted"])

            raw_dir = os.path.join(controller.session_dir, "raw")
            jsonl_files = glob.glob(os.path.join(raw_dir, "*.jsonl"))
            stream_names = {os.path.basename(f) for f in jsonl_files}

            self.assertTrue(any(name.startswith("events_segment") for name in stream_names), stream_names)
            self.assertTrue(any(name.startswith("health_segment") for name in stream_names), stream_names)

            # every *.jsonl written must be valid line-delimited JSON
            for path in jsonl_files:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            json.loads(line)

    def test_inventory_mode_completes_without_starting_collectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = SessionController("inventory", sessions_root=tmp)
            controller.start()  # inventory mode shuts itself down inside start()

            manifest = json.loads(controller.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["session_complete"])
            self.assertEqual(manifest["execution_mode"], "inventory")
            report_path = os.path.join(controller.session_dir, "reports", "port_inventory.json")
            self.assertTrue(os.path.isfile(report_path))

    def test_session_summary_json_has_expected_shape(self):
        """Task 3: session_summary.json sibling to manifest.json, built
        from what each collector already exposes."""
        with tempfile.TemporaryDirectory() as tmp:
            controller = SessionController("mock", sessions_root=tmp)
            controller.start()
            controller.run_for(1.5)

            summary_path = os.path.join(controller.session_dir, "session_summary.json")
            self.assertTrue(os.path.isfile(summary_path))
            with open(summary_path, encoding="utf-8") as fh:
                summary = json.load(fh)

            for field in ("session_id", "start_utc", "end_utc", "session_duration_s", "streams"):
                self.assertIn(field, summary)

            self.assertIn("vive_tracker", summary["streams"])
            vive_stats = summary["streams"]["vive_tracker"]
            for field in (
                "record_count",
                "valid_count",
                "first_t_pc_rx_ns",
                "last_t_pc_rx_ns",
                "packet_gap_count",
                "actual_rate_hz",
                "writer_pending_at_shutdown",
            ):
                self.assertIn(field, vive_stats)
            self.assertGreater(vive_stats["record_count"], 0)
            self.assertIsNotNone(vive_stats["actual_rate_hz"])

            self.assertIn("imu3_tip", summary["streams"])
            imu_stats = summary["streams"]["imu3_tip"]
            self.assertGreater(imu_stats["record_count"], 0)
            self.assertIsNone(imu_stats["valid_count"], "not exposed by IMURS232Collector.stats() -- must be null, not fabricated")

    def test_start_barrier_releases_all_pending_launches_together(self):
        """Task 5: _release_start_barrier is the whole mechanism (a real
        collector.start()/thread.start() is just one more zero-arg
        callable to it) -- test it directly rather than depending on
        collector timing."""
        with tempfile.TemporaryDirectory() as tmp:
            controller = SessionController("mock", sessions_root=tmp)
            call_times: list[float] = []
            lock = threading.Lock()

            def make_launch():
                def _launch():
                    with lock:
                        call_times.append(time.monotonic())

                return _launch

            n = 6
            controller._pending_launches = [make_launch() for _ in range(n)]
            controller._release_start_barrier()

            deadline = time.monotonic() + 5.0
            while len(call_times) < n and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertEqual(len(call_times), n)
            self.assertLess(max(call_times) - min(call_times), 1.0, "barrier should release all parties together")
            self.assertEqual(controller._pending_launches, [])

    def test_start_barrier_is_a_no_op_with_nothing_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = SessionController("mock", sessions_root=tmp)
            controller._pending_launches = []
            controller._release_start_barrier()  # must return immediately, never raise/hang

    def test_real_mode_imu_matches_sensor_map_and_opens_configured_baud(self):
        """Task 1: real IMU port resolution is entirely sensor_map-driven.
        `serial.Serial` and `core.port_discovery.list_ports` are both
        faked -- this never opens a real serial port."""
        fake_ports = [
            {
                "device": "COM99",
                "vid": 0x1A86,
                "pid": 0x7523,
                "serial_number": "FAKE-IMU0",
                "description": "",
                "manufacturer": "",
                "location": "",
            }
        ]
        opened = []

        class _FakeSerial:
            def __init__(self, port, baud, timeout=None):
                opened.append((port, baud))
                self.in_waiting = 0

            def read(self, n=1):
                time.sleep(0.01)
                return b""

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            sensor_map_path = os.path.join(tmp, "sensor_map.yaml")
            with open(sensor_map_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "sensor_map:\n"
                    "  imu0_base:\n"
                    "    vid: 6790\n"
                    "    pid: 29987\n"
                    "    serial_number: \"FAKE-IMU0\"\n"
                )

            with mock.patch("core.port_discovery.list_ports", return_value=fake_ports), mock.patch(
                "serial.Serial", _FakeSerial
            ), mock.patch("openvr.init", side_effect=_FAKE_OPENVR_INIT_ERROR):
                controller = SessionController(
                    "passive", sessions_root=tmp, sensor_map_path=sensor_map_path, imu_baud=57600
                )
                controller.start()
                deadline = time.monotonic() + 5.0
                while not opened and time.monotonic() < deadline:
                    time.sleep(0.02)
                controller.shutdown(interrupted=True)

            self.assertEqual(opened, [("COM99", 57600)])
            manifest = json.loads(controller.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["collector_status"]["imu0_base"], "started")
            # the other 3 IMUs have no matching entry in this sensor_map --
            # must fail closed, never guess a port.
            self.assertIn("unavailable", manifest["collector_status"]["imu1_link_root"])

    def test_real_mode_imu_without_sensor_map_stays_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("openvr.init", side_effect=_FAKE_OPENVR_INIT_ERROR):
                controller = SessionController("passive", sessions_root=tmp)
                controller.start()
                controller.shutdown(interrupted=True)

            manifest = json.loads(controller.manifest_path.read_text(encoding="utf-8"))
            status = manifest["collector_status"]["imu0_base"]
            self.assertIn("unavailable", status)
            self.assertIn("no sensor_map configured", status)

    def test_real_mode_encoder_without_factory_or_sensor_map_stays_unavailable(self):
        """No encoder_collector_factory override AND no sensor_map ->
        fails closed (never a guessed COM port), same message shape as
        the IMU real-mode gate."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("openvr.init", side_effect=_FAKE_OPENVR_INIT_ERROR):
                controller = SessionController("passive", sessions_root=tmp)
                controller.start()
                controller.shutdown(interrupted=True)

            manifest = json.loads(controller.manifest_path.read_text(encoding="utf-8"))
            status = manifest["collector_status"]["encoder"]
            self.assertIn("unavailable", status)
            self.assertIn("no sensor_map configured", status)

    def test_real_mode_encoder_defaults_to_esp32_state_poll_collector(self):
        """Without an encoder_collector_factory override, real mode wires
        the actual real-hardware collector (Esp32StatePollCollector) by
        default, gated on sensor_map -- same pattern as IMU real mode.
        `serial.Serial` is faked; this never opens a real port."""
        fake_ports = [
            {
                "device": "COM77",
                "vid": 0x10C4,
                "pid": 0xEA60,
                "serial_number": "FAKE-ESP32",
                "description": "",
                "manufacturer": "",
                "location": "",
            }
        ]
        opened = []

        class _FakeSerial:
            def __init__(self, port, baud, timeout=None):
                opened.append((port, baud))

            def write(self, data):
                pass

            def read(self, n=1):
                return b""  # unparsable -> a valid=False record, still proves the wiring works end to end

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            sensor_map_path = os.path.join(tmp, "sensor_map.yaml")
            with open(sensor_map_path, "w", encoding="utf-8") as fh:
                fh.write("sensor_map:\n  esp32_encoder:\n    vid: 4292\n    pid: 60000\n    serial_number: \"FAKE-ESP32\"\n")

            with mock.patch("core.port_discovery.list_ports", return_value=fake_ports), mock.patch(
                "serial.Serial", _FakeSerial
            ), mock.patch("openvr.init", side_effect=_FAKE_OPENVR_INIT_ERROR):
                controller = SessionController("passive", sessions_root=tmp, sensor_map_path=sensor_map_path)
                controller.start()
                controller.run_for(0.5)

            self.assertEqual(opened, [("COM77", 115200)])
            manifest = json.loads(controller.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["collector_status"]["encoder"], "started")
            enc_files = glob.glob(os.path.join(controller.session_dir, "raw", "encoder_segment*.jsonl"))
            self.assertTrue(enc_files)
            with open(enc_files[0], encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]
            self.assertGreater(len(lines), 0)
            self.assertEqual(lines[0]["stream"], "encoder")

    def test_real_mode_encoder_pluggable_factory_gets_wired(self):
        """Task 1's integration point: any factory returning a pull-style
        `.read_record() -> dict` object gets wired the same way vive_tracker
        already is, with zero changes to session_controller.py."""
        from core import schemas

        class _FakeRealEncoderCollector:
            def __init__(self):
                self.n = 0

            def read_record(self):
                self.n += 1
                return {"valid": True, "t_pc_rx_ns": schemas.monotonic_ns(), "stream": "encoder", "n": self.n}

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("openvr.init", side_effect=_FAKE_OPENVR_INIT_ERROR):
                controller = SessionController(
                    "passive",
                    sessions_root=tmp,
                    encoder_collector_factory=lambda ctrl: _FakeRealEncoderCollector(),
                )
                controller.start()
                controller.run_for(0.5)

            manifest = json.loads(controller.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["collector_status"]["encoder"], "started")
            enc_files = glob.glob(os.path.join(controller.session_dir, "raw", "encoder_segment*.jsonl"))
            self.assertTrue(enc_files, "encoder stream file must exist")
            with open(enc_files[0], encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]
            self.assertGreater(len(lines), 0)
            self.assertEqual(lines[0]["stream"], "encoder")


if __name__ == "__main__":
    unittest.main()
