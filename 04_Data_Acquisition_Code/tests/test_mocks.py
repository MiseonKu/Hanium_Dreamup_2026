"""Self-check tests for mocks/ (mock_imu, mock_esp32, mock_vive,
fault_injector). Plain stdlib unittest, no fixtures/frameworks.

Run with:
    python -m pytest tests -q
or:
    python tests/test_mocks.py
from the `04_Data_Acquisition_Code/` directory.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mocks"))

from mock_imu import MockWT901CImu  # noqa: E402
from mock_esp32 import MockESP32Encoder, JOINT_NAMES  # noqa: E402
from mock_vive import MockViveTracker  # noqa: E402
from fault_injector import (  # noqa: E402
    with_corrupted_checksum,
    with_random_byte_noise,
    with_duplicate_packets,
    with_reordered_packets,
    with_disconnect,
    with_frozen_values,
)


def _checksum_ok(pkt: bytes) -> bool:
    """Same formula as `07_WT901C/WT901C_imu0/collect_4imu.py:extract_frames`."""
    return len(pkt) == 11 and pkt[0] == 0x55 and (sum(pkt[:10]) & 0xFF) == pkt[10]


class TestMockImu(unittest.TestCase):
    def test_packet_is_valid_wt901c_frame(self):
        imu = MockWT901CImu(seed=1)
        for _ in range(20):
            pkt = imu.next_packet_bytes()
            self.assertTrue(_checksum_ok(pkt), pkt.hex())

    def test_frame_types_rotate_through_all_four(self):
        imu = MockWT901CImu(seed=1)
        seen = {imu.next_packet_bytes()[1] for _ in range(4)}
        self.assertEqual(seen, {0x51, 0x52, 0x53, 0x59})

    def test_sequence_is_external_and_settable(self):
        imu = MockWT901CImu(seed=1)
        imu.next_packet_bytes()
        imu.next_packet_bytes()
        self.assertEqual(imu.sequence, 2)
        imu.sequence = 999  # fault injector must be able to reach in and rewind/set this
        self.assertEqual(imu.sequence, 999)

    def test_settled_vs_moving_produce_different_gyro_magnitude(self):
        import struct

        def gyro_x_rms(imu, n_ticks):
            vals = []
            for _ in range(n_ticks):
                for _ in range(4):  # one full accel/gyro/angle/quat sample tick
                    pkt = imu.next_packet_bytes()
                    if pkt[1] == 0x52:
                        vals.append(struct.unpack("<h", pkt[2:4])[0])
            return (sum(v * v for v in vals) / len(vals)) ** 0.5

        settled_rms = gyro_x_rms(MockWT901CImu(seed=1, moving=False), n_ticks=40)
        moving_rms = gyro_x_rms(
            MockWT901CImu(seed=1, moving=True, motion_amplitude_dps=90.0), n_ticks=40
        )
        # settled = small noise near zero; moving = a real sinusoidal burst.
        # Compare by ratio (not an absolute threshold) so this isn't sensitive
        # to exactly which phase of the sine the sampled window lands on.
        self.assertGreater(moving_rms, settled_rms * 10)


class TestMockEsp32(unittest.TestCase):
    LINE_RE = re.compile(
        r"^STREAM t_ms=(\d+) seq=(\d+) fresh=(\d) armed=(\d) cal=(\d) fault=(\d) "
        r"fault_axis=(-?\d+) mode=(\S+) enabled=(0x[0-9A-Fa-f]+) "
        r"pos=([\-\d,]+) upd=([\d,]+) deg=([\-\d.,]+) target=([\-\d.,]+)$"
    )

    def test_lines_match_documented_stream_format(self):
        enc = MockESP32Encoder(period_ms=50, seed=1)
        enc.set_target([10, 20, 30, 40, 50])
        for _ in range(5):
            line = enc.next_line()
            m = self.LINE_RE.match(line)
            self.assertIsNotNone(m, line)
            self.assertEqual(len(m.group(10).split(",")), len(JOINT_NAMES))
            self.assertEqual(len(m.group(12).split(",")), len(JOINT_NAMES))
            self.assertEqual(len(m.group(13).split(",")), len(JOINT_NAMES))

    def test_encoder_converges_toward_target(self):
        enc = MockESP32Encoder(period_ms=50, move_rate_deg_s=200.0, seed=1)
        enc.set_target([80.0, 35.0, 77.0, 155.0, 50.0])
        line = ""
        for _ in range(50):
            line = enc.next_line()
        deg = [float(v) for v in line.split("deg=")[1].split(" ")[0].split(",")]
        for got, want in zip(deg, enc.q_cmd_deg):
            self.assertAlmostEqual(got, want, places=1)


class TestMockVive(unittest.TestCase):
    def test_valid_sample_has_all_pose_fields(self):
        vt = MockViveTracker(seed=1)
        s = vt.next_sample()
        self.assertTrue(s["tracking_valid"])
        self.assertIsNotNone(s["position_m"])
        self.assertIsNotNone(s["quaternion_xyzw"])
        self.assertIsNotNone(s["pose_matrix_3x4"])
        self.assertEqual(len(s["position_m"]), 3)
        self.assertEqual(len(s["quaternion_xyzw"]), 4)

    def test_tracking_lost_returns_null_fields_not_stale_pose(self):
        vt = MockViveTracker(seed=1)
        first_good = vt.next_sample()["position_m"]
        vt.set_tracking_lost(True)
        lost = vt.next_sample()
        self.assertFalse(lost["tracking_valid"])
        self.assertIsNone(lost["position_m"])
        self.assertIsNone(lost["quaternion_xyzw"])
        self.assertIsNone(lost["pose_matrix_3x4"])
        self.assertNotEqual(first_good, "stale")  # sanity: previous value did exist, just isn't reused

    def test_recovers_with_fresh_pose_not_forward_filled(self):
        vt = MockViveTracker(seed=1)
        vt.set_tracking_lost(True)
        vt.next_sample()
        vt.set_tracking_lost(False)
        recovered = vt.next_sample()
        self.assertTrue(recovered["tracking_valid"])
        self.assertIsNotNone(recovered["position_m"])


class TestFaultInjector(unittest.TestCase):
    def test_corrupted_checksum_actually_breaks_checksum(self):
        imu = with_corrupted_checksum(MockWT901CImu(seed=1), every_n=2)
        results = [imu.next_packet_bytes() for _ in range(10)]
        self.assertTrue(any(not _checksum_ok(p) for p in results))
        self.assertTrue(any(_checksum_ok(p) for p in results))

    def test_random_byte_noise_prepends_junk_requiring_resync(self):
        imu = with_random_byte_noise(MockWT901CImu(seed=1), every_n=2, num_bytes=3, seed=7)
        lengths = [len(imu.next_packet_bytes()) for _ in range(6)]
        self.assertTrue(any(length > 11 for length in lengths))

    def test_duplicate_packets_repeats_identical_output(self):
        imu = with_duplicate_packets(MockWT901CImu(seed=1), rate=1.0)  # always duplicate after first
        first = imu.next_packet_bytes()
        second = imu.next_packet_bytes()
        self.assertEqual(first, second)

    def test_reordered_packets_changes_emission_order(self):
        # The mock's frame type normally cycles accel/gyro/angle/quat in strict
        # order; a reorder buffer mixing packets across that cycle boundary
        # should break that strict cyclic pattern.
        reordered_mock = with_reordered_packets(MockWT901CImu(seed=1), window=5, seed=42)
        types = [reordered_mock.next_packet_bytes()[1] for _ in range(20)]
        cyclic = [(0x51, 0x52, 0x53, 0x59)[i % 4] for i in range(20)]
        self.assertNotEqual(types, cyclic, "reordering should break the normal cyclic frame order")

    def test_disconnect_raises_then_recovers(self):
        imu = with_disconnect(MockWT901CImu(seed=1), after_n=3, duration_calls=2)
        statuses = []
        for _ in range(7):
            try:
                imu.next_packet_bytes()
                statuses.append("ok")
            except ConnectionError:
                statuses.append("down")
        self.assertEqual(statuses, ["ok", "ok", "ok", "down", "down", "ok", "ok"])

    def test_frozen_values_freeze_imu_payload_but_keep_sequence_moving(self):
        imu = with_frozen_values(MockWT901CImu(seed=1, moving=True), after_n=8)
        pkts = [imu.next_packet_bytes() for _ in range(16)]
        # after freeze (index >= 8), same frame type must repeat identical payload bytes
        by_type_after = {}
        for pkt in pkts[8:]:
            ftype = pkt[1]
            if ftype in by_type_after:
                self.assertEqual(pkt, by_type_after[ftype], "frozen payload must repeat verbatim")
            else:
                by_type_after[ftype] = pkt
        # the underlying mock's own sequence counter must still have advanced normally
        self.assertEqual(imu._inner.sequence, 16)

    def test_frozen_values_freeze_esp32_deg_but_keep_seq_advancing(self):
        enc = MockESP32Encoder(period_ms=10, move_rate_deg_s=500.0, seed=1)
        enc.set_target([90, 90, 90, 90, 90])
        frozen = with_frozen_values(enc, after_n=1)
        lines = [frozen.next_line() for _ in range(5)]
        degs = [ln.split("deg=")[1].split(" ")[0] for ln in lines]
        seqs = [int(re.search(r"seq=(\d+)", ln).group(1)) for ln in lines]
        self.assertEqual(len(set(degs[1:])), 1, "deg must be pinned after freeze point")
        self.assertEqual(seqs, list(range(seqs[0], seqs[0] + len(seqs))), "seq must keep advancing while frozen")

    def test_frozen_values_freeze_vive_pose_not_valid_flag(self):
        vt = MockViveTracker(seed=1)
        frozen = with_frozen_values(vt, after_n=1)
        samples = [frozen.next_sample() for _ in range(4)]
        positions = [tuple(s["position_m"]) for s in samples[1:]]
        self.assertEqual(len(set(positions)), 1, "position_m must be pinned after freeze point")
        timestamps = [s["source_timestamp"] for s in samples]
        self.assertEqual(timestamps, sorted(set(timestamps)))  # timestamps still advance


if __name__ == "__main__":
    unittest.main()
