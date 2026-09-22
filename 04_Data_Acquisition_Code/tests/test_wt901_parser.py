"""Tests for `collectors/collect_imu_rs232.py`'s WT901C protocol layer
(`extract_frames`/`decode_frame`/`s16`), reimplemented from
`07_WT901C/WT901C_imu0/collect_4imu.py` (commit
`3a0c757e913454ade5007cbacbb52bae48f321a0`; see
`docs/07_WT901C_CODE_AUDIT.md`). That source file has zero tests for this
parser -- the audit's #1 flagged gap for it -- these are the first.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import time
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)
sys.path.insert(0, os.path.join(WORKSPACE, "mocks"))

from collectors.collect_imu_rs232 import (  # noqa: E402
    FRAME_ACCEL,
    FRAME_ANGLE,
    FRAME_GYRO,
    FRAME_MAGNETIC,
    FRAME_PORT_STATUS,
    FRAME_QUATERNION,
    FRAME_TIME,
    HEADER,
    IMURS232Collector,
    decode_frame,
    extract_frames,
)
from mock_imu import MockWT901CImu  # noqa: E402
from fault_injector import with_duplicate_packets, with_random_byte_noise  # noqa: E402


def _checksummed_frame(frame_type: int, payload8: bytes) -> bytes:
    body = bytes((HEADER, frame_type)) + payload8
    return body + bytes((sum(body) & 0xFF,))


class ExtractFramesTests(unittest.TestCase):
    def test_valid_frame_extracted_cleanly(self):
        frame = _checksummed_frame(FRAME_ACCEL, bytes(range(8)))
        buf = bytearray(frame)
        frames, carry = extract_frames(buf)
        self.assertEqual(frames, [(frame, False)])
        self.assertEqual(carry, 0)
        self.assertEqual(len(buf), 0)

    def test_corrupted_checksum_discards_exactly_the_bad_frame_and_resyncs(self):
        # Payload bytes deliberately avoid the value 0x55 so the header
        # hunt skips the WHOLE corrupted frame deterministically, not a
        # coincidental early match on a stray 0x55 inside its payload.
        good_a = _checksummed_frame(FRAME_ACCEL, bytes([1, 2, 3, 4, 5, 6, 7, 8]))
        bad_b = bytearray(_checksummed_frame(FRAME_GYRO, bytes([1, 2, 3, 4, 5, 6, 7, 8])))
        bad_b[-1] ^= 0xFF  # corrupt checksum only -- payload/type/header untouched
        good_c = _checksummed_frame(FRAME_ANGLE, bytes([9, 8, 7, 6, 5, 4, 3, 2]))

        stream = bytearray(good_a + bytes(bad_b) + good_c)
        frames, carry = extract_frames(stream)

        self.assertEqual(len(frames), 2, frames)
        (frame_a, resync_a), (frame_c, resync_c) = frames
        self.assertEqual(frame_a, good_a)
        self.assertFalse(resync_a)
        self.assertEqual(frame_c, good_c)
        self.assertTrue(resync_c, "the frame following a resync must be flagged")
        self.assertEqual(len(stream), 0, "the whole corrupted frame must be discarded")
        self.assertEqual(carry, 0)

    def test_partial_frame_split_across_two_reads_is_buffered(self):
        frame = _checksummed_frame(FRAME_QUATERNION, bytes([10, 20, 30, 40, 50, 60, 70, 80]))
        buf = bytearray()
        buf.extend(frame[:5])  # first "read()" -- incomplete frame
        frames, carry = extract_frames(buf)
        self.assertEqual(frames, [])
        self.assertEqual(bytes(buf), frame[:5], "partial bytes must stay buffered untouched")

        buf.extend(frame[5:])  # second "read()" completes it
        frames, carry = extract_frames(buf, carry)
        self.assertEqual(frames, [(frame, False)])
        self.assertEqual(len(buf), 0)

    def test_random_byte_noise_between_frames_is_resynced_past(self):
        imu = with_random_byte_noise(MockWT901CImu(seed=3), every_n=3, num_bytes=4, seed=9)
        buf = bytearray()
        carry = 0
        seen_resync = False
        decoded_count = 0
        for _ in range(40):
            buf.extend(imu.next_packet_bytes())
            frames, carry = extract_frames(buf, carry)
            for frame_bytes, resynced in frames:
                decoded_count += 1
                seen_resync = seen_resync or resynced
                decode_frame(frame_bytes)  # must not raise
        self.assertTrue(seen_resync, "expected at least one resync from injected noise")
        self.assertGreater(decoded_count, 30, "noise must not have swallowed most real frames")


class DecodeFrameTests(unittest.TestCase):
    def test_accel_frame_decodes_to_mps2_near_1g(self):
        imu = MockWT901CImu(seed=1, moving=False)
        pkt = imu.next_packet_bytes()
        while pkt[1] != FRAME_ACCEL:
            pkt = imu.next_packet_bytes()
        decoded = decode_frame(pkt)
        self.assertIn("accel_mps2", decoded)
        self.assertEqual(len(decoded["accel_mps2"]), 3)
        self.assertAlmostEqual(decoded["accel_mps2"][2], 9.80665, delta=0.5)

    def test_quaternion_reordered_from_wire_wxyz_to_schema_xyzw(self):
        raw = [16384, 0, 0, 0]  # wire order (w,x,y,z): w=0.5, rest 0
        payload = b"".join(struct.pack("<h", v) for v in raw)
        body = bytes((HEADER, FRAME_QUATERNION)) + payload
        frame = body + bytes((sum(body) & 0xFF,))
        decoded = decode_frame(frame)
        qx, qy, qz, qw = decoded["quaternion_xyzw"]
        self.assertAlmostEqual(qw, 0.5, places=3)
        self.assertAlmostEqual(qx, 0.0, places=3)
        self.assertAlmostEqual(qy, 0.0, places=3)
        self.assertAlmostEqual(qz, 0.0, places=3)

    def test_magnetic_frame_decodes_raw_counts_only_no_invented_scale(self):
        # mx,my,mz raw counts + word4 -- see decode_frame's FRAME_MAGNETIC
        # branch for why word4 is NOT decoded as temperature_c here (real
        # hardware evidence in docs/IMU_DATA_COVERAGE.md).
        payload = struct.pack("<hhhh", 1500, -900, 12800, 4321)
        body = bytes((HEADER, FRAME_MAGNETIC)) + payload
        frame = body + bytes((sum(body) & 0xFF,))
        decoded = decode_frame(frame)
        self.assertEqual(decoded["mag_raw"], [1500, -900, 12800])
        self.assertEqual(decoded["mag_packet_word4_raw"], 4321)
        self.assertNotIn("mag_uT", decoded, "no protocol-confirmed count->uT scale factor -- must not invent one")
        self.assertNotIn("temperature_c", decoded, "magnetic word4 must not be asserted as temperature")

    def test_angle_frame_word4_is_raw_not_mislabeled_temperature(self):
        # 195.82C investigation (docs/IMU_DATA_COVERAGE.md): the angle
        # packet's word4 is a firmware/version number (WitMotion protocol:
        # Version=(VH<<8)|VL), not temperature -- confirmed constant across
        # 4 independent physical sensors in real captured data. Regression
        # test: decode_frame must NOT populate temperature_c for FRAME_ANGLE,
        # and must preserve the raw int16 word honestly.
        payload = struct.pack("<hhhh", 100, 200, 300, 19582)
        body = bytes((HEADER, FRAME_ANGLE)) + payload
        frame = body + bytes((sum(body) & 0xFF,))
        decoded = decode_frame(frame)
        self.assertEqual(decoded["angle_packet_word4_raw"], 19582)
        self.assertNotIn("temperature_c", decoded, "angle packet must not claim a temperature reading")
        self.assertIn("euler_rad", decoded)

    def test_time_frame_decodes_calendar_bytes_not_int16_pairs(self):
        # WitMotion time packet layout: YY MM DD HH MN SS MSL MSH (raw
        # bytes, not signed int16 axis pairs like every other frame type).
        payload = bytes((26, 8, 4, 14, 3, 29, 0xE7, 0x03))  # 2026-08-04 14:03:29.999
        body = bytes((HEADER, FRAME_TIME)) + payload
        frame = body + bytes((sum(body) & 0xFF,))
        decoded = decode_frame(frame)
        self.assertEqual(decoded["t_sensor"], "2026-08-04T14:03:29.999")

    def test_port_status_frame_header_and_type_byte_both_0x55_decode_correctly(self):
        # 0x55 doubles as both the frame header AND the port_status frame
        # type code (docs/07_WT901C_CODE_AUDIT.md's flagged protocol
        # ambiguity). A genuine port_status frame is `0x55 0x55 <payload>
        # <checksum>` -- extract_frames must not choke on/misinterpret the
        # repeated byte, and decode_frame must label it "port_status", not
        # "unknown".
        frame = _checksummed_frame(FRAME_PORT_STATUS, bytes([1, 2, 3, 4, 5, 6, 7, 8]))
        self.assertEqual(frame[0], 0x55)
        self.assertEqual(frame[1], 0x55)
        buf = bytearray(frame)
        frames, carry = extract_frames(buf)
        self.assertEqual(frames, [(frame, False)], "the double-0x55 frame must be extracted as exactly one frame")
        self.assertEqual(len(buf), 0)
        decoded = decode_frame(frame)
        self.assertEqual(decoded["packet_type"], "port_status")
        self.assertEqual(decoded["frame_type"], 0x55)

    def test_garbage_0x55_byte_does_not_masquerade_as_port_status_frame(self):
        # A lone stray 0x55 byte (not a real frame start) followed by a
        # genuine accel frame must resync past the stray byte rather than
        # emitting a bogus 11-byte "frame" built from noise.
        stray = bytes([0x55, 0x01, 0x02])  # too short to even reach checksum check
        good = _checksummed_frame(FRAME_ACCEL, bytes([9, 8, 7, 6, 5, 4, 3, 2]))
        buf = bytearray(stray + good)
        frames, carry = extract_frames(buf)
        self.assertEqual(frames, [(good, True)], "stray byte must be discarded, only the real frame emitted")


class ReaderAgainstMockTests(unittest.TestCase):
    def test_short_synthetic_session_yields_valid_imu_records(self):
        imu = MockWT901CImu(seed=5, output_rate_hz=100.0)
        with tempfile.TemporaryDirectory() as tmp:
            collector = IMURS232Collector(
                sensor_id="imu3_tip",
                session_id="session_parser_test",
                segment_id=0,
                output_dir=tmp,
                byte_source=imu.next_packet_bytes,
                queue_expected_rate_hz=400.0,
            )
            collector.start()
            try:
                deadline = time.time() + 5.0
                while collector.frames_emitted < 40 and time.time() < deadline:
                    time.sleep(0.02)
            finally:
                collector.stop()

            self.assertGreaterEqual(collector.frames_emitted, 40, "collector.stats(): " + str(collector.stats()))

            with open(collector.writer.final_path, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh]
            self.assertGreater(len(records), 0)
            for rec in records:
                self.assertTrue(rec["checksum_ok"])
                self.assertTrue(rec["valid"])
                self.assertEqual(rec["parse_status"], "ok")
                self.assertEqual(rec["schema"], "alphaf.imu.raw.v1")
                self.assertEqual(rec["sensor_id"], "imu3_tip")
                self.assertEqual(rec["connection_epoch"], 1)
            host_seqs = [rec["host_seq"] for rec in records]
            self.assertEqual(host_seqs, list(range(1, len(records) + 1)), "host_seq must be gap-free on a clean stream")


class DuplicatePacketTests(unittest.TestCase):
    """`mocks/fault_injector.py:with_duplicate_packets` producing duplicate
    wire bytes was already covered at the mock level (`tests/test_mocks.py`)
    but never end-to-end through `IMURS232Collector` -- the `duplicate` flag
    it's actually meant to feed (`_handle_frame`'s exact-byte-repeat check,
    `collectors/collect_imu_rs232.py`) had no collector-level regression
    test. `out_of_order` is intentionally NOT tested here: the collector's
    own docstring (`_handle_frame`, `out_of_order=False` comment) explains
    the WT901C wire format carries no in-frame sequence, so there is no
    independent ordering signal this collector could use -- it is always
    `False` by design, not a detector that could be exercised."""

    def test_duplicate_wire_bytes_flag_records_as_duplicate_end_to_end(self):
        imu = with_duplicate_packets(MockWT901CImu(seed=7, output_rate_hz=100.0), rate=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            collector = IMURS232Collector(
                sensor_id="imu3_tip",
                session_id="session_duplicate_test",
                segment_id=0,
                output_dir=tmp,
                byte_source=imu.next_packet_bytes,
                queue_expected_rate_hz=400.0,
            )
            collector.start()
            try:
                deadline = time.time() + 5.0
                while collector.frames_emitted < 30 and time.time() < deadline:
                    time.sleep(0.02)
            finally:
                collector.stop()

            with open(collector.writer.final_path, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh]
            self.assertGreater(len(records), 0)
            self.assertTrue(
                any(rec["duplicate"] for rec in records),
                "with_duplicate_packets(rate=1.0) must produce at least one duplicate-flagged record",
            )
            self.assertFalse(
                records[0]["duplicate"], "the very first record has no predecessor to duplicate"
            )
            # out_of_order is structurally always False for this collector
            # (see class docstring) -- assert that documented invariant.
            self.assertFalse(any(rec["out_of_order"] for rec in records))


if __name__ == "__main__":
    unittest.main()
