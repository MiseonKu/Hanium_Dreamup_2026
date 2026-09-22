import json
import os
import sys
import tempfile
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

from core.recovery import recover_partial_jsonl


class JsonlRecoveryTests(unittest.TestCase):
    def _write_partial(self, folder):
        path = os.path.join(folder, "imu3_tip_segment000.jsonl.partial")
        records = [
            {"host_seq": 0, "v": 0},
            {"host_seq": 1, "v": 1},
            {"host_seq": 5, "v": 5},  # deliberate gap: 1 -> 5
            {"host_seq": 6, "v": 6},
        ]
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
            fh.write('{"host_seq": 7, "v": "trunc')  # deliberately truncated, no closing brace/newline
        return path, records

    def test_recovers_valid_records_and_flags_truncated_tail_and_seq_gap(self):
        with tempfile.TemporaryDirectory() as folder:
            path, records = self._write_partial(folder)
            with open(path, "rb") as fh:
                original_bytes = fh.read()

            result = recover_partial_jsonl(path)

            # original .partial must be untouched
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), original_bytes)

            with open(result.recovered_path, encoding="utf-8") as fh:
                recovered = [json.loads(line) for line in fh]
            self.assertEqual(recovered, records)

            self.assertEqual(result.valid_record_count, 4)
            self.assertEqual(result.total_lines, 5)
            self.assertEqual(len(result.dropped_lines), 1)
            self.assertIn("trunc", result.dropped_lines[0]["raw_text_truncated"])
            self.assertEqual(result.dropped_lines[0]["starting_line_number"], 5)

            self.assertEqual(result.seq_gaps, [[1, 5]])

            with open(result.report_path, encoding="utf-8") as fh:
                report = json.load(fh)
            self.assertEqual(report["valid_records_recovered"], 4)
            self.assertEqual(report["host_seq_gaps"], [[1, 5]])
            self.assertEqual(len(report["corrupt_or_incomplete_lines_dropped"]), 1)

    def test_fully_valid_partial_file_has_no_dropped_lines_or_gaps(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "encoder_segment000.jsonl.partial")
            with open(path, "w", encoding="utf-8") as fh:
                for i in range(5):
                    fh.write(json.dumps({"host_seq": i}) + "\n")

            result = recover_partial_jsonl(path)
            self.assertEqual(result.valid_record_count, 5)
            self.assertEqual(result.dropped_lines, [])
            self.assertEqual(result.seq_gaps, [])

    def test_truncated_text_is_actually_truncated_for_a_long_fragment(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "vive_tracker_segment000.jsonl.partial")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"host_seq": 0}) + "\n")
                fh.write("{" + "x" * 1000)  # long garbage tail, no closing

            result = recover_partial_jsonl(path)
            frag = result.dropped_lines[0]
            self.assertTrue(frag["truncated"])
            self.assertLessEqual(len(frag["raw_text_truncated"]), 500)


if __name__ == "__main__":
    unittest.main()
