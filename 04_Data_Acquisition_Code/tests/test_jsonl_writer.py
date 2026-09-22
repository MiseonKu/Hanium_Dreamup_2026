import json
import os
import sys
import tempfile
import threading
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

from core.jsonl_writer import JsonlWriter


class JsonlWriterTests(unittest.TestCase):
    def test_clean_close_renames_partial_to_final_in_order(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = JsonlWriter(folder, "imu3_tip", 0, flush_interval_s=0.05, fsync_interval_s=0.1)
            for i in range(20):
                writer.put({"host_seq": i, "value": i * 2})
            writer.close(clean=True)

            self.assertTrue(os.path.exists(writer.final_path))
            self.assertFalse(os.path.exists(writer.partial_path))
            self.assertTrue(writer.final_path.endswith("imu3_tip_segment000.jsonl"))

            with open(writer.final_path, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh]
            self.assertEqual(len(records), 20)
            self.assertEqual([rec["host_seq"] for rec in records], list(range(20)))

    def test_interrupted_close_leaves_partial_file(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = JsonlWriter(folder, "imu3_tip", 1, flush_interval_s=0.05, fsync_interval_s=0.1)
            writer.put({"host_seq": 0})
            writer.close_interrupted()

            self.assertTrue(os.path.exists(writer.partial_path))
            self.assertFalse(os.path.exists(writer.final_path))

            # data already put() before the interrupt is still durable
            with open(writer.partial_path, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh]
            self.assertEqual(records, [{"host_seq": 0}])

    def test_close_with_clean_false_param_also_leaves_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = JsonlWriter(folder, "imu3_tip", 2, flush_interval_s=0.05, fsync_interval_s=0.1)
            writer.put({"host_seq": 0})
            writer.close(clean=False)
            self.assertTrue(os.path.exists(writer.partial_path))
            self.assertFalse(os.path.exists(writer.final_path))

    def test_concurrent_producers_all_land_in_file(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = JsonlWriter(folder, "encoder", 0, flush_interval_s=0.05, fsync_interval_s=0.1)

            def produce(producer_id):
                for i in range(50):
                    writer.put({"producer": producer_id, "i": i})

            workers = [threading.Thread(target=produce, args=(pid,)) for pid in range(4)]
            for w in workers:
                w.start()
            for w in workers:
                w.join()
            writer.close(clean=True)

            with open(writer.final_path, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh]
            self.assertEqual(len(records), 200)
            # every (producer, i) pair present exactly once
            seen = {(r["producer"], r["i"]) for r in records}
            expected = {(pid, i) for pid in range(4) for i in range(50)}
            self.assertEqual(seen, expected)

    def test_start_new_segment_bumps_id_and_does_not_touch_old_file(self):
        with tempfile.TemporaryDirectory() as folder:
            writer0 = JsonlWriter(folder, "vive_tracker", 0, flush_interval_s=0.05, fsync_interval_s=0.1)
            writer0.put({"host_seq": 0})
            writer1 = writer0.start_new_segment()
            try:
                writer1.put({"host_seq": 1})
                self.assertTrue(os.path.exists(writer0.final_path))
                self.assertEqual(writer1.segment_id, 1)
                self.assertTrue(writer1.final_path.endswith("vive_tracker_segment001.jsonl"))
            finally:
                writer1.close(clean=True)

    def test_cannot_reuse_a_segment_id_whose_partial_file_still_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = JsonlWriter(folder, "imu3_tip", 5, flush_interval_s=0.05, fsync_interval_s=0.1)
            writer.put({"host_seq": 0})
            writer.close_interrupted()
            # a crashed segment's .partial must not be silently resumed
            with self.assertRaises(Exception):
                JsonlWriter(folder, "imu3_tip", 5, flush_interval_s=0.05, fsync_interval_s=0.1)

    def test_fsync_interval_must_not_be_shorter_than_flush_interval(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                JsonlWriter(folder, "imu3_tip", 0, flush_interval_s=0.5, fsync_interval_s=0.1)


if __name__ == "__main__":
    unittest.main()
