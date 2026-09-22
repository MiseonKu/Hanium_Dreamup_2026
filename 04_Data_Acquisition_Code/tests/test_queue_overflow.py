import os
import sys
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

from core.bounded_queue import CRITICAL, DEGRADED, NORMAL, OVERFLOW, WARN, BoundedQueue


class BoundedQueueOverflowTests(unittest.TestCase):
    def test_state_transitions_as_queue_fills(self):
        q = BoundedQueue(expected_rate_hz=100, buffer_seconds=1.0)  # maxsize=100
        self.assertEqual(q.maxsize, 100)
        self.assertEqual(q.depth_state(), NORMAL)

        for i in range(59):
            self.assertTrue(q.put_nowait_tracked(i, seq=i))
        self.assertEqual(q.depth_state(), NORMAL)  # 59%

        q.put_nowait_tracked(59, seq=59)
        self.assertEqual(q.depth_state(), WARN)  # 60%

        for i in range(60, 79):
            q.put_nowait_tracked(i, seq=i)
        self.assertEqual(q.depth_state(), WARN)  # 79%

        q.put_nowait_tracked(79, seq=79)
        self.assertEqual(q.depth_state(), DEGRADED)  # 80%

        for i in range(80, 94):
            q.put_nowait_tracked(i, seq=i)
        self.assertEqual(q.depth_state(), DEGRADED)  # 94%

        q.put_nowait_tracked(94, seq=94)
        self.assertEqual(q.depth_state(), CRITICAL)  # 95%

        for i in range(95, 99):
            q.put_nowait_tracked(i, seq=i)
        self.assertEqual(q.depth_state(), CRITICAL)  # 99%

        q.put_nowait_tracked(99, seq=99)
        self.assertEqual(q.depth_state(), OVERFLOW)  # 100%

    def test_overflow_never_silently_drops(self):
        q = BoundedQueue(expected_rate_hz=10, buffer_seconds=1.0)  # maxsize=10
        for i in range(10):
            self.assertTrue(q.put_nowait_tracked(i, seq=100 + i))
        self.assertEqual(q.depth_state(), OVERFLOW)

        stats_before = q.drop_stats()
        self.assertIsNone(stats_before["first_dropped_seq"])
        self.assertIsNone(stats_before["last_dropped_seq"])
        self.assertEqual(stats_before["drop_count"], 0)

        ok1 = q.put_nowait_tracked("overflow-item-1", seq=200)
        self.assertFalse(ok1)
        ok2 = q.put_nowait_tracked("overflow-item-2", seq=201)
        self.assertFalse(ok2)

        stats = q.drop_stats()
        self.assertEqual(stats["first_dropped_seq"], 200)
        self.assertEqual(stats["last_dropped_seq"], 201)
        self.assertEqual(stats["drop_count"], 2)

        # queue itself still only holds the original 10 items, in order --
        # nothing corrupted by the failed overflow puts.
        self.assertEqual(q.qsize(), 10)
        self.assertEqual(q.get_nowait(), 0)

    def test_maxsize_computed_from_rate_and_buffer_seconds(self):
        q = BoundedQueue(expected_rate_hz=200, buffer_seconds=10)
        self.assertEqual(q.maxsize, 2000)

        q_default_buffer = BoundedQueue(expected_rate_hz=50)
        self.assertEqual(q_default_buffer.maxsize, 500)  # default buffer_seconds=10

    def test_low_rate_never_produces_zero_capacity_queue(self):
        q = BoundedQueue(expected_rate_hz=0.01, buffer_seconds=1.0)
        self.assertGreaterEqual(q.maxsize, 1)


if __name__ == "__main__":
    unittest.main()
