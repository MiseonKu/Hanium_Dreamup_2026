import json
import os
import random
import sys
import tempfile
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

from tools.audit_sampling import (
    compute_packet_type_rates,
    compute_sampling_stats,
    uart_budget_report,
    uart_byte_capacity_per_sec,
)


class SamplingAuditTests(unittest.TestCase):
    def test_uniform_rate_matches_configured_within_tolerance(self):
        rate_hz = 100.0
        period_ns = int(1e9 / rate_hz)
        n = 500
        ts = [i * period_ns for i in range(n)]

        stats = compute_sampling_stats(ts, configured_rate_hz=rate_hz)

        self.assertEqual(stats["sample_count"], n)
        self.assertAlmostEqual(stats["measured_rate_hz"], rate_hz, delta=0.5)
        self.assertAlmostEqual(stats["median_period_ms"], period_ns / 1e6, places=6)
        self.assertLess(stats["jitter_p95_ms"], 0.01)  # perfectly uniform spacing -> ~0 jitter
        self.assertAlmostEqual(stats["max_gap_ms"], period_ns / 1e6, places=6)
        self.assertAlmostEqual(stats["packet_loss_ratio"], 0.0, delta=1e-6)

    def test_known_jitter_distribution_p95_within_expected_bound(self):
        rng = random.Random(42)
        rate_hz = 200.0
        period_ns = int(1e9 / rate_hz)
        jitter_ns = 500_000  # +-0.5ms uniform jitter around the nominal period
        n = 2000

        t = 0
        ts = [0]
        for _ in range(n - 1):
            t += period_ns + rng.randint(-jitter_ns, jitter_ns)
            ts.append(t)

        stats = compute_sampling_stats(ts, configured_rate_hz=rate_hz)

        # |uniform(-0.5, 0.5)ms| has support [0, 0.5]ms -- loose bound, not
        # a tight statistical claim, just "jitter was actually measured".
        self.assertGreater(stats["jitter_p95_ms"], 0.0)
        self.assertLess(stats["jitter_p95_ms"], 1.0)
        self.assertAlmostEqual(stats["measured_rate_hz"], rate_hz, delta=2.0)

    def test_injected_large_gap_is_detected_as_packet_loss(self):
        rate_hz = 100.0
        period_ns = int(1e9 / rate_hz)
        n = 300
        gap_index = 150
        gap_ns = period_ns * 50  # a 50-sample-equivalent gap simulating dropped packets

        ts = [i * period_ns for i in range(n)]
        ts = ts[:gap_index] + [t + gap_ns for t in ts[gap_index:]]

        stats = compute_sampling_stats(ts, configured_rate_hz=rate_hz)

        self.assertEqual(stats["sample_count"], n)
        self.assertGreater(stats["max_gap_ms"], (gap_ns * 0.9) / 1e6)
        self.assertEqual(stats["max_gap_index"], gap_index)
        self.assertLess(stats["measured_rate_hz"], rate_hz)  # the gap drags the overall average down
        self.assertGreater(stats["packet_loss_ratio"], 0.05)  # meaningfully nonzero

    def test_fewer_than_two_samples_returns_none_stats_not_a_crash(self):
        empty = compute_sampling_stats([])
        self.assertEqual(empty["sample_count"], 0)
        self.assertIsNone(empty["measured_rate_hz"])

        single = compute_sampling_stats([123])
        self.assertEqual(single["sample_count"], 1)
        self.assertIsNone(single["measured_rate_hz"])
        self.assertIsNone(single["max_gap_ms"])

    def test_unsorted_input_is_handled_defensively(self):
        rate_hz = 50.0
        period_ns = int(1e9 / rate_hz)
        n = 100
        ts = [i * period_ns for i in range(n)]
        shuffled = list(ts)
        random.Random(1).shuffle(shuffled)

        stats_sorted = compute_sampling_stats(ts, configured_rate_hz=rate_hz)
        stats_shuffled = compute_sampling_stats(shuffled, configured_rate_hz=rate_hz)
        self.assertEqual(stats_sorted, stats_shuffled)


class UartBudgetTests(unittest.TestCase):
    def test_uart_byte_capacity_9600_8n1(self):
        # 9600 baud / 10 bits-per-byte (1 start + 8 data + 1 stop) = 960 B/s.
        self.assertAlmostEqual(uart_byte_capacity_per_sec(9600), 960.0)

    def test_compute_packet_type_rates_on_synthetic_imu_file(self):
        # 4 packet types, 10 records each, evenly spread over 1 second.
        period_ns = int(1e9 / 40)
        packet_types = ["acceleration", "angular_velocity", "angle", "magnetic_field"]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "imu0_base_segment000.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                for i in range(40):
                    rec = {"t_pc_rx_ns": i * period_ns, "packet_type": packet_types[i % 4]}
                    fh.write(json.dumps(rec) + "\n")

            rates = compute_packet_type_rates(path)
            self.assertEqual(rates["packet_type_counts"], {pt: 10 for pt in packet_types})
            self.assertEqual(rates["combined_record_count"], 40)
            # compute_packet_type_rates uses count/span_s (not (n-1)/span_s
            # like compute_sampling_stats) -- at only 40 samples that's a
            # ~2.5% high bias vs. the nominal 40Hz; negligible at the
            # thousands-of-samples scale of the real IMU files.
            self.assertAlmostEqual(rates["combined_rate_hz"], 40.0, delta=2.0)
            for pt in packet_types:
                self.assertAlmostEqual(rates["packet_type_rate_hz"][pt], 10.0, delta=0.5)

            # At 9600 baud (960 B/s) this combined rate uses well under the
            # raw byte budget (40 * 11 = 440 B/s of 960 B/s available).
            budget = uart_budget_report(9600, rates)
            self.assertAlmostEqual(budget["required_bytes_per_sec"], 440.0, delta=20.0)
            self.assertLess(budget["utilization_ratio"], 1.0)
            self.assertAlmostEqual(budget["max_per_packet_type_rate_hz_at_capacity"], 960 / 11 / 4, places=3)


if __name__ == "__main__":
    unittest.main()
