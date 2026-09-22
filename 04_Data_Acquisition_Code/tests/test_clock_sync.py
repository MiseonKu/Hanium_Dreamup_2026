import os
import random
import sys
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

from core.clock_sync import ClockSyncEstimator, SyncSample


class ClockSyncTests(unittest.TestCase):
    def test_recovers_known_offset_and_drift_with_noise(self):
        rng = random.Random(1234)
        true_a, true_b = 2.0, 1000.0
        rtt = 2000  # ns, fixed round trip for every sample

        estimator = ClockSyncEstimator()
        for seq, x in enumerate(range(0, 10000, 100)):
            noise = rng.uniform(-5.0, 5.0)
            target_mid = true_a * x + true_b + noise
            # integer arithmetic only, so rtt comes out exactly `rtt` for
            # every sample regardless of where target_mid rounds.
            t_pc_send = round(target_mid) - rtt // 2
            t_pc_receive = t_pc_send + rtt
            estimator.add_sample(
                SyncSample(
                    seq=seq,
                    t_pc_send_ns=t_pc_send,
                    t_esp_receive_us=x,
                    t_esp_send_us=x,
                    t_pc_receive_ns=t_pc_receive,
                )
            )

        report = estimator.report()
        self.assertEqual(report["sample_count"], 100)
        self.assertAlmostEqual(report["clock_drift"], true_a, delta=0.01)
        self.assertAlmostEqual(report["clock_offset"], true_b, delta=10.0)
        self.assertGreaterEqual(report["clock_sync_rmse"], 0.0)
        self.assertLess(report["clock_sync_rmse"], 10.0)
        self.assertGreaterEqual(report["clock_sync_p95"], 0.0)
        self.assertGreaterEqual(report["clock_sync_p95"], report["clock_sync_rmse"] * 0.5)
        self.assertAlmostEqual(report["rtt_mean_ns"], float(rtt), delta=0.001)
        self.assertAlmostEqual(report["rtt_p95_ns"], float(rtt), delta=0.001)

    def test_insufficient_samples_returns_none_fit_not_a_fabricated_number(self):
        estimator = ClockSyncEstimator()
        estimator.add_sample(
            SyncSample(seq=0, t_pc_send_ns=0, t_esp_receive_us=0, t_esp_send_us=0, t_pc_receive_ns=100)
        )
        report = estimator.report()
        self.assertIsNone(report["clock_drift"])
        self.assertIsNone(report["clock_offset"])
        self.assertIsNone(report["clock_sync_rmse"])
        self.assertIsNone(report["clock_sync_p95"])
        self.assertEqual(report["sample_count"], 1)
        # rtt stats don't need a fit -- available even from one sample
        self.assertEqual(report["rtt_mean_ns"], 100.0)

    def test_zero_samples_reports_none_everywhere(self):
        report = ClockSyncEstimator().report()
        self.assertEqual(report["sample_count"], 0)
        self.assertIsNone(report["clock_drift"])
        self.assertIsNone(report["rtt_mean_ns"])

    def test_report_never_mutates_the_input_samples(self):
        s0 = SyncSample(seq=0, t_pc_send_ns=0, t_esp_receive_us=10, t_esp_send_us=10, t_pc_receive_ns=200)
        s1 = SyncSample(seq=1, t_pc_send_ns=100, t_esp_receive_us=20, t_esp_send_us=20, t_pc_receive_ns=300)
        estimator = ClockSyncEstimator()
        estimator.add_sample(s0)
        estimator.add_sample(s1)
        estimator.report()
        # SyncSample is a frozen dataclass; any attempted mutation would
        # raise, so simply re-checking the original values is sufficient
        # to prove report() didn't (and can't) rewrite them.
        self.assertEqual(s0.t_pc_send_ns, 0)
        self.assertEqual(s1.t_pc_receive_ns, 300)
        with self.assertRaises(Exception):
            s0.t_pc_send_ns = 999  # frozen -- must reject mutation


if __name__ == "__main__":
    unittest.main()
