"""python -m unittest discover -s tests   (numpy + pandas only; no TensorFlow needed)"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pbrtqc.data import Preprocessor, episode_days, episode_pairs, inject_bias, load_minutes, split_days
from pbrtqc.detectors import CUSUM, EWMA, MovingAverage, rolling_mean
from pbrtqc.evaluate import Replay, calibrate, detection_delays, evaluate, first_alarm, summarise, sweep
from pbrtqc.lstm_ae import LstmAEDetector, make_windows


class DataTests(unittest.TestCase):
    def test_inject_bias_only_after_onset_and_does_not_mutate(self):
        x = np.full(10, 100.0)
        y = inject_bias(x, 0.1, 6)
        self.assertTrue(np.allclose(y[:6], 100) and np.allclose(y[6:], 110))
        self.assertTrue(np.allclose(x, 100))

    def test_split_is_chronological_and_disjoint(self):
        days = [np.array([i]) for i in range(10)]
        tr, va, te = split_days(days, 0.6, 0.2)
        self.assertEqual((len(tr), len(va), len(te)), (6, 2, 2))
        self.assertEqual([d[0] for d in tr + va + te], list(range(10)))

    def test_episode_days_filters_short_days(self):
        self.assertEqual(len(episode_days([np.zeros(399), np.zeros(400)], 300, 100)), 1)

    def test_preprocessor_truncation_keeps_original_positions(self):
        pre = Preprocessor.fit([np.array([10.0, 20.0, 500.0])], hi=100.0)
        z, keep = pre.transform(np.array([10.0, 500.0, 20.0]))
        self.assertEqual(list(keep), [0, 2])
        self.assertAlmostEqual(pre.mu, 15.0)
        self.assertEqual(len(z), 2)


class DetectorTests(unittest.TestCase):
    def test_ma_matches_pandas_rolling(self):
        z = np.random.default_rng(0).normal(size=200)
        ours = rolling_mean(z, 25)
        ref = pd.Series(z).rolling(25).mean().to_numpy()
        self.assertTrue(np.allclose(ours[24:], ref[24:]) and np.isnan(ours[:24]).all())

    def test_all_detectors_flag_a_step_and_stay_quiet_on_clean(self):
        rng = np.random.default_rng(1)
        clean = rng.normal(size=600)
        shifted = clean.copy()
        shifted[300:] += 1.5
        for det, lim in ((MovingAverage(50), 0.6), (EWMA(0.1), 0.95), (CUSUM(0.5), 8.0)):
            self.assertFalse((det.score(clean) > lim).any(), det.label)
            a = np.flatnonzero(det.score(shifted) > lim)
            self.assertTrue(a.size and a[0] >= 300, det.label)

    def test_first_alarm_maps_through_truncation(self):
        score = np.array([0.0, 5.0, 5.0])
        keep = np.array([0, 4, 7])  # results 1-3 and 5-6 were truncated away
        self.assertEqual(first_alarm(score, keep, 1.0), 4)
        self.assertEqual(first_alarm(score, keep, 1.0, start=5), 7)
        self.assertIsNone(first_alarm(score, keep, 10.0))

    def test_nan_warmup_never_alarms(self):
        s = MovingAverage(50).score(np.full(30, 99.0))
        self.assertTrue(np.isnan(s).all())
        self.assertIsNone(first_alarm(s, np.arange(30), 0.0))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.days = [rng.normal(100, 5, 700) for _ in range(12)]
        self.pre = Preprocessor.fit(self.days[:6])

    def test_detection_delay_is_measured_from_onset_and_calibrated_limit_controls_far(self):
        det = MovingAverage(50)
        lim = calibrate(det, self.pre, self.days[:6], 0.9)
        delays = detection_delays(det, self.pre, lim, self.days[6:], 0.20, 300)
        self.assertTrue(all(d is not None and 0 <= d <= 60 for d in delays))
        rows = evaluate(det, self.pre, lim, self.days[6:], [0.2, -0.2], 300, 100)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["detect_rate"], 1.0)

    def test_properly_scaled_ma_sees_5pct_that_a_3sd_limit_misses(self):
        det = MovingAverage(50)
        lim = calibrate(det, self.pre, self.days[:6], 0.9)
        good = evaluate(det, self.pre, lim, self.days[6:], [0.05], 300, 100)[0]
        paper = evaluate(MovingAverage(50, fixed_limit=3.0), self.pre, 3.0, self.days[6:], [0.05], 300, 100)[0]
        self.assertEqual(good["detect_rate"], 1.0)
        self.assertEqual(paper["detect_rate"], 0.0)


class MinutesAndSweepTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.days = [rng.normal(100, 5, 700) for _ in range(12)]
        self.pre = Preprocessor.fit(self.days[:6])
        self.test = self.days[6:]

    def test_delay_in_minutes_uses_the_recorded_times(self):
        det = MovingAverage(20)
        minutes = [np.arange(700) * 2.0 for _ in self.test]          # a result every 2 minutes
        replay = Replay(det, self.pre, self.test, [0.2], 300)
        row = summarise(det, calibrate(det, self.pre, self.days[:6], 0.9), replay, minutes)[0]
        self.assertAlmostEqual(row["median_delay_min"], 2.0 * row["median_delay"])

    def test_undetected_shifts_have_infinite_minutes_and_no_minutes_column_without_times(self):
        det = MovingAverage(20)
        replay = Replay(det, self.pre, self.test, [0.0], 300)         # no shift: nothing to detect
        row = summarise(det, 99.0, replay, [np.arange(700.0) for _ in self.test])[0]
        self.assertTrue(np.isinf(row["median_delay_min"]) and row["detect_rate"] == 0.0)
        self.assertNotIn("median_delay_min", summarise(det, 99.0, replay, None)[0])

    def test_sweep_trades_false_alarms_for_detection_monotonically(self):
        det = MovingAverage(20)
        replay = Replay(det, self.pre, self.test, [0.02], 300)
        rows = sweep(det, self.pre, self.days[:6], replay, None, [0.5, 0.8, 0.95])
        fa = [r["fa_day_rate"] for r in rows]
        dr = [r["detect_rate"] for r in rows]
        self.assertEqual([r["q"] for r in rows], [0.5, 0.8, 0.95])
        self.assertTrue(fa == sorted(fa, reverse=True) and dr == sorted(dr, reverse=True))

    def test_sweep_skips_fixed_limit_detectors(self):
        det = MovingAverage(50, fixed_limit=3.0)
        replay = Replay(det, self.pre, self.test, [0.05], 300)
        self.assertEqual(sweep(det, self.pre, self.days[:6], replay, None, [0.9]), [])

    def test_load_minutes_parallel_to_days_and_optional(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.csv")
            pd.DataFrame({"Day": [1, 1, 2], "Result": [1.0, 2.0, 3.0], "Minute": [10, 20, 30]}).to_csv(p, index=False)
            self.assertEqual([m.tolist() for m in load_minutes(p)], [[10.0, 20.0], [30.0]])
            pd.DataFrame({"Day": [1], "Result": [1.0]}).to_csv(p, index=False)
            self.assertIsNone(load_minutes(p))

    def test_episode_pairs_keeps_minutes_aligned(self):
        days = [np.zeros(399), np.zeros(450)]
        eps, mins = episode_pairs(days, [np.arange(399.0), np.arange(450.0)], 300, 100)
        self.assertEqual((len(eps), len(mins[0])), (1, 450))
        self.assertEqual(episode_pairs(days, None, 300, 100)[1], None)


class ShiftThenTruncateTests(unittest.TestCase):
    def test_shift_is_applied_before_truncation_so_a_positive_shift_pushes_results_out_of_the_stream(self):
        day = np.full(500, 140.0)                                  # below the 150 cap before any shift
        pre = Preprocessor(mu=140.0, sd=10.0, lo=-np.inf, hi=150.0)
        det = MovingAverage(20)
        up = Replay(det, pre, [day], [0.2], 300).shifted[0.2][0][1]  # keep indices of the shifted replay
        down = Replay(det, pre, [day], [-0.2], 300).shifted[-0.2][0][1]
        self.assertEqual(len(up), 300)       # 168 > 150 after +20%, so results 300.. are dropped
        self.assertEqual(len(down), 500)     # 112 stays inside the cap

    def test_proportional_shift_scales_spread_as_well_as_mean(self):
        x = np.random.default_rng(5).normal(100, 5, 2000)
        y = inject_bias(x, 0.2, 0)
        self.assertAlmostEqual(y.std() / x.std(), 1.2, places=6)
        self.assertAlmostEqual(y.mean() / x.mean(), 1.2, places=6)


class _ZeroModel:
    def predict(self, w, verbose=0):
        return np.zeros_like(w)


class LstmPlumbingTests(unittest.TestCase):
    def test_windows_never_cross_days(self):
        w = make_windows([np.arange(10.0), np.arange(100.0, 110.0)], T=4, ma_window=0)
        self.assertEqual(w.shape, (14, 4, 1))
        self.assertFalse(((w[:, :, 0] < 50).any(axis=1) & (w[:, :, 0] > 50).any(axis=1)).any())

    def test_scorer_is_causal_and_aligned(self):
        z = np.arange(20.0)
        det = LstmAEDetector(_ZeroModel(), T=5, ma_window=0, scale=1.0, label="x")
        s = det.score(z)
        self.assertTrue(np.isnan(s[:4]).all())
        self.assertAlmostEqual(s[4], np.mean(z[0:5] ** 2))   # window ending at t=4
        self.assertAlmostEqual(s[19], np.mean(z[15:20] ** 2))
        z2 = z.copy()
        z2[19] = 1e6                                          # future change must not move past scores
        self.assertTrue(np.allclose(det.score(z2)[:19], s[:19], equal_nan=True))

    def test_ma_warmup_offsets_first_score(self):
        det = LstmAEDetector(_ZeroModel(), T=5, ma_window=3, scale=1.0, label="x")
        s = det.score(np.ones(20))
        self.assertEqual(int(np.isnan(s).sum()), (3 - 1) + (5 - 1))


if __name__ == "__main__":
    unittest.main()
