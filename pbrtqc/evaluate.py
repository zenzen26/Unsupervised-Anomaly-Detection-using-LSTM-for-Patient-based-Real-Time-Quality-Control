"""Calibration, parameter selection and episode-based evaluation.

Protocol (the standard PBRTQC simulation):
  * Limits are calibrated on CLEAN days only, from the distribution of daily
    maxima of the score (no anomaly labels used): limit = q-quantile of daily
    extremes, so every detector has the same false-alarm-day budget (~1-q).
  * Detector parameters are picked on validation days, never on the test set.
  * Each test day is replayed twice: clean (-> false-alarm rate) and with a step
    shift injected at `onset` (-> detection rate and delay in results since onset).
  * Delay is counted in original results (truncated results still take time).
"""
import numpy as np

from .data import episode_days, episode_pairs, inject_bias


def _scores(det, pre, x):
    z, keep = pre.transform(x)
    return det.score(z), keep


def first_alarm(score, keep, limit, start=0):
    """Original index of the first alarm at/after `start`, else None. NaN never alarms."""
    hit = keep[np.flatnonzero(score > limit)]
    hit = hit[hit >= start]
    return int(hit[0]) if hit.size else None


def calibrate(det, pre, calib_days, q):
    if det.fixed_limit is not None:
        return float(det.fixed_limit)
    return float(np.quantile(daily_extremes(det, pre, calib_days), q))


def detection_delays(det, pre, limit, days, bias, onset):
    """Delay (results after onset) per episode; None if never alarmed."""
    out = []
    for x in days:
        s, keep = _scores(det, pre, inject_bias(x, bias, onset))
        a = first_alarm(s, keep, limit, start=onset)
        out.append(None if a is None else a - onset)
    return out


def select(candidates, pre, calib_days, val_days, ref_bias, onset, min_post, q):
    """Pick the candidate with the lowest mean delay on validation (undetected = censored)."""
    val = episode_days(val_days, onset, min_post)
    best, best_obj = None, np.inf
    for det in candidates:
        lim = calibrate(det, pre, calib_days, q)
        obj = []
        for b in (ref_bias, -ref_bias):
            for x, d in zip(val, detection_delays(det, pre, lim, val, b, onset)):
                obj.append(len(x) - onset if d is None else d)
        if np.mean(obj) < best_obj:
            best, best_obj = det, float(np.mean(obj))
    return best


def daily_extremes(det, pre, days):
    ext = []
    for x in days:
        s, _ = _scores(det, pre, x)
        s = s[np.isfinite(s)]
        if s.size:
            ext.append(s.max())
    return ext


class Replay:
    """Scores of every clean day and every shifted replay, computed once and reused for any limit."""

    def __init__(self, det, pre, days, biases, onset):
        self.onset = onset
        self.clean = [_scores(det, pre, x) for x in days]
        self.shifted = {b: [_scores(det, pre, inject_bias(x, b, onset)) for x in days] for b in biases}


def summarise(det, limit, replay, minutes, analyte="", extra=None):
    """One row per bias at a fixed limit. Delay is in results and, if minutes are given, in minutes."""
    onset = replay.onset
    fa = float(np.mean([first_alarm(s, k, limit) is not None for s, k in replay.clean]))
    rows = []
    for b, runs in replay.shifted.items():
        res, mins = [], []
        for i, (s, k) in enumerate(runs):
            a = first_alarm(s, k, limit, start=onset)
            res.append(np.inf if a is None else a - onset)
            mins.append(np.inf if a is None or minutes is None else float(minutes[i][a] - minutes[i][onset]))
        res, det_r = np.array(res), np.array([v for v in res if np.isfinite(v)])
        row = dict(analyte=analyte, detector=det.label, bias=b, limit=round(limit, 4), n_days=len(runs),
                   fa_day_rate=round(fa, 3), detect_rate=round(len(det_r) / len(res), 3),
                   detect_within_100=round(float(np.mean(res <= 100)), 3),
                   median_delay=float(np.median(res)),
                   mean_delay_detected=round(float(det_r.mean()), 1) if det_r.size else np.nan)
        if minutes is not None:
            row["median_delay_min"] = float(np.median(mins))
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def sweep(det, pre, calib_days, replay, minutes, qs, analyte=""):
    """Operating curve: the same detector at several false-alarm budgets (limit = q-quantile of daily maxima)."""
    if det.fixed_limit is not None:
        return []
    ext = daily_extremes(det, pre, calib_days)
    rows = []
    for q in qs:
        rows.extend(summarise(det, float(np.quantile(ext, q)), replay, minutes, analyte, extra=dict(q=q)))
    return rows


def evaluate(det, pre, limit, test_days, biases, onset, min_post, analyte="", test_minutes=None):
    """One row per bias; false-alarm day rate comes from the clean replay."""
    days, minutes = episode_pairs(test_days, test_minutes, onset, min_post)
    return summarise(det, limit, Replay(det, pre, days, biases, onset), minutes, analyte)
