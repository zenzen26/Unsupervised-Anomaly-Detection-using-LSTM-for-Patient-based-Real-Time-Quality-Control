"""Streaming detectors. Each maps a standardised stream z -> a score array.

score[t] uses only z[:t+1] (causal) and is NaN during warm-up. An alarm is
score > limit. Limits are calibrated by evaluate.calibrate (same false-alarm
budget for every detector), unless a detector carries a `fixed_limit`.
"""
import numpy as np


def rolling_mean(z, n):
    """Trailing mean of window n; first n-1 entries NaN."""
    z = np.asarray(z, float)
    out = np.full(len(z), np.nan)
    if len(z) >= n:
        c = np.cumsum(np.insert(z, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


class Detector:
    label = "detector"
    fixed_limit = None

    def score(self, z):
        raise NotImplementedError


class MovingAverage(Detector):
    def __init__(self, n, fixed_limit=None, label=None):
        self.n = n
        self.fixed_limit = fixed_limit
        self.label = label or f"MA(n={n})"

    def score(self, z):
        return np.abs(rolling_mean(z, self.n))


class EWMA(Detector):
    def __init__(self, lam):
        self.lam = lam
        self.label = f"EWMA(lam={lam})"

    def score(self, z):
        out = np.empty(len(z))
        e = 0.0
        for i, v in enumerate(z):
            e = self.lam * v + (1.0 - self.lam) * e
            out[i] = abs(e)
        return out


class CUSUM(Detector):
    def __init__(self, k):
        self.k = k
        self.label = f"CUSUM(k={k})"

    def score(self, z):
        out = np.empty(len(z))
        sp = sm = 0.0
        for i, v in enumerate(z):
            sp = max(0.0, sp + v - self.k)
            sm = max(0.0, sm - v - self.k)
            out[i] = max(sp, sm)
        return out
