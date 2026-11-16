"""Data loading, chronological day-wise splits, bias injection, preprocessing.

Every detector sees exactly the same streams: one stream = one day of results,
processed independently (no state carried across days, so yesterday's biased
tail can never contaminate today's "clean" results).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


def load_days(csv_path):
    """Return a chronological list of 1-D arrays, one per Day (rows keep file order)."""
    df = pd.read_csv(csv_path).dropna(subset=["Result"])
    return [g["Result"].to_numpy(float) for _, g in df.groupby("Day", sort=True)]


def load_minutes(csv_path):
    """Minute-of-day of each result, parallel to load_days (None if the CSV has no Minute column)."""
    df = pd.read_csv(csv_path).dropna(subset=["Result"])
    if "Minute" not in df.columns:
        return None
    return [g["Minute"].to_numpy(float) for _, g in df.groupby("Day", sort=True)]


def split_days(days, train_frac=0.6, val_frac=0.2):
    """Chronological train / validation / test split by whole days."""
    n = len(days)
    a = int(round(n * train_frac))
    b = int(round(n * (train_frac + val_frac)))
    return days[:a], days[a:b], days[b:]


def inject_bias(x, bias, onset):
    """Multiplicative step shift: results from index `onset` onward scaled by (1 + bias)."""
    y = np.array(x, dtype=float)
    y[onset:] *= 1.0 + bias
    return y


def episode_days(days, onset, min_post):
    """Days long enough to hold a shift episode: >= onset + min_post results."""
    return [d for d in days if len(d) >= onset + min_post]


def episode_pairs(days, minutes, onset, min_post):
    """Like episode_days but keeps the parallel minute arrays aligned (minutes may be None)."""
    keep = [i for i, d in enumerate(days) if len(d) >= onset + min_post]
    return [days[i] for i in keep], (None if minutes is None else [minutes[i] for i in keep])


@dataclass
class Preprocessor:
    """Truncation + standardisation, fitted on training days only.

    Truncation drops results outside [lo, hi] from the monitored stream (standard
    PBRTQC practice). `transform` also returns the original positions of the kept
    results so alarms can be mapped back to "number of results since onset".
    """
    mu: float
    sd: float
    lo: float = -np.inf
    hi: float = np.inf

    @classmethod
    def fit(cls, train_days, lo=None, hi=None):
        lo = -np.inf if lo is None else lo
        hi = np.inf if hi is None else hi
        x = np.concatenate(train_days)
        x = x[(x >= lo) & (x <= hi)]
        return cls(mu=float(x.mean()), sd=float(x.std()), lo=lo, hi=hi)

    def transform(self, x):
        x = np.asarray(x, float)
        keep = np.flatnonzero((x >= self.lo) & (x <= self.hi))
        return (x[keep] - self.mu) / self.sd, keep
