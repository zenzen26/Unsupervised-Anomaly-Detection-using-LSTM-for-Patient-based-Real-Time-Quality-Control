"""One-command PBRTQC experiment: python run_experiments.py --analyte sodium [--lstm]

Runs every detector on identical day-wise streams with limits calibrated on clean
data and parameters chosen on validation days. Writes results/<analyte>.csv and
results/<analyte>_config.json. Classical detectors need only numpy + pandas;
--lstm additionally needs tensorflow (train on your GPU machine / Colab).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from pbrtqc.data import Preprocessor, episode_pairs, load_days, load_minutes, split_days
from pbrtqc.detectors import CUSUM, EWMA, MovingAverage
from pbrtqc.evaluate import Replay, calibrate, select, summarise, sweep

TRUNCATION = {"sodium": (None, None), "creatinine": (None, 150.0)}  # (lo, hi) in raw units

FAMILIES = {
    "MA": [MovingAverage(n) for n in (20, 50, 100)],
    "EWMA": [EWMA(l) for l in (0.02, 0.05, 0.1, 0.2)],
    "CUSUM": [CUSUM(k) for k in (0.25, 0.5)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyte", required=True, choices=sorted(TRUNCATION))
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--biases", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05, 0.10, 0.20])
    ap.add_argument("--q", type=float, default=0.9, help="daily-extreme quantile for limits (~1-q false-alarm days)")
    ap.add_argument("--ref-bias", type=float, default=0.05, help="bias magnitude used to select parameters on validation")
    ap.add_argument("--onset", type=int, default=300)
    ap.add_argument("--min-post", type=int, default=100)
    ap.add_argument("--sweep-q", type=float, nargs="+", default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.98],
                    help="false-alarm budgets for the operating-curve sweep (results/<analyte>_sweep.csv)")
    ap.add_argument("--lstm", action="store_true")
    ap.add_argument("--T", type=int, default=70)
    ap.add_argument("--ma-window", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    path = os.path.join(a.data_dir, f"{a.analyte}.csv")
    days = load_days(path)
    train, val, test = split_days(days)
    minutes = load_minutes(path)  # minute-of-day per result, used to report delay in minutes
    test_eps, test_min = episode_pairs(test, None if minutes is None else split_days(minutes)[2], a.onset, a.min_post)
    pre = Preprocessor.fit(train, *TRUNCATION[a.analyte])
    both = train + val  # classical detectors have no trained parameters: calibrate on all clean history
    biases = sorted(set(a.biases) | {-b for b in a.biases})

    rows, sweep_rows, chosen = [], [], {}

    def run(det, calib):
        lim = calibrate(det, pre, calib, a.q)
        replay = Replay(det, pre, test_eps, biases, a.onset)  # scores computed once, reused for every limit
        rows.extend(summarise(det, lim, replay, test_min, a.analyte))
        sweep_rows.extend(sweep(det, pre, calib, replay, test_min, a.sweep_q, a.analyte))
        chosen[det.label] = dict(limit=lim)

    def save():
        """Write results after every detector, so a stopped run keeps what finished."""
        os.makedirs(a.out, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{a.analyte}.csv"), index=False)
        pd.DataFrame(sweep_rows).to_csv(os.path.join(a.out, f"{a.analyte}_sweep.csv"), index=False)
        with open(os.path.join(a.out, f"{a.analyte}_config.json"), "w") as f:
            json.dump(dict(args=vars(a), n_days=dict(train=len(train), val=len(val), test=len(test)),
                           preprocessor=dict(mu=pre.mu, sd=pre.sd, lo=str(pre.lo), hi=str(pre.hi)),
                           detectors=chosen), f, indent=2)

    # Baseline exactly as published: MA, +-3 raw SD limits, fixed (not calibrated).
    run(MovingAverage(50, fixed_limit=3.0, label="MA(n=50) paper limits"), both)
    for fam, cands in FAMILIES.items():
        run(select(cands, pre, both, val, a.ref_bias, a.onset, a.min_post, a.q), both)
    save()

    if a.lstm:
        from pbrtqc.lstm_ae import LstmAEDetector, train_detector
        zs = lambda ds: [pre.transform(x)[0] for x in ds]
        os.makedirs("models", exist_ok=True)
        for ma, name in ((0, "LSTM-AE"), (a.ma_window, f"LSTM-AE-MA(w={a.ma_window})")):
            label = f"{name}(T={a.T})"
            prefix = os.path.join("models", f"{a.analyte}_{name}_T{a.T}_seed{a.seed}")
            if os.path.exists(prefix + ".keras") and os.path.exists(prefix + ".json"):
                print(f"Resuming: loading trained {label} from {prefix}")  # skip retraining after a stop
                det = LstmAEDetector.load(prefix, label)
            else:
                det = train_detector(zs(train), zs(val), a.T, ma, label, seed=a.seed)
                det.save(prefix)
            run(det, val)  # LSTM limit from held-out validation days only
            save()

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
