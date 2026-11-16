# Unsupervised anomaly detection for PBRTQC (LSTM autoencoder vs statistical detectors)

Patient-based real-time quality control (PBRTQC) on sodium and creatinine results from Westmead
Hospital (31 Dec 2022 - 31 Mar 2023). Detectors: moving average (MA), EWMA, CUSUM and unsupervised
LSTM autoencoders (with and without MA preprocessing). Every detector sees identical day-wise
streams, has its limit calibrated on clean days only (no anomaly labels), and has its parameters
chosen on validation days. Each test day is replayed clean and with a step shift of +-1 to 20%
injected after the 300th result.

**Headline result:** calibrated MA/EWMA/CUSUM detect 3-5% sodium shifts within about 10-40 results;
the LSTM autoencoders give no advantage over the best statistical detector (CUSUM) at any
false-alarm budget tested. See `paper/paper.tex`.

## Layout
```
pbrtqc/              Pipeline package: data split + bias injection, detectors, LSTM-AE, evaluation
run_experiments.py   One-command experiment runner
tests/               Unit tests (numpy + pandas only; no TensorFlow needed)
data/                sodium.csv, creatinine.csv (columns Day, Result, Minute); DeIdentified_Data.xlsx = raw extract
results/             Output of run_experiments.py (see below)
paper/               LaTeX paper (paper.tex, reference.bib, resources/ figures)
notebooks/           colab_run.ipynb - runs the LSTM experiments on a Colab GPU
docs/                Compiled paper (Thesis Report.pdf)
```

## Run
```
python -m unittest discover -s tests
python run_experiments.py --analyte sodium --out scratch      # statistical detectors only, no TensorFlow
python run_experiments.py --analyte creatinine --lstm         # + LSTM-AE / LSTM-AE-MA (needs tensorflow)
```
- **Write statistical-only runs to another folder (`--out scratch`).** A run without `--lstm`
  rewrites `results/<analyte>.csv` without the LSTM rows.
- **LSTM runs:** use Colab (`notebooks/colab_run.ipynb`, T4 GPU) or Python 3.10-3.12 with
  `pip install tensorflow pandas numpy`. About 10-20 minutes per model on a T4.
- **Resumable:** results are written after every detector, and trained models are saved to `models/`
  (git-ignored). Re-running the same command reloads a saved model (it prints "Resuming") instead of
  retraining, so a stopped Colab session only loses the model that was training.
- Options: `--q` (false-alarm budget, default 0.9), `--sweep-q` (operating curve), `--biases`,
  `--onset`, `--T`, `--ma-window`, `--seed`.

## Outputs
| File | Content |
|---|---|
| `results/<analyte>.csv` | One row per detector and signed shift at the main budget (q = 0.9): false-alarm day rate, detection rate, median delay in results and in minutes |
| `results/<analyte>_sweep.csv` | The same at several budgets `q` (operating curves; calibrated detectors only) |
| `results/<analyte>_config.json` | Arguments, split sizes, preprocessing constants, calibrated limits |

Paper tables: Table I = `<analyte>.csv` (positive and negative shifts averaged); Table II =
`<analyte>_sweep.csv`.

## Data
- `sodium.csv` / `creatinine.csv`: one row per result, in recorded order. `Day` = recorded date,
  `Minute` = minute of day it was recorded. Results are standardised on training days only.
- `DeIdentified_Data.xlsx` is the raw extract released as de-identified quality-improvement data.
  It still carries a pseudonymised MRN (re-scrambled in this repository), accession numbers, ward and
  timestamps, and is not read by any code. Do not share it outside the data-governance agreement.
