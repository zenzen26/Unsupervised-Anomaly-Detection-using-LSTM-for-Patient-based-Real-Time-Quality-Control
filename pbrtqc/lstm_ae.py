"""LSTM autoencoder detector (same architecture as the paper), evaluated causally.

score[t] = reconstruction MSE of the window of the last T (optionally
moving-averaged) results ending at t, so the LSTM raises alarms per result and
its detection delay is measured exactly like the statistical detectors.
Training windows never cross a day boundary; early stopping uses whole held-out
validation days (not the tail of an overlapping-window array).
TensorFlow is imported lazily so the rest of the package runs without it.
"""
import json

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .detectors import Detector, rolling_mean


def _process(z, ma_window):
    """MA preprocessing; returns (series, first_valid_index)."""
    if ma_window and ma_window > 1:
        return rolling_mean(z, ma_window), ma_window - 1
    return np.asarray(z, float), 0


def make_windows(streams, T, ma_window, stride=1):
    """Windows within each stream (day) only. streams are standardised arrays."""
    out = []
    for z in streams:
        p, first = _process(z, ma_window)
        p = p[first:]
        if len(p) >= T:
            out.append(sliding_window_view(p, T)[::stride])
    return np.concatenate(out)[:, :, None]


def build_model(T):
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
    inp = Input(shape=(T, 1))
    x = Dropout(0.2)(LSTM(64, return_sequences=True)(inp))
    x = Dropout(0.2)(LSTM(32, return_sequences=True)(x))
    x = LSTM(16)(x)
    x = RepeatVector(T)(x)
    x = Dropout(0.2)(LSTM(16, return_sequences=True)(x))
    x = Dropout(0.2)(LSTM(32, return_sequences=True)(x))
    model = Model(inp, TimeDistributed(Dense(1))(x))
    model.compile(optimizer="adam", loss="mse")
    return model


class LstmAEDetector(Detector):
    def __init__(self, model, T, ma_window, scale, label):
        self.model, self.T, self.ma_window, self.scale, self.label = model, T, ma_window, scale, label

    def score(self, z):
        out = np.full(len(z), np.nan)
        p, first = _process(z, self.ma_window)
        p = p[first:]
        if len(p) < self.T:
            return out
        w = sliding_window_view(p, self.T)[:, :, None] / self.scale
        rec = self.model.predict(w, verbose=0)
        out[first + self.T - 1:] = np.mean((w - rec) ** 2, axis=(1, 2))
        return out

    def save(self, path_prefix):
        self.model.save(f"{path_prefix}.keras")
        with open(f"{path_prefix}.json", "w") as f:
            json.dump(dict(T=self.T, ma_window=self.ma_window, scale=self.scale), f)


def _load(cls, path_prefix, label):
    import tensorflow as tf
    with open(f"{path_prefix}.json") as f:
        meta = json.load(f)
    model = tf.keras.models.load_model(f"{path_prefix}.keras")
    return cls(model, meta["T"], meta["ma_window"], meta["scale"], label)


LstmAEDetector.load = classmethod(_load)


def train_detector(train_streams, val_streams, T, ma_window, label, seed=0,
                   epochs=100, patience=5, batch_size=32, stride=1):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    xtr = make_windows(train_streams, T, ma_window, stride)
    xva = make_windows(val_streams, T, ma_window, stride)
    scale = float(xtr.std()) or 1.0  # constant rescale only; shifts of the level are preserved
    model = build_model(T)
    model.fit(xtr / scale, xtr / scale, validation_data=(xva / scale, xva / scale), epochs=epochs,
              batch_size=batch_size, shuffle=True, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)])
    return LstmAEDetector(model, T, ma_window, scale, label)
