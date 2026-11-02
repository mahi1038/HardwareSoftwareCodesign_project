"""Linux-side inference for M4/M4S/M5 and the FFT features.
numpy (+ joblib for the forests) only. Copied to the board unchanged, together with features.py."""
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from features import features_from

RATE_HZ = {"r50": 50.0, "r25": 25.0}


def fft_features(W, rate_hz):
    """W: (n, N, 3) int16, already windowed/decimated. Returns (n, 24) float32.
    Per signal [x, y, z, mag]: 4 band energies, dominant frequency (Hz), peak magnitude."""
    a = W.astype(np.float64)
    mag = np.sqrt((a * a).sum(axis=2))
    n, N = mag.shape
    nb = N // 2
    out = []
    for s in (a[:, :, 0], a[:, :, 1], a[:, :, 2], mag):
        X = np.abs(np.fft.rfft(s - s.mean(axis=1, keepdims=True), axis=1))[:, 1:nb + 1]
        bands = (X ** 2 / N).reshape(n, 4, nb // 4).sum(axis=2)
        out += [bands[:, 0], bands[:, 1], bands[:, 2], bands[:, 3],
                (X.argmax(axis=1) + 1) * rate_hz / N, X.max(axis=1)]
    return np.stack(out, axis=1).astype(np.float32)


def rf_features(W, rate, stat_only):
    s = features_from(W)
    return s if stat_only else np.hstack([s, fft_features(W, RATE_HZ[rate])])


def _conv1d(x, w, b):                      # x (n,Cin,L), w (Cout,Cin,K) -> (n,Cout,L-K+1)
    win = sliding_window_view(x, w.shape[2], axis=2)
    return np.einsum("nclk,ock->nol", win, w, optimize=True) + b[None, :, None]


def _relu_pool(x):                         # ReLU then MaxPool1d(2), floor like torch
    x = np.maximum(x, 0)
    n, c, L = x.shape
    return x[:, :, : L // 2 * 2].reshape(n, c, L // 2, 2).max(axis=3)


def cnn_logits(W, p):
    x = ((W.astype(np.float32) - p["mean"]) / p["std"]).transpose(0, 2, 1)
    x = _relu_pool(_conv1d(x, p["w1"], p["b1"]))
    x = _relu_pool(_conv1d(x, p["w2"], p["b2"]))
    return x.mean(axis=2) @ p["wf"].T + p["bf"]


def softmax_max(L):
    L = L.astype(np.float64)
    e = np.exp(L - L.max(axis=1, keepdims=True))
    return (e.max(axis=1) / e.sum(axis=1)).astype(np.float32)


def load(path, mid):
    path = Path(path)
    if mid == "M5":
        d = np.load(path / "model.npz")
        return {k: d[k] for k in d.files}
    import joblib
    return joblib.load(path / "model.joblib")


def predict(mid, model, W, rate):
    """W: (n, N, 3) int16 already windowed/decimated. Returns (labels, confidence)."""
    if mid == "M5":
        L = cnn_logits(W, model)
        return L.argmax(axis=1), softmax_max(L)
    P = model.predict_proba(rf_features(W, rate, stat_only=(mid == "M4S")))
    return P.argmax(axis=1), P.max(axis=1).astype(np.float32)