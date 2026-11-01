"""Reference feature extraction. Must match docs/feature_spec.md exactly.
Input: int16 windows (n, 128, 3) in mg. Output: float32 (n, 24)."""
import numpy as np

WINDOWS = {"w128": 128, "w64": 64}
RATES = {"r50": 1, "r25": 2}
SIGNALS = ("x", "y", "z", "mag")
FEATS = ("mean", "std", "min", "max", "energy", "zc")
FEATURE_NAMES = [f"{s}_{f}" for s in SIGNALS for f in FEATS]   # index = 6*sig + feat
N_FEATURES = len(FEATURE_NAMES)                                # 24


def quantize(X_g):
    """g (float) -> int16 mg. Done once in Python; the board never sees floats."""
    q = np.floor(np.asarray(X_g, dtype=np.float64) * 1000.0 + 0.5)
    return np.clip(q, -32768, 32767).astype(np.int16)


def select(Xq, window, rate):
    """Spec section 2: first n samples of the base window, then every k-th."""
    return Xq[:, :WINDOWS[window]:RATES[rate], :]


def features_from(W):
    """W: (n, N, 3) int16, already windowed/decimated. Returns (n, 24) float32."""
    a = W.astype(np.int64)
    n, N, _ = a.shape
    out = np.zeros((n, N_FEATURES), dtype=np.float64)

    # x, y, z: integer-exact
    S = a.sum(axis=1)                          # (n, 3)
    Q = (a * a).sum(axis=1)
    d = N * a - S[:, None, :]                  # N*x[i] - S, exact int64
    zc = ((d[:, :-1, :] * d[:, 1:, :]) < 0).sum(axis=1)
    cols = {
        0: S / N,                                          # mean
        1: np.sqrt((N * Q - S * S) / (N * N)),             # std, ddof=0
        2: a.min(axis=1),
        3: a.max(axis=1),
        4: Q / N,                                          # energy
        5: zc,
    }
    for s in range(3):
        for f in range(6):
            out[:, 6 * s + f] = cols[f][:, s]

    # magnitude: float32 values, spec section 3
    sq = (a * a).sum(axis=2)                               # (n, N) exact int64
    assert sq.max() < 2**24, "x^2+y^2+z^2 not exact in float32; spec assumption broken"
    m = np.sqrt(sq.astype(np.float32))                     # float32, correctly rounded
    m64 = m.astype(np.float64)
    mean = m64.mean(axis=1)
    dm = np.sign(m64 - mean[:, None])
    out[:, 18] = mean
    out[:, 19] = m64.std(axis=1)                           # ddof=0
    out[:, 20] = m64.min(axis=1)
    out[:, 21] = m64.max(axis=1)
    out[:, 22] = sq.sum(axis=1) / N                        # mean(x^2+y^2+z^2)
    out[:, 23] = ((dm[:, :-1] * dm[:, 1:]) < 0).sum(axis=1)
    return out.astype(np.float32)


def extract(Xq, window="w128", rate="r50"):
    Xq = np.asarray(Xq)
    if Xq.ndim == 2:
        Xq = Xq[None]
    return features_from(select(Xq, window, rate))