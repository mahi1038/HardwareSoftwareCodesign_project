import numpy as np
from data import load_har, ACTIVITIES
from features import quantize, select, extract, features_from, FEATURE_NAMES

# 1. hand-computed case: x = [1,-1,1,-1] -> mean 0, std 1, min -1, max 1, energy 1, zc 3
W = np.zeros((1, 4, 3), dtype=np.int16)
W[0, :, 0] = [1, -1, 1, -1]
f = features_from(W)[0]
assert list(f[:6]) == [0, 1, -1, 1, 1, 3], f[:6]

# 2. tie case: x = [0,1,2] has mean 1; the sample equal to the mean never counts -> zc 0
W = np.zeros((1, 3, 3), dtype=np.int16)
W[0, :, 0] = [0, 1, 2]
assert features_from(W)[0][5] == 0

# 3. constant window: std exactly 0, zc 0
W = np.full((1, 64, 3), 500, dtype=np.int16)
f = features_from(W)[0]
assert f[1] == 0 and f[5] == 0 and f[0] == 500

# 4. whole test set vs. a naive float64 reference, for all 4 window/rate combos
d = load_har()
Xq = quantize(d["Xte"])
assert Xq.dtype == np.int16 and np.abs(Xq).max() < 32767


def naive(w):                                   # w: (N,3) int16
    x = w.astype(np.float64)
    m = np.sqrt((x ** 2).sum(axis=1)).astype(np.float32).astype(np.float64)
    res = []
    for sig in (x[:, 0], x[:, 1], x[:, 2], m):
        mu = sig.mean()
        s = np.sign(sig - mu)
        res += [mu, sig.std(), sig.min(), sig.max(), (sig ** 2).mean(),
                ((s[:-1] * s[1:]) < 0).sum()]
    return np.array(res)


rng = np.random.default_rng(0)
idx = rng.choice(len(Xq), 300, replace=False)
for win in ("w128", "w64"):
    for rate in ("r50", "r25"):
        F = extract(Xq[idx], win, rate)
        R = np.stack([naive(select(Xq[idx], win, rate)[i]) for i in range(len(idx))])
        for j, name in enumerate(FEATURE_NAMES):
            if name.endswith("_zc"):
                tol = 1 if name.startswith("mag") else 0
                assert np.abs(F[:, j] - R[:, j]).max() <= tol, (win, rate, name)
            else:
                err = np.abs(F[:, j] - R[:, j])
                assert (err <= 1e-3 + 1e-4 * np.abs(R[:, j])).all(), (win, rate, name, err.max())
        print(f"{win}/{rate}: features shape {F.shape}  OK vs naive reference")

# 5. sanity: features separate the classes
F = extract(Xq, "w128", "r50")
y = d["yte"]
for k, a in enumerate(ACTIVITIES):
    print(f"{a:20s} x_mean={F[y==k, 0].mean():7.0f}  mag_std={F[y==k, 19].mean():7.1f}")
print("ALL FEATURE TESTS PASSED")