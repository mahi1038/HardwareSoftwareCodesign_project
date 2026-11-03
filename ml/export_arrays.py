"""Convert sklearn models (M1-M4S) to numpy-only model.npz and verify against sklearn
on all 2,947 test windows. Usage: python ml/export_arrays.py"""
import numpy as np, joblib
from pathlib import Path
from data import load_har
from features import quantize, select
import linux_infer as LI

ROOT = Path(__file__).resolve().parents[1]


def trees_to_arrays(trees):
    T, N = len(trees), max(e.tree_.node_count for e in trees)
    left = np.tile(np.arange(N, dtype=np.int32), (T, 1)); right = left.copy()
    feat = np.zeros((T, N), np.int32); thr = np.full((T, N), np.inf)
    val = np.zeros((T, N, 6), np.float32)
    for i, e in enumerate(trees):
        t, n = e.tree_, e.tree_.node_count
        leaf = t.children_left == -1
        idx = np.arange(n)
        left[i, :n] = np.where(leaf, idx, t.children_left)
        right[i, :n] = np.where(leaf, idx, t.children_right)
        feat[i, :n] = np.where(leaf, 0, t.feature)
        thr[i, :n] = np.where(leaf, np.inf, t.threshold)      # leaf: x <= inf -> stays on itself
        v = t.value[:, 0, :]
        val[i, :n] = v / v.sum(axis=1, keepdims=True)
    return dict(left=left, right=right, feat=feat, thr=thr, val=val,
                depth=np.array(max(e.get_depth() for e in trees)))


def convert(est, mid):
    if mid == "M1":
        return trees_to_arrays([est])
    if mid in ("M4", "M4S"):
        return trees_to_arrays(est.estimators_)
    sc = est.named_steps["standardscaler"]
    if mid == "M2":
        lr = est.named_steps["logisticregression"]
        return dict(mean=sc.mean_, scale=sc.scale_, w=lr.coef_, b=lr.intercept_)
    m = est.named_steps["mlpclassifier"]
    return dict(mean=sc.mean_, scale=sc.scale_, w1=m.coefs_[0], b1=m.intercepts_[0],
                w2=m.coefs_[1], b2=m.intercepts_[1])


d = load_har()
Xq = quantize(d["Xte"])
total = 0
for p in sorted((ROOT / "exports").glob("M*_fp32")):
    mid, window, rate, _ = p.name.split("_")
    if mid == "M5":
        continue
    est = joblib.load(p / "model.joblib")
    np.savez_compressed(p / "model.npz", **convert(est, mid))
    W = select(Xq, window, rate)
    F = LI.rf_features(W, rate, stat_only=(mid != "M4"))
    ref = est.predict(F)
    lab, _ = LI.predict(mid, LI.load(p, mid), W, rate)
    mism = int((lab != ref).sum()); total += mism
    print(f"{p.name}: numpy vs sklearn {mism} mismatches / {len(ref)}   npz {(p/'model.npz').stat().st_size/1024:.0f} KiB")
    assert mism <= 3
print("ALL ARRAY EXPORTS OK, total mismatches", total)