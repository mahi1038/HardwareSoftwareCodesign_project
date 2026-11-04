"""Experiment C: adaptive inference. theta chosen on TRAIN subjects (GroupKFold out-of-fold),
reported on TEST subjects. Usage: python framework/expC.py   (first run ~5 min: trains CNNs per fold)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from data import load_har
from features import quantize, select, extract
import linux_infer as LI
import train_linux
import db, selector

FIRST = ("M2", "M3")                       # MCU model (P4, w128/r50)
ESC = {"M5_w128_r50": ("w128", "r50"), "M5_w64_r25": ("w64", "r25")}   # escalation target
GRID = [round(x, 2) for x in np.arange(0.40, 1.0, 0.05)] + [1.01]
TOL = 0.005                                # accept theta within 0.5 pt of best OOF accuracy


def oof_predictions(d):
    cache = ROOT / "data" / "expC_oof.npz"
    if cache.exists():
        return dict(np.load(cache))
    Q = quantize(d["Xtr"]); y = d["ytr"]; g = d["str_"]
    F = extract(Q, "w128", "r50")
    folds = list(GroupKFold(n_splits=5).split(F, y, g))
    out = {}
    for m in FIRST:
        est = joblib.load(ROOT / "exports" / f"{m}_w128_r50_fp32" / "model.joblib")
        lab = np.zeros(len(y), int); conf = np.zeros(len(y), np.float32)
        for tr, va in folds:
            e = clone(est).fit(F[tr], y[tr])
            P = e.predict_proba(F[va])
            lab[va] = e.classes_[P.argmax(1)]; conf[va] = P.max(1)
        out[f"{m}_lab"], out[f"{m}_conf"] = lab, conf
        print(f"  OOF {m}: acc {(lab == y).mean():.4f}")
    for name, (w, r) in ESC.items():
        Qs = select(Q, w, r)
        lab = np.zeros(len(y), int)
        for k, (tr, va) in enumerate(folds):
            _, _, pred = train_linux.train_cnn(Qs[tr], y[tr], g[tr], Qs[va])
            lab[va] = pred
        out[f"{name}_lab"] = lab
        print(f"  OOF {name}: acc {(lab == y).mean():.4f}")
    np.savez(cache, **out)
    return out


def board_estimate(df, first, esc_name, th):
    """100 board windows: escalation by expected confidence, latency from A's measured per-window e2e."""
    g = df.groupby(["config_id", "window_idx"]).e2e_ms.median()
    p4 = g[f"P4_{first}_w128_r50_fp32"].sort_index().values
    p2 = g[f"P2_{esc_name}_fp32"].sort_index().values
    b1 = pd.read_csv(ROOT / "exports" / f"{first}_w128_r50_fp32" / "board100_pred.csv")
    b2 = pd.read_csv(ROOT / "exports" / f"{esc_name}_fp32" / "board100_pred.csv")
    import json
    true = np.array(json.load(open(ROOT / "golden" / "test_selection.json"))["labels"])
    esc = b1.conf.values < th
    final = np.where(esc, b2.label.values, b1.label.values)
    return esc.mean(), (final == true).mean(), (p4 + esc * p2).mean()


def main():
    d = load_har()
    print("[expC] out-of-fold predictions on training subjects")
    oof = oof_predictions(d)
    ytr, yte = d["ytr"], d["yte"]
    t = selector.load_table()
    src = t.source.iloc[0]
    lat = t.set_index("config_id").lat_ms
    front = selector.pareto(t.dropna(subset=["lat_ms"]))
    real = db.load()
    df = real if len(real) else db.load(ROOT / "results" / "measurements_synth.csv")

    Fte = extract(quantize(d["Xte"]), "w128", "r50")
    rows, chosen = [], []
    for first in FIRST:
        m1 = LI.load(ROOT / "exports" / f"{first}_w128_r50_fp32", first)
        p1, c1 = LI.predict_features(first, m1, Fte)
        L1 = lat[f"P4_{first}_w128_r50_fp32"]
        for esc_name, (w, r) in ESC.items():
            m5 = LI.load(ROOT / "exports" / f"{esc_name}_fp32", "M5")
            p2, _ = LI.predict("M5", m5, select(quantize(d["Xte"]), w, r), r)
            L2 = lat[f"P2_{esc_name}_fp32"]
            recs = []
            for th in GRID:
                eo = oof[f"{first}_conf"] < th
                fo = np.where(eo, oof[f"{esc_name}_lab"], oof[f"{first}_lab"])
                et = c1 < th
                ft = np.where(et, p2, p1)
                recs.append(dict(first=first, escalate_to=esc_name, theta=th,
                                 oof_esc=round(eo.mean(), 3), oof_acc=round((fo == ytr).mean(), 4),
                                 test_esc=round(et.mean(), 3), test_acc=round((ft == yte).mean(), 4),
                                 exp_lat_ms=round(L1 + et.mean() * L2, 1), L1_ms=round(L1, 1), L2_ms=round(L2, 1)))
            R = pd.DataFrame(recs)
            ok = R[R.oof_acc >= R.oof_acc.max() - TOL]          # decided on TRAIN data only
            pick = ok.sort_values("oof_esc").iloc[0]            # cheapest acceptable theta
            R["chosen"] = R.theta == pick.theta
            be, ba, bl = board_estimate(df, first, esc_name, pick.theta)
            beats = pick.test_acc > front[front.lat_ms <= pick.exp_lat_ms].accuracy.max()
            chosen.append(dict(first=first, escalate_to=esc_name, theta=pick.theta,
                               test_esc=pick.test_esc, test_acc=pick.test_acc, exp_lat_ms=pick.exp_lat_ms,
                               board100_esc=round(be, 2), board100_acc=round(ba, 3),
                               board100_lat_ms=round(bl, 1), above_pareto_front=bool(beats)))
            rows.append(R)
    out = pd.concat(rows); out["source"] = src
    out.to_csv(ROOT / "results" / "expC_adaptive.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 220)
    print(f"\nalways-M3 at P4: acc {accuracy(p1, yte):.4f} | M5 w128/r50 at P2: {lat['P2_M5_w128_r50_fp32']:.1f} ms"
          f" | M5 w64/r25 at P2: {lat['P2_M5_w64_r25_fp32']:.1f} ms")
    print("\nTheta chosen on TRAIN subjects, evaluated on TEST subjects:")
    print(pd.DataFrame(chosen).to_string(index=False))
    if src == "SYNTHETIC":
        print("*** SYNTHETIC latencies: do not use ***")


def accuracy(p, y):
    return float((p == y).mean())


if __name__ == "__main__":
    main()