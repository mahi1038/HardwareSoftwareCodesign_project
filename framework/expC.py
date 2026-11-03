"""Experiment C offline part: accuracy and escalation rate vs confidence threshold.
Usage: python framework/expC.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from data import load_har
from features import quantize, select, extract
import linux_infer as LI
import selector

W, R = "w128", "r50"


def main():
    d = load_har()
    Q = select(quantize(d["Xte"]), W, R)
    y = d["yte"]
    F = extract(quantize(d["Xte"]), W, R)
    m5 = LI.load(ROOT / "exports" / f"M5_{W}_{R}_fp32", "M5")
    p2, _ = LI.predict("M5", m5, Q, R)
    acc2 = (p2 == y).mean()

    t = selector.load_table().set_index("config_id")
    L2 = t.loc[f"P2_M5_{W}_{R}_fp32", "lat_ms"]
    rows = []
    for first in ("M2", "M3"):
        m = LI.load(ROOT / "exports" / f"{first}_{W}_{R}_fp32", first)
        p1, c1 = LI.predict_features(first, m, F)
        L1 = t.loc[f"P4_{first}_{W}_{R}_fp32", "lat_ms"]
        for th in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]:
            esc = c1 < th
            final = np.where(esc, p2, p1)
            rows.append(dict(first=first, threshold=th, esc_rate=round(esc.mean(), 3),
                             accuracy=round((final == y).mean(), 4),
                             acc_kept=round((p1[~esc] == y[~esc]).mean(), 4) if (~esc).any() else np.nan,
                             acc_escalated_first=round((p1[esc] == y[esc]).mean(), 4) if esc.any() else np.nan,
                             exp_lat_ms=round(L1 + esc.mean() * L2, 1),
                             L1_ms=round(L1, 1), L2_ms=round(L2, 1)))
    out = pd.DataFrame(rows)
    out["source"] = t.source.iloc[0]
    out.to_csv(ROOT / "results" / "expC_offline.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 200)
    print(f"always-M5 accuracy {acc2:.4f} (latency {L2:.1f} ms)")
    print(out.drop(columns=["source"]).to_string(index=False))
    for first in ("M2", "M3"):
        o = out[out["first"] == first]
        r0, r1 = o.iloc[0], o.iloc[-1]
        gain = acc2 - r0.accuracy
        print(f"{first}: max gain from escalation {gain:+.4f}; "
              f"break-even escalation rate vs always-M5 = {(L2 - r0.L1_ms) / L2:.2f}")
    if out.source.iloc[0] == "SYNTHETIC":
        print("*** latencies are synthetic; accuracy/escalation columns are real ***")


if __name__ == "__main__":
    main()