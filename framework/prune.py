"""Experiment D: predictor-guided pruning vs exhaustive search. Usage: python framework/prune.py"""
from pathlib import Path
import pandas as pd
import db
from predictor import predict_all, CALIB_CONFIGS
from selector import pareto

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def candidates(t, margin):
    """Configs whose predicted point is not dominated by more than `margin` (relative) in latency."""
    keep = []
    for r in t.itertuples():
        better = t[(t.accuracy >= r.accuracy) & (t.pred_ms * (1 + margin) < r.pred_ms)]
        if better.empty:
            keep.append(r.config_id)
    return set(keep)


def main():
    cfg, _ = predict_all()
    real = db.load()
    df, src = (real, "REAL") if len(real) else (db.load(RES / "measurements_synth.csv"), "SYNTHETIC")
    meas = db.summarize(df)[["config_id", "e2e_med_ms"]].rename(columns={"e2e_med_ms": "lat_ms"})
    t = cfg.merge(meas, on="config_id")
    t["pred_ms"] = t.raw_pred_ms + 1.0          # same overhead rule as predictor.py (ms)
    true_front = set(pareto(t).config_id)
    print(f"[prune] measurements: {src} | exhaustive: {len(t)} runs, front size {len(true_front)}")

    rows = []
    for m in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
        cand = candidates(t, m) | set(CALIB_CONFIGS)
        found = set(pareto(t[t.config_id.isin(cand)]).config_id)
        rec = len(found & true_front) / len(true_front)
        rows.append(dict(margin=m, runs=len(cand), saved_pct=round(100 * (1 - len(cand) / len(t)), 1),
                         front_recall=round(rec, 2), false_front=len(found - true_front)))
    out = pd.DataFrame(rows)
    out["source"] = src
    out.to_csv(RES / "expD_pruning.csv", index=False, lineterminator="\n")
    print(out.to_string(index=False))
    if src == "SYNTHETIC":
        print("\n*** SYNTHETIC: predictor is near-perfect by construction; rerun on real data ***")


if __name__ == "__main__":
    main()