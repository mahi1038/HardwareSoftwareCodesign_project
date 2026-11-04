"""Experiment D: predictor-guided pruning vs exhaustive search. Usage: python framework/prune.py"""
from pathlib import Path
import pandas as pd
import db
from predictor import fitted_predictions, CALIB_CONFIGS
from selector import pareto

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def candidates(t, margin):
    """Configs not dominated in predicted latency by more than `margin` (relative)."""
    keep = []
    for r in t.itertuples():
        better = t[(t.accuracy >= r.accuracy) & (t.pred_ms * (1 + margin) < r.pred_ms)]
        if better.empty:
            keep.append(r.config_id)
    return set(keep)


def main():
    real = db.load()
    df, src = (real, "REAL") if len(real) else (db.load(RES / "measurements_synth.csv"), "SYNTHETIC")
    meas = db.summarize(df)[["config_id", "e2e_med_ms"]].rename(columns={"e2e_med_ms": "meas_ms"})
    t, over = fitted_predictions(meas)
    t["lat_ms"] = t.meas_ms
    print(f"[prune] measurements: {src} | overheads fitted on {len(CALIB_CONFIGS)} configs: "
          f"{over.round(2).to_dict()}")

    true_front = set(pareto(t).config_id)
    calib = set(CALIB_CONFIGS)
    scored = true_front - calib          # front configs the predictor was NOT calibrated on
    print(f"exhaustive: {len(t)} runs, front size {len(true_front)} "
          f"({len(scored)} not in calibration set)")

    rows = []
    for m in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]:
        cand = candidates(t, m)
        measured = cand | calib          # calibration runs are always paid for
        found = set(pareto(t[t.config_id.isin(measured)]).config_id)
        rows.append(dict(margin=m, runs=len(measured),
                         saved_pct=round(100 * (1 - len(measured) / len(t)), 1),
                         front_recall=round(len(found & true_front) / len(true_front), 2),
                         recall_excl_calib=round(len((found & true_front) - calib) / max(len(scored), 1), 2),
                         false_front=len(found - true_front)))
    out = pd.DataFrame(rows)
    out["source"] = src
    out.to_csv(RES / "expD_pruning.csv", index=False, lineterminator="\n")
    print(out.to_string(index=False))
    if src == "SYNTHETIC":
        print("\n*** SYNTHETIC: do not use these numbers ***")


if __name__ == "__main__":
    main()