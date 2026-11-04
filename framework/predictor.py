"""Stage-level latency predictor, calibrated on 4 configs, evaluated on the rest.
Usage: python framework/predictor.py"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import db

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CALIB_CONFIGS = ["P1_M5_w128_r50_fp32", "P2_M5_w128_r50_fp32",
                 "P3_M1_w128_r50_fp32", "P4_M1_w128_r50_fp32"]
PLACEHOLDER_MS = {"M1": 0.15, "M2": 0.15, "M3": 0.2, "M4": 12.0, "M4S": 9.0, "M5": 4.0}


def linux_table():
    p = RES / "calibration_linux.csv"      # columns: stage,model,window,rate,ms
    if p.exists():
        return pd.read_csv(p), "calibration_linux.csv"
    return None, "PLACEHOLDER"


def linux_ms(tab, part, model, w, r):
    if part == "P4":
        return 0.0
    if tab is None:
        return PLACEHOLDER_MS[model]
    stage = "from_features" if part == "P3" else "from_window"
    q = tab[(tab.stage == stage) & (tab.model == model) & (tab.window == w) & (tab.rate == r)]
    if q.empty:     # fall back to the same model at any window/rate
        q = tab[(tab.stage == stage) & (tab.model == model)]
    return float(q.ms.median())


def predict_all():
    cfg = pd.read_csv(RES / "configs_valid.csv")
    tab, src = linux_table()
    cfg["linux_ms_pred"] = [linux_ms(tab, c.partition, c.model, c.window, c.rate)
                            for c in cfg.itertuples()]
    cfg["raw_pred_ms"] = cfg.bridge_ms + cfg.mcu_us / 1000 + cfg.linux_ms_pred
    return cfg, src


def main():
    cfg, src = predict_all()
    real = db.load()
    df, msrc = (real, "REAL") if len(real) else (db.load(RES / "measurements_synth.csv"), "SYNTHETIC")
    s = db.summarize(df)[["config_id", "e2e_med_ms"]].rename(columns={"e2e_med_ms": "meas_ms"})
    t = cfg.merge(s, on="config_id")
    print(f"[predictor] linux times: {src} | measurements: {msrc}")

    cal = t[t.config_id.isin(CALIB_CONFIGS)]
    over = (cal.meas_ms - cal.raw_pred_ms).groupby(cal.partition).median()
    print("fitted overhead per partition (ms):", over.round(2).to_dict())
    t["pred_ms"] = t.raw_pred_ms + t.partition.map(over)
    t["err_ms"] = t.pred_ms - t.meas_ms
    t["err_pct"] = 100 * t.err_ms / t.meas_ms
    ev = t[~t.config_id.isin(CALIB_CONFIGS)]

    print(f"\nHeld-out configs: {len(ev)}")
    print(f"  MAPE            {ev.err_pct.abs().mean():.1f} %")
    print(f"  max |error|     {ev.err_pct.abs().max():.1f} %  ({ev.loc[ev.err_pct.abs().idxmax(), 'config_id']})")
    print(f"  Spearman rank   {spearmanr(ev.pred_ms, ev.meas_ms)[0]:.3f}")
    print(ev.groupby("partition").err_pct.apply(lambda x: x.abs().mean()).round(1).to_string())
    t[["config_id", "partition", "model", "accuracy", "pred_ms", "meas_ms", "err_pct"]].to_csv(
        RES / "predictions.csv", index=False, lineterminator="\n")
    if msrc == "SYNTHETIC" or src == "PLACEHOLDER":
        print("\n*** PLUMBING CHECK ONLY: inputs are synthetic/placeholder, errors mean nothing ***")

def fitted_predictions(meas):
    """meas: DataFrame with config_id, meas_ms. Returns configs with pred_ms using
    per-partition overheads fitted ONLY on CALIB_CONFIGS, plus the overhead dict."""
    cfg, _ = predict_all()
    t = cfg.merge(meas, on="config_id")
    cal = t[t.config_id.isin(CALIB_CONFIGS)]
    over = (cal.meas_ms - cal.raw_pred_ms).groupby(cal.partition).median()
    t["pred_ms"] = t.raw_pred_ms + t.partition.map(over)
    return t, over


if __name__ == "__main__":
    main()