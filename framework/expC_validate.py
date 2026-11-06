"""Validate Experiment C on the board: measured adaptive latency vs P4 + escalated x P2 estimate.
Usage: python framework/expC_validate.py"""
from pathlib import Path
import numpy as np, pandas as pd
import db

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main():
    a = pd.read_csv(RES / "measurements_adapt.csv")
    m = db.load()
    print("path values:", a.path.value_counts().to_dict())
    med = m.groupby(["config_id", "window_idx"]).e2e_ms.median()
    exp_pyth = {"ADAPT_M3_M5_w64_r25_th0.80": 0.200, "ADAPT_M3_M5_w128_r50_th0.65": 0.103}   # offline test_esc

    rows = []
    for cid, g in a.groupby("config_id"):
        esc_name = cid.split("_th")[0].replace("ADAPT_M3_", "")          # M5_w64_r25
        p4 = med["P4_M3_w128_r50_fp32"]
        p2 = med[f"P2_{esc_name}_fp32"]
        g = g.copy()
        g["esc"] = g.path == "escalated"
        assert (g.esc == (g.conf < g.theta)).mean() > 0.99, "escalation flag disagrees with conf < theta"
        g["est"] = g.window_idx.map(p4) + g.esc * g.window_idx.map(p2)
        kept, esc = g[~g.esc], g[g.esc]
        rows.append(dict(
            config=cid, esc_rate=round(g.esc.mean(), 3), python_test_esc=exp_pyth[cid],
            acc_board100=round((g.pred_label == g.true_label).mean(), 3),
            measured_ms=round(g.e2e_ms.mean(), 1), estimated_ms=round(g.est.mean(), 1),
            diff_pct=round(100 * (g.e2e_ms.mean() / g.est.mean() - 1), 1),
            kept_ms=round(kept.e2e_ms.mean(), 1), kept_vs_P4_ms=round(kept.e2e_ms.mean() - kept.window_idx.map(p4).mean(), 2),
            esc_ms=round(esc.e2e_ms.mean(), 1) if len(esc) else np.nan,
            esc_vs_sum_ms=round(esc.e2e_ms.mean() - esc.est.mean(), 2) if len(esc) else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "expC_validation.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 220)
    print(out.T.to_string(header=False))


if __name__ == "__main__":
    main()