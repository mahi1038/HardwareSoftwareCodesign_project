"""Feasibility filter, Pareto front, ranking, Experiment A. Usage: python framework/selector.py"""
from pathlib import Path
import pandas as pd, yaml
import db

FW = Path(__file__).resolve().parent
ROOT = FW.parent
RES = ROOT / "results"


def load_table():
    real = db.load()
    if len(real):
        df, src = real, "REAL"
    else:
        df, src = db.load(RES / "measurements_synth.csv"), "SYNTHETIC"
    print(f"[selector] measurement source: {src} ({len(df)} rows)")
    cfg = pd.read_csv(RES / "configs_valid.csv")[
        ["config_id", "partition", "model", "window", "rate", "precision",
         "accuracy", "macro_f1", "mcu_flash_total_B", "mcu_ram_B"]]
    s = db.summarize(df)
    t = cfg.merge(s, on="config_id", how="left")
    t = t.rename(columns={"e2e_med_ms": "lat_ms"})
    t["source"] = src
    return t


def reasons(t, c):
    out = []
    for r in t.itertuples():
        x = []
        if pd.isna(r.lat_ms):
            x.append("not measured")
        else:
            if r.lat_ms > c["max_latency_ms"]:
                x.append(f"latency {r.lat_ms:.1f} ms > {c['max_latency_ms']} ms")
            if r.rss_MB > c["linux_ram_budget_mb"]:
                x.append(f"Linux RAM {r.rss_MB:.0f} MB > {c['linux_ram_budget_mb']} MB")
        if r.accuracy < c["min_accuracy"]:
            x.append(f"accuracy {r.accuracy:.3f} < {c['min_accuracy']}")
        if r.mcu_flash_total_B > c["mcu_flash_budget_kb"] * 1024:
            x.append("MCU flash budget")
        if r.mcu_ram_B > c["mcu_ram_budget_kb"] * 1024:
            x.append("MCU RAM budget")
        out.append("; ".join(x))
    return out


def pareto(t, lat="lat_ms", acc="accuracy"):
    """Non-dominated set: no other config is at least as fast AND at least as accurate (one strictly better)."""
    s = t.sort_values([lat, acc], ascending=[True, False])
    best, keep = -1.0, []
    for i, r in s.iterrows():
        if r[acc] > best:
            keep.append(i); best = r[acc]
    return t.loc[keep].sort_values(lat)


def choose(t, c):
    t = t.copy()
    t["reject_reason"] = reasons(t, c)
    ok = t[t.reject_reason == ""]
    if ok.empty:
        return None, t, ok
    if c["objective"] == "maximize_accuracy":
        best = ok.sort_values(["accuracy", "lat_ms"], ascending=[False, True]).iloc[0]
    else:
        best = ok.sort_values(["lat_ms", "accuracy"], ascending=[True, False]).iloc[0]
    return best, t, ok


def main():
    t = load_table()
    src = t.source.iloc[0]
    sc = yaml.safe_load(open(FW / "constraints.yaml"))["scenarios"]

    # ---- Pareto front over all measured valid configs
    p = pareto(t.dropna(subset=["lat_ms"]))
    p.to_csv(RES / "pareto.csv", index=False, lineterminator="\n")
    print(f"\nPareto front ({len(p)} of {len(t)} configs):")
    print(p[["config_id", "accuracy", "lat_ms"]].to_string(index=False))

    # ---- scenarios
    rej = []
    print("\nScenarios:")
    for name, c in sc.items():
        best, tt, ok = choose(t, c)
        tt["scenario"] = name
        rej.append(tt[tt.reject_reason != ""])
        if best is None:
            print(f"  {name:12s} feasible 0 -> NO FEASIBLE CONFIGURATION")
        else:
            print(f"  {name:12s} feasible {len(ok):2d} -> {best.config_id}  "
                  f"acc {best.accuracy:.4f}  lat {best.lat_ms:.1f} ms")
    pd.concat(rej)[["scenario", "config_id", "accuracy", "lat_ms", "reject_reason"]].to_csv(
        RES / "rejected_by_scenario.csv", index=False, lineterminator="\n")

    # ---- Experiment A: tighten the latency limit
    base = dict(sc["loose"], min_accuracy=0.80, objective="maximize_accuracy")
    top = t.dropna(subset=["lat_ms"]).sort_values("accuracy", ascending=False).iloc[0]
    rows = []
    for lim in [1000, 200, 150, 100, 75, 50, 30, 20, 10, 8, 7]:
        c = dict(base, max_latency_ms=lim)
        best, _, ok = choose(t, c)
        r = dict(max_latency_ms=lim, n_feasible=len(ok),
                 naive_config=top.config_id, naive_acc=round(top.accuracy, 4),
                 naive_lat_ms=round(top.lat_ms, 1), naive_meets_limit=top.lat_ms <= lim)
        if best is not None:
            r.update(chosen=best.config_id, chosen_acc=round(best.accuracy, 4),
                     chosen_lat_ms=round(best.lat_ms, 1),
                     acc_given_up=round(top.accuracy - best.accuracy, 4))
        else:
            r.update(chosen="NONE")
        rows.append(r)
    ea = pd.DataFrame(rows)
    ea["source"] = src
    ea.to_csv(RES / "expA_latency.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 220)
    print(f"\nExperiment A (min accuracy 0.80; naive = always {top.config_id}):")
    print(ea[["max_latency_ms", "n_feasible", "naive_meets_limit", "chosen",
              "chosen_acc", "chosen_lat_ms", "acc_given_up"]].to_string(index=False))
    if src == "SYNTHETIC":
        print("\n*** SYNTHETIC DATA: do not use these numbers in the report ***")


if __name__ == "__main__":
    main()