"""Fake measurements for developing selector/predictor. NOT real data; written to a separate file.
Linux compute times below are invented placeholders."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import db

ROOT = Path(__file__).resolve().parents[1]
LINUX_MS = {"M1": 0.15, "M2": 0.15, "M3": 0.2, "M4": 12.0, "M4S": 9.0, "M5": 4.0}   # placeholders
REPS = 3


def main():
    rng = np.random.default_rng(1)
    cfgs = pd.read_csv(ROOT / "results" / "configs_valid.csv")
    true = json.load(open(ROOT / "golden" / "test_selection.json"))["labels"]
    rows = []
    for c in cfgs.itertuples():
        cid4 = f"{c.model}_{c.window}_{c.rate}_{c.precision}"
        pred = pd.read_csv(ROOT / "exports" / cid4 / "board100_pred.csv").label.values
        sha = db.model_sha(c.config_id)
        lin = 0.0 if c.partition == "P4" else LINUX_MS[c.model]
        for w in range(100):
            for rep in range(REPS):
                br = c.bridge_ms * rng.normal(1, 0.01)
                mcu = c.mcu_us * rng.normal(1, 0.01)
                lx = lin * rng.lognormal(0, 0.1) if lin else 0.0
                rows.append(dict(
                    config_id=c.config_id, window_idx=w, rep=rep,
                    e2e_ms=round(br + mcu / 1000 + lx + 1.0, 3), mcu_us=round(mcu, 1),
                    linux_ms=round(lx, 3), bridge_ms=round(br, 3),
                    pred_label=int(pred[w]), true_label=int(true[w]),
                    mcu_flash_B=c.mcu_flash_total_B, mcu_ram_B=c.mcu_ram_B,
                    linux_rss_MB=c.linux_rss_est_mb, cpu_load=0.0, cpu_governor="synthetic",
                    py_version=db.CURRENT["py_version"], numpy_version=db.CURRENT["numpy_version"],
                    date="2026-11-03", n_repeats=REPS, spec_version="v1", model_sha=sha,
                    n_board_windows=100))
    df = pd.DataFrame(rows)[db.SCHEMA]
    out = ROOT / "results" / "measurements_synth.csv"
    df.to_csv(out, index=False, lineterminator="\n")
    print(f"{len(df)} synthetic rows -> {out}")
    st = db.status(df, cfgs.config_id)
    print("status:", st.value_counts().to_dict())
    assert (st == "fresh").all()
    # stale test: change a condition, the config must be reported stale
    df2 = df.copy(); df2.loc[df2.config_id == cfgs.config_id[0], "numpy_version"] = "2.0.0"
    assert db.status(df2, cfgs.config_id)[cfgs.config_id[0]] == "stale"
    # missing test: drop a config
    assert db.status(df[df.config_id != cfgs.config_id[1]], cfgs.config_id)[cfgs.config_id[1]] == "missing"
    s = db.summarize(df)
    print(s[["config_id", "acc_board", "e2e_med_ms"]].head(6).to_string(index=False))
    print("DB CHECKS PASSED")


if __name__ == "__main__":
    main()