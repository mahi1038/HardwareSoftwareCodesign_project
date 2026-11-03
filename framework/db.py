"""Measurement store: schema check, freshness (reuse vs re-run), per-config summary."""
import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ["config_id", "window_idx", "rep", "e2e_ms", "mcu_us", "linux_ms", "bridge_ms",
          "pred_label", "true_label", "mcu_flash_B", "mcu_ram_B", "linux_rss_MB", "cpu_load",
          "cpu_governor", "py_version", "numpy_version", "date", "n_repeats",
          "spec_version", "model_sha", "n_board_windows"]
CURRENT = dict(spec_version="v1", n_board_windows=100, numpy_version="2.5.4", py_version="3.13.9")
MIN_REPS = 3


def model_sha(config_id):
    """Hash of model content. P4 runs the C header on the MCU; other cuts run model.npz on Linux."""
    part, rest = config_id.split("_", 1)
    cid = "_".join(rest.split("_")[:4])
    d = ROOT / "exports" / cid
    h = hashlib.sha1()
    if part == "P4":
        h.update((d / f"model_{cid}.h").read_bytes())
    else:
        z = np.load(d / "model.npz")
        for k in sorted(z.files):
            h.update(k.encode()); h.update(np.ascontiguousarray(z[k]).tobytes())
    return h.hexdigest()[:8]


def load(path=None):
    path = Path(path or ROOT / "results" / "measurements.csv")
    if not path.exists():
        return pd.DataFrame(columns=SCHEMA)
    df = pd.read_csv(path)
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        sys.exit(f"measurements file lacks columns: {missing}")
    return df


def status(df, configs):
    """Per config: fresh (reusable), stale (conditions changed), missing (never measured)."""
    out = {}
    for cid in configs:
        g = df[df.config_id == cid]
        if g.empty:
            out[cid] = "missing"; continue
        ok = ((g.model_sha == model_sha(cid)).all()
              and all((g[k].astype(str) == str(v)).all() for k, v in CURRENT.items())
              and g.groupby("window_idx").rep.nunique().min() >= MIN_REPS
              and g.window_idx.nunique() == CURRENT["n_board_windows"])
        out[cid] = "fresh" if ok else "stale"
    return pd.Series(out, name="status")


def summarize(df):
    """One row per config: board accuracy (agreement with truth) and latency statistics."""
    g = df.groupby("config_id")
    s = pd.DataFrame({
        "n_rows": g.size(),
        "acc_board": g.apply(lambda x: (x.pred_label == x.true_label).mean(), include_groups=False),
        "e2e_med_ms": g.e2e_ms.median(), "e2e_p95_ms": g.e2e_ms.quantile(0.95),
        "e2e_max_ms": g.e2e_ms.max(), "mcu_med_us": g.mcu_us.median(),
        "linux_med_ms": g.linux_ms.median(), "bridge_med_ms": g.bridge_ms.median(),
        "rss_MB": g.linux_rss_MB.max()})
    return s.reset_index()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = pd.read_csv(ROOT / "results" / "configs_valid.csv").config_id
    df = load(path)
    print(status(df, cfg).value_counts().to_dict(), f"({len(df)} rows)")