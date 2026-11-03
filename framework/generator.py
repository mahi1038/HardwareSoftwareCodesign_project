"""Enumerate the design space, apply validity rules, then constraint feasibility.
Usage: python framework/generator.py"""
import itertools
from pathlib import Path
import pandas as pd, yaml

FW = Path(__file__).resolve().parent
ROOT = FW.parent
PARTS = ["P1", "P2", "P3", "P4"]
MODELS = ["M1", "M2", "M3", "M4", "M4S", "M5"]
WINDOWS, RATES, PRECS = ["w128", "w64"], ["r50", "r25"], ["fp32", "int8"]
N_BASE = {"w128": 128, "w64": 64}
DECIM = {"r50": 1, "r25": 2}
MCU_MODELS, NEEDS_RAW, INT8_OK = {"M1", "M2", "M3"}, {"M4", "M5"}, {"M2", "M3"}
N_FEAT = 24


def load(name):
    return yaml.safe_load(open(FW / name))


def validity(part, model, prec):
    r = []
    if prec == "int8" and model not in INT8_OK:
        r.append(f"int8 not implemented for {model}")
    elif prec == "int8" and part != "P4":
        r.append("int8 exists only as MCU C code (needs P4)")
    if part == "P4" and model not in MCU_MODELS:
        r.append(f"{model} is Linux-only but P4 classifies on the MCU")
    if part == "P3" and model in NEEDS_RAW:
        r.append(f"{model} needs the raw window on Linux but P3 sends only 24 features")
    return r


def payload_chars(part, n, b):
    h = b["hex_chars_per_int16"]
    return {"P1": 128 * 3 * h, "P2": n * 3 * h, "P3": N_FEAT * b["feature_chars_each"],
            "P4": b["label_chars"]}[part]


def generate():
    plat = load("platform.yaml")
    b, m, lx = plat["bridge"], plat["mcu"], plat["linux"]
    acc = pd.read_csv(ROOT / "results" / "model_accuracy.csv").set_index("config_id")
    rows = []
    for i, (part, model, w, r, prec) in enumerate(
            itertools.product(PARTS, MODELS, WINDOWS, RATES, PRECS)):
        n = N_BASE[w] // DECIM[r]
        mid = f"{model}_{w}_{r}_{prec}"
        why = validity(part, model, prec)
        chars = payload_chars(part, n, b)
        if chars > b["mcu_to_linux"]["max_chars_tested"]:
            why.append(f"payload {chars} chars exceeds tested limit")
        bridge_ms = b["mcu_to_linux"]["fixed_ms"] + b["mcu_to_linux"]["per_char_ms"] * chars
        mcu_us = 0.0
        if part in ("P3", "P4"):
            mcu_us += m["feature_us"][f"{w}_{r}"]
        if part == "P4":
            mcu_us += m["predict_us"].get(f"{model}_{prec}", 0.0)
        model_flash = m["model_flash_bytes"].get(f"{model}_{prec}", 0) if part == "P4" else 0
        flash = m["flash_baseline_bytes"] + m["test_windows_flash_bytes"] + model_flash
        if flash > m["flash_total_bytes"]:
            why.append("MCU flash exceeded")
        a = acc.loc[mid] if mid in acc.index else None
        if a is None and not why:
            raise KeyError(f"no accuracy row for valid config {mid}")
        pbytes = float(a["param_bytes"]) if a is not None else 0.0
        rss = lx["baseline_rss_mb"] + (0 if part == "P4" else lx["rss_growth_factor"] * pbytes / 1e6)
        rows.append(dict(
            config_id=f"{part}_{mid}", num_id=i, partition=part, model=model, window=w, rate=r,
            precision=prec, n_samples=n, payload_chars=chars, bridge_ms=round(bridge_ms, 2),
            mcu_us=round(mcu_us, 1), lat_lb_ms=round(bridge_ms + mcu_us / 1000, 2),
            accuracy=None if a is None else a["accuracy"], macro_f1=None if a is None else a["macro_f1"],
            param_bytes=pbytes, linux_rss_est_mb=round(rss, 1), mcu_flash_total_B=flash,
            mcu_ram_B=m["ram_static_bytes"] + m["ram_stack_estimate_bytes"],
            status="invalid" if why else "valid", reason="; ".join(why)))
    return pd.DataFrame(rows)


def apply_constraints(valid, c):
    why = []
    for _, r in valid.iterrows():
        x = []
        if r.accuracy < c["min_accuracy"]:
            x.append(f"accuracy {r.accuracy:.3f} < {c['min_accuracy']}")
        if r.lat_lb_ms > c["max_latency_ms"]:
            x.append(f"latency lower bound {r.lat_lb_ms:.1f} ms > {c['max_latency_ms']} ms")
        if r.linux_rss_est_mb > c["linux_ram_budget_mb"]:
            x.append(f"Linux RAM est {r.linux_rss_est_mb:.0f} MB > {c['linux_ram_budget_mb']} MB")
        if r.mcu_flash_total_B > c["mcu_flash_budget_kb"] * 1024:
            x.append("MCU flash budget")
        if r.mcu_ram_B > c["mcu_ram_budget_kb"] * 1024:
            x.append("MCU RAM budget")
        why.append("; ".join(x))
    out = valid.copy()
    out["reject_reason"] = why
    out["feasible"] = [w == "" for w in why]
    return out


def main():
    df = generate()
    res = ROOT / "results"
    cols = list(df.columns)
    df.to_csv(res / "configs_all.csv", index=False, lineterminator="\n")
    valid = df[df.status == "valid"].drop(columns=["status", "reason"])
    valid.to_csv(res / "configs_valid.csv", index=False, lineterminator="\n")
    df[df.status == "invalid"].to_csv(res / "configs_rejected.csv", index=False, lineterminator="\n")
    print(f"raw product {len(df)} | valid {len(valid)} | rejected {len(df) - len(valid)}")
    print(valid.groupby("partition").size().to_dict())
    assert len(df) == 192

    print("\nPreview (latency is a LOWER bound: bridge + MCU only, no Linux compute):")
    for name, c in load("constraints.yaml")["scenarios"].items():
        f = apply_constraints(valid, c)
        ok = f[f.feasible]
        if ok.empty:
            print(f"  {name:12s} feasible 0 -> no feasible configuration")
            continue
        best = (ok.sort_values(["accuracy", "lat_lb_ms"], ascending=[False, True]).iloc[0]
                if c["objective"] == "maximize_accuracy"
                else ok.sort_values(["lat_lb_ms", "accuracy"], ascending=[True, False]).iloc[0])
        print(f"  {name:12s} feasible {len(ok):2d} -> {best.config_id}  "
              f"acc {best.accuracy:.4f}  lat>= {best.lat_lb_ms:.1f} ms")


if __name__ == "__main__":
    main()