"""Sweep: for each window/rate, train M1-M3, export fp32 C, quantize M2/M3 to int8,
then collect all meta.json into results/model_accuracy.csv.  Usage: python ml/train_all.py"""
import json, time
from pathlib import Path
import pandas as pd
import train, export_c, quantize

ROOT = Path(__file__).resolve().parents[1]
# w128/r50 last, so the shared console output ends on the default configuration
COMBOS = [("w128", "r25"), ("w64", "r50"), ("w64", "r25"), ("w128", "r50")]
COLS = ["config_id", "model", "window", "rate", "precision", "accuracy", "macro_f1",
        "accuracy_board100", "cv_macro_f1", "param_bytes", "header_bytes"]


def main():
    t0 = time.time()
    for w, r in COMBOS:
        print(f"\n######## {w} / {r} ########")
        for m in ("M1", "M2", "M3"):
            meta = train.train_one(m, w, r, verbose=False)
            print(f"{meta['config_id']}: acc {meta['accuracy']:.4f} F1 {meta['macro_f1']:.4f} "
                  f"CV {meta['cv_macro_f1']:.4f} {meta['best_params']}")
        export_c.main(w, r)
        quantize.main(w, r)

    rows = [json.loads(p.read_text()) for p in sorted((ROOT / "exports").glob("M*_*/meta.json"))]
    df = pd.DataFrame(rows).reindex(columns=COLS)
    df = df.sort_values(["window", "rate", "model", "precision"], ascending=[False, False, True, True])
    out = ROOT / "results" / "model_accuracy.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False, lineterminator="\n")
    pd.set_option("display.width", 200)
    print("\n", df[["config_id", "accuracy", "macro_f1", "param_bytes"]].to_string(index=False))
    print(f"\n{len(df)} configurations -> {out}   ({time.time() - t0:.0f} s)")
    assert len(df) == 20, "expected 20 model configurations"


if __name__ == "__main__":
    main()