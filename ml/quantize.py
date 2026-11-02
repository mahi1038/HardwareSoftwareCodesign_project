"""int8 post-training quantization of M2/M3. Python emulation mirrors the generated C exactly.
Usage: python ml/quantize.py [--window w128] [--rate r50]"""
import json, sys
from pathlib import Path
import numpy as np, joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from train import get_data
from export_c import cf, carr, write, softmax_max
from features import N_FEATURES

ROOT = Path(__file__).resolve().parents[1]
f32 = np.float32


def iarr(name, a):
    a = np.asarray(a)
    if a.ndim == 1:
        return f"static const int8_t {name}[{a.shape[0]}] = {{{','.join(map(str, a))}}};\n"
    rows = ",\n  ".join("{" + ",".join(map(str, r)) + "}" for r in a)
    return f"static const int8_t {name}[{a.shape[0]}][{a.shape[1]}] = {{\n  {rows}\n}};\n"


def qrows(W):
    """per-row symmetric int8: returns (int8 matrix, float32 row scales)"""
    s = np.maximum(np.abs(W).max(axis=1), 1e-12) / 127.0
    return np.clip(np.rint(W / s[:, None]), -127, 127).astype(np.int8), s.astype(f32)


def qin(X, mean, inv):
    return np.clip(np.floor((X - mean) * inv + f32(0.5)), -127, 127).astype(np.int64)


def build(mid, est, Xtr):
    sc = est.named_steps["standardscaler"]
    mean, scale = sc.mean_, sc.scale_
    Z = (Xtr - mean) / scale
    s_in = np.maximum(np.percentile(np.abs(Z), 99.9, axis=0), 1e-6) / 127.0     # per feature
    P = dict(mean=mean.astype(f32), inv=(1.0 / (scale * s_in)).astype(f32))
    if mid == "M2":
        lr = est.named_steps["logisticregression"]
        P["Wq"], P["sw"] = qrows(lr.coef_ * s_in[None, :])
        P["b"] = lr.intercept_.astype(f32)
    else:
        mlp = est.named_steps["mlpclassifier"]
        W1T, W2T = mlp.coefs_[0].T, mlp.coefs_[1].T
        P["W1q"], P["sw1"] = qrows(W1T * s_in[None, :])
        P["b1"], P["b2"] = (b.astype(f32) for b in mlp.intercepts_)
        xq = qin(Xtr, P["mean"], P["inv"])
        h = np.maximum(P["sw1"] * (xq @ P["W1q"].T.astype(np.int64)).astype(f32) + P["b1"], 0)
        s_h = max(np.percentile(h, 99.9), 1e-6) / 127.0
        P["invh"] = f32(1.0 / s_h)
        P["W2q"], sw2 = qrows(W2T)
        P["swh"] = (sw2 * s_h).astype(f32)
    return P


def emulate(mid, P, X):
    xq = qin(X.astype(f32), P["mean"], P["inv"])
    if mid == "M2":
        L = P["sw"] * (xq @ P["Wq"].T.astype(np.int64)).astype(f32) + P["b"]
    else:
        h = np.maximum(P["sw1"] * (xq @ P["W1q"].T.astype(np.int64)).astype(f32) + P["b1"], 0)
        hq = np.clip(np.floor(h * P["invh"] + f32(0.5)), 0, 127).astype(np.int64)
        L = P["swh"] * (hq @ P["W2q"].T.astype(np.int64)).astype(f32) + P["b2"]
    return L.argmax(1), softmax_max(L)


SOFTMAX_C = ("  float sum = 0.0f;\n  for (int c = 0; c < N_CLASSES; c++) sum += expf(lg[c] - lg[best]);\n"
             "  *conf = 1.0f / sum;\n  return best;\n}\n")
QIN_C = ("  int8_t xq[N_FEATURES];\n  for (int i = 0; i < N_FEATURES; i++) {{\n"
         "    float v = floorf((f[i] - {U}_MEAN[i]) * {U}_INV[i] + 0.5f);\n"
         "    if (v > 127.0f) v = 127.0f;\n    if (v < -127.0f) v = -127.0f;\n"
         "    xq[i] = (int8_t)v;\n  }}\n")


def header_m2(P, Pn, U):
    consts = (carr(f"{U}_MEAN", P["mean"]) + carr(f"{U}_INV", P["inv"]) + iarr(f"{U}_W", P["Wq"])
              + carr(f"{U}_SW", P["sw"]) + carr(f"{U}_B", P["b"]))
    code = (f"static inline int {Pn}_predict(const float *f, float *conf) {{\n" + QIN_C.format(U=U)
            + f"  float lg[N_CLASSES];\n  int best = 0;\n  for (int c = 0; c < N_CLASSES; c++) {{\n"
            f"    int32_t acc = 0;\n    for (int i = 0; i < N_FEATURES; i++) acc += (int32_t){U}_W[c][i] * xq[i];\n"
            f"    lg[c] = {U}_SW[c] * (float)acc + {U}_B[c];\n    if (lg[c] > lg[best]) best = c;\n  }}\n" + SOFTMAX_C)
    nbytes = P["Wq"].size + 4 * (P["sw"].size + P["b"].size + 2 * len(P["mean"]))
    return consts, code, nbytes


def header_m3(P, Pn, U):
    H = P["W1q"].shape[0]
    consts = (carr(f"{U}_MEAN", P["mean"]) + carr(f"{U}_INV", P["inv"]) + iarr(f"{U}_W1", P["W1q"])
              + carr(f"{U}_SW1", P["sw1"]) + carr(f"{U}_B1", P["b1"])
              + f"static const float {U}_INVH = {cf(P['invh'])};\n"
              + iarr(f"{U}_W2", P["W2q"]) + carr(f"{U}_SWH", P["swh"]) + carr(f"{U}_B2", P["b2"]))
    code = (f"static inline int {Pn}_predict(const float *f, float *conf) {{\n" + QIN_C.format(U=U)
            + f"  int8_t hq[{H}];\n  for (int j = 0; j < {H}; j++) {{\n"
            f"    int32_t acc = 0;\n    for (int i = 0; i < N_FEATURES; i++) acc += (int32_t){U}_W1[j][i] * xq[i];\n"
            f"    float h = {U}_SW1[j] * (float)acc + {U}_B1[j];\n    if (h < 0.0f) h = 0.0f;\n"
            f"    float v = floorf(h * {U}_INVH + 0.5f);\n    if (v > 127.0f) v = 127.0f;\n    hq[j] = (int8_t)v;\n  }}\n"
            f"  float lg[N_CLASSES];\n  int best = 0;\n  for (int c = 0; c < N_CLASSES; c++) {{\n"
            f"    int32_t acc = 0;\n    for (int j = 0; j < {H}; j++) acc += (int32_t){U}_W2[c][j] * hq[j];\n"
            f"    lg[c] = {U}_SWH[c] * (float)acc + {U}_B2[c];\n    if (lg[c] > lg[best]) best = c;\n  }}\n" + SOFTMAX_C)
    nbytes = (P["W1q"].size + P["W2q"].size
              + 4 * (P["sw1"].size + P["b1"].size + P["swh"].size + P["b2"].size + 2 * len(P["mean"]) + 1))
    return consts, code, nbytes


def main(window="w128", rate="r50"):
    Xtr, ytr, gtr, Xte, yte = get_data(window, rate)
    sel = json.load(open(ROOT / "golden" / "test_selection.json"))["har_test_indices"]
    G = json.load(open(ROOT / "golden" / "golden_v1.json"))
    Xg = np.array([w["features"][f"{window}_{rate}"] for w in G["windows"]], dtype=f32)
    golden_h = ("// generated by ml/quantize.py - do not edit\n#pragma once\n#include <stdint.h>\n"
                f"#ifndef GOLDEN_N\n#define GOLDEN_N {len(Xg)}\n#endif\n")
    for mid, hdr in (("M2", header_m2), ("M3", header_m3)):
        cfp = f"{mid}_{window}_{rate}_fp32"
        cid = f"{mid}_{window}_{rate}_int8"
        est = joblib.load(ROOT / "exports" / cfp / "model.joblib")
        P = build(mid, est, Xtr)
        lab, conf = emulate(mid, P, Xte)
        ref = est.predict(Xte)
        acc, f1 = accuracy_score(yte, lab), f1_score(yte, lab, average="macro")
        fp = json.loads((ROOT / "exports" / cfp / "meta.json").read_text())
        agree = float((lab == ref).mean())

        out = ROOT / "exports" / cid
        out.mkdir(parents=True, exist_ok=True)
        consts, code, nbytes = hdr(P, cid.lower(), cid.upper())
        write(out / f"model_{cid}.h",
              f"// generated by ml/quantize.py - do not edit\n// config_id: {cid}\n"
              "// input: 24 float32 features (spec order); returns class 0..5, conf in [0,1]\n"
              "#pragma once\n#include <stdint.h>\n#include <math.h>\n#ifndef N_FEATURES\n#define N_FEATURES 24\n#endif\n"
              "#ifndef N_CLASSES\n#define N_CLASSES 6\n#endif\n" + consts + code)
        np.savetxt(out / "confusion.csv", confusion_matrix(yte, lab, labels=range(6)), fmt="%d", delimiter=",")
        write(out / "board100_pred.csv", "window_idx,label,conf\n" + "\n".join(
            f"{i},{a},{c:.6f}" for i, (a, c) in enumerate(zip(lab[sel], conf[sel]))) + "\n")
        gl, gc = emulate(mid, P, Xg)
        golden_h += (f"static const uint8_t GOLDEN_{cid.upper()}_LABEL[GOLDEN_N] = {{{','.join(map(str, gl))}}};\n"
                     f"static const float GOLDEN_{cid.upper()}_CONF[GOLDEN_N] = {{{','.join(cf(c) for c in gc)}}};\n")
        meta = dict(config_id=cid, model=mid, window=window, rate=rate, precision="int8",
                    n_features=N_FEATURES, accuracy=acc, macro_f1=f1,
                    accuracy_board100=accuracy_score(yte[sel], lab[sel]),
                    fp32_accuracy=fp["accuracy"], fp32_macro_f1=fp["macro_f1"],
                    agreement_with_fp32=agree, param_bytes=int(nbytes), fp32_param_bytes=fp["param_bytes"],
                    header_bytes=(out / f"model_{cid}.h").stat().st_size)
        write(out / "meta.json", json.dumps(meta, indent=2))
        print(f"{cid}: acc {acc:.4f} (fp32 {fp['accuracy']:.4f}, delta {acc - fp['accuracy']:+.4f}) "
              f"macro-F1 {f1:.4f} | agreement with fp32 {agree:.4f} | "
              f"params {nbytes} B vs {fp['param_bytes']} B ({nbytes / fp['param_bytes']:.0%})")
    write(ROOT / "golden" / "model_golden_int8_v1.h", golden_h)
    print("ALL QUANTIZATION STEPS DONE")


if __name__ == "__main__":
    kw = {a[2:]: sys.argv[i + 1] for i, a in enumerate(sys.argv) if a in ("--window", "--rate")}
    main(**kw)