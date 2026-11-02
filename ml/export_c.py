"""Export M1-M3 (fp32) to C headers, verify a C-equivalent float32 model vs sklearn,
and write model golden labels. Usage: python ml/export_c.py [--window w128] [--rate r50]"""
import json, sys
from pathlib import Path
import numpy as np, joblib
from data import load_har
from features import quantize, extract

ROOT = Path(__file__).resolve().parents[1]
f32 = np.float32


def write(path, text):
    with open(path, "w", newline="\n") as f:
        f.write(text)


def cf(v):
    s = f"{float(f32(v)):.9g}"
    if not any(c in s for c in ".en"):
        s += ".0"
    return s + "f"


def carr(name, a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        body = ",".join(cf(v) for v in a)
        return f"static const float {name}[{a.shape[0]}] = {{{body}}};\n"
    rows = ",\n  ".join("{" + ",".join(cf(v) for v in r) + "}" for r in a)
    return f"static const float {name}[{a.shape[0]}][{a.shape[1]}] = {{\n  {rows}\n}};\n"


def thr32(t):
    """largest float32 <= t, so (float32 x <= t) == (x <= thr32) for every float32 x"""
    t32 = f32(t)
    if float(t32) > t:
        t32 = np.nextafter(t32, f32(-np.inf))
    return t32


# ---------------------------------------------------------------- M1 (tree)
def export_m1(est, P, U):
    t = est.tree_
    lines = []

    def rec(n, ind):
        pad = "  " * ind
        if t.children_left[n] == -1:
            v = t.value[n][0]
            lines.append(f"{pad}*conf = {cf(v.max() / v.sum())}; return {int(np.argmax(v))};")
        else:
            lines.append(f"{pad}if (f[{t.feature[n]}] <= {cf(thr32(t.threshold[n]))}) {{")
            rec(t.children_left[n], ind + 1)
            lines.append(f"{pad}}} else {{")
            rec(t.children_right[n], ind + 1)
            lines.append(f"{pad}}}")

    rec(0, 1)
    code = (f"static inline int {P}_predict(const float *f, float *conf) {{\n"
            + "\n".join(lines) + "\n}\n")
    n_int = int((t.children_left != -1).sum())
    nbytes = n_int * 8 + (t.node_count - n_int) * 8          # approximate
    return code, nbytes


def emu_m1(est, X):
    t = est.tree_
    thr = np.array([thr32(v) if t.children_left[i] != -1 else 0 for i, v in enumerate(t.threshold)],
                   dtype=f32)
    node = np.zeros(len(X), dtype=int)
    while True:
        internal = t.children_left[node] != -1
        if not internal.any():
            break
        i = np.where(internal)[0]
        go_left = X[i, t.feature[node[i]]] <= thr[node[i]]
        node[i] = np.where(go_left, t.children_left[node[i]], t.children_right[node[i]])
    v = t.value[node][:, 0, :]
    return v.argmax(1), (v.max(1) / v.sum(1)).astype(f32)


# ---------------------------------------------------------------- M2 (logistic regression)
def export_m2(est, P, U):
    sc, lr = est.named_steps["standardscaler"], est.named_steps["logisticregression"]
    consts = (carr(f"{U}_MEAN", sc.mean_) + carr(f"{U}_SCALE", sc.scale_)
              + carr(f"{U}_W", lr.coef_) + carr(f"{U}_B", lr.intercept_))
    code = (f"static inline int {P}_predict(const float *f, float *conf) {{\n"
            f"  float z[N_FEATURES], lg[N_CLASSES];\n"
            f"  for (int i = 0; i < N_FEATURES; i++) z[i] = (f[i] - {U}_MEAN[i]) / {U}_SCALE[i];\n"
            f"  int best = 0;\n"
            f"  for (int c = 0; c < N_CLASSES; c++) {{\n"
            f"    float s = {U}_B[c];\n"
            f"    for (int i = 0; i < N_FEATURES; i++) s += {U}_W[c][i] * z[i];\n"
            f"    lg[c] = s;\n"
            f"    if (s > lg[best]) best = c;\n"
            f"  }}\n"
            f"  float sum = 0.0f;\n"
            f"  for (int c = 0; c < N_CLASSES; c++) sum += expf(lg[c] - lg[best]);\n"
            f"  *conf = 1.0f / sum;\n"
            f"  return best;\n}}\n")
    return consts, code, 4 * (lr.coef_.size + lr.intercept_.size + 2 * len(sc.mean_))


def emu_m2(est, X):
    sc, lr = est.named_steps["standardscaler"], est.named_steps["logisticregression"]
    Z = (X - sc.mean_.astype(f32)) / sc.scale_.astype(f32)
    L = Z @ lr.coef_.astype(f32).T + lr.intercept_.astype(f32)
    return L.argmax(1), softmax_max(L)


# ---------------------------------------------------------------- M3 (MLP)
def export_m3(est, P, U):
    sc, mlp = est.named_steps["standardscaler"], est.named_steps["mlpclassifier"]
    W1, W2 = mlp.coefs_
    H = W1.shape[1]
    consts = (carr(f"{U}_MEAN", sc.mean_) + carr(f"{U}_SCALE", sc.scale_)
              + carr(f"{U}_W1T", W1.T) + carr(f"{U}_B1", mlp.intercepts_[0])
              + carr(f"{U}_W2T", W2.T) + carr(f"{U}_B2", mlp.intercepts_[1]))
    code = (f"static inline int {P}_predict(const float *f, float *conf) {{\n"
            f"  float z[N_FEATURES], h[{H}], lg[N_CLASSES];\n"
            f"  for (int i = 0; i < N_FEATURES; i++) z[i] = (f[i] - {U}_MEAN[i]) / {U}_SCALE[i];\n"
            f"  for (int j = 0; j < {H}; j++) {{\n"
            f"    float s = {U}_B1[j];\n"
            f"    for (int i = 0; i < N_FEATURES; i++) s += {U}_W1T[j][i] * z[i];\n"
            f"    h[j] = s > 0.0f ? s : 0.0f;\n"
            f"  }}\n"
            f"  int best = 0;\n"
            f"  for (int c = 0; c < N_CLASSES; c++) {{\n"
            f"    float s = {U}_B2[c];\n"
            f"    for (int j = 0; j < {H}; j++) s += {U}_W2T[c][j] * h[j];\n"
            f"    lg[c] = s;\n"
            f"    if (s > lg[best]) best = c;\n"
            f"  }}\n"
            f"  float sum = 0.0f;\n"
            f"  for (int c = 0; c < N_CLASSES; c++) sum += expf(lg[c] - lg[best]);\n"
            f"  *conf = 1.0f / sum;\n"
            f"  return best;\n}}\n")
    n = sum(a.size for a in (W1, W2, *mlp.intercepts_)) + 2 * len(sc.mean_)
    return consts, code, 4 * n


def emu_m3(est, X):
    sc, mlp = est.named_steps["standardscaler"], est.named_steps["mlpclassifier"]
    W1, W2 = (w.astype(f32) for w in mlp.coefs_)
    Z = (X - sc.mean_.astype(f32)) / sc.scale_.astype(f32)
    Hh = np.maximum(Z @ W1 + mlp.intercepts_[0].astype(f32), 0)
    L = Hh @ W2 + mlp.intercepts_[1].astype(f32)
    return L.argmax(1), softmax_max(L)


def softmax_max(L):
    L = L.astype(np.float64)
    e = np.exp(L - L.max(1, keepdims=True))
    return (e.max(1) / e.sum(1)).astype(f32)


# ---------------------------------------------------------------- main
def main(window="w128", rate="r50"):
    d = load_har()
    Xte, yte = extract(quantize(d["Xte"]), window, rate), d["yte"]
    sel = json.load(open(ROOT / "golden" / "test_selection.json"))["har_test_indices"]
    G = json.load(open(ROOT / "golden" / "golden_v1.json"))
    key = f"{window}_{rate}"
    Xg = np.array([w["features"][key] for w in G["windows"]], dtype=f32)
    gidx = [w["window_idx"] for w in G["windows"]]

    golden_h = ("// generated by ml/export_c.py - do not edit\n#pragma once\n#include <stdint.h>\n"
                f"#ifndef GOLDEN_N\n#define GOLDEN_N {len(Xg)}\n#endif\n"
                "// expected output of each model on the golden windows (features from golden_v1)\n")
    golden_j = {}

    for mid, fn in (("M1", None), ("M2", export_m2), ("M3", export_m3)):
        cid = f"{mid}_{window}_{rate}_fp32"
        out = ROOT / "exports" / cid
        est = joblib.load(out / "model.joblib")
        P, U = cid.lower(), cid.upper()

        if mid == "M1":
            code, nbytes = export_m1(est, P, U)
            consts, emu = "", emu_m1
        else:
            consts, code, nbytes = fn(est, P, U)
            emu = emu_m2 if mid == "M2" else emu_m3

        # ---- verification: C-equivalent float32 model vs sklearn on ALL test windows
        ref = est.predict(Xte)
        lab, conf = emu(est, Xte)
        mism = int((lab != ref).sum())
        print(f"{cid}: C-equivalent vs sklearn: {mism} mismatches / {len(ref)}"
              f"  (acc {float((lab == yte).mean()):.4f}, ~{nbytes} B params)")
        assert mism == 0 if mid == "M1" else mism <= 3, "export does not match sklearn"

        header = (f"// generated by ml/export_c.py - do not edit\n// config_id: {cid}\n"
                  "// input: 24 float32 features in feature_spec.md order; returns class 0..5, conf in [0,1]\n"
                  "#pragma once\n#include <math.h>\n#ifndef N_FEATURES\n#define N_FEATURES 24\n#endif\n"
                  "#ifndef N_CLASSES\n#define N_CLASSES 6\n#endif\n" + consts + code)
        write(out / f"model_{cid}.h", header)

        # ---- expected outputs for A
        rows = ["window_idx,label,conf"] + [f"{i},{lab_},{c:.6f}" for i, (lab_, c) in
                                             enumerate(zip(lab[sel], conf[sel]))]
        write(out / "board100_pred.csv", "\n".join(rows) + "\n")
        gl, gc = emu(est, Xg)
        golden_h += (f"static const uint8_t GOLDEN_{U}_LABEL[GOLDEN_N] = {{{','.join(map(str, gl))}}};\n"
                     f"static const float GOLDEN_{U}_CONF[GOLDEN_N] = {{{','.join(cf(c) for c in gc)}}};\n")
        golden_j[cid] = {"label": gl.tolist(), "conf": [float(c) for c in gc]}

        meta = json.loads((out / "meta.json").read_text())
        meta.update(param_bytes=int(nbytes), header_bytes=(out / f"model_{cid}.h").stat().st_size,
                    export_mismatches_vs_sklearn=mism)
        write(out / "meta.json", json.dumps(meta, indent=2))

        tag = "" if (window, rate) == ("w128", "r50") else f"_{key}"
        write(ROOT / "golden" / f"model_golden_v1{tag}.h", golden_h)
        write(ROOT / "golden" / f"model_golden_v1{tag}.json",
            json.dumps({"golden_window_idx": gidx, "models": golden_j}))
    print("ALL EXPORT CHECKS PASSED")


if __name__ == "__main__":
    kw = {a[2:]: sys.argv[i + 1] for i, a in enumerate(sys.argv) if a in ("--window", "--rate")}
    main(**kw)