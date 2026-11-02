"""Train M4 (RF, stat+FFT), M4S (RF, stat only), M5 (1D-CNN). Subject-wise validation only.
Usage: python ml/train_linux.py"""
import json, time
from pathlib import Path
import numpy as np, sklearn, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data import load_har
from features import quantize, select
import linux_infer as LI

ROOT = Path(__file__).resolve().parents[1]
SEED = 0
COMBOS = [("w128", "r50"), ("w128", "r25"), ("w64", "r50"), ("w64", "r25")]
RF_GRID = {"max_depth": [8, 16], "min_samples_leaf": [3, 10]}      # 100 trees; capped to keep files small
COLS = ["config_id", "model", "window", "rate", "precision", "accuracy", "macro_f1",
        "accuracy_board100", "cv_macro_f1", "param_bytes", "header_bytes",
        "linux_only", "needs_raw_window"]


def write(path, text):
    with open(path, "w", newline="\n") as f:
        f.write(text)


def train_cnn(Qtr, ytr, gtr, Qte, epochs=60):
    """Small 1D-CNN. 4 training subjects are held out for epoch selection (subject-wise)."""
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    va = np.isin(gtr, rng.choice(np.unique(gtr), 4, replace=False))
    tr = ~va
    flat = Qtr[tr].astype(np.float32).reshape(-1, 3)
    mean, std = flat.mean(axis=0), flat.std(axis=0)

    def prep(Q):
        return torch.tensor(((Q.astype(np.float32) - mean) / std).transpose(0, 2, 1).copy())

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.c1, s.c2, s.fc = nn.Conv1d(3, 16, 5), nn.Conv1d(16, 32, 5), nn.Linear(32, 6)

        def forward(s, x):
            x = F.max_pool1d(F.relu(s.c1(x)), 2)
            x = F.max_pool1d(F.relu(s.c2(x)), 2)
            return s.fc(x.mean(dim=2))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt, yt = prep(Qtr[tr]), torch.tensor(ytr[tr].astype(np.int64))
    Xv, yv = prep(Qtr[va]), ytr[va]
    best, state = -1.0, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            F.cross_entropy(net(Xt[b]), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            f1 = f1_score(yv, net(Xv).argmax(1).numpy(), average="macro")
        if f1 > best:
            best, state = f1, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        torch_pred = net(prep(Qte)).argmax(1).numpy()
    g = lambda k: state[k].numpy().astype(np.float32)
    params = dict(mean=mean, std=std, w1=g("c1.weight"), b1=g("c1.bias"), w2=g("c2.weight"),
                  b2=g("c2.bias"), wf=g("fc.weight"), bf=g("fc.bias"))
    return params, best, torch_pred


def finish(cid, mid, window, rate, files, pred, conf, yte, sel, cv, params):
    out = ROOT / "exports" / cid
    acc, f1 = accuracy_score(yte, pred), f1_score(yte, pred, average="macro")
    np.savetxt(out / "confusion.csv", confusion_matrix(yte, pred, labels=range(6)), fmt="%d", delimiter=",")
    write(out / "board100_pred.csv", "window_idx,label,conf\n" + "\n".join(
        f"{i},{a},{c:.6f}" for i, (a, c) in enumerate(zip(pred[sel], conf[sel]))) + "\n")
    meta = dict(config_id=cid, model=mid, window=window, rate=rate, precision="fp32",
                accuracy=acc, macro_f1=f1, accuracy_board100=accuracy_score(yte[sel], pred[sel]),
                cv_macro_f1=cv, best_params=params, seed=SEED, sklearn=sklearn.__version__,
                param_bytes=int(sum((out / f).stat().st_size for f in files)),
                linux_only=True, needs_raw_window=mid in ("M4", "M5"))
    write(out / "meta.json", json.dumps(meta, indent=2, default=str))
    print(f"{cid}: acc {acc:.4f}  macro-F1 {f1:.4f}  val/CV {cv:.4f}  "
          f"model file {meta['param_bytes'] / 1024:.0f} KiB  {params if mid != 'M5' else ''}")


def main():
    t0 = time.time()
    d = load_har()
    sel = json.load(open(ROOT / "golden" / "test_selection.json"))["har_test_indices"]
    ytr, gtr, yte = d["ytr"], d["str_"], d["yte"]
    for window, rate in COMBOS:
        Qtr = select(quantize(d["Xtr"]), window, rate)
        Qte = select(quantize(d["Xte"]), window, rate)
        for mid in ("M4", "M4S", "M5"):
            cid = f"{mid}_{window}_{rate}_fp32"
            out = ROOT / "exports" / cid
            out.mkdir(parents=True, exist_ok=True)
            if mid in ("M4", "M4S"):
                st = mid == "M4S"
                gs = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
                                  RF_GRID, cv=GroupKFold(n_splits=5), scoring="f1_macro")
                gs.fit(LI.rf_features(Qtr, rate, st), ytr, groups=gtr)
                joblib.dump(gs.best_estimator_, out / "model.joblib", compress=3)
                model = LI.load(out, mid)                       # the board path: reload from disk
                pred, conf = LI.predict(mid, model, Qte, rate)
                ref = gs.best_estimator_.predict(LI.rf_features(Qte, rate, st))
                assert (pred == ref).all(), "linux_infer disagrees with sklearn"
                finish(cid, mid, window, rate, ["model.joblib"], pred, conf, yte, sel,
                       gs.best_score_, gs.best_params_)
            else:
                params, val_f1, tpred = train_cnn(Qtr, ytr, gtr, Qte)
                np.savez(out / "model.npz", **params)
                model = LI.load(out, mid)
                pred, conf = LI.predict(mid, model, Qte, rate)
                mism = int((pred != tpred).sum())
                print(f"  {cid}: NumPy vs PyTorch: {mism} mismatches / {len(pred)}")
                assert mism <= 3, "NumPy CNN does not match PyTorch"
                finish(cid, mid, window, rate, ["model.npz"], pred, conf, yte, sel, val_f1, "")

    rows = [json.loads(p.read_text()) for p in sorted((ROOT / "exports").glob("M*_*/meta.json"))]
    df = pd.DataFrame(rows).reindex(columns=COLS)
    df["linux_only"] = df["model"].isin(["M4", "M4S", "M5"])
    df["needs_raw_window"] = df["model"].isin(["M4", "M5"])
    df = df.sort_values(["window", "rate", "model", "precision"], ascending=[False, False, True, True])
    df.to_csv(ROOT / "results" / "model_accuracy.csv", index=False, lineterminator="\n")
    print(f"\n{len(df)} configurations -> results/model_accuracy.csv  ({time.time() - t0:.0f} s)")
    assert len(df) == 32, "expected 32 configurations (20 + 12)"


if __name__ == "__main__":
    main()