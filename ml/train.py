"""Train M1-M3 for one (window, rate). Tuning is subject-wise (GroupKFold) on train subjects only.
Usage: python ml/train.py [M1 M2 M3] [--window w128] [--rate r50]"""
import json, sys, time
from pathlib import Path
import numpy as np, sklearn, joblib
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data import load_har, ACTIVITIES
from features import quantize, extract, N_FEATURES

ROOT = Path(__file__).resolve().parents[1]
SEED = 0

MODELS = {
    "M1": (DecisionTreeClassifier(random_state=SEED),
           {"max_depth": [3, 4, 5], "min_samples_leaf": [1, 5, 20]}),
    "M2": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
           {"logisticregression__C": [0.01, 0.1, 1, 10]}),
    "M3": (make_pipeline(StandardScaler(),
                         MLPClassifier((32,), max_iter=800, random_state=SEED)),
           {"mlpclassifier__alpha": [1e-4, 1e-2, 1e-1]}),
}


def get_data(window, rate):
    d = load_har()
    return (extract(quantize(d["Xtr"]), window, rate), d["ytr"], d["str_"],
            extract(quantize(d["Xte"]), window, rate), d["yte"])


def train_one(mid, window="w128", rate="r50", verbose=True):
    Xtr, ytr, gtr, Xte, yte = get_data(window, rate)
    est, grid = MODELS[mid]
    t0 = time.time()
    gs = GridSearchCV(est, grid, cv=GroupKFold(n_splits=5), scoring="f1_macro", n_jobs=-1)
    gs.fit(Xtr, ytr, groups=gtr)
    pred = gs.predict(Xte)
    acc, f1 = accuracy_score(yte, pred), f1_score(yte, pred, average="macro")
    cm = confusion_matrix(yte, pred, labels=range(6))

    # the same 100 windows that are on the board (golden/test_selection.json)
    sel = json.load(open(ROOT / "golden" / "test_selection.json"))["har_test_indices"]
    acc100 = accuracy_score(yte[sel], pred[sel])

    cid = f"{mid}_{window}_{rate}_fp32"
    out = ROOT / "exports" / cid
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_estimator_, out / "model.joblib")
    np.savetxt(out / "confusion.csv", cm, fmt="%d", delimiter=",")
    meta = dict(config_id=cid, model=mid, window=window, rate=rate, precision="fp32",
                n_features=N_FEATURES, accuracy=acc, macro_f1=f1, accuracy_board100=acc100,
                cv_macro_f1=gs.best_score_, best_params=gs.best_params_, seed=SEED,
                n_train=len(ytr), n_test=len(yte), sklearn=sklearn.__version__,
                train_seconds=round(time.time() - t0, 1))
    (out / "meta.json").write_text(json.dumps(meta, indent=2), newline="\n")
    if verbose:
        print(f"\n== {cid} ==  best {gs.best_params_}")
        print(f"CV macro-F1 (subject-wise): {gs.best_score_:.4f}")
        print(f"TEST (9 held-out subjects, n={len(yte)}): acc={acc:.4f}  macro-F1={f1:.4f}")
        print(f"board-100 subset acc: {acc100:.4f}")
        print("confusion (rows=true, cols=pred):", [a[:5] for a in ACTIVITIES])
        print(cm)
    return meta


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ids = [a for a in args if a in MODELS] or list(MODELS)
    kw = {}
    for i, a in enumerate(sys.argv):
        if a in ("--window", "--rate"):
            kw[a[2:]] = sys.argv[i + 1]
    for m in ids:
        train_one(m, **kw)