"""Load raw UCI HAR total_acc windows. Subject-wise split as provided by UCI."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HAR = ROOT / "data" / "UCI HAR Dataset"
CACHE = ROOT / "data" / "har_total_acc.npz"
ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]


def _load_split(split):
    sig = HAR / split / "Inertial Signals"
    X = np.stack([np.loadtxt(sig / f"total_acc_{a}_{split}.txt") for a in "xyz"],
                 axis=-1)                                   # (N, 128, 3), float64, in g
    y = np.loadtxt(HAR / split / f"y_{split}.txt", dtype=int) - 1        # 0..5
    s = np.loadtxt(HAR / split / f"subject_{split}.txt", dtype=int)
    return X, y, s


def load_har(use_cache=True):
    """Returns dict with Xtr, ytr, str_, Xte, yte, ste."""
    if use_cache and CACHE.exists():
        d = np.load(CACHE)
        return {k: d[k] for k in d.files}
    Xtr, ytr, st = _load_split("train")
    Xte, yte, se = _load_split("test")
    out = dict(Xtr=Xtr, ytr=ytr, str_=st, Xte=Xte, yte=yte, ste=se)
    np.savez_compressed(CACHE, **out)       # data/ is gitignored
    return out


if __name__ == "__main__":
    load_har(use_cache=False)
    print("cached to", CACHE)