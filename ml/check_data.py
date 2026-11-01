from collections import Counter
import numpy as np
from data import load_har, ACTIVITIES

d = load_har()
Xtr, ytr, st = d["Xtr"], d["ytr"], d["str_"]
Xte, yte, se = d["Xte"], d["yte"], d["ste"]

# shapes
assert Xtr.shape[1:] == (128, 3) and Xte.shape[1:] == (128, 3), "window shape wrong"
assert len(Xtr) == len(ytr) == len(st) and len(Xte) == len(yte) == len(se)

# the key check: no subject in both train and test
overlap = set(np.unique(st)) & set(np.unique(se))
assert not overlap, f"LEAKAGE: subjects in both splits: {overlap}"

# all six classes in both splits
assert set(np.unique(ytr)) == set(np.unique(yte)) == set(range(6))

print(f"train: {Xtr.shape}, {len(np.unique(st))} subjects")
print(f"test : {Xte.shape}, {len(np.unique(se))} subjects")
print("train subjects:", sorted(np.unique(st)))
print("test  subjects:", sorted(np.unique(se)))
for name, y in (("train", ytr), ("test", yte)):
    c = Counter(y.tolist())
    print(name, {ACTIVITIES[k]: c[k] for k in range(6)})

# range check, needed later for int16 scaling (value * 1000 must fit in int16)
m = max(np.abs(Xtr).max(), np.abs(Xte).max())
print(f"max |value| = {m:.3f} g  ->  x1000 = {m*1000:.0f}  (int16 limit 32767)")
assert m * 1000 < 32767, "int16 x1000 scaling would saturate"
print("per-axis mean (g):", Xtr.reshape(-1, 3).mean(axis=0).round(3))
print("ALL CHECKS PASSED")