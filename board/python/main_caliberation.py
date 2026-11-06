from arduino.app_utils import *
import os, re, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import linux_infer as li
from features import features_from

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "models")
done = False
COMBOS = [("w128", "r50", 128, 1), ("w128", "r25", 128, 2), ("w64", "r50", 64, 1), ("w64", "r25", 64, 2)]
MODELS_W = ["M1", "M2", "M3", "M4", "M4S", "M5"]   # from_window
MODELS_F = ["M1", "M2", "M3", "M4S"]               # from_features


def load_windows():
    txt = open(os.path.join(HERE, "test_windows.h")).read().split("= {", 1)[1]
    return np.array(re.findall(r"-?\d+", txt), dtype=np.int16).reshape(-1, 128, 3)


def gov():
    try:
        return open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except Exception:
        return "unknown"


def med(fn, reps=5, n_win=10):
    for _ in range(5):
        fn(0)
    t = []
    for i in range(n_win):
        for _ in range(reps):
            t0 = time.perf_counter(); fn(i); t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t)), len(t)


def loop():
    global done
    if done:
        time.sleep(1); return
    done = True
    time.sleep(3)
    W = load_windows()
    print("governor:", gov(), "windows:", W.shape, flush=True)
    rows = []
    for wn, rn, span, step in COMBOS:
        Wc = np.ascontiguousarray(W[:, :span:step, :])          # (100, N, 3)
        F = features_from(Wc).astype(np.float32)               # (100, 24)
        for mid in MODELS_W:
            m = li.load(os.path.join(MD, f"{mid}_{wn}_{rn}_fp32"), mid)
            ms, n = med(lambda i: li.predict(mid, m, Wc[i:i+1], rn))
            rows.append(("from_window", mid, wn, rn, ms))
            print(f"from_window {mid} {wn} {rn}: {ms:.3f} ms (n={n})", flush=True)
            if mid in MODELS_F:
                ms, n = med(lambda i: li.predict_features(mid, m, F[i:i+1]))
                rows.append(("from_features", mid, wn, rn, ms))
                print(f"from_features {mid} {wn} {rn}: {ms:.3f} ms (n={n})", flush=True)
    print("=== CSV ===", flush=True)
    print("stage,model,window,rate,ms", flush=True)
    for r in rows:
        print(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.4f}", flush=True)
    print("=== done === governor:", gov(), flush=True)


App.run(user_loop=loop)