from arduino.app_utils import *
import glob, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
done = False

CODE = r'''
import sys, time, warnings, os
import numpy as np, joblib
import sklearn, sklearn.ensemble, sklearn.tree
def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"):
            return int(l.split()[1]) / 1024
path = sys.argv[1]
r0 = rss(); t0 = time.perf_counter()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    if path.endswith(".npz"):
        d = np.load(path)
        obj = {k: d[k] for k in d.files}
        info = "npz arrays=%d" % len(obj)
    else:
        obj = joblib.load(path)
        info = type(obj).__name__
        if hasattr(obj, "predict") and hasattr(obj, "n_features_in_"):
            obj.predict(np.zeros((1, obj.n_features_in_)))
            info += " predict_ok n_features=%d" % obj.n_features_in_
dt = time.perf_counter() - t0; r1 = rss()
print("OK %s | file %.2f MB | load %.2f s | RSS before %.1f MB -> after %.1f MB | growth %.1f MB"
      % (info, os.path.getsize(path) / 1e6, dt, r0, r1, r1 - r0))
for x in w:
    print("WARNING:", str(x.message)[:200])
'''

def loop():
    global done
    if done:
        time.sleep(1)
        return
    done = True
    import sklearn
    print("board sklearn:", sklearn.__version__, flush=True)
    paths = sorted(glob.glob(os.path.join(HERE, "models", "*", "model.joblib")) +
                   glob.glob(os.path.join(HERE, "models", "*", "model.npz")))
    if not paths:
        print("no model files found under python/models/*/", flush=True)
    for p in paths:
        print(os.path.relpath(p, HERE), flush=True)
        try:
            r = subprocess.run([sys.executable, "-c", CODE, p], capture_output=True, text=True, timeout=300)
            print("  ", r.stdout.strip() or "(no output)", flush=True)
            if r.returncode != 0:
                print("   ERROR:", r.stderr.strip()[-400:], flush=True)
        except Exception as e:
            print("   FAILED:", e, flush=True)
    print("=== done ===", flush=True)

App.run(user_loop=loop)