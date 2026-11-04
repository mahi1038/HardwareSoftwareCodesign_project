from arduino.app_utils import *
import csv, os, re, struct, time, platform, warnings
from datetime import datetime
import numpy as np
warnings.filterwarnings("ignore")
import psutil
import linux_infer as li
from features import features_from

SMOKE = False            # True: 4 configs x 5 windows x 1 rep -> measurements_smoke.csv
                        # False: full run -> measurements.csv

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "models")
N_WIN = 5 if SMOKE else 100
REPS = 1 if SMOKE else 3
OUT = os.path.join(HERE, "measurements_smoke.csv" if SMOKE else "measurements.csv")

SCHEMA = ["config_id", "window_idx", "rep", "e2e_ms", "mcu_us", "linux_ms", "bridge_ms",
          "pred_label", "true_label", "mcu_flash_B", "mcu_ram_B", "linux_rss_MB", "cpu_load",
          "cpu_governor", "py_version", "numpy_version", "date", "n_repeats",
          "spec_version", "model_sha", "n_board_windows"]
COMBO = {("w128", "r50"): 0, ("w128", "r25"): 1, ("w64", "r50"): 2, ("w64", "r25"): 3}
SPAN = {"w128": 128, "w64": 64}
STEP = {"r50": 1, "r25": 2}
KIDX = {("M1", "fp32"): 0, ("M2", "fp32"): 1, ("M3", "fp32"): 2, ("M2", "int8"): 3, ("M3", "int8"): 4}

ready = False
queue, DONE, MODELS = [], set(), {}
W = LAB = SHA = None
MASK = 0
total_cfgs = 0


def rss_mb():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"):
            return int(l.split()[1]) / 1024.0


def governor():
    try:
        return open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except Exception:
        return "unknown"


def load_windows():
    txt = open(os.path.join(HERE, "test_windows.h")).read().split("= {", 1)[1]
    return np.array(re.findall(r"-?\d+", txt), dtype=np.int16).reshape(-1, 128, 3)


def load_labels():
    body = open(os.path.join(HERE, "labels.h")).read().split("= {", 1)[1].split("}")[0]
    return np.array([int(x) for x in re.findall(r"-?\d+", body)])


def get_model(mid, wn, rn):
    key = (mid, wn, rn)
    if key not in MODELS:
        MODELS[key] = li.load(os.path.join(MD, f"{mid}_{wn}_{rn}_fp32"), mid)
    return MODELS[key]


def init():
    global W, LAB, SHA, MASK, DONE, queue, total_cfgs
    W, LAB = load_windows(), load_labels()
    print("windows:", W.shape, "labels:", len(LAB), "first 10:", LAB[:10].tolist(),
          "counts:", np.bincount(LAB).tolist(), flush=True)
    cfgs = list(csv.DictReader(open(os.path.join(HERE, "configs_valid.csv"))))
    SHA = {r["config_id"]: r["model_sha"] for r in csv.DictReader(open(os.path.join(HERE, "model_sha_map.csv")))}
    MASK = None
    t0 = time.time()
    while time.time() - t0 < 40:
        try:
            MASK = Bridge.call("combo_mask")
            break
        except Exception as e:
            print(f"waiting for MCU ({time.time()-t0:.0f}s): {str(e)[:60]}", flush=True)
            time.sleep(3)
    if MASK is None:
        raise SystemExit("MCU never registered its functions: image too big or MCU stuck")
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            DONE.add((r["config_id"], int(r["window_idx"]), int(r["rep"])))
    order = {"P4": 0, "P3": 1, "P2": 2, "P1": 3}
    cfgs.sort(key=lambda r: order[r["partition"]])
    if SMOKE:
        cfgs = [r for r in cfgs if r["model"] == "M1" and r["window"] == "w128" and r["rate"] == "r50"]
    for r in cfgs:
        have = sum(1 for i in range(N_WIN) for p in range(REPS) if (r["config_id"], i, p) in DONE)
        if have >= N_WIN * REPS:
            continue
        if r["partition"] == "P4" and not (MASK >> COMBO[(r["window"], r["rate"])]) & 1:
            print("skip (not compiled in this sketch):", r["config_id"], flush=True)
            continue
        queue.append(r)
    total_cfgs = len(queue)
    print(f"SMOKE={SMOKE} combo_mask={MASK:04b} configs to run: {total_cfgs}", flush=True)
    psutil.cpu_percent(None)


def one_call(cfg, idx, model):
    part, mid, wn, rn, prec = cfg["partition"], cfg["model"], cfg["window"], cfg["rate"], cfg["precision"]
    combo, span, step = COMBO[(wn, rn)], SPAN[wn], STEP[rn]
    t0 = time.perf_counter()
    if part == "P4":
        r = Bridge.call("mcu_p4", idx * 100 + combo * 10 + KIDX[(mid, prec)])
        t1 = time.perf_counter()
        lab, conf, us = r.split(",")
        return int(lab), (t1 - t0) * 1e3, int(us), 0.0
    if part == "P3":
        r = Bridge.call("mcu_feat", idx * 10 + combo)
        t1 = time.perf_counter()
        h, us = r.rsplit("|", 1)
        f = np.array([struct.unpack(">f", bytes.fromhex(x))[0] for x in h.split(",")],
                     dtype=np.float32).reshape(1, -1)
        l, c = li.predict_features(mid, model, f)
    elif part == "P2":
        r = Bridge.call("mcu_cut", idx * 10 + combo)
        t1 = time.perf_counter()
        h, us = r.rsplit("|", 1)
        w = np.frombuffer(bytes.fromhex(h), dtype=">i2").astype(np.int16).reshape(1, -1, 3)
        l, c = li.predict(mid, model, w, rn)
    else:  # P1
        r = Bridge.call("mcu_raw", idx)
        t1 = time.perf_counter()
        h, us = r.rsplit("|", 1)
        w = np.frombuffer(bytes.fromhex(h), dtype=">i2").astype(np.int16).reshape(1, 128, 3)
        w = np.ascontiguousarray(w[:, :span:step, :])
        l, c = li.predict(mid, model, w, rn)
    t2 = time.perf_counter()
    return int(l[0]), (t1 - t0) * 1e3, int(us), (t2 - t1) * 1e3


def run_config(cfg, n_done):
    cid, part = cfg["config_id"], cfg["partition"]
    model = None if part == "P4" else get_model(cfg["model"], cfg["window"], cfg["rate"])
    gov, rows, correct, n0, feat_flips = governor(), [], 0, 0, 0
    for i in range(3):                       # warm-up, not recorded
        one_call(cfg, i, model)
    new_file = not os.path.exists(OUT)
    try:
        for idx in range(N_WIN):
            for rep in range(REPS):
                if (cid, idx, rep) in DONE:
                    continue
                pred, call_ms, mcu_us, linux_ms = one_call(cfg, idx, model)
                if part == "P3" and rep == 0:       # does MCU feature parsing agree with NumPy?
                    span, step = SPAN[cfg["window"]], STEP[cfg["rate"]]
                    F = features_from(np.ascontiguousarray(W[idx:idx+1, :span:step, :])).astype(np.float32)
                    lp, _ = li.predict_features(cfg["model"], model, F)
                    feat_flips += int(lp[0]) != pred
                if rep == 0:
                    n0 += 1; correct += int(pred == LAB[idx])
                rows.append({
                    "config_id": cid, "window_idx": idx, "rep": rep,
                    "e2e_ms": round(call_ms + linux_ms, 3), "mcu_us": mcu_us,
                    "linux_ms": round(linux_ms, 3), "bridge_ms": round(max(call_ms - mcu_us / 1e3, 0.0), 3),
                    "pred_label": pred, "true_label": int(LAB[idx]),
                    "mcu_flash_B": cfg["mcu_flash_total_B"], "mcu_ram_B": cfg["mcu_ram_B"],
                    "linux_rss_MB": round(rss_mb(), 1), "cpu_load": psutil.cpu_percent(None),
                    "cpu_governor": gov, "py_version": platform.python_version(),
                    "numpy_version": np.__version__, "date": datetime.now().isoformat(timespec="seconds"),
                    "n_repeats": REPS, "spec_version": "v1", "model_sha": SHA[cid], "n_board_windows": 100})
    finally:                                  # save what we have, even on a bridge failure
        if rows:
            with open(OUT, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=SCHEMA)
                if new_file:
                    w.writeheader()
                w.writerows(rows)
            for r in rows:
                DONE.add((r["config_id"], r["window_idx"], r["rep"]))
    e2e = sum(r["e2e_ms"] for r in rows) / max(len(rows), 1)
    extra = f" | P3 feature-parse label flips vs NumPy: {feat_flips}/{n0}" if part == "P3" else ""
    print(f"[{n_done}/{total_cfgs}] {cid}: rows={len(rows)} avg e2e={e2e:.1f} ms "
          f"acc(rep0)={correct}/{n0} (B's acc col: {float(cfg['accuracy']):.3f}){extra}", flush=True)


finished = False
n_run = 0


def loop():
    global ready, finished, n_run
    if not ready:
        time.sleep(3)
        init()
        ready = True
        return
    if not queue:
        if not finished:
            print("=== ALL DONE ===  file:", OUT, flush=True)
            finished = True
        time.sleep(5)
        return
    cfg = queue.pop(0)
    n_run += 1
    try:
        run_config(cfg, n_run)
    except Exception as e:
        print("STOPPED at", cfg["config_id"], "->", repr(e)[:200], flush=True)
        print("Bridge probably hung. Stop the app and run it again: it resumes where it stopped.", flush=True)
        queue.clear()
        finished = True


App.run(user_loop=loop)