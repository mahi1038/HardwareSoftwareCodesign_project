from arduino.app_utils import *
import csv, os, re, time, platform, warnings, hashlib, struct
from datetime import datetime
import numpy as np
warnings.filterwarnings("ignore")
import psutil
import linux_infer as li

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "models")
OUT = os.path.join(HERE, "measurements_adapt.csv")
MEAS = os.path.join(HERE, "measurements.csv")
N_WIN, REPS, K_M3 = 100, 3, 2          # K_M3: M3 fp32 in the sketch dispatch

# (window, rate, combo code in the sketch, threshold)
ADAPT = [("w64", "r25", 3, 0.80), ("w128", "r50", 0, 0.65)]

SCHEMA = ["config_id", "window_idx", "rep", "e2e_ms", "mcu_us", "linux_ms", "bridge_ms",
          "pred_label", "true_label", "mcu_flash_B", "mcu_ram_B", "linux_rss_MB", "cpu_load",
          "cpu_governor", "py_version", "numpy_version", "date", "n_repeats",
          "spec_version", "model_sha", "n_board_windows"]
EXTRA = ["path", "conf", "theta", "p4_call_ms", "cut_call_ms", "m5_ms"]

ready = finished = False
queue, DONE = [], set()
W = LAB = SHA = CFG = None


def rss_mb():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"):
            return int(l.split()[1]) / 1024.0


def governor():
    try:
        return open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except Exception:
        return "unknown"


def load_labels():
    body = open(os.path.join(HERE, "labels.h")).read().split("= {", 1)[1].split("}")[0]
    return np.array([int(x) for x in re.findall(r"-?\d+", body)])


def cid_of(wn, rn, th):
    return f"ADAPT_M3_M5_{wn}_{rn}_th{th:.2f}"


def init():
    global LAB, SHA, CFG
    LAB = load_labels()
    CFG = {r["config_id"]: r for r in csv.DictReader(open(os.path.join(HERE, "configs_valid.csv")))}
    SHA = {r["config_id"]: r["model_sha"] for r in csv.DictReader(open(os.path.join(HERE, "model_sha_map.csv")))}
    mask = Bridge.call("combo_mask")
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            DONE.add((r["config_id"], int(r["window_idx"]), int(r["rep"])))
    for wn, rn, combo, th in ADAPT:
        cid = cid_of(wn, rn, th)
        if not (mask >> combo) & 1:
            print("skip (combo not compiled in this sketch):", cid, flush=True)
            continue
        if sum(1 for i in range(N_WIN) for p in range(REPS) if (cid, i, p) in DONE) >= N_WIN * REPS:
            continue
        queue.append((wn, rn, combo, th))
    print(f"combo_mask={mask:04b} configs to run: {len(queue)}", flush=True)
    psutil.cpu_percent(None)


def one_window(idx, combo, th_int, rn, m5):
    t0 = time.perf_counter()
    r = Bridge.call("mcu_p4", idx * 100 + combo * 10 + K_M3)
    t1 = time.perf_counter()
    lab, conf, us = r.split(",")
    lab, conf, us = int(lab), int(conf), int(us)
    if conf < th_int:                                   # escalate
        r2 = Bridge.call("mcu_cut", idx * 10 + combo)
        t2 = time.perf_counter()
        h, us2 = r2.rsplit("|", 1)
        w = np.frombuffer(bytes.fromhex(h), dtype=">i2").astype(np.int16).reshape(1, -1, 3)
        l, _ = li.predict("M5", m5, w, rn)
        t3 = time.perf_counter()
        final, path, us2 = int(l[0]), "escalated", int(us2)
    else:
        t2 = t3 = time.perf_counter()
        final, path, us2 = lab, "kept", 0
    return dict(final=final, m3=lab, conf=conf, path=path, mcu_us=us + us2,
                e2e=(t3 - t0) * 1e3, p4=(t1 - t0) * 1e3, cut=(t2 - t1) * 1e3, m5=(t3 - t2) * 1e3,
                bridge=max((t1 - t0) * 1e3 - us / 1e3, 0) + (max((t2 - t1) * 1e3 - us2 / 1e3, 0) if path == "escalated" else 0))


def compare_with_estimate(cid, wn, rn, rows):
    """B's estimate: per window, P4_M3 e2e + escalated * P2_M5 e2e, from measurements.csv."""
    try:
        acc = {}
        for r in csv.DictReader(open(MEAS)):
            if r["config_id"] in (f"P4_M3_{wn}_{rn}_fp32", f"P2_M5_{wn}_{rn}_fp32"):
                acc.setdefault((r["config_id"], int(r["window_idx"])), []).append(float(r["e2e_ms"]))
        mean = lambda k: sum(acc[k]) / len(acc[k])
        per = {}
        for r in rows:
            per.setdefault(r["window_idx"], []).append(r)
        est = meas = 0.0
        for i, rs in per.items():
            esc = rs[0]["path"] == "escalated"
            est += mean((f"P4_M3_{wn}_{rn}_fp32", i)) + (mean((f"P2_M5_{wn}_{rn}_fp32", i)) if esc else 0)
            meas += sum(x["e2e"] for x in rs) / len(rs)
        n = len(per)
        print(f"   estimate P4+esc*P2 = {est/n:.2f} ms | measured adaptive = {meas/n:.2f} ms | "
              f"extra = {(meas-est)/n:+.2f} ms per window", flush=True)
    except Exception as e:
        print("   (estimate comparison skipped:", repr(e)[:120], ")", flush=True)


def run_config(wn, rn, combo, th):
    cid = cid_of(wn, rn, th)
    th_int = int(round(th * 10000))
    m5 = li.load(os.path.join(MD, f"M5_{wn}_{rn}_fp32"), "M5")
    p4_row = CFG[f"P4_M3_{wn}_{rn}_fp32"]
    sha = hashlib.sha1((SHA[f"P4_M3_{wn}_{rn}_fp32"] + SHA[f"P2_M5_{wn}_{rn}_fp32"]).encode()).hexdigest()[:8]
    gov = governor()
    for i in range(3):                                   # warm-up: normal and forced-escalation path
        one_window(i, combo, th_int, rn, m5)
        one_window(i, combo, 10001, rn, m5)
    new_file = not os.path.exists(OUT)
    out_rows, mine = [], []
    try:
        for idx in range(N_WIN):
            for rep in range(REPS):
                if (cid, idx, rep) in DONE:
                    continue
                x = one_window(idx, combo, th_int, rn, m5)
                mine.append(dict(x, window_idx=str(idx), rep=rep))
                out_rows.append({
                    "config_id": cid, "window_idx": idx, "rep": rep, "e2e_ms": round(x["e2e"], 3),
                    "mcu_us": x["mcu_us"], "linux_ms": round(x["m5"], 3), "bridge_ms": round(x["bridge"], 3),
                    "pred_label": x["final"], "true_label": int(LAB[idx]),
                    "mcu_flash_B": p4_row["mcu_flash_total_B"], "mcu_ram_B": p4_row["mcu_ram_B"],
                    "linux_rss_MB": round(rss_mb(), 1), "cpu_load": psutil.cpu_percent(None),
                    "cpu_governor": gov, "py_version": platform.python_version(),
                    "numpy_version": np.__version__, "date": datetime.now().isoformat(timespec="seconds"),
                    "n_repeats": REPS, "spec_version": "v1", "model_sha": sha, "n_board_windows": 100,
                    "path": x["path"], "conf": round(x["conf"] / 10000, 4), "theta": th,
                    "p4_call_ms": round(x["p4"], 3), "cut_call_ms": round(x["cut"], 3), "m5_ms": round(x["m5"], 3)})
    finally:
        if out_rows:
            with open(OUT, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=SCHEMA + EXTRA)
                if new_file:
                    w.writeheader()
                w.writerows(out_rows)
            for r in out_rows:
                DONE.add((r["config_id"], r["window_idx"], r["rep"]))
    r0 = [m for m in mine if m["rep"] == 0]
    if not r0:
        return
    esc = [m for m in r0 if m["path"] == "escalated"]
    kept = [m for m in r0 if m["path"] == "kept"]
    avg = lambda L, k: sum(m[k] for m in L) / len(L) if L else float("nan")
    acc_final = sum(int(m["final"] == LAB[int(m["window_idx"])]) for m in r0)
    acc_m3 = sum(int(m["m3"] == LAB[int(m["window_idx"])]) for m in r0)
    print(f"{cid}: escalated {len(esc)}/{len(r0)} | acc adaptive {acc_final}/{len(r0)} vs M3-only {acc_m3}/{len(r0)} | "
          f"e2e kept {avg(kept,'e2e'):.1f} ms, escalated {avg(esc,'e2e'):.1f} ms, all {avg(mine,'e2e'):.1f} ms", flush=True)
    print(f"   escalated call split: p4 {avg(esc,'p4'):.1f} + cut {avg(esc,'cut'):.1f} + (M5 + parse) {avg(esc,'m5'):.1f} ms", flush=True)
    compare_with_estimate(cid, wn, rn, mine)


def loop():
    global ready, finished
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
    try:
        run_config(*cfg)
    except Exception as e:
        print("STOPPED at", cfg, "->", repr(e)[:200], flush=True)
        print("Stop the app (power-cycle the board if needed) and run again: it resumes.", flush=True)
        queue.clear()
        finished = True


App.run(user_loop=loop)