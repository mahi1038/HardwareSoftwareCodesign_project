from arduino.app_utils import *
import csv, os, re, struct, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import linux_infer as li

DEMO = ["P2_M5_w64_r50_fp32", "P2_M5_w64_r25_fp32", "P4_M3_w128_r50_int8"]   # <- put the selector's choices here
N = 20                                                   # windows per config, live

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "models")
COMBO = {("w128", "r50"): 0, ("w128", "r25"): 1, ("w64", "r50"): 2, ("w64", "r25"): 3}
SPAN = {"w128": 128, "w64": 64}
STEP = {"r50": 1, "r25": 2}
KIDX = {("M1", "fp32"): 0, ("M2", "fp32"): 1, ("M3", "fp32"): 2, ("M2", "int8"): 3, ("M3", "int8"): 4}
done = False


def labels():
    body = open(os.path.join(HERE, "labels.h")).read().split("= {", 1)[1].split("}")[0]
    return [int(x) for x in re.findall(r"-?\d+", body)]


def bench_avg():
    acc = {}
    for r in csv.DictReader(open(os.path.join(HERE, "measurements.csv"))):
        acc.setdefault(r["config_id"], []).append(float(r["e2e_ms"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def one(cid, idx, model):
    part, mid, wn, rn, prec = cid.split("_")
    combo, span, step = COMBO[(wn, rn)], SPAN[wn], STEP[rn]
    t0 = time.perf_counter()
    if part == "P4":
        r = Bridge.call("mcu_p4", idx * 100 + combo * 10 + KIDX[(mid, prec)])
        return int(r.split(",")[0]), (time.perf_counter() - t0) * 1e3
    if part == "P3":
        r = Bridge.call("mcu_feat", idx * 10 + combo)
        h = r.rsplit("|", 1)[0]
        f = np.array([struct.unpack(">f", bytes.fromhex(x))[0] for x in h.split(",")],
                     dtype=np.float32).reshape(1, -1)
        l, _ = li.predict_features(mid, model, f)
    elif part == "P2":
        h = Bridge.call("mcu_cut", idx * 10 + combo).rsplit("|", 1)[0]
        w = np.frombuffer(bytes.fromhex(h), dtype=">i2").astype(np.int16).reshape(1, -1, 3)
        l, _ = li.predict(mid, model, w, rn)
    else:  # P1
        h = Bridge.call("mcu_raw", idx).rsplit("|", 1)[0]
        w = np.frombuffer(bytes.fromhex(h), dtype=">i2").astype(np.int16).reshape(1, 128, 3)
        l, _ = li.predict(mid, model, np.ascontiguousarray(w[:, :span:step, :]), rn)
    return int(l[0]), (time.perf_counter() - t0) * 1e3


def loop():
    global done
    if done:
        time.sleep(1)
        return
    done = True
    time.sleep(3)
    lab, ref = labels(), bench_avg()
    print("=== LIVE DEMO ===", flush=True)
    for cid in DEMO:
        part, mid, wn, rn, prec = cid.split("_")
        model = None if part == "P4" else li.load(os.path.join(MD, f"{mid}_{wn}_{rn}_fp32"), mid)
        for i in range(3):
            one(cid, i, model)                       # warm-up
        t, ok = [], 0
        for i in range(N):
            p, ms = one(cid, i, model)
            t.append(ms)
            ok += int(p == lab[i])
        print(f"{cid}: live avg {sum(t)/N:.1f} ms | live accuracy {ok}/{N} | "
              f"benchmark avg {ref.get(cid, float('nan')):.1f} ms", flush=True)
    print("=== done ===", flush=True)


App.run(user_loop=loop)