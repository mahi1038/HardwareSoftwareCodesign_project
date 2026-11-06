from arduino.app_utils import *
import csv, os, re, time

HERE = os.path.dirname(os.path.abspath(__file__))
done = False

# (k, name, precision, board100 csv, golden header, golden array prefix)
MODELS = [
    (0, "M1", "fp32", "board100_M1.csv",  "model_golden_v1.h",      "GOLDEN_M1_W128_R50_FP32"),
    (1, "M2", "fp32", "board100_M2.csv",  "model_golden_v1.h",      "GOLDEN_M2_W128_R50_FP32"),
    (2, "M3", "fp32", "board100_M3.csv",  "model_golden_v1.h",      "GOLDEN_M3_W128_R50_FP32"),
    (3, "M2", "int8", "board100_M2I.csv", "model_golden_int8_v1.h", "GOLDEN_M2_W128_R50_INT8"),  # ASSUMED
    (4, "M3", "int8", "board100_M3I.csv", "model_golden_int8_v1.h", "GOLDEN_M3_W128_R50_INT8"),  # ASSUMED
]


def ref_csv(fn):
    rows = list(csv.DictReader(open(os.path.join(HERE, fn))))
    return [int(r["label"]) for r in rows], [float(r["conf"]) for r in rows]


def arr(txt, name):
    m = re.search(name + r"\[[^\]]*\]\s*=\s*\{([^}]*)\}", txt)
    return [x.strip().rstrip("f") for x in m.group(1).split(",")]


def ref_golden(fn, prefix):
    txt = open(os.path.join(HERE, fn)).read()
    labels = [int(x) for x in arr(txt, prefix + "_LABEL")]
    confs = [float(x) for x in arr(txt, prefix + "_CONF")]
    return labels, confs


def loop():
    global done
    if done:
        time.sleep(1)
        return
    done = True
    time.sleep(3)
    print("ping_len:", Bridge.call("ping_len", "abc"), flush=True)
    mode = Bridge.call("mode")
    mask = Bridge.call("models_mask")
    print("MODE:", "100 board windows" if mode == 1 else "golden windows", " mask:", bin(mask), flush=True)

    for k, name, prec, csvfn, gfn, gprefix in MODELS:
        if not (mask >> k) & 1:
            continue
        tag = f"{name} {prec}"
        try:
            ref_l, ref_c = ref_csv(csvfn) if mode == 1 else ref_golden(gfn, gprefix)
        except Exception as e:
            print(f"{tag}: reference unavailable ({e}); skipping comparison", flush=True)
            ref_l = ref_c = None
        n = len(ref_l) if ref_l else (100 if mode == 1 else 23)
        flips, errs, worst, big = [], 0, (0.0, -1), 0
        for i in range(n):
            r = Bridge.call("model_run", i * 5 + k)
            if r < 0:
                errs += 1
                continue
            label, conf = r // 100000, (r % 100000) / 10000.0
            if ref_l:
                if label != ref_l[i]:
                    flips.append((i, label, ref_l[i]))
                d = abs(conf - ref_c[i])
                if d > worst[0]:
                    worst = (d, i)
                if d > 1e-3:
                    big += 1
        print(f"{tag}: windows={n} label_mismatches={len(flips)} errors={errs} "
              f"max_conf_diff={worst[0]:.5f} (window {worst[1]}) conf_diff>1e-3: {big}", flush=True)
        if flips:
            print(f"  (window, board, expected): {flips[:10]}", flush=True)
        print(f"{tag}: predict = {Bridge.call('model_time_us', k) / 1000.0:.2f} us (upper bound, incl. loop overhead)", flush=True)

    print("=== Feature timing (us per extraction) ===", flush=True)
    for c, nm in enumerate(["w128_r50", "w128_r25", "w64_r50", "w64_r25"]):
        print(nm, Bridge.call("feat_time_us", c) / 10.0, flush=True)
    print("=== done ===", flush=True)


App.run(user_loop=loop)