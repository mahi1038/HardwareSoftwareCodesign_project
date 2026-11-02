from arduino.app_utils import *
import json, os, struct, time

HERE = os.path.dirname(os.path.abspath(__file__))
COMBOS = ["w128_r50", "w128_r25", "w64_r50", "w64_r25"]
done = False


def decode(s):
    return [struct.unpack(">f", bytes.fromhex(h))[0] for h in s.split(",")]


def expected(golden, g, c):
    # ADAPT to the real keys in golden_v1.json (see the inspect step)
    return golden["windows"][g]["features"][COMBOS[c]]


def check(got, ref):
    bad = []
    for i, (a, r) in enumerate(zip(got, ref)):
        sig, f = divmod(i, 6)
        if f in (2, 3, 5) and sig < 3:
            ok = (a == r)                      # integer-exact
        elif f == 5:
            ok = abs(a - r) <= 1               # mag zc: +-1 allowed
        else:
            ok = abs(a - r) <= max(1e-4 * abs(r), 1e-3)
        if not ok:
            bad.append((i, a, r))
    return bad


def loop():
    global done
    if done:
        time.sleep(1)
        return
    done = True

    with open(os.path.join(HERE, "golden_v1.json")) as fh:
        golden = json.load(fh)
    print("JSON top-level keys:", list(golden.keys()) if isinstance(golden, dict) else type(golden), flush=True)

    print("=== Golden test ===", flush=True)
    n_gold = len(golden["windows"]) if isinstance(golden, dict) and "windows" in golden else 23
    fails = 0
    for g in range(n_gold):
        row = []
        for c in range(4):
            try:
                got = decode(Bridge.call("feat_golden", g * 4 + c))
                bad = check(got, expected(golden, g, c))
            except Exception as e:
                print(f"window {g} combo {c} ERROR: {e}", flush=True)
                return
            row.append("PASS" if not bad else "FAIL")
            if bad:
                fails += 1
                print(f"  window {g} {COMBOS[c]} mismatches (idx, C, ref): {bad[:5]}", flush=True)
        print(f"window {g:2d}: " + "  ".join(f"{COMBOS[c]}={row[c]}" for c in range(4)), flush=True)
    print(f"TOTAL FAILS: {fails} of {n_gold * 4}", flush=True)

    print("=== Feature timing (per call, us) ===", flush=True)
    for c in range(4):
        print(COMBOS[c], Bridge.call("feat_time_us", c) / 10.0, flush=True)

    print("=== done ===", flush=True)


App.run(user_loop=loop)