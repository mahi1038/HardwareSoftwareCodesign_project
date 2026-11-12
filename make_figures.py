"""Regenerates all five report figures from results/ CSVs (needs only numpy + matplotlib)."""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
COL = {"P1": "tab:red", "P2": "tab:orange", "P3": "tab:green", "P4": "tab:blue"}
plt.rcParams.update({"font.size": 10})

# ---------- load predictions.csv (accuracy, predicted and measured latency for all 84 configs)
pr = list(csv.DictReader(open(f"{R}/predictions.csv")))
for r in pr:
    r["acc"] = float(r["accuracy"]); r["lat"] = float(r["meas_ms"]); r["pred"] = float(r["pred_ms"])
assert len(pr) == 84

def pareto(rows):
    best, front = -1.0, []
    for r in sorted(rows, key=lambda r: (r["lat"], -r["acc"])):
        if r["acc"] > best:
            front.append(r); best = r["acc"]
    return front

front = pareto(pr)

# ---------- Figure 1: latency by cut point
fig, ax = plt.subplots(figsize=(7, 4))
rng = np.random.default_rng(0)
for i, p in enumerate(["P1", "P2", "P3", "P4"]):
    ys = [r["lat"] for r in pr if r["partition"] == p]
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(ys)), ys, color=COL[p], alpha=0.8, s=25)
ax.set_yscale("log")
ax.set_xticks(range(4))
ax.set_xticklabels(["P1\nfull window", "P2\ncut window", "P3\nfeatures", "P4\nlabel only"])
ax.set_ylabel("End-to-end latency (ms, log scale)")
ax.set_title("Latency by cut point (each dot = one configuration)")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_latency_by_cut.png", dpi=200); plt.close(fig)

# ---------- Figure 2: accuracy vs latency, Pareto front
fig, ax = plt.subplots(figsize=(7.6, 4.5))
for p in ["P1", "P2", "P3", "P4"]:
    rows = [r for r in pr if r["partition"] == p]
    ax.scatter([r["lat"] for r in rows], [r["acc"] for r in rows], color=COL[p], label=p, alpha=0.8, s=25)
ax.step([r["lat"] for r in front], [r["acc"] for r in front], where="post", color="black", lw=1.2, label="Pareto front")
ax.set_xscale("log")
ax.set_xlabel("End-to-end latency (ms, log scale)")
ax.set_ylabel("Accuracy (held-out subjects)")
ax.set_title("Accuracy vs latency, 84 configurations")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.savefig(f"{OUT}/fig2_pareto.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- Figure 3: Experiment A (values from the selector output)
lim = np.array([1000, 200, 150, 100, 75, 50, 30, 20, 10, 8, 7], float)
ca = np.array([0.8670, 0.8670, 0.8670, 0.8670, 0.8629, 0.8629, 0.8483, 0.8483, 0.8483, 0.8483, 0.8327])
cl = np.array([80.1, 80.1, 80.1, 80.1, 45.3, 45.3, 7.5, 7.5, 7.5, 7.5, 6.9])
naive = [r for r in pr if r["config_id"] == "P1_M5_w128_r50_fp32"][0]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(lim, ca, "o-", label="framework choice")
a1.axhline(naive["acc"], ls="--", color="gray", label="always most accurate")
a1.set_xscale("log"); a1.invert_xaxis()
a1.set_xlabel("Latency limit (ms)"); a1.set_ylabel("Accuracy of chosen configuration")
a1.set_title("Accuracy kept as the limit tightens"); a1.grid(True, alpha=0.3); a1.legend(loc="lower left")
a2.plot(lim, cl, "o-", label="framework choice")
a2.plot(lim, lim, ":", color="red", label="latency limit")
a2.axhline(naive["lat"], ls="--", color="gray", label="always most accurate")
a2.set_xscale("log"); a2.set_yscale("log"); a2.invert_xaxis()
a2.set_xlabel("Latency limit (ms)"); a2.set_ylabel("Latency of chosen configuration (ms)")
a2.set_title("Latency of the chosen configuration"); a2.grid(True, alpha=0.3); a2.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_expA.png", dpi=200); plt.close(fig)

# ---------- Figure 4: adaptive threshold sweep vs fixed configurations
ec = list(csv.DictReader(open(f"{R}/expC_offline.csv")))
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.scatter([r["lat"] for r in pr], [r["acc"] for r in pr], color="lightgray", s=18, label="fixed configurations (84)", zorder=1)
ax.step([r["lat"] for r in front], [r["acc"] for r in front], where="post", color="black", lw=1.2, label="fixed-configuration Pareto front", zorder=2)
for first, c in [("M3", "tab:purple"), ("M2", "tab:cyan")]:
    rows = [r for r in ec if r["first"] == first and float(r["threshold"]) < 1.0]
    x = [float(r["exp_lat_ms"]) for r in rows]; y = [float(r["accuracy"]) for r in rows]
    ax.plot(x, y, "o-", color=c, ms=4, label=f"adaptive: {first} first, escalate to M5", zorder=3)
    for r in rows:
        if first == "M3" and float(r["threshold"]) in (0.6, 0.7, 0.8, 0.9):
            ax.annotate(f"{float(r['threshold']):.1f}", (float(r["exp_lat_ms"]), float(r["accuracy"])),
                        textcoords="offset points", xytext=(4, -10), fontsize=8, color=c)
ax.set_xscale("log")
ax.set_xlabel("Expected latency (ms, log scale)")
ax.set_ylabel("Accuracy (held-out subjects)")
ax.set_title("Adaptive inference vs fixed configurations (labels: confidence threshold)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
fig.savefig(f"{OUT}/fig4_adaptive.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- Figure 5: predictor accuracy and pruning
ed = list(csv.DictReader(open(f"{R}/expD_pruning.csv")))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
for p in ["P1", "P2", "P3", "P4"]:
    rows = [r for r in pr if r["partition"] == p]
    if p == "P1":   # P1 and P2 overlap near 148 ms, so draw P1 as larger hollow rings
        a1.scatter([r["lat"] for r in rows], [r["pred"] for r in rows], facecolors="none", edgecolors=COL[p], s=60, lw=1.2, label=p)
    else:
        a1.scatter([r["lat"] for r in rows], [r["pred"] for r in rows], color=COL[p], s=22, alpha=0.85, label=p)
lo, hi = min(r["lat"] for r in pr) * 0.9, max(r["lat"] for r in pr) * 1.1
a1.plot([lo, hi], [lo, hi], "k--", lw=1)
a1.set_xscale("log"); a1.set_yscale("log")
err = np.array([abs(float(r["err_pct"])) for r in pr])
a1.set_xlabel("Measured latency (ms)"); a1.set_ylabel("Predicted latency (ms)")
a1.set_title(f"Predicted vs measured (mean error {err.mean():.1f}%, max {err.max():.1f}%)")
a1.grid(True, alpha=0.3); a1.legend()
m = [float(r["margin"]) * 100 for r in ed]
runs = [int(r["runs"]) for r in ed]
rec = [float(r["front_recall"]) for r in ed]
a2.plot(runs, rec, "o-", color="tab:blue")
seen = set()
offsets = {0.0: (-4, 10), 1.0: (4, -16)}     # keep the 0% and 1% labels apart
for x, y, mm in zip(runs, rec, m):
    if (x, y) in seen:
        continue
    seen.add((x, y))
    lab = f"margin {mm:g}%" if mm < 2 else "margin 2% and above"
    off = offsets.get(mm, (6, -14))
    a2.annotate(lab, (x, y), textcoords="offset points", xytext=off, fontsize=8,
                ha="right" if mm == 0.0 else "left")
a2.axvline(84, color="gray", ls="--"); a2.text(83.5, 0.885, "all 84 runs", ha="right", fontsize=8, color="gray")
a2.set_xlim(10, 88); a2.set_ylim(0.8, 1.04)
a2.set_xlabel("On-board runs needed (of 84)"); a2.set_ylabel("Pareto front recovered (fraction)")
a2.set_title("Pruning: recall vs runs (labels: safety margin)")
a2.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_predictor_pruning.png", dpi=200); plt.close(fig)
print("saved 5 figures in", OUT)