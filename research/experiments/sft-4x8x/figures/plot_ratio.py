import csv, sys, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(sys.argv[1])
res = root / "VLMEvalKit/vlmeval/results"
out = pathlib.Path(sys.argv[2])

TEX_2X = {
    "fixed": {"MMBench": 74.39, "DocVQA": 37.44},
    "random": {"MMBench": 72.68, "DocVQA": 41.95},
    "drip": {"MMBench": 75.08, "DocVQA": 80.04},
}
SFT_KEY = {"fixed pooling + LLaVA@66K": "fixed", "random pruning + LLaVA@66K": "random",
           "DRIP + LLaVA@66K two-tailed 5:5": "drip"}
TF_KEY = {"fixed pooling": "fixed", "random pruning": "random", "ToMe": "tome"}

sft = {k: {2.0: v} for k, v in TEX_2X.items()}
for r in csv.DictReader(open(res / "training_based.csv")):
    if r["Model"] in SFT_KEY:
        sft[SFT_KEY[r["Model"]]][float(r["compression"])] = {m: float(r[m]) for m in ("MMBench", "DocVQA")}
tf = {k: {} for k in TF_KEY.values()}
for r in csv.DictReader(open(res / "training_free_2B.csv")):
    if r["Model"] in TF_KEY:
        tf[TF_KEY[r["Model"]]][float(r["compression"])] = {m: float(r[m]) for m in ("MMBench", "DocVQA")}

COL = {"fixed": "#2a78d6", "random": "#eb6834", "drip": "#1baf7a", "tome": "#eda100"}
NAME = {"fixed": "fixed pooling", "random": "random pruning", "drip": "DRIP 5:5", "tome": "ToMe"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
for ax, m, base in zip(axes, ("MMBench", "DocVQA"), (76.20, 89.86)):
    ax.axhline(base, color="#888", lw=1, ls=":")
    ax.text(8.0, base + 1, "NEO-2B-SFT", color="#555", fontsize=8, ha="right")
    for k, d in tf.items():
        xs = sorted(d); ax.plot(xs, [d[x][m] for x in xs], ls="--", lw=1.5, marker="o", ms=5,
                                mfc="white", color=COL[k], label=f"{NAME[k]} (training-free)")
    placed = []
    for k, d in sorted(sft.items(), key=lambda kv: -kv[1][max(kv[1])][m]):
        xs = sorted(d); ys = [d[x][m] for x in xs]
        ax.plot(xs, ys, ls="-", lw=2, marker="o", ms=7, color=COL[k], label=f"{NAME[k]} + SFT")
        ly = min([ys[-1]] + [p - 3.5 for p in placed])
        placed.append(ly)
        ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=(xs[-1] * 1.07, ly), textcoords="data",
                    fontsize=8, color="#333", va="center")
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8]); ax.set_xticklabels(["2x", "4x", "8x"])
    ax.set_xlim(1.8, 9.5); ax.set_title(m, fontsize=11); ax.set_xlabel("visual-token compression")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("score")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
fig.suptitle("NEO-2B: LLaVA@66K SFT (solid) vs training-free (dashed); 8x SFT pending", fontsize=10)
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig(out)
print(out)
