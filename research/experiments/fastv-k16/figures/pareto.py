"""DocVQA and MMBench vs prefill TFLOPs at NEO-2B: FastV-2, FastV-16, PyramidDrop-16 (2x/4x/8x) + baseline.
Comparators from tab:results-2B in VTC-Native-Eval.tex; FastV-16 from e002 (None = cell pending); TF from flops.csv.
    python pareto.py   # writes pareto_2b.png next to this file
"""
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = {"tf": 13.55, "DocVQA": 89.86, "MMBench": 76.20}
ROWS = {  # label: (color, [(ratio, tf, DocVQA, MMBench)])
    "FastV-2": ("#2a78d6", [(2, 6.19, 82.00, 73.80), (4, 3.23, 63.92, 65.46), (8, 1.94, 39.99, 56.62)]),
    "PyramidDrop-16": ("#eb6834", [(2, 8.92, 81.54, 74.22), (4, 7.01, 55.87, 69.75), (8, 6.24, 29.68, 58.16)]),
    "FastV-16": ("#1baf7a", [(2, 8.905, 85.75, 76.03), (4, 7.042, 69.07, 73.63), (8, 6.226, 42.63, 69.85)]),
    "FastV-24": ("#eda100", [(2, 10.456, 86.92, 75.26)]),  # e003, 2x only
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
for ax, metric in zip(axes, ["DocVQA", "MMBench"]):
    k = 2 if metric == "DocVQA" else 3
    for label, (color, pts) in ROWS.items():
        pts = [p for p in pts if p[k] is not None]
        if not pts:
            continue
        ax.plot([p[1] for p in pts], [p[k] for p in pts], "-o", color=color, lw=2, ms=8, label=label,
                markeredgecolor="white", markeredgewidth=2)
        for p in pts:
            ax.annotate(f"{p[0]}x", (p[1], p[k]), textcoords="offset points", xytext=(6, -12), fontsize=8, color="#555")
    ax.plot([BASE["tf"]], [BASE[metric]], "s", color="#444", ms=8, label="NEO-2B-SFT (no VTC)")
    ax.set_xlabel("prefill TFLOPs (2048x2048 image)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs cost, NEO-2B", fontsize=11)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].legend(frameon=False, fontsize=9, loc="lower right")
out = osp.join(osp.dirname(osp.abspath(__file__)), "pareto_2b.png")
fig.savefig(out, dpi=150)
print("wrote", out)
