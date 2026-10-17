import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv("Fixed_pooling_trend.csv")

# Benchmark groups
coarse_benchmarks = ["MMBench", "MME", "MMMU", "RealWorldQA"]
ocr_benchmarks = ["TextVQA", "DocVQA", "OCRBench", "ChartQA"]

all_benchmarks = coarse_benchmarks + ocr_benchmarks

# ============================================================
# Normalize relative to NEO-2B-SFT
# ============================================================

baseline = df[df["Model"] == "NEO-2B-SFT"].iloc[0]

for benchmark in all_benchmarks:
    df[benchmark] = df[benchmark] / baseline[benchmark] * 100

# Aggregate relative performance
df["Coarse"] = df[coarse_benchmarks].mean(axis=1)
df["OCR"] = df[ocr_benchmarks].mean(axis=1)

# ============================================================
# Model settings
# ============================================================

settings = {
    "fixed pooling": {
        "label": "Fixed Pooling",
        "color": "tab:blue",
        "marker": "o",
        "linestyle": "-"
    },
    "fixed poooling + KL": {
        "label": "Fixed Pooling + KL",
        "color": "tab:orange",
        "marker": "s",
        "linestyle": "-"
    }
}

# ============================================================
# Plot
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
})

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5),
    sharey=True
)

groups = [
    ("Coarse-Grained Benchmarks", "Coarse"),
    ("OCR Benchmarks", "OCR")
]

for ax, (title, metric) in zip(axes, groups):

    # Baseline reference
    ax.axhline(
        100,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7
    )

    ax.scatter(
        0, 100,
        marker="*",
        s=250,
        color="black",
        zorder=10,
        label="NEO-2B-SFT"
    )

    # Plot each setting independently
    for model_name, style in settings.items():

        subset = df[df["Model"] == model_name].copy()
        subset = subset.sort_values("training_data")

        if subset.empty:
            continue

        ax.plot(
            subset["training_data"],
            subset[metric],
            marker=style["marker"],
            markersize=8,
            linewidth=2.5,
            linestyle=style["linestyle"],
            color=style["color"],
            label=style["label"],
            zorder=5
        )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Training Data (K Samples)")

    ax.set_xticks([0, 66, 332])
    ax.set_xticklabels(["0K", "66K", "332K"])

    ax.set_xlim(-15, 350)
    ax.set_ylim(0, 110)

    ax.grid(
        True,
        linestyle=":",
        alpha=0.4
    )

    ax.legend(loc="best", frameon=True)

axes[0].set_ylabel(
    "Average Relative Performance to NEO-2B-SFT (%)"
)

fig.suptitle(
    "Fixed Pooling (2×): Performance vs. SFT Data Size",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "fixed_pooling_sft_scaling.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "fixed_pooling_sft_scaling.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()