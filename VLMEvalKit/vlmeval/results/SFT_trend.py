import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("Fixed_pooling_trend.csv")

# Benchmark groups
coarse_benchmarks = ["MMBench", "MME", "MMMU", "RealWorldQA"]
ocr_benchmarks = ["TextVQA", "DocVQA", "OCRBench", "ChartQA"]

# Baseline
baseline = df[df["Model"] == "NEO-2B-SFT"].iloc[0]

# Fixed pooling
fixed = df[df["Model"] == "fixed pooling"].copy()
fixed = fixed.sort_values("training_data")

# Normalize
for benchmark in coarse_benchmarks + ocr_benchmarks:
    fixed[benchmark] = fixed[benchmark] / baseline[benchmark] * 100

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

groups = [
    ("Coarse-Grained Benchmarks", coarse_benchmarks),
    ("OCR Benchmarks", ocr_benchmarks),
]

for ax, (title, benchmarks) in zip(axes, groups):

    ax.axhline(100, color="gray", linestyle="--", alpha=0.7)

    ax.scatter(
        0, 100,
        marker="*",
        s=250,
        color="black",
        zorder=10,
        label="NEO-2B-SFT"
    )

    for benchmark in benchmarks:
        ax.plot(
            fixed["training_data"],
            fixed[benchmark],
            marker="o",
            markersize=7,
            linewidth=2,
            label=benchmark
        )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Training Data (K Samples)")
    ax.set_xticks([0, 66, 332])
    ax.set_xticklabels(["0K", "66K", "332K"])
    ax.set_xlim(-15, 350)
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=10)

axes[0].set_ylabel("Relative Performance to NEO-2B-SFT (%)")

fig.suptitle(
    "Fixed Pooling (2×): Performance vs. SFT Data Size",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig("fixed_pooling_sft_scaling.pdf", bbox_inches="tight")
plt.savefig("fixed_pooling_sft_scaling.png", dpi=300, bbox_inches="tight")
plt.show()