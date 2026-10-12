import pandas as pd
import matplotlib.pyplot as plt


"""
cd VLMEvalKit/vlmeval/results

"""

csv_path = "training_free.csv"

coarse_benchmarks = [
    "MMBench",
    "MME",
    "MMMU",
    "RealWorldQA",
]

ocr_benchmarks = [
    "TextVQA",
    "DocVQA",
    "OCRBench",
    "ChartQA",
]

baseline_model = "NEO-2B-SFT"


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(csv_path)

baseline = df[df["Model"] == baseline_model].iloc[0]


# ============================================================
# Compute relative aggregate performance
#
# For each benchmark:
#
#     relative = method_score / baseline_score * 100
#
# Then average across benchmarks.
#
# Thus:
#     100% = same performance as NEO-2B-SFT
#      80% = retains 80% of baseline performance
# ============================================================

def aggregate_relative_performance(row, benchmarks):
    relative_scores = [
        row[b] / baseline[b] * 100
        for b in benchmarks
    ]
    return sum(relative_scores) / len(relative_scores)


df["Coarse"] = df.apply(
    lambda row: aggregate_relative_performance(row, coarse_benchmarks),
    axis=1,
)

df["OCR"] = df.apply(
    lambda row: aggregate_relative_performance(row, ocr_benchmarks),
    axis=1,
)


# ============================================================
# Keep training-free compression methods
#
# NEO-2B-SFT is added separately as the 1x reference point.
# NEO-2B-MT is excluded.
# ============================================================

methods = [
    "fixed pooling",
    "spatial pooling",
    "random pruning",
    "ToMe",
    "PruneSID",
]

method_styles = {
    "fixed pooling":   {"marker": "o"},
    "spatial pooling": {"marker": "s"},
    "random pruning":  {"marker": "^"},
    "ToMe":            {"marker": "D"},
    "PruneSID":        {"marker": "P"},
}


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5),
    sharey=True,
)

panels = [
    (axes[0], "Coarse", "Coarse-grained Benchmarks"),
    (axes[1], "OCR", "OCR Benchmarks"),
]

for ax, metric, title in panels:

    # --------------------------------------------------------
    # Plot each compression method
    # --------------------------------------------------------
    for method in methods:

        method_df = (
            df[df["Model"] == method]
            .sort_values("compression")
        )

        if len(method_df) == 0:
            continue

        ax.plot(
            method_df["compression"],
            method_df[metric],
            marker=method_styles[method]["marker"],
            markersize=8,
            linewidth=2,
            label=method,
        )


    # --------------------------------------------------------
    # Uncompressed model references
    # --------------------------------------------------------

    # Final SFT model
    ax.scatter(
        [1.0],
        [100.0],
        marker="*",
        s=180,
        color="black",
        label="NEO-2B-SFT",
        zorder=10,
    )
    ax.axhline(
        100,
        linestyle="--",
        linewidth=1.2,
        color="gray",
        alpha=0.7,
    )


    # Mid-training checkpoint
    mt_row = df[df["Model"] == "NEO-2B-MT"].iloc[0]

    ax.scatter(
        [1.0],
        [mt_row[metric]],
        marker="X",
        s=120,
        color="gray",
        label="NEO-2B-MT",
        zorder=9,
    )


    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Token Compression Ratio", fontsize=12)

    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1×", "2×", "4×", "8×"])

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )


axes[0].set_ylabel(
    "Relative Performance vs. NEO-2B-SFT (%)",
    fontsize=12,
)


# ============================================================
# Shared legend
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=len(methods)+2,
    frameon=False,
)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.savefig(
    "training_free_compression.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "training_free_compression.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()