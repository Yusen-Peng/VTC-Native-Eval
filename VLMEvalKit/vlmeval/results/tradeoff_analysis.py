import pandas as pd
import matplotlib.pyplot as plt

"""
cd VLMEvalKit/vlmeval/results
python tradeoff_analysis.py
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

mt_row = df[df["Model"] == "NEO-2B-MT"].iloc[0]



# ============================================================
# Compression methods
# ============================================================

methods = [
    "fixed pooling",
    "spatial pooling",
    "random pruning",
    "ToMe",
    "PruneSID",
]

method_styles = {
    "fixed pooling":   {"marker": "o", "color": "tab:blue"},
    "spatial pooling": {"marker": "s", "color": "tab:orange"},
    "random pruning":  {"marker": "^", "color": "tab:green"},
    "ToMe":            {"marker": "D", "color": "tab:red"},
    "PruneSID":        {"marker": "P", "color": "tab:purple"},
}


# ============================================================
# Reusable plotting function
# ============================================================

def plot_performance(x_col, xlabel, filename):

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

        # ----------------------------------------------------
        # Compression methods
        # ----------------------------------------------------
        for method in methods:

            method_df = (
                df[df["Model"] == method]
                .sort_values(x_col)
            )

            if method_df.empty:
                continue

            style = method_styles[method]

            ax.plot(
                method_df[x_col],
                method_df[metric],
                marker=style["marker"],
                color=style["color"],
                markersize=8,
                linewidth=2,
                label=method,
            )

        # ----------------------------------------------------
        # NEO-2B-SFT reference
        # ----------------------------------------------------
        ax.scatter(
            [baseline[x_col]],
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

        # ----------------------------------------------------
        # NEO-2B-MT reference
        # ----------------------------------------------------
        ax.scatter(
            [mt_row[x_col]],
            [mt_row[metric]],
            marker="X",
            s=120,
            color="gray",
            label="NEO-2B-MT",
            zorder=9,
        )

        # ----------------------------------------------------
        # Formatting
        # ----------------------------------------------------
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)

        ax.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

        if x_col == "compression":
            ax.set_xticks([1, 2, 4, 8])
            ax.set_xticklabels(["1×", "2×", "4×", "8×"])

        elif x_col == "TFLOPs":
            ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
            ax.set_xlim(0, 14.5)

    axes[0].set_ylabel(
        "Relative Performance vs. NEO-2B-SFT (%)",
        fontsize=12,
    )

    # ========================================================
    # Shared legend
    # ========================================================

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(methods) + 2,
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # ========================================================
    # Save figures
    # ========================================================

    plt.savefig(
        f"{filename}.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# Figure 1: Performance vs. Compression Ratio
# ============================================================

plot_performance(
    x_col="compression",
    xlabel="Token Compression Ratio",
    filename="training_free_compression",
)


# ============================================================
# Figure 2: Performance vs. TFLOPs
# ============================================================

plot_performance(
    x_col="TFLOPs",
    xlabel="Computational Cost (TFLOPs)",
    filename="training_free_tflops",
)