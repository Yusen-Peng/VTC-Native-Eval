import pandas as pd
import matplotlib.pyplot as plt

"""
cd VLMEvalKit/vlmeval/results
module load miniconda3/24.1.2-py310
conda activate neo
python tradeoff_analysis.py
"""

baseline_model = "NEO-9B-SFT" # NEO-2B-SFT or NEO-9B-SFT
csv_path = "training_free_9B.csv" # training_free_2B.csv or training_free_9B.csv

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



# ============================================================
# Load data
# ============================================================

df = pd.read_csv(csv_path)

flops_df = pd.read_csv("flops.csv")

# Select the corresponding model scale
scale = "2B" if baseline_model == "NEO-2B-SFT" else "9B"

flops_df = flops_df[flops_df["scale"] == scale]

# Merge latency measurements
df = df.merge(
    flops_df[
        ["Model", "compression", "total_latency_ms"]
    ],
    on=["Model", "compression"],
    how="left",
    validate="one_to_one",
)

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




# ============================================================
# Compression methods
# ============================================================

methods = [
    "fixed pooling",
    "spatial pooling",
    "random pruning",
    "ToMe",
    "PruneSID",
    "CDPruner",
    "FastV-2",
    "PyramidDrop-16",
]

method_styles = {
    "fixed pooling":   {"marker": "o", "color": "tab:blue"},
    "spatial pooling": {"marker": "s", "color": "tab:orange"},
    "random pruning":  {"marker": "^", "color": "tab:green"},
    "ToMe":            {"marker": "D", "color": "tab:red"},
    "PruneSID":        {"marker": "P", "color": "tab:purple"},
    "CDPruner":        {"marker": "H", "color": "tab:cyan"},
    "FastV-2":         {"marker": "v", "color": "tab:gray"},
    "PyramidDrop-16":  {"marker": "X", "color": "tab:brown"},
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
        # Baseline model reference
        # ----------------------------------------------------
        ax.scatter(
            [baseline[x_col]],
            [100.0],
            marker="*",
            s=180,
            color="black",
            label=f"{baseline_model}",
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
            if baseline_model == "NEO-2B-SFT":
                ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
                ax.set_xlim(0, 14.5)
            elif baseline_model == "NEO-9B-SFT":
                ax.set_xticks([0, 5, 10, 20, 30, 40, 45])
                ax.set_xlim(0, 50)

            elif x_col == "total_latency_ms":
                ax.set_xlim(left=0)

    axes[0].set_ylabel(
        f"Relative Performance vs. {baseline_model} (%)",
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

    plt.show()


# ============================================================
# Figure 1: Performance vs. Compression Ratio
# ============================================================

plot_performance(
    x_col="compression",
    xlabel="Token Compression Ratio",
    filename=f"training_free_compression_{baseline_model}",
)


# ============================================================
# Figure 2: Performance vs. TFLOPs
# ============================================================

plot_performance(
    x_col="TFLOPs",
    xlabel="Computational Cost (TFLOPs)",
    filename=f"training_free_tflops_{baseline_model}",
)


# ============================================================
# Figure 3: Performance vs. Latency
# ============================================================

plot_performance(
    x_col="total_latency_ms",
    xlabel="Prefill Latency (ms)",
    filename=f"training_free_latency_{baseline_model}",
)

