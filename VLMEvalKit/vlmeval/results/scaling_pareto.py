import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


"""
cd VLMEvalKit/vlmeval/results
python scaling_pareto.py
"""


# ============================================================
# Configuration
# ============================================================

CSV_2B = "training_free_2B.csv"
CSV_9B = "training_free_9B.csv"

COARSE = ["MMBench", "MME", "MMMU", "RealWorldQA"]
OCR = ["TextVQA", "DocVQA", "OCRBench", "ChartQA"]
BENCHMARKS = COARSE + OCR

COLORS = {
    "2B": "#1f77b4",
    "9B": "#d62728",
}

MARKERS = {
    "fixed pooling": "o",
    "spatial pooling": "s",
    "random pruning": "^",
    "ToMe": "D",
    "PruneSID": "P",
    "baseline": "*",
}

# ============================================================
# Load data
# ============================================================

df_2b = pd.read_csv(CSV_2B)
df_9b = pd.read_csv(CSV_9B)

df_2b["Size"] = "2B"
df_9b["Size"] = "9B"

# Use uncompressed NEO-9B-SFT as the common reference.
baseline_9b = df_9b[
    (df_9b["Model"] == "NEO-9B-SFT")
    & (df_9b["compression"] == 1.0)
]

assert len(baseline_9b) == 1, "Cannot find unique NEO-9B-SFT baseline."

baseline_9b = baseline_9b.iloc[0]

# Combine data.
df = pd.concat([df_2b, df_9b], ignore_index=True)

# Keep SFT baselines and compression methods.
df = df[
    ~df["Model"].str.endswith("-MT")
].copy()

# ============================================================
# Normalize benchmark performance
# ============================================================

for benchmark in BENCHMARKS:
    df[f"{benchmark}_relative"] = (
        df[benchmark] / baseline_9b[benchmark] * 100
    )

df["Coarse"] = df[
    [f"{b}_relative" for b in COARSE]
].mean(axis=1)

df["OCR"] = df[
    [f"{b}_relative" for b in OCR]
].mean(axis=1)

df["Method"] = df["Model"].apply(
    lambda x: "baseline" if x.startswith("NEO-") else x
)

# ============================================================
# Pareto frontier
# ============================================================

def pareto_frontier(data, metric):
    """
    Minimize TFLOPs, maximize performance.

    A point is Pareto optimal if no other point has:
        FLOPs <= current FLOPs
        performance >= current performance
    with at least one strict inequality.
    """

    points = data[["TFLOPs", metric]].to_numpy(dtype=float)

    is_pareto = np.ones(len(points), dtype=bool)

    for i, (cost_i, perf_i) in enumerate(points):
        dominated = (
            (points[:, 0] <= cost_i)
            & (points[:, 1] >= perf_i)
            & (
                (points[:, 0] < cost_i)
                | (points[:, 1] > perf_i)
            )
        )
        is_pareto[i] = not np.any(dominated)

    frontier = data.loc[is_pareto].copy()
    frontier = frontier.sort_values(["TFLOPs", metric])

    return frontier


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, metric, title in zip(
    axes,
    ["Coarse", "OCR"],
    ["Coarse-grained Benchmarks", "OCR Benchmarks"],
):

    # Plot all configurations.
    for _, row in df.iterrows():

        size = row["Size"]
        method = row["Method"]

        ax.scatter(
            row["TFLOPs"],
            row[metric],
            color=COLORS[size],
            marker=MARKERS.get(method, "o"),
            s=190 if method == "baseline" else 75,
            edgecolor="black" if method == "baseline" else "none",
            linewidth=0.8,
            alpha=0.9,
            zorder=4,
        )

    # Unified Pareto frontier across BOTH model sizes.
    frontier = pareto_frontier(df, metric)

    ax.plot(
        frontier["TFLOPs"],
        frontier[metric],
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        zorder=2,
    )

    # 2B baseline reference.
    baseline_2b = df[
        (df["Size"] == "2B")
        & (df["Method"] == "baseline")
    ].iloc[0]

    ax.axhline(
        baseline_2b[metric],
        color=COLORS["2B"],
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
    )

    # 9B baseline reference.
    ax.axhline(
        100,
        color=COLORS["9B"],
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Computational Cost (TFLOPs)", fontsize=12)
    ax.set_ylabel(
        "Performance Relative to NEO-9B-SFT (%)",
        fontsize=12,
    )

    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)

# ============================================================
# Legends
# ============================================================

size_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        color="w",
        markerfacecolor=color,
        markersize=9,
        label=size,
    )
    for size, color in COLORS.items()
]

method_handles = [
    Line2D(
        [0], [0],
        marker=marker,
        linestyle="none",
        color="black",
        markersize=8,
        label=method,
    )
    for method, marker in MARKERS.items()
]

frontier_handle = Line2D(
    [0], [0],
    color="black",
    linestyle="--",
    linewidth=2,
    label="Pareto Frontier",
)

fig.legend(
    handles=size_handles + method_handles + [frontier_handle],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.04),
    ncol=5,
    frameon=False,
    fontsize=10,
)

fig.suptitle(
    "Compute-Performance Pareto Frontier across 2B and 9B",
    fontsize=17,
    y=1.02,
)

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.savefig("scaling_pareto.pdf", bbox_inches="tight")
plt.show()

# ============================================================
# Print Pareto-optimal configurations
# ============================================================

for metric in ["Coarse", "OCR"]:

    frontier = pareto_frontier(df, metric)

    print(f"\n{'=' * 60}")
    print(f"Pareto Frontier: {metric}")
    print(f"{'=' * 60}")

    print(
        frontier[
            ["Size", "Model", "compression", "TFLOPs", metric]
        ].to_string(index=False)
    )