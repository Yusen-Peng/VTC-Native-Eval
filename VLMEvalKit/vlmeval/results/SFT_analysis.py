import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

CSV_PATH = "SFT_dynamics.csv"

COARSE_METRICS = [
    "MMBench",
    "MME",
    "MMMU",
    "RealWorldQA",
]

OCR_METRICS = [
    "TextVQA",
    "DocVQA",
    "OCRBench",
    "ChartQA",
]

# Names as they appear in the CSV
ORIGINAL = "NEO-2B-SFT"

RUNS = {
    # model name: (training pipeline, data size in K)
    "NEO-2B-SFT":                  ("SFT", 0),
    "NEO-2B-SFT (LLaVA@66K)":      ("SFT", 66),
    "NEO-2B-SFT (LLaVA@332K)":     ("SFT", 332),
    "NEO-2B-SFT (LLaVA@66K+KL)":   ("SFT + KL", 66),
    "NEO-2B-SFT (LLaVA@332K+KL)":  ("SFT + KL", 332),
}


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(CSV_PATH)

# Keep only relevant runs
df = df[df["Model"].isin(RUNS)].copy()

df["Pipeline"] = df["Model"].map(lambda x: RUNS[x][0])
df["DataSize"] = df["Model"].map(lambda x: RUNS[x][1])

# Convert benchmark columns to numeric.
# Pending entries such as "training!" automatically become NaN.
all_metrics = COARSE_METRICS + OCR_METRICS

for metric in all_metrics:
    df[metric] = pd.to_numeric(df[metric], errors="coerce")


# ============================================================
# Normalize against original NEO-2B-SFT
# ============================================================

baseline = df[df["Model"] == ORIGINAL].iloc[0]

for metric in all_metrics:
    df[f"{metric}_relative"] = (
        df[metric] / baseline[metric] * 100
    )

coarse_relative = [f"{m}_relative" for m in COARSE_METRICS]
ocr_relative = [f"{m}_relative" for m in OCR_METRICS]

df["Coarse Aggregate"] = df[coarse_relative].mean(axis=1)
df["OCR Aggregate"] = df[ocr_relative].mean(axis=1)


# ============================================================
# Important:
# Both trajectories start from the SAME original model.
# ============================================================

baseline_row = df[df["Model"] == ORIGINAL].iloc[0]

plot_rows = []

# Shared starting point
for pipeline in ["SFT", "SFT + KL"]:
    plot_rows.append({
        "Pipeline": pipeline,
        "DataSize": 0,
        "Coarse Aggregate": baseline_row["Coarse Aggregate"],
        "OCR Aggregate": baseline_row["OCR Aggregate"],
    })

# Add post-training checkpoints
for _, row in df[df["Model"] != ORIGINAL].iterrows():
    plot_rows.append({
        "Pipeline": row["Pipeline"],
        "DataSize": row["DataSize"],
        "Coarse Aggregate": row["Coarse Aggregate"],
        "OCR Aggregate": row["OCR Aggregate"],
    })

plot_df = pd.DataFrame(plot_rows)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(11, 4.2),
    sharex=True
)

panels = [
    (
        axes[0],
        "Coarse Aggregate",
        "Coarse-Grained Benchmarks",
    ),
    (
        axes[1],
        "OCR Aggregate",
        "OCR-Centric Benchmarks",
    ),
]

markers = {
    "SFT": "o",
    "SFT + KL": "s",
}

linestyles = {
    "SFT": "-",
    "SFT + KL": "--",
}

for ax, metric, title in panels:

    for pipeline in ["SFT", "SFT + KL"]:

        sub = (
            plot_df[plot_df["Pipeline"] == pipeline]
            .dropna(subset=[metric])
            .sort_values("DataSize")
        )

        ax.plot(
            sub["DataSize"],
            sub[metric],
            marker=markers[pipeline],
            linestyle=linestyles[pipeline],
            linewidth=2,
            markersize=7,
            label=pipeline,
        )

    # Original model = 100
    ax.axhline(
        100,
        linestyle=":",
        linewidth=1.2,
        alpha=0.6,
    )

    ax.set_title(title)
    ax.set_xlabel("LLaVA SFT Data Size")
    ax.set_ylabel("Relative Aggregate Performance (%)")

    ax.set_xticks(
        [0, 66, 332],
        ["Original", "66K\n(10%)", "332K\n(50%)"],
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    ax.legend(frameon=False)

plt.tight_layout()
plt.savefig(
    "SFT_dynamics.pdf",
    bbox_inches="tight",
)
plt.savefig(
    "SFT_dynamics.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
