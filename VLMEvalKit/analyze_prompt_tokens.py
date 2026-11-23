
"""
Analyze text prompt token distributions across VLMEvalKit benchmarks.

Measures:
    - Text tokens produced by dataset.build_prompt()
    - No image payloads or visual embeddings
    - No model inference or GPU required

Outputs:
    text_stats.csv
    summary.csv
    failures.csv
    text_tokens_boxplot.pdf
    text_tokens_boxplot.png

Example:
    python analyze_prompt_tokens.py --tokenizer Qwen/Qwen3-1.7B
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm
from transformers import AutoTokenizer

from vlmeval.dataset import build_dataset


DEFAULT_DATASETS = [
    "MMBench_DEV_EN",
    "MME",
    "MMMU_DEV_VAL",
    "RealWorldQA",
    "TextVQA_VAL",
    "DocVQA_VAL",
    "OCRBench",
    "ChartQA_TEST",
]

BENCHMARK_INFO = {
    "MMBench_DEV_EN": ("MMBench", "coarse"),
    "MME": ("MME", "coarse"),
    "MMMU_DEV_VAL": ("MMMU", "coarse"),
    "RealWorldQA": ("RealWorldQA", "coarse"),
    "TextVQA_VAL": ("TextVQA", "ocr"),
    "DocVQA_VAL": ("DocVQA", "ocr"),
    "OCRBench": ("OCRBench", "ocr"),
    "ChartQA_TEST": ("ChartQA", "ocr"),
}

COLORS = {
    "coarse": "#A8C8E8",
    "ocr": "#E8B4B8",
    "median": "#C65D3B",
    "mean": "#287D46",
    "edge": "#333333",
    "grid": "#E5E5E5",
}

# Remove literal image placeholders if present in text.
IMAGE_MARKER = re.compile(r"<image>", re.IGNORECASE)


# ============================================================
# Prompt extraction
# ============================================================

def get_text(message):
    """
    Recursively extract text from VLMEvalKit prompts.

    Handles:
      - strings
      - lists of text/image dictionaries
      - nested role/content dictionaries

    Image entries are ignored.
    """
    if isinstance(message, str):
        return message

    if isinstance(message, (list, tuple)):
        parts = [
            get_text(item)
            for item in message
        ]
        return "\n".join(x for x in parts if x)

    if isinstance(message, dict):
        kind = message.get("type")

        if kind == "image":
            return ""

        if kind == "text":
            return get_text(message.get("value", ""))

        # Handle chat-message dictionaries.
        if "content" in message:
            return get_text(message["content"])

        # Fallback for untyped prompt wrappers.
        if kind is None:
            parts = [
                get_text(value)
                for key, value in message.items()
                if key not in {
                    "role", "index", "metadata",
                    "image", "images"
                }
            ]
            return "\n".join(x for x in parts if x)

    return ""


def count_text_tokens(message, tokenizer):
    """
    Count raw text tokens without model-specific
    chat-template overhead.
    """
    text = get_text(message)

    # Image placeholders are not text content.
    text = IMAGE_MARKER.sub("", text)

    # Avoid tokenizing artificial leading/trailing whitespace.
    text = text.strip()

    if not text:
        return text, 0

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    return text, len(token_ids)


# ============================================================
# Dataset analysis
# ============================================================

def analyze_dataset(name, tokenizer):
    print(f"\nLoading {name}...", flush=True)

    dataset = build_dataset(name)

    if dataset is None:
        raise ValueError(
            f"Could not build dataset: {name}"
        )

    records = []
    failures = []

    for i in tqdm(range(len(dataset)), desc=name):
        row = dataset.data.iloc[i]

        idx = (
            row["index"]
            if "index" in row.index
            else i
        )

        try:
            message = dataset.build_prompt(row)

            text, num_tokens = count_text_tokens(
                message,
                tokenizer,
            )

            if not text:
                raise ValueError(
                    "No text found in benchmark prompt"
                )

            records.append({
                "dataset": name,
                "index": idx,
                "text_characters": len(text),
                "text_words": len(text.split()),
                "text_tokens": num_tokens,
            })

        except Exception as e:
            failures.append({
                "dataset": name,
                "index": idx,
                "error": str(e),
            })

    print(
        f"{name}: {len(records)} samples, "
        f"{len(failures)} failures",
        flush=True,
    )

    if failures:
        print("Example failures:", failures[:5])

    return records, failures


# ============================================================
# Summary statistics
# ============================================================

def summarize(df):
    metrics = [
        "text_characters",
        "text_words",
        "text_tokens",
    ]

    stats = [
        "count",
        "min",
        "p25",
        "median",
        "p75",
        "p90",
        "p95",
        "max",
        "mean",
    ]

    grouped = df.groupby("dataset")[metrics].agg([
        "count",
        "min",
        lambda x: x.quantile(0.25),
        "median",
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.90),
        lambda x: x.quantile(0.95),
        "max",
        "mean",
    ])

    grouped.columns = [
        f"{metric}_{stat}"
        for metric in metrics
        for stat in stats
    ]

    summary = grouped.reset_index()

    # Preserve benchmark order.
    summary["dataset"] = pd.Categorical(
        summary["dataset"],
        categories=DEFAULT_DATASETS,
        ordered=True,
    )

    return summary.sort_values("dataset")


# ============================================================
# Publication-style boxplot
# ============================================================

def make_boxplots(df, output_dir):

    # --------------------------------------------------------
    # Publication style
    # --------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "DejaVu Serif",
            "Times New Roman",
            "Times",
        ],
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })

    # --------------------------------------------------------
    # Benchmark configuration
    # --------------------------------------------------------
    datasets = [
        name for name in BENCHMARK_INFO
        if name in df["dataset"].values
    ]

    labels = [
        BENCHMARK_INFO[d][0]
        for d in datasets
    ]

    groups = [
        BENCHMARK_INFO[d][1]
        for d in datasets
    ]

    values = [
        df.loc[
            df["dataset"] == d,
            "text_tokens"
        ].dropna().to_numpy()
        for d in datasets
    ]

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 3.8))

    bp = ax.boxplot(
        values,
        positions=np.arange(1, len(datasets) + 1),
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        whis=1.5,

        boxprops={
            "linewidth": 1.0,
            "edgecolor": COLORS["edge"],
        },

        medianprops={
            "color": COLORS["median"],
            "linewidth": 1.8,
        },

        whiskerprops={
            "color": COLORS["edge"],
            "linewidth": 1.0,
        },

        capprops={
            "color": COLORS["edge"],
            "linewidth": 1.0,
        },

        meanprops={
            "marker": "^",
            "markerfacecolor": COLORS["mean"],
            "markeredgecolor": "white",
            "markeredgewidth": 0.5,
            "markersize": 5.5,
        },
    )

    # --------------------------------------------------------
    # Box colors
    # --------------------------------------------------------
    for patch, group in zip(bp["boxes"], groups):
        patch.set_facecolor(COLORS[group])
        patch.set_alpha(0.65)

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------
    ax.set_ylabel("Number of text tokens")

    ax.set_xticks(
        np.arange(1, len(datasets) + 1)
    )

    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)

    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        color=COLORS["grid"],
    )

    ax.xaxis.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(COLORS["edge"])
    ax.spines["bottom"].set_color(COLORS["edge"])

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3,
        width=0.8,
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------
    legend_handles = [
        Patch(
            facecolor=COLORS["coarse"],
            edgecolor=COLORS["edge"],
            alpha=0.65,
            label="General",
        ),
        Patch(
            facecolor=COLORS["ocr"],
            edgecolor=COLORS["edge"],
            alpha=0.65,
            label="OCR-centric",
        ),
        Line2D(
            [0], [0],
            color=COLORS["median"],
            linewidth=1.8,
            label="Median",
        ),
        Line2D(
            [0], [0],
            marker="^",
            linestyle="none",
            markerfacecolor=COLORS["mean"],
            markeredgecolor="white",
            markersize=6,
            label="Mean",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=2,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.2,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    fig.tight_layout(pad=0.4)

    fig.savefig(
        output_dir / "text_tokens_boxplot.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "text_tokens_boxplot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze benchmark text token distributions "
            "without loading a VLM."
        )
    )

    parser.add_argument(
        "--data",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="VLMEvalKit dataset names",
    )

    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-1.7B",
        help=(
            "HuggingFace tokenizer identifier "
            "or local tokenizer directory"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="./text_token_analysis",
        help="Output directory",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load tokenizer (CPU only)
    # --------------------------------------------------------
    print(
        f"Loading tokenizer: {args.tokenizer}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        trust_remote_code=False,
    )

    print(
        f"Tokenizer vocabulary size: {len(tokenizer)}",
        flush=True,
    )

    # --------------------------------------------------------
    # Analyze benchmarks
    # --------------------------------------------------------
    all_records = []
    all_failures = []

    for dataset_name in args.data:
        try:
            records, failures = analyze_dataset(
                dataset_name,
                tokenizer,
            )

            all_records.extend(records)
            all_failures.extend(failures)

        except Exception as e:
            print(
                f"FAILED {dataset_name}: {e}",
                flush=True,
            )

            all_failures.append({
                "dataset": dataset_name,
                "index": None,
                "error": str(e),
            })

    if not all_records:
        raise RuntimeError(
            "No benchmark prompts could be analyzed."
        )

    # --------------------------------------------------------
    # Save per-sample records
    # --------------------------------------------------------
    df = pd.DataFrame(all_records)

    df.to_csv(
        output_dir / "text_stats.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save failures
    # --------------------------------------------------------
    pd.DataFrame(
        all_failures,
        columns=["dataset", "index", "error"],
    ).to_csv(
        output_dir / "failures.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------
    summary = summarize(df)

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    print("\nText token statistics:")

    columns = [
        "dataset",
        "text_tokens_count",
        "text_tokens_min",
        "text_tokens_p25",
        "text_tokens_median",
        "text_tokens_p75",
        "text_tokens_p90",
        "text_tokens_p95",
        "text_tokens_max",
        "text_tokens_mean",
    ]

    print(
        summary[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    make_boxplots(df, output_dir)

    print(
        f"\nSaved results to {output_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
