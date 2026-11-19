
import argparse
import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from vlmeval.dataset import build_dataset


"""
module load miniconda3/24.1.2-py310
conda activate neo
cd VLMEvalKit
export LMUData=/fs/scratch/PAS2836/yusenpeng_dataset/VLMEvalKit_data
python analyze_image_tokens.py
"""

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

PATCH_SIZE = 16
DOWNSAMPLE_FACTOR = 2


def get_images(message):
    """Recursively extract image paths or image objects from prompts."""
    images = []

    if isinstance(message, (list, tuple)):
        for item in message:
            images.extend(get_images(item))

    elif isinstance(message, dict):
        if message.get("type") == "image":
            value = message.get("value")
            if value is not None:
                images.append(value)
        elif "type" not in message:
            for value in message.values():
                images.extend(get_images(value))

    return images


def image_size(source):
    """Read source image dimensions without decoding full pixel arrays."""
    if isinstance(source, Image.Image):
        return source.size

    if isinstance(source, (str, os.PathLike)):
        source = str(source)

        if source.startswith("data:image/"):
            import base64

            payload = source.split(",", 1)[1]
            with Image.open(io.BytesIO(base64.b64decode(payload))) as img:
                return img.size

        with Image.open(source) as img:
            return img.size

    raise TypeError(f"Unsupported image source: {type(source)}")



import math


def round_by_factor(number, factor):
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


def smart_resize(
    height,
    width,
    factor=32,
    min_pixels=65536,
    max_pixels=4194304,
):
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Aspect ratio exceeds 200")

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))

    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


def estimate_tokens(width, height):
    resized_h, resized_w = smart_resize(
        height,
        width,
        factor=32,
        min_pixels=65536,
        max_pixels=4194304,
    )

    return (resized_h // 32) * (resized_w // 32)



def analyze_dataset(name):
    print(f"\nLoading {name}...", flush=True)
    dataset = build_dataset(name)

    if dataset is None:
        raise ValueError(f"Could not build dataset: {name}")

    records = []
    failures = []

    for i in tqdm(range(len(dataset)), desc=name):
        row = dataset.data.iloc[i]
        idx = row["index"]

        try:
            message = dataset.build_prompt(row)
            images = get_images(message)

            if not images:
                failures.append((idx, "No images found"))
                continue

            for j, source in enumerate(images):
                width, height = image_size(source)

                records.append({
                    "dataset": name,
                    "index": idx,
                    "image_id": j,
                    "width": width,
                    "height": height,
                    "pixels": width * height,
                    "megapixels": width * height / 1e6,
                    "estimated_tokens": estimate_tokens(width, height),
                })

        except Exception as e:
            failures.append((idx, str(e)))

    print(
        f"{name}: {len(records)} images, "
        f"{len(failures)} failed samples",
        flush=True,
    )

    if failures:
        print("Example failures:", failures[:5])

    return records




def make_boxplots(df, output_dir):
    # ============================================================
    # Publication style
    # ============================================================
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
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

    # ============================================================
    # Benchmark configuration
    # ============================================================
    benchmark_info = {
        "MMBench_DEV_EN": ("MMBench", "coarse"),
        "MME": ("MME", "coarse"),
        "MMMU_DEV_VAL": ("MMMU", "coarse"),
        "RealWorldQA": ("RealWorldQA", "coarse"),
        "TextVQA_VAL": ("TextVQA", "ocr"),
        "DocVQA_VAL": ("DocVQA", "ocr"),
        "OCRBench": ("OCRBench", "ocr"),
        "ChartQA_TEST": ("ChartQA", "ocr"),
    }

    datasets = [
        name for name in benchmark_info
        if name in df["dataset"].values
    ]

    labels = [benchmark_info[d][0] for d in datasets]
    groups = [benchmark_info[d][1] for d in datasets]

    values = [
        df.loc[
            df["dataset"] == d,
            "estimated_tokens"
        ].dropna().values
        for d in datasets
    ]

    # ============================================================
    # Colors
    # ============================================================
    COLORS = {
        "coarse": "#A8C8E8",   # Pastel blue
        "ocr": "#E8B4B8",      # Pastel coral
        "median": "#C65D3B",   # Muted orange
        "mean": "#287D46",     # Dark green
        "edge": "#333333",
        "grid": "#E5E5E5",
    }

    # ============================================================
    # Figure
    # ============================================================
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

    # Fill boxes by benchmark category
    for patch, group in zip(bp["boxes"], groups):
        patch.set_facecolor(COLORS[group])
        patch.set_alpha(0.65)

    # ============================================================
    # Axes
    # ============================================================
    ax.set_ylabel("Number of visual tokens")

    ax.set_xticks(np.arange(1, len(datasets) + 1))
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

    # Remove unnecessary spines
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

    # ============================================================
    # Legend
    # ============================================================
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

    # ============================================================
    # Save
    # ============================================================
    fig.tight_layout(pad=0.4)

    fig.savefig(
        output_dir / "tokens_boxplot.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "tokens_boxplot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def summarize(df):
    metrics = ["pixels", "megapixels", "estimated_tokens"]

    summary = (
        df.groupby("dataset")[metrics]
        .agg([
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
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric in metrics
        for stat in [
            "count", "min", "p25", "median",
            "p75", "p90", "p95", "max", "mean"
        ]
    ]

    return summary.reset_index()






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        nargs="+",
        default=DEFAULT_DATASETS,
    )
    parser.add_argument(
        "--output-dir",
        default="./image_token_analysis",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    for dataset_name in args.data:
        try:
            all_records.extend(analyze_dataset(dataset_name))
        except Exception as e:
            print(f"FAILED {dataset_name}: {e}", flush=True)

    if not all_records:
        raise RuntimeError("No images could be analyzed.")

    df = pd.DataFrame(all_records)
    df.to_csv(output_dir / "image_stats.csv", index=False)

    summary = summarize(df)
    summary.to_csv(output_dir / "summary.csv", index=False)

    print("\nVisual token statistics:")
    columns = [
        "dataset",
        "estimated_tokens_min",
        "estimated_tokens_p25",
        "estimated_tokens_median",
        "estimated_tokens_p75",
        "estimated_tokens_p90",
        "estimated_tokens_max",
    ]
    print(summary[columns].to_string(index=False))

    make_boxplots(df, output_dir)

    print(f"\nSaved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
