"""Collect Table 2 scores from VLMEvalKit/outputs/<model>/<model>_<dataset>_{acc,score}.csv.

    python scripts/summarize_table2.py                       # markdown table, NEO-2B
    python scripts/summarize_table2.py --scale 9B --latex    # rows to paste into VTC-Native-Eval.tex

Conventions match the existing table: MME = perception score; MMBench (dev split), MMMU (validation
split) and RealWorldQA are Overall accuracy x100; TextVQA / DocVQA (ANLS) / ChartQA are Overall (already
in percent); OCRBench is "Final Score Norm" (final score / 10). Missing/failed runs show as '--'.
TFLOPs (prefill, one reference image) come from VLMEvalKit/vlmeval/results/flops.csv (scripts/flops.py).
"""
import argparse
import json
import os.path as osp

import pandas as pd

ROOT = osp.join(osp.dirname(osp.abspath(__file__)), "..", "VLMEvalKit", "outputs")
FLOPS = osp.join(osp.dirname(osp.abspath(__file__)), "..", "VLMEvalKit", "vlmeval", "results", "flops.csv")
RATIOS = ["2x", "4x", "8x"]
# (config key, table label, ratios run); spatial 2x (1x2 blocks) is skipped: it equals fixed 2x on even-width grids
METHODS = [("Fixed", "fixed pooling", RATIOS), ("Random", "random pruning", RATIOS), ("ToMe", "ToMe", RATIOS),
           ("PruneSID", "PruneSID", RATIOS), ("Spatial", "spatial pooling", ["4x", "8x"])]
# dataset -> (file suffix, split row or None, column, scale)
DATASETS = {
    "MMBench": ("MMBench_DEV_EN_acc.csv", "dev", "Overall", 100),
    "MME": ("MME_score.csv", None, "perception", 1),
    "MMMU": ("MMMU_DEV_VAL_acc.csv", "validation", "Overall", 100),
    "RealWorldQA": ("RealWorldQA_acc.csv", None, "Overall", 100),
    "TextVQA": ("TextVQA_VAL_acc.csv", None, "Overall", 1),
    "DocVQA": ("DocVQA_VAL_acc.csv", None, "Overall", 1),
    "OCRBench": ("OCRBench_score.json", None, "Final Score Norm", 1),
    "ChartQA": ("ChartQA_TEST_acc.csv", None, "Overall", 1),
}


def score(root, model, suffix, split, col, scale):
    path = osp.join(root, model, f"{model}_{suffix}")
    if not osp.exists(path):
        return None
    if path.endswith(".json"):
        with open(path) as f:
            return float(json.load(f)[col]) * scale
    df = pd.read_csv(path)
    if split is not None:
        df = df[df["split"] == split]
    return float(df[col].iloc[0]) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--root", default=ROOT, help="VLMEvalKit outputs dir")
    ap.add_argument("--scale", default="2B", choices=["2B", "9B"])
    args = ap.parse_args()

    base = f"NEO-{args.scale}-SFT"
    flops = pd.read_csv(FLOPS) if osp.exists(FLOPS) else None

    def tflops(label, ratio):
        if flops is None:
            return None
        hit = flops[(flops["scale"] == args.scale) & (flops["Model"] == label) & (flops["compression"] == ratio)]
        return float(hit["TFLOPs"].iloc[0]) if len(hit) else None

    rows = [("1x", base, [score(args.root, base, *DATASETS[d]) for d in DATASETS] + [tflops(base, 1.0)])]
    for ratio in RATIOS:
        for key, label, ratios in METHODS:
            if ratio not in ratios:
                continue
            model = f"{base}-{key}-{ratio}"
            rows.append((ratio, label, [score(args.root, model, *DATASETS[d]) for d in DATASETS]
                         + [tflops(label, float(ratio[:-1]))]))

    columns = list(DATASETS) + ["TFLOPs"]
    fmt = lambda v: "--" if v is None else (f"{v:.1f}" if v > 200 else f"{v:.2f}")
    if args.latex:
        print(f"        {base} & " + " & ".join(fmt(v) for v in rows[0][2]) + " \\\\")
        for ratio in RATIOS:
            print(f"        \\rowcolor{{green!10}}\\multicolumn{{{len(columns) + 1}}}{{c}}{{{ratio} compression}} \\\\")
            for r, label, vals in rows[1:]:
                if r == ratio:
                    print(f"        + {label} & " + " & ".join(fmt(v) for v in vals) + " \\\\")
    else:
        print("| model | " + " | ".join(columns) + " |")
        print("|---|" + "---|" * len(columns))
        for ratio, label, vals in rows:
            print(f"| {label} {ratio} | " + " | ".join(fmt(v) for v in vals) + " |")


if __name__ == "__main__":
    main()
