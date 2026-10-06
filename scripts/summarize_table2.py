"""Collect Table 2 scores from VLMEvalKit/outputs/<model>/<model>_<dataset>_{acc,score}.csv.

    python scripts/summarize_table2.py            # markdown table
    python scripts/summarize_table2.py --latex    # rows to paste into VTC-Native-Eval.tex

Conventions match the existing table: MME = perception score; MMBench (dev split), MMMU (validation
split) and RealWorldQA are Overall accuracy x100; TextVQA / DocVQA (ANLS) / ChartQA are Overall (already
in percent); OCRBench is "Final Score Norm" (final score / 10). Missing/failed runs show as '--'.
"""
import argparse
import json
import os.path as osp

import pandas as pd

ROOT = osp.join(osp.dirname(osp.abspath(__file__)), "..", "VLMEvalKit", "outputs")
RATIOS = ["2x", "4x", "8x"]
# (config key, table label, ratios run); spatial 2x (1x2 blocks) is skipped: it equals fixed 2x on even-width grids
METHODS = [("Fixed", "fixed pooling", RATIOS), ("Random", "random pruning", RATIOS), ("ToMe", "ToMe (re-implement)", RATIOS),
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
    args = ap.parse_args()

    rows = []
    for ratio in RATIOS:
        for key, label, ratios in METHODS:
            if ratio not in ratios:
                continue
            model = f"NEO-2B-SFT-{key}-{ratio}"
            rows.append((ratio, label, [score(args.root, model, *DATASETS[d]) for d in DATASETS]))

    fmt = lambda v: "--" if v is None else (f"{v:.1f}" if v > 200 else f"{v:.2f}")
    if args.latex:
        for ratio in RATIOS:
            print(f"        \\rowcolor{{green!10}}\\multicolumn{{9}}{{c}}{{{ratio} compression}} \\\\")
            for r, label, vals in rows:
                if r == ratio:
                    print(f"        + {label} & " + " & ".join(fmt(v) for v in vals) + " \\\\")
    else:
        print("| model | " + " | ".join(DATASETS) + " |")
        print("|---|" + "---|" * len(DATASETS))
        for ratio, label, vals in rows:
            print(f"| {label} {ratio} | " + " | ".join(fmt(v) for v in vals) + " |")


if __name__ == "__main__":
    main()
