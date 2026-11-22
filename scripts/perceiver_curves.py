"""Training-loss curves of every Perceiver SFT run (direction perceiver-learned).

Collects `trainer_state.json` of each run under the checkpoint root into one compact JSON, then plots loss vs
step for all runs (the six existing Perceiver / PerceiverLearned runs plus any new ones). Two modes:

    # on the cluster (stdlib only): checkpoint dirs -> compact JSON on stdout
    python scripts/perceiver_curves.py collect /fs/scratch/PAS2836/jacktaylor/VTC-Native-Eval-ckpts/output
    # on the desktop: fetch the JSON over ssh (or read a cached one) and draw the PNG
    python scripts/perceiver_curves.py plot --ssh ascend-agent --out research/experiments/perceiver-learned/figures/loss_curves.png

Runs are matched by `NEO_continue_SFT_LLaVA_OV_llava66k_<Name>_<r>x[<suffix>]`; `plot --cache f.json` reuses a
collected JSON (and `--ssh` refreshes it).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

PREFIX = "NEO_continue_SFT_LLaVA_OV_llava66k_"


def collect(root):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, PREFIX + "Perceiver*"))):
        p = os.path.join(d, "trainer_state.json")   # written at the end of training
        if not os.path.isfile(p):
            # in-progress or interrupted run: newest checkpoint-*/trainer_state.json
            ckpts = sorted(glob.glob(os.path.join(d, "checkpoint-*", "trainer_state.json")),
                           key=lambda q: int(q.split("checkpoint-")[1].split(os.sep)[0]))
            if not ckpts:
                continue
            p = ckpts[-1]
        s = json.load(open(p))
        hist = [h for h in s["log_history"] if "loss" in h]
        out[os.path.basename(d)[len(PREFIX):]] = {"step": [h["step"] for h in hist],
                                                   "loss": [round(h["loss"], 4) for h in hist],
                                                   "global_step": s.get("global_step")}
    return out


def smooth(v, k):
    if k <= 1:
        return v
    out, acc = [], []
    for x in v:
        acc.append(x)
        if len(acc) > k:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def plot(curves, out, k=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # colour by ratio, line style by variant; new runs (any other name) get a thicker solid line
    ratio_colour = {"2x": "#1b6ca8", "4x": "#d1495b", "8x": "#30a46c"}
    style = {"Perceiver": "--", "PerceiverLearned": ":"}
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    for name in sorted(curves):
        m = re.match(r"(.+?)_(\d+x)(.*)$", name)
        variant, ratio, suffix = (m.group(1), m.group(2), m.group(3)) if m else (name, "2x", "")
        c = curves[name]
        ls = style.get(variant, "-")
        lw = 1.1 if variant in style else 1.8
        label = f"{variant}{suffix} {ratio} (final {c['loss'][-1]:.3f})"
        ax.plot(c["step"], smooth(c["loss"], k), ls, color=ratio_colour.get(ratio, "#7a7a7a"), lw=lw, label=label)
    ax.set_xlabel("optimizer step (global batch 128)")
    ax.set_ylabel(f"training loss (moving average, {k} steps)")
    ax.set_ylim(0.6, 1.5)
    ax.grid(alpha=0.3)
    ax.set_title("NEO-2B + Perceiver LoRA SFT on LLaVA@66K: pooled (dashed) vs learned (dotted) queries")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("root")
    p = sub.add_parser("plot")
    p.add_argument("--ssh", default=None, help="host alias; runs `collect` there and caches the JSON")
    p.add_argument("--root", default="/fs/scratch/PAS2836/jacktaylor/VTC-Native-Eval-ckpts/output")
    p.add_argument("--cache", default=None, help="compact JSON (written by --ssh, read without it)")
    p.add_argument("--out", required=True)
    p.add_argument("--smooth", type=int, default=10)
    args = ap.parse_args()
    if args.cmd == "collect":
        json.dump(collect(args.root), sys.stdout)
        return
    cache = args.cache or os.path.splitext(args.out)[0] + ".json"
    if args.ssh:
        remote = f"/fs/scratch/PAS2836/jacktaylor/wt/vtc/perceiver-learned/scripts/{os.path.basename(__file__)}"
        txt = subprocess.check_output(["ssh", args.ssh, f"python3 {remote} collect {args.root}"], text=True)
        curves = json.loads(txt)
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        json.dump(curves, open(cache, "w"))
    else:
        curves = json.load(open(cache))
    plot(curves, args.out, args.smooth)


if __name__ == "__main__":
    main()
