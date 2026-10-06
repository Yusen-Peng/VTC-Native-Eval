"""Effective compression ratio of each Table 2 method on each benchmark, without running the model.

The number of visual tokens a static compressor keeps depends only on each image's native grid, so
this replays NEO's preprocessing on the image sizes alone (CPU, no weights):
    image -> (MMMU 2x upscale) -> smart_resize(factor 32, min/max_pixels) -> grid (h, w) = size / 32
    -> compressor.num_output_tokens(h, w)
and reports  effective ratio = total native tokens / total kept tokens  per (method, benchmark).
Spatial pooling keeps whole partial edge blocks, so its effective ratio is below nominal on grids not
divisible by the block; the other methods are (near-)exact.

    cd VLMEvalKit && python ../scripts/token_budget.py              # all Table 2 benchmarks
    cd VLMEvalKit && python ../scripts/token_budget.py --data MME   # subset

Needs LMUData set as for the evals (images are dumped there on first use, as run.py does).
"""
import argparse
from collections import defaultdict

from PIL import Image

from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.smp import listinstr, toliststr
from vlmeval.vlm.neo.local_models.compressors import build_compressor
from vlmeval.vlm.neo.utils import smart_resize

DATASETS = ["MMBench_DEV_EN", "MME", "MMMU_DEV_VAL", "RealWorldQA",
            "TextVQA_VAL", "DocVQA_VAL", "OCRBench", "ChartQA_TEST"]
METHODS = {"Fixed": ["2x", "4x", "8x"], "Random": ["2x", "4x", "8x"], "ToMe": ["2x", "4x", "8x"],
           "PruneSID": ["2x", "4x", "8x"], "Spatial": ["4x", "8x"]}


def native_grids(dataset_name, min_pixels, max_pixels, factor):
    """(h, w) native token grid of every image NEO sees on this benchmark (mirrors NEOChat.generate_inner)."""
    if listinstr(["OCRBench"], dataset_name):  # NEOChat.set_max_num override
        min_pixels = 256 * 256
    dataset = build_dataset(dataset_name)
    grids = []
    for i in range(len(dataset.data)):
        paths = toliststr(dataset.dump_image(dataset.data.iloc[i]))
        for j, path in enumerate(paths):
            with Image.open(path) as im:  # header only, no decode
                W, H = im.size
            if j == 0 and listinstr(["MMMU"], dataset_name):  # NEOChat upscales the first MMMU image 2x
                W, H = W * 2, H * 2
            rh, rw = smart_resize(H, W, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
            grids.append((rh // factor, rw // factor))
    return grids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=DATASETS)
    args = ap.parse_args()

    models = [f"NEO-2B-SFT-{m}-{r}" for m, ratios in METHODS.items() for r in ratios]
    kw = supported_VLM[models[0]].keywords
    factor = int(kw["patch_size"] / kw["downsample_ratio"])
    for m in models:  # all methods must share the preprocessing, so one grid list serves them all
        k = supported_VLM[m].keywords
        assert (k["min_pixels"], k["max_pixels"], int(k["patch_size"] / k["downsample_ratio"])) == \
            (kw["min_pixels"], kw["max_pixels"], factor), m
    compressors = {m: build_compressor(supported_VLM[m].keywords["vtc_method"],
                                       compression_ratio=supported_VLM[m].keywords["compression_ratio"])
                   for m in models}

    table = defaultdict(dict)
    for d in args.data:
        grids = native_grids(d, kw["min_pixels"], kw["max_pixels"], factor)
        n_native = sum(h * w for h, w in grids)
        print(f"{d}: {len(grids)} images, mean {n_native / len(grids):.0f} native tokens/image", flush=True)
        for m, comp in compressors.items():
            table[m][d] = n_native / sum(comp.num_output_tokens(h, w) for h, w in grids)

    print("\nEffective compression ratio (total native tokens / total kept tokens)\n")
    print("| model | " + " | ".join(args.data) + " |")
    print("|---|" + "---|" * len(args.data))
    for m in models:
        print(f"| {m.removeprefix('NEO-2B-SFT-')} | " + " | ".join(f"{table[m][d]:.2f}x" for d in args.data) + " |")


if __name__ == "__main__":
    main()
