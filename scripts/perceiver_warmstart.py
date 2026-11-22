"""Warm-start weights for the paper-faithful ``learned``-query Perceiver (direction perceiver-learned, item 3).

e001 found the collapse mechanism of ``queries=learned``: trunc-normal(0.02) latents are drowned by the O(1)
attention residual, so every query emits the same vector. This builds an initial ``perceiver.bin`` for the
``learned`` mode (the architecture is unchanged; the file loads with strict=True) in which

  - latents    = per-cell mean over N LLaVA images of the native token map adaptively average-pooled to the
                 [G, G] latent grid, rescaled so the grid's RMS equals the mean native-token RMS (the scale the
                 pooled queries have in the modes that train);
  - o_proj and the last MLP linear are zero, so the first forward is exactly norm_out(resized latents) * RMS(x):
                 the residual branches start at 0 and grow, instead of starting at O(1) and burying the queries;
  - the other projections / LayerNorms are copied from a trained ``pooled`` run (``--from``; default: the
                 pooled run of the same ratio), norm_out is fresh (weight 1, bias 0).

Run from any directory (absolute paths; neo env, HF_HOME, HF_HUB_OFFLINE=1, LLaVA annotation on scratch):
    python $REPO/scripts/perceiver_warmstart.py --ratio 2 --n 256 --out $SCRATCH/VTC-Native-Eval-ckpts/init/perceiver_learned_warm_2x.bin
Then train with  QUERIES=learned INIT_FROM=<out> SFT_perceiver.sh  (run name PerceiverLearnedWarm).
``--selftest`` (CPU, no data, no checkpoints) exercises build_init + the strict load + the first-forward identity.

Item 3b (e008: the grid's common direction, 82% of its RMS, drowns the attention branch in norm_out): derive a
centred init from an existing one on the CPU (seconds, any node, no data):
    python $REPO/scripts/perceiver_warmstart.py --center-from <warm.bin> --out <warm_c.bin> [--rms keep|native]
latents <- latents - mean over the grid cells (the 18% spatial part only); ``--rms keep`` (default) leaves it at
its own RMS, ``native`` rescales it back to the source grid's RMS. Everything else is copied unchanged.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "VLMEvalKit"))
sys.path.insert(0, REPO)

CKPTS = "/fs/scratch/PAS2836/jacktaylor/VTC-Native-Eval-ckpts/output"
COPIED = ("norm_q", "norm_kv", "q_proj", "k_proj", "v_proj", "norm_mlp", "mlp.0")
ZEROED = ("o_proj", "mlp.2")


def build_init(learned_sd, pooled_sd, mean_grid, native_rms):
    """State dict for a ``learned`` PerceiverCompressor: pooled weights copied, residual outputs zeroed,
    latents = mean_grid [D, G, G] rescaled to native_rms. learned_sd supplies the key set (and norm_out)."""
    out = {}
    for k, v in learned_sd.items():
        prefix = k.rsplit(".", 1)[0]
        if k == "latents":
            g = mean_grid.to(v.dtype)
            out[k] = (g * (native_rms / g.pow(2).mean().sqrt())).reshape(v.shape).clone()
        elif prefix in ZEROED:
            out[k] = torch.zeros_like(v)
        elif prefix in COPIED:
            out[k] = pooled_sd[k].to(v.dtype).clone()
        elif prefix == "norm_out":
            out[k] = v.clone()  # fresh LayerNorm
        else:
            raise KeyError(f"unexpected learned-mode key {k}")
    return out


def grid_stats(lat):
    """lat [D, G, G]: RMS, RMS of the per-cell deviation from the mean cell, and the mean pairwise cosine
    between cells (1.0 = every cell the same direction)."""
    cells = lat.flatten(1).t().float()  # [G*G, D]
    dev = cells - cells.mean(dim=0, keepdim=True)
    u = F.normalize(cells, dim=1)
    cos = u @ u.t()
    n = cos.shape[0]
    return {"rms": cells.pow(2).mean().sqrt().item(), "spatial_dev_rms": dev.pow(2).mean().sqrt().item(),
            "cell_pairwise_cosine": ((cos.sum() - n) / (n * (n - 1))).item()}


def center_init(sd, rms="keep"):
    """Same state dict with the latents centred over the grid cells. rms: keep (the spatial part at its own
    RMS) | native (rescaled to the source grid's RMS). Returns (new state dict, stats before, stats after)."""
    out = {k: v.clone() for k, v in sd.items()}
    lat = sd["latents"]
    g = lat.reshape(lat.shape[-3], lat.shape[-2], lat.shape[-1]).float()  # [D, G, G]
    before = grid_stats(g)
    c = g - g.mean(dim=(1, 2), keepdim=True)
    if rms == "native":
        c = c * (before["rms"] / c.pow(2).mean().sqrt())
    elif rms != "keep":
        raise ValueError(rms)
    out["latents"] = c.reshape(lat.shape).to(lat.dtype)
    return out, before, grid_stats(c)


def selftest():
    # compressor package without the rest of vlmeval (runs on the desktop CPU, stdlib + torch)
    from scripts.test_perceiver_modes import load_package
    PerceiverCompressor = load_package()[1].PerceiverCompressor
    torch.manual_seed(0)
    D, hd, G = 64, 16, 8
    # centring: a grid that is one common direction plus a small spatial part
    common = torch.randn(D, 1, 1)
    sd0 = {"latents": (common + 0.2 * torch.randn(D, G, G)).reshape(1, D, G, G), "other": torch.ones(3)}
    for rms in ("keep", "native"):
        sd1, b, a = center_init(sd0, rms)
        g1 = sd1["latents"][0]
        assert g1.mean(dim=(1, 2)).abs().max() < 1e-5, "not centred"
        assert b["cell_pairwise_cosine"] > 0.9 and abs(a["cell_pairwise_cosine"]) < 0.1, (b, a)
        assert torch.equal(sd1["other"], sd0["other"])
        want = b["rms"] if rms == "native" else b["spatial_dev_rms"]
        assert abs(a["rms"] - want) < 1e-4, (rms, a, b)
    print(f"center_init: cosine {b['cell_pairwise_cosine']:.3f} -> {a['cell_pairwise_cosine']:.3f}, keep/native RMS ok")
    for ratio in (2, 4, 8):
        pooled = PerceiverCompressor(1 / ratio, hidden_size=D, head_dim=hd, queries="pooled", latent_grid=G)
        for p in pooled.parameters():  # a "trained" pooled module: non-trivial weights everywhere
            p.data.normal_(0, 0.1)
        learned = PerceiverCompressor(1 / ratio, hidden_size=D, head_dim=hd, queries="learned", latent_grid=G)
        mean_grid = torch.randn(D, G, G)
        sd = build_init(learned.state_dict(), pooled.state_dict(), mean_grid, native_rms=4.5)
        learned.load_state_dict(sd, strict=True)
        assert abs(learned.latents.pow(2).mean().sqrt().item() - 4.5) < 1e-4
        assert torch.equal(learned.q_proj.weight, pooled.q_proj.weight) and torch.equal(learned.mlp[0].bias, pooled.mlp[0].bias)
        assert learned.o_proj.weight.abs().sum() == 0 and learned.mlp[2].weight.abs().sum() == 0
        x = torch.randn(2, 6, 10, D)
        y = learned(x)
        gh, gw = learned._cell_grid(6, 10)
        q = F.interpolate(learned.latents, size=(gh, gw), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
        ref = learned.norm_out(q.expand(2, -1, -1)) * x.reshape(2, -1, D).pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
        assert y.shape == ref.shape and torch.allclose(y, ref, atol=1e-5), "first forward is not the pure query grid"
        # at init every position emits the same direction only up to the latent grid's spatial structure
        y.pow(2).sum().backward()
        assert learned.latents.grad.abs().sum() > 0 and learned.o_proj.weight.grad.abs().sum() > 0
        assert learned.mlp[2].weight.grad.abs().sum() > 0
        print(f"{ratio}x: strict load, first forward == norm_out(latent grid) * rms(x), grads to latents/o_proj/mlp.2: ok")
    print("selftest passed")


class _Stash(Exception):
    pass


@torch.no_grad()
def mean_native_grid(batches, model, device, G):
    """Per-cell mean over images of adaptive_avg_pool2d(native token map [h, w, D] -> [G, G]); also the mean
    native-token RMS. The LLM forward is aborted right after the vision embedding (the compressor input)."""
    from scripts.perceiver_probe import to_device
    comp = model.compressor
    orig = comp.forward
    acc, rms_sum, n = None, 0.0, 0

    def grab(x):
        raise _Stash(x.detach().float())

    comp.forward = grab
    try:
        for b in batches:
            inp = to_device(b, device)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model(**inp)
            except _Stash as s:
                x = s.args[0][0]  # [h, w, D]
                g = F.adaptive_avg_pool2d(x.permute(2, 0, 1)[None], G)[0]  # [D, G, G]
                acc = g if acc is None else acc + g
                rms_sum += x.pow(2).mean().sqrt().item()
                n += 1
            else:
                raise RuntimeError("compressor.forward was not called (no image in this sample?)")
    finally:
        comp.forward = orig
    return acc / n, rms_sum / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ratio", type=int, default=2)
    ap.add_argument("--n", type=int, default=256, help="LLaVA images for the latent-grid mean")
    ap.add_argument("--seed", type=int, default=1, help="sample selection seed (e001/e003 probes use 0)")
    ap.add_argument("--latent-grid", type=int, default=32)
    ap.add_argument("--from", dest="src", default=None, help="trained pooled perceiver.bin (default: pooled run of --ratio)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset", default="llava_665k")
    ap.add_argument("--center-from", default=None, help="existing warm init bin: write a centred copy to --out (CPU)")
    ap.add_argument("--rms", default="keep", choices=("keep", "native"), help="latent RMS after centring")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    assert args.out, "--out required"
    if args.center_from:
        sd = torch.load(args.center_from, map_location="cpu", weights_only=True)
        new, before, after = center_init(sd, args.rms)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save(new, args.out)
        src_meta = os.path.splitext(args.center_from)[0] + ".json"
        meta = {"center_from": args.center_from, "rms": args.rms, "before": before, "after": after,
                "source_meta": json.load(open(src_meta)) if os.path.isfile(src_meta) else None}
        with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[warmstart] centred latents ({args.rms}): rms {before['rms']:.4f} -> {after['rms']:.4f}, "
              f"spatial dev {before['spatial_dev_rms']:.4f} -> {after['spatial_dev_rms']:.4f}, cell cosine "
              f"{before['cell_pairwise_cosine']:.4f} -> {after['cell_pairwise_cosine']:.4f}; wrote {args.out} ({len(new)} tensors)")
        return
    from VLMEvalKit.vlmeval.vlm.neo.local_models.compressors.perceiver import PerceiverCompressor
    from scripts.perceiver_probe import build_batches, data_args_like_sft, load_model, model_args  # noqa: E402
    from VLMEvalKit.vlmeval.VLMTrainKit.neo.train.build import build_tokenizer

    src = args.src or f"{CKPTS}/NEO_continue_SFT_LLaVA_OV_llava66k_Perceiver_{args.ratio}x/perceiver.bin"
    t0 = time.time()
    device = torch.device("cuda")
    data_args = data_args_like_sft(args.dataset)
    tokenizer = build_tokenizer(model_args("none", 1.0, "pooled"), data_args)
    batches, picked = build_batches(tokenizer, data_args, args.n, args.seed)
    print(f"[warmstart] {len(batches)} samples in {time.time() - t0:.0f}s", flush=True)
    model = load_model(dict(lora=None, vtc="perceiver", ratio=1 / args.ratio, queries="learned", perceiver=None),
                       data_args, device)
    mean_grid, native_rms, n = mean_native_grid(batches, model, device, args.latent_grid)
    raw_rms = mean_grid.pow(2).mean().sqrt().item()
    # spatial structure of the mean grid: RMS of the per-cell deviation from the grid's mean vector
    dev_rms = (mean_grid - mean_grid.mean(dim=(1, 2), keepdim=True)).pow(2).mean().sqrt().item()
    print(f"[warmstart] {n} images: native RMS {native_rms:.4f}, mean-grid RMS {raw_rms:.4f} "
          f"(spatial deviation {dev_rms:.4f}), scale x{native_rms / raw_rms:.3f}", flush=True)

    pooled_sd = torch.load(src, map_location="cpu", weights_only=True)
    fresh = PerceiverCompressor(1 / args.ratio, queries="learned", latent_grid=args.latent_grid)
    sd = build_init(fresh.state_dict(), pooled_sd, mean_grid.cpu(), native_rms)
    fresh.load_state_dict(sd, strict=True)  # the trainer will do the same
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(sd, args.out)
    meta = {"ratio": args.ratio, "n_images": n, "seed": args.seed, "from": src, "native_rms": native_rms,
            "mean_grid_rms_raw": raw_rms, "mean_grid_spatial_dev_rms": dev_rms, "latent_grid": args.latent_grid,
            "sample_ids": picked[:20], "copied": COPIED, "zeroed": ZEROED}
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[warmstart] wrote {args.out} ({len(sd)} tensors) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
