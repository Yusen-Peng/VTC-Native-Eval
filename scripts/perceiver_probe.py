"""Diagnose the collapsed learned-query Perceiver (direction perceiver-learned, backlog item 1).

On N fixed, image-bearing LLaVA-665K samples (seeded, same preprocessing and packing as SFT_perceiver.sh) this
measures the NEO-2B training loss under several (LoRA adapter, compressor, visual input) combinations:

    learned/trained      PerceiverLearned-2x adapter + its trained compressor
    learned/zeros        same adapter, the compressor's output replaced by zeros
    learned/noise        same adapter, the output replaced by Gaussian noise at the native-token RMS
    pooled/trained       Perceiver-2x (pooled queries) adapter + its trained compressor
    pooled/zeros         same adapter, zeros
    base/fixed2x         no adapter, fixed pooling 2x        (= step 0 of the pooled run)
    base/fixed2x-zeros   no adapter, zeros                   (text-only floor of the base LLM)
    base/learned-init    no adapter, freshly initialised learned-query Perceiver (= step 0 of the learned run)
    base/none            no adapter, no compression          (uncompressed reference)
    residual/trained     PerceiverLearnedResidual-2x adapter + its trained compressor (e002)
    residual/no-latents  same, latent grid zeroed (pooled queries + the trained gated residuals)
    residual/fixed       same adapter, plain fixed pooling (= the compressor at step 0)
    residual/zeros       same adapter, zeros
    warm/{trained,zeros,noise}   (--warm-ckpt, e007) warm-started paper-faithful `learned` module + adapter at a checkpoint
    warm-init/trained    (--warm-init) the warm-start init .bin under no adapter (= step 0 of the warm run)

and dumps, for each Perceiver compressor, statistics of its attention and output on the first K images:
attention entropy over keys per query relative to uniform (log N), max attention weight, mean pairwise cosine
of the output tokens, output RMS / native-token RMS, and (learned mode) pairwise cosine of the resized latent
queries. If the LLM ignores its visual tokens, trained == zeros == noise within noise; if the compressor emits
near-identical tokens, the pairwise cosine is near 1.

Run from any directory (absolute paths only):
    python $REPO/scripts/perceiver_probe.py --out probe.json --n 128
Default checkpoints are the six runs under $SCRATCH/VTC-Native-Eval-ckpts/output (ratio 2x). Needs the neo env,
HF_HOME + HF_HUB_OFFLINE=1 and the LLaVA-665K annotation on scratch (see scripts/perceiver_probe.sbatch).
"""
import argparse
import json
import math
import os
import random
import sys
import time
import types

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# training-code import style (VLMEvalKit.vlmeval.*); the second entry makes `import vlmeval` inside config.py
# resolve to this checkout instead of the editable install of another checkout
sys.path.insert(0, os.path.join(REPO, "VLMEvalKit"))
sys.path.insert(0, REPO)

from transformers import set_seed  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

from VLMEvalKit.vlmeval.VLMTrainKit.neo.data.data_processor import (  # noqa: E402
    FlattenedDataCollatorForSupervisedDataset, LazySupervisedDataset)
from VLMEvalKit.vlmeval.VLMTrainKit.neo.train.argument import DataArguments, ModelArguments  # noqa: E402
from VLMEvalKit.vlmeval.VLMTrainKit.neo.train.build import build_model_and_tokenizer, build_tokenizer  # noqa: E402
from VLMEvalKit.vlmeval.vlm.neo.local_models.compressors.fixed_pooling import FixedPoolingCompressor  # noqa: E402
from VLMEvalKit.vlmeval.vlm.neo.local_models.compressors.perceiver import PerceiverCompressor, _rotate_2d  # noqa: E402

CKPTS = "/fs/scratch/PAS2836/jacktaylor/VTC-Native-Eval-ckpts/output"
BASE = "Paranioar/NEO1_0-2B-SFT"


def data_args_like_sft(dataset):
    """DataArguments with the values SFT_perceiver.sh passes to train.py."""
    return DataArguments(image_size=512, patch_size=16, max_pixels=4194304, min_pixels=1310720, dataset_use=dataset,
                         data_flatten=True, downsample_ratio=0.5, dynamic_image_size="native_resolution",
                         max_seq_length=18432, loss_reduction="square")


def model_args(vtc_method, ratio, queries):
    return ModelArguments(model_name_or_path=BASE, tokenizer_name_or_path=BASE, model_max_length=18432,
                          vtc_method=vtc_method, compression_ratio=ratio, perceiver_queries=queries)


def build_batches(tokenizer, data_args, n, seed):
    """n collated single-sample batches of image-bearing LLaVA samples, fixed by seed; condition-independent."""
    ds = LazySupervisedDataset(tokenizer, data_args=data_args)
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    collator = FlattenedDataCollatorForSupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, training_args=types.SimpleNamespace(per_device_train_batch_size=1))
    batches, picked = [], []
    for i in order:
        src = ds.list_data_dict[i]
        if "image" not in src and "images" not in src:
            continue
        try:
            sample = ds.item_fn([dict(src)])
        except Exception as e:  # unreadable image etc.: skip, keep the set deterministic
            print(f"skip sample {i}: {type(e).__name__}: {e}", flush=True)
            continue
        if len(sample["pixel_values"]) != 1:
            continue
        batches.append(collator([sample]))
        picked.append(src.get("id", i))
        if len(batches) == n:
            break
    return batches, picked


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


def load_model(spec, data_args, device):
    """spec: dict(lora, vtc, ratio, queries, perceiver)."""
    set_seed(42)
    model, tokenizer = build_model_and_tokenizer(model_args(spec["vtc"], spec["ratio"], spec["queries"]), data_args)
    if spec.get("perceiver"):
        model.compressor.load_perceiver_weights(spec["perceiver"])
    if spec.get("lora"):
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(model.language_model, spec["lora"], is_trainable=False)
    model.config.use_cache = False
    return model.to(device).eval()


class VisualOverride:
    """Monkeypatch compressor.forward (called directly by forward_with_indices, so hooks would not fire)."""

    def __init__(self, compressor, mode, stash=None, stash_max=0):
        self.c, self.mode, self.stash, self.stash_max = compressor, mode, stash, stash_max
        self.orig = compressor.forward

    def __enter__(self):
        orig = self.orig

        def patched(x):
            if self.mode == "no-latents":
                # learned_residual with its latent grid zeroed: pooled queries + the trained gated residuals
                saved = self.c.latents.data.clone()
                self.c.latents.data.zero_()
                try:
                    out = orig(x)
                finally:
                    self.c.latents.data.copy_(saved)
            elif self.mode == "fixed":
                # plain fixed pooling of the same native tokens (what the compressor emitted at step 0)
                out = FixedPoolingCompressor.forward(self.c, self.c._flatten_tokens(x))
            else:
                out = orig(x)
            if self.stash is not None and len(self.stash) < self.stash_max:
                self.stash.append(x.detach().cpu())
            if self.mode == "zeros":
                return torch.zeros_like(out)
            if self.mode == "noise":
                rms = x.float().pow(2).mean().sqrt()
                return (torch.randn_like(out, dtype=torch.float32) * rms).to(out.dtype)
            return out

        self.c.forward = patched
        return self

    def __exit__(self, *a):
        self.c.forward = self.orig


@torch.no_grad()
def run_condition(model, batches, device, visual="real", stash=None, stash_max=0):
    losses, seq_lens, vis_tokens = [], [], []
    img_id = model.img_context_token_id
    with VisualOverride(model.compressor, visual, stash, stash_max):
        for b in batches:
            inp = to_device(b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**inp)
            losses.append(out.loss.float().item())
            ids = inp["input_ids"][0]
            seq_lens.append(int(ids.numel()))
            h, w = model._get_native_hw(inp["image_grid_hw"][0][0, 0], inp["image_grid_hw"][0][0, 1])
            vis_tokens.append(int(model.compressor.num_output_tokens(h, w)))
            del out
    return {"loss_mean": sum(losses) / len(losses), "loss_std": (torch.tensor(losses).std().item()),
            "n": len(losses), "losses": losses, "seq_len_mean_native": sum(seq_lens) / len(seq_lens),
            "visual_tokens_mean": sum(vis_tokens) / len(vis_tokens)}


@torch.no_grad()
def perceiver_stats(comp: PerceiverCompressor, images):
    """Mirror of PerceiverCompressor.forward that also returns attention statistics. images: list of [1,h,w,D]."""
    dev = comp.q_proj.weight.device
    ent, maxp, cos_out, cos_q, rms_ratio, cos_to_pooled, dev_from_pooled, lat_ratio = [], [], [], [], [], [], [], []
    for x in images:
        x = x.to(dev)
        B, h, w, D = x.shape
        xf = x.reshape(B, h * w, D).to(comp.q_proj.weight.dtype)
        key_h = torch.arange(h, device=dev, dtype=torch.float32).repeat_interleave(w)
        key_w = torch.arange(w, device=dev, dtype=torch.float32).repeat(h)
        if comp.queries != "learned":
            queries = FixedPoolingCompressor.forward(comp, xf)
            q_h, q_w = FixedPoolingCompressor.forward(comp, torch.stack((key_h, key_w), dim=-1)[None])[0].unbind(-1)
            if comp.queries == "learned_residual":  # mirror of PerceiverCompressor.forward
                grid = torch.stack(((q_w + 0.5) / w * 2 - 1, (q_h + 0.5) / h * 2 - 1), dim=-1)
                lat = F.grid_sample(comp.latents, grid[None, None], mode="bilinear",
                                    padding_mode="border", align_corners=False)[0, :, 0].transpose(0, 1)
                lat = lat.to(queries.dtype)
                # size of the learned query shift relative to the pooled query it is added to
                lat_ratio.append((lat.float().pow(2).mean().sqrt() / queries.float().pow(2).mean().sqrt()).item())
                queries = queries + lat
        else:
            gh, gw = comp._cell_grid(h, w)
            queries = F.interpolate(comp.latents, size=(gh, gw), mode="bilinear", align_corners=False)
            queries = queries.flatten(2).transpose(1, 2).expand(B, -1, -1)
            q_h = (torch.arange(gh, device=dev, dtype=torch.float32) * comp.cell_h + (comp.cell_h - 1) / 2).repeat_interleave(gw)
            q_w = (torch.arange(gw, device=dev, dtype=torch.float32) * comp.cell_w + (comp.cell_w - 1) / 2).repeat(gh)
        S = queries.shape[1]
        q = comp.q_proj(comp.norm_q(queries)).view(B, S, comp.num_heads, comp.head_dim).transpose(1, 2)
        kv = comp.norm_kv(xf)
        k = comp.k_proj(kv).view(B, -1, comp.num_heads, comp.head_dim).transpose(1, 2)
        q, k = _rotate_2d(q, q_h, q_w), _rotate_2d(k, key_h, key_w)
        attn = ((q @ k.transpose(-2, -1)) * comp.head_dim ** -0.5).float().softmax(dim=-1)  # [1, heads, S, N]
        N = attn.shape[-1]
        ent.append((-(attn * (attn + 1e-12).log()).sum(-1).mean() / math.log(N)).item())
        maxp.append(attn.max(-1).values.mean().item())
        y = comp.forward(x).float()[0]  # [S, D], the tokens the LLM receives
        yn = F.normalize(y, dim=-1)
        C = yn @ yn.T
        cos_out.append(((C.sum() - C.diagonal().sum()) / (S * (S - 1))).item())
        qn = F.normalize(queries.float()[0], dim=-1)
        Cq = qn @ qn.T
        cos_q.append(((Cq.sum() - Cq.diagonal().sum()) / (S * (S - 1))).item())
        rms_ratio.append((y.pow(2).mean().sqrt() / xf.float().pow(2).mean().sqrt()).item())
        pooled = FixedPoolingCompressor.forward(comp, xf).float()[0]
        if pooled.shape == y.shape:
            cos_to_pooled.append(F.cosine_similarity(y, pooled, dim=-1).mean().item())
            # relative RMS of the change the trained module makes to the fixed-pooling tokens
            dev_from_pooled.append(((y - pooled).pow(2).mean().sqrt() / pooled.pow(2).mean().sqrt()).item())
    mean = lambda v: (sum(v) / len(v)) if v else None  # noqa: E731
    out = {"images": len(images), "attn_entropy_over_log_n": mean(ent), "attn_max_weight": mean(maxp),
           "output_pairwise_cosine": mean(cos_out), "query_pairwise_cosine": mean(cos_q),
           "output_rms_over_native_rms": mean(rms_ratio), "output_cosine_to_fixed_pooling": mean(cos_to_pooled),
           "output_rel_rms_change_from_fixed_pooling": mean(dev_from_pooled)}
    if comp.queries != "learned":
        out["attn_gate_tanh"] = comp.attn_gate.tanh().item()
        out["mlp_gate_tanh"] = comp.mlp_gate.tanh().item()
    if comp.queries != "pooled":
        out["latents_rms"] = comp.latents.float().pow(2).mean().sqrt().item()
    if comp.queries == "learned_residual":
        out["latent_term_rms_over_pooled_query_rms"] = mean(lat_ratio)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="JSON results path")
    ap.add_argument("--n", type=int, default=128, help="number of LLaVA samples")
    ap.add_argument("--k-stats", type=int, default=32, help="images for the compressor statistics")
    ap.add_argument("--seed", type=int, default=0, help="sample selection seed")
    ap.add_argument("--dataset", default="llava_665k")
    ap.add_argument("--ratio", type=int, default=2)
    ap.add_argument("--ckpts", default=CKPTS)
    ap.add_argument("--conditions", nargs="*", default=None, help="subset of condition names")
    ap.add_argument("--warm-ckpt", default=None,
                    help="checkpoint-* dir (adapter + perceiver.bin) of a warm-started `learned` run (e007); enables warm/*")
    ap.add_argument("--warm-init", default=None,
                    help="perceiver_warmstart.py init .bin (step 0 of the warm run, no adapter); enables warm-init/trained")
    args = ap.parse_args()
    device = torch.device("cuda")
    r = args.ratio
    L = f"{args.ckpts}/NEO_continue_SFT_LLaVA_OV_llava66k_PerceiverLearned_{r}x"
    P = f"{args.ckpts}/NEO_continue_SFT_LLaVA_OV_llava66k_Perceiver_{r}x"
    R = f"{args.ckpts}/NEO_continue_SFT_LLaVA_OV_llava66k_PerceiverLearnedResidual_{r}x"
    specs = {
        "learned": dict(lora=L, vtc="perceiver", ratio=1 / r, queries="learned", perceiver=f"{L}/perceiver.bin"),
        "pooled": dict(lora=P, vtc="perceiver", ratio=1 / r, queries="pooled", perceiver=f"{P}/perceiver.bin"),
        "residual": dict(lora=R, vtc="perceiver", ratio=1 / r, queries="learned_residual", perceiver=f"{R}/perceiver.bin"),
        "base-fixed": dict(lora=None, vtc="fixed", ratio=1 / r, queries="pooled", perceiver=None),
        "base-learned-init": dict(lora=None, vtc="perceiver", ratio=1 / r, queries="learned", perceiver=None),
        "base-none": dict(lora=None, vtc="none", ratio=1.0, queries="pooled", perceiver=None),
    }
    if args.warm_ckpt:  # e007: warm-started paper-faithful `learned` module + its LoRA at a training checkpoint
        specs["warm"] = dict(lora=args.warm_ckpt, vtc="perceiver", ratio=1 / r, queries="learned",
                             perceiver=f"{args.warm_ckpt}/perceiver.bin")
    if args.warm_init:  # the same module at step 0 (no adapter): what the LLM saw at the first forward
        specs["warm-init"] = dict(lora=None, vtc="perceiver", ratio=1 / r, queries="learned", perceiver=args.warm_init)
    conditions = [  # (name, spec key, visual)
        ("learned/trained", "learned", "real"), ("learned/zeros", "learned", "zeros"), ("learned/noise", "learned", "noise"),
        ("pooled/trained", "pooled", "real"), ("pooled/zeros", "pooled", "zeros"),
        (f"base/fixed{r}x", "base-fixed", "real"), (f"base/fixed{r}x-zeros", "base-fixed", "zeros"),
        ("base/learned-init", "base-learned-init", "real"), ("base/none", "base-none", "real"),
        # learned_residual (e002+): trained; latents zeroed; plain fixed pooling under the same adapter; zeros
        ("residual/trained", "residual", "real"), ("residual/no-latents", "residual", "no-latents"),
        ("residual/fixed", "residual", "fixed"), ("residual/zeros", "residual", "zeros"),
        # warm-started `learned` (e007): trained module; zeros; noise; the step-0 module under no adapter
        ("warm/trained", "warm", "real"), ("warm/zeros", "warm", "zeros"), ("warm/noise", "warm", "noise"),
        ("warm-init/trained", "warm-init", "real"),
    ]
    conditions = [c for c in conditions if c[1] in specs]
    if args.conditions:
        conditions = [c for c in conditions if c[0] in args.conditions]

    t0 = time.time()
    data_args = data_args_like_sft(args.dataset)
    tokenizer = build_tokenizer(model_args("none", 1.0, "pooled"), data_args)
    hf_logging.set_verbosity_warning()
    batches, picked = build_batches(tokenizer, data_args, args.n, args.seed)
    print(f"[probe] {len(batches)} samples built in {time.time() - t0:.0f}s; ids {picked[:5]}...", flush=True)

    results = {"args": vars(args), "sample_ids": picked, "conditions": {}, "compressor_stats": {}}
    for key in dict.fromkeys(c[1] for c in conditions):
        t1 = time.time()
        model = load_model(specs[key], data_args, device)
        hf_logging.set_verbosity_warning()
        print(f"[probe] model {key} loaded in {time.time() - t1:.0f}s", flush=True)
        stash = [] if specs[key]["vtc"] == "perceiver" else None
        for name, k2, visual in conditions:
            if k2 != key:
                continue
            t2 = time.time()
            res = run_condition(model, batches, device, visual, stash, args.k_stats)
            results["conditions"][name] = res
            print(f"[probe] {name:24s} loss {res['loss_mean']:.4f} +- {res['loss_std']:.3f}  "
                  f"(visual tokens {res['visual_tokens_mean']:.0f}, {time.time() - t2:.0f}s)", flush=True)
        if stash:
            st = perceiver_stats(model.compressor, stash[:args.k_stats])
            results["compressor_stats"][key] = st
            print(f"[probe] stats {key}: " + ", ".join(f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}"
                                                        for a, b in st.items()), flush=True)
        del model
        torch.cuda.empty_cache()
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)

    # paired summary
    c = results["conditions"]
    print("\n| condition | loss | +- | n |\n|---|---|---|---|")
    for name, res in c.items():
        print(f"| {name} | {res['loss_mean']:.4f} | {res['loss_std']:.3f} | {res['n']} |")
    pairs = [("learned/trained", "learned/zeros"), ("learned/trained", "learned/noise"), ("pooled/trained", "pooled/zeros"),
             ("learned/trained", "pooled/trained"), (f"base/fixed{r}x", f"base/fixed{r}x-zeros"),
             ("residual/trained", "residual/no-latents"), ("residual/trained", "residual/fixed"),
             ("residual/trained", "residual/zeros"), ("residual/trained", "pooled/trained"),
             ("warm/trained", "warm/zeros"), ("warm/trained", "warm/noise"), ("warm/trained", "learned/trained"),
             ("warm/trained", "warm-init/trained")]
    print("\n| a | b | mean(a-b) | sd(a-b) |\n|---|---|---|---|")
    for a, b in pairs:
        if a in c and b in c:
            d = torch.tensor(c[a]["losses"]) - torch.tensor(c[b]["losses"])
            print(f"| {a} | {b} | {d.mean():+.4f} | {d.std():.4f} |")
    print(f"[probe] wrote {args.out} in {time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
