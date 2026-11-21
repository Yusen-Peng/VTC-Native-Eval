"""Prefill TFLOPs of every training-free VTC method, counted with fvcore on one fixed reference input.

Each base model is loaded once; the compressor inside NEOVisionEmbeddings is then swapped for every
(method, ratio) and one prefill pass is counted, with the lm_head applied to the last position only
(as generate() does). Decoding is not counted.
  - LLM (>99% of the total): fvcore, via jit.trace.
  - vision embedding + compressor: torch FlopCounterMode, run eagerly, halved to fvcore's convention.
    The compressors use tensor shapes as Python ints (seeds, clamp bounds, merge loops), which
    jit.trace turns into tensors, so they cannot be traced.
Both count one multiply-add as one FLOP; elementwise ops (softmax, RMSNorm, RoPE) are not counted.

The reference input is a synthetic image at the max_pixels cap (2048x2048 -> 64x64 = 4096 native
visual tokens) plus a short fixed prompt, so the numbers are per-image and benchmark-independent.

    pip install fvcore                                   # once, in the neo env
    cd VLMEvalKit && python ../scripts/flops.py          # NEO-2B-SFT and NEO-9B-SFT
    cd VLMEvalKit && python ../scripts/flops.py --models NEO-2B-SFT --image-size 1536

Writes vlmeval/results/flops.csv (one row per scale x method x ratio), read by tradeoff_analysis.py.
"""
import argparse
import gc
import os.path as osp

import pandas as pd
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from torch.utils.flop_counter import FlopCounterMode

from vlmeval.config import supported_VLM
from vlmeval.vlm.neo.local_models.compressors import build_compressor
from vlmeval.vlm.neo.local_models.compressors import FastV, PyramidDrop
from vlmeval.vlm.neo.local_models.conversation import get_conv_template
from vlmeval.vlm.neo.utils import load_image_native, preprocess_pixel_values, smart_resize

# vtc_method -> (label used in training_free.csv, ratios); spatial 2x equals fixed 2x, as in summarize_table2.py
METHODS = {
    "fixed": ("fixed pooling", [2, 4, 8]),
    "spatial": ("spatial pooling", [4, 8]),
    "random": ("random pruning", [2, 4, 8]),
    "tome": ("ToMe", [2, 4, 8]),
    "prunesid": ("PruneSID", [2, 4, 8]),
    "cdpruner": ("CDPruner", [2, 4, 8]),
    "perceiver": ("Perceiver", [2, 4, 8]),  # untrained weights: FLOPs don't depend on them
    "drip": ("VLVLM", [2, 4, 8]),
    "perceiver_learned": ("Perceiver (learned queries)", [2, 4, 8]),
    "perceiver_learned_residual": ("Perceiver (pooled + learned residual queries)", [2, 4, 8]),
}

# LLM-stage compression methods
LLM_METHODS = {
    "pdrop-16": ("PyramidDrop-16", [2, 4, 8]),
    "fastv-2": ("FastV-2", [2, 4, 8]),
}

IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = "<img>", "</img>", "<IMG_CONTEXT>"


class PrefillLM(nn.Module):
    """LLM part of one NEOChatModel.generate() prefill, returning only the last-position logits."""

    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model

    def forward(self, input_embeds, indexes, visual_mask):
        outputs = self.language_model(
            inputs_embeds=input_embeds, indexes=indexes, visual_mask=visual_mask, use_cache=False, logits_to_keep=1
        )
        return outputs.logits


def vision_prefill(model, pixel_values, grid_hw, input_ids):
    """Vision embedding + compressor, run eagerly as in NEOChatModel.generate().

    Returns the LLM inputs (input_embeds, indexes) and the vision FLOPs (one multiply-add = one FLOP).
    """
    with FlopCounterMode(display=False) as counter:
        text_embeds = None
        if model.vision_model.embeddings.vtc_method == "cdpruner":
            text_mask = ((input_ids != model.img_context_token_id) & (input_ids != model.img_start_token_id))
            if hasattr(model, "img_end_token_id") and model.img_end_token_id is not None:
                text_mask &= input_ids != model.img_end_token_id

            text_input_ids = input_ids[text_mask]
            text_embeds = model.language_model.get_input_embeddings()(text_input_ids)
        vit_embeds, visual_positions = model.extract_feature_with_positions(pixel_values, grid_hw=grid_hw, text_embeds=text_embeds)

    indexes = model.get_thw_indexes(input_ids[0], grid_hw, visual_positions=visual_positions)
    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    selected = input_ids == model.img_context_token_id
    input_embeds[selected] = vit_embeds.reshape(-1, input_embeds.shape[-1]).to(input_embeds.dtype)
    return input_embeds, indexes, counter.get_total_flops() / 2


def reference_image(kw, image_path, image_size, device):
    """(pixel_values, grid_hw) of the reference image, preprocessed exactly as NEOChat does."""
    factor = int(kw["patch_size"] / kw["downsample_ratio"])
    if image_path is not None:
        pixel_values, grid_hw = load_image_native(
            image_path, patch_size=kw["patch_size"], downsample_ratio=kw["downsample_ratio"],
            min_pixels=kw["min_pixels"], max_pixels=kw["max_pixels"],
        )
    else:
        h, w = smart_resize(image_size, image_size, factor=factor,
                            min_pixels=kw["min_pixels"], max_pixels=kw["max_pixels"])
        gen = torch.Generator().manual_seed(0)
        pixel_values, grid_hw = preprocess_pixel_values(torch.randn(3, h, w, generator=gen), patch_size=kw["patch_size"])
    return pixel_values.to(device, torch.bfloat16), grid_hw.to(device)


def build_input_ids(model, tokenizer, prompt, grid_hw, device):
    """Same query construction as NEOChatModel.chat(), for the currently installed compressor."""
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], "<image>\n" + prompt)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    num_tokens = model._get_num_visual_tokens(grid_hw[0, 0], grid_hw[0, 1])
    query = query.replace("<image>", IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_tokens + IMG_END_TOKEN, 1)
    return tokenizer(query, return_tensors="pt")["input_ids"].to(device), num_tokens


def count_llm(prefill_lm, input_embeds, indexes, visual_mask):
    fca = FlopCountAnalysis(prefill_lm, (input_embeds, indexes, visual_mask))
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    fca.tracer_warnings("none")
    return fca.total(), fca.unsupported_ops()


def measure_latency(fn, warmup=10, repeats=30):
    """Measure GPU execution latency using CUDA events. Returns average latency in milliseconds."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn()

        torch.cuda.synchronize()

        times = []

        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            fn()
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    return sum(times) / len(times)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["NEO-2B-SFT", "NEO-9B-SFT"])
    ap.add_argument("--image", default=None, help="real reference image instead of the synthetic one")
    ap.add_argument("--image-size", type=int, default=2048, help="side of the synthetic square image")
    ap.add_argument("--prompt", default="What is shown in this image? Answer with a single word or phrase.")
    ap.add_argument("--out", default=osp.join("vlmeval", "results", "flops.csv"))
    args = ap.parse_args()

    rows = []
    for key in args.models:
        kw = supported_VLM[key].keywords
        scale = key.split("-")[1]  # NEO-2B-SFT -> 2B
        vlm = supported_VLM[key]()
        model, tokenizer = vlm.model, vlm.tokenizer
        model.eval()
        model.system_message = vlm.system_prompt  # as NEOChat.generate_inner does
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        prefill_lm = PrefillLM(model.language_model).eval()
        model.language_model.model.llm_compressor = None
        pixel_values, grid_hw = reference_image(kw, args.image, args.image_size, vlm.device)
        embeddings = model.vision_model.embeddings

        configs = [("none", f"NEO-{scale}-SFT", 1)]
        configs += [(m, label, r) for m, (label, ratios) in METHODS.items() for r in ratios]
        for vtc_method, label, ratio in configs:
            kwargs = {}
            if vtc_method.startswith("perceiver"):
                kwargs = {"hidden_size": embeddings.llm_embed_dim,
                          "queries": vtc_method[len("perceiver_"):] or "pooled"}
                vtc_method = "perceiver"
            
            if vtc_method == "drip":
                kwargs = {
                    "hidden_size": embeddings.llm_embed_dim,
                    "intermediate_size": embeddings.llm_embed_dim * 2,
                }

            embeddings.vtc_method = vtc_method
            embeddings.compression_ratio = 1.0 / ratio
            embeddings.compressor = build_compressor(vtc_method, compression_ratio=1.0 / ratio, **kwargs).to(
                device=vlm.device, dtype=model.dtype).eval()
            input_ids, num_visual = build_input_ids(model, tokenizer, args.prompt, grid_hw, vlm.device)
            with torch.no_grad():
                input_embeds, indexes, vision = vision_prefill(model, pixel_values, grid_hw, input_ids)
                visual_mask = input_ids[0] == model.img_context_token_id
                llm, unsupported = count_llm(prefill_lm, input_embeds, indexes, visual_mask)
            total = vision + llm

            # Vision encoder + compression latency
            text_embeds = None
            if vtc_method == "cdpruner":
                text_mask = (
                    (input_ids != model.img_context_token_id)
                    & (input_ids != model.img_start_token_id)
                )
                if hasattr(model, "img_end_token_id") and model.img_end_token_id is not None:
                    text_mask &= input_ids != model.img_end_token_id
                text_input_ids = input_ids[text_mask]
                text_embeds = model.language_model.get_input_embeddings()(text_input_ids)
            vision_latency = measure_latency(lambda: model.extract_feature_with_positions(pixel_values, grid_hw=grid_hw, text_embeds=text_embeds))
            
            # LLM prefill latency
            llm_latency = measure_latency(lambda: prefill_lm(input_embeds, indexes, visual_mask))
            total_latency = vision_latency + llm_latency

            rows.append({
                "Model": label, "scale": scale, "compression": float(ratio),
                "visual_tokens": num_visual, "seq_len": input_ids.shape[1],
                "TFLOPs": total / 1e12, "vision_TFLOPs": vision / 1e12, "llm_TFLOPs": llm / 1e12,
                "vision_latency_ms": vision_latency, "llm_latency_ms": llm_latency, "total_latency_ms": total_latency,
            })
            print(f"{key} {vtc_method} {ratio}x: {num_visual} visual / {input_ids.shape[1]} total tokens, "
                  f"{total / 1e12:.3f} TFLOPs (vision {vision / 1e12:.4f}, LLM {llm / 1e12:.3f}) | latency {total_latency:.3f} (vision {vision_latency:.3f}, LLM {llm_latency:.3f}) ms", flush=True)
        print(f"{key}: ops not counted by fvcore: {sorted(unsupported)}", flush=True)


        # LLM-stage compression: restore uncompressed vision tokens
        embeddings.vtc_method = "none"
        embeddings.compression_ratio = 1.0
        embeddings.compressor = build_compressor("none", compression_ratio=1.0).to(device=vlm.device, dtype=model.dtype)
        input_ids, num_visual = build_input_ids(model, tokenizer, args.prompt, grid_hw, vlm.device)
        with torch.no_grad():
            input_embeds, indexes, vision = vision_prefill(model, pixel_values, grid_hw, input_ids)
        visual_mask = input_ids[0] == model.img_context_token_id
        for method, (label, ratios) in LLM_METHODS.items():
            method_name, layer = method.rsplit("-", 1)
            layer = int(layer)

            for ratio in ratios:
                if method_name == "pdrop":
                    model.language_model.model.llm_compressor = PyramidDrop(
                        compression_ratio=1.0 / ratio,
                        compression_layer=layer,
                    ).to(device=vlm.device, dtype=model.dtype)

                elif method_name == "fastv":
                    model.language_model.model.llm_compressor = FastV(
                        compression_ratio=1.0 / ratio,
                        compression_layer=layer,
                    ).to(device=vlm.device, dtype=model.dtype)
                else:
                    raise ValueError(f"Unknown LLM compressor: {method_name}")
                with torch.no_grad():
                    llm, unsupported = count_llm(prefill_lm, input_embeds, indexes, visual_mask)
                total = vision + llm

                vision_latency = measure_latency(
                    lambda: model.extract_feature_with_positions(
                        pixel_values,
                        grid_hw=grid_hw,
                    )
                )
                llm_latency = measure_latency(lambda: prefill_lm(input_embeds, indexes, visual_mask))
                total_latency = vision_latency + llm_latency

                rows.append({
                    "Model": label,
                    "scale": scale,
                    "compression": float(ratio),
                    "visual_tokens": num_visual,
                    "seq_len": input_ids.shape[1],
                    "TFLOPs": total / 1e12,
                    "vision_TFLOPs": vision / 1e12,
                    "llm_TFLOPs": llm / 1e12,
                    "vision_latency_ms": vision_latency,
                    "llm_latency_ms": llm_latency,
                    "total_latency_ms": total_latency,
                })
                print(
                    f"{key} {method} {ratio}x: "
                    f"{num_visual} visual / {input_ids.shape[1]} total tokens, "
                    f"{total / 1e12:.3f} TFLOPs "
                    f"(vision {vision / 1e12:.4f}, LLM {llm / 1e12:.3f}) | "
                    f"latency {total_latency:.3f} "
                    f"(vision {vision_latency:.3f}, LLM {llm_latency:.3f}) ms",
                    flush=True,
                )

        model.language_model.model.llm_compressor = None

        del prefill_lm, model, vlm
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
