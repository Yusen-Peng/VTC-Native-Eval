"""Visualize LLM-stage DRIP boundaries using existing encoder-DRIP plotting code.

Place this file alongside visualize_neo_drip.py in VLMEvalKit.

Example:
  python visualize_neo_llm_drip.py --dataset general
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

import visualize_neo_drip as shared
from vlmeval.vlm.neo.local_models.configuration_neo_chat import NEOChatConfig
from vlmeval.vlm.neo.local_models.modeling_neo_chat import NEOChatModel

DEFAULT_CHECKPOINT = "/fs/scratch/PAS2836/yusenpeng_checkpoint/VTC-Native-Eval-ckpts/output/NEO_continue_SFT_LLaVA_OV_llava66k_LLM_DRIP_2x_SMOKE/checkpoint-10/llm_drip.bin"

def build_model(model_base, checkpoint_path, compression_ratio, compression_layer, temperature):
    config = NEOChatConfig.from_pretrained(model_base)
    config.vision_config.vtc_method = "none"
    config.vision_config.compression_ratio = 1.0
    config.llm_config.llm_vtc_method = "llm_drip"
    config.llm_config.compression_ratio = compression_ratio
    config.llm_config.compression_layer = compression_layer
    config.llm_config.llm_drip_temperature = temperature

    model = NEOChatModel.from_pretrained(
        model_base,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    compressor = model.language_model.model.llm_compressor
    if compressor is None:
        raise RuntimeError("LLMDRIP was not initialized; check inner llm_config.")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    compressor.load_state_dict(state, strict=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True)
    print(f"Loaded LLMDRIP from {checkpoint_path}", flush=True)
    return model, tokenizer


@torch.no_grad()
def get_llm_drip_boundaries(model, pixel_values, grid_hw):
    """Adapter for visualize_neo_drip.get_neo_drip_boundaries()."""
    model.eval()

    # Existing NEO chat API builds image-context tokens and invokes LLM prefill.
    model.chat(
        tokenizer=model._llm_drip_viz_tokenizer,
        pixel_values=pixel_values,
        question=model._llm_drip_viz_prompt,
        generation_config={"max_new_tokens": 1, "do_sample": False},
        grid_hw=grid_hw,
    )

    compressor = model.language_model.model.llm_compressor
    if not compressor.last_boundaries or not compressor.last_soft_boundaries:
        raise RuntimeError(
            "No LLMDRIP boundaries cached. Check that generate() passes visual_mask "
            "and prefill reaches the compression layer."
        )
    if len(compressor.last_boundaries) != 1:
        raise RuntimeError(
            f"Expected one image, found {len(compressor.last_boundaries)} boundary caches"
        )

    hard = compressor.last_boundaries[0].detach().float().cpu().reshape(-1)
    soft = compressor.last_soft_boundaries[0].detach().float().cpu().reshape(-1)
    factor = model.vision_model.embeddings.downsample_factor
    grid_h = int(grid_hw[0, 0].item()) // factor
    grid_w = int(grid_hw[0, 1].item()) // factor
    expected = grid_h * grid_w

    if hard.numel() != expected or soft.numel() != expected:
        raise RuntimeError(
            f"Boundary count mismatch: hard={hard.numel()}, soft={soft.numel()}, "
            f"expected={expected} ({grid_h}x{grid_w})"
        )
    return hard, soft, grid_h, grid_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["general", "docvqa"], default="docvqa")
    parser.add_argument("--model_base", default=shared.MODEL_BASE)
    parser.add_argument("--llm_drip_path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--compression_layer", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--min_pixels", type=int, default=shared.MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=shared.MAX_PIXELS)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--output_dir", default=shared.OUTPUT_DIR)
    args = parser.parse_args()

    directory = shared.GENERAL_IMAGE_DIR if args.dataset == "general" else shared.DOCVQA_IMAGE_DIR
    image_paths = [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.lower().endswith(shared.IMAGE_EXTENSIONS)
    ]
    if len(image_paths) != 10:
        raise ValueError(f"Expected exactly 10 images in {directory}; found {len(image_paths)}")

    model, tokenizer = build_model(
        args.model_base,
        args.llm_drip_path,
        args.compression_ratio,
        args.compression_layer,
        args.temperature,
    )
    model._llm_drip_viz_tokenizer = tokenizer
    model._llm_drip_viz_prompt = args.prompt

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_name = os.path.basename(os.path.dirname(args.llm_drip_path))
    filename = f"NEO_LLM_DRIP_{checkpoint_name}_{args.dataset}.png"
    out_path = os.path.join(args.output_dir, filename)

    # Reuse BOTH the existing overlay renderer and 2x5 plotting code.
    # Only swap the boundary-extraction function for this process.
    old_get = shared.get_neo_drip_boundaries
    shared.get_neo_drip_boundaries = get_llm_drip_boundaries
    try:
        shared.visualize_neo_10_images_2x5(
            model=model,
            image_paths=image_paths,
            save_path=out_path,
            alpha=args.alpha,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    finally:
        shared.get_neo_drip_boundaries = old_get


if __name__ == "__main__":
    main()
