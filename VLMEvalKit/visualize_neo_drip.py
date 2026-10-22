import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from vlmeval.vlm.neo.local_models.configuration_neo_chat import NEOChatConfig
from vlmeval.vlm.neo.local_models.modeling_neo_chat import NEOChatModel
from vlmeval.vlm.neo.utils import load_image_native

"""
python VLMEvalKit/visualize_neo_drip.py --dataset general
python VLMEvalKit/visualize_neo_drip.py --dataset docvqa
"""


MODEL_BASE = "Paranioar/NEO1_0-2B-SFT"
DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/VTC-Native-Eval-ckpts/output/NEO_continue_SFT_LLaVA_OV_llava66k_DRIP_2x_DEBUG/checkpoint-10/drip.bin"
COMPRESSION_RATIO = 0.5


GENERAL_IMAGE_DIR = "/users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl/boundaries/image_examples"
DOCVQA_IMAGE_DIR = "/users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl/boundaries/docvqa_examples"
OUTPUT_DIR = "/users/PAS2912/yusenpeng/VTC-Eval/VLMEvalKit/neo_drip_visualizations"

MIN_PIXELS = 1280 * 32 * 32
MAX_PIXELS = 4096 * 32 * 32

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


def build_neo_drip(model_base=MODEL_BASE, drip_weight_path=DRIP_WEIGHT_PATH, compression_ratio=COMPRESSION_RATIO):
    print(f"Loading NEO model: {model_base}")
    config = NEOChatConfig.from_pretrained(model_base)
    config.vision_config.vtc_method = "drip"
    config.vision_config.compression_ratio = compression_ratio
    config.vision_config.drip_path = None

    model = NEOChatModel.from_pretrained(
        model_base,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    # Load DRIP after Hugging Face finishes model initialization.
    print(f"Loading DRIP checkpoint: {drip_weight_path}")
    model.compressor.load_drip_weights(drip_weight_path)
    model.eval()
    print("NEO + DRIP ready!")
    return model


def prepare_neo_image(img_path, model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    original_img = Image.open(img_path).convert("RGB")
    embeddings = model.vision_model.embeddings

    patch_size = embeddings.patch_size
    downsample_ratio = 1.0 / embeddings.downsample_factor

    pixel_values, grid_hw = load_image_native(
        img_path,
        patch_size=patch_size,
        downsample_ratio=downsample_ratio,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    device = embeddings.patch_embedding.weight.device
    dtype = embeddings.patch_embedding.weight.dtype
    pixel_values = pixel_values.to(device=device,dtype=dtype)
    grid_hw = grid_hw.to(device)
    return original_img, pixel_values, grid_hw


@torch.no_grad()
def get_neo_drip_boundaries(model, pixel_values, grid_hw):
    model.eval()

    output = model.vision_model(pixel_values=pixel_values, grid_hw=grid_hw)
    compressor = model.compressor
    embeddings = model.vision_model.embeddings

    hard = compressor.last_boundaries[0].detach().float().cpu()
    soft = compressor.last_soft_boundaries[0].detach().float().cpu()

    assert grid_hw.shape[0] == 1, "Expected a single image per visualization call."
    factor = embeddings.downsample_factor

    grid_h = int(grid_hw[0, 0].item()) // factor
    grid_w = int(grid_hw[0, 1].item()) // factor

    expected_tokens = grid_h * grid_w

    assert hard.numel() == expected_tokens, f"Hard boundary mismatch: expected={expected_tokens}, actual={hard.numel()}"
    assert soft.numel() == expected_tokens, f"Soft boundary mismatch: expected={expected_tokens}, actual={soft.numel()}"

    num_boundaries = int((hard > 0.5).sum().item())

    assert output.last_hidden_state.shape[0] == num_boundaries, f"Compressed tokens mismatch: output={output.last_hidden_state.shape[0]}, boundaries={num_boundaries}"
    return hard, soft, grid_h, grid_w


@torch.no_grad()
def overlay_neo_drip_boundaries(
    model,
    img_path,
    alpha=0.4,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
):
    (
        original_img,
        pixel_values,
        grid_hw,
    ) = prepare_neo_image(
        img_path,
        model,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    (
        hard_1d,
        soft_1d,
        grid_h,
        grid_w,
    ) = get_neo_drip_boundaries(
        model,
        pixel_values,
        grid_hw,
    )

    hard_mask = hard_1d.view(
        grid_h,
        grid_w,
    ).numpy()

    soft_mask = soft_1d.view(
        grid_h,
        grid_w,
    ).numpy()

    embeddings = model.vision_model.embeddings

    effective_patch_size = (
        embeddings.patch_size
        * embeddings.downsample_factor
    )

    processed_img = original_img.resize(
        (
            grid_w * effective_patch_size,
            grid_h * effective_patch_size,
        ),
        resample=Image.Resampling.BICUBIC,
    )

    orig_np = np.asarray(processed_img).copy()

    img_h, img_w = orig_np.shape[:2]

    assert img_h % grid_h == 0
    assert img_w % grid_w == 0

    cell_h = img_h // grid_h
    cell_w = img_w // grid_w

    overlay_np = orig_np.copy()

    for i in range(grid_h):
        for j in range(grid_w):

            if hard_mask[i, j] > 0.5:

                y0 = i * cell_h
                y1 = (i + 1) * cell_h

                x0 = j * cell_w
                x1 = (j + 1) * cell_w

                patch = overlay_np[
                    y0:y1,
                    x0:x1,
                ]

                red = np.zeros_like(patch)
                red[..., 0] = 255

                overlay_np[
                    y0:y1,
                    x0:x1,
                ] = (
                    (1 - alpha) * patch
                    + alpha * red
                ).astype(np.uint8)

    num_boundaries = int(
        (hard_1d > 0.5).sum().item()
    )

    return (
        Image.fromarray(overlay_np),
        hard_mask,
        soft_mask,
        num_boundaries,
        grid_h,
        grid_w,
        grid_hw.detach().cpu(),
    )

def visualize_neo_soft_probs(
    soft_masks,
    titles,
    save_path,
    figsize=(20, 8),
    dpi=300,
):
    fig, axes = plt.subplots(
        2,
        5,
        figsize=figsize,
    )

    axes = axes.flatten()

    for ax, soft_mask, title in zip(
        axes,
        soft_masks,
        titles,
    ):
        soft_mask = np.asarray(soft_mask)

        im = ax.imshow(
            soft_mask,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )

        grid_h, grid_w = soft_mask.shape

        ax.set_title(
            f"{title}: {grid_h}×{grid_w}",
            fontsize=10,
        )

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    plt.savefig(
        save_path,
        bbox_inches="tight",
        dpi=dpi,
    )

    plt.close(fig)

    print(f"Saved soft probabilities: {save_path}")


def visualize_neo_10_images_2x5(
    model,
    image_paths,
    save_path,
    alpha=0.4,
    titles=None,
    figsize=(20, 8),
    dpi=300,
    title_fontsize=12,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
):
    assert len(image_paths) == 10, (
        f"Expected 10 images, got {len(image_paths)}"
    )

    overlays = []
    soft_masks = []
    stats = []

    for p in image_paths:

        (
            overlay,
            hard_mask,
            soft_mask,
            num_boundaries,
            grid_h,
            grid_w,
            grid_hw,
        ) = overlay_neo_drip_boundaries(
            model,
            p,
            alpha=alpha,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        total_tokens = grid_h * grid_w

        overlays.append(overlay)
        soft_masks.append(soft_mask)

        stats.append(
            (
                num_boundaries,
                total_tokens,
                grid_h,
                grid_w,
            )
        )

        print(
            f"{os.path.basename(p)}: "
            f"grid={grid_h}x{grid_w}, "
            f"L={total_tokens}, "
            f"boundaries={num_boundaries}, "
            f"rate={num_boundaries / total_tokens:.3f}, "
            f"soft_mean={soft_mask.mean():.4f}, "
            f"soft_std={soft_mask.std():.4f}, "
            f"soft_min={soft_mask.min():.4f}, "
            f"soft_max={soft_mask.max():.4f}, "
            f"grid_hw={grid_hw.tolist()}"
        )

    if titles is None:
        titles = [
            os.path.splitext(
                os.path.basename(p)
            )[0]
            for p in image_paths
        ]

    # ========================================================
    # Hard-boundary overlays
    # ========================================================

    fig, axes = plt.subplots(
        2,
        5,
        figsize=figsize,
    )

    axes = axes.flatten()

    for (
        ax,
        overlay,
        title,
        stat,
    ) in zip(
        axes,
        overlays,
        titles,
        stats,
    ):
        (
            num_boundaries,
            total_tokens,
            grid_h,
            grid_w,
        ) = stat

        ax.imshow(overlay)

        ax.set_title(
            f"{title}\n"
            f"{num_boundaries}/{total_tokens} "
            f"({100 * num_boundaries / total_tokens:.1f}%), "
            f"{grid_h}×{grid_w}",
            fontsize=title_fontsize,
        )

        ax.axis("off")

    plt.tight_layout()

    save_dir = os.path.dirname(save_path)

    if save_dir:
        os.makedirs(
            save_dir,
            exist_ok=True,
        )

    plt.savefig(
        save_path,
        bbox_inches="tight",
        dpi=dpi,
    )

    plt.close(fig)

    print(f"Saved hard boundaries: {save_path}")

    # ========================================================
    # Soft-boundary probabilities
    # ========================================================

    root, _ = os.path.splitext(save_path)

    soft_save_path = root + "_soft_probs.pdf"

    visualize_neo_soft_probs(
        soft_masks=soft_masks,
        titles=titles,
        save_path=soft_save_path,
        figsize=figsize,
        dpi=dpi,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["general", "docvqa"],
        default="docvqa",
    )

    parser.add_argument(
        "--drip_path",
        type=str,
        default=DRIP_WEIGHT_PATH,
    )

    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=COMPRESSION_RATIO,
    )

    parser.add_argument(
        "--min_pixels",
        type=int,
        default=MIN_PIXELS,
    )

    parser.add_argument(
        "--max_pixels",
        type=int,
        default=MAX_PIXELS,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
    )

    args = parser.parse_args()

    image_dir = (
        GENERAL_IMAGE_DIR
        if args.dataset == "general"
        else DOCVQA_IMAGE_DIR
    )

    image_paths = [
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    assert len(image_paths) == 10, (
        f"Expected exactly 10 images in {image_dir}, "
        f"found {len(image_paths)}"
    )

    model = build_neo_drip(
        drip_weight_path=args.drip_path,
        compression_ratio=args.compression_ratio,
    )

    checkpoint_name = os.path.basename(
        os.path.dirname(args.drip_path)
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        f"NEO_DRIP_{checkpoint_name}_{args.dataset}.png",
    )

    visualize_neo_10_images_2x5(
        model=model,
        image_paths=image_paths,
        save_path=save_path,
        alpha=args.alpha,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )


if __name__ == "__main__":
    main()