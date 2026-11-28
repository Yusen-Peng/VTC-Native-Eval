import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import visualize_neo_drip as shared
import visualize_neo_llm_drip as llm_viz


"""
cd VLMEvalKit
module load miniconda3/24.1.2-py310
conda activate neo

python visualize_neo_tsne.py --layer 2 --num_tokens 1500
"""


class FeatureCapture:
    """Capture the same visual tokens before and after two LLM layers."""

    def __init__(self, model, layer=2):
        self.model = model
        self.encoder_features = None
        self.llm_features = None
        self.visual_mask = None
        self.handles = []

        language_model = model.language_model
        if hasattr(language_model, "get_base_model"):
            language_model = language_model.get_base_model()

        qwen = language_model.model

        # Post vision encoder.
        self.handles.append(model.vision_model.register_forward_hook(self._vision_hook))

        # Save visual-token positions from the multimodal prefill.
        self.handles.append(
            qwen.register_forward_pre_hook(
                self._llm_pre_hook,
                with_kwargs=True,
            )
        )

        # Output of LLM layer N, before the compressor executes.
        self.handles.append(
            qwen.layers[layer - 1].register_forward_hook(
                self._layer_hook
            )
        )

    def _vision_hook(self, module, inputs, output):
        if self.encoder_features is not None:
            return

        x = output.last_hidden_state
        if x.ndim == 3:
            assert x.shape[0] == 1
            x = x[0]

        self.encoder_features = x.detach().float().cpu()

    def _llm_pre_hook(self, module, args, kwargs):
        mask = kwargs.get("visual_mask")

        # Capture only prefill, not decoding.
        if mask is not None and self.visual_mask is None:
            self.visual_mask = mask.detach().bool().cpu().reshape(-1)

    def _layer_hook(self, module, inputs, output):
        if self.visual_mask is None or self.llm_features is not None:
            return

        x = output[0] if isinstance(output, tuple) else output

        if x.ndim != 3 or x.shape[0] != 1:
            raise RuntimeError(f"Unexpected LLM shape: {x.shape}")

        mask = self.visual_mask.to(x.device)

        if x.shape[1] != mask.numel():
            raise RuntimeError(
                f"Token alignment mismatch: "
                f"hidden={x.shape[1]}, mask={mask.numel()}"
            )

        self.llm_features = x[0, mask].detach().float().cpu()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.no_grad()
def extract_features(model, tokenizer, image_path, prompt, layer,
                     min_pixels, max_pixels):

    _, pixel_values, grid_hw = shared.prepare_neo_image(
        image_path,
        model,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    capture = FeatureCapture(model, layer=layer)

    try:
        model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config={
                "max_new_tokens": 1,
                "do_sample": False,
            },
            grid_hw=grid_hw,
        )
    finally:
        capture.close()

    encoder = capture.encoder_features
    llm = capture.llm_features

    if encoder is None or llm is None:
        raise RuntimeError(
            "Feature capture failed. Check that LLM prefill "
            "passes visual_mask and reaches the requested layer."
        )

    if encoder.shape != llm.shape:
        raise RuntimeError(
            f"Feature shape mismatch: encoder={encoder.shape}, "
            f"LLM={llm.shape}"
        )

    factor = model.vision_model.embeddings.downsample_factor
    grid_h = int(grid_hw[0, 0]) // factor
    grid_w = int(grid_hw[0, 1]) // factor

    assert encoder.shape[0] == grid_h * grid_w, (
        f"Expected {grid_h * grid_w} tokens, "
        f"got {encoder.shape[0]}"
    )

    return encoder, llm, grid_h, grid_w


def compute_joint_tsne(encoder, llm, sample_indices, seed):
    # Identical token indices for both representations.
    a = encoder[sample_indices]
    b = llm[sample_indices]

    # Normalize directions to reduce influence of magnitude.
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    combined = torch.cat([a, b], dim=0).numpy()

    # Shared PCA and t-SNE fit: meaningful within-pair comparison.
    pca_dim = min(50, combined.shape[0] - 1, combined.shape[1])

    reduced = PCA(
        n_components=pca_dim,
        random_state=seed,
    ).fit_transform(combined)

    embedded = TSNE(
        n_components=2,
        perplexity=min(30, (len(combined) - 1) / 3),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(reduced)

    n = len(sample_indices)
    return embedded[:n], embedded[n:]


def plot_scatter_pair(ax_left, ax_right, coords_a, coords_b,
                      colors, title, layer, axis_name):

    for ax, coords, subtitle in [
        (ax_left, coords_a, "Post vision encoder"),
        (ax_right, coords_b, f"Post LLM layer {layer}"),
    ]:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=9,
            alpha=0.8,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")

    # Both panels use identical coordinate bounds.
    merged = np.concatenate([coords_a, coords_b], axis=0)
    low = merged.min(axis=0)
    high = merged.max(axis=0)
    pad = np.maximum((high - low) * 0.06, 1e-3)

    for ax in [ax_left, ax_right]:
        ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
        ax.set_ylim(low[1] - pad[1], high[1] + pad[1])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_base", default=shared.MODEL_BASE)
    parser.add_argument(
        "--llm_drip_path",
        default=llm_viz.DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="Optional trained LoRA checkpoint directory",
    )
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--compression_ratio", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--num_tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_pixels", type=int, default=shared.MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=shared.MAX_PIXELS)
    parser.add_argument("--output_dir", default=shared.OUTPUT_DIR)

    args = parser.parse_args()

    image_dir = "/users/PAS2912/yusenpeng/VTC-Eval/VLMEvalKit/image_examples"

    image_paths = [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if name.lower().endswith(shared.IMAGE_EXTENSIONS)
    ]


    model, tokenizer = llm_viz.build_model(
        args.model_base,
        args.llm_drip_path,
        args.compression_ratio,
        args.layer,
        args.temperature,
    )

    if args.adapter_path:
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            args.adapter_path,
            is_trainable=False,
        )
        print(f"Loaded LoRA adapter: {args.adapter_path}")
    else:
        print("Using base NEO-SFT language model (no trained LoRA).")

    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    # Two figures, each with 10 paired comparisons.
    figures = {}
    for color_axis in ("horizontal", "vertical"):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
        figures[color_axis] = (fig, axes)

    for i, image_path in enumerate(image_paths):
        name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n[{i + 1}/4] Processing {name}", flush=True)

        encoder, llm, h, w = extract_features(
            model, tokenizer, image_path,
            args.prompt, args.layer,
            args.min_pixels, args.max_pixels,
        )

        n = encoder.shape[0]
        rng = np.random.default_rng(args.seed)
        indices = np.sort(
            rng.choice(
                n,
                size=min(args.num_tokens, n),
                replace=False,
            )
        )

        coords_a, coords_b = compute_joint_tsne(
            encoder, llm, indices, args.seed
        )

        # Row-major visual token ordering.
        rows = indices // w
        cols = indices % w

        horizontal_colors = cols / max(w - 1, 1)
        vertical_colors = rows / max(h - 1, 1)

        plot_row = i // 2
        plot_col = (i % 2) * 2

        for color_axis, colors in [
            ("horizontal", horizontal_colors),
            ("vertical", vertical_colors),
        ]:
            fig, axes = figures[color_axis]

            plot_scatter_pair(
                axes[plot_row, plot_col],
                axes[plot_row, plot_col + 1],
                coords_a,
                coords_b,
                colors,
                name,
                args.layer,
                color_axis,
            )

        print(
            f"  Grid={h}x{w}, total={n}, "
            f"sampled={len(indices)}, "
            f"encoder_norm={encoder.norm(dim=-1).mean():.3f}, "
            f"llm_norm={llm.norm(dim=-1).mean():.3f}",
            flush=True,
        )

    for color_axis, (fig, axes) in figures.items():
        fig.suptitle(
            f"NEO visual token t-SNE: encoder vs LLM layer {args.layer}\n"
            f"Points colored by original {color_axis} patch position",
            fontsize=15,
        )

        path = os.path.join(
            args.output_dir,
            f"NEO_tsne_layer{args.layer}_{color_axis}.png",
        )

        fig.savefig(path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
