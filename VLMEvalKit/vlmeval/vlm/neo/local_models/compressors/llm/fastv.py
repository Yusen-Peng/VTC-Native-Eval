import torch

from .base import BaseLLMVTCCompressor


class FastV(BaseLLMVTCCompressor):
    """FastV (training free): https://arxiv.org/pdf/2403.06764

    ``compression_layer`` is the paper's K: pruning happens once, at prefill, after K full layers.
    ``compression_ratio`` is the fraction of each image's tokens kept: the paper's 1 - R.
    """

    @torch.no_grad()
    def select_indices(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        visual_mask: torch.Tensor,
        attn_weights: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Choose the tokens that continue past layer K.

        Args:
            hidden_states: [1, L, D], unused by FastV (ranking only needs the attention)
            indexes: [3, L] t/h/w positions. All tokens of one image share the same t, so indexes[0]
                separates the images of a multi-image prompt.
            visual_mask: [L], True for visual tokens
            attn_weights: [1, heads, L, L] attention weights of layer K-1 (row = query, column = key).
                Softmaxed, in the model's dtype (bf16 on GPU). Masked entries are ~0.
        Returns:
            keep_indices: [L_new] sorted LongTensor of sequence positions. Must contain every non-visual
            position, and for each image ``max(1, int(n * self.compression_ratio))`` of its n tokens, the
            count every vision-stage compressor uses (BaseVTCCompressor._get_target_num_tokens).
        """
        device = attn_weights.device
        visual_mask = visual_mask.to(device=device, dtype=torch.bool)
        image_times = indexes[0].to(device)

        scores = attn_weights[0, :, -1].float().mean(dim=0)  # [heads, L] -> [L]

        keep_mask = ~visual_mask
        for image_time in image_times[visual_mask].unique():
            image_indices = torch.where(visual_mask & (image_times == image_time))[0]
            k = max(1, int(image_indices.numel() * self.compression_ratio))
            keep_mask[image_indices[scores[image_indices].topk(k).indices]] = True

        return torch.where(keep_mask)[0]
