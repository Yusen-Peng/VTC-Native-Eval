from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .base import BaseVTCCompressor
from .registry import register_compressor


@torch.no_grad()
def cdpruner_select(visual_features: torch.Tensor, text_embeds: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """
    CDPruner greedy DPP MAP selection.
    """
    N, D = visual_features.shape
    K = num_tokens

    assert text_embeds.ndim == 2
    assert text_embeds.shape[-1] == D
    assert 1 <= K <= N

    image_normalized = F.normalize(visual_features.float(), dim=-1)

    similarity = image_normalized @ image_normalized.T

    text_normalized = F.normalize(text_embeds.float(), dim=-1)

    relevance = image_normalized @ text_normalized.T

    # Preserve the Qwen3-VL implementation.
    relevance = (-relevance).mean(dim=-1)

    relevance_min = relevance.min()
    relevance_max = relevance.max()

    relevance = (relevance - relevance_min + 1e-6) / (relevance_max - relevance_min + 1e-6)
    kernel = relevance.unsqueeze(-1) * similarity * relevance.unsqueeze(0)

    cis = torch.zeros(K, N, device=kernel.device, dtype=kernel.dtype)
    di2s = torch.diagonal(kernel).clone()
    select_idx = torch.empty(K, device=kernel.device, dtype=torch.long)

    for i in range(K):
        j = torch.argmax(di2s)
        select_idx[i] = j
        numerator = (
            kernel[j]
            - torch.sum(
                cis[:i, j].unsqueeze(-1) * cis[:i],
                dim=0,
            )
        )
        denominator = torch.sqrt(di2s[j].clamp_min(1e-12))
        eis = numerator / denominator
        cis[i] = eis
        di2s -= eis.square()
        di2s[j] = -float("inf")
    return select_idx.sort().values


@register_compressor("cdpruner")
class CDPrunerCompressor(BaseVTCCompressor):
    """CDPruner: Training-free, instruction-aware visual token pruning."""

    def __init__(self, compression_ratio: float = 1.0):
        super().__init__(compression_ratio)

    def kept_indices(
        self,
        h: int,
        w: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "CDPruner requires visual features and instruction embeddings. "
            "Use forward_with_indices()."
        )

    def num_output_tokens(self, h: int, w: int) -> int:
        return self._get_target_num_tokens(h * w)

    @torch.no_grad()
    def forward_with_indices(self, x: torch.Tensor, text_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # [1, H, W, D] -> [1, N, D]
        x = self._flatten_tokens(x)
        B, N, D = x.shape
        assert B == 1, ("CDPruner currently supports one image at a time.")
        K = self._get_target_num_tokens(N)

        if K == N:
            return x, torch.arange(N, device=x.device)
        if text_embeds is None:
            raise ValueError("CDPruner requires instruction embeddings.")
        indices = cdpruner_select(visual_features=x[0], text_embeds=text_embeds, num_tokens=K)
        assert indices.numel() == K, (f"CDPruner selected {indices.numel()} tokens, expected {K}")
        return x[:, indices, :], indices

    def forward(self, x: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        return self.forward_with_indices(x, text_embeds=text_embeds)[0]

