from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .base import BaseVTCCompressor
from .registry import register_compressor


# Adapted from: https://github.com/facebookresearch/ToMe/blob/main/tome/merge.py
def bipartite_soft_matching(metric: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokens are split alternately into A (even positions) and B (odd positions). Every A token
    picks its most similar B token; the r most similar (A, B) pairs get merged.

    Args:
        metric: [B, N, C] features used to measure similarity
        r: number of tokens to remove, 0 < r <= N // 2

    Returns (all indexes are positions inside A or inside B, not inside the full sequence):
        unm_idx: [B, |A| - r] A tokens that are kept
        src_idx: [B, r] A tokens that are merged away
        dst_idx: [B, r] B token that each src token is merged into
    """
    with torch.no_grad():
        metric = F.normalize(metric.float(), dim=-1)
        a, b = metric[:, ::2], metric[:, 1::2]
        scores = a @ b.transpose(-1, -2)  # [B, |A|, |B|] cosine similarity

        node_max, node_idx = scores.max(dim=-1)  # best B token for every A token
        edge_idx = node_max.argsort(dim=-1, descending=True)  # A tokens, most similar first

        unm_idx = edge_idx[:, r:]  # Unmerged Tokens
        src_idx = edge_idx[:, :r]  # Merged Tokens
        dst_idx = node_idx.gather(dim=-1, index=src_idx)

    return unm_idx, src_idx, dst_idx


@register_compressor("tome")
class ToMeCompressor(BaseVTCCompressor):
    """Rule Based (Training Free) Token Merging: https://arxiv.org/pdf/2210.09461

    NEO has no vision transformer blocks to merge in between, so merging is applied at a
    single site (after the 2x2 downsample + projection, before the decoder): bipartite soft
    matching is repeated until exactly ``num_output_tokens`` tokens remain.

    Differences to the paper:
        - similarity is measured on the token features (no attention keys available)
        - a merged token takes the position of its destination (B) token
        - no proportional attention
    """

    def __init__(self, compression_ratio: float = 1.0, max_merge_ratio: float = 0.25):
        super().__init__(compression_ratio)
        # one round of bipartite matching can remove at most half of the tokens
        if not (0.0 < max_merge_ratio <= 0.5):
            raise ValueError(f"max_merge_ratio must be in (0, 0.5], got {max_merge_ratio}")
        self.max_merge_ratio = max_merge_ratio

    def kept_indices(self, h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
        raise NotImplementedError(
            "ToMe chooses its tokens from the image features, not from (h, w): "
            "use forward_with_indices (training and precomputed visual_features are not supported)."
        )

    def num_output_tokens(self, h: int, w: int) -> int:
        return self._get_target_num_tokens(h * w)

    def _merge_round(self, x, size, idx, r):
        """Remove r tokens. x: [B, N, D], size: [B, N, 1], idx: [B, N]"""
        D = x.shape[-1]
        unm_idx, src_idx, dst_idx = bipartite_soft_matching(x, r)

        x_a, x_b = x[:, ::2], x[:, 1::2]
        size_a, size_b = size[:, ::2], size[:, 1::2]
        idx_a, idx_b = idx[:, ::2], idx[:, 1::2]

        # size-weighted average of each destination token with the tokens merged into it
        src_size = size_a.gather(1, src_idx[..., None])
        src_x = x_a.gather(1, src_idx[..., None].expand(-1, -1, D)) * src_size
        dst_size = size_b.scatter_add(1, dst_idx[..., None], src_size)
        dst_x = (x_b * size_b).scatter_add(1, dst_idx[..., None].expand(-1, -1, D), src_x) / dst_size

        x = torch.cat([x_a.gather(1, unm_idx[..., None].expand(-1, -1, D)), dst_x], dim=1)
        size = torch.cat([size_a.gather(1, unm_idx[..., None]), dst_size], dim=1)
        # merged tokens keep the position of their destination token
        idx = torch.cat([idx_a.gather(1, unm_idx), idx_b], dim=1)
        return x, size, idx

    def _merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge down to the target count. Returns tokens, sizes and native indices in raster order."""
        dtype = x.dtype
        x = self._flatten_tokens(x).float()  # [B, N, D], merge in float32
        B, N, D = x.shape
        assert B == 1, "ToMe compresses one image at a time"
        K = self._get_target_num_tokens(N)

        size = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)  # patches represented by each token
        idx = torch.arange(N, device=x.device).unsqueeze(0)  # native row-major index of each token

        while x.shape[1] > K:
            n_cur = x.shape[1]
            r = min(n_cur - K, max(1, int(n_cur * self.max_merge_ratio)), n_cur // 2)
            x, size, idx = self._merge_round(x, size, idx, r)

        # merging scrambles the order: restore raster order for the LLM
        idx, order = idx.sort(dim=-1)
        x = x.gather(1, order[..., None].expand(-1, -1, D))
        size = size.gather(1, order[..., None])

        return x.to(dtype), size, idx

    def forward_with_indices(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, _, idx = self._merge(x)
        return x, idx[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_indices(x)[0]
