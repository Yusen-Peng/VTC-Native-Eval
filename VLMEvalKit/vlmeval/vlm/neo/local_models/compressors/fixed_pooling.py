from typing import Optional

import torch
import torch.nn as nn

from ..helpers import downsample
from .base import BaseVTCCompressor
from .registry import register_compressor


@register_compressor("fixed")
class FixedPoolingCompressor(BaseVTCCompressor):
    """Fixed 1D pooling."""

    def __init__(self, compression_ratio: float = 1.0):
        super().__init__(compression_ratio)

        self.pooling_factor = int(round(1.0 / compression_ratio))

        if self.pooling_factor < 1:
            raise ValueError(f"Invalid compression ratio: {compression_ratio}")
        self.null_group = nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)

    def kept_indices(self, h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
        # use the final token in each pooling segment as the representative;
        # the final (possibly incomplete) segment is represented by the last token.
        N = h * w
        idx = (torch.arange(self.num_output_tokens(h, w), device=device) + 1) * self.pooling_factor - 1
        return idx.clamp_(max=N - 1)

    def num_output_tokens(self, h: int, w: int) -> int:
        return (h * w + self.pooling_factor - 1) // self.pooling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, W, D] -> [B, N, D]
        x = self._flatten_tokens(x)
        if self.pooling_factor == 1:
            return x

        B, N, D = x.shape
        boundaries = torch.zeros(B, N, device=x.device, dtype=x.dtype) # [B, N]

        boundaries[:, self.pooling_factor - 1::self.pooling_factor] = 1.0
        # required: always close the final segment
        boundaries[:, -1] = 1.0

        hidden = x.transpose(0, 1) # [N, B, D]
        null_group = torch.zeros(1, B, D, device=x.device, dtype=x.dtype) # [1, B, D]
        compressed = downsample(boundaries=boundaries, hidden=hidden, null_group=null_group) # [S, B, D]
        compressed = compressed.transpose(0, 1) # [B, S, D]
        return compressed
