from typing import Optional

import torch

from .base import BaseVTCCompressor
from .registry import register_compressor


@register_compressor("none")
class IdentityCompressor(BaseVTCCompressor):
    def __init__(self, compression_ratio: float = 1.0, max_merge_ratio: float = 0.25):
        super().__init__(compression_ratio)

        self.max_merge_ratio = max_merge_ratio

    def kept_indices(self, h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.arange(h * w, device=device)

    def num_output_tokens(self, h: int, w: int) -> int:
        return h * w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._flatten_tokens(x)
