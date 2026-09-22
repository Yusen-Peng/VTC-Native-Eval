from typing import Optional

import torch

from .base import BaseVTCCompressor
from .registry import register_compressor


@register_compressor("random")
class RandomPruningCompressor(BaseVTCCompressor):
    """Uniformly drop indices w/out replacement"""

    def __init__(self, compression_ratio: float = 1.0, seed: int = 0):
        super().__init__(compression_ratio)
        self._seed = seed

    def kept_indices(self, h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
        curr_seed = self._seed * 1_000_003 + h * 10_007 + w
        gen = torch.Generator(device='cpu').manual_seed(curr_seed)
        K = self._get_target_num_tokens(h*w)
        N = h * w
        indices = torch.randperm(N, generator=gen)[:K].to(device)
        return torch.sort(indices).values

    def num_output_tokens(self, h: int, w: int) -> int:
        return self._get_target_num_tokens(h*w) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, W, D] -> [B, N, D]
        assert x.ndim==4
        B,H,W,D = x.shape

        indices = self.kept_indices(H,W, x.device)
        
        x = self._flatten_tokens(x)
        return x[:, indices]
