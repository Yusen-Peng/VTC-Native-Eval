from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .base import BaseVTCCompressor
from .registry import register_compressor

# pooling factor -> (rows, cols) of each block; wide blocks keep lines of text apart
DEFAULT_BLOCKS = {1: (1, 1), 2: (1, 2), 4: (2, 2), 8: (2, 4)}

@register_compressor("spatial")
class SpatialPoolingCompressor(BaseVTCCompressor):
    """Fixed 2D pooling: average each (bh, bw) block of the native token grid.

    Edge blocks of grids not divisible by the block are partial and averaged over the
    tokens they contain (no zero padding), so the output has ceil(h/bh) * ceil(w/bw) tokens.
    Each block is placed at its bottom-right native token (clamped to the grid), i.e. its
    last token in raster order, as in fixed 1D pooling.
    """

    def __init__(self, compression_ratio: float = 1.0, block: Optional[Tuple[int, int]] = None):
        super().__init__(compression_ratio)

        if block is None:
            pooling_factor = int(round(1.0 / compression_ratio))
            if pooling_factor not in DEFAULT_BLOCKS:
                raise ValueError(
                    f"No default block for compression ratio {compression_ratio}; "
                    f"pass block=(bh, bw) or use one of 1/{sorted(DEFAULT_BLOCKS)}"
                )
            block = DEFAULT_BLOCKS[pooling_factor]
        self.bh, self.bw = block

    def _out_grid(self, h: int, w: int) -> Tuple[int, int]:
        return -(-h // self.bh), -(-w // self.bw)

    def kept_indices(self, h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
        oh, ow = self._out_grid(h, w)
        rows = ((torch.arange(oh, device=device) + 1) * self.bh - 1).clamp_(max=h - 1)
        cols = ((torch.arange(ow, device=device) + 1) * self.bw - 1).clamp_(max=w - 1)
        # block-major raster order -> sorted
        return (rows[:, None] * w + cols[None, :]).flatten()

    def num_output_tokens(self, h: int, w: int) -> int:
        oh, ow = self._out_grid(h, w)
        return oh * ow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Spatial pooling needs the token grid [B,H,W,D], got {tuple(x.shape)}")
        if (self.bh, self.bw) == (1, 1):
            return self._flatten_tokens(x)

        # [B, H, W, D] -> [B, D, H, W]; ceil_mode averages partial edge blocks over their real tokens
        pooled = F.avg_pool2d(
            x.permute(0, 3, 1, 2).float(),
            kernel_size=(self.bh, self.bw),
            stride=(self.bh, self.bw),
            ceil_mode=True,
        )
        # [B, D, H', W'] -> [B, H'*W', D]
        return pooled.flatten(2).transpose(1, 2).to(x.dtype)
