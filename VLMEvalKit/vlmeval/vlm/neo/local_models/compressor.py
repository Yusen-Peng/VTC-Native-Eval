import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import downsample


class BaseVTCCompressor(nn.Module):
    """Base class for visual token compression (VTC) methods.
    Input:
        x: [B, H, W, D] or [B, N, D]

    Output:
        x: [B, N', D]
    """

    def __init__(self, compression_ratio: float = 1.0):
        super().__init__()
        if not (0.0 < compression_ratio <= 1.0):
            raise ValueError(f"compression_ratio must be in (0, 1], got {compression_ratio}")
        self.compression_ratio = compression_ratio

    def _flatten_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """make sure the visual features are in shape [B, N, D]."""
        if x.ndim == 4:
            # [B, H, W, D] -> [B, H*W, D]
            B, H, W, D = x.shape
            x = x.reshape(B, H * W, D)
        elif x.ndim == 3:
            pass
        else:
            raise ValueError(f"Expected input with shape [B,H,W,D] or [B,N,D], got {tuple(x.shape)}")
        return x

    def _get_target_num_tokens(self, num_tokens: int) -> int:
        """Number of tokens retained after compression."""
        return max(1, int(num_tokens * self.compression_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FixedPoolingCompressor(BaseVTCCompressor):
    """Fixed 1D pooling."""

    def __init__(self, compression_ratio: float = 1.0):
        super().__init__(compression_ratio)

        self.pooling_factor = int(round(1.0 / compression_ratio))

        if self.pooling_factor < 1:
            raise ValueError(f"Invalid compression ratio: {compression_ratio}")
        self.null_group = nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)

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
