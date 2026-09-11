import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseVTCCompressor(nn.Module):
    """
    Base class for visual token compression (VTC) methods.

    Args:
        compression_ratio:
            Fraction of visual tokens retained after compression.

            1.0  -> no compression
            0.5  -> keep 1/2 tokens  (2x compression)
            0.25 -> keep 1/4 tokens  (4x compression)

    Input:
        x: [B, H, W, D] or [B, N, D]

    Output:
        x: [B, N', D]
    """

    def __init__(self, compression_ratio: float = 1.0):
        super().__init__()

        if not (0.0 < compression_ratio <= 1.0):
            raise ValueError(
                f"compression_ratio must be in (0, 1], "
                f"got {compression_ratio}"
            )

        self.compression_ratio = compression_ratio

    def _flatten_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert visual features into [B, N, D].
        """
        if x.ndim == 4:
            # [B, H, W, D] -> [B, H*W, D]
            B, H, W, D = x.shape
            x = x.reshape(B, H * W, D)

        elif x.ndim == 3:
            # already [B, N, D]
            pass

        else:
            raise ValueError(
                f"Expected input with shape [B,H,W,D] or [B,N,D], "
                f"got {tuple(x.shape)}"
            )

        return x

    def _get_target_num_tokens(self, num_tokens: int) -> int:
        """
        Number of tokens retained after compression.
        """
        return max(1, int(num_tokens * self.compression_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError