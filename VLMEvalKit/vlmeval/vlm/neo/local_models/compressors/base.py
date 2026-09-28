from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseVTCCompressor(nn.Module):
    """Base class for visual token compression (VTC) methods.

    A compressor sees ONE image at a time, as its native token grid of shape (h, w)
    (i.e. after NEO's 2x2 downsample), flattened in row-major order.

    Input:
        x: [B, H, W, D] or [B, N, D]

    Output:
        x: [B, N', D]

    Besides ``forward``, the language model needs to know, per image:
        - how many tokens come out          -> ``num_output_tokens``
        - which (h, w) position each one has -> ``output_positions``
        - which IMG_CONTEXT slots survive    -> ``kept_indices``
    Subclasses whose output tokens each correspond to one native token (pooling, pruning)
    only implement ``kept_indices``; the other two are derived from it.

    NOTE: ``num_output_tokens`` must depend on (h, w) only, never on the features: the
    prompt (number of IMG_CONTEXT slots) is built before the vision forward pass.

    Static methods (pooling, pruning) choose their representatives from (h, w) alone and
    only implement ``kept_indices`` + ``forward``.
    Data-dependent methods (e.g. ToMe) choose them from the features: they override
    ``forward_with_indices`` and leave ``kept_indices`` unimplemented.
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

    def kept_indices(self, h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Row-major index (into the h*w native grid) of the token representing each output token.

        Returns a sorted 1D LongTensor of length ``num_output_tokens(h, w)``.
        """
        raise NotImplementedError

    def num_output_tokens(self, h: int, w: int) -> int:
        """Number of visual tokens the LLM receives for an image with native grid (h, w)."""
        return self.kept_indices(h, w).numel()

    def output_positions(self, h: int, w: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """(pos_h, pos_w) of each output token, used for the LLM's 2D position indexes."""
        idx = self.kept_indices(h, w, device=device)
        return idx // w, idx % w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_with_indices(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress one image and report which native token represents each output token.

        Args:
            x: [1, H, W, D]

        Returns:
            tokens: [1, N', D]
            idx: [N'] sorted LongTensor of row-major indices into the H*W native grid,
                aligned with ``tokens`` (tokens[:, i] sits at position idx[i]).
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape [B,H,W,D], got {tuple(x.shape)}")
        h, w = x.shape[1:3]
        return self.forward(x), self.kept_indices(h, w, device=x.device)
