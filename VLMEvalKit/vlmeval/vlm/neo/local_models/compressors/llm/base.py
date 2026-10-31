import torch
import torch.nn as nn


class BaseLLMVTCCompressor(nn.Module):
    """Base class for LLM-stage visual token compression."""

    def __init__(
        self,
        compression_ratio: float = 1.0,
        compression_layer: int = 2,
    ):
        super().__init__()

        if not (0.0 < compression_ratio <= 1.0):
            raise ValueError(
                f"compression_ratio must be in (0, 1], got {compression_ratio}"
            )

        if compression_layer < 1:
            raise ValueError(
                f"compression_layer must be positive, got {compression_layer}"
            )

        self.compression_ratio = compression_ratio
        self.compression_layer = compression_layer

    def select_indices(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        visual_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Select tokens to retain after an LLM decoder layer.

        Args:
            hidden_states: [B, L, D]
            indexes: [3, L], NEO temporal/spatial position indexes
            visual_mask: [L], True for visual tokens

        Returns:
            keep_indices: [L_new], sorted indices into the input sequence.
                All nonvisual tokens must be preserved.
        """
        raise NotImplementedError