import torch
import torch.nn as nn
from .base import BaseVTCCompressor
from .registry import register_compressor
from .drip_modules.BP import BoundaryPredictor, downsample

@register_compressor("perceiver")
class PerceiverCompressor(BaseVTCCompressor):
    def __init__(
        self,
        compression_ratio: float = 1.0,
    ):
        super().__init__(compression_ratio)
        
    def load_perceiver_weights(self, drip_path):
        ...

    def num_output_tokens(self, h: int, w: int) -> int:
        N = h * w
        return max(1, min(int(N * self.compression_ratio), N))

    def kept_indices(self, h, w, device=None):
        if self.last_kept_indices is None:
            raise RuntimeError("Perceiver kept_indices() called before forward_with_indices().")

        indices = self.last_kept_indices
        return indices.to(device=device)
    

    def _predict_boundaries(self, x):
        ...

    def forward_with_indices(self, x):
        ...

    def forward(self, x):
        compressed, _ = self.forward_with_indices(x)
        return compressed
