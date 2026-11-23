
import torch

from ...helpers import downsample
from .base import BaseLLMVTCCompressor


class FixedPooling(BaseLLMVTCCompressor):
    """Training-free LLM-stage fixed 1D pooling."""

    def __init__(
        self,
        compression_ratio: float = 1.0,
        compression_layer: int = 2,
    ):
        super().__init__(compression_ratio, compression_layer)

        self.pooling_factor = int(round(1.0 / compression_ratio))

        if self.pooling_factor < 1:
            raise ValueError(f"Invalid compression ratio: {compression_ratio}")

    def pool(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        visual_mask: torch.Tensor,
        return_indices: bool = False,
    ):
        device = hidden_states.device
        visual_mask = visual_mask.to(device=device, dtype=torch.bool)
        image_times = indexes[0].to(device)

        keep_mask = ~visual_mask
        pooled_results = []

        for image_time in image_times[visual_mask].unique():
            image_indices = torch.where(
                visual_mask & (image_times == image_time)
            )[0]

            # [B, N, D]
            features = hidden_states[:, image_indices, :]
            B, N, D = features.shape

            if self.pooling_factor == 1:
                keep_mask[image_indices] = True
                continue

            # Same fixed boundaries as encoder-stage FixedPooling
            boundaries = torch.zeros(
                B, N, device=device, dtype=features.dtype
            )
            boundaries[:, self.pooling_factor - 1::self.pooling_factor] = 1.0
            boundaries[:, -1] = 1.0

            # [B, N, D] -> [N, B, D]
            hidden = features.transpose(0, 1)

            null_group = torch.zeros(
                1, B, D, device=device, dtype=features.dtype
            )

            compressed = downsample(
                boundaries=boundaries,
                hidden=hidden,
                null_group=null_group,
            )

            # [S, B, D] -> [B, S, D]
            compressed = compressed.transpose(0, 1)

            # Same representative selection as encoder-stage FixedPooling:
            # final token of each segment, including incomplete final segment.
            num_output = (N + self.pooling_factor - 1) // self.pooling_factor

            kept = (
                (torch.arange(num_output, device=device) + 1)
                * self.pooling_factor
                - 1
            )

            kept = torch.minimum(
                kept,
                torch.as_tensor(N - 1, device=device, dtype=kept.dtype),
            )

            representatives = image_indices[kept]

            keep_mask[representatives] = True
            pooled_results.append((representatives, compressed))

        keep_indices = torch.where(keep_mask)[0]

        new_hidden_states = hidden_states[:, keep_indices, :].clone()

        for representatives, compressed in pooled_results:
            positions = torch.searchsorted(
                keep_indices, representatives
            )
            new_hidden_states[:, positions, :] = compressed

        result = (
            new_hidden_states,
            indexes[:, keep_indices],
            visual_mask[keep_indices],
        )

        if return_indices:
            return (*result, keep_indices)
        return result

