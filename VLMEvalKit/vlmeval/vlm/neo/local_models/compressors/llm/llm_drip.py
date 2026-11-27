
import torch
import torch.nn as nn

from .base import BaseLLMVTCCompressor
from ..drip_modules.BP import BoundaryPredictor, downsample


class LLMDRIP(BaseLLMVTCCompressor):
    """
    LLM-stage DRIP/VLVLM with learned segment boundaries.

    Compresses visual tokens after a selected LLM layer.
    Text tokens remain unchanged.

    Supports both training and inference.
    """

    def __init__(
        self,
        compression_ratio: float = 1.0,
        compression_layer: int = 2,
        hidden_size: int = 2048,
        intermediate_size: int = 4096,
        temperature: float = 0.1,
        threshold: float = 0.5,
        smart_init: bool = False,
    ):
        super().__init__(compression_ratio, compression_layer)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.temperature = temperature

        self.boundary_predictor = BoundaryPredictor(
            d_model=hidden_size,
            d_inner=intermediate_size,
            activation_function="gelu",
            temp=temperature,
            prior=compression_ratio,
            bp_type="gumbel",
            threshold=threshold,
            smart_init=smart_init,
        )

        self.null_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.last_boundary_loss = None
        self.last_boundaries = None
        self.last_soft_boundaries = None

    def _predict_boundaries(self, features):
        # [B, N, D] -> [N, B, D]
        hidden = features.transpose(0, 1)
        if self.training:
            soft_boundaries, hard_boundaries = self.boundary_predictor(hidden)
        else:
            soft_boundaries, hard_boundaries = self.boundary_predictor.inference(hidden)
        self.last_soft_boundaries = soft_boundaries.detach()
        # Ensure the final segment always terminates.
        hard_boundaries = torch.cat([hard_boundaries[:, :-1], torch.ones_like(hard_boundaries[:, -1:])],dim=-1)
        return soft_boundaries, hard_boundaries

    def pool(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        visual_mask: torch.Tensor,
        return_indices: bool = False,
    ):
        if hidden_states.shape[0] != 1:
            raise NotImplementedError("LLMDRIP currently supports batch size 1.")
        device = hidden_states.device
        visual_mask = visual_mask.to(device=device, dtype=torch.bool)
        image_times = indexes[0].to(device)
        keep_mask = ~visual_mask
        pooled_results = []
        boundary_losses = []
        # Reset state for this forward pass.
        self.last_boundary_loss = None
        self.last_boundaries = []
        self.last_soft_boundaries = []

        for image_time in image_times[visual_mask].unique():
            image_indices = torch.where(visual_mask & (image_times == image_time))[0]
            # [1, N, D]
            features = hidden_states[:, image_indices, :]
            if features.shape[-1] != self.hidden_size:
                raise ValueError(f"Expected hidden size {self.hidden_size}, got {features.shape[-1]}")

            if self.compression_ratio == 1.0:
                keep_mask[image_indices] = True
                continue

            # Use FP32 for the learned boundary predictor.
            soft_boundaries, hard_boundaries = self._predict_boundaries(features)
            # Representative positions are segment endpoints.
            kept = torch.where(hard_boundaries[0].detach() > 0.5)[0]
            representatives = image_indices[kept]
            keep_mask[representatives] = True

            # [N, B, D]
            compressed = downsample(
                boundaries=hard_boundaries,
                hidden=features.transpose(0, 1),
                null_group=self.null_token,
            )

            # [S, B, D] -> [B, S, D]
            compressed = compressed.transpose(0, 1)

            # Return to the LLM hidden-state dtype.
            compressed = compressed.to(
                dtype=hidden_states.dtype
            )

            assert compressed.shape[1] == kept.numel(), (
                f"DRIP token mismatch: "
                f"{compressed.shape[1]} vs {kept.numel()}"
            )

            pooled_results.append(
                (representatives, compressed)
            )

            self.last_boundaries.append(
                hard_boundaries.detach()
            )
            self.last_soft_boundaries.append(
                soft_boundaries.detach()
            )

            if self.training:
                boundary_losses.append(
                    self.boundary_predictor.calc_loss(
                        soft_boundaries
                    )
                )

        # Preserve text tokens and selected visual positions.
        keep_indices = torch.where(keep_mask)[0]

        new_hidden_states = hidden_states[
            :, keep_indices, :
        ].clone()

        # Substitute the pooled representations.
        for representatives, compressed in pooled_results:
            positions = torch.searchsorted(
                keep_indices,
                representatives,
            )

            new_hidden_states[:, positions, :] = compressed

        if boundary_losses:
            self.last_boundary_loss = torch.stack(boundary_losses).mean()
        result = (
            new_hidden_states,
            indexes[:, keep_indices],
            visual_mask[keep_indices],
        )

        if return_indices:
            return (*result, keep_indices)

        return result
