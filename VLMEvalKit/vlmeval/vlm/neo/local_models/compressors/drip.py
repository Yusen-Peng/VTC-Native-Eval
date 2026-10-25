import torch
import torch.nn as nn
from .base import BaseVTCCompressor
from .registry import register_compressor
from .drip_modules.BP import BoundaryPredictor, downsample

@register_compressor("drip")
class DRIPCompressor(BaseVTCCompressor):
    def __init__(
        self,
        compression_ratio: float = 1.0,
        hidden_size: int = 2048,
        intermediate_size: int = 4096,
        temperature: float = 0.1,
        threshold: float = 0.5,
        smart_init: bool = False,
        drip_path: str = None,
    ):
        super().__init__(compression_ratio)

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

        self.null_token = nn.Parameter(
            torch.zeros(1, 1, hidden_size)
        )

        self.last_boundary_loss = None
        self.last_boundaries = None
        self.last_soft_boundaries = None
        
        self.last_kept_indices = None

    def load_drip_weights(self, drip_path):
        print(f"🌊 Loading DRIP weights from {drip_path}")
        sd = torch.load(drip_path, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        # Support Qwen, LLaVA, and standalone DRIP checkpoints.
        bp_sd = {
            k.split("boundary_predictor.", 1)[1]: v
            for k, v in sd.items()
            if "boundary_predictor." in k
        }

        self.boundary_predictor.load_state_dict(bp_sd, strict=True)

        # Load null token if available and compatible.
        null_token = next(
            (v for k, v in sd.items() if k.endswith("null_token")),
            None,
        )

        if null_token is not None:
            if null_token.shape == self.null_token.shape:
                with torch.no_grad():
                    self.null_token.copy_(null_token)
            else:
                print("⚠️ Skipping null_token: shape mismatch")

        print("✅ DRIP weights loaded")


    def num_output_tokens(self, h: int, w: int) -> int:
        N = h * w
        return max(1, min(int(N * self.compression_ratio), N))

    def kept_indices(self, h, w, device=None):
        if self.last_kept_indices is None:
            raise RuntimeError("DRIP kept_indices() called before forward_with_indices().")

        indices = self.last_kept_indices
        if indices.numel() != self.num_output_tokens(h, w):
            raise RuntimeError(f"DRIP expected {self.num_output_tokens(h, w)} tokens, but selected {indices.numel()}.")
        return indices.to(device=device)
    

    def _predict_boundaries(self, x):
        hidden = x.transpose(0, 1)
        if self.training:
            soft_boundaries, hard_boundaries = self.boundary_predictor(hidden)
        else:
            soft_boundaries, hard_boundaries = self.boundary_predictor.inference(hidden)

        self.last_soft_boundaries = soft_boundaries.detach()
        
        # Force the final token to terminate the last segment.
        hard_boundaries = torch.cat(
            [
                hard_boundaries[:, :-1],
                torch.ones_like(hard_boundaries[:, -1:]),
            ],
            dim=-1,
        )
        return soft_boundaries, hard_boundaries

    def forward_with_indices(self, x):
        x = self._flatten_tokens(x)

        B, N, D = x.shape

        if B != 1:
            raise NotImplementedError("DRIP currently supports one image per call.")

        if D != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {D}")

        if self.compression_ratio == 1.0:
            indices = torch.arange(N, device=x.device)
            self.last_boundary_loss = x.new_zeros(())
            self.last_boundaries = torch.ones(1, N, device=x.device, dtype=x.dtype)
            return x, indices
        soft_boundaries, hard_boundaries = self._predict_boundaries(x)

        K = self.num_output_tokens(1, N)
        # Select exactly K boundaries, always including the last token
        scores = soft_boundaries[:, :-1]
        indices = scores[0].topk(K - 1).indices
        indices = torch.cat([indices, torch.tensor([N - 1], device=x.device)]).sort().values

        # Construct hard Top-K boundaries
        topk_boundaries = torch.zeros_like(soft_boundaries)
        topk_boundaries[:, indices] = 1.0

        # Straight-through estimator: hard forward, soft backward
        hard_boundaries = topk_boundaries.detach() - soft_boundaries.detach() + soft_boundaries

        # Cache selected indices for training sequence compression
        self.last_kept_indices = indices.detach()

        compressed = downsample(boundaries=hard_boundaries, hidden=x.transpose(0, 1), null_group=self.null_token) # [S, B, D]
        compressed = compressed.transpose(0, 1)

        # Original DRIP boundary regularization.
        if self.training:
            self.last_boundary_loss = self.boundary_predictor.calc_loss(soft_boundaries)
        else:
            self.last_boundary_loss = None
        self.last_boundaries = hard_boundaries.detach()

        assert compressed.shape[1] == indices.numel(), (
            f"DRIP token mismatch: "
            f"{compressed.shape[1]} compressed tokens vs "
            f"{indices.numel()} boundaries"
        )
        return compressed, indices

    def forward(self, x):
        compressed, _ = self.forward_with_indices(x)
        return compressed
