import torch
import torch.nn as nn
from .fixed_pooling import FixedPoolingCompressor
from .registry import register_compressor


@register_compressor("perceiver")
class PerceiverCompressor(FixedPoolingCompressor):
    """Perceiver-style resampler (https://arxiv.org/abs/2103.03206), trained.

    The queries are the fixed-pooling outputs (one per pooling segment), which cross-attend to
    all native tokens of the image and then pass through an MLP. Token count and positions are
    therefore those of ``fixed``. Both residual updates are scaled by tanh-gates that start at 0
    (as in Flamingo), so at init the module is exactly fixed pooling.

    The native tokens already carry NEO's 2D RoPE (applied to the patch embeddings), so the
    keys need no extra position encoding.
    """

    def __init__(
        self,
        compression_ratio: float = 1.0,
        hidden_size: int = 2048,
        head_dim: int = 128,
        mlp_ratio: int = 4,
    ):
        super().__init__(compression_ratio)
        # unused by FixedPoolingCompressor.forward; would be an unused trainable parameter once unfrozen
        del self.null_group
        if hidden_size % head_dim != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by head_dim {head_dim}")
        self.num_heads = hidden_size // head_dim
        self.head_dim = head_dim

        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_ratio * hidden_size),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_size, hidden_size),
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.mlp_gate = nn.Parameter(torch.zeros(1))

    def reset_parameters(self):
        """(Re)initialise every weight. Call after ``from_pretrained``: it re-initialises (or leaves
        uninitialised) all weights missing from the checkpoint, which includes this whole module."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LayerNorm)):
                m.reset_parameters()
        nn.init.zeros_(self.attn_gate)
        nn.init.zeros_(self.mlp_gate)

    def load_perceiver_weights(self, perceiver_path):
        sd = torch.load(perceiver_path, map_location="cpu", weights_only=True)
        self.load_state_dict(sd, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, W, D] -> [B, N, D]
        x = self._flatten_tokens(x)
        if self.pooling_factor == 1: return x

        #weights may be fp32 (training) while the rest of NEO runs in bf16
        out_dtype = x.dtype
        x = x.to(self.q_proj.weight.dtype)
        queries = super().forward(x)  # [B, S, D] fixed-pooling outputs
        B, S, D = queries.shape
        q = self.q_proj(self.norm_q(queries)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.norm_kv(x)
        k = self.k_proj(kv).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5  # [B, heads, S, N]
        attn = attn.float().softmax(dim=-1).to(v.dtype)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)

        queries = queries + self.attn_gate.tanh() * self.o_proj(out)
        queries = queries + self.mlp_gate.tanh() * self.mlp(self.norm_mlp(queries))
        return queries.to(out_dtype)
