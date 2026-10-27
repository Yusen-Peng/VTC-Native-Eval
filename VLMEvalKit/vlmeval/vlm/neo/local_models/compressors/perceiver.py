import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fixed_pooling import FixedPoolingCompressor
from .registry import register_compressor


def _cell_shape(pooling_factor: int):
    """Split a pooling factor r into a (rows, cols) cell: 2 -> (1, 2), 4 -> (2, 2), 8 -> (2, 4)."""
    rows = 2 ** (int(math.log2(pooling_factor)) // 2)
    return rows, pooling_factor // rows


def _rotate_2d(t: torch.Tensor, pos_h: torch.Tensor, pos_w: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """2D RoPE inside attention: the first half of each head rotates with the row, the second with the column.

    t: [B, heads, L, head_dim]; pos_h, pos_w: [L] (may be fractional, e.g. cell centres).
    """
    d = t.shape[-1] // 2
    inv_freq = 1.0 / theta ** (torch.arange(0, d, 2, device=t.device, dtype=torch.float32) / d)
    halves = []
    for x, pos in ((t[..., :d], pos_h), (t[..., d:], pos_w)):
        angle = pos.float()[:, None] * inv_freq  # [L, d/2]
        cos, sin = angle.cos(), angle.sin()
        x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
        halves.append(torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2))
    return torch.cat(halves, dim=-1).to(t.dtype)


@register_compressor("perceiver")
class PerceiverCompressor(FixedPoolingCompressor):
    """Perceiver-style resampler (https://arxiv.org/abs/2103.03206), trained.

    One cross-attention block (queries attend to all native tokens of the image) + MLP. Two query modes:

    - ``pooled``: the queries are the fixed-pooling outputs, so token count and positions are those of
      ``fixed``. Both residual updates are scaled by tanh-gates that start at 0 (as in Flamingo), so at
      init the module is exactly fixed pooling.
    - ``learned`` (as in the paper): the queries are a learned [D, G, G] latent grid, bilinearly resized
      to the image's output grid of (h / cell_h) x (w / cell_w) cells, cell_h * cell_w = 1 / ratio. The
      queries depend on the image's size but not its content; the count still follows the ratio. Output
      positions are the bottom-right native token of each cell. A final LayerNorm, rescaled to the RMS
      of the native tokens, keeps the output on the scale the LLM expects.

    In both modes queries and keys get 2D RoPE inside the attention (queries at their segment / cell
    centre), so the scores depend on relative position consistently across resolutions.
    """

    def __init__(
        self,
        compression_ratio: float = 1.0,
        hidden_size: int = 2048,
        head_dim: int = 128,
        mlp_ratio: int = 4,
        queries: str = "pooled",
        latent_grid: int = 32,
    ):
        super().__init__(compression_ratio)
        # unused by FixedPoolingCompressor.forward; would be an unused trainable parameter once unfrozen
        del self.null_group
        if hidden_size % head_dim != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by head_dim {head_dim}")
        if queries not in ("pooled", "learned"):
            raise ValueError(f"queries must be 'pooled' or 'learned', got {queries}")
        self.queries = queries
        self.num_heads = hidden_size // head_dim
        self.head_dim = head_dim
        self.cell_h, self.cell_w = _cell_shape(self.pooling_factor)

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
        if queries == "pooled":
            self.attn_gate = nn.Parameter(torch.zeros(1))
            self.mlp_gate = nn.Parameter(torch.zeros(1))
        else:
            self.latents = nn.Parameter(torch.zeros(1, hidden_size, latent_grid, latent_grid))
            self.norm_out = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        """(Re)initialise every weight. Call after ``from_pretrained``: it re-initialises (or leaves
        uninitialised) all weights missing from the checkpoint, which includes this whole module."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LayerNorm)):
                m.reset_parameters()
        if self.queries == "pooled":
            nn.init.zeros_(self.attn_gate)
            nn.init.zeros_(self.mlp_gate)
        else:
            nn.init.trunc_normal_(self.latents, std=0.02)

    def load_perceiver_weights(self, perceiver_path):
        print(f"🔭 Loading Perceiver weights from {perceiver_path}")
        sd = torch.load(perceiver_path, map_location="cpu", weights_only=True)
        self.load_state_dict(sd, strict=True)
        print(f"✅ Perceiver weights loaded, {self.queries} queries, compression ratio {self.compression_ratio}")

    def _cell_grid(self, h: int, w: int):
        return -(-h // self.cell_h), -(-w // self.cell_w)

    def num_output_tokens(self, h: int, w: int) -> int:
        if self.queries == "pooled":
            return super().num_output_tokens(h, w)
        gh, gw = self._cell_grid(h, w)
        return gh * gw

    def kept_indices(self, h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if self.queries == "pooled":
            return super().kept_indices(h, w, device=device)
        # bottom-right native token of each cell (cf. fixed pooling's last token per segment)
        gh, gw = self._cell_grid(h, w)
        rows = ((torch.arange(gh, device=device) + 1) * self.cell_h - 1).clamp_(max=h - 1)
        cols = ((torch.arange(gw, device=device) + 1) * self.cell_w - 1).clamp_(max=w - 1)
        return (rows[:, None] * w + cols[None, :]).flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape [B,H,W,D], got {tuple(x.shape)}")
        B, h, w, D = x.shape
        # [B, H, W, D] -> [B, N, D]
        x = self._flatten_tokens(x)
        if self.pooling_factor == 1:
            return x

        # weights may be fp32 (training) while the rest of NEO runs in bf16
        out_dtype = x.dtype
        x = x.to(self.q_proj.weight.dtype)
        key_h = torch.arange(h, device=x.device, dtype=torch.float32).repeat_interleave(w)
        key_w = torch.arange(w, device=x.device, dtype=torch.float32).repeat(h)

        if self.queries == "pooled":
            queries = super().forward(x)  # [B, S, D] fixed-pooling outputs
            # query position = centroid of its pooling segment
            q_h, q_w = super().forward(torch.stack((key_h, key_w), dim=-1)[None])[0].unbind(-1)
        else:
            gh, gw = self._cell_grid(h, w)
            queries = F.interpolate(self.latents, size=(gh, gw), mode="bilinear", align_corners=False)
            queries = queries.flatten(2).transpose(1, 2).expand(B, -1, -1)  # [B, S, D]
            # query position = centre of its cell
            q_h = (torch.arange(gh, device=x.device, dtype=torch.float32) * self.cell_h
                   + (self.cell_h - 1) / 2).repeat_interleave(gw)
            q_w = (torch.arange(gw, device=x.device, dtype=torch.float32) * self.cell_w
                   + (self.cell_w - 1) / 2).repeat(gh)

        S = queries.shape[1]
        q = self.q_proj(self.norm_q(queries)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.norm_kv(x)
        k = self.k_proj(kv).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = _rotate_2d(q, q_h, q_w), _rotate_2d(k, key_h, key_w)

        # explicit attention instead of SDPA so fvcore counts its FLOPs (scripts/flops.py)
        attn = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5  # [B, heads, S, N]
        attn = attn.float().softmax(dim=-1).to(v.dtype)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)

        if self.queries == "pooled":
            queries = queries + self.attn_gate.tanh() * self.o_proj(out)
            queries = queries + self.mlp_gate.tanh() * self.mlp(self.norm_mlp(queries))
        else:
            queries = queries + self.o_proj(out)
            queries = queries + self.mlp(self.norm_mlp(queries))
            # unit-RMS tokens rescaled to the RMS of this image's native tokens
            queries = self.norm_out(queries) * x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
        return queries.to(out_dtype)